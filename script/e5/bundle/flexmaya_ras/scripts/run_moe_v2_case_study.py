#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gc
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Mapping

import flexmaya_ras as fm

import run_moe_v2_matrix as matrix


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the FlexEva MoE v2 case study with one Maya-full calibration and many FlexEva candidates."
    )
    parser.add_argument("--world-size", type=int, default=32)
    parser.add_argument("--ep-group-size", type=int, default=8)
    parser.add_argument("--micro-batches", type=int, default=64)
    parser.add_argument("--layers", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--variants-per-candidate", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--time-budget-s", type=float, default=3300.0)
    parser.add_argument("--maya-trace-ras-limit", type=int, default=0)
    parser.add_argument("--memory-counts", "--memory-candidate-counts", dest="memory_counts", default="1,8,32")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--memory-worker-mode", action="store_true")
    parser.add_argument("--memory-mode", choices=("maya_full", "maya_trace_ras", "flexeva_selected"), default=None)
    parser.add_argument("--memory-count", type=int, default=None)
    return parser.parse_args()


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return time.perf_counter() - start, value


def parse_counts(value: str) -> list[int]:
    counts = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not counts or any(count <= 0 for count in counts):
        raise ValueError(f"invalid memory counts: {value!r}")
    return counts


def current_rss_bytes() -> int:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def mb(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def workload_args(args: argparse.Namespace | Mapping[str, Any]) -> dict[str, Any]:
    source = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
    return {
        "world_size": int(source["world_size"]),
        "ep_group_size": int(source["ep_group_size"]),
        "micro_batches": int(source["micro_batches"]),
        "layers": int(source["layers"]),
        "seq_len": int(source["seq_len"]),
        "hidden_size": int(source["hidden_size"]),
    }


def load_case_candidates(
    count: int,
    *,
    candidate_limit: int | None = None,
    variants_per_candidate: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case = matrix.build_maya_v2_load_skew_case()
    anchor = asdict(case.anchor)
    base_candidates = [
        asdict(candidate)
        for candidate in case.candidates
        if candidate.candidate_id != case.anchor.candidate_id
    ]
    if candidate_limit is not None:
        base_candidates = base_candidates[: max(int(candidate_limit), 0)]
    if variants_per_candidate is not None:
        count = len(base_candidates) * int(variants_per_candidate)
    return anchor, matrix.expand_candidate_variants(base_candidates, count)


def run_maya_full_calibration(args: Mapping[str, Any], anchor: Mapping[str, Any]) -> dict[str, Any]:
    gen_s, raw = timed(
        lambda: matrix.synthetic_moe_events(
            args,
            candidate_id=str(anchor["candidate_id"]),
            base_candidate_id=matrix.base_candidate_id(anchor),
            ranks=range(int(args["world_size"])),
        )
    )
    trace, replay, phases = matrix.build_and_replay_full(raw)
    total_s = gen_s + sum(phases.values())
    return {
        "candidate_id": str(anchor["candidate_id"]),
        "measured": True,
        "phases_s": {"synthetic_hook_capture_s": gen_s, **phases},
        "total_s": total_s,
        "raw_event_count": len(raw),
        "trace": fm.trace_summary(trace),
        "feedback": replay.to_dict(),
    }


def evaluate_flexeva_case_candidate(payload: tuple[dict[str, Any], dict[str, Any], dict[int, list[int]], float, dict[str, Any]]) -> dict[str, Any]:
    candidate, args, groups, anchor_total_runtime_us, calibration = payload
    flex = matrix.evaluate_flexeva_selected_candidate(candidate, args, groups, anchor_total_runtime_us)
    calibration_raw_count = int(calibration["raw_event_count"])
    calibration_total_s = float(calibration["total_s"])
    selected_raw_count = int(flex["raw_event_count"])
    selected_trace_count = int(flex["trace"]["event_count"])
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "base_candidate_id": matrix.base_candidate_id(candidate),
        "variant_index": int(candidate.get("variant_index", 0)),
        "variant_scale": matrix.candidate_variant_scale(candidate),
        "change_surface": str(candidate["change_surface"]),
        "entry": str(candidate["entry"]),
        "semantic_diffs": list(candidate.get("semantic_diffs", ())),
        "selected_code_partitions": list(flex["selected_code_partitions"]),
        "representative_ranks": list(flex["representative_ranks"]),
        "flexeva_selected": flex,
        "reuse": {
            "raw_event_reuse_rate_vs_maya_full_calibration": 1.0 - (selected_raw_count / max(calibration_raw_count, 1)),
            "rank_capture_reduction_rate": 1.0 - (len(flex["representative_ranks"]) / max(int(args["world_size"]), 1)),
            "selected_trace_event_count": selected_trace_count,
        },
        "speedup": {
            "flexeva_vs_one_round_maya_full_wall": calibration_total_s / max(float(flex["total_s"]), 1e-12),
        },
    }


def run_memory_worker(args: argparse.Namespace) -> dict[str, Any]:
    if args.memory_mode is None or args.memory_count is None:
        raise SystemExit("--memory-worker-mode requires --memory-mode and --memory-count")
    config = workload_args(args)
    groups = matrix.rank_groups(args.world_size, args.ep_group_size)
    _, candidates = load_case_candidates(args.memory_count)
    retained: list[object] = []
    raw_events = 0
    trace_events = 0
    logical_events = 0
    replay_time_us = 0.0

    gc.collect()
    baseline_rss = current_rss_bytes()
    baseline_peak = peak_rss_bytes()
    start = time.perf_counter()

    if args.memory_mode == "flexeva_selected":
        anchor_raw = matrix.synthetic_moe_events(config, candidate_id="anchor_baseline", ranks=range(args.world_size))
        anchor_trace = fm.build_rank_grouped_trace_ras(anchor_raw, groups)
        anchor_feedback = fm.replay_trace_once(anchor_trace)
        retained.append(("anchor", anchor_raw, anchor_trace, anchor_feedback))
        raw_events += len(anchor_raw)
        trace_events += len(anchor_trace.events)
        logical_events += int(anchor_trace.logical_event_count)
        replay_time_us += float(anchor_feedback.total_time_us)

    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if args.memory_mode == "maya_full":
            raw = matrix.synthetic_moe_events(
                config,
                candidate_id=candidate_id,
                base_candidate_id=matrix.base_candidate_id(candidate),
                profile_scale=matrix.candidate_variant_scale(candidate),
                ranks=range(args.world_size),
            )
            trace = fm.build_trace_ras(raw)
        elif args.memory_mode == "maya_trace_ras":
            raw = matrix.synthetic_moe_events(
                config,
                candidate_id=candidate_id,
                base_candidate_id=matrix.base_candidate_id(candidate),
                profile_scale=matrix.candidate_variant_scale(candidate),
                ranks=range(args.world_size),
            )
            trace = fm.build_rank_grouped_trace_ras(raw, groups)
        else:
            selected_partitions = matrix.selected_code_partitions(candidate)
            representatives = tuple(sorted(groups))
            raw = matrix.synthetic_moe_events(
                config,
                candidate_id=candidate_id,
                base_candidate_id=matrix.base_candidate_id(candidate),
                profile_scale=matrix.candidate_variant_scale(candidate),
                ranks=representatives,
                capture_code_partitions=selected_partitions,
            )
            spec = matrix.spec_for_candidate(candidate, config)
            trace = fm.build_selected_trace(
                spec,
                raw,
                selected_ranks=representatives,
                selected_code_partitions=selected_partitions,
                rank_groups=groups,
            )
        feedback = fm.replay_trace_once(trace)
        retained.append((candidate_id, raw, trace, feedback))
        raw_events += len(raw)
        trace_events += len(trace.events)
        logical_events += int(trace.logical_event_count)
        replay_time_us += float(feedback.total_time_us)

    gc.collect()
    current_rss = current_rss_bytes()
    peak_rss = peak_rss_bytes()
    return {
        "mode": args.memory_mode,
        "candidate_count": args.memory_count,
        "measurement_kind": "measured",
        "baseline_rss_mb": mb(baseline_rss),
        "current_rss_mb": mb(current_rss),
        "retained_rss_delta_mb": mb(max(current_rss - baseline_rss, 0)),
        "peak_rss_delta_mb": mb(max(peak_rss - baseline_peak, 0)),
        "raw_event_count": raw_events,
        "trace_event_count": trace_events,
        "logical_event_count": logical_events,
        "replay_time_us": replay_time_us,
        "wall_time_s": time.perf_counter() - start,
    }


def run_memory_child(args: argparse.Namespace, mode: str, count: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--memory-worker-mode",
        "--memory-mode",
        mode,
        "--memory-count",
        str(count),
        "--world-size",
        str(args.world_size),
        "--ep-group-size",
        str(args.ep_group_size),
        "--micro-batches",
        str(args.micro_batches),
        "--layers",
        str(args.layers),
        "--seq-len",
        str(args.seq_len),
        "--hidden-size",
        str(args.hidden_size),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True, cwd=ROOT)
    return json.loads(completed.stdout)


def project_row(row: Mapping[str, Any], count: int) -> dict[str, Any]:
    base_count = max(int(row["candidate_count"]), 1)
    scale = float(count) / float(base_count)
    projected = dict(row)
    projected["candidate_count"] = count
    projected["measurement_kind"] = "projected_from_one_round"
    for key in (
        "retained_rss_delta_mb",
        "peak_rss_delta_mb",
        "raw_event_count",
        "trace_event_count",
        "logical_event_count",
        "replay_time_us",
        "wall_time_s",
    ):
        projected[key] = float(row[key]) * scale
    return projected


def run_memory_scaling(args: argparse.Namespace, counts: list[int]) -> dict[str, Any]:
    measured_full = run_memory_child(args, "maya_full", 1)
    measured_trace = run_memory_child(args, "maya_trace_ras", 1)
    rows: list[dict[str, Any]] = []
    for count in counts:
        rows.append(measured_full if count == 1 else project_row(measured_full, count))
    for count in counts:
        rows.append(measured_trace if count == 1 else project_row(measured_trace, count))
    for count in counts:
        rows.append(run_memory_child(args, "flexeva_selected", count))
    flex_max = next(row for row in rows if row["mode"] == "flexeva_selected" and int(row["candidate_count"]) == max(counts))
    comparisons: dict[str, Any] = {}
    for baseline in ("maya_full", "maya_trace_ras"):
        base = next(row for row in rows if row["mode"] == baseline and int(row["candidate_count"]) == max(counts))
        comparisons[baseline] = {
            "retained_memory_reduction": 1.0
            - (float(flex_max["retained_rss_delta_mb"]) / max(float(base["retained_rss_delta_mb"]), 1e-9)),
            "trace_event_reduction": 1.0
            - (float(flex_max["trace_event_count"]) / max(float(base["trace_event_count"]), 1.0)),
        }
    return {
        "configuration": {**workload_args(args), "candidate_counts": counts},
        "rows": rows,
        "summary": {
            "max_count": max(counts),
            "comparisons_at_max_count": comparisons,
        },
    }


def write_outputs(out_dir: Path, result: Mapping[str, Any], memory: Mapping[str, Any] | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_manifest.json").write_text(json.dumps(result["run_manifest"], indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "matrix.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "oracle_fidelity.json").write_text(json.dumps(result["oracle_fidelity"], indent=2, sort_keys=True), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "candidate_id",
            "base_candidate_id",
            "variant_scale",
            "predicted_runtime_us",
            "flexeva_wall_s",
            "speedup_vs_one_round_maya_full",
            "raw_event_reuse_rate_vs_maya_full",
            "rank_capture_reduction_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["candidates"]:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "base_candidate_id": row["base_candidate_id"],
                    "variant_scale": row["variant_scale"],
                    "predicted_runtime_us": row["flexeva_selected"]["predicted_candidate_total_runtime_us"],
                    "flexeva_wall_s": row["flexeva_selected"]["total_s"],
                    "speedup_vs_one_round_maya_full": row["speedup"]["flexeva_vs_one_round_maya_full_wall"],
                    "raw_event_reuse_rate_vs_maya_full": row["reuse"]["raw_event_reuse_rate_vs_maya_full_calibration"],
                    "rank_capture_reduction_rate": row["reuse"]["rank_capture_reduction_rate"],
                }
            )
    if memory is not None:
        (out_dir / "memory_scaling.json").write_text(json.dumps(memory, indent=2, sort_keys=True), encoding="utf-8")
        with (out_dir / "memory_scaling.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "mode",
                "candidate_count",
                "measurement_kind",
                "retained_rss_delta_mb",
                "peak_rss_delta_mb",
                "raw_event_count",
                "trace_event_count",
                "logical_event_count",
                "wall_time_s",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in memory["rows"]:
                writer.writerow({field: row[field] for field in fieldnames})
    write_tables(out_dir, result, memory)


def write_tables(out_dir: Path, result: Mapping[str, Any], memory: Mapping[str, Any] | None) -> None:
    candidates = list(result["candidates"])
    winner = result["oracle_fidelity"]["predicted_winner"]
    lines = [
        "# FlexEva MoE V2 Case Study",
        "",
        f"- Predicted winner: `{winner}`",
        f"- Maya-full calibration wall time: {result['maya_full_calibration']['total_s']:.6f} s",
        f"- Candidate count: {len(candidates)}",
        "",
        "## Candidate Matrix",
        "",
        "| Candidate | Base | FlexEva wall s | Speedup vs one Maya-full | Raw event reuse |",
        "|---|---|---:|---:|---:|",
    ]
    for row in candidates[: min(len(candidates), 12)]:
        lines.append(
            "| {candidate} | {base} | {wall:.6f} | {speedup:.2f} | {reuse:.2%} |".format(
                candidate=row["candidate_id"],
                base=row["base_candidate_id"],
                wall=float(row["flexeva_selected"]["total_s"]),
                speedup=float(row["speedup"]["flexeva_vs_one_round_maya_full_wall"]),
                reuse=float(row["reuse"]["raw_event_reuse_rate_vs_maya_full_calibration"]),
            )
        )
    if len(candidates) > 12:
        lines.append(f"| ... | ... | ... | ... | ... |")
    if memory is not None:
        lines += [
            "",
            "## Memory Scaling",
            "",
            "| Mode | K | Kind | Retained RSS MB | Trace events |",
            "|---|---:|---|---:|---:|",
        ]
        for row in memory["rows"]:
            lines.append(
                "| {mode} | {count} | {kind} | {rss:.2f} | {events:.0f} |".format(
                    mode=row["mode"],
                    count=int(row["candidate_count"]),
                    kind=row["measurement_kind"],
                    rss=float(row["retained_rss_delta_mb"]),
                    events=float(row["trace_event_count"]),
                )
            )
    (out_dir / "case_study_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.memory_worker_mode:
        print(json.dumps(run_memory_worker(args), sort_keys=True))
        return
    if args.out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = ROOT / "output" / f"case_study_moe_v2_large_{stamp}"
    start = time.perf_counter()
    config = workload_args(args)
    groups = matrix.rank_groups(args.world_size, args.ep_group_size)
    anchor, candidates = load_case_candidates(
        args.candidate_count,
        candidate_limit=args.candidate_limit,
        variants_per_candidate=args.variants_per_candidate,
    )
    anchor_payload = matrix.build_anchor(config, anchor, groups)
    anchor_total_runtime_us = float(anchor_payload["anchor"].feedback.total_time_us)
    calibration = run_maya_full_calibration(config, anchor)

    worker_payloads = [(candidate, config, groups, anchor_total_runtime_us, calibration) for candidate in candidates]
    if args.workers <= 1:
        rows = [evaluate_flexeva_case_candidate(payload) for payload in worker_payloads]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(evaluate_flexeva_case_candidate, worker_payloads))
    rows.sort(key=lambda item: int(item["variant_index"]))
    for row in rows:
        row["maya_trace_ras"] = None
    if args.maya_trace_ras_limit > 0:
        by_id = {row["candidate_id"]: row for row in rows}
        for candidate in candidates[: args.maya_trace_ras_limit]:
            by_id[str(candidate["candidate_id"])]["maya_trace_ras"] = matrix.evaluate_maya_trace_ras_candidate(
                candidate,
                config,
                groups,
            )

    predicted_order = [
        row["candidate_id"]
        for row in sorted(rows, key=lambda item: float(item["flexeva_selected"]["predicted_candidate_total_runtime_us"]))
    ]
    elapsed_before_memory = time.perf_counter() - start
    memory = None
    memory_status = "skipped_by_flag" if args.skip_memory else "not_started"
    if not args.skip_memory:
        counts = parse_counts(args.memory_counts)
        if elapsed_before_memory < args.time_budget_s * 0.75:
            memory = run_memory_scaling(args, counts)
            memory_status = "completed"
        else:
            memory_status = "skipped_due_to_time_budget"

    result = {
        "case_id": matrix.MAYA_V2_LOAD_SKEW_CASE_ID,
        "round_id": matrix.MAYA_V2_LOAD_SKEW_ROUND_ID,
        "mode": "case_study_one_maya_full_calibration_many_flexeva",
        "designs": {
            "maya_full": "measured exactly once as cold calibration",
            "flexeva_selected": "measured for every expanded case-study candidate",
            "maya_trace_ras": "optional diagnostic sample controlled by --maya-trace-ras-limit",
        },
        "run_manifest": {
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": {
                **config,
                "candidate_count": args.candidate_count,
                "workers": args.workers,
                "time_budget_s": args.time_budget_s,
                "memory_counts": parse_counts(args.memory_counts),
            },
            "designs": {
                "maya_full_calibration": "one full synthetic hook capture, C++ collation, and Python replay on anchor_baseline",
                "flexeva_selected": "selected representative hook capture, C++ selected placement, one Python replay, anchor-delta feedback",
                "memory_projection": "Maya-full and Maya-trace-RAS K>1 memory rows are projected from measured K=1 to keep the case study inside one hour",
            },
        },
        "rank_groups": {str(rep): members for rep, members in groups.items()},
        "anchor": anchor_payload["summary"],
        "maya_full_calibration": calibration,
        "candidates": rows,
        "oracle_fidelity": {
            "oracle_coverage": 0.0,
            "oracle_winner": None,
            "predicted_winner": predicted_order[0] if predicted_order else None,
            "winner_match": None,
            "ranking_match": None,
            "uncertainty_flags": [row["candidate_id"] for row in rows],
            "note": "native oracle was not run in this bounded case-study command",
        },
        "memory_status": memory_status,
        "elapsed_s": time.perf_counter() - start,
    }
    write_outputs(args.out_dir, result, memory)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "matrix_json": str(args.out_dir / "matrix.json"),
                "predicted_winner": result["oracle_fidelity"]["predicted_winner"],
                "memory_status": memory_status,
                "elapsed_s": result["elapsed_s"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
