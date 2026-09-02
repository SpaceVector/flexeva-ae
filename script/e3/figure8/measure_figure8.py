#!/usr/bin/env python3
"""Collect the three paper Figure 8 scale-sensitivity points."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import flexmaya_ras as fm

from gpt_scale_base import (
    SOURCE_OPS,
    build_full,
    build_trace_ras,
    feedback_generation_time,
    feedback_payload,
    load_region_windows,
    marker_step_window_seconds,
    selected_refresh,
    selected_region_wall_time_s,
    timed,
)
from measure_maya_megatron_fakecuda_similarity import (
    load_step_window,
    raw_event_from_record,
    run_maya_megatron_case,
)
from measure_megatron_trace_similarity import MegatronCase, default_cases


MUTATION = "attention_backward"
CASE_NAMES = (
    "megatron_2p7b_8gpu",
    "megatron_2p7b_16gpu_dp2",
    "megatron_18p4b_16gpu",
)
MAX_STEP_BEGIN_SKEW_S = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--maya-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--proot", type=Path, required=True)
    parser.add_argument("--local-device-count", type=int, default=8)
    parser.add_argument("--keep-raw-traces", action="store_true")
    parser.set_defaults(
        reuse_existing_traces=False,
        source_region_markers=True,
        sync_before_step_window=True,
    )
    return parser.parse_args()


def paper_cases() -> tuple[MegatronCase, ...]:
    by_name = {case.name: case for case in default_cases()}
    return tuple(by_name[name] for name in CASE_NAMES)


def progress(message: str) -> None:
    print(f"[figure8] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}", file=sys.stderr, flush=True)


@contextmanager
def single_node_launch() -> Iterator[None]:
    keys = ("FLEXMAYA_NNODES", "FLEXMAYA_NODE_RANK")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update({"FLEXMAYA_NNODES": "1", "FLEXMAYA_NODE_RANK": "0"})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def capture_case(args: argparse.Namespace, case: MegatronCase, case_dir: Path) -> dict[str, object]:
    case_dir.mkdir(parents=True, exist_ok=True)
    if case.world_size == 8:
        with single_node_launch():
            run = run_maya_megatron_case(args, case, case_dir)
    else:
        run = run_maya_megatron_case(args, case, case_dir)
    if int(run["return_code"]) != 0:
        raise RuntimeError(f"{case.name} failed; see {run['stdout']} and {run['stderr']}")
    return run


def step_begin_skew_s(case: MegatronCase, trace_dir: Path) -> float:
    starts: list[int] = []
    for rank in range(case.world_size):
        with (trace_dir / f"rank_{rank}_markers.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("kind") == "step_begin" and record.get("label") == "training_step":
                    timestamp = int(record.get("realtime_ns") or record.get("monotonic_ns") or 0)
                    if timestamp <= 0:
                        raise RuntimeError(f"{case.name}: rank {rank} lacks a comparable step timestamp")
                    starts.append(timestamp)
                    break
    if len(starts) != case.world_size:
        raise RuntimeError(f"{case.name}: expected {case.world_size} step markers, found {len(starts)}")
    return (max(starts) - min(starts)) / 1.0e9


def parse_case_raw_events_with_regions(case: MegatronCase, trace_dir: Path) -> list[object]:
    """Preserve the Figure 8 event population, including context operations."""
    events: list[object] = []
    next_id = 1
    for rank in range(case.world_size):
        markers_path = trace_dir / f"rank_{rank}_markers.jsonl"
        step_start, step_end = load_step_window(markers_path)
        windows = load_region_windows(markers_path, step_start, step_end)
        window_index = 0
        with (trace_dir / f"rank_{rank}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                timestamp = int(record.get("ts") or 0)
                if not step_start <= timestamp <= step_end:
                    continue
                if str(record.get("type", "")) in {"marker", "other"}:
                    continue
                event = raw_event_from_record(record, case=case, rank=rank, event_id=next_id)
                while window_index < len(windows) and timestamp > windows[window_index][1]:
                    window_index += 1
                if window_index < len(windows):
                    begin, end, label = windows[window_index]
                    if begin <= timestamp <= end:
                        event.code_partition = label
                events.append(event)
                next_id += 1
    return events


def write_source_files(
    root: Path,
    case: MegatronCase,
    *,
    anchor_paths: dict[str, Path] | None = None,
    changed_ops: tuple[str, ...] = (),
) -> dict[str, Path]:
    source_dir = root / "source_partitions"
    source_dir.mkdir(parents=True, exist_ok=True)
    anchor_paths = anchor_paths or {}
    paths: dict[str, Path] = {}
    for stage in range(case.pp):
        partition = f"stage_{stage:03d}"
        if partition in anchor_paths:
            paths[partition] = anchor_paths[partition]
            continue
        path = source_dir / f"{partition}.py"
        path.write_text(
            f"# Megatron {case.parameter_scale}/{case.world_size}-rank stage {stage}\n"
            f"PIPELINE_STAGE = {stage}\n",
            encoding="utf-8",
        )
        paths[partition] = path
    for operation in SOURCE_OPS:
        if operation not in changed_ops and operation in anchor_paths:
            paths[operation] = anchor_paths[operation]
            continue
        path = source_dir / f"{operation}.py"
        suffix = "\n# fixed Figure 8 source mutation\n" if operation in changed_ops else ""
        path.write_text(
            f"# Megatron source partition: {operation}\n"
            f"OP = {operation!r}\n"
            f"PARAMETER_SCALE = {case.parameter_scale!r}\n"
            f"{suffix}",
            encoding="utf-8",
        )
        paths[operation] = path
    return paths


def cleanup_traces(args: argparse.Namespace, run: dict[str, object]) -> None:
    if args.keep_raw_traces:
        return
    for path in Path(str(run["trace_dir"])).glob("rank_*.jsonl"):
        path.unlink(missing_ok=True)


def measure_case(args: argparse.Namespace, case: MegatronCase) -> dict[str, object]:
    progress(f"{case.name}: capture")
    case_dir = args.out_dir / case.name
    anchor_paths = write_source_files(case_dir / "anchor", case)
    candidate_paths = write_source_files(
        case_dir / "candidate_attention_backward",
        case,
        anchor_paths=anchor_paths,
        changed_ops=(MUTATION,),
    )
    run = capture_case(args, case, case_dir)
    trace_dir = Path(str(run["trace_dir"]))
    skew_s = step_begin_skew_s(case, trace_dir)
    if skew_s > MAX_STEP_BEGIN_SKEW_S:
        raise RuntimeError(f"{case.name}: step_begin skew {skew_s:.6f}s exceeds {MAX_STEP_BEGIN_SKEW_S:.1f}s")

    progress(f"{case.name}: parse and evaluate")
    step_window = marker_step_window_seconds(case, trace_dir)
    parse_s, raw_events = timed(lambda: parse_case_raw_events_with_regions(case, trace_dir))
    full_trace, full_replay, full_phases = build_full(raw_events)
    anchor_trace, anchor_replay, anchor_phases, anchor_spec = build_trace_ras(case, raw_events, anchor_paths)
    candidate_trace, candidate_replay, candidate_phases, candidate_spec = build_trace_ras(
        case, raw_events, candidate_paths
    )
    _, anchor_hashes = timed(lambda: fm.source_hashes(anchor_spec))
    source_analysis_s, candidate_hashes = timed(lambda: fm.source_hashes(candidate_spec))
    anchor = fm.FlexMayaAnchor(
        spec=anchor_spec,
        source_hashes=anchor_hashes,
        trace=anchor_trace,
        feedback=anchor_replay,
        summary=feedback_payload(anchor_trace, anchor_replay),
    )
    plan_s, plan = timed(
        lambda: fm.plan_candidate_refresh(
            anchor,
            candidate_spec,
            candidate_trace,
            candidate_source_hashes=candidate_hashes,
        )
    )
    selected_trace, selected_replay, selected_phases = selected_refresh(candidate_trace, plan)

    full_feedback = feedback_payload(full_trace, full_replay)
    candidate_feedback = feedback_payload(candidate_trace, candidate_replay)
    selected_feedback = feedback_payload(selected_trace, selected_replay)
    full_feedback_s = feedback_generation_time(full_feedback)
    candidate_feedback_s = feedback_generation_time(candidate_feedback)
    selected_feedback_s = feedback_generation_time(selected_feedback)
    selective_emulation_s = selected_region_wall_time_s(case, trace_dir, (MUTATION,))
    emulation_s = float(run["elapsed_s"])

    maya_full_s = (
        emulation_s
        + parse_s
        + full_phases["trace_build_s"]
        + full_phases["python_replay_s"]
        + full_feedback_s
    )
    maya_trace_ras_s = (
        emulation_s
        + parse_s
        + candidate_phases["trace_build_s"]
        + candidate_phases["python_replay_s"]
        + candidate_feedback_s
    )
    flexeva_refresh_s = (
        source_analysis_s
        + plan_s
        + selective_emulation_s
        + selected_phases["trace_filter_s"]
        + selected_phases["python_replay_s"]
        + selected_feedback_s
    )
    total_events = int(candidate_trace.logical_event_count)
    regenerated_events = int(selected_trace.logical_event_count)
    cuda_get_device_events = sum(event.api == "cudaGetDevice" for event in raw_events)

    row = {
        "case": asdict(case),
        "label": f"{case.parameter_scale}/{case.world_size}GPU TP{case.tp}-PP{case.pp}-DP{case.dp}",
        "mutation": MUTATION,
        "physical_gpu_count": case.world_size,
        "run": run,
        "step_begin_skew_s": skew_s,
        "step_window_s": step_window,
        "raw_events": len(raw_events),
        "api_audit": {
            "cudaGetDevice_modeled_count": cuda_get_device_events,
            "cudaGetDevice_replay_count": cuda_get_device_events,
        },
        "maya_full": {
            **full_feedback,
            "phases_s": {
                "maya_emulation_s": emulation_s,
                "jsonl_parse_s": parse_s,
                **full_phases,
                "feedback_generation_s": full_feedback_s,
                "total_s": maya_full_s,
            },
        },
        "maya_trace_ras": {
            **candidate_feedback,
            "phases_s": {
                "maya_emulation_s": emulation_s,
                "jsonl_parse_s": parse_s,
                **candidate_phases,
                "feedback_generation_s": candidate_feedback_s,
                "total_s": maya_trace_ras_s,
            },
        },
        "flexeva_refresh": {
            "plan": asdict(plan),
            "selected_trace": selected_feedback["trace"],
            "feedback": selected_feedback["feedback"],
            "phases_s": {
                "source_analysis_s": source_analysis_s,
                "refresh_plan_s": plan_s,
                "selective_emulation_s": selective_emulation_s,
                **selected_phases,
                "feedback_generation_s": selected_feedback_s,
                "total_s": flexeva_refresh_s,
            },
        },
        "metrics": {
            "total_trace_events": total_events,
            "compact_trace_events": len(candidate_trace.events),
            "regenerated_trace_events": regenerated_events,
            "reused_trace_events": max(total_events - regenerated_events, 0),
            "refresh_ratio": regenerated_events / max(total_events, 1),
            "trace_ras_compaction_ratio": len(candidate_trace.events) / max(len(full_trace.events), 1),
            "maya_full_s": maya_full_s,
            "maya_trace_ras_s": maya_trace_ras_s,
            "flexeva_refresh_s": flexeva_refresh_s,
            "speedup_vs_maya_full": maya_full_s / max(flexeva_refresh_s, 1.0e-12),
            "speedup_vs_maya_trace_ras": maya_trace_ras_s / max(flexeva_refresh_s, 1.0e-12),
        },
    }
    cleanup_traces(args, run)
    progress(f"{case.name}: done")
    return row


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(args: argparse.Namespace, results: list[dict[str, object]]) -> None:
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": {
            "workload": "Megatron trace-shape workload through fake-CUDA frun",
            "target_scale": "paper Figure 8 model/cluster and trace-size sensitivity",
            "mutation": MUTATION,
            "physical_execution": (
                "2.7B/8GPU uses one guarded eight-A100 node; both 16GPU cases use two guarded "
                "eight-A100 nodes with eight ranks per node"
            ),
            "rank_synchronization": (
                "host-side Gloo barrier immediately before every measured step window; "
                f"maximum accepted step_begin skew is {MAX_STEP_BEGIN_SKEW_S:.1f}s"
            ),
            "mutation_scope": (
                "controlled attention_backward source-manifest change over the captured trace; "
                "not a historical agent patch or model-output correctness experiment"
            ),
            "maya_full": "candidate process wall + JSONL parse + full trace build/replay + feedback serialization",
            "maya_trace_ras": (
                "candidate process wall + JSONL parse + active-lane trace-RAS build/full replay + feedback; "
                "author ablation, not an original Maya feature"
            ),
            "flexeva_refresh": (
                "source analysis + RAS refresh plan + marker-derived selective emulation + selected trace replay/feedback"
            ),
            "sample_count": 1,
        },
        "results": results,
    }
    (args.out_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    summary_rows: list[dict[str, object]] = []
    scale_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    for row in results:
        case = row["case"]
        metrics = row["metrics"]
        summary_rows.append(
            {
                "case": case["name"],
                "label": row["label"],
                "parameter_scale": case["parameter_scale"],
                "world_size": case["world_size"],
                "tp": case["tp"],
                "pp": case["pp"],
                "dp": case["dp"],
                "global_batch_size": case["global_batch_size"],
                "micro_batches": case["micro_batches"],
                "mutation": row["mutation"],
                "raw_events": row["raw_events"],
                "total_trace_events": metrics["total_trace_events"],
                "compact_trace_events": metrics["compact_trace_events"],
                "regenerated_trace_events": metrics["regenerated_trace_events"],
                "reused_trace_events": metrics["reused_trace_events"],
                "refresh_ratio": metrics["refresh_ratio"],
                "trace_ras_compaction_ratio": metrics["trace_ras_compaction_ratio"],
                "affected_trace_partitions": row["flexeva_refresh"]["plan"]["affected_trace_partition_count"],
                "maya_full_s": metrics["maya_full_s"],
                "maya_trace_ras_s": metrics["maya_trace_ras_s"],
                "flexeva_refresh_s": metrics["flexeva_refresh_s"],
                "speedup_vs_maya_full": metrics["speedup_vs_maya_full"],
                "speedup_vs_maya_trace_ras": metrics["speedup_vs_maya_trace_ras"],
                "candidate_process_wall_s": row["run"]["elapsed_s"],
                "candidate_step_window_mean_s": row["step_window_s"]["mean"],
            }
        )
        for system, key in (
            ("Maya-full", "maya_full_s"),
            ("Maya-trace-RAS", "maya_trace_ras_s"),
            ("FlexEva refresh", "flexeva_refresh_s"),
        ):
            scale_rows.append({"label": row["label"], "system": system, "seconds": metrics[key]})
            trace_rows.append(
                {
                    "label": row["label"],
                    "system": system,
                    "total_trace_events": metrics["total_trace_events"],
                    "regenerated_trace_events": metrics["regenerated_trace_events"],
                    "seconds": metrics[key],
                    "refresh_ratio": metrics["refresh_ratio"],
                }
            )

    write_csv(args.out_dir / "summary.csv", list(summary_rows[0]), summary_rows)
    write_csv(args.out_dir / "figure8a.csv", ["label", "system", "seconds"], scale_rows)
    write_csv(
        args.out_dir / "figure8b.csv",
        ["label", "system", "total_trace_events", "regenerated_trace_events", "seconds", "refresh_ratio"],
        trace_rows,
    )
    lines = [
        "# Figure 8 Scale Sensitivity",
        "",
        f"Fixed mutation: `{MUTATION}`. Each row is one synchronized capture.",
        "",
        "| Target | Total events | Refresh ratio | Maya-full (s) | Maya-trace-RAS (s) | FlexEva (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['label']} | {int(row['total_trace_events']):,} | {100 * float(row['refresh_ratio']):.2f}% | "
            f"{float(row['maya_full_s']):.3f} | {float(row['maya_trace_ras_s']):.3f} | "
            f"{float(row['flexeva_refresh_s']):.3f} |"
        )
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_peer(args: argparse.Namespace) -> int:
    for case in paper_cases():
        if case.world_size == 16:
            progress(f"{case.name}: peer capture")
            capture_case(args, case, args.out_dir / case.name)
    print(json.dumps({"peer_node_rank": int(os.environ["FLEXMAYA_NODE_RANK"]), "captures": 2}))
    return 0


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("FLEXMAYA_NODE_RANK", "0")) != 0:
        return run_peer(args)
    results = [measure_case(args, case) for case in paper_cases()]
    write_outputs(args, results)
    print(json.dumps({"result": str(args.out_dir / "result.json"), "figure8a": str(args.out_dir / "figure8a.csv"), "figure8b": str(args.out_dir / "figure8b.csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
