#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import gc
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import flexmaya_ras as fm

import run_moe_v2_matrix as matrix


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure retained memory while evaluating many Maya/FlexEva candidates.")
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--ep-group-size", type=int, default=8)
    parser.add_argument("--micro-batches", type=int, default=8)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--candidate-counts", default="1,2,4,8,16,32")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--mode", choices=("maya_full", "maya_trace_ras", "flexeva_selected"), default=None)
    parser.add_argument("--candidate-count", type=int, default=None)
    return parser.parse_args()


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


def candidate_cycle() -> list[dict[str, Any]]:
    case = matrix.build_maya_v2_load_skew_case()
    candidates = [
        asdict(candidate)
        for candidate in case.candidates
        if candidate.candidate_id != case.anchor.candidate_id
    ]
    if not candidates:
        raise RuntimeError("Maya v2 load-skew case has no candidates")
    return candidates


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode is None or args.candidate_count is None:
        raise SystemExit("--worker-mode requires --mode and --candidate-count")
    config = vars(args).copy()
    groups = matrix.rank_groups(args.world_size, args.ep_group_size)
    candidates = candidate_cycle()
    retained: list[object] = []
    raw_events = 0
    trace_events = 0
    logical_events = 0
    replay_time_us = 0.0

    gc.collect()
    baseline_rss = current_rss_bytes()
    baseline_peak = peak_rss_bytes()
    start = time.perf_counter()

    if args.mode == "flexeva_selected":
        anchor_candidate = {
            "candidate_id": "anchor_baseline",
            "entry": candidates[0]["entry"],
            "change_surface": "anchor",
            "semantic_diffs": (),
        }
        anchor_raw = matrix.synthetic_moe_events(
            config,
            candidate_id="anchor_baseline",
            ranks=range(args.world_size),
        )
        anchor_trace = fm.build_rank_grouped_trace_ras(anchor_raw, groups)
        anchor_feedback = fm.replay_trace_once(anchor_trace)
        retained.append(("anchor", anchor_raw, anchor_trace, anchor_feedback))
        raw_events += len(anchor_raw)
        trace_events += len(anchor_trace.events)
        logical_events += int(anchor_trace.logical_event_count)
        replay_time_us += float(anchor_feedback.total_time_us)

    for index in range(args.candidate_count):
        candidate = candidates[index % len(candidates)]
        candidate_id = str(candidate["candidate_id"])
        if args.mode == "maya_full":
            raw = matrix.synthetic_moe_events(config, candidate_id=candidate_id, ranks=range(args.world_size))
            trace = fm.build_trace_ras(raw)
        elif args.mode == "maya_trace_ras":
            raw = matrix.synthetic_moe_events(config, candidate_id=candidate_id, ranks=range(args.world_size))
            trace = fm.build_rank_grouped_trace_ras(raw, groups)
        else:
            selected_partitions = matrix.selected_code_partitions(candidate)
            representatives = tuple(sorted(groups))
            raw = matrix.synthetic_moe_events(
                config,
                candidate_id=candidate_id,
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
        "mode": args.mode,
        "candidate_count": args.candidate_count,
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


def parse_counts(value: str) -> list[int]:
    counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not counts or any(item <= 0 for item in counts):
        raise ValueError(f"invalid candidate counts: {value!r}")
    return sorted(dict.fromkeys(counts))


def run_child(args: argparse.Namespace, mode: str, count: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode",
        "--mode",
        mode,
        "--candidate-count",
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


def slope(rows: list[dict[str, Any]], field: str) -> float:
    if len(rows) < 2:
        return 0.0
    first = rows[0]
    last = rows[-1]
    delta_count = int(last["candidate_count"]) - int(first["candidate_count"])
    if delta_count <= 0:
        return 0.0
    return (float(last[field]) - float(first[field])) / float(delta_count)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(str(row["mode"]), []).append(row)
    for mode_rows in by_mode.values():
        mode_rows.sort(key=lambda item: int(item["candidate_count"]))
    summary: dict[str, Any] = {"slopes_mb_per_candidate": {}, "max_count": max(int(row["candidate_count"]) for row in rows)}
    for mode, mode_rows in sorted(by_mode.items()):
        summary["slopes_mb_per_candidate"][mode] = slope(mode_rows, "retained_rss_delta_mb")
    max_rows = {
        mode: mode_rows[-1]
        for mode, mode_rows in by_mode.items()
        if mode_rows
    }
    flex = max_rows.get("flexeva_selected")
    if flex is not None:
        comparisons: dict[str, Any] = {}
        for baseline in ("maya_full", "maya_trace_ras"):
            base = max_rows.get(baseline)
            if base is None:
                continue
            retained = float(base["retained_rss_delta_mb"])
            flex_retained = float(flex["retained_rss_delta_mb"])
            comparisons[baseline] = {
                "retained_memory_reduction": 1.0 - (flex_retained / max(retained, 1e-9)),
                "retained_memory_ratio": flex_retained / max(retained, 1e-9),
                "trace_event_reduction": 1.0 - (
                    float(flex["trace_event_count"]) / max(float(base["trace_event_count"]), 1.0)
                ),
            }
        summary["max_count_comparisons"] = comparisons
    return summary


def write_outputs(out_dir: Path, result: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "memory_scaling.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    with (out_dir / "memory_scaling.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "mode",
            "candidate_count",
            "retained_rss_delta_mb",
            "peak_rss_delta_mb",
            "raw_event_count",
            "trace_event_count",
            "logical_event_count",
            "wall_time_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({field: row[field] for field in fieldnames})
    (out_dir / "result.json").write_text(
        json.dumps({"memory_scaling_json": str(out_dir / "memory_scaling.json")}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.worker_mode:
        print(json.dumps(run_worker(args), sort_keys=True))
        return
    if args.out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = ROOT / "output" / f"candidate_memory_{stamp}"
    counts = parse_counts(args.candidate_counts)
    modes = ("maya_full", "maya_trace_ras", "flexeva_selected")
    rows = [run_child(args, mode, count) for mode in modes for count in counts]
    result = {
        "configuration": {
            "world_size": args.world_size,
            "ep_group_size": args.ep_group_size,
            "micro_batches": args.micro_batches,
            "layers": args.layers,
            "seq_len": args.seq_len,
            "hidden_size": args.hidden_size,
            "candidate_counts": counts,
            "measurement": "fresh subprocess per mode/count; retained RSS includes C++ pybind allocations",
        },
        "rows": rows,
        "summary": summarize(rows),
    }
    write_outputs(args.out_dir, result)
    print(json.dumps({"memory_scaling_json": str(args.out_dir / "memory_scaling.json"), "summary": result["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
