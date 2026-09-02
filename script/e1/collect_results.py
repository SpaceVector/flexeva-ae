#!/usr/bin/env python3
"""Combine current E1 traces, routing runs, and source diffs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path


ROUND_NAMES = (
    "round_00_baseline",
    "round_01_packed_variable_dispatch",
    "round_02_top1_confident_fallback",
    "round_03_sdpa_attention_recovery",
    "round_04_switch_top1_routing",
)
STAGES = ("Base", "S1", "S2", "S3", "S4")
MUTATIONS = (
    "clean historical sparse-MoE baseline",
    "packed variable-length dispatch",
    "confident-primary routing fallback",
    "remove fallback and use SDPA attention",
    "Switch-style top-1 routing",
)
METRICS = {
    "time": "step_time_s",
    "a2a": "estimated_a2a_bytes",
    "drop": "tokens_dropped",
    "reroute": "tokens_rerouted",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty E1 input: {path}")
    return rows


def routing_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    summary = payload.get("summary", payload)
    if not isinstance(summary, dict) or summary.get("status") != "complete":
        raise ValueError(f"incomplete E1 routing run: {path}")
    if (int(summary.get("world_size", 0)), int(summary.get("rank", -1))) != (128, 0):
        raise ValueError(f"E1 routing run is not logical rank 0 of 128: {path}")
    if int(summary.get("benchmark_samples", 0)) != 1:
        raise ValueError(f"E1 routing run must contain one measured step: {path}")
    for field in ("tokens_dropped", "tokens_rerouted"):
        value = float(summary[field])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid E1 {field}: {path}")
    return summary


def changed_lines(before: Path, after: Path) -> tuple[int, int]:
    completed = subprocess.run(
        ["git", "diff", "--no-index", "--numstat", "--", str(before), str(after)],
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(completed.stderr.strip() or "git diff --no-index failed")
    added = deleted = 0
    for line in completed.stdout.splitlines():
        left, right, path = line.split("\t", 2)
        if "/__pycache__/" in path:
            continue
        if left == "-" or right == "-":
            raise ValueError("E1 source snapshots contain a binary change")
        added += int(left)
        deleted += int(right)
    return added, deleted


def collect(trace_metrics: Path, routing_root: Path, workload_root: Path) -> list[dict[str, object]]:
    traces = read_csv(trace_metrics)
    if [(int(row["round"]), row["stage"]) for row in traces] != list(enumerate(STAGES)):
        raise ValueError("E1 trace round order differs")
    values: list[dict[str, object]] = []
    for round_id, (round_name, stage, mutation, trace) in enumerate(
        zip(ROUND_NAMES, STAGES, MUTATIONS, traces, strict=True)
    ):
        routing = routing_summary(routing_root / round_name / "benchmark_result.json")
        if round_id == 0:
            added = deleted = 0
        else:
            added, deleted = changed_lines(workload_root / ROUND_NAMES[round_id - 1], workload_root / round_name)
        values.append(
            {
                "round": round_id,
                "stage": stage,
                "mutation": mutation,
                "added_lines": added,
                "deleted_lines": deleted,
                "total_changed_lines": added + deleted,
                "step_time_s": trace["rank0_marker_step_s"],
                "estimated_a2a_bytes": trace["trace_rank0_a2a_bytes"],
                "tokens_dropped": routing["tokens_dropped"],
                "tokens_rerouted": routing["tokens_rerouted"],
                "benchmark_samples": routing["benchmark_samples"],
                "world_size": routing["world_size"],
                "seed": 1234,
                "trace_dir": trace["trace_dir"],
            }
        )

    baseline = values[0]
    for row in values:
        for prefix, field in METRICS.items():
            base = float(baseline[field])
            if base <= 0.0:
                raise ValueError(f"E1 baseline {field} must be positive")
            reduction = (base - float(row[field])) / base
            row[f"{prefix}_reduction_vs_baseline"] = reduction
            row[f"{prefix}_normalized_improvement"] = 1.0 + reduction
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-metrics", type=Path, required=True)
    parser.add_argument("--routing-root", type=Path, required=True)
    parser.add_argument("--workload-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = collect(args.trace_metrics, args.routing_root, args.workload_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
