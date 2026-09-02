#!/usr/bin/env python3
"""Validate freshly generated E2 tables and traces."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


TABLE4_CASES = (
    ("GPT", "megatron_2p7b_8gpu", 8),
    ("GPT", "megatron_18p4b_16gpu", 16),
    ("GPT", "megatron_2p7b_16gpu_dp2", 16),
    ("GPT", "megatron_18p4b_16gpu_dp2", 16),
    ("Routed-MoE", "routed_moe_intra_group_0_1", 16),
    ("Routed-MoE", "routed_moe_cross_group_0_8", 16),
    ("Routed-MoE", "routed_moe_cross_group_0_15", 16),
    ("Routed-MoE", "routed_moe_boundary_7_8", 16),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty E2 output: {path}")
    return rows


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-9, abs_tol=1.0e-9)


def validate_table4(path: Path, require_traces: bool) -> None:
    current = read_rows(path)
    if len(current) != len(TABLE4_CASES):
        raise ValueError(f"Table 4 requires {len(TABLE4_CASES)} rows, found {len(current)}")
    for row, expected in zip(current, TABLE4_CASES, strict=True):
        actual = row["workload"], row["case"], int(row["world_size"])
        if actual != expected:
            raise ValueError(f"unexpected Table 4 case: {actual}")
        if min(float(row["logical_event_coverage"]), float(row["weighted_event_jaccard"])) < 0.9999:
            raise ValueError(f"Table 4 similarity below threshold: {row['case']}")
        if min(int(row["raw_events"]), int(row["maya_events"]), int(row["flexeva_events"])) <= 0:
            raise ValueError(f"Table 4 has no events: {row['case']}")
        if require_traces:
            trace_dir = Path(row["trace_dir"])
            for rank in range(int(row["world_size"])):
                for suffix in (".jsonl", "_markers.jsonl"):
                    trace = trace_dir / f"rank_{rank}{suffix}"
                    if not trace.is_file() or trace.stat().st_size == 0:
                        raise ValueError(f"missing generated trace: {trace}")


def validate_error(row: dict[str, str], label: str) -> None:
    predicted = float(row["predicted_runtime_us"])
    actual = float(row["actual_runtime_us"])
    error = float(row["oracle_error_pct"])
    if min(predicted, actual) <= 0.0:
        raise ValueError(f"Figure 5 runtime is non-positive: {label}")
    recomputed = 100.0 * abs(predicted - actual) / actual
    if not close(error, recomputed):
        raise ValueError(f"Figure 5 arithmetic differs: {label}")
    if not close(error, float(row["maya_error_pct"])) or not close(error, float(row["flexeva_error_pct"])):
        raise ValueError(f"Figure 5 evaluator feedback differs: {label}")


def validate_figure5(result_dir: Path) -> None:
    expected_gpt = (
        ("8", "native_gpu"),
        ("16", "native_gpu"),
        ("32", "large_cluster_trace"),
        ("64", "large_cluster_trace"),
        ("128", "large_cluster_trace"),
    )
    gpt = read_rows(result_dir / "figure5a.csv")
    if len(gpt) != len(expected_gpt):
        raise ValueError(f"Figure 5(a) requires five rows, found {len(gpt)}")
    for row, expected in zip(gpt, expected_gpt, strict=True):
        if (row["gpu_scale"], row["source_mode"]) != expected:
            raise ValueError(f"unexpected Figure 5(a) source: {row}")
        validate_error(row, f"GPT-{row['gpu_scale']}")
    if not all(float(row["oracle_error_pct"]) < 4.0 for row in gpt):
        raise ValueError("Figure 5(a) does not satisfy the <4% criterion")

    labels = ("Base MoE", "Intra 0-1", "Cross 0-8", "Cross 0-15", "Boundary 7-8")
    moe = read_rows(result_dir / "figure5b.csv")
    if len(moe) != len(labels):
        raise ValueError(f"Figure 5(b) requires five rows, found {len(moe)}")
    for row, label in zip(moe, labels, strict=True):
        if row["case"] != label or row["source_mode"] != "native_gpu" or int(row["world_size"]) != 16:
            raise ValueError(f"unexpected Figure 5(b) source: {row}")
        validate_error(row, label)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table4-result", type=Path)
    parser.add_argument("--require-table4-traces", action="store_true")
    parser.add_argument("--figure5-result-dir", type=Path)
    args = parser.parse_args()
    if args.require_table4_traces and args.table4_result is None:
        parser.error("--require-table4-traces requires --table4-result")
    if args.table4_result is None and args.figure5_result_dir is None:
        parser.error("select --table4-result and/or --figure5-result-dir")
    if args.table4_result is not None:
        validate_table4(args.table4_result, args.require_table4_traces)
    if args.figure5_result_dir is not None:
        validate_figure5(args.figure5_result_dir)
    print("E2 validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
