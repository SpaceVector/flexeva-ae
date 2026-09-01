#!/usr/bin/env python3
"""Validate E2 reference ledgers and freshly generated experiment results."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "result" / "e2"
GENERATED_TABLE4 = RESULT_DIR / "table4_from_trace.csv"
GENERATED_FIGURE5 = RESULT_DIR / "generated_figure5"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty E2 ledger: {path}")
    return rows


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-9, abs_tol=1.0e-9)


def validate_table4() -> None:
    expected = (
        ("GPT", "megatron_2p7b_8gpu", 8, "1.000000"),
        ("GPT", "megatron_18p4b_16gpu", 16, "0.999992"),
        ("GPT", "megatron_2p7b_16gpu_dp2", 16, "0.999951"),
        ("GPT", "megatron_18p4b_16gpu_dp2", 16, "0.999984"),
        ("Routed-MoE", "routed_moe_intra_group_0_1", 16, "1.000000"),
        ("Routed-MoE", "routed_moe_cross_group_0_8", 16, "1.000000"),
        ("Routed-MoE", "routed_moe_cross_group_0_15", 16, "1.000000"),
        ("Routed-MoE", "routed_moe_boundary_7_8", 16, "1.000000"),
    )
    rows = read_rows(RESULT_DIR / "table4.csv")
    if len(rows) != len(expected):
        raise ValueError(f"Table 4 requires {len(expected)} rows, found {len(rows)}")
    for row, (workload, case, world_size, paper_jaccard) in zip(rows, expected, strict=True):
        if (row["workload"], row["case"], int(row["world_size"])) != (workload, case, world_size):
            raise ValueError(f"unexpected Table 4 case: {row}")
        coverage = float(row["logical_event_coverage"])
        jaccard = float(row["weighted_event_jaccard"])
        threshold = 0.9999 if workload == "GPT" else 1.0
        if coverage < threshold or jaccard < threshold:
            raise ValueError(f"Table 4 similarity below threshold: {case}")
        if f"{jaccard:.6f}" != paper_jaccard:
            raise ValueError(f"Table 4 paper value changed: {case}")


def validate_error_ledger(path: Path, key: str, expected: tuple[tuple[object, str], ...], precision: int) -> None:
    rows = read_rows(path)
    if len(rows) != len(expected):
        raise ValueError(f"{path.name} requires {len(expected)} rows, found {len(rows)}")
    for row, (expected_key, expected_display) in zip(rows, expected, strict=True):
        if row[key] != str(expected_key):
            raise ValueError(f"unexpected {path.name} row: {row[key]}")
        predicted = float(row["predicted_runtime_us"])
        actual = float(row["actual_runtime_us"])
        if min(predicted, actual) <= 0.0:
            raise ValueError(f"non-positive runtime: {row[key]}")
        recomputed = 100.0 * abs(predicted - actual) / actual
        stored = float(row["oracle_error_pct"])
        if not close(recomputed, stored):
            raise ValueError(f"error arithmetic differs: {row[key]}")
        if not close(stored, float(row["maya_error_pct"])) or not close(stored, float(row["flexeva_error_pct"])):
            raise ValueError(f"evaluator feedback differs: {row[key]}")
        if f"{stored:.{precision}f}" != expected_display or f"{recomputed:.{precision}f}" != expected_display:
            raise ValueError(f"paper rounding differs: {row[key]}")


def validate_generated_table4() -> None:
    rows = read_rows(GENERATED_TABLE4)
    if len(rows) != 8:
        raise ValueError(f"generated Table 4 requires eight rows, found {len(rows)}")
    for row in rows:
        coverage = float(row["logical_event_coverage"])
        jaccard = float(row["weighted_event_jaccard"])
        if min(coverage, jaccard) < 0.9999:
            raise ValueError(f"generated Table 4 similarity below threshold: {row['case']}")
        if int(row["raw_events"]) <= 0 or int(row["maya_events"]) <= 0 or int(row["flexeva_events"]) <= 0:
            raise ValueError(f"generated Table 4 has no events: {row['case']}")
        trace_dir = ROOT / row["trace_dir"]
        world_size = int(row["world_size"])
        for rank in range(world_size):
            for suffix in (".jsonl", "_markers.jsonl"):
                path = trace_dir / f"rank_{rank}{suffix}"
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"missing generated trace file: {path}")


def validate_generated_figure5(result_dir: Path) -> None:
    expected_gpt = (
        ("8", "native_gpu"),
        ("16", "native_gpu"),
        ("32", "large_cluster_trace"),
        ("64", "large_cluster_trace"),
        ("128", "large_cluster_trace"),
    )
    gpt_rows = read_rows(result_dir / "figure5a.csv")
    if len(gpt_rows) != len(expected_gpt):
        raise ValueError(f"generated Figure 5(a) requires five rows, found {len(gpt_rows)}")
    for row, (scale, source_mode) in zip(gpt_rows, expected_gpt, strict=True):
        if (row["gpu_scale"], row["source_mode"]) != (scale, source_mode):
            raise ValueError(f"unexpected generated Figure 5(a) source: {row}")
        validate_generated_error(row, f"GPT-{scale}")
    if not all(float(row["oracle_error_pct"]) < 4.0 for row in gpt_rows):
        raise ValueError("generated Figure 5(a) does not satisfy the <4% criterion")

    expected_moe = ("Base MoE", "Intra 0-1", "Cross 0-8", "Cross 0-15", "Boundary 7-8")
    moe_rows = read_rows(result_dir / "figure5b.csv")
    if len(moe_rows) != len(expected_moe):
        raise ValueError(f"generated Figure 5(b) requires five rows, found {len(moe_rows)}")
    for row, case in zip(moe_rows, expected_moe, strict=True):
        if row["case"] != case or row["source_mode"] != "native_gpu" or int(row["world_size"]) != 16:
            raise ValueError(f"unexpected generated Figure 5(b) source: {row}")
        validate_generated_error(row, case)


def validate_generated_error(row: dict[str, str], label: str) -> None:
    predicted = float(row["predicted_runtime_us"])
    actual = float(row["actual_runtime_us"])
    error = float(row["oracle_error_pct"])
    if min(predicted, actual) <= 0.0:
        raise ValueError(f"generated Figure 5 runtime is non-positive: {label}")
    recomputed = 100.0 * abs(predicted - actual) / actual
    if not close(error, recomputed):
        raise ValueError(f"generated Figure 5 arithmetic differs: {label}")
    if not close(error, float(row["maya_error_pct"])) or not close(error, float(row["flexeva_error_pct"])):
        raise ValueError(f"generated evaluator feedback differs: {label}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="validate without producing outputs")
    parser.add_argument("--require-traces", action="store_true", help="require generated Table 4 traces")
    parser.add_argument("--table4-only", action="store_true", help="skip Figure 5 validation")
    parser.add_argument("--require-figure5", action="store_true", help="require generated Figure 5 data")
    parser.add_argument("--figure5-only", action="store_true", help="skip Table 4 validation")
    parser.add_argument("--figure5-result-dir", type=Path, default=GENERATED_FIGURE5)
    args = parser.parse_args()
    if args.table4_only and (args.require_figure5 or args.figure5_only):
        parser.error("--table4-only cannot be combined with Figure 5 options")
    if args.figure5_only and args.require_traces:
        parser.error("--figure5-only cannot be combined with --require-traces")
    if not args.figure5_only:
        validate_table4()
        if args.require_traces:
            validate_generated_table4()
    if args.table4_only:
        figure5_source = "Figure 5 skipped"
    elif args.require_figure5:
        validate_generated_figure5(args.figure5_result_dir)
        figure5_source = "generated Figure 5"
    else:
        validate_error_ledger(
            RESULT_DIR / "figure5a.csv",
            "gpu_scale",
            ((8, "0.75"), (16, "3.66"), (32, "1.24"), (64, "0.73"), (128, "1.95")),
            2,
        )
        rows = read_rows(RESULT_DIR / "figure5a.csv")
        if not all(float(row["oracle_error_pct"]) < 4.0 for row in rows):
            raise ValueError("Figure 5(a) does not satisfy the <4% criterion")
        validate_error_ledger(
            RESULT_DIR / "figure5b.csv",
            "case",
            (
                ("Base MoE", "7.7056"),
                ("Intra 0-1", "7.1199"),
                ("Cross 0-8", "3.8771"),
                ("Cross 0-15", "4.5084"),
                ("Boundary 7-8", "1.3687"),
            ),
            4,
        )
        figure5_source = "reference Figure 5"
    table4_source = "Table 4 skipped" if args.figure5_only else "8 Table 4 rows"
    suffix = ", fresh Table 4 traces" if args.require_traces else ""
    print(f"E2 validation: PASS ({table4_source}, {figure5_source}{suffix})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
