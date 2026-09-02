#!/usr/bin/env python3
"""Validate generated E2 tables and Figure 5 mode contracts."""

from __future__ import annotations

import argparse
import csv
import math
import re
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
TRACE_GPT = (
    ("8", "paper_run_record", "0.75", "4ac6907ebcd895f59c4823a493a91600252ef486c62fb4658d4ab642a2cfe439"),
    ("16", "paper_run_record", "3.66", "22bcde3e6393ab0f09a6c6f72c56b87d889ff2714279fef4311d4f07e9f7a74b"),
    ("32", "provided_trace", "1.24", "c6e65ca3bd4adc55815a69d7aa66fe6321655809253ef4e73e0bf37570de4069"),
    ("64", "provided_trace", "0.73", "11bed4032890063ca62285c43f7c5472d04b0812c978e90bd82993f5f213101b"),
    ("128", "provided_trace", "1.95", "35cb4cafe5acd0274459765489f825bfeaefd70a212ef0570ad2b2fcfec63dee"),
)
TRACE_MOE = (
    ("Base MoE", "7.7056", "26fc6597f58fd1c498b7c8a3fc86ad7e8324c6ce0beb9a48ec166490d8f69152"),
    ("Intra 0-1", "7.1199", "4fc56b407f3f399124ed020b886a5052045718a6a262e99669cc2fa79b114591"),
    ("Cross 0-8", "3.8771", "abac228eedbe0fd73519435d1e3c90a97628ee8962ad2928f3f63d6029ac19af"),
    ("Cross 0-15", "4.5084", "5ef3cde33b7b7dcd0dcce6cc5057ccede8bb2ad45b7a9a66615440e14d7e4283"),
    ("Boundary 7-8", "1.3687", "394f2ee5214ac8649462e766bf6031990e324f494dc05b8769a3e9235b06c05e"),
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


def validate_error(row: dict[str, str], label: str, mode: str) -> float:
    predicted = float(row["predicted_runtime_us"])
    actual = float(row["actual_runtime_us"])
    error = float(row["evaluator_error_pct"])
    if min(predicted, actual) <= 0.0:
        raise ValueError(f"Figure 5 runtime is non-positive: {label}")
    recomputed = 100.0 * abs(predicted - actual) / actual
    if not close(error, recomputed):
        raise ValueError(f"Figure 5 arithmetic differs: {label}")
    if row.get("shared_trace", "").lower() not in {"true", "1"}:
        raise ValueError(f"Figure 5 shared-trace contract is absent: {label}")
    if row.get("contract_mode") != mode:
        raise ValueError(f"Figure 5 contract mode differs: {label}")
    if mode == "trace":
        if re.fullmatch(r"[0-9a-f]{64}", row.get("source_sha256", "")) is None:
            raise ValueError(f"Figure 5 source provenance is absent: {label}")
    elif row.get("source_mode") != "native_gpu" or not row.get("trace_dir"):
        raise ValueError(f"Figure 5 native provenance is absent: {label}")
    return error


def validate_figure5(result_dir: Path, mode: str) -> None:
    gpt = read_rows(result_dir / "figure5a.csv")
    moe = read_rows(result_dir / "figure5b.csv")
    if mode == "trace":
        if len(gpt) != len(TRACE_GPT):
            raise ValueError(f"trace Figure 5(a) requires five rows, found {len(gpt)}")
        for row, (scale, source_mode, displayed, source_sha256) in zip(gpt, TRACE_GPT, strict=True):
            if (row["gpu_scale"], row["source_mode"]) != (scale, source_mode):
                raise ValueError(f"unexpected trace Figure 5(a) source: {row}")
            if row["source_sha256"] != source_sha256:
                raise ValueError(f"Figure 5(a) source fingerprint changed: GPT-{scale}")
            error = validate_error(row, f"GPT-{scale}", mode)
            if f"{error:.2f}" != displayed:
                raise ValueError(f"Figure 5(a) paper value changed: GPT-{scale}")
        if not all(float(row["evaluator_error_pct"]) < 4.0 for row in gpt):
            raise ValueError("Figure 5(a) does not satisfy the paper's <4% criterion")
        if len(moe) != len(TRACE_MOE):
            raise ValueError(f"trace Figure 5(b) requires five rows, found {len(moe)}")
        for row, (label, displayed, source_sha256) in zip(moe, TRACE_MOE, strict=True):
            if row["case"] != label or row["source_mode"] != "paper_run_record":
                raise ValueError(f"unexpected trace Figure 5(b) source: {row}")
            if row["source_sha256"] != source_sha256:
                raise ValueError(f"Figure 5(b) source fingerprint changed: {label}")
            error = validate_error(row, label, mode)
            if f"{error:.4f}" != displayed:
                raise ValueError(f"Figure 5(b) paper value changed: {label}")
        return

    if [row["gpu_scale"] for row in gpt] != ["8", "16"]:
        raise ValueError("native Figure 5(a) must contain only the 8- and 16-GPU points")
    for row in gpt:
        validate_error(row, f"GPT-{row['gpu_scale']}", mode)
    expected_labels = [label for label, *_ in TRACE_MOE]
    if [row["case"] for row in moe] != expected_labels:
        raise ValueError("native Figure 5(b) has the wrong MoE cases")
    for row in moe:
        if int(row["world_size"]) != 16:
            raise ValueError(f"native Figure 5(b) is not a 16-GPU run: {row['case']}")
        validate_error(row, row["case"], mode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table4-result", type=Path)
    parser.add_argument("--require-table4-traces", action="store_true")
    parser.add_argument("--figure5-result-dir", type=Path)
    parser.add_argument("--figure5-mode", choices=("trace", "native"), default="trace")
    args = parser.parse_args()
    if args.require_table4_traces and args.table4_result is None:
        parser.error("--require-table4-traces requires --table4-result")
    if args.table4_result is None and args.figure5_result_dir is None:
        parser.error("select --table4-result and/or --figure5-result-dir")
    if args.table4_result is not None:
        validate_table4(args.table4_result, args.require_table4_traces)
    if args.figure5_result_dir is not None:
        validate_figure5(args.figure5_result_dir, args.figure5_mode)
    print("E2 validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
