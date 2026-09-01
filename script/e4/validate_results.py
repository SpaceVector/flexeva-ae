#!/usr/bin/env python3
"""Validate E4 paper ledgers and an optional fresh Table 6 run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "result" / "e4"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        result = list(csv.DictReader(stream))
    if not result:
        raise ValueError(f"empty E4 ledger: {path}")
    return result


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-9, abs_tol=1.0e-9)


def validate_paper_ledgers() -> None:
    table6_expected = (
        ("ResNet", "torch.compile", "0.778", "3.62", "79.15"),
        ("BERT", "FSDP sharding", "0.861", "4.82", "91.95"),
        ("ViT", "activation checkpointing", "0.862", "4.38", "89.12"),
    )
    table6 = rows(RESULT_DIR / "table6.csv")
    if len(table6) != len(table6_expected):
        raise ValueError(f"Table 6 requires {len(table6_expected)} rows, found {len(table6)}")
    for row, expected in zip(table6, table6_expected, strict=True):
        actual = (
            row["Workload"],
            row["Optimization"],
            f"{float(row['Init metric']):.3f}",
            f"{float(row['Refresh speedup']):.2f}",
            f"{float(row['Reuse rate']):.2f}",
        )
        if actual != expected:
            raise ValueError(f"Table 6 paper row changed: {actual}")

    table7_expected = (
        ("GPT", "attention backward", "84.77", "6.57"),
        ("GPT", "MLP backward", "84.14", "6.31"),
        ("GPT", "optimizer step", "84.72", "6.54"),
        ("Routed-MoE", "optimizer step", "67.03", "3.03"),
        ("Routed-MoE", "attention backward", "66.99", "3.03"),
        ("Routed-MoE", "router backward", "66.95", "3.03"),
    )
    table7 = rows(RESULT_DIR / "table7.csv")
    if len(table7) != len(table7_expected):
        raise ValueError(f"Table 7 requires {len(table7_expected)} rows, found {len(table7)}")
    for row, expected in zip(table7, table7_expected, strict=True):
        actual = (
            row["Workload"],
            row["Mutation"],
            f"{float(row['Reuse rate']):.2f}",
            f"{float(row['Speedup']):.2f}",
        )
        if actual != expected:
            raise ValueError(f"Table 7 paper row changed: {actual}")


def validate_generated_table6() -> None:
    summary_path = RESULT_DIR / "generated" / "table6" / "summary.csv"
    generated = rows(summary_path)
    if [row["workload"] for row in generated] != ["ResNet", "BERT", "ViT"]:
        raise ValueError("generated Table 6 workload order differs")
    case_names = {"ResNet": "resnet_compile", "BERT": "bert_fsdp", "ViT": "vit_checkpoint"}
    for row in generated:
        if row["status"] != "ok" or int(row["world_size"]) != 16:
            raise ValueError(f"generated Table 6 case failed: {row}")
        compact = int(row["candidate_compact_events"])
        selected = int(row["selected_events"])
        reuse = float(row["reuse_rate"])
        if compact <= 0 or not 0 <= selected <= compact or not close(1.0 - selected / compact, reuse):
            raise ValueError(f"generated Table 6 reuse accounting differs: {row['workload']}")
        if min(float(row["init_metric_tinit_over_tb"]), float(row["core_refresh_speedup"])) <= 0.0:
            raise ValueError(f"generated Table 6 metric is non-positive: {row['workload']}")
        for variant in ("anchor", "candidate"):
            trace_dir = ROOT / "trace" / "e4" / "table6" / case_names[row["workload"]] / variant / "traces"
            for rank in range(16):
                for suffix in (".jsonl", "_markers.jsonl"):
                    path = trace_dir / f"rank_{rank}{suffix}"
                    if not path.is_file() or path.stat().st_size == 0:
                        raise ValueError(f"missing generated Table 6 trace: {path}")

    analysis_path = RESULT_DIR / "generated" / "table6" / "source_partition_analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    rows_by_workload = {row["workload"]: row for row in analysis["results"]}
    if set(rows_by_workload) != set(case_names):
        raise ValueError("generated Table 6 source analysis workload set differs")
    for workload, row in rows_by_workload.items():
        if not row["matches_recorded_plan"] or not row["inferred_partitions"] or not row["source_sites"]:
            raise ValueError(f"generated Table 6 source analysis failed: {workload}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-generated", action="store_true", help="also require fresh Table 6 traces")
    args = parser.parse_args()
    validate_paper_ledgers()
    if args.require_generated:
        validate_generated_table6()
    suffix = "; generated Table 6 trace" if args.require_generated else ""
    print(f"E4 validation: PASS (Table 6: 3 rows; Table 7: 6 rows{suffix})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
