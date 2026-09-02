#!/usr/bin/env python3
"""Validate Tables 6 and 7 from the current E4 run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        result = list(csv.DictReader(stream))
    if not result:
        raise ValueError(f"empty E4 output: {path}")
    return result


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-9, abs_tol=1.0e-9)


def validate_table6(result_root: Path, trace_root: Path) -> None:
    generated = rows(result_root / "table6" / "summary.csv")
    if [row["workload"] for row in generated] != ["ResNet", "BERT", "ViT"]:
        raise ValueError("Table 6 workload order differs")
    case_names = {"ResNet": "resnet_compile", "BERT": "bert_fsdp", "ViT": "vit_checkpoint"}
    for row in generated:
        if row["status"] != "ok" or int(row["world_size"]) != 16:
            raise ValueError(f"Table 6 case failed: {row}")
        compact = int(row["candidate_compact_events"])
        selected = int(row["selected_events"])
        reuse = float(row["reuse_rate"])
        if compact <= 0 or not 0 <= selected <= compact or not close(1.0 - selected / compact, reuse):
            raise ValueError(f"Table 6 reuse accounting differs: {row['workload']}")
        if min(float(row["init_metric_tinit_over_tb"]), float(row["core_refresh_speedup"])) <= 0.0:
            raise ValueError(f"Table 6 metric is non-positive: {row['workload']}")
        for variant in ("anchor", "candidate"):
            trace_dir = trace_root / "table6" / case_names[row["workload"]] / variant / "traces"
            for rank in range(16):
                for suffix in (".jsonl", "_markers.jsonl"):
                    path = trace_dir / f"rank_{rank}{suffix}"
                    if not path.is_file() or path.stat().st_size == 0:
                        raise ValueError(f"missing Table 6 trace: {path}")

    analysis = json.loads((result_root / "table6" / "source_partition_analysis.json").read_text(encoding="utf-8"))
    by_workload = {row["workload"]: row for row in analysis["results"]}
    if set(by_workload) != set(case_names):
        raise ValueError("Table 6 source analysis workload set differs")
    for workload, row in by_workload.items():
        if not row["matches_recorded_plan"] or not row["inferred_partitions"] or not row["source_sites"]:
            raise ValueError(f"Table 6 source analysis failed: {workload}")

    table = rows(result_root / "table6.csv")
    if len(table) != 3:
        raise ValueError(f"Table 6 requires three rows, found {len(table)}")
    for output, source in zip(table, generated, strict=True):
        if output["Workload"] != source["workload"]:
            raise ValueError("Table 6 final workload order differs")
        if not close(float(output["Init metric"]), float(source["init_metric_tinit_over_tb"])):
            raise ValueError(f"Table 6 init metric differs: {output['Workload']}")
        if not close(float(output["Refresh speedup"]), float(source["core_refresh_speedup"])):
            raise ValueError(f"Table 6 speedup differs: {output['Workload']}")
        if not close(float(output["Reuse rate"]), 100.0 * float(source["reuse_rate"])):
            raise ValueError(f"Table 6 reuse rate differs: {output['Workload']}")


def validate_table7(result_root: Path) -> None:
    gpt = rows(result_root / "table7" / "gpt" / "summary.csv")
    moe = rows(result_root / "table7" / "moe" / "summary.csv")
    generated = gpt + moe
    expected_cases = (
        "gpt_attention_backward",
        "gpt_mlp_backward",
        "gpt_optimizer_step",
        "moe_optimizer_step",
        "moe_attention_backward",
        "moe_router_backward",
    )
    if tuple(row["case"] for row in generated) != expected_cases:
        raise ValueError("Table 7 case order differs")
    for row in generated:
        total = int(row["total_partitions"])
        reusable = int(row["reusable_partitions"])
        if total <= 0 or not 0 <= reusable <= total:
            raise ValueError(f"Table 7 partition accounting differs: {row['case']}")
        if not close(float(row["partition_reuse_rate"]), reusable / total):
            raise ValueError(f"Table 7 reuse arithmetic differs: {row['case']}")
        if min(float(row["full_phase_median_s"]), float(row["refresh_phase_median_s"]), float(row["phase_speedup"])) <= 0.0:
            raise ValueError(f"Table 7 timing is non-positive: {row['case']}")
        if not close(
            float(row["phase_speedup"]),
            float(row["full_phase_median_s"]) / float(row["refresh_phase_median_s"]),
        ):
            raise ValueError(f"Table 7 speedup arithmetic differs: {row['case']}")

    table = rows(result_root / "table7.csv")
    if len(table) != 6:
        raise ValueError(f"Table 7 requires six rows, found {len(table)}")
    for output, source in zip(table, generated, strict=True):
        if output["Mutation"] != source["mutation"]:
            raise ValueError(f"Table 7 mutation differs: {source['case']}")
        if not close(float(output["Reuse rate"]), 100.0 * float(source["partition_reuse_rate"])):
            raise ValueError(f"Table 7 reuse rate differs: {source['case']}")
        if not close(float(output["Speedup"]), float(source["phase_speedup"])):
            raise ValueError(f"Table 7 speedup differs: {source['case']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    args = parser.parse_args()
    validate_table6(args.result_root.resolve(), args.trace_root.resolve())
    validate_table7(args.result_root.resolve())
    print("E4 validation: PASS (Tables 6 and 7 from current run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
