#!/usr/bin/env python3
"""Combine the fresh GPT and Routed-MoE Table 4 summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "result" / "e2"
OUTPUT = RESULT_DIR / "table4_from_trace.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty summary: {path}")
    return rows


def check_summary(rows: list[dict[str, str]], workload: str, trace_subdir: str) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        if int(row["return_code"]) != 0:
            raise ValueError(f"{workload} case failed: {row['case']}")
        case = row["case"]
        if int(row["world_size"]) <= 0 or int(row["raw_events"]) <= 0:
            raise ValueError(f"{workload} case has no trace events: {case}")
        trace_dir = Path("trace/e2/table4") / trace_subdir / case / "traces"
        if not (ROOT / trace_dir).is_dir():
            raise FileNotFoundError(f"missing trace directory for {case}: {ROOT / trace_dir}")
        normalized.append(
            {
                "workload": workload,
                "case": case,
                "world_size": row["world_size"],
                "configuration": (
                    f"{row['parameter_scale']}/TP{row['tp']}-PP{row['pp']}-DP{row['dp']}"
                    if workload == "GPT"
                    else row.get("route_label", "")
                ),
                "logical_event_coverage": row["logical_event_coverage"],
                "weighted_event_jaccard": row["weighted_event_jaccard"],
                "weighted_event_cosine": row["weighted_event_cosine"],
                "raw_events": row["raw_events"],
                "maya_events": row["maya_events"],
                "flexeva_events": row["flexeva_events"],
                "trace_dir": str(trace_dir),
            }
        )
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpt-summary",
        type=Path,
        default=RESULT_DIR / "generated_table4" / "gpt" / "summary.csv",
    )
    parser.add_argument(
        "--moe-summary",
        type=Path,
        default=RESULT_DIR / "generated_table4" / "routed-moe" / "summary.csv",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = check_summary(read_rows(args.gpt_summary), "GPT", "gpt")
    rows.extend(check_summary(read_rows(args.moe_summary), "Routed-MoE", "routed-moe"))
    if len(rows) != 8:
        raise ValueError(f"Table 4 requires eight generated rows, found {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
