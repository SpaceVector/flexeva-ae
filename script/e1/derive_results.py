#!/usr/bin/env python3
"""Validate the historical E1 ledger and derive Figure 1(b/c) CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "large-cluster" / "e1" / "trajectory.csv"
EXPECTED_STAGES = ("Base", "S1", "S2", "S3", "S4")
METRICS = {
    "time": "step_time_s",
    "a2a": "estimated_a2a_bytes",
    "drop": "tokens_dropped",
    "reroute": "tokens_rerouted",
}


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate(rows: list[dict[str, str]]) -> None:
    if len(rows) != 5:
        raise ValueError(f"E1 requires five ledger rows, found {len(rows)}")

    baseline = rows[0]
    for index, (row, stage) in enumerate(zip(rows, EXPECTED_STAGES, strict=True)):
        if (int(row["round"]), row["stage"]) != (index, stage):
            raise ValueError(f"unexpected E1 row order at {stage}")
        if (int(row["world_size"]), int(row["seed"]), int(row["benchmark_samples"])) != (128, 1234, 1):
            raise ValueError(f"E1 execution contract differs at {stage}")

        added = int(row["added_lines"])
        deleted = int(row["deleted_lines"])
        if int(row["total_changed_lines"]) != added + deleted:
            raise ValueError(f"E1 patch accounting differs at {stage}")
        if Path(row["result_source_file"]).is_absolute():
            raise ValueError(f"E1 result source must be relative at {stage}")
        for prefix, field in METRICS.items():
            value = float(row[field])
            base = float(baseline[field])
            if value < 0.0 or base <= 0.0:
                raise ValueError(f"invalid E1 {field} at {stage}")
            reduction = (base - value) / base
            normalized = 1.0 + reduction
            if not close(reduction, float(row[f"{prefix}_reduction_vs_baseline"])):
                raise ValueError(f"E1 {prefix} reduction differs at {stage}")
            if not close(normalized, float(row[f"{prefix}_normalized_improvement"])):
                raise ValueError(f"E1 {prefix} normalization differs at {stage}")

    if float(rows[2]["time_reduction_vs_baseline"]) >= 0.0:
        raise ValueError("E1 must preserve the failed S2 round")
    if f"{100.0 * float(rows[-1]['time_reduction_vs_baseline']):.2f}" != "61.30":
        raise ValueError("E1 final time reduction must be 61.30%")
    if f"{100.0 * float(rows[-1]['a2a_reduction_vs_baseline']):.2f}" != "95.03":
        raise ValueError("E1 final A2A reduction must be 95.03%")


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def derive(rows: list[dict[str, str]], result_dir: Path) -> None:
    write_csv(
        result_dir / "figure1b.csv",
        ("Optimization Stage", "Total", "Added", "Deleted"),
        [
            {
                "Optimization Stage": row["stage"],
                "Total": row["total_changed_lines"],
                "Added": row["added_lines"],
                "Deleted": row["deleted_lines"],
            }
            for row in rows
        ],
    )
    write_csv(
        result_dir / "figure1c.csv",
        ("Optimization Stage", "Time", "A2A", "Drop", "Reroute"),
        [
            {
                "Optimization Stage": row["stage"],
                "Time": row["time_normalized_improvement"],
                "A2A": row["a2a_normalized_improvement"],
                "Drop": row["drop_normalized_improvement"],
                "Reroute": row["reroute_normalized_improvement"],
            }
            for row in rows
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.input)
    validate(rows)
    if not args.check_only:
        if args.result_dir is None:
            parser.error("--result-dir is required unless --check-only is used")
        derive(rows, args.result_dir)
    print(f"E1 historical ledger: PASS ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
