#!/usr/bin/env python3
"""Create Tables 6 and 7 from the current E4 run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty E4 summary: {path}")
    return rows


def write(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_optimization(value: str) -> str:
    prefix = r"\texttt{"
    return value[len(prefix) : -1] if value.startswith(prefix) and value.endswith("}") else value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_root

    table6 = read(root / "table6" / "summary.csv")
    if len(table6) != 3:
        raise ValueError(f"Table 6 requires three rows, found {len(table6)}")
    write(
        root / "table6.csv",
        ("Workload", "Optimization", "Init metric", "Refresh speedup", "Reuse rate"),
        [
            {
                "Workload": row["workload"],
                "Optimization": clean_optimization(row["optimization"]),
                "Init metric": row["init_metric_tinit_over_tb"],
                "Refresh speedup": row["core_refresh_speedup"],
                "Reuse rate": 100.0 * float(row["reuse_rate"]),
            }
            for row in table6
        ],
    )

    table7 = read(root / "table7" / "gpt" / "summary.csv") + read(root / "table7" / "moe" / "summary.csv")
    if len(table7) != 6:
        raise ValueError(f"Table 7 requires six rows, found {len(table7)}")
    write(
        root / "table7.csv",
        ("Workload", "Mutation", "Reuse rate", "Speedup"),
        [
            {
                "Workload": "GPT" if row["workload"].startswith("GPT") else "Routed-MoE",
                "Mutation": row["mutation"],
                "Reuse rate": 100.0 * float(row["partition_reuse_rate"]),
                "Speedup": row["phase_speedup"],
            }
            for row in table7
        ],
    )
    print(f"wrote {root / 'table6.csv'} and {root / 'table7.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
