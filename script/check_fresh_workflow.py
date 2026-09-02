#!/usr/bin/env python3
"""Reject workflows that consume generated results as experiment inputs."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_CSV_INPUTS = {"large-cluster/e1/trajectory.csv"}


def tracked_result_files() -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files", "--", "result"],
        text=True,
    )
    return [line for line in output.splitlines() if line and (ROOT / line).exists()]


def tracked_csv_files() -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files", "--", "*.csv"],
        text=True,
    )
    return [line for line in output.splitlines() if line and (ROOT / line).exists()]


def reject(path: str, fragments: tuple[str, ...]) -> list[str]:
    text = (ROOT / path).read_text(encoding="utf-8")
    return [f"{path}: {fragment}" for fragment in fragments if fragment in text]


def main() -> int:
    errors = [f"tracked experiment output: {path}" for path in tracked_result_files()]
    errors.extend(f"tracked CSV: {path}" for path in tracked_csv_files() if path not in ALLOWED_CSV_INPUTS)
    if not (ROOT / "large-cluster/e1/trajectory.csv").is_file():
        errors.append("missing historical E1 ledger")
    errors.extend(
        reject(
            "script/run_all",
            (
                '"E1: Figure 1"',
                '"E2: Table 4"',
                '"E3: Figure 6"',
                '"E4: Tables 6 and 7"',
                '"E5: Table 8 and per-round speedup"',
                "--check-only",
            ),
        )
    )
    errors.extend(
        reject(
            "script/e1/derive_results.py",
            ('ROOT / "result" / "e1" / "trajectory.csv"',),
        )
    )
    errors.extend(
        reject(
            "script/run_e1",
            ("capture_emulated.py", "FLEXSIM_MAYA_SAFE_ROUTING", "E1_ROUTING_TRACE_ROOT"),
        )
    )
    errors.extend(
        reject(
            "script/e2/validate_results.py",
            (
                'RESULT_DIR / "table4.csv"',
                'RESULT_DIR / "figure5a.csv"',
                'RESULT_DIR / "figure5b.csv"',
            ),
        )
    )
    errors.extend(
        reject(
            "script/e3/validate_results.py",
            ("validate_paper_ledgers()", "validate_figure8_paper_ledgers()"),
        )
    )
    errors.extend(
        reject(
            "script/e4/validate_results.py",
            ("validate_paper_ledgers()",),
        )
    )
    for forbidden in (
        "script/e1/agent_interactions",
        "script/e4/backend/bundle/inputs/table7_anchor_20260901",
        "script/e5/bundle/paper/table8-expected.json",
    ):
        path = ROOT / forbidden
        if path.is_file() or (path.is_dir() and any(item.is_file() for item in path.rglob("*"))):
            errors.append(f"preloaded experiment data: {forbidden}")
    run_all = (ROOT / "script/run_all").read_text(encoding="utf-8")
    for runner in range(1, 6):
        token = f'"$ROOT/script/run_e{runner}"'
        if run_all.count(token) != 1:
            errors.append(f"script/run_all must call run_e{runner} exactly once")
    if errors:
        print("fresh workflow contract: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("fresh workflow contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
