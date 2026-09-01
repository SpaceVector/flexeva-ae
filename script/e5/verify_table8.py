#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "result" / "e5" / "submitted" / "memory_scaling.json"
EXPECTED_PATH = Path(__file__).resolve().parent / "bundle" / "paper" / "table8-expected.json"
MODE_ORDER = ("maya_full", "maya_trace_ras", "flexeva_selected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render and verify E5 Table 8 from memory_scaling.json")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--require-paper-values",
        action="store_true",
        help="fail unless the input rounds to every submitted Table 8 value",
    )
    parser.add_argument("--write-derived-json", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def derive(payload: dict) -> dict:
    config = payload["configuration"]
    required_config = {
        "world_size": 16,
        "ep_group_size": 8,
        "micro_batches": 64,
        "layers": 64,
        "seq_len": 256,
        "hidden_size": 512,
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in required_config.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"paper-shape configuration mismatch: {mismatches}")

    counts = (1, 2, 4, 8, 16, 32)
    by_mode_count = {
        (str(row["mode"]), int(row["candidate_count"])): row
        for row in payload["rows"]
    }
    expected_keys = {(mode, count) for mode in MODE_ORDER for count in counts}
    if len(payload["rows"]) != len(by_mode_count) or set(by_mode_count) != expected_keys:
        raise ValueError("expected one measurement for each of 3 modes x 6 candidate counts")

    for key, row in by_mode_count.items():
        retained = float(row["retained_rss_delta_mb"])
        peak = float(row["peak_rss_delta_mb"])
        current_delta = float(row["current_rss_mb"]) - float(row["baseline_rss_mb"])
        if retained <= 0.0 or peak < retained or not math.isclose(current_delta, retained, abs_tol=1.0e-9):
            raise ValueError(f"invalid RSS accounting for {key}")
    for mode in MODE_ORDER:
        retained = [float(by_mode_count[(mode, count)]["retained_rss_delta_mb"]) for count in counts]
        if retained != sorted(retained) or retained[0] == retained[-1]:
            raise ValueError(f"retained RSS does not grow with candidate count for {mode}")

    rows = {}
    for mode in MODE_ORDER:
        rss = {
            count: float(by_mode_count[(mode, count)]["retained_rss_delta_mb"])
            for count in (1, 8, 32)
        }
        rows[mode] = {
            "source_retained_rss_delta_mb": {str(count): rss[count] for count in (1, 8, 32)},
            "paper_values": {
                "k1_gib": round(rss[1] / 1024.0, 2),
                "k8_gib": round(rss[8] / 1024.0, 2),
                "k32_gib": round(rss[32] / 1024.0, 2),
                "marginal_mb_per_candidate": round((rss[32] - rss[1]) / 31.0, 2),
            },
        }

    flex_k32 = rows["flexeva_selected"]["source_retained_rss_delta_mb"]["32"]
    maya_k32 = rows["maya_full"]["source_retained_rss_delta_mb"]["32"]
    trace_ras_k32 = rows["maya_trace_ras"]["source_retained_rss_delta_mb"]["32"]
    flex_slope = rows["flexeva_selected"]["paper_values"]["marginal_mb_per_candidate"]
    if not (flex_k32 < maya_k32 and flex_k32 < trace_ras_k32):
        raise ValueError("fresh K=32 result does not preserve the submitted memory ordering")
    if not all(
        flex_slope < rows[mode]["paper_values"]["marginal_mb_per_candidate"]
        for mode in ("maya_full", "maya_trace_ras")
    ):
        raise ValueError("fresh result does not preserve the submitted marginal-memory ordering")

    return {
        "input": str(payload.get("source", "memory_scaling.json")),
        "configuration": config,
        "configuration_mismatches": mismatches,
        "source_field": "retained_rss_delta_mb",
        "rows": rows,
        "k32_reductions": {
            "vs_maya_full_pct": round((1.0 - flex_k32 / maya_k32) * 100.0, 1),
            "vs_maya_trace_ras_pct": round((1.0 - flex_k32 / trace_ras_k32) * 100.0, 1),
        },
    }


def compare_expected(derived: dict, expected: dict) -> list[str]:
    errors = []
    for mode in MODE_ORDER:
        actual = derived["rows"][mode]["paper_values"]
        target = expected["rows"][mode]
        for field in ("k1_gib", "k8_gib", "k32_gib", "marginal_mb_per_candidate"):
            if actual[field] != target[field]:
                errors.append(f"{mode}.{field}: expected {target[field]}, got {actual[field]}")
    for field, target in expected["k32_reductions"].items():
        actual = derived["k32_reductions"][field]
        if actual != target:
            errors.append(f"k32_reductions.{field}: expected {target}, got {actual}")
    return errors


def print_table(derived: dict, expected: dict) -> None:
    print("| Evaluator | K=1 | K=8 | K=32 | Marginal delta RSS/candidate (K=1 to K=32) |")
    print("|---|---:|---:|---:|---:|")
    for mode in MODE_ORDER:
        values = derived["rows"][mode]["paper_values"]
        label = expected["rows"][mode]["label"]
        print(
            f"| {label} | {values['k1_gib']:.2f} GiB | {values['k8_gib']:.2f} GiB | "
            f"{values['k32_gib']:.2f} GiB | {values['marginal_mb_per_candidate']:.2f} MiB |"
        )
    reductions = derived["k32_reductions"]
    print()
    print(
        "K=32 retained-RSS reduction: "
        f"{reductions['vs_maya_full_pct']:.1f}% vs Maya-full; "
        f"{reductions['vs_maya_trace_ras_pct']:.1f}% vs Maya-trace-RAS."
    )
    print("Source field: retained_rss_delta_mb (not peak_rss_delta_mb).")


def main() -> int:
    args = parse_args()
    derived = derive(load_json(args.input))
    expected = load_json(EXPECTED_PATH)
    print_table(derived, expected)
    if args.write_derived_json is not None:
        args.write_derived_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_derived_json.write_text(
            json.dumps(derived, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    errors = compare_expected(derived, expected)
    must_match = args.require_paper_values or args.input.resolve() == DEFAULT_INPUT.resolve()
    if errors and must_match:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if errors:
        print("NOTE: fresh values differ from the allocator-sensitive archived values.")
        print("PASS: fresh measurements preserve the Table 8 contract and memory ordering.")
        return 0
    print("PASS: selected measurements reproduce all displayed Table 8 values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
