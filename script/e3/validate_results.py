#!/usr/bin/env python3
"""Validate freshly generated E3 figure data and plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIGURE8_SYSTEMS = ("Maya-full", "Maya-trace-RAS", "FlexEva refresh")
FIGURE8_CASES = {
    "2.7B/8GPU TP1-PP8-DP1": ("megatron_2p7b_8gpu", 8, 1, 8, 1),
    "2.7B/16GPU TP1-PP8-DP2": ("megatron_2p7b_16gpu_dp2", 16, 1, 8, 2),
    "18.4B/16GPU TP2-PP8-DP1": ("megatron_18p4b_16gpu", 16, 2, 8, 1),
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        result = list(csv.DictReader(stream))
    if not result:
        raise ValueError(f"empty E3 output: {path}")
    return result


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-9, abs_tol=1.0e-9)


def validate_trajectory(label: str, maya: list[float], flexeva: list[float]) -> None:
    if not all(right > left for left, right in zip(maya, maya[1:])):
        raise ValueError(f"Figure 6 {label} Maya trajectory is not cumulative")
    if not all(right > left for left, right in zip(flexeva, flexeva[1:])):
        raise ValueError(f"Figure 6 {label} FlexEva trajectory is not cumulative")
    if any(flexeva[index] >= maya[index] for index in range(1, 4)):
        raise ValueError(f"Figure 6 {label} efficiency trend differs")


def validate_capture(capture: dict[str, object], label: str) -> None:
    run = capture["run"]
    if int(run["return_code"]) != 0:
        raise ValueError(f"failed Figure 6 capture: {label}")
    distributed = run["distributed"]
    if (int(distributed["nnodes"]), int(distributed["nproc_per_node"])) != (2, 8):
        raise ValueError(f"Figure 6 capture is not 2x8 ranks: {label}")
    if int(run["peer_trace_transfer"]["files"]) != 16:
        raise ValueError(f"Figure 6 peer traces are incomplete: {label}")
    if capture["api_audit"] != {"cudaGetDevice_modeled_count": 0, "cudaGetDevice_replay_count": 0}:
        raise ValueError(f"Figure 6 models cudaGetDevice: {label}")
    if capture["maya_full"]["feedback"]["cycle_detected"]:
        raise ValueError(f"Figure 6 replay cycle: {label}")
    trace_dir = Path(run["trace_dir"])
    for rank in range(16):
        for suffix in (".jsonl", "_markers.jsonl"):
            path = trace_dir / f"rank_{rank}{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing Figure 6 trace: {path}")


def validate_figure6(generated: Path) -> None:
    summary = json.loads((generated / "result.json").read_text(encoding="utf-8"))
    if summary["execution"]["nodes"] != 2 or summary["execution"]["physical_a100s"] != 16:
        raise ValueError("Figure 6 physical execution contract differs")
    gpt = json.loads((generated / "gpt" / "breakdown" / "e2e_result.json").read_text(encoding="utf-8"))
    moe = json.loads((generated / "moe" / "result.json").read_text(encoding="utf-8"))
    for panel, payload in (("gpt", gpt), ("moe", moe)):
        captures = [payload["anchor"], *payload["candidates"]]
        if len(captures) != 5:
            raise ValueError(f"Figure 6 {panel} requires an anchor and four candidates")
        for index, capture in enumerate(captures):
            validate_capture(capture, f"{panel}-{index}")
        for candidate in payload["candidates"]:
            refresh = candidate["flexeva_refresh"]
            if int(refresh["source_analysis_count"]) != 1 or refresh["feedback"]["cycle_detected"]:
                raise ValueError(f"Figure 6 {panel} source-analysis/replay contract differs")

    figure = rows(generated / "figure6.csv")
    phases = rows(generated / "phase_breakdown.csv")
    if len(figure) != 8 or len(phases) != 16:
        raise ValueError("Figure 6 generated row count differs")
    phase_by_key = {(row["panel"], int(row["round"]), row["system"]): row for row in phases}
    for panel in ("gpt", "moe"):
        panel_rows = sorted((row for row in figure if row["panel"] == panel), key=lambda row: int(row["round"]))
        if [int(row["round"]) for row in panel_rows] != [1, 2, 3, 4]:
            raise ValueError(f"Figure 6 {panel} rounds differ")
        validate_trajectory(
            panel,
            [float(row["maya_normalized"]) for row in panel_rows],
            [float(row["flexeva_normalized"]) for row in panel_rows],
        )
    for row in figure:
        key = row["panel"], int(row["round"])
        full = phase_by_key[(*key, "Maya-full")]
        flexeva = phase_by_key[(*key, "FlexEva")]
        base = float(row["normalization_base_s"])
        if not close(float(full["total_s"]), float(row["maya_full_cumulative_s"])):
            raise ValueError(f"Figure 6 Maya cumulative mismatch: {key}")
        if not close(float(flexeva["total_s"]), float(row["flexeva_cumulative_s"])):
            raise ValueError(f"Figure 6 FlexEva cumulative mismatch: {key}")
        if not close(float(row["maya_normalized"]), float(row["maya_full_cumulative_s"]) / base):
            raise ValueError(f"Figure 6 Maya normalization mismatch: {key}")
        if not close(float(row["flexeva_normalized"]), float(row["flexeva_cumulative_s"]) / base):
            raise ValueError(f"Figure 6 FlexEva normalization mismatch: {key}")
    for name in ("figure6a.pdf", "figure6b.pdf"):
        plot = ROOT / "plot" / name
        if not plot.is_file() or plot.stat().st_size == 0:
            raise ValueError(f"missing Figure 6 plot: {plot}")


def figure8_csv(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    current = rows(path)
    labels = list(dict.fromkeys(row["label"] for row in current))
    indexed = {(row["label"], row["system"]): row for row in current}
    expected = {(label, system) for label in FIGURE8_CASES for system in FIGURE8_SYSTEMS}
    if labels != list(FIGURE8_CASES) or set(indexed) != expected:
        raise ValueError(f"Figure 8 case/system set differs: {path}")
    return indexed


def validate_figure8_capture(row: dict[str, object]) -> None:
    label = str(row["label"])
    expected = FIGURE8_CASES[label]
    case = row["case"]
    if (case["name"], case["world_size"], case["tp"], case["pp"], case["dp"]) != expected:
        raise ValueError(f"Figure 8 case geometry differs: {label}")
    world_size = expected[1]
    if int(row["physical_gpu_count"]) != world_size:
        raise ValueError(f"Figure 8 physical GPU count differs: {label}")
    run = row["run"]
    launch = (1, 8) if world_size == 8 else (2, 8)
    distributed = run["distributed"]
    if int(run["return_code"]) != 0 or (int(distributed["nnodes"]), int(distributed["nproc_per_node"])) != launch:
        raise ValueError(f"Figure 8 capture failed or used the wrong launch: {label}")
    if int(run["peer_trace_transfer"]["files"]) != (0 if world_size == 8 else 16):
        raise ValueError(f"Figure 8 peer trace transfer differs: {label}")
    if float(row["step_begin_skew_s"]) > 1.0:
        raise ValueError(f"Figure 8 rank synchronization differs: {label}")
    if row["flexeva_refresh"]["plan"]["changed_partitions"] != ["attention_backward"]:
        raise ValueError(f"Figure 8 mutation scope differs: {label}")
    if any(section["feedback"]["cycle_detected"] for section in (row["maya_full"], row["maya_trace_ras"], row["flexeva_refresh"])):
        raise ValueError(f"Figure 8 replay cycle detected: {label}")
    trace_dir = Path(run["trace_dir"])
    for rank in range(world_size):
        for suffix in (".jsonl", "_markers.jsonl"):
            path = trace_dir / f"rank_{rank}{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing Figure 8 trace: {path}")
    metrics = row["metrics"]
    total = int(metrics["total_trace_events"])
    regenerated = int(metrics["regenerated_trace_events"])
    if not 0 < regenerated <= total or not close(regenerated / total, float(metrics["refresh_ratio"])):
        raise ValueError(f"Figure 8 refresh accounting differs: {label}")
    if min(float(metrics[name]) for name in ("maya_full_s", "maya_trace_ras_s", "flexeva_refresh_s")) <= 0.0:
        raise ValueError(f"Figure 8 timing is non-positive: {label}")
    if float(metrics["flexeva_refresh_s"]) >= float(metrics["maya_full_s"]):
        raise ValueError(f"Figure 8 FlexEva trend differs: {label}")


def validate_figure8(generated: Path) -> None:
    payload = json.loads((generated / "result.json").read_text(encoding="utf-8"))
    if payload["method"]["mutation"] != "attention_backward" or int(payload["method"]["sample_count"]) != 1:
        raise ValueError("Figure 8 method differs")
    by_label = {str(row["label"]): row for row in payload["results"]}
    if set(by_label) != set(FIGURE8_CASES):
        raise ValueError("Figure 8 generated cases differ")
    scale = figure8_csv(generated / "figure8a.csv")
    trace = figure8_csv(generated / "figure8b.csv")
    for label, row in by_label.items():
        validate_figure8_capture(row)
        metrics = row["metrics"]
        for system, metric in zip(
            FIGURE8_SYSTEMS,
            ("maya_full_s", "maya_trace_ras_s", "flexeva_refresh_s"),
            strict=True,
        ):
            key = label, system
            seconds = float(metrics[metric])
            if not close(float(scale[key]["seconds"]), seconds) or not close(float(trace[key]["seconds"]), seconds):
                raise ValueError(f"Figure 8 generated CSV differs: {label}/{system}")
            if int(trace[key]["total_trace_events"]) != int(metrics["total_trace_events"]):
                raise ValueError(f"Figure 8 event count differs: {label}/{system}")
    plot = generated / "figure8.pdf"
    if not plot.is_file() or plot.stat().st_size == 0:
        raise ValueError(f"missing Figure 8 plot: {plot}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure6-generated-dir", type=Path)
    parser.add_argument("--figure8-generated-dir", type=Path)
    args = parser.parse_args()
    if args.figure6_generated_dir is None and args.figure8_generated_dir is None:
        parser.error("select --figure6-generated-dir and/or --figure8-generated-dir")
    if args.figure6_generated_dir is not None:
        validate_figure6(args.figure6_generated_dir.resolve())
    if args.figure8_generated_dir is not None:
        validate_figure8(args.figure8_generated_dir.resolve())
    print("E3 validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
