#!/usr/bin/env python3
"""Validate the retained and freshly generated E3 figure ledgers."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "result" / "e3"
DEFAULT_GENERATED = RESULT_DIR / "generated" / "figure6"
DEFAULT_FIGURE8_GENERATED = RESULT_DIR / "generated" / "figure8"
FIGURE8_SYSTEMS = ("Maya-full", "Maya-trace-RAS", "FlexEva refresh")
FIGURE8_EXPECTED = {
    "2.7B/8GPU TP1-PP8-DP1": {
        "case": "megatron_2p7b_8gpu",
        "topology": (8, 1, 8, 1),
        "events": (5_109_807, 1_827_036),
        "ratio": 0.3575547961009095,
        "seconds": (133.1263225711882, 129.8060109578073, 27.429122076628445),
    },
    "2.7B/16GPU TP1-PP8-DP2": {
        "case": "megatron_2p7b_16gpu_dp2",
        "topology": (16, 1, 8, 2),
        "events": (10_234_016, 3_653_214),
        "ratio": 0.3569677827355361,
        "seconds": (235.98281895183027, 182.76972949504852, 27.99616958781421),
    },
    "18.4B/16GPU TP2-PP8-DP1": {
        "case": "megatron_18p4b_16gpu",
        "topology": (16, 2, 8, 1),
        "events": (26_980_072, 9_205_510),
        "ratio": 0.3411966432113302,
        "seconds": (605.9359602481127, 455.1708391327411, 77.34714144642425),
    },
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        result = list(csv.DictReader(stream))
    if not result:
        raise ValueError(f"empty E3 ledger: {path}")
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


def validate_paper_ledgers() -> None:
    expected = {
        "a": (
            ("R1", 1.0, 1.0398405393781895),
            ("R2", 2.0054422245022687, 1.1510639397947473),
            ("R3", 3.010403005356908, 1.2516062794528466),
            ("R4", 3.5687609760590244, 1.4766067761442887),
        ),
        "b": (
            ("R1", 1.0, 1.1240813790628736),
            ("R2", 2.0038156040488335, 1.2574243229030382),
            ("R3", 3.0227582793627397, 1.393415694906547),
            ("R4", 4.03059414436752, 2.1119045149635918),
        ),
    }
    for panel, target in expected.items():
        current = rows(RESULT_DIR / f"figure6{panel}.csv")
        if len(current) != 4:
            raise ValueError(f"Figure 6({panel}) requires four rounds")
        for row, (round_name, maya, flexeva) in zip(current, target, strict=True):
            if row["Cumulative Round"] != round_name:
                raise ValueError(f"Figure 6({panel}) round order differs")
            if not close(float(row["Maya"]), maya) or not close(float(row["FlexEva"]), flexeva):
                raise ValueError(f"Figure 6({panel}) target changed: {round_name}")
        validate_trajectory(
            panel,
            [float(row["Maya"]) for row in current],
            [float(row["FlexEva"]) for row in current],
        )


def validate_capture(capture: dict[str, object], *, label: str) -> None:
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
        raise ValueError(f"Figure 6 Maya replay cycle: {label}")
    trace_dir = Path(run["trace_dir"])
    missing = [
        str(trace_dir / f"rank_{rank}{suffix}")
        for rank in range(16)
        for suffix in (".jsonl", "_markers.jsonl")
        if not (trace_dir / f"rank_{rank}{suffix}").is_file()
        or (trace_dir / f"rank_{rank}{suffix}").stat().st_size == 0
    ]
    if missing:
        raise ValueError(f"Figure 6 raw traces are incomplete: {label}: {missing[0]}")


def figure8_by_key(path: Path) -> tuple[list[str], dict[tuple[str, str], dict[str, str]]]:
    current = rows(path)
    labels = list(dict.fromkeys(row["label"] for row in current))
    by_key = {(row["label"], row["system"]): row for row in current}
    expected_keys = {(label, system) for label in FIGURE8_EXPECTED for system in FIGURE8_SYSTEMS}
    if labels != list(FIGURE8_EXPECTED) or set(by_key) != expected_keys:
        raise ValueError(f"Figure 8 case/system set differs: {path}")
    return labels, by_key


def validate_figure8_paper_ledgers() -> None:
    _, scale = figure8_by_key(RESULT_DIR / "figure8a.csv")
    _, trace = figure8_by_key(RESULT_DIR / "figure8b.csv")
    for label, expected in FIGURE8_EXPECTED.items():
        total, regenerated = expected["events"]
        for system, seconds in zip(FIGURE8_SYSTEMS, expected["seconds"], strict=True):
            key = (label, system)
            if not close(float(scale[key]["seconds"]), seconds):
                raise ValueError(f"Figure 8(a) paper target changed: {label}/{system}")
            row = trace[key]
            if not close(float(row["seconds"]), seconds):
                raise ValueError(f"Figure 8 panel time mismatch: {label}/{system}")
            if int(row["total_trace_events"]) != total or int(row["regenerated_trace_events"]) != regenerated:
                raise ValueError(f"Figure 8 event target changed: {label}/{system}")
            if not close(float(row["refresh_ratio"]), expected["ratio"]):
                raise ValueError(f"Figure 8 refresh ratio changed: {label}/{system}")


def validate_figure8_capture(row: dict[str, object]) -> None:
    label = str(row["label"])
    expected = FIGURE8_EXPECTED[label]
    case = row["case"]
    world_size, tp, pp, dp = expected["topology"]
    if (case["name"], case["world_size"], case["tp"], case["pp"], case["dp"]) != (
        expected["case"], world_size, tp, pp, dp
    ):
        raise ValueError(f"Figure 8 case geometry differs: {label}")
    if int(row["physical_gpu_count"]) != world_size:
        raise ValueError(f"Figure 8 physical GPU count differs: {label}")
    run = row["run"]
    if int(run["return_code"]) != 0:
        raise ValueError(f"Figure 8 capture failed: {label}")
    distributed = run["distributed"]
    expected_launch = (1, 8) if world_size == 8 else (2, 8)
    if (int(distributed["nnodes"]), int(distributed["nproc_per_node"])) != expected_launch:
        raise ValueError(f"Figure 8 launch is not paper-scale: {label}")
    expected_transfer_files = 0 if world_size == 8 else 16
    if int(run["peer_trace_transfer"]["files"]) != expected_transfer_files:
        raise ValueError(f"Figure 8 peer trace transfer differs: {label}")
    if float(row["step_begin_skew_s"]) > 1.0:
        raise ValueError(f"Figure 8 rank synchronization differs: {label}")
    api_audit = row["api_audit"]
    if int(api_audit["cudaGetDevice_modeled_count"]) <= 0 or int(
        api_audit["cudaGetDevice_modeled_count"]
    ) != int(api_audit["cudaGetDevice_replay_count"]):
        raise ValueError(f"Figure 8 context-event population differs: {label}")
    if row["flexeva_refresh"]["plan"]["changed_partitions"] != ["attention_backward"]:
        raise ValueError(f"Figure 8 mutation scope differs: {label}")
    if any(
        section["feedback"]["cycle_detected"]
        for section in (row["maya_full"], row["maya_trace_ras"], row["flexeva_refresh"])
    ):
        raise ValueError(f"Figure 8 replay cycle detected: {label}")
    trace_dir = Path(run["trace_dir"])
    missing = [
        trace_dir / f"rank_{rank}{suffix}"
        for rank in range(world_size)
        for suffix in (".jsonl", "_markers.jsonl")
        if not (trace_dir / f"rank_{rank}{suffix}").is_file()
        or (trace_dir / f"rank_{rank}{suffix}").stat().st_size == 0
    ]
    if missing:
        raise ValueError(f"Figure 8 raw traces are incomplete: {label}: {missing[0]}")

    metrics = row["metrics"]
    phases = (
        (
            row["maya_full"]["phases_s"],
            ("maya_emulation_s", "jsonl_parse_s", "trace_build_s", "python_replay_s", "feedback_generation_s"),
            "maya_full_s",
        ),
        (
            row["maya_trace_ras"]["phases_s"],
            ("maya_emulation_s", "jsonl_parse_s", "trace_build_s", "python_replay_s", "feedback_generation_s"),
            "maya_trace_ras_s",
        ),
        (
            row["flexeva_refresh"]["phases_s"],
            (
                "source_analysis_s",
                "refresh_plan_s",
                "selective_emulation_s",
                "trace_filter_s",
                "python_replay_s",
                "feedback_generation_s",
            ),
            "flexeva_refresh_s",
        ),
    )
    for current, components, metric in phases:
        if not close(sum(float(current[name]) for name in components), float(current["total_s"])):
            raise ValueError(f"Figure 8 phase sum differs: {label}/{metric}")
        if not close(float(current["total_s"]), float(metrics[metric])):
            raise ValueError(f"Figure 8 metric/phase mismatch: {label}/{metric}")
    total = int(metrics["total_trace_events"])
    regenerated = int(metrics["regenerated_trace_events"])
    if not 0 < regenerated <= total or not close(regenerated / total, float(metrics["refresh_ratio"])):
        raise ValueError(f"Figure 8 refresh accounting differs: {label}")
    if float(metrics["flexeva_refresh_s"]) >= float(metrics["maya_full_s"]):
        raise ValueError(f"Figure 8 FlexEva trend differs: {label}")


def validate_figure8_generated(generated: Path, max_timing_drift_rel: float) -> None:
    payload = json.loads((generated / "result.json").read_text(encoding="utf-8"))
    if payload["method"]["mutation"] != "attention_backward" or int(payload["method"]["sample_count"]) != 1:
        raise ValueError("Figure 8 method differs")
    result_by_label = {str(row["label"]): row for row in payload["results"]}
    if set(result_by_label) != set(FIGURE8_EXPECTED):
        raise ValueError("Figure 8 generated cases differ")
    _, scale = figure8_by_key(generated / "figure8a.csv")
    _, trace = figure8_by_key(generated / "figure8b.csv")
    for label, row in result_by_label.items():
        validate_figure8_capture(row)
        metrics = row["metrics"]
        expected = FIGURE8_EXPECTED[label]
        paper_total, _ = expected["events"]
        if abs(int(metrics["total_trace_events"]) / paper_total - 1.0) > 0.001:
            raise ValueError(f"Figure 8 event-count drift exceeds 0.1%: {label}")
        if abs(float(metrics["refresh_ratio"]) - float(expected["ratio"])) > 0.001:
            raise ValueError(f"Figure 8 refresh-ratio drift exceeds 0.1 percentage point: {label}")
        for system, metric, paper_seconds in zip(
            FIGURE8_SYSTEMS,
            ("maya_full_s", "maya_trace_ras_s", "flexeva_refresh_s"),
            expected["seconds"],
            strict=True,
        ):
            key = (label, system)
            current_seconds = float(metrics[metric])
            if not close(float(scale[key]["seconds"]), current_seconds) or not close(
                float(trace[key]["seconds"]), current_seconds
            ):
                raise ValueError(f"Figure 8 generated CSV differs: {label}/{system}")
            if abs(current_seconds / paper_seconds - 1.0) > max_timing_drift_rel:
                raise ValueError(
                    f"Figure 8 timing drift exceeds {100 * max_timing_drift_rel:.1f}%: {label}/{system}"
                )
            if int(trace[key]["total_trace_events"]) != int(metrics["total_trace_events"]):
                raise ValueError(f"Figure 8 generated event count differs: {label}/{system}")
    plot = generated / "figure8.pdf"
    if not plot.is_file() or plot.stat().st_size == 0:
        raise ValueError(f"missing Figure 8 plot: {plot}")


def validate_generated(generated: Path) -> None:
    summary = json.loads((generated / "result.json").read_text(encoding="utf-8"))
    if summary["execution"]["nodes"] != 2 or summary["execution"]["physical_a100s"] != 16:
        raise ValueError("Figure 6 physical execution contract differs")

    gpt = json.loads((generated / "gpt" / "breakdown" / "e2e_result.json").read_text(encoding="utf-8"))
    moe = json.loads((generated / "moe" / "result.json").read_text(encoding="utf-8"))
    gpt_captures = [gpt["anchor"], *gpt["candidates"]]
    moe_captures = [moe["anchor"], *moe["candidates"]]
    if len(gpt_captures) != 5 or len(moe_captures) != 5:
        raise ValueError("Figure 6 requires anchor plus four candidates per panel")
    for index, capture in enumerate(gpt_captures):
        validate_capture(capture, label=f"gpt-{index}")
    for index, capture in enumerate(moe_captures):
        validate_capture(capture, label=f"moe-{index}")
    for candidate in [*gpt["candidates"], *moe["candidates"]]:
        refresh = candidate["flexeva_refresh"]
        if int(refresh["source_analysis_count"]) != 1 or refresh["feedback"]["cycle_detected"]:
            raise ValueError("Figure 6 source-analysis/replay contract differs")

    figure = rows(generated / "figure6.csv")
    phases = rows(generated / "phase_breakdown.csv")
    if len(figure) != 8 or len(phases) != 16:
        raise ValueError("Figure 6 generated row count differs")
    for panel in ("gpt", "moe"):
        panel_rows = sorted(
            (row for row in figure if row["panel"] == panel),
            key=lambda row: int(row["round"]),
        )
        if [int(row["round"]) for row in panel_rows] != [1, 2, 3, 4]:
            raise ValueError(f"Figure 6 {panel} rounds differ")
        maya = [float(row["maya_normalized"]) for row in panel_rows]
        flexeva = [float(row["flexeva_normalized"]) for row in panel_rows]
        validate_trajectory(panel, maya, flexeva)
    phase_by_key = {(row["panel"], int(row["round"]), row["system"]): row for row in phases}
    for row in figure:
        key = (row["panel"], int(row["round"]))
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
        path = ROOT / "plot" / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing Figure 6 plot: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-generated", action="store_true")
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_GENERATED)
    parser.add_argument("--require-figure8-generated", action="store_true")
    parser.add_argument("--figure8-generated-dir", type=Path, default=DEFAULT_FIGURE8_GENERATED)
    parser.add_argument("--figure8-max-timing-drift-rel", type=float, default=0.10)
    args = parser.parse_args()
    if not 0.0 <= args.figure8_max_timing_drift_rel <= 1.0:
        raise ValueError("--figure8-max-timing-drift-rel must be between 0 and 1")
    validate_paper_ledgers()
    validate_figure8_paper_ledgers()
    if args.require_generated:
        validate_generated(args.generated_dir.resolve())
    if args.require_figure8_generated:
        validate_figure8_generated(
            args.figure8_generated_dir.resolve(),
            args.figure8_max_timing_drift_rel,
        )
    fresh = []
    if args.require_generated:
        fresh.append("Figure 6 fresh two-node run")
    if args.require_figure8_generated:
        fresh.append("Figure 8 fresh 8/16-GPU run")
    suffix = f"; {', '.join(fresh)}" if fresh else ""
    print(f"E3 validation: PASS (retained Figures 6 and 8{suffix})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
