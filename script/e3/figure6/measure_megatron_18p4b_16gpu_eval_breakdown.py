#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Callable

import flexmaya_ras as fm

from measure_maya_megatron_fakecuda_similarity import load_step_window
from measure_megatron_18p4b_16gpu_e2e_speedup import (
    ANCHOR,
    CANDIDATES,
    SOURCE_OPS,
    apply_figure6_wall_time_contract,
    cumulative_table,
    load_region_windows,
    spec_for,
    update_timing_method,
)


BREAKDOWN_COMPONENTS = (
    "maya_emulation_s",
    "trace_processing_s",
    "trace_ras_compaction_s",
    "code_analysis_s",
    "source_ras_partition_update_s",
    "grounding_s",
    "selective_emulation_s",
    "trace_patching_collation_s",
    "event_simulation_s",
    "feedback_generation_s",
)

MAYA_FULL = "Maya-style full"
MAYA_TRACE_RAS_ABLATION = "Maya-style + FlexEva trace-RAS"
FLEXEVA_REFRESH = "FlexEva refresh"
FLEXEVA_CUMULATIVE = "FlexEva cumulative"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Figure 6 breakdown tables for the 18.4B/16-rank run."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def timed(fn: Callable[[], object]) -> tuple[float, object]:
    start = time.perf_counter()
    value = fn()
    return time.perf_counter() - start, value


def empty_components() -> dict[str, float]:
    return {component: 0.0 for component in BREAKDOWN_COMPONENTS}


def total(components: dict[str, float]) -> float:
    return sum(float(components.get(component, 0.0)) for component in BREAKDOWN_COMPONENTS)


def add_components(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {
        component: float(left.get(component, 0.0)) + float(right.get(component, 0.0))
        for component in BREAKDOWN_COMPONENTS
    }


def feedback_generation_time(payload: object) -> float:
    elapsed_s, _ = timed(lambda: json.dumps(payload, sort_keys=True))
    return elapsed_s


def source_paths_for(input_dir: Path, case_dir_name: str, case, anchor_paths: dict[str, Path]) -> dict[str, Path]:
    case_source_dir = input_dir / case_dir_name / "source_partitions"
    paths: dict[str, Path] = {}
    for stage in range(case.pp):
        partition_id = f"stage_{stage:03d}"
        candidate_path = case_source_dir / f"{partition_id}.py"
        paths[partition_id] = candidate_path if candidate_path.exists() else anchor_paths[partition_id]
    for op in SOURCE_OPS:
        candidate_path = case_source_dir / f"{op}.py"
        paths[op] = candidate_path if candidate_path.exists() else anchor_paths[op]
    return paths


def selected_region_wall_time_s(trace_dir: Path, case, selected_ops: tuple[str, ...]) -> float:
    selected = set(selected_ops)
    per_rank_s: list[float] = []
    for rank in range(case.world_size):
        markers_path = trace_dir / f"rank_{rank}_markers.jsonl"
        step_start, step_end = load_step_window(markers_path)
        windows = load_region_windows(markers_path, step_start, step_end)
        duration_us = sum(max(end - start, 0) for start, end, label in windows if label in selected)
        per_rank_s.append(duration_us / 1.0e6)
    return max(per_rank_s) if per_rank_s else 0.0


def stored_or_measured_feedback_time(section: dict[str, object], payload: object) -> tuple[float, str]:
    stored = section.get("phases_s", {}).get("feedback_generation_s")
    if stored is not None:
        return float(stored), "stored_in_e2e_driver"
    return feedback_generation_time(payload), "retimed_during_figure6_postprocess"


def baseline_components(
    section: dict[str, object],
    *,
    emulation_s: float,
    feedback_payload: object,
    trace_ras_ablation: bool,
) -> tuple[dict[str, float], dict[str, str]]:
    phases = section["phases_s"]
    feedback_s, feedback_basis = stored_or_measured_feedback_time(section, feedback_payload)
    parse_s = float(phases.get("jsonl_parse_s", 0.0))
    build_s = float(phases.get("trace_build_s", 0.0))
    components = empty_components()
    components.update(
        {
            "maya_emulation_s": emulation_s,
            "trace_processing_s": parse_s + (0.0 if trace_ras_ablation else build_s),
            "trace_ras_compaction_s": build_s if trace_ras_ablation else 0.0,
            "event_simulation_s": float(phases.get("python_replay_s", 0.0)),
            "feedback_generation_s": feedback_s,
        }
    )
    return components, {"feedback_timing_basis": feedback_basis}


def anchor_components(anchor_row: dict[str, object]) -> tuple[dict[str, float], dict[str, str]]:
    phases = anchor_row["flexeva_anchor_init"]["phases_s"]
    feedback_s, feedback_basis = stored_or_measured_feedback_time(
        anchor_row["flexeva_anchor_init"], anchor_row["flexeva_anchor_init"]
    )
    components = empty_components()
    components.update(
        {
            "maya_emulation_s": float(anchor_row["run"]["elapsed_s"]),
            "trace_processing_s": float(phases.get("jsonl_parse_s", 0.0)),
            "trace_ras_compaction_s": float(phases.get("trace_build_s", 0.0)),
            "code_analysis_s": float(phases.get("source_hash_s", 0.0)),
            "event_simulation_s": float(phases.get("python_replay_s", 0.0)),
            "feedback_generation_s": feedback_s,
        }
    )
    return components, {
        "feedback_timing_basis": feedback_basis,
        "process_wall_s": float(anchor_row["run"]["elapsed_s"]),
        "max_marker_step_window_s": float(anchor_row["step_window_s"]["max"]),
    }


def refresh_components(
    input_dir: Path,
    candidate_row: dict[str, object],
    candidate,
    anchor_paths: dict[str, Path],
) -> tuple[dict[str, float], dict[str, object]]:
    refresh = candidate_row["flexeva_refresh"]
    phases = refresh["phases_s"]
    plan = refresh["plan"]
    source_paths = source_paths_for(input_dir, candidate.name, candidate.case, anchor_paths)
    spec = spec_for(candidate.case, source_paths)
    stored_code_analysis_s = phases.get("source_hash_s")
    if stored_code_analysis_s is None:
        code_analysis_s, _ = timed(lambda: fm.source_hashes(spec))
        code_analysis_basis = "retimed_during_figure6_postprocess"
    else:
        code_analysis_s = float(stored_code_analysis_s)
        code_analysis_basis = "stored_in_e2e_driver"
    source_update_s = float(phases.get("refresh_plan_s", 0.0))
    feedback_s, feedback_basis = stored_or_measured_feedback_time(refresh, refresh)

    if bool(plan.get("configuration_changed", False)):
        selective_emulation_s = float(candidate_row["run"]["elapsed_s"])
        selective_emulation_method = "full candidate process wall time because the parallel config changed"
    elif phases.get("selective_emulation_s") is not None:
        selective_emulation_s = float(phases["selective_emulation_s"])
        selective_emulation_method = "stored marker-derived selected-region window from the e2e driver"
    else:
        selective_emulation_s = selected_region_wall_time_s(
            Path(candidate_row["run"]["trace_dir"]),
            candidate.case,
            tuple(candidate.changed_ops),
        )
        selective_emulation_method = "max per-rank wall time of selected source-region marker windows"

    components = empty_components()
    components.update(
        {
            "code_analysis_s": code_analysis_s,
            "source_ras_partition_update_s": source_update_s,
            "grounding_s": 0.0,
            "selective_emulation_s": selective_emulation_s,
            "trace_patching_collation_s": float(phases.get("trace_filter_s", 0.0)),
            "event_simulation_s": float(phases.get("python_replay_s", 0.0)),
            "feedback_generation_s": feedback_s,
        }
    )
    return components, {
        "selective_emulation_method": selective_emulation_method,
        "code_analysis_timing_basis": code_analysis_basis,
        "feedback_timing_basis": feedback_basis,
    }


def write_wide(path: Path, rows: list[dict[str, object]], *, include_round: bool = True) -> None:
    fields = ["round", "x_label", "system", *BREAKDOWN_COMPONENTS, "total_s"] if include_round else [
        "system",
        *BREAKDOWN_COMPONENTS,
        "total_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_long(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "x_label", "system", "component", "seconds"])
        writer.writeheader()
        for row in rows:
            for component in BREAKDOWN_COMPONENTS:
                writer.writerow(
                    {
                        "round": row["round"],
                        "x_label": row["x_label"],
                        "system": row["system"],
                        "component": component,
                        "seconds": row[component],
                    }
                )


def main() -> int:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    result = json.loads((args.input_dir / "result.json").read_text(encoding="utf-8"))
    anchor_row = result["anchor"]
    apply_figure6_wall_time_contract(anchor_row, result["candidates"])
    update_timing_method(result["method"])
    result["cumulative"] = cumulative_table(
        float(anchor_row["metrics"]["anchor_init_s"]), result["candidates"]
    )
    (args.out_dir / "e2e_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "e2e_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["cumulative"][0]))
        writer.writeheader()
        writer.writerows(result["cumulative"])
    candidates_by_name = {row["candidate"]: row for row in result["candidates"]}
    anchor_dir = str(result.get("method", {}).get("anchor_dir", "anchor"))
    anchor_paths = source_paths_for(args.input_dir, anchor_dir, ANCHOR, {})

    anchor_init, anchor_diagnostic = anchor_components(anchor_row)
    per_round_rows: list[dict[str, object]] = []
    cumulative_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    cumulative_maya_full = empty_components()
    cumulative_maya_trace = empty_components()
    cumulative_flexeva = dict(anchor_init)

    for round_index, candidate in enumerate(CANDIDATES, start=1):
        row = candidates_by_name[candidate.name]
        emulation_wall_s = float(row["run"]["elapsed_s"])
        maya_full, maya_full_diagnostic = baseline_components(
            row["maya_full"],
            emulation_s=emulation_wall_s,
            feedback_payload=row["maya_full"],
            trace_ras_ablation=False,
        )
        maya_trace, maya_trace_diagnostic = baseline_components(
            row["maya_trace_ras"],
            emulation_s=emulation_wall_s,
            feedback_payload=row["maya_trace_ras"],
            trace_ras_ablation=True,
        )
        refresh, diagnostic = refresh_components(args.input_dir, row, candidate, anchor_paths)

        for system, components in (
            (MAYA_FULL, maya_full),
            (MAYA_TRACE_RAS_ABLATION, maya_trace),
            (FLEXEVA_REFRESH, refresh),
        ):
            per_round_rows.append(
                {
                    "round": round_index,
                    "x_label": row["label"],
                    "system": system,
                    **components,
                    "total_s": total(components),
                }
            )

        cumulative_maya_full = add_components(cumulative_maya_full, maya_full)
        cumulative_maya_trace = add_components(cumulative_maya_trace, maya_trace)
        cumulative_flexeva = add_components(cumulative_flexeva, refresh)
        for system, components in (
            (MAYA_FULL, cumulative_maya_full),
            (MAYA_TRACE_RAS_ABLATION, cumulative_maya_trace),
            (FLEXEVA_CUMULATIVE, cumulative_flexeva),
        ):
            cumulative_rows.append(
                {
                    "round": round_index,
                    "x_label": row["label"],
                    "system": system,
                    **components,
                    "total_s": total(components),
                }
            )
        diagnostics.append(
            {
                "round": round_index,
                "label": row["label"],
                "changed_ops": row["changed_ops"],
                "configuration_changed": row["flexeva_refresh"]["plan"]["configuration_changed"],
                "refresh_event_reuse": row["metrics"]["refresh_event_reuse"],
                "affected_trace_partitions": row["flexeva_refresh"]["plan"]["affected_trace_partition_count"],
                "candidate_process_wall_s": emulation_wall_s,
                "candidate_max_marker_step_window_s": float(row["step_window_s"]["max"]),
                "maya_full": maya_full_diagnostic,
                "maya_trace_ras_ablation": maya_trace_diagnostic,
                **diagnostic,
            }
        )

    write_wide(args.out_dir / "breakdown_per_round_wide.csv", per_round_rows)
    write_wide(args.out_dir / "breakdown_cumulative_wide.csv", cumulative_rows)
    write_long(args.out_dir / "breakdown_cumulative_long.csv", cumulative_rows)

    output = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input_dir": str(args.input_dir),
        "method": {
            "workload": "Megatron 18.4B/16-rank maya_megatron.py fake-CUDA trace-shape workload",
            "physical_execution": "two guarded eight-A100 hosts; torchrun launches eight ranks per host",
            "timing_boundary": (
                "Figure 6 accounting; full fake-CUDA emulation uses subprocess wall time "
                "including startup and teardown, while source-selective emulation uses marked region windows"
            ),
            "baseline_attribution": (
                "Maya-style full is the local paper-aligned implementation, not the unavailable original Maya source; "
                "Maya-style + FlexEva trace-RAS is an author-constructed ablation"
            ),
            "maya_emulation_s": "complete fake-CUDA subprocess wall time",
            "trace_processing_s": "JSONL parsing plus ordinary full-trace construction for Maya-style full; JSONL parsing for RAS paths",
            "trace_ras_compaction_s": "FlexEva trace-RAS construction only; always zero for Maya-style full",
            "code_analysis_s": "source hash analysis recorded by the e2e driver (legacy inputs are explicitly retimed)",
            "source_ras_partition_update_s": "refresh-plan wall time after separately measured source hash analysis",
            "grounding_s": "zero for dense Megatron",
            "selective_emulation_s": (
                "source rounds use marker-derived selected-region wall time; "
                "the TP/DP config round uses the full candidate process wall time"
            ),
            "trace_patching_collation_s": "selected trace filter/patch for refresh; zero for both full-evaluation baselines",
            "event_simulation_s": "Python replay wall time from the completed e2e run",
            "feedback_generation_s": "JSON feedback serialization recorded by the e2e driver (legacy inputs are explicitly retimed)",
        },
        "anchor_diagnostic": anchor_diagnostic,
        "anchor_init_components": anchor_init,
        "per_round": per_round_rows,
        "cumulative": cumulative_rows,
        "diagnostics": diagnostics,
    }
    (args.out_dir / "result.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Megatron 18.4B / 16-rank Evaluation Breakdown",
        "",
        "Input: `maya_megatron.py` fake-CUDA traces from controlled cumulative source-region selections. Full emulation uses the Figure 6 process-wall boundary; marker windows remain diagnostics and define source-selective emulation.",
        "`Maya-style + FlexEva trace-RAS` is an author-constructed ablation, not a feature of original Maya.",
        "",
        "| Round | Candidate | Maya-style full (s) | Maya-style + FlexEva trace-RAS (s) | FlexEva cumulative (s) | FlexEva refresh (s) |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for round_index, candidate in enumerate(CANDIDATES, start=1):
        label = candidates_by_name[candidate.name]["label"]
        full_total = next(row["total_s"] for row in cumulative_rows if row["round"] == round_index and row["system"] == MAYA_FULL)
        trace_total = next(row["total_s"] for row in cumulative_rows if row["round"] == round_index and row["system"] == MAYA_TRACE_RAS_ABLATION)
        flex_total = next(
            row["total_s"] for row in cumulative_rows if row["round"] == round_index and row["system"] == FLEXEVA_CUMULATIVE
        )
        refresh_total = next(
            row["total_s"] for row in per_round_rows if row["round"] == round_index and row["system"] == FLEXEVA_REFRESH
        )
        lines.append(
            f"| {round_index} | {label} | {full_total:.3f} | {trace_total:.3f} | {flex_total:.3f} | {refresh_total:.3f} |"
        )
    lines.extend(
        [
            "",
            "CSV files:",
            "- `breakdown_per_round_wide.csv`: one row per round/system.",
            "- `breakdown_cumulative_wide.csv`: cumulative stacked-bar data.",
            "- `breakdown_cumulative_long.csv`: long-form cumulative data.",
        ]
    )
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str(args.out_dir / "result.json"),
                "per_round": str(args.out_dir / "breakdown_per_round_wide.csv"),
                "cumulative": str(args.out_dir / "breakdown_cumulative_wide.csv"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
