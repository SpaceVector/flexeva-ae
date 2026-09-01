#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import flexmaya_ras as fm

from measure_maya_megatron_fakecuda_similarity import (
    load_step_window,
    raw_event_from_record,
    run_maya_megatron_case,
)
from measure_megatron_trace_similarity import MegatronCase


SOURCE_OPS = ("attention_backward", "mlp_backward", "optimizer_step")


@dataclass(frozen=True)
class CandidateRound:
    name: str
    label: str
    case: MegatronCase
    changed_ops: tuple[str, ...]
    mutation: str = "source"


def case_for(name: str, *, tp: int, pp: int, dp: int) -> MegatronCase:
    world_size = 16
    if tp * pp * dp != world_size:
        raise ValueError(f"invalid 16-rank geometry: TP{tp}-PP{pp}-DP{dp}")
    return MegatronCase(
        name=name,
        parameter_scale="2.7B",
        steps=1,
        global_batch_size=256,
        seq_len=2048,
        hidden_size=2560,
        num_layers=32,
        num_heads=32,
        vocab_size=32000,
        tp=tp,
        pp=pp,
        dp=dp,
        world_size=world_size,
        micro_batches=256,
        schedule="1f1b",
        dtype="bf16",
    )


ANCHOR = case_for("megatron_2p7b_16rank_tp1_pp8_dp2_anchor", tp=1, pp=8, dp=2)
TP_DP_CASE = case_for("megatron_2p7b_16rank_tp2_pp8_dp1", tp=2, pp=8, dp=1)
CANDIDATES = (
    CandidateRound("round1_attention_backward", "Attn", ANCHOR, ("attention_backward",)),
    CandidateRound("round2_attention_mlp_backward", "Attn+MLP", ANCHOR, ("attention_backward", "mlp_backward")),
    CandidateRound("round3_attention_mlp_optimizer", "Attn+MLP+Opt", ANCHOR, SOURCE_OPS),
    CandidateRound(
        "round4_attention_mlp_optimizer_tp_dp",
        "Attn+MLP+Opt+TP/DP",
        TP_DP_CASE,
        SOURCE_OPS,
        mutation="source+parallel_config",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the 2.7B/16-rank Megatron cumulative evaluator core-time breakdown."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--maya-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--proot", type=Path, required=True)
    parser.add_argument("--local-device-count", type=int, default=8)
    parser.add_argument("--keep-raw-traces", action="store_true")
    parser.set_defaults(reuse_existing_traces=False, source_region_markers=True)
    return parser.parse_args()


def timed(fn: Callable[[], object]) -> tuple[float, object]:
    start = time.perf_counter()
    value = fn()
    return time.perf_counter() - start, value


def marker_step_window_seconds(case: MegatronCase, trace_dir: Path) -> dict[str, object]:
    per_rank: dict[str, float] = {}
    for rank in range(case.world_size):
        start_ts, end_ts = load_step_window(trace_dir / f"rank_{rank}_markers.jsonl")
        per_rank[str(rank)] = max((end_ts - start_ts) / 1.0e6, 0.0)
    values = list(per_rank.values())
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values), "per_rank": per_rank}


def selected_region_wall_time_s(
    case: MegatronCase,
    trace_dir: Path,
    selected_ops: tuple[str, ...],
) -> float:
    selected = set(selected_ops)
    per_rank_s: list[float] = []
    for rank in range(case.world_size):
        step_start, step_end = load_step_window(trace_dir / f"rank_{rank}_markers.jsonl")
        windows = load_region_windows(
            trace_dir / f"rank_{rank}_markers.jsonl", step_start, step_end
        )
        duration_us = sum(
            max(end - start, 0)
            for start, end, label in windows
            if label in selected
        )
        per_rank_s.append(duration_us / 1.0e6)
    return max(per_rank_s) if per_rank_s else 0.0


def feedback_payload(trace: object, replay: object) -> dict[str, object]:
    return {"trace": fm.trace_summary(trace), "feedback": replay.to_dict()}


def feedback_generation_time(payload: object) -> float:
    elapsed_s, _ = timed(lambda: json.dumps(payload, sort_keys=True))
    return elapsed_s


def load_region_windows(markers_path: Path, step_start: int, step_end: int) -> list[tuple[int, int, str]]:
    stacks: dict[str, list[int]] = {label: [] for label in SOURCE_OPS}
    windows: list[tuple[int, int, str]] = []
    with markers_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            label = str(record.get("label", ""))
            if label not in stacks:
                continue
            ts = record.get("trace_ts")
            if ts is None:
                continue
            trace_ts = int(ts)
            if record.get("kind") == "region_begin":
                stacks[label].append(trace_ts)
            elif record.get("kind") == "region_end" and stacks[label]:
                begin = stacks[label].pop()
                end = trace_ts
                if end < step_start or begin > step_end:
                    continue
                windows.append((max(begin, step_start), min(end, step_end), label))
    return sorted(windows)


def region_label_for_ts(ts: int, windows: list[tuple[int, int, str]]) -> str | None:
    for start, end, label in windows:
        if ts < start:
            return None
        if start <= ts <= end:
            return label
    return None


def parse_case_raw_events_with_regions(case: MegatronCase, trace_dir: Path) -> list[object]:
    rows: list[object] = []
    next_id = 1
    for rank in range(case.world_size):
        markers_path = trace_dir / f"rank_{rank}_markers.jsonl"
        start_ts, end_ts = load_step_window(markers_path)
        windows = load_region_windows(markers_path, start_ts, end_ts)
        window_idx = 0
        trace_path = trace_dir / f"rank_{rank}.jsonl"
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                ts = int(record.get("ts") or 0)
                if ts < start_ts or ts > end_ts:
                    continue
                if record.get("api") == "cudaGetDevice":
                    continue
                kind = str(record.get("type", ""))
                if kind in {"marker", "other"}:
                    continue
                event = raw_event_from_record(record, case=case, rank=rank, event_id=next_id)
                while window_idx < len(windows) and ts > windows[window_idx][1]:
                    window_idx += 1
                label = None
                if window_idx < len(windows):
                    start, end, current_label = windows[window_idx]
                    if start <= ts <= end:
                        label = current_label
                if label is not None:
                    event.code_partition = label
                rows.append(event)
                next_id += 1
    return rows


def write_source_files(case_dir: Path, case: MegatronCase, anchor_paths: dict[str, Path] | None = None, changed_ops: tuple[str, ...] = ()) -> dict[str, Path]:
    source_dir = case_dir / "source_partitions"
    source_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    anchor_paths = anchor_paths or {}
    for stage in range(case.pp):
        partition_id = f"stage_{stage:03d}"
        if partition_id in anchor_paths:
            paths[partition_id] = anchor_paths[partition_id]
            continue
        path = source_dir / f"{partition_id}.py"
        path.write_text(f"# Megatron {case.parameter_scale}/16-rank stage partition {stage}\nPIPELINE_STAGE = {stage}\n", encoding="utf-8")
        paths[partition_id] = path
    for op in SOURCE_OPS:
        if op not in changed_ops and op in anchor_paths:
            paths[op] = anchor_paths[op]
            continue
        suffix = "\n# source mutation\n" if op in changed_ops else ""
        path = source_dir / f"{op}.py"
        path.write_text(f"# Megatron source partition: {op}\nOP = {op!r}\n{suffix}", encoding="utf-8")
        paths[op] = path
    return paths


def spec_for(case: MegatronCase, source_paths: dict[str, Path]) -> fm.FlexMayaWorkloadSpec:
    partitions: list[fm.CodePartitionSpec] = []
    for stage in range(case.pp):
        partition_id = f"stage_{stage:03d}"
        partitions.append(
            fm.CodePartitionSpec(
                partition_id=partition_id,
                path=str(source_paths[partition_id]),
                active_ranks=fm.megatron_pp_stage_active_ranks(case.world_size, case.tp, case.pp, stage),
            )
        )
    for op in SOURCE_OPS:
        partitions.append(fm.CodePartitionSpec(partition_id=op, path=str(source_paths[op])))
    return fm.FlexMayaWorkloadSpec(
        workload_id=case.name,
        world_size=case.world_size,
        tp=case.tp,
        pp=case.pp,
        dp=case.dp,
        code_partitions=tuple(partitions),
        rank_group_policy="active_lane_set",
        notes=("maya_megatron.py fake-CUDA trace-shape workload with source-region markers",),
    )


def build_full(raw_events: list[object]) -> tuple[object, object, dict[str, float]]:
    build_s, trace = timed(lambda: fm.build_trace_ras(raw_events))
    replay_s, replay = timed(lambda: fm.replay_trace_once(trace))
    return trace, replay, {"trace_build_s": build_s, "python_replay_s": replay_s}


def build_trace_ras(case: MegatronCase, raw_events: list[object], source_paths: dict[str, Path]) -> tuple[object, object, dict[str, float], fm.FlexMayaWorkloadSpec]:
    spec = spec_for(case, source_paths)
    build_s, trace = timed(lambda: fm.build_rank_grouped_trace_ras(raw_events, fm.active_lane_rank_groups(spec)))
    replay_s, replay = timed(lambda: fm.replay_trace_once(trace))
    return trace, replay, {"trace_build_s": build_s, "python_replay_s": replay_s}, spec


def selected_refresh(trace: object, plan: fm.FlexMayaRefreshPlan) -> tuple[object, object, dict[str, float]]:
    filter_s, selected_trace = timed(lambda: fm.filter_trace_partitions(trace, list(plan.affected_trace_partitions)))
    replay_s, replay = timed(lambda: fm.replay_trace_once(selected_trace))
    return selected_trace, replay, {"trace_filter_s": filter_s, "python_replay_s": replay_s}


def event_ids_for_partitions(trace: object, partition_ids: tuple[int, ...]) -> tuple[int, ...]:
    selected = set(partition_ids)
    event_ids: set[int] = set()
    for partition in trace.sync_partitions:
        if int(partition.id) in selected:
            event_ids.update(int(event_id) for event_id in partition.event_ids)
    return tuple(sorted(event_ids))


def run_case(args: argparse.Namespace, case: MegatronCase, case_dir: Path) -> dict[str, object]:
    case_dir.mkdir(parents=True, exist_ok=True)
    run = run_maya_megatron_case(args, case, case_dir)
    if int(run["return_code"]) != 0:
        raise RuntimeError(f"{case.name} failed; see {run['stdout']} and {run['stderr']}")
    return run


def cleanup_traces(args: argparse.Namespace, run: dict[str, object]) -> None:
    if args.keep_raw_traces:
        return
    for path in Path(str(run["trace_dir"])).glob("rank_*.jsonl"):
        path.unlink(missing_ok=True)


def measure_anchor(args: argparse.Namespace) -> tuple[fm.FlexMayaAnchor, dict[str, object], dict[str, Path]]:
    case_dir = args.out_dir / "anchor"
    source_paths = write_source_files(case_dir, ANCHOR)
    run = run_case(args, ANCHOR, case_dir)
    trace_dir = Path(str(run["trace_dir"]))
    step_window_s = marker_step_window_seconds(ANCHOR, trace_dir)
    parse_s, raw_events = timed(lambda: parse_case_raw_events_with_regions(ANCHOR, trace_dir))
    full_trace, full_replay, full_phases = build_full(raw_events)
    trace_ras_trace, trace_ras_replay, trace_ras_phases, spec = build_trace_ras(ANCHOR, raw_events, source_paths)
    source_hash_s, source_hashes = timed(lambda: fm.source_hashes(spec))
    full_summary = feedback_payload(full_trace, full_replay)
    trace_ras_summary = feedback_payload(trace_ras_trace, trace_ras_replay)
    full_feedback_s = feedback_generation_time(full_summary)
    trace_ras_feedback_s = feedback_generation_time(trace_ras_summary)
    emulation_s = float(step_window_s["max"])
    maya_full_s = (
        emulation_s
        + parse_s
        + full_phases["trace_build_s"]
        + full_phases["python_replay_s"]
        + full_feedback_s
    )
    init_s = (
        emulation_s
        + parse_s
        + source_hash_s
        + trace_ras_phases["trace_build_s"]
        + trace_ras_phases["python_replay_s"]
        + trace_ras_feedback_s
    )
    anchor = fm.FlexMayaAnchor(
        spec=spec,
        source_hashes=source_hashes,
        trace=trace_ras_trace,
        feedback=trace_ras_replay,
        summary=trace_ras_summary,
    )
    row = {
        "case": asdict(ANCHOR),
        "run": run,
        "step_window_s": step_window_s,
        "raw_events": len(raw_events),
        "api_audit": {"cudaGetDevice_modeled_count": 0, "cudaGetDevice_replay_count": 0},
        "maya_full": {
            **full_summary,
            "phases_s": {
                "maya_emulation_s": emulation_s,
                "process_wall_s": float(run["elapsed_s"]),
                "jsonl_parse_s": parse_s,
                **full_phases,
                "feedback_generation_s": full_feedback_s,
                "total_s": maya_full_s,
            },
        },
        "flexeva_anchor_init": {
            **trace_ras_summary,
            "phases_s": {
                "maya_emulation_s": emulation_s,
                "process_wall_s": float(run["elapsed_s"]),
                "jsonl_parse_s": parse_s,
                "source_hash_s": source_hash_s,
                **trace_ras_phases,
                "feedback_generation_s": trace_ras_feedback_s,
                "total_s": init_s,
            },
        },
        "metrics": {"anchor_init_s": init_s, "anchor_init_over_maya_full": init_s / max(maya_full_s, 1.0e-12)},
    }
    cleanup_traces(args, run)
    return anchor, row, source_paths


def measure_candidate(args: argparse.Namespace, anchor: fm.FlexMayaAnchor, candidate: CandidateRound, anchor_source_paths: dict[str, Path]) -> dict[str, object]:
    case_dir = args.out_dir / candidate.name
    source_paths = write_source_files(case_dir, candidate.case, anchor_source_paths, candidate.changed_ops)
    run = run_case(args, candidate.case, case_dir)
    trace_dir = Path(str(run["trace_dir"]))
    step_window_s = marker_step_window_seconds(candidate.case, trace_dir)
    parse_s, raw_events = timed(lambda: parse_case_raw_events_with_regions(candidate.case, trace_dir))
    full_trace, full_replay, full_phases = build_full(raw_events)
    trace_ras_trace, trace_ras_replay, trace_ras_phases, spec = build_trace_ras(candidate.case, raw_events, source_paths)
    source_hash_s, candidate_source_hashes = timed(lambda: fm.source_hashes(spec))
    plan_s, plan = timed(
        lambda: fm.plan_candidate_refresh(
            anchor,
            spec,
            trace_ras_trace,
            candidate_source_hashes=candidate_source_hashes,
        )
    )
    selected_trace, selected_replay, selected_phases = selected_refresh(trace_ras_trace, plan)
    affected_event_ids = event_ids_for_partitions(trace_ras_trace, plan.affected_trace_partitions)
    full_summary = feedback_payload(full_trace, full_replay)
    trace_ras_summary = feedback_payload(trace_ras_trace, trace_ras_replay)
    selected_summary = feedback_payload(selected_trace, selected_replay)
    full_feedback_s = feedback_generation_time(full_summary)
    trace_ras_feedback_s = feedback_generation_time(trace_ras_summary)
    selected_feedback_s = feedback_generation_time(selected_summary)
    emulation_s = float(step_window_s["max"])
    selective_emulation_s = (
        emulation_s
        if plan.configuration_changed
        else selected_region_wall_time_s(candidate.case, trace_dir, candidate.changed_ops)
    )
    maya_full_s = (
        emulation_s
        + parse_s
        + full_phases["trace_build_s"]
        + full_phases["python_replay_s"]
        + full_feedback_s
    )
    maya_trace_ras_s = (
        emulation_s
        + parse_s
        + trace_ras_phases["trace_build_s"]
        + trace_ras_phases["python_replay_s"]
        + trace_ras_feedback_s
    )
    refresh_s = (
        source_hash_s
        + selective_emulation_s
        + plan_s
        + selected_phases["trace_filter_s"]
        + selected_phases["python_replay_s"]
        + selected_feedback_s
    )
    row = {
        "candidate": candidate.name,
        "label": candidate.label,
        "mutation": candidate.mutation,
        "case": asdict(candidate.case),
        "changed_ops": list(candidate.changed_ops),
        "setting": f"{candidate.label} ({'TP%d-PP%d-DP%d' % (candidate.case.tp, candidate.case.pp, candidate.case.dp)})",
        "run": run,
        "step_window_s": step_window_s,
        "raw_events": len(raw_events),
        "api_audit": {"cudaGetDevice_modeled_count": 0, "cudaGetDevice_replay_count": 0},
        "maya_full": {
            **full_summary,
            "phases_s": {
                "maya_emulation_s": emulation_s,
                "process_wall_s": float(run["elapsed_s"]),
                "jsonl_parse_s": parse_s,
                **full_phases,
                "feedback_generation_s": full_feedback_s,
                "total_s": maya_full_s,
            },
        },
        "maya_trace_ras": {
            **trace_ras_summary,
            "phases_s": {
                "maya_emulation_s": emulation_s,
                "process_wall_s": float(run["elapsed_s"]),
                "jsonl_parse_s": parse_s,
                **trace_ras_phases,
                "feedback_generation_s": trace_ras_feedback_s,
                "total_s": maya_trace_ras_s,
            },
        },
        "flexeva_refresh": {
            "plan": asdict(plan),
            "source_analysis_count": 1,
            "affected_event_count": len(affected_event_ids),
            "selected_trace": selected_summary["trace"],
            "feedback": selected_summary["feedback"],
            "phases_s": {
                **selected_phases,
                "source_hash_s": source_hash_s,
                "refresh_plan_s": plan_s,
                "selective_emulation_s": selective_emulation_s,
                "feedback_generation_s": selected_feedback_s,
                "total_s": refresh_s,
            },
        },
        "metrics": {
            "maya_full_s": maya_full_s,
            "maya_trace_ras_s": maya_trace_ras_s,
            "flexeva_refresh_s": refresh_s,
            "speedup_vs_maya_full": maya_full_s / max(refresh_s, 1.0e-12),
            "speedup_vs_maya_trace_ras": maya_trace_ras_s / max(refresh_s, 1.0e-12),
            "maya_trace_ras_event_reuse": 1.0 - (len(trace_ras_trace.events) / max(len(full_trace.events), 1)),
            "refresh_event_reuse": 1.0 - (len(selected_trace.events) / max(len(trace_ras_trace.events), 1)),
            "refresh_logical_event_reuse": 1.0 - (int(selected_trace.logical_event_count) / max(int(trace_ras_trace.logical_event_count), 1)),
        },
    }
    cleanup_traces(args, run)
    return row


def cumulative_table(anchor_init_s: float, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    running_full = running_trace = running_refresh = 0.0
    table: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        metrics = row["metrics"]
        refresh = row["flexeva_refresh"]
        case = row["case"]
        maya_full_s = float(metrics["maya_full_s"])
        maya_trace_ras_s = float(metrics["maya_trace_ras_s"])
        refresh_s = float(metrics["flexeva_refresh_s"])
        running_full += maya_full_s
        running_trace += maya_trace_ras_s
        running_refresh += refresh_s
        table.append(
            {
                "round": index,
                "x_label": row["label"],
                "setting": row["setting"],
                "mutation": row["mutation"],
                "tp": case["tp"],
                "pp": case["pp"],
                "dp": case["dp"],
                "changed_ops": "+".join(row["changed_ops"]),
                "maya_full_s": maya_full_s,
                "maya_trace_ras_s": maya_trace_ras_s,
                "flexeva_refresh_s": refresh_s,
                "speedup_vs_maya_full": float(metrics["speedup_vs_maya_full"]),
                "speedup_vs_maya_trace_ras": float(metrics["speedup_vs_maya_trace_ras"]),
                "amortized_speedup_vs_maya_full": running_full / max(anchor_init_s + running_refresh, 1.0e-12),
                "amortized_speedup_vs_maya_trace_ras": running_trace / max(anchor_init_s + running_refresh, 1.0e-12),
                "maya_trace_ras_event_reuse": float(metrics["maya_trace_ras_event_reuse"]),
                "refresh_event_reuse": float(metrics["refresh_event_reuse"]),
                "refresh_logical_event_reuse": float(metrics["refresh_logical_event_reuse"]),
                "raw_events": row["raw_events"],
                "maya_full_events": row["maya_full"]["trace"]["event_count"],
                "maya_trace_ras_events": row["maya_trace_ras"]["trace"]["event_count"],
                "selected_events": refresh["selected_trace"]["event_count"],
                "affected_trace_partitions": refresh["plan"]["affected_trace_partition_count"],
                "candidate_process_wall_s": row["run"]["elapsed_s"],
                "candidate_step_window_mean_s": row["step_window_s"]["mean"],
            }
        )
    return table


def write_outputs(args: argparse.Namespace, anchor_row: dict[str, object], rows: list[dict[str, object]]) -> None:
    anchor_init_s = float(anchor_row["metrics"]["anchor_init_s"])
    table = cumulative_table(anchor_init_s, rows)
    result = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": {
            "workload": "Megatron 2.7B 16-rank trace-shape workload through fake-CUDA frun",
            "physical_execution": "one guarded eight-A100 host; 16 modeled ranks map rank modulo 8 to local devices",
            "anchor": "Megatron 2.7B / 16 modeled ranks / TP1-PP8-DP2",
            "anchor_dir": "anchor",
            "candidate_selection": (
                "controlled cumulative source-region selections: attention backward; +MLP backward; "
                "+optimizer step; then TP2-PP8-DP1"
            ),
            "timing_boundary": (
                "core evaluator work; emulation uses marked training-step/selected-region windows and excludes "
                "process startup and teardown"
            ),
            "maya_full": (
                "Maya-style full: marked candidate emulation + JSONL parse + ordinary full-trace construction "
                "+ full replay + feedback serialization; local paper-aligned implementation"
            ),
            "maya_trace_ras": (
                "author ablation, Maya-style + FlexEva trace-RAS: marked candidate emulation + JSONL parse "
                "+ active-lane trace-RAS construction + full replay + feedback serialization; not an original Maya feature"
            ),
            "flexeva_refresh": (
                "source hash + source/config RAS refresh plan + marker-derived selective emulation "
                "+ selected trace filtering/replay + feedback serialization"
            ),
            "mutation_scope": (
                "R1-R3 change source-partition manifests while executing the same trace-shape workload; "
                "R4 also changes the modeled parallel configuration"
            ),
            "amortized": "sum of baseline candidate times / (anchor initialization + sum of refresh times)",
        },
        "anchor": anchor_row,
        "candidates": rows,
        "cumulative": table,
    }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    plot_fields = ["round", "x_label", "speedup_vs_maya_full", "speedup_vs_maya_trace_ras", "amortized_speedup_vs_maya_full", "amortized_speedup_vs_maya_trace_ras"]
    with (args.out_dir / "line_plot.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=plot_fields)
        writer.writeheader()
        for item in table:
            writer.writerow({key: item[key] for key in plot_fields})
    lines = [
        "# Megatron 2.7B / 16-rank Evaluator Core-Time Speedup",
        "",
        "Anchor: `TP1-PP8-DP2`. The 16 modeled ranks run on one guarded eight-A100 host. "
        "R1--R3 are controlled source-partition selections over the same trace-shape workload; R4 changes to `TP2-PP8-DP1`.",
        "Core time excludes process startup/teardown. `Maya-style + FlexEva trace-RAS` is an author ablation, not an original Maya feature.",
        "",
        "| Round | Candidate | Maya-style full (s) | Maya-style + FlexEva trace-RAS (s) | FlexEva refresh (s) | Speedup vs full | Speedup vs trace-RAS ablation | Amortized vs full | Amortized vs trace-RAS ablation | Refresh reuse |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in table:
        lines.append(
            f"| {item['round']} | {item['setting']} | {float(item['maya_full_s']):.3f} | "
            f"{float(item['maya_trace_ras_s']):.3f} | {float(item['flexeva_refresh_s']):.3f} | "
            f"{float(item['speedup_vs_maya_full']):.2f}x | {float(item['speedup_vs_maya_trace_ras']):.2f}x | "
            f"{float(item['amortized_speedup_vs_maya_full']):.2f}x | {float(item['amortized_speedup_vs_maya_trace_ras']):.2f}x | "
            f"{100.0 * float(item['refresh_event_reuse']):.2f}% |"
        )
    lines.extend(["", f"- Anchor initialization: `{anchor_init_s:.3f}s`", "- Use `line_plot.csv` for plotting."])
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    anchor, anchor_row, anchor_source_paths = measure_anchor(args)
    rows = [measure_candidate(args, anchor, candidate, anchor_source_paths) for candidate in CANDIDATES]
    write_outputs(args, anchor_row, rows)
    print(json.dumps({"result": str(args.out_dir / "result.json"), "summary": str(args.out_dir / "summary.csv"), "line_plot": str(args.out_dir / "line_plot.csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
