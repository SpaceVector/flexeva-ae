#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import flexmaya_ras as fm

from measure_routed_moe_fakecuda_similarity import (
    ROUTE_CASES,
    RouteCase,
    RoutedMoeConfig,
    load_step_window,
    raw_event_from_record,
    run_routed_moe_path,
)


SOURCE_OPS = ("router_backward", "attention_backward", "optimizer_step")
ROUTE_PARTITION = "route_path"
REGION_PARTITIONS = SOURCE_OPS + (ROUTE_PARTITION,)
BREAKDOWN_COMPONENTS = (
    "maya_emulation_s",
    "lower_trace_ras_compaction_s",
    "code_analysis_s",
    "source_ras_partition_update_s",
    "grounding_s",
    "selective_emulation_s",
    "trace_patching_collation_s",
    "event_simulation_s",
    "feedback_generation_s",
)

ANCHOR_ROUTE = ROUTE_CASES[0]
BOUNDARY_ROUTE = ROUTE_CASES[3]
MAX_STEP_BEGIN_SKEW_S = 1.0
FIGURE6_CONFIG = RoutedMoeConfig(
    backend="ns3",
    binary="extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default",
    world_size=16,
    ep_size=16,
    dp=1,
    steps=1,
    global_batch_size=128,
    seq_len=64,
    hidden_size=128,
    num_layers=32,
    num_heads=4,
    vocab_size=32000,
    num_experts=16,
    top_k=2,
    capacity_factor=1.25,
    micro_batches=8,
    dtype="bf16",
)


@dataclass(frozen=True)
class CandidateRound:
    name: str
    label: str
    route_case: RouteCase
    changed_ops: tuple[str, ...]
    mutation: str = "source"


CANDIDATES = (
    CandidateRound("round1_router_backward", "Router", ANCHOR_ROUTE, ("router_backward",)),
    CandidateRound(
        "round2_router_attention_backward",
        "Router+Attn",
        ANCHOR_ROUTE,
        ("router_backward", "attention_backward"),
    ),
    CandidateRound("round3_router_attention_optimizer", "Router+Attn+Opt", ANCHOR_ROUTE, SOURCE_OPS),
    CandidateRound(
        "round4_router_attention_optimizer_route",
        "Router+Attn+Opt+Route",
        BOUNDARY_ROUTE,
        SOURCE_OPS + (ROUTE_PARTITION,),
        mutation="source+route_path",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure real Routed-MoE evaluator speedup and evaluation-time breakdown."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--maya-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--proot", type=Path, required=True)
    parser.add_argument("--local-device-count", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=5200)
    parser.add_argument("--keep-raw-traces", action="store_true")
    parser.add_argument("--reuse-existing-traces", action="store_true")
    parser.add_argument(
        "--no-route-p2p-probe",
        action="store_true",
        help="Disable sparse P2P route probes. Enabled by default for path-aware routed-MoE runs.",
    )
    parser.set_defaults(source_region_markers=True, sync_before_step_window=True)
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


def marker_step_window_seconds(config: RoutedMoeConfig, trace_dir: Path) -> dict[str, object]:
    per_rank: dict[str, float] = {}
    for rank in range(config.world_size):
        start_ts, end_ts = load_step_window(trace_dir / f"rank_{rank}_markers.jsonl")
        per_rank[str(rank)] = max((end_ts - start_ts) / 1.0e6, 0.0)
    values = list(per_rank.values())
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values), "per_rank": per_rank}


def step_begin_skew_s(config: RoutedMoeConfig, trace_dir: Path) -> float:
    starts: list[int] = []
    for rank in range(config.world_size):
        with (trace_dir / f"rank_{rank}_markers.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("kind") == "step_begin" and record.get("label") == "training_step":
                    starts.append(int(record.get("realtime_ns") or record["monotonic_ns"]))
                    break
    if len(starts) != config.world_size:
        raise RuntimeError(f"expected {config.world_size} step_begin markers, found {len(starts)}")
    skew_s = (max(starts) - min(starts)) / 1e9
    if skew_s > MAX_STEP_BEGIN_SKEW_S:
        raise RuntimeError(f"rank step_begin skew {skew_s:.6f}s exceeds {MAX_STEP_BEGIN_SKEW_S:.1f}s")
    return skew_s


def load_region_windows(markers_path: Path, step_start: int, step_end: int) -> list[tuple[int, int, str]]:
    stacks: dict[str, list[int]] = {label: [] for label in REGION_PARTITIONS}
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


def parse_case_raw_events_with_regions(config: RoutedMoeConfig, trace_dir: Path) -> list[object]:
    rows: list[object] = []
    next_id = 1
    for rank in range(config.world_size):
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
                event = raw_event_from_record(record, config=config, rank=rank, event_id=next_id)
                while window_idx < len(windows) and ts > windows[window_idx][1]:
                    window_idx += 1
                if window_idx < len(windows):
                    start, end, label = windows[window_idx]
                    if start <= ts <= end:
                        event.code_partition = label
                rows.append(event)
                next_id += 1
    return rows


def selected_region_wall_time_s(trace_dir: Path, config: RoutedMoeConfig, selected_ops: tuple[str, ...]) -> float:
    selected = set(op for op in selected_ops if op in REGION_PARTITIONS)
    if not selected:
        return 0.0
    per_rank_s: list[float] = []
    for rank in range(config.world_size):
        markers_path = trace_dir / f"rank_{rank}_markers.jsonl"
        step_start, step_end = load_step_window(markers_path)
        windows = load_region_windows(markers_path, step_start, step_end)
        duration_us = sum(max(end - start, 0) for start, end, label in windows if label in selected)
        per_rank_s.append(duration_us / 1.0e6)
    return max(per_rank_s) if per_rank_s else 0.0


def write_source_files(
    case_dir: Path,
    *,
    config: RoutedMoeConfig,
    route_case: RouteCase,
    anchor_paths: dict[str, Path] | None = None,
    changed_ops: tuple[str, ...] = (),
) -> dict[str, Path]:
    source_dir = case_dir / "source_partitions"
    source_dir.mkdir(parents=True, exist_ok=True)
    anchor_paths = anchor_paths or {}
    changed = set(changed_ops)
    paths: dict[str, Path] = {}
    for rank in range(config.world_size):
        partition_id = f"expert_rank_{rank:03d}"
        if partition_id in anchor_paths:
            paths[partition_id] = anchor_paths[partition_id]
            continue
        path = source_dir / f"{partition_id}.py"
        path.write_text(
            f"# Routed-MoE expert-rank partition {rank}\nEXPERT_RANK = {rank}\n",
            encoding="utf-8",
        )
        paths[partition_id] = path
    for op in SOURCE_OPS:
        if op not in changed and op in anchor_paths:
            paths[op] = anchor_paths[op]
            continue
        suffix = "\n# source mutation\n" if op in changed else ""
        path = source_dir / f"{op}.py"
        path.write_text(f"# Routed-MoE source partition: {op}\nOP = {op!r}\n{suffix}", encoding="utf-8")
        paths[op] = path
    if ROUTE_PARTITION not in changed and ROUTE_PARTITION in anchor_paths:
        paths[ROUTE_PARTITION] = anchor_paths[ROUTE_PARTITION]
    else:
        suffix = "\n# route-path mutation\n" if ROUTE_PARTITION in changed else ""
        path = source_dir / f"{ROUTE_PARTITION}.py"
        path.write_text(
            "# Routed-MoE forced route path\n"
            f"PATH_ID = {route_case.path_id}\n"
            f"EXPERTS = {tuple(route_case.experts)!r}\n"
            f"LABEL = {route_case.label!r}\n"
            f"{suffix}",
            encoding="utf-8",
        )
        paths[ROUTE_PARTITION] = path
    return paths


def spec_for(
    config: RoutedMoeConfig,
    *,
    route_case: RouteCase,
    source_paths: dict[str, Path],
) -> fm.FlexMayaWorkloadSpec:
    partitions: list[fm.CodePartitionSpec] = []
    for rank in range(config.world_size):
        partition_id = f"expert_rank_{rank:03d}"
        partitions.append(
            fm.CodePartitionSpec(
                partition_id=partition_id,
                path=str(source_paths[partition_id]),
                active_ranks=(rank,),
            )
        )
    for op in SOURCE_OPS:
        partitions.append(fm.CodePartitionSpec(partition_id=op, path=str(source_paths[op])))
    partitions.append(
        fm.CodePartitionSpec(
            partition_id=ROUTE_PARTITION,
            path=str(source_paths[ROUTE_PARTITION]),
            requires_grounding=True,
        )
    )
    return fm.FlexMayaWorkloadSpec(
        workload_id=f"routed_moe_{route_case.name}",
        world_size=config.world_size,
        tp=1,
        pp=1,
        dp=config.dp,
        code_partitions=tuple(partitions),
        rank_group_policy="active_lane_set",
        notes=(
            "real workload/routed-moe/moe_topk.py source-region marker trace",
            f"route_experts={tuple(route_case.experts)!r}",
        ),
    )


def build_full(raw_events: list[object]) -> tuple[object, object, dict[str, float]]:
    build_s, trace = timed(lambda: fm.build_trace_ras(raw_events))
    replay_s, replay = timed(lambda: fm.replay_trace_once(trace))
    return trace, replay, {"trace_build_s": build_s, "python_replay_s": replay_s}


def build_trace_ras(
    config: RoutedMoeConfig,
    *,
    route_case: RouteCase,
    raw_events: list[object],
    source_paths: dict[str, Path],
) -> tuple[object, object, dict[str, float], fm.FlexMayaWorkloadSpec]:
    spec = spec_for(config, route_case=route_case, source_paths=source_paths)
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


def run_case(
    args: argparse.Namespace,
    config: RoutedMoeConfig,
    *,
    route_case: RouteCase,
    case_dir: Path,
    seed: int,
) -> dict[str, object]:
    case_dir.mkdir(parents=True, exist_ok=True)
    run = run_routed_moe_path(args, config, route_case=route_case, seed=seed, case_dir=case_dir)
    if int(run["return_code"]) != 0:
        raise RuntimeError(f"{route_case.name} failed; see {run['stdout']} and {run['stderr']}")
    return run


def cleanup_traces(args: argparse.Namespace, run: dict[str, object]) -> None:
    if args.keep_raw_traces:
        return
    for path in Path(str(run["trace_dir"])).glob("rank_*.jsonl"):
        if path.name.endswith("_markers.jsonl"):
            continue
        path.unlink(missing_ok=True)


def baseline_components(section: dict[str, object], *, feedback_payload: object) -> dict[str, float]:
    phases = section["phases_s"]
    components = empty_components()
    components.update(
        {
            "maya_emulation_s": float(phases.get("maya_emulation_s", 0.0)),
            "trace_patching_collation_s": float(phases.get("jsonl_parse_s", 0.0)),
            "lower_trace_ras_compaction_s": float(phases.get("trace_build_s", 0.0)),
            "event_simulation_s": float(phases.get("python_replay_s", 0.0)),
            "feedback_generation_s": feedback_generation_time(feedback_payload),
        }
    )
    return components


def anchor_components(anchor_row: dict[str, object]) -> dict[str, float]:
    phases = anchor_row["flexeva_anchor_init"]["phases_s"]
    components = empty_components()
    components.update(
        {
            "maya_emulation_s": float(phases.get("maya_emulation_s", 0.0)),
            "trace_patching_collation_s": float(phases.get("jsonl_parse_s", 0.0)),
            "lower_trace_ras_compaction_s": float(phases.get("trace_build_s", 0.0)),
            "code_analysis_s": float(phases.get("source_hash_s", 0.0)),
            "event_simulation_s": float(phases.get("python_replay_s", 0.0)),
            "feedback_generation_s": feedback_generation_time(anchor_row["flexeva_anchor_init"]),
        }
    )
    return components


def refresh_breakdown_components(
    candidate_row: dict[str, object],
    candidate: CandidateRound,
    *,
    config: RoutedMoeConfig,
) -> tuple[dict[str, float], dict[str, object]]:
    refresh = candidate_row["flexeva_refresh"]
    phases = refresh["phases_s"]
    code_analysis_s = float(phases.get("source_hash_s", 0.0))
    source_update_s = float(phases.get("refresh_plan_s", 0.0))
    route_changed = ROUTE_PARTITION in candidate.changed_ops
    if route_changed:
        grounding_s = selected_region_wall_time_s(
            Path(str(candidate_row["run"]["trace_dir"])),
            config,
            (ROUTE_PARTITION,),
        )
        selective_emulation_s = max(float(candidate_row["run"]["elapsed_s"]) - grounding_s, 0.0)
        selective_emulation_method = (
            "route-path round splits the full candidate fake-CUDA wall time into "
            "route_path grounding marker time and the remaining selective emulation time"
        )
    else:
        grounding_s = 0.0
        selective_emulation_s = selected_region_wall_time_s(
            Path(str(candidate_row["run"]["trace_dir"])),
            config,
            candidate.changed_ops,
        )
        selective_emulation_method = "max per-rank wall time of selected source-region marker windows"
    components = empty_components()
    components.update(
        {
            "code_analysis_s": code_analysis_s,
            "source_ras_partition_update_s": source_update_s,
            "grounding_s": grounding_s,
            "selective_emulation_s": selective_emulation_s,
            "trace_patching_collation_s": float(phases.get("trace_filter_s", 0.0)),
            "event_simulation_s": float(phases.get("python_replay_s", 0.0)),
            "feedback_generation_s": feedback_generation_time(refresh),
        }
    )
    return components, {"selective_emulation_method": selective_emulation_method}


def measure_anchor(
    args: argparse.Namespace,
    config: RoutedMoeConfig,
) -> tuple[fm.FlexMayaAnchor, dict[str, object], dict[str, Path]]:
    case_dir = args.out_dir / "anchor_route_0_1"
    source_paths = write_source_files(case_dir, config=config, route_case=ANCHOR_ROUTE)
    run = run_case(
        args,
        config,
        route_case=ANCHOR_ROUTE,
        case_dir=case_dir,
        seed=int(args.seed_base),
    )
    trace_dir = Path(str(run["trace_dir"]))
    step_window_s = marker_step_window_seconds(config, trace_dir)
    begin_skew_s = step_begin_skew_s(config, trace_dir)
    parse_s, raw_events = timed(lambda: parse_case_raw_events_with_regions(config, trace_dir))
    full_trace, full_replay, full_phases = build_full(raw_events)
    trace_ras_trace, trace_ras_replay, trace_ras_phases, spec = build_trace_ras(
        config,
        route_case=ANCHOR_ROUTE,
        raw_events=raw_events,
        source_paths=source_paths,
    )
    source_hash_s, source_hashes = timed(lambda: fm.source_hashes(spec))
    emulation_s = float(run["elapsed_s"])
    maya_full_s = emulation_s + parse_s + full_phases["trace_build_s"] + full_phases["python_replay_s"]
    init_s = emulation_s + parse_s + source_hash_s + trace_ras_phases["trace_build_s"] + trace_ras_phases["python_replay_s"]
    anchor = fm.FlexMayaAnchor(
        spec=spec,
        source_hashes=source_hashes,
        trace=trace_ras_trace,
        feedback=trace_ras_replay,
        summary={"trace": fm.trace_summary(trace_ras_trace), "feedback": trace_ras_replay.to_dict()},
    )
    row = {
        "route_case": asdict(ANCHOR_ROUTE),
        "config": asdict(config),
        "run": run,
        "step_window_s": step_window_s,
        "step_begin_skew_s": begin_skew_s,
        "raw_events": len(raw_events),
        "api_audit": {"cudaGetDevice_modeled_count": 0, "cudaGetDevice_replay_count": 0},
        "maya_full": {
            "trace": fm.trace_summary(full_trace),
            "feedback": full_replay.to_dict(),
            "phases_s": {
                "maya_emulation_s": emulation_s,
                "jsonl_parse_s": parse_s,
                **full_phases,
                "total_s": maya_full_s,
            },
        },
        "flexeva_anchor_init": {
            "trace": fm.trace_summary(trace_ras_trace),
            "feedback": trace_ras_replay.to_dict(),
            "phases_s": {
                "maya_emulation_s": emulation_s,
                "jsonl_parse_s": parse_s,
                "source_hash_s": source_hash_s,
                **trace_ras_phases,
                "total_s": init_s,
            },
        },
        "metrics": {"anchor_init_s": init_s, "anchor_init_over_maya_full": init_s / max(maya_full_s, 1.0e-12)},
    }
    cleanup_traces(args, run)
    return anchor, row, source_paths


def measure_candidate(
    args: argparse.Namespace,
    config: RoutedMoeConfig,
    anchor: fm.FlexMayaAnchor,
    candidate: CandidateRound,
    anchor_source_paths: dict[str, Path],
    *,
    round_index: int,
) -> dict[str, object]:
    case_dir = args.out_dir / candidate.name
    source_paths = write_source_files(
        case_dir,
        config=config,
        route_case=candidate.route_case,
        anchor_paths=anchor_source_paths,
        changed_ops=candidate.changed_ops,
    )
    run = run_case(
        args,
        config,
        route_case=candidate.route_case,
        case_dir=case_dir,
        seed=int(args.seed_base),
    )
    trace_dir = Path(str(run["trace_dir"]))
    step_window_s = marker_step_window_seconds(config, trace_dir)
    begin_skew_s = step_begin_skew_s(config, trace_dir)
    parse_s, raw_events = timed(lambda: parse_case_raw_events_with_regions(config, trace_dir))
    full_trace, full_replay, full_phases = build_full(raw_events)
    trace_ras_trace, trace_ras_replay, trace_ras_phases, spec = build_trace_ras(
        config,
        route_case=candidate.route_case,
        raw_events=raw_events,
        source_paths=source_paths,
    )
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
    emulation_s = float(run["elapsed_s"])
    maya_full_s = emulation_s + parse_s + full_phases["trace_build_s"] + full_phases["python_replay_s"]
    maya_trace_ras_s = emulation_s + parse_s + trace_ras_phases["trace_build_s"] + trace_ras_phases["python_replay_s"]
    refresh_s = source_hash_s + plan_s + selected_phases["trace_filter_s"] + selected_phases["python_replay_s"]
    row = {
        "candidate": candidate.name,
        "label": candidate.label,
        "mutation": candidate.mutation,
        "route_case": asdict(candidate.route_case),
        "config": asdict(config),
        "changed_ops": list(candidate.changed_ops),
        "setting": f"{candidate.label} ({tuple(candidate.route_case.experts)!r})",
        "run": run,
        "step_window_s": step_window_s,
        "step_begin_skew_s": begin_skew_s,
        "raw_events": len(raw_events),
        "api_audit": {"cudaGetDevice_modeled_count": 0, "cudaGetDevice_replay_count": 0},
        "maya_full": {
            "trace": fm.trace_summary(full_trace),
            "feedback": full_replay.to_dict(),
            "phases_s": {
                "maya_emulation_s": emulation_s,
                "jsonl_parse_s": parse_s,
                **full_phases,
                "total_s": maya_full_s,
            },
        },
        "maya_trace_ras": {
            "trace": fm.trace_summary(trace_ras_trace),
            "feedback": trace_ras_replay.to_dict(),
            "phases_s": {
                "maya_emulation_s": emulation_s,
                "jsonl_parse_s": parse_s,
                **trace_ras_phases,
                "total_s": maya_trace_ras_s,
            },
        },
        "flexeva_refresh": {
            "plan": asdict(plan),
            "source_analysis_count": 1,
            "affected_event_count": len(affected_event_ids),
            "selected_trace": fm.trace_summary(selected_trace),
            "feedback": selected_replay.to_dict(),
            "phases_s": {
                **selected_phases,
                "source_hash_s": source_hash_s,
                "refresh_plan_s": plan_s,
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
            "refresh_logical_event_reuse": 1.0
            - (int(selected_trace.logical_event_count) / max(int(trace_ras_trace.logical_event_count), 1)),
        },
    }
    cleanup_traces(args, run)
    return row


def cumulative_table(anchor_init_s: float, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    running_full = running_trace = running_refresh = 0.0
    table: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        metrics = row["metrics"]
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
                "route_experts": json.dumps(row["route_case"]["experts"]),
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
                "selected_events": row["flexeva_refresh"]["selected_trace"]["event_count"],
                "affected_trace_partitions": row["flexeva_refresh"]["plan"]["affected_trace_partition_count"],
                "candidate_emulation_s": row["run"]["elapsed_s"],
                "candidate_step_window_mean_s": row["step_window_s"]["mean"],
            }
        )
    return table


def write_wide(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["round", "x_label", "system", *BREAKDOWN_COMPONENTS, "total_s"]
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


def write_outputs(
    args: argparse.Namespace,
    config: RoutedMoeConfig,
    anchor_row: dict[str, object],
    rows: list[dict[str, object]],
) -> None:
    anchor_init_s = float(anchor_row["metrics"]["anchor_init_s"])
    table = cumulative_table(anchor_init_s, rows)
    anchor_init = anchor_components(anchor_row)
    per_round_rows: list[dict[str, object]] = []
    cumulative_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    cumulative_maya_full = empty_components()
    cumulative_maya_trace = empty_components()
    cumulative_flexeva = dict(anchor_init)

    for round_index, (candidate, row) in enumerate(zip(CANDIDATES, rows), start=1):
        maya_full = baseline_components(row["maya_full"], feedback_payload=row["maya_full"])
        maya_trace = baseline_components(row["maya_trace_ras"], feedback_payload=row["maya_trace_ras"])
        refresh, diagnostic = refresh_breakdown_components(row, candidate, config=config)

        for system, components in (
            ("Maya-full", maya_full),
            ("Maya-trace-RAS", maya_trace),
            ("FlexEva refresh", refresh),
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
            ("Maya-full", cumulative_maya_full),
            ("Maya-trace-RAS", cumulative_maya_trace),
            ("FlexEva cumulative", cumulative_flexeva),
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
                "route_case": row["route_case"],
                "refresh_event_reuse": row["metrics"]["refresh_event_reuse"],
                "affected_trace_partitions": row["flexeva_refresh"]["plan"]["affected_trace_partition_count"],
                **diagnostic,
            }
        )

    write_wide(args.out_dir / "breakdown_per_round_wide.csv", per_round_rows)
    write_wide(args.out_dir / "breakdown_cumulative_wide.csv", cumulative_rows)
    write_long(args.out_dir / "breakdown_cumulative_long.csv", cumulative_rows)

    result = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": {
            "workload": "workload/routed-moe/moe_topk.py through fake-CUDA frun",
            "physical_execution": "two guarded eight-A100 hosts; torchrun launches eight ranks per host",
            "config": asdict(config),
            "anchor": "forced top-2 expert route (0,1)",
            "candidate_selection": (
                "cumulative edits: router backward; +attention backward; "
                "+optimizer step; +forced route path (7,8)"
            ),
            "maya_full": "candidate fake-CUDA emulation wall time + JSONL parse + ordinary ungrouped trace build + full replay",
            "maya_trace_ras": (
                "candidate fake-CUDA emulation wall time + JSONL parse + singleton expert-rank "
                "trace-RAS build + replay_trace_once"
            ),
            "flexeva_refresh": "source/route RAS refresh plan + selected trace partition filter + selected replay",
            "breakdown": {
                "maya_emulation_s": "real fake-CUDA process wall time",
                "lower_trace_ras_compaction_s": "measured trace build time",
                "code_analysis_s": "single source hash analysis recorded by the evaluator",
                "source_ras_partition_update_s": "refresh-plan wall time using the recorded source hashes",
                "grounding_s": "router marker time for the route-path round; zero for pure source rounds",
                "selective_emulation_s": (
                    "source rounds use marker-derived selected-region wall time; "
                    "the route-path round uses the remainder of the full candidate fake-CUDA wall time "
                    "after route_path grounding"
                ),
                "trace_patching_collation_s": "JSONL parse for baselines; selected trace filter/patch for refresh",
                "event_simulation_s": "Python replay wall time",
                "feedback_generation_s": "JSON feedback serialization measured in this pass",
            },
            "amortized": "sum baseline candidate times divided by anchor initialization plus sum refresh times",
            "controlled_input": "anchor and candidates use one fixed seed; R1--R3 vary only source-manifest refresh scope",
            "source_analysis": "each candidate source manifest is hashed once and reused by refresh planning",
            "rank_synchronization": (
                "host-side Gloo barrier immediately before every measured step window; "
                f"maximum accepted step_begin skew is {MAX_STEP_BEGIN_SKEW_S:.1f} s"
            ),
        },
        "anchor": anchor_row,
        "candidates": rows,
        "cumulative": table,
        "anchor_init_components": anchor_init,
        "per_round_breakdown": per_round_rows,
        "cumulative_breakdown": cumulative_rows,
        "diagnostics": diagnostics,
    }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    plot_fields = [
        "round",
        "x_label",
        "speedup_vs_maya_full",
        "speedup_vs_maya_trace_ras",
        "amortized_speedup_vs_maya_full",
        "amortized_speedup_vs_maya_trace_ras",
    ]
    with (args.out_dir / "line_plot.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=plot_fields)
        writer.writeheader()
        for item in table:
            writer.writerow({key: item[key] for key in plot_fields})

    lines = [
        "# Routed-MoE Real Evaluation-Time Breakdown",
        "",
        "Anchor: forced top-2 expert route `(0,1)`. Candidates use cumulative source/route mutations: `Router`, `Router+Attn`, `Router+Attn+Opt`, and `Router+Attn+Opt+Route` with route `(7,8)`.",
        "",
        "| Round | Candidate | Maya-full (s) | Maya-trace-RAS (s) | FlexEva refresh (s) | Speedup vs Maya-full | Speedup vs Maya-trace-RAS | FlexEva cumulative (s) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in table:
        flex_total = next(
            row["total_s"]
            for row in cumulative_rows
            if row["round"] == item["round"] and row["system"] == "FlexEva cumulative"
        )
        lines.append(
            f"| {item['round']} | {item['setting']} | {float(item['maya_full_s']):.3f} | "
            f"{float(item['maya_trace_ras_s']):.3f} | {float(item['flexeva_refresh_s']):.3f} | "
            f"{float(item['speedup_vs_maya_full']):.2f}x | "
            f"{float(item['speedup_vs_maya_trace_ras']):.2f}x | {float(flex_total):.3f} |"
        )
    lines.extend(
        [
            "",
            f"- Anchor initialization: `{anchor_init_s:.3f}s`",
            "- `summary.csv` and `line_plot.csv` are for the end-to-end speedup plot.",
            "- `breakdown_cumulative_long.csv` is the stacked-bar input for evaluation-time breakdown.",
        ]
    )
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = FIGURE6_CONFIG
    if int(os.environ.get("FLEXMAYA_NODE_RANK", "0")) != 0:
        captures = [("anchor_route_0_1", ANCHOR_ROUTE), *((candidate.name, candidate.route_case) for candidate in CANDIDATES)]
        for directory, route_case in captures:
            run_case(
                args,
                config,
                route_case=route_case,
                case_dir=args.out_dir / directory,
                seed=int(args.seed_base),
            )
        print(json.dumps({"peer_node_rank": int(os.environ["FLEXMAYA_NODE_RANK"]), "captures": len(captures)}))
        return 0
    anchor, anchor_row, anchor_source_paths = measure_anchor(args, config)
    rows = [
        measure_candidate(args, config, anchor, candidate, anchor_source_paths, round_index=index)
        for index, candidate in enumerate(CANDIDATES, start=1)
    ]
    write_outputs(args, config, anchor_row, rows)
    print(
        json.dumps(
            {
                "result": str(args.out_dir / "result.json"),
                "summary": str(args.out_dir / "summary.csv"),
                "line_plot": str(args.out_dir / "line_plot.csv"),
                "breakdown": str(args.out_dir / "breakdown_cumulative_wide.csv"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
