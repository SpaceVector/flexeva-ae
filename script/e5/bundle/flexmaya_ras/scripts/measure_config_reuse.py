#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import flexmaya_ras as fm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure FlexMaya RAS reuse for parallel-strategy changes.")
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--anchor-tp", type=int, default=2)
    parser.add_argument("--anchor-pp", type=int, default=8)
    parser.add_argument("--anchor-dp", type=int, default=1)
    parser.add_argument("--candidate", action="append", default=None)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=64)
    parser.add_argument("--micro-batches", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--diagnostic-full-refresh-replay",
        action="store_true",
        help="Also replay the full refreshed compact trace for comparison. The default FlexEva path replays only selected events.",
    )
    return parser.parse_args()


def parse_candidate(value: str) -> tuple[int, int, int]:
    fields = tuple(int(part.strip()) for part in value.split(","))
    if len(fields) != 3:
        raise ValueError(f"candidate must be TP,PP,DP: {value!r}")
    return fields


def spec_for(args: argparse.Namespace, *, tp: int, pp: int, dp: int, label: str) -> fm.FlexMayaWorkloadSpec:
    return fm.FlexMayaWorkloadSpec(
        workload_id=label,
        world_size=args.world_size,
        tp=tp,
        pp=pp,
        dp=dp,
        code_partitions=tuple(
            fm.CodePartitionSpec(
                partition_id=f"layer_{layer:03d}",
                path=__file__,
                active_ranks=fm.megatron_pp_stage_active_ranks(args.world_size, tp, pp, layer % max(pp, 1)),
            )
            for layer in range(args.num_layers)
        ),
        rank_group_policy="active_lane_set",
        notes=("parallel-strategy reuse measurement",),
    )


def synthetic_events(
    args: argparse.Namespace,
    *,
    ranks: tuple[int, ...],
    tp: int,
    pp: int,
    dp: int,
) -> list[object]:
    rows: list[object] = []
    ts = 100
    # Use the physical layer id as code partition so strategy changes do not
    # look like source edits. Code events store active lane sets; the lower
    # trace materializes runtime events only for active ranked workers.
    del dp
    selected_ranks = set(ranks)
    for micro_batch in range(args.micro_batches):
        for layer in range(args.num_layers):
            stage = layer % max(pp, 1)
            partition = f"layer_{layer:03d}"
            active_ranks = tuple(
                rank
                for rank in fm.megatron_pp_stage_active_ranks(args.world_size, tp, pp, stage)
                if rank in selected_ranks
            )
            for rank in active_ranks:
                rows.append(
                    fm.make_event(
                        "cudaLaunchKernel",
                        "kernel_launch",
                        rank=rank,
                        stream=0,
                        timestamp_ns=ts,
                        count=args.hidden_size * args.seq_len,
                        code_partition=partition,
                    )
                )
                ts += 10
                rows.append(
                    fm.make_event(
                        "cublasGemmEx",
                        "blas_compute",
                        rank=rank,
                        stream=3,
                        timestamp_ns=ts,
                        count=args.hidden_size * args.hidden_size,
                        code_partition=partition,
                    )
                )
                ts += 10
            for group_idx, group_members in enumerate(
                fm.megatron_tp_groups_for_stage(args.world_size, tp, pp, stage)
            ):
                members = tuple(rank for rank in group_members if rank in selected_ranks)
                if not members:
                    continue
                group = f"tp{tp}:stage{stage}:layer{layer}:mb{micro_batch}:group{group_idx}"
                for rank in members:
                    rows.append(
                        fm.make_event(
                            "ncclAllReduce",
                            "nccl_collective",
                            rank=rank,
                            stream=3,
                            timestamp_ns=ts,
                            bytes=args.hidden_size * args.seq_len * 2,
                            count=args.hidden_size * args.seq_len,
                            collective_group=group,
                            code_partition=partition,
                        )
                    )
                    ts += 5
    return rows


def timed(label: str, fn):
    start = time.perf_counter()
    value = fn()
    return label, time.perf_counter() - start, value


def _rank_groups_for_spec(spec: fm.FlexMayaWorkloadSpec) -> dict[int, list[int]]:
    return fm.active_lane_rank_groups(spec)


def build_and_replay_maya_full(raw_events: list[object]) -> tuple[object, object, dict[str, float]]:
    phases: dict[str, float] = {}
    _label, phases["trace_build_s"], trace = timed("trace_build", lambda: fm.build_trace_ras(raw_events))
    _label, phases["python_replay_s"], replay = timed("python_replay", lambda: fm.replay_trace_once(trace))
    return trace, replay, phases


def build_and_replay_maya_trace_ras(spec: fm.FlexMayaWorkloadSpec, raw_events: list[object]) -> tuple[object, object, dict[str, float]]:
    phases: dict[str, float] = {}
    _label, phases["trace_build_s"], trace = timed(
        "trace_build",
        lambda: fm.build_rank_grouped_trace_ras(raw_events, _rank_groups_for_spec(spec)),
    )
    _label, phases["python_replay_s"], replay = timed("python_replay", lambda: fm.replay_trace_once(trace))
    return trace, replay, phases


def build_maya_trace_ras(spec: fm.FlexMayaWorkloadSpec, raw_events: list[object]) -> tuple[object, dict[str, float]]:
    phases: dict[str, float] = {}
    _label, phases["trace_build_s"], trace = timed(
        "trace_build",
        lambda: fm.build_rank_grouped_trace_ras(raw_events, _rank_groups_for_spec(spec)),
    )
    return trace, phases


def trace_window_count(trace: object) -> int:
    return sum(1 for partition in trace.sync_partitions if str(partition.kind) == "trace_window")


def event_ids_for_partitions(trace: object, partition_ids: tuple[int, ...]) -> tuple[int, ...]:
    selected = set(partition_ids)
    event_ids: set[int] = set()
    for partition in trace.sync_partitions:
        if int(partition.id) in selected:
            event_ids.update(int(event_id) for event_id in partition.event_ids)
    return tuple(sorted(event_ids))


def selected_refresh_replay(trace: object, plan: fm.FlexMayaRefreshPlan) -> tuple[object, object, dict[str, float]]:
    phases: dict[str, float] = {}
    _label, phases["trace_filter_s"], selected_trace = timed(
        "trace_filter",
        lambda: fm.filter_trace_partitions(trace, list(plan.affected_trace_partitions)),
    )
    _label, phases["python_replay_s"], replay = timed("python_replay", lambda: fm.replay_trace_once(selected_trace))
    return selected_trace, replay, phases


def refresh_plan_summary(plan: fm.FlexMayaRefreshPlan, trace: object) -> dict[str, object]:
    affected_count = (
        plan.affected_trace_partition_count
        if plan.affected_trace_partition_count is not None
        else len(plan.affected_trace_partitions)
    )
    affected_event_ids = event_ids_for_partitions(trace, plan.affected_trace_partitions)
    windows = trace_window_count(trace)
    return {
        "changed_partitions": list(plan.changed_partitions),
        "configuration_changed": plan.configuration_changed,
        "refresh_scope": plan.refresh_scope,
        "affected_rank_groups": list(plan.affected_rank_groups),
        "affected_rank_group_count": len(plan.affected_rank_groups),
        "affected_trace_partition_count": affected_count,
        "affected_trace_partition_sample": list(plan.affected_trace_partitions[:16]),
        "affected_trace_event_count": len(affected_event_ids),
        "trace_window_count": windows,
        "trace_partition_reuse_ratio": 1.0 - (affected_count / max(windows, 1)),
        "trace_event_reuse_ratio": 1.0 - (len(affected_event_ids) / max(len(trace.events), 1)),
        "fallback_reasons": list(plan.fallback_reasons),
    }


def code_ras_summary(spec: fm.FlexMayaWorkloadSpec) -> dict[str, object]:
    return {
        "code_event_count": len(spec.code_partitions),
        "active_lane_memberships": sum(len(partition.active_ranks) for partition in spec.code_partitions),
        "active_lane_sets": {
            partition.partition_id: list(partition.active_ranks)
            for partition in spec.code_partitions[:8]
        },
    }


def code_reuse_summary(anchor_spec: fm.FlexMayaWorkloadSpec, candidate_spec: fm.FlexMayaWorkloadSpec) -> dict[str, object]:
    anchor = {partition.partition_id: set(partition.active_ranks) for partition in anchor_spec.code_partitions}
    candidate = {partition.partition_id: set(partition.active_ranks) for partition in candidate_spec.code_partitions}
    shared_ids = set(anchor) & set(candidate)
    candidate_memberships = sum(len(ranks) for ranks in candidate.values())
    reused_memberships = sum(len(anchor[partition_id] & candidate[partition_id]) for partition_id in shared_ids)
    changed_active_sets = sum(
        1
        for partition_id in shared_ids
        if anchor[partition_id] != candidate[partition_id]
    )
    return {
        "code_event_count": len(candidate),
        "code_event_object_reuse_ratio": len(shared_ids) / max(len(candidate), 1),
        "active_lane_membership_reuse_ratio": reused_memberships / max(candidate_memberships, 1),
        "changed_active_lane_set_count": changed_active_sets + len(set(candidate) - shared_ids),
    }


def measure_one(args: argparse.Namespace, anchor: fm.FlexMayaAnchor, candidate: tuple[int, int, int]) -> dict[str, object]:
    tp, pp, dp = candidate
    spec = spec_for(args, tp=tp, pp=pp, dp=dp, label=f"candidate_tp{tp}_pp{pp}_dp{dp}")
    groups = _rank_groups_for_spec(spec)
    representative_ranks = tuple(sorted(groups))

    _label, full_gen_s, full_raw = timed(
        "maya_full_raw_generation",
        lambda: synthetic_events(args, ranks=tuple(range(args.world_size)), tp=tp, pp=pp, dp=dp),
    )
    maya_full_trace, maya_full_replay, maya_full_phases = build_and_replay_maya_full(full_raw)
    maya_trace_ras_trace, maya_trace_ras_replay, maya_trace_ras_phases = build_and_replay_maya_trace_ras(spec, full_raw)

    _label, refresh_gen_s, refresh_raw = timed(
        "flexeva_representative_raw_generation",
        lambda: synthetic_events(args, ranks=representative_ranks, tp=tp, pp=pp, dp=dp),
    )
    refresh_trace, refresh_phases = build_maya_trace_ras(spec, refresh_raw)
    _label, plan_s, refresh_plan = timed(
        "refresh_plan",
        lambda: fm.plan_candidate_refresh(anchor, spec, refresh_trace),
    )
    selected_trace, selected_replay, selected_phases = selected_refresh_replay(refresh_trace, refresh_plan)
    full_refresh_replay = None
    full_refresh_replay_s = 0.0
    if args.diagnostic_full_refresh_replay:
        _label, full_refresh_replay_s, full_refresh_replay = timed(
            "diagnostic_full_refresh_python_replay",
            lambda: fm.replay_trace_once(refresh_trace),
        )

    maya_full_total = full_gen_s + maya_full_phases["trace_build_s"] + maya_full_phases["python_replay_s"]
    maya_trace_ras_total = full_gen_s + maya_trace_ras_phases["trace_build_s"] + maya_trace_ras_phases["python_replay_s"]
    refresh_build_total = refresh_gen_s + refresh_phases["trace_build_s"] + plan_s
    full_refresh_total = refresh_build_total + full_refresh_replay_s if full_refresh_replay is not None else None
    selected_total = (
        refresh_gen_s
        + refresh_phases["trace_build_s"]
        + plan_s
        + selected_phases["trace_filter_s"]
        + selected_phases["python_replay_s"]
    )
    code_reuse = code_reuse_summary(anchor.spec, spec)
    plan_summary = refresh_plan_summary(refresh_plan, refresh_trace)
    row = {
        "candidate": {"tp": tp, "pp": pp, "dp": dp},
        "code_ras": code_reuse,
        "representative_ranks": representative_ranks,
        "profiled_rank_count": len(representative_ranks),
        "world_size": args.world_size,
        "refresh_plan": plan_summary,
        "maya_full": {
            "raw_events": len(full_raw),
            "trace": fm.trace_summary(maya_full_trace),
            "feedback": maya_full_replay.to_dict(),
            "phases_s": {"raw_generation_s": full_gen_s, **maya_full_phases, "total_s": maya_full_total},
        },
        "maya_trace_ras": {
            "raw_events": len(full_raw),
            "trace": fm.trace_summary(maya_trace_ras_trace),
            "feedback": maya_trace_ras_replay.to_dict(),
            "phases_s": {"raw_generation_s": full_gen_s, **maya_trace_ras_phases, "total_s": maya_trace_ras_total},
            "rank_groups_from_active_lane_sets": _rank_groups_for_spec(spec),
        },
        "flexeva_refresh": {
            "raw_events": len(refresh_raw),
            "trace": fm.trace_summary(refresh_trace),
            "mode": "selected_replay_default",
            "diagnostic_full_feedback": full_refresh_replay.to_dict() if full_refresh_replay is not None else None,
            "phases_s": {
                "raw_generation_s": refresh_gen_s,
                **refresh_phases,
                "refresh_plan_s": plan_s,
                "diagnostic_full_python_replay_s": full_refresh_replay_s if full_refresh_replay is not None else None,
                "diagnostic_full_total_s": full_refresh_total,
                "total_s": refresh_build_total,
            },
        },
        "flexeva_selected_event_replay": {
            "selected_trace": fm.trace_summary(selected_trace),
            "feedback": selected_replay.to_dict(),
            "phases_s": {
                "raw_generation_s": refresh_gen_s,
                "trace_build_s": refresh_phases["trace_build_s"],
                "refresh_plan_s": plan_s,
                **selected_phases,
                "total_s": selected_total,
            },
        },
        "benefit": {
            "code_ras_reuse_ratio": code_reuse["code_event_object_reuse_ratio"],
            "active_lane_membership_reuse": code_reuse["active_lane_membership_reuse_ratio"],
            "maya_trace_ras_event_reuse": 1.0 - (len(maya_trace_ras_trace.events) / max(len(maya_full_trace.events), 1)),
            "maya_trace_ras_speedup_vs_maya_full": maya_full_total / max(maya_trace_ras_total, 1.0e-12),
            "profiled_rank_reduction": 1.0 - (len(representative_ranks) / args.world_size),
            "raw_event_reduction": 1.0 - (len(refresh_raw) / max(len(full_raw), 1)),
            "compact_event_reduction_vs_maya_trace_ras": 1.0 - (len(refresh_trace.events) / max(len(maya_trace_ras_trace.events), 1)),
            "trace_partition_reuse_ratio": plan_summary["trace_partition_reuse_ratio"],
            "trace_event_reuse_ratio": plan_summary["trace_event_reuse_ratio"],
            "selected_event_reduction_vs_refresh_trace": 1.0 - (len(selected_trace.events) / max(len(refresh_trace.events), 1)),
            "flexeva_full_replay_speedup_vs_maya_full": (
                maya_full_total / max(full_refresh_total, 1.0e-12) if full_refresh_total is not None else None
            ),
            "flexeva_full_replay_speedup_vs_maya_trace_ras": (
                maya_trace_ras_total / max(full_refresh_total, 1.0e-12) if full_refresh_total is not None else None
            ),
            "flexeva_speedup_vs_maya_full": maya_full_total / max(selected_total, 1.0e-12),
            "flexeva_speedup_vs_maya_trace_ras": maya_trace_ras_total / max(selected_total, 1.0e-12),
            "flexeva_selected_event_speedup_vs_maya_full": maya_full_total / max(selected_total, 1.0e-12),
            "flexeva_selected_event_speedup_vs_maya_trace_ras": maya_trace_ras_total / max(selected_total, 1.0e-12),
            "candidate_speedup_vs_cold_full": maya_full_total / max(selected_total, 1.0e-12),
        },
    }
    row["maya_dedup"] = row["maya_trace_ras"]
    row["benefit"]["maya_rank_dedup_event_reuse"] = row["benefit"]["maya_trace_ras_event_reuse"]
    row["benefit"]["maya_dedup_speedup_vs_maya_full"] = row["benefit"]["maya_trace_ras_speedup_vs_maya_full"]
    row["benefit"]["compact_event_reduction_vs_maya_dedup"] = row["benefit"]["compact_event_reduction_vs_maya_trace_ras"]
    row["benefit"]["flexeva_speedup_vs_maya_dedup"] = row["benefit"]["flexeva_speedup_vs_maya_trace_ras"]
    return row


def main() -> int:
    args = parse_args()
    if args.candidate is None:
        args.candidate = ["4,4,1", "1,16,1", "2,4,2"]
    out_dir = args.out_dir or (
        Path(__file__).resolve().parents[2]
        / "output"
        / f"flexmaya_ras_config_reuse_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor_spec = spec_for(
        args,
        tp=args.anchor_tp,
        pp=args.anchor_pp,
        dp=args.anchor_dp,
        label=f"anchor_tp{args.anchor_tp}_pp{args.anchor_pp}_dp{args.anchor_dp}",
    )
    _label, anchor_gen_s, anchor_raw = timed(
        "anchor_raw_generation",
        lambda: synthetic_events(
            args,
            ranks=tuple(range(args.world_size)),
            tp=args.anchor_tp,
            pp=args.anchor_pp,
            dp=args.anchor_dp,
        ),
    )
    anchor_maya_full_trace, anchor_maya_full_replay, anchor_maya_full_phases = build_and_replay_maya_full(anchor_raw)
    anchor_trace, anchor_replay, anchor_phases = build_and_replay_maya_trace_ras(anchor_spec, anchor_raw)
    _label, source_hash_s, anchor_source_hashes = timed("source_hashes", lambda: fm.source_hashes(anchor_spec))
    anchor = fm.FlexMayaAnchor(
        spec=anchor_spec,
        source_hashes=anchor_source_hashes,
        trace=anchor_trace,
        feedback=anchor_replay,
        summary={
            "kind": "anchor_init_measurement",
            "trace": fm.trace_summary(anchor_trace),
            "feedback": anchor_replay.to_dict(),
        },
    )
    anchor_maya_full_total = (
        anchor_gen_s + anchor_maya_full_phases["trace_build_s"] + anchor_maya_full_phases["python_replay_s"]
    )
    anchor_total = anchor_gen_s + source_hash_s + anchor_phases["trace_build_s"] + anchor_phases["python_replay_s"]
    rows = [measure_one(args, anchor, parse_candidate(candidate)) for candidate in args.candidate]
    result = {
        "anchor": {
            "config": {"tp": args.anchor_tp, "pp": args.anchor_pp, "dp": args.anchor_dp},
            "code_ras": code_ras_summary(anchor_spec),
            "ras_layers": {
                "maya_trace_ras": "lower trace/event RAS; active-lane-set dedup and sync/collective partitions",
                "flexeva_source_ras": "upper source/code RAS; code partitions, AL edges, and selective refresh",
            },
            "raw_events": len(anchor_raw),
            "maya_full": {
                "trace": fm.trace_summary(anchor_maya_full_trace),
                "feedback": anchor_maya_full_replay.to_dict(),
                "phases_s": {
                    "raw_generation_s": anchor_gen_s,
                    **anchor_maya_full_phases,
                    "total_s": anchor_maya_full_total,
                },
            },
            "flexeva_anchor": {
                "trace": fm.trace_summary(anchor_trace),
                "feedback": anchor_replay.to_dict(),
                "phases_s": {
                    "raw_generation_s": anchor_gen_s,
                    "code_ras_hash_s": source_hash_s,
                    **anchor_phases,
                    "total_s": anchor_total,
                },
            },
            "trace": fm.trace_summary(anchor_trace),
            "feedback": anchor_replay.to_dict(),
            "phases_s": {"raw_generation_s": anchor_gen_s, "code_ras_hash_s": source_hash_s, **anchor_phases, "total_s": anchor_total},
            "comparison": {
                "maya_trace_ras_event_reuse": 1.0
                - (len(anchor_trace.events) / max(len(anchor_maya_full_trace.events), 1)),
                "maya_rank_dedup_event_reuse": 1.0
                - (len(anchor_trace.events) / max(len(anchor_maya_full_trace.events), 1)),
                "flexeva_anchor_overhead_vs_maya_full": anchor_total / max(anchor_maya_full_total, 1.0e-12),
            },
        },
        "candidates": rows,
    }
    path = out_dir / "result.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(path), **result}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
