"""High-level FlexMaya RAS orchestration API.

The C++ layer owns the hot path: fake-CUDA hook memory, lane placement,
collation, sync partitions, and the lower Maya trace-RAS.  This module owns the
FlexEva-facing source RAS: code partitions, AL metadata, refresh planning, and
the single Python replay used for prediction and feedback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Mapping

from ._flexmaya_ras import (
    RawEvent,
    build_deduplicated_trace_ras,
    build_rank_grouped_selected_trace_ras,
    build_rank_grouped_trace_ras,
    build_trace_ras,
    filter_trace_partitions,
)
from .simulator import ReplayReport, replay_trace_once


@dataclass(frozen=True)
class CodePartitionSpec:
    partition_id: str
    path: str
    start_line: int | None = None
    end_line: int | None = None
    active_ranks: tuple[int, ...] = ()
    boundary_marker: str | None = None
    requires_grounding: bool = False


@dataclass(frozen=True)
class SourceHash:
    path: str
    sha1: str
    line_count: int


@dataclass(frozen=True)
class FlexMayaWorkloadSpec:
    workload_id: str
    world_size: int
    tp: int = 1
    pp: int = 1
    dp: int = 1
    included_files: tuple[str, ...] = ()
    code_partitions: tuple[CodePartitionSpec, ...] = ()
    rank_group_policy: str = "active_lane_set"
    launch_command: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FlexMayaRefreshPlan:
    changed_partitions: tuple[str, ...]
    affected_rank_groups: tuple[int, ...]
    affected_trace_partitions: tuple[int, ...]
    configuration_changed: bool = False
    affected_trace_partition_count: int | None = None
    refresh_scope: str = "selected"
    fallback_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class FlexMayaAnchor:
    spec: FlexMayaWorkloadSpec
    source_hashes: tuple[SourceHash, ...]
    trace: object
    feedback: ReplayReport
    summary: Mapping[str, object]


@dataclass(frozen=True)
class FlexMayaAnchorContext:
    spec: FlexMayaWorkloadSpec
    source_hashes: tuple[SourceHash, ...]
    rank_groups: Mapping[int, tuple[int, ...]]
    summary: Mapping[str, object]


@dataclass(frozen=True)
class FlexMayaEvaluationReport:
    anchor: FlexMayaAnchor
    candidate_spec: FlexMayaWorkloadSpec
    refresh_plan: FlexMayaRefreshPlan
    trace: object
    feedback: ReplayReport
    selected_trace: object
    selected_feedback: ReplayReport
    summary: Mapping[str, object]


def _read_selected_text(path: Path, start_line: int | None, end_line: int | None) -> str:
    text = path.read_text(encoding="utf-8")
    if start_line is None and end_line is None:
        return text
    lines = text.splitlines()
    start = 1 if start_line is None else max(start_line, 1)
    end = len(lines) if end_line is None else max(end_line, start)
    return "\n".join(lines[start - 1:end]) + "\n"


def source_hashes(spec: FlexMayaWorkloadSpec) -> tuple[SourceHash, ...]:
    selected: list[SourceHash] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    for path_text in spec.included_files:
        path = Path(path_text).resolve()
        key = (str(path), None, None)
        if key in seen:
            continue
        seen.add(key)
        text = _read_selected_text(path, None, None)
        selected.append(
            SourceHash(path=str(path), sha1=hashlib.sha1(text.encode("utf-8")).hexdigest(), line_count=len(text.splitlines()))
        )
    for partition in spec.code_partitions:
        path = Path(partition.path).resolve()
        key = (str(path), partition.start_line, partition.end_line)
        if key in seen:
            continue
        seen.add(key)
        text = _read_selected_text(path, partition.start_line, partition.end_line)
        selected.append(
            SourceHash(path=str(path), sha1=hashlib.sha1(text.encode("utf-8")).hexdigest(), line_count=len(text.splitlines()))
        )
    return tuple(selected)


def megatron_pp_stage_rank_groups(world_size: int, tp: int, pp: int) -> dict[int, list[int]]:
    model_parallel = tp * pp
    if model_parallel <= 0 or world_size % model_parallel != 0:
        raise ValueError(f"invalid Megatron rank geometry: world_size={world_size}, tp={tp}, pp={pp}")
    dp = world_size // model_parallel
    groups: dict[int, list[int]] = {}
    for pp_rank in range(pp):
        members: list[int] = []
        for dp_rank in range(dp):
            for tp_rank in range(tp):
                rank = dp_rank * model_parallel + tp_rank * pp + pp_rank
                members.append(rank)
        groups[min(members)] = sorted(members)
    return groups


def megatron_pp_stage_active_ranks(world_size: int, tp: int, pp: int, pp_rank: int) -> tuple[int, ...]:
    model_parallel = tp * pp
    if model_parallel <= 0 or world_size % model_parallel != 0:
        raise ValueError(f"invalid Megatron rank geometry: world_size={world_size}, tp={tp}, pp={pp}")
    if pp <= 0:
        raise ValueError(f"invalid pipeline parallelism: pp={pp}")
    stage = pp_rank % pp
    dp = world_size // model_parallel
    ranks: list[int] = []
    for dp_rank in range(dp):
        for tp_rank in range(tp):
            ranks.append(dp_rank * model_parallel + tp_rank * pp + stage)
    return tuple(sorted(ranks))


def megatron_tp_groups_for_stage(world_size: int, tp: int, pp: int, pp_rank: int) -> tuple[tuple[int, ...], ...]:
    model_parallel = tp * pp
    if model_parallel <= 0 or world_size % model_parallel != 0:
        raise ValueError(f"invalid Megatron rank geometry: world_size={world_size}, tp={tp}, pp={pp}")
    if pp <= 0:
        raise ValueError(f"invalid pipeline parallelism: pp={pp}")
    stage = pp_rank % pp
    dp = world_size // model_parallel
    groups: list[tuple[int, ...]] = []
    for dp_rank in range(dp):
        members = tuple(dp_rank * model_parallel + tp_rank * pp + stage for tp_rank in range(tp))
        groups.append(members)
    return tuple(groups)


def active_lane_rank_groups(spec: FlexMayaWorkloadSpec) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for partition in spec.code_partitions:
        ranks = tuple(sorted(dict.fromkeys(int(rank) for rank in partition.active_ranks)))
        if not ranks:
            continue
        representative = ranks[0]
        current = groups.get(representative)
        if current is not None and tuple(current) != ranks:
            raise ValueError(
                "active-lane-set dedup requires one active set per representative rank; "
                f"rank {representative} appears in both {tuple(current)} and {ranks}"
            )
        groups[representative] = list(ranks)
    if groups:
        return dict(sorted(groups.items()))
    return megatron_pp_stage_rank_groups(spec.world_size, spec.tp, spec.pp)


def _build_trace(spec: FlexMayaWorkloadSpec, raw_events: Iterable[RawEvent]):
    raw = tuple(raw_events)
    if spec.rank_group_policy == "none":
        return build_trace_ras(raw)
    if spec.rank_group_policy == "pattern":
        return build_deduplicated_trace_ras(raw)
    if spec.rank_group_policy == "active_lane_set":
        return build_rank_grouped_trace_ras(raw, active_lane_rank_groups(spec))
    if spec.rank_group_policy == "megatron_pp_stage":
        return build_rank_grouped_trace_ras(raw, megatron_pp_stage_rank_groups(spec.world_size, spec.tp, spec.pp))
    raise ValueError(f"unsupported rank_group_policy={spec.rank_group_policy!r}")


def build_selected_trace(
    spec: FlexMayaWorkloadSpec,
    raw_events: Iterable[RawEvent],
    *,
    selected_ranks: Iterable[int] = (),
    selected_code_partitions: Iterable[str] = (),
    rank_groups: Mapping[int, Iterable[int]] | None = None,
):
    """Build a lower trace directly from selected representative events.

    This is the FlexEva refresh hot path: AL/code analysis identifies rank
    groups and code partitions, then C++ filters and places only those raw
    events before the single Python replay.
    """
    groups = {
        int(rep): [int(rank) for rank in ranks]
        for rep, ranks in (rank_groups or active_lane_rank_groups(spec)).items()
    }
    return build_rank_grouped_selected_trace_ras(
        tuple(raw_events),
        groups,
        [int(rank) for rank in selected_ranks],
        [str(partition) for partition in selected_code_partitions],
    )


def trace_summary(trace: object) -> dict[str, object]:
    return {
        "event_count": len(trace.events),
        "logical_event_count": int(trace.logical_event_count),
        "lane_count": len(trace.lanes),
        "edge_count": len(trace.edges),
        "sync_partition_count": len(trace.sync_partitions),
        "deduplicated": bool(trace.deduplicated),
        "dedup_group_count": len(trace.dedup_groups),
        "dedup_groups": [
            {
                "id": int(group.id),
                "representative_rank": int(group.representative_rank),
                "ranks": list(group.ranks),
                "representative_event_count": int(group.representative_event_count),
                "logical_event_count": int(group.logical_event_count),
            }
            for group in trace.dedup_groups
        ],
        "lineage_edge_count": len(trace.lineage_edges),
    }


def init_anchor(spec: FlexMayaWorkloadSpec, raw_events: Iterable[RawEvent]) -> FlexMayaAnchor:
    trace = _build_trace(spec, raw_events)
    feedback = replay_trace_once(trace)
    summary = {
        "kind": "anchor_init",
        "spec": asdict(spec),
        "trace": trace_summary(trace),
        "feedback": feedback.to_dict(),
    }
    return FlexMayaAnchor(
        spec=spec,
        source_hashes=source_hashes(spec),
        trace=trace,
        feedback=feedback,
        summary=summary,
    )


def anchor_context(
    anchor: FlexMayaAnchor,
    rank_groups: Mapping[int, Iterable[int]] | None = None,
) -> FlexMayaAnchorContext:
    rank_groups = {
        int(rep): tuple(int(rank) for rank in ranks)
        for rep, ranks in (rank_groups or active_lane_rank_groups(anchor.spec)).items()
    }
    return FlexMayaAnchorContext(
        spec=anchor.spec,
        source_hashes=anchor.source_hashes,
        rank_groups=rank_groups,
        summary={
            "kind": "anchor_context",
            "workload_id": anchor.spec.workload_id,
            "rank_group_count": len(rank_groups),
            "rank_groups": {str(rep): list(ranks) for rep, ranks in rank_groups.items()},
            "trace": trace_summary(anchor.trace),
            "feedback": anchor.feedback.to_dict(),
        },
    )


def _changed_partitions(anchor: FlexMayaAnchor, candidate: FlexMayaWorkloadSpec) -> tuple[str, ...]:
    before = {(item.path, item.line_count): item.sha1 for item in anchor.source_hashes}
    after_hashes = source_hashes(candidate)
    after = {(item.path, item.line_count): item.sha1 for item in after_hashes}
    if before == after:
        return ()
    changed: list[str] = []
    for partition in candidate.code_partitions:
        path = str(Path(partition.path).resolve())
        text = _read_selected_text(Path(partition.path).resolve(), partition.start_line, partition.end_line)
        sha1 = hashlib.sha1(text.encode("utf-8")).hexdigest()
        matched = [
            old_hash
            for (old_path, _line_count), old_hash in before.items()
            if old_path == path
        ]
        if not matched or sha1 not in matched:
            changed.append(partition.partition_id)
    if not changed and before != after:
        changed.append("__file_scope__")
    return tuple(dict.fromkeys(changed))


def _affected_trace_partitions(trace: object, changed_partitions: tuple[str, ...]) -> tuple[int, ...]:
    if not changed_partitions:
        return ()
    changed = set(changed_partitions)
    trace_partition_ids = {int(partition.id) for partition in trace.sync_partitions}
    touched = {
        int(edge.lower_partition)
        for edge in trace.lineage_edges
        if str(edge.upper_partition) in changed
        and str(edge.edge_kind) == "code_to_trace_partition"
        and int(edge.lower_partition) in trace_partition_ids
    }
    return tuple(sorted(touched))


def _event_ids_in_trace_partitions(trace: object, partition_ids: tuple[int, ...]) -> tuple[int, ...]:
    selected = set(partition_ids)
    event_ids: set[int] = set()
    for partition in trace.sync_partitions:
        if int(partition.id) not in selected:
            continue
        event_ids.update(int(event_id) for event_id in partition.event_ids)
    return tuple(sorted(event_ids))


def _affected_groups_from_events(trace: object, event_ids: tuple[int, ...]) -> tuple[int, ...]:
    if not event_ids:
        return ()
    selected = set(event_ids)
    groups = {
        int(event.dedup_group_id)
        for event in trace.events
        if int(event.id) in selected and int(event.dedup_group_id) != 0
    }
    return tuple(sorted(groups))


def _code_partition_active_rank_map(spec: FlexMayaWorkloadSpec) -> dict[str, tuple[int, ...]]:
    return {
        partition.partition_id: tuple(sorted(dict.fromkeys(int(rank) for rank in partition.active_ranks)))
        for partition in spec.code_partitions
    }


def _trace_window_partitions(trace: object) -> tuple[object, ...]:
    return tuple(partition for partition in trace.sync_partitions if str(partition.kind) == "trace_window")


def _event_is_config_dependent(event: object) -> bool:
    kind = str(getattr(event, "kind", ""))
    api = str(getattr(event, "api", "")).lower()
    if kind == "nccl_collective":
        return True
    communication_tokens = (
        "nccl",
        "allreduce",
        "all_reduce",
        "alltoall",
        "all_to_all",
        "allgather",
        "all_gather",
        "reduce_scatter",
        "broadcast",
        "send",
        "recv",
        "isend",
        "irecv",
    )
    return any(token in api for token in communication_tokens)


def _config_affected_trace_partitions(
    anchor_spec: FlexMayaWorkloadSpec,
    candidate_spec: FlexMayaWorkloadSpec,
    trace: object,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    anchor_active = _code_partition_active_rank_map(anchor_spec)
    candidate_active = _code_partition_active_rank_map(candidate_spec)
    event_by_id = {int(event.id): event for event in trace.events}
    selected: set[int] = set()
    missing_active_metadata: set[str] = set()
    changed_active_sets: set[str] = set()
    communication_window_count = 0

    for partition in _trace_window_partitions(trace):
        code_partition = str(partition.code_partition)
        if not code_partition:
            continue
        anchor_ranks = anchor_active.get(code_partition)
        candidate_ranks = candidate_active.get(code_partition)
        if not anchor_ranks or not candidate_ranks:
            missing_active_metadata.add(code_partition)
            selected.add(int(partition.id))
            continue
        if anchor_ranks != candidate_ranks:
            changed_active_sets.add(code_partition)
            selected.add(int(partition.id))
            continue
        if any(
            _event_is_config_dependent(event_by_id[event_id])
            for event_id in (int(item) for item in partition.event_ids)
            if event_id in event_by_id
        ):
            selected.add(int(partition.id))
            communication_window_count += 1

    reasons = ["parallel strategy changed; refreshing config-dependent trace windows"]
    if changed_active_sets:
        reasons.append(
            "active-lane sets changed for "
            + str(len(changed_active_sets))
            + " code partitions; refreshing their trace windows"
        )
    if communication_window_count:
        reasons.append(
            "communication placement may change; refreshing "
            + str(communication_window_count)
            + " communication trace windows"
        )
    if missing_active_metadata:
        reasons.append(
            "missing active-lane metadata for "
            + str(len(missing_active_metadata))
            + " code partitions; refreshing their trace windows conservatively"
        )
    if not selected:
        reasons.append("active-lane sets and communication trace windows are unchanged")
    return tuple(sorted(selected)), tuple(reasons)


def _configuration_changed(anchor: FlexMayaAnchor, candidate: FlexMayaWorkloadSpec) -> bool:
    return (
        anchor.spec.world_size,
        anchor.spec.tp,
        anchor.spec.pp,
        anchor.spec.dp,
        anchor.spec.rank_group_policy,
    ) != (
        candidate.world_size,
        candidate.tp,
        candidate.pp,
        candidate.dp,
        candidate.rank_group_policy,
    )


def plan_candidate_refresh(
    anchor: FlexMayaAnchor,
    candidate_spec: FlexMayaWorkloadSpec,
    trace: object,
) -> FlexMayaRefreshPlan:
    changed_partitions = _changed_partitions(anchor, candidate_spec)
    config_changed = _configuration_changed(anchor, candidate_spec)
    if config_changed:
        changed_partitions = tuple(dict.fromkeys(changed_partitions + ("__parallel_strategy__",)))
    fallback_reasons: list[str] = []
    if config_changed:
        affected_trace_partitions, config_reasons = _config_affected_trace_partitions(anchor.spec, candidate_spec, trace)
        fallback_reasons.extend(config_reasons)
        affected_rank_groups = _affected_groups_from_events(
            trace,
            _event_ids_in_trace_partitions(trace, affected_trace_partitions),
        )
    else:
        affected_trace_partitions = _affected_trace_partitions(trace, changed_partitions)
        affected_rank_groups = _affected_groups_from_events(
            trace,
            _event_ids_in_trace_partitions(trace, affected_trace_partitions),
        )
    if any(partition.requires_grounding for partition in candidate_spec.code_partitions):
        fallback_reasons.append("candidate has grounding partitions; V1 records them but does not satisfy MoE grounding")
    affected_trace_partition_count = len(affected_trace_partitions)
    return FlexMayaRefreshPlan(
        changed_partitions=changed_partitions,
        affected_rank_groups=affected_rank_groups,
        affected_trace_partitions=affected_trace_partitions,
        configuration_changed=config_changed,
        affected_trace_partition_count=affected_trace_partition_count,
        refresh_scope="config_trace_partitions" if config_changed else "selected",
        fallback_reasons=tuple(fallback_reasons),
    )


def replay_selected_refresh(trace: object, refresh_plan: FlexMayaRefreshPlan) -> tuple[object, ReplayReport]:
    """Replay only event windows selected by the FlexEva source-RAS plan."""
    selected_trace = filter_trace_partitions(trace, list(refresh_plan.affected_trace_partitions))
    return selected_trace, replay_trace_once(selected_trace)


def evaluate_candidate(
    anchor: FlexMayaAnchor,
    candidate_spec: FlexMayaWorkloadSpec,
    raw_events: Iterable[RawEvent],
) -> FlexMayaEvaluationReport:
    trace = _build_trace(candidate_spec, raw_events)
    refresh_plan = plan_candidate_refresh(anchor, candidate_spec, trace)
    selected_trace, selected_feedback = replay_selected_refresh(trace, refresh_plan)
    summary = {
        "kind": "candidate_eval",
        "candidate_spec": asdict(candidate_spec),
        "refresh_plan": asdict(refresh_plan),
        "trace": trace_summary(trace),
        "selected_trace": trace_summary(selected_trace),
        "feedback": selected_feedback.to_dict(),
        "selected_feedback": selected_feedback.to_dict(),
        "reuse_ratio": 1.0 - (
            (refresh_plan.affected_trace_partition_count or len(refresh_plan.affected_trace_partitions))
            / max(len(trace.sync_partitions), 1)
        ),
    }
    return FlexMayaEvaluationReport(
        anchor=anchor,
        candidate_spec=candidate_spec,
        refresh_plan=refresh_plan,
        trace=trace,
        feedback=selected_feedback,
        selected_trace=selected_trace,
        selected_feedback=selected_feedback,
        summary=summary,
    )


def commit_anchor(report: FlexMayaEvaluationReport) -> FlexMayaAnchor:
    return FlexMayaAnchor(
        spec=report.candidate_spec,
        source_hashes=source_hashes(report.candidate_spec),
        trace=report.trace,
        feedback=report.feedback,
        summary={"kind": "commit_anchor", "from_report": report.summary},
    )
