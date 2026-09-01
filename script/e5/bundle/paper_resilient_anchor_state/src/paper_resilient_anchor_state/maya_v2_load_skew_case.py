from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


MAYA_V2_LOAD_SKEW_CASE_ID = "maya_v2_load_skew_r1"
MAYA_V2_LOAD_SKEW_ROUND_ID = "round_load_skew_r1"
MAYA_V2_LOAD_SKEW_GOAL_ID = "straggler_load_skew"

DEFAULT_MAYA_V2_SOURCE_ROOT = (
    Path(__file__).resolve().parents[3]
    / "external"
    / "maya-native-source-package-20260514"
)
DEFAULT_MAYA_V2_WORKLOAD_MANIFEST = (
    DEFAULT_MAYA_V2_SOURCE_ROOT
    / "tests"
    / "workloads"
    / "fake_cuda"
    / "moe_workload_family_v2"
    / "workload_family.json"
)

_TEST_SIM_FAKE_CUDA_PREFIX = "/home/muxi/test-sim/tests/workloads/fake_cuda/"
_TRACE_MODELING_EVENT_TYPES = frozenset(
    {
        "kernel_launch",
        "blas_compute",
        "nccl_collective",
        "mem_copy",
        "mem_alloc",
        "stream_op",
    }
)
_SYNC_APIS = frozenset(
    {
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
        "cudaEventSynchronize",
        "cudaStreamWaitEvent",
        "cudaEventRecord",
        "cudaEventRecordWithFlags",
        "ncclAllReduce",
        "ncclAllGather",
        "ncclAllToAll",
        "ncclAllToAllv",
        "ncclBroadcast",
        "ncclReduce",
        "ncclReduceScatter",
        "ncclSend",
        "ncclRecv",
    }
)
_MATERIAL_STREAM_APIS = frozenset(
    {
        "cudaLaunchKernel",
        "cublasSgemm_v2",
        "cublasGemmEx",
        "cublasGemmStridedBatchedEx",
        "cublasGemmBatchedEx",
        "cublasLtMatmul",
        "cudaMemcpy",
        "cudaMemcpyAsync",
    }
) | _SYNC_APIS


@dataclass(frozen=True)
class MayaV2CandidateSpec:
    candidate_id: str
    parent: str | None
    goal_ids: tuple[str, ...]
    change_surface: str
    entry: str
    semantic_diffs: tuple[str, ...]


@dataclass(frozen=True)
class MayaV2RoundSpec:
    round_id: str
    goal_id: str
    parallel_budget: int
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class MoEGroundingSnapshot:
    source: str
    route_decisions_by_rank: Mapping[int, tuple[int, ...]]
    expert_loads: Mapping[int, int]
    overflow_tokens: int
    dropped_tokens: int
    rerouted_tokens: int
    local_dispatch_tokens: int
    remote_dispatch_tokens: int
    collective_stage: str
    required_grounding_points: tuple[str, ...]

    @property
    def remote_token_fraction(self) -> float:
        total = self.local_dispatch_tokens + self.remote_dispatch_tokens
        if total == 0:
            return 0.0
        return self.remote_dispatch_tokens / total


@dataclass(frozen=True)
class TraceLane:
    lane_id: str
    rank: int
    lane_kind: str
    stream_id: str | None = None
    event_count: int = 0


@dataclass(frozen=True)
class TracePartition:
    partition_id: str
    lane_id: str
    rank: int
    event_ids: tuple[str, ...]
    start_order: int
    end_order: int
    boundary_kind: str
    source_boundary_id: str


@dataclass(frozen=True)
class TraceRASProjection:
    lanes: tuple[TraceLane, ...]
    partitions: tuple[TracePartition, ...]
    raw_event_count: int
    modeling_event_count: int
    notes: tuple[str, ...]


@dataclass(frozen=True)
class SelectiveRefreshPlan:
    candidate_id: str
    logical_world_size: int
    affected_boundary_ids: tuple[str, ...]
    affected_rank_groups: tuple[tuple[int, ...], ...]
    refreshed_ranks: tuple[int, ...]
    reused_ranks: tuple[int, ...]
    cached_partition_ids: tuple[str, ...]
    escalation_required: bool
    escalation_reason: str | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class MayaV2LoadSkewCase:
    case_id: str
    family_id: str
    workload_kind: str
    title: str
    manifest_path: str
    source_root: str
    round: MayaV2RoundSpec
    anchor: MayaV2CandidateSpec
    candidates: tuple[MayaV2CandidateSpec, ...]
    grounding_targets: tuple[str, ...]
    trace_layer_policy: Mapping[str, Any]
    live_capture_mode: str
    real_cluster_role: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "family_id": self.family_id,
            "workload_kind": self.workload_kind,
            "title": self.title,
            "manifest_path": self.manifest_path,
            "source_root": self.source_root,
            "round": asdict(self.round),
            "anchor": asdict(self.anchor),
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "grounding_targets": list(self.grounding_targets),
            "trace_layer_policy": dict(self.trace_layer_policy),
            "live_capture_mode": self.live_capture_mode,
            "real_cluster_role": dict(self.real_cluster_role),
        }


def build_maya_v2_load_skew_case(
    *,
    manifest_path: Path | str = DEFAULT_MAYA_V2_WORKLOAD_MANIFEST,
    source_root: Path | str = DEFAULT_MAYA_V2_SOURCE_ROOT,
    round_id: str = MAYA_V2_LOAD_SKEW_ROUND_ID,
) -> MayaV2LoadSkewCase:
    manifest_file = Path(manifest_path)
    source_root_path = Path(source_root)
    manifest = _load_json(manifest_file)
    round_payload = _round_by_id(manifest, round_id)
    if round_payload["goal_id"] != MAYA_V2_LOAD_SKEW_GOAL_ID:
        raise ValueError(f"{round_id} is not a load-skew round")

    candidates_by_id = {
        item["candidate_id"]: _candidate_from_payload(item, source_root_path=source_root_path)
        for item in manifest["candidates"]
    }
    anchor_id = manifest["anchor"]["candidate_id"]
    anchor = candidates_by_id[anchor_id]
    candidate_ids = tuple(round_payload["candidate_order"][: int(round_payload["parallel_budget"])])
    selected = (anchor,) + tuple(candidates_by_id[candidate_id] for candidate_id in candidate_ids)

    return MayaV2LoadSkewCase(
        case_id=MAYA_V2_LOAD_SKEW_CASE_ID,
        family_id=str(manifest["family_id"]),
        workload_kind=str(manifest["workload_kind"]),
        title=str(manifest["title"]),
        manifest_path=str(manifest_file),
        source_root=str(source_root_path),
        round=MayaV2RoundSpec(
            round_id=str(round_payload["round_id"]),
            goal_id=str(round_payload["goal_id"]),
            parallel_budget=int(round_payload["parallel_budget"]),
            candidate_ids=candidate_ids,
        ),
        anchor=anchor,
        candidates=selected,
        grounding_targets=(
            "route_decision",
            "expert_load",
            "overflow_state",
            "remote_dispatch",
            "collective_stage",
        ),
        trace_layer_policy={
            "raw_trace_role": "audit_input_evidence",
            "modeling_surface": "normalized_collated_maya_events",
            "lane_model": "rank_local_host_plus_cuda_stream_lanes",
            "partitioning": "source_boundary_then_lane_synchronization",
            "partial_refresh": "rerun_affected_rank_groups_and_reuse_cached_partitions",
        },
        live_capture_mode="optional_fake_cuda_proot_integration",
        real_cluster_role=dict(manifest["live_example"]["real_cluster_role"]),
    )


def build_moe_load_skew_grounding_snapshot(
    *,
    route_decisions_by_rank: Mapping[int, Iterable[int]],
    local_experts_by_rank: Mapping[int, Iterable[int]] | None = None,
    overflow_tokens: int = 0,
    dropped_tokens: int = 0,
    rerouted_tokens: int = 0,
    collective_stage: str = "dispatch_then_expert_compute",
    source: str = "minimal_cpu_semantic_pass",
) -> MoEGroundingSnapshot:
    normalized_routes = {
        int(rank): tuple(int(expert) for expert in experts)
        for rank, experts in sorted(route_decisions_by_rank.items())
    }
    normalized_local = {
        int(rank): frozenset(int(expert) for expert in experts)
        for rank, experts in (local_experts_by_rank or {}).items()
    }
    expert_loads = Counter(expert for experts in normalized_routes.values() for expert in experts)
    local_dispatch_tokens = 0
    remote_dispatch_tokens = 0
    for rank, experts in normalized_routes.items():
        local_experts = normalized_local.get(rank)
        for expert in experts:
            if local_experts is not None and expert in local_experts:
                local_dispatch_tokens += 1
            else:
                remote_dispatch_tokens += 1

    return MoEGroundingSnapshot(
        source=source,
        route_decisions_by_rank=normalized_routes,
        expert_loads=dict(sorted(expert_loads.items())),
        overflow_tokens=int(overflow_tokens),
        dropped_tokens=int(dropped_tokens),
        rerouted_tokens=int(rerouted_tokens),
        local_dispatch_tokens=local_dispatch_tokens,
        remote_dispatch_tokens=remote_dispatch_tokens,
        collective_stage=collective_stage,
        required_grounding_points=(
            "route_decision",
            "expert_load",
            "overflow_state",
            "remote_dispatch",
            "collective_stage",
        ),
    )


def build_trace_ras_projection(
    trace_records_by_rank: Mapping[int, Iterable[Mapping[str, Any]]],
    *,
    source_boundary_id: str = "source:unknown",
) -> TraceRASProjection:
    lane_events: dict[str, list[tuple[int, str, str, bool]]] = {}
    lane_meta: dict[str, tuple[int, str, str | None]] = {}
    raw_event_count = 0
    modeling_event_count = 0

    for rank, records in sorted(trace_records_by_rank.items()):
        rank_id = int(rank)
        host_lane_id = _host_lane_id(rank_id)
        lane_meta.setdefault(host_lane_id, (rank_id, "host", None))
        for ordinal, record in enumerate(records):
            raw_event_count += 1
            event_id = _event_id(rank_id, ordinal, record)
            lane_id, lane_kind, stream_id = _lane_for_record(rank_id, record)
            lane_meta.setdefault(lane_id, (rank_id, lane_kind, stream_id))
            if not _is_modeling_event(record):
                continue
            modeling_event_count += 1
            boundary_id = str(record.get("source_boundary_id") or source_boundary_id)
            lane_events.setdefault(lane_id, []).append(
                (ordinal, event_id, boundary_id, _is_sync_boundary(record))
            )

    lanes = tuple(
        TraceLane(
            lane_id=lane_id,
            rank=rank,
            lane_kind=lane_kind,
            stream_id=stream_id,
            event_count=len(lane_events.get(lane_id, ())),
        )
        for lane_id, (rank, lane_kind, stream_id) in sorted(lane_meta.items())
    )
    partitions = _partitions_from_lane_events(lane_events)
    return TraceRASProjection(
        lanes=lanes,
        partitions=partitions,
        raw_event_count=raw_event_count,
        modeling_event_count=modeling_event_count,
        notes=(
            "raw Maya rank traces remain audit evidence",
            "normalized modeling events are partitioned by rank-local lanes",
            "synchronization records become trace RAS partition boundaries",
        ),
    )


def build_selective_refresh_plan(
    *,
    candidate_id: str,
    logical_world_size: int,
    rank_groups: Iterable[Iterable[int]],
    affected_group_indices: Iterable[int],
    affected_boundary_ids: Iterable[str],
    cached_partition_ids: Iterable[str] = (),
    collective_shape_changed: bool = False,
    collective_member_ranks: Iterable[int] = (),
    escalation_reason: str | None = None,
) -> SelectiveRefreshPlan:
    groups = tuple(tuple(int(rank) for rank in group) for group in rank_groups)
    affected_indices = tuple(int(index) for index in affected_group_indices)
    affected_groups = tuple(groups[index] for index in affected_indices)
    if collective_shape_changed:
        expanded = tuple(int(rank) for rank in collective_member_ranks)
        if not expanded:
            expanded = tuple(rank for group in affected_groups for rank in group)
        refreshed_ranks = tuple(sorted(set(expanded)))
        reason = escalation_reason or "collective membership/order/shape changed"
    else:
        refreshed_ranks = tuple(sorted({min(group) for group in affected_groups if group}))
        reason = None
    reused_ranks = tuple(
        rank for rank in range(int(logical_world_size)) if rank not in set(refreshed_ranks)
    )
    return SelectiveRefreshPlan(
        candidate_id=candidate_id,
        logical_world_size=int(logical_world_size),
        affected_boundary_ids=tuple(str(item) for item in affected_boundary_ids),
        affected_rank_groups=affected_groups,
        refreshed_ranks=refreshed_ranks,
        reused_ranks=reused_ranks,
        cached_partition_ids=tuple(str(item) for item in cached_partition_ids),
        escalation_required=bool(collective_shape_changed),
        escalation_reason=reason,
        notes=(
            "default refresh reruns one representative per affected semantic rank group",
            "cached trace partitions remain reusable for unaffected ranks and windows",
            "collective-visible changes expand refresh to the required collective members",
        ),
    )


def build_optional_live_capture_command(
    case: MayaV2LoadSkewCase,
    *,
    candidate_id: str,
    output_dir: Path | str,
    logical_world_size: int,
    profiled_rank_groups: str,
    steps: int = 1,
) -> tuple[str, ...]:
    candidate = _candidate_by_id(case.candidates, candidate_id)
    source_root = Path(case.source_root)
    return (
        "python",
        "-m",
        "flexsim.maya_lite.capture_emulated",
        "--output-dir",
        str(output_dir),
        "--logical-world-size",
        str(int(logical_world_size)),
        "--profiled-rank-groups",
        profiled_rank_groups,
        "--collective-mode",
        "trace_only",
        "--trace-surface",
        "all",
        "--host-timing-mode",
        "measure",
        "--trim-to-step-window",
        "--frun",
        str(source_root / "fake-cuda" / "frun"),
        candidate.entry,
        "--steps",
        str(int(steps)),
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _round_by_id(manifest: Mapping[str, Any], round_id: str) -> Mapping[str, Any]:
    for item in manifest["round_specs"]:
        if item["round_id"] == round_id:
            return item
    raise KeyError(round_id)


def _candidate_from_payload(
    payload: Mapping[str, Any],
    *,
    source_root_path: Path,
) -> MayaV2CandidateSpec:
    return MayaV2CandidateSpec(
        candidate_id=str(payload["candidate_id"]),
        parent=None if payload.get("parent") is None else str(payload["parent"]),
        goal_ids=tuple(str(item) for item in payload["goal_ids"]),
        change_surface=str(payload["change_surface"]),
        entry=_normalize_maya_workload_entry(str(payload["entry"]), source_root_path=source_root_path),
        semantic_diffs=tuple(str(item) for item in payload["semantic_diffs"]),
    )


def _normalize_maya_workload_entry(entry: str, *, source_root_path: Path) -> str:
    if entry.startswith(_TEST_SIM_FAKE_CUDA_PREFIX):
        relative = entry.removeprefix(_TEST_SIM_FAKE_CUDA_PREFIX)
        return str(source_root_path / "tests" / "workloads" / "fake_cuda" / relative)
    return entry


def _candidate_by_id(
    candidates: Iterable[MayaV2CandidateSpec],
    candidate_id: str,
) -> MayaV2CandidateSpec:
    for candidate in candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise KeyError(candidate_id)


def _host_lane_id(rank: int) -> str:
    return f"rank:{rank}/host"


def _stream_lane_id(rank: int, stream_id: str) -> str:
    return f"rank:{rank}/stream:{stream_id}"


def _event_id(rank: int, ordinal: int, record: Mapping[str, Any]) -> str:
    raw = record.get("id") or record.get("event_id")
    if raw is not None:
        return str(raw)
    return f"rank:{rank}:event:{ordinal:04d}"


def _lane_for_record(rank: int, record: Mapping[str, Any]) -> tuple[str, str, str | None]:
    if _record_targets_stream(record):
        stream_id = _stream_id(record)
        return _stream_lane_id(rank, stream_id), "device_stream", stream_id
    return _host_lane_id(rank), "host", None


def _stream_id(record: Mapping[str, Any]) -> str:
    for key in ("stream_id", "launch_stream_id", "canonical_stream_id", "raw_stream_id", "stream"):
        value = record.get(key)
        if value not in (None, "", "0", "0x0"):
            return str(value)
    return "default"


def _record_targets_stream(record: Mapping[str, Any]) -> bool:
    op_type = _op_type(record)
    api = _api(record)
    return op_type in _TRACE_MODELING_EVENT_TYPES or api in _MATERIAL_STREAM_APIS


def _is_modeling_event(record: Mapping[str, Any]) -> bool:
    op_type = _op_type(record)
    api = _api(record)
    if op_type == "host_delay" or api == "__hostDelay__":
        return True
    return _record_targets_stream(record)


def _is_sync_boundary(record: Mapping[str, Any]) -> bool:
    if bool(record.get("synchronization") or record.get("sync")):
        return True
    op_type = _op_type(record)
    api = _api(record)
    return op_type == "nccl_collective" or api in _SYNC_APIS


def _api(record: Mapping[str, Any]) -> str:
    return str(record.get("api") or record.get("name") or "")


def _op_type(record: Mapping[str, Any]) -> str:
    return str(record.get("type") or record.get("op_type") or record.get("event_type") or "").lower()


def _partitions_from_lane_events(
    lane_events: Mapping[str, list[tuple[int, str, str, bool]]],
) -> tuple[TracePartition, ...]:
    partitions: list[TracePartition] = []
    for lane_id, events in sorted(lane_events.items()):
        current: list[tuple[int, str, str, bool]] = []
        partition_index = 0
        for event in events:
            if current and event[2] != current[-1][2]:
                partitions.append(_partition_from_events(lane_id, current, partition_index))
                partition_index += 1
                current = []
            if event[3]:
                if current:
                    partitions.append(_partition_from_events(lane_id, current, partition_index))
                    partition_index += 1
                    current = []
                partitions.append(_partition_from_events(lane_id, [event], partition_index))
                partition_index += 1
                continue
            current.append(event)
        if current:
            partitions.append(_partition_from_events(lane_id, current, partition_index))
    return tuple(partitions)


def _partition_from_events(
    lane_id: str,
    events: list[tuple[int, str, str, bool]],
    partition_index: int,
) -> TracePartition:
    rank = int(lane_id.split("/", 1)[0].split(":", 1)[1])
    boundary_kind = "sync" if len(events) == 1 and events[0][3] else "window"
    return TracePartition(
        partition_id=f"{lane_id}:p{partition_index:04d}",
        lane_id=lane_id,
        rank=rank,
        event_ids=tuple(event_id for _, event_id, _, _ in events),
        start_order=events[0][0],
        end_order=events[-1][0],
        boundary_kind=boundary_kind,
        source_boundary_id=events[0][2],
    )
