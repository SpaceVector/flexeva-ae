"""
Schema definitions for Maya-lite low-level trace ingestion.

The contract is intentionally narrow and low-level:

- traces are per-rank JSONL files
- each record describes one backend API observation
- rank identity comes from the file path, not workload semantics
- no SPSD, logical-op, or semantic-scope fields appear here
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .filters import canonicalize_trace_api


class TraceSource(str, Enum):
    REAL = "real"
    FAKE = "fake"
    UNKNOWN = "unknown"


_CUBLAS_COMPUTE_APIS = {
    "cublasSgemm_v2",
    "cublasGemmEx",
    "cublasGemmStridedBatchedEx",
    "cublasGemmBatchedEx",
    "cublasLtMatmul",
}

_CUBLAS_STREAM_APIS = {
    "cublasSetStream_v2",
}

_CUBLAS_CONTEXT_APIS = {
    "cublasCreate_v2",
    "cublasDestroy_v2",
    "cublasSetMathMode",
    "cublasSetWorkspace_v2",
    "cublasLtCreate",
    "cublasLtDestroy",
    "cublasLtMatmulDescCreate",
    "cublasLtMatmulDescSetAttribute",
    "cublasLtMatmulDescDestroy",
    "cublasLtMatmulPreferenceCreate",
    "cublasLtMatmulPreferenceSetAttribute",
    "cublasLtMatmulPreferenceDestroy",
    "cublasLtMatrixLayoutCreate",
    "cublasLtMatrixLayoutDestroy",
}


def _normalize_canonical_op_type(api: str, op_type: str) -> str:
    if api == "cudaLaunchKernel":
        return "kernel_launch"
    if api in _CUBLAS_COMPUTE_APIS:
        return "blas_compute"
    if api in _CUBLAS_STREAM_APIS:
        return "stream_op"
    if api in _CUBLAS_CONTEXT_APIS:
        return "context_op"
    if api in {
        "ncclAllReduce",
        "ncclAllGather",
        "ncclAllToAll",
        "ncclAllToAllv",
        "ncclBroadcast",
        "ncclReduce",
        "ncclReduceScatter",
        "ncclSend",
        "ncclRecv",
    }:
        return "nccl_collective"
    if api in {"cudaMemcpy", "cudaMemcpyAsync"}:
        return "mem_copy"
    if api in {"cudaMalloc", "cudaMallocAsync", "cudaFree", "cudaFreeAsync"}:
        return "mem_alloc"
    if api.startswith("cudaStream") or api.startswith("cudaEvent"):
        return "stream_op"
    return op_type


def normalize_op_type(api: str, op_type: str) -> str:
    return _normalize_canonical_op_type(canonicalize_trace_api(api), op_type)


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One low-level API event from a rank trace."""

    rank: int
    ordinal: int
    source: TraceSource
    ts: int
    pid: int
    tid: int
    module: str
    api: str
    op_type: str
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json_record(
        cls,
        record: Mapping[str, Any],
        *,
        rank: int,
        ordinal: int,
        source: TraceSource,
    ) -> "TraceEvent":
        required = ("ts", "pid", "tid", "mod", "api", "type")
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"trace record missing required fields: {missing}")

        extras = {
            str(key): value
            for key, value in record.items()
            if key not in {"ts", "pid", "tid", "mod", "api", "type"}
        }
        api = canonicalize_trace_api(str(record["api"]))
        return cls(
            rank=rank,
            ordinal=ordinal,
            source=source,
            ts=int(record["ts"]),
            pid=int(record["pid"]),
            tid=int(record["tid"]),
            module=str(record["mod"]),
            api=api,
            op_type=_normalize_canonical_op_type(api, str(record["type"])),
            extras=extras,
        )


@dataclass(frozen=True, slots=True)
class RankTrace:
    """All events loaded from one rank_*.jsonl file."""

    rank: int
    path: Path
    source: TraceSource
    events: tuple[TraceEvent, ...]

    @property
    def num_events(self) -> int:
        return len(self.events)


@dataclass(frozen=True, slots=True)
class FidelityWindow:
    """Resolved per-rank fidelity envelope metadata.

    Public source taxonomy is intentionally narrow:

    - manifest
    - trace_markers
    - workload_heuristic
    - boundary_fallback

    Diagnostic transport or widening details belong in ``extras`` rather than
    inventing new source strings.
    """

    start_ts: int
    end_ts: int
    source: str
    is_paper_valid_step_window: bool
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "start_ts": int(self.start_ts),
            "end_ts": int(self.end_ts),
            "source": str(self.source),
            "is_paper_valid_step_window": bool(self.is_paper_valid_step_window),
        }
        payload.update(self.extras)
        return payload


@dataclass(frozen=True, slots=True)
class TraceBundle:
    """A directory of rank traces treated as one low-level workload trace."""

    trace_dir: Path
    source: TraceSource
    rank_traces: tuple[RankTrace, ...]
    original_world_size: int | None = None
    captured_world_size: int | None = None
    profiled_rank_groups: dict[int, tuple[int, ...]] = field(default_factory=dict)
    rank_host_machines: dict[int, str] = field(default_factory=dict)
    rank_host_dispatch_queues: dict[int, str] = field(default_factory=dict)
    communicator_memberships: dict[str, tuple[int, ...]] = field(default_factory=dict)
    host_timing_dispatch_scope_resolved: str | None = None
    logical_rank_materialized: bool = False
    trace_window: str = "full"
    step_windows: dict[int, tuple[int, int]] = field(default_factory=dict)
    fidelity_windows: dict[int, FidelityWindow] = field(default_factory=dict)

    @property
    def world_size(self) -> int:
        return self.original_world_size or len(self.rank_traces)

    @property
    def profiled_world_size(self) -> int:
        return self.captured_world_size or len(self.rank_traces)

    @property
    def total_events(self) -> int:
        return sum(rank_trace.num_events for rank_trace in self.rank_traces)

    def rank_ids(self) -> tuple[int, ...]:
        return tuple(rank_trace.rank for rank_trace in self.rank_traces)


@dataclass(frozen=True, slots=True)
class TraceDirectorySummary:
    """Lightweight inspection result for a rank-trace directory."""

    trace_dir: Path
    source: TraceSource
    rank_files: tuple[Path, ...]
    rank_ids: tuple[int, ...]
    sample_event_counts: dict[int, int]
    observed_keys: tuple[str, ...]
    observed_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CollectiveGroup:
    """Paper-aligned low-level collective grouping for replay."""

    id: str
    api: str
    op_type: str
    ranks: tuple[int, ...]
    event_ids: tuple[str, ...]
    communicator_id: str | None = None
    sequence_number: int | None = None
    communicator_size: int | None = None
    participant_count: int | None = None
    root: int | None = None
    match_basis: str = "payload_signature"


@dataclass(frozen=True, slots=True)
class CollatedEvent:
    """Trace event lifted into a deterministic low-level replay order."""

    id: str
    rank: int
    ordinal: int
    source: TraceSource
    ts: int
    pid: int
    tid: int
    module: str
    api: str
    op_type: str
    extras: dict[str, Any] = field(default_factory=dict)
    # Previous event on the same host-dispatch lane. This is lane-local rather
    # than "previous event in rank order" once multiple host lanes are
    # preserved explicitly.
    prev_event_id: str | None = None
    collective_group_id: str | None = None

    @property
    def is_collective(self) -> bool:
        return self.collective_group_id is not None


@dataclass(frozen=True, slots=True)
class CollatedTrace:
    """Collated low-level trace used as Maya-lite replay input."""

    trace_dir: Path
    source: TraceSource
    rank_events: dict[int, tuple[CollatedEvent, ...]]
    global_events: tuple[CollatedEvent, ...]
    collective_groups: dict[str, CollectiveGroup]
    original_world_size: int | None = None
    captured_world_size: int | None = None
    profiled_rank_groups: dict[int, tuple[int, ...]] = field(default_factory=dict)
    rank_host_machines: dict[int, str] = field(default_factory=dict)
    rank_host_dispatch_queues: dict[int, str] = field(default_factory=dict)
    communicator_memberships: dict[str, tuple[int, ...]] = field(default_factory=dict)
    host_timing_dispatch_scope_resolved: str | None = None
    logical_rank_materialized: bool = False
    trace_window: str = "full"
    fidelity_windows: dict[int, FidelityWindow] = field(default_factory=dict)

    @property
    def world_size(self) -> int:
        return self.original_world_size or len(self.rank_events)

    @property
    def profiled_world_size(self) -> int:
        return self.captured_world_size or len(self.rank_events)

    @property
    def total_events(self) -> int:
        return len(self.global_events)


@dataclass(frozen=True, slots=True)
class AnnotatedEvent(CollatedEvent):
    """Collated event with attached duration estimate."""

    duration_us: float = 0.0
    duration_source: str = "unknown"


@dataclass(frozen=True, slots=True)
class AnnotatedTrace:
    """Collated low-level trace with runtime annotations."""

    trace_dir: Path
    source: TraceSource
    rank_events: dict[int, tuple[AnnotatedEvent, ...]]
    global_events: tuple[AnnotatedEvent, ...]
    collective_groups: dict[str, CollectiveGroup]
    original_world_size: int | None = None
    captured_world_size: int | None = None
    profiled_rank_groups: dict[int, tuple[int, ...]] = field(default_factory=dict)
    rank_host_machines: dict[int, str] = field(default_factory=dict)
    rank_host_dispatch_queues: dict[int, str] = field(default_factory=dict)
    communicator_memberships: dict[str, tuple[int, ...]] = field(default_factory=dict)
    host_timing_dispatch_scope_resolved: str | None = None
    logical_rank_materialized: bool = False
    trace_window: str = "full"
    fidelity_windows: dict[int, FidelityWindow] = field(default_factory=dict)

    @property
    def world_size(self) -> int:
        return self.original_world_size or len(self.rank_events)

    @property
    def profiled_world_size(self) -> int:
        return self.captured_world_size or len(self.rank_events)

    @property
    def total_events(self) -> int:
        return len(self.global_events)


@dataclass(frozen=True, slots=True)
class SimulatedEvent:
    """Replay-time view of an annotated low-level event."""

    event_id: str
    rank: int
    api: str
    op_type: str
    start_us: float
    end_us: float
    duration_us: float
    duration_source: str = "unknown"
    collective_group_id: str | None = None
    resource_kind: str = "host"
    host_machine_id: str | None = None
    host_dispatch_queue_id: str | None = None
    host_pid: int | None = None
    host_tid: int | None = None
    stream_id: str | None = None


@dataclass(frozen=True, slots=True)
class RankReplayMetrics:
    """Per-rank replay metrics."""

    rank: int
    compute_time_us: float
    communication_time_us: float
    memory_time_us: float
    other_time_us: float
    total_time_us: float
    num_events: int
    utilization: float
    start_offset_us: float = 0.0
    end_time_us: float = 0.0


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """End-to-end output of Maya-lite low-level replay."""

    total_time_us: float
    critical_path_us: float
    global_makespan_us: float
    rank0_time_us: float | None
    success: bool
    rank_metrics: tuple[RankReplayMetrics, ...]
    simulated_events: tuple[SimulatedEvent, ...]


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """One Maya-lite black-box evaluation result."""

    candidate_name: str
    trace_dir: Path
    source: TraceSource
    world_size: int
    total_events: int
    total_time_us: float
    critical_path_us: float
    global_makespan_us: float
    rank0_time_us: float | None
    average_utilization: float
    rank_metrics: tuple[RankReplayMetrics, ...]
    profiled_world_size: int | None = None
    profiled_rank_groups: dict[int, tuple[int, ...]] = field(default_factory=dict)
    annotation_diagnostics: dict[str, Any] = field(default_factory=dict)
    trace_window: str = "full"
    paper_valid_step_window_rank_count: int = 0
    step_window_sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "trace_dir": str(self.trace_dir),
            "source": self.source.value,
            "world_size": self.world_size,
            "profiled_world_size": self.profiled_world_size or len(self.rank_metrics),
            "trace_window": self.trace_window,
            "paper_valid_step_window_rank_count": int(self.paper_valid_step_window_rank_count),
            "step_window_sources": list(self.step_window_sources),
            "total_events": self.total_events,
            "total_time_us": self.total_time_us,
            "critical_path_us": self.critical_path_us,
            "global_makespan_us": self.global_makespan_us,
            "rank0_time_us": self.rank0_time_us,
            "average_utilization": self.average_utilization,
            "profiled_rank_groups": {
                str(rank): list(ranks) for rank, ranks in sorted(self.profiled_rank_groups.items())
            },
            "annotation_diagnostics": dict(self.annotation_diagnostics),
            "rank_metrics": [
                {
                    "rank": metric.rank,
                    "compute_time_us": metric.compute_time_us,
                    "communication_time_us": metric.communication_time_us,
                    "memory_time_us": metric.memory_time_us,
                    "other_time_us": metric.other_time_us,
                    "total_time_us": metric.total_time_us,
                    "start_offset_us": metric.start_offset_us,
                    "end_time_us": metric.end_time_us,
                    "num_events": metric.num_events,
                    "utilization": metric.utilization,
                }
                for metric in self.rank_metrics
            ],
        }
