"""
Low-level runtime annotation for Maya-lite.

This module attaches duration estimates to collated low-level events. The
annotation path is intentionally low-level and may use a trace-derived or other
black-box estimator, but it does not use SPSD semantic scopes.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import os
from threading import Lock
from time import perf_counter
from typing import Any

from flexsim.estimator import Estimator

from .filters import is_ignorable_setup_api, is_low_overhead_api
from .host_delay_profile import HostDelayProfile, HostGapProfile
from .material_signature import (
    canonical_gemm_material_signature,
    canonical_gemm_signature_inputs,
    is_gemm_material_api,
)
from .schema import AnnotatedEvent, AnnotatedTrace, CollatedEvent, CollatedTrace

def is_ignorable_setup_event(event: CollatedEvent) -> bool:
    return is_ignorable_setup_api(event.api)


def _int_payload(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _float_payload(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _env_flag_enabled(*keys: str) -> bool:
    for key in keys:
        if os.environ.get(key) == "1":
            return True
    return False


_OBSERVED_WRAPPER_RUNTIME_APIS = {
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cudaMemcpy",
    "cudaMalloc",
    "cudaFree",
    "cudaMemGetInfo",
    "ncclCommCount",
    "ncclCommInitRank",
    "ncclCommInitRankConfig",
    "ncclCommUserRank",
}

_OBSERVED_CONTROL_PLANE_WRAPPER_APIS = {
    "cublasCreate_v2",
    "cublasSetStream_v2",
    "cudaGetDevice",
    "cudaGetLastError",
    "cudaSetDevice",
    "cudaStreamCreate",
    "cudaEventCreateWithFlags",
    "cudaEventDestroy",
    "cudaEventQuery",
    "cudaEventRecord",
    "cudaStreamWaitEvent",
    "ncclCommGetAsyncError",
    "ncclGroupStart",
    "ncclGroupEnd",
    "ncclGetUniqueId",
    "ncclGetVersion",
}

_DIRECT_RUNTIME_CONTRACT_VALUES = {
    "direct_runtime",
    "semantic_runtime",
    "async_runtime",
}

_EXPLICIT_DIRECT_RUNTIME_FIELDS = (
    "direct_runtime_us",
    "observed_runtime_us",
)

_ASYNC_RUNTIME_OBSERVATION_SOURCE = "capture_real_cuda_event"
_ACTUAL_DEVICE_RUNTIME_MEASUREMENT_KIND = "existing_cpp_event_async_runtime_observer"
_APPENDIX_AB_P2P_DIAGNOSTIC_ENV_KEYS = (
    "MAYA_ENABLE_P2P_PER_BLOCK_COMPONENT_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_P2P_PER_BLOCK_COMPONENT_DIAGNOSTICS",
)
_APPENDIX_AB_ALLREDUCE_DIAGNOSTIC_ENV_KEYS = (
    "MAYA_ENABLE_APPENDIX_AB_SELECTED_NCCL_ALLREDUCE_PER_BLOCK_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_APPENDIX_AB_SELECTED_NCCL_ALLREDUCE_PER_BLOCK_DIAGNOSTICS",
)
_APPENDIX_AB_P2P_SELECTED_APIS = {"ncclSend", "ncclRecv"}
_APPENDIX_AB_BLOCK_BOUNDARY_APIS = {"ncclSend", "ncclRecv", "ncclAllReduce"}
_APPENDIX_AB_KERNEL_APIS = {"cudaLaunchKernel"}
_APPENDIX_AB_GEMM_APIS = {"cublasGemmEx"}
_APPENDIX_AB_STRIDED_GEMM_APIS = {"cublasGemmStridedBatchedEx"}
_APPENDIX_AB_P2P_SELECTION_BASIS = "opt_in_rank_order_semantic_block_api_counts_v1"
_APPENDIX_AB_ALLREDUCE_SELECTION_BASIS = "opt_in_rank_order_semantic_block_api_counts_v1"
_APPENDIX_AB_SELECTED_ALLREDUCE_MOTIF_KEY = (
    "boundary=ncclAllReduce|kernel=4-8|gemm=2-3|strided=2-3|send=0|recv=0|allreduce=1"
)
_APPENDIX_AB_SELECTED_ALLREDUCE_FAMILY_LABEL = (
    "nccl_allreduce_kernel4_8_gemm2_3_strided2_3"
)
_NCCL_P2P_MEMBER_APIS = {"ncclSend", "ncclRecv"}
_NCCL_P2P_GROUP_API = "ncclP2P"


@dataclass(frozen=True)
class _WrapperTimingInfo:
    direct_runtime_us: float | None
    raw_wrapper_us: float | None
    has_wrapper_timing_field: bool
    strict_runtime_wrapper_timing_contract: str


@dataclass
class _CollectiveGroupAggregate:
    representative: AnnotatedEvent
    max_duration_us: float
    event_count: int


def _is_p2p_collective_group_member(event: AnnotatedEvent) -> bool:
    collective_api = str(event.extras.get("collective_api") or "")
    collective = str(event.extras.get("collective") or "").strip().lower()
    return (
        event.api in _NCCL_P2P_MEMBER_APIS
        or collective_api in _NCCL_P2P_MEMBER_APIS
        or collective_api == _NCCL_P2P_GROUP_API
        or collective in {"send", "recv", "p2p", "ncclsend", "ncclrecv", "ncclp2p"}
    )


def _normalize_p2p_group_runtime_payload(payload: dict[str, Any]) -> dict[str, Any]:
    api = str(payload.get("api") or "")
    collective_api = str(payload.get("collective_api") or "")
    collective = str(payload.get("collective") or "").strip().lower()
    is_p2p_group = (
        api in _NCCL_P2P_MEMBER_APIS
        or api == _NCCL_P2P_GROUP_API
        or collective_api in _NCCL_P2P_MEMBER_APIS
        or collective_api == _NCCL_P2P_GROUP_API
        or collective in {"send", "recv", "p2p", "ncclsend", "ncclrecv", "ncclp2p"}
    )
    if not is_p2p_group:
        return payload
    if payload.get("member_api") in (None, ""):
        if api in _NCCL_P2P_MEMBER_APIS:
            payload["member_api"] = api
        elif collective_api in _NCCL_P2P_MEMBER_APIS:
            payload["member_api"] = collective_api
    if payload.get("member_collective") in (None, "") and collective in {"send", "recv"}:
        payload["member_collective"] = collective
    original_world_size = payload.get("world_size")
    if original_world_size not in (None, "") and payload.get("trace_world_size") in (None, ""):
        if _int_payload(original_world_size) != 2:
            payload["trace_world_size"] = original_world_size
    payload["api"] = _NCCL_P2P_GROUP_API
    payload["type"] = "nccl_collective"
    payload["collective"] = "p2p"
    payload["collective_api"] = _NCCL_P2P_GROUP_API
    payload["world_size"] = 2
    payload["communicator_size"] = 2
    payload["participant_count"] = 2
    return payload


def _stream_id(event: CollatedEvent) -> str | None:
    value = event.extras.get("launch_stream_id", event.extras.get("stream_id"))
    if value in (None, ""):
        return None
    return str(value)


def _provider_row_window_fields(event: AnnotatedEvent) -> tuple[str | None, float | None, float | None]:
    window_id = event.extras.get("paper_valid_step_window_id")
    if window_id in (None, ""):
        window_id = event.extras.get("step_window_id")
    start = _float_payload(event.extras.get("window_start_us"))
    end = _float_payload(event.extras.get("window_end_us"))
    return (str(window_id) if window_id not in (None, "") else None, start, end)


def export_predicted_provider_rows(trace: AnnotatedTrace) -> tuple[dict[str, Any], ...]:
    """Export metadata-only predicted provider rows from annotated events.

    The export is intentionally read-only: it serializes existing annotation
    decisions and computes deterministic occurrence ordinals for joining.  It
    does not change duration selection, provider selection, or replay behavior.
    """

    counters: dict[tuple[object, ...], int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    for event in trace.global_events:
        expected = event.extras.get("provider_duration_source_expected")
        if expected in (None, ""):
            continue
        material_api = str(event.extras.get("material_api") or event.api)
        material_signature = event.extras.get("material_signature")
        if material_signature in (None, ""):
            material_signature_text = "signature:unavailable"
        else:
            material_signature_text = str(material_signature)
        window_id, window_start_us, window_end_us = _provider_row_window_fields(event)
        stream = _stream_id(event)
        counter_key = (
            event.rank,
            window_id,
            stream,
            material_api,
            material_signature_text,
        )
        provider_ordinal = counters[counter_key]
        counters[counter_key] += 1
        rows.append(
            {
                "duration_us": float(event.duration_us),
                "duration_source": event.duration_source,
                "provider_duration_source_expected": str(expected),
                "rank": int(event.rank),
                "ordinal": int(event.ordinal),
                "event_id": event.id,
                "api": event.api,
                "op_type": event.op_type,
                "stream": stream,
                "material_api": material_api,
                "material_signature": material_signature_text,
                "material_signature_inputs": event.extras.get("material_signature_inputs", {}),
                "raw_stream_id": event.extras.get("raw_stream_id", stream),
                "canonical_stream_id": event.extras.get("canonical_stream_id", stream),
                "provider_counterpart_key": event.extras.get(
                    "provider_counterpart_key",
                    _provider_counterpart_key(event, material_api, material_signature_text),
                ),
                "paper_valid_step_window_id": window_id,
                "window_start_us": window_start_us,
                "window_end_us": window_end_us,
                "provider_ordinal_within_rank_window_api_signature": provider_ordinal,
            }
        )
    return tuple(rows)


@dataclass
class AnnotationTimingRecorder:
    """Paper-facing timing counters for the annotation / predictor pass.

    Figure 6 describes the Kernel Runtime Estimator as annotating compute
    operations with predicted durations. The implementation still has to walk
    every low-level event to build the annotated trace, but that pass-through
    bookkeeping is not the same thing as runtime prediction. This recorder
    keeps those two costs visible instead of charging all annotation wall time
    to the paper-facing predictor stage.
    """

    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    rank_runtime_estimation_seconds_by_rank: dict[int, float] = field(default_factory=dict)
    rank_runtime_estimation_event_count_by_rank: dict[int, int] = field(default_factory=dict)
    rank_annotated_event_count_by_rank: dict[int, int] = field(default_factory=dict)
    rank_pass_through_event_count_by_rank: dict[int, int] = field(default_factory=dict)
    collective_group_estimation_seconds: float = 0.0
    collective_group_estimation_attempt_count: int = 0
    collective_group_estimation_hit_count: int = 0
    duration_source_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    duration_source_counts_by_api: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    duration_source_us_by_api: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )
    collective_group_duration_basis_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def record_rank(
        self,
        rank: int,
        *,
        runtime_estimation_seconds: float,
        runtime_estimation_event_count: int,
        annotated_event_count: int,
        pass_through_event_count: int,
    ) -> None:
        with self._lock:
            self.rank_runtime_estimation_seconds_by_rank[rank] = float(
                runtime_estimation_seconds
            )
            self.rank_runtime_estimation_event_count_by_rank[rank] = int(
                runtime_estimation_event_count
            )
            self.rank_annotated_event_count_by_rank[rank] = int(annotated_event_count)
            self.rank_pass_through_event_count_by_rank[rank] = int(
                pass_through_event_count
            )

    def record_collective_group_estimation(
        self,
        *,
        elapsed_seconds: float,
        hit: bool,
    ) -> None:
        with self._lock:
            self.collective_group_estimation_seconds += float(elapsed_seconds)
            self.collective_group_estimation_attempt_count += 1
            if hit:
                self.collective_group_estimation_hit_count += 1

    def record_duration_source(
        self,
        *,
        api: str,
        source: str,
        duration_us: float,
    ) -> None:
        with self._lock:
            self.duration_source_counts[source] += 1
            self.duration_source_counts_by_api[api][source] += 1
            self.duration_source_us_by_api[api][source] += float(duration_us)

    def record_collective_group_duration_basis(self, basis: str) -> None:
        with self._lock:
            self.collective_group_duration_basis_counts[basis] += 1

    def summary(self, *, total_annotation_seconds: float) -> dict[str, object]:
        rank_seconds = dict(sorted(self.rank_runtime_estimation_seconds_by_rank.items()))
        rank_event_counts = dict(
            sorted(self.rank_runtime_estimation_event_count_by_rank.items())
        )
        rank_wall_seconds = max(rank_seconds.values(), default=0.0)
        rank_thread_seconds = sum(rank_seconds.values())
        runtime_estimation_wall_seconds = (
            rank_wall_seconds + self.collective_group_estimation_seconds
        )
        annotated_event_count = sum(self.rank_annotated_event_count_by_rank.values())
        runtime_estimation_event_count = sum(rank_event_counts.values())
        pass_through_event_count = sum(self.rank_pass_through_event_count_by_rank.values())
        pass_through_annotation_seconds = max(
            float(total_annotation_seconds) - runtime_estimation_wall_seconds,
            0.0,
        )
        return {
            "basis": "runtime_estimation_only",
            "total_annotation_seconds": float(total_annotation_seconds),
            "runtime_estimation_wall_seconds": float(runtime_estimation_wall_seconds),
            "rank_runtime_estimation_wall_seconds": float(rank_wall_seconds),
            "rank_runtime_estimation_thread_seconds": float(rank_thread_seconds),
            "collective_group_estimation_seconds": float(
                self.collective_group_estimation_seconds
            ),
            "pass_through_annotation_seconds": float(pass_through_annotation_seconds),
            "annotated_event_count": int(annotated_event_count),
            "runtime_estimation_event_count": int(runtime_estimation_event_count),
            "pass_through_event_count": int(pass_through_event_count),
            "collective_group_estimation_attempt_count": int(
                self.collective_group_estimation_attempt_count
            ),
            "collective_group_estimation_hit_count": int(
                self.collective_group_estimation_hit_count
            ),
            "duration_source_counts": dict(sorted(self.duration_source_counts.items())),
            "duration_source_counts_by_api": {
                api: dict(sorted(counts.items()))
                for api, counts in sorted(self.duration_source_counts_by_api.items())
            },
            "duration_source_us_by_api": {
                api: {source: float(value) for source, value in sorted(values.items())}
                for api, values in sorted(self.duration_source_us_by_api.items())
            },
            "collective_group_duration_basis_counts": dict(
                sorted(self.collective_group_duration_basis_counts.items())
            ),
            "rank_runtime_estimation_seconds_by_rank": {
                str(rank): seconds for rank, seconds in rank_seconds.items()
            },
            "rank_runtime_estimation_event_count_by_rank": {
                str(rank): count for rank, count in rank_event_counts.items()
            },
            "rank_annotated_event_count_by_rank": {
                str(rank): count
                for rank, count in sorted(self.rank_annotated_event_count_by_rank.items())
            },
            "rank_pass_through_event_count_by_rank": {
                str(rank): count
                for rank, count in sorted(self.rank_pass_through_event_count_by_rank.items())
            },
        }


def _resolve_wrapper_timing_info(event: CollatedEvent | AnnotatedEvent) -> _WrapperTimingInfo:
    extras = event.extras
    direct_runtime_us: float | None = None
    for field in _EXPLICIT_DIRECT_RUNTIME_FIELDS:
        observed = _float_payload(extras.get(field))
        if observed is not None:
            direct_runtime_us = max(float(observed), 0.0)
            break

    host_duration_field_present = extras.get("host_duration_us") not in (None, "")
    end_ts_field_present = extras.get("end_ts") not in (None, "")
    raw_wrapper_us: float | None = None
    observed_host_duration = _float_payload(extras.get("host_duration_us"))
    if observed_host_duration is not None:
        raw_wrapper_us = max(float(observed_host_duration), 0.0)
    else:
        end_ts = _float_payload(extras.get("end_ts"))
        if end_ts is not None:
            raw_wrapper_us = max(float(end_ts) - float(event.ts), 0.0)

    has_wrapper_timing_field = (
        direct_runtime_us is not None
        or host_duration_field_present
        or end_ts_field_present
    )
    if not _is_strict_runtime_signal_api_type(event.api, event.op_type):
        strict_runtime_wrapper_timing_contract = "not_strict_runtime"
    elif direct_runtime_us is not None:
        strict_runtime_wrapper_timing_contract = "direct_runtime"
    else:
        raw_contract = str(extras.get("wrapper_runtime_contract") or "").strip().lower()
        if raw_contract in _DIRECT_RUNTIME_CONTRACT_VALUES:
            strict_runtime_wrapper_timing_contract = "direct_runtime"
        elif has_wrapper_timing_field:
            strict_runtime_wrapper_timing_contract = "dispatch_only"
        else:
            strict_runtime_wrapper_timing_contract = "missing"

    return _WrapperTimingInfo(
        direct_runtime_us=direct_runtime_us,
        raw_wrapper_us=raw_wrapper_us,
        has_wrapper_timing_field=has_wrapper_timing_field,
        strict_runtime_wrapper_timing_contract=strict_runtime_wrapper_timing_contract,
    )


def _explicit_direct_runtime_us(event: CollatedEvent | AnnotatedEvent) -> float | None:
    return _resolve_wrapper_timing_info(event).direct_runtime_us


def _raw_wrapper_timing_duration_us(event: CollatedEvent | AnnotatedEvent) -> float | None:
    return _resolve_wrapper_timing_info(event).raw_wrapper_us


def _has_wrapper_timing_field(event: CollatedEvent | AnnotatedEvent) -> bool:
    return _resolve_wrapper_timing_info(event).has_wrapper_timing_field


def _strict_runtime_wrapper_timing_contract(event: CollatedEvent | AnnotatedEvent) -> str:
    return _resolve_wrapper_timing_info(event).strict_runtime_wrapper_timing_contract


def _observed_wrapper_duration_us(
    event: CollatedEvent,
    *,
    use_observed_control_plane_wrapper_durations: bool = True,
    use_observed_semantic_wrapper_durations: bool = True,
) -> float | None:
    timing_info = _resolve_wrapper_timing_info(event)
    direct_runtime_us = timing_info.direct_runtime_us
    raw_wrapper_us = timing_info.raw_wrapper_us
    if direct_runtime_us is not None and direct_runtime_us <= 0.0:
        direct_runtime_us = None
    if raw_wrapper_us is not None and raw_wrapper_us <= 0.0:
        raw_wrapper_us = None
    if event.api in _OBSERVED_WRAPPER_RUNTIME_APIS:
        return direct_runtime_us if direct_runtime_us is not None else raw_wrapper_us
    if use_observed_semantic_wrapper_durations and event.op_type in _STRICT_RUNTIME_SIGNAL_TYPES:
        if timing_info.strict_runtime_wrapper_timing_contract == "direct_runtime":
            return direct_runtime_us if direct_runtime_us is not None else raw_wrapper_us
        return None
    if (
        use_observed_control_plane_wrapper_durations
        and event.api in _OBSERVED_CONTROL_PLANE_WRAPPER_APIS
        and str(event.extras.get("wrapper_runtime_contract") or "").strip().lower()
        != "dispatch_only"
    ):
        return raw_wrapper_us
    return None


_STRICT_RUNTIME_SIGNAL_APIS = {
    "cudaLaunchKernel",
}

_STRICT_RUNTIME_SIGNAL_TYPES = {
    "kernel_launch",
    "blas_compute",
    "nccl_collective",
}


def _is_strict_runtime_signal_api_type(api: str, op_type: str) -> bool:
    return api in _STRICT_RUNTIME_SIGNAL_APIS or op_type in _STRICT_RUNTIME_SIGNAL_TYPES


def _requires_strict_runtime_signal(event: CollatedEvent) -> bool:
    return _is_strict_runtime_signal_api_type(event.api, event.op_type)


def _uses_weak_estimator_fallback(source: str) -> bool:
    return source in {"type_stats", "global_fallback"}


def _memory_event_us(event: CollatedEvent) -> float | None:
    if event.api in {"cudaMemcpy", "cudaMemcpyAsync"}:
        size_bytes = (
            _int_payload(event.extras.get("bytes"))
            or _int_payload(event.extras.get("count"))
            or _int_payload(event.extras.get("size"))
        )
        if size_bytes is None:
            return 10.0
        kind = str(event.extras.get("kind", ""))
        # CUDA memcpy kind 3 is device-to-device and is far faster than host transfers.
        bandwidth_bytes_per_us = {
            "1": 25_000.0,   # H2D ~25 GB/s
            "2": 25_000.0,   # D2H ~25 GB/s
            "3": 600_000.0,  # D2D on modern GPU memory path
        }.get(kind, 25_000.0)
        overhead_us = 3.0 if kind == "3" else 10.0
        return overhead_us + (float(size_bytes) / bandwidth_bytes_per_us)

    if event.api in {"cudaMalloc", "cudaMallocAsync"}:
        size_bytes = _int_payload(event.extras.get("bytes")) or _int_payload(event.extras.get("size"))
        if size_bytes is None:
            return 25.0
        return 25.0 + min(float(size_bytes) / 10_000_000.0, 50.0)

    if event.api in {"cudaFree", "cudaFreeAsync"}:
        return 10.0

    return None


def _kernel_launch_event_us(
    event: CollatedEvent,
    *,
    rank_kernel_launch_count: int | None = None,
) -> float | None:
    if event.api != "cudaLaunchKernel":
        return None
    # These traces currently observe launch boundaries, not full kernel completion
    # records. Use a calibrated fallback until richer kernel payloads exist.
    #
    # Two regimes show up in current validation:
    # - GPT / soft-MoE traces: fewer, heavier kernel launches; 11.5 ms works well.
    # - top-k MoE traces: thousands of fine-grained launches; the same fallback
    #   grossly overcounts. Detect that regime from the per-rank launch count and
    #   use a much smaller launch cost.
    launch_count = rank_kernel_launch_count
    if launch_count is None:
        launch_count = _int_payload(event.extras.get("rank_kernel_launch_count")) or 0
    if launch_count >= 3_000:
        return 1_500.0
    return 11_500.0


def _material_signature(event: CollatedEvent) -> str | None:
    """Build a stable ledger-only material signature from existing metadata."""

    if is_gemm_material_api(event.api):
        return canonical_gemm_material_signature(event.extras)

    parts: list[str] = []
    kernel_name = event.extras.get("launch_kernel_name", event.extras.get("kernel"))
    if kernel_name not in (None, ""):
        parts.append(f"kernel={kernel_name}")
    for label, keys in (
        ("grid", ("grid_x", "grid_y", "grid_z")),
        ("block", ("block_x", "block_y", "block_z")),
    ):
        existing = event.extras.get(f"launch_{label}")
        if existing not in (None, ""):
            parts.append(f"{label}={existing}")
            continue
        values = tuple(event.extras.get(key) for key in keys)
        if any(value is not None for value in values):
            parts.append(f"{label}=" + "x".join("" if value is None else str(value) for value in values))
    shared_mem = event.extras.get("launch_shared_mem_bytes", event.extras.get("shared_mem"))
    if shared_mem is not None:
        parts.append(f"shared_mem={shared_mem}")
    stream_id = event.extras.get("launch_stream_id", event.extras.get("stream_id"))
    if stream_id is not None:
        parts.append(f"stream={stream_id}")
    gemm_fields = (
        "m",
        "n",
        "k",
        "lda",
        "ldb",
        "ldc",
        "batch_count",
        "stride_a",
        "stride_b",
        "stride_c",
        "compute_type",
        "cuda_data_type",
        "dtype",
        "transa",
        "transb",
    )
    for key in gemm_fields:
        value = event.extras.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    return ";".join(parts) if parts else None


def _material_signature_inputs(event: CollatedEvent) -> dict[str, object]:
    if is_gemm_material_api(event.api):
        return canonical_gemm_signature_inputs(event.extras)

    keys = (
        "launch_kernel_name", "kernel",
        "launch_grid", "grid_x", "grid_y", "grid_z",
        "launch_block", "block_x", "block_y", "block_z",
        "launch_shared_mem_bytes", "shared_mem",
        "launch_stream_id", "stream_id", "stream",
        "m", "n", "k", "lda", "ldb", "ldc",
        "batch_count", "batchCount",
        "stride_a", "stride_b", "stride_c", "strideA", "strideB", "strideC",
        "compute_type", "computeType", "cuda_data_type", "dtype", "transa", "transb",
    )
    return {key: event.extras.get(key) for key in keys if event.extras.get(key) not in (None, "")}


def _appendix_ab_p2p_component_diagnostics_enabled() -> bool:
    return _env_flag_enabled(*_APPENDIX_AB_P2P_DIAGNOSTIC_ENV_KEYS)


def _appendix_ab_allreduce_component_diagnostics_enabled() -> bool:
    return _env_flag_enabled(*_APPENDIX_AB_ALLREDUCE_DIAGNOSTIC_ENV_KEYS)


def _appendix_ab_count_bucket(count: int) -> str:
    if count <= 1:
        return str(max(int(count), 0))
    if count <= 3:
        return "2-3"
    if count <= 8:
        return "4-8"
    if count <= 32:
        return "9-32"
    if count <= 96:
        return "33-96"
    return "97+"


def _appendix_ab_block_motif_key(boundary_api: str, counts: dict[str, int]) -> str:
    return "|".join(
        [
            f"boundary={boundary_api}",
            f"kernel={_appendix_ab_count_bucket(counts['kernel'])}",
            f"gemm={_appendix_ab_count_bucket(counts['gemm'])}",
            f"strided={_appendix_ab_count_bucket(counts['strided'])}",
            f"send={_appendix_ab_count_bucket(counts['send'])}",
            f"recv={_appendix_ab_count_bucket(counts['recv'])}",
            f"allreduce={_appendix_ab_count_bucket(counts['allreduce'])}",
        ]
    )


def _appendix_ab_count_event(event: AnnotatedEvent, counts: dict[str, int]) -> None:
    if event.api in _APPENDIX_AB_KERNEL_APIS:
        counts["kernel"] += 1
    elif event.api in _APPENDIX_AB_GEMM_APIS:
        counts["gemm"] += 1
    elif event.api in _APPENDIX_AB_STRIDED_GEMM_APIS:
        counts["strided"] += 1
    elif event.api == "ncclSend":
        counts["send"] += 1
    elif event.api == "ncclRecv":
        counts["recv"] += 1
    elif event.api == "ncclAllReduce":
        counts["allreduce"] += 1


def _attach_appendix_ab_p2p_selection_metadata(
    rank_events: dict[int, tuple[AnnotatedEvent, ...]],
    global_events: tuple[AnnotatedEvent, ...],
) -> tuple[dict[int, tuple[AnnotatedEvent, ...]], tuple[AnnotatedEvent, ...]]:
    if not _appendix_ab_p2p_component_diagnostics_enabled():
        return rank_events, global_events

    selected_by_id: dict[str, AnnotatedEvent] = {}
    updated_rank_events: dict[int, tuple[AnnotatedEvent, ...]] = {}
    zero_counts = {
        "kernel": 0,
        "gemm": 0,
        "strided": 0,
        "send": 0,
        "recv": 0,
        "allreduce": 0,
    }
    for rank, events in sorted(rank_events.items()):
        counts = dict(zero_counts)
        updated_events: list[AnnotatedEvent] = []
        for event in events:
            _appendix_ab_count_event(event, counts)
            if event.api not in _APPENDIX_AB_BLOCK_BOUNDARY_APIS:
                updated_events.append(event)
                continue

            motif_key = _appendix_ab_block_motif_key(event.api, counts)
            if (
                event.api in _APPENDIX_AB_P2P_SELECTED_APIS
                and motif_key
                in {
                    "boundary=ncclSend|kernel=1|gemm=0|strided=0|send=1|recv=0|allreduce=0",
                    "boundary=ncclRecv|kernel=1|gemm=0|strided=0|send=0|recv=1|allreduce=0",
                }
            ):
                updated_event = replace(
                    event,
                    extras={
                        **event.extras,
                        "appendix_ab_p2p_motif_key": motif_key,
                        "appendix_ab_p2p_kernel_bucket": "1",
                        "appendix_ab_p2p_selection_basis": _APPENDIX_AB_P2P_SELECTION_BASIS,
                    },
                )
                selected_by_id[event.id] = updated_event
                updated_events.append(updated_event)
            else:
                updated_events.append(event)
            counts = dict(zero_counts)
        updated_rank_events[rank] = tuple(updated_events)

    if not selected_by_id:
        return updated_rank_events, global_events
    updated_global_events = tuple(selected_by_id.get(event.id, event) for event in global_events)
    return updated_rank_events, updated_global_events


def _attach_appendix_ab_allreduce_selection_metadata(
    rank_events: dict[int, tuple[AnnotatedEvent, ...]],
    global_events: tuple[AnnotatedEvent, ...],
) -> tuple[dict[int, tuple[AnnotatedEvent, ...]], tuple[AnnotatedEvent, ...]]:
    if not _appendix_ab_allreduce_component_diagnostics_enabled():
        return rank_events, global_events

    selected_by_id: dict[str, AnnotatedEvent] = {}
    updated_rank_events: dict[int, tuple[AnnotatedEvent, ...]] = {}
    zero_counts = {
        "kernel": 0,
        "gemm": 0,
        "strided": 0,
        "send": 0,
        "recv": 0,
        "allreduce": 0,
    }
    for rank, events in sorted(rank_events.items()):
        counts = dict(zero_counts)
        updated_events: list[AnnotatedEvent] = []
        for event in events:
            _appendix_ab_count_event(event, counts)
            if event.api not in _APPENDIX_AB_BLOCK_BOUNDARY_APIS:
                updated_events.append(event)
                continue

            motif_key = _appendix_ab_block_motif_key(event.api, counts)
            if event.api == "ncclAllReduce" and motif_key == _APPENDIX_AB_SELECTED_ALLREDUCE_MOTIF_KEY:
                updated_event = replace(
                    event,
                    extras={
                        **event.extras,
                        "appendix_ab_allreduce_motif_key": motif_key,
                        "appendix_ab_allreduce_family_label": (
                            _APPENDIX_AB_SELECTED_ALLREDUCE_FAMILY_LABEL
                        ),
                        "appendix_ab_allreduce_kernel_bucket": "4-8",
                        "appendix_ab_allreduce_gemm_bucket": "2-3",
                        "appendix_ab_allreduce_strided_bucket": "2-3",
                        "appendix_ab_allreduce_selection_basis": (
                            _APPENDIX_AB_ALLREDUCE_SELECTION_BASIS
                        ),
                    },
                )
                selected_by_id[event.id] = updated_event
                updated_events.append(updated_event)
            else:
                updated_events.append(event)
            counts = dict(zero_counts)
        updated_rank_events[rank] = tuple(updated_events)

    if not selected_by_id:
        return updated_rank_events, global_events
    updated_global_events = tuple(selected_by_id.get(event.id, event) for event in global_events)
    return updated_rank_events, updated_global_events


def _provider_counterpart_key(event: CollatedEvent, material_api: str, signature: str | None) -> str:
    signature_text = signature if signature not in (None, "") else "signature:unavailable"
    return f"rank={event.rank}|api={material_api}|ordinal={event.ordinal}|{signature_text}"


def _async_observer_actual_device_runtime_us(event: CollatedEvent) -> float | None:
    if str(event.extras.get("wrapper_runtime_contract") or "").strip().lower() != "async_runtime":
        return None
    if str(event.extras.get("runtime_observation_source") or "").strip() != _ASYNC_RUNTIME_OBSERVATION_SOURCE:
        return None
    observed_runtime = _float_payload(event.extras.get("observed_runtime_us"))
    if observed_runtime is None:
        return None
    return max(float(observed_runtime), 0.0)


def _with_actual_device_runtime_metadata(event: CollatedEvent, extras: dict[str, object]) -> dict[str, object]:
    """Expose existing async-runtime observer output as ledger-only metadata.

    This is intentionally additive.  It does not change duration selection,
    provider selection, replay behavior, or CppEvent/CUDA instrumentation.  The
    actual-device-runtime field is emitted only when the trace already carries
    the existing CppEvent async observer contract.
    """

    actual_runtime_us = _async_observer_actual_device_runtime_us(event)
    if actual_runtime_us is None:
        return extras
    if extras is event.extras:
        extras = dict(event.extras)
    extras.setdefault("actual_device_runtime_us", float(actual_runtime_us))
    extras.setdefault(
        "actual_runtime_measurement_kind",
        _ACTUAL_DEVICE_RUNTIME_MEASUREMENT_KIND,
    )
    extras.setdefault("material_api", event.api)
    material_api = str(extras.get("material_api") or event.api)
    if is_gemm_material_api(material_api):
        material_signature = _material_signature(event)
        if material_signature is not None:
            extras["material_signature"] = material_signature
        extras["material_signature_inputs"] = _material_signature_inputs(event)
    else:
        material_signature = str(extras.get("material_signature") or "").strip() or None
        if material_signature is None:
            material_signature = _material_signature(event)
            if material_signature is not None:
                extras.setdefault("material_signature", material_signature)
    extras.setdefault(
        "provider_counterpart_key",
        _provider_counterpart_key(event, material_api, material_signature),
    )
    extras.setdefault("actual_counterpart_rank", int(event.rank))
    stream = extras.get("launch_stream_id", extras.get("stream_id"))
    if stream is not None:
        extras.setdefault("stream", stream)
    extras.setdefault("actual_counterpart_window", "event")
    extras.setdefault("actual_counterpart_window_event_id", event.id)
    return extras


def estimate_low_level_event_us(
    event: CollatedEvent,
    estimator: Estimator,
    *,
    percentile: str = "p50",
    prev_api: str | None = None,
    world_size: int = 1,
    allow_kernel_launch_heuristic_fallback: bool = False,
    allow_weak_runtime_fallback: bool = False,
    use_observed_control_plane_wrapper_durations: bool = True,
    use_observed_semantic_wrapper_durations: bool = True,
) -> float:
    duration_us, _, _, _ = _estimate_low_level_event_with_source(
        event,
        estimator,
        percentile=percentile,
        prev_api=prev_api,
        world_size=world_size,
        allow_kernel_launch_heuristic_fallback=allow_kernel_launch_heuristic_fallback,
        allow_weak_runtime_fallback=allow_weak_runtime_fallback,
        use_observed_control_plane_wrapper_durations=use_observed_control_plane_wrapper_durations,
        use_observed_semantic_wrapper_durations=use_observed_semantic_wrapper_durations,
    )
    return duration_us


def _estimate_low_level_event_with_source(
    event: CollatedEvent,
    estimator: Estimator,
    *,
    percentile: str = "p50",
    prev_api: str | None = None,
    world_size: int = 1,
    allow_kernel_launch_heuristic_fallback: bool = False,
    allow_weak_runtime_fallback: bool = False,
    use_observed_control_plane_wrapper_durations: bool = True,
    use_observed_semantic_wrapper_durations: bool = True,
    rank_kernel_launch_count: int | None = None,
) -> tuple[float, str, float, bool]:
    """Estimate duration for one low-level event."""
    if event.op_type == "host_delay" or event.api == "__hostDelay__":
        return (
            float(event.extras.get("observed_gap_us", 0.0)),
            "observed_host_delay",
            0.0,
            False,
        )
    observed_wrapper_duration = _observed_wrapper_duration_us(
        event,
        use_observed_control_plane_wrapper_durations=use_observed_control_plane_wrapper_durations,
        use_observed_semantic_wrapper_durations=use_observed_semantic_wrapper_durations,
    )
    if observed_wrapper_duration is not None:
        return observed_wrapper_duration, "observed_wrapper_duration", 0.0, False
    if is_ignorable_setup_event(event):
        return 0.0, "ignored_setup", 0.0, False
    if is_low_overhead_api(event.api):
        return 1.0, "low_overhead_api", 0.0, False
    if allow_kernel_launch_heuristic_fallback:
        kernel_duration = _kernel_launch_event_us(
            event,
            rank_kernel_launch_count=rank_kernel_launch_count,
        )
        if kernel_duration is not None:
            return kernel_duration, "heuristic_kernel_launch", 0.0, False
    memory_duration = _memory_event_us(event)
    if memory_duration is not None:
        return memory_duration, "heuristic_memory_model", 0.0, False
    payload = {
        "api": event.api,
        "type": event.op_type,
        "module": event.module,
        "rank": event.rank,
        "ordinal": event.ordinal,
        "pid": event.pid,
        "tid": event.tid,
        "source": event.source.value,
        "world_size": world_size,
        "collective_group_id": event.collective_group_id,
        **event.extras,
    }
    if prev_api:
        payload["prev_api"] = prev_api
    estimator_start = perf_counter()
    decision = estimator.estimate_event_with_details(payload, percentile=percentile)
    estimator_seconds = perf_counter() - estimator_start
    if (
        _requires_strict_runtime_signal(event)
        and not allow_weak_runtime_fallback
        and _uses_weak_estimator_fallback(decision.source)
    ):
        raise RuntimeError(
            "Strict runtime estimation required for "
            f"{event.api}; refusing weak fallback source {decision.source}. "
            "Provide a calibrated provider/api-specific estimator signal or "
            "explicitly opt in to weak fallback."
        )
    source = f"estimator_{decision.source}"
    if decision.provider_name:
        source = f"{source}:{decision.provider_name}"
    return float(decision.duration_us), source, estimator_seconds, True


def _annotate_rank_events(
    rank: int,
    events: tuple[CollatedEvent, ...],
    *,
    estimator: Estimator,
    percentile: str,
    duration_source: str,
    world_size: int,
    allow_kernel_launch_heuristic_fallback: bool,
    allow_weak_runtime_fallback: bool,
    use_observed_control_plane_wrapper_durations: bool,
    use_observed_semantic_wrapper_durations: bool,
    host_delay_profile: HostDelayProfile | None = None,
    host_gap_profile: HostGapProfile | None = None,
    timing_recorder: AnnotationTimingRecorder | None = None,
) -> tuple[int, tuple[AnnotatedEvent, ...]]:
    annotated_rank_events: list[AnnotatedEvent] = []
    previous_api: str | None = None
    rank_runtime_estimation_seconds = 0.0
    rank_runtime_estimation_event_count = 0
    pass_through_event_count = 0
    kernel_launch_count = sum(1 for event in events if event.api == "cudaLaunchKernel")
    kernel_launch_count_text = str(kernel_launch_count) if kernel_launch_count else None
    next_api_by_host_delay_id: dict[str, str] = {}
    next_materialized_api: str | None = None
    for candidate in reversed(events):
        if candidate.api == "__hostDelay__" or candidate.op_type == "host_delay":
            if next_materialized_api is not None:
                next_api_by_host_delay_id[candidate.id] = next_materialized_api
        else:
            next_materialized_api = candidate.api
    for event in events:
        if event.op_type == "host_delay" or event.api == "__hostDelay__":
            pass_through_event_count += 1
            profiled_gap_us = (
                host_gap_profile.profiled_gap_us(
                    previous_api,
                    next_api_by_host_delay_id.get(event.id),
                    rank=event.rank,
                )
                if host_gap_profile is not None
                else None
            )
            host_delay_us = (
                profiled_gap_us
                if profiled_gap_us is not None
                else float(event.extras.get("observed_gap_us", 0.0))
            )
            duration_source_text = (
                "profiled_host_gap" if profiled_gap_us is not None else "observed_host_delay"
            )
            host_delay_extras = event.extras
            if profiled_gap_us is not None:
                host_delay_extras = dict(event.extras)
                host_delay_extras["host_gap_profile_source"] = "profiled_transition_gap"
                host_delay_extras["host_gap_profile_prev_api"] = previous_api
                host_delay_extras["host_gap_profile_next_api"] = next_api_by_host_delay_id.get(event.id)
                host_delay_extras["host_gap_profile_duration_us"] = float(profiled_gap_us)
            annotated_rank_events.append(
                AnnotatedEvent(
                    id=event.id,
                    rank=event.rank,
                    ordinal=event.ordinal,
                    source=event.source,
                    ts=event.ts,
                    pid=event.pid,
                    tid=event.tid,
                    module=event.module,
                    api=event.api,
                    op_type=event.op_type,
                    extras=host_delay_extras,
                    prev_event_id=event.prev_event_id,
                    collective_group_id=event.collective_group_id,
                    duration_us=max(host_delay_us, 0.0),
                    duration_source=duration_source_text,
                )
            )
            previous_api = event.api
            continue
        if not use_observed_control_plane_wrapper_durations:
            if is_ignorable_setup_event(event):
                pass_through_event_count += 1
                annotated_rank_events.append(
                    AnnotatedEvent(
                        id=event.id,
                        rank=event.rank,
                        ordinal=event.ordinal,
                        source=event.source,
                        ts=event.ts,
                        pid=event.pid,
                        tid=event.tid,
                        module=event.module,
                        api=event.api,
                        op_type=event.op_type,
                        extras=event.extras,
                        prev_event_id=event.prev_event_id,
                        collective_group_id=event.collective_group_id,
                        duration_us=0.0,
                        duration_source="ignored_setup",
                    )
                )
                previous_api = event.api
                continue
            if is_low_overhead_api(event.api):
                pass_through_event_count += 1
                annotated_rank_events.append(
                    AnnotatedEvent(
                        id=event.id,
                        rank=event.rank,
                        ordinal=event.ordinal,
                        source=event.source,
                        ts=event.ts,
                        pid=event.pid,
                        tid=event.tid,
                        module=event.module,
                        api=event.api,
                        op_type=event.op_type,
                        extras=event.extras,
                        prev_event_id=event.prev_event_id,
                        collective_group_id=event.collective_group_id,
                        duration_us=1.0,
                        duration_source="low_overhead_api",
                    )
                )
                previous_api = event.api
                continue
        event_for_estimation = event
        observed_dispatch_us = None
        if host_delay_profile is None:
            raw_observed_dispatch = event.extras.get("host_duration_us")
            if raw_observed_dispatch is not None:
                try:
                    observed_dispatch_us = max(float(raw_observed_dispatch), 0.0)
                except (TypeError, ValueError):
                    observed_dispatch_us = None
        profile_dispatch_us = (
            host_delay_profile.dispatch_duration_us(event)
            if host_delay_profile is not None
            else None
        )
        if profile_dispatch_us is not None:
            profiled_extras = dict(event.extras)
            profiled_extras["host_duration_us"] = float(profile_dispatch_us)
            profiled_extras["wrapper_runtime_contract"] = "dispatch_only"
            profiled_extras["host_delay_profile_source"] = "profiled_host_api_delay"
            event_for_estimation = replace(event, extras=profiled_extras)
        elif observed_dispatch_us is not None:
            observed_extras = dict(event.extras)
            observed_extras["host_duration_us"] = float(observed_dispatch_us)
            observed_extras["wrapper_runtime_contract"] = "dispatch_only"
            observed_extras["host_delay_profile_source"] = "emulation_observed_host_api_delay"
            event_for_estimation = replace(event, extras=observed_extras)
        (
            duration_us,
            resolved_duration_source,
            estimator_seconds,
            estimator_called,
        ) = _estimate_low_level_event_with_source(
            event_for_estimation,
            estimator,
            percentile=percentile,
            prev_api=previous_api,
            world_size=world_size,
            allow_kernel_launch_heuristic_fallback=allow_kernel_launch_heuristic_fallback,
            allow_weak_runtime_fallback=allow_weak_runtime_fallback,
            use_observed_control_plane_wrapper_durations=use_observed_control_plane_wrapper_durations,
            use_observed_semantic_wrapper_durations=use_observed_semantic_wrapper_durations,
            rank_kernel_launch_count=kernel_launch_count,
        )
        if estimator_called:
            rank_runtime_estimation_seconds += estimator_seconds
            rank_runtime_estimation_event_count += 1
        else:
            pass_through_event_count += 1
        if timing_recorder is not None:
            timing_recorder.record_duration_source(
                api=event_for_estimation.api,
                source=resolved_duration_source,
                duration_us=float(duration_us),
            )
        annotated_extras = event_for_estimation.extras
        if (
            kernel_launch_count_text is not None
            and event_for_estimation.api == "cudaLaunchKernel"
            and "rank_kernel_launch_count" not in annotated_extras
        ):
            annotated_extras = dict(event_for_estimation.extras)
            annotated_extras["rank_kernel_launch_count"] = kernel_launch_count_text
        if (
            resolved_duration_source.startswith("estimator_")
            and "provider_duration_source_expected" not in annotated_extras
        ):
            if annotated_extras is event_for_estimation.extras:
                annotated_extras = dict(event_for_estimation.extras)
            annotated_extras["provider_duration_source_expected"] = resolved_duration_source
            annotated_extras.setdefault("material_api", event_for_estimation.api)
            material_api = str(annotated_extras.get("material_api") or event_for_estimation.api)
            material_signature = _material_signature(event_for_estimation)
            if is_gemm_material_api(material_api):
                if material_signature is not None:
                    annotated_extras["material_signature"] = material_signature
                annotated_extras["material_signature_inputs"] = _material_signature_inputs(event_for_estimation)
            else:
                if material_signature is not None:
                    annotated_extras.setdefault("material_signature", material_signature)
                annotated_extras.setdefault("material_signature_inputs", _material_signature_inputs(event_for_estimation))
            stream_id = _stream_id(event_for_estimation)
            annotated_extras.setdefault("raw_stream_id", stream_id)
            annotated_extras.setdefault("canonical_stream_id", stream_id)
            annotated_extras.setdefault(
                "provider_counterpart_key",
                _provider_counterpart_key(event_for_estimation, material_api, material_signature),
            )
        annotated_extras = _with_actual_device_runtime_metadata(
            event_for_estimation,
            annotated_extras,
        )
        annotated = AnnotatedEvent(
            id=event_for_estimation.id,
            rank=event_for_estimation.rank,
            ordinal=event_for_estimation.ordinal,
            source=event_for_estimation.source,
            ts=event_for_estimation.ts,
            pid=event_for_estimation.pid,
            tid=event_for_estimation.tid,
            module=event_for_estimation.module,
            api=event_for_estimation.api,
            op_type=event_for_estimation.op_type,
            extras=annotated_extras,
            prev_event_id=event_for_estimation.prev_event_id,
            collective_group_id=event_for_estimation.collective_group_id,
            duration_us=max(duration_us, 0.0),
            duration_source=(
                duration_source
                if duration_source != "estimator" and resolved_duration_source.startswith("estimator_")
                else resolved_duration_source
            ),
        )
        annotated_rank_events.append(annotated)
        previous_api = event_for_estimation.api
    if timing_recorder is not None:
        timing_recorder.record_rank(
            rank,
            runtime_estimation_seconds=rank_runtime_estimation_seconds,
            runtime_estimation_event_count=rank_runtime_estimation_event_count,
            annotated_event_count=len(events),
            pass_through_event_count=pass_through_event_count,
        )
    return rank, tuple(annotated_rank_events)


def _attach_collective_group_duration_metadata(
    rank_events: dict[int, tuple[AnnotatedEvent, ...]],
    global_events: tuple[AnnotatedEvent, ...],
    *,
    estimator: Estimator,
    percentile: str,
    world_size: int,
    timing_recorder: AnnotationTimingRecorder | None = None,
    allow_collective_group_fallback: bool = False,
) -> tuple[dict[int, tuple[AnnotatedEvent, ...]], tuple[AnnotatedEvent, ...]]:
    grouped_events: dict[str, _CollectiveGroupAggregate] = {}
    for events in rank_events.values():
        for event in events:
            if event.collective_group_id is not None:
                aggregate = grouped_events.get(event.collective_group_id)
                if aggregate is None:
                    grouped_events[event.collective_group_id] = _CollectiveGroupAggregate(
                        representative=event,
                        max_duration_us=float(event.duration_us),
                        event_count=1,
                    )
                    continue
                if (event.ts, event.rank, event.ordinal) < (
                    aggregate.representative.ts,
                    aggregate.representative.rank,
                    aggregate.representative.ordinal,
                ):
                    aggregate.representative = event
                aggregate.max_duration_us = max(aggregate.max_duration_us, float(event.duration_us))
                aggregate.event_count += 1

    group_duration_us: dict[str, float] = {}
    group_duration_basis: dict[str, str] = {}
    for group_id, aggregate in grouped_events.items():
        fallback_duration_us = aggregate.max_duration_us
        fallback_basis = "max_member_duration"
        representative = aggregate.representative
        participant_count = max(
            _int_payload(representative.extras.get("participant_count")) or aggregate.event_count,
            1,
        )
        payload = {
            "api": representative.api,
            "type": representative.op_type,
            "module": representative.module,
            "rank": representative.rank,
            "ordinal": representative.ordinal,
            "pid": representative.pid,
            "tid": representative.tid,
            "source": representative.source.value,
            "world_size": world_size,
            "collective_group_id": representative.collective_group_id,
            **representative.extras,
            "participant_count": participant_count,
        }
        if payload.get("collective_api") in (None, ""):
            payload["collective_api"] = representative.api
        if payload.get("communicator_size") in (None, ""):
            payload["communicator_size"] = participant_count
        payload = _normalize_p2p_group_runtime_payload(payload)

        group_estimation_start = perf_counter()
        decision = estimator.estimate_collective_group_with_details(
            payload,
            percentile=percentile,
        )
        group_estimation_seconds = perf_counter() - group_estimation_start
        if timing_recorder is not None:
            timing_recorder.record_collective_group_estimation(
                elapsed_seconds=group_estimation_seconds,
                hit=decision is not None,
            )
        if decision is None:
            if not allow_collective_group_fallback:
                raise RuntimeError(
                    "missing collective group duration estimate for "
                    f"group_id={group_id}, api={representative.api}, "
                    f"participant_count={participant_count}; paper-facing replay must not "
                    "fall back to observed max_member_duration"
                )
            group_duration_us[group_id] = fallback_duration_us
            group_duration_basis[group_id] = fallback_basis
            if timing_recorder is not None:
                timing_recorder.record_collective_group_duration_basis(fallback_basis)
            continue
        group_duration_us[group_id] = float(decision.duration_us)
        if decision.provider_name:
            group_duration_basis[group_id] = f"group_provider:{decision.provider_name}"
        else:
            group_duration_basis[group_id] = f"group_{decision.source}"
        if timing_recorder is not None:
            timing_recorder.record_collective_group_duration_basis(group_duration_basis[group_id])

    if not group_duration_us:
        return rank_events, global_events

    updated_rank_events: dict[int, tuple[AnnotatedEvent, ...]] = {}
    updated_collective_events_by_id: dict[str, AnnotatedEvent] = {}
    for rank, events in sorted(rank_events.items()):
        updated_events: list[AnnotatedEvent] = []
        for event in events:
            if event.collective_group_id is None:
                updated_events.append(event)
                continue
            extras = {
                **event.extras,
                "collective_group_duration_us": group_duration_us[event.collective_group_id],
                "collective_group_duration_basis": group_duration_basis[event.collective_group_id],
            }
            if _is_p2p_collective_group_member(event):
                extras.setdefault("member_api", event.api)
                if event.extras.get("collective") not in (None, ""):
                    extras.setdefault("member_collective", event.extras.get("collective"))
                extras.setdefault("collective_group_runtime_api", _NCCL_P2P_GROUP_API)
                extras.setdefault("collective_group_runtime_collective", "p2p")
                extras.setdefault("collective_group_runtime_collective_api", _NCCL_P2P_GROUP_API)
                extras.setdefault("collective_group_runtime_participant_count", 2)
            updated_event = replace(
                event,
                extras=extras,
            )
            updated_events.append(updated_event)
            updated_collective_events_by_id[event.id] = updated_event
        updated_rank_events[rank] = tuple(updated_events)

    global_events = tuple(
        updated_collective_events_by_id.get(event.id, event)
        for event in global_events
    )
    return updated_rank_events, global_events


def collective_group_duration_summary(trace: AnnotatedTrace) -> dict[str, Any]:
    group_basis_by_id: dict[str, str] = {}
    group_ids_with_duration_metadata: set[str] = set()
    groups_with_metadata = 0
    duration_source_counts: dict[str, int] = defaultdict(int)
    strict_runtime_signal_duration_source_counts: dict[str, int] = defaultdict(int)
    strict_runtime_signal_wrapper_timing_contract_counts: dict[str, int] = defaultdict(int)
    strict_runtime_signal_count_by_api: dict[str, int] = defaultdict(int)
    strict_runtime_signal_duration_source_counts_by_api: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    strict_runtime_signal_wrapper_timing_contract_counts_by_api: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    wrapper_timing_field_count = 0
    explicit_direct_runtime_field_count = 0
    direct_wrapper_runtime_count = 0
    strict_runtime_signal_count = 0
    strict_runtime_signal_with_wrapper_timing_field_count = 0
    strict_runtime_signal_with_explicit_direct_runtime_field_count = 0
    strict_runtime_signal_with_direct_wrapper_runtime_count = 0
    strict_runtime_signal_with_direct_runtime_contract_count = 0
    strict_runtime_signal_observed_wrapper_duration_count = 0
    strict_runtime_signal_observed_wrapper_duration_dispatch_only_count = 0
    strict_runtime_signal_observed_wrapper_duration_direct_runtime_count = 0
    for event in trace.global_events:
        duration_source_counts[str(event.duration_source)] += 1
        timing_info = _resolve_wrapper_timing_info(event)
        direct_runtime_observation = None
        direct_runtime_us = timing_info.direct_runtime_us
        raw_wrapper_us = timing_info.raw_wrapper_us
        if direct_runtime_us is not None and direct_runtime_us <= 0.0:
            direct_runtime_us = None
        if raw_wrapper_us is not None and raw_wrapper_us <= 0.0:
            raw_wrapper_us = None
        if event.api in _OBSERVED_WRAPPER_RUNTIME_APIS:
            direct_runtime_observation = (
                direct_runtime_us if direct_runtime_us is not None else raw_wrapper_us
            )
        elif event.op_type in _STRICT_RUNTIME_SIGNAL_TYPES:
            if timing_info.strict_runtime_wrapper_timing_contract == "direct_runtime":
                direct_runtime_observation = (
                    direct_runtime_us if direct_runtime_us is not None else raw_wrapper_us
                )
        has_wrapper_timing_field = timing_info.has_wrapper_timing_field
        has_explicit_direct_runtime_field = timing_info.direct_runtime_us is not None
        has_direct_wrapper_runtime = (
            direct_runtime_observation is not None and direct_runtime_observation > 0.0
        )
        if has_wrapper_timing_field:
            wrapper_timing_field_count += 1
        if has_explicit_direct_runtime_field:
            explicit_direct_runtime_field_count += 1
        if has_direct_wrapper_runtime:
            direct_wrapper_runtime_count += 1
        if _is_strict_runtime_signal_api_type(event.api, event.op_type):
            strict_runtime_signal_count += 1
            strict_runtime_signal_count_by_api[event.api] += 1
            strict_runtime_signal_duration_source_counts[str(event.duration_source)] += 1
            strict_runtime_signal_duration_source_counts_by_api[event.api][
                str(event.duration_source)
            ] += 1
            contract = timing_info.strict_runtime_wrapper_timing_contract
            strict_runtime_signal_wrapper_timing_contract_counts[contract] += 1
            strict_runtime_signal_wrapper_timing_contract_counts_by_api[event.api][contract] += 1
            if has_wrapper_timing_field:
                strict_runtime_signal_with_wrapper_timing_field_count += 1
            if has_explicit_direct_runtime_field:
                strict_runtime_signal_with_explicit_direct_runtime_field_count += 1
            if has_direct_wrapper_runtime:
                strict_runtime_signal_with_direct_wrapper_runtime_count += 1
            if contract == "direct_runtime":
                strict_runtime_signal_with_direct_runtime_contract_count += 1
            if str(event.duration_source) == "observed_wrapper_duration":
                strict_runtime_signal_observed_wrapper_duration_count += 1
                if contract == "dispatch_only":
                    strict_runtime_signal_observed_wrapper_duration_dispatch_only_count += 1
                if contract == "direct_runtime":
                    strict_runtime_signal_observed_wrapper_duration_direct_runtime_count += 1
        if event.collective_group_id is None:
            continue
        raw_duration = event.extras.get("collective_group_duration_us")
        if raw_duration not in (None, ""):
            groups_with_metadata += 1
            group_ids_with_duration_metadata.add(event.collective_group_id)
        basis = str(event.extras.get("collective_group_duration_basis") or "missing")
        group_basis_by_id.setdefault(event.collective_group_id, basis)

    basis_counts: dict[str, int] = defaultdict(int)
    for basis in group_basis_by_id.values():
        basis_counts[basis] += 1

    return {
        "collective_group_count": len(group_basis_by_id),
        "collective_group_with_duration_metadata_count": len(group_ids_with_duration_metadata),
        "collective_group_duration_basis_counts": dict(sorted(basis_counts.items())),
        "collective_event_with_duration_metadata_count": groups_with_metadata,
        "duration_source_counts": dict(sorted(duration_source_counts.items())),
        "strict_runtime_signal_duration_source_counts": dict(
            sorted(strict_runtime_signal_duration_source_counts.items())
        ),
        "strict_runtime_signal_wrapper_timing_contract_counts": dict(
            sorted(strict_runtime_signal_wrapper_timing_contract_counts.items())
        ),
        "strict_runtime_signal_event_count_by_api": {
            api: int(count)
            for api, count in sorted(strict_runtime_signal_count_by_api.items())
        },
        "strict_runtime_signal_duration_source_counts_by_api": {
            api: dict(sorted(counts.items()))
            for api, counts in sorted(strict_runtime_signal_duration_source_counts_by_api.items())
        },
        "strict_runtime_signal_wrapper_timing_contract_counts_by_api": {
            api: dict(sorted(counts.items()))
            for api, counts in sorted(strict_runtime_signal_wrapper_timing_contract_counts_by_api.items())
        },
        "event_with_wrapper_timing_field_count": wrapper_timing_field_count,
        "event_with_explicit_direct_runtime_field_count": explicit_direct_runtime_field_count,
        "event_with_direct_wrapper_runtime_count": direct_wrapper_runtime_count,
        "strict_runtime_signal_event_count": strict_runtime_signal_count,
        "strict_runtime_signal_event_with_wrapper_timing_field_count": (
            strict_runtime_signal_with_wrapper_timing_field_count
        ),
        "strict_runtime_signal_event_with_explicit_direct_runtime_field_count": (
            strict_runtime_signal_with_explicit_direct_runtime_field_count
        ),
        "strict_runtime_signal_event_with_direct_wrapper_runtime_count": (
            strict_runtime_signal_with_direct_wrapper_runtime_count
        ),
        "strict_runtime_signal_event_with_direct_runtime_contract_count": (
            strict_runtime_signal_with_direct_runtime_contract_count
        ),
        "strict_runtime_signal_observed_wrapper_duration_count": (
            strict_runtime_signal_observed_wrapper_duration_count
        ),
        "strict_runtime_signal_observed_wrapper_duration_dispatch_only_count": (
            strict_runtime_signal_observed_wrapper_duration_dispatch_only_count
        ),
        "strict_runtime_signal_observed_wrapper_duration_direct_runtime_count": (
            strict_runtime_signal_observed_wrapper_duration_direct_runtime_count
        ),
    }


def annotate_collated_trace(
    collated: CollatedTrace,
    estimator: Estimator,
    *,
    percentile: str = "p50",
    duration_source: str = "estimator",
    allow_kernel_launch_heuristic_fallback: bool = False,
    allow_weak_runtime_fallback: bool = False,
    allow_collective_group_fallback: bool = False,
    use_observed_control_plane_wrapper_durations: bool = True,
    use_observed_semantic_wrapper_durations: bool = True,
    host_delay_profile: HostDelayProfile | None = None,
    host_gap_profile: HostGapProfile | None = None,
    parallel_workers: int = 1,
    timing_recorder: AnnotationTimingRecorder | None = None,
) -> AnnotatedTrace:
    """Attach runtime estimates to all events in a collated low-level trace."""
    rank_events: dict[int, tuple[AnnotatedEvent, ...]] = {}

    rank_items = list(sorted(collated.rank_events.items()))
    max_workers = max(int(parallel_workers), 1)
    if max_workers > 1 and len(rank_items) > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(rank_items))) as executor:
            futures = [
                executor.submit(
                    _annotate_rank_events,
                    rank,
                    events,
                    estimator=estimator,
                    percentile=percentile,
                    duration_source=duration_source,
                    world_size=collated.world_size,
                    allow_kernel_launch_heuristic_fallback=allow_kernel_launch_heuristic_fallback,
                    allow_weak_runtime_fallback=allow_weak_runtime_fallback,
                    use_observed_control_plane_wrapper_durations=use_observed_control_plane_wrapper_durations,
                    use_observed_semantic_wrapper_durations=use_observed_semantic_wrapper_durations,
                    host_delay_profile=host_delay_profile,
                    host_gap_profile=host_gap_profile,
                    timing_recorder=timing_recorder,
                )
                for rank, events in rank_items
            ]
            for future in futures:
                rank, annotated_rank_events = future.result()
                rank_events[rank] = annotated_rank_events
    else:
        for rank, events in rank_items:
            rank, annotated_rank_events = _annotate_rank_events(
                rank,
                events,
                estimator=estimator,
                percentile=percentile,
                duration_source=duration_source,
                world_size=collated.world_size,
                allow_kernel_launch_heuristic_fallback=allow_kernel_launch_heuristic_fallback,
                allow_weak_runtime_fallback=allow_weak_runtime_fallback,
                use_observed_control_plane_wrapper_durations=use_observed_control_plane_wrapper_durations,
                use_observed_semantic_wrapper_durations=use_observed_semantic_wrapper_durations,
                host_delay_profile=host_delay_profile,
                host_gap_profile=host_gap_profile,
                timing_recorder=timing_recorder,
            )
            rank_events[rank] = annotated_rank_events

    annotated_events_by_id = {
        event.id: event
        for events in rank_events.values()
        for event in events
    }
    annotated_global = tuple(
        annotated_events_by_id[event.id]
        for event in collated.global_events
    )
    rank_events, annotated_global = _attach_collective_group_duration_metadata(
        rank_events,
        annotated_global,
        estimator=estimator,
        percentile=percentile,
        world_size=collated.world_size,
        timing_recorder=timing_recorder,
        allow_collective_group_fallback=allow_collective_group_fallback,
    )
    rank_events, annotated_global = _attach_appendix_ab_p2p_selection_metadata(
        rank_events,
        annotated_global,
    )
    rank_events, annotated_global = _attach_appendix_ab_allreduce_selection_metadata(
        rank_events,
        annotated_global,
    )
    return AnnotatedTrace(
        trace_dir=collated.trace_dir,
        source=collated.source,
        rank_events=rank_events,
        global_events=annotated_global,
        collective_groups=dict(collated.collective_groups),
        original_world_size=collated.world_size,
        captured_world_size=collated.profiled_world_size,
        profiled_rank_groups=dict(collated.profiled_rank_groups),
        rank_host_machines=dict(collated.rank_host_machines),
        rank_host_dispatch_queues=dict(collated.rank_host_dispatch_queues),
        communicator_memberships=dict(collated.communicator_memberships),
        host_timing_dispatch_scope_resolved=collated.host_timing_dispatch_scope_resolved,
        logical_rank_materialized=collated.logical_rank_materialized,
        trace_window=collated.trace_window,
        fidelity_windows=dict(collated.fidelity_windows),
    )
