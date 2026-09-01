"""
Capture real CUDA/NCCL/cuBLAS wrapper events into Maya-lite rank_*.jsonl files.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import runpy
import shlex
import socket
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .communicators import recover_communicator_topology_from_events
from .io import (
    estimate_rank_trace_window,
    fidelity_window_from_payload,
    iter_rank_trace_events,
)
from .markers import load_step_markers, resolve_step_window_from_markers
from .planner import plan_profiled_rank_groups, profiled_ranks_for_groups
from .schema import TraceSource, normalize_op_type
from .launch_neighborhood import (
    LAUNCH_NEIGHBORHOOD_EQUIVALENCE_APIS,
    LAUNCH_NEIGHBORHOOD_EQUIVALENCE_ENV_KEYS,
    build_launch_neighborhood_equivalence_metadata,
)
from .material_signature import canonical_gemm_material_signature, is_gemm_material_api

_ASYNC_RUNTIME_OBSERVATION_SOURCE = "capture_real_cuda_event"
_ACTUAL_CUDA_EVENT_COUNTERPART_SCHEMA_VERSION = "actual_cuda_event_record_wait_release_v1"
_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_SCHEMA_VERSION = (
    "appendix_ab_p2p_actual_counterpart_release_metadata_v1"
)
_SHARED_PHASE_ANCHOR_COUNTERPART_SCHEMA_VERSION = (
    "shared_all_rank_phase_anchor_counterpart_replay_causal_edge_metadata_v1"
)
_SHARED_PHASE_ANCHOR_COMMON_BASIS_SCHEMA_VERSION = (
    "shared_phase_anchor_common_basis_key_fields_v1"
)
_SELECTED_ALLREDUCE_RELEASE_PARTICIPANT_HOST_DISPATCH_PHASE_SCHEMA_VERSION = (
    "selected_allreduce_release_participant_host_dispatch_phase_counterpart_metadata_v1"
)
_NCCL_WAIT_RELEASE_COUNTERPART_SCHEMA_VERSION = (
    "nccl_wait_release_stream_namespace_counterpart_metadata_v1"
)
_HOST_CONTROL_BOUNDARY_COUNTERPART_SCHEMA_VERSION = (
    "host_control_boundary_visibility_unblocker_v2_row_evidence_v1"
)
_HOST_CONTROL_ENVELOPE_COUNTERPART_SCHEMA_VERSION = (
    "host_control_replay_envelope_counterpart_metadata_v1"
)
_HOST_CONTROL_PRODUCER_VISIBILITY_SCHEMA_VERSION = (
    "host_control_producer_visibility_nonoverlap_v1"
)
_HOST_CONTROL_LAUNCH_NEIGHBORHOOD_SCHEMA_VERSION = (
    "host_control_launch_neighborhood_visibility_counterpart_isolation_v1"
)
_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_SCHEMA_VERSION = (
    "generic_replay_placement_envelope_actual_counterpart_metadata_v1"
)
_COMPONENT_STRICT_COUNTERPART_ACTUAL_SCHEMA_VERSION = (
    "component_strict_counterpart_actual_metadata_evidence_v1"
)
_GEMM_ADJACENT_HOSTDELAY_BOUNDARY_SCHEMA_VERSION = (
    "gemm_adjacent_hostdelay_boundary_counterpart_visibility_count_once_metadata_v1"
)
_GEMM_ADJACENT_PRODUCER_VISIBILITY_SCHEMA_VERSION = (
    "gemm_adjacent_hostdelay_producer_visibility_v1"
)
_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_SCHEMA_VERSION = (
    "cudaLaunch_GEMM_hostdispatch_strict_occurrence_gap_metadata_v1"
)
_JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_SCHEMA_VERSION = (
    "joined_gemm_stream_queue_wait_actual_counterpart_metadata_v1"
)
_ACTUAL_CUDA_EVENT_COUNTERPART_ENV_KEYS = (
    "MAYA_ENABLE_ACTUAL_CUDA_EVENT_COUNTERPART_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_ACTUAL_CUDA_EVENT_COUNTERPART_DIAGNOSTICS",
)
_HOST_CONTROL_BOUNDARY_COUNTERPART_ENV_KEYS = (
    "MAYA_ENABLE_HOST_CONTROL_BOUNDARY_COUNTERPART_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_HOST_CONTROL_BOUNDARY_COUNTERPART_DIAGNOSTICS",
)
_HOST_CONTROL_ENVELOPE_COUNTERPART_ENV_KEYS = (
    "MAYA_ENABLE_HOST_CONTROL_ENVELOPE_COUNTERPART_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_HOST_CONTROL_ENVELOPE_COUNTERPART_DIAGNOSTICS",
)
_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_ENV_KEYS = (
    "MAYA_ENABLE_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_DIAGNOSTICS",
)
_SHARED_PHASE_ANCHOR_COUNTERPART_ENV_KEYS = (
    "MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
)
_SHARED_PHASE_ANCHOR_COMMON_BASIS_ENV_KEYS = (
    "MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
)
_SELECTED_ALLREDUCE_RELEASE_PARTICIPANT_HOST_DISPATCH_PHASE_ENV_KEYS = (
    "MAYA_ENABLE_SELECTED_ALLREDUCE_RELEASE_PARTICIPANT_HOST_DISPATCH_PHASE_COUNTERPART_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_SELECTED_ALLREDUCE_RELEASE_PARTICIPANT_HOST_DISPATCH_PHASE_COUNTERPART_DIAGNOSTICS",
)
_NCCL_WAIT_RELEASE_COUNTERPART_ENV_KEYS = (
    "MAYA_ENABLE_NCCL_WAIT_RELEASE_COUNTERPART_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_NCCL_WAIT_RELEASE_COUNTERPART_DIAGNOSTICS",
)
_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_ENV_KEYS = (
    "MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_DIAGNOSTICS",
)
_COMPONENT_STRICT_COUNTERPART_METADATA_ENV_KEYS = (
    "MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
)
_GEMM_ADJACENT_HOSTDELAY_BOUNDARY_ACTUAL_COUNTERPART_ENV_KEYS = (
    "MAYA_ENABLE_GEMM_ADJACENT_HOSTDELAY_BOUNDARY_ACTUAL_COUNTERPART_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_GEMM_ADJACENT_HOSTDELAY_BOUNDARY_ACTUAL_COUNTERPART_DIAGNOSTICS",
    "MAYA_ENABLE_GEMM_HOSTDELAY_BOUNDARY_COUNTERPART_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_GEMM_HOSTDELAY_BOUNDARY_COUNTERPART_DIAGNOSTICS",
)
_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_ENV_KEYS = (
    "MAYA_ENABLE_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_METADATA_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_METADATA_DIAGNOSTICS",
)
_JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_ENV_KEYS = (
    "MAYA_ENABLE_JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_METADATA_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_METADATA_DIAGNOSTICS",
)
_HOST_CONTROL_BOUNDARY_APIS = {
    "__cudaPopCallConfiguration",
    "cudaGetDevice",
    "cudaLaunchKernel",
    "cublasSetStream_v2",
    "cudaEventRecord",
    "cudaEventRecordWithFlags",
    "cudaStreamWaitEvent",
}
_HOST_CONTROL_SELECTED_BOUNDARY_FAMILIES = {
    "__cudaPopCallConfiguration -> cudaLaunchKernel",
    "cudaGetDevice -> cublasSetStream_v2",
    "cudaGetDevice -> cudaEventRecord",
    "cudaGetDevice -> cudaEventRecordWithFlags",
    "cudaGetDevice -> cudaStreamWaitEvent",
}
_HOST_CONTROL_PRODUCER_VISIBILITY_APIS = {
    "cudaGetDevice",
    "cudaLaunchKernel",
    "cublasSetStream_v2",
    "cudaEventRecord",
    "cudaEventRecordWithFlags",
    "cudaStreamWaitEvent",
}
_GEMM_ADJACENT_HOSTDELAY_GEMM_APIS = {
    "cublasGemmEx",
    "cublasGemmStridedBatchedEx",
}
_GEMM_ADJACENT_HOSTDELAY_ADJACENT_APIS = {
    "cublasSetStream_v2",
    "cudaLaunchKernel",
}
_GEMM_ADJACENT_HOSTDELAY_ENDPOINT_APIS = (
    _GEMM_ADJACENT_HOSTDELAY_GEMM_APIS
    | _GEMM_ADJACENT_HOSTDELAY_ADJACENT_APIS
)
_CUDALAUNCH_GEMM_HOSTDISPATCH_TARGET_APIS = {
    "cudaLaunchKernel",
    "cublasGemmEx",
    "cublasGemmStridedBatchedEx",
}
_HOSTDELAY_SEMANTIC_PREDECESSOR_CONTROL_QUERY_APIS = {
    "cudaGetDevice",
    "cudaGetDeviceCount",
    "cudaGetDeviceProperties",
    "cudaGetLastError",
    "cudaPeekAtLastError",
    "cudaEventQuery",
    "cudaSetDevice",
    "ncclCommGetAsyncError",
}
_JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_TARGET_APIS = set(
    _CUDALAUNCH_GEMM_HOSTDISPATCH_TARGET_APIS
)
_BOUNDARY_VISIBILITY_JSON_FIELDS = {
    "boundary_visibility_segments",
    "host_control_producer_visibility_segments",
}
_BOUNDARY_VISIBILITY_FLOAT_FIELDS = {
    "wrapper_segment_sum_us",
    "wrapper_segment_unattributed_us",
    "paper_visible_host_duration_us",
    "instrumentation_only_duration_us",
    "caller_visible_elapsed_us",
    "actual_launch_control_dispatch_us",
    "actual_launch_api_body_us",
    "actual_launch_instrumentation_only_us",
}
_HOST_TIMING_LINE_DISABLED_PAYLOAD: dict[str, object] = {
    "host_timing_paper_alignment_line": "disabled",
    "host_timing_line_family": "disabled",
    "host_timing_line_contract_version": "phase4_v1",
    "host_timing_profile_backed": False,
    "host_timing_paper_alignment_ready": False,
}
_ROUTE_METADATA_CONFLICT_KEYS = {
    "figure13_route",
    "auto_profiled_strategy",
    "dynamic_first_iteration_dedup",
    "collective_mode",
    "host_timing_mode",
    "host_timing_dispatch_scope",
    "host_timing_schedule_surface",
    "validation_mode",
    "workload_args",
}
_REQUIRED_ROUTE_METADATA_KEYS = set(_ROUTE_METADATA_CONFLICT_KEYS)


def _module_for_api(api_name: str) -> str:
    if api_name.startswith("nccl"):
        return "libnccl.so.2"
    if api_name.startswith("cublasLt"):
        return "libcublasLt.so.12"
    if api_name.startswith("cublas"):
        return "libcublas.so.12"
    if api_name.startswith("cu") and not api_name.startswith("cuda"):
        return "libcuda.so.1"
    return "libcudart.so.12"


def _type_for_event(api_name: str, kind_name: str) -> str:
    resolved = "other"
    if api_name == "cudaLaunchKernel":
        resolved = "kernel_launch"
    elif api_name in {
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
        resolved = "nccl_collective"
    elif kind_name in {
        "AllReduce",
        "AllGather",
        "AllToAll",
        "AllToAllv",
        "Broadcast",
        "Reduce",
        "ReduceScatter",
        "Collective",
    }:
        resolved = "nccl_collective"
    elif api_name in {"cudaMemcpy", "cudaMemcpyAsync"}:
        resolved = "mem_copy"
    elif kind_name in {"MemcpyHostToDevice", "MemcpyDeviceToHost", "MemcpyDeviceToDevice"}:
        resolved = "mem_copy"
    elif api_name in {"cudaMalloc", "cudaMallocAsync", "cudaFree", "cudaFreeAsync"}:
        resolved = "mem_alloc"
    elif kind_name in {"MemoryAllocation", "MemoryFree"}:
        resolved = "mem_alloc"
    elif kind_name == "ComputeKernel":
        resolved = "kernel_launch"
    elif api_name.startswith("cudaStream") or api_name.startswith("cudaEvent"):
        resolved = "stream_op"
    elif api_name.startswith("cuda") or api_name.startswith("cu"):
        resolved = "context_op"
    return normalize_op_type(api_name, resolved)


def _wrapper_runtime_contract_for_event(api_name: str, event_type: str) -> str | None:
    if event_type in {"kernel_launch", "blas_compute", "nccl_collective"}:
        return "dispatch_only"
    if api_name in {
        "cudaMemcpyAsync",
        "cudaMallocAsync",
        "cudaFreeAsync",
        "cudaEventRecord",
        "cudaEventRecordWithFlags",
        "cudaStreamWaitEvent",
        "cublasSetStream_v2",
    }:
        return "dispatch_only"
    if api_name in {
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
    }:
        return "direct_runtime"
    return None


def _float_payload(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _normalized_payload_text(value: object) -> str | None:
    if value in (None, ""):
        return None
    resolved = str(value).strip()
    return resolved or None


def _cupti_activity_timestamp_text(record: dict[str, object] | None, key: str) -> str | None:
    if record is None:
        return None
    return _normalized_payload_text(record.get(key))


def _cupti_activity_stream_order_gap_ticks(
    previous_kernel_end: str | None,
    current_kernel_start: str | None,
) -> int | None:
    if previous_kernel_end is None or current_kernel_start is None:
        return None
    try:
        return int(current_kernel_start) - int(previous_kernel_end)
    except ValueError:
        return None


def _cupti_activity_previous_kernel_start_text(
    record: dict[str, object] | None,
) -> str | None:
    return _cupti_activity_timestamp_text(
        record,
        "cupti_activity_last_kernel_start",
    ) or _cupti_activity_timestamp_text(
        record,
        "cupti_activity_first_kernel_start",
    )


def _cupti_activity_previous_kernel_end_text(
    record: dict[str, object] | None,
) -> str | None:
    return _cupti_activity_timestamp_text(
        record,
        "cupti_activity_last_kernel_end",
    ) or _cupti_activity_timestamp_text(
        record,
        "cupti_activity_first_kernel_end",
    )


def _cupti_activity_previous_kernel_stream_id_text(
    record: dict[str, object] | None,
) -> str | None:
    if record is None:
        return None
    return _normalized_payload_text(
        record.get("cupti_activity_last_kernel_stream_id")
        or record.get("cupti_activity_first_kernel_stream_id")
    )


def _has_cupti_activity_device_predecessor_timing(
    record: dict[str, object] | None,
) -> bool:
    return _cupti_activity_previous_kernel_end_text(record) is not None


def _cupti_activity_stream_id_pair_status(
    previous_kernel_stream_id: str | None,
    current_kernel_stream_id: str | None,
) -> str:
    if previous_kernel_stream_id is None or current_kernel_stream_id is None:
        return "unavailable"
    if previous_kernel_stream_id == current_kernel_stream_id:
        return "same_cupti_stream_id_observed"
    return "different_cupti_stream_id_observed"


def _int_payload(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value), 0)
    except (TypeError, ValueError):
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return None


def _env_flag_enabled(*keys: str) -> bool:
    for key in keys:
        value = os.environ.get(key)
        if value == "1":
            return True
    return False


def _env_flag_truthy(*keys: str) -> bool:
    truthy = {"1", "true", "yes", "on"}
    for key in keys:
        value = os.environ.get(key)
        if value is not None and value.strip().lower() in truthy:
            return True
    return False


def _timedelta_to_microseconds(value) -> int:
    return (
        (int(value.days) * 24 * 60 * 60 + int(value.seconds)) * 1_000_000
        + int(value.microseconds)
    )


def _is_strict_runtime_signal_event(api_name: str, event_type: str) -> bool:
    return api_name == "cudaLaunchKernel" or event_type in {
        "kernel_launch",
        "blas_compute",
        "nccl_collective",
    }


def _has_authoritative_async_runtime_observation(
    record: dict[str, object],
    *,
    api_name: str,
    event_type: str,
) -> bool:
    return (
        str(record.get("runtime_observation_source") or "").strip()
        == _ASYNC_RUNTIME_OBSERVATION_SOURCE
        and _is_strict_runtime_signal_event(api_name, event_type)
        and str(record.get("wrapper_runtime_contract") or "").strip().lower()
        == "async_runtime"
        and _float_payload(record.get("observed_runtime_us")) is not None
    )


def _rank_from_env() -> int:
    for key in ("RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value is not None:
            return int(value)
    return 0


def _world_size_from_env() -> int | None:
    for key in ("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS"):
        value = os.environ.get(key)
        if value is not None:
            return int(value)
    return None


def _host_machine_id_from_env() -> str:
    for key in (
        "FLEXSIM_HOST_MACHINE_ID",
        "SLURMD_NODENAME",
        "HOSTNAME",
        "HOST",
    ):
        value = os.environ.get(key)
        if value is not None and value.strip():
            return value.strip()
    return socket.gethostname().strip()


def _host_dispatch_queue_id_from_env(*, rank: int, host_machine_id: str) -> str:
    default_queue_id = f"{host_machine_id}:rank:{int(rank)}"
    for key in (
        "FLEXSIM_HOST_DISPATCH_QUEUE_ID",
        "FLEXSIM_HOST_DISPATCH_ID",
    ):
        value = os.environ.get(key)
        if value is not None and value.strip():
            return value.strip()
    return default_queue_id


def _parse_rank_list(value: str | None) -> tuple[int, ...]:
    if value is None or not value.strip():
        return ()
    return tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))


def _parse_profiled_rank_groups(value: str | None) -> dict[int, tuple[int, ...]]:
    if value is None or not value.strip():
        return {}
    groups: dict[int, tuple[int, ...]] = {}
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        head, members = item.split(":", 1)
        representative = int(head.strip())
        ranks = tuple(int(part.strip()) for part in members.split(",") if part.strip())
        groups[representative] = tuple(sorted(ranks))
    return groups


def _parse_route_metadata(raw_items: list[str] | None) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for raw_item in raw_items or []:
        if "=" not in raw_item:
            raise ValueError(f"route metadata must be key=value, got: {raw_item!r}")
        key, value = raw_item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"route metadata key must be non-empty: {raw_item!r}")
        normalized = value.strip().lower()
        if normalized == "true":
            metadata[key] = True
        elif normalized == "false":
            metadata[key] = False
        else:
            metadata[key] = value
    return metadata


def _profiled_ranks_from_env() -> tuple[int, ...]:
    return _parse_rank_list(os.environ.get("FLEXSIM_PROFILED_RANKS"))


def _profiled_rank_groups_from_env() -> dict[int, tuple[int, ...]]:
    return _parse_profiled_rank_groups(os.environ.get("FLEXSIM_PROFILED_RANK_GROUPS"))


def _normalize_boundary_visibility_payload(record: dict[str, object]) -> None:
    for field in _BOUNDARY_VISIBILITY_JSON_FIELDS:
        value = record.get(field)
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            record[field] = parsed
    for field in _BOUNDARY_VISIBILITY_FLOAT_FIELDS:
        value = record.get(field)
        if isinstance(value, bool) or value in (None, ""):
            continue
        try:
            record[field] = float(value)
        except (TypeError, ValueError):
            continue


def _raw_trace_event_id(rank: int, ordinal: int) -> str:
    return f"rank:{int(rank)}:raw_ordinal:{int(ordinal)}"


def _generic_actual_counterpart_row_id(
    *,
    rank: int,
    raw_event_id: str,
    raw_ordinal: int,
    api_name: str,
    host_dispatch_queue_id: str,
    stream_id: object | None,
) -> str:
    payload = json.dumps(
        [
            int(rank),
            raw_event_id,
            int(raw_ordinal),
            api_name,
            host_dispatch_queue_id,
            None if stream_id in (None, "") else str(stream_id),
        ],
        separators=(",", ":"),
        sort_keys=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"generic_replay_placement_envelope_actual:{digest}"


def _host_control_boundary_occurrence_id(
    *,
    rank: int,
    previous_ordinal: int | None,
    current_ordinal: int,
) -> str:
    previous = (
        "leading"
        if previous_ordinal is None
        else f"raw_ordinal:{int(previous_ordinal)}"
    )
    return (
        f"rank:{int(rank)}:host_control_boundary:{previous}"
        f"->raw_ordinal:{int(current_ordinal)}"
    )


def _event_counterpart_pair_id(rank: int, event_handle: str, event_version: int) -> str:
    return f"rank:{int(rank)}:cuda_event:{event_handle}:version:{int(event_version)}"


def _p2p_actual_row_id(rank: int, raw_ordinal: int) -> str:
    return f"rank:{int(rank)}:p2p_actual_row:raw_ordinal:{int(raw_ordinal)}"


def _p2p_occurrence_id(rank: int, api_name: str, occurrence: int | None) -> str:
    occurrence_label = (
        f"pair_seq:{int(occurrence)}"
        if occurrence is not None
        else "pair_seq:unavailable"
    )
    return f"rank:{int(rank)}:p2p:{api_name}:{occurrence_label}"


def _p2p_collective_name(api_name: str, record: dict[str, object]) -> str:
    value = _normalized_payload_text(record.get("collective"))
    if value is not None:
        return value
    if api_name == "ncclSend":
        return "send"
    return "recv"


def _p2p_count_or_numel(record: dict[str, object]) -> object | None:
    for key in ("numel", "count", "sendcount", "recvcount"):
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _p2p_datatype_or_dtype_code(record: dict[str, object]) -> object | None:
    for key in ("dtype_code", "datatype"):
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _allreduce_semantic_call_index(
    record: dict[str, object],
) -> tuple[int | None, str, str | None]:
    raw_call_idx = record.get("call_idx")
    parsed_call_idx = _int_payload(raw_call_idx)
    if parsed_call_idx is not None:
        return (
            int(parsed_call_idx),
            "raw_semantic_call_idx",
            None,
        )
    group_id = _normalized_payload_text(record.get("collective_group_id"))
    if group_id is not None:
        for part in group_id.split("|"):
            if not part.startswith("call:"):
                continue
            parsed_group_call = _int_payload(part.split(":", 1)[1])
            if parsed_group_call is not None:
                return (
                    int(parsed_group_call),
                    "recovered_collective_group_sequence_call_ordinal",
                    None,
                )
            break
    if raw_call_idx not in (None, ""):
        return (
            None,
            "unavailable_invalid_raw_call_idx_and_missing_group_sequence",
            "common_group_call_order_unavailable_invalid_raw_call_idx_and_missing_group_sequence",
        )
    return (
        None,
        "unavailable_missing_raw_call_idx_and_group_sequence",
        "common_group_call_order_unavailable_missing_raw_call_idx_and_group_sequence",
    )


def _p2p_comm_members(record: dict[str, object]) -> list[int] | None:
    for key in (
        "collective_communicator_members",
        "communicator_members",
        "comm_members",
        "actual_comm_members",
    ):
        value = record.get(key)
        if value in (None, ""):
            continue
        parsed: object = value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = [part.strip() for part in stripped.split(",")]
        if not isinstance(parsed, (list, tuple)):
            continue
        members: list[int] = []
        for member in parsed:
            parsed_member = _int_payload(member)
            if parsed_member is None:
                members = []
                break
            members.append(int(parsed_member))
        if members:
            return members
    return None


def _p2p_pair_members(
    rank: int,
    peer: object | None,
    communicator_members: list[int] | None,
) -> list[int] | None:
    if peer in (None, "") or communicator_members is None:
        return None
    peer_local_rank = _int_payload(peer)
    if (
        peer_local_rank is None
        or peer_local_rank < 0
        or peer_local_rank >= len(communicator_members)
        or int(rank) not in communicator_members
    ):
        return None
    peer_rank = int(communicator_members[peer_local_rank])
    members = [int(rank), peer_rank]
    return sorted(
        members,
        key=lambda value: value,
    )


def _p2p_shape_signature(
    *,
    group_api: str,
    collective: str,
    count: object | None,
    datatype: object | None,
) -> str:
    return (
        f"group_api={group_api};collective={collective};"
        f"count={'' if count is None else count};"
        f"datatype={'' if datatype is None else datatype}"
    )


def _common_membership_signature(members: list[int] | None) -> str:
    if not members:
        return "members:unavailable"
    return "members:" + "-".join(str(int(member)) for member in sorted(members))


def _common_payload_signature(
    *,
    group_api: str,
    collective_kind: str,
    api_name: str,
    members: list[int] | None,
    count: object | None,
    datatype: object | None,
    op: object | None,
) -> tuple[str, dict[str, object]]:
    membership_signature = _common_membership_signature(members)
    pair_members = list(members) if collective_kind == "p2p" and members else None
    inputs: dict[str, object] = {
        "api": api_name,
        "group_api": group_api,
        "collective_kind": collective_kind,
        "count": count,
        "datatype": datatype,
        "op": op,
        "membership_signature": membership_signature,
        "pair_members": pair_members,
    }
    signature = (
        f"group_api={group_api};kind={collective_kind};api={api_name};"
        f"members={membership_signature};count={'' if count is None else count};"
        f"datatype={'' if datatype is None else datatype};"
        f"op={'null' if op is None else op}"
    )
    return signature, inputs


def _boundary_family(previous_api: object | None, current_api: object | None) -> str:
    apis = [str(api) for api in (previous_api, current_api) if api not in (None, "")]
    return " -> ".join(apis)


def _is_hostdelay_semantic_predecessor_control_query_api(api_name: object | None) -> bool:
    return str(api_name or "") in _HOSTDELAY_SEMANTIC_PREDECESSOR_CONTROL_QUERY_APIS


def _diagnostic_float_gap_us(
    *,
    previous_end_ts_us: object | None,
    current_ts_us: object | None,
) -> float | None:
    if previous_end_ts_us in (None, "") or current_ts_us in (None, ""):
        return None
    try:
        return max(float(current_ts_us) - float(previous_end_ts_us), 0.0)
    except (TypeError, ValueError):
        return None


def _host_control_raw_row_snapshot(
    record: dict[str, object],
    *,
    raw_event_id: str,
    raw_ordinal: int,
) -> dict[str, object]:
    material_fields = {
        key: record.get(key)
        for key in (
            "material_signature",
            "m",
            "n",
            "k",
            "lda",
            "ldb",
            "ldc",
            "batch_count",
            "batchCount",
            "stride_a",
            "strideA",
            "stride_b",
            "strideB",
            "stride_c",
            "strideC",
            "compute_type",
            "computeType",
            "cuda_data_type",
            "dtype",
            "transa",
            "transb",
            "algorithm",
            "algo",
            "kernel",
            "grid_x",
            "grid_y",
            "grid_z",
            "block_x",
            "block_y",
            "block_z",
            "shared_mem",
        )
        if record.get(key) not in (None, "")
    }
    cupti_activity_fields = {
        key: record.get(key)
        for key in (
            "cupti_activity_first_kernel_start",
            "cupti_activity_first_kernel_end",
            "cupti_activity_last_kernel_start",
            "cupti_activity_last_kernel_end",
            "cupti_activity_first_kernel_stream_id",
            "cupti_activity_last_kernel_stream_id",
            "cupti_activity_kernel_stream_id_unique_count",
            "cupti_activity_kernel_stream_id_status",
            "cupti_activity_kernel_stream_id_basis",
            "cupti_activity_device_activity_timing_status",
            "cupti_activity_common_clock_status",
            "cupti_activity_strict_wait_timing",
        )
        if record.get(key) not in (None, "")
    }
    return {
        "raw_event_id": raw_event_id,
        "raw_ordinal": int(raw_ordinal),
        "api": record.get("api"),
        "ts": record.get("ts"),
        "end_ts": record.get("end_ts"),
        "host_duration_us": record.get("host_duration_us"),
        "host_machine_id": record.get("host_machine_id"),
        "host_dispatch_queue_id": record.get("host_dispatch_queue_id"),
        "stream_id": record.get("stream_id"),
        **material_fields,
        **cupti_activity_fields,
    }


def _add_generic_replay_placement_envelope_actual_counterpart_diagnostics(
    record: dict[str, object],
    *,
    rank: int,
    raw_event_id: str,
    raw_ordinal: int,
    previous_record: dict[str, object] | None,
    host_machine_id: str,
    host_dispatch_queue_id: str,
) -> None:
    api_name = str(record.get("api") or "")
    event_type = str(record.get("type") or "")
    stream_id = record.get("stream_id")
    actual_stream_resource_id = (
        f"rank:{int(rank)}:stream:{stream_id}"
        if stream_id not in (None, "")
        else None
    )
    previous_raw_event_id = (
        str(previous_record.get("raw_event_id")) if previous_record is not None else None
    )
    previous_api = (
        str(previous_record.get("api") or "") if previous_record is not None else None
    )
    strict_timing_unavailable_reason = (
        "capture_real exports wrapper endpoint provenance only; strict actual "
        "replay interval timing, wait-map release timing, and runtime "
        "counterpart timing require a later reviewed offline join or producer"
    )
    nonoverlap_unavailable_reason = (
        "capture_real endpoint rows cannot observe replay interval overlap, "
        "wait-map overlap, rank/global envelope overlap, or Figure 6 count-once "
        "grouping"
    )
    row_id = _generic_actual_counterpart_row_id(
        rank=rank,
        raw_event_id=raw_event_id,
        raw_ordinal=raw_ordinal,
        api_name=api_name,
        host_dispatch_queue_id=host_dispatch_queue_id,
        stream_id=stream_id,
    )

    record.update(
        {
            "generic_actual_counterpart_schema_version": (
                _GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_SCHEMA_VERSION
            ),
            "generic_actual_counterpart_opt_in_flag": True,
            "generic_source_side": "actual_endpoint_provenance",
            "generic_actual_counterpart_row_id": row_id,
            "generic_actual_counterpart_candidate_kind": "actual_api_endpoint_row",
            "generic_actual_rank": int(rank),
            "generic_actual_api": api_name,
            "generic_actual_type": event_type,
            "generic_actual_raw_event_id": raw_event_id,
            "generic_actual_raw_ordinal": int(raw_ordinal),
            "generic_actual_trace_id": None,
            "generic_actual_trace_id_unavailable_reason": (
                "trace_directory_identifier_not_available_in_rank_row_writer"
            ),
            "generic_actual_host_machine_id": host_machine_id,
            "generic_actual_host_dispatch_queue_id": host_dispatch_queue_id,
            "generic_actual_prev_raw_event_id": previous_raw_event_id,
            "generic_actual_prev_api": previous_api,
            "generic_actual_next_raw_event_id": None,
            "generic_actual_next_api": None,
            "generic_actual_next_unavailable_reason": (
                "capture_real_streaming_writer_exports_previous_endpoint_only"
            ),
            "generic_actual_paper_valid_window_id": None,
            "generic_actual_in_paper_valid_window": None,
            "generic_actual_paper_valid_window_unavailable_reason": (
                "paper_valid_step_window_resolved_later_by_collate_or_offline_ledger"
            ),
            "generic_phase1_stable_component_row_id": None,
            "generic_phase1_component_row_type": None,
            "generic_phase1_component_kind": None,
            "generic_phase1_count_once_group_id": None,
            "generic_phase1_stable_replay_edge_id": None,
            "generic_counterpart_join_key": {
                "generic_actual_rank": int(rank),
                "generic_actual_api": api_name,
                "generic_actual_type": event_type,
                "generic_actual_raw_event_id": raw_event_id,
                "generic_actual_raw_ordinal": int(raw_ordinal),
                "generic_actual_host_dispatch_queue_id": host_dispatch_queue_id,
                "generic_actual_stream_resource_id": actual_stream_resource_id,
            },
            "generic_counterpart_join_basis": (
                "actual_endpoint_metadata_only_predicted_phase1_join_deferred"
            ),
            "generic_counterpart_join_attempted_during_capture": False,
            "generic_counterpart_join_status": (
                "actual_metadata_export_only_predicted_phase1_join_deferred"
            ),
            "generic_counterpart_join_confidence": "unavailable",
            "generic_counterpart_unavailable_reason": (
                "predicted generic phase1 rows and paper-valid window membership "
                "are not available during raw capture"
            ),
            "generic_actual_timing_status": (
                "endpoint_context_only_strict_counterpart_unavailable"
            ),
            "generic_actual_timing_basis": "wrapper_endpoint_provenance_only",
            "generic_actual_timing_unavailable_reason": strict_timing_unavailable_reason,
            "generic_actual_start_us": None,
            "generic_actual_end_us": None,
            "generic_actual_duration_us": None,
            "generic_actual_wait_start_us": None,
            "generic_actual_release_us": None,
            "generic_actual_waited_us": None,
            "generic_actual_release_reason": None,
            "generic_actual_released_by_event_id": None,
            "generic_actual_release_source_kind": None,
            "generic_actual_endpoint_ts_us": record.get("ts"),
            "generic_actual_endpoint_end_ts_us": record.get("end_ts"),
            "generic_actual_endpoint_host_duration_us": record.get("host_duration_us"),
            "generic_actual_observed_runtime_us": record.get("observed_runtime_us"),
            "generic_actual_endpoint_context_only": True,
            "generic_actual_endpoint_timestamps_used_as_strict_timing": False,
            "generic_actual_endpoint_end_ts_used_as_wait_release": False,
            "generic_actual_endpoint_end_ts_used_as_release": False,
            "generic_actual_runtime_direct_substitution": False,
            "generic_actual_stream_id": stream_id,
            "generic_actual_raw_stream_id": stream_id,
            "generic_actual_stream_resource_id": actual_stream_resource_id,
            "generic_actual_stream_namespace_basis": (
                "rank_scoped_actual_raw_stream_id_process_local_when_stream_id_present_else_unavailable"
            ),
            "generic_predicted_stream_resource_id": None,
            "generic_predicted_stream_id": None,
            "generic_stream_namespace_alignment_status": (
                "actual_only_unresolved_predicted_namespace_not_joined"
            ),
            "generic_stream_namespace_alignment_unavailable_reason": (
                "predicted replay stream resource id is available only in "
                "phase1/offline join context"
            ),
            "generic_actual_count_once_group_id": None,
            "generic_actual_count_once_interval_id": None,
            "generic_count_once_status": (
                "actual_endpoint_metadata_only_not_strict_non_overlap_proof"
            ),
            "generic_count_once_non_overlap_status": "unavailable",
            "generic_count_once_non_overlap_unavailable_reason": nonoverlap_unavailable_reason,
            "generic_double_counting_overlap_status": "unavailable",
            "generic_double_counting_overlap_unavailable_reason": nonoverlap_unavailable_reason,
            "generic_wait_map_safety_status": "unavailable",
            "generic_wait_map_safety_unavailable_reason": (
                "strict actual wait-map release/source timing and replay "
                "wait-map non-overlap are unavailable during raw capture"
            ),
            "generic_diagnostic_only": True,
            "generic_repair_ready": False,
            "generic_safe_to_use_as_repair_evidence": False,
            "generic_safe_to_use_as_subtraction_delta": False,
            "generic_safe_to_use_as_repair_evidence_reason": (
                "metadata export only; no paper/Maya semantic mismatch or "
                "strict non-overlap proof is available"
            ),
            "generic_safe_to_use_as_subtraction_delta_reason": (
                "actual endpoint timing is provenance only and not a prediction "
                "subtraction delta"
            ),
            "generic_paper_facing_closure_claimed": False,
            "generic_native_capture_or_compare_run_for_this_metadata": False,
            "diagnostic_only": True,
            "repair_ready": False,
            "safe_to_use_as_repair_evidence": False,
            "safe_to_use_as_subtraction_delta": False,
            "safe_to_use_as_repair_evidence_reason": (
                "metadata export only; no paper/Maya semantic mismatch or "
                "strict non-overlap proof is available"
            ),
            "safe_to_use_as_subtraction_delta_reason": (
                "actual endpoint timing is provenance only and not a prediction "
                "subtraction delta"
            ),
            "paper_facing_closure_claimed": False,
            "native_capture_or_compare_run_for_this_metadata": False,
        }
    )


def _host_control_producer_visibility_segments(api_name: str) -> list[dict[str, object]]:
    source = f"capture_real::_add_host_control_boundary_counterpart_diagnostics::{api_name}"
    return [
        {
            "name": "real_api_call_envelope",
            "visibility_kind": "mixed_or_unresolved",
            "duration_us": None,
            "clock": "unmeasured",
            "source_file_function": source,
            "classification_basis": (
                "structural_label_only_internal_clocks_disabled_to_preserve_start_time_end_time"
            ),
            "included_in_paper_visible_host_duration": False,
            "included_in_instrumentation_only_duration": False,
        },
        {
            "name": "producer_payload_and_recording_overhead",
            "visibility_kind": "producer_instrumentation_unavailable",
            "duration_us": None,
            "clock": "unmeasured",
            "source_file_function": source,
            "classification_basis": (
                "not_mechanically_bracketed_outside_measured_wrapper_interval"
            ),
            "included_in_paper_visible_host_duration": False,
            "included_in_instrumentation_only_duration": False,
        },
    ]


def _component_strict_actual_stream_resource_id(
    *,
    rank: int,
    stream_id: object,
) -> str | None:
    if stream_id in (None, ""):
        return None
    return f"rank:{int(rank)}:stream:{stream_id}"


def _component_strict_cuda_launch_material_signature(
    record: dict[str, object],
) -> str | None:
    if record.get("api") != "cudaLaunchKernel":
        return None
    kernel = record.get("kernel")
    shared_mem = record.get("shared_mem")
    stream = record.get("stream_id")
    grid_parts = [record.get("grid_x"), record.get("grid_y"), record.get("grid_z")]
    block_parts = [
        record.get("block_x"),
        record.get("block_y"),
        record.get("block_z"),
    ]
    required_parts = [kernel, shared_mem, stream, *grid_parts, *block_parts]
    if any(part in (None, "") for part in required_parts):
        return None
    grid = "x".join(str(part) for part in grid_parts)
    block = "x".join(str(part) for part in block_parts)
    return (
        f"kernel={kernel};grid={grid};block={block};"
        f"shared_mem={shared_mem};stream={stream}"
    )


def _component_strict_actual_material_signature(
    record: dict[str, object],
) -> object | None:
    api_name = str(record.get("api") or "")
    if is_gemm_material_api(api_name):
        return (
            canonical_gemm_material_signature(record)
            or record.get("material_signature")
        )
    return (
        record.get("material_signature")
        or _component_strict_cuda_launch_material_signature(record)
    )


def _add_component_strict_counterpart_actual_metadata_diagnostics(
    record: dict[str, object],
    *,
    rank: int,
    world_size: int | None,
    raw_event_id: str,
    raw_ordinal: int,
    previous_record: dict[str, object] | None,
    host_machine_id: str,
    host_dispatch_queue_id: str,
) -> None:
    api_name = str(record.get("api") or "")
    event_type = str(record.get("type") or "")
    stream_id = record.get("stream_id")
    actual_stream_resource_id = _component_strict_actual_stream_resource_id(
        rank=rank,
        stream_id=stream_id,
    )
    previous_api = (
        str(previous_record.get("api") or "") if previous_record is not None else None
    )
    material_signature = _component_strict_actual_material_signature(record)
    strict_timing_unavailable_reason = (
        "capture_real exports actual endpoint/common-basis provenance only; "
        "endpoint ts/end_ts, host_duration_us, and observed runtime are not strict "
        "same-component timing and are not prediction substitutions"
    )
    nonoverlap_unavailable_reason = (
        "raw capture rows cannot prove replay count-once, wait-map, stream FIFO, "
        "host-queue, rank-envelope, or global-envelope non-overlap"
    )
    row_id = (
        f"rank:{int(rank)}:component_strict_actual:raw_ordinal:{int(raw_ordinal)}:"
        f"api:{api_name or 'unknown'}"
    )
    record.update(
        {
            "actual_counterpart_schema_version": (
                _COMPONENT_STRICT_COUNTERPART_ACTUAL_SCHEMA_VERSION
            ),
            "actual_counterpart_opt_in_flag": True,
            "component_strict_counterpart_schema_version": (
                _COMPONENT_STRICT_COUNTERPART_ACTUAL_SCHEMA_VERSION
            ),
            "component_strict_counterpart_opt_in_flag": True,
            "source_side": "actual_counterpart_metadata",
            "actual_counterpart_row_id": row_id,
            "actual_counterpart_candidate_kind": "actual_trace_row_or_actual_interval_candidate",
            "actual_trace_id": None,
            "actual_trace_id_unavailable_reason": (
                "trace_directory_identifier_not_available_in_rank_row_writer"
            ),
            "actual_rank": int(rank),
            "actual_world_size": None if world_size is None else int(world_size),
            "actual_raw_event_id": raw_event_id,
            "actual_raw_ordinal": int(raw_ordinal),
            "actual_api": api_name,
            "actual_type": event_type,
            "actual_paper_valid_window_id": None,
            "actual_in_paper_valid_window": None,
            "actual_paper_valid_window_unavailable_reason": (
                "paper-valid window is resolved later by collate or offline ledger"
            ),
            "actual_host_machine_id": host_machine_id,
            "actual_host_dispatch_queue_id": host_dispatch_queue_id,
            "actual_stream_id": stream_id,
            "actual_stream_resource_id": actual_stream_resource_id,
            "actual_collective_group_id": record.get("collective_group_id"),
            "actual_cuda_event_id": record.get("event_id"),
            "actual_material_signature": material_signature,
            "actual_material_signature_status": (
                "available" if material_signature not in (None, "") else "unavailable"
            ),
            "common_basis_version": "component_strict_counterpart_common_basis_v1",
            "common_basis_kind": "actual_trace_row_metadata_only",
            "common_basis_rank": int(rank),
            "common_basis_paper_window": None,
            "common_basis_raw_ordinal": int(raw_ordinal),
            "common_basis_api_sequence_ordinal": int(raw_ordinal),
            "common_basis_stream_sequence_ordinal": None,
            "common_basis_host_queue_sequence_ordinal": int(raw_ordinal),
            "common_basis_collective_sequence_ordinal": None,
            "common_basis_cuda_event_sequence_ordinal": None,
            "common_basis_material_signature": material_signature,
            "common_basis_boundary_family": _boundary_family(previous_api, api_name),
            "common_basis_confidence": "metadata_only_unreviewed",
            "common_basis_unavailable_reason": (
                "predicted stable component row ids are not available during capture"
            ),
            "predicted_component_row_id": None,
            "predicted_interval_row_id": None,
            "predicted_edge_row_id": None,
            "actual_counterpart_join_status": (
                "actual_metadata_export_only_predicted_join_deferred"
            ),
            "actual_counterpart_join_basis": "not_joined_during_capture",
            "actual_counterpart_unavailable_reason": (
                "offline strict counterpart join has not been run"
            ),
            "strict_actual_timing_status": "unavailable",
            "strict_actual_timing_available": False,
            "strict_actual_timing_unavailable_reason": strict_timing_unavailable_reason,
            "actual_start_us": None,
            "actual_end_us": None,
            "actual_duration_us": None,
            "actual_timing_basis": "unavailable_endpoint_context_only",
            "actual_timing_clock_domain": None,
            "actual_timing_common_clock_review_status": "unavailable",
            "actual_timing_source_event_ids": [],
            "actual_endpoint_ts_us": record.get("ts"),
            "actual_endpoint_end_ts_us": record.get("end_ts"),
            "actual_endpoint_host_duration_us": record.get("host_duration_us"),
            "actual_observed_runtime_us": record.get("observed_runtime_us"),
            "actual_endpoint_timestamps_used_as_strict_timing": False,
            "actual_host_duration_used_as_strict_timing": False,
            "actual_runtime_direct_substitution": False,
            "actual_observed_runtime_used_as_prediction": False,
            "predicted_stream_id": None,
            "predicted_stream_resource_id": None,
            "predicted_stream_namespace_basis": None,
            "actual_stream_namespace_basis": (
                "rank_scoped_actual_raw_stream_id_process_local"
                if actual_stream_resource_id is not None
                else "unavailable_missing_actual_stream_id"
            ),
            "stream_namespace_alignment_status": (
                "actual_only_unresolved_predicted_namespace_not_joined"
            ),
            "stream_namespace_alignment_basis": (
                "predicted replay stream metadata unavailable during capture"
            ),
            "stream_namespace_alignment_evidence": None,
            "stream_namespace_mismatch_reason": None,
            "exact_stream_identity_proven": False,
            "default_stream_equivalence_reviewed": False,
            "cross_trace_stream_namespace_review_status": "unavailable",
            "predicted_count_once_group_id": None,
            "actual_count_once_group_id": None,
            "count_once_group_basis": "actual_endpoint_row_only",
            "count_once_status": "unavailable",
            "nonoverlap_status": "unavailable",
            "nonoverlap_unavailable_reason": nonoverlap_unavailable_reason,
            "wait_map_safety_status": "unavailable",
            "wait_map_safety_unavailable_reason": (
                "actual wait-map release/source and replay wait-map non-overlap are unavailable"
            ),
            "producer_visibility_status": "unavailable",
            "producer_visibility_basis": "capture_real_no_non_perturbing_visibility_split",
            "producer_visibility_unavailable_reason": (
                "internal producer visibility split is not measured to preserve wrapper timing"
            ),
            "paper_maya_tags": [
                "actual_endpoint_provenance",
                "metadata_only",
                "no_repair",
                "no_runtime_substitution",
                "no_endpoint_timestamp_substitution",
            ],
            "paper_maya_semantic_component_status": (
                "actual_candidate_metadata_only_no_predicted_component_join"
            ),
            "diagnostic_only": True,
            "repair_ready": False,
            "safe_to_use_as_repair_evidence": False,
            "safe_to_use_as_subtraction_delta": False,
            "safe_to_use_as_repair_evidence_reason": (
                "strict actual timing, count-once/non-overlap, wait-map safety, "
                "stream alignment, and producer visibility are unavailable"
            ),
            "safe_to_use_as_subtraction_delta_reason": (
                "actual endpoint metadata is provenance only and not a subtraction delta"
            ),
            "native_capture_or_compare_run_for_this_metadata": False,
            "paper_facing_closure_claimed": False,
        }
    )


def _gemm_adjacent_boundary_in_scope(
    previous_api: str | None,
    current_api: str | None,
) -> bool:
    apis = {api for api in (previous_api, current_api) if api}
    return bool(apis & _GEMM_ADJACENT_HOSTDELAY_GEMM_APIS) and bool(
        apis & _GEMM_ADJACENT_HOSTDELAY_ADJACENT_APIS
    )


def _gemm_adjacent_shape_signature(record: dict[str, object]) -> str | None:
    return canonical_gemm_material_signature(record)


def _gemm_adjacent_material_metadata_status(
    api_name: str,
    record: dict[str, object],
) -> str:
    if api_name not in _GEMM_ADJACENT_HOSTDELAY_GEMM_APIS:
        return "not_applicable_non_gemm_endpoint"
    required = ("m", "n", "k")
    if all(record.get(key) not in (None, "") for key in required):
        return "shape_metadata_available"
    if any(record.get(key) not in (None, "") for key in required):
        return "partial_shape_metadata"
    return "unavailable"


def _gemm_adjacent_producer_visibility_segments(api_name: str) -> list[dict[str, object]]:
    source = (
        "capture_real::_add_gemm_adjacent_hostdelay_boundary_actual_counterpart_"
        f"diagnostics::{api_name}"
    )
    return [
        {
            "name": "actual_wrapper_endpoint_envelope",
            "visibility_kind": "mixed_or_unresolved",
            "duration_us": None,
            "clock": "unmeasured",
            "source_file_function": source,
            "classification_basis": (
                "structural_label_only_internal_clocks_disabled_to_preserve_wrapper_timing"
            ),
            "included_in_paper_visible_host_duration": False,
            "included_in_instrumentation_only_duration": False,
        },
        {
            "name": "producer_payload_recording_or_library_call_region",
            "visibility_kind": "producer_visibility_split_unavailable",
            "duration_us": None,
            "clock": "unmeasured",
            "source_file_function": source,
            "classification_basis": (
                "not_mechanically_bracketed_outside_existing_start_time_end_time_interval"
            ),
            "included_in_paper_visible_host_duration": False,
            "included_in_instrumentation_only_duration": False,
        },
    ]


def _add_gemm_adjacent_hostdelay_boundary_actual_counterpart_diagnostics(
    record: dict[str, object],
    *,
    rank: int,
    raw_event_id: str,
    raw_ordinal: int,
    previous_record: dict[str, object] | None,
    host_machine_id: str,
    host_dispatch_queue_id: str,
) -> None:
    api_name = str(record.get("api") or "")
    previous_api = (
        str(previous_record.get("api") or "") if previous_record is not None else None
    )
    if api_name not in _GEMM_ADJACENT_HOSTDELAY_ENDPOINT_APIS:
        return
    if not _gemm_adjacent_boundary_in_scope(previous_api, api_name):
        return
    occurrence_id = _host_control_boundary_occurrence_id(
        rank=rank,
        previous_ordinal=(
            int(previous_record["raw_ordinal"]) if previous_record is not None else None
        ),
        current_ordinal=raw_ordinal,
    )
    previous_raw_event_id = (
        str(previous_record.get("raw_event_id")) if previous_record is not None else None
    )
    previous_ts_us = previous_record.get("ts") if previous_record is not None else None
    previous_end_ts_us = (
        previous_record.get("end_ts") if previous_record is not None else None
    )
    family_prev_to_current = _boundary_family(previous_api, api_name)
    stream_id = record.get("stream_id")
    target_gemm_api = (
        api_name
        if api_name in _GEMM_ADJACENT_HOSTDELAY_GEMM_APIS
        else (
            previous_api
            if previous_api in _GEMM_ADJACENT_HOSTDELAY_GEMM_APIS
            else None
        )
    )
    material_record = (
        record
        if api_name in _GEMM_ADJACENT_HOSTDELAY_GEMM_APIS
        else previous_record
        if previous_api in _GEMM_ADJACENT_HOSTDELAY_GEMM_APIS
        else record
    )
    adjacent_api = (
        api_name
        if api_name in _GEMM_ADJACENT_HOSTDELAY_ADJACENT_APIS
        else (
            previous_api
            if previous_api in _GEMM_ADJACENT_HOSTDELAY_ADJACENT_APIS
            else None
        )
    )
    visibility_unavailable_reason = (
        "mechanical_visibility_split_not_measured_to_preserve_wrapper_host_timing"
    )
    nonoverlap_unavailable_reason = (
        "requires_offline_replay_wait_map_host_queue_stream_fifo_and_rank_envelope_ledger"
    )
    record.update(
        {
            "gemm_adjacent_actual_counterpart_schema_version": (
                _GEMM_ADJACENT_HOSTDELAY_BOUNDARY_SCHEMA_VERSION
            ),
            "gemm_adjacent_actual_counterpart_opt_in_flag": True,
            "gemm_adjacent_source_side": "actual_endpoint_provenance",
            "gemm_adjacent_actual_counterpart_row_id": (
                f"rank:{int(rank)}:gemm_adjacent_actual:raw_ordinal:{int(raw_ordinal)}"
            ),
            "gemm_adjacent_actual_counterpart_candidate_kind": (
                "actual_api_endpoint_row"
            ),
            "gemm_adjacent_actual_rank": int(rank),
            "gemm_adjacent_actual_api": api_name,
            "gemm_adjacent_actual_type": record.get("type"),
            "gemm_adjacent_actual_raw_event_id": raw_event_id,
            "gemm_adjacent_actual_raw_ordinal": int(raw_ordinal),
            "gemm_adjacent_actual_trace_id": None,
            "gemm_adjacent_actual_host_machine_id": host_machine_id,
            "gemm_adjacent_actual_host_dispatch_queue_id": host_dispatch_queue_id,
            "gemm_adjacent_actual_paper_valid_window_id": None,
            "gemm_adjacent_actual_in_paper_valid_window": None,
            "gemm_adjacent_actual_prev_raw_event_id": previous_raw_event_id,
            "gemm_adjacent_actual_prev_api": previous_api,
            "gemm_adjacent_actual_prev_ts_us": previous_ts_us,
            "gemm_adjacent_actual_prev_end_ts_us": previous_end_ts_us,
            "gemm_adjacent_actual_next_raw_event_id": None,
            "gemm_adjacent_actual_next_api": None,
            "gemm_adjacent_actual_next_ts_us": None,
            "gemm_adjacent_actual_next_end_ts_us": None,
            "gemm_adjacent_actual_raw_boundary_family_prev_to_current": (
                family_prev_to_current
            ),
            "gemm_adjacent_actual_raw_boundary_family_current_to_next": None,
            "gemm_adjacent_actual_boundary_family_in_design_scope": (
                _gemm_adjacent_boundary_in_scope(previous_api, api_name)
            ),
            "gemm_adjacent_target_gemm_api": target_gemm_api,
            "gemm_adjacent_adjacent_api": adjacent_api,
            "gemm_adjacent_actual_endpoint_ts_us": record.get("ts"),
            "gemm_adjacent_actual_endpoint_end_ts_us": record.get("end_ts"),
            "gemm_adjacent_actual_endpoint_host_duration_us": record.get(
                "host_duration_us"
            ),
            "gemm_adjacent_actual_observed_runtime_us": record.get(
                "observed_runtime_us"
            ),
            "gemm_adjacent_actual_wrapper_runtime_contract": record.get(
                "wrapper_runtime_contract"
            ),
            "gemm_adjacent_actual_timing_status": (
                "endpoint_context_only_strict_counterpart_unavailable"
            ),
            "gemm_adjacent_actual_timing_basis": "wrapper_endpoint_provenance_only",
            "gemm_adjacent_actual_timing_unavailable_reason": (
                "actual endpoint timestamps and host_duration_us are provenance only; "
                "strict hostDelay counterpart timing requires offline join and "
                "non-overlap review"
            ),
            "gemm_adjacent_actual_endpoint_context_only": True,
            "gemm_adjacent_actual_endpoint_timestamps_used_as_strict_timing": False,
            "gemm_adjacent_actual_runtime_direct_substitution": False,
            "gemm_adjacent_actual_stream_id": stream_id,
            "gemm_adjacent_actual_raw_stream_id": stream_id,
            "gemm_adjacent_actual_stream_resource_id": (
                None if stream_id in (None, "") else f"rank:{int(rank)}:stream:{stream_id}"
            ),
            "gemm_adjacent_actual_stream_namespace_basis": (
                "actual_raw_stream_id_process_local"
                if stream_id not in (None, "")
                else "actual_stream_id_unavailable"
            ),
            "gemm_adjacent_actual_material_signature": (
                _component_strict_actual_material_signature(material_record)
                if material_record is not None
                else None
            ),
            "gemm_adjacent_actual_algorithm": (
                material_record.get("algorithm")
                if material_record is not None
                and material_record.get("algorithm") not in (None, "")
                else material_record.get("algo")
                if material_record is not None
                else None
            ),
            "gemm_adjacent_actual_gemm_shape_signature": (
                _gemm_adjacent_shape_signature(material_record or record)
            ),
            "gemm_adjacent_actual_gemm_metadata_status": (
                _gemm_adjacent_material_metadata_status(
                    str(target_gemm_api or api_name),
                    material_record or record,
                )
            ),
            "gemm_adjacent_predicted_stable_boundary_row_id": None,
            "gemm_adjacent_predicted_materialized_hostdelay_event_id": None,
            "gemm_adjacent_predicted_count_once_group_id": None,
            "gemm_adjacent_counterpart_join_key": occurrence_id,
            "gemm_adjacent_counterpart_join_basis": (
                "actual_endpoint_rank_raw_ordinal_boundary_family_stream_material_metadata"
            ),
            "gemm_adjacent_counterpart_join_attempted_during_capture": False,
            "gemm_adjacent_counterpart_join_status": (
                "actual_metadata_export_only_predicted_hostdelay_boundary_join_deferred"
            ),
            "gemm_adjacent_counterpart_join_confidence": "unavailable",
            "gemm_adjacent_counterpart_unavailable_reason": (
                "predicted materialized hostDelay boundary rows are not joined during capture"
            ),
            "gemm_adjacent_producer_visibility_schema_version": (
                _GEMM_ADJACENT_PRODUCER_VISIBILITY_SCHEMA_VERSION
            ),
            "gemm_adjacent_producer_visibility_status": "structural_unavailable",
            "gemm_adjacent_producer_visibility_basis": (
                "capture_real_gemm_adjacent_endpoint_structural_metadata_no_internal_wrapper_clocks"
            ),
            "gemm_adjacent_producer_visibility_unavailable_reason": (
                visibility_unavailable_reason
            ),
            "gemm_adjacent_boundary_origin_kind": "mixed_or_unresolved",
            "gemm_adjacent_boundary_visibility_kind": "mixed_or_unresolved",
            "gemm_adjacent_classification_basis": (
                "structural_endpoint_and_adjacent_boundary_metadata_only"
            ),
            "gemm_adjacent_paper_visible_host_duration_us": None,
            "gemm_adjacent_instrumentation_only_duration_us": None,
            "gemm_adjacent_unresolved_mixed_duration_us": None,
            "gemm_adjacent_actual_control_dispatch_us": None,
            "gemm_adjacent_actual_api_body_us": None,
            "gemm_adjacent_actual_instrumentation_only_us": None,
            "gemm_adjacent_wrapper_segment_sum_us": None,
            "gemm_adjacent_wrapper_segment_unattributed_us": None,
            "gemm_adjacent_producer_visibility_segments": (
                _gemm_adjacent_producer_visibility_segments(api_name)
            ),
            "gemm_adjacent_boundary_visibility_segments": [],
            "gemm_adjacent_split_sum_check_status": "unavailable",
            "gemm_adjacent_split_sum_check_delta_us": None,
            "gemm_adjacent_predicted_count_once_interval_id": None,
            "gemm_adjacent_actual_count_once_group_id": None,
            "gemm_adjacent_actual_count_once_interval_id": None,
            "gemm_adjacent_count_once_status": "unavailable",
            "gemm_adjacent_count_once_non_overlap_status": "unavailable",
            "gemm_adjacent_count_once_non_overlap_unavailable_reason": (
                nonoverlap_unavailable_reason
            ),
            "gemm_adjacent_double_counting_overlap_status": "unavailable",
            "gemm_adjacent_double_counting_overlap_unavailable_reason": (
                nonoverlap_unavailable_reason
            ),
            "gemm_adjacent_wait_map_safety_status": "unavailable",
            "gemm_adjacent_wait_map_safety_unavailable_reason": (
                nonoverlap_unavailable_reason
            ),
            "gemm_adjacent_stream_fifo_nonoverlap_status": "unavailable",
            "gemm_adjacent_host_queue_nonoverlap_status": "unavailable",
            "gemm_adjacent_rank_envelope_nonoverlap_status": "unavailable",
            "gemm_adjacent_strict_nonoverlap_proof_basis": None,
            "gemm_adjacent_repair_ready": False,
            "gemm_adjacent_safe_to_use_as_repair_evidence": False,
            "gemm_adjacent_safe_to_use_as_subtraction_delta": False,
            "repair_ready": False,
            "safe_to_use_as_repair_evidence": False,
            "safe_to_use_as_subtraction_delta": False,
            "safe_to_use_as_repair_evidence_reason": (
                "GEMM-adjacent metadata is structural only; strict actual counterpart, "
                "producer visibility split, and count-once non-overlap are unavailable"
            ),
            "safe_to_use_as_subtraction_delta_reason": (
                "actual endpoint provenance is not a prediction subtraction delta"
            ),
        }
    )


def _cuda_gemm_hostdispatch_stream_resource_id(
    *,
    rank: int,
    stream_id: object,
) -> str | None:
    if stream_id in (None, ""):
        return None
    return f"rank:{int(rank)}:stream:{stream_id}"


def _cuda_gemm_hostdispatch_algorithm(record: dict[str, object]) -> object | None:
    if record.get("algorithm") not in (None, ""):
        return record.get("algorithm")
    if record.get("algo") not in (None, ""):
        return record.get("algo")
    return None


def _cuda_gemm_hostdispatch_strict_occurrence_key(
    *,
    rank: int,
    paper_valid_window_id: object | None,
    host_dispatch_queue_id: str,
    api_name: str,
    component_role: str,
    api_sequence_ordinal: int | None,
    host_queue_sequence_ordinal: int | None,
    stream_sequence_ordinal: int | None,
    material_signature: object | None,
    algorithm: object | None,
    boundary_family: str | None,
) -> str:
    material_text = (
        "unavailable" if material_signature in (None, "") else str(material_signature)
    )
    algorithm_text = "unavailable" if algorithm in (None, "") else str(algorithm)
    parts = [
        f"rank:{int(rank)}",
        f"queue:{host_dispatch_queue_id or 'unavailable'}",
        f"api:{api_name or 'unknown'}",
        f"api_seq:{api_sequence_ordinal if api_sequence_ordinal is not None else 'unavailable'}",
        (
            "host_queue_seq:"
            f"{host_queue_sequence_ordinal if host_queue_sequence_ordinal is not None else 'unavailable'}"
        ),
        f"stream_seq:{stream_sequence_ordinal if stream_sequence_ordinal is not None else 'unavailable'}",
        f"material:{material_text}",
        f"algorithm:{algorithm_text}",
        f"boundary:{boundary_family or 'unavailable'}",
    ]
    return "|".join(parts)


def _cuda_gemm_hostdispatch_material_without_embedded_algo(
    material_signature: object | None,
) -> object | None:
    if material_signature in (None, ""):
        return material_signature
    parts = [
        part
        for part in str(material_signature).split("|")
        if not part.startswith("algo=")
    ]
    return "|".join(parts) if parts else None


def _cuda_gemm_hostdispatch_boundary_target_side(
    boundary_family: str | None,
) -> str | None:
    if boundary_family in (None, ""):
        return None
    pieces = str(boundary_family).split(" -> ")
    targets = _CUDALAUNCH_GEMM_HOSTDISPATCH_TARGET_APIS
    if len(pieces) == 2:
        previous_api, current_api = pieces
        if previous_api in targets and current_api in targets:
            return "target_to_target"
        if current_api in targets:
            return "incoming_to_target"
        if previous_api in targets:
            return "target_to_outgoing"
    return str(boundary_family)


def _cuda_gemm_hostdispatch_projection_key(
    *,
    rank: int,
    host_dispatch_queue_id: str,
    api_name: str,
    api_sequence_ordinal: int | None,
    material_signature: object | None,
    algorithm: object | None,
    boundary_target_side: str | None = None,
) -> str:
    material_text = (
        "unavailable" if material_signature in (None, "") else str(material_signature)
    )
    algorithm_text = "unavailable" if algorithm in (None, "") else str(algorithm)
    parts = [
        f"rank:{int(rank)}",
        f"queue:{host_dispatch_queue_id or 'unavailable'}",
        f"api:{api_name or 'unknown'}",
        f"api_seq:{api_sequence_ordinal if api_sequence_ordinal is not None else 'unavailable'}",
        f"material:{material_text}",
        f"algorithm:{algorithm_text}",
    ]
    if boundary_target_side is not None:
        parts.append(f"boundary_target_side:{boundary_target_side}")
    return "|".join(parts)


def _cuda_gemm_hostdispatch_strict_occurrence_common_fields(
    *,
    rank: int,
    source_side: str,
    count_basis_side: str,
    api_name: str,
    component_role: str,
    paper_valid_window_id: object | None,
    host_dispatch_queue_id: str,
    stream_id: object | None,
    api_sequence_ordinal: int | None,
    host_queue_sequence_ordinal: int | None,
    stream_sequence_ordinal: int | None,
    material_signature: object | None,
    algorithm: object | None,
    gemm_shape_signature: object | None,
    boundary_family: str | None,
    predicted_count_once_group_id: object | None,
    actual_count_once_group_id: object | None,
    key_completeness_status: str,
) -> dict[str, object]:
    actual_stream_resource_id = (
        _cuda_gemm_hostdispatch_stream_resource_id(rank=rank, stream_id=stream_id)
        if source_side == "actual_endpoint_metadata"
        else None
    )
    predicted_stream_resource_id = (
        _cuda_gemm_hostdispatch_stream_resource_id(rank=rank, stream_id=stream_id)
        if source_side != "actual_endpoint_metadata"
        else None
    )
    common_key = _cuda_gemm_hostdispatch_strict_occurrence_key(
        rank=rank,
        paper_valid_window_id=paper_valid_window_id,
        host_dispatch_queue_id=host_dispatch_queue_id,
        api_name=api_name,
        component_role=component_role,
        api_sequence_ordinal=api_sequence_ordinal,
        host_queue_sequence_ordinal=host_queue_sequence_ordinal,
        stream_sequence_ordinal=stream_sequence_ordinal,
        material_signature=material_signature,
        algorithm=algorithm,
        boundary_family=boundary_family,
    )
    material_without_algo = _cuda_gemm_hostdispatch_material_without_embedded_algo(
        material_signature
    )
    boundary_target_side = _cuda_gemm_hostdispatch_boundary_target_side(boundary_family)
    endpoint_identity_projection_key = _cuda_gemm_hostdispatch_projection_key(
        rank=rank,
        host_dispatch_queue_id=host_dispatch_queue_id,
        api_name=api_name,
        api_sequence_ordinal=api_sequence_ordinal,
        material_signature=material_without_algo,
        algorithm=algorithm,
    )
    boundary_target_side_projection_key = _cuda_gemm_hostdispatch_projection_key(
        rank=rank,
        host_dispatch_queue_id=host_dispatch_queue_id,
        api_name=api_name,
        api_sequence_ordinal=api_sequence_ordinal,
        material_signature=material_without_algo,
        algorithm=algorithm,
        boundary_target_side=boundary_target_side,
    )
    return {
        "cuda_gemm_hostdispatch_strict_occurrence_gap_schema_version": (
            _CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_SCHEMA_VERSION
        ),
        "cuda_gemm_hostdispatch_strict_occurrence_gap_opt_in_flag": True,
        "cuda_gemm_hostdispatch_strict_occurrence_gap_source_side": source_side,
        "strict_occurrence_common_basis_key": common_key,
        "strict_occurrence_material_without_embedded_algo": material_without_algo,
        "strict_occurrence_boundary_target_side": boundary_target_side,
        "strict_occurrence_endpoint_identity_projection_key": (
            endpoint_identity_projection_key
        ),
        "strict_occurrence_boundary_target_side_projection_key": (
            boundary_target_side_projection_key
        ),
        "strict_occurrence_projection_keys_status": (
            "diagnostic_only_projection_not_strict_join_key"
        ),
        "strict_occurrence_projection_keys_basis": (
            "field_ablation_followup_endpoint_identity_and_boundary_target_side;"
            "drops_host_queue_seq_stream_seq_and_exact_boundary_for_blocker_review"
        ),
        "strict_occurrence_projection_keys_repair_ready": False,
        "strict_occurrence_key_parts": {
            "rank": int(rank),
            "paper_valid_window_id": paper_valid_window_id,
            "host_dispatch_queue_id": host_dispatch_queue_id,
            "api_family": api_name,
            "component_role": component_role,
            "api_sequence_ordinal_in_window": api_sequence_ordinal,
            "host_queue_sequence_ordinal_in_window": host_queue_sequence_ordinal,
            "stream_sequence_ordinal_in_window": stream_sequence_ordinal,
            "material_signature": material_signature,
            "algorithm": algorithm,
            "gemm_shape_signature": gemm_shape_signature,
            "boundary_family": boundary_family,
        },
        "strict_occurrence_count_basis_side": count_basis_side,
        "paper_valid_window_id": paper_valid_window_id,
        "rank": int(rank),
        "host_dispatch_queue_id": host_dispatch_queue_id,
        "api_family": api_name,
        "component_role": component_role,
        "api_sequence_ordinal_in_window": api_sequence_ordinal,
        "host_queue_sequence_ordinal_in_window": host_queue_sequence_ordinal,
        "stream_sequence_ordinal_in_window": stream_sequence_ordinal,
        "material_signature": material_signature,
        "algorithm": algorithm,
        "gemm_shape_signature": gemm_shape_signature,
        "boundary_family": boundary_family,
        "key_completeness_status": key_completeness_status,
        "actual_mechanical_dispatch_split_status": "unavailable",
        "actual_mechanical_dispatch_split_basis": (
            "metadata_only_no_non_perturbing_actual_mechanical_dispatch_split"
        ),
        "actual_control_dispatch_us": None,
        "actual_api_body_us": None,
        "actual_instrumentation_only_us": None,
        "actual_endpoint_timestamps_used_as_dispatch_split": False,
        "actual_host_duration_used_as_dispatch_split": False,
        "actual_runtime_used_as_dispatch_split": False,
        "actual_mechanical_dispatch_split_unavailable_reason": (
            "endpoint ts/end_ts, host_duration_us, and observed runtime are "
            "context only and are not a strict mechanical dispatch split"
        ),
        "predicted_stream_resource_id": predicted_stream_resource_id,
        "actual_stream_resource_id": actual_stream_resource_id,
        "stream_namespace_basis": (
            "rank_scoped_actual_raw_stream_id"
            if source_side == "actual_endpoint_metadata"
            and actual_stream_resource_id is not None
            else "rank_scoped_predicted_stream_id"
            if predicted_stream_resource_id is not None
            else "unavailable_missing_stream_id"
        ),
        "stream_alignment_status": (
            "actual_only_unresolved_predicted_namespace_not_joined"
            if source_side == "actual_endpoint_metadata"
            else "predicted_only_actual_alignment_unavailable"
        ),
        "exact_stream_identity_proven": False,
        "default_stream_equivalence_reviewed": False,
        "predicted_count_once_group_id": predicted_count_once_group_id,
        "actual_count_once_group_id": actual_count_once_group_id,
        "count_once_status": (
            "metadata_only_count_once_group_not_strict_nonoverlap_proof"
            if predicted_count_once_group_id is not None
            or actual_count_once_group_id is not None
            else "unavailable"
        ),
        "nonoverlap_status": "unavailable",
        "provider_runtime_overlap_status": "unavailable",
        "host_dispatch_overlap_status": "unavailable",
        "hostDelay_overlap_status": "unavailable",
        "stream_wait_overlap_status": "unavailable",
        "wait_map_blocking_overlap_status": "unavailable",
        "hostdispatch_producer_visibility_status": "unavailable",
        "hostdispatch_producer_visibility_basis": (
            "no_non_perturbing_producer_visibility_split_available"
        ),
        "paper_visible_host_dispatch_us": None,
        "instrumentation_only_host_dispatch_us": None,
        "unresolved_mixed_host_dispatch_us": None,
        "producer_side_safe_for_repair_design": False,
        "wait_map_safety_status": "unavailable",
        "predicted_wait_map_edge_ids": [],
        "actual_wait_release_source_status": "unavailable",
        "dependency_release_timing_preserved": False,
        "cuda_event_wait_safety_status": "unavailable",
        "collective_wait_safety_status": "unavailable",
        "stream_queue_wait_safety_status": "unavailable",
        "strict_occurrence_join_ready": False,
        "strict_actual_timing_or_mechanical_split_ready": False,
        "strict_apples_to_apples_delta_ready": False,
        "repair_ready": False,
        "safe_to_use_as_repair_evidence": False,
        "safe_to_use_as_subtraction_delta": False,
        "safe_to_use_for_runtime_substitution": False,
        "safe_to_use_for_endpoint_timestamp_substitution": False,
    }


def _add_cuda_gemm_hostdispatch_strict_occurrence_gap_actual_metadata_diagnostics(
    record: dict[str, object],
    *,
    rank: int,
    raw_event_id: str,
    raw_ordinal: int,
    api_sequence_ordinal: int,
    host_queue_sequence_ordinal: int,
    stream_sequence_ordinal: int | None,
    previous_record: dict[str, object] | None,
    host_dispatch_queue_id: str,
    semantic_previous_record: dict[str, object] | None = None,
) -> None:
    api_name = str(record.get("api") or "")
    if api_name not in _CUDALAUNCH_GEMM_HOSTDISPATCH_TARGET_APIS:
        return
    previous_api = (
        str(previous_record.get("api") or "") if previous_record is not None else None
    )
    semantic_previous_api = (
        str(semantic_previous_record.get("api") or "")
        if semantic_previous_record is not None
        else previous_api
    )
    raw_boundary_family = _boundary_family(previous_api, api_name)
    semantic_boundary_family = _boundary_family(semantic_previous_api, api_name)
    material_signature = _component_strict_actual_material_signature(record)
    algorithm = _cuda_gemm_hostdispatch_algorithm(record)
    shape_signature = _gemm_adjacent_shape_signature(record)
    actual_count_once_group_id = f"actual_endpoint:{raw_event_id}"
    fields = _cuda_gemm_hostdispatch_strict_occurrence_common_fields(
        rank=rank,
        source_side="actual_endpoint_metadata",
        count_basis_side="actual_endpoint_row",
        api_name=api_name,
        component_role="actual_mechanical_dispatch_split_candidate",
        paper_valid_window_id=None,
        host_dispatch_queue_id=host_dispatch_queue_id,
        stream_id=record.get("stream_id"),
        api_sequence_ordinal=api_sequence_ordinal,
        host_queue_sequence_ordinal=host_queue_sequence_ordinal,
        stream_sequence_ordinal=stream_sequence_ordinal,
        material_signature=material_signature,
        algorithm=algorithm,
        gemm_shape_signature=shape_signature,
        boundary_family=semantic_boundary_family,
        predicted_count_once_group_id=None,
        actual_count_once_group_id=actual_count_once_group_id,
        key_completeness_status=(
            "actual_endpoint_key_parts_available_without_paper_window_or_predicted_join"
        ),
    )
    record.update(
        {
            **fields,
            "actual_raw_event_id": raw_event_id,
            "actual_raw_ordinal": int(raw_ordinal),
            "actual_api": api_name,
            "actual_raw_immediate_boundary_family": raw_boundary_family,
            "actual_semantic_predecessor_boundary_family": semantic_boundary_family,
            "actual_semantic_predecessor_prev_api": semantic_previous_api,
            "actual_semantic_predecessor_prev_raw_event_id": (
                str(semantic_previous_record.get("raw_event_id"))
                if semantic_previous_record is not None
                else (
                    str(previous_record.get("raw_event_id"))
                    if previous_record is not None
                    else None
                )
            ),
            "actual_boundary_namespace_basis": (
                "semantic_predecessor_control_query_filtered"
                if semantic_boundary_family != raw_boundary_family
                else "raw_immediate_predecessor"
            ),
            "actual_endpoint_ts_us": record.get("ts"),
            "actual_endpoint_end_ts_us": record.get("end_ts"),
            "actual_endpoint_host_duration_us": record.get("host_duration_us"),
            "actual_observed_runtime_us": record.get("observed_runtime_us"),
            "actual_endpoint_context_only": True,
            "actual_endpoint_timestamps_used_as_strict_timing": False,
            "actual_host_duration_used_as_strict_timing": False,
            "actual_runtime_direct_substitution": False,
            "actual_observed_runtime_used_as_prediction": False,
            "strict_actual_timing_status": "unavailable",
            "strict_actual_timing_available": False,
            "actual_start_us": None,
            "actual_end_us": None,
            "actual_duration_us": None,
            "actual_timing_basis": "unavailable_endpoint_context_only",
            "actual_counterpart_join_status": (
                "actual_metadata_export_only_predicted_hostdispatch_join_deferred"
            ),
            "actual_counterpart_join_basis": "not_joined_during_capture",
            "actual_counterpart_unavailable_reason": (
                "predicted host_dispatch/hostDelay occurrence rows are not joined during capture"
            ),
            "cuda_gemm_hostdispatch_strict_occurrence_gap_actual_row_id": (
                f"rank:{int(rank)}:strict_occurrence_gap_actual:raw_ordinal:{int(raw_ordinal)}"
            ),
            "cuda_gemm_hostdispatch_strict_occurrence_gap_actual_endpoint_context_only": True,
        }
    )


def _add_joined_gemm_stream_queue_wait_actual_counterpart_metadata_diagnostics(
    record: dict[str, object],
    *,
    rank: int,
    raw_event_id: str,
    raw_ordinal: int,
    stream_sequence_ordinal: int | None,
    previous_same_stream_record: dict[str, object] | None,
    previous_same_stream_device_record: dict[str, object] | None,
    host_machine_id: str,
    host_dispatch_queue_id: str,
) -> None:
    api_name = str(record.get("api") or "")
    if api_name not in _JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_TARGET_APIS:
        return
    stream_id = record.get("stream_id")
    actual_stream_resource_id = _component_strict_actual_stream_resource_id(
        rank=rank,
        stream_id=stream_id,
    )
    previous_raw_event_id = (
        str(previous_same_stream_record.get("raw_event_id"))
        if previous_same_stream_record is not None
        else None
    )
    previous_api = (
        str(previous_same_stream_record.get("api") or "")
        if previous_same_stream_record is not None
        else None
    )
    previous_material_signature = (
        _component_strict_actual_material_signature(previous_same_stream_record)
        if previous_same_stream_record is not None
        else None
    )
    previous_algorithm = (
        _cuda_gemm_hostdispatch_algorithm(previous_same_stream_record)
        if previous_same_stream_record is not None
        else None
    )
    current_material_signature = _component_strict_actual_material_signature(record)
    current_algorithm = _cuda_gemm_hostdispatch_algorithm(record)
    current_shape_signature = (
        canonical_gemm_material_signature(record)
        if is_gemm_material_api(api_name)
        else None
    )
    previous_stream_sequence_ordinal = (
        previous_same_stream_record.get("stream_sequence_ordinal")
        if previous_same_stream_record is not None
        else None
    )
    previous_cupti_kernel_start = _cupti_activity_previous_kernel_start_text(
        previous_same_stream_record
    )
    previous_cupti_kernel_end = _cupti_activity_previous_kernel_end_text(
        previous_same_stream_record
    )
    current_cupti_kernel_start = _cupti_activity_timestamp_text(
        record,
        "cupti_activity_first_kernel_start",
    )
    current_cupti_kernel_end = _cupti_activity_timestamp_text(
        record,
        "cupti_activity_last_kernel_end",
    ) or _cupti_activity_timestamp_text(
        record,
        "cupti_activity_first_kernel_end",
    )
    current_cupti_kernel_stream_id = _normalized_payload_text(
        record.get("cupti_activity_first_kernel_stream_id")
    )
    previous_cupti_kernel_stream_id = _cupti_activity_previous_kernel_stream_id_text(
        previous_same_stream_record
    )
    cupti_kernel_stream_id_pair_status = _cupti_activity_stream_id_pair_status(
        previous_cupti_kernel_stream_id,
        current_cupti_kernel_stream_id,
    )
    cupti_common_clock_status = _normalized_payload_text(
        record.get("cupti_activity_common_clock_status")
    )
    cupti_gap_ticks = _cupti_activity_stream_order_gap_ticks(
        previous_cupti_kernel_end,
        current_cupti_kernel_start,
    )
    previous_device_raw_event_id = (
        str(previous_same_stream_device_record.get("raw_event_id"))
        if previous_same_stream_device_record is not None
        else None
    )
    previous_device_api = (
        str(previous_same_stream_device_record.get("api") or "")
        if previous_same_stream_device_record is not None
        else None
    )
    previous_device_material_signature = (
        _component_strict_actual_material_signature(previous_same_stream_device_record)
        if previous_same_stream_device_record is not None
        else None
    )
    previous_device_algorithm = (
        _cuda_gemm_hostdispatch_algorithm(previous_same_stream_device_record)
        if previous_same_stream_device_record is not None
        else None
    )
    previous_device_stream_sequence_ordinal = (
        previous_same_stream_device_record.get("stream_sequence_ordinal")
        if previous_same_stream_device_record is not None
        else None
    )
    previous_device_cupti_kernel_start = _cupti_activity_previous_kernel_start_text(
        previous_same_stream_device_record
    )
    previous_device_cupti_kernel_end = _cupti_activity_previous_kernel_end_text(
        previous_same_stream_device_record
    )
    previous_device_cupti_kernel_stream_id = (
        _cupti_activity_previous_kernel_stream_id_text(previous_same_stream_device_record)
    )
    previous_device_cupti_kernel_stream_id_pair_status = (
        _cupti_activity_stream_id_pair_status(
            previous_device_cupti_kernel_stream_id,
            current_cupti_kernel_stream_id,
        )
    )
    previous_device_stream_order_gap_ticks = _cupti_activity_stream_order_gap_ticks(
        previous_device_cupti_kernel_end,
        current_cupti_kernel_start,
    )
    previous_device_predecessor_source = (
        "rank_local_previous_same_stream_cupti_backed_device_predecessor"
        if previous_same_stream_device_record is not None
        else "unavailable"
    )
    if previous_same_stream_device_record is None:
        previous_device_predecessor_status = (
            "unavailable_missing_previous_same_stream_cupti_device_predecessor"
        )
    elif previous_device_cupti_kernel_end is None:
        previous_device_predecessor_status = (
            "unavailable_missing_previous_device_kernel_end_cupti_timestamp"
        )
    elif current_cupti_kernel_start is None:
        previous_device_predecessor_status = (
            "unavailable_missing_current_cupti_kernel_start"
        )
    else:
        previous_device_predecessor_status = (
            "available_previous_same_stream_device_predecessor_gap_unreviewed_clock"
        )
    has_cupti_stream_order_timing = (
        previous_cupti_kernel_end is not None or current_cupti_kernel_start is not None
    )
    previous_key = previous_raw_event_id if previous_raw_event_id else "leading"
    stream_order_pair_id = (
        f"rank:{int(rank)}:stream:{stream_id if stream_id not in (None, '') else 'unavailable'}:"
        f"previous:{previous_key}->current:{raw_event_id}"
    )
    if has_cupti_stream_order_timing:
        source_side = "actual_stream_order_cupti_activity_metadata"
        release_timing_status = (
            "partial_available_cupti_previous_same_stream_kernel_end_unreviewed_clock"
            if previous_cupti_kernel_end is not None
            else "unavailable_missing_previous_same_stream_cupti_kernel_end"
        )
        wait_timing_status = (
            "partial_available_cupti_current_kernel_start_no_enqueue_wait_start"
            if current_cupti_kernel_start is not None
            else "unavailable_missing_current_cupti_kernel_start"
        )
        unavailable_reason = (
            "CUPTI activity exposes device-side same-stream kernel timestamps for "
            "release provenance, but enqueue wait_start/common-clock strict delta is "
            "not reviewed and no actual timing is used as prediction."
        )
    else:
        source_side = "actual_stream_order_endpoint_metadata"
        release_timing_status = "unavailable"
        wait_timing_status = "unavailable"
        unavailable_reason = (
            "capture_real exports same-stream endpoint order metadata only; endpoint "
            "timestamps, host_duration_us, and observed runtime are not wait-map release "
            "timing, stream_queue_wait timing, or strict prediction deltas"
        )
    record.update(
        {
            "joined_gemm_stream_queue_wait_actual_counterpart_schema_version": (
                _JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_SCHEMA_VERSION
            ),
            "joined_gemm_stream_queue_wait_actual_counterpart_opt_in_flag": True,
            "joined_gemm_stream_queue_wait_source_side": source_side,
            "joined_gemm_stream_queue_wait_actual_row_id": (
                f"rank:{int(rank)}:joined_gemm_stream_queue_wait_actual:"
                f"raw_ordinal:{int(raw_ordinal)}"
            ),
            "joined_gemm_stream_queue_wait_actual_rank": int(rank),
            "joined_gemm_stream_queue_wait_actual_api": api_name,
            "joined_gemm_stream_queue_wait_actual_raw_event_id": raw_event_id,
            "joined_gemm_stream_queue_wait_actual_raw_ordinal": int(raw_ordinal),
            "joined_gemm_stream_queue_wait_actual_host_machine_id": host_machine_id,
            "joined_gemm_stream_queue_wait_actual_host_dispatch_queue_id": (
                host_dispatch_queue_id
            ),
            "joined_gemm_stream_queue_wait_actual_stream_id": stream_id,
            "joined_gemm_stream_queue_wait_actual_stream_resource_id": (
                actual_stream_resource_id
            ),
            "joined_gemm_stream_queue_wait_actual_stream_namespace_basis": (
                "rank_scoped_actual_raw_stream_id_process_local"
                if actual_stream_resource_id is not None
                else "unavailable_missing_actual_stream_id"
            ),
            "joined_gemm_stream_queue_wait_actual_stream_sequence_ordinal": (
                stream_sequence_ordinal
            ),
            "joined_gemm_stream_queue_wait_previous_same_stream_raw_event_id": (
                previous_raw_event_id
            ),
            "joined_gemm_stream_queue_wait_previous_same_stream_raw_ordinal": (
                previous_same_stream_record.get("raw_ordinal")
                if previous_same_stream_record is not None
                else None
            ),
            "joined_gemm_stream_queue_wait_previous_same_stream_api": previous_api,
            "joined_gemm_stream_queue_wait_previous_same_stream_material_signature": (
                previous_material_signature
            ),
            "joined_gemm_stream_queue_wait_previous_same_stream_algorithm": (
                previous_algorithm
            ),
            "joined_gemm_stream_queue_wait_previous_same_stream_sequence_ordinal": (
                previous_stream_sequence_ordinal
            ),
            "joined_gemm_stream_queue_wait_previous_device_predecessor_raw_event_id": (
                previous_device_raw_event_id
            ),
            "joined_gemm_stream_queue_wait_previous_device_predecessor_raw_ordinal": (
                previous_same_stream_device_record.get("raw_ordinal")
                if previous_same_stream_device_record is not None
                else None
            ),
            "joined_gemm_stream_queue_wait_previous_device_predecessor_api": (
                previous_device_api
            ),
            "joined_gemm_stream_queue_wait_previous_device_predecessor_material_signature": (
                previous_device_material_signature
            ),
            "joined_gemm_stream_queue_wait_previous_device_predecessor_algorithm": (
                previous_device_algorithm
            ),
            "joined_gemm_stream_queue_wait_previous_device_predecessor_sequence_ordinal": (
                previous_device_stream_sequence_ordinal
            ),
            "joined_gemm_stream_queue_wait_previous_device_predecessor_status": (
                previous_device_predecessor_status
            ),
            "joined_gemm_stream_queue_wait_previous_device_predecessor_source": (
                previous_device_predecessor_source
            ),
            "joined_gemm_stream_queue_wait_previous_device_predecessor_kernel_start_cupti_timestamp": (
                previous_device_cupti_kernel_start
            ),
            "joined_gemm_stream_queue_wait_previous_device_predecessor_kernel_end_cupti_timestamp": (
                previous_device_cupti_kernel_end
            ),
            "joined_gemm_stream_queue_wait_previous_device_predecessor_cupti_kernel_stream_id": (
                previous_device_cupti_kernel_stream_id
            ),
            "joined_gemm_stream_queue_wait_previous_device_predecessor_cupti_kernel_stream_id_pair_status": (
                previous_device_cupti_kernel_stream_id_pair_status
            ),
            "joined_gemm_stream_queue_wait_previous_device_predecessor_stream_order_gap_cupti_ticks": (
                previous_device_stream_order_gap_ticks
            ),
            "joined_gemm_stream_queue_wait_actual_material_signature": (
                current_material_signature
            ),
            "joined_gemm_stream_queue_wait_actual_algorithm": current_algorithm,
            "joined_gemm_stream_queue_wait_actual_gemm_shape_signature": (
                current_shape_signature
            ),
            "joined_gemm_stream_queue_wait_actual_stream_order_pair_id": (
                stream_order_pair_id
            ),
            "joined_gemm_stream_queue_wait_actual_stream_order_pair_basis": (
                "rank_local_previous_endpoint_on_same_raw_stream_id"
            ),
            "joined_gemm_stream_queue_wait_actual_counterpart_join_status": (
                "actual_stream_order_metadata_only_predicted_wait_edge_join_deferred"
            ),
            "joined_gemm_stream_queue_wait_actual_counterpart_join_basis": (
                "not_joined_during_capture"
            ),
            "joined_gemm_stream_queue_wait_actual_release_timing_status": (
                release_timing_status
            ),
            "joined_gemm_stream_queue_wait_actual_wait_timing_status": (
                wait_timing_status
            ),
            "joined_gemm_stream_queue_wait_actual_wait_release_timing_unavailable_reason": (
                unavailable_reason
            ),
            "joined_gemm_stream_queue_wait_actual_wait_start_us": None,
            "joined_gemm_stream_queue_wait_actual_release_us": None,
            "joined_gemm_stream_queue_wait_actual_waited_us": None,
            "joined_gemm_stream_queue_wait_actual_device_timing_source": (
                "cupti_activity_concurrent_kernel"
                if has_cupti_stream_order_timing
                else "unavailable"
            ),
            "joined_gemm_stream_queue_wait_actual_previous_kernel_start_cupti_timestamp": (
                previous_cupti_kernel_start
            ),
            "joined_gemm_stream_queue_wait_actual_previous_kernel_end_cupti_timestamp": (
                previous_cupti_kernel_end
            ),
            "joined_gemm_stream_queue_wait_actual_current_kernel_start_cupti_timestamp": (
                current_cupti_kernel_start
            ),
            "joined_gemm_stream_queue_wait_actual_current_kernel_end_cupti_timestamp": (
                current_cupti_kernel_end
            ),
            "joined_gemm_stream_queue_wait_actual_current_cupti_kernel_stream_id": (
                current_cupti_kernel_stream_id
            ),
            "joined_gemm_stream_queue_wait_actual_previous_cupti_kernel_stream_id": (
                previous_cupti_kernel_stream_id
            ),
            "joined_gemm_stream_queue_wait_actual_cupti_kernel_stream_id_pair_status": (
                cupti_kernel_stream_id_pair_status
            ),
            "joined_gemm_stream_queue_wait_actual_cupti_common_clock_status": (
                cupti_common_clock_status or "unavailable"
            ),
            "joined_gemm_stream_queue_wait_actual_stream_order_gap_cupti_ticks": (
                cupti_gap_ticks
            ),
            "joined_gemm_stream_queue_wait_actual_endpoint_ts_us": record.get("ts"),
            "joined_gemm_stream_queue_wait_actual_endpoint_end_ts_us": record.get("end_ts"),
            "joined_gemm_stream_queue_wait_actual_endpoint_host_duration_us": (
                record.get("host_duration_us")
            ),
            "joined_gemm_stream_queue_wait_actual_observed_runtime_us": (
                record.get("observed_runtime_us")
            ),
            "joined_gemm_stream_queue_wait_actual_endpoint_context_only": True,
            "joined_gemm_stream_queue_wait_endpoint_timestamps_used_as_wait_release": False,
            "joined_gemm_stream_queue_wait_endpoint_timestamps_used_as_strict_delta": False,
            "joined_gemm_stream_queue_wait_actual_runtime_direct_substitution": False,
            "joined_gemm_stream_queue_wait_actual_observed_runtime_used_as_prediction": False,
            "joined_gemm_stream_queue_wait_strict_actual_timing_available": False,
            "joined_gemm_stream_queue_wait_strict_delta_calculable": False,
            "joined_gemm_stream_queue_wait_count_once_status": "unavailable",
            "joined_gemm_stream_queue_wait_count_once_nonoverlap_status": "unavailable",
            "joined_gemm_stream_queue_wait_nonoverlap_status": "unavailable",
            "joined_gemm_stream_queue_wait_wait_map_safety_status": "unavailable",
            "joined_gemm_stream_queue_wait_wait_map_safety_proven": False,
            "joined_gemm_stream_queue_wait_producer_visibility_status": "unavailable",
            "joined_gemm_stream_queue_wait_repair_ready": False,
            "joined_gemm_stream_queue_wait_safe_to_use_as_repair_evidence": False,
            "joined_gemm_stream_queue_wait_safe_to_use_as_subtraction_delta": False,
            "joined_gemm_stream_queue_wait_safe_to_use_for_runtime_substitution": False,
            "joined_gemm_stream_queue_wait_safe_to_use_for_endpoint_timestamp_substitution": False,
        }
    )


def _add_host_control_boundary_counterpart_diagnostics(
    record: dict[str, object],
    *,
    rank: int,
    raw_event_id: str,
    raw_ordinal: int,
    previous_record: dict[str, object] | None,
    host_machine_id: str,
    host_dispatch_queue_id: str,
    enable_launch_neighborhood_equivalence: bool = False,
) -> None:
    api_name = str(record.get("api") or "")
    previous_api = (
        str(previous_record.get("api") or "") if previous_record is not None else None
    )
    boundary_apis = {api for api in (api_name, previous_api) if api}
    interesting_apis = set(_HOST_CONTROL_BOUNDARY_APIS)
    if enable_launch_neighborhood_equivalence:
        interesting_apis |= LAUNCH_NEIGHBORHOOD_EQUIVALENCE_APIS
    if (
        not (boundary_apis & interesting_apis)
    ):
        return

    family = _boundary_family(previous_api, api_name)
    occurrence_id = _host_control_boundary_occurrence_id(
        rank=rank,
        previous_ordinal=(
            int(previous_record["raw_ordinal"]) if previous_record is not None else None
        ),
        current_ordinal=raw_ordinal,
    )
    previous_raw_event_id = (
        str(previous_record.get("raw_event_id")) if previous_record is not None else None
    )
    current_ts_us = record.get("ts")
    current_end_ts_us = record.get("end_ts")
    previous_ts_us = previous_record.get("ts") if previous_record is not None else None
    previous_end_ts_us = (
        previous_record.get("end_ts") if previous_record is not None else None
    )
    inter_host_gap_us = _diagnostic_float_gap_us(
        previous_end_ts_us=previous_end_ts_us,
        current_ts_us=current_ts_us,
    )
    selected_status = (
        "selected_family"
        if family in _HOST_CONTROL_SELECTED_BOUNDARY_FAMILIES
        else "context_touching_selected_host_control_api"
    )
    split_unavailable_reason = (
        "mechanical_visibility_split_not_measured_to_preserve_wrapper_host_timing"
    )
    producer_visibility_unavailable_reason = (
        "structural_labels_only_internal_clocks_disabled_to_preserve_start_time_end_time"
    )
    producer_nonoverlap_unavailable_reason = (
        "producer_cannot_observe_replay_wait_map_stream_collective_host_sync_or_rank_global_overlap"
    )
    exact_counterpart_unavailable_reason = (
        "requires_offline_emulated_selected_occurrence_join_not_available_during_capture"
    )
    producer_supported = api_name in _HOST_CONTROL_PRODUCER_VISIBILITY_APIS
    producer_visibility_status = (
        "structural_unavailable"
        if producer_supported
        else "unavailable_current_api_not_in_real_wrapper_visibility_surface"
    )

    record.update(
        {
            "host_control_visibility_schema_version": (
                _HOST_CONTROL_LAUNCH_NEIGHBORHOOD_SCHEMA_VERSION
            ),
            "host_control_visibility_opt_in_flag": True,
            "host_control_envelope_counterpart_schema_version": (
                _HOST_CONTROL_ENVELOPE_COUNTERPART_SCHEMA_VERSION
            ),
            "host_control_envelope_counterpart_opt_in_flag": True,
            "host_control_envelope_counterpart_key": occurrence_id,
            "hostdelay_counterpart_key": occurrence_id,
            "host_control_envelope_actual_row_id": raw_event_id,
            "host_control_envelope_actual_interval_id": (
                f"{occurrence_id}:actual_endpoint_gap"
            ),
            "host_control_envelope_actual_interval_kind": (
                "actual_endpoint_gap_context_only"
            ),
            "host_control_envelope_prev_raw_event_id": previous_raw_event_id,
            "host_control_envelope_current_raw_event_id": raw_event_id,
            "host_control_envelope_prev_api": previous_api,
            "host_control_envelope_current_api": api_name,
            "host_control_envelope_rank": int(rank),
            "host_control_envelope_stream_id": record.get("stream_id"),
            "host_control_envelope_host_dispatch_queue_id": host_dispatch_queue_id,
            "host_control_envelope_paper_valid_window_id": None,
            "host_control_envelope_prev_raw_ordinal": (
                int(previous_record["raw_ordinal"])
                if previous_record is not None
                else None
            ),
            "host_control_envelope_current_raw_ordinal": int(raw_ordinal),
            "host_control_envelope_timestamp_basis": (
                "actual_previous_end_ts_to_current_ts"
            ),
            "host_control_envelope_interval_start_ts_us": previous_end_ts_us,
            "host_control_envelope_interval_end_ts_us": current_ts_us,
            "host_control_envelope_interval_duration_us": inter_host_gap_us,
            "host_control_envelope_interval_time_basis": (
                "actual_endpoint_gap_context_only"
            ),
            "host_control_envelope_visibility_basis_status": (
                "structural_metadata_only_no_mechanical_visibility_split"
            ),
            "host_control_envelope_visibility_kind": "mixed_or_unresolved",
            "host_control_envelope_replay_overlap_status": "unavailable",
            "host_control_envelope_replay_overlap_unavailable_reason": (
                "capture_real_cannot_observe_replay_wait_map_stream_fifo_or_count_once_overlap"
            ),
            "selected_occurrence_id": occurrence_id,
            "paper_valid_window_id": None,
            "paper_valid_window_unavailable_reason": (
                "paper_valid_step_window_resolved_later_by_collate_or_ledger"
            ),
            "split_sum_check_status": "unavailable",
            "split_tolerance_us": None,
            "classification_basis": (
                "capture_real_adjacent_raw_row_export_without_internal_wrapper_clocks"
            ),
            "classification_unavailable_reason": split_unavailable_reason,
            "host_control_boundary_counterpart_schema_version": (
                _HOST_CONTROL_BOUNDARY_COUNTERPART_SCHEMA_VERSION
            ),
            "host_control_boundary_row_id": raw_event_id,
            "host_control_boundary_occurrence_id": occurrence_id,
            "host_control_boundary_selection_status": selected_status,
            "host_control_boundary_prev_raw_event_id": previous_raw_event_id,
            "host_control_boundary_current_raw_event_id": raw_event_id,
            "host_control_boundary_prev_api": previous_api,
            "host_control_boundary_current_api": api_name,
            "host_control_boundary_family": family,
            "host_control_boundary_prev_ts_us": previous_ts_us,
            "host_control_boundary_prev_end_ts_us": previous_end_ts_us,
            "host_control_boundary_prev_host_duration_us": (
                previous_record.get("host_duration_us")
                if previous_record is not None
                else None
            ),
            "host_control_boundary_current_ts_us": current_ts_us,
            "host_control_boundary_current_end_ts_us": current_end_ts_us,
            "host_control_boundary_current_host_duration_us": record.get(
                "host_duration_us"
            ),
            "host_control_boundary_host_machine_id": host_machine_id,
            "host_control_boundary_dispatch_queue_id": host_dispatch_queue_id,
            "host_control_visibility_split_status": "unavailable",
            "host_control_visibility_split_unavailable_reason": split_unavailable_reason,
            "host_control_visibility_split_basis": (
                "capture_real_adjacent_raw_row_export_without_internal_wrapper_clocks"
            ),
            "mechanical_visibility_split_status": "unavailable",
            "mechanical_visibility_split_unavailable_reason": split_unavailable_reason,
            "actual_counterpart_id": occurrence_id,
            "actual_counterpart_status": (
                "actual_boundary_row_id_exported_selected_occurrence_join_not_attempted"
            ),
            "actual_counterpart_unavailable_reason": exact_counterpart_unavailable_reason,
            "emulated_occurrence_id": None,
            "emulated_occurrence_id_unavailable_reason": (
                "emulated_selected_occurrence_join_not_available_during_capture"
            ),
            "actual_trace_id": None,
            "actual_trace_id_unavailable_reason": (
                "trace_directory_identifier_not_available_in_rank_row_writer"
            ),
            "actual_rank": int(rank),
            "actual_paper_valid_window_id": None,
            "actual_paper_valid_window_unavailable_reason": (
                "paper_valid_step_window_resolved_later_by_collate_or_ledger"
            ),
            "actual_raw_prev_event_id": previous_raw_event_id,
            "actual_raw_current_event_id": raw_event_id,
            "actual_boundary_family": family,
            "counterpart_join_key": occurrence_id,
            "counterpart_join_method": (
                "actual_row_id_only_emulated_selected_occurrence_join_not_attempted"
            ),
            "counterpart_join_confidence": "unavailable",
            "counterpart_unavailable_reason": exact_counterpart_unavailable_reason,
            "comparable_actual_context_only": True,
            "actual_counterpart_component_id": "host_inter_op_overhead",
            "actual_counterpart_rank": int(rank),
            "actual_counterpart_window_id": None,
            "actual_counterpart_window_unavailable_reason": (
                "paper_valid_step_window_resolved_later_by_collate_or_ledger"
            ),
            "actual_counterpart_prev_event_id": previous_raw_event_id,
            "actual_counterpart_current_event_id": raw_event_id,
            "actual_counterpart_boundary_family": family,
            "actual_counterpart_dispatch_queue_id": host_dispatch_queue_id,
            "actual_counterpart_visibility_kind": "mixed_or_unresolved",
            "actual_inter_host_op_gap_us": inter_host_gap_us,
            "actual_inter_host_op_gap_unavailable_reason": (
                None
                if inter_host_gap_us is not None
                else "previous_raw_end_ts_or_current_ts_unavailable"
            ),
            "exact_counterpart_status": "unavailable",
            "exact_counterpart_unavailable_reason": exact_counterpart_unavailable_reason,
            "wait_map_safety_status": "unavailable",
            "wait_map_non_overlap_unavailable_reason": (
                "requires_replay_wait_edge_non_overlap_ledger_not_available_during_capture"
            ),
            "double_counting_overlap_status": "unavailable",
            "double_counting_overlap_unavailable_reason": (
                "requires_figure6_count_once_interval_ledger_not_available_during_capture"
            ),
            "affected_interval_id": None,
            "affected_interval_unavailable_reason": (
                "collate_or_replay_export_required_for_materialized_interval_id"
            ),
            "candidate_subinterval_id": None,
            "candidate_subinterval_unavailable_reason": split_unavailable_reason,
            "interval_kind": "actual_endpoint_gap_context_only",
            "start_ts_us": previous_end_ts_us,
            "end_ts_us": current_ts_us,
            "duration_us": inter_host_gap_us,
            "host_dispatch_interval_ids": [],
            "stream_order_interval_ids": [],
            "cuda_event_wait_edge_ids": [],
            "collective_wait_edge_ids": [],
            "host_sync_interval_ids": [],
            "rank_completion_context_id": None,
            "global_completion_context_id": None,
            "count_once_status": "unavailable",
            "safe_to_use_as_subtraction_delta": False,
            "safe_to_use_as_repair_evidence": False,
            "safe_to_use_as_subtraction_delta_reason": (
                "diagnostic_row_id_export_only_not_actual_runtime_substitution"
            ),
            "safe_to_use_as_repair_evidence_reason": (
                "requires_mechanical_visibility_split_exact_selected_occurrence_join_and_nonoverlap_review"
            ),
        }
    )
    record.setdefault("boundary_origin_kind", "mixed_or_unresolved")
    record.setdefault("boundary_visibility_kind", "mixed_or_unresolved")
    record.setdefault("boundary_origin_classification_basis", split_unavailable_reason)
    record.setdefault("wrapper_segment_coverage", "unavailable")
    record.setdefault(
        "host_control_producer_visibility_schema_version",
        _HOST_CONTROL_PRODUCER_VISIBILITY_SCHEMA_VERSION,
    )
    record.setdefault("host_control_producer_visibility_status", producer_visibility_status)
    record.setdefault(
        "host_control_producer_visibility_unavailable_reason",
        (
            producer_visibility_unavailable_reason
            if producer_supported
            else "current_api_not_in_reviewed_real_wrapper_visibility_surface"
        ),
    )
    record.setdefault(
        "host_control_producer_visibility_basis",
        "capture_real_default_off_structural_metadata_no_internal_wrapper_clocks",
    )
    record.setdefault(
        "host_control_producer_visibility_segments",
        _host_control_producer_visibility_segments(api_name) if producer_supported else [],
    )
    record.setdefault("host_control_producer_numeric_split_status", "unavailable")
    record.setdefault(
        "host_control_producer_numeric_split_unavailable_reason",
        "real_api_body_or_instrumentation_split_not_emitted_without_nonperturbing_brackets",
    )
    record.setdefault("host_control_producer_nonoverlap_status", "unavailable")
    record.setdefault(
        "host_control_producer_nonoverlap_unavailable_reason",
        producer_nonoverlap_unavailable_reason,
    )
    record.setdefault("host_control_producer_wait_map_nonoverlap_status", "unavailable")
    record.setdefault(
        "host_control_producer_wait_map_nonoverlap_unavailable_reason",
        producer_nonoverlap_unavailable_reason,
    )
    record.setdefault(
        "host_control_producer_double_counting_nonoverlap_status",
        "unavailable",
    )
    record.setdefault(
        "host_control_producer_double_counting_nonoverlap_unavailable_reason",
        "producer_cannot_observe_figure6_count_once_interval_nonoverlap",
    )
    record.setdefault("paper_visible_host_duration_us", None)
    record.setdefault("instrumentation_only_duration_us", None)
    record.setdefault("wrapper_internal_duration_us", None)
    record.setdefault("fake_api_body_duration_us", None)
    record.setdefault("runtime_or_framework_duration_us", None)
    record.setdefault("payload_enrichment_duration_us", None)
    record.setdefault("trace_serialization_duration_us", None)
    record.setdefault("mis_materialized_duration_us", None)
    record.setdefault("unresolved_mixed_duration_us", None)
    if enable_launch_neighborhood_equivalence:
        record.update(
            build_launch_neighborhood_equivalence_metadata(
                rank=rank,
                previous_api=previous_api,
                current_api=api_name,
                previous_raw_event_id=previous_raw_event_id,
                current_raw_event_id=raw_event_id,
                previous_raw_ordinal=(
                    int(previous_record["raw_ordinal"])
                    if previous_record is not None
                    else None
                ),
                current_raw_ordinal=int(raw_ordinal),
                host_dispatch_queue_id=host_dispatch_queue_id,
                stream_id=record.get("stream_id"),
                paper_valid_window_id=None,
                role="actual_wrapper_control_interleaved_neighborhood",
            )
        )
    if api_name in _HOST_CONTROL_BOUNDARY_APIS:
        record.setdefault("actual_launch_visibility_kind", "mixed_or_unresolved")
        record.setdefault("actual_launch_unavailable_reason", split_unavailable_reason)
    if api_name == "cudaLaunchKernel":
        record.setdefault(
            "host_control_compat_launch_pop_coverage_status",
            "unavailable_not_exported_by_current_real_wrapper_producer",
        )
        record.setdefault(
            "host_control_compat_launch_pop_coverage_unavailable_reason",
            (
                "__cudaPopCallConfiguration_interposition_not_proven_for_real_libcudart;"
                "do_not_synthesize_compat_launch_family_from_cudaLaunchKernel"
            ),
        )


def _add_actual_cuda_event_counterpart_diagnostics(
    record: dict[str, object],
    *,
    rank: int,
    raw_event_id: str,
    event_versions: dict[str, int],
    latest_record_by_key: dict[tuple[str, int], dict[str, object]],
) -> None:
    api_name = str(record.get("api") or "")
    if api_name not in {"cudaEventRecord", "cudaEventRecordWithFlags", "cudaStreamWaitEvent"}:
        return
    event_handle = record.get("event_id")
    if event_handle in (None, ""):
        record.update(
            {
                "actual_cuda_event_counterpart_schema_version": (
                    _ACTUAL_CUDA_EVENT_COUNTERPART_SCHEMA_VERSION
                ),
                "actual_cuda_event_counterpart_unavailable_reason": "missing_raw_event_id_handle",
                "safe_to_use_as_subtraction_delta": False,
                "safe_to_use_as_repair_evidence": False,
            }
        )
        return

    handle = str(event_handle)
    stream_id = record.get("stream_id")
    ts_us = record.get("ts")
    end_ts_us = record.get("end_ts")
    if api_name in {"cudaEventRecord", "cudaEventRecordWithFlags"}:
        version = int(event_versions.get(handle, 0)) + 1
        event_versions[handle] = version
        pair_id = _event_counterpart_pair_id(rank, handle, version)
        latest_record_by_key[(handle, version)] = {
            "raw_event_id": raw_event_id,
            "api": api_name,
            "ts_us": ts_us,
            "end_ts_us": end_ts_us,
            "stream_id": stream_id,
        }
        record.update(
            {
                "actual_cuda_event_counterpart_schema_version": (
                    _ACTUAL_CUDA_EVENT_COUNTERPART_SCHEMA_VERSION
                ),
                "actual_cuda_event_handle": handle,
                "actual_cuda_event_version": version,
                "actual_record_wait_pair_id": pair_id,
                "actual_record_raw_event_id": raw_event_id,
                "actual_record_api": api_name,
                "actual_record_ts_us": ts_us,
                "actual_record_end_ts_us": end_ts_us,
                "actual_record_stream_id": stream_id,
                "actual_release_us": None,
                "actual_released_by_event_id": raw_event_id,
                "actual_release_reason": "record_operation_identified_release_timing_unavailable",
                "actual_wait_start_us": None,
                "actual_waited_us": None,
                "actual_stream_namespace_basis": "actual_raw_stream_id_process_local",
                "actual_stream_namespace_alignment": (
                    "actual_only_unresolved_predicted_namespace_not_joined"
                ),
                "actual_cuda_event_counterpart_source_provenance": (
                    "capture_real_opt_in_event_handle_versioning"
                ),
                "actual_cuda_event_counterpart_unavailable_reason": (
                    "device_stream_release_timing_not_observable_from_wrapper_endpoint"
                ),
                "safe_to_use_as_subtraction_delta": False,
                "safe_to_use_as_repair_evidence": False,
            }
        )
        return

    version = int(event_versions.get(handle, 0))
    pair_id = _event_counterpart_pair_id(rank, handle, version)
    record_info = latest_record_by_key.get((handle, version))
    release_reason = (
        "record_operation_identified_release_timing_unavailable"
        if record_info is not None
        else "missing_record_unresolved"
    )
    record.update(
        {
            "actual_cuda_event_counterpart_schema_version": (
                _ACTUAL_CUDA_EVENT_COUNTERPART_SCHEMA_VERSION
            ),
            "actual_cuda_event_handle": handle,
            "actual_cuda_event_version": version,
            "actual_record_wait_pair_id": pair_id,
            "actual_wait_raw_event_id": raw_event_id,
            "actual_wait_api": api_name,
            "actual_wait_api_ts_us": ts_us,
            "actual_wait_api_end_ts_us": end_ts_us,
            "actual_wait_stream_id": stream_id,
            "actual_wait_start_us": None,
            "actual_release_us": None,
            "actual_waited_us": None,
            "actual_release_reason": release_reason,
            "actual_released_by_event_id": (
                record_info.get("raw_event_id") if record_info is not None else None
            ),
            "actual_record_raw_event_id": (
                record_info.get("raw_event_id") if record_info is not None else None
            ),
            "actual_record_api": record_info.get("api") if record_info is not None else None,
            "actual_record_ts_us": record_info.get("ts_us") if record_info is not None else None,
            "actual_record_end_ts_us": (
                record_info.get("end_ts_us") if record_info is not None else None
            ),
            "actual_record_stream_id": (
                record_info.get("stream_id") if record_info is not None else None
            ),
            "actual_stream_namespace_basis": "actual_raw_stream_id_process_local",
            "actual_stream_namespace_alignment": (
                "actual_only_unresolved_predicted_namespace_not_joined"
            ),
            "actual_cuda_event_counterpart_source_provenance": (
                "capture_real_opt_in_event_handle_versioning"
            ),
            "actual_cuda_event_counterpart_unavailable_reason": (
                "device_stream_wait_start_and_release_timing_not_observable_from_wrapper_endpoint"
            ),
            "safe_to_use_as_subtraction_delta": False,
            "safe_to_use_as_repair_evidence": False,
        }
    )


def _add_appendix_ab_p2p_actual_counterpart_diagnostics(
    record: dict[str, object],
    *,
    rank: int,
    raw_event_id: str,
    raw_ordinal: int,
    normalized_call_orders: dict[tuple[object, ...], int],
) -> None:
    api_name = str(record.get("api") or "")
    if api_name not in {"ncclSend", "ncclRecv"}:
        return

    group_api = "ncclP2P"
    p2p_direction = _p2p_collective_name(api_name, record)
    collective = "p2p"
    comm_id = _normalized_payload_text(record.get("comm_id"))
    canonical_comm_id = _normalized_payload_text(
        record.get("collective_communicator_id")
    ) or comm_id
    peer = record.get("peer")
    comm_members = _p2p_comm_members(record)
    pair_members = _p2p_pair_members(rank, peer, comm_members)
    count_or_numel = _p2p_count_or_numel(record)
    datatype_or_dtype_code = _p2p_datatype_or_dtype_code(record)
    shape_signature = _p2p_shape_signature(
        group_api=group_api,
        collective=collective,
        count=count_or_numel,
        datatype=datatype_or_dtype_code,
    )
    if canonical_comm_id is not None and pair_members is not None:
        call_order_key: tuple[object, ...] | None = (
            "communicator_pair_sequence",
            "p2p",
            canonical_comm_id,
            tuple(pair_members),
        )
        normalized_call_order = normalized_call_orders.get(call_order_key, 0)
        normalized_call_orders[call_order_key] = normalized_call_order + 1
        pair_seq_unavailable_reason = None
    else:
        call_order_key = None
        normalized_call_order = None
        pair_seq_unavailable_reason = (
            "communicator_resolved_pair_members_unavailable_during_raw_rank_write"
        )
    row_id = _p2p_actual_row_id(rank, raw_ordinal)
    occurrence_id = _p2p_occurrence_id(rank, api_name, normalized_call_order)
    stream_id = record.get("stream_id")
    actual_stream_resource_id = (
        f"rank:{int(rank)}:stream:{stream_id}"
        if stream_id not in (None, "")
        else None
    )

    release_unavailable_reason = (
        "strict device-stream wait-map release timing is not observable from "
        "actual wrapper API endpoints; actual_api_end_ts_us is endpoint "
        "provenance only and is not a release or block timestamp"
    )
    record.update(
        {
            "p2p_actual_counterpart_schema_version": (
                _APPENDIX_AB_P2P_ACTUAL_COUNTERPART_SCHEMA_VERSION
            ),
            "p2p_actual_counterpart_opt_in_flag": True,
            "actual_p2p_row_id": row_id,
            "actual_p2p_occurrence_id": occurrence_id,
            "actual_rank": int(rank),
            "actual_api": api_name,
            "actual_raw_event_id": raw_event_id,
            "actual_raw_ordinal": int(raw_ordinal),
            "actual_trace_id": None,
            "actual_trace_id_unavailable_reason": (
                "trace_directory_identifier_not_available_in_rank_row_writer"
            ),
            "actual_paper_valid_window_id": None,
            "actual_in_paper_valid_window": None,
            "actual_paper_valid_window_unavailable_reason": (
                "paper_valid_step_window_resolved_later_by_collate_or_offline_ledger"
            ),
            "actual_comm_id": comm_id,
            "actual_canonical_comm_id": canonical_comm_id,
            "actual_comm_members": comm_members,
            "actual_comm_members_unavailable_reason": (
                None
                if comm_members is not None
                else "communicator_membership_recovered_later_by_collate_or_offline_ledger"
            ),
            "actual_pair_members": pair_members,
            "actual_pair_members_basis": (
                "communicator_membership_resolved_peer_local_rank"
                if pair_members is not None
                else "unavailable_without_communicator_membership"
            ),
            "actual_pair_members_unavailable_reason": (
                None
                if pair_members is not None
                else "raw_peer_payload_is_communicator_local_and_not_safe_as_global_rank_without_membership"
            ),
            "actual_peer": peer,
            "actual_pair_seq": (
                int(normalized_call_order)
                if normalized_call_order is not None
                else None
            ),
            "actual_pair_seq_key": (
                list(call_order_key[:-1]) + [list(call_order_key[-1])]
                if call_order_key is not None
                else None
            ),
            "actual_pair_seq_unavailable_reason": pair_seq_unavailable_reason,
            "actual_call_idx": record.get("call_idx"),
            "actual_group_api": group_api,
            "actual_collective": collective,
            "actual_p2p_direction": p2p_direction,
            "actual_count_or_numel": count_or_numel,
            "actual_datatype_or_dtype_code": datatype_or_dtype_code,
            "actual_shape_signature": shape_signature,
            "actual_normalized_call_order": (
                int(normalized_call_order)
                if normalized_call_order is not None
                else None
            ),
            "actual_stream_id": stream_id,
            "actual_raw_stream_id": stream_id,
            "actual_canonical_stream_id": stream_id,
            "actual_stream_namespace_basis": (
                "actual_raw_stream_id_process_local"
                if stream_id not in (None, "")
                else "unavailable_missing_actual_stream_id"
            ),
            "predicted_stream_resource_id": None,
            "actual_stream_resource_id": actual_stream_resource_id,
            "stream_namespace_alignment": (
                "actual_only_unresolved_predicted_namespace_not_joined"
            ),
            "actual_api_ts_us": record.get("ts"),
            "actual_api_end_ts_us": record.get("end_ts"),
            "actual_api_host_duration_us": record.get("host_duration_us"),
            "actual_observed_runtime_us": record.get("observed_runtime_us"),
            "actual_endpoint_context_only": True,
            "actual_api_end_ts_used_as_release": False,
            "actual_api_end_ts_used_as_block_end": False,
            "actual_wait_start_us": None,
            "actual_release_us": None,
            "actual_waited_us": None,
            "actual_release_reason": None,
            "actual_released_by_event_id": None,
            "actual_released_by_raw_event_id": None,
            "actual_release_source_kind": None,
            "actual_release_observability_status": "strict_release_timing_unavailable",
            "actual_release_unavailable_reason": release_unavailable_reason,
            "actual_block_start_us": None,
            "actual_block_end_us": None,
            "actual_block_duration_us": None,
            "actual_block_timing_unavailable_reason": (
                "strict actual per-block timing is not exported by capture_real; "
                "actual API endpoint timestamps are retained only as endpoint context"
            ),
            "predicted_stable_block_id": None,
            "predicted_collective_group_id": None,
            "predicted_pair_seq": None,
            "predicted_wait_edge_id": None,
            "predicted_release_us": None,
            "actual_counterpart_join_key": {
                "rank": int(rank),
                "api": api_name,
                "group_api": group_api,
                "canonical_comm_id": canonical_comm_id,
                "pair_members": pair_members,
                "peer": peer,
                "shape_signature": shape_signature,
                "normalized_call_order": (
                    int(normalized_call_order)
                    if normalized_call_order is not None
                    else None
                ),
            },
            "actual_counterpart_join_method": (
                "actual_row_metadata_export_only_predicted_join_not_attempted_during_capture"
            ),
            "actual_counterpart_join_confidence": "unavailable",
            "actual_counterpart_status": (
                "actual_p2p_row_id_exported_predicted_join_not_attempted"
            ),
            "double_counting_overlap_status": "unavailable",
            "double_counting_overlap_unavailable_reason": (
                "requires_offline_count_once_overlap_ledger_not_available_during_capture"
            ),
            "wait_map_safety_status": "unavailable",
            "wait_map_safety_unavailable_reason": (
                "strict_actual_wait_map_release_source_timing_unavailable"
            ),
            "safe_to_use_as_repair_evidence": False,
            "safe_to_use_as_subtraction_delta": False,
            "safe_to_use_as_repair_evidence_reason": (
                "diagnostic_metadata_only_no_paper_contract_violation_proven"
            ),
            "safe_to_use_as_subtraction_delta_reason": (
                "actual_runtime_endpoint_timing_is_not_a_prediction_substitution_delta"
            ),
            "diagnostic_only": True,
            "repair_ready": False,
            "native_capture_or_compare_run_for_this_metadata": False,
        }
    )


def _shared_phase_anchor_collective_metadata(
    api_name: str,
    record: dict[str, object],
    *,
    rank: int,
    normalized_call_orders: dict[tuple[object, ...], int],
    include_common_basis: bool = False,
) -> dict[str, object] | None:
    common_call_order_key: tuple[object, ...] | None = None
    common_call_order_basis = "unavailable_unsupported_api"
    common_call_order_index: int | None = None
    common_group_id_call_index: int | None = None
    common_pair_seq: int | None = None
    common_key_unavailable_reasons: list[str] = []
    if api_name in {"ncclSend", "ncclRecv"}:
        group_api = "ncclP2P"
        collective = "p2p"
        peer = record.get("peer")
        comm_members = _p2p_comm_members(record)
        pair_members = _p2p_pair_members(rank, peer, comm_members)
        canonical_comm_id = _normalized_payload_text(
            record.get("collective_communicator_id")
        ) or _normalized_payload_text(record.get("comm_id"))
        shape_signature = _p2p_shape_signature(
            group_api=group_api,
            collective=collective,
            count=_p2p_count_or_numel(record),
            datatype=_p2p_datatype_or_dtype_code(record),
        )
        call_order_key = (
            "shared_phase_anchor",
            group_api,
            canonical_comm_id,
            tuple(pair_members or ()),
            shape_signature,
            api_name,
        )
        common_call_order_key = (
            "shared_phase_anchor_common_basis",
            group_api,
            canonical_comm_id,
            tuple(pair_members or ()),
            shape_signature,
        )
        pair_id = (
            f"rank:{int(rank)}:p2p:{api_name}:pair_seq:"
            f"{normalized_call_orders.get(call_order_key, 0)}"
        )
        participant_rank_ids = pair_members
        peer_rank = None if pair_members is None else next(
            (member for member in pair_members if int(member) != int(rank)),
            None,
        )
        raw_peer = peer
        raw_peer_local_rank = _int_payload(peer)
        peer_rank_unavailable_reason = (
            None
            if peer_rank is not None
            else "raw_peer_payload_is_communicator_local_and_not_safe_as_global_rank_without_membership"
        )
        if canonical_comm_id is not None and pair_members is not None:
            common_pair_seq = normalized_call_orders.get(common_call_order_key, 0)
            common_call_order_index = common_pair_seq
            common_call_order_basis = "communicator_pair_sequence"
        else:
            common_call_order_basis = "unavailable_missing_resolved_pair_members"
            common_key_unavailable_reasons.append("common_pair_seq_unavailable")
    elif api_name == "ncclAllReduce":
        group_api = "ncclAllReduce"
        collective = "allreduce"
        peer = None
        raw_peer = None
        raw_peer_local_rank = None
        pair_members = None
        pair_id = None
        canonical_comm_id = _normalized_payload_text(
            record.get("collective_communicator_id")
        ) or _normalized_payload_text(record.get("comm_id"))
        participant_rank_ids = _p2p_comm_members(record)
        shape_signature = _p2p_shape_signature(
            group_api=group_api,
            collective=collective,
            count=_p2p_count_or_numel(record),
            datatype=_p2p_datatype_or_dtype_code(record),
        )
        call_order_key = (
            "shared_phase_anchor",
            group_api,
            canonical_comm_id,
            tuple(participant_rank_ids or ()),
            shape_signature,
        )
        peer_rank = None
        peer_rank_unavailable_reason = None
        (
            common_group_id_call_index,
            common_call_order_basis,
            group_call_unavailable_reason,
        ) = _allreduce_semantic_call_index(record)
        common_call_order_index = common_group_id_call_index
        if group_call_unavailable_reason is not None:
            common_key_unavailable_reasons.append(group_call_unavailable_reason)
    else:
        return None

    normalized_call_order = normalized_call_orders.get(call_order_key, 0)
    normalized_call_orders[call_order_key] = normalized_call_order + 1
    if (
        common_call_order_key is not None
        and common_call_order_index is not None
        and common_call_order_basis == "communicator_pair_sequence"
    ):
        normalized_call_orders[common_call_order_key] = common_call_order_index + 1
    members_for_payload = pair_members if collective == "p2p" else participant_rank_ids
    count_or_numel = _p2p_count_or_numel(record)
    datatype_or_dtype_code = _p2p_datatype_or_dtype_code(record)
    reduction_op = record.get("op")
    common_payload, common_payload_inputs = _common_payload_signature(
        group_api=group_api,
        collective_kind=collective,
        api_name=api_name,
        members=members_for_payload,
        count=count_or_numel,
        datatype=datatype_or_dtype_code,
        op=reduction_op,
    )
    if members_for_payload is None:
        common_key_unavailable_reasons.append("common_membership_unavailable")
    metadata: dict[str, object] = {
        "group_api": group_api,
        "canonical_communicator_id": canonical_comm_id,
        "pair_id": pair_id,
        "pair_members": pair_members,
        "participant_rank_ids": participant_rank_ids,
        "peer_rank": peer_rank,
        "peer_rank_unavailable_reason": peer_rank_unavailable_reason,
        "raw_peer": raw_peer,
        "raw_peer_local_rank": raw_peer_local_rank,
        "raw_peer_semantics": (
            "communicator_local_rank_provenance_only"
            if raw_peer not in (None, "")
            else None
        ),
        "shape_signature": shape_signature,
        "payload_signature": shape_signature,
        "normalized_call_order": int(normalized_call_order),
    }
    if include_common_basis:
        metadata.update(
            {
                "common_basis_schema_version": (
                    _SHARED_PHASE_ANCHOR_COMMON_BASIS_SCHEMA_VERSION
                ),
                "common_call_order_basis": common_call_order_basis,
                "common_call_order_index": common_call_order_index,
                "common_group_id_call_index": common_group_id_call_index,
                "common_pair_seq": common_pair_seq,
                "common_rank_window_index": int(normalized_call_order),
                "common_payload_signature": common_payload,
                "common_payload_signature_inputs": common_payload_inputs,
                "payload_basis": "raw_operation_semantics_not_stream_only_key",
                "common_api": api_name,
                "common_group_api": group_api,
                "common_api_direction": (
                    "send"
                    if api_name == "ncclSend"
                    else "recv"
                    if api_name == "ncclRecv"
                    else None
                ),
                "common_collective_kind": collective,
                "common_count": count_or_numel,
                "common_datatype": datatype_or_dtype_code,
                "common_reduction_op": reduction_op,
                "common_membership_signature": _common_membership_signature(
                    members_for_payload
                ),
                "common_pair_members": (
                    list(pair_members)
                    if collective == "p2p" and pair_members is not None
                    else None
                ),
                "common_tensor_or_count_shape": count_or_numel,
                "common_key_unavailable_reason": (
                    ";".join(common_key_unavailable_reasons)
                    if common_key_unavailable_reasons
                    else None
                ),
            }
        )
    return metadata


def _add_shared_phase_anchor_actual_counterpart_diagnostics(
    record: dict[str, object],
    *,
    rank: int,
    raw_event_id: str,
    raw_ordinal: int,
    previous_record: dict[str, object] | None,
    host_dispatch_queue_id: str,
    normalized_call_orders: dict[tuple[object, ...], int],
    include_common_basis: bool = False,
    include_selected_allreduce_release_participant_host_dispatch_phase: bool = False,
    include_nccl_wait_release_counterpart: bool = False,
) -> None:
    api_name = str(record.get("api") or "")
    collective = _shared_phase_anchor_collective_metadata(
        api_name,
        record,
        rank=rank,
        normalized_call_orders=normalized_call_orders,
        include_common_basis=include_common_basis,
    )
    if collective is None:
        return

    previous_raw_event_id = (
        str(previous_record.get("raw_event_id")) if previous_record is not None else None
    )
    previous_api = (
        str(previous_record.get("api") or "") if previous_record is not None else None
    )
    occurrence_id = _host_control_boundary_occurrence_id(
        rank=rank,
        previous_ordinal=(
            int(previous_record["raw_ordinal"]) if previous_record is not None else None
        ),
        current_ordinal=raw_ordinal,
    )
    unavailable_reason = (
        "strict actual device-stream wait start/release/source timing is not "
        "observable from current wrapper endpoint rows; endpoint ts/end_ts are "
        "provenance only and must not be used as release or block timing"
    )
    stream_id = record.get("stream_id")
    record.update(
        {
            "shared_anchor_actual_counterpart_schema_version": (
                _SHARED_PHASE_ANCHOR_COUNTERPART_SCHEMA_VERSION
            ),
            "diagnostic_opt_in_flag": True,
            "source_side": "actual_endpoint_provenance",
            "actual_raw_event_id": raw_event_id,
            "actual_rank": int(rank),
            "actual_api": api_name,
            "actual_raw_ordinal": int(raw_ordinal),
            "actual_trace_id": None,
            "actual_trace_id_unavailable_reason": (
                "trace_directory_identifier_not_available_in_rank_row_writer"
            ),
            "actual_timestamp_basis": "wrapper_endpoint_provenance_only",
            "actual_paper_window_id": None,
            "actual_in_paper_window": None,
            "actual_paper_window_unavailable_reason": (
                "paper_valid_step_window_resolved_later_by_collate_or_offline_ledger"
            ),
            "actual_stream_id": stream_id,
            "actual_raw_stream_id": stream_id,
            "actual_canonical_stream_id": stream_id,
            "actual_stream_namespace_basis": (
                "actual_raw_stream_id_process_local"
                if stream_id not in (None, "")
                else "unavailable_missing_actual_stream_id"
            ),
            "host_dispatch_queue_id": host_dispatch_queue_id,
            "host_dispatch_occurrence_id": occurrence_id,
            "communicator_id": record.get("comm_id"),
            "group_id": record.get("collective_group_id"),
            "phase_counterpart_id": None,
            "phase_counterpart_unavailable_reason": (
                "predicted_phase_anchor_join_not_attempted_during_raw_capture"
            ),
            "block_counterpart_id": None,
            "block_counterpart_unavailable_reason": (
                "predicted_block_join_not_attempted_during_raw_capture"
            ),
            "adjacent_prev_raw_event_id": previous_raw_event_id,
            "adjacent_prev_api": previous_api,
            "adjacent_next_raw_event_id": None,
            "adjacent_next_api": None,
            "adjacent_next_unavailable_reason": (
                "capture_real_streaming_writer_exports_previous_endpoint_only"
            ),
            "actual_endpoint_ts_us": record.get("ts"),
            "actual_endpoint_end_ts_us": record.get("end_ts"),
            "actual_endpoint_host_duration_us": record.get("host_duration_us"),
            "actual_observed_runtime_us": record.get("observed_runtime_us"),
            "actual_endpoint_context_only": True,
            "actual_endpoint_end_ts_used_as_release": False,
            "actual_endpoint_end_ts_used_as_block_end": False,
            "strict_actual_wait_start_us": None,
            "strict_actual_release_us": None,
            "strict_actual_waited_us": None,
            "strict_actual_release_reason": None,
            "strict_actual_released_by_event_id": None,
            "strict_actual_release_observability_status": (
                "unavailable_without_native_device_stream_release_observer"
            ),
            "actual_block_start_us": None,
            "actual_block_end_us": None,
            "actual_block_duration_us": None,
            "actual_block_timing_unavailable_reason": (
                "strict actual per-block timing is not exported by capture_real; "
                "actual endpoint ts/end_ts are retained only as endpoint context"
            ),
            "stream_namespace_alignment_status": (
                "actual_only_unresolved_predicted_namespace_not_joined"
            ),
            "safe_to_use_as_repair_evidence": False,
            "safe_to_use_as_subtraction_delta": False,
            "unavailable_reason": unavailable_reason,
        }
    )
    record.update(collective)
    if include_nccl_wait_release_counterpart and api_name == "ncclAllReduce":
        stream_id = record.get("actual_stream_id")
        actual_stream_resource_id = (
            f"rank:{int(rank)}:stream:{stream_id}"
            if stream_id not in (None, "")
            else None
        )
        members = collective.get("participant_rank_ids")
        membership_signature = _common_membership_signature(members)
        wait_release_counterpart_id = (
            "actual:ncclAllReduce:"
            f"rank:{int(rank)}:"
            f"call:{collective.get('common_call_order_index')}:"
            f"{membership_signature or 'members:unavailable'}:"
            f"{collective.get('common_payload_signature') or collective.get('shape_signature')}"
        )
        release_unavailable_reason = (
            "strict actual collective all-participants-ready wait start/release/source "
            "timing is not observable from wrapper endpoint rows; endpoint ts/end_ts "
            "are retained only as provenance and must not be used as wait-map release "
            "or duration fields"
        )
        record.update(
            {
                "nccl_wait_release_counterpart_schema_version": (
                    _NCCL_WAIT_RELEASE_COUNTERPART_SCHEMA_VERSION
                ),
                "nccl_wait_release_counterpart_opt_in_flag": True,
                "actual_collective_wait_release_counterpart_id": (
                    wait_release_counterpart_id
                ),
                "actual_collective_wait_release_scope": (
                    "actual_ncclAllReduce_endpoint_metadata_for_strict_runtime_waitmap_join"
                ),
                "actual_collective_wait_release_api": api_name,
                "actual_collective_wait_release_rank": int(rank),
                "actual_collective_wait_release_raw_event_id": raw_event_id,
                "actual_collective_wait_release_raw_ordinal": int(raw_ordinal),
                "actual_collective_group_api": collective.get("group_api"),
                "actual_collective_kind": collective.get("common_collective_kind")
                or collective.get("group_api"),
                "actual_collective_members": (
                    list(members) if isinstance(members, list) else members
                ),
                "actual_collective_membership_signature": membership_signature,
                "actual_collective_shape_signature": collective.get("shape_signature"),
                "actual_collective_payload_signature": collective.get(
                    "common_payload_signature"
                ),
                "actual_collective_call_order_index": collective.get(
                    "common_call_order_index"
                ),
                "actual_collective_call_order_basis": collective.get(
                    "common_call_order_basis"
                ),
                "actual_collective_call_order_unavailable_reason": collective.get(
                    "common_key_unavailable_reason"
                ),
                "actual_collective_count": collective.get("common_count"),
                "actual_collective_datatype": collective.get("common_datatype"),
                "actual_collective_reduction_op": collective.get(
                    "common_reduction_op"
                ),
                "actual_stream_resource_id": actual_stream_resource_id,
                "actual_stream_namespace_basis": (
                    "rank_scoped_actual_raw_stream_id_process_local"
                    if actual_stream_resource_id is not None
                    else "unavailable_missing_actual_stream_id"
                ),
                "actual_stream_namespace_replay_comparable": False,
                "actual_stream_namespace_alignment": (
                    "actual_only_unresolved_predicted_namespace_not_joined"
                ),
                "stream_namespace_alignment": (
                    "actual_only_unresolved_predicted_namespace_not_joined"
                ),
                "stream_namespace_alignment_status": (
                    "actual_only_unresolved_predicted_namespace_not_joined"
                ),
                "stream_namespace_alignment_unavailable_reason": (
                    "predicted replay stream_resource_id is not available during "
                    "actual capture; offline join must compare rank-scoped actual "
                    "stream resource to predicted stream resource"
                ),
                "predicted_stream_resource_id": None,
                "actual_collective_wait_start_us": None,
                "actual_collective_release_us": None,
                "actual_collective_waited_us": None,
                "actual_collective_release_reason": None,
                "actual_collective_released_by_event_id": None,
                "actual_collective_released_by_raw_event_id": None,
                "actual_collective_release_source_kind": None,
                "actual_collective_release_observability_status": (
                    "strict_release_timing_unavailable_from_wrapper_endpoint"
                ),
                "actual_collective_release_unavailable_reason": (
                    release_unavailable_reason
                ),
                "actual_wait_start_us": None,
                "actual_release_us": None,
                "actual_waited_us": None,
                "actual_release_reason": None,
                "actual_released_by_event_id": None,
                "actual_released_by_raw_event_id": None,
                "actual_release_source_kind": None,
                "actual_release_observability_status": (
                    "strict_release_timing_unavailable_from_wrapper_endpoint"
                ),
                "actual_release_unavailable_reason": release_unavailable_reason,
                "wait_map_counterpart_id": None,
                "wait_map_counterpart_id_unavailable_reason": (
                    "predicted wait edge id is only available during replay/offline join"
                ),
                "strict_runtime_delta_safe": False,
                "strict_waitmap_delta_safe": False,
                "strict_delta_safety_status": (
                    "blocked_missing_stream_alignment_and_actual_wait_release_timing"
                ),
                "strict_delta_safety_unavailable_reason": (
                    "metadata export only; strict deltas require offline counterpart "
                    "join plus actual collective release timing/source evidence"
                ),
                "diagnostic_only": True,
                "repair_ready": False,
                "native_capture_or_compare_run_for_this_metadata": False,
            }
        )
    if (
        include_selected_allreduce_release_participant_host_dispatch_phase
        and api_name == "ncclAllReduce"
    ):
        previous_ordinal = (
            int(previous_record["raw_ordinal"]) if previous_record is not None else None
        )
        previous_ts_us = (
            previous_record.get("ts") if previous_record is not None else None
        )
        previous_end_ts_us = (
            previous_record.get("end_ts") if previous_record is not None else None
        )
        previous_host_duration_us = (
            previous_record.get("host_duration_us")
            if previous_record is not None
            else None
        )
        record.update(
            {
                "selected_allreduce_release_participant_host_dispatch_phase_schema_version": (
                    _SELECTED_ALLREDUCE_RELEASE_PARTICIPANT_HOST_DISPATCH_PHASE_SCHEMA_VERSION
                ),
                "selected_allreduce_release_participant_host_dispatch_phase_opt_in_flag": True,
                "selected_allreduce_release_participant_scope": (
                    "actual_ncclAllReduce_rows_for_common_basis_selected_release_participant_join"
                ),
                "actual_release_participant_candidate": True,
                "actual_release_participant_api": api_name,
                "actual_release_participant_rank": int(rank),
                "actual_release_participant_raw_event_id": raw_event_id,
                "actual_release_participant_raw_ordinal": int(raw_ordinal),
                "actual_release_participant_host_dispatch_queue_id": (
                    host_dispatch_queue_id
                ),
                "actual_release_participant_host_queue_position": int(raw_ordinal),
                "actual_release_participant_host_queue_position_basis": (
                    "rank_trace_raw_ordinal_within_host_dispatch_queue"
                ),
                "actual_release_participant_host_phase_order_basis": (
                    "raw_rank_writer_order_plus_common_call_order_payload_join"
                ),
                "actual_release_participant_host_phase_timing_status": (
                    "endpoint_provenance_only_strict_phase_timing_unavailable"
                ),
                "actual_release_participant_host_phase_timing_unavailable_reason": (
                    "capture_real records API endpoint rows but does not observe "
                    "strict host-dispatch queue arrival/release phase timing on the "
                    "same basis as replay host_dispatch_end"
                ),
                "actual_release_participant_prev_raw_event_id": previous_raw_event_id,
                "actual_release_participant_prev_raw_ordinal": previous_ordinal,
                "actual_release_participant_prev_api": previous_api,
                "actual_release_participant_prev_ts_us": previous_ts_us,
                "actual_release_participant_prev_end_ts_us": previous_end_ts_us,
                "actual_release_participant_prev_host_duration_us": (
                    previous_host_duration_us
                ),
                "actual_release_participant_next_raw_event_id": None,
                "actual_release_participant_next_api": None,
                "actual_release_participant_next_unavailable_reason": (
                    "capture_real_streaming_writer_exports_previous_endpoint_only"
                ),
                "actual_release_participant_endpoint_ts_us": record.get("ts"),
                "actual_release_participant_endpoint_end_ts_us": record.get("end_ts"),
                "actual_release_participant_endpoint_host_duration_us": record.get(
                    "host_duration_us"
                ),
                "actual_release_participant_endpoint_timing_context_only": True,
                "actual_release_participant_endpoint_ts_used_as_phase_arrival": False,
                "actual_release_participant_endpoint_end_ts_used_as_phase_release": False,
                "actual_release_participant_endpoint_end_ts_used_as_wait_release": False,
                "actual_release_participant_endpoint_end_ts_used_as_block_timing": False,
                "actual_host_dispatch_phase_arrival_us": None,
                "actual_host_dispatch_phase_release_us": None,
                "actual_host_dispatch_phase_duration_us": None,
                "actual_host_dispatch_phase_queue_wait_us": None,
                "actual_host_dispatch_phase_strict_timing_status": "unavailable",
                "actual_host_dispatch_phase_strict_timing_unavailable_reason": (
                    "strict actual host-dispatch phase timing is not observable from "
                    "wrapper endpoint ts/end_ts without a separate reviewed producer"
                ),
                "actual_wait_map_release_timing_status": (
                    "unavailable_not_endpoint_timing"
                ),
                "actual_wait_map_release_timing_unavailable_reason": (
                    "endpoint end_ts is provenance/order only and is not a collective "
                    "wait-map release timestamp"
                ),
                "counterpart_join_basis": (
                    "actual_api + common_payload_signature + common_membership_signature "
                    "+ rank + common_call_order_index"
                ),
                "counterpart_join_attempted_during_capture": False,
                "counterpart_join_status": (
                    "actual_metadata_export_only_predicted_selected_row_join_deferred"
                ),
                "trace_window_compatibility": (
                    "raw extras are additive; paper-valid window resolution remains "
                    "in collate/offline ledger"
                ),
                "paper_valid_window_id": None,
                "paper_valid_window_unavailable_reason": (
                    "paper_valid_step_window_resolved_later_by_collate_or_offline_ledger"
                ),
                "diagnostic_only": True,
                "repair_ready": False,
                "safe_to_use_as_repair_evidence": False,
                "safe_to_use_as_subtraction_delta": False,
                "safe_to_use_as_repair_evidence_reason": (
                    "metadata export only; no paper/Maya semantic mismatch proven"
                ),
                "safe_to_use_as_subtraction_delta_reason": (
                    "actual endpoint/order metadata is not a prediction timing delta"
                ),
            }
        )


def _write_rank_trace(output_dir: Path, events) -> Path:
    rank = _rank_from_env()
    world_size = _world_size_from_env()
    host_machine_id = _host_machine_id_from_env()
    host_dispatch_queue_id = _host_dispatch_queue_id_from_env(
        rank=rank,
        host_machine_id=host_machine_id,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rank_{rank}.jsonl"
    enable_actual_cuda_event_counterpart_diagnostics = _env_flag_enabled(
        *_ACTUAL_CUDA_EVENT_COUNTERPART_ENV_KEYS
    )
    enable_host_control_boundary_counterpart_diagnostics = _env_flag_enabled(
        *_HOST_CONTROL_BOUNDARY_COUNTERPART_ENV_KEYS
    ) or _env_flag_enabled(*_HOST_CONTROL_ENVELOPE_COUNTERPART_ENV_KEYS)
    enable_launch_neighborhood_equivalence_diagnostics = _env_flag_enabled(
        *LAUNCH_NEIGHBORHOOD_EQUIVALENCE_ENV_KEYS
    )
    enable_host_control_boundary_counterpart_diagnostics = (
        enable_host_control_boundary_counterpart_diagnostics
        or enable_launch_neighborhood_equivalence_diagnostics
    )
    enable_appendix_ab_p2p_actual_counterpart_diagnostics = _env_flag_enabled(
        *_APPENDIX_AB_P2P_ACTUAL_COUNTERPART_ENV_KEYS
    )
    enable_shared_phase_anchor_common_basis_diagnostics = _env_flag_truthy(
        *_SHARED_PHASE_ANCHOR_COMMON_BASIS_ENV_KEYS
    )
    enable_selected_allreduce_release_participant_host_dispatch_phase_diagnostics = (
        _env_flag_truthy(
            *_SELECTED_ALLREDUCE_RELEASE_PARTICIPANT_HOST_DISPATCH_PHASE_ENV_KEYS
        )
    )
    enable_nccl_wait_release_counterpart_diagnostics = _env_flag_enabled(
        *_NCCL_WAIT_RELEASE_COUNTERPART_ENV_KEYS
    )
    enable_generic_replay_placement_envelope_actual_counterpart_diagnostics = (
        _env_flag_truthy(
            *_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ACTUAL_COUNTERPART_ENV_KEYS
        )
    )
    enable_component_strict_counterpart_metadata_diagnostics = _env_flag_truthy(
        *_COMPONENT_STRICT_COUNTERPART_METADATA_ENV_KEYS
    )
    enable_gemm_adjacent_hostdelay_boundary_actual_counterpart_diagnostics = (
        _env_flag_truthy(
            *_GEMM_ADJACENT_HOSTDELAY_BOUNDARY_ACTUAL_COUNTERPART_ENV_KEYS
        )
    )
    enable_cuda_gemm_hostdispatch_strict_occurrence_gap_diagnostics = (
        _env_flag_truthy(
            *_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_ENV_KEYS
        )
    )
    enable_joined_gemm_stream_queue_wait_actual_counterpart_diagnostics = (
        _env_flag_truthy(
            *_JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_ENV_KEYS
        )
    )
    enable_shared_phase_anchor_explicit_diagnostics = _env_flag_truthy(
        *_SHARED_PHASE_ANCHOR_COUNTERPART_ENV_KEYS
    )
    enable_shared_phase_anchor_counterpart_diagnostics = (
        enable_shared_phase_anchor_explicit_diagnostics
        or enable_shared_phase_anchor_common_basis_diagnostics
        or enable_selected_allreduce_release_participant_host_dispatch_phase_diagnostics
        or enable_nccl_wait_release_counterpart_diagnostics
    )
    actual_cuda_event_versions: dict[str, int] = {}
    actual_cuda_event_latest_record_by_key: dict[tuple[str, int], dict[str, object]] = {}
    appendix_ab_p2p_normalized_call_orders: dict[tuple[object, ...], int] = {}
    shared_phase_anchor_normalized_call_orders: dict[tuple[object, ...], int] = {}
    previous_host_control_record: dict[str, object] | None = None
    semantic_predecessor_record: dict[str, object] | None = None
    api_sequence_counts: dict[str, int] = {}
    stream_sequence_counts: dict[object, int] = {}
    previous_same_stream_record_by_stream: dict[object, dict[str, object]] = {}
    previous_same_stream_device_record_by_stream: dict[object, dict[str, object]] = {}

    with output_path.open("w", encoding="utf-8") as handle:
        for raw_ordinal, event in enumerate(events):
            kind_name = event.kind.name
            end_timestamp = getattr(event, "end_timestamp", None)
            host_duration = getattr(event, "host_duration", None)
            record = {
                "ts": _timedelta_to_microseconds(event.timestamp),
                "pid": int(getattr(event, "process_id", os.getpid())),
                "tid": int(getattr(event, "thread_id", 0)),
                "mod": _module_for_api(event.api_name),
                "api": event.api_name,
                "type": _type_for_event(event.api_name, kind_name),
                "host_machine_id": host_machine_id,
                "host_dispatch_queue_id": host_dispatch_queue_id,
            }
            payload = getattr(getattr(event, "payload", None), "attributes", None)
            if payload:
                record.update({str(key): value for key, value in dict(payload).items()})
                _normalize_boundary_visibility_payload(record)
            has_authoritative_async_runtime = _has_authoritative_async_runtime_observation(
                record,
                api_name=event.api_name,
                event_type=str(record["type"]),
            )
            wrapper_runtime_contract = _wrapper_runtime_contract_for_event(
                event.api_name,
                str(record["type"]),
            )
            if end_timestamp is not None:
                record["end_ts"] = _timedelta_to_microseconds(end_timestamp)
            if host_duration is not None:
                record["host_duration_us"] = max(
                    float(_timedelta_to_microseconds(host_duration)),
                    0.0,
                )
            if has_authoritative_async_runtime:
                record["wrapper_runtime_contract"] = "async_runtime"
                record.pop("direct_runtime_us", None)
            elif wrapper_runtime_contract is not None:
                record["wrapper_runtime_contract"] = wrapper_runtime_contract
                if wrapper_runtime_contract == "direct_runtime":
                    if "host_duration_us" in record:
                        record["direct_runtime_us"] = float(record["host_duration_us"])
                    elif "end_ts" in record:
                        record["direct_runtime_us"] = max(
                            float(record["end_ts"]) - float(record["ts"]),
                            0.0,
                        )
                else:
                    record.pop("direct_runtime_us", None)
            if world_size is not None and "world_size" not in record:
                record["world_size"] = str(world_size)
            raw_event_id = _raw_trace_event_id(rank, raw_ordinal)
            api_sequence_ordinal = api_sequence_counts.get(event.api_name, 0)
            api_sequence_counts[event.api_name] = api_sequence_ordinal + 1
            stream_id_for_sequence = record.get("stream_id")
            stream_sequence_ordinal: int | None = None
            previous_same_stream_record = None
            previous_same_stream_device_record = None
            if stream_id_for_sequence not in (None, ""):
                previous_same_stream_record = previous_same_stream_record_by_stream.get(
                    stream_id_for_sequence
                )
                previous_same_stream_device_record = (
                    previous_same_stream_device_record_by_stream.get(
                        stream_id_for_sequence
                    )
                )
                stream_sequence_ordinal = stream_sequence_counts.get(
                    stream_id_for_sequence,
                    0,
                )
                stream_sequence_counts[stream_id_for_sequence] = (
                    stream_sequence_ordinal + 1
                )
            if enable_component_strict_counterpart_metadata_diagnostics:
                _add_component_strict_counterpart_actual_metadata_diagnostics(
                    record,
                    rank=rank,
                    world_size=world_size,
                    raw_event_id=raw_event_id,
                    raw_ordinal=raw_ordinal,
                    previous_record=previous_host_control_record,
                    host_machine_id=host_machine_id,
                    host_dispatch_queue_id=host_dispatch_queue_id,
                )
            if enable_generic_replay_placement_envelope_actual_counterpart_diagnostics:
                _add_generic_replay_placement_envelope_actual_counterpart_diagnostics(
                    record,
                    rank=rank,
                    raw_event_id=raw_event_id,
                    raw_ordinal=raw_ordinal,
                    previous_record=previous_host_control_record,
                    host_machine_id=host_machine_id,
                    host_dispatch_queue_id=host_dispatch_queue_id,
                )
            if enable_gemm_adjacent_hostdelay_boundary_actual_counterpart_diagnostics:
                if _gemm_adjacent_boundary_in_scope(
                    (
                        str(previous_host_control_record.get("api") or "")
                        if previous_host_control_record is not None
                        else None
                    ),
                    event.api_name,
                ):
                    _add_host_control_boundary_counterpart_diagnostics(
                        record,
                        rank=rank,
                        raw_event_id=raw_event_id,
                        raw_ordinal=raw_ordinal,
                        previous_record=previous_host_control_record,
                        host_machine_id=host_machine_id,
                        host_dispatch_queue_id=host_dispatch_queue_id,
                        enable_launch_neighborhood_equivalence=(
                            enable_launch_neighborhood_equivalence_diagnostics
                        ),
                    )
                _add_gemm_adjacent_hostdelay_boundary_actual_counterpart_diagnostics(
                    record,
                    rank=rank,
                    raw_event_id=raw_event_id,
                    raw_ordinal=raw_ordinal,
                    previous_record=previous_host_control_record,
                    host_machine_id=host_machine_id,
                    host_dispatch_queue_id=host_dispatch_queue_id,
                )
            if enable_host_control_boundary_counterpart_diagnostics:
                _add_host_control_boundary_counterpart_diagnostics(
                    record,
                    rank=rank,
                    raw_event_id=raw_event_id,
                    raw_ordinal=raw_ordinal,
                    previous_record=previous_host_control_record,
                    host_machine_id=host_machine_id,
                    host_dispatch_queue_id=host_dispatch_queue_id,
                    enable_launch_neighborhood_equivalence=(
                        enable_launch_neighborhood_equivalence_diagnostics
                    ),
                )
            if enable_actual_cuda_event_counterpart_diagnostics:
                _add_actual_cuda_event_counterpart_diagnostics(
                    record,
                    rank=rank,
                    raw_event_id=raw_event_id,
                    event_versions=actual_cuda_event_versions,
                    latest_record_by_key=actual_cuda_event_latest_record_by_key,
                )
            if enable_appendix_ab_p2p_actual_counterpart_diagnostics:
                _add_appendix_ab_p2p_actual_counterpart_diagnostics(
                    record,
                    rank=rank,
                    raw_event_id=raw_event_id,
                    raw_ordinal=raw_ordinal,
                    normalized_call_orders=appendix_ab_p2p_normalized_call_orders,
                )
            if enable_shared_phase_anchor_counterpart_diagnostics and (
                event.api_name == "ncclAllReduce"
                or (
                    not enable_selected_allreduce_release_participant_host_dispatch_phase_diagnostics
                    and not enable_nccl_wait_release_counterpart_diagnostics
                )
                or enable_shared_phase_anchor_explicit_diagnostics
                or enable_shared_phase_anchor_common_basis_diagnostics
            ):
                _add_shared_phase_anchor_actual_counterpart_diagnostics(
                    record,
                    rank=rank,
                    raw_event_id=raw_event_id,
                    raw_ordinal=raw_ordinal,
                    previous_record=previous_host_control_record,
                    host_dispatch_queue_id=host_dispatch_queue_id,
                    normalized_call_orders=shared_phase_anchor_normalized_call_orders,
                    include_common_basis=(
                        enable_shared_phase_anchor_common_basis_diagnostics
                        or (
                            enable_selected_allreduce_release_participant_host_dispatch_phase_diagnostics
                            and event.api_name == "ncclAllReduce"
                        )
                        or (
                            enable_nccl_wait_release_counterpart_diagnostics
                            and event.api_name == "ncclAllReduce"
                        )
                    ),
                    include_selected_allreduce_release_participant_host_dispatch_phase=(
                        enable_selected_allreduce_release_participant_host_dispatch_phase_diagnostics
                    ),
                    include_nccl_wait_release_counterpart=(
                        enable_nccl_wait_release_counterpart_diagnostics
                    ),
                )
            if enable_cuda_gemm_hostdispatch_strict_occurrence_gap_diagnostics:
                _add_cuda_gemm_hostdispatch_strict_occurrence_gap_actual_metadata_diagnostics(
                    record,
                    rank=rank,
                    raw_event_id=raw_event_id,
                    raw_ordinal=raw_ordinal,
                    api_sequence_ordinal=api_sequence_ordinal,
                    host_queue_sequence_ordinal=raw_ordinal,
                    stream_sequence_ordinal=stream_sequence_ordinal,
                    previous_record=previous_host_control_record,
                    host_dispatch_queue_id=host_dispatch_queue_id,
                    semantic_previous_record=semantic_predecessor_record,
                )
            if enable_joined_gemm_stream_queue_wait_actual_counterpart_diagnostics:
                _add_joined_gemm_stream_queue_wait_actual_counterpart_metadata_diagnostics(
                    record,
                    rank=rank,
                    raw_event_id=raw_event_id,
                    raw_ordinal=raw_ordinal,
                    stream_sequence_ordinal=stream_sequence_ordinal,
                    previous_same_stream_record=previous_same_stream_record,
                    previous_same_stream_device_record=(
                        previous_same_stream_device_record
                    ),
                    host_machine_id=host_machine_id,
                    host_dispatch_queue_id=host_dispatch_queue_id,
                )
            handle.write(json.dumps(record) + "\n")
            raw_row_snapshot = _host_control_raw_row_snapshot(
                record,
                raw_event_id=raw_event_id,
                raw_ordinal=raw_ordinal,
            )
            raw_row_snapshot["stream_sequence_ordinal"] = stream_sequence_ordinal
            previous_host_control_record = raw_row_snapshot
            if not _is_hostdelay_semantic_predecessor_control_query_api(event.api_name):
                semantic_predecessor_record = raw_row_snapshot
            if stream_id_for_sequence not in (None, ""):
                previous_same_stream_record_by_stream[stream_id_for_sequence] = (
                    raw_row_snapshot
                )
                if _has_cupti_activity_device_predecessor_timing(raw_row_snapshot):
                    previous_same_stream_device_record_by_stream[
                        stream_id_for_sequence
                    ] = raw_row_snapshot

    return output_path


def _write_capture_manifest(
    output_dir: Path,
    *,
    world_size: int | None,
    profiled_ranks: tuple[int, ...],
    profiled_rank_groups: dict[int, tuple[int, ...]],
    fidelity_windows: dict[int, dict[str, object]] | None = None,
    rank_host_machines: dict[int, str] | None = None,
    rank_host_dispatch_queues: dict[int, str] | None = None,
    communicator_memberships: dict[str, tuple[int, ...]] | None = None,
    communicator_aliases: dict[int, dict[str, str]] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "capture_manifest.json"
    canonical_fidelity_windows = {
        int(rank): fidelity_window_from_payload(window, default_source="manifest")
        for rank, window in sorted((fidelity_windows or {}).items())
    }
    resolved_fidelity_windows = {
        str(rank): fidelity_window.to_dict()
        for rank, fidelity_window in canonical_fidelity_windows.items()
        if fidelity_window is not None
    }
    resolved_step_windows = {
        str(rank): fidelity_window.to_dict()
        for rank, fidelity_window in canonical_fidelity_windows.items()
        if fidelity_window is not None and fidelity_window.is_paper_valid_step_window
    }
    payload = {
        "original_world_size": world_size,
        "profiled_ranks": list(profiled_ranks),
        "profiled_rank_groups": {
            str(rank): list(ranks) for rank, ranks in sorted(profiled_rank_groups.items())
        },
        "rank_host_machines": {
            str(rank): str(host_machine_id)
            for rank, host_machine_id in sorted((rank_host_machines or {}).items())
        },
        "rank_host_dispatch_queues": {
            str(rank): str(host_dispatch_queue_id)
            for rank, host_dispatch_queue_id in sorted(
                (rank_host_dispatch_queues or {}).items()
            )
        },
        "communicators": {
            str(comm_id): {"members": [int(member) for member in members]}
            for comm_id, members in sorted((communicator_memberships or {}).items())
        },
        "communicator_aliases": {
            str(rank): {
                str(local_comm_id): str(canonical_comm_id)
                for local_comm_id, canonical_comm_id in sorted(alias_map.items())
            }
            for rank, alias_map in sorted((communicator_aliases or {}).items())
        },
        "step_windows": resolved_step_windows,
        "fidelity_windows": resolved_fidelity_windows,
    }
    payload.update(_HOST_TIMING_LINE_DISABLED_PAYLOAD)
    _atomic_write_json(manifest_path, payload)
    return manifest_path


def _read_manifest_payload(manifest_path: Path) -> dict[str, object]:
    if not manifest_path.exists():
        return {}
    raw = manifest_path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


@contextmanager
def _manifest_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".capture_manifest.lock"
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _merge_capture_manifest(
    output_dir: Path,
    *,
    world_size: int | None,
    profiled_ranks: tuple[int, ...],
    profiled_rank_groups: dict[int, tuple[int, ...]],
    rank: int | None = None,
    fidelity_window: dict[str, object] | None = None,
    host_machine_id: str | None = None,
    host_dispatch_queue_id: str | None = None,
    communicator_memberships: dict[str, tuple[int, ...]] | None = None,
    communicator_aliases: dict[int, dict[str, str]] | None = None,
    route_metadata: dict[str, object] | None = None,
) -> Path:
    with _manifest_lock(output_dir):
        manifest_path = output_dir / "capture_manifest.json"
        payload = _read_manifest_payload(manifest_path)
        payload["original_world_size"] = world_size
        payload["profiled_ranks"] = list(profiled_ranks)
        payload["profiled_rank_groups"] = {
            str(group_rank): list(ranks)
            for group_rank, ranks in sorted(profiled_rank_groups.items())
        }
        rank_host_machines_payload = payload.get("rank_host_machines", {})
        if not isinstance(rank_host_machines_payload, dict):
            rank_host_machines_payload = {}
        if rank is not None and host_machine_id is not None:
            rank_host_machines_payload[str(rank)] = str(host_machine_id)
        payload["rank_host_machines"] = rank_host_machines_payload
        rank_host_dispatch_queues_payload = payload.get("rank_host_dispatch_queues", {})
        if not isinstance(rank_host_dispatch_queues_payload, dict):
            rank_host_dispatch_queues_payload = {}
        if rank is not None and host_dispatch_queue_id is not None:
            rank_host_dispatch_queues_payload[str(rank)] = str(host_dispatch_queue_id)
        payload["rank_host_dispatch_queues"] = rank_host_dispatch_queues_payload
        communicators_payload = payload.get("communicators", {})
        if not isinstance(communicators_payload, dict):
            communicators_payload = {}
        for comm_id, members in sorted((communicator_memberships or {}).items()):
            communicators_payload[str(comm_id)] = {
                "members": [int(member) for member in members]
            }
        payload["communicators"] = communicators_payload
        communicator_aliases_payload = payload.get("communicator_aliases", {})
        if not isinstance(communicator_aliases_payload, dict):
            communicator_aliases_payload = {}
        for alias_rank, alias_map in sorted((communicator_aliases or {}).items()):
            rank_payload = communicator_aliases_payload.get(str(alias_rank), {})
            if not isinstance(rank_payload, dict):
                rank_payload = {}
            for local_comm_id, canonical_comm_id in sorted(alias_map.items()):
                rank_payload[str(local_comm_id)] = str(canonical_comm_id)
            communicator_aliases_payload[str(alias_rank)] = rank_payload
        payload["communicator_aliases"] = communicator_aliases_payload
        for key, value in _HOST_TIMING_LINE_DISABLED_PAYLOAD.items():
            payload[key] = value
        if route_metadata:
            existing_route_metadata = payload.get("route_metadata", {})
            if not isinstance(existing_route_metadata, dict):
                existing_route_metadata = {}
            existing_route_metadata.update(route_metadata)
            payload["route_metadata"] = existing_route_metadata
        fidelity_windows_payload = payload.get("fidelity_windows", {})
        if not isinstance(fidelity_windows_payload, dict):
            fidelity_windows_payload = {}
        canonical_fidelity_window = (
            fidelity_window_from_payload(fidelity_window, default_source="manifest")
            if fidelity_window is not None
            else None
        )
        if rank is not None and canonical_fidelity_window is not None:
            fidelity_windows_payload[str(rank)] = canonical_fidelity_window.to_dict()
        payload["fidelity_windows"] = fidelity_windows_payload

        step_windows_payload = payload.get("step_windows", {})
        if not isinstance(step_windows_payload, dict):
            step_windows_payload = {}
        if rank is not None:
            if (
                canonical_fidelity_window is not None
                and canonical_fidelity_window.is_paper_valid_step_window
            ):
                step_windows_payload[str(rank)] = canonical_fidelity_window.to_dict()
            else:
                step_windows_payload.pop(str(rank), None)
        payload["step_windows"] = step_windows_payload
        _atomic_write_json(manifest_path, payload)
        return manifest_path


def _recover_output_dir_communicators(
    output_dir: Path,
) -> tuple[dict[str, tuple[int, ...]], dict[int, dict[str, str]]]:
    trace_files = sorted(
        trace_file
        for trace_file in output_dir.glob("rank_*.jsonl")
        if not trace_file.name.endswith(".markers.jsonl")
    )
    if not trace_files:
        return {}, {}

    recovery = recover_communicator_topology_from_events(
        event
        for trace_file in trace_files
        for event in iter_rank_trace_events(trace_file, source=TraceSource.REAL)
    )
    alias_map_by_rank: dict[int, dict[str, str]] = {}
    for (alias_rank, local_comm_id), canonical_comm_id in sorted(
        recovery.local_comm_aliases.items()
    ):
        alias_map_by_rank.setdefault(int(alias_rank), {})[str(local_comm_id)] = str(
            canonical_comm_id
        )
    return dict(recovery.memberships), alias_map_by_rank


def merge_real_trace_nodes(node_dirs: list[Path], merged_dir: Path) -> None:
    fidelity_windows: dict[str, dict[str, object]] = {}
    step_windows: dict[str, dict[str, object]] = {}
    profiled_ranks: set[int] = set()
    profiled_rank_groups: dict[str, list[int]] = {}
    rank_host_machines: dict[str, str] = {}
    rank_host_dispatch_queues: dict[str, str] = {}
    communicators: dict[str, object] = {}
    communicator_aliases: dict[str, object] = {}
    route_metadata: dict[str, object] = {}
    original_world_size: int | None = None
    merged_from: list[str] = []
    trace_copies: list[tuple[Path, str]] = []
    for node_dir in node_dirs:
        merged_from.append(str(node_dir))
        for rank_file in sorted(node_dir.glob("rank_*.jsonl")):
            if rank_file.name.endswith(".markers.jsonl"):
                continue
            markers = rank_file.with_name(f"{rank_file.stem}.markers.jsonl")
            trace_copies.append((rank_file, rank_file.name))
            if markers.exists():
                trace_copies.append((markers, markers.name))
            try:
                profiled_ranks.add(int(rank_file.stem.split("_", 1)[1]))
            except Exception:
                pass
        manifest = node_dir / "capture_manifest.json"
        if not manifest.exists():
            raise ValueError(f"missing capture_manifest.json for real trace node: {node_dir}")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        world_size = payload.get("original_world_size")
        if world_size not in (None, ""):
            original_world_size = int(world_size)
        for rank in payload.get("profiled_ranks", []):
            profiled_ranks.add(int(rank))
        for representative, ranks in payload.get("profiled_rank_groups", {}).items():
            profiled_rank_groups[str(representative)] = [int(rank) for rank in ranks]
        for rank, value in payload.get("rank_host_machines", {}).items():
            rank_host_machines[str(rank)] = str(value)
        for rank, value in payload.get("rank_host_dispatch_queues", {}).items():
            rank_host_dispatch_queues[str(rank)] = str(value)
        communicators.update(payload.get("communicators", {}) or {})
        communicator_aliases.update(payload.get("communicator_aliases", {}) or {})
        node_route_metadata = payload.get("route_metadata", {})
        if isinstance(node_route_metadata, dict):
            for key, value in node_route_metadata.items():
                if key == "capture_command":
                    route_metadata.setdefault("node_capture_commands", []).append(value)
                else:
                    if key in route_metadata and route_metadata[key] != value:
                        if key in _ROUTE_METADATA_CONFLICT_KEYS:
                            raise ValueError(
                                "conflicting route metadata while merging real traces: "
                                f"key={key!r} existing={route_metadata[key]!r} "
                                f"node={node_dir} value={value!r}"
                            )
                        route_metadata[f"node_{node_dir.name}_{key}"] = value
                    else:
                        route_metadata.setdefault(key, value)
        raw_fidelity_windows = payload.get("fidelity_windows", {})
        if not isinstance(raw_fidelity_windows, dict) or not raw_fidelity_windows:
            raw_fidelity_windows = payload.get("step_windows", {})
        if isinstance(raw_fidelity_windows, dict):
            for rank, window in raw_fidelity_windows.items():
                fidelity_window = fidelity_window_from_payload(window, default_source="manifest")
                if fidelity_window is None:
                    continue
                fidelity_windows[str(rank)] = fidelity_window.to_dict()
                if fidelity_window.is_paper_valid_step_window:
                    step_windows[str(rank)] = fidelity_window.to_dict()
    missing_route_metadata = sorted(_REQUIRED_ROUTE_METADATA_KEYS - set(route_metadata))
    if missing_route_metadata:
        raise ValueError(
            "merged real trace is missing required route_metadata keys: "
            + ", ".join(missing_route_metadata)
        )
    merged_manifest = {
        "original_world_size": original_world_size,
        "profiled_ranks": sorted(profiled_ranks),
        "profiled_rank_groups": profiled_rank_groups,
        "rank_host_machines": rank_host_machines,
        "rank_host_dispatch_queues": rank_host_dispatch_queues,
        "communicators": communicators,
        "communicator_aliases": communicator_aliases,
        "fidelity_windows": fidelity_windows,
        "step_windows": step_windows,
        "route_metadata": route_metadata,
        "merged_from": merged_from,
    }
    merged_manifest.update(_HOST_TIMING_LINE_DISABLED_PAYLOAD)
    merged_dir.mkdir(parents=True, exist_ok=True)
    for src, name in trace_copies:
        (merged_dir / name).write_bytes(src.read_bytes())
    _atomic_write_json(merged_dir / "capture_manifest.json", merged_manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture a real workload into Maya-lite rank_*.jsonl traces"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profiled-ranks",
        default=None,
        help="Comma-separated ranks to trace. Other launched ranks still execute but do not emit rank_*.jsonl traces.",
    )
    parser.add_argument(
        "--profiled-rank-groups",
        default=None,
        help="Optional explicit representative-rank mapping like '0:0,1;2:2,3'. Written into capture_manifest.json for later reconstruction-aware evaluation.",
    )
    parser.add_argument(
        "--auto-profiled-strategy",
        choices=["single", "pairwise", "identity"],
        default=None,
        help="Automatically choose representative ranks from WORLD_SIZE when explicit profiled ranks are not provided.",
    )
    parser.add_argument(
        "--route-metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Route/provenance metadata to merge into capture_manifest.json under route_metadata; may be repeated.",
    )
    parser.add_argument("script", type=Path, help="Python script to execute in-process")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rank = _rank_from_env()
    world_size = _world_size_from_env()
    host_machine_id = _host_machine_id_from_env()
    host_dispatch_queue_id = _host_dispatch_queue_id_from_env(
        rank=rank,
        host_machine_id=host_machine_id,
    )
    profiled_ranks = _parse_rank_list(args.profiled_ranks) or _profiled_ranks_from_env()
    profiled_rank_groups = _parse_profiled_rank_groups(args.profiled_rank_groups) or _profiled_rank_groups_from_env()
    route_metadata = _parse_route_metadata(args.route_metadata)
    workload_args = list(
        args.script_args[1:]
        if args.script_args and args.script_args[0] == "--"
        else args.script_args
    )
    route_metadata.setdefault(
        "capture_command", " ".join(shlex.quote(item) for item in sys.argv)
    )
    route_metadata.setdefault("workload_script", str(args.script))
    route_metadata.setdefault("workload_args", workload_args)
    if args.auto_profiled_strategy:
        route_metadata.setdefault("auto_profiled_strategy", args.auto_profiled_strategy)
    if args.profiled_ranks:
        route_metadata.setdefault("profiled_ranks_arg", args.profiled_ranks)
    if args.profiled_rank_groups:
        route_metadata.setdefault("profiled_rank_groups_arg", args.profiled_rank_groups)
    if not profiled_ranks and not profiled_rank_groups and args.auto_profiled_strategy and world_size:
        profiled_rank_groups = plan_profiled_rank_groups(
            world_size,
            strategy=args.auto_profiled_strategy,
        )
        profiled_ranks = profiled_ranks_for_groups(profiled_rank_groups)
    if profiled_ranks and not profiled_rank_groups:
        profiled_rank_groups = {profiled_rank: (profiled_rank,) for profiled_rank in profiled_ranks}
    should_trace_rank = not profiled_ranks or rank in set(profiled_ranks)

    _merge_capture_manifest(
        args.output_dir,
        world_size=world_size,
        profiled_ranks=profiled_ranks,
        profiled_rank_groups=profiled_rank_groups,
        rank=rank,
        host_machine_id=host_machine_id,
        host_dispatch_queue_id=host_dispatch_queue_id,
        route_metadata=route_metadata,
    )

    cpp_event = None
    log = None
    if should_trace_rank:
        import cpp_event_py as cpp_event  # type: ignore[no-redef]

        if getattr(cpp_event, "__fallback__", False):
            raise RuntimeError(
                "capture_real requires native cpp_event_py/cpp_event_tls extensions "
                "on profiled ranks; cpp_event_py resolved to fallback. Build or "
                "install the native CppEvent Python extensions before capture."
            )

        context = cpp_event.EventContext()
        log = cpp_event.EventLog(context)
        adapter = cpp_event.EventLogRecorderAdapter(log)
        cpp_event.set_recorder(adapter)
        marker_path = args.output_dir / f"rank_{rank}.markers.jsonl"
        marker_path.unlink(missing_ok=True)
        os.environ["FLEXSIM_MAYA_MARKERS_PATH"] = str(marker_path)
        os.environ["FLEXSIM_CAPTURE_REAL_ENABLE_ASYNC_RUNTIME"] = "1"

    script_argv = [str(args.script)]
    if args.script_args and args.script_args[0] == "--":
        script_argv.extend(args.script_args[1:])
    else:
        script_argv.extend(args.script_args)

    old_argv = sys.argv[:]
    exit_code = 0
    try:
        sys.argv = script_argv
        runpy.run_path(str(args.script), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        exit_code = int(code) if isinstance(code, int) else 1
    finally:
        sys.argv = old_argv
        if should_trace_rank and log is not None and cpp_event is not None:
            try:
                cpp_event.resolve_async_runtime_observations(log)
                resolve_cupti_metadata = getattr(
                    cpp_event,
                    "resolve_cupti_activity_metadata_observations",
                    None,
                )
                if resolve_cupti_metadata is not None:
                    resolve_cupti_metadata(log)
                events = log.snapshot()
                output_path = _write_rank_trace(args.output_dir, events)
                marker_records = load_step_markers(args.output_dir / f"rank_{rank}.markers.jsonl")
                step_window = resolve_step_window_from_markers(
                    marker_records,
                    source=TraceSource.REAL,
                )
                if step_window is not None:
                    step_window = {
                        **step_window,
                        "source": "trace_markers",
                    }
                if step_window is None:
                    step_window = estimate_rank_trace_window(output_path, source=TraceSource.REAL)
                communicator_memberships, communicator_aliases = _recover_output_dir_communicators(
                    args.output_dir
                )
                _merge_capture_manifest(
                    args.output_dir,
                    world_size=world_size,
                    profiled_ranks=profiled_ranks,
                    profiled_rank_groups=profiled_rank_groups,
                    rank=rank,
                    fidelity_window=step_window,
                    host_machine_id=host_machine_id,
                    host_dispatch_queue_id=host_dispatch_queue_id,
                    communicator_memberships=communicator_memberships,
                    communicator_aliases=communicator_aliases,
                    route_metadata=route_metadata,
                )
                print(f"[capture_real] wrote {len(events)} events to {output_path}", file=sys.stderr)
            finally:
                cpp_event.reset_recorder()
                cpp_event.clear_async_runtime_observations()
                clear_cupti_metadata = getattr(
                    cpp_event,
                    "clear_cupti_activity_metadata_observations",
                    None,
                )
                if clear_cupti_metadata is not None:
                    clear_cupti_metadata()
                os.environ.pop("FLEXSIM_MAYA_MARKERS_PATH", None)
                os.environ.pop("FLEXSIM_CAPTURE_REAL_ENABLE_ASYNC_RUNTIME", None)
        else:
            print(
                f"[capture_real] rank {rank} executed without tracing; profiled_ranks={profiled_ranks}",
                file=sys.stderr,
            )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
