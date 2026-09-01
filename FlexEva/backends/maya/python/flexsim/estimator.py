"""
Data-Driven Timing Estimator for FlexSim.

Learns per-API-call durations from real GPU traces (timestamp deltas),
then applies those durations when simulating fake-cuda traces. The
trace-derived lookup remains the default path, but the estimator can also
host optional event-level timing providers for richer operator metadata.

Usage:
    # Fit from real traces
    est = Estimator.fit_from_traces("paper/traces/real/e1")

    # Estimate a single API call
    dur_us = est.estimate("cudaLaunchKernel", "kernel_launch")

    # Annotate a fake trace with estimated durations
    annotated = est.annotate_trace("paper/traces/fake/e1/rank_0.jsonl")

    # Get the full duration table
    est.summary()
"""

from __future__ import annotations

import base64
import concurrent.futures
import importlib.util
import json
import math
import os
import pickle
import hashlib
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]


def trace_dir_fingerprint(trace_dir: str | Path) -> dict[str, Any]:
    """Stable lightweight fingerprint for estimator provenance checks."""
    root = Path(trace_dir).resolve()
    h = hashlib.sha256()
    rank_count = 0
    total_size = 0
    manifest = root / "capture_manifest.json"
    if manifest.exists():
        data = manifest.read_bytes()
        h.update(b"manifest\0")
        h.update(data)
        total_size += len(data)
    for path in sorted(root.glob("rank_*.jsonl")):
        if ".markers" in path.name or path.name.endswith(".raw.jsonl"):
            continue
        stat = path.stat()
        rank_count += 1
        total_size += int(stat.st_size)
        h.update(path.name.encode("utf-8"))
        h.update(str(stat.st_size).encode("ascii"))
        with path.open("rb") as f:
            h.update(f.read(65536))
            if stat.st_size > 65536:
                f.seek(max(stat.st_size - 65536, 0))
                h.update(f.read(65536))
    return {
        "path": str(root),
        "rank_file_count": rank_count,
        "total_size_bytes": total_size,
        "sha256": h.hexdigest(),
    }
DEFAULT_GPU_ESTIMATOR_BUNDLE = (
    REPO_ROOT
    / "gpu_estimator"
    / "fake_driver_xgboost_provider_runtime_only_bundle"
)
_CAPTURE_MANIFEST = "capture_manifest.json"
_PAPER_VALID_FIT_WINDOW_SOURCES = frozenset(
    {"manifest", "trace_markers", "workload_heuristic"}
)
_RANK_TRACE_FILE_RE = re.compile(r"rank_(\d+)\.jsonl$")
_CUDA_LAUNCH_TRACE_LEARNED_CONTRACT_GUARD_ENV = (
    "MAYA_ENABLE_CUDALAUNCH_TRACE_LEARNED_CONTRACT_GUARD"
)
_GEMM_TRACE_LEARNED_FEATURE_COVERAGE_GUARD_ENV = (
    "MAYA_ENABLE_TRACE_LEARNED_GEMM_FEATURE_COVERAGE_GUARD"
)

_CANONICAL_API_ALIASES = {
    "cudaEventRecordWithFlags": "cudaEventRecord",
    "cudaEventCreate": "cudaEventCreateWithFlags",
    "cudaStreamCreateWithFlags": "cudaStreamCreate",
    "cudaStreamCreateWithPriority": "cudaStreamCreate",
    "ncclBcast": "ncclBroadcast",
}
_NCCL_P2P_MEMBER_APIS = {"ncclSend", "ncclRecv"}
_NCCL_P2P_GROUP_API = "ncclP2P"


def _nested_float_defaultdict() -> defaultdict:
    return defaultdict(float)


def _env_flag_enabled(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _split_contiguous_chunks(items: list[Any], chunk_count: int) -> list[list[Any]]:
    if chunk_count <= 1 or len(items) <= 1:
        return [list(items)]
    chunk_count = max(1, min(chunk_count, len(items)))
    chunk_size = math.ceil(len(items) / chunk_count)
    return [
        items[index:index + chunk_size]
        for index in range(0, len(items), chunk_size)
        if items[index:index + chunk_size]
    ]


def _rank_from_trace_path(path: Path) -> int | None:
    match = _RANK_TRACE_FILE_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _coerce_communicator_members(payload: object) -> tuple[int, ...] | None:
    if isinstance(payload, Mapping):
        payload = payload.get("members")
    if not isinstance(payload, (list, tuple)):
        return None
    members: list[int] = []
    for item in payload:
        try:
            members.append(int(item))
        except (TypeError, ValueError):
            return None
    return tuple(members) if members else None


def _communicator_context_from_manifest(
    trace_path: Path,
) -> tuple[dict[int, dict[str, str]], dict[str, tuple[int, ...]]]:
    """Load real-trace communicator topology used by group-level timing fits."""

    manifest_path = trace_path / "capture_manifest.json"
    if not manifest_path.exists():
        return {}, {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, {}
    if not isinstance(manifest, Mapping):
        return {}, {}

    memberships: dict[str, tuple[int, ...]] = {}
    raw_communicators = manifest.get("communicators")
    if isinstance(raw_communicators, Mapping):
        for raw_comm_id, raw_payload in raw_communicators.items():
            members = _coerce_communicator_members(raw_payload)
            if members is not None:
                memberships[str(raw_comm_id)] = members

    aliases_by_rank: dict[int, dict[str, str]] = {}
    raw_aliases = manifest.get("communicator_aliases")
    if isinstance(raw_aliases, Mapping):
        for raw_rank, raw_mapping in raw_aliases.items():
            try:
                rank = int(raw_rank)
            except (TypeError, ValueError):
                continue
            if not isinstance(raw_mapping, Mapping):
                continue
            rank_aliases = {
                str(local_comm_id): str(canonical_comm_id)
                for local_comm_id, canonical_comm_id in raw_mapping.items()
                if local_comm_id not in (None, "") and canonical_comm_id not in (None, "")
            }
            if rank_aliases:
                aliases_by_rank[rank] = rank_aliases

    return aliases_by_rank, memberships


def _apply_communicator_topology_to_payload(
    payload: dict[str, Any],
    *,
    communicator_aliases: Mapping[str, str] | None,
    communicator_memberships: Mapping[str, tuple[int, ...]] | None,
) -> dict[str, Any]:
    api = _normalize_api_name(payload.get("api", ""))
    typ = _normalize_low_level_op_type(api, str(payload.get("type", "other")))
    if typ != "nccl_collective":
        return payload

    local_comm_id = payload.get("comm_id")
    canonical_comm_id = None
    if local_comm_id not in (None, ""):
        local_comm_id_text = str(local_comm_id)
        canonical_comm_id = (
            str(communicator_aliases.get(local_comm_id_text))
            if communicator_aliases and local_comm_id_text in communicator_aliases
            else local_comm_id_text
        )
        payload["comm_id"] = canonical_comm_id
        payload["collective_communicator_id"] = canonical_comm_id

    members = (
        communicator_memberships.get(canonical_comm_id)
        if communicator_memberships is not None and canonical_comm_id is not None
        else None
    )
    if not members:
        return payload

    communicator_size = len(members)
    # For group-level collective runtime features, "world_size" means the
    # active communicator topology, not the process-global training world.
    payload["communicator_size"] = communicator_size
    payload["world_size"] = communicator_size
    if api in {"ncclSend", "ncclRecv"}:
        payload["participant_count"] = 2
    else:
        payload["participant_count"] = communicator_size
    return payload


def _fit_window_from_payload(payload: object) -> tuple[int, int, str] | None:
    if not isinstance(payload, Mapping):
        return None
    try:
        start_ts = int(float(payload.get("start_ts")))
        end_ts = int(float(payload.get("end_ts")))
    except (TypeError, ValueError):
        return None
    if end_ts < start_ts:
        return None
    source = str(payload.get("source") or "manifest").strip().lower()
    return start_ts, end_ts, source


def _fit_windows_from_manifest(
    trace_path: Path,
    *,
    trace_window: str,
) -> dict[int, tuple[int, int]]:
    normalized = str(trace_window or "auto").strip().lower()
    if normalized == "full":
        return {}
    if normalized not in {"auto", "step"}:
        raise ValueError(
            f"fit trace_window must be one of auto/full/step, got {trace_window!r}"
        )

    manifest_path = trace_path / _CAPTURE_MANIFEST
    if not manifest_path.exists():
        if normalized == "step":
            raise FileNotFoundError(
                f"step-scoped estimator fitting requires {_CAPTURE_MANIFEST} in {trace_path}"
            )
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if normalized == "step":
            raise ValueError(f"invalid capture manifest: {manifest_path}") from exc
        return {}
    if not isinstance(manifest, Mapping):
        if normalized == "step":
            raise ValueError(f"capture manifest must be a JSON object: {manifest_path}")
        return {}

    raw_windows = manifest.get("fidelity_windows")
    if not isinstance(raw_windows, Mapping) or not raw_windows:
        raw_windows = manifest.get("step_windows")
    if not isinstance(raw_windows, Mapping) or not raw_windows:
        if normalized == "step":
            raise ValueError(f"no step/fidelity windows found in {manifest_path}")
        return {}

    windows: dict[int, tuple[int, int]] = {}
    invalid_sources: dict[int, str] = {}
    for raw_rank, raw_window in raw_windows.items():
        try:
            rank = int(raw_rank)
        except (TypeError, ValueError):
            continue
        resolved = _fit_window_from_payload(raw_window)
        if resolved is None:
            continue
        start_ts, end_ts, source = resolved
        if source not in _PAPER_VALID_FIT_WINDOW_SOURCES:
            invalid_sources[rank] = source
            continue
        windows[rank] = (start_ts, end_ts)

    if not windows and normalized == "step":
        raise ValueError(
            "no paper-valid step windows found in "
            f"{manifest_path}; invalid sources={invalid_sources}"
        )
    return windows


class EventTimingProvider(Protocol):
    """Optional event-level provider layered ahead of trace statistics."""

    name: str

    def estimate_us(
        self,
        event: Mapping[str, Any],
        percentile: str = "p50",
    ) -> float | None:
        """Return an event duration in microseconds or None if unsupported."""


@dataclass(frozen=True)
class EstimatorDecision:
    """Resolved timing decision for one low-level event."""

    duration_us: float
    source: str
    calibrated: bool
    provider_name: str | None = None


@dataclass(frozen=True)
class ProviderLoadStatus:
    provider: EventTimingProvider | None
    error: str | None = None


def _normalize_api_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return _CANONICAL_API_ALIASES.get(text, text)


def _p2p_group_api_name(value: Any) -> str:
    api_name = _normalize_api_name(value)
    return _NCCL_P2P_GROUP_API if api_name in _NCCL_P2P_MEMBER_APIS else api_name


def _normalize_estimator_key(api: Any, typ: Any) -> tuple[str, str]:
    canonical_api = _normalize_api_name(api)
    canonical_typ = _normalize_low_level_op_type(canonical_api, str(typ or "other"))
    return canonical_api, canonical_typ


def _split_serialized_estimator_key(key_str: Any) -> tuple[str, str] | None:
    parts = str(key_str).split("::", 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _normalize_collective_group_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    pipe_index = text.find("|")
    hash_index = text.find("#")
    separator_index = -1
    if pipe_index >= 0 and hash_index >= 0:
        separator_index = min(pipe_index, hash_index)
    elif pipe_index >= 0:
        separator_index = pipe_index
    elif hash_index >= 0:
        separator_index = hash_index
    if separator_index < 0:
        return _p2p_group_api_name(text)
    prefix = _p2p_group_api_name(text[:separator_index])
    return prefix + text[separator_index:]


def _normalize_token(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _api_family(api_name: str) -> str:
    lowered = api_name.lower()
    if lowered.startswith("nccl"):
        return "nccl"
    if lowered.startswith("cublaslt"):
        return "cublaslt"
    if lowered.startswith("cublas"):
        return "cublas"
    if lowered.startswith("cuda"):
        return "cuda"
    if lowered.startswith("cu"):
        return "cuda_driver"
    return "other"


def _kernel_family_from_symbol(symbol: str) -> str:
    lowered = symbol.lower()
    if not lowered:
        return "kernel_other"
    if "flash" in lowered:
        return "flash_attention"
    if "gemm" in lowered or "xmma" in lowered or "matmul" in lowered:
        return "gemm_family"
    if "layer_norm" in lowered or "layernorm" in lowered:
        return "layer_norm"
    if "softmax" in lowered:
        return "softmax"
    if "dropout" in lowered:
        return "dropout"
    if "reduce" in lowered or "sum_and_scatter" in lowered:
        return "reduction"
    if "scan" in lowered or "radixsort" in lowered or "cubdevicescan" in lowered:
        return "scan_sort"
    if "catarraybatchedcopy" in lowered or "copy" in lowered or "memcpy" in lowered:
        return "tensor_copy"
    if "indexselect" in lowered or "index_" in lowered or "gather" in lowered:
        return "indexing"
    if "gelu" in lowered:
        return "activation"
    if "elementwise" in lowered:
        return "elementwise"
    if "fillfunctor" in lowered:
        return "fill"
    return "kernel_other"


def _canonicalize_kernel_signature(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    canonical = text
    canonical = re.sub(r"\{\s*lambda\([^)]*\)#\d+\s*\}", "{lambda#}", canonical)
    canonical = re.sub(r"function_traits<[^>]+>::result_type", "function_traits<sig>::result_type", canonical)
    for pattern in (
        r"\bc10::BFloat16\b",
        r"\bc10::Half\b",
        r"\bat::Half\b",
        r"\b__half\b",
        r"\bhalf\b",
        r"\bfloat\b",
        r"\bdouble\b",
    ):
        canonical = re.sub(pattern, "__dtype__", canonical)
    canonical = re.sub(r"\s+", " ", canonical).strip()
    return canonical


def event_operator_family(event: Mapping[str, Any]) -> str:
    api_name = _normalize_api_name(
        event.get("api") or event.get("api_name") or event.get("name")
    )
    typ = _normalize_low_level_op_type(api_name, str(event.get("type") or "other"))
    if api_name in _CUBLAS_COMPUTE_APIS:
        return "gemm_family"
    if api_name == "cudaLaunchKernel" or typ == "kernel_launch":
        kernel_name = str(event.get("kernel") or event.get("kernel_name") or "")
        return _kernel_family_from_symbol(kernel_name)
    if api_name.startswith("nccl"):
        collective = _normalize_token(
            event.get("collective")
            or event.get("kind")
            or event.get("name")
            or api_name
        )
        if "allreduce" in collective or api_name == "ncclAllReduce":
            return "nccl_allreduce"
        if "allgather" in collective or api_name == "ncclAllGather":
            return "nccl_allgather"
        if "alltoall" in collective or api_name in {"ncclAllToAll", "ncclAllToAllv"}:
            return "nccl_alltoall"
        if "reducescatter" in collective or api_name == "ncclReduceScatter":
            return "nccl_reduce_scatter"
        if collective in {"reduce", "ncclreduce"} or api_name == "ncclReduce":
            return "nccl_reduce"
        if "broadcast" in collective or api_name == "ncclBroadcast":
            return "nccl_broadcast"
        if api_name in {"ncclSend", "ncclRecv", "ncclP2P"}:
            return "nccl_p2p"
        return "nccl_other"
    if api_name in {"cudaMemcpy", "cudaMemcpyAsync"}:
        return "memcpy"
    if api_name in {"cudaMalloc", "cudaMallocAsync", "cudaFree", "cudaFreeAsync"}:
        return "memory_management"
    if typ == "stream_op":
        return "stream_sync"
    if typ == "context_op":
        return "context_op"
    if typ == "host_delay":
        return "host_delay"
    family = _api_family(api_name)
    if family != "other":
        return family
    return _normalize_token(api_name or typ or "other")


def event_operator_label(event: Mapping[str, Any]) -> str:
    api_name = _normalize_api_name(
        event.get("api") or event.get("api_name") or event.get("name")
    )
    if api_name == "cudaLaunchKernel" or str(event.get("type") or "") == "kernel_launch":
        kernel_name = str(event.get("kernel") or event.get("kernel_name") or "").strip()
        if kernel_name:
            return kernel_name
    return api_name or str(event.get("type") or "other")


def is_modeled_operator_event(event: Mapping[str, Any]) -> bool:
    api_name = _normalize_api_name(
        event.get("api") or event.get("api_name") or event.get("name")
    )
    typ = _normalize_low_level_op_type(api_name, str(event.get("type") or "other"))
    if api_name in _CUBLAS_COMPUTE_APIS:
        return True
    if api_name == "cudaLaunchKernel" or typ == "kernel_launch":
        return True
    if api_name in {
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
        return True
    if api_name in {"cudaMemcpy", "cudaMemcpyAsync"}:
        return True
    return False


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


def _normalize_low_level_op_type(api: str, op_type: str) -> str:
    api = _normalize_api_name(api)
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
        "ncclP2P",
    }:
        return "nccl_collective"
    if api in {"cudaMemcpy", "cudaMemcpyAsync"}:
        return "mem_copy"
    if api in {"cudaMalloc", "cudaMallocAsync", "cudaFree", "cudaFreeAsync"}:
        return "mem_alloc"
    if api.startswith("cudaStream") or api.startswith("cudaEvent"):
        return "stream_op"
    return op_type


def _control_plane_api_cap_us(api_name: str, typ: str) -> float | None:
    """
    Cap host-side trace targets for asynchronous/control-plane APIs.

    Real traces record API entry timestamps, not GPU completion. Long gaps after
    enqueue/control APIs should not be attributed wholesale to the API itself.
    """
    if typ in {"kernel_launch", "blas_compute", "nccl_collective", "mem_copy"}:
        return None
    if _trust_observed_wrapper_runtime(api_name, typ):
        return None
    if typ == "stream_op":
        return 100.0
    if typ == "context_op":
        return 50.0
    lowered = api_name.lower()
    if lowered.startswith("ncclget") or lowered.startswith("ncclcomm"):
        return 100.0
    if lowered.startswith("cuda") or lowered.startswith("cu"):
        return 100.0
    return None


def _normalize_observed_target_us(api_name: str, typ: str, delta_us: float) -> float:
    clamped = max(float(delta_us), 0.0)
    cap_us = _control_plane_api_cap_us(api_name, typ)
    if cap_us is not None:
        clamped = min(clamped, cap_us)
    if clamped > 1_000_000.0:
        clamped = 1_000_000.0
    return clamped


def _parse_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _raw_observed_wrapper_runtime_us(record: Mapping[str, Any]) -> float | None:
    observed = _parse_optional_float(record.get("host_duration_us"))
    if observed is not None:
        return max(float(observed), 0.0)
    start_ts = _parse_optional_float(record.get("ts"))
    end_ts = _parse_optional_float(record.get("end_ts"))
    if start_ts is None or end_ts is None:
        return None
    return max(float(end_ts) - float(start_ts), 0.0)


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

_DIRECT_RUNTIME_CONTRACT_VALUES = {
    "direct_runtime",
    "semantic_runtime",
    "async_runtime",
}

_EXPLICIT_DIRECT_RUNTIME_FIELDS = (
    "direct_runtime_us",
    "observed_runtime_us",
)


def _explicit_direct_runtime_observation_us(record: Mapping[str, Any]) -> float | None:
    for field in _EXPLICIT_DIRECT_RUNTIME_FIELDS:
        observed = _parse_optional_float(record.get(field))
        if observed is not None:
            return max(float(observed), 0.0)
    return None


def _has_wrapper_timing_field(record: Mapping[str, Any]) -> bool:
    return (
        _explicit_direct_runtime_observation_us(record) is not None
        or _raw_observed_wrapper_runtime_us(record) is not None
    )


def _wrapper_runtime_contract(record: Mapping[str, Any], *, api_name: str, typ: str) -> str:
    raw_contract = str(record.get("wrapper_runtime_contract") or "").strip().lower()
    if raw_contract:
        if raw_contract in _DIRECT_RUNTIME_CONTRACT_VALUES:
            return "direct_runtime"
        return raw_contract
    if _explicit_direct_runtime_observation_us(record) is not None:
        return "direct_runtime"
    if api_name in _OBSERVED_WRAPPER_RUNTIME_APIS:
        return "direct_runtime"
    if _is_async_modeled_device_work_event(api_name, typ):
        return "dispatch_only"
    return "missing"


def _trust_observed_wrapper_runtime(api_name: str, typ: str) -> bool:
    if api_name in _OBSERVED_WRAPPER_RUNTIME_APIS:
        return True
    return False


def _observed_wrapper_runtime_us(record: Mapping[str, Any], *, api_name: str, typ: str) -> float | None:
    explicit_direct_runtime_us = _explicit_direct_runtime_observation_us(record)
    if explicit_direct_runtime_us is not None:
        return explicit_direct_runtime_us
    contract = _wrapper_runtime_contract(record, api_name=api_name, typ=typ)
    if contract != "direct_runtime" and not _trust_observed_wrapper_runtime(api_name, typ):
        return None
    if contract != "direct_runtime" and api_name not in _OBSERVED_WRAPPER_RUNTIME_APIS:
        return None
    return _raw_observed_wrapper_runtime_us(record)


def _resolved_observed_target_us(
    api_name: str,
    typ: str,
    recorded_host_us: float | None,
    fallback_us: float,
) -> float:
    """
    Resolve the best available training target for one traced event.

    Async device work often records near-zero host-side wrapper time even when
    the downstream GPU work is substantial. Treat non-positive host durations as
    unusable for modeled device work so we can fall back to gap/wait attribution.
    """
    fallback = max(float(fallback_us), 0.0)
    if recorded_host_us is None:
        return fallback
    observed = max(float(recorded_host_us), 0.0)
    if _is_device_work_event(api_name, typ) and observed <= 0.0:
        return fallback
    return observed


def _distribution_weight(api_name: str, typ: str) -> float:
    if typ == "blas_compute":
        return 1.5
    if typ == "nccl_collective":
        return 1.5
    if typ == "mem_copy":
        return 1.0
    if typ == "mem_alloc":
        return 0.8
    if typ == "kernel_launch":
        return 0.2
    if typ == "stream_op":
        return 0.1
    if typ == "context_op":
        return 0.05
    if _api_family(api_name) in {"cuda", "cuda_driver"}:
        return 0.05
    return 0.25


def _is_blocking_sync_api(api_name: str) -> bool:
    return api_name in {
        "cudaStreamSynchronize",
        "cudaEventSynchronize",
        "cudaDeviceSynchronize",
    }


def _normalize_runtime_stream_id(value: Any) -> str:
    if value in (None, "", "0", "0x0"):
        return "__default_stream__"
    return str(value)


def _stream_id_from_payload(
    api_name: str,
    payload: Mapping[str, Any],
    handle_streams: Mapping[str, str],
) -> str:
    stream_id = payload.get("stream_id")
    if stream_id not in (None, "", "0", "0x0"):
        return _normalize_runtime_stream_id(stream_id)
    handle_id = payload.get("handle_id")
    if handle_id not in (None, "", "0", "0x0"):
        return handle_streams.get(str(handle_id), "__default_stream__")
    if api_name == "cudaLaunchKernel":
        return "__default_stream__"
    return "__default_stream__"


def _is_device_work_event(api_name: str, typ: str) -> bool:
    if api_name in {
        "cublasCreate_v2",
        "cublasDestroy_v2",
        "cublasSetMathMode",
        "cublasSetStream_v2",
        "cublasSetWorkspace_v2",
    }:
        return False
    return typ in {"kernel_launch", "blas_compute", "nccl_collective", "mem_copy"}


def _is_async_modeled_device_work_event(api_name: str, typ: str) -> bool:
    if typ in {"kernel_launch", "blas_compute", "nccl_collective"}:
        return True
    return api_name == "cudaMemcpyAsync"


def _observed_wrapper_log2_bucket(runtime_us: float | None) -> int:
    if runtime_us is None or runtime_us <= 0.0:
        return 0
    return min(20, int(math.log2(max(float(runtime_us), 1.0))))


def _dtype_elem_bytes(payload: Mapping[str, Any]) -> int:
    for key in ("dtype_code", "datatype", "dtype", "Atype", "Btype", "Ctype", "compute_type", "computeType"):
        value = payload.get(key)
        if value in (None, ""):
            continue
        normalized = _normalize_cublas_dtype_code(value)
        if normalized in {0, 2}:
            return 2
        if normalized == 1:
            return 4
        normalized_nccl = _normalize_nccl_dtype_code(value)
        if normalized_nccl in {0, 1}:
            return 2
        if normalized_nccl == 2:
            return 4
    return 2


def _device_work_weight(
    api_name: str,
    typ: str,
    payload: Mapping[str, Any],
) -> float:
    if typ == "blas_compute":
        m = max(_safe_int(payload.get("m"), 1), 1)
        n = max(_safe_int(payload.get("n"), 1), 1)
        k = max(_safe_int(payload.get("k"), 1), 1)
        batch_count = max(
            _safe_int(payload.get("batch_count"), _safe_int(payload.get("batchCount"), 1)),
            1,
        )
        return float(max(m * n * k * batch_count, 1))
    if typ == "nccl_collective":
        numel = max(
            _safe_int(payload.get("numel"), _safe_int(payload.get("count"), 1)),
            1,
        )
        participant_extent = max(
            _safe_int(
                payload.get(
                    "participant_count",
                    _safe_int(
                        payload.get("communicator_size"),
                        _safe_int(payload.get("world_size"), 1),
                    ),
                ),
                1,
            ),
            1,
        )
        return float(max(numel * _dtype_elem_bytes(payload) * participant_extent, 1))
    if typ == "mem_copy":
        size_bytes = max(
            _safe_int(
                payload.get("bytes"),
                _safe_int(payload.get("count"), _safe_int(payload.get("size"), 1)),
            ),
            1,
        )
        return float(size_bytes)
    if api_name == "cudaLaunchKernel" or typ == "kernel_launch":
        return 1.0
    return 1.0


def _distribute_wait_budget(
    entries: list[dict[str, Any]],
    active_pending_ids: set[int],
    candidate_ids: list[int] | tuple[int, ...],
    wait_budget_us: float,
) -> None:
    if wait_budget_us <= 0.0:
        return
    filtered = [event_id for event_id in candidate_ids if event_id in active_pending_ids]
    if not filtered:
        return
    total_weight = sum(float(entries[event_id]["device_weight"]) for event_id in filtered)
    if total_weight <= 0.0:
        total_weight = float(len(filtered))
        for event_id in filtered:
            entries[event_id]["device_weight"] = 1.0
    for event_id in filtered:
        weight = float(entries[event_id]["device_weight"])
        entries[event_id]["target_us"] += (wait_budget_us * weight) / total_weight


def _remove_pending_ids(
    pending_by_stream: dict[str, list[int]],
    active_pending_ids: set[int],
    event_ids: list[int] | tuple[int, ...],
) -> None:
    to_remove = {event_id for event_id in event_ids if event_id in active_pending_ids}
    if not to_remove:
        return
    for stream_id, pending_ids in list(pending_by_stream.items()):
        remaining = [event_id for event_id in pending_ids if event_id not in to_remove]
        if remaining:
            pending_by_stream[stream_id] = remaining
        else:
            pending_by_stream.pop(stream_id, None)
    active_pending_ids.difference_update(to_remove)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


_FEATURE_INT_KEYS = {
    "m",
    "n",
    "k",
    "lda",
    "ldb",
    "ldc",
    "stride_a",
    "stride_b",
    "stride_c",
    "algorithm",
    "grid_x",
    "grid_y",
    "grid_z",
    "block_x",
    "block_y",
    "block_z",
    "shared_mem",
    "batch_count",
    "transa",
    "transb",
    "world_size",
    "communicator_size",
    "participant_count",
    "numel",
    "count",
    "bytes",
    "size",
    "dtype_code",
    "datatype",
    "Atype",
    "Btype",
    "Ctype",
    "compute_type",
    "computeType",
    "collective_sequence_number",
    "collective_root",
    "has_wrapper_timing_field",
    "has_positive_observed_wrapper_runtime",
    "observed_wrapper_log2_bucket",
}


def _canonicalize_gemm_feature_aliases(payload: dict[str, Any], api_name: str) -> None:
    if _api_family(api_name) not in {"cublas", "cublaslt"}:
        return
    if "computeType" in payload and "compute_type" not in payload:
        payload["compute_type"] = payload["computeType"]
    if "batchCount" in payload and "batch_count" not in payload:
        payload["batch_count"] = payload["batchCount"]
    for source_key, canonical_key in (
        ("strideA", "stride_a"),
        ("strideB", "stride_b"),
        ("strideC", "stride_c"),
    ):
        if source_key in payload and canonical_key not in payload:
            payload[canonical_key] = payload[source_key]
    if "algo" in payload and "algorithm" not in payload:
        payload["algorithm"] = payload["algo"]


def build_learned_timing_features(event: Mapping[str, Any]) -> dict[str, Any]:
    """
    Build a low-level feature map for learned timing providers.

    This intentionally stays below the semantic-recovery layer. It uses only
    API-local and execution-context fields that are present in the raw/collated
    low-level events or supplied directly by a runtime wrapper.
    """
    api_name = _normalize_api_name(
        event.get("api") or event.get("api_name") or event.get("name")
    )
    event_payload = dict(event)
    _canonicalize_gemm_feature_aliases(event_payload, api_name)
    op_type = _normalize_low_level_op_type(api_name, str(event.get("type") or "other"))
    module = str(event_payload.get("module") or event_payload.get("mod") or "unknown")
    prev_api = _normalize_api_name(event_payload.get("prev_api") or "")
    ordinal = max(_safe_int(event_payload.get("ordinal"), 0), 0)
    world_size = max(_safe_int(event_payload.get("world_size"), 1), 1)
    rank = max(_safe_int(event_payload.get("rank"), 0), 0)
    thread_id = max(_safe_int(event_payload.get("tid"), 0), 0)
    pid = max(_safe_int(event_payload.get("pid"), 0), 0)

    features: dict[str, Any] = {
        "api": api_name,
        "type": op_type,
        "module": module,
        "api_family": _api_family(api_name),
        "module_family": module.split(".", 1)[0],
        "rank": rank,
        "world_size": world_size,
        "ordinal_bucket": min(12, int(math.log2(ordinal + 1))) if ordinal > 0 else 0,
        "thread_mod_8": thread_id % 8,
        "pid_mod_8": pid % 8,
        "is_cross_rank": 1 if world_size > 1 else 0,
        "has_prev_api": 1 if prev_api else 0,
        "has_collective_group": 1 if event_payload.get("collective_group_id") else 0,
    }
    if prev_api:
        features["prev_api"] = prev_api
        features["prev_api_family"] = _api_family(prev_api)

    for key in (
        "m",
        "n",
        "k",
        "lda",
        "ldb",
        "ldc",
        "kernel_id",
        "kernel",
        "grid_x",
        "grid_y",
        "grid_z",
        "block_x",
        "block_y",
        "block_z",
        "shared_mem",
        "batch_count",
        "world_size",
        "numel",
        "count",
        "bytes",
        "size",
        "dtype",
        "dtype_code",
        "datatype",
        "collective",
        "collective_api",
        "collective_match_basis",
        "collective_communicator_id",
        "collective_sequence_number",
        "collective_root",
        "communicator_size",
        "participant_count",
        "has_wrapper_timing_field",
        "has_positive_observed_wrapper_runtime",
        "observed_wrapper_us",
        "observed_wrapper_log2_bucket",
        "reduction",
        "op",
        "kind",
        "transa",
        "transb",
        "Atype",
        "Btype",
        "Ctype",
        "compute_type",
        "computeType",
        "stride_a",
        "stride_b",
        "stride_c",
        "algorithm",
    ):
        value = event_payload.get(key)
        if value is not None:
            if key in _FEATURE_INT_KEYS:
                features[key] = _safe_int(value)
            elif key == "kernel":
                features[key] = _canonicalize_kernel_signature(value)
            elif key == "collective_api":
                features[key] = _normalize_api_name(value)
            else:
                features[key] = value

    # Preserve direct wrapper observations symmetrically across fit and
    # inference. Fit-time samples may already materialize these fields
    # explicitly, but inference paths often only carry raw wrapper timing
    # fields such as `host_duration_us` or `end_ts`.
    api_name = _normalize_api_name(
        event.get("api") or event.get("api_name") or event.get("name")
    )
    typ = _normalize_low_level_op_type(api_name, str(event.get("type") or "other"))
    has_wrapper_timing_field = _has_wrapper_timing_field(event)
    observed_wrapper_us = _observed_wrapper_runtime_us(
        event,
        api_name=api_name,
        typ=typ,
    )
    wrapper_runtime_contract = _wrapper_runtime_contract(
        event,
        api_name=api_name,
        typ=typ,
    )
    if "has_wrapper_timing_field" not in features:
        features["has_wrapper_timing_field"] = 1 if has_wrapper_timing_field else 0
    if "has_positive_observed_wrapper_runtime" not in features:
        features["has_positive_observed_wrapper_runtime"] = (
            1 if observed_wrapper_us is not None and observed_wrapper_us > 0.0 else 0
        )
    if "wrapper_runtime_contract" not in features and wrapper_runtime_contract != "missing":
        features["wrapper_runtime_contract"] = wrapper_runtime_contract
    if observed_wrapper_us is not None and "observed_wrapper_us" not in features:
        features["observed_wrapper_us"] = float(observed_wrapper_us)
    if observed_wrapper_us is not None and "observed_wrapper_log2_bucket" not in features:
        features["observed_wrapper_log2_bucket"] = _observed_wrapper_log2_bucket(
            observed_wrapper_us
        )
    return features


_COLLECTIVE_GROUP_TIMING_FEATURE_KEYS = (
    "api",
    "type",
    "api_family",
    "world_size",
    "collective",
    "collective_api",
    "collective_match_basis",
    "collective_communicator_id",
    "communicator_size",
    "participant_count",
    "collective_root",
    "numel",
    "count",
    "bytes",
    "size",
    "dtype",
    "dtype_code",
    "datatype",
    "reduction",
    "op",
    "kind",
)


def _normalize_p2p_collective_group_payload(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(event)
    api_name = _normalize_api_name(
        payload.get("api") or payload.get("api_name") or payload.get("name")
    )
    collective_api = _normalize_api_name(payload.get("collective_api"))
    collective_name = _normalize_token(payload.get("collective"))
    is_p2p_group = (
        api_name in _NCCL_P2P_MEMBER_APIS
        or api_name == _NCCL_P2P_GROUP_API
        or collective_api in _NCCL_P2P_MEMBER_APIS
        or collective_api == _NCCL_P2P_GROUP_API
        or collective_name in {"send", "recv", "p2p", "ncclsend", "ncclrecv", "ncclp2p"}
    )
    if not is_p2p_group:
        return payload

    if payload.get("member_api") in (None, ""):
        if api_name in _NCCL_P2P_MEMBER_APIS:
            payload["member_api"] = api_name
        elif collective_api in _NCCL_P2P_MEMBER_APIS:
            payload["member_api"] = collective_api
    if payload.get("member_collective") in (None, "") and collective_name in {"send", "recv"}:
        payload["member_collective"] = collective_name
    original_world_size = payload.get("world_size")
    if original_world_size not in (None, "") and payload.get("trace_world_size") in (None, ""):
        if _safe_int(original_world_size, 0) != 2:
            payload["trace_world_size"] = original_world_size
    payload["api"] = _NCCL_P2P_GROUP_API
    payload["type"] = "nccl_collective"
    payload["collective"] = "p2p"
    payload["collective_api"] = _NCCL_P2P_GROUP_API
    payload["world_size"] = 2
    payload["communicator_size"] = 2
    payload["participant_count"] = 2
    return payload


def build_collective_group_timing_features(event: Mapping[str, Any]) -> dict[str, Any]:
    """
    Build Maya-style group-level collective runtime features.

    The paper models collective on-wire duration as a group-level black-box
    prediction after all participants join. That predictor should be driven by
    operation shape and topology inputs, not by trace bookkeeping such as rank,
    pid/tid, ordinal buckets, or collective sequence number.
    """
    event = _normalize_p2p_collective_group_payload(canonicalize_gpu_estimator_event(event))
    base = build_learned_timing_features(event)
    features = {
        key: base[key]
        for key in _COLLECTIVE_GROUP_TIMING_FEATURE_KEYS
        if key in base and base[key] not in (None, "")
    }
    if "count" in features and "numel" not in features:
        features["numel"] = features["count"]
    if "op" in features and "reduction" not in features:
        features["reduction"] = _normalize_nccl_reduction(features["op"])
    if "datatype" in features and "dtype_code" not in features:
        features["dtype_code"] = _normalize_nccl_dtype_code(features["datatype"])
    return features


def _learned_provider_features(event: Mapping[str, Any]) -> dict[str, Any]:
    event = canonicalize_gpu_estimator_event(event)
    api_name = _normalize_api_name(
        event.get("api") or event.get("api_name") or event.get("name")
    )
    op_type = _normalize_low_level_op_type(api_name, str(event.get("type") or "other"))
    if op_type == "nccl_collective":
        return build_collective_group_timing_features(event)
    return build_learned_timing_features(event)


_CUDA_LAUNCH_MATERIAL_METADATA_KEYS = frozenset(
    {
        "numel",
        "count",
        "bytes",
        "size",
        "shape",
        "dtype",
        "dtype_code",
        "datatype",
        "tensor",
        "tensor_shape",
        "input_shape",
        "output_shape",
    }
)

_TRACE_LEARNED_GEMM_MATERIAL_FEATURE_KEYS = frozenset(
    {
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
        "transa",
        "transb",
        "Atype",
        "Btype",
        "Ctype",
        "compute_type",
        "dtype_code",
        "algorithm",
    }
)


def _trace_learned_cuda_launch_support_key(features: Mapping[str, Any]) -> str:
    return _signature_key_from_features(
        {
            key: features[key]
            for key in (
                "api",
                "type",
                "kernel",
                "grid_x",
                "grid_y",
                "grid_z",
                "block_x",
                "block_y",
                "block_z",
                "shared_mem",
            )
            if key in features and features[key] not in (None, "")
        }
    )


def _trace_learned_cuda_launch_has_material_metadata(
    features: Mapping[str, Any],
) -> bool:
    return any(
        features.get(key) not in (None, "")
        for key in _CUDA_LAUNCH_MATERIAL_METADATA_KEYS
    )


def _trace_learned_cuda_launch_is_eligible(
    features: Mapping[str, Any],
    support_keys: set[str],
) -> bool:
    if not _env_flag_enabled(_CUDA_LAUNCH_TRACE_LEARNED_CONTRACT_GUARD_ENV):
        return True
    if features.get("api") != "cudaLaunchKernel":
        return True
    if _trace_learned_cuda_launch_has_material_metadata(features):
        return True
    return _trace_learned_cuda_launch_support_key(features) in support_keys


def _trace_learned_cuda_launch_support_keys(
    samples: Iterable[Mapping[str, Any]],
) -> set[str]:
    keys: set[str] = set()
    for sample in samples:
        features = _learned_provider_features(sample)
        if features.get("api") != "cudaLaunchKernel":
            continue
        if _trace_learned_cuda_launch_has_material_metadata(features):
            continue
        key = _trace_learned_cuda_launch_support_key(features)
        if key:
            keys.add(key)
    return keys


_COLLECTIVE_NAME_TO_NCCL_API = {
    "allreduce": "ncclAllReduce",
    "allgather": "ncclAllGather",
    "alltoall": "ncclAllToAll",
    "alltoallv": "ncclAllToAllv",
    "broadcast": "ncclBroadcast",
    "bcast": "ncclBroadcast",
    "reduce": "ncclReduce",
    "reducescatter": "ncclReduceScatter",
    "reduce_scatter": "ncclReduceScatter",
    "p2p": _NCCL_P2P_GROUP_API,
    "send": _NCCL_P2P_GROUP_API,
    "recv": _NCCL_P2P_GROUP_API,
}


def _trace_learned_nccl_collective_support_key(
    features: Mapping[str, Any],
) -> str:
    api_name = _p2p_group_api_name(features.get("collective_api") or features.get("api"))
    if api_name:
        typ = _normalize_low_level_op_type(api_name, str(features.get("type") or "other"))
        if typ == "nccl_collective":
            return api_name
    collective_name = _normalize_token(features.get("collective"))
    return _COLLECTIVE_NAME_TO_NCCL_API.get(collective_name, "")


def _trace_learned_nccl_collective_is_eligible(
    features: Mapping[str, Any],
    support_keys: set[str],
) -> bool:
    if features.get("type") != "nccl_collective":
        return True
    key = _trace_learned_nccl_collective_support_key(features)
    if not key:
        return False
    return key in support_keys


def _trace_learned_nccl_collective_support_keys(
    samples: Iterable[Mapping[str, Any]],
) -> set[str]:
    keys: set[str] = set()
    for sample in samples:
        features = _learned_provider_features(sample)
        if features.get("type") != "nccl_collective":
            continue
        key = _trace_learned_nccl_collective_support_key(features)
        if key:
            keys.add(key)
    return keys


def _trace_learned_vectorizer_feature_names(vectorizer: Any) -> list[str]:
    for getter_name in ("get_feature_names_out", "get_feature_names"):
        getter = getattr(vectorizer, getter_name, None)
        if callable(getter):
            try:
                return [str(name) for name in getter()]
            except Exception:
                pass
    names = getattr(vectorizer, "feature_names_", None)
    if names is None:
        return []
    try:
        return [str(name) for name in names]
    except Exception:
        return []


def _trace_learned_feature_is_supported(
    feature_names: set[str],
    key: str,
    value: Any,
) -> bool:
    if key in feature_names:
        return True
    return f"{key}={value}" in feature_names


def _trace_learned_gemm_missing_material_feature_columns(
    features: Mapping[str, Any],
    feature_names: set[str],
) -> tuple[str, ...]:
    api_name = _normalize_api_name(features.get("api") or "")
    if api_name not in _CUBLAS_COMPUTE_APIS:
        return ()
    missing: list[str] = []
    for key in sorted(_TRACE_LEARNED_GEMM_MATERIAL_FEATURE_KEYS):
        value = features.get(key)
        if value in (None, ""):
            continue
        if not _trace_learned_feature_is_supported(feature_names, key, value):
            missing.append(key)
    return tuple(missing)


def _trace_learned_gemm_material_features_are_eligible(
    features: Mapping[str, Any],
    feature_names: set[str],
) -> bool:
    if not _env_flag_enabled(_GEMM_TRACE_LEARNED_FEATURE_COVERAGE_GUARD_ENV):
        return True
    return not _trace_learned_gemm_missing_material_feature_columns(
        features,
        feature_names,
    )


def _trace_learned_nccl_collective_support_keys_from_vectorizer(
    vectorizer: Any,
) -> set[str]:
    keys: set[str] = set()
    for feature_name in _trace_learned_vectorizer_feature_names(vectorizer):
        if "=" not in feature_name:
            continue
        field_name, raw_value = feature_name.split("=", 1)
        if field_name in {"api", "collective_api", "member_api"}:
            key = _trace_learned_nccl_collective_support_key(
                {"api": raw_value, "type": "nccl_collective"}
            )
        elif field_name == "collective":
            key = _COLLECTIVE_NAME_TO_NCCL_API.get(_normalize_token(raw_value), "")
        else:
            key = ""
        if key:
            keys.add(key)
    return keys


_PROVIDER_COVERAGE_CACHE_IGNORED_KEYS = frozenset(
    {
        "rank",
        "ordinal_bucket",
        "thread_mod_8",
        "pid_mod_8",
        "prev_api",
        "prev_api_family",
        "has_prev_api",
        "collective_sequence_number",
        "collective_communicator_id",
        "has_wrapper_timing_field",
        "has_positive_observed_wrapper_runtime",
        "observed_wrapper_us",
        "observed_wrapper_log2_bucket",
    }
)


def _provider_coverage_cache_key(event: Mapping[str, Any]) -> str:
    # Coverage asks whether a provider can model an operator shape, not whether
    # a particular rank/timestamp occurrence differs.  Keeping this key stable
    # avoids re-querying external providers for millions of identical shapes.
    payload = {
        str(key): value
        for key, value in event.items()
        if str(key) not in _PROVIDER_COVERAGE_CACHE_IGNORED_KEYS
        and not str(key).startswith("_")
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _normalize_nccl_dtype_code(value: Any) -> Any:
    token = str(value).strip().lower()
    if token in {"6", "ncclfloat16", "ncclhalf"}:
        return 0
    if token in {"9", "ncclbfloat16"}:
        return 1
    if token in {"7", "ncclfloat32"}:
        return 2
    return value


def _normalize_nccl_reduction(value: Any) -> Any:
    token = str(value).strip().lower()
    if token in {"0", "ncclsum", "sum"}:
        return "sum"
    return value


def _normalize_cublas_dtype_code(value: Any) -> Any:
    if isinstance(value, (int, float)) and int(value) in {0, 1, 2}:
        return int(value)
    token = str(value).strip().lower()
    if token in {"2", "fp16", "half", "float16", "f16", "16f", "cuda_r_16f"}:
        return 0
    if token in {
        "0",
        "1",
        "68",
        "69",
        "74",
        "77",
        "fp32",
        "float",
        "float32",
        "f32",
        "32f",
        "cuda_r_32f",
        "cublas_compute_32f",
        "cublas_compute_32f_pedantic",
        "cublas_compute_32f_fast_16f",
        "cublas_compute_32f_fast_tf32",
    }:
        return 1
    if token in {
        "14",
        "75",
        "bf16",
        "bfloat16",
        "16bf",
        "cuda_r_16bf",
        "cublas_compute_32f_fast_16bf",
    }:
        return 2
    return value


def canonicalize_gpu_estimator_event(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(event)
    api_name = _normalize_api_name(
        payload.get("api") or payload.get("api_name") or payload.get("name")
    )
    if api_name:
        payload["api"] = api_name
        payload["type"] = _normalize_low_level_op_type(
            api_name,
            str(payload.get("type") or "other"),
        )
    prev_api = payload.get("prev_api")
    if prev_api not in (None, ""):
        payload["prev_api"] = _normalize_api_name(prev_api)
    collective_api = payload.get("collective_api")
    if collective_api not in (None, ""):
        payload["collective_api"] = _normalize_api_name(collective_api)
    if _api_family(api_name) in {"cublas", "cublaslt"}:
        _canonicalize_gemm_feature_aliases(payload, api_name)
        cublas_dtype = None
        for key in ("Atype", "Btype", "Ctype", "dtype_code", "dtype", "compute_type", "computeType"):
            if key in payload:
                normalized = _normalize_cublas_dtype_code(payload[key])
                payload[key] = normalized
                if cublas_dtype is None and normalized in {0, 1, 2}:
                    cublas_dtype = normalized
        if cublas_dtype is not None:
            payload["dtype_code"] = cublas_dtype
    nccl_collective_defaults = {
        "ncclAllReduce": ("allreduce", "ncclAllReduce"),
        "ncclAllGather": ("allgather", "ncclAllGather"),
        "ncclAllToAll": ("alltoall", "ncclAllToAll"),
        "ncclAllToAllv": ("alltoallv", "ncclAllToAllv"),
        "ncclBroadcast": ("broadcast", "ncclBroadcast"),
        "ncclReduce": ("reduce", "ncclReduce"),
        "ncclReduceScatter": ("reducescatter", "ncclReduceScatter"),
        "ncclSend": ("send", "ncclP2P"),
        "ncclRecv": ("recv", "ncclP2P"),
    }
    if api_name in nccl_collective_defaults:
        if "count" in payload and "numel" not in payload:
            payload["numel"] = payload["count"]
        if "dtype_code" in payload:
            payload["dtype_code"] = _normalize_nccl_dtype_code(payload["dtype_code"])
        elif "datatype" in payload:
            payload["dtype_code"] = _normalize_nccl_dtype_code(payload["datatype"])
        if "reduction" in payload:
            payload["reduction"] = _normalize_nccl_reduction(payload["reduction"])
        elif "op" in payload:
            payload["reduction"] = _normalize_nccl_reduction(payload["op"])
        collective_name, collective_api = nccl_collective_defaults[api_name]
        if "collective" not in payload:
            payload["collective"] = collective_name
        if "collective_api" not in payload:
            payload["collective_api"] = collective_api
        if payload.get("collective_communicator_id") in (None, "") and payload.get("comm_id") not in (None, ""):
            payload["collective_communicator_id"] = payload["comm_id"]
        if payload.get("collective_sequence_number") in (None, "") and payload.get("call_idx") not in (None, ""):
            payload["collective_sequence_number"] = payload["call_idx"]
        communicator_size = payload.get("communicator_size")
        if communicator_size in (None, ""):
            communicator_size = (
                payload.get("nranks")
                or payload.get("num_ranks")
                or payload.get("nRanks")
                or payload.get("world_size")
            )
            if communicator_size not in (None, ""):
                payload["communicator_size"] = communicator_size
        if communicator_size not in (None, ""):
            original_world_size = payload.get("world_size")
            if original_world_size not in (None, "") and payload.get("trace_world_size") in (None, ""):
                if _safe_int(original_world_size, 0) != _safe_int(communicator_size, 0):
                    payload["trace_world_size"] = original_world_size
            payload["world_size"] = communicator_size
        if "participant_count" not in payload:
            if api_name in {"ncclSend", "ncclRecv"}:
                payload["participant_count"] = 2
            elif communicator_size not in (None, ""):
                payload["participant_count"] = communicator_size
        if "collective_root" not in payload and payload.get("root") not in (None, ""):
            payload["collective_root"] = payload["root"]
    return payload


class GPUXGBoostTimingProvider:
    """Adapter for the external gpu_estimator runtime bundle."""

    name = "gpu_estimator_xgboost"
    supports_collective_group_timing = True

    def __init__(self, runtime_provider: Any, bundle_dir: str | Path):
        self._runtime_provider = runtime_provider
        self.bundle_dir = Path(bundle_dir)

    def __getstate__(self) -> dict[str, str]:
        return {"bundle_dir": str(self.bundle_dir)}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        bundle_dir = Path(state["bundle_dir"])
        status = probe_gpu_estimator_provider(bundle_dir)
        if not isinstance(status.provider, GPUXGBoostTimingProvider):
            raise RuntimeError(
                f"failed to reload gpu_estimator runtime bundle: {status.error}"
            )
        self.bundle_dir = bundle_dir
        self._runtime_provider = status.provider._runtime_provider

    def estimate_us(
        self,
        event: Mapping[str, Any],
        percentile: str = "p50",
    ) -> float | None:
        del percentile  # external runtime provider predicts a single value
        event = canonicalize_gpu_estimator_event(event)
        api_name = event.get("api") or event.get("api_name") or event.get("name")
        try:
            if not self._runtime_provider.supports(api_name=api_name, event=event):
                return None
            predicted_ms = float(
                self._runtime_provider.predict_ms(api_name=api_name, event=event)
            )
        except Exception:
            return None
        if not math.isfinite(predicted_ms) or predicted_ms < 0:
            return None
        return predicted_ms * 1000.0

    def estimate_many_us(
        self,
        events: list[Mapping[str, Any]],
        percentile: str = "p50",
    ) -> list[float | None]:
        return [self.estimate_us(event, percentile=percentile) for event in events]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": "gpu_estimator_xgboost",
            "bundle_dir": str(self.bundle_dir),
        }


def _collective_group_representative_sort_key(sample: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        max(_safe_int(sample.get("ts"), 0), 0),
        max(_safe_int(sample.get("rank"), 0), 0),
        max(_safe_int(sample.get("ordinal"), 0), 0),
    )


def _strip_training_metadata(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in sample.items()
        if not str(key).startswith("_") and str(key) != "collective_group_id"
    }


def _collective_group_training_id(sample: Mapping[str, Any]) -> str:
    explicit_group_id = _normalize_collective_group_id(
        sample.get("_collective_group_id") or sample.get("collective_group_id")
    )
    if explicit_group_id:
        return explicit_group_id
    communicator_id = (
        sample.get("_collective_communicator_id")
        or sample.get("collective_communicator_id")
        or sample.get("comm_id")
    )
    sequence_number = (
        sample.get("_collective_sequence_number")
        or sample.get("collective_sequence_number")
        or sample.get("call_idx")
    )
    if communicator_id in (None, "") or sequence_number in (None, ""):
        return ""
    collective_api = _p2p_group_api_name(
        sample.get("collective_api") or sample.get("api") or "nccl_collective"
    )
    return f"{collective_api}|{communicator_id}|{sequence_number}"


def _collapse_collective_group_training_pairs(
    samples: list[dict[str, Any]],
    targets_us: list[float],
) -> list[tuple[dict[str, Any], float]]:
    """
    Collapse collective-group training targets to the group completion target.

    Replay finishes a collective when the whole group completes, so training
    group-capable providers on per-member targets creates a contract mismatch.
    For NCCL events carrying ``collective_group_id``, keep one representative
    sample per group and lift its target to the group's max target.
    """
    collapsed_items: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, Any]] = {}

    for index, (sample, target_us) in enumerate(zip(samples, targets_us)):
        api_name = _normalize_api_name(sample.get("api") or sample.get("api_name") or sample.get("name"))
        typ = _normalize_low_level_op_type(api_name, str(sample.get("type") or "other"))
        collective_group_id = _collective_group_training_id(sample)
        normalized_target_us = max(float(target_us), 0.0)
        if typ != "nccl_collective" or collective_group_id in (None, ""):
            collapsed_items.append(
                {
                    "sample": _strip_training_metadata(sample),
                    "target_us": normalized_target_us,
                    "first_index": index,
                }
            )
            continue

        group_id = str(collective_group_id)
        representative_key = _collective_group_representative_sort_key(sample)
        existing = grouped.get(group_id)
        if existing is None:
            grouped[group_id] = {
                "sample": _strip_training_metadata(sample),
                "target_us": normalized_target_us,
                "first_index": index,
                "representative_key": representative_key,
            }
            continue
        existing["target_us"] = max(float(existing["target_us"]), normalized_target_us)
        if representative_key < existing["representative_key"]:
            existing["sample"] = _strip_training_metadata(sample)
            existing["representative_key"] = representative_key

    collapsed_items.extend(grouped.values())
    return [
        (payload["sample"], float(payload["target_us"]))
        for payload in sorted(collapsed_items, key=lambda item: int(item["first_index"]))
    ]


class TraceLearnedTimingProvider:
    """
    Learned low-level timing model fit from real traces.

    The training targets still come from the observed trace timing signal, but
    the regressor can generalize across low-level context fields beyond a flat
    (api, type) table.
    """

    name = "trace_learned_sklearn"
    supports_collective_group_timing = True

    def __init__(
        self,
        vectorizer: Any,
        model: Any,
        cuda_launch_support_keys: set[str] | None = None,
        nccl_collective_support_keys: set[str] | None = None,
    ):
        self._vectorizer = vectorizer
        self._model = model
        self._cuda_launch_support_keys = set(cuda_launch_support_keys or ())
        if nccl_collective_support_keys is None:
            nccl_collective_support_keys = (
                _trace_learned_nccl_collective_support_keys_from_vectorizer(vectorizer)
            )
        self._nccl_collective_support_keys = set(nccl_collective_support_keys or ())
        self._vectorizer_feature_names = set(
            _trace_learned_vectorizer_feature_names(vectorizer)
        )
        self._prediction_cache: dict[str, float | None] = {}

    def _nccl_collective_support_keys_for_estimation(self) -> set[str]:
        support_keys = getattr(self, "_nccl_collective_support_keys", None)
        if support_keys is None:
            support_keys = _trace_learned_nccl_collective_support_keys_from_vectorizer(
                self._vectorizer
            )
            self._nccl_collective_support_keys = set(support_keys)
        return set(support_keys or ())

    def _vectorizer_feature_names_for_estimation(self) -> set[str]:
        feature_names = getattr(self, "_vectorizer_feature_names", None)
        if feature_names is None:
            feature_names = set(_trace_learned_vectorizer_feature_names(self._vectorizer))
            self._vectorizer_feature_names = set(feature_names)
        return set(feature_names or ())

    @classmethod
    def fit(
        cls,
        samples: list[dict[str, Any]],
        targets_us: list[float],
        *,
        max_samples: int = 20000,
    ) -> "TraceLearnedTimingProvider | None":
        return probe_trace_learned_provider(
            samples,
            targets_us,
            max_samples=max_samples,
        ).provider

    def estimate_us(
        self,
        event: Mapping[str, Any],
        percentile: str = "p50",
    ) -> float | None:
        del percentile
        try:
            import numpy as np
        except Exception:
            return None

        features = _learned_provider_features(event)
        if not _trace_learned_cuda_launch_is_eligible(
            features,
            getattr(self, "_cuda_launch_support_keys", set()),
        ):
            return None
        if not _trace_learned_nccl_collective_is_eligible(
            features,
            self._nccl_collective_support_keys_for_estimation(),
        ):
            return None
        if not _trace_learned_gemm_material_features_are_eligible(
            features,
            self._vectorizer_feature_names_for_estimation(),
        ):
            return None
        cache_key = _signature_key_from_features(features)
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
        matrix = self._vectorizer.transform([features])
        prediction = float(np.expm1(self._model.predict(matrix)[0]))
        if not math.isfinite(prediction) or prediction < 0:
            self._prediction_cache[cache_key] = None
            return None
        self._prediction_cache[cache_key] = prediction
        return prediction

    def estimate_many_us(
        self,
        events: list[Mapping[str, Any]],
        percentile: str = "p50",
    ) -> list[float | None]:
        del percentile
        if not events:
            return []
        try:
            import numpy as np
        except Exception:
            return [None for _ in events]

        results: list[float | None] = [None for _ in events]
        missed_features: list[dict[str, Any]] = []
        missed_keys: list[str] = []
        missed_indexes: list[int] = []
        for index, event in enumerate(events):
            features = _learned_provider_features(event)
            if not _trace_learned_cuda_launch_is_eligible(
                features,
                getattr(self, "_cuda_launch_support_keys", set()),
            ):
                continue
            if not _trace_learned_nccl_collective_is_eligible(
                features,
                self._nccl_collective_support_keys_for_estimation(),
            ):
                continue
            if not _trace_learned_gemm_material_features_are_eligible(
                features,
                self._vectorizer_feature_names_for_estimation(),
            ):
                continue
            cache_key = _signature_key_from_features(features)
            if cache_key in self._prediction_cache:
                results[index] = self._prediction_cache[cache_key]
                continue
            missed_features.append(features)
            missed_keys.append(cache_key)
            missed_indexes.append(index)

        if missed_features:
            matrix = self._vectorizer.transform(missed_features)
            predictions = np.expm1(self._model.predict(matrix))
            for index, cache_key, raw_prediction in zip(
                missed_indexes,
                missed_keys,
                predictions,
            ):
                prediction = float(raw_prediction)
                if not math.isfinite(prediction) or prediction < 0:
                    self._prediction_cache[cache_key] = None
                    results[index] = None
                else:
                    self._prediction_cache[cache_key] = prediction
                    results[index] = prediction
        return results

    def to_jsonable(self) -> dict[str, Any]:
        payload = pickle.dumps(
            {
                "vectorizer": self._vectorizer,
                "model": self._model,
                "cuda_launch_support_keys": sorted(
                    getattr(self, "_cuda_launch_support_keys", set())
                ),
                "nccl_collective_support_keys": sorted(
                    getattr(self, "_nccl_collective_support_keys", set())
                ),
            }
        )
        return {
            "type": "trace_learned_sklearn",
            "pickle_b64": base64.b64encode(payload).decode("ascii"),
        }

    @classmethod
    def from_jsonable(cls, payload: Mapping[str, Any]) -> "TraceLearnedTimingProvider":
        blob = base64.b64decode(str(payload["pickle_b64"]).encode("ascii"))
        state = pickle.loads(blob)
        nccl_collective_support_keys = state.get("nccl_collective_support_keys")
        if nccl_collective_support_keys is None:
            nccl_collective_support_keys = (
                _trace_learned_nccl_collective_support_keys_from_vectorizer(
                    state["vectorizer"]
                )
            )
        return cls(
            state["vectorizer"],
            state["model"],
            cuda_launch_support_keys=set(state.get("cuda_launch_support_keys", ())),
            nccl_collective_support_keys=set(nccl_collective_support_keys or ()),
        )


def probe_trace_learned_provider(
    samples: list[dict[str, Any]],
    targets_us: list[float],
    *,
    max_samples: int = 20000,
) -> ProviderLoadStatus:
    collapsed_pairs = _collapse_collective_group_training_pairs(samples, targets_us)
    if len(collapsed_pairs) < 8:
        return ProviderLoadStatus(
            provider=None,
            error=(
                "trace_learned_sklearn unavailable: insufficient collapsed "
                f"training samples ({len(collapsed_pairs)} < 8)"
            ),
        )

    try:
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_extraction import DictVectorizer
    except Exception as exc:
        return ProviderLoadStatus(
            provider=None,
            error=(
                "trace_learned_sklearn unavailable: missing numpy/sklearn "
                f"dependency ({exc!r})"
            ),
        )

    if len(collapsed_pairs) > max_samples:
        step = max(1, len(collapsed_pairs) // max_samples)
        sampled_pairs = [
            collapsed_pairs[index]
            for index in range(0, len(collapsed_pairs), step)
        ][:max_samples]
    else:
        sampled_pairs = list(collapsed_pairs)

    sampled_samples = [sample for sample, _ in sampled_pairs]
    sampled_targets = [target for _, target in sampled_pairs]

    try:
        vectorizer = DictVectorizer(sparse=True)
        matrix = vectorizer.fit_transform(sampled_samples)
    except Exception as exc:
        return ProviderLoadStatus(
            provider=None,
            error=(
                "trace_learned_sklearn unavailable: failed to vectorize "
                f"training samples ({exc!r})"
            ),
        )
    if getattr(matrix, "shape", (0, 0))[0] < 8:
        return ProviderLoadStatus(
            provider=None,
            error=(
                "trace_learned_sklearn unavailable: insufficient vectorized "
                f"rows ({getattr(matrix, 'shape', (0, 0))[0]} < 8)"
            ),
        )

    try:
        log_targets = np.log1p(np.asarray(sampled_targets, dtype=np.float64))
        model = RandomForestRegressor(
            n_estimators=96,
            max_depth=18,
            min_samples_leaf=4,
            random_state=0,
            n_jobs=1,
        )
        model.fit(matrix, log_targets)
    except Exception as exc:
        return ProviderLoadStatus(
            provider=None,
            error=(
                "trace_learned_sklearn unavailable: model fit failed "
                f"({exc!r})"
            ),
        )
    return ProviderLoadStatus(
        provider=TraceLearnedTimingProvider(
            vectorizer,
            model,
            cuda_launch_support_keys=_trace_learned_cuda_launch_support_keys(sampled_samples),
            nccl_collective_support_keys=_trace_learned_nccl_collective_support_keys(sampled_samples),
        ),
        error=None,
    )


_SIGNATURE_STAT_KEYS = (
    "api",
    "type",
    "kernel_signature",
    "grid_x",
    "grid_y",
    "grid_z",
    "block_x",
    "block_y",
    "block_z",
    "shared_mem",
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
    "transa",
    "transb",
    "algorithm",
    "dtype_code",
    "numel",
    "world_size",
    "collective",
    "collective_api",
    "reduction",
    "communicator_size",
    "participant_count",
    "collective_root",
    "bytes",
)

_LEGACY_SIGNATURE_STAT_KEYS = (
    "api",
    "type",
    "kernel_signature",
    "grid_x",
    "grid_y",
    "grid_z",
    "block_x",
    "block_y",
    "block_z",
    "shared_mem",
    "m",
    "n",
    "k",
    "batch_count",
    "transa",
    "transb",
    "dtype_code",
    "numel",
    "world_size",
    "collective",
    "reduction",
    "bytes",
)

_KERNEL_SIGNATURE_FALLBACK_MATERIAL_KEYS = (
    "api",
    "type",
    "grid_x",
    "grid_y",
    "grid_z",
    "block_x",
    "block_y",
    "block_z",
    "shared_mem",
    "world_size",
    "collective",
    "collective_api",
    "communicator_size",
    "participant_count",
    "collective_root",
    "numel",
    "bytes",
)


def _normalize_signature_features(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = canonicalize_gpu_estimator_event(event)
    features = _learned_provider_features(payload)
    kernel_signature = (
        features.get("kernel_signature")
        or payload.get("kernel_signature")
        or features.get("kernel")
        or features.get("kernel_id")
    )
    if kernel_signature not in (None, ""):
        features["kernel_signature"] = str(kernel_signature)
    if "batchCount" in features and "batch_count" not in features:
        features["batch_count"] = features["batchCount"]
    if "count" in features and "numel" not in features:
        features["numel"] = features["count"]
    if "op" in features and "reduction" not in features:
        features["reduction"] = _normalize_nccl_reduction(features["op"])
    if "datatype" in features and "dtype_code" not in features:
        features["dtype_code"] = _normalize_nccl_dtype_code(features["datatype"])
    if "Atype" in features and "dtype_code" not in features:
        features["dtype_code"] = _normalize_cublas_dtype_code(features["Atype"])
    if "Btype" in features and "dtype_code" not in features:
        features["dtype_code"] = _normalize_cublas_dtype_code(features["Btype"])
    if "Ctype" in features and "dtype_code" not in features:
        features["dtype_code"] = _normalize_cublas_dtype_code(features["Ctype"])
    return {
        key: features[key]
        for key in _SIGNATURE_STAT_KEYS
        if key in features and features[key] not in (None, "")
    }


def _legacy_signature_features(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = canonicalize_gpu_estimator_event(event)
    features = _learned_provider_features(payload)
    kernel_signature = (
        features.get("kernel_signature")
        or payload.get("kernel_signature")
        or features.get("kernel")
        or features.get("kernel_id")
    )
    if kernel_signature not in (None, ""):
        features["kernel_signature"] = str(kernel_signature)
    if "batchCount" in features and "batch_count" not in features:
        features["batch_count"] = features["batchCount"]
    if "count" in features and "numel" not in features:
        features["numel"] = features["count"]
    if "op" in features and "reduction" not in features:
        features["reduction"] = _normalize_nccl_reduction(features["op"])
    if "datatype" in features and "dtype_code" not in features:
        features["dtype_code"] = _normalize_nccl_dtype_code(features["datatype"])
    if "Atype" in features and "dtype_code" not in features:
        features["dtype_code"] = _normalize_cublas_dtype_code(features["Atype"])
    if "Btype" in features and "dtype_code" not in features:
        features["dtype_code"] = _normalize_cublas_dtype_code(features["Btype"])
    if "Ctype" in features and "dtype_code" not in features:
        features["dtype_code"] = _normalize_cublas_dtype_code(features["Ctype"])
    return {
        key: features[key]
        for key in _LEGACY_SIGNATURE_STAT_KEYS
        if key in features and features[key] not in (None, "")
    }


def _signature_key_from_features(features: Mapping[str, Any]) -> str:
    return json.dumps(
        {key: features[key] for key in sorted(features)},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _kernel_signature_material_key(features: Mapping[str, Any]) -> str:
    return _signature_key_from_features(
        {
            key: features[key]
            for key in _KERNEL_SIGNATURE_FALLBACK_MATERIAL_KEYS
            if key in features and features[key] not in (None, "")
        }
    )


def _normalize_serialized_signature_key(key: Any) -> str:
    text = str(key)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(payload, dict):
        return text
    normalized_features = _normalize_signature_features(
        {str(feature_key): value for feature_key, value in payload.items()}
    )
    if not normalized_features:
        return text
    return _signature_key_from_features(normalized_features)


def _canonicalize_signature_p50_us_map(
    raw_map: Mapping[str, float],
) -> dict[str, float]:
    canonical_entries: list[tuple[str, float]] = []
    alias_entries: list[tuple[str, float]] = []
    for key, value in raw_map.items():
        key_text = str(key)
        normalized_key = _normalize_serialized_signature_key(key_text)
        entry = (normalized_key, float(value))
        if normalized_key == key_text:
            canonical_entries.append(entry)
        else:
            alias_entries.append(entry)
    resolved: dict[str, float] = {}
    for key, value in canonical_entries:
        resolved[key] = value
    for key, value in alias_entries:
        resolved.setdefault(key, value)
    return resolved


def _canonicalize_signature_stats_us_map(
    raw_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    canonical_entries: list[tuple[str, dict[str, float]]] = []
    alias_entries: list[tuple[str, dict[str, float]]] = []
    for key, raw_stats in raw_map.items():
        if not isinstance(raw_stats, Mapping):
            continue
        key_text = str(key)
        normalized_key = _normalize_serialized_signature_key(key_text)
        stats: dict[str, float] = {}
        for stat_name in ("p50", "mean", "p95", "count"):
            value = raw_stats.get(stat_name)
            if value in (None, ""):
                continue
            stats[stat_name] = float(value)
        if "p50" not in stats:
            continue
        stats.setdefault("mean", stats["p50"])
        stats.setdefault("p95", stats["p50"])
        stats.setdefault("count", 1.0)
        entry = (normalized_key, stats)
        if normalized_key == key_text:
            canonical_entries.append(entry)
        else:
            alias_entries.append(entry)
    resolved: dict[str, dict[str, float]] = {}
    for key, stats in canonical_entries:
        resolved[key] = stats
    for key, stats in alias_entries:
        resolved.setdefault(key, stats)
    return resolved


def _timing_stats_us(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return None
    p95_index = min(int(0.95 * len(sorted_values)), len(sorted_values) - 1)
    return {
        "p50": float(statistics.median(sorted_values)),
        "mean": float(statistics.mean(sorted_values)),
        "p95": float(sorted_values[p95_index]),
        "count": float(len(sorted_values)),
    }


def _canonicalize_collective_group_event(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = _normalize_p2p_collective_group_payload(
        canonicalize_gpu_estimator_event(event)
    )
    collective_group_id = payload.get("collective_group_id")
    if collective_group_id not in (None, ""):
        payload["collective_group_id"] = _normalize_collective_group_id(
            collective_group_id
        )
    return payload


def _canonicalize_event_for_provider(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = canonicalize_gpu_estimator_event(event)
    collective_group_id = payload.get("collective_group_id")
    if collective_group_id not in (None, ""):
        payload["collective_group_id"] = _normalize_collective_group_id(
            collective_group_id
        )
    return payload


class TraceSignatureTimingProvider:
    """
    Dependency-free event-signature statistics provider.

    This sits between exact provider-backed models and coarse `(api, type)` lookup.
    It uses a normalized subset of low-level operator metadata so shape-aware
    events can still resolve through finer-grained timing statistics even when
    optional sklearn/xgboost dependencies are unavailable.
    """

    name = "trace_signature_stats"
    supports_collective_group_timing = True

    def __init__(
        self,
        signature_p50_us: Mapping[str, float],
        kernel_signature_p50_us: Mapping[str, float] | None = None,
        signature_stats_us: Mapping[str, Mapping[str, Any]] | None = None,
        kernel_signature_stats_us: Mapping[str, Mapping[str, Any]] | None = None,
        kernel_signature_material_keys: Mapping[str, Sequence[str]] | None = None,
    ):
        self._signature_p50_us = _canonicalize_signature_p50_us_map(signature_p50_us)
        self._kernel_signature_p50_us = dict(kernel_signature_p50_us or {})
        self._signature_stats_us = _canonicalize_signature_stats_us_map(signature_stats_us or {})
        self._kernel_signature_stats_us = _canonicalize_signature_stats_us_map(
            kernel_signature_stats_us or {}
        )
        if kernel_signature_material_keys is None:
            self._kernel_signature_material_keys = {}
        else:
            self._kernel_signature_material_keys = {
                str(key): frozenset(str(value) for value in values)
                for key, values in kernel_signature_material_keys.items()
            }
        for key, value in self._signature_p50_us.items():
            self._signature_stats_us.setdefault(
                key,
                {"p50": float(value), "mean": float(value), "p95": float(value), "count": 1.0},
            )
        for key, value in self._kernel_signature_p50_us.items():
            self._kernel_signature_stats_us.setdefault(
                key,
                {"p50": float(value), "mean": float(value), "p95": float(value), "count": 1.0},
            )
        if kernel_signature_material_keys is None:
            self._kernel_signature_material_keys = self._derive_kernel_signature_material_keys()

    @classmethod
    def fit(
        cls,
        samples: list[dict[str, Any]],
        targets_us: list[float],
    ) -> "TraceSignatureTimingProvider | None":
        if not samples or not targets_us:
            return None
        buckets: dict[str, list[float]] = defaultdict(list)
        legacy_buckets: dict[str, list[float]] = defaultdict(list)
        kernel_signature_buckets: dict[str, list[float]] = defaultdict(list)
        kernel_signature_material_keys: dict[str, set[str]] = defaultdict(set)
        collective_group_samples: dict[str, tuple[dict[str, Any], dict[str, Any], str | None]] = {}
        collective_group_targets: dict[str, float] = {}
        for sample, target_us in zip(samples, targets_us):
            clean_sample = _strip_training_metadata(sample)
            features = _normalize_signature_features(clean_sample)
            if not features:
                continue
            normalized_target_us = max(float(target_us), 0.0)
            api_name = _normalize_api_name(clean_sample.get("api") or clean_sample.get("api_name") or clean_sample.get("name"))
            typ = _normalize_low_level_op_type(api_name, str(clean_sample.get("type") or "other"))
            if _is_device_work_event(api_name, typ) and normalized_target_us <= 0.0:
                continue
            legacy_features = _legacy_signature_features(clean_sample)
            kernel_signature = features.get("kernel_signature")
            collective_group_id = _collective_group_training_id(sample)
            if typ == "nccl_collective" and collective_group_id not in (None, ""):
                group_id = str(collective_group_id)
                collective_group_samples.setdefault(
                    group_id,
                    (
                        features,
                        legacy_features,
                        str(kernel_signature) if kernel_signature not in (None, "") else None,
                    ),
                )
                collective_group_targets[group_id] = max(
                    collective_group_targets.get(group_id, 0.0),
                    normalized_target_us,
                )
                continue
            buckets[_signature_key_from_features(features)].append(normalized_target_us)
            if legacy_features:
                legacy_buckets[_signature_key_from_features(legacy_features)].append(
                    normalized_target_us
                )
            if kernel_signature not in (None, ""):
                kernel_signature_buckets[str(kernel_signature)].append(normalized_target_us)
                kernel_signature_material_keys[str(kernel_signature)].add(
                    _kernel_signature_material_key(features)
                )

        for group_id, group_target_us in collective_group_targets.items():
            features, legacy_features, kernel_signature = collective_group_samples[group_id]
            buckets[_signature_key_from_features(features)].append(group_target_us)
            if legacy_features:
                legacy_buckets[_signature_key_from_features(legacy_features)].append(
                    group_target_us
                )
            if kernel_signature not in (None, ""):
                kernel_signature_buckets[str(kernel_signature)].append(group_target_us)
                kernel_signature_material_keys[str(kernel_signature)].add(
                    _kernel_signature_material_key(features)
                )
        if not buckets:
            return None
        signature_stats_us = {
            key: stats
            for key, values in buckets.items()
            for stats in (_timing_stats_us(values),)
            if stats is not None
        }
        signature_p50_us = {
            key: float(stats["p50"])
            for key, stats in signature_stats_us.items()
        }
        for key, values in legacy_buckets.items():
            if values and key not in signature_p50_us:
                stats = _timing_stats_us(values)
                if stats is None:
                    continue
                signature_stats_us[key] = stats
                signature_p50_us[key] = float(stats["p50"])
        if not signature_p50_us:
            return None
        kernel_signature_stats_us = {
            key: stats
            for key, values in kernel_signature_buckets.items()
            for stats in (_timing_stats_us(values),)
            if stats is not None
        }
        kernel_signature_p50_us = {
            key: float(stats["p50"])
            for key, stats in kernel_signature_stats_us.items()
        }
        return cls(
            signature_p50_us,
            kernel_signature_p50_us,
            signature_stats_us=signature_stats_us,
            kernel_signature_stats_us=kernel_signature_stats_us,
            kernel_signature_material_keys=kernel_signature_material_keys,
        )

    def estimate_us(
        self,
        event: Mapping[str, Any],
        percentile: str = "p50",
    ) -> float | None:
        features = _normalize_signature_features(event)
        if not features:
            return None
        stat_name = percentile if percentile in {"p50", "mean", "p95"} else "p50"
        exact_stats = self._signature_stats_us.get(_signature_key_from_features(features))
        if exact_stats is not None:
            return exact_stats.get(stat_name, exact_stats["p50"])
        legacy_features = _legacy_signature_features(event)
        if legacy_features:
            legacy_stats = self._signature_stats_us.get(_signature_key_from_features(legacy_features))
            if legacy_stats is not None:
                return legacy_stats.get(stat_name, legacy_stats["p50"])
        kernel_signature = features.get("kernel_signature")
        if kernel_signature not in (None, ""):
            kernel_stats = self._kernel_signature_stats_us.get(str(kernel_signature))
            if kernel_stats is not None and self._kernel_signature_fallback_is_safe(
                str(kernel_signature), features
            ):
                return kernel_stats.get(stat_name, kernel_stats["p50"])
        return None

    def _derive_kernel_signature_material_keys(self) -> dict[str, frozenset[str]]:
        derived_kernel_material_keys: dict[str, set[str]] = defaultdict(set)
        for signature_key in self._signature_stats_us:
            try:
                signature_features = json.loads(str(signature_key))
            except json.JSONDecodeError:
                continue
            if not isinstance(signature_features, Mapping):
                continue
            trained_kernel_signature = signature_features.get("kernel_signature")
            if trained_kernel_signature in (None, ""):
                continue
            derived_kernel_material_keys[str(trained_kernel_signature)].add(
                _kernel_signature_material_key(signature_features)
            )
        return {
            key: frozenset(values)
            for key, values in derived_kernel_material_keys.items()
        }

    def _kernel_signature_fallback_is_safe(
        self,
        kernel_signature: str,
        features: Mapping[str, Any],
    ) -> bool:
        if not hasattr(self, "_kernel_signature_material_keys"):
            self._kernel_signature_material_keys = self._derive_kernel_signature_material_keys()
        material_keys = self._kernel_signature_material_keys.get(str(kernel_signature))
        if material_keys is None or len(material_keys) != 1:
            return False
        return _kernel_signature_material_key(features) == next(iter(material_keys))

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": "trace_signature_stats",
            "signature_p50_us": self._signature_p50_us,
            "kernel_signature_p50_us": self._kernel_signature_p50_us,
            "signature_stats_us": self._signature_stats_us,
            "kernel_signature_stats_us": self._kernel_signature_stats_us,
            "kernel_signature_material_keys": {
                key: sorted(values)
                for key, values in self._kernel_signature_material_keys.items()
            },
        }

    @classmethod
    def from_jsonable(cls, payload: Mapping[str, Any]) -> "TraceSignatureTimingProvider":
        raw = payload.get("signature_p50_us", {})
        raw_kernel = payload.get("kernel_signature_p50_us", {})
        raw_stats = payload.get("signature_stats_us", {})
        raw_kernel_stats = payload.get("kernel_signature_stats_us", {})
        raw_kernel_material_keys = payload.get("kernel_signature_material_keys", None)
        return cls(
            {str(key): float(value) for key, value in raw.items()},
            {str(key): float(value) for key, value in raw_kernel.items()},
            signature_stats_us=(
                raw_stats if isinstance(raw_stats, Mapping) else {}
            ),
            kernel_signature_stats_us=(
                raw_kernel_stats if isinstance(raw_kernel_stats, Mapping) else {}
            ),
            kernel_signature_material_keys=(
                raw_kernel_material_keys
                if isinstance(raw_kernel_material_keys, Mapping)
                else None
            ),
        )


def probe_gpu_estimator_provider(
    bundle_dir: str | Path = DEFAULT_GPU_ESTIMATOR_BUNDLE,
) -> ProviderLoadStatus:
    """
    Load the optional gpu_estimator runtime bundle.

    Returns structured status so callers can decide whether to fail closed or
    gracefully degrade when dependencies like xgboost are unavailable.
    """
    bundle_path = Path(bundle_dir)
    module_path = bundle_path / "src" / "fake_driver_xgboost_provider.py"
    if not module_path.exists():
        return ProviderLoadStatus(
            provider=None,
            error=f"runtime bundle entrypoint not found: {module_path}",
        )

    module_name = "_flexsim_gpu_estimator_runtime_provider"
    module = sys.modules.get(module_name)

    extra_import_paths = [
        str(candidate)
        for candidate in (
            bundle_path / ".vendor",
            bundle_path / "vendor",
            REPO_ROOT / ".vendor",
        )
        if candidate.exists()
    ]

    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return ProviderLoadStatus(
                provider=None,
                error=f"failed to create import spec for: {module_path}",
            )

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        src_dir = str(module_path.parent)
        added_paths: list[str] = []
        for candidate in [src_dir, *extra_import_paths]:
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
                added_paths.append(candidate)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            return ProviderLoadStatus(
                provider=None,
                error=f"failed to import gpu_estimator runtime bundle: {exc!r}",
            )
        finally:
            for candidate in reversed(added_paths):
                try:
                    sys.path.remove(candidate)
                except ValueError:
                    continue

    try:
        runtime_provider = module.FakeDriverXGBoostProvider()
    except Exception as exc:
        return ProviderLoadStatus(
            provider=None,
            error=f"failed to initialize gpu_estimator runtime provider: {exc!r}",
        )
    return ProviderLoadStatus(
        provider=GPUXGBoostTimingProvider(runtime_provider, bundle_path),
        error=None,
    )


def load_gpu_estimator_provider(
    bundle_dir: str | Path = DEFAULT_GPU_ESTIMATOR_BUNDLE,
) -> EventTimingProvider | None:
    return probe_gpu_estimator_provider(bundle_dir).provider


class Estimator:
    """
    Data-driven timing model learned from real GPU traces.

    Learns per-(api, type) duration distributions by computing
    inter-event timestamp deltas from real traces.
    """

    def __init__(self, providers: Optional[list[EventTimingProvider]] = None):
        # (api, type) -> list of durations in microseconds
        self._raw: dict[tuple[str, str], list[float]] = defaultdict(list)
        # Computed stats: (api, type) -> {p50, mean, p95, count}
        self._stats: dict[tuple[str, str], dict] = {}
        # Type-level fallback: type -> {p50, mean, p95, count}
        self._type_stats: dict[str, dict] = {}
        # Global fallback
        self._global_p50: float = 1.0  # 1 us default
        # Optional event-level providers checked before trace statistics
        self._providers: list[EventTimingProvider] = list(providers or [])
        self._provider_diagnostics: dict[str, str] = {}
        self._provenance: dict[str, Any] = {}
        # Samples used to optionally fit a learned low-level regressor
        self._feature_samples: list[dict[str, Any]] = []
        self._feature_targets_us: list[float] = []
        # Heavy-hitter operator-family accounting from fitted trace targets
        self._operator_family_totals_us: dict[str, float] = defaultdict(float)
        self._operator_family_counts: dict[str, int] = defaultdict(int)
        self._operator_family_examples: dict[str, dict[str, float]] = defaultdict(
            _nested_float_defaultdict
        )
        self._kernel_launch_metadata = {
            "total_kernel_launches": 0,
            "with_kernel_name": 0,
            "with_launch_shape": 0,
            "with_stream_id": 0,
            "with_host_duration": 0,
        }
        self._transparent_profiling_metadata = {
            "modeled_event_count": 0,
            "modeled_event_with_wrapper_timing_field_count": 0,
            "modeled_event_with_explicit_direct_runtime_field_count": 0,
            "modeled_event_with_direct_wrapper_runtime_count": 0,
            "modeled_event_with_dispatch_only_wrapper_contract_count": 0,
            "modeled_event_with_direct_runtime_contract_count": 0,
            "modeled_target_us_total": 0.0,
            "modeled_target_us_with_direct_wrapper_runtime": 0.0,
            "modeled_observed_wrapper_us_total": 0.0,
            "async_modeled_device_event_count": 0,
            "async_modeled_device_event_with_wrapper_timing_field_count": 0,
            "async_modeled_device_event_with_explicit_direct_runtime_field_count": 0,
            "async_modeled_device_event_with_direct_wrapper_runtime_count": 0,
            "async_modeled_device_event_with_dispatch_only_wrapper_contract_count": 0,
            "async_modeled_device_event_with_direct_runtime_contract_count": 0,
            "async_modeled_device_target_us_total": 0.0,
            "async_modeled_device_target_us_with_direct_wrapper_runtime": 0.0,
            "async_modeled_device_observed_wrapper_us_total": 0.0,
        }

    @classmethod
    def fit_from_traces(
        cls,
        trace_dir: str,
        max_files: int = 0,
        providers: Optional[list[EventTimingProvider]] = None,
        learned_method: str = "trace_stats",
        gpu_estimator_bundle: str | Path = DEFAULT_GPU_ESTIMATOR_BUNDLE,
        fit_workers: int = 1,
        trace_window: str = "auto",
    ) -> "Estimator":
        """
        Fit estimator from a directory of real trace JSONL files.

        Args:
            trace_dir: Directory containing rank_*.jsonl files
            max_files: Max files to load (0 = all)
        """
        est = cls(providers=providers)
        trace_path = Path(trace_dir)
        est._provenance = {"training_trace_fingerprints": [trace_dir_fingerprint(trace_path)]}

        files = est._list_rank_trace_files(trace_path)
        if not files:
            raise FileNotFoundError(f"No rank_*.jsonl files in {trace_dir}")

        if max_files > 0:
            files = files[:max_files]

        world_size = len(files)
        fit_windows = _fit_windows_from_manifest(trace_path, trace_window=trace_window)
        communicator_aliases_by_rank, communicator_memberships = (
            _communicator_context_from_manifest(trace_path)
        )
        ingest_items = [
            (
                str(f),
                world_size,
                fit_windows.get(_rank_from_trace_path(f)),
                communicator_aliases_by_rank.get(_rank_from_trace_path(f), {}),
                communicator_memberships,
            )
            for f in files
        ]
        est._ingest_items(
            ingest_items,
            fit_workers=fit_workers,
        )

        est._compute_stats()
        est._attach_learned_methods(
            learned_method,
            bundle_dir=gpu_estimator_bundle,
        )
        return est

    @classmethod
    def fit_from_maya_trace_bundle(
        cls,
        bundle: Any,
        providers: Optional[list[EventTimingProvider]] = None,
        learned_method: str = "trace_stats",
        gpu_estimator_bundle: str | Path = DEFAULT_GPU_ESTIMATOR_BUNDLE,
    ) -> "Estimator":
        """Fit from an already-loaded Maya-lite TraceBundle.

        This avoids a second JSONL parse when Maya-lite has already loaded the
        profiling trace into memory.  It preserves the same step-window and
        communicator topology contracts used by ``fit_from_traces``.
        """
        est = cls(providers=providers)
        est._provenance = {"training_trace_fingerprints": [trace_dir_fingerprint(Path(bundle.trace_dir))]}
        communicator_memberships = getattr(bundle, "communicator_memberships", {})
        communicator_aliases_by_rank = getattr(bundle, "communicator_aliases_by_rank", {})
        world_size = int(getattr(bundle, "world_size", 1) or 1)
        trace_window = getattr(bundle, "trace_window", "auto")
        step_windows = getattr(bundle, "step_windows", {})
        for rank_trace in getattr(bundle, "rank_traces", ()):  # maya_lite.schema.RankTrace
            active_window = None
            if trace_window == "step":
                active_window = step_windows.get(rank_trace.rank)
            est._ingest_maya_trace_events(
                tuple(rank_trace.events),
                world_size=world_size,
                active_window=active_window,
                communicator_aliases=communicator_aliases_by_rank.get(rank_trace.rank, {}),
                communicator_memberships=communicator_memberships,
            )
        est._compute_stats()
        est._attach_learned_methods(
            learned_method,
            bundle_dir=gpu_estimator_bundle,
        )
        return est

    @classmethod
    def fit_from_multiple(
        cls,
        trace_dirs: list[str],
        providers: Optional[list[EventTimingProvider]] = None,
        learned_method: str = "trace_stats",
        gpu_estimator_bundle: str | Path = DEFAULT_GPU_ESTIMATOR_BUNDLE,
        fit_workers: int = 1,
        trace_window: str = "auto",
    ) -> "Estimator":
        """Fit from multiple experiment directories for broader coverage."""
        est = cls(providers=providers)
        est._provenance = {"training_trace_fingerprints": [trace_dir_fingerprint(Path(d)) for d in trace_dirs]}
        ingest_items: list[tuple[str, int, tuple[int, int] | None]] = []
        for d in trace_dirs:
            trace_path = Path(d)
            files = est._list_rank_trace_files(trace_path)
            world_size = len(files)
            fit_windows = _fit_windows_from_manifest(trace_path, trace_window=trace_window)
            communicator_aliases_by_rank, communicator_memberships = (
                _communicator_context_from_manifest(trace_path)
            )
            ingest_items.extend(
                (
                    str(f),
                    world_size,
                    fit_windows.get(_rank_from_trace_path(f)),
                    communicator_aliases_by_rank.get(_rank_from_trace_path(f), {}),
                    communicator_memberships,
                )
                for f in files
            )
        est._ingest_items(
            ingest_items,
            fit_workers=fit_workers,
        )
        est._compute_stats()
        est._attach_learned_methods(
            learned_method,
            bundle_dir=gpu_estimator_bundle,
        )
        return est

    def add_provider(self, provider: EventTimingProvider, *, prepend: bool = False):
        """Register an event-level provider ahead of trace-stat fallbacks."""
        if prepend:
            self._providers.insert(0, provider)
        else:
            self._providers.append(provider)

    def attach_learned_methods(
        self,
        learned_method: str,
        *,
        gpu_estimator_bundle: str | Path = DEFAULT_GPU_ESTIMATOR_BUNDLE,
    ) -> None:
        """Attach learned/provider timing methods to an already fitted estimator."""
        self._attach_learned_methods(
            learned_method,
            bundle_dir=gpu_estimator_bundle,
        )

    def provider_names(self) -> tuple[str, ...]:
        return tuple(
            getattr(provider, "name", type(provider).__name__)
            for provider in self._providers
        )

    def provider_diagnostics(self) -> dict[str, str]:
        return dict(self._provider_diagnostics)

    def has_event_providers(self) -> bool:
        return bool(self._providers)

    def has_trace_statistics(self) -> bool:
        return bool(self._stats or self._type_stats)

    def is_calibrated(self) -> bool:
        """
        Return whether this estimator has any learned/provider-backed timing signal.

        A bare ``Estimator()`` with no fitted statistics and no providers should not
        be treated as suitable for absolute timing comparisons.
        """
        return self.has_event_providers() or self.has_trace_statistics()

    def operator_family_summary(self, limit: int = 10) -> list[dict[str, Any]]:
        total_us = sum(self._operator_family_totals_us.values())
        rows: list[dict[str, Any]] = []
        for family, family_us in sorted(
            self._operator_family_totals_us.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            example_rows = sorted(
                self._operator_family_examples.get(family, {}).items(),
                key=lambda item: item[1],
                reverse=True,
            )
            rows.append(
                {
                    "family": family,
                    "sample_count": int(self._operator_family_counts.get(family, 0)),
                    "total_target_us": float(family_us),
                    "time_share": (float(family_us) / total_us) if total_us > 0 else 0.0,
                    "top_examples": [
                        {
                            "label": label,
                            "total_target_us": float(example_us),
                        }
                        for label, example_us in example_rows[:3]
                    ],
                }
            )
        if limit > 0:
            return rows[:limit]
        return rows

    def kernel_launch_metadata_summary(self) -> dict[str, Any]:
        total = int(self._kernel_launch_metadata.get("total_kernel_launches", 0))

        def _share(count_key: str) -> float:
            count = int(self._kernel_launch_metadata.get(count_key, 0))
            return (float(count) / float(total)) if total > 0 else 0.0

        return {
            **{key: int(value) for key, value in self._kernel_launch_metadata.items()},
            "kernel_name_share": _share("with_kernel_name"),
            "launch_shape_share": _share("with_launch_shape"),
            "stream_id_share": _share("with_stream_id"),
            "host_duration_share": _share("with_host_duration"),
            "missing_kernel_name_count": max(
                total - int(self._kernel_launch_metadata.get("with_kernel_name", 0)),
                0,
            ),
            "missing_launch_shape_count": max(
                total - int(self._kernel_launch_metadata.get("with_launch_shape", 0)),
                0,
            ),
        }

    def transparent_profiling_summary(self) -> dict[str, Any]:
        modeled_total = int(self._transparent_profiling_metadata["modeled_event_count"])
        modeled_field = int(
            self._transparent_profiling_metadata["modeled_event_with_wrapper_timing_field_count"]
        )
        modeled_explicit = int(
            self._transparent_profiling_metadata[
                "modeled_event_with_explicit_direct_runtime_field_count"
            ]
        )
        modeled_direct = int(
            self._transparent_profiling_metadata["modeled_event_with_direct_wrapper_runtime_count"]
        )
        modeled_dispatch_contract = int(
            self._transparent_profiling_metadata[
                "modeled_event_with_dispatch_only_wrapper_contract_count"
            ]
        )
        modeled_direct_contract = int(
            self._transparent_profiling_metadata[
                "modeled_event_with_direct_runtime_contract_count"
            ]
        )
        modeled_target_total = float(self._transparent_profiling_metadata["modeled_target_us_total"])
        modeled_target_direct = float(
            self._transparent_profiling_metadata["modeled_target_us_with_direct_wrapper_runtime"]
        )
        async_total = int(self._transparent_profiling_metadata["async_modeled_device_event_count"])
        async_field = int(
            self._transparent_profiling_metadata["async_modeled_device_event_with_wrapper_timing_field_count"]
        )
        async_explicit = int(
            self._transparent_profiling_metadata[
                "async_modeled_device_event_with_explicit_direct_runtime_field_count"
            ]
        )
        async_direct = int(
            self._transparent_profiling_metadata["async_modeled_device_event_with_direct_wrapper_runtime_count"]
        )
        async_dispatch_contract = int(
            self._transparent_profiling_metadata[
                "async_modeled_device_event_with_dispatch_only_wrapper_contract_count"
            ]
        )
        async_direct_contract = int(
            self._transparent_profiling_metadata[
                "async_modeled_device_event_with_direct_runtime_contract_count"
            ]
        )
        async_target_total = float(
            self._transparent_profiling_metadata["async_modeled_device_target_us_total"]
        )
        async_target_direct = float(
            self._transparent_profiling_metadata["async_modeled_device_target_us_with_direct_wrapper_runtime"]
        )
        return {
            "modeled_event_count": modeled_total,
            "modeled_event_with_wrapper_timing_field_count": modeled_field,
            "modeled_event_with_explicit_direct_runtime_field_count": modeled_explicit,
            "modeled_event_with_direct_wrapper_runtime_count": modeled_direct,
            "modeled_event_with_dispatch_only_wrapper_contract_count": modeled_dispatch_contract,
            "modeled_event_with_direct_runtime_contract_count": modeled_direct_contract,
            "modeled_wrapper_timing_field_share": (
                float(modeled_field) / float(modeled_total) if modeled_total > 0 else 0.0
            ),
            "modeled_explicit_direct_runtime_field_share": (
                float(modeled_explicit) / float(modeled_total) if modeled_total > 0 else 0.0
            ),
            "modeled_direct_wrapper_runtime_share": (
                float(modeled_direct) / float(modeled_total) if modeled_total > 0 else 0.0
            ),
            "modeled_dispatch_only_wrapper_contract_share": (
                float(modeled_dispatch_contract) / float(modeled_total) if modeled_total > 0 else 0.0
            ),
            "modeled_direct_runtime_contract_share": (
                float(modeled_direct_contract) / float(modeled_total) if modeled_total > 0 else 0.0
            ),
            "modeled_target_us_total": modeled_target_total,
            "modeled_target_us_with_direct_wrapper_runtime": modeled_target_direct,
            "modeled_target_time_share_with_direct_wrapper_runtime": (
                modeled_target_direct / modeled_target_total if modeled_target_total > 0 else 0.0
            ),
            "modeled_observed_wrapper_us_total": float(
                self._transparent_profiling_metadata["modeled_observed_wrapper_us_total"]
            ),
            "async_modeled_device_event_count": async_total,
            "async_modeled_device_event_with_wrapper_timing_field_count": async_field,
            "async_modeled_device_event_with_explicit_direct_runtime_field_count": async_explicit,
            "async_modeled_device_event_with_direct_wrapper_runtime_count": async_direct,
            "async_modeled_device_event_with_dispatch_only_wrapper_contract_count": (
                async_dispatch_contract
            ),
            "async_modeled_device_event_with_direct_runtime_contract_count": async_direct_contract,
            "async_modeled_device_wrapper_timing_field_share": (
                float(async_field) / float(async_total) if async_total > 0 else 0.0
            ),
            "async_modeled_device_explicit_direct_runtime_field_share": (
                float(async_explicit) / float(async_total) if async_total > 0 else 0.0
            ),
            "async_modeled_device_direct_wrapper_runtime_share": (
                float(async_direct) / float(async_total) if async_total > 0 else 0.0
            ),
            "async_modeled_device_dispatch_only_wrapper_contract_share": (
                float(async_dispatch_contract) / float(async_total) if async_total > 0 else 0.0
            ),
            "async_modeled_device_direct_runtime_contract_share": (
                float(async_direct_contract) / float(async_total) if async_total > 0 else 0.0
            ),
            "async_modeled_device_target_us_total": async_target_total,
            "async_modeled_device_target_us_with_direct_wrapper_runtime": async_target_direct,
            "async_modeled_device_target_time_share_with_direct_wrapper_runtime": (
                async_target_direct / async_target_total if async_target_total > 0 else 0.0
            ),
            "async_modeled_device_observed_wrapper_us_total": float(
                self._transparent_profiling_metadata["async_modeled_device_observed_wrapper_us_total"]
            ),
        }

    def provider_coverage_summary(
        self,
        provider_name: str | None = None,
        *,
        limit: int = 10,
        modeled_ops_only: bool = True,
        providers: Sequence[EventTimingProvider] | None = None,
    ) -> dict[str, Any]:
        provider_pool = tuple(providers) if providers is not None else tuple(self._providers)
        matching_providers = [
            provider
            for provider in provider_pool
            if provider_name is None
            or getattr(provider, "name", type(provider).__name__) == provider_name
        ]
        filtered_pairs = [
            (sample, target_us)
            for sample, target_us in zip(self._feature_samples, self._feature_targets_us)
            if (not modeled_ops_only) or is_modeled_operator_event(sample)
        ]
        total_target_us = float(sum(target_us for _, target_us in filtered_pairs))
        total_samples = len(filtered_pairs)
        if not matching_providers:
            return {
                "provider_name": provider_name or "all",
                "matched_provider_names": [],
                "modeled_ops_only": modeled_ops_only,
                "covered_target_us": 0.0,
                "covered_sample_count": 0,
                "covered_time_share": 0.0,
                "covered_event_share": 0.0,
                "top_covered_families": [],
                "top_uncovered_families": self.operator_family_summary(limit=limit),
            }

        covered_target_us = 0.0
        covered_sample_count = 0
        covered_by_family: dict[str, float] = defaultdict(float)
        uncovered_by_family: dict[str, float] = defaultdict(float)
        uncovered_counts: dict[str, int] = defaultdict(int)
        coverage_cache: dict[str, bool] = {}
        for sample, target_us in filtered_pairs:
            cache_key = _provider_coverage_cache_key(sample)
            covered = coverage_cache.get(cache_key)
            if covered is None:
                covered = False
                for provider in matching_providers:
                    try:
                        duration_us = provider.estimate_us(sample, "p50")
                    except Exception:
                        duration_us = None
                    if duration_us is not None:
                        covered = True
                        break
                coverage_cache[cache_key] = covered
            family = event_operator_family(sample)
            if covered:
                covered_target_us += float(target_us)
                covered_sample_count += 1
                covered_by_family[family] += float(target_us)
            else:
                uncovered_by_family[family] += float(target_us)
                uncovered_counts[family] += 1

        def _family_rows(
            family_totals: Mapping[str, float],
            family_counts: Mapping[str, int] | None = None,
        ) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for family, family_us in sorted(
                family_totals.items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                rows.append(
                    {
                        "family": family,
                        "total_target_us": float(family_us),
                        "time_share": (float(family_us) / total_target_us)
                        if total_target_us > 0
                        else 0.0,
                        "sample_count": int((family_counts or self._operator_family_counts).get(family, 0)),
                    }
                )
            if limit > 0:
                return rows[:limit]
            return rows

        return {
            "provider_name": provider_name or "all",
            "matched_provider_names": [
                getattr(provider, "name", type(provider).__name__)
                for provider in matching_providers
            ],
            "modeled_ops_only": modeled_ops_only,
            "covered_target_us": float(covered_target_us),
            "covered_sample_count": int(covered_sample_count),
            "covered_time_share": (
                float(covered_target_us) / total_target_us if total_target_us > 0 else 0.0
            ),
            "covered_event_share": (
                float(covered_sample_count) / float(total_samples) if total_samples > 0 else 0.0
            ),
            "top_covered_families": _family_rows(covered_by_family),
            "top_uncovered_families": _family_rows(uncovered_by_family, uncovered_counts),
        }

    @staticmethod
    def _list_rank_trace_files(trace_path: Path) -> list[Path]:
        return sorted(
            candidate
            for candidate in trace_path.glob("rank_*.jsonl")
            if _RANK_TRACE_FILE_RE.match(candidate.name)
        )

    def attach_optional_gpu_estimator(
        self,
        bundle_dir: str | Path = DEFAULT_GPU_ESTIMATOR_BUNDLE,
        *,
        prepend: bool = True,
    ) -> bool:
        status = probe_gpu_estimator_provider(bundle_dir)
        provider = status.provider
        if provider is None:
            if status.error:
                self._provider_diagnostics["gpu_estimator_xgboost"] = status.error
            return False
        self.add_provider(provider, prepend=prepend)
        self._provider_diagnostics.pop("gpu_estimator_xgboost", None)
        return True

    def _attach_learned_methods(
        self,
        learned_method: str,
        *,
        bundle_dir: str | Path = DEFAULT_GPU_ESTIMATOR_BUNDLE,
    ) -> None:
        method = (learned_method or "trace_stats").strip().lower()
        if method in {"trace_stats", "stats", "lookup"}:
            return
        signature_provider = None
        learned_provider = None
        if method in {"gpu_xgboost", "xgboost"}:
            signature_provider = TraceSignatureTimingProvider.fit(
                self._feature_samples,
                self._feature_targets_us,
            )
            if signature_provider is not None:
                self.add_provider(signature_provider)
        if method in {"learned_trace", "trace_learned", "hybrid", "hybrid_trace"}:
            learned_status = probe_trace_learned_provider(
                self._feature_samples,
                self._feature_targets_us,
            )
            learned_provider = learned_status.provider
            signature_provider = TraceSignatureTimingProvider.fit(
                self._feature_samples,
                self._feature_targets_us,
            )
            if learned_provider is not None:
                self.add_provider(learned_provider)
                self._provider_diagnostics.pop("trace_learned_sklearn", None)
            elif learned_status.error:
                self._provider_diagnostics["trace_learned_sklearn"] = learned_status.error
            if signature_provider is not None:
                self.add_provider(signature_provider, prepend=True)
        if method in {"gpu_xgboost", "xgboost", "hybrid"}:
            loaded = self.attach_optional_gpu_estimator(
                bundle_dir,
                prepend=method in {"gpu_xgboost", "xgboost"},
            )
            if method in {"gpu_xgboost", "xgboost"} and not loaded:
                detail = self._provider_diagnostics.get(
                    "gpu_estimator_xgboost",
                    "gpu_estimator provider unavailable",
                )
                raise RuntimeError(
                    "gpu_xgboost estimator requested but unavailable: "
                    f"{detail}"
                )

    def _ingest_file(
        self,
        path: Path,
        *,
        world_size: int = 1,
        active_window: tuple[int, int] | None = None,
        communicator_aliases: Mapping[str, str] | None = None,
        communicator_memberships: Mapping[str, tuple[int, ...]] | None = None,
    ):
        """
        Read a single JSONL trace and compute per-call durations.

        Real traces record API entry timestamps in microseconds. We estimate
        a richer training target in two stages:

        1. host-visible call cost from thread-local deltas between consecutive
           API-entry observations on the same `(pid, tid)` stream
        2. device-side wait attribution by pushing blocking sync time from
           `cuda*Synchronize` calls back onto the pending stream work that the
           host was waiting for

        This keeps setup/control APIs cheap while letting asynchronous device
        work absorb a better approximation of true execution time.
        """
        thread_events: dict[tuple[int, int], list[int]] = defaultdict(list)
        entries: list[dict[str, Any]] = []
        with open(path) as f:
            for ordinal, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = rec.get("ts")
                if ts is None:
                    continue
                ts_int = int(ts)
                if active_window is not None:
                    start_ts, end_ts = active_window
                    if ts_int < int(start_ts) or ts_int > int(end_ts):
                        continue
                api = _normalize_api_name(rec.get("api", ""))
                typ = _normalize_low_level_op_type(api, str(rec.get("type", "other")))
                pid = int(rec.get("pid", 0))
                tid = int(rec.get("tid", 0))
                feature_payload = {
                    **rec,
                    "api": api,
                    "type": typ,
                    "ordinal": ordinal,
                    "world_size": world_size,
                }
                feature_payload = _apply_communicator_topology_to_payload(
                    feature_payload,
                    communicator_aliases=communicator_aliases,
                    communicator_memberships=communicator_memberships,
                )
                if rec.get("collective_api") not in (None, ""):
                    feature_payload["collective_api"] = _normalize_api_name(
                        rec.get("collective_api")
                    )
                entry_id = len(entries)
                if api == "cudaLaunchKernel" or typ == "kernel_launch":
                    self._kernel_launch_metadata["total_kernel_launches"] += 1
                    if rec.get("kernel") not in (None, "") or rec.get("kernel_name") not in (None, ""):
                        self._kernel_launch_metadata["with_kernel_name"] += 1
                    has_launch_shape = any(
                        rec.get(key) not in (None, "")
                        for key in (
                            "grid_x",
                            "grid_y",
                            "grid_z",
                            "block_x",
                            "block_y",
                            "block_z",
                            "shared_mem",
                        )
                    )
                    if has_launch_shape:
                        self._kernel_launch_metadata["with_launch_shape"] += 1
                    if rec.get("stream_id") not in (None, ""):
                        self._kernel_launch_metadata["with_stream_id"] += 1
                    if _parse_optional_float(rec.get("host_duration_us")) is not None:
                        self._kernel_launch_metadata["with_host_duration"] += 1
                entries.append(
                    {
                        "id": entry_id,
                        "ts": ts_int,
                        "ordinal": ordinal,
                        "api": api,
                        "typ": typ,
                        "pid": pid,
                        "tid": tid,
                        "payload": feature_payload,
                        "recorded_has_wrapper_timing_field": _has_wrapper_timing_field(rec),
                        "recorded_has_explicit_direct_runtime_field": (
                            _explicit_direct_runtime_observation_us(rec) is not None
                        ),
                        "recorded_wrapper_us": _raw_observed_wrapper_runtime_us(rec),
                        "recorded_direct_runtime_us": _observed_wrapper_runtime_us(
                            rec,
                            api_name=api,
                            typ=typ,
                        ),
                        "wrapper_runtime_contract": _wrapper_runtime_contract(
                            rec,
                            api_name=api,
                            typ=typ,
                        ),
                        "observed_us": 0.1,
                        "target_us": 0.0,
                        "device_weight": 1.0,
                    }
                )
                thread_events[(pid, tid)].append(entry_id)

        if not entries:
            return

        for stream_ids in thread_events.values():
            stream_ids.sort(key=lambda event_id: (entries[event_id]["ts"], entries[event_id]["ordinal"]))
            if not stream_ids:
                continue

            segments: list[tuple[int, int, int]] = []
            seg_start = 0
            for i in range(1, len(stream_ids)):
                if entries[stream_ids[i]]["ts"] != entries[stream_ids[seg_start]]["ts"]:
                    segments.append((seg_start, i, int(entries[stream_ids[seg_start]]["ts"])))
                    seg_start = i
            segments.append((seg_start, len(stream_ids), int(entries[stream_ids[seg_start]]["ts"])))
            previous_api = ""

            for si in range(len(segments) - 1):
                s_start, s_end, s_ts = segments[si]
                _, _, next_ts = segments[si + 1]
                delta_us = max(float(next_ts - s_ts), 0.0)
                seg_event_ids = stream_ids[s_start:s_end]

                weighted_events: list[tuple[int, str, str, dict[str, Any], float]] = []
                total_weight = 0.0
                for event_id in seg_event_ids:
                    api = str(entries[event_id]["api"])
                    typ = str(entries[event_id]["typ"])
                    feature_payload = dict(entries[event_id]["payload"])
                    feature_payload = dict(feature_payload)
                    if previous_api:
                        feature_payload["prev_api"] = previous_api
                    weight = _distribution_weight(api, typ)
                    weighted_events.append((event_id, api, typ, feature_payload, weight))
                    total_weight += weight
                    previous_api = api
                if total_weight <= 0.0:
                    total_weight = float(len(weighted_events)) or 1.0
                    weighted_events = [
                        (event_id, api, typ, feature_payload, 1.0)
                        for event_id, api, typ, feature_payload, _ in weighted_events
                    ]

                for event_id, api, typ, feature_payload, weight in weighted_events:
                    recorded_host_us = entries[event_id]["recorded_direct_runtime_us"]
                    fallback_us = (weight / total_weight) * delta_us
                    share_us = _resolved_observed_target_us(
                        api,
                        typ,
                        recorded_host_us,
                        fallback_us,
                    )
                    entries[event_id]["payload"] = feature_payload
                    entries[event_id]["observed_us"] = share_us
                    entries[event_id]["target_us"] += _normalize_observed_target_us(api, typ, share_us)

            # Last segment: assign minimal duration
            for event_id in stream_ids[segments[-1][0]:segments[-1][1]]:
                api = str(entries[event_id]["api"])
                typ = str(entries[event_id]["typ"])
                feature_payload = dict(entries[event_id]["payload"])
                feature_payload = dict(feature_payload)
                if previous_api:
                    feature_payload["prev_api"] = previous_api
                entries[event_id]["payload"] = feature_payload
                recorded_host_us = entries[event_id]["recorded_direct_runtime_us"]
                final_observed_us = _resolved_observed_target_us(
                    api,
                    typ,
                    recorded_host_us,
                    0.1,
                )
                entries[event_id]["observed_us"] = final_observed_us
                entries[event_id]["target_us"] += _normalize_observed_target_us(
                    api,
                    typ,
                    final_observed_us,
                )
                previous_api = api

        handle_streams: dict[str, str] = {}
        pending_by_stream: dict[str, list[int]] = defaultdict(list)
        active_pending_ids: set[int] = set()
        event_record_pending: dict[str, tuple[int, ...]] = {}

        for entry in sorted(entries, key=lambda item: (item["ts"], item["ordinal"])):
            event_id = int(entry["id"])
            api = str(entry["api"])
            typ = str(entry["typ"])
            payload = dict(entry["payload"])

            if api == "cublasCreate_v2":
                handle_id = payload.get("handle_id")
                if handle_id not in (None, "", "0", "0x0"):
                    handle_streams[str(handle_id)] = "__default_stream__"
            elif api == "cublasSetStream_v2":
                handle_id = payload.get("handle_id")
                if handle_id not in (None, "", "0", "0x0"):
                    handle_streams[str(handle_id)] = _stream_id_from_payload(api, payload, handle_streams)

            if api in {"cudaEventRecord", "cudaEventRecordWithFlags"}:
                event_token = payload.get("event_id")
                if event_token not in (None, "", "0", "0x0"):
                    stream_id = _stream_id_from_payload(api, payload, handle_streams)
                    event_record_pending[str(event_token)] = tuple(pending_by_stream.get(stream_id, ()))
            elif api == "cudaStreamWaitEvent":
                event_token = payload.get("event_id")
                if event_token not in (None, "", "0", "0x0"):
                    stream_id = _stream_id_from_payload(api, payload, handle_streams)
                    imported_pending_ids = tuple(event_record_pending.get(str(event_token), ()))
                    if imported_pending_ids:
                        existing_ids = set(pending_by_stream.get(stream_id, ()))
                        for pending_event_id in imported_pending_ids:
                            if pending_event_id in active_pending_ids and pending_event_id not in existing_ids:
                                pending_by_stream[stream_id].append(pending_event_id)
                                existing_ids.add(pending_event_id)

            if _is_device_work_event(api, typ):
                stream_id = _stream_id_from_payload(api, payload, handle_streams)
                entry["device_weight"] = _device_work_weight(api, typ, payload)
                pending_by_stream[stream_id].append(event_id)
                active_pending_ids.add(event_id)
                continue

            if not _is_blocking_sync_api(api):
                continue

            wait_budget_us = max(float(entry["observed_us"]) - float(entry["target_us"]), 0.0)
            if wait_budget_us <= 0.0:
                continue

            candidate_ids: tuple[int, ...] = ()
            if api == "cudaStreamSynchronize":
                stream_id = _stream_id_from_payload(api, payload, handle_streams)
                candidate_ids = tuple(pending_by_stream.get(stream_id, ()))
            elif api == "cudaEventSynchronize":
                event_token = payload.get("event_id")
                if event_token not in (None, "", "0", "0x0"):
                    candidate_ids = tuple(event_record_pending.get(str(event_token), ()))
            elif api == "cudaDeviceSynchronize":
                candidate_ids = tuple(sorted(active_pending_ids))

            _distribute_wait_budget(entries, active_pending_ids, candidate_ids, wait_budget_us)
            _remove_pending_ids(pending_by_stream, active_pending_ids, candidate_ids)

        for entry in entries:
            api = str(entry["api"])
            typ = str(entry["typ"])
            final_target_us = max(min(float(entry["target_us"]), 1_000_000.0), 0.0)
            self._raw[(api, typ)].append(final_target_us)
            feature_payload = dict(entry["payload"])
            recorded_direct_runtime_us = entry.get("recorded_direct_runtime_us")
            wrapper_runtime_contract = str(entry.get("wrapper_runtime_contract") or "missing")
            has_wrapper_timing_field = bool(entry.get("recorded_has_wrapper_timing_field"))
            has_explicit_direct_runtime_field = bool(
                entry.get("recorded_has_explicit_direct_runtime_field")
            )
            has_direct_wrapper_runtime = (
                recorded_direct_runtime_us is not None and float(recorded_direct_runtime_us) > 0.0
            )
            feature_payload["has_wrapper_timing_field"] = 1 if has_wrapper_timing_field else 0
            feature_payload["has_positive_observed_wrapper_runtime"] = (
                1 if has_direct_wrapper_runtime else 0
            )
            if wrapper_runtime_contract != "missing":
                feature_payload["wrapper_runtime_contract"] = wrapper_runtime_contract
            if has_direct_wrapper_runtime:
                feature_payload["observed_wrapper_us"] = float(recorded_direct_runtime_us)
                feature_payload["observed_wrapper_log2_bucket"] = _observed_wrapper_log2_bucket(
                    float(recorded_direct_runtime_us)
                )
            entry["payload"] = feature_payload
            feature_sample = _learned_provider_features(feature_payload)
            if typ == "nccl_collective":
                collective_payload = canonicalize_gpu_estimator_event(feature_payload)
                for payload_key, metadata_key in (
                    ("collective_group_id", "_collective_group_id"),
                    ("collective_communicator_id", "_collective_communicator_id"),
                    ("collective_sequence_number", "_collective_sequence_number"),
                ):
                    value = collective_payload.get(payload_key)
                    if value not in (None, ""):
                        feature_sample[metadata_key] = value
            modeled_operator = is_modeled_operator_event(entry["payload"])
            if modeled_operator:
                self._feature_samples.append(feature_sample)
                self._feature_targets_us.append(final_target_us)
                family = event_operator_family(entry["payload"])
                label = event_operator_label(entry["payload"])
                self._operator_family_totals_us[family] += final_target_us
                self._operator_family_counts[family] += 1
                self._operator_family_examples[family][label] += final_target_us
                self._transparent_profiling_metadata["modeled_event_count"] += 1
                self._transparent_profiling_metadata["modeled_target_us_total"] += final_target_us
                if has_wrapper_timing_field:
                    self._transparent_profiling_metadata[
                        "modeled_event_with_wrapper_timing_field_count"
                    ] += 1
                if has_explicit_direct_runtime_field:
                    self._transparent_profiling_metadata[
                        "modeled_event_with_explicit_direct_runtime_field_count"
                    ] += 1
                if wrapper_runtime_contract == "dispatch_only":
                    self._transparent_profiling_metadata[
                        "modeled_event_with_dispatch_only_wrapper_contract_count"
                    ] += 1
                if wrapper_runtime_contract == "direct_runtime":
                    self._transparent_profiling_metadata[
                        "modeled_event_with_direct_runtime_contract_count"
                    ] += 1
                if has_direct_wrapper_runtime:
                    self._transparent_profiling_metadata[
                        "modeled_event_with_direct_wrapper_runtime_count"
                    ] += 1
                    self._transparent_profiling_metadata[
                        "modeled_target_us_with_direct_wrapper_runtime"
                    ] += final_target_us
                    self._transparent_profiling_metadata[
                        "modeled_observed_wrapper_us_total"
                    ] += float(recorded_direct_runtime_us)
                if _is_async_modeled_device_work_event(api, typ):
                    self._transparent_profiling_metadata["async_modeled_device_event_count"] += 1
                    self._transparent_profiling_metadata[
                        "async_modeled_device_target_us_total"
                    ] += final_target_us
                    if has_wrapper_timing_field:
                        self._transparent_profiling_metadata[
                            "async_modeled_device_event_with_wrapper_timing_field_count"
                        ] += 1
                    if has_explicit_direct_runtime_field:
                        self._transparent_profiling_metadata[
                            "async_modeled_device_event_with_explicit_direct_runtime_field_count"
                        ] += 1
                    if wrapper_runtime_contract == "dispatch_only":
                        self._transparent_profiling_metadata[
                            "async_modeled_device_event_with_dispatch_only_wrapper_contract_count"
                        ] += 1
                    if wrapper_runtime_contract == "direct_runtime":
                        self._transparent_profiling_metadata[
                            "async_modeled_device_event_with_direct_runtime_contract_count"
                        ] += 1
                    if has_direct_wrapper_runtime:
                        self._transparent_profiling_metadata[
                            "async_modeled_device_event_with_direct_wrapper_runtime_count"
                        ] += 1
                        self._transparent_profiling_metadata[
                            "async_modeled_device_target_us_with_direct_wrapper_runtime"
                        ] += final_target_us
                        self._transparent_profiling_metadata[
                            "async_modeled_device_observed_wrapper_us_total"
                        ] += float(recorded_direct_runtime_us)

    def _ingest_maya_trace_events(
        self,
        events: Sequence[Any],
        *,
        world_size: int = 1,
        active_window: tuple[int, int] | None = None,
        communicator_aliases: Mapping[str, str] | None = None,
        communicator_memberships: Mapping[str, tuple[int, ...]] | None = None,
    ) -> None:
        """Ingest already-loaded Maya-lite TraceEvent objects without JSONL I/O."""
        thread_events: dict[tuple[int, int], list[int]] = defaultdict(list)
        entries: list[dict[str, Any]] = []
        for event in events:
            ts_int = int(getattr(event, "ts"))
            if active_window is not None:
                start_ts, end_ts = active_window
                if ts_int < int(start_ts) or ts_int > int(end_ts):
                    continue
            api = _normalize_api_name(getattr(event, "api", ""))
            typ = _normalize_low_level_op_type(api, str(getattr(event, "op_type", "other")))
            pid = int(getattr(event, "pid", 0))
            tid = int(getattr(event, "tid", 0))
            ordinal = int(getattr(event, "ordinal", len(entries)))
            rec = {
                **dict(getattr(event, "extras", {}) or {}),
                "ts": ts_int,
                "pid": pid,
                "tid": tid,
                "mod": getattr(event, "module", ""),
                "api": api,
                "type": typ,
            }
            feature_payload = {
                **rec,
                "api": api,
                "type": typ,
                "ordinal": ordinal,
                "world_size": world_size,
            }
            feature_payload = _apply_communicator_topology_to_payload(
                feature_payload,
                communicator_aliases=communicator_aliases,
                communicator_memberships=communicator_memberships,
            )
            if rec.get("collective_api") not in (None, ""):
                feature_payload["collective_api"] = _normalize_api_name(
                    rec.get("collective_api")
                )
            entry_id = len(entries)
            if api == "cudaLaunchKernel" or typ == "kernel_launch":
                self._kernel_launch_metadata["total_kernel_launches"] += 1
                if rec.get("kernel") not in (None, "") or rec.get("kernel_name") not in (None, ""):
                    self._kernel_launch_metadata["with_kernel_name"] += 1
                has_launch_shape = any(
                    rec.get(key) not in (None, "")
                    for key in (
                        "grid_x",
                        "grid_y",
                        "grid_z",
                        "block_x",
                        "block_y",
                        "block_z",
                        "shared_mem",
                    )
                )
                if has_launch_shape:
                    self._kernel_launch_metadata["with_launch_shape"] += 1
                if rec.get("stream_id") not in (None, ""):
                    self._kernel_launch_metadata["with_stream_id"] += 1
                if _parse_optional_float(rec.get("host_duration_us")) is not None:
                    self._kernel_launch_metadata["with_host_duration"] += 1
            entries.append(
                {
                    "id": entry_id,
                    "ts": ts_int,
                    "ordinal": ordinal,
                    "api": api,
                    "typ": typ,
                    "pid": pid,
                    "tid": tid,
                    "payload": feature_payload,
                    "recorded_has_wrapper_timing_field": _has_wrapper_timing_field(rec),
                    "recorded_has_explicit_direct_runtime_field": (
                        _explicit_direct_runtime_observation_us(rec) is not None
                    ),
                    "recorded_wrapper_us": _raw_observed_wrapper_runtime_us(rec),
                    "recorded_direct_runtime_us": _observed_wrapper_runtime_us(
                        rec,
                        api_name=api,
                        typ=typ,
                    ),
                    "wrapper_runtime_contract": _wrapper_runtime_contract(
                        rec,
                        api_name=api,
                        typ=typ,
                    ),
                    "observed_us": 0.1,
                    "target_us": 0.0,
                    "device_weight": 1.0,
                }
            )
            thread_events[(pid, tid)].append(entry_id)

        if not entries:
            return

        for stream_ids in thread_events.values():
            stream_ids.sort(key=lambda event_id: (entries[event_id]["ts"], entries[event_id]["ordinal"]))
            if not stream_ids:
                continue

            segments: list[tuple[int, int, int]] = []
            seg_start = 0
            for i in range(1, len(stream_ids)):
                if entries[stream_ids[i]]["ts"] != entries[stream_ids[seg_start]]["ts"]:
                    segments.append((seg_start, i, int(entries[stream_ids[seg_start]]["ts"])))
                    seg_start = i
            segments.append((seg_start, len(stream_ids), int(entries[stream_ids[seg_start]]["ts"])))
            previous_api = ""

            for si in range(len(segments) - 1):
                s_start, s_end, s_ts = segments[si]
                _, _, next_ts = segments[si + 1]
                delta_us = max(float(next_ts - s_ts), 0.0)
                seg_event_ids = stream_ids[s_start:s_end]

                weighted_events: list[tuple[int, str, str, dict[str, Any], float]] = []
                total_weight = 0.0
                for event_id in seg_event_ids:
                    api = str(entries[event_id]["api"])
                    typ = str(entries[event_id]["typ"])
                    feature_payload = dict(entries[event_id]["payload"])
                    feature_payload = dict(feature_payload)
                    if previous_api:
                        feature_payload["prev_api"] = previous_api
                    weight = _distribution_weight(api, typ)
                    weighted_events.append((event_id, api, typ, feature_payload, weight))
                    total_weight += weight
                    previous_api = api
                if total_weight <= 0.0:
                    total_weight = float(len(weighted_events)) or 1.0
                    weighted_events = [
                        (event_id, api, typ, feature_payload, 1.0)
                        for event_id, api, typ, feature_payload, _ in weighted_events
                    ]

                for event_id, api, typ, feature_payload, weight in weighted_events:
                    recorded_host_us = entries[event_id]["recorded_direct_runtime_us"]
                    fallback_us = (weight / total_weight) * delta_us
                    share_us = _resolved_observed_target_us(
                        api,
                        typ,
                        recorded_host_us,
                        fallback_us,
                    )
                    entries[event_id]["payload"] = feature_payload
                    entries[event_id]["observed_us"] = share_us
                    entries[event_id]["target_us"] += _normalize_observed_target_us(api, typ, share_us)

            # Last segment: assign minimal duration
            for event_id in stream_ids[segments[-1][0]:segments[-1][1]]:
                api = str(entries[event_id]["api"])
                typ = str(entries[event_id]["typ"])
                feature_payload = dict(entries[event_id]["payload"])
                feature_payload = dict(feature_payload)
                if previous_api:
                    feature_payload["prev_api"] = previous_api
                entries[event_id]["payload"] = feature_payload
                recorded_host_us = entries[event_id]["recorded_direct_runtime_us"]
                final_observed_us = _resolved_observed_target_us(
                    api,
                    typ,
                    recorded_host_us,
                    0.1,
                )
                entries[event_id]["observed_us"] = final_observed_us
                entries[event_id]["target_us"] += _normalize_observed_target_us(
                    api,
                    typ,
                    final_observed_us,
                )
                previous_api = api

        handle_streams: dict[str, str] = {}
        pending_by_stream: dict[str, list[int]] = defaultdict(list)
        active_pending_ids: set[int] = set()
        event_record_pending: dict[str, tuple[int, ...]] = {}

        for entry in sorted(entries, key=lambda item: (item["ts"], item["ordinal"])):
            event_id = int(entry["id"])
            api = str(entry["api"])
            typ = str(entry["typ"])
            payload = dict(entry["payload"])

            if api == "cublasCreate_v2":
                handle_id = payload.get("handle_id")
                if handle_id not in (None, "", "0", "0x0"):
                    handle_streams[str(handle_id)] = "__default_stream__"
            elif api == "cublasSetStream_v2":
                handle_id = payload.get("handle_id")
                if handle_id not in (None, "", "0", "0x0"):
                    handle_streams[str(handle_id)] = _stream_id_from_payload(api, payload, handle_streams)

            if api in {"cudaEventRecord", "cudaEventRecordWithFlags"}:
                event_token = payload.get("event_id")
                if event_token not in (None, "", "0", "0x0"):
                    stream_id = _stream_id_from_payload(api, payload, handle_streams)
                    event_record_pending[str(event_token)] = tuple(pending_by_stream.get(stream_id, ()))
            elif api == "cudaStreamWaitEvent":
                event_token = payload.get("event_id")
                if event_token not in (None, "", "0", "0x0"):
                    stream_id = _stream_id_from_payload(api, payload, handle_streams)
                    imported_pending_ids = tuple(event_record_pending.get(str(event_token), ()))
                    if imported_pending_ids:
                        existing_ids = set(pending_by_stream.get(stream_id, ()))
                        for pending_event_id in imported_pending_ids:
                            if pending_event_id in active_pending_ids and pending_event_id not in existing_ids:
                                pending_by_stream[stream_id].append(pending_event_id)
                                existing_ids.add(pending_event_id)

            if _is_device_work_event(api, typ):
                stream_id = _stream_id_from_payload(api, payload, handle_streams)
                entry["device_weight"] = _device_work_weight(api, typ, payload)
                pending_by_stream[stream_id].append(event_id)
                active_pending_ids.add(event_id)
                continue

            if not _is_blocking_sync_api(api):
                continue

            wait_budget_us = max(float(entry["observed_us"]) - float(entry["target_us"]), 0.0)
            if wait_budget_us <= 0.0:
                continue

            candidate_ids: tuple[int, ...] = ()
            if api == "cudaStreamSynchronize":
                stream_id = _stream_id_from_payload(api, payload, handle_streams)
                candidate_ids = tuple(pending_by_stream.get(stream_id, ()))
            elif api == "cudaEventSynchronize":
                event_token = payload.get("event_id")
                if event_token not in (None, "", "0", "0x0"):
                    candidate_ids = tuple(event_record_pending.get(str(event_token), ()))
            elif api == "cudaDeviceSynchronize":
                candidate_ids = tuple(sorted(active_pending_ids))

            _distribute_wait_budget(entries, active_pending_ids, candidate_ids, wait_budget_us)
            _remove_pending_ids(pending_by_stream, active_pending_ids, candidate_ids)

        for entry in entries:
            api = str(entry["api"])
            typ = str(entry["typ"])
            final_target_us = max(min(float(entry["target_us"]), 1_000_000.0), 0.0)
            self._raw[(api, typ)].append(final_target_us)
            feature_payload = dict(entry["payload"])
            recorded_direct_runtime_us = entry.get("recorded_direct_runtime_us")
            wrapper_runtime_contract = str(entry.get("wrapper_runtime_contract") or "missing")
            has_wrapper_timing_field = bool(entry.get("recorded_has_wrapper_timing_field"))
            has_explicit_direct_runtime_field = bool(
                entry.get("recorded_has_explicit_direct_runtime_field")
            )
            has_direct_wrapper_runtime = (
                recorded_direct_runtime_us is not None and float(recorded_direct_runtime_us) > 0.0
            )
            feature_payload["has_wrapper_timing_field"] = 1 if has_wrapper_timing_field else 0
            feature_payload["has_positive_observed_wrapper_runtime"] = (
                1 if has_direct_wrapper_runtime else 0
            )
            if wrapper_runtime_contract != "missing":
                feature_payload["wrapper_runtime_contract"] = wrapper_runtime_contract
            if has_direct_wrapper_runtime:
                feature_payload["observed_wrapper_us"] = float(recorded_direct_runtime_us)
                feature_payload["observed_wrapper_log2_bucket"] = _observed_wrapper_log2_bucket(
                    float(recorded_direct_runtime_us)
                )
            entry["payload"] = feature_payload
            feature_sample = _learned_provider_features(feature_payload)
            if typ == "nccl_collective":
                collective_payload = canonicalize_gpu_estimator_event(feature_payload)
                for payload_key, metadata_key in (
                    ("collective_group_id", "_collective_group_id"),
                    ("collective_communicator_id", "_collective_communicator_id"),
                    ("collective_sequence_number", "_collective_sequence_number"),
                ):
                    value = collective_payload.get(payload_key)
                    if value not in (None, ""):
                        feature_sample[metadata_key] = value
            modeled_operator = is_modeled_operator_event(entry["payload"])
            if modeled_operator:
                self._feature_samples.append(feature_sample)
                self._feature_targets_us.append(final_target_us)
                family = event_operator_family(entry["payload"])
                label = event_operator_label(entry["payload"])
                self._operator_family_totals_us[family] += final_target_us
                self._operator_family_counts[family] += 1
                self._operator_family_examples[family][label] += final_target_us
                self._transparent_profiling_metadata["modeled_event_count"] += 1
                self._transparent_profiling_metadata["modeled_target_us_total"] += final_target_us
                if has_wrapper_timing_field:
                    self._transparent_profiling_metadata[
                        "modeled_event_with_wrapper_timing_field_count"
                    ] += 1
                if has_explicit_direct_runtime_field:
                    self._transparent_profiling_metadata[
                        "modeled_event_with_explicit_direct_runtime_field_count"
                    ] += 1
                if wrapper_runtime_contract == "dispatch_only":
                    self._transparent_profiling_metadata[
                        "modeled_event_with_dispatch_only_wrapper_contract_count"
                    ] += 1
                if wrapper_runtime_contract == "direct_runtime":
                    self._transparent_profiling_metadata[
                        "modeled_event_with_direct_runtime_contract_count"
                    ] += 1
                if has_direct_wrapper_runtime:
                    self._transparent_profiling_metadata[
                        "modeled_event_with_direct_wrapper_runtime_count"
                    ] += 1
                    self._transparent_profiling_metadata[
                        "modeled_target_us_with_direct_wrapper_runtime"
                    ] += final_target_us
                    self._transparent_profiling_metadata[
                        "modeled_observed_wrapper_us_total"
                    ] += float(recorded_direct_runtime_us)
                if _is_async_modeled_device_work_event(api, typ):
                    self._transparent_profiling_metadata["async_modeled_device_event_count"] += 1
                    self._transparent_profiling_metadata[
                        "async_modeled_device_target_us_total"
                    ] += final_target_us
                    if has_wrapper_timing_field:
                        self._transparent_profiling_metadata[
                            "async_modeled_device_event_with_wrapper_timing_field_count"
                        ] += 1
                    if has_explicit_direct_runtime_field:
                        self._transparent_profiling_metadata[
                            "async_modeled_device_event_with_explicit_direct_runtime_field_count"
                        ] += 1
                    if wrapper_runtime_contract == "dispatch_only":
                        self._transparent_profiling_metadata[
                            "async_modeled_device_event_with_dispatch_only_wrapper_contract_count"
                        ] += 1
                    if wrapper_runtime_contract == "direct_runtime":
                        self._transparent_profiling_metadata[
                            "async_modeled_device_event_with_direct_runtime_contract_count"
                        ] += 1
                    if has_direct_wrapper_runtime:
                        self._transparent_profiling_metadata[
                            "async_modeled_device_event_with_direct_wrapper_runtime_count"
                        ] += 1
                        self._transparent_profiling_metadata[
                            "async_modeled_device_target_us_with_direct_wrapper_runtime"
                        ] += final_target_us
                        self._transparent_profiling_metadata[
                            "async_modeled_device_observed_wrapper_us_total"
                        ] += float(recorded_direct_runtime_us)


    def _ingest_items(
        self,
        ingest_items: list[
            tuple[
                str,
                int,
                tuple[int, int] | None,
                Mapping[str, str],
                Mapping[str, tuple[int, ...]],
            ]
        ],
        *,
        fit_workers: int = 1,
    ) -> None:
        if not ingest_items:
            return
        worker_count = max(int(fit_workers), 1)
        use_parallel = (
            worker_count > 1
            and len(ingest_items) > 1
            and os.name != "nt"
        )
        if not use_parallel:
            for (
                path_str,
                world_size,
                active_window,
                communicator_aliases,
                communicator_memberships,
            ) in ingest_items:
                self._ingest_file(
                    Path(path_str),
                    world_size=world_size,
                    active_window=active_window,
                    communicator_aliases=communicator_aliases,
                    communicator_memberships=communicator_memberships,
                )
            return

        chunks = _split_contiguous_chunks(ingest_items, worker_count)
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(_fit_trace_chunk, chunk_index, chunk)
                for chunk_index, chunk in enumerate(chunks)
            ]
            partials = [future.result() for future in futures]
        for partial in sorted(partials, key=lambda item: int(item["chunk_index"])):
            self._merge_fit_partial(partial)

    def _merge_fit_partial(self, partial: Mapping[str, Any]) -> None:
        for key_str, durations in partial.get("raw", {}).items():
            parts = _split_serialized_estimator_key(key_str)
            if parts is None:
                continue
            api, typ = _normalize_estimator_key(parts[0], parts[1])
            self._raw[(api, typ)].extend(float(value) for value in durations)
        self._feature_samples.extend(list(partial.get("feature_samples", ())))
        self._feature_targets_us.extend(
            float(value) for value in partial.get("feature_targets_us", ())
        )
        for family, total_us in partial.get("operator_family_totals_us", {}).items():
            self._operator_family_totals_us[str(family)] += float(total_us)
        for family, count in partial.get("operator_family_counts", {}).items():
            self._operator_family_counts[str(family)] += int(count)
        for family, examples in partial.get("operator_family_examples", {}).items():
            family_key = str(family)
            for label, total_us in dict(examples).items():
                self._operator_family_examples[family_key][str(label)] += float(total_us)
        for key, value in partial.get("kernel_launch_metadata", {}).items():
            if key in self._kernel_launch_metadata:
                self._kernel_launch_metadata[key] += int(value)
        for key, value in partial.get("transparent_profiling_metadata", {}).items():
            if key in self._transparent_profiling_metadata:
                self._transparent_profiling_metadata[key] += value

    def _compute_stats(self):
        """Compute summary statistics from raw durations."""
        type_raw: dict[str, list[float]] = defaultdict(list)
        all_durations = []

        for (api, typ), durations in self._raw.items():
            if len(durations) < 2:
                p50 = durations[0] if durations else 1.0
                self._stats[(api, typ)] = {
                    "p50": p50, "mean": p50, "p95": p50,
                    "count": len(durations),
                }
            else:
                sorted_d = sorted(durations)
                p50 = statistics.median(sorted_d)
                mean = statistics.mean(sorted_d)
                p95_idx = int(len(sorted_d) * 0.95)
                p95 = sorted_d[min(p95_idx, len(sorted_d) - 1)]
                self._stats[(api, typ)] = {
                    "p50": p50, "mean": mean, "p95": p95,
                    "count": len(durations),
                }

            type_raw[typ].extend(durations)
            all_durations.extend(durations)

        # Type-level aggregates
        for typ, durations in type_raw.items():
            sorted_d = sorted(durations)
            p50 = statistics.median(sorted_d)
            mean = statistics.mean(sorted_d)
            p95_idx = int(len(sorted_d) * 0.95)
            p95 = sorted_d[min(p95_idx, len(sorted_d) - 1)]
            self._type_stats[typ] = {
                "p50": p50, "mean": mean, "p95": p95,
                "count": len(durations),
            }

        # Global fallback
        if all_durations:
            self._global_p50 = statistics.median(sorted(all_durations))

    def estimate(self, api: str, typ: str = "other", percentile: str = "p50") -> float:
        """
        Estimate duration for an API call in microseconds.

        Lookup order: (api, type) -> type -> global fallback.
        """
        api, typ = _normalize_estimator_key(api, typ)
        key = (api, typ)
        if key in self._stats:
            return self._stats[key].get(percentile, self._stats[key]["p50"])
        if typ in self._type_stats:
            return self._type_stats[typ].get(percentile, self._type_stats[typ]["p50"])
        return self._global_p50

    def estimate_ns(self, api: str, typ: str = "other", percentile: str = "p50") -> int:
        """Estimate duration in nanoseconds (for C++ integration)."""
        return int(self.estimate(api, typ, percentile) * 1000)

    def estimate_event(
        self,
        event: Mapping[str, Any],
        percentile: str = "p50",
    ) -> float:
        return self.estimate_event_with_details(event, percentile=percentile).duration_us

    def estimate_event_with_details(
        self,
        event: Mapping[str, Any],
        percentile: str = "p50",
    ) -> EstimatorDecision:
        """
        Estimate an event duration in microseconds.

        Event-level providers get the first chance when rich operator
        metadata is present. If none apply, we fall back to the
        trace-derived (api, type) statistics.
        """
        normalized_event = _canonicalize_event_for_provider(event)
        for provider in self._providers:
            try:
                duration_us = provider.estimate_us(normalized_event, percentile)
            except Exception:
                continue
            if duration_us is None:
                continue
            if math.isfinite(duration_us) and duration_us >= 0:
                return EstimatorDecision(
                    duration_us=float(duration_us),
                    source="provider",
                    calibrated=True,
                    provider_name=getattr(provider, "name", type(provider).__name__),
                )

        api = _normalize_api_name(
            normalized_event.get("api")
            or normalized_event.get("api_name")
            or normalized_event.get("name")
            or ""
        )
        typ = _normalize_low_level_op_type(api, str(normalized_event.get("type") or "other"))
        key = (api, typ)
        if key in self._stats:
            return EstimatorDecision(
                duration_us=float(self._stats[key].get(percentile, self._stats[key]["p50"])),
                source="api_stats",
                calibrated=True,
            )
        if typ in self._type_stats:
            return EstimatorDecision(
                duration_us=float(self._type_stats[typ].get(percentile, self._type_stats[typ]["p50"])),
                source="type_stats",
                calibrated=True,
            )
        return EstimatorDecision(
            duration_us=float(self._global_p50),
            source="global_fallback",
            calibrated=self.is_calibrated(),
        )

    def estimate_collective_group_with_details(
        self,
        event: Mapping[str, Any],
        percentile: str = "p50",
    ) -> EstimatorDecision | None:
        """
        Estimate a collective group's completion time from group-stable providers.

        This intentionally does not fall back to flat `(api, type)` tables. It is
        meant to surface topology-aware group-level timing only when a provider
        explicitly advertises support for collective-group timing.
        """
        return self.estimate_collective_groups_with_details(
            [event],
            percentile=percentile,
        )[0]

    def estimate_collective_groups_with_details(
        self,
        events: list[Mapping[str, Any]],
        percentile: str = "p50",
    ) -> list[EstimatorDecision | None]:
        """
        Estimate collective-group completion times in provider-sized batches.

        This preserves the same provider order and per-event decisions as
        ``estimate_collective_group_with_details`` while allowing deterministic
        providers to evaluate many group-level events in one model call.
        """
        if not events:
            return []
        normalized_events = [
            _canonicalize_collective_group_event(event)
            for event in events
        ]
        decisions: list[EstimatorDecision | None] = [None for _ in normalized_events]
        remaining_indexes = list(range(len(normalized_events)))
        for provider in self._providers:
            if not getattr(provider, "supports_collective_group_timing", False):
                continue
            provider_name = getattr(provider, "name", type(provider).__name__)
            next_remaining: list[int] = []
            batch_estimator = getattr(provider, "estimate_many_us", None)
            if callable(batch_estimator):
                try:
                    durations = batch_estimator(
                        [normalized_events[index] for index in remaining_indexes],
                        percentile,
                    )
                except Exception:
                    durations = [None for _ in remaining_indexes]
                for index, duration_us in zip(remaining_indexes, durations):
                    if (
                        duration_us is not None
                        and math.isfinite(float(duration_us))
                        and float(duration_us) >= 0
                    ):
                        decisions[index] = EstimatorDecision(
                            duration_us=float(duration_us),
                            source="provider",
                            calibrated=True,
                            provider_name=provider_name,
                        )
                    else:
                        next_remaining.append(index)
                remaining_indexes = next_remaining
                if not remaining_indexes:
                    break
                continue

            for index in remaining_indexes:
                normalized_event = normalized_events[index]
                try:
                    duration_us = provider.estimate_us(normalized_event, percentile)
                except Exception:
                    duration_us = None
                if (
                    duration_us is not None
                    and math.isfinite(float(duration_us))
                    and float(duration_us) >= 0
                ):
                    decisions[index] = EstimatorDecision(
                        duration_us=float(duration_us),
                        source="provider",
                        calibrated=True,
                        provider_name=provider_name,
                    )
                else:
                    next_remaining.append(index)
            remaining_indexes = next_remaining
            if not remaining_indexes:
                break
        return decisions

    def annotate_trace(self, trace_path: str, percentile: str = "p50") -> list[dict]:
        """
        Load a trace file and annotate each event with estimated duration.

        Returns list of dicts with added 'dur_us' field.
        """
        events = []
        with open(trace_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec["dur_us"] = self.estimate_event(rec, percentile)
                events.append(rec)
        return events

    def annotate_trace_to_file(
        self, trace_path: str, output_path: str, percentile: str = "p50"
    ):
        """Annotate a trace and write to output JSONL file."""
        events = self.annotate_trace(trace_path, percentile)
        with open(output_path, "w") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

    def to_lookup_table(self) -> dict[str, dict]:
        """
        Export as a flat lookup table for serialization.

        Returns: { "api::type": {p50, mean, p95, count}, ... }
        """
        table = {}
        for (api, typ), stats in sorted(self._stats.items()):
            table[f"{api}::{typ}"] = stats
        return table

    def save(self, path: str):
        """Save estimator to JSON file."""
        data = {
            "api_stats": {
                f"{api}::{typ}": stats
                for (api, typ), stats in self._stats.items()
            },
            "type_stats": self._type_stats,
            "global_p50_us": self._global_p50,
            "providers": [
                provider.to_jsonable()
                for provider in self._providers
                if hasattr(provider, "to_jsonable")
            ],
            "provenance": self._provenance,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(
        cls,
        path: str,
        providers: Optional[list[EventTimingProvider]] = None,
        gpu_estimator_bundle: str | Path | None = None,
    ) -> "Estimator":
        """Load estimator from JSON file."""
        est = cls(providers=providers)
        with open(path) as f:
            data = json.load(f)

        canonical_api_stats: list[tuple[tuple[str, str], dict[str, Any]]] = []
        alias_api_stats: list[tuple[tuple[str, str], dict[str, Any]]] = []
        for key_str, stats in data.get("api_stats", {}).items():
            parts = _split_serialized_estimator_key(key_str)
            if parts is None:
                continue
            raw_api, raw_typ = parts
            normalized_key = _normalize_estimator_key(raw_api, raw_typ)
            stats_payload = dict(stats)
            if normalized_key == (raw_api, raw_typ):
                canonical_api_stats.append((normalized_key, stats_payload))
            else:
                alias_api_stats.append((normalized_key, stats_payload))

        for key, stats in canonical_api_stats:
            est._stats[key] = stats
        for key, stats in alias_api_stats:
            est._stats.setdefault(key, stats)

        est._type_stats = data.get("type_stats", {})
        est._global_p50 = data.get("global_p50_us", 1.0)
        est._provenance = dict(data.get("provenance", {}))
        for provider_payload in data.get("providers", []):
            provider_type = provider_payload.get("type")
            if provider_type == "trace_learned_sklearn":
                try:
                    est.add_provider(TraceLearnedTimingProvider.from_jsonable(provider_payload))
                except Exception as exc:
                    est._provider_diagnostics["trace_learned_sklearn"] = (
                        f"failed to restore saved sklearn provider: {exc!r}"
                    )
            elif provider_type == "trace_signature_stats":
                est.add_provider(TraceSignatureTimingProvider.from_jsonable(provider_payload))
            elif provider_type == "gpu_estimator_xgboost":
                bundle_dir = (
                    gpu_estimator_bundle
                    if gpu_estimator_bundle is not None
                    else provider_payload.get("bundle_dir", DEFAULT_GPU_ESTIMATOR_BUNDLE)
                )
                status = probe_gpu_estimator_provider(bundle_dir)
                provider = status.provider
                if provider is not None:
                    est.add_provider(provider)
                elif status.error:
                    est._provider_diagnostics["gpu_estimator_xgboost"] = status.error
        return est

    def summary(self, top_n: int = 30):
        """Print summary of learned durations."""
        print(f"Estimator: {len(self._stats)} (api, type) entries, "
              f"{len(self._type_stats)} type entries")
        print(f"Global p50: {self._global_p50:.1f} us")
        print()

        # Sort by total observed time (count * p50) descending
        ranked = sorted(
            self._stats.items(),
            key=lambda x: x[1]["count"] * x[1]["p50"],
            reverse=True,
        )

        print(f"{'API':<45} {'Type':<20} {'p50(us)':>10} {'p95(us)':>10} {'Count':>8}")
        print("-" * 97)
        for (api, typ), s in ranked[:top_n]:
            print(f"{api:<45} {typ:<20} {s['p50']:>10.1f} {s['p95']:>10.1f} {s['count']:>8}")

        print()
        print(f"{'Type':<20} {'p50(us)':>10} {'p95(us)':>10} {'Count':>8}")
        print("-" * 52)
        for typ, s in sorted(self._type_stats.items(), key=lambda x: -x[1]["count"]):
            print(f"{typ:<20} {s['p50']:>10.1f} {s['p95']:>10.1f} {s['count']:>8}")


def _fit_trace_chunk(
    chunk_index: int,
    ingest_items: list[
        tuple[
            str,
            int,
            tuple[int, int] | None,
            Mapping[str, str],
            Mapping[str, tuple[int, ...]],
        ]
    ],
) -> dict[str, Any]:
    est = Estimator()
    for (
        path_str,
        world_size,
        active_window,
        communicator_aliases,
        communicator_memberships,
    ) in ingest_items:
        est._ingest_file(
            Path(path_str),
            world_size=world_size,
            active_window=active_window,
            communicator_aliases=communicator_aliases,
            communicator_memberships=communicator_memberships,
        )
    return {
        "chunk_index": int(chunk_index),
        "raw": {
            f"{api}::{typ}": list(durations)
            for (api, typ), durations in est._raw.items()
        },
        "feature_samples": list(est._feature_samples),
        "feature_targets_us": list(est._feature_targets_us),
        "operator_family_totals_us": dict(est._operator_family_totals_us),
        "operator_family_counts": dict(est._operator_family_counts),
        "operator_family_examples": {
            family: dict(examples)
            for family, examples in est._operator_family_examples.items()
        },
        "kernel_launch_metadata": dict(est._kernel_launch_metadata),
        "transparent_profiling_metadata": dict(est._transparent_profiling_metadata),
    }


def fit_all_experiments(
    base_dir: str = "paper/traces/real",
    providers: Optional[list[EventTimingProvider]] = None,
) -> Estimator:
    """Convenience: fit from all real trace experiments."""
    base = Path(base_dir)
    dirs = sorted(d for d in base.iterdir() if d.is_dir())
    return Estimator.fit_from_multiple([str(d) for d in dirs], providers=providers)


# --- Integration with CppEvent simulation ---

def simulate_with_estimator(
    est: Estimator,
    fake_trace_dir: str,
    world_size: int = 2,
    percentile: str = "p50",
) -> dict:
    """
    Simulate a fake trace using data-driven durations.

    Loads fake traces, annotates with estimated durations,
    then runs a simple per-rank timeline simulation.

    Returns simulation result dict compatible with flexsim.simulate output.
    """
    trace_path = Path(fake_trace_dir)
    rank_files = sorted(trace_path.glob("rank_*.jsonl"))

    if not rank_files:
        raise FileNotFoundError(f"No rank_*.jsonl in {fake_trace_dir}")

    # Collect per-rank events with durations
    rank_events: dict[int, list[dict]] = {}
    for rf in rank_files:
        rank_id = int(rf.stem.split("_")[1])
        rank_events[rank_id] = est.annotate_trace(str(rf), percentile)

    # Simple sequential simulation per rank
    rank_metrics = []
    for rank_id in sorted(rank_events.keys()):
        events = rank_events[rank_id]
        compute_us = 0.0
        comm_us = 0.0
        mem_us = 0.0
        other_us = 0.0

        for ev in events:
            dur = ev["dur_us"]
            typ = ev.get("type", "other")
            if typ in ("kernel_launch", "blas_compute"):
                compute_us += dur
            elif typ == "nccl_collective":
                comm_us += dur
            elif typ in ("mem_copy", "mem_alloc"):
                mem_us += dur
            else:
                other_us += dur

        total_us = compute_us + comm_us + mem_us + other_us
        rank_metrics.append({
            "rank": rank_id,
            "compute_time_ms": compute_us / 1000,
            "communication_time_ms": comm_us / 1000,
            "memory_time_ms": mem_us / 1000,
            "other_time_ms": other_us / 1000,
            "total_time_ms": total_us / 1000,
            "num_events": len(events),
            "utilization": compute_us / total_us if total_us > 0 else 0.0,
        })

    # Cross-rank: total time is max across ranks (critical path)
    total_ms = max(rm["total_time_ms"] for rm in rank_metrics) if rank_metrics else 0
    critical_path_ms = total_ms  # simplified: no dependency tracking here

    return {
        "total_time_ms": total_ms,
        "critical_path_ms": critical_path_ms,
        "success": True,
        "rank_metrics": rank_metrics,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FlexSim Data-Driven Estimator")
    sub = parser.add_subparsers(dest="cmd")

    # fit
    fit_p = sub.add_parser("fit", help="Fit estimator from real traces")
    fit_p.add_argument("trace_dir", help="Directory with rank_*.jsonl (or parent with e1/e2/...)")
    fit_p.add_argument("-o", "--output", default="estimator.json", help="Output file")
    fit_p.add_argument("--all", action="store_true", help="Fit from all subdirectories")

    # estimate
    est_p = sub.add_parser("estimate", help="Estimate duration for an API call")
    est_p.add_argument("model", help="Estimator JSON file")
    est_p.add_argument("api", help="API function name")
    est_p.add_argument("--type", default="other", help="Operation type")

    # annotate
    ann_p = sub.add_parser("annotate", help="Annotate a fake trace with durations")
    ann_p.add_argument("model", help="Estimator JSON file")
    ann_p.add_argument("trace", help="Fake trace JSONL file")
    ann_p.add_argument("-o", "--output", help="Output JSONL file")

    # simulate
    sim_p = sub.add_parser("simulate", help="Simulate fake trace with data-driven model")
    sim_p.add_argument("model", help="Estimator JSON file")
    sim_p.add_argument("trace_dir", help="Directory with fake rank_*.jsonl files")
    sim_p.add_argument("-w", "--world-size", type=int, default=2)

    # summary
    sum_p = sub.add_parser("summary", help="Print estimator summary")
    sum_p.add_argument("model", help="Estimator JSON file")
    sum_p.add_argument("-n", "--top", type=int, default=30)

    args = parser.parse_args()

    if args.cmd == "fit":
        if args.all:
            est = fit_all_experiments(args.trace_dir)
        else:
            est = Estimator.fit_from_traces(args.trace_dir)
        est.save(args.output)
        print(f"Saved estimator to {args.output}")
        est.summary()

    elif args.cmd == "estimate":
        est = Estimator.load(args.model)
        dur = est.estimate(args.api, args.type)
        print(f"{args.api} ({args.type}): {dur:.1f} us")

    elif args.cmd == "annotate":
        est = Estimator.load(args.model)
        out = args.output or args.trace.replace(".jsonl", "_annotated.jsonl")
        est.annotate_trace_to_file(args.trace, out)
        print(f"Annotated trace saved to {out}")

    elif args.cmd == "simulate":
        est = Estimator.load(args.model)
        result = simulate_with_estimator(est, args.trace_dir, args.world_size)
        print(json.dumps(result, indent=2))

    elif args.cmd == "summary":
        est = Estimator.load(args.model)
        est.summary(args.top)

    else:
        parser.print_help()
