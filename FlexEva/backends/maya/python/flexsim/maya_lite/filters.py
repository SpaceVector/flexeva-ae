"""
Low-level event filters shared across Maya-lite stages.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache


_SETUP_API_PREFIXES = (
    "__cudaRegister",
    "__cudaPopCallConfiguration",
    "__cudaPushCallConfiguration",
    "__cudaUnregister",
)

_CANONICAL_API_ALIASES = {
    "cudaEventRecordWithFlags": "cudaEventRecord",
    "cudaEventCreate": "cudaEventCreateWithFlags",
    "cudaStreamCreateWithFlags": "cudaStreamCreate",
    "cudaStreamCreateWithPriority": "cudaStreamCreate",
    "ncclBcast": "ncclBroadcast",
}

_COMPAT_ONLY_APIS = {
    "cublasCreate_v2",
    "cublasLtCreate",
    "cudaGetDevice",
    "cudaGetDeviceCount",
    "cudaGetDeviceProperties",
    "cudaDeviceGetStreamPriorityRange",
    "cudaGetLastError",
    "cudaSetDevice",
    "cudaStreamCreate",
    "cudaStreamIsCapturing",
    "cudaEventCreateWithFlags",
    "cudaEventDestroy",
    "cublasSetMathMode",
    "cublasSetWorkspace_v2",
    "cublasLtMatmulDescCreate",
    "cublasLtMatmulDescSetAttribute",
    "cublasLtMatmulPreferenceCreate",
    "cublasLtMatmulPreferenceSetAttribute",
    "cublasLtMatrixLayoutCreate",
    "cublasLtMatrixLayoutDestroy",
    "cuCtxGetCurrent",
    "cuDevicePrimaryCtxGetState",
    "ncclCommGetAsyncError",
    "ncclCommCount",
    "ncclCommUserRank",
    "ncclGroupEnd",
    "ncclGroupStart",
    "ncclGetUniqueId",
    "ncclGetVersion",
}

_COMPAT_ONLY_APIS |= {
    "cublasDestroy_v2",
    "cublasLtDestroy",
    "cublasLtMatmulDescDestroy",
    "cublasLtMatmulPreferenceDestroy",
    "cublasLtMatrixLayoutDestroy",
    "cudaStreamDestroy",
    "ncclCommDestroy",
}

_LOW_OVERHEAD_APIS = {
    "cublasSetStream_v2",
    "cudaEventRecord",
    "cudaEventRecordWithFlags",
    "cudaEventQuery",
    "cudaStreamWaitEvent",
}

_STREAM_DEVICE_OP_TYPES = {
    "kernel_launch",
    "blas_compute",
    "nccl_collective",
    "mem_copy",
}

_THREAD_SYNC_APIS = {
    "cudaEventSynchronize",
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
}

_CUDA_EVENT_RECORD_APIS = {
    "cudaEventRecord",
}

_HANDLE_HOST_APIS = {
    "cublasCreate_v2",
    "cublasSetStream_v2",
}


_COLLECTIVE_APIS = {
    "ncclAllGather",
    "ncclAllReduce",
    "ncclAllToAll",
    "ncclAllToAllv",
    "ncclBcast",
    "ncclBroadcast",
    "ncclGather",
    "ncclRecv",
    "ncclReduce",
    "ncclReduceScatter",
    "ncclScatter",
    "ncclSend",
}

_SEMANTIC_TRACED_APIS = {
    "cudaLaunchKernel",
    "cudaMemcpy",
    "cudaMemcpyAsync",
    "cudaMalloc",
    "cudaMallocAsync",
    "cudaFree",
    "cudaFreeAsync",
    "cudaMemGetInfo",
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaStreamWaitEvent",
    "cudaEventRecord",
    "cudaEventQuery",
    "cudaEventSynchronize",
    "cublasSetStream_v2",
    "cublasGemmEx",
    "cublasGemmStridedBatchedEx",
    "cublasGemmBatchedEx",
    "cublasLtMatmul",
    "ncclCommInitRank",
    "ncclCommInitRankConfig",
    "ncclAllGather",
    "ncclAllReduce",
    "ncclAllToAll",
    "ncclAllToAllv",
    "ncclBroadcast",
    "ncclReduce",
    "ncclReduceScatter",
    "ncclSend",
    "ncclRecv",
}


class TraceApiBucket(str, Enum):
    SEMANTIC_TRACED = "semantic_traced"
    COMPAT_ONLY = "compat_only"
    UNSUPPORTED = "unsupported"


@lru_cache(maxsize=None)
def canonicalize_trace_api(api: str) -> str:
    return _CANONICAL_API_ALIASES.get(api, api)


@lru_cache(maxsize=None)
def is_ignorable_setup_api(api: str) -> bool:
    api = canonicalize_trace_api(api)
    return (
        api.startswith(_SETUP_API_PREFIXES)
        or classify_trace_api_bucket(api, "other") is TraceApiBucket.COMPAT_ONLY
    )


def is_ignorable_emulator_boundary_api(api: str) -> bool:
    return is_ignorable_setup_api(api)


@lru_cache(maxsize=None)
def is_teardown_api(api: str) -> bool:
    api = canonicalize_trace_api(api)
    return api in {
        "cublasDestroy_v2",
        "cublasLtDestroy",
        "cublasLtMatmulDescDestroy",
        "cublasLtMatmulPreferenceDestroy",
        "cublasLtMatrixLayoutDestroy",
        "cudaStreamDestroy",
        "ncclCommDestroy",
    }


@lru_cache(maxsize=None)
def is_ignorable_dedup_api(api: str, op_type: str) -> bool:
    """Return True for events that should not define worker-uniqueness.

    Maya-style worker dedup is meant to collapse redundant work patterns, not
    split otherwise-equivalent workers because of setup/control-plane noise or
    allocator jitter. We therefore ignore setup APIs and pure allocation events
    when building pattern signatures.
    """

    return is_ignorable_setup_api(api) or op_type == "mem_alloc"


@lru_cache(maxsize=None)
def is_collective_api(api: str, op_type: str) -> bool:
    api = canonicalize_trace_api(api)
    if op_type == "nccl_collective":
        return api in _COLLECTIVE_APIS
    return api in _COLLECTIVE_APIS


@lru_cache(maxsize=None)
def is_low_overhead_api(api: str) -> bool:
    api = canonicalize_trace_api(api)
    return api in _LOW_OVERHEAD_APIS


@lru_cache(maxsize=None)
def targets_stream_resource(api: str, op_type: str) -> bool:
    api = canonicalize_trace_api(api)
    if api in _CUDA_EVENT_RECORD_APIS:
        return True
    if api == "cudaStreamWaitEvent":
        return True
    if api in _THREAD_SYNC_APIS:
        return False
    if api in _HANDLE_HOST_APIS:
        return False
    if op_type == "host_delay":
        return False
    if op_type in _STREAM_DEVICE_OP_TYPES:
        return True
    return False


@lru_cache(maxsize=None)
def occupies_host_dispatch_resource(api: str, op_type: str) -> bool:
    return not targets_stream_resource(api, op_type)


@lru_cache(maxsize=None)
def classify_trace_api_bucket(api: str, op_type: str) -> TraceApiBucket:
    api = canonicalize_trace_api(api)
    if api in _SEMANTIC_TRACED_APIS:
        return TraceApiBucket.SEMANTIC_TRACED
    if op_type in {"kernel_launch", "blas_compute", "mem_copy", "mem_alloc", "nccl_collective"}:
        return TraceApiBucket.SEMANTIC_TRACED
    if api.startswith(_SETUP_API_PREFIXES) or api in _COMPAT_ONLY_APIS:
        return TraceApiBucket.COMPAT_ONLY
    return TraceApiBucket.UNSUPPORTED


@lru_cache(maxsize=None)
def is_compat_only_api(api: str, op_type: str) -> bool:
    return classify_trace_api_bucket(api, op_type) is TraceApiBucket.COMPAT_ONLY


@lru_cache(maxsize=None)
def is_supported_trace_api(api: str, op_type: str) -> bool:
    return classify_trace_api_bucket(api, op_type) is not TraceApiBucket.UNSUPPORTED


@lru_cache(maxsize=None)
def is_semantic_traced_api(api: str, op_type: str) -> bool:
    return classify_trace_api_bucket(api, op_type) is TraceApiBucket.SEMANTIC_TRACED


@lru_cache(maxsize=None)
def is_host_timing_traced_api(api: str, op_type: str) -> bool:
    """Return True when an API should retain explicit timing in emulation.

    Maya-lite uses a stricter semantic conformance policy than its host-timing
    policy:

    - semantic_traced APIs participate in ordering/state/replay validation
    - compat_only APIs must *not* affect semantic conformance conclusions
    - but compat_only APIs may still carry explicit host/control-plane time in
      fake-cuda so host_delay faithfully preserves the observed wall-clock
      envelope around semantic events

    This helper encodes that second policy directly so host-timing logic does
    not have to piggyback on the looser "supported" notion implicitly.
    """

    bucket = classify_trace_api_bucket(api, op_type)
    return bucket in {TraceApiBucket.SEMANTIC_TRACED, TraceApiBucket.COMPAT_ONLY}
