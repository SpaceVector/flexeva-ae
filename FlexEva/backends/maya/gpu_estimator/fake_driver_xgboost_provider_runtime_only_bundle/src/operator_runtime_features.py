from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict


TARGET_COLUMN = "time_ms"

CUBLAS_DTYPE_NAMES = {
    0: "fp16",
    1: "fp32",
    2: "bf16",
}

NCCL_DTYPE_NAMES = {
    0: "fp16",
    1: "bf16",
    2: "fp32",
}

CUBLAS_FEATURE_COLUMNS = [
    "m",
    "n",
    "k",
    "batch_count",
    "transa",
    "transb",
    "dtype_code",
    "elem_bytes",
    "lda",
    "ldb",
    "ldc",
    "stride_a",
    "stride_b",
    "stride_c",
    "matrix_a_bytes",
    "matrix_b_bytes",
    "matrix_c_bytes",
    "total_flops",
    "memory_bytes",
    "arithmetic_intensity",
    "is_batched",
    "log_m",
    "log_n",
    "log_k",
    "log_batch",
    "log_flops",
    "log_memory_bytes",
]

NCCL_ALLREDUCE_FEATURE_COLUMNS = [
    "numel",
    "world_size",
    "dtype_code",
    "elem_bytes",
    "message_bytes",
    "cluster_aggregate_bytes",
    "effective_comm_bytes",
    "collective_factor",
    "log_numel",
    "log_message_bytes",
    "log_effective_comm_bytes",
    "log_world_size",
]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def cublas_dtype_name(dtype_code: int) -> str:
    return CUBLAS_DTYPE_NAMES[int(dtype_code)]


def nccl_dtype_name(dtype_code: int) -> str:
    return NCCL_DTYPE_NAMES[int(dtype_code)]


def cublas_elem_bytes(dtype_code: int) -> int:
    return 4 if int(dtype_code) == 1 else 2


def nccl_elem_bytes(dtype_code: int) -> int:
    return 4 if int(dtype_code) == 2 else 2


def _log2p1(value: float) -> float:
    return math.log2(float(value) + 1.0)


def build_cublas_features(
    *,
    m: int,
    n: int,
    k: int,
    batch_count: int,
    transa: int,
    transb: int,
    dtype_code: int,
) -> Dict[str, Any]:
    m = int(m)
    n = int(n)
    k = int(k)
    batch_count = max(1, int(batch_count))
    transa = int(transa)
    transb = int(transb)
    dtype_code = int(dtype_code)

    elem_bytes = cublas_elem_bytes(dtype_code)
    lda = m if transa == 0 else k
    ldb = k if transb == 0 else n
    ldc = m
    stride_a = m * k
    stride_b = k * n
    stride_c = m * n

    matrix_a_bytes = batch_count * stride_a * elem_bytes
    matrix_b_bytes = batch_count * stride_b * elem_bytes
    matrix_c_bytes = batch_count * stride_c * elem_bytes
    memory_bytes = matrix_a_bytes + matrix_b_bytes + matrix_c_bytes
    total_flops = 2.0 * batch_count * m * n * k
    arithmetic_intensity = total_flops / max(float(memory_bytes), 1.0)

    return {
        "operator_name": "cublas_gemm",
        "api_name": "cublasGemmEx" if batch_count == 1 else "cublasGemmStridedBatchedEx",
        "dtype_name": cublas_dtype_name(dtype_code),
        "m": m,
        "n": n,
        "k": k,
        "batch_count": batch_count,
        "transa": transa,
        "transb": transb,
        "dtype_code": dtype_code,
        "elem_bytes": elem_bytes,
        "lda": lda,
        "ldb": ldb,
        "ldc": ldc,
        "stride_a": stride_a,
        "stride_b": stride_b,
        "stride_c": stride_c,
        "matrix_a_bytes": matrix_a_bytes,
        "matrix_b_bytes": matrix_b_bytes,
        "matrix_c_bytes": matrix_c_bytes,
        "total_flops": total_flops,
        "memory_bytes": memory_bytes,
        "arithmetic_intensity": arithmetic_intensity,
        "is_batched": 1 if batch_count > 1 else 0,
        "log_m": _log2p1(m),
        "log_n": _log2p1(n),
        "log_k": _log2p1(k),
        "log_batch": _log2p1(batch_count),
        "log_flops": _log2p1(total_flops),
        "log_memory_bytes": _log2p1(memory_bytes),
    }


def build_nccl_allreduce_features(
    *,
    numel: int,
    world_size: int,
    dtype_code: int,
) -> Dict[str, Any]:
    numel = int(numel)
    world_size = max(1, int(world_size))
    dtype_code = int(dtype_code)

    elem_bytes = nccl_elem_bytes(dtype_code)
    message_bytes = numel * elem_bytes
    cluster_aggregate_bytes = world_size * message_bytes
    collective_factor = 2.0 * (world_size - 1) / world_size if world_size > 1 else 1.0
    effective_comm_bytes = message_bytes * collective_factor

    return {
        "operator_name": "nccl_allreduce",
        "api_name": "ncclAllReduce",
        "collective": "all_reduce",
        "reduction": "sum",
        "dtype_name": nccl_dtype_name(dtype_code),
        "numel": numel,
        "world_size": world_size,
        "dtype_code": dtype_code,
        "elem_bytes": elem_bytes,
        "message_bytes": message_bytes,
        "cluster_aggregate_bytes": cluster_aggregate_bytes,
        "effective_comm_bytes": effective_comm_bytes,
        "collective_factor": collective_factor,
        "log_numel": _log2p1(numel),
        "log_message_bytes": _log2p1(message_bytes),
        "log_effective_comm_bytes": _log2p1(effective_comm_bytes),
        "log_world_size": _log2p1(world_size),
    }
