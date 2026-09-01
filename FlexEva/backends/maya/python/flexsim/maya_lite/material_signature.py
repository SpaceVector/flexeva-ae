"""Shared material signature helpers for Maya-lite provider row metadata."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

GEMM_APIS = frozenset({"cublasGemmEx", "cublasGemmStridedBatchedEx"})

_GEMM_FIELD_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("m", ("m",)),
    ("n", ("n",)),
    ("k", ("k",)),
    ("lda", ("lda",)),
    ("ldb", ("ldb",)),
    ("ldc", ("ldc",)),
    ("batch_count", ("batch_count", "batchCount")),
    ("stride_a", ("stride_a", "strideA")),
    ("stride_b", ("stride_b", "strideB")),
    ("stride_c", ("stride_c", "strideC")),
    ("compute_type", ("compute_type", "computeType")),
    ("cuda_data_type", ("cuda_data_type", "dtype")),
    ("transa", ("transa",)),
    ("transb", ("transb",)),
    ("algorithm", ("algorithm", "algo")),
)


def is_gemm_material_api(api: object) -> bool:
    return str(api or "") in GEMM_APIS


def canonical_material_value(value: Any) -> str:
    """Serialize numeric values by value while preserving non-numeric tokens."""

    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return format(value, ".15g")
    text = str(value).strip()
    try:
        decimal = Decimal(text)
    except (InvalidOperation, ValueError):
        return text
    if not decimal.is_finite():
        return text
    if decimal == decimal.to_integral_value():
        return str(int(decimal))
    return format(decimal.normalize(), "f")


def canonical_gemm_signature_inputs(record: Mapping[str, Any]) -> dict[str, str]:
    """Return canonical GEMM material inputs, excluding placement metadata."""

    inputs: dict[str, str] = {}
    for canonical_key, aliases in _GEMM_FIELD_ALIASES:
        for alias in aliases:
            value = record.get(alias)
            if value in (None, ""):
                continue
            inputs[canonical_key] = canonical_material_value(value)
            break
    return inputs


def canonical_gemm_material_signature(record: Mapping[str, Any]) -> str | None:
    inputs = canonical_gemm_signature_inputs(record)
    if not inputs:
        return None
    return ";".join(f"{key}={value}" for key, value in inputs.items())
