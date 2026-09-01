from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import xgboost as xgb

try:
    from .operator_runtime_features import build_cublas_features, build_nccl_allreduce_features
except ImportError:
    from operator_runtime_features import build_cublas_features, build_nccl_allreduce_features


DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "xgboost" / "runtime_provider" / "manifest.json"
)

_CUBLAS_APIS = {
    "cublassgemm",
    "cublassgemm_v2",
    "cublasgemmex",
    "cublasgemmstridedbatchedex",
    "cublasgemmbatchedex",
    "cublasltmatmul",
}

_NCCL_ALLREDUCE_APIS = {
    "ncclallreduce",
}

_ALLREDUCE_ALIASES = {
    "allreduce",
    "all_reduce",
    "ncclallreduce",
}

_SUM_ALIASES = {
    "sum",
    "ncclsum",
}

_CUBLAS_DTYPE_CODE_BY_NAME = {
    "fp16": 0,
    "half": 0,
    "float16": 0,
    "f16": 0,
    "16f": 0,
    "cuda_r_16f": 0,
    "fp32": 1,
    "float": 1,
    "float32": 1,
    "f32": 1,
    "32f": 1,
    "cuda_r_32f": 1,
    "bf16": 2,
    "bfloat16": 2,
    "16bf": 2,
    "cuda_r_16bf": 2,
}

_NCCL_DTYPE_CODE_BY_NAME = {
    "fp16": 0,
    "half": 0,
    "float16": 0,
    "f16": 0,
    "ncclfloat16": 0,
    "ncclhalf": 0,
    "bf16": 1,
    "bfloat16": 1,
    "ncclbfloat16": 1,
    "fp32": 2,
    "float": 2,
    "float32": 2,
    "f32": 2,
    "ncclfloat32": 2,
}


def _normalize_token(value: Any) -> str:
    token = str(value).strip().lower()
    return token.replace(" ", "").replace("-", "").replace("__", "_")


def _first(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _to_int(value: Any, field_name: str) -> int:
    if value is None:
        raise ValueError(f"missing required field: {field_name}")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value)
    try:
        return int(float(str(value)))
    except ValueError as exc:
        raise ValueError(f"could not parse integer field {field_name} from {value!r}") from exc


def _normalize_transpose(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, np.integer, bool)):
        return 0 if int(value) == 0 else 1
    token = _normalize_token(value)
    if token in {"n", "nontranspose", "notranspose", "0"}:
        return 0
    if token in {"t", "transpose", "c", "conjugatetranspose", "1"}:
        return 1
    raise ValueError(f"unsupported transpose value: {value!r}")


def _normalize_cublas_dtype_code(value: Any) -> int:
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1, 2}:
        return int(value)
    token = _normalize_token(value)
    if token in _CUBLAS_DTYPE_CODE_BY_NAME:
        return _CUBLAS_DTYPE_CODE_BY_NAME[token]
    raise ValueError(f"unsupported cuBLAS dtype value: {value!r}")


def _normalize_nccl_dtype_code(value: Any) -> int:
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1, 2}:
        return int(value)
    token = _normalize_token(value)
    if token in _NCCL_DTYPE_CODE_BY_NAME:
        return _NCCL_DTYPE_CODE_BY_NAME[token]
    raise ValueError(f"unsupported NCCL dtype value: {value!r}")


def _normalize_api_name(value: Any) -> str:
    if value is None:
        return ""
    return _normalize_token(value)


@dataclass
class _ModelBundle:
    model: xgb.XGBRegressor
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]
    prediction_cache: dict[tuple[float, ...], float] = field(default_factory=dict)

    def predict_ms(self, feature_row: Mapping[str, Any]) -> float:
        key = tuple(float(feature_row.get(feature_name, 0.0)) for feature_name in self.feature_names)
        cached = self.prediction_cache.get(key)
        if cached is not None:
            return cached
        matrix = np.array(
            [key],
            dtype=np.float32,
        )
        prediction = float(self.model.predict(matrix)[0])
        prediction = max(prediction, 1e-9)
        self.prediction_cache[key] = prediction
        return prediction


class FakeDriverXGBoostProvider:
    """
    Thin runtime provider that consumes fake-driver hook params directly.

    Supported compute APIs:
    - cublasSgemm / cublasGemmEx / cublasGemmStridedBatchedEx / cublasGemmBatchedEx
    - cublasLtMatmul (mapped onto the same GEMM feature space)

    Supported communication APIs:
    - ncclAllReduce (sum)
    """

    def __init__(self, manifest_path: str | Path = DEFAULT_MANIFEST_PATH):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"runtime-provider manifest not found: {self.manifest_path}")
        self.root_dir = self.manifest_path.parent
        manifest = json.loads(self.manifest_path.read_text())
        model_entries = manifest.get("models", {})
        self._models = {
            operator_name: self._load_model_bundle(entry)
            for operator_name, entry in model_entries.items()
        }

    def _load_model_bundle(self, entry: Mapping[str, Any]) -> _ModelBundle:
        model_path = self.root_dir / str(entry["model_path"])
        feature_names_path = self.root_dir / str(entry["feature_names_path"])
        metadata_path = self.root_dir / str(entry["metadata_path"])

        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
        feature_names = tuple(
            line.strip() for line in feature_names_path.read_text().splitlines() if line.strip()
        )
        metadata = json.loads(metadata_path.read_text())
        return _ModelBundle(model=model, feature_names=feature_names, metadata=metadata)

    def available_operators(self) -> tuple[str, ...]:
        return tuple(sorted(self._models))

    def supports(self, api_name: str | None = None, event: Mapping[str, Any] | None = None) -> bool:
        payload = dict(event or {})
        resolved_api = _normalize_api_name(api_name or _first(payload, "api_name", "name"))
        if resolved_api in _CUBLAS_APIS or resolved_api in _NCCL_ALLREDUCE_APIS:
            return True
        kind = _normalize_token(_first(payload, "kind", "collective") or "")
        return kind in {"gemm", "allreduce", "all_reduce"}

    def predict_ms(
        self,
        hook_params: Mapping[str, Any] | None = None,
        *,
        api_name: str | None = None,
        event: Mapping[str, Any] | None = None,
    ) -> float:
        if hook_params is not None and event is not None:
            raise ValueError("pass either hook_params or event, not both")
        payload = dict(event or hook_params or {})
        if api_name is not None and "api_name" not in payload and "name" not in payload:
            payload["api_name"] = api_name

        resolved_api = _normalize_api_name(_first(payload, "api_name", "name"))
        if self._is_cublas_payload(resolved_api, payload):
            return self.predict_cublas_ms(payload)
        if self._is_nccl_allreduce_payload(resolved_api, payload):
            return self.predict_nccl_allreduce_ms(payload)
        raise ValueError(f"unsupported hook payload for runtime provider: api_name={api_name!r}, keys={sorted(payload)}")

    def predict_cublas_ms(self, hook_params: Mapping[str, Any]) -> float:
        payload = dict(hook_params)
        api_name = _normalize_api_name(_first(payload, "api_name", "name"))

        dtype_value = _first(payload, "dtype_code", "dtype", "dtype_name", "Atype", "Btype", "Ctype", "compute_type")
        if dtype_value is None:
            if api_name in {"cublassgemm", "cublassgemm_v2"}:
                dtype_code = 1
            else:
                raise ValueError("missing cuBLAS dtype information")
        else:
            dtype_code = _normalize_cublas_dtype_code(dtype_value)

        batch_count = _first(payload, "batch_count", "batch", "batchCount")
        if batch_count is None:
            batch_count = 1
        batch_count = max(1, _to_int(batch_count, "batch_count"))

        feature_row = build_cublas_features(
            m=_to_int(_first(payload, "m", "M"), "m"),
            n=_to_int(_first(payload, "n", "N"), "n"),
            k=_to_int(_first(payload, "k", "K"), "k"),
            batch_count=batch_count,
            transa=_normalize_transpose(_first(payload, "transa", "transA")),
            transb=_normalize_transpose(_first(payload, "transb", "transB")),
            dtype_code=dtype_code,
        )
        return self._models["cublas_gemm"].predict_ms(feature_row)

    def predict_nccl_allreduce_ms(self, hook_params: Mapping[str, Any]) -> float:
        payload = dict(hook_params)

        collective = _first(payload, "collective", "name", "api_name")
        if collective is not None and _normalize_token(collective) not in _ALLREDUCE_ALIASES:
            raise ValueError(f"unsupported NCCL collective for current provider: {collective!r}")

        reduction = _first(payload, "reduction", "op")
        if reduction is not None and _normalize_token(reduction) not in _SUM_ALIASES:
            raise ValueError(f"unsupported NCCL reduction op for current provider: {reduction!r}")

        dtype_code = _normalize_nccl_dtype_code(
            _first(payload, "dtype_code", "dtype", "datatype", "dtype_name")
        )
        world_size = _to_int(_first(payload, "world_size", "nranks", "rank_count"), "world_size")

        numel_value = _first(payload, "numel", "count")
        if numel_value is None:
            bytes_value = _first(payload, "bytes")
            if bytes_value is None:
                raise ValueError("missing NCCL count/numel/bytes information")
            message_bytes = _to_int(bytes_value, "bytes")
            elem_bytes = 2 if dtype_code in {0, 1} else 4
            if message_bytes % elem_bytes != 0:
                raise ValueError(
                    f"NCCL bytes={message_bytes} is not divisible by elem_bytes={elem_bytes}"
                )
            numel = message_bytes // elem_bytes
        else:
            numel = _to_int(numel_value, "numel")

        feature_row = build_nccl_allreduce_features(
            numel=numel,
            world_size=world_size,
            dtype_code=dtype_code,
        )
        return self._models["nccl_allreduce"].predict_ms(feature_row)

    def _is_cublas_payload(self, api_name: str, payload: Mapping[str, Any]) -> bool:
        if api_name in _CUBLAS_APIS:
            return True
        kind = _normalize_token(_first(payload, "kind") or "")
        return kind == "gemm"

    def _is_nccl_allreduce_payload(self, api_name: str, payload: Mapping[str, Any]) -> bool:
        if api_name in _NCCL_ALLREDUCE_APIS:
            return True
        kind = _normalize_token(_first(payload, "kind", "collective") or "")
        return kind in {"allreduce", "all_reduce"}
