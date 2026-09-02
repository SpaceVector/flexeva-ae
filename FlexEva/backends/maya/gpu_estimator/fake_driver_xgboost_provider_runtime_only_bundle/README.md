# Fake Driver XGBoost Runtime Provider Bundle

This bundle provides operator-level latency prediction for the fake-driver
integration. It does not require profiling or retraining.

Included runtime support:

- cuBLAS GEMM-like APIs:
  - `cublasSgemm`
  - `cublasGemmEx`
  - `cublasGemmStridedBatchedEx`
  - `cublasGemmBatchedEx`
  - `cublasLtMatmul`
- NCCL communication API:
  - `ncclAllReduce` with reduction op `sum`

## Layout

- `src/`: provider source code
- `models/xgboost/runtime_provider/`: trained XGBoost artifacts
- `examples/`: minimal usage example
- `requirements.txt`: Python runtime dependencies
- `FILELIST.txt`: exact bundled files

## Runtime dependencies

Install:

```bash
pip install -r requirements.txt
```

Required packages:

- `numpy`
- `xgboost`

## Minimal usage

```python
import sys
sys.path.insert(0, "/path/to/fake_driver_xgboost_provider_bundle/src")

from fake_driver_xgboost_provider import FakeDriverXGBoostProvider

provider = FakeDriverXGBoostProvider()

gemm_ms = provider.predict_ms({
    "api_name": "cublasGemmEx",
    "m": 4096,
    "n": 4096,
    "k": 4096,
    "transa": "N",
    "transb": "N",
    "dtype": "bf16",
})

allreduce_ms = provider.predict_ms({
    "api_name": "ncclAllReduce",
    "bytes": 1048576,
    "nranks": 4,
    "dtype": "fp32",
    "op": "sum",
})
```

## Hook parameter conventions

cuBLAS payload fields:
- required: `m`, `n`, `k`
- optional: `batch` or `batch_count`
- optional: `transa`/`transb` or `transA`/`transB`
- optional: `dtype` or `dtype_code`

NCCL payload fields:
- required: `nranks` or `world_size`
- one of: `numel`, `count`, or `bytes`
- required: `dtype` or `datatype`
- optional: `op`, currently only `sum`
