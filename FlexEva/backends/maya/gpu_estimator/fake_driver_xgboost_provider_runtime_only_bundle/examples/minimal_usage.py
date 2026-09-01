from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fake_driver_xgboost_provider import FakeDriverXGBoostProvider


provider = FakeDriverXGBoostProvider()

gemm_payload = {
    "api_name": "cublasGemmEx",
    "m": 4096,
    "n": 4096,
    "k": 4096,
    "transa": "N",
    "transb": "N",
    "dtype": "bf16",
}

allreduce_payload = {
    "api_name": "ncclAllReduce",
    "bytes": 1048576,
    "nranks": 4,
    "dtype": "fp32",
    "op": "sum",
}

print("gemm_ms =", provider.predict_ms(gemm_payload))
print("allreduce_ms =", provider.predict_ms(allreduce_payload))
