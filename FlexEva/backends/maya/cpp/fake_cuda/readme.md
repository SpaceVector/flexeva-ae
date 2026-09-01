# FakeCUDA

FakeCUDA emulates the CUDA environment so CUDA-facing programs can be tested
without executing real GPU kernels. It provides the API surface and trace data
used by the artifact's evaluator workflows.

## Features

- **CUDA API emulation:** implements the commonly used CUDA APIs needed by the
  included workloads without performing the underlying GPU computation.
- **Event logging:** records API calls and errors for trace collection and
  debugging.
- **Device management:** emulates multiple devices, including device selection
  and per-device state.

## Trace payloads

cuBLAS GEMM wrappers preserve operation and layout metadata for Maya-lite
provider features. `cublasGemmEx` and `cublasGemmStridedBatchedEx` emit shape
and leading-dimension fields; strided-batched calls also emit batch and stride
fields. When the wrapped API has an `algo` argument, FakeCUDA emits both
`algorithm` and the raw alias `algo` with the same numeric value so Python
consumers can canonicalize the material signature without losing the original
payload.

## Build

From the Maya backend root:

```bash
cmake -S fake-cuda -B fake-cuda/build
cmake --build fake-cuda/build -j
```

## Run

The reference environment uses:

```text
Python 3.12
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers
```

`fake-cuda/frun` requires an executable `fake-cuda/proot`. If the default
libraries are not found, check the PyTorch and CUDA library discovery logic in
`fake-cuda/frun`.

Run a workload with:

```bash
chmod +x fake-cuda/frun
fake-cuda/frun python tests/workloads/fake_cuda/test.py
```
