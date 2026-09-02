# FakeCUDA workload API coverage

This file tracks the APIs required by the included training workloads.

## Current status

| Workload | Status | Blocking Issue |
|----------|--------|----------------|
| GPT Data Parallel | Supported | None |
| GPT 3D Parallel (DP+TP+PP) | Supported | None |
| MoE (Expert Parallel) | Supported | Local workload uses CPU `topk` |
| ResNet (Vision) | Unsupported | cuDNN is not initialized |
| GPT-2 (transformers) | Supported | None |

## Maya-lite trace payload coverage

The implemented cuBLAS GEMM APIs used by Maya-lite preserve
paper-relevant operation/layout metadata in trace payloads. `cublasGemmEx` and
`cublasGemmStridedBatchedEx` emit GEMM dimensions, leading dimensions, operand
types, compute type, and the cuBLAS algorithm. Strided-batched GEMM additionally
emits batch count and stride fields. For the algorithm argument, fake-cuda emits
both canonical `algorithm` and raw alias `algo`; Python consumers canonicalize
the alias for material signatures and estimator features.

## cuDNN gap

ResNet and other CNN-based workloads fail with `CUDNN_STATUS_NOT_INITIALIZED`.

### Missing cuDNN functions

```c
// Initialization
cudnnCreate()
cudnnDestroy()
cudnnSetStream()
cudnnGetStream()

// Tensor operations
cudnnCreateTensorDescriptor()
cudnnDestroyTensorDescriptor()
cudnnSetTensor4dDescriptor()
cudnnSetTensorNdDescriptor()

// Convolution
cudnnCreateConvolutionDescriptor()
cudnnDestroyConvolutionDescriptor()
cudnnSetConvolution2dDescriptor()
cudnnSetConvolutionNdDescriptor()
cudnnGetConvolutionForwardAlgorithm_v7()
cudnnConvolutionForward()
cudnnConvolutionBackwardData()
cudnnConvolutionBackwardFilter()

// Pooling
cudnnCreatePoolingDescriptor()
cudnnDestroyPoolingDescriptor()
cudnnSetPooling2dDescriptor()
cudnnPoolingForward()
cudnnPoolingBackward()

// Batch Normalization
cudnnBatchNormalizationForwardTraining()
cudnnBatchNormalizationBackward()
cudnnBatchNormalizationForwardInference()

// Activation
cudnnCreateActivationDescriptor()
cudnnDestroyActivationDescriptor()
cudnnSetActivationDescriptor()
cudnnActivationForward()
cudnnActivationBackward()

// Softmax (also used in MoE routing)
cudnnSoftmaxForward()
cudnnSoftmaxBackward()
```

### Library requirement

A matching `libcudnn.so.9` is not currently provided.

## MoE top-k workaround

`torch.topk()` on an emulated CUDA tensor causes a floating-point exception.

### Current workaround

Move topk to CPU in MoE code:

```python
top_k_logits, top_k_indices = torch.topk(gate_logits.cpu(), k, dim=-1)
top_k_logits = top_k_logits.to(device)
top_k_indices = top_k_indices.to(device)
```

### Full emulation

Full FakeCUDA support would require emulating the sorting and selection kernel.
The relevant entry point is `cudaLaunchKernel()`. The included MoE workload
does not require this path.

## Unimplemented CUDA runtime APIs

The PyTorch 2.8.0 inventory lists 347 unimplemented APIs.

### CUDA graph APIs (102 missing)

Not needed for PyTorch 2.8.0, but required for PyTorch 2.9+:

```c
cudaGraphAddEmptyNode()
cudaGraphCreate()
cudaGraphAddKernelNode()
cudaGraphAddMemcpyNode()
cudaGraphAddDependencies()
// ... and 97 more
```

### Memory pool APIs

```c
cudaMemPoolCreate()
cudaMemPoolDestroy()
cudaMemPoolSetAttribute()
cudaMemPoolGetAttribute()
cudaMemAllocFromPoolAsync()
```

### Device management

```c
cudaDeviceGetP2PAttribute()
cudaDeviceGetGraphMemAttribute()
cudaDeviceSetGraphMemAttribute()
cudaDeviceGraphMemTrim()
cudaInitDevice()
```

## NCCL APIs

Core NCCL APIs used by the Maya-lite workloads are implemented in
`cpp/fake_cuda/src/nccl/common.cpp`. Additional NCCL APIs should be added only
when a concrete workload requires them.

```c
ncclCommInitRank()
ncclCommDestroy()
ncclAllReduce()
ncclBroadcast()
ncclReduce()
ncclAllGather()
ncclReduceScatter()
ncclSend()
ncclRecv()
ncclGroupStart()
ncclGroupEnd()
```

## Test commands

```bash
# Test GPT DP through the current fake-cuda launcher
fake-cuda/frun python tests/workloads/fake_cuda/gpt2.py --steps 2

# Test GPT 3D
fake-cuda/frun python tests/workloads/fake_cuda/gpt_3d.py --steps 2

# Test MoE
fake-cuda/frun python tests/workloads/fake_cuda/moe.py --steps 2

# Test ResNet (currently expected to fail until cuDNN stubs exist)
fake-cuda/frun python tests/workloads/fake_cuda/resnet_dp.py --steps 2
```

## File locations

- Missing CUDA Runtime APIs: `cpp/fake_cuda/missing_apis.txt` (347 APIs)
- Fake library source: `cpp/fake_cuda/src/`
- Workload tests: `tests/workloads/fake_cuda/`
