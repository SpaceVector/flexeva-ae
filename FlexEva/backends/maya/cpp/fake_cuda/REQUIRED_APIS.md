# Fake CUDA - Required APIs for Workload Support

This document lists the APIs needed to support various AI training workloads.

## Current Status

| Workload | Status | Blocking Issue |
|----------|--------|----------------|
| GPT Data Parallel | ✅ Working | - |
| GPT 3D Parallel (DP+TP+PP) | ✅ Working | - |
| MoE (Expert Parallel) | ✅ Working | CPU topk workaround used by local workload |
| ResNet (Vision) | ❌ Crash | cuDNN not initialized |
| GPT-2 (transformers) | ✅ Working | - |

## Maya-lite Trace Payload Coverage

The implemented cuBLAS GEMM APIs used by Maya-lite should preserve
paper-relevant operation/layout metadata in trace payloads. `cublasGemmEx` and
`cublasGemmStridedBatchedEx` emit GEMM dimensions, leading dimensions, operand
types, compute type, and the cuBLAS algorithm. Strided-batched GEMM additionally
emits batch count and stride fields. For the algorithm argument, fake-cuda emits
both canonical `algorithm` and raw alias `algo`; Python consumers canonicalize
the alias for material signatures and estimator features.

## Priority 1: cuDNN APIs (for CNN workloads)

ResNet and other CNN-based workloads fail with `CUDNN_STATUS_NOT_INITIALIZED`.

### Required cuDNN Functions

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

### Library to Create

Need `libcudnn.so.9` (or version matching PyTorch 2.8.0).

## Priority 2: MoE-specific APIs (RESOLVED)

**Root cause found:** `torch.topk()` on CUDA causes floating point exception.

### Workaround Applied

Move topk to CPU in MoE code:
```python
# Instead of:
# top_k_logits, top_k_indices = torch.topk(gate_logits, k, dim=-1)

# Use:
top_k_logits, top_k_indices = torch.topk(gate_logits.cpu(), k, dim=-1)
top_k_logits = top_k_logits.to(device)
top_k_indices = top_k_indices.to(device)
```

### Proper Fix (for colleague)

The topk CUDA kernel implementation needs to handle the sorting/selection without FP exception. The kernel is likely `cub::DeviceRadixSort` or similar.

```c
// APIs that may need proper implementation for topk:
cudaLaunchKernel()    // topk uses custom CUDA kernels for radix sort
```

## Priority 3: Missing CUDA Runtime APIs

From analysis of PyTorch 2.8.0 requirements (347 total missing):

### CUDA Graph APIs (102 missing)

Not needed for PyTorch 2.8.0, but required for PyTorch 2.9+:

```c
cudaGraphAddEmptyNode()
cudaGraphCreate()
cudaGraphAddKernelNode()
cudaGraphAddMemcpyNode()
cudaGraphAddDependencies()
// ... and 97 more
```

### Memory Pool APIs

```c
cudaMemPoolCreate()
cudaMemPoolDestroy()
cudaMemPoolSetAttribute()
cudaMemPoolGetAttribute()
cudaMemAllocFromPoolAsync()
```

### Device Management

```c
cudaDeviceGetP2PAttribute()
cudaDeviceGetGraphMemAttribute()
cudaDeviceSetGraphMemAttribute()
cudaDeviceGraphMemTrim()
cudaInitDevice()
```

## Priority 4: NCCL APIs (for multi-GPU)

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

## Testing Commands

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

## Implementation Notes

### For cuDNN Stubs

Minimal implementation that returns success without actual computation:

```c
cudnnStatus_t cudnnCreate(cudnnHandle_t *handle) {
    *handle = (cudnnHandle_t)malloc(sizeof(void*));
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionForward(...) {
    // Just return success, output tensor stays zeros
    return CUDNN_STATUS_SUCCESS;
}
```

### For MoE Fix

Debug with:
```bash
FAKECUDA_LOG_LEVEL=4 fake-cuda/frun python tests/workloads/fake_cuda/moe.py --steps 1
```

## Files Reference

- Missing CUDA Runtime APIs: `cpp/fake_cuda/missing_apis.txt` (347 APIs)
- Fake library source: `cpp/fake_cuda/src/`
- Workload tests: `tests/workloads/fake_cuda/`
