// Minimal CUDA runtime stubs for wrapper compilation
// These are opaque type definitions - actual implementations are in fake-cuda

#ifndef CUDA_RUNTIME_STUB_H
#define CUDA_RUNTIME_STUB_H

#include <stddef.h>

// CUDA error type
typedef int cudaError_t;
#define cudaSuccess 0

// Opaque handle types
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;

// Memory copy kinds
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

// Dim3 for kernel launch
struct dim3 {
    unsigned int x, y, z;
};

struct uint3 {
    unsigned int x, y, z;
};

#endif // CUDA_RUNTIME_STUB_H
