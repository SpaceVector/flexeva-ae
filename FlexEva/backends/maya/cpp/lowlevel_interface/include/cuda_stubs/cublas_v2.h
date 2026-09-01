// Minimal cuBLAS stubs for wrapper compilation
// These are opaque type definitions - actual implementations are in fake-cuda

#ifndef CUBLAS_V2_STUB_H
#define CUBLAS_V2_STUB_H

#include <stddef.h>
#include "cuda_runtime.h"

// cuBLAS status type
typedef int cublasStatus_t;
#define CUBLAS_STATUS_SUCCESS 0

// Opaque handle type
typedef struct cublasContext* cublasHandle_t;

// Operation types
typedef enum {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2
} cublasOperation_t;

// Data types
typedef int cudaDataType;
typedef int cudaDataType_t;  // Also used by cublasGemmEx
typedef int cublasComputeType_t;
typedef int cublasGemmAlgo_t;

#endif // CUBLAS_V2_STUB_H
