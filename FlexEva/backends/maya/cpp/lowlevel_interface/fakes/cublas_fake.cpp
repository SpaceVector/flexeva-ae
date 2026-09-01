/**
 * cuBLAS API fake implementation
 *
 * This file declares cuBLAS functions for wrapper generation.
 * The scan_fakes tool parses this to generate config/cublas_wrappers.json.
 *
 * Actual implementation is provided by fake-cuda runtime libraries.
 * These stubs just need correct signatures for wrapper generation.
 */

#include <cstddef>
#include <cstdint>

// cuBLAS types (minimal definitions for signature parsing)
typedef int cublasStatus_t;
typedef void* cublasHandle_t;
typedef void* cudaStream_t;

enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2
};

enum cublasGemmAlgo_t {
    CUBLAS_GEMM_DEFAULT = -1
};

enum cublasComputeType_t {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_64F = 70
};

enum cudaDataType_t {
    CUDA_R_16F = 2,
    CUDA_R_32F = 0,
    CUDA_R_64F = 1
};

extern "C" {

// ============================================================================
// Handle Management
// ============================================================================

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle) {
    (void)handle;
    return 0;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
    (void)handle;
    return 0;
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId) {
    (void)handle; (void)streamId;
    return 0;
}

cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t* streamId) {
    (void)handle; (void)streamId;
    return 0;
}

// ============================================================================
// GEMM Operations - Single Precision
// ============================================================================

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float* alpha,
                           const float* A, int lda,
                           const float* B, int ldb,
                           const float* beta,
                           float* C, int ldc) {
    (void)handle; (void)transa; (void)transb;
    (void)m; (void)n; (void)k;
    (void)alpha; (void)A; (void)lda;
    (void)B; (void)ldb; (void)beta;
    (void)C; (void)ldc;
    return 0;
}

// ============================================================================
// GEMM Operations - Double Precision
// ============================================================================

cublasStatus_t cublasDgemm_v2(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double* alpha,
                           const double* A, int lda,
                           const double* B, int ldb,
                           const double* beta,
                           double* C, int ldc) {
    (void)handle; (void)transa; (void)transb;
    (void)m; (void)n; (void)k;
    (void)alpha; (void)A; (void)lda;
    (void)B; (void)ldb; (void)beta;
    (void)C; (void)ldc;
    return 0;
}

// ============================================================================
// GEMM Operations - Half Precision
// ============================================================================

cublasStatus_t cublasHgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const void* alpha,
                           const void* A, int lda,
                           const void* B, int ldb,
                           const void* beta,
                           void* C, int ldc) {
    (void)handle; (void)transa; (void)transb;
    (void)m; (void)n; (void)k;
    (void)alpha; (void)A; (void)lda;
    (void)B; (void)ldb; (void)beta;
    (void)C; (void)ldc;
    return 0;
}

// ============================================================================
// GEMM Operations - Generic (GemmEx)
// ============================================================================

cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const void* alpha,
                            const void* A, cudaDataType_t Atype, int lda,
                            const void* B, cudaDataType_t Btype, int ldb,
                            const void* beta,
                            void* C, cudaDataType_t Ctype, int ldc,
                            cublasComputeType_t computeType,
                            cublasGemmAlgo_t algo) {
    (void)handle; (void)transa; (void)transb;
    (void)m; (void)n; (void)k;
    (void)alpha; (void)A; (void)Atype; (void)lda;
    (void)B; (void)Btype; (void)ldb;
    (void)beta; (void)C; (void)Ctype; (void)ldc;
    (void)computeType; (void)algo;
    return 0;
}

// ============================================================================
// GEMM Operations - Strided Batched
// ============================================================================

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle,
                                         cublasOperation_t transa, cublasOperation_t transb,
                                         int m, int n, int k,
                                         const float* alpha,
                                         const float* A, int lda, long long strideA,
                                         const float* B, int ldb, long long strideB,
                                         const float* beta,
                                         float* C, int ldc, long long strideC,
                                         int batchCount) {
    (void)handle; (void)transa; (void)transb;
    (void)m; (void)n; (void)k;
    (void)alpha; (void)A; (void)lda; (void)strideA;
    (void)B; (void)ldb; (void)strideB;
    (void)beta; (void)C; (void)ldc; (void)strideC;
    (void)batchCount;
    return 0;
}

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
                                          cublasOperation_t transa, cublasOperation_t transb,
                                          int m, int n, int k,
                                          const void* alpha,
                                          const void* A, cudaDataType_t Atype, int lda, long long strideA,
                                          const void* B, cudaDataType_t Btype, int ldb, long long strideB,
                                          const void* beta,
                                          void* C, cudaDataType_t Ctype, int ldc, long long strideC,
                                          int batchCount,
                                          cublasComputeType_t computeType,
                                          cublasGemmAlgo_t algo) {
    (void)handle; (void)transa; (void)transb;
    (void)m; (void)n; (void)k;
    (void)alpha; (void)A; (void)Atype; (void)lda; (void)strideA;
    (void)B; (void)Btype; (void)ldb; (void)strideB;
    (void)beta; (void)C; (void)Ctype; (void)ldc; (void)strideC;
    (void)batchCount; (void)computeType; (void)algo;
    return 0;
}

} // extern "C"
