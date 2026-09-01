#include "fake_types.h"
#include "fake_runtime_api.hpp"
#include "cublas.hpp"
#include "utils.hpp"
#include "../../include/common/trace_log.hpp"

namespace {
std::uint64_t opaque_id(const void* value) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(value));
}
}

API cublasStatus_t cublasCreate_v2(cublasHandle_t* handle){
    LOG_DEBUG(CUBLAS, "cublasCreate_v2() called.");
    // 分配内存
    cublasContext* ctx = (struct cublasContext*)malloc(sizeof(struct cublasContext));
    if (!ctx) return CUBLAS_STATUS_ALLOC_FAILED;
    // 初始化内部状态
    ctx->magic_number = 0xBEEF; // 标记，用于 debug
    ctx->stream = 0;            // 默认流 (cudaStreamDefault)
    *handle = ctx; // 赋值给输出指针
    TracePayloadBuilder payload;
    payload.add_uint64("handle_id", opaque_id(ctx));
    TRACE_API_EX(CUBLAS, "cublasCreate_v2", "context_op", payload);
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDestroy_v2(cublasHandle_t handle){
    LOG_DEBUG(CUBLAS, "cublasDestroy_v2() called.");
    if (!handle || handle->magic_number != 0xBEEF) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    TracePayloadBuilder payload;
    payload.add_uint64("handle_id", opaque_id(handle));
    TRACE_API_EX(CUBLAS, "cublasDestroy_v2", "context_op", payload);
    // 释放内存
    free(handle);
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc){
    LOG_DEBUG(CUBLAS, "cublasDgemm_v2() called.");
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t  cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float* beta, float* C, int ldc, long long int strideC, int batchCount){
    LOG_DEBUG(CUBLAS, "cublasSgemmStridedBatched() called.");
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result){
    LOG_DEBUG(CUBLAS, "cublasDdot_v2() called.");
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc){
    LOG_DEBUG(CUBLAS, "cublasSgemm_v2() called.");
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCdotu_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result){
    LOG_DEBUG(CUBLAS, "cublasCdotu_v2() called.");
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* const A[], int lda, double* const B[], int ldb, int batchCount){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZdotc_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex* const Aarray[], int lda, const int* devIpiv, cuDoubleComplex* const Barray[], int ldb, int* info, int batchSize){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, double* const Aarray[], int lda, double* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCgeqrfBatched(cublasHandle_t handle, int m, int n, cuComplex* const Aarray[], int lda, cuComplex* const TauArray[], int* info, int batchSize){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, long long int strideA, const cuDoubleComplex* B, int ldb, long long int strideB, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc, long long int strideC, int batchCount){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx,  const cuDoubleComplex* beta, cuDoubleComplex* y, int incy){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZgeqrfBatched(cublasHandle_t handle, int m, int n, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const TauArray[], int* info, int batchSize){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t* mode){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* const A[], int lda, cuComplex* const B[], int ldb, int batchCount){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, float* const Aarray[], int lda, float* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSgeqrfBatched(cublasHandle_t handle, int m, int n, float* const Aarray[], int lda, float* const TauArray[], int* info, int batchSize){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda, double* B, int ldb){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* const A[], int lda, float* const B[], int ldb, int batchCount){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle, int n, float* const A[], int lda, int* P, int* info, int batchSize){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float* const Aarray[], int lda, const int* devIpiv, float* const Barray[], int ldb, int* info, int batchSize){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* B, int ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int ldc, long long int strideC, int batchCount){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCdotc_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle, int n, double* const A[], int lda, int* P, int* info, int batchSize){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDotEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, const void* y, cudaDataType yType, int incy, void* result, cudaDataType resultType, cudaDataType executionType){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuComplex* const Aarray[], int lda, cuComplex* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo){
    LOG_DEBUG(CUBLAS, "cublasGemmEx() called.");
    const bool defer_trace_until_wrapper_exit =
        fakecuda::host_timing::should_defer_trace_until_wrapper_exit();
    TracePayloadBuilder payload;
    payload.add_uint64("handle_id", opaque_id(handle));
    cudaStream_t stream = handle ? handle->stream : nullptr;
    fake_stream_mark_work_enqueued(stream);
    payload.add_uint64("stream_id", trace_stream_id(stream));
    payload.add_int("transa", static_cast<int>(transa));
    payload.add_int("transb", static_cast<int>(transb));
    payload.add_int("m", m);
    payload.add_int("n", n);
    payload.add_int("k", k);
    payload.add_int("lda", lda);
    payload.add_int("ldb", ldb);
    payload.add_int("ldc", ldc);
    payload.add_int("Atype", static_cast<int>(Atype));
    payload.add_int("Btype", static_cast<int>(Btype));
    payload.add_int("Ctype", static_cast<int>(Ctype));
    payload.add_int("computeType", static_cast<int>(computeType));
    payload.add_int("algorithm", static_cast<int>(algo));
    payload.add_int("algo", static_cast<int>(algo));
    if (!defer_trace_until_wrapper_exit) {
        TRACE_API_EX(CUBLAS, "cublasGemmEx", "blas_compute", payload);
        TRACE_SYNTHETIC_CUDA_GET_LAST_ERROR("cublasGemmEx");
    }
    LOG_INFO(CUDART,
        "[cublasGemmEx] %c%c  M=%-7d N=%-7d K=%-7d"
        "  ldA=%-6d ldB=%-6d ldC=%-6d  A=%p B=%p C=%p  α=%.4f",
        transa==CUBLAS_OP_N?'N':'T', transb==CUBLAS_OP_N?'N':'T',
        m, n, k, lda, ldb, ldc, A, B, C,
        *reinterpret_cast<const float*>(alpha));
    if (defer_trace_until_wrapper_exit) {
        TRACE_API_EX(CUBLAS, "cublasGemmEx", "blas_compute", payload);
        TRACE_SYNTHETIC_CUDA_GET_LAST_ERROR("cublasGemmEx");
    }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, cuComplex* B, int ldb){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void* workspace, size_t workspaceSizeInBytes){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex* const Aarray[], int lda, const int* devIpiv, cuComplex* const Barray[], int ldb, int* info, int batchSize){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* const A[], int lda, cuDoubleComplex* const B[], int ldb, int batchCount){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle, int n, cuComplex* const A[], int lda, int* P, int* info, int batchSize){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, long long int strideA, const double* B, int ldb, long long int strideB, const double* beta, double* C, int ldc, long long int strideC, int batchCount){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDgeqrfBatched(cublasHandle_t handle, int m, int n, double* const Aarray[], int lda, double* const TauArray[], int* info, int batchSize){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle, int n, cuDoubleComplex* const A[], int lda, int* P, int* info, int batchSize){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const float* beta, void* C, cudaDataType Ctype, int ldc){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasStrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, float* B, int ldb){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, long long int strideA, const void* B, cudaDataType Btype, int ldb, long long int strideB, const void* beta, void* C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo){
    LOG_DEBUG(CUBLAS, "cublasGemmStridedBatchedEx() called.");
    const bool defer_trace_until_wrapper_exit =
        fakecuda::host_timing::should_defer_trace_until_wrapper_exit();
    TracePayloadBuilder payload;
    payload.add_uint64("handle_id", opaque_id(handle));
    cudaStream_t stream = handle ? handle->stream : nullptr;
    fake_stream_mark_work_enqueued(stream);
    payload.add_uint64("stream_id", trace_stream_id(stream));
    payload.add_int("transa", static_cast<int>(transa));
    payload.add_int("transb", static_cast<int>(transb));
    payload.add_int("m", m);
    payload.add_int("n", n);
    payload.add_int("k", k);
    payload.add_int("batch_count", batchCount);
    payload.add_int("lda", lda);
    payload.add_int("ldb", ldb);
    payload.add_int("ldc", ldc);
    payload.add_int64("strideA", static_cast<std::int64_t>(strideA));
    payload.add_int64("strideB", static_cast<std::int64_t>(strideB));
    payload.add_int64("strideC", static_cast<std::int64_t>(strideC));
    payload.add_int("Atype", static_cast<int>(Atype));
    payload.add_int("Btype", static_cast<int>(Btype));
    payload.add_int("Ctype", static_cast<int>(Ctype));
    payload.add_int("computeType", static_cast<int>(computeType));
    payload.add_int("algorithm", static_cast<int>(algo));
    payload.add_int("algo", static_cast<int>(algo));
    if (!defer_trace_until_wrapper_exit) {
        TRACE_API_EX(CUBLAS, "cublasGemmStridedBatchedEx", "blas_compute", payload);
        TRACE_SYNTHETIC_CUDA_GET_LAST_ERROR("cublasGemmStridedBatchedEx");
    }
    LOG_INFO(CUDART,
        "[cublasGemmStridedBatchedEx] %c%c"
        "  M=%-7d N=%-7d K=%-7d  batch=%d"
        "  ldA=%-5d ldB=%-5d ldC=%-5d"
        "  strA=%-10lld strB=%-10lld strC=%-10lld"
        "  A=%p B=%p C=%p  α=%.4f",
        transa==CUBLAS_OP_N?'N':'T', transb==CUBLAS_OP_N?'N':'T',
        m, n, k, batchCount, lda, ldb, ldc, strideA, strideB, strideC, A, B, C,
        *reinterpret_cast<const float*>(alpha));
    if (defer_trace_until_wrapper_exit) {
        TRACE_API_EX(CUBLAS, "cublasGemmStridedBatchedEx", "blas_compute", payload);
        TRACE_SYNTHETIC_CUDA_GET_LAST_ERROR("cublasGemmStridedBatchedEx");
    }
    return CUBLAS_STATUS_SUCCESS;
}
