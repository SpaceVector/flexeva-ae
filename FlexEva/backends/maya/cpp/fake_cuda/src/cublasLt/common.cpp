#include "fake_types.h"
#include "fake_runtime_api.hpp"
#include "utils.hpp"
#include "cublas.hpp"
#include "../../include/common/trace_log.hpp"

namespace {
std::uint64_t opaque_id(const void* value) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(value));
}
}

API cublasStatus_t cublasLtCreate(cublasLtHandle_t* lightHandle){
    LOG_DEBUG(CUBLASLT, "cublasLtCreate() called.");
    if (lightHandle == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    cublasLtContext* ctx = (struct cublasLtContext*)malloc(sizeof(struct cublasLtContext));
    if (!ctx) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    ctx->magic_number = 0xBEEF;
    ctx->stream = 0;
    *lightHandle = ctx;
    TracePayloadBuilder payload;
    payload.add_uint64("handle_id", opaque_id(ctx));
    TRACE_API_EX(CUBLASLT, "cublasLtCreate", "context_op", payload);
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc, cublasLtMatmulPreference_t preference, int requestedAlgoCount, cublasLtMatmulHeuristicResult_t heuristicResultsArray[], int* returnAlgoCount){
    LOG_DEBUG(CUBLASLT, "cublasLtMatmulAlgoGetHeuristic() called.");
    // 关键：必须返回至少一个算法
    *returnAlgoCount = (requestedAlgoCount > 0) ? 1 : 0;
    if (requestedAlgoCount > 0) std::memset(heuristicResultsArray, 0, sizeof(*heuristicResultsArray));
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc){
    LOG_DEBUG(CUBLASLT, "cublasLtMatmulDescDestroy() called.");
    free(matmulDesc);
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* pref){
    LOG_DEBUG(CUBLASLT, "cublasLtMatmulPreferenceCreate() called.");
    *pref = (cublasLtMatmulPreference_t)malloc(sizeof(int));  // 必须分配内存
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr, const void* buf, size_t sizeInBytes){
    LOG_DEBUG(CUBLASLT, "cublasLtMatmulDescSetAttribute() called.");
    auto* p = (FakeDesc*)matmulDesc;
    if (attr == CUBLASLT_MATMUL_DESC_TRANSA) std::memcpy(&p->transa, buf, sizeInBytes);
    if (attr == CUBLASLT_MATMUL_DESC_TRANSB) std::memcpy(&p->transb, buf, sizeInBytes);
    if (attr == CUBLASLT_MATMUL_DESC_SCALE_TYPE)
        std::memcpy(&p->scaleType, buf, sizeInBytes);
    // 其余属性（bias、epilogue 等）不需要读取，直接忽略
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref){
    LOG_DEBUG(CUBLASLT, "cublasLtMatmulPreferenceDestroy() called.");
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* matLayout, cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld){
    LOG_DEBUG(CUBLASLT, "cublasLtMatrixLayoutCreate() called.");
    auto* p = new FakeLayout{};
    p->rows = rows; p->cols = cols; p->ld = ld;
    *matLayout = (cublasLtMatrixLayout_t)p;
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout){
    delete (FakeLayout*)matLayout;
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr, const void* buf, size_t sizeInBytes){
    LOG_DEBUG(CUBLASLT, "cublasLtMatmulPreferenceSetAttribute() called.");
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* matmulDesc, cublasComputeType_t computeType, cudaDataType_t scaleType){
    LOG_DEBUG(CUBLASLT, "cublasLtMatmulDescCreate() called.");
    *matmulDesc = (cublasLtMatmulDesc_t)malloc(sizeof(int));
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, const void* alpha, /* host or device pointer */ const void* A, cublasLtMatrixLayout_t Adesc, const void* B, cublasLtMatrixLayout_t Bdesc, const void* beta, /* host or device pointer */ const void* C, cublasLtMatrixLayout_t Cdesc, void* D, cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t* algo, void* workspace, size_t workspaceSizeInBytes, cudaStream_t stream){
    LOG_DEBUG(CUBLASLT, "cublasLtMatmul() called.");
    auto* md = (FakeDesc*)computeDesc;
    auto* la = (FakeLayout*)Adesc;
    auto* lb = (FakeLayout*)Bdesc;
    auto* lc = (FakeLayout*)Cdesc;
    cublasOperation_t ta = md->transa, tb = md->transb;
    int64_t M = (ta==CUBLAS_OP_N) ? la->rows : la->cols;
    int64_t K = (ta==CUBLAS_OP_N) ? la->cols : la->rows;
    int64_t N = (tb==CUBLAS_OP_N) ? lb->cols : lb->rows;
    float fa;
    if (md->scaleType == CUDA_R_16F) {
        uint16_t h; std::memcpy(&h, alpha, 2);
        // half to float
        uint32_t f = ((uint32_t)(h & 0x8000) << 16)
                | ((uint32_t)((h & 0x7C00) + 0x1C000) << 13)
                | ((uint32_t)(h & 0x03FF) << 13);
        std::memcpy(&fa, &f, 4);
    } else {
        std::memcpy(&fa, alpha, 4);
    }
    TracePayloadBuilder payload;
    fake_stream_mark_work_enqueued(stream);
    payload.add_uint64("stream_id", trace_stream_id(stream));
    payload.add_int("m", static_cast<int>(M));
    payload.add_int("n", static_cast<int>(N));
    payload.add_int("k", static_cast<int>(K));
    payload.add_int("batch_count", static_cast<int>(la->batch));
    TRACE_API_EX(CUBLASLT, "cublasLtMatmul", "blas_compute", payload);
    if (la->batch <= 1) {
        LOG_INFO(CUDART,
            "[cublasLtMatmul GEMM]  %c%c"
            "  M=%-8ld N=%-8ld K=%-8ld"
            "  ldA=%-7ld ldB=%-7ld ldC=%-7ld"
            "  A=%p B=%p D=%p  α=%.4f",
            ta==CUBLAS_OP_N?'N':'T', tb==CUBLAS_OP_N?'N':'T',
            M, N, K, la->ld, lb->ld, lc->ld, A, B, D, fa);
    } else {
        LOG_INFO(CUDART,
            "[cublasLtMatmul BGEMM] %c%c"
            "  M=%-8ld N=%-8ld K=%-8ld  batch=%d"
            "  ldA=%-6ld ldB=%-6ld ldC=%-6ld"
            "  strA=%-12ld strB=%-12ld strC=%-12ld"
            "  A=%p B=%p D=%p  α=%.4f",
            ta==CUBLAS_OP_N?'N':'T', tb==CUBLAS_OP_N?'N':'T',
            M, N, K, la->batch,
            la->ld, lb->ld, lc->ld,
            la->stride, lb->stride, lc->stride,
            A, B, D, fa);
    }
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr, const void* buf, size_t sizeInBytes){
    LOG_DEBUG(CUBLASLT, "cublasLtMatrixLayoutSetAttribute() called.");
    auto* p = (FakeLayout*)matLayout;
    if (attr == CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT)
        std::memcpy(&p->batch,  buf, sizeInBytes);
    if (attr == CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET)
        std::memcpy(&p->stride, buf, sizeInBytes);
    return CUBLAS_STATUS_SUCCESS;
}
