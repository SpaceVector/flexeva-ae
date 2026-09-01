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


API cublasStatus_t  cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId){
    LOG_DEBUG(CUBLAS, "cublasSetStream_v2() called.");
    if (handle != nullptr) {
        handle->stream = streamId;
    }
    TracePayloadBuilder payload;
    payload.add_uint64("handle_id", opaque_id(handle));
    payload.add_uint64("stream_id", trace_stream_id(streamId));
    TRACE_API_EX(CUBLAS, "cublasSetStream_v2", "stream_op", payload);
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasZdotu_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result){
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double* const Aarray[], int lda, const int* devIpiv, double* const Barray[], int ldb, int* info, int batchSize){
    return CUBLAS_STATUS_SUCCESS;
}
