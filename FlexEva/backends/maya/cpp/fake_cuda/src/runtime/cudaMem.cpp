#include "fake_runtime_api.hpp"
#include "utils.hpp"
#include "fake_device_core.h"
#include "function_registry.hpp"
#include "trace_log.hpp"


API cudaError_t cudaMemGetInfo ( size_t* free, size_t* total ){
    LOG_DEBUG(CUDART, "cudaMemGetInfo() called.");
    if(free != nullptr && total != nullptr){
        *total = fake_getDeviceTotalMem();
        *free = fake_getDeviceFreeMem();
        TracePayloadBuilder payload;
        payload.add_size("free_bytes", *free);
        payload.add_size("total_bytes", *total);
        TRACE_API_EX(CUDART, "cudaMemGetInfo", "context_op", payload);
        fake_setLastError(cudaSuccess);
        return cudaSuccess;
    }else{
        fake_setLastError(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
}

API cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind){
    LOG_DEBUG(CUDART, "cudaMemcpy() called.");
    if (dst == nullptr || src == nullptr || count == 0) {
        fake_setLastError(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    if (count <= METADATA_THRESHOLD) {
        memcpy(dst, src, count);
    }
    TracePayloadBuilder payload;
    payload.add_size("bytes", count);
    payload.add_int("kind", static_cast<int>(kind));
    TRACE_API_EX(CUDART, "cudaMemcpy", "mem_copy", payload);
    // if (kind == cudaMemcpyDeviceToHost) {
    //     memset(dst, 0, count); 
    // }
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMemcpy);

API cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)){
    LOG_DEBUG(CUDART, "cudaMemcpy2DAsync() called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMemcpy2DAsync);

API cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)){
    LOG_DEBUG(CUDART, "cudaMemcpyAsync() called.");
    fake_stream_mark_work_enqueued(stream);
    memcpy(dst, src, count);
    TracePayloadBuilder payload;
    payload.add_size("bytes", count);
    payload.add_int("kind", static_cast<int>(kind));
    payload.add_uint64("stream_id", trace_stream_id(stream));
    TRACE_API_EX(CUDART, "cudaMemcpyAsync", "mem_copy", payload);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMemcpyAsync);

API cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream __dv(0)){
    LOG_DEBUG(CUDART, "cudaMemcpyPeerAsync() called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMemcpyPeerAsync);

API cudaError_t cudaMemset(void *devPtr, int value, size_t count){
    LOG_DEBUG(CUDART, "cudaMemset() called.");
    TracePayloadBuilder payload;
    payload.add_size("bytes", count);
    payload.add_int("value", value);
    TRACE_API_EX(CUDART, "cudaMemset", "mem_copy", payload);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMemset);

API cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream __dv(0)){
    LOG_DEBUG(CUDART, "cudaMemsetAsync() called.");
    fake_stream_mark_work_enqueued(stream);
    if (devPtr && count > 0) {
        memset(devPtr, value, count);
    }
    TracePayloadBuilder payload;
    payload.add_size("bytes", count);
    payload.add_int("value", value);
    payload.add_uint64("stream_id", trace_stream_id(stream));
    TRACE_API_EX(CUDART, "cudaMemsetAsync", "mem_copy", payload);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMemsetAsync);

API cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value ){
    LOG_DEBUG(CUDART, "cudaMemPoolGetAttribute called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMemPoolGetAttribute);

API cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const struct cudaMemAccessDesc *descList, size_t count){
    LOG_DEBUG(CUDART, "cudaMemPoolSetAccess called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMemPoolSetAccess);

API cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value ){
    LOG_DEBUG(CUDART, "cudaMemPoolSetAttribute called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMemPoolSetAttribute);

API cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep){
    LOG_DEBUG(CUDART, "cudaMemPoolTrimTo called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaMemPoolTrimTo);
