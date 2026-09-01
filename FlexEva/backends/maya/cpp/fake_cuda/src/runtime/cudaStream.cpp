#include "fake_runtime_api.hpp"
#include "utils.hpp"
#include "function_registry.hpp"
#include "trace_log.hpp"


API cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void *userData, unsigned int flags){
    LOG_DEBUG(CUDART, "cudaStreamAddCallback() called.");

    if (callback) {
        // 立即执行callback，因为我们没有真实的异步工作
        // 对PyTorch来说效果等同于GPU工作立即完成
        callback(stream, cudaSuccess, userData);
        LOG_DEBUG(CUDART, "  Callback executed immediately");
    }

    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaStreamAddCallback);

API cudaError_t cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode){
    LOG_DEBUG(CUDART, "cudaStreamBeginCapture() called.");

    FakeStream* fs = to_fake_stream(stream);
    if (!fs) {
        // default stream不支持capture
        fake_setLastError(cudaErrorStreamCaptureUnsupported);
        return cudaErrorStreamCaptureUnsupported;
    }
    
    {
        std::lock_guard<std::mutex> lock(g_streamMutex);
        if (g_validStreams.find(fs) == g_validStreams.end()) {
            fake_setLastError(cudaErrorInvalidResourceHandle);
            return cudaErrorInvalidResourceHandle;
        }
        
        fs->isCapturing = true;
        fs->captureMode = mode;
        fs->captureId = g_captureIdCounter++;
    }
    
    LOG_INFO(CUDART, "  Stream id=%llu began capture, captureId=%llu", 
              fs->id, fs->captureId);

    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaStreamBeginCapture);

API cudaError_t cudaStreamCreate(cudaStream_t *pStream){
    LOG_DEBUG(CUDART, "cudaStreamCreate() called.");
    if (!pStream) {
        fake_setLastError(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    
    FakeStream* stream = new FakeStream();
    stream->id = g_streamIdCounter++;
    stream->flags = 0;
    stream->priority = 0;
    
    {
        std::lock_guard<std::mutex> lock(g_streamMutex);
        g_validStreams.insert(stream);
    }
    
    *pStream = to_cuda_stream(stream);
    TracePayloadBuilder payload;
    payload.add_uint64("stream_id", stream->id);
    TRACE_API_EX(CUDART, "cudaStreamCreate", "stream_op", payload);
    
    LOG_INFO(CUDART, "  Created stream id=%llu handle=%p", stream->id, *pStream);

    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaStreamCreate);

API cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority){
    LOG_DEBUG(CUDART, "cudaStreamCreateWithPriority() called.");

    if (!pStream) {
        fake_setLastError(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    
    FakeStream* stream = new FakeStream();
    stream->id = g_streamIdCounter++;
    stream->flags = flags;
    stream->priority = priority;
    
    {
        std::lock_guard<std::mutex> lock(g_streamMutex);
        g_validStreams.insert(stream);
    }
    
    *pStream = to_cuda_stream(stream);
    TracePayloadBuilder payload;
    payload.add_uint64("stream_id", stream->id);
    payload.add_uint("flags", flags);
    payload.add_int("priority", priority);
    TRACE_API_EX(CUDART, "cudaStreamCreateWithPriority", "stream_op", payload);
    
    // 为什么记录priority：
    // PyTorch高优先级stream用于通信overlap，
    // 低优先级用于计算，记录可以复现调度分析
    LOG_INFO(CUDART, "  Created stream id=%llu handle=%p priority=%d", 
              stream->id, *pStream, priority);

    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaStreamCreateWithPriority);

API cudaError_t cudaStreamDestroy(cudaStream_t stream){
    LOG_DEBUG(CUDART, "cudaStreamDestroy() called.");
    unsigned long long stream_id = trace_stream_id(stream);

    if (stream == nullptr) {
        // default stream不能被destroy
        fake_setLastError(cudaErrorInvalidResourceHandle);
        return cudaErrorInvalidResourceHandle;
    }
    
    FakeStream* fs = to_fake_stream(stream);
    
    {
        std::lock_guard<std::mutex> lock(g_streamMutex);
        auto it = g_validStreams.find(fs);
        if (it == g_validStreams.end()) {
            LOG_WARN(CUDART, "  Destroying invalid stream handle=%p", stream);
            fake_setLastError(cudaErrorInvalidResourceHandle);
            return cudaErrorInvalidResourceHandle;
        }
        g_validStreams.erase(it);
    }
    
    LOG_INFO(CUDART, "  Destroyed stream id=%llu handle=%p", fs->id, stream);
    TracePayloadBuilder payload;
    payload.add_uint64("stream_id", stream_id);
    TRACE_API_EX(CUDART, "cudaStreamDestroy", "stream_op", payload);
    delete fs;

    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaStreamDestroy);

API cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph){
    LOG_DEBUG(CUDART, "cudaStreamEndCapture() called.");

    FakeStream* fs = to_fake_stream(stream);
    if (!fs) {
        fake_setLastError(cudaErrorInvalidResourceHandle);
        return cudaErrorInvalidResourceHandle;
    }
    
    {
        std::lock_guard<std::mutex> lock(g_streamMutex);
        if (g_validStreams.find(fs) == g_validStreams.end()) {
            fake_setLastError(cudaErrorInvalidResourceHandle);
            return cudaErrorInvalidResourceHandle;
        }
        
        LOG_INFO(CUDART, "  Stream id=%llu ended capture, captureId=%llu", 
                  fs->id, fs->captureId);
        
        fs->isCapturing = false;
        fs->captureId = 0;
    }
    
    // 为什么pGraph可以返回nullptr：
    // 我们不真实执行graph，但必须给PyTorch一个非null的图句柄
    // 否则PyTorch在后续cudaGraphInstantiate时会崩溃
    // 这里返回一个fake指针作为占位
    if (pGraph) {
        // TODO: 如果你需要实现Graph相关API，这里需要返回真实的fake graph句柄
        *pGraph = reinterpret_cast<cudaGraph_t>(0xDEADBEEF);
    }

    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaStreamEndCapture);

API cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, enum cudaStreamCaptureStatus *captureStatus_out, unsigned long long *id_out __dv(0), cudaGraph_t *graph_out __dv(0), const cudaGraphNode_t **dependencies_out __dv(0), size_t *numDependencies_out __dv(0)){
    LOG_DEBUG(CUDART, "cudaStreamGetCaptureInfo() called.");

    if (!captureStatus_out) {
        fake_setLastError(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    
    if (stream == nullptr) {
        *captureStatus_out = cudaStreamCaptureStatusNone;
        if (id_out) *id_out = 0;
        if (graph_out) *graph_out = nullptr;
        if (dependencies_out) *dependencies_out = nullptr;
        if (numDependencies_out) *numDependencies_out = 0;
        fake_setLastError(cudaSuccess);
        return cudaSuccess;
    }
    
    FakeStream* fs = to_fake_stream(stream);
    
    {
        std::lock_guard<std::mutex> lock(g_streamMutex);
        if (g_validStreams.find(fs) == g_validStreams.end()) {
            fake_setLastError(cudaErrorInvalidResourceHandle);
            return cudaErrorInvalidResourceHandle;
        }
        
        *captureStatus_out = fs->isCapturing ? 
                             cudaStreamCaptureStatusActive : 
                             cudaStreamCaptureStatusNone;
        if (id_out) *id_out = fs->captureId;
        
        // 我们没有真实的graph节点
        if (graph_out) *graph_out = nullptr;
        if (dependencies_out) *dependencies_out = nullptr;
        if (numDependencies_out) *numDependencies_out = 0;
    }

    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaStreamGetCaptureInfo);
API cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, enum cudaStreamCaptureStatus *captureStatus_out, unsigned long long *id_out __dv(0), cudaGraph_t *graph_out __dv(0), const cudaGraphNode_t **dependencies_out __dv(0), size_t *numDependencies_out __dv(0)){
    return cudaStreamGetCaptureInfo(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);
}
REGISTER_CUDA_FUNCTION(cudaStreamGetCaptureInfo_v2);

// 为什么GetPriority需要返回真实值：
// PyTorch会根据priority来决定stream的调度顺序
// 返回错误的priority会影响调度分析的准确性
API cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int *priority){
    LOG_DEBUG(CUDART, "cudaStreamGetPriority() called.");

    if (!priority) {
        fake_setLastError(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    
    if (hStream == nullptr) {
        *priority = 0;  // default stream的priority为0
        fake_setLastError(cudaSuccess);
        return cudaSuccess;
    }
    
    FakeStream* fs = to_fake_stream(hStream);
    
    {
        std::lock_guard<std::mutex> lock(g_streamMutex);
        if (g_validStreams.find(fs) == g_validStreams.end()) {
            fake_setLastError(cudaErrorInvalidResourceHandle);
            return cudaErrorInvalidResourceHandle;
        }
        *priority = fs->priority;
    }
    
    LOG_INFO(CUDART, "  Stream id=%llu priority=%d", fs->id, *priority);

    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaStreamGetPriority);

API cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus){
    LOG_DEBUG(CUDART, "cudaStreamIsCapturing() called.");

    if (!pCaptureStatus) {
        fake_setLastError(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    
    if (stream == nullptr) {
        // default stream永远不在capture状态
        *pCaptureStatus = cudaStreamCaptureStatusNone;
        fake_setLastError(cudaSuccess);
        return cudaSuccess;
    }
    
    FakeStream* fs = to_fake_stream(stream);
    
    {
        std::lock_guard<std::mutex> lock(g_streamMutex);
        if (g_validStreams.find(fs) == g_validStreams.end()) {
            fake_setLastError(cudaErrorInvalidResourceHandle);
            return cudaErrorInvalidResourceHandle;
        }
        
        *pCaptureStatus = fs->isCapturing ? 
                          cudaStreamCaptureStatusActive : 
                          cudaStreamCaptureStatusNone;
    }
    
    LOG_INFO(CUDART, "  Stream id=%llu isCapturing=%d", fs->id, fs->isCapturing);

    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaStreamIsCapturing);

API cudaError_t cudaStreamQuery(cudaStream_t stream){
    LOG_DEBUG(CUDART, "cudaStreamQuery() called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaStreamQuery);

API cudaError_t cudaStreamSynchronize(cudaStream_t stream){
    LOG_DEBUG(CUDART, "cudaStreamSynchronize() called.");
    fake_stream_mark_completed_through(stream, fake_stream_enqueued_seq(stream));
    fake_mark_ready_events_visible(stream);
    TracePayloadBuilder payload;
    payload.add_uint64("stream_id", trace_stream_id(stream));
    TRACE_API_EX(CUDART, "cudaStreamSynchronize", "stream_op", payload);

    // 永远返回完成（cudaSuccess表示stream中所有工作已完成）

    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaStreamSynchronize);

API cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags __dv(0)){
    LOG_DEBUG(CUDART, "cudaStreamWaitEvent() called.");
    fake_stream_mark_work_enqueued(stream);
    TracePayloadBuilder payload;
    payload.add_uint64("stream_id", trace_stream_id(stream));
    payload.add_uint64("event_id", trace_event_id(event));
    payload.add_uint("flags", flags);
    TRACE_API_EX(CUDART, "cudaStreamWaitEvent", "stream_op", payload);

    // 记录stream-event依赖关系（可选，用于调度分析）
    // TODO: 如果需要分析stream间依赖，在这里记录

    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaStreamWaitEvent);
