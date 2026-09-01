#include "fake_runtime_api.hpp"
#include "utils.hpp"
#include "fake_device_core.h"
#include "function_registry.hpp"
#include "trace_log.hpp"


API cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice){
    LOG_DEBUG(CUDART, "cudaDeviceCanAccessPeer() called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaDeviceCanAccessPeer);


API cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags){
    LOG_DEBUG(CUDART, "cudaDeviceEnablePeerAccess() called.");
    if(flags != 0){
        fake_setLastError(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    if(peerDevice < 0 || peerDevice >= fake_getDeviceCount()){
        fake_setLastError(cudaErrorInvalidDevice);
        return cudaErrorInvalidDevice;
    }
    // fprintf(stderr, "[fakecuda] Peer access to device %d enabled.\n", peerDevice);
    int current_device;
    fake_getDevice(&current_device);
    // fprintf(stderr, "[fakecuda] now device %d .\n",current_device);
    return cudaSuccess;
    
}
REGISTER_CUDA_FUNCTION(cudaDeviceEnablePeerAccess);

API cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device){
    LOG_DEBUG(CUDART, "cudaDeviceGetAttribute() called.");
    *value = fake_getDevProps(static_cast<CUdevice_attribute>(attr), device);
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaDeviceGetAttribute);

API cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool, int device){
    LOG_DEBUG(CUDART, "cudaDeviceGetDefaultMemPool() called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaDeviceGetDefaultMemPool);

API cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device){
    LOG_DEBUG(CUDART, "cudaDeviceGetPCIBusId() called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaDeviceGetPCIBusId);

API cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority){
    LOG_DEBUG(CUDART, "cudaDeviceGetStreamPriorityRange() called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaDeviceGetStreamPriorityRange);

API cudaError_t cudaDeviceSynchronize(void){
    LOG_DEBUG(CUDART, "cudaDeviceSynchronize() called.");
    // Maya paper-facing workloads close the measured step after an explicit
    // device synchronize. fake-cuda must therefore advance all outstanding
    // stream work before returning, otherwise the step marker can close while
    // virtual device work is still pending.
    fake_mark_all_streams_completed();
    fake_mark_ready_events_visible();
    int current_device = 0;
    fake_getDevice(&current_device);
    TracePayloadBuilder payload;
    payload.add_int("device", current_device);
    TRACE_API_EX(CUDART, "cudaDeviceSynchronize", "stream_op", payload);
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaDeviceSynchronize);
