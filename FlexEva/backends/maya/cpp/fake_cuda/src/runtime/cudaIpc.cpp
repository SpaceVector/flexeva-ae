#include "fake_runtime_api.hpp"
#include "utils.hpp"
#include "fake_device_core.h"
#include "function_registry.hpp"


API cudaError_t cudaIpcCloseMemHandle(void *devPtr){
    fprintf(stderr, "[fakecuda] cudaIpcCloseMemHandle() called.\n");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaIpcCloseMemHandle);

API cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle, cudaEvent_t event){
    fprintf(stderr, "[fakecuda] cudaIpcGetEventHandle() called.\n");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaIpcGetEventHandle);

API cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr){
    fprintf(stderr, "[fakecuda] cudaIpcGetMemHandle() called.\n");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaIpcGetMemHandle);

API cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event, cudaIpcEventHandle_t handle){
    fprintf(stderr, "[fakecuda] cudaIpcOpenEventHandle() called.\n");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaIpcOpenEventHandle);

API cudaError_t cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags){
    fprintf(stderr, "[fakecuda] cudaIpcOpenMemHandle() called.\n");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaIpcOpenMemHandle);