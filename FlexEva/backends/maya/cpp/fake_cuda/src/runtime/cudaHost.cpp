#include "fake_runtime_api.hpp"
#include "utils.hpp"
#include "fake_device_core.h"
#include "function_registry.hpp"


API cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags){
    LOG_DEBUG(CUDART, "cudaHostAlloc() called.");
    void *ptr = mmap(nullptr, size, 
                     PROT_READ | PROT_WRITE, 
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, 
                     -1, 0);
    if (ptr == MAP_FAILED) {
        return cudaErrorMemoryAllocation;
    }
    {
        std::lock_guard<std::mutex> lock(g_host_mutex);
        g_host_allocations[ptr] = size;
    }
    *pHost = ptr;
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaHostAlloc);

API cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags){
    LOG_DEBUG(CUDART, "cudaHostRegister() called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaHostRegister);

API cudaError_t cudaHostUnregister(void *ptr){
    LOG_DEBUG(CUDART, "cudaHostUnregister() called.");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaHostUnregister);