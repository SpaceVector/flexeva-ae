#include <stdio.h>
#include <dlfcn.h>
#include <cuda_runtime_api.h>

// 定义函数指针类型
typedef cudaError_t (*cudaGetDriverEntryPointByVersion_func)(
    const char *symbolName,
    void **funcPtr,
    unsigned int driverVersion,
    unsigned long long flags, 
    cudaDriverEntryPointQueryResult *driverStatus
);

typedef cudaError_t (*cudaMalloc_func)(void **devPtr, size_t size);

extern "C" {
    // Hook 目标函数
    cudaError_t cudaGetDriverEntryPointByVersion(
        const char *symbolName,
        void **funcPtr,
        unsigned int driverVersion,
        unsigned long long flags,
        cudaDriverEntryPointQueryResult *driverStatus) {

        static cudaGetDriverEntryPointByVersion_func original_func = 
            (cudaGetDriverEntryPointByVersion_func)dlsym(RTLD_NEXT, "cudaGetDriverEntryPointByVersion");

        if (original_func) {
            printf(">> [Hook] cudaGetDriverEntryPointByVersion: %s\n", symbolName);
            return original_func(symbolName, funcPtr, driverVersion, flags, driverStatus);
        }
        return cudaErrorUnknown;
    }

    // Hook 一个常用函数来验证 dlsym/RTLD_NEXT 转发是否生效
    cudaError_t cudaMalloc(void **devPtr, size_t size) {
        static cudaMalloc_func original_malloc = 
            (cudaMalloc_func)dlsym(RTLD_NEXT, "cudaMalloc");

        // 打印日志证明 Hook 库已加载
        printf(">> [Hook] cudaMalloc called (size: %zu)\n", size);

        if (original_malloc) {
            return original_malloc(devPtr, size);
        }
        return cudaErrorUnknown;
    }
}
