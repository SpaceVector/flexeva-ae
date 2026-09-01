#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>

#define RED   "\x1B[31m"
#define GRN   "\x1B[32m"
#define RESET "\x1B[0m"

void check_val(const char* name, int value, int min_val) {
    if (value < min_val) {
        printf(RED "[FAIL] %s is %d (Expected >= %d)\n" RESET, name, value, min_val);
        exit(1); // 直接退出，模拟崩溃风险
    } else {
        printf(GRN "[OK]   %s: %d\n" RESET, name, value);
    }
}

int main() {
    printf("=== Starting Fake CUDA Properties Verification ===\n");

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf(RED "Failed to get device count: %d\n" RESET, err);
        return 1;
    }
    printf("Device Count: %d\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        
        if (err != cudaSuccess) {
            printf(RED "Failed to get properties for device %d: %d\n" RESET, dev, err);
            continue;
        }

        printf("--- Checking Device %d: %s ---\n", dev, prop.name);

        // 1. 检查会导致 SIGFPE (除零错误) 的关键参数
        check_val("warpSize", prop.warpSize, 32); 
        check_val("maxThreadsPerBlock", prop.maxThreadsPerBlock, 1);
        check_val("multiProcessorCount", prop.multiProcessorCount, 1);
        
        // 2. 检查 Grid/Block 维度 (防止 Kernel Launch 失败)
        check_val("maxGridSize[0]", prop.maxGridSize[0], 1);
        check_val("maxThreadsDim[0]", prop.maxThreadsDim[0], 1);

        // 3. 检查逻辑一致性 (防止 Occupancy 计算器崩溃)
        // 之前你的配置里 SM最大线程数(80) < Block最大线程数(1024)，这会导致 block_per_sm 计算结果为 0
        if (prop.maxThreadsPerMultiProcessor < prop.maxThreadsPerBlock) {
             printf(RED "[FAIL] maxThreadsPerMultiProcessor (%d) < maxThreadsPerBlock (%d)\n"
                    "       This will cause Occupancy Calculator to return 0 blocks!\n" RESET, 
                    prop.maxThreadsPerMultiProcessor, prop.maxThreadsPerBlock);
             exit(1);
        } else {
            printf(GRN "[OK]   Logic Check: maxThreadsPerSM (%d) >= maxThreadsPerBlock (%d)\n" RESET,
                   prop.maxThreadsPerMultiProcessor, prop.maxThreadsPerBlock);
        }

        // 4. 打印其他重要信息
        printf("[INFO] Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("[INFO] Total Global Mem: %zu bytes\n", prop.totalGlobalMem);
        printf("[INFO] Shared Mem Per Block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("\n");
    }

    printf(GRN "=== ALL CHECKS PASSED. PyTorch should be safe from SIGFPE. ===\n" RESET);
    return 0;
}
