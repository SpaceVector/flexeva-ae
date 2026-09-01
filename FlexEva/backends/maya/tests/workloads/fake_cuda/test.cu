#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <cuda_runtime.h>

// 定义一个结构体来记录分配的指针和大小
struct MemBlock {
    void* ptr;
    size_t size;
};

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    // 初始化随机种子
    srand((unsigned int)time(NULL));

    std::vector<MemBlock> blocks;
    size_t total_allocated = 0;
    cudaError_t err;
    
    printf("=== 第一阶段：随机分配直到 OOM ===\n");

    while (1) {
        // 随机生成 100MB 到 2GB 之间的大小
        // (rand() % 1900 + 100) -> 100~1999 MB
        size_t current_size = (size_t)(rand() % 1900 + 100) * 1024 * 1024;
        
        void* d_ptr = NULL;
        err = cudaMalloc(&d_ptr, current_size);

        if (err != cudaSuccess) {
            printf("\n[停止分配] 遇到错误: %s (Code: %d)\n", cudaGetErrorString(err), err);
            printf("最终分配总量: %.2f GB\n", (double)total_allocated / (1024 * 1024 * 1024));
            printf("总共分配块数: %zu 个\n", blocks.size());
            break;
        }

        // 记录分配成功的指针
        blocks.push_back({d_ptr, current_size});
        total_allocated += current_size;

        printf("\r[分配中] 块大小: %4zu MB | 总显存: %6.2f GB", 
               current_size / (1024 * 1024), 
               (double)total_allocated / (1024 * 1024 * 1024));
        fflush(stdout);
    }

    printf("\n\n=== 第二阶段：开始释放内存 ===\n");

    // 倒序释放（模拟栈的后进先出，避免内存碎片化干扰测试，虽然对cudaFree不强制）
    while (!blocks.empty()) {
        MemBlock block = blocks.back();
        blocks.pop_back();

        err = cudaFree(block.ptr);
        if (err != cudaSuccess) {
            printf("释放失败! Ptr: %p, Error: %s\n", block.ptr, cudaGetErrorString(err));
        }

        total_allocated -= block.size;
        
        // 每释放 5 个块打印一次状态，避免刷屏
        if (blocks.size() % 5 == 0) {
            printf("\r[释放中] 剩余块数: %4zu | 剩余显存占用: %6.2f GB", 
                   blocks.size(), 
                   (double)total_allocated / (1024 * 1024 * 1024));
            fflush(stdout);
        }
    }

    printf("\n\n=== 测试结束，显存已清空 ===\n");
    return 0;
}
