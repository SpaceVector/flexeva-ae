#include <cstdlib>
#include <string>

#include "cpp/fake_cuda/include/common/utils.hpp"
#include "cpp/fake_cuda/include/common/trace_log.hpp"
#include "cpp/fake_cuda/include/runtime/fake_runtime_api.hpp"
#include "cpp/fake_cuda/include/cublas/cublas.hpp"

extern "C" cudaError_t cudaGetDevice(int* device);
extern "C" cudaError_t cudaGetLastError();
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, void* stream);
extern "C" cudaError_t __cudaPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem, void** stream);
extern "C" cudaError_t cudaEventCreate(cudaEvent_t* event);
extern "C" cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
extern "C" cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
extern "C" cublasStatus_t cublasCreate_v2(cublasHandle_t* handle);
extern "C" cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
extern "C" cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId);
extern "C" cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
extern "C" cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, long long int strideA, const void* B, cudaDataType Btype, int ldb, long long int strideB, const void* beta, void* C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);

namespace {
void fake_kernel_for_boundary_origin_smoke() {}

void register_safe_fake_kernel() {
    static int fatbin_anchor = 0;
    static void* fatbin_handle = &fatbin_anchor;
    static void** fatbin_handle_ptr = &fatbin_handle;
    KernelRegistry::getInstance().RegisterFunction(
        fatbin_handle_ptr,
        reinterpret_cast<const void*>(&fake_kernel_for_boundary_origin_smoke),
        "boundary_origin_smoke_kernel",
        "boundary_origin_smoke_kernel",
        0
    );
}
}

int main(int argc, char** argv) {
    if (argc < 2) {
        return 2;
    }
    setenv("FAKECUDA_TRACE", "1", 1);
    setenv("FAKECUDA_TRACE_PATH", argv[1], 1);
    setenv("FAKECUDA_HOST_TIMING_MODE", "measure", 1);

    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return 10;
    }

    dim3 pushed_grid(2, 1, 1);
    dim3 pushed_block(4, 1, 1);
    cudaStream_t default_stream = nullptr;
    __cudaPushCallConfiguration(pushed_grid, pushed_block, 16, default_stream);
    dim3 popped_grid;
    dim3 popped_block;
    size_t popped_shared = 0;
    void* popped_stream = nullptr;
    if (__cudaPopCallConfiguration(&popped_grid, &popped_block, &popped_shared, &popped_stream) != cudaSuccess) {
        return 12;
    }

    cublasHandle_t handle = nullptr;
    if (cublasCreate_v2(&handle) != CUBLAS_STATUS_SUCCESS || handle == nullptr) {
        return 13;
    }
    if (cublasSetStream_v2(handle, default_stream) != CUBLAS_STATUS_SUCCESS) {
        return 14;
    }
    float alpha = 1.0f;
    float beta = 0.0f;
    float a[16] = {};
    float b[16] = {};
    float c[16] = {};
    if (cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            4,
            4,
            4,
            &alpha,
            a,
            CUDA_R_32F,
            4,
            b,
            CUDA_R_32F,
            4,
            &beta,
            c,
            CUDA_R_32F,
            4,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
        return 20;
    }
    if (cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            4,
            4,
            4,
            &alpha,
            a,
            CUDA_R_32F,
            4,
            16,
            b,
            CUDA_R_32F,
            4,
            16,
            &beta,
            c,
            CUDA_R_32F,
            4,
            16,
            1,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
        return 21;
    }
    if (cudaGetLastError() != cudaSuccess) {
        return 22;
    }

    cudaEvent_t event = nullptr;
    if (cudaEventCreate(&event) != cudaSuccess || event == nullptr) {
        return 15;
    }
    if (cudaEventRecord(event, default_stream) != cudaSuccess) {
        return 16;
    }
    if (cudaStreamWaitEvent(default_stream, event, 0) != cudaSuccess) {
        return 17;
    }

    register_safe_fake_kernel();
    if (cudaLaunchKernel(
            reinterpret_cast<const void*>(&fake_kernel_for_boundary_origin_smoke),
            dim3(1, 1, 1),
            dim3(1, 1, 1),
            nullptr,
            0,
            default_stream) != cudaSuccess) {
        return 18;
    }

    if (cublasDestroy_v2(handle) != CUBLAS_STATUS_SUCCESS) {
        return 19;
    }
    return 0;
}
