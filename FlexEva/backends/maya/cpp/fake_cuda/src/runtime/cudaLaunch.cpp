#include "fake_runtime_api.hpp"
#include "utils.hpp"
#include "fake_device_core.h"
#include "function_registry.hpp"
#include "kernel_arg_parsers.h"
#include "trace_log.hpp"


API cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData){
    fprintf(stderr, "[fakecuda] cudaLaunchHostFunc() called.\n");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaLaunchHostFunc);

// ============================================================================
// 函数: cudaLaunchKernel
// 作用: 实际启动 kernel 的函数
// 参数:
//   - func: kernel 函数指针（对应之前注册的 hostFun）
//   - gridDim/blockDim: 网格和块的维度
//   - args: kernel 参数数组
//   - sharedMem: 动态共享内存大小
//   - stream: CUDA 流
// ============================================================================
API cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream){
    LOG_DEBUG(CUDART, "cudaLaunchKernel() called.");
    fake_stream_mark_work_enqueued(stream);
    auto* ki = KernelRegistry::getInstance().FindByHost(func);
    std::string kernel_name = ki ? ki->readableName : std::string("(unknown)");
    const bool defer_trace_until_wrapper_exit =
        fakecuda::host_timing::should_defer_trace_until_wrapper_exit();
    TracePayloadBuilder payload;
    payload.add_string("kernel", kernel_name);
    payload.add_uint("grid_x", gridDim.x);
    payload.add_uint("grid_y", gridDim.y);
    payload.add_uint("grid_z", gridDim.z);
    payload.add_uint("block_x", blockDim.x);
    payload.add_uint("block_y", blockDim.y);
    payload.add_uint("block_z", blockDim.z);
    payload.add_size("shared_mem", sharedMem);
    payload.add_uint64("stream_id", trace_stream_id(stream));
    if (!defer_trace_until_wrapper_exit) {
        TRACE_API_EX(CUDART, "cudaLaunchKernel", "kernel_launch", payload);
        TRACE_SYNTHETIC_CUDA_GET_LAST_ERROR("cudaLaunchKernel");
    }
    LOG_INFO(CUDART, "cudaLaunchKernel %s grid=(%u,%u,%u) block=(%u,%u,%u) smem=%zu stream=%p",
             kernel_name.c_str(),
             gridDim.x, gridDim.y, gridDim.z,
             blockDim.x, blockDim.y, blockDim.z,
             sharedMem, (void*)stream);
    dispatchKernelArgs(ki->readableName, args, gridDim, blockDim, sharedMem); // ← 新增这一行
    if (defer_trace_until_wrapper_exit) {
        TRACE_API_EX(CUDART, "cudaLaunchKernel", "kernel_launch", payload);
        TRACE_SYNTHETIC_CUDA_GET_LAST_ERROR("cudaLaunchKernel");
    }
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaLaunchKernel);

API cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config, const void *func, void **args){
    fprintf(stderr, "[fakecuda] cudaLaunchKernelExC() called.\n");
    fake_setLastError(cudaSuccess);
    return cudaSuccess;
}
REGISTER_CUDA_FUNCTION(cudaLaunchKernelExC);
