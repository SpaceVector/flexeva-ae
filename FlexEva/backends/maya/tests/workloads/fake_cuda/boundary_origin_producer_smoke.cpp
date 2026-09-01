#include <cstdlib>
#include <string>
#include <unistd.h>

#include "cpp/fake_cuda/include/common/utils.hpp"
#include "cpp/fake_cuda/include/common/trace_log.hpp"

extern "C" int fake_getDevProps(CUdevice_attribute attrib, int dev) {
    (void)attrib;
    (void)dev;
    return 8;
}

namespace {
void emit_selected_api(int lib, const char* api, const char* type, bool lightweight) {
    fakecuda::trace::maybe_record_wrapper_entry_api(api);
    usleep(1000);
    TracePayloadBuilder payload;
    payload.add_int("smoke_marker", 1);
    if (std::string(api) == "cudaGetDevice") {
        payload.add_int("device", 0);
    } else if (std::string(api) == "__cudaPushCallConfiguration") {
        payload.add_uint("grid_x", 1);
        payload.add_uint("block_x", 1);
        payload.add_size("shared_mem", 0);
    } else if (std::string(api) == "cudaGetLastError") {
        payload.add_int("error", 0);
    } else if (std::string(api) == "cublasGemmEx") {
        payload.add_uint64("handle_id", 1);
        payload.add_uint64("stream_id", 0);
        payload.add_int("m", 16);
        payload.add_int("n", 16);
        payload.add_int("k", 16);
        payload.add_int("algorithm", -1);
    } else if (std::string(api) == "cublasGemmStridedBatchedEx") {
        payload.add_uint64("handle_id", 1);
        payload.add_uint64("stream_id", 0);
        payload.add_int("m", 16);
        payload.add_int("n", 16);
        payload.add_int("k", 16);
        payload.add_int("batch_count", 2);
        payload.add_int("algorithm", -1);
    } else if (std::string(api) == "__cudaPopCallConfiguration") {
        payload.add_uint("grid_x", 1);
        payload.add_uint("block_x", 1);
    } else if (std::string(api) == "cudaLaunchKernel") {
        payload.add_string("kernel", "producer_smoke_kernel");
        payload.add_uint64("stream_id", 0);
    } else if (std::string(api) == "cublasSetStream_v2") {
        payload.add_uint64("handle_id", 1);
        payload.add_uint64("stream_id", 0);
    } else if (std::string(api) == "cudaEventRecord") {
        payload.add_uint64("event_id", 1);
        payload.add_uint64("stream_id", 0);
    } else if (std::string(api) == "cudaStreamWaitEvent") {
        payload.add_uint64("event_id", 1);
        payload.add_uint64("stream_id", 0);
        payload.add_uint("flags", 0);
    }
    if (lightweight) {
        TRACE_API_LIGHT_EX(lib, api, type, payload);
    } else {
        TRACE_API_EX(lib, api, type, payload);
    }
}
}

int main(int argc, char** argv) {
    if (argc < 2) {
        return 2;
    }
    setenv("FAKECUDA_TRACE", "1", 1);
    setenv("FAKECUDA_TRACE_PATH", argv[1], 1);
    setenv("FAKECUDA_HOST_TIMING_MODE", "measure", 1);
    // Keep the synthetic smoke envelope large enough that sub-microsecond
    // body brackets survive JSON millisecond-style rounding to microseconds.

    emit_selected_api(CUDART, "cudaGetDevice", "context_op", true);
    emit_selected_api(CUDART, "__cudaPushCallConfiguration", "other", true);
    emit_selected_api(CUDART, "cudaGetLastError", "other", true);
    emit_selected_api(CUBLAS, "cublasGemmEx", "blas_compute", false);
    emit_selected_api(CUBLAS, "cublasGemmStridedBatchedEx", "blas_compute", false);
    emit_selected_api(CUDART, "__cudaPopCallConfiguration", "other", true);
    emit_selected_api(CUDART, "cudaLaunchKernel", "kernel_launch", false);
    emit_selected_api(CUBLAS, "cublasSetStream_v2", "stream_op", false);
    emit_selected_api(CUDART, "cudaEventRecord", "stream_op", false);
    emit_selected_api(CUDART, "cudaStreamWaitEvent", "stream_op", false);
    return 0;
}
