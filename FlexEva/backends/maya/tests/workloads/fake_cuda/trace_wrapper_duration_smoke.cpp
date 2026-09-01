#include <unistd.h>

#include "../../../cpp/fake_cuda/include/common/utils.hpp"
#include "../../../cpp/fake_cuda/include/common/trace_log.hpp"

extern "C" int fake_getDevProps(CUdevice_attribute attrib, int dev) {
    (void)dev;
    switch (attrib) {
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
            return 8;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
            return 0;
        default:
            return 0;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        return 2;
    }

    setenv("FAKECUDA_TRACE", "1", 1);
    setenv("FAKECUDA_TRACE_PATH", argv[1], 1);
    setenv("FAKECUDA_HOST_TIMING_MODE", "measure", 1);

    LOG_DEBUG(CUDART, "cudaStreamSynchronize() called.");
    usleep(20 * 1000);
    TRACE_API(CUDART, "cudaStreamSynchronize", "stream_op");

    LOG_DEBUG(CUDART, "cudaMemGetInfo() called.");
    usleep(10 * 1000);
    TracePayloadBuilder payload;
    payload.add_size("free_bytes", 1024);
    payload.add_size("total_bytes", 2048);
    TRACE_API_EX(CUDART, "cudaMemGetInfo", "context_op", payload);

    return 0;
}
