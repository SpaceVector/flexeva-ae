/**
 * CUDA Runtime API fake implementation
 *
 * This file declares CUDA runtime functions for wrapper generation.
 * The scan_fakes tool parses this to generate config/cuda_wrappers.json.
 *
 * Actual implementation is provided by fake-cuda runtime libraries.
 * These stubs just need correct signatures for wrapper generation.
 */

#include <cstddef>
#include <cstdint>

// CUDA types (minimal definitions for signature parsing)
typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;

struct dim3 {
    unsigned int x, y, z;
};

struct uint3 {
    unsigned int x, y, z;
};

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

extern "C" {

// ============================================================================
// Memory Management
// ============================================================================

cudaError_t cudaMalloc(void** devPtr, size_t size) {
    (void)devPtr; (void)size;
    return 0;
}

cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream) {
    (void)devPtr; (void)size; (void)stream;
    return 0;
}

cudaError_t cudaFree(void* devPtr) {
    (void)devPtr;
    return 0;
}

cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t stream) {
    (void)devPtr; (void)stream;
    return 0;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    (void)dst; (void)src; (void)count; (void)kind;
    return 0;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream) {
    (void)dst; (void)src; (void)count; (void)kind; (void)stream;
    return 0;
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    (void)devPtr; (void)value; (void)count;
    return 0;
}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
    (void)devPtr; (void)value; (void)count; (void)stream;
    return 0;
}

// ============================================================================
// Device Management
// ============================================================================

cudaError_t cudaSetDevice(int device) {
    (void)device;
    return 0;
}

cudaError_t cudaGetDevice(int* device) {
    (void)device;
    return 0;
}

cudaError_t cudaGetDeviceCount(int* count) {
    (void)count;
    return 0;
}

cudaError_t cudaDeviceSynchronize() {
    return 0;
}

// ============================================================================
// Stream Management
// ============================================================================

cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
    (void)pStream;
    return 0;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) {
    (void)pStream; (void)flags;
    return 0;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority) {
    (void)pStream; (void)flags; (void)priority;
    return 0;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    (void)stream;
    return 0;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    (void)stream;
    return 0;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    (void)stream; (void)event; (void)flags;
    return 0;
}

// ============================================================================
// Event Management
// ============================================================================

cudaError_t cudaEventCreate(cudaEvent_t* event) {
    (void)event;
    return 0;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    (void)event; (void)flags;
    return 0;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    (void)event;
    return 0;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    (void)event; (void)stream;
    return 0;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    (void)event;
    return 0;
}
cudaError_t cudaEventQuery(cudaEvent_t event) {
    (void)event;
    return 0;
}

cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) {
    (void)event; (void)stream; (void)flags;
    return 0;
}

cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
    (void)free; (void)total;
    return 0;
}
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    (void)ms; (void)start; (void)end;
    return 0;
}

// ============================================================================
// Kernel Launch
// ============================================================================

cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                             void** args, size_t sharedMem, cudaStream_t stream) {
    (void)func; (void)gridDim; (void)blockDim;
    (void)args; (void)sharedMem; (void)stream;
    return 0;
}

// ============================================================================
// CUDA Internal Registration API
// ============================================================================

void** __cudaRegisterFatBinary(void* fatCubin) {
    (void)fatCubin;
    return nullptr;
}

void __cudaRegisterFatBinaryEnd(void** fatCubinHandle) {
    (void)fatCubinHandle;
}

void __cudaUnregisterFatBinary(void** fatCubinHandle) {
    (void)fatCubinHandle;
}

void __cudaRegisterVar(void** fatCubinHandle, char* hostVar, char* deviceAddress,
                       const char* deviceName, int ext, size_t size,
                       int constant, int global) {
    (void)fatCubinHandle; (void)hostVar; (void)deviceAddress;
    (void)deviceName; (void)ext; (void)size; (void)constant; (void)global;
}

void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun,
                            char* deviceFun, const char* deviceName,
                            int thread_limit, uint3* tid, uint3* bid,
                            dim3* bDim, dim3* gDim, int* wSize) {
    (void)fatCubinHandle; (void)hostFun; (void)deviceFun; (void)deviceName;
    (void)thread_limit; (void)tid; (void)bid; (void)bDim; (void)gDim; (void)wSize;
}

void __cudaInitModule(void** fatCubinHandle) {
    (void)fatCubinHandle;
}

cudaError_t __cudaPopCallConfiguration(dim3* gridDim, dim3* blockDim,
                                       size_t* sharedMem, void** stream) {
    (void)gridDim; (void)blockDim; (void)sharedMem; (void)stream;
    return 0;
}

unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                     size_t sharedMem, void* stream) {
    (void)gridDim; (void)blockDim; (void)sharedMem; (void)stream;
    return 0;
}

// ============================================================================
// Error Handling
// ============================================================================

cudaError_t cudaGetLastError() {
    return 0;
}

cudaError_t cudaPeekAtLastError() {
    return 0;
}

} // extern "C"

