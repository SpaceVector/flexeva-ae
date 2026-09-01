// test_driver.cpp
// Minimal CUDA Driver-API tester via dlsym, no CUDA headers required.
// Build: g++ -std=c++17 -O2 -ldl -o test_driver test_driver.cpp
// Run:   (set LD_LIBRARY_PATH to where your fake libcuda.so is)

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

using i32 = int;
using u32 = unsigned int;
using u64 = unsigned long long;
using CUdevice = int;
using CUresult = int;
using CUdeviceptr = unsigned long long;

struct CUctx_st 
{
    u64 id;
    int dev;
};
using CUcontext = CUctx_st *;
struct CUstream_st
{
    u64 id;
    int dev;
};
using CUstream = CUstream_st *;
struct CUevent_st
{
    u64 id;
    int dev;
};
using CUevent = CUevent_st *;

// Common CUDA Driver return codes we care about
static constexpr CUresult CUDA_SUCCESS = 0;

#define LOAD_SYM(var, name)                                 \
    var = reinterpret_cast<decltype(var)>(dlsym(h, name));  \
    if (!var)                                               \
    {                                                       \
        fprintf(stderr, "[ERR] missing symbol %s\n", name); \
        ok = false;                                         \
    }

#define CHECK(call)                                                                               \
    do                                                                                            \
    {                                                                                             \
        CUresult _r = (call);                                                                     \
        if (_r != CUDA_SUCCESS)                                                                   \
        {                                                                                         \
            fprintf(stderr, "[ERR] %s failed with %d at %s:%d\n", #call, _r, __FILE__, __LINE__); \
            exit(2);                                                                              \
        }                                                                                         \
    } while (0)

int main()
{
    // --------------- dlopen libcuda.so ---------------
    const char *so = "libcuda.so";
    void *h = dlopen(so, RTLD_NOW | RTLD_LOCAL);
    if (!h)
    {
        fprintf(stderr, "[FATAL] dlopen(%s) failed: %s\n", so, dlerror());
        return 1;
    }

    // --------------- resolve a minimal set of symbols ---------------
    bool ok = true;

    CUresult (*cuInit)(u32) = nullptr;
    CUresult (*cuDriverGetVersion)(int *) = nullptr;
    CUresult (*cuDeviceGetCount)(int *) = nullptr;
    CUresult (*cuDeviceGet)(CUdevice *, int) = nullptr;
    CUresult (*cuDeviceGetName)(char *, int, CUdevice) = nullptr;
    CUresult (*cuDeviceGetAttribute)(int *, int, CUdevice) = nullptr;
    CUresult (*cuMemAddressReserve)(u64*, size_t, size_t, u64, u64) = nullptr;
    CUresult (*cuMemRelease)(u64) = nullptr;
    CUresult (*cuMemMap)(u64, size_t, size_t, u64, u64) = nullptr;
    CUresult (*cuMemAddressFree)(u64, size_t) = nullptr;
    CUresult (*cuMemSetAccess)(u64, size_t, const void*, size_t) = nullptr;
    CUresult (*cuMemUnmap)(u64, size_t) = nullptr;
    CUresult (*cuMemCreate)(u64*, size_t, const void*, u64) = nullptr;
    CUresult (*cuMemGetAllocationGranularity)(size_t*, const void*, int) = nullptr;
    CUresult (*cuMemExportToShareableHandle)(void*, u64, int, u64) = nullptr;
    CUresult (*cuMemImportFromShareableHandle)(u64*, void*, int) = nullptr;
    CUresult (*cuMemsetD32Async)(u64, u32, size_t, CUstream) = nullptr;
    CUresult (*cuStreamWriteValue32)(CUstream, u64, u32, u32) = nullptr;
    CUresult (*cuGetErrorString)(CUresult, const char**) = nullptr;

    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice) = nullptr;
    CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
    CUresult (*cuCtxGetDevice)(CUdevice *) = nullptr;
    CUresult (*cuCtxEnablePeerAccess)(CUcontext, unsigned int) = nullptr;
    CUresult (*cuDeviceCanAccessPeer)(int *, CUdevice, CUdevice) = nullptr;

    CUresult (*cuStreamCreate)(CUstream *, unsigned int) = nullptr;
    CUresult (*cuStreamSynchronize)(CUstream) = nullptr;
    CUresult (*cuStreamDestroy_v2)(CUstream) = nullptr;

    CUresult (*cuMemAlloc_v2)(CUdeviceptr *, size_t) = nullptr;
    CUresult (*cuMemFree_v2)(CUdeviceptr) = nullptr;
    CUresult (*cuMemcpyHtoD_v2)(CUdeviceptr, const void *, size_t) = nullptr;
    CUresult (*cuMemcpyDtoH_v2)(void *, CUdeviceptr, size_t) = nullptr;
    CUresult (*cuMemcpyPeer)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t) = nullptr;
    CUresult (*cuMemsetD8_v2)(CUdeviceptr, unsigned char, size_t) = nullptr;

    CUresult (*cuEventCreate)(CUevent *, unsigned int) = nullptr;
    CUresult (*cuEventRecord)(CUevent, CUstream) = nullptr;
    CUresult (*cuEventSynchronize)(CUevent) = nullptr;
    CUresult (*cuEventDestroy_v2)(CUevent) = nullptr;

    CUresult (*cuCtxSynchronize)() = nullptr;

    LOAD_SYM(cuInit, "cuInit");
    LOAD_SYM(cuDriverGetVersion, "cuDriverGetVersion");
    LOAD_SYM(cuDeviceGetCount, "cuDeviceGetCount");
    LOAD_SYM(cuDeviceGet, "cuDeviceGet");
    LOAD_SYM(cuDeviceGetName, "cuDeviceGetName");
    LOAD_SYM(cuDeviceGetAttribute, "cuDeviceGetAttribute");
    LOAD_SYM(cuMemAddressReserve, "cuMemAddressReserve");
    LOAD_SYM(cuMemRelease, "cuMemRelease");
    LOAD_SYM(cuMemMap, "cuMemMap");
    LOAD_SYM(cuMemAddressFree, "cuMemAddressFree");
    LOAD_SYM(cuMemSetAccess, "cuMemSetAccess");
    LOAD_SYM(cuMemUnmap, "cuMemUnmap");
    LOAD_SYM(cuMemCreate, "cuMemCreate");
    LOAD_SYM(cuMemGetAllocationGranularity, "cuMemGetAllocationGranularity");
    LOAD_SYM(cuMemExportToShareableHandle, "cuMemExportToShareableHandle");
    LOAD_SYM(cuMemImportFromShareableHandle, "cuMemImportFromShareableHandle");
    LOAD_SYM(cuMemsetD32Async, "cuMemsetD32Async");
    LOAD_SYM(cuStreamWriteValue32, "cuStreamWriteValue32");
    LOAD_SYM(cuGetErrorString, "cuGetErrorString");

    LOAD_SYM(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
    LOAD_SYM(cuCtxSetCurrent, "cuCtxSetCurrent");
    LOAD_SYM(cuCtxGetDevice, "cuCtxGetDevice");
    LOAD_SYM(cuCtxEnablePeerAccess, "cuCtxEnablePeerAccess");
    LOAD_SYM(cuDeviceCanAccessPeer, "cuDeviceCanAccessPeer");

    LOAD_SYM(cuStreamCreate, "cuStreamCreate");
    LOAD_SYM(cuStreamSynchronize, "cuStreamSynchronize");
    LOAD_SYM(cuStreamDestroy_v2, "cuStreamDestroy_v2");

    LOAD_SYM(cuMemAlloc_v2, "cuMemAlloc_v2");
    LOAD_SYM(cuMemFree_v2, "cuMemFree_v2");
    LOAD_SYM(cuMemcpyHtoD_v2, "cuMemcpyHtoD_v2");
    LOAD_SYM(cuMemcpyDtoH_v2, "cuMemcpyDtoH_v2");
    LOAD_SYM(cuMemcpyPeer, "cuMemcpyPeer");
    LOAD_SYM(cuMemsetD8_v2, "cuMemsetD8_v2");

    LOAD_SYM(cuEventCreate, "cuEventCreate");
    LOAD_SYM(cuEventRecord, "cuEventRecord");
    LOAD_SYM(cuEventSynchronize, "cuEventSynchronize");
    LOAD_SYM(cuEventDestroy_v2, "cuEventDestroy_v2");

    LOAD_SYM(cuCtxSynchronize, "cuCtxSynchronize");

    if (!ok)
    {
        fprintf(stderr, "[FATAL] Some required symbols were not found. Check your fake libcuda exports.\n");
        return 3;
    }

    // --------------- basic init & device enumeration ---------------
    CHECK(cuInit(0));
    int drv = 0;
    CHECK(cuDriverGetVersion(&drv));
    printf("[info] driver version: %d\n", drv);
    const char* err_str = nullptr;
    CHECK(cuGetErrorString(CUDA_SUCCESS, &err_str));
    printf("[info] cuGetErrorString(CUDA_SUCCESS): %s\n", err_str ? err_str : "<null>");

    int ndev = 0;
    CHECK(cuDeviceGetCount(&ndev));
    printf("[info] device count  : %d\n", ndev);
    if (ndev < 1)
    {
        fprintf(stderr, "[ERR] No device reported. Fake driver should report at least 1.\n");
        return 4;
    }

    std::vector<CUdevice> devs;
    for (int i = 0; i < ndev; ++i)
    {
        CUdevice d;
        CHECK(cuDeviceGet(&d, i));
        char name[256];
        CHECK(cuDeviceGetName(name, 256, d));
        printf("  - dev %d name: %s\n", i, name);

        auto get_attr = [&](int attr, const char *aname)
        {
            int v = 0;
            CHECK(cuDeviceGetAttribute(&v, attr, d));
            printf("    %s: %d\n", aname, v);
        };
        // A few common attributes (IDs are stable across CUDA versions)
        get_attr(10, "warpSize");            // CU_DEVICE_ATTRIBUTE_WARP_SIZE
        get_attr(1, "maxThreadsPerBlock");   // CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        get_attr(16, "multiProcessorCount"); // CU_DEVICE_ATTRIBUTE_MULTI_PROCESSOR_COUNT
        get_attr(13, "clockRate(kHz)");      // CU_DEVICE_ATTRIBUTE_CLOCK_RATE
        get_attr(44, "l2CacheSize(bytes)");  // CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE

        devs.push_back(d);
    }

    // --------------- create/retain primary contexts, set current ---------------
    std::vector<CUcontext> ctxs(ndev, nullptr);
    for (int i = 0; i < ndev; ++i)
    {
        CHECK(cuDevicePrimaryCtxRetain(&ctxs[i], devs[i]));
        CHECK(cuCtxSetCurrent(ctxs[i]));
        int cur_dev = -1;
        CHECK(cuCtxGetDevice(&cur_dev));
        printf("[info] current device after retain(%d): %d\n", i, cur_dev);
    }

    // --------------- streams on dev0 (and dev1 if exists) ---------------
    CUstream s0 = nullptr, s1 = nullptr;
    CHECK(cuCtxSetCurrent(ctxs[0]));
    CHECK(cuStreamCreate(&s0, /*flags=*/0));
    if (ndev > 1)
    {
        CHECK(cuCtxSetCurrent(ctxs[1]));
        CHECK(cuStreamCreate(&s1, 0));
    }

    // --------------- memory alloc/copies on dev0 ---------------
    const size_t N = 1024;
    std::vector<unsigned char> hbuf(N, 42);
    std::vector<unsigned char> hout(N, 0);

    CUdeviceptr d0 = 0, d1 = 0;
    CHECK(cuCtxSetCurrent(ctxs[0]));
    CHECK(cuMemAlloc_v2(&d0, N));
    CHECK(cuMemsetD8_v2(d0, 0, N));
    CHECK(cuMemcpyHtoD_v2(d0, hbuf.data(), N));
    CHECK(cuMemcpyDtoH_v2(hout.data(), d0, N));
    printf("[info] dev0 memcpy roundtrip OK, hout[0]=%u\n", (unsigned)hout[0]);

    // --------------- peer access + peer copy (if 2+ devices) ---------------
    if (ndev > 1)
    {
        int can = 0;
        CHECK(cuDeviceCanAccessPeer(&can, devs[0], devs[1]));
        printf("[info] dev0 canAccess dev1: %d\n", can);
        if (can)
        {
            CHECK(cuCtxSetCurrent(ctxs[0]));
            CHECK(cuCtxEnablePeerAccess(ctxs[1], 0));

            CHECK(cuCtxSetCurrent(ctxs[1]));
            CHECK(cuMemAlloc_v2(&d1, N));
            CHECK(cuMemsetD8_v2(d1, 7, N));

            // copy d0 -> d1
            CHECK(cuMemcpyPeer(d1, ctxs[1], d0, ctxs[0], N));

            std::vector<unsigned char> h2(N, 0);
            CHECK(cuMemcpyDtoH_v2(h2.data(), d1, N));
            printf("[info] peer copy OK, h2[0]=%u\n", (unsigned)h2[0]);
        }
    }

    // --------------- event record/sync on dev0 stream ---------------
    CHECK(cuCtxSetCurrent(ctxs[0]));
    CUevent e0 = nullptr;
    CHECK(cuEventCreate(&e0, 0));
    CHECK(cuEventRecord(e0, s0));
    CHECK(cuEventSynchronize(e0));
    CHECK(cuStreamSynchronize(s0));
    CHECK(cuEventDestroy_v2(e0));
    CHECK(cuCtxSynchronize());

    // --------------- cleanup ---------------
    if (d1)
        CHECK(cuMemFree_v2(d1));
    if (d0)
        CHECK(cuMemFree_v2(d0));
    if (s1)
        CHECK(cuStreamDestroy_v2(s1));
    if (s0)
        CHECK(cuStreamDestroy_v2(s0));

    printf("[info] done.\n");
    return 0;
}
