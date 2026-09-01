#include <cstdlib> 

// API 可见性
#define API extern "C" __attribute__((visibility("default")))


struct FakeLayout {
    uint64_t rows = 0, cols = 0;
    int64_t  ld   = 0;
    int32_t  batch  = 1;
    int64_t  stride = 0;
};
struct FakeDesc {
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    cudaDataType_t    scaleType = CUDA_R_32F;
};