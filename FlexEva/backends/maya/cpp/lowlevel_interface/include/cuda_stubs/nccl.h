// Minimal NCCL stubs for wrapper compilation
// These are opaque type definitions - actual implementations are in fake-cuda

#ifndef NCCL_STUB_H
#define NCCL_STUB_H

#include <stddef.h>
#include "cuda_runtime.h"

// NCCL result type
typedef int ncclResult_t;
#define ncclSuccess 0

// Opaque handle types
typedef struct ncclComm* ncclComm_t;

// Unique ID for communicator initialization
typedef struct {
    char internal[128];
} ncclUniqueId;

// Configuration structure
typedef struct {
    int blocking;
    // Other fields as needed
} ncclConfig_t;

// Data types
typedef enum {
    ncclInt8 = 0,
    ncclChar = 0,
    ncclUint8 = 1,
    ncclInt32 = 2,
    ncclInt = 2,
    ncclUint32 = 3,
    ncclInt64 = 4,
    ncclUint64 = 5,
    ncclFloat16 = 6,
    ncclHalf = 6,
    ncclFloat32 = 7,
    ncclFloat = 7,
    ncclFloat64 = 8,
    ncclDouble = 8,
    ncclBfloat16 = 9
} ncclDataType_t;

// Reduction operations
typedef enum {
    ncclSum = 0,
    ncclProd = 1,
    ncclMax = 2,
    ncclMin = 3,
    ncclAvg = 4
} ncclRedOp_t;

#endif // NCCL_STUB_H
