/**
 * NCCL API fake implementation
 *
 * This file declares NCCL functions for wrapper generation.
 * The scan_fakes tool parses this to generate config/nccl_wrappers.json.
 *
 * Actual implementation is provided by fake-cuda runtime libraries.
 * These stubs just need correct signatures for wrapper generation.
 */

#include <cstddef>
#include <cstdint>

// NCCL types (minimal definitions for signature parsing)
typedef int ncclResult_t;
typedef void* ncclComm_t;
typedef void* cudaStream_t;

struct ncclUniqueId {
    char internal[128];
};

typedef struct ncclConfig_v21700 {
    size_t size;
    unsigned int magic;
    unsigned int version;
    int blocking;
    int cgaClusterSize;
    int minCTAs;
    int maxCTAs;
    const char* netName;
    int splitShare;
} ncclConfig_t;

enum ncclDataType_t {
    ncclInt8 = 0, ncclChar = 0,
    ncclUint8 = 1,
    ncclInt32 = 2, ncclInt = 2,
    ncclUint32 = 3,
    ncclInt64 = 4,
    ncclUint64 = 5,
    ncclFloat16 = 6, ncclHalf = 6,
    ncclFloat32 = 7, ncclFloat = 7,
    ncclFloat64 = 8, ncclDouble = 8,
    ncclBfloat16 = 9
};

enum ncclRedOp_t {
    ncclSum = 0,
    ncclProd = 1,
    ncclMax = 2,
    ncclMin = 3,
    ncclAvg = 4
};

extern "C" {

// ============================================================================
// Initialization
// ============================================================================

ncclResult_t ncclGetVersion(int* version) {
    (void)version;
    return 0;
}

ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
    (void)uniqueId;
    return 0;
}

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
    (void)comm; (void)nranks; (void)commId; (void)rank;
    return 0;
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config) {
    (void)comm; (void)nranks; (void)commId; (void)rank; (void)config;
    return 0;
}

ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
    (void)comms; (void)ndev; (void)devlist;
    return 0;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    (void)comm;
    return 0;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) {
    (void)comm;
    return 0;
}

ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
    (void)comm; (void)count;
    return 0;
}

ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device) {
    (void)comm; (void)device;
    return 0;
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
    (void)comm; (void)rank;
    return 0;
}

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError) {
    (void)comm; (void)asyncError;
    return 0;
}

// ============================================================================
// Collective Operations
// ============================================================================

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
    (void)sendbuff; (void)recvbuff; (void)count;
    (void)datatype; (void)op; (void)comm; (void)stream;
    return 0;
}

ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, int root,
                           ncclComm_t comm, cudaStream_t stream) {
    (void)sendbuff; (void)recvbuff; (void)count;
    (void)datatype; (void)root; (void)comm; (void)stream;
    return 0;
}

ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream) {
    (void)sendbuff; (void)recvbuff; (void)count;
    (void)datatype; (void)op; (void)root; (void)comm; (void)stream;
    return 0;
}

ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                           ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
    (void)sendbuff; (void)recvbuff; (void)sendcount;
    (void)datatype; (void)comm; (void)stream;
    return 0;
}

ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                               ncclDataType_t datatype, ncclRedOp_t op,
                               ncclComm_t comm, cudaStream_t stream) {
    (void)sendbuff; (void)recvbuff; (void)recvcount;
    (void)datatype; (void)op; (void)comm; (void)stream;
    return 0;
}

// ============================================================================
// Point-to-Point Operations
// ============================================================================

ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
    (void)sendbuff; (void)count; (void)datatype;
    (void)peer; (void)comm; (void)stream;
    return 0;
}

ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
    (void)recvbuff; (void)count; (void)datatype;
    (void)peer; (void)comm; (void)stream;
    return 0;
}

// ============================================================================
// Group Operations
// ============================================================================

ncclResult_t ncclGroupStart() {
    return 0;
}

ncclResult_t ncclGroupEnd() {
    return 0;
}

} // extern "C"

