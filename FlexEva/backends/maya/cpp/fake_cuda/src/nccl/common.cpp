#include "fake_types.h"
#include "fake_runtime_api.hpp"
#include "nccl.hpp"
#include <cstring>
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <stdio.h>
#include <unistd.h>
#include "utils.hpp"
#include "../../include/common/trace_log.hpp"

namespace {
std::atomic<unsigned long long> g_nccl_uid_counter{1};
std::mutex g_nccl_trace_mutex;
std::unordered_map<ncclComm_t, unsigned long long> g_nccl_call_counters;

std::uint64_t opaque_id(const void* value) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(value));
}

std::string nccl_comm_id_string(const ncclComm_t comm) {
    if (comm == nullptr) {
        return "";
    }
    size_t len = 0;
    while (len < sizeof(comm->commId.internal) && comm->commId.internal[len] != '\0') {
        ++len;
    }
    if (len > 0) {
        return std::string(comm->commId.internal, comm->commId.internal + len);
    }
    return std::to_string(opaque_id(comm));
}

unsigned long long nccl_next_call_idx(ncclComm_t comm) {
    std::lock_guard<std::mutex> lock(g_nccl_trace_mutex);
    return g_nccl_call_counters[comm]++;
}

TracePayloadBuilder nccl_payload_common(
    const char* collective_name,
    ncclComm_t comm,
    size_t count,
    ncclDataType_t datatype,
    cudaStream_t stream
) {
    TracePayloadBuilder payload;
    payload.add_string("collective", collective_name);
    payload.add_string("comm_id", nccl_comm_id_string(comm));
    payload.add_uint64("call_idx", nccl_next_call_idx(comm));
    payload.add_size("count", count);
    payload.add_int("datatype", static_cast<int>(datatype));
    payload.add_int("nranks", comm ? comm->nranks : 0);
    payload.add_uint64("stream_id", trace_stream_id(stream));
    return payload;
}

ncclResult_t nccl_comm_init_common(
    const char* api_name,
    ncclComm_t* comm,
    int nranks,
    ncclUniqueId commId,
    int rank,
    int blocking
) {
    if (comm == nullptr) {
        LOG_ERROR(NCCL, "comm pointer is NULL\n");
        return ncclInvalidArgument;
    }

    *comm = (ncclComm_t)calloc(1, sizeof(struct ncclComm));
    if (*comm == nullptr) {
        LOG_ERROR(NCCL, "Failed to allocate comm\n");
        return ncclSystemError;
    }

    (*comm)->rank = rank;
    (*comm)->nranks = nranks;
    (*comm)->asyncError = ncclSuccess;
    (*comm)->commId = commId;
    (*comm)->blocking = blocking ? 1 : 0;

    TracePayloadBuilder payload;
    payload.add_string("comm_id", nccl_comm_id_string(*comm));
    payload.add_int("rank", rank);
    payload.add_int("nranks", nranks);
    payload.add_int("world_size", nranks);
    TRACE_API_EX(NCCL, api_name, "other", payload);
    return ncclSuccess;
}
}  // namespace


API ncclResult_t  ncclGetUniqueId(ncclUniqueId* uniqueId){
    LOG_DEBUG(NCCL, "ncclGetUniqueId() called.");
    if (uniqueId == nullptr) {
        return ncclInvalidArgument;
    }
    std::memset(uniqueId->internal, 0, sizeof(uniqueId->internal));
    unsigned long long uid = g_nccl_uid_counter.fetch_add(1);
    snprintf(uniqueId->internal, sizeof(uniqueId->internal), "fake-nccl-%llu", uid);
    TracePayloadBuilder payload;
    payload.add_string("comm_id", uniqueId->internal);
    TRACE_API_EX(NCCL, "ncclGetUniqueId", "other", payload);
    return ncclSuccess;
}

API ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream){
    LOG_DEBUG(NCCL, "ncclAllReduce() called.");
    const bool defer_trace_until_wrapper_exit =
        fakecuda::host_timing::should_defer_trace_until_wrapper_exit();
    TracePayloadBuilder payload = nccl_payload_common("allreduce", comm, count, datatype, stream);
    payload.add_int("op", static_cast<int>(op));
    if (!defer_trace_until_wrapper_exit) {
        TRACE_API_EX(NCCL, "ncclAllReduce", "nccl_collective", payload);
    }
    
    if (sendbuff == nullptr || recvbuff == nullptr || count == 0) {
        if (comm != nullptr) comm->asyncError = ncclSuccess;
        if (defer_trace_until_wrapper_exit) {
            TRACE_API_EX(NCCL, "ncclAllReduce", "nccl_collective", payload);
        }
        return sendbuff == nullptr || recvbuff == nullptr ? ncclInvalidArgument : ncclSuccess;
    }
    
    size_t byte_count = count * get_dtype_size(datatype);
    
    // 只处理 metadata
    if (byte_count <= METADATA_THRESHOLD) {
        memcpy(recvbuff, sendbuff, byte_count);
        LOG_INFO(NCCL, "ncclAllReduce: copied %zu bytes", byte_count);
    } else {
        LOG_INFO(NCCL, "ncclAllReduce: skipped %zu bytes", byte_count);
    }
    
    if (comm != nullptr) {
        comm->asyncError = ncclSuccess;
    }

    if (defer_trace_until_wrapper_exit) {
        TRACE_API_EX(NCCL, "ncclAllReduce", "nccl_collective", payload);
    }
    return ncclSuccess;    
}

API ncclResult_t  ncclCommAbort(ncclComm_t comm){
    LOG_DEBUG(NCCL, "ncclCommAbort() called.");
    return ncclSuccess;
}

API ncclResult_t  ncclMemFree(void *ptr){
    LOG_DEBUG(NCCL, "ncclMemFree() called.");
    free(ptr);
    return ncclSuccess;
}

API ncclResult_t  ncclCommUserRank(const ncclComm_t comm, int* rank){
    LOG_DEBUG(NCCL, "ncclCommUserRank() called.");
    if (comm == nullptr || rank == nullptr) {
        return ncclInvalidArgument;
    }
    *rank = comm->rank;
    TracePayloadBuilder payload;
    payload.add_string("comm_id", nccl_comm_id_string(comm));
    TRACE_API_EX(NCCL, "ncclCommUserRank", "other", payload);
    return ncclSuccess;
}

API ncclResult_t  ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle){
    LOG_DEBUG(NCCL, "ncclCommRegister() called.");
    if (comm == nullptr || buff == nullptr || handle == nullptr) {
        return ncclInvalidArgument;
    }
    *handle = buff;
    return ncclSuccess;
}

API ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm){
    LOG_DEBUG(NCCL, "ncclRedOpDestroy() called.");
    return ncclSuccess;
}

API ncclResult_t  ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm){
    LOG_DEBUG(NCCL, "ncclRedOpCreatePreMulSum() called.");
    return ncclSuccess;
}

API ncclResult_t  ncclCommFinalize(ncclComm_t comm){
    LOG_DEBUG(NCCL, "ncclCommFinalize() called.");
    return ncclSuccess;
}

API ncclResult_t  ncclCommDeregister(const ncclComm_t comm, void* handle){
    LOG_DEBUG(NCCL, "ncclCommDeregister() called.");
    return ncclSuccess;
}

API ncclResult_t  ncclGroupSimulateEnd(ncclSimInfo_t* simInfo){
    LOG_DEBUG(NCCL, "ncclGroupSimulateEnd() called.");
    return ncclSuccess;
}

API ncclResult_t  ncclGroupEnd(){
    LOG_DEBUG(NCCL, "ncclGroupEnd() called.");
    TRACE_API(NCCL, "ncclGroupEnd", "other");
    return ncclSuccess;
}

API ncclResult_t  ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream){
    LOG_DEBUG(NCCL, "ncclBroadcast() called.");
    const bool defer_trace_until_wrapper_exit =
        fakecuda::host_timing::should_defer_trace_until_wrapper_exit();
    TracePayloadBuilder payload = nccl_payload_common("broadcast", comm, count, datatype, stream);
    payload.add_int("root", root);
    if (!defer_trace_until_wrapper_exit) {
        TRACE_API_EX(NCCL, "ncclBroadcast", "nccl_collective", payload);
    }
    if (recvbuff == nullptr || count == 0) {
        if (defer_trace_until_wrapper_exit) {
            TRACE_API_EX(NCCL, "ncclBroadcast", "nccl_collective", payload);
        }
        return recvbuff == nullptr ? ncclInvalidArgument : ncclSuccess;
    }
    
    size_t byte_count = count * get_dtype_size(datatype);
    
    // 只处理 metadata
    if (byte_count <= METADATA_THRESHOLD && sendbuff != nullptr && sendbuff != recvbuff) {
        memcpy(recvbuff, sendbuff, byte_count);
        LOG_INFO(NCCL, "ncclBroadcast: copied %zu bytes", byte_count);
    } else {
        LOG_INFO(NCCL, "ncclBroadcast: skipped %zu bytes", byte_count);
    }

    if (defer_trace_until_wrapper_exit) {
        TRACE_API_EX(NCCL, "ncclBroadcast", "nccl_collective", payload);
    }
    return ncclSuccess;    
}

API ncclResult_t  ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream){
    LOG_DEBUG(NCCL, "ncclBcast() called.");
    const bool defer_trace_until_wrapper_exit =
        fakecuda::host_timing::should_defer_trace_until_wrapper_exit();
    TracePayloadBuilder payload = nccl_payload_common("broadcast", comm, count, datatype, stream);
    payload.add_int("root", root);
    if (!defer_trace_until_wrapper_exit) {
        TRACE_API_EX(NCCL, "ncclBcast", "nccl_collective", payload);
    }
    if (defer_trace_until_wrapper_exit) {
        TRACE_API_EX(NCCL, "ncclBcast", "nccl_collective", payload);
    }
    return ncclSuccess;
}

API ncclResult_t  ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError){
    LOG_DEBUG_NOENTRY(NCCL, "ncclCommGetAsyncError() called.");
    if (asyncError != nullptr) {
        *asyncError = ncclSuccess;
    }
    TracePayloadBuilder payload;
    payload.add_string("comm_id", nccl_comm_id_string(comm));
    payload.add_int("async_error", asyncError ? static_cast<int>(*asyncError) : static_cast<int>(ncclSuccess));
    TRACE_API_LIGHT_EX(NCCL, "ncclCommGetAsyncError", "other", payload);
    return ncclSuccess;
}

API const char*  ncclGetLastError(ncclComm_t comm){
    LOG_DEBUG(NCCL, "ncclGetLastError() called.");
    return "fake ncclGetLastError";
}

API ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags){
    LOG_DEBUG(NCCL, "ncclCommWindowRegister() called.");
    if (comm == nullptr || buff == nullptr || win == nullptr) {
        return ncclInvalidArgument;
    }
    *win = reinterpret_cast<ncclWindow_t>(buff);
    return ncclSuccess;
}

API ncclResult_t  ncclCommDestroy(ncclComm_t comm){
    // fprintf(stderr, "[fakenccl] ncclCommDestroy called, comm=%p\n", (void*)comm);
    LOG_DEBUG(NCCL, "ncclCommDestroy() called.");
    TracePayloadBuilder payload;
    payload.add_string("comm_id", nccl_comm_id_string(comm));
    payload.add_int("nranks", comm ? comm->nranks : 0);
    TRACE_API_EX(NCCL, "ncclCommDestroy", "other", payload);
    if (comm) {
        std::lock_guard<std::mutex> lock(g_nccl_trace_mutex);
        g_nccl_call_counters.erase(comm);
        free(comm);  // ⭐ 释放内存
    }
    return ncclSuccess;
}

API ncclResult_t  ncclCommCount(const ncclComm_t comm, int* count){
    LOG_DEBUG(NCCL, "ncclCommCount() called.");
    if (comm == nullptr || count == nullptr) {
        return ncclInvalidArgument;
    }
    *count = comm->nranks;
    TracePayloadBuilder payload;
    payload.add_string("comm_id", nccl_comm_id_string(comm));
    TRACE_API_EX(NCCL, "ncclCommCount", "other", payload);
    return ncclSuccess;
}

API ncclResult_t  ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist){
    LOG_DEBUG(NCCL, "ncclCommInitAll() called.");
    return ncclSuccess;
}

API ncclResult_t  ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config){
    // fprintf(stderr, "[fakenccl] ncclCommInitRankConfig called, rank=%d, nranks=%d\n", rank, nranks);
    LOG_DEBUG(NCCL, "ncclCommInitRankConfig() called.");
    return nccl_comm_init_common(
        "ncclCommInitRankConfig",
        comm,
        nranks,
        commId,
        rank,
        (config && config->blocking) ? 1 : 0
    );
}

API ncclResult_t  ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream){
    LOG_DEBUG(NCCL, "ncclRecv() called.");
    const bool defer_trace_until_wrapper_exit =
        fakecuda::host_timing::should_defer_trace_until_wrapper_exit();
    TracePayloadBuilder payload = nccl_payload_common("recv", comm, count, datatype, stream);
    payload.add_int("peer", peer);
    if (!defer_trace_until_wrapper_exit) {
        TRACE_API_EX(NCCL, "ncclRecv", "nccl_collective", payload);
    }
    if (defer_trace_until_wrapper_exit) {
        TRACE_API_EX(NCCL, "ncclRecv", "nccl_collective", payload);
    }
    return ncclSuccess;    
}

API ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win){
    LOG_DEBUG(NCCL, "ncclCommWindowDeregister() called.");
    return ncclSuccess; 
}

API ncclResult_t  ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank){
    LOG_DEBUG(NCCL, "ncclCommInitRank() called.");
    return nccl_comm_init_common("ncclCommInitRank", comm, nranks, commId, rank, 0);
}

API ncclResult_t  ncclGetVersion(int *version){
    // fprintf(stderr, "[fakenccl][PID:%d] ncclGetVersion called\n", getpid());
    LOG_DEBUG(NCCL, "ncclGetVersion() called.");
    TRACE_API(NCCL, "ncclGetVersion", "other");
    if (version) *version = 22620; // NCCL 2.26.2
    return ncclSuccess;
}

API const char*  ncclGetErrorString(ncclResult_t result){
    LOG_DEBUG(NCCL, "ncclGetErrorString() called.");
    return "fake ncclGetErrorString";
}

API ncclResult_t  ncclMemAlloc(void** ptr, size_t size){
    LOG_DEBUG(NCCL, "ncclMemAlloc() called.");
    if (ptr == nullptr) {
        return ncclInvalidArgument;
    }
    void* allocated = malloc(size > 0 ? size : 1);
    if (allocated == nullptr) {
        return ncclSystemError;
    }
    *ptr = allocated;
    return ncclSuccess;
}

API ncclResult_t  ncclGroupStart(){
    LOG_DEBUG(NCCL, "ncclGroupStart() called.");
    TRACE_API(NCCL, "ncclGroupStart", "other");
    return ncclSuccess;
}

API ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream){
    LOG_DEBUG(NCCL, "ncclSend() called.");
    const bool defer_trace_until_wrapper_exit =
        fakecuda::host_timing::should_defer_trace_until_wrapper_exit();
    TracePayloadBuilder payload = nccl_payload_common("send", comm, count, datatype, stream);
    payload.add_int("peer", peer);
    if (!defer_trace_until_wrapper_exit) {
        TRACE_API_EX(NCCL, "ncclSend", "nccl_collective", payload);
    }
    if (defer_trace_until_wrapper_exit) {
        TRACE_API_EX(NCCL, "ncclSend", "nccl_collective", payload);
    }
    return ncclSuccess;
}

API ncclResult_t ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commIds, ncclConfig_t* config){
    LOG_DEBUG(NCCL, "ncclCommInitRankScalable() called.");
    return ncclSuccess;
}

API ncclResult_t  ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream){
    LOG_DEBUG(NCCL, "ncclReduceScatter() called.");
    const bool defer_trace_until_wrapper_exit =
        fakecuda::host_timing::should_defer_trace_until_wrapper_exit();
    TracePayloadBuilder payload = nccl_payload_common("reducescatter", comm, recvcount, datatype, stream);
    payload.add_int("op", static_cast<int>(op));
    if (!defer_trace_until_wrapper_exit) {
        TRACE_API_EX(NCCL, "ncclReduceScatter", "nccl_collective", payload);
    }

    if (sendbuff == nullptr || recvbuff == nullptr || recvcount == 0) {
        if (defer_trace_until_wrapper_exit) {
            TRACE_API_EX(NCCL, "ncclReduceScatter", "nccl_collective", payload);
        }
        return sendbuff == nullptr || recvbuff == nullptr ? ncclInvalidArgument : ncclSuccess;
    }
    
    size_t recv_bytes = recvcount * get_dtype_size(datatype);
    
    // 只处理 metadata
    if (recv_bytes <= METADATA_THRESHOLD) {
        int rank = (comm != nullptr) ? comm->rank : 0;
        void* src = (char*)sendbuff + (rank * recv_bytes);
        memcpy(recvbuff, src, recv_bytes);
        LOG_INFO(NCCL, "ncclReduceScatter: copied %zu bytes", recv_bytes);
    } else {
        LOG_INFO(NCCL, "ncclReduceScatter: skipped %zu bytes", recv_bytes);
    }

    if (defer_trace_until_wrapper_exit) {
        TRACE_API_EX(NCCL, "ncclReduceScatter", "nccl_collective", payload);
    }
    return ncclSuccess;
}

API ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream){
    LOG_DEBUG(NCCL, "ncclAllGather() called.");
    const bool defer_trace_until_wrapper_exit =
        fakecuda::host_timing::should_defer_trace_until_wrapper_exit();
    TracePayloadBuilder payload = nccl_payload_common("allgather", comm, sendcount, datatype, stream);
    if (!defer_trace_until_wrapper_exit) {
        TRACE_API_EX(NCCL, "ncclAllGather", "nccl_collective", payload);
    }
    if (sendbuff == nullptr || recvbuff == nullptr || sendcount == 0) {
        if (defer_trace_until_wrapper_exit) {
            TRACE_API_EX(NCCL, "ncclAllGather", "nccl_collective", payload);
        }
        return sendbuff == nullptr || recvbuff == nullptr ? ncclInvalidArgument : ncclSuccess;
    }

    size_t dtype_size = get_dtype_size(datatype);
    size_t send_bytes = sendcount * dtype_size;
    
    // 只处理 metadata（小数据）
    if (send_bytes <= METADATA_THRESHOLD) {
        int rank = (comm != nullptr) ? comm->rank : 0;
        int nranks = (comm != nullptr) ? comm->nranks : 1;
        
        // 将 sendbuff 复制到 recvbuff 的所有位置（模拟所有 rank 数据一致）
        for (int i = 0; i < nranks; i++) {
            void* dst = (char*)recvbuff + (i * send_bytes);
            memcpy(dst, sendbuff, send_bytes);
        }
        LOG_INFO(NCCL, "ncclAllGather: copied %zu bytes", send_bytes);
    } else {
        LOG_INFO(NCCL, "ncclAllGather: skipped %zu bytes", send_bytes);
    }

    if (defer_trace_until_wrapper_exit) {
        TRACE_API_EX(NCCL, "ncclAllGather", "nccl_collective", payload);
    }
    return ncclSuccess;
}

API ncclResult_t  ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t* config){
    LOG_DEBUG(NCCL, "ncclCommSplit() called.");
    if (newcomm == nullptr) {
        return ncclInvalidArgument;
    }
    *newcomm = nullptr;
    if (comm == nullptr) {
        return ncclInvalidArgument;
    }
    if (color < 0) {
        TracePayloadBuilder payload;
        payload.add_string("comm_id", nccl_comm_id_string(comm));
        payload.add_string("parent_comm_id", nccl_comm_id_string(comm));
        payload.add_int("color", color);
        payload.add_int("key", key);
        payload.add_string("new_comm_id", "");
        TRACE_API_EX(NCCL, "ncclCommSplit", "other", payload);
        return ncclSuccess;
    }
    *newcomm = (ncclComm_t)calloc(1, sizeof(struct ncclComm));
    if (*newcomm == nullptr) {
        return ncclSystemError;
    }
    (*newcomm)->rank = key;
    (*newcomm)->nranks = comm->nranks;
    (*newcomm)->asyncError = ncclSuccess;
    (*newcomm)->commId = comm->commId;
    (*newcomm)->blocking = config && config->blocking ? 1 : comm->blocking;
    TracePayloadBuilder payload;
    payload.add_string("comm_id", nccl_comm_id_string(comm));
    payload.add_string("parent_comm_id", nccl_comm_id_string(comm));
    payload.add_int("color", color);
    payload.add_int("key", key);
    payload.add_string("new_comm_id", nccl_comm_id_string(*newcomm));
    TRACE_API_EX(NCCL, "ncclCommSplit", "other", payload);
    return ncclSuccess;
}

API ncclResult_t  ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream){
    LOG_DEBUG(NCCL, "ncclReduce() called.");
    const bool defer_trace_until_wrapper_exit =
        fakecuda::host_timing::should_defer_trace_until_wrapper_exit();
    TracePayloadBuilder payload = nccl_payload_common("reduce", comm, count, datatype, stream);
    payload.add_int("op", static_cast<int>(op));
    payload.add_int("root", root);
    if (!defer_trace_until_wrapper_exit) {
        TRACE_API_EX(NCCL, "ncclReduce", "nccl_collective", payload);
    }
    // if (recvbuff) {
    //     memset(recvbuff, 0, count);
    // }
    if (defer_trace_until_wrapper_exit) {
        TRACE_API_EX(NCCL, "ncclReduce", "nccl_collective", payload);
    }
    return ncclSuccess;
}
