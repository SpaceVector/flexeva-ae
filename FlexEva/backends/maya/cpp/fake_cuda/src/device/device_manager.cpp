#include "device_manager.hpp"
#include "fake_types.h"
#include <fstream>
#include <sys/mman.h>

using json = nlohmann::json;


// 初始化线程局部静态成员
thread_local int DeviceManager::currentDevice = 0;  // 默认设备0
thread_local cudaError_t DeviceManager::lastError = cudaSuccess; // 默认无错误

DeviceManager& DeviceManager::getInstance() {
    static DeviceManager instance;
    return instance;
}


// Function to parse UUID string (e.g., "432cc443-e848-f714-df0e-3e0256064883") into 16 bytes
void parse_uuid_string(const std::string& uuid_str, char* bytes) {
    if (uuid_str.length() != 36) return;  // Standard UUID string length
    size_t byte_index = 0;
    for (size_t i = 0; i < 36 && byte_index < 16; ++i) {
        if (uuid_str[i] == '-') continue;  // Skip hyphens
        int byte_val;
        if (sscanf(uuid_str.substr(i, 2).c_str(), "%02x", &byte_val) != 1) return;
        bytes[byte_index++] = static_cast<char>(byte_val);
        ++i;  // Skip the next char (we processed 2 hex digits)
    }
    return;  // Ensure exactly 16 bytes
}

namespace fs = std::filesystem;
std::string get_config_path() {
    // PROJECT_ROOT_DIR 是由 CMake 注入的字符串常量
    fs::path root_path(PROJECT_ROOT_DIR);
    fs::path config_file = root_path / "config" / "config.json";
    
    return config_file.string();
}

bool DeviceManager::doInitialize() {
    // 在这里添加实际的初始化逻辑
    try{
        // 配置文件路径
        std::string config_path = get_config_path();
        std::ifstream config_file(config_path);
        if (!config_file.is_open()) {
                fprintf(stderr, "[fakecuda] WARN: Can't open config file!\n");
                return false;
        }
        json config_data = json::parse(config_file, nullptr, true, true);
        try{
            DeviceManager::driver_version = config_data["driver_version"];
            DeviceManager::device_count = config_data["device_count"];
            for(int i = 0; i < DeviceManager::device_count; i++) {
                DevProps temp;
                temp.name = config_data["devices"][i]["name"];
                parse_uuid_string(config_data["devices"][i]["uuid"], temp.uuid.bytes);
                temp.id = i;
                temp.DeviceTotalMem = config_data["devices"][i]["DeviceTotalMem"];
                temp.CU_DEVICE_ATTRIBUTE_CLOCK_RATE = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_CLOCK_RATE"];
                temp.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE"];
                temp.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH"];
                temp.CU_DEVICE_ATTRIBUTE_MAX_PITCH = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAX_PITCH"];
                temp.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR"];
                temp.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR"];
                temp.CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO"];
                temp.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK"];
                temp.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X"];
                temp.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y"];
                temp.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z"];
                temp.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X"];
                temp.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y"];
                temp.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z"];
                temp.CU_DEVICE_ATTRIBUTE_WARP_SIZE = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_WARP_SIZE"];
                temp.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT"];
                temp.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR"];
                temp.CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR"];
                temp.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK"];
                temp.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR"];
                temp.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN"];
                temp.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY"];
                temp.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE"];
                temp.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK"];
                temp.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR"];
                temp.CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED"];
                temp.CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS"];
                temp.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING"];
                temp.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY"];
                temp.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS"];
                temp.CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST"];
                temp.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS"];
                temp.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES"];
                temp.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED"];
                temp.CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED"];
                temp.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS"];
                temp.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT"];
                temp.CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED"];
                temp.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH"];
                temp.CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED"];
                temp.CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED"];
                temp.CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE"];
                temp.CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED"];
                temp.CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES"];
                temp.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED"];
                temp.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_COMPUTE_MODE"];
                temp.CU_DEVICE_ATTRIBUTE_ECC_ENABLED = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_ECC_ENABLED"];
                temp.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY"];
                temp.CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED"];
                temp.CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT"];
                temp.CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT"];
                temp.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH"];
                temp.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT"];
                temp.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH"];
                temp.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT"];
                temp.CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT"];
                temp.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_PCI_BUS_ID"];
                temp.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID"];
                temp.CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID"];
                temp.CU_DEVICE_ATTRIBUTE_TCC_DRIVER = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_TCC_DRIVER"];
                temp.CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED"];
                temp.CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED"];
                temp.CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM"];
                temp.CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH"];
                temp.CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE"];
                temp.CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED"];
                temp.CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED"];
                temp.CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED"];
                temp.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS"];
                temp.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING"];
                temp.CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED"];
                temp.CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = config_data["devices"][i]["CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH"];
                DeviceManager::devices.push_back(std::move(temp));
            }
        }catch(...){
            fprintf(stderr, "[fakecuda] WARN: Invalid config file format!\n");
            DeviceManager::devices.clear();
            return false;
        }
        std::lock_guard<std::mutex> lock(ctx_mutex);
        for (int i = 0; i < DeviceManager::device_count; ++i) {
            CUctx_st* ctx = new CUctx_st();
            ctx->id = i;
            ctx->magic = 0xCAFE1234;
            fake_contexts.push_back(ctx);
        }
        return true;
    }catch(...){
        DeviceManager::devices.clear();
        return false;
    }
}

bool DeviceManager::initialize(){
    if (initialized.load(std::memory_order_acquire))
        return true; // 快速路径：已初始化，无需加锁
    std::lock_guard<std::mutex> lock(init_mutex);
    if (initialized.load(std::memory_order_relaxed))
        return true; // 双重检查（Double-Checked Locking），防止重复初始化
    // 执行实际初始化逻辑
    bool success = doInitialize(); // 你的初始化函数，可能失败
    if (success)
        initialized.store(true, std::memory_order_release);
    return success;
}

bool DeviceManager::isInitialized() const noexcept {
    return initialized.load(std::memory_order_acquire);
}

int DeviceManager::getDriverVersion() const noexcept {
    return driver_version;
}

int DeviceManager::getDeviceCount() const noexcept {
    return device_count;
}

std::string DeviceManager::getDeviceName(int dev) const noexcept{
    return devices[dev].name;
}

size_t DeviceManager::getDeviceTotalMem(int device_id) const noexcept{
    if(device_id == -1){
        device_id = currentDevice;
    }
    return devices[device_id].DeviceTotalMem;
}

size_t DeviceManager::getDeviceFreeMem(int device_id) const noexcept{
    if(device_id == -1){
        device_id = currentDevice;
    }
    return devices[device_id].DeviceTotalMem - devices[device_id].DeviceUsedMem;
}

int DeviceManager::getDevProps(CUdevice_attribute attrib, int dev) const noexcept{
    switch (attrib){
        case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
            return devices[dev].CU_DEVICE_ATTRIBUTE_CLOCK_RATE;
        case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE;
        case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH:
            return devices[dev].CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH;
        case CU_DEVICE_ATTRIBUTE_MAX_PITCH:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAX_PITCH;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
            return devices[dev].CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
            return devices[dev].CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
        case CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO:
            return devices[dev].CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO;
        case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z;
        case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
            return devices[dev].CU_DEVICE_ATTRIBUTE_WARP_SIZE;
        case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
        case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR;
        case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK;
        case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR;
        case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN;
        case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY:
            return devices[dev].CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY;
        case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE:
            return devices[dev].CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE;
        case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK;
        case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR:  
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR;
        case CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED:
            return devices[dev].CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED;
        case CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS:
            return devices[dev].CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS;
        case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING:
            return devices[dev].CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING;
        case CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY;
        case CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS:
            return devices[dev].CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS;
        case CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST:
            return devices[dev].CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST;
        case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS:
            return devices[dev].CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS;
        case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES:
            return devices[dev].CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES;
        case CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED:
            // Keep driver-level VMM symbols available for modern PyTorch loader
            // compatibility, but conservatively report VMM unsupported so fake-cuda
            // does not advertise semantics we do not model paper-faithfully.
            return 0;
        case CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED;
        case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS:
            return devices[dev].CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS;
        case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT:
            return devices[dev].CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT;
        case CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED:
            return devices[dev].CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED;
        case CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH:
            return devices[dev].CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH;
        case CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED:
            return devices[dev].CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED;
        case CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED:
            return devices[dev].CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED;
        case CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE;
        case CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED:
            return devices[dev].CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED;
        case CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES;
        case CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED:
            return 0;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE:
            return devices[dev].CU_DEVICE_ATTRIBUTE_COMPUTE_MODE;
        case CU_DEVICE_ATTRIBUTE_ECC_ENABLED:
            return devices[dev].CU_DEVICE_ATTRIBUTE_ECC_ENABLED;
        case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY:
            return devices[dev].CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY;
        case CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED:
            return devices[dev].CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED;
        case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT:
            return devices[dev].CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT;
        case CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT:
            return devices[dev].CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT;
        case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH;
        case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT;
        case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH;
        case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT;
        case CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT:
            return devices[dev].CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT;
        case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID:
            return devices[dev].CU_DEVICE_ATTRIBUTE_PCI_BUS_ID;
        case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID:
            return devices[dev].CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID;
        case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID:
            return devices[dev].CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID;
        case CU_DEVICE_ATTRIBUTE_TCC_DRIVER:
            return devices[dev].CU_DEVICE_ATTRIBUTE_TCC_DRIVER;
        case CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED:
            return devices[dev].CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED:
            return devices[dev].CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED;
        case CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM:
            return devices[dev].CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM;
        case CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH:
            return devices[dev].CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH;
        case CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE:
            return devices[dev].CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE;
        case CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED:
            return devices[dev].CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED;
        case CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED:
            return devices[dev].CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED;
        case CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED:
            return devices[dev].CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED;
        case CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS:
            return devices[dev].CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS;
        case CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING:
            return devices[dev].CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING;
        case CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED:
            return devices[dev].CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED;
        case CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH:
            return devices[dev].CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH;
        default:
            break;
    }
    return -1; // 未知属性返回-1
}

cudaUUID_t DeviceManager::getDeviceUUID(int dev) const noexcept{
    return devices[dev].uuid;
}

void DeviceManager::setDevice(int device_id) const noexcept{
    // 简单设置当前线程的设备ID
    currentDevice = device_id;
}

void DeviceManager::getDevice(int* device_id) const noexcept{
    *device_id = currentDevice;
}

cudaError_t DeviceManager::getLastError() const noexcept{
    return lastError;
}

void DeviceManager::setLastError(cudaError_t error)const noexcept{
    lastError = error;
}

CUcontext DeviceManager::getCurrentContext()const noexcept{
    // 获取当前线程的设备 ID
    int dev_id = currentDevice; 
    
    // 边界检查
    if (dev_id < 0 || dev_id >= fake_contexts.size()) {
        return nullptr;
    }
    // 返回预创建的 Context 指针
    return fake_contexts[dev_id];
}

// 根据设备 ID 获取 Context (用于 cuDevicePrimaryCtxRetain)
CUcontext DeviceManager::getContextForDevice(int dev_id) const noexcept{
    if (dev_id < 0 || dev_id >= fake_contexts.size()) {
        return nullptr;
    }
    return fake_contexts[dev_id];
}


cudaError_t DeviceManager::allocateMemory(void** devPtr, size_t size) noexcept{
    std::lock_guard<std::mutex> lock(alloc_mutex);
    // 1. 检查是否越界 (模拟 OOM)
    if (devices[currentDevice].DeviceUsedMem + size > devices[currentDevice].DeviceTotalMem) {
        return cudaErrorMemoryAllocation;
    }
    // 2. 分配虚拟地址，但不分配物理页
    // MAP_PRIVATE | MAP_ANONYMOUS: 匿名映射
    // MAP_NORESERVE: 关键！不预留交换空间。
    // PROT_NONE: 甚至可以设为不可读写，防止意外访问（可选 PROT_READ | PROT_WRITE 以防万一）
    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (ptr == MAP_FAILED) {
        return cudaErrorMemoryAllocation;
    }
    // 3. 记录分配信息并更新使用量
    allocation_map[ptr] = {size, currentDevice};
    devices[currentDevice].DeviceUsedMem += size;
    *devPtr = ptr;
    return cudaSuccess;
}

cudaError_t DeviceManager::freeMemory(void* devPtr) noexcept{
    if (devPtr == nullptr) return cudaSuccess;
    std::lock_guard<std::mutex> lock(alloc_mutex);
    auto it = allocation_map.find(devPtr);
    if (it == allocation_map.end()) {
        return cudaErrorInvalidValue;
    }
    size_t size = it->second.size;
    int dev_id = it->second.device_id;
    // 4. 使用 munmap 释放虚拟内存
    munmap(devPtr, size);
    // 更新统计
    if (devices[dev_id].DeviceUsedMem >= size) {
        devices[dev_id].DeviceUsedMem -= size;
    } else {
        devices[dev_id].DeviceUsedMem = 0;
    }
    
    allocation_map.erase(it);
    
    return cudaSuccess;
}
