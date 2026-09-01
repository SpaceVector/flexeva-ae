#include "cpp_event/cluster_state.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

// Simple JSON parsing without external dependency
// For production, consider using nlohmann/json
namespace {

std::string trim(const std::string &s) {
  size_t start = s.find_first_not_of(" \t\n\r");
  if (start == std::string::npos)
    return "";
  size_t end = s.find_last_not_of(" \t\n\r");
  return s.substr(start, end - start + 1);
}

} // namespace

namespace cpp_event {

// Default GPU memory for A100 (80GB)
constexpr std::size_t kDefaultGpuMemoryBytes = 80ULL * 1024 * 1024 * 1024;

// Network constants for A100/NVLink topology
constexpr double kNvlinkBandwidthGBps = 600.0;       // GB/s per NVLink
constexpr double kPcieBandwidthGBps = 64.0;          // GB/s for PCIe 4.0 x16
constexpr double kInfinibandBandwidthGBps = 200.0;   // GB/s for HDR InfiniBand
constexpr std::int64_t kNvlinkLatencyNs = 1000;      // 1 microsecond
constexpr std::int64_t kPcieLatencyNs = 5000;        // 5 microseconds
constexpr std::int64_t kNetworkLatencyNs = 10000;    // 10 microseconds

ClusterState::ClusterState(const ClusterTopology &topology)
    : topology_(topology) {
  initialize_device_states();
}

ClusterState ClusterState::create_default(std::int32_t world_size) {
  return ClusterState(create_default_topology(world_size));
}

void ClusterState::initialize_device_states() {
  for (const auto &machine : topology_.machines) {
    for (const auto &gpu : machine.gpus) {
      DeviceState state;
      state.device_id = gpu.id;
      state.memory_capacity_bytes = kDefaultGpuMemoryBytes;
      state.memory_used_bytes = 0;
      state.is_available = true;
      device_states_[gpu.id] = state;
    }
  }
}

std::string ClusterState::device_id_for_rank(RankId rank) const {
  auto binding = topology_.binding_for_rank(rank);
  if (!binding) {
    return "";
  }
  const auto *machine = topology_.find_machine(binding->machine_id);
  if (!machine || binding->gpu_index >= machine->gpus.size()) {
    return "";
  }
  return machine->gpus[binding->gpu_index].id;
}

bool ClusterState::is_device_available(RankId rank) const {
  const auto *state = get_device_state(rank);
  return state != nullptr && state->is_available;
}

void ClusterState::allocate_memory(RankId rank, std::size_t bytes) {
  auto *state = get_device_state_mut(rank);
  if (state) {
    state->memory_used_bytes += bytes;
  }
}

void ClusterState::free_memory(RankId rank, std::size_t bytes) {
  auto *state = get_device_state_mut(rank);
  if (state && state->memory_used_bytes >= bytes) {
    state->memory_used_bytes -= bytes;
  }
}

std::size_t ClusterState::get_memory_usage(RankId rank) const {
  const auto *state = get_device_state(rank);
  return state ? state->memory_used_bytes : 0;
}

std::size_t ClusterState::get_memory_capacity(RankId rank) const {
  const auto *state = get_device_state(rank);
  return state ? state->memory_capacity_bytes : 0;
}

std::chrono::nanoseconds ClusterState::estimate_network_latency(RankId src,
                                                                 RankId dst) const {
  if (src == dst) {
    return std::chrono::nanoseconds(0);
  }

  if (are_ranks_on_same_machine(src, dst)) {
    // Intra-node: NVLink latency
    return std::chrono::nanoseconds(kNvlinkLatencyNs);
  }

  // Inter-node: Network latency
  return std::chrono::nanoseconds(kNetworkLatencyNs);
}

bool ClusterState::are_ranks_on_same_machine(RankId a, RankId b) const {
  auto binding_a = topology_.binding_for_rank(a);
  auto binding_b = topology_.binding_for_rank(b);

  if (!binding_a || !binding_b) {
    return false;
  }

  return binding_a->machine_id == binding_b->machine_id;
}

double ClusterState::get_network_bandwidth(RankId src, RankId dst) const {
  if (src == dst) {
    return std::numeric_limits<double>::infinity();
  }

  if (are_ranks_on_same_machine(src, dst)) {
    return kNvlinkBandwidthGBps;
  }

  return kInfinibandBandwidthGBps;
}

const DeviceState *ClusterState::get_device_state(RankId rank) const {
  std::string device_id = device_id_for_rank(rank);
  auto it = device_states_.find(device_id);
  return it != device_states_.end() ? &it->second : nullptr;
}

DeviceState *ClusterState::get_device_state_mut(RankId rank) {
  std::string device_id = device_id_for_rank(rank);
  auto it = device_states_.find(device_id);
  return it != device_states_.end() ? &it->second : nullptr;
}

ClusterTopology create_default_topology(std::int32_t world_size) {
  ClusterTopology topology;

  // Create a single machine with world_size GPUs
  Machine machine;
  machine.id = "node0";
  machine.hostname = "localhost";

  // Single CPU and NIC
  Cpu cpu;
  cpu.id = "cpu:0";
  cpu.numa_node = "numa:0";
  machine.cpus.push_back(cpu);

  Nic nic;
  nic.id = "mlx5:0";
  nic.model = "ConnectX-6";
  for (std::int32_t i = 0; i < world_size; ++i) {
    nic.gpu_group.push_back(static_cast<std::size_t>(i));
  }
  machine.nics.push_back(nic);

  // Create GPUs
  for (std::int32_t i = 0; i < world_size; ++i) {
    Gpu gpu;
    gpu.id = "gpu:" + std::to_string(i);
    gpu.model = "A100";
    gpu.cpu_index = 0;
    gpu.nic_index = 0;
    gpu.nic_group = 0;
    machine.gpus.push_back(gpu);
  }

  topology.machines.push_back(machine);

  // Create rank bindings
  for (std::int32_t i = 0; i < world_size; ++i) {
    RankBinding binding;
    binding.rank = i;
    binding.machine_id = "node0";
    binding.gpu_index = static_cast<std::size_t>(i);
    topology.ranks.push_back(binding);
  }

  return topology;
}

ClusterTopology load_topology_from_json(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open topology file: " + path);
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  return parse_topology_from_json(buffer.str());
}

// Simple JSON parser for topology (handles the specific format we need)
ClusterTopology parse_topology_from_json(const std::string &json_str) {
  // For now, return a default topology
  // A full JSON parser would be implemented here
  // This is a placeholder that creates a 4-GPU single-node topology

  // In production, use nlohmann/json or similar library
  // For Phase 3, we primarily use create_default_topology()

  return create_default_topology(4);
}

} // namespace cpp_event
