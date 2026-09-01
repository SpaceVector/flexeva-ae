#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cpp_event {

enum class ResourceScope {
  kCluster,
  kMachine,
  kNumaDomain,
  kDevice,
};

enum class DeviceType {
  kCpu,
  kGpu,
  kNic,
};

struct Cpu {
  std::string id{};
  std::string numa_node{};
};

struct Nic {
  std::string id{};
  std::string model{};
  std::vector<std::size_t> gpu_group{};
};

struct Gpu {
  std::string id{};
  std::string model{};
  std::size_t cpu_index{};
  std::size_t nic_index{};
  std::size_t nic_group{};
};

struct Machine {
  std::string id{};
  std::string hostname{};
  std::vector<Cpu> cpus{};
  std::vector<Nic> nics{};
  std::vector<Gpu> gpus{};
};

struct RankBinding {
  std::int32_t rank{};
  std::string machine_id{};
  std::size_t gpu_index{};
};

struct ClusterTopology {
  std::vector<Machine> machines{};
  std::vector<RankBinding> ranks{};

  [[nodiscard]] const Machine *
  find_machine(const std::string &machine_id) const noexcept;

  [[nodiscard]] std::optional<RankBinding>
  binding_for_rank(std::int32_t rank) const noexcept;
};

enum class AccessMode {
  kExclusive,
  kShared,
};

struct DeviceReference {
  std::string machine_id{};
  DeviceType type{DeviceType::kGpu};
  std::string id{};
};

struct DeviceAccess {
  DeviceReference device{};
  AccessMode mode{AccessMode::kExclusive};
};

std::vector<DeviceAccess> devices_for_rank(const ClusterTopology &topology,
                                           std::int32_t rank);

std::vector<std::int32_t> ranks_for_device(const ClusterTopology &topology,
                                           const DeviceReference &device);

inline const Machine *
ClusterTopology::find_machine(const std::string &machine_id) const noexcept {
  for (const auto &machine : machines) {
    if (machine.id == machine_id) {
      return &machine;
    }
  }
  return nullptr;
}

inline std::optional<RankBinding>
ClusterTopology::binding_for_rank(std::int32_t rank) const noexcept {
  for (const auto &binding : ranks) {
    if (binding.rank == rank) {
      return binding;
    }
  }
  return std::nullopt;
}

inline std::vector<DeviceAccess>
devices_for_rank(const ClusterTopology &topology, std::int32_t rank) {
  std::vector<DeviceAccess> result;
  const auto binding = topology.binding_for_rank(rank);
  if (!binding) {
    return result;
  }

  const auto *machine = topology.find_machine(binding->machine_id);
  if (!machine) {
    return result;
  }
  if (binding->gpu_index >= machine->gpus.size()) {
    return result;
  }

  const auto &gpu = machine->gpus[binding->gpu_index];
  result.push_back(DeviceAccess{
      .device = DeviceReference{machine->id, DeviceType::kGpu, gpu.id},
      .mode = AccessMode::kExclusive,
  });

  if (gpu.cpu_index < machine->cpus.size()) {
    const auto &cpu = machine->cpus[gpu.cpu_index];
    result.push_back(DeviceAccess{
        .device = DeviceReference{machine->id, DeviceType::kCpu, cpu.id},
        .mode = AccessMode::kShared,
    });
  }

  if (gpu.nic_index < machine->nics.size()) {
    const auto &nic = machine->nics[gpu.nic_index];
    result.push_back(DeviceAccess{
        .device = DeviceReference{machine->id, DeviceType::kNic, nic.id},
        .mode = AccessMode::kShared,
    });
  }

  return result;
}

inline std::vector<std::int32_t>
ranks_for_device(const ClusterTopology &topology,
                 const DeviceReference &device) {
  std::vector<std::int32_t> ranks;
  const auto *machine = topology.find_machine(device.machine_id);
  if (!machine) {
    return ranks;
  }

  auto gather_from_gpu_index = [&](std::size_t gpu_index) {
    for (const auto &binding : topology.ranks) {
      if (binding.machine_id == device.machine_id &&
          binding.gpu_index == gpu_index) {
        ranks.push_back(binding.rank);
      }
    }
  };

  switch (device.type) {
  case DeviceType::kGpu: {
    for (std::size_t idx = 0; idx < machine->gpus.size(); ++idx) {
      if (machine->gpus[idx].id == device.id) {
        gather_from_gpu_index(idx);
        break;
      }
    }
    break;
  }
  case DeviceType::kCpu: {
    for (std::size_t cpu_index = 0; cpu_index < machine->cpus.size();
         ++cpu_index) {
      if (machine->cpus[cpu_index].id != device.id) {
        continue;
      }
      for (const auto &binding : topology.ranks) {
        if (binding.machine_id != device.machine_id) {
          continue;
        }
        if (binding.gpu_index >= machine->gpus.size()) {
          continue;
        }
        if (machine->gpus[binding.gpu_index].cpu_index == cpu_index) {
          ranks.push_back(binding.rank);
        }
      }
      break;
    }
    break;
  }
  case DeviceType::kNic: {
    for (std::size_t nic_index = 0; nic_index < machine->nics.size();
         ++nic_index) {
      if (machine->nics[nic_index].id != device.id) {
        continue;
      }
      for (const auto &binding : topology.ranks) {
        if (binding.machine_id != device.machine_id) {
          continue;
        }
        if (binding.gpu_index >= machine->gpus.size()) {
          continue;
        }
        if (machine->gpus[binding.gpu_index].nic_index == nic_index) {
          ranks.push_back(binding.rank);
        }
      }
      break;
    }
    break;
  }
  }

  return ranks;
}

} // namespace cpp_event
