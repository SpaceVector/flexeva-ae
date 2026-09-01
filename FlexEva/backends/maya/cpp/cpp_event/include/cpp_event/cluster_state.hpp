#pragma once

#include "cpp_event/device_topology.hpp"
#include "cpp_event/event_schema.hpp"

#include <chrono>
#include <string>
#include <unordered_map>

namespace cpp_event {

/// Runtime state for a single device (GPU).
struct DeviceState {
  std::string device_id{};
  std::size_t memory_capacity_bytes{0};     // Total GPU memory
  std::size_t memory_used_bytes{0};         // Currently allocated
  bool is_available{true};                  // Available for scheduling
};

/// Runtime state for the cluster, built from ClusterTopology.
/// Tracks device availability, memory usage, and provides network topology helpers.
class ClusterState {
public:
  /// Initialize from topology configuration.
  explicit ClusterState(const ClusterTopology &topology);

  /// Initialize with default topology for given world size.
  static ClusterState create_default(std::int32_t world_size);

  /// Get the underlying topology.
  [[nodiscard]] const ClusterTopology &topology() const noexcept {
    return topology_;
  }

  /// Get number of ranks.
  [[nodiscard]] std::int32_t world_size() const noexcept {
    return static_cast<std::int32_t>(topology_.ranks.size());
  }

  // Device availability
  [[nodiscard]] bool is_device_available(RankId rank) const;

  // Memory tracking
  void allocate_memory(RankId rank, std::size_t bytes);
  void free_memory(RankId rank, std::size_t bytes);
  [[nodiscard]] std::size_t get_memory_usage(RankId rank) const;
  [[nodiscard]] std::size_t get_memory_capacity(RankId rank) const;

  // Network topology helpers
  [[nodiscard]] std::chrono::nanoseconds
  estimate_network_latency(RankId src, RankId dst) const;

  [[nodiscard]] bool are_ranks_on_same_machine(RankId a, RankId b) const;

  [[nodiscard]] double get_network_bandwidth(RankId src, RankId dst) const;

  // Device state access
  [[nodiscard]] const DeviceState *get_device_state(RankId rank) const;
  DeviceState *get_device_state_mut(RankId rank);

private:
  ClusterTopology topology_;
  std::unordered_map<std::string, DeviceState> device_states_;

  /// Initialize device states from topology.
  void initialize_device_states();

  /// Get device ID for a rank.
  [[nodiscard]] std::string device_id_for_rank(RankId rank) const;
};

/// Load ClusterTopology from JSON file.
[[nodiscard]] ClusterTopology load_topology_from_json(const std::string &path);

/// Load ClusterTopology from JSON string.
[[nodiscard]] ClusterTopology
parse_topology_from_json(const std::string &json_str);

/// Create a default single-node topology with the specified number of GPUs.
[[nodiscard]] ClusterTopology create_default_topology(std::int32_t world_size);

} // namespace cpp_event
