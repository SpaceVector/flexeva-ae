#include "cpp_event/timing_model.hpp"

#include <algorithm>
#include <cmath>

namespace cpp_event {

namespace {

// Default operation sizes when not specified in event payload
constexpr std::uint64_t kDefaultComputeFlops = 1'000'000'000;   // 1 GFLOP
constexpr std::uint64_t kDefaultMemoryBytes = 100'000'000;      // 100 MB
constexpr std::uint64_t kDefaultCollectiveBytes = 100'000'000;  // 100 MB

// Minimum durations to avoid zero times
constexpr std::int64_t kMinComputeNs = 100'000;       // 100 us minimum compute
constexpr std::int64_t kMinMemoryNs = 10'000;         // 10 us minimum memory
constexpr std::int64_t kMinCollectiveNs = 50'000;     // 50 us minimum collective
constexpr std::int64_t kMinLatencyNs = 1'000;         // 1 us minimum anything

} // namespace

std::chrono::nanoseconds
SimpleTimingModel::estimate(const EventRecord &event,
                            const ClusterState &state) {
  switch (event.kind) {
  case EventKind::kComputeKernel:
    return estimate_compute(event);

  case EventKind::kMemcpyHostToDevice:
  case EventKind::kMemcpyDeviceToHost:
  case EventKind::kMemcpyDeviceToDevice:
    return estimate_memory_op(event);

  case EventKind::kAllReduce:
  case EventKind::kBroadcast:
  case EventKind::kAllGather:
  case EventKind::kReduceScatter:
  case EventKind::kCollective:
    return estimate_collective(event, state);

  case EventKind::kSend:
  case EventKind::kRecv:
  case EventKind::kPointToPoint:
    return estimate_point_to_point(event, state);

  case EventKind::kBarrier:
    // Barrier time depends on network latency and slowest rank
    return std::chrono::nanoseconds(
        static_cast<std::int64_t>(params_.network_latency_us * 1000));

  case EventKind::kMemoryAllocation:
  case EventKind::kMemoryFree:
    // Memory allocation is typically fast on GPU
    return std::chrono::nanoseconds(kMinLatencyNs);

  case EventKind::kRuntimeCall:
    // Runtime overhead
    return std::chrono::nanoseconds(kMinLatencyNs);

  case EventKind::kFileRead:
  case EventKind::kFileWrite:
    // I/O is slow - use bytes if available
    {
      std::uint64_t bytes = get_bytes(event);
      if (bytes == 0)
        bytes = kDefaultMemoryBytes;
      // Assume ~1 GB/s I/O bandwidth
      double time_ns = (static_cast<double>(bytes) / 1e9) * 1e9;
      return std::chrono::nanoseconds(
          std::max(static_cast<std::int64_t>(time_ns), 1'000'000L));
    }

  default:
    return std::chrono::nanoseconds(kMinLatencyNs);
  }
}

std::chrono::nanoseconds
SimpleTimingModel::estimate_compute(const EventRecord &event) const {
  std::uint64_t flops = get_flops(event);
  if (flops == 0) {
    flops = kDefaultComputeFlops;
  }

  // Time = FLOPs / (peak TFLOPs * 1e12)
  // Result in seconds, convert to nanoseconds
  double peak_flops_per_sec = params_.peak_fp16_tflops * 1e12;
  double time_sec = static_cast<double>(flops) / peak_flops_per_sec;
  auto time_ns = static_cast<std::int64_t>(time_sec * 1e9);

  return std::chrono::nanoseconds(std::max(time_ns, kMinComputeNs));
}

std::chrono::nanoseconds
SimpleTimingModel::estimate_memory_op(const EventRecord &event) const {
  std::uint64_t bytes = get_bytes(event);
  if (bytes == 0) {
    bytes = kDefaultMemoryBytes;
  }

  // Time = bytes / bandwidth
  // bandwidth is in GB/s, convert to bytes/s
  double bandwidth_bytes_per_sec = params_.memory_bandwidth_gbps * 1e9;
  double time_sec = static_cast<double>(bytes) / bandwidth_bytes_per_sec;
  auto time_ns = static_cast<std::int64_t>(time_sec * 1e9);

  return std::chrono::nanoseconds(std::max(time_ns, kMinMemoryNs));
}

std::chrono::nanoseconds
SimpleTimingModel::estimate_collective(const EventRecord &event,
                                       const ClusterState &state) const {
  std::uint64_t bytes = get_bytes(event);
  if (bytes == 0) {
    bytes = kDefaultCollectiveBytes;
  }

  std::int32_t num_ranks = get_num_ranks(event);
  if (num_ranks <= 0) {
    num_ranks = state.world_size();
  }
  if (num_ranks <= 1) {
    return std::chrono::nanoseconds(kMinLatencyNs);
  }

  // Ring AllReduce model: 2 * (n-1) / n * bytes / bandwidth
  // where n = number of ranks
  double efficiency = params_.ring_allreduce_efficiency;
  double effective_bytes =
      2.0 * (num_ranks - 1) / num_ranks * static_cast<double>(bytes);

  // Determine bandwidth based on topology
  // Simplified: assume all ranks on same machine use NVLink
  double bandwidth_gbps = params_.nvlink_bandwidth_gbps;
  double latency_us = params_.nvlink_latency_us;

  // If ranks span machines, use network bandwidth
  // For now, use NVLink since default is single-node

  double bandwidth_bytes_per_sec = bandwidth_gbps * 1e9 * efficiency;
  double transfer_time_sec = effective_bytes / bandwidth_bytes_per_sec;

  // Add latency for each step in the ring (2 * (n-1) steps)
  double latency_sec = 2.0 * (num_ranks - 1) * latency_us * 1e-6;

  double total_time_sec = transfer_time_sec + latency_sec;
  auto time_ns = static_cast<std::int64_t>(total_time_sec * 1e9);

  return std::chrono::nanoseconds(std::max(time_ns, kMinCollectiveNs));
}

std::chrono::nanoseconds
SimpleTimingModel::estimate_point_to_point(const EventRecord &event,
                                           const ClusterState &state) const {
  std::uint64_t bytes = get_bytes(event);
  if (bytes == 0) {
    bytes = kDefaultMemoryBytes;
  }

  // Get source and destination ranks from event payload
  // For now, assume same machine (NVLink)
  double bandwidth_gbps = params_.nvlink_bandwidth_gbps;
  double latency_us = params_.nvlink_latency_us;

  double bandwidth_bytes_per_sec = bandwidth_gbps * 1e9;
  double transfer_time_sec =
      static_cast<double>(bytes) / bandwidth_bytes_per_sec;
  double latency_sec = latency_us * 1e-6;

  double total_time_sec = transfer_time_sec + latency_sec;
  auto time_ns = static_cast<std::int64_t>(total_time_sec * 1e9);

  return std::chrono::nanoseconds(std::max(time_ns, kMinLatencyNs));
}

std::uint64_t SimpleTimingModel::get_flops(const EventRecord &event) const {
  auto it = event.payload.attributes.find("flops");
  if (it != event.payload.attributes.end()) {
    try {
      return std::stoull(it->second);
    } catch (...) {
      return 0;
    }
  }
  return 0;
}

std::uint64_t SimpleTimingModel::get_bytes(const EventRecord &event) const {
  auto it = event.payload.attributes.find("bytes");
  if (it != event.payload.attributes.end()) {
    try {
      return std::stoull(it->second);
    } catch (...) {
      return 0;
    }
  }
  // Also check "size" attribute
  it = event.payload.attributes.find("size");
  if (it != event.payload.attributes.end()) {
    try {
      return std::stoull(it->second);
    } catch (...) {
      return 0;
    }
  }
  return 0;
}

std::int32_t SimpleTimingModel::get_num_ranks(const EventRecord &event) const {
  // Check event's active_group first
  if (!event.active_group.empty()) {
    return static_cast<std::int32_t>(event.active_group.size());
  }

  // Check payload attributes
  auto it = event.payload.attributes.find("world_size");
  if (it != event.payload.attributes.end()) {
    try {
      return std::stoi(it->second);
    } catch (...) {
      return 0;
    }
  }
  it = event.payload.attributes.find("num_ranks");
  if (it != event.payload.attributes.end()) {
    try {
      return std::stoi(it->second);
    } catch (...) {
      return 0;
    }
  }
  return 0;
}

std::unique_ptr<TimingModel> create_default_timing_model() {
  return std::make_unique<SimpleTimingModel>();
}

} // namespace cpp_event
