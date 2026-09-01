#pragma once

#include "cpp_event/cluster_state.hpp"
#include "cpp_event/event_schema.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

namespace cpp_event {

/// Abstract interface for timing estimation.
/// Different models can implement hardware-specific or analytical timing.
class TimingModel {
public:
  virtual ~TimingModel() = default;

  /// Estimate execution time for an event given cluster state.
  [[nodiscard]] virtual std::chrono::nanoseconds
  estimate(const EventRecord &event, const ClusterState &state) = 0;

  /// Get model name for debugging/logging.
  [[nodiscard]] virtual std::string name() const = 0;
};

/// Hardware parameters for timing calculations.
struct HardwareParams {
  // Compute (A100 defaults)
  double peak_fp16_tflops{312.0};   // FP16 Tensor Core peak
  double peak_fp32_tflops{19.5};    // FP32 peak
  double peak_int8_tops{624.0};     // INT8 Tensor Core peak

  // Memory bandwidth
  double memory_bandwidth_gbps{2000.0}; // 2 TB/s HBM bandwidth

  // Network (intra-node)
  double nvlink_bandwidth_gbps{600.0};  // NVLink bandwidth
  double nvlink_latency_us{1.0};        // NVLink base latency

  // Network (inter-node)
  double network_bandwidth_gbps{200.0}; // InfiniBand HDR
  double network_latency_us{5.0};       // Network base latency

  // Collective efficiency factors
  double ring_allreduce_efficiency{0.9};
  double tree_broadcast_efficiency{0.85};
};

/// Simple analytical timing model based on hardware parameters.
/// Estimates time using:
/// - Compute: FLOPs / peak_flops
/// - Memory: bytes / bandwidth
/// - Collective: latency + bytes / network_bw
class SimpleTimingModel : public TimingModel {
public:
  explicit SimpleTimingModel(HardwareParams params = HardwareParams{})
      : params_(params) {}

  [[nodiscard]] std::chrono::nanoseconds
  estimate(const EventRecord &event, const ClusterState &state) override;

  [[nodiscard]] std::string name() const override { return "SimpleTimingModel"; }

  /// Get mutable access to hardware parameters for tuning.
  HardwareParams &params() { return params_; }
  [[nodiscard]] const HardwareParams &params() const { return params_; }

private:
  HardwareParams params_;

  // Estimation helpers
  [[nodiscard]] std::chrono::nanoseconds
  estimate_compute(const EventRecord &event) const;

  [[nodiscard]] std::chrono::nanoseconds
  estimate_memory_op(const EventRecord &event) const;

  [[nodiscard]] std::chrono::nanoseconds
  estimate_collective(const EventRecord &event, const ClusterState &state) const;

  [[nodiscard]] std::chrono::nanoseconds
  estimate_point_to_point(const EventRecord &event,
                          const ClusterState &state) const;

  // Extract numeric attributes from event payload
  [[nodiscard]] std::uint64_t get_flops(const EventRecord &event) const;
  [[nodiscard]] std::uint64_t get_bytes(const EventRecord &event) const;
  [[nodiscard]] std::int32_t get_num_ranks(const EventRecord &event) const;
};

/// Create a timing model with default A100 parameters.
[[nodiscard]] std::unique_ptr<TimingModel> create_default_timing_model();

} // namespace cpp_event
