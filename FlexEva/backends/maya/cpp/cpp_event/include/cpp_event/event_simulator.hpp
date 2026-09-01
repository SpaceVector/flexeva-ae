#pragma once

#include "cpp_event/cluster_state.hpp"
#include "cpp_event/event_context.hpp"
#include "cpp_event/event_schema.hpp"
#include "cpp_event/spmd_execution_controller.hpp"
#include "cpp_event/timing_model.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cpp_event {

/// Per-rank metrics from simulation.
struct RankMetrics {
  RankId rank{0};
  std::chrono::nanoseconds compute_time{0};
  std::chrono::nanoseconds communication_time{0};
  std::chrono::nanoseconds blocked_time{0};
  std::chrono::nanoseconds total_time{0};
  std::size_t num_events{0};
  double utilization{0.0}; // compute_time / total_time
};

/// Result of running a simulation.
struct SimulationResult {
  std::chrono::nanoseconds total_time{0};
  std::chrono::nanoseconds critical_path{0};
  std::vector<RankMetrics> rank_metrics{};
  EventGraph final_graph{};
  bool success{false};
  std::string error_message{};
};

/// Simulated event with timing information.
struct SimulatedEvent {
  EventRecord record{};
  RankId rank{0};
  std::chrono::nanoseconds start_time{0};
  std::chrono::nanoseconds duration{0};
  std::chrono::nanoseconds end_time{0};
};

/// Discrete-event simulator for SPSD traces.
/// Orchestrates simulation loop with proper timing estimation.
class EventSimulator {
public:
  /// Create simulator with cluster state and context.
  EventSimulator(ClusterTopology topology, EventContext &ctx);

  /// Create simulator with default topology for given world size.
  static EventSimulator create_default(std::int32_t world_size,
                                       EventContext &ctx);

  /// Set the timing model (takes ownership).
  void set_timing_model(std::unique_ptr<TimingModel> model);

  /// Load events from a trace file (JSON format).
  bool load_events(const std::string &trace_path);

  /// Load events from an EventGraph directly.
  void load_events(const EventGraph &graph);

  /// Add a single event to simulate.
  void add_event(RankId rank, const EventRecord &event);

  /// Run the simulation.
  [[nodiscard]] SimulationResult run();

  /// Get the current simulation clock for a rank.
  [[nodiscard]] std::chrono::nanoseconds
  get_rank_clock(RankId rank) const;

  /// Get the maximum simulation clock across all ranks.
  [[nodiscard]] std::chrono::nanoseconds get_total_time() const;

  /// Get the critical path length (max rank completion time).
  [[nodiscard]] std::chrono::nanoseconds get_critical_path() const;

  /// Get metrics for all ranks.
  [[nodiscard]] std::vector<RankMetrics> get_rank_metrics() const;

  /// Get the cluster state.
  [[nodiscard]] const ClusterState &cluster_state() const {
    return cluster_state_;
  }

  /// Get simulated events (available after run()).
  [[nodiscard]] const std::vector<SimulatedEvent> &simulated_events() const {
    return simulated_events_;
  }

private:
  ClusterState cluster_state_;
  EventContext &context_;
  SPMDExecutionController controller_;
  std::unique_ptr<TimingModel> timing_model_;

  // Per-rank simulation state
  std::unordered_map<RankId, std::chrono::nanoseconds> rank_clocks_;
  std::unordered_map<RankId, RankMetrics> rank_metrics_;

  // Events to simulate, organized by rank
  std::unordered_map<RankId, std::vector<EventRecord>> pending_events_;

  // Results
  std::vector<SimulatedEvent> simulated_events_;

  /// Initialize rank state for simulation.
  void initialize_ranks();

  /// Process a single event for a rank.
  void process_event(RankId rank, const EventRecord &event);

  /// Update metrics after processing an event.
  void update_metrics(RankId rank, const EventRecord &event,
                      std::chrono::nanoseconds duration);

  /// Handle blocking dependencies (e.g., recv waiting for send).
  void handle_dependencies(RankId rank, const EventRecord &event);

  /// Check if an event is a communication event.
  [[nodiscard]] bool is_communication_event(EventKind kind) const;

  /// Parse events from JSON trace file.
  [[nodiscard]] bool parse_trace_file(const std::string &path);
};

} // namespace cpp_event
