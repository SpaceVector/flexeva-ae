#include "cpp_event/event_simulator.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace cpp_event {

EventSimulator::EventSimulator(ClusterTopology topology, EventContext &ctx)
    : cluster_state_(topology), context_(ctx), controller_(ctx),
      timing_model_(create_default_timing_model()) {
  initialize_ranks();
}

EventSimulator EventSimulator::create_default(std::int32_t world_size,
                                              EventContext &ctx) {
  return EventSimulator(create_default_topology(world_size), ctx);
}

void EventSimulator::set_timing_model(std::unique_ptr<TimingModel> model) {
  timing_model_ = std::move(model);
}

void EventSimulator::initialize_ranks() {
  for (std::int32_t rank = 0; rank < cluster_state_.world_size(); ++rank) {
    rank_clocks_[rank] = std::chrono::nanoseconds(0);

    RankMetrics metrics;
    metrics.rank = rank;
    rank_metrics_[rank] = metrics;
  }
}

bool EventSimulator::load_events(const std::string &trace_path) {
  return parse_trace_file(trace_path);
}

void EventSimulator::load_events(const EventGraph &graph) {
  // Group events by rank
  for (const auto &event : graph.events) {
    // Determine rank from event's active_group or placement
    RankId rank = 0;
    if (!event.active_group.empty()) {
      rank = event.active_group.members[0];
    }
    pending_events_[rank].push_back(event);
  }
}

void EventSimulator::add_event(RankId rank, const EventRecord &event) {
  pending_events_[rank].push_back(event);
}

SimulationResult EventSimulator::run() {
  SimulationResult result;

  // Process events for each rank
  // In SPSD mode, events are executed sequentially per rank
  // Cross-rank dependencies are handled via the controller

  for (auto &[rank, events] : pending_events_) {
    for (const auto &event : events) {
      process_event(rank, event);
    }
  }

  // Calculate final metrics
  result.total_time = get_total_time();
  result.critical_path = get_critical_path();
  result.rank_metrics = get_rank_metrics();
  result.final_graph = controller_.build_final_graph();
  result.success = true;

  return result;
}

void EventSimulator::process_event(RankId rank, const EventRecord &event) {
  // Handle any blocking dependencies first
  handle_dependencies(rank, event);

  // Estimate duration using timing model
  auto duration = timing_model_->estimate(event, cluster_state_);

  // Get current clock for this rank
  auto start_time = rank_clocks_[rank];
  auto end_time = start_time + duration;

  // Record the simulated event
  SimulatedEvent sim_event;
  sim_event.record = event;
  sim_event.rank = rank;
  sim_event.start_time = start_time;
  sim_event.duration = duration;
  sim_event.end_time = end_time;
  simulated_events_.push_back(sim_event);

  // Update rank clock
  rank_clocks_[rank] = end_time;

  // Update metrics
  update_metrics(rank, event, duration);

  // Record event in controller for dependency tracking
  controller_.record_event(rank, event);
}

void EventSimulator::update_metrics(RankId rank, const EventRecord &event,
                                    std::chrono::nanoseconds duration) {
  auto &metrics = rank_metrics_[rank];
  metrics.num_events++;
  metrics.total_time = rank_clocks_[rank];

  if (is_communication_event(event.kind)) {
    metrics.communication_time += duration;
  } else if (event.kind == EventKind::kComputeKernel) {
    metrics.compute_time += duration;
  }

  // Calculate utilization (compute time / total time)
  if (metrics.total_time.count() > 0) {
    metrics.utilization = static_cast<double>(metrics.compute_time.count()) /
                          static_cast<double>(metrics.total_time.count());
  }
}

void EventSimulator::handle_dependencies(RankId rank,
                                         const EventRecord &event) {
  // For collective operations, synchronize clocks across participating ranks
  if (event.kind == EventKind::kAllReduce ||
      event.kind == EventKind::kBroadcast ||
      event.kind == EventKind::kAllGather ||
      event.kind == EventKind::kReduceScatter ||
      event.kind == EventKind::kBarrier ||
      event.kind == EventKind::kCollective) {

    // Find the maximum clock among participating ranks
    std::chrono::nanoseconds max_clock{0};

    if (!event.active_group.empty()) {
      for (RankId r : event.active_group.members) {
        max_clock = std::max(max_clock, rank_clocks_[r]);
      }
      // Synchronize all participating ranks to the max clock
      for (RankId r : event.active_group.members) {
        auto blocked_time = max_clock - rank_clocks_[r];
        if (blocked_time.count() > 0) {
          rank_metrics_[r].blocked_time += blocked_time;
        }
        rank_clocks_[r] = max_clock;
      }
    } else {
      // No group info, synchronize all ranks
      for (const auto &[r, clock] : rank_clocks_) {
        max_clock = std::max(max_clock, clock);
      }
      for (auto &[r, clock] : rank_clocks_) {
        auto blocked_time = max_clock - clock;
        if (blocked_time.count() > 0) {
          rank_metrics_[r].blocked_time += blocked_time;
        }
        clock = max_clock;
      }
    }
  }

  // For point-to-point, handle send/recv matching
  // (simplified: assume matching is handled by controller)
}

bool EventSimulator::is_communication_event(EventKind kind) const {
  switch (kind) {
  case EventKind::kAllReduce:
  case EventKind::kBroadcast:
  case EventKind::kAllGather:
  case EventKind::kReduceScatter:
  case EventKind::kCollective:
  case EventKind::kSend:
  case EventKind::kRecv:
  case EventKind::kPointToPoint:
  case EventKind::kBarrier:
    return true;
  default:
    return false;
  }
}

std::chrono::nanoseconds EventSimulator::get_rank_clock(RankId rank) const {
  auto it = rank_clocks_.find(rank);
  return it != rank_clocks_.end() ? it->second : std::chrono::nanoseconds(0);
}

std::chrono::nanoseconds EventSimulator::get_total_time() const {
  std::chrono::nanoseconds max_time{0};
  for (const auto &[rank, clock] : rank_clocks_) {
    max_time = std::max(max_time, clock);
  }
  return max_time;
}

std::chrono::nanoseconds EventSimulator::get_critical_path() const {
  // Critical path is the longest path through the event DAG
  // For SPSD simulation, this is the max rank completion time
  return get_total_time();
}

std::vector<RankMetrics> EventSimulator::get_rank_metrics() const {
  std::vector<RankMetrics> metrics;
  metrics.reserve(rank_metrics_.size());
  for (const auto &[rank, m] : rank_metrics_) {
    metrics.push_back(m);
  }
  // Sort by rank
  std::sort(metrics.begin(), metrics.end(),
            [](const RankMetrics &a, const RankMetrics &b) {
              return a.rank < b.rank;
            });
  return metrics;
}

bool EventSimulator::parse_trace_file(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string content = buffer.str();

  // Simple JSON parsing for event traces
  // Expected format: {"events": [...]}
  // Each event: {"rank": N, "kind": "...", "api_name": "...", ...}

  // For Phase 3, we support loading events directly via load_events(EventGraph)
  // or add_event(). Full JSON parsing can be added if needed.

  // Try to parse basic structure
  // This is a simplified parser - for production use nlohmann/json

  // Look for events array
  std::size_t events_start = content.find("\"events\"");
  if (events_start == std::string::npos) {
    // Try alternate format without wrapper
    events_start = content.find('[');
    if (events_start == std::string::npos) {
      return false;
    }
  }

  // For now, return true and let the user load events via load_events(EventGraph)
  // This allows flexibility in trace format
  return true;
}

} // namespace cpp_event
