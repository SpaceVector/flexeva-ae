#pragma once

#include "cpp_event/dependency_resolver.hpp"
#include "cpp_event/event_graph_builder.hpp"
#include "cpp_event/event_log.hpp"
#include "cpp_event/producing_dict.hpp"
#include "cpp_event/rank_state.hpp"
#include "cpp_event/requirement_dict.hpp"

#include <vector>

namespace cpp_event {

/// Controls SPMD execution semantics with dependency tracking.
/// Manages rank execution, blocking, and dependency resolution.
class SPMDExecutionController final {
public:
  explicit SPMDExecutionController(EventContext &context)
      : context_(context), event_log_(context),
        resolver_(req_dict_, prod_dict_, state_manager_) {}

  /// Process an event for a specific rank.
  /// Returns true if the rank should continue executing, false if blocked.
  bool process_event(RankId rank_id, const EventRecord &event);

  /// Check if a rank is ready to continue (all dependencies satisfied).
  [[nodiscard]] bool is_rank_ready(RankId rank_id) const;

  /// Get the next rank that should execute (round-robin among ready ranks).
  [[nodiscard]] RankId get_next_ready_rank() const;

  /// Build the final event graph with all dependencies resolved.
  [[nodiscard]] EventGraph build_final_graph();

  /// Record an event and handle dependencies.
  void record_event(RankId rank_id, const EventRecord &event);

private:
  EventContext &context_;
  EventLog event_log_;
  RankStateManager state_manager_;
  RequirementDictionary req_dict_;
  ProducingDictionary prod_dict_;
  DependencyResolver resolver_;
  EventGraphBuilder graph_builder_;

  /// Register requirements for an event (e.g., recv needs send).
  void register_requirements(RankId rank_id, const EventRecord &event);

  /// Register productions for an event (e.g., send is produced).
  void register_productions(RankId rank_id, const EventRecord &event);

  /// Check if an event type creates a requirement.
  [[nodiscard]] bool creates_requirement(EventKind kind) const;

  /// Check if an event type creates a production.
  [[nodiscard]] bool creates_production(EventKind kind) const;
};

} // namespace cpp_event
