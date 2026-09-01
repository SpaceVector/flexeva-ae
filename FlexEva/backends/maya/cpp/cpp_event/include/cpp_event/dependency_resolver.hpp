#pragma once

#include "cpp_event/event_schema.hpp"
#include "cpp_event/producing_dict.hpp"
#include "cpp_event/requirement_dict.hpp"
#include "cpp_event/rank_state.hpp"

#include <unordered_map>
#include <vector>

namespace cpp_event {

/// Resolves dependencies between events across ranks in SPMD mode.
/// Matches requirements with productions and generates dependency edges.
class DependencyResolver final {
public:
  DependencyResolver(RequirementDictionary &req_dict,
                     ProducingDictionary &prod_dict,
                     RankStateManager &state_manager)
      : req_dict_(req_dict), prod_dict_(prod_dict),
        state_manager_(state_manager) {}

  /// Resolve dependencies for a specific consumer rank.
  /// Returns a list of dependency edges that should be added to the graph.
  [[nodiscard]] std::vector<EventEdge>
  resolve_dependencies(RankId consumer_rank);

  /// Resolve all dependencies across all ranks.
  [[nodiscard]] std::vector<EventEdge> resolve_all_dependencies();

  /// Check if a requirement can be satisfied given current productions.
  [[nodiscard]] bool can_satisfy_requirement(RankId consumer_rank,
                                              EventId required_event_id);

  /// Handle bidirectional dependencies by collapsing into a single node.
  /// Returns a map of event IDs to merge (key -> value means merge value into
  /// key).
  [[nodiscard]] std::unordered_map<EventId, EventId>
  detect_bidirectional_dependencies(const EventList &events) const;

private:
  RequirementDictionary &req_dict_;
  ProducingDictionary &prod_dict_;
  RankStateManager &state_manager_;

  /// Resolve point-to-point communication (send/recv).
  std::vector<EventEdge> resolve_p2p(RankId consumer_rank, EventId recv_id);

  /// Resolve collective communication (reduce, allreduce, etc.).
  std::vector<EventEdge>
  resolve_collective(RankId consumer_rank, EventId collective_id,
                     const EventRecord &event);

  /// Check if all required producers have produced matching events.
  bool check_all_producers_satisfied(
      RankId consumer_rank, EventId required_event_id,
      const std::unordered_set<RankId> &required_producers);
};

} // namespace cpp_event
