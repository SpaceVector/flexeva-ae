#include "cpp_event/dependency_resolver.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace cpp_event {

std::vector<EventEdge>
DependencyResolver::resolve_dependencies(RankId consumer_rank) {
  std::vector<EventEdge> edges;

  if (!req_dict_.has_requirements(consumer_rank)) {
    return edges;
  }

  auto requirements = req_dict_.get_requirements(consumer_rank);
  for (EventId required_event_id : requirements) {
    // Get which producers can satisfy this requirement
    auto required_producers =
        req_dict_.get_required_producers(consumer_rank, required_event_id);

    // Check if all required producers have satisfied this requirement
    if (check_all_producers_satisfied(consumer_rank, required_event_id,
                                      required_producers)) {
      // Generate dependency edges from each producer to consumer
      for (RankId producer_rank : required_producers) {
        // Find the produced event that satisfies this requirement
        // For now, we use a simple matching: find the most recent production
        // of matching kind
        // In practice, this should match specific event IDs based on the
        // requirement
        auto prod_events =
            prod_dict_.get_productions_by_kind(producer_rank, EventKind::kSend);
        if (prod_events.empty()) {
          // Try other kinds
          prod_events = prod_dict_.get_productions(producer_rank);
        }

        if (!prod_events.empty()) {
          // Use the most recent production (last in list)
          EventId producer_event_id = prod_events.back();

          EventEdge edge;
          edge.from = producer_event_id;
          edge.to = required_event_id;
          edge.reason = "dependency";
          edges.push_back(edge);

          // Mark requirement as satisfied
          req_dict_.remove_requirement(consumer_rank, required_event_id);
        }
      }

      // Update rank state: unblock if all requirements are satisfied
      if (!req_dict_.has_requirements(consumer_rank)) {
        state_manager_.set_state(consumer_rank, RankExecutionState::kRunning);
      }
    }
  }

  return edges;
}

std::vector<EventEdge> DependencyResolver::resolve_all_dependencies() {
  std::vector<EventEdge> all_edges;

  // Get all ranks with requirements
  // We'll need to track which ranks have requirements
  // For now, iterate through known ranks in state manager
  auto all_ranks = state_manager_.get_all_ranks();
  for (RankId rank : all_ranks) {
    if (req_dict_.has_requirements(rank)) {
      auto edges = resolve_dependencies(rank);
      all_edges.insert(all_edges.end(), edges.begin(), edges.end());
    }
  }

  return all_edges;
}

bool DependencyResolver::can_satisfy_requirement(RankId consumer_rank,
                                                  EventId required_event_id) {
  auto required_producers =
      req_dict_.get_required_producers(consumer_rank, required_event_id);
  return check_all_producers_satisfied(consumer_rank, required_event_id,
                                       required_producers);
}

std::unordered_map<EventId, EventId> DependencyResolver::
    detect_bidirectional_dependencies(const EventList &events) const {
  std::unordered_map<EventId, EventId> merge_map;

  // For bidirectional dependencies, we look for events where:
  // - Event A depends on Event B
  // - Event B depends on Event A
  // These should be collapsed into a single node (like collective operations)

  // This is a simplified implementation - in practice, we'd need to analyze
  // the dependency graph more carefully
  // For now, we detect obvious cases like matching send/recv pairs

  for (const auto &event1 : events) {
    if (event1.kind != EventKind::kSend && event1.kind != EventKind::kRecv) {
      continue;
    }

    for (const auto &event2 : events) {
      if (event2.id == event1.id) {
        continue;
      }

      // Check if these form a bidirectional pair
      if ((event1.kind == EventKind::kSend && event2.kind == EventKind::kRecv) ||
          (event1.kind == EventKind::kRecv && event2.kind == EventKind::kSend)) {
        // Check if they're in the same group and match
        if (!event1.active_group.empty() && !event2.active_group.empty()) {
          // For bidirectional, merge the recv into the send
          if (event1.kind == EventKind::kSend &&
              event2.kind == EventKind::kRecv) {
            merge_map[event2.id] = event1.id;
          }
        }
      }
    }
  }

  return merge_map;
}

std::vector<EventEdge>
DependencyResolver::resolve_p2p(RankId consumer_rank, EventId recv_id) {
  std::vector<EventEdge> edges;
  // Point-to-point resolution is handled in resolve_dependencies
  // This method can be extended for more specific P2P logic
  return edges;
}

std::vector<EventEdge>
DependencyResolver::resolve_collective(RankId consumer_rank,
                                       EventId collective_id,
                                       const EventRecord &event) {
  std::vector<EventEdge> edges;
  // Collective resolution requires checking all participants
  // This is handled in resolve_dependencies with multiple producers
  return edges;
}

bool DependencyResolver::check_all_producers_satisfied(
    RankId consumer_rank, EventId required_event_id,
    const std::unordered_set<RankId> &required_producers) {
  // Check if all required producers have produced matching events
  for (RankId producer_rank : required_producers) {
    // For now, we check if the producer has any productions
    // In practice, we'd match specific event kinds/types
    if (prod_dict_.get_productions(producer_rank).empty()) {
      return false;
    }
  }
  return true;
}

} // namespace cpp_event
