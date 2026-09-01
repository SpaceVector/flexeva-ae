#include "cpp_event/event_graph_builder.hpp"

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace cpp_event {

EventGraph EventGraphBuilder::build_graph(const EventList &events) {
  EventGraph graph;
  graph.events = events;

  // Assign IDs to events
  assign_event_ids(graph.events);

  // Generate program-order edges
  generate_program_order_edges(graph);

  return graph;
}

EventGraph
EventGraphBuilder::build_program_order_graph(const EventList &events) {
  return build_graph(events);
}

void EventGraphBuilder::assign_event_ids(EventList &events) {
  EventId id = 1;
  for (auto &event : events) {
    event.id = id++;
  }
  next_event_id_ = id;
}

void EventGraphBuilder::generate_program_order_edges(EventGraph &graph) {
  // Group events by rank
  auto rank_groups = group_events_by_rank(graph.events);

  // For each rank, generate program-order edges
  for (const auto &[rank, event_ids] : rank_groups) {
    if (event_ids.size() < 2) {
      continue; // Need at least 2 events for an edge
    }

    // Sort event IDs by their timestamp
    std::vector<std::pair<EventId, std::chrono::nanoseconds>> timed_events;
    for (EventId eid : event_ids) {
      // Find the event with this ID
      auto it = std::find_if(
          graph.events.begin(), graph.events.end(),
          [eid](const EventRecord &rec) { return rec.id == eid; });
      if (it != graph.events.end()) {
        timed_events.emplace_back(eid, it->timestamp);
      }
    }

    // Sort by timestamp
    std::sort(timed_events.begin(), timed_events.end(),
              [](const auto &a, const auto &b) {
                return a.second < b.second;
              });

    // Generate edges between consecutive events
    for (size_t i = 0; i < timed_events.size() - 1; ++i) {
      EventEdge edge;
      edge.from = timed_events[i].first;
      edge.to = timed_events[i + 1].first;
      edge.reason = "program_order";
      graph.edges.push_back(edge);
    }
  }
}

std::unordered_map<RankId, std::vector<EventId>>
EventGraphBuilder::group_events_by_rank(const EventList &events) const {
  std::unordered_map<RankId, std::vector<EventId>> rank_map;

  for (const auto &event : events) {
    if (event.active_group.empty()) {
      // If no active group, we can't determine rank - skip for now
      continue;
    }

    // For events with active groups, add to all ranks in the group
    // This handles both local and cross-rank events
    for (RankId rank : event.active_group.members) {
      rank_map[rank].push_back(event.id);
    }
  }

  return rank_map;
}

} // namespace cpp_event
