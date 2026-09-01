#pragma once

#include "cpp_event/event_schema.hpp"

#include <unordered_map>
#include <vector>

namespace cpp_event {

/// Builds an event graph from a collection of event records.
/// Assigns event IDs and generates program-order edges within rank flows.
class EventGraphBuilder final {
public:
  EventGraphBuilder() = default;

  /// Builds an event graph from a list of event records.
  /// Assigns sequential IDs and generates program-order edges.
  [[nodiscard]] EventGraph build_graph(const EventList &events);

  /// Builds an event graph with program-order edges only (no dependency
  /// resolution). Useful for SPSD mode or initial graph construction.
  [[nodiscard]] EventGraph build_program_order_graph(const EventList &events);

private:
  /// Assigns sequential IDs to events in the list.
  void assign_event_ids(EventList &events);

  /// Generates program-order edges within each rank's event flow.
  /// Events are grouped by rank (from active_group) and ordered by timestamp.
  void generate_program_order_edges(EventGraph &graph);

  /// Groups events by rank for processing.
  [[nodiscard]] std::unordered_map<RankId, std::vector<EventId>>
  group_events_by_rank(const EventList &events) const;

  EventId next_event_id_{1};
};

} // namespace cpp_event
