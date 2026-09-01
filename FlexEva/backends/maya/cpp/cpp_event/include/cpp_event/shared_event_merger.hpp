#pragma once

#include "cpp_event/event_schema.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cpp_event {

/// Merges shared events in SPSD mode.
/// Shared events (collectives, barriers) are collapsed into a single node.
class SharedEventMerger final {
public:
  SharedEventMerger() = default;

  /// Merge shared events in an event list.
  /// Returns a new event list with shared events merged, and a mapping from
  /// original event IDs to merged event IDs.
  std::pair<EventList, std::unordered_map<EventId, EventId>>
  merge_shared_events(const EventList &events);

  /// Check if an event should be merged (has shared scope).
  [[nodiscard]] static bool is_shared_event(const EventRecord &event);

  /// Find all events that should be merged together.
  [[nodiscard]] std::vector<std::vector<EventId>>
  find_shared_groups(const EventList &events) const;

private:
  /// Merge a group of events into a single event.
  [[nodiscard]] EventRecord
  merge_event_group(const EventList &all_events,
                    const std::vector<EventId> &event_ids) const;
};

} // namespace cpp_event
