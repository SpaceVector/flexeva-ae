#include "cpp_event/shared_event_merger.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cpp_event {

std::pair<EventList, std::unordered_map<EventId, EventId>>
SharedEventMerger::merge_shared_events(const EventList &events) {
  EventList merged_events;
  std::unordered_map<EventId, EventId> id_mapping;

  // Find shared event groups
  auto shared_groups = find_shared_groups(events);

  // Create a set of all event IDs that are in shared groups
  std::unordered_set<EventId> shared_event_ids;
  for (const auto &group : shared_groups) {
    for (EventId eid : group) {
      shared_event_ids.insert(eid);
    }
  }

  // Process each shared group
  EventId next_id = 1;
  for (const auto &group : shared_groups) {
    // Collect events in this group
    EventList group_events;
    for (EventId eid : group) {
      auto it = std::find_if(events.begin(), events.end(),
                             [eid](const EventRecord &e) {
                               return e.id == eid;
                             });
      if (it != events.end()) {
        group_events.push_back(*it);
      }
    }

    if (!group_events.empty()) {
      // Merge into a single event
      EventRecord merged = merge_event_group(events, group);
      merged.id = next_id++;
      merged_events.push_back(merged);

      // Map all original IDs to the merged ID
      for (EventId original_id : group) {
        id_mapping[original_id] = merged.id;
      }
    }
  }

  // Add non-shared events
  for (const auto &event : events) {
    if (shared_event_ids.find(event.id) == shared_event_ids.end()) {
      EventRecord copy = event;
      copy.id = next_id++;
      merged_events.push_back(copy);
      id_mapping[event.id] = copy.id;
    }
  }

  return {merged_events, id_mapping};
}

bool SharedEventMerger::is_shared_event(const EventRecord &event) {
  // Events with cross-rank scope are shared
  if (event.scope == EventScope::kCrossRank) {
    return true;
  }

  // Collective operations are shared
  switch (event.kind) {
  case EventKind::kCollective:
  case EventKind::kAllReduce:
  case EventKind::kBroadcast:
  case EventKind::kAllGather:
  case EventKind::kReduceScatter:
  case EventKind::kBarrier:
    return true;
  default:
    return false;
  }
}

std::vector<std::vector<EventId>>
SharedEventMerger::find_shared_groups(const EventList &events) const {
  std::vector<std::vector<EventId>> groups;

  // Group events by their shared characteristics
  // Events that should be merged together:
  // - Same kind
  // - Same active_group
  // - Same timestamp (approximately)

  std::unordered_map<std::string, std::vector<EventId>> group_map;

  for (const auto &event : events) {
    if (!is_shared_event(event)) {
      continue;
    }

    // Create a key based on kind, group ID, and approximate timestamp
    std::string key = std::to_string(static_cast<int>(event.kind)) + "_" +
                      event.active_group.id + "_" +
                      std::to_string(event.timestamp.count() / 1000000); // ms

    group_map[key].push_back(event.id);
  }

  // Convert map to vector
  for (auto &[key, event_ids] : group_map) {
    if (event_ids.size() > 1) {
      groups.push_back(std::move(event_ids));
    }
  }

  return groups;
}

EventRecord SharedEventMerger::merge_event_group(
    const EventList &all_events, const std::vector<EventId> &event_ids) const {
  if (event_ids.empty()) {
    return EventRecord{};
  }

  // Get the first event as base
  auto it = std::find_if(all_events.begin(), all_events.end(),
                         [&event_ids](const EventRecord &e) {
                           return e.id == event_ids[0];
                         });
  if (it == all_events.end()) {
    return EventRecord{};
  }

  EventRecord merged = *it;

  // Merge metadata from all events in the group
  // For shared events, we keep the same kind and scope
  // but ensure scope is marked as cross-rank
  merged.scope = EventScope::kCrossRank;

  // Merge active groups (should be the same for shared events)
  // Keep the first one

  return merged;
}

} // namespace cpp_event
