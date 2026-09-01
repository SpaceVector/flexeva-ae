#pragma once

#include "cpp_event/event_schema.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cpp_event {

/// Tracks events produced by ranks that may satisfy other ranks' requirements.
/// Key: producer rank ID
/// Value: map of event kind -> set of produced event IDs
class ProducingDictionary final {
public:
  ProducingDictionary() = default;

  /// Register a produced event: producer_rank produced an event of kind.
  void add_production(RankId producer_rank, EventKind event_kind,
                      EventId produced_event_id);

  /// Get all events produced by a rank.
  [[nodiscard]] std::vector<EventId>
  get_productions(RankId producer_rank) const;

  /// Get all events of a specific kind produced by a rank.
  [[nodiscard]] std::vector<EventId>
  get_productions_by_kind(RankId producer_rank, EventKind event_kind) const;

  /// Check if a rank has produced a specific event.
  [[nodiscard]] bool has_produced(RankId producer_rank,
                                   EventId produced_event_id) const;

  /// Remove a production record (for cleanup).
  void remove_production(RankId producer_rank, EventId produced_event_id);

  /// Clear all productions for a rank.
  void clear_productions(RankId producer_rank);

  /// Get all producer ranks.
  [[nodiscard]] std::vector<RankId> get_all_producers() const;

private:
  // producer_rank -> event_kind -> set of produced event IDs
  std::unordered_map<
      RankId, std::unordered_map<EventKind, std::unordered_set<EventId>>>
      productions_;
};

} // namespace cpp_event
