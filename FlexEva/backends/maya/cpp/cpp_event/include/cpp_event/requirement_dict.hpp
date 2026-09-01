#pragma once

#include "cpp_event/event_schema.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cpp_event {

/// Tracks which events must be produced by which ranks before a rank can proceed.
/// Key: consumer rank ID
/// Value: map of required event IDs -> set of producer rank IDs
class RequirementDictionary final {
public:
  RequirementDictionary() = default;

  /// Register a requirement: consumer_rank needs an event of kind from
  /// producer_rank.
  void add_requirement(RankId consumer_rank, RankId producer_rank,
                       EventKind event_kind, EventId required_event_id);

  /// Get all requirements for a consumer rank.
  [[nodiscard]] std::vector<EventId>
  get_requirements(RankId consumer_rank) const;

  /// Get specific requirement: which producers can satisfy this requirement.
  [[nodiscard]] std::unordered_set<RankId>
  get_required_producers(RankId consumer_rank, EventId required_event_id) const;

  /// Check if a requirement is satisfied by a producer.
  [[nodiscard]] bool is_requirement_satisfied(RankId consumer_rank,
                                               EventId required_event_id,
                                               RankId producer_rank) const;

  /// Remove a requirement (when satisfied).
  void remove_requirement(RankId consumer_rank, EventId required_event_id);

  /// Remove all requirements for a rank.
  void clear_requirements(RankId consumer_rank);

  /// Check if a rank has any pending requirements.
  [[nodiscard]] bool has_requirements(RankId consumer_rank) const;

private:
  // consumer_rank -> required_event_id -> set of producer ranks
  std::unordered_map<RankId,
                     std::unordered_map<EventId, std::unordered_set<RankId>>>
      requirements_;
};

} // namespace cpp_event
