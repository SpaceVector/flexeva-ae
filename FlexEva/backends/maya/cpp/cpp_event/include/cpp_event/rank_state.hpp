#pragma once

#include "cpp_event/event_schema.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cpp_event {

/// Execution state of a rank.
enum class RankExecutionState {
  kRunning,   ///< Rank is executing normally.
  kBlocked,   ///< Rank is blocked waiting for dependencies.
  kCompleted, ///< Rank has finished execution.
};

/// Tracks the execution state and position for each rank.
struct RankState final {
  RankId rank_id{};
  RankExecutionState state{RankExecutionState::kRunning};
  EventId current_event_id{0}; ///< Last processed event ID for this rank.
  std::vector<EventId> pending_requirements{}; ///< Events this rank is waiting for.
};

/// Manages execution state for all ranks in the system.
class RankStateManager final {
public:
  RankStateManager() = default;

  /// Initialize state for a rank.
  void initialize_rank(RankId rank_id);

  /// Get the current state of a rank.
  [[nodiscard]] RankExecutionState get_state(RankId rank_id) const;

  /// Set the state of a rank.
  void set_state(RankId rank_id, RankExecutionState state);

  /// Get the current event ID for a rank.
  [[nodiscard]] EventId get_current_event(RankId rank_id) const;

  /// Update the current event ID for a rank.
  void set_current_event(RankId rank_id, EventId event_id);

  /// Add a pending requirement for a rank.
  void add_requirement(RankId rank_id, EventId required_event_id);

  /// Get all pending requirements for a rank.
  [[nodiscard]] const std::vector<EventId> &
  get_requirements(RankId rank_id) const;

  /// Clear requirements for a rank.
  void clear_requirements(RankId rank_id);

  /// Check if a rank has any pending requirements.
  [[nodiscard]] bool has_requirements(RankId rank_id) const;

  /// Get all registered ranks.
  [[nodiscard]] std::vector<RankId> get_all_ranks() const;

private:
  std::unordered_map<RankId, RankState> rank_states_;
};

} // namespace cpp_event
