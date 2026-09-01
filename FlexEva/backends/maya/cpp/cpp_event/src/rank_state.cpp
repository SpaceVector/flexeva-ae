#include "cpp_event/rank_state.hpp"

#include <stdexcept>

namespace cpp_event {

void RankStateManager::initialize_rank(RankId rank_id) {
  RankState state;
  state.rank_id = rank_id;
  state.state = RankExecutionState::kRunning;
  rank_states_[rank_id] = state;
}

RankExecutionState RankStateManager::get_state(RankId rank_id) const {
  auto it = rank_states_.find(rank_id);
  if (it == rank_states_.end()) {
    return RankExecutionState::kCompleted;
  }
  return it->second.state;
}

void RankStateManager::set_state(RankId rank_id, RankExecutionState state) {
  auto it = rank_states_.find(rank_id);
  if (it == rank_states_.end()) {
    initialize_rank(rank_id);
    it = rank_states_.find(rank_id);
  }
  it->second.state = state;
}

EventId RankStateManager::get_current_event(RankId rank_id) const {
  auto it = rank_states_.find(rank_id);
  if (it == rank_states_.end()) {
    return 0;
  }
  return it->second.current_event_id;
}

void RankStateManager::set_current_event(RankId rank_id, EventId event_id) {
  auto it = rank_states_.find(rank_id);
  if (it == rank_states_.end()) {
    initialize_rank(rank_id);
    it = rank_states_.find(rank_id);
  }
  it->second.current_event_id = event_id;
}

void RankStateManager::add_requirement(RankId rank_id,
                                       EventId required_event_id) {
  auto it = rank_states_.find(rank_id);
  if (it == rank_states_.end()) {
    initialize_rank(rank_id);
    it = rank_states_.find(rank_id);
  }
  it->second.pending_requirements.push_back(required_event_id);
}

const std::vector<EventId> &
RankStateManager::get_requirements(RankId rank_id) const {
  static const std::vector<EventId> empty_vec;
  auto it = rank_states_.find(rank_id);
  if (it == rank_states_.end()) {
    return empty_vec;
  }
  return it->second.pending_requirements;
}

void RankStateManager::clear_requirements(RankId rank_id) {
  auto it = rank_states_.find(rank_id);
  if (it != rank_states_.end()) {
    it->second.pending_requirements.clear();
  }
}

bool RankStateManager::has_requirements(RankId rank_id) const {
  auto it = rank_states_.find(rank_id);
  if (it == rank_states_.end()) {
    return false;
  }
  return !it->second.pending_requirements.empty();
}

std::vector<RankId> RankStateManager::get_all_ranks() const {
  std::vector<RankId> ranks;
  ranks.reserve(rank_states_.size());
  for (const auto &[rank_id, state] : rank_states_) {
    ranks.push_back(rank_id);
  }
  return ranks;
}

} // namespace cpp_event
