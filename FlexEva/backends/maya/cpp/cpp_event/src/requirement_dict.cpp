#include "cpp_event/requirement_dict.hpp"

namespace cpp_event {

void RequirementDictionary::add_requirement(RankId consumer_rank,
                                            RankId producer_rank,
                                            EventKind /* event_kind */,
                                            EventId required_event_id) {
  requirements_[consumer_rank][required_event_id].insert(producer_rank);
}

std::vector<EventId>
RequirementDictionary::get_requirements(RankId consumer_rank) const {
  std::vector<EventId> result;
  auto it = requirements_.find(consumer_rank);
  if (it != requirements_.end()) {
    for (const auto &[event_id, producers] : it->second) {
      result.push_back(event_id);
    }
  }
  return result;
}

std::unordered_set<RankId> RequirementDictionary::get_required_producers(
    RankId consumer_rank, EventId required_event_id) const {
  std::unordered_set<RankId> result;
  auto it = requirements_.find(consumer_rank);
  if (it != requirements_.end()) {
    auto event_it = it->second.find(required_event_id);
    if (event_it != it->second.end()) {
      result = event_it->second;
    }
  }
  return result;
}

bool RequirementDictionary::is_requirement_satisfied(
    RankId consumer_rank, EventId required_event_id,
    RankId producer_rank) const {
  auto it = requirements_.find(consumer_rank);
  if (it == requirements_.end()) {
    return false;
  }

  auto event_it = it->second.find(required_event_id);
  if (event_it == it->second.end()) {
    return false;
  }

  return event_it->second.find(producer_rank) != event_it->second.end();
}

void RequirementDictionary::remove_requirement(RankId consumer_rank,
                                               EventId required_event_id) {
  auto it = requirements_.find(consumer_rank);
  if (it != requirements_.end()) {
    it->second.erase(required_event_id);
    if (it->second.empty()) {
      requirements_.erase(it);
    }
  }
}

void RequirementDictionary::clear_requirements(RankId consumer_rank) {
  requirements_.erase(consumer_rank);
}

bool RequirementDictionary::has_requirements(RankId consumer_rank) const {
  auto it = requirements_.find(consumer_rank);
  if (it == requirements_.end()) {
    return false;
  }
  return !it->second.empty();
}

} // namespace cpp_event
