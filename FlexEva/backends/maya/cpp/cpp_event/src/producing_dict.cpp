#include "cpp_event/producing_dict.hpp"

namespace cpp_event {

void ProducingDictionary::add_production(RankId producer_rank,
                                         EventKind event_kind,
                                         EventId produced_event_id) {
  productions_[producer_rank][event_kind].insert(produced_event_id);
}

std::vector<EventId>
ProducingDictionary::get_productions(RankId producer_rank) const {
  std::vector<EventId> result;
  auto it = productions_.find(producer_rank);
  if (it != productions_.end()) {
    for (const auto &[kind, event_ids] : it->second) {
      for (EventId event_id : event_ids) {
        result.push_back(event_id);
      }
    }
  }
  return result;
}

std::vector<EventId>
ProducingDictionary::get_productions_by_kind(RankId producer_rank,
                                             EventKind event_kind) const {
  std::vector<EventId> result;
  auto it = productions_.find(producer_rank);
  if (it != productions_.end()) {
    auto kind_it = it->second.find(event_kind);
    if (kind_it != it->second.end()) {
      for (EventId event_id : kind_it->second) {
        result.push_back(event_id);
      }
    }
  }
  return result;
}

bool ProducingDictionary::has_produced(RankId producer_rank,
                                       EventId produced_event_id) const {
  auto it = productions_.find(producer_rank);
  if (it == productions_.end()) {
    return false;
  }

  for (const auto &[kind, event_ids] : it->second) {
    if (event_ids.find(produced_event_id) != event_ids.end()) {
      return true;
    }
  }
  return false;
}

void ProducingDictionary::remove_production(RankId producer_rank,
                                            EventId produced_event_id) {
  auto it = productions_.find(producer_rank);
  if (it != productions_.end()) {
    for (auto &[kind, event_ids] : it->second) {
      event_ids.erase(produced_event_id);
    }
    // Clean up empty entries
    auto kind_it = it->second.begin();
    while (kind_it != it->second.end()) {
      if (kind_it->second.empty()) {
        kind_it = it->second.erase(kind_it);
      } else {
        ++kind_it;
      }
    }
    if (it->second.empty()) {
      productions_.erase(it);
    }
  }
}

void ProducingDictionary::clear_productions(RankId producer_rank) {
  productions_.erase(producer_rank);
}

std::vector<RankId> ProducingDictionary::get_all_producers() const {
  std::vector<RankId> producers;
  producers.reserve(productions_.size());
  for (const auto &[rank_id, productions] : productions_) {
    producers.push_back(rank_id);
  }
  return producers;
}

} // namespace cpp_event
