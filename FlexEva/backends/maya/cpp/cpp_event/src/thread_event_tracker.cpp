#include "cpp_event/thread_event_tracker.hpp"

namespace cpp_event {

void ThreadEventTracker::record_thread_creation(EventId thread_id,
                                                 RankId rank_id,
                                                 EventId parent_event_id) {
  ThreadInfo info;
  info.rank_id = rank_id;
  info.parent_event_id = parent_event_id;
  info.is_active = true;
  threads_[thread_id] = info;
  rank_threads_[rank_id].push_back(thread_id);
}

void ThreadEventTracker::record_thread_destruction(EventId thread_id) {
  auto it = threads_.find(thread_id);
  if (it != threads_.end()) {
    it->second.is_active = false;
  }
}

void ThreadEventTracker::record_process_creation(EventId process_id,
                                                  RankId rank_id,
                                                  EventId parent_event_id) {
  // Processes are tracked the same way as threads
  record_thread_creation(process_id, rank_id, parent_event_id);
}

void ThreadEventTracker::record_process_destruction(EventId process_id) {
  record_thread_destruction(process_id);
}

std::vector<EventId>
ThreadEventTracker::get_threads_for_rank(RankId rank_id) const {
  auto it = rank_threads_.find(rank_id);
  if (it != rank_threads_.end()) {
    return it->second;
  }
  return {};
}

bool ThreadEventTracker::is_active(EventId thread_id) const {
  auto it = threads_.find(thread_id);
  if (it == threads_.end()) {
    return false;
  }
  return it->second.is_active;
}

EventId ThreadEventTracker::get_parent(EventId thread_id) const {
  auto it = threads_.find(thread_id);
  if (it == threads_.end()) {
    return 0;
  }
  return it->second.parent_event_id;
}

} // namespace cpp_event
