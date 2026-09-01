#pragma once

#include "cpp_event/event_schema.hpp"

#include <unordered_map>
#include <vector>

namespace cpp_event {

/// Tracks thread and process creation/destruction events.
/// Maintains separate event flows for each thread/process.
class ThreadEventTracker final {
public:
  ThreadEventTracker() = default;

  /// Track thread creation.
  void record_thread_creation(EventId thread_id, RankId rank_id,
                              EventId parent_event_id);

  /// Track thread destruction.
  void record_thread_destruction(EventId thread_id);

  /// Track process creation.
  void record_process_creation(EventId process_id, RankId rank_id,
                               EventId parent_event_id);

  /// Track process destruction.
  void record_process_destruction(EventId process_id);

  /// Get all threads/processes for a rank.
  [[nodiscard]] std::vector<EventId> get_threads_for_rank(RankId rank_id) const;

  /// Check if a thread/process is active.
  [[nodiscard]] bool is_active(EventId thread_id) const;

  /// Get parent event for a thread/process.
  [[nodiscard]] EventId get_parent(EventId thread_id) const;

private:
  struct ThreadInfo {
    RankId rank_id{};
    EventId parent_event_id{0};
    bool is_active{true};
  };

  std::unordered_map<EventId, ThreadInfo> threads_;
  std::unordered_map<RankId, std::vector<EventId>> rank_threads_;
};

} // namespace cpp_event
