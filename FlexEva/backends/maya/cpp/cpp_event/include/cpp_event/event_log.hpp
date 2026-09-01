#pragma once

#include "cpp_event/event_context.hpp"
#include "cpp_event/event_schema.hpp"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <sys/syscall.h>
#include <unistd.h>

namespace cpp_event {

/// In-memory collector that materialises EventRecord instances.
class EventLog final {
public:
  using Clock = std::chrono::steady_clock;

  explicit EventLog(EventContext &context) : context_(context) {}

  void append(std::string_view api_name, EventKind kind,
              Clock::time_point start_time,
              const EventPayload &payload = {});
  void append(std::string_view api_name, EventKind kind,
              Clock::time_point start_time, Clock::time_point end_time,
              const EventPayload &payload = {});

  /// Returns a copy of the collected events.
  [[nodiscard]] EventList snapshot() const;

  template <typename Fn>
  void with_mutable_events(Fn &&visitor) {
    std::lock_guard<std::mutex> guard(mutex_);
    std::forward<Fn>(visitor)(events_);
  }

  /// Clears all recorded events.
  void clear();

private:
  [[nodiscard]] static EventDomain infer_domain(EventKind kind) noexcept;

  mutable std::mutex mutex_;
  EventList events_;
  EventContext &context_;
};

inline void EventLog::append(std::string_view api_name, EventKind kind,
                             Clock::time_point start_time,
                             const EventPayload &payload) {
  append(api_name, kind, start_time, start_time, payload);
}

inline void EventLog::append(std::string_view api_name, EventKind kind,
                             Clock::time_point start_time,
                             Clock::time_point end_time,
                             const EventPayload &payload) {
  const auto timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
      start_time.time_since_epoch());
  auto end_timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_time.time_since_epoch());
  if (end_timestamp < timestamp) {
    end_timestamp = timestamp;
  }
  std::lock_guard<std::mutex> guard(mutex_);

  EventRecord record{};
  record.id = 0; // IDs assigned later when graph is materialised.
  record.domain = infer_domain(kind);
  record.kind = kind;
  record.process_id = static_cast<std::int64_t>(::getpid());
  record.thread_id = static_cast<std::int64_t>(::syscall(SYS_gettid));
  const auto context_snapshot = context_.snapshot();
  record.scope = context_snapshot.scope;
  record.active_group = context_snapshot.active_group;
  record.api_name.assign(api_name.begin(), api_name.end());
  record.placement = context_snapshot.placement;
  record.timestamp = timestamp;
  record.end_timestamp = end_timestamp;
  record.host_duration = end_timestamp - timestamp;
  record.payload = payload;
  events_.push_back(std::move(record));
}

inline EventList EventLog::snapshot() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return events_;
}

inline void EventLog::clear() {
  std::lock_guard<std::mutex> guard(mutex_);
  events_.clear();
}

inline EventDomain EventLog::infer_domain(EventKind kind) noexcept {
  switch (kind) {
  case EventKind::kComputeKernel:
    return EventDomain::kCompute;
  case EventKind::kCollective:
  case EventKind::kPointToPoint:
  case EventKind::kBarrier:
  case EventKind::kAllReduce:
  case EventKind::kBroadcast:
  case EventKind::kAllGather:
  case EventKind::kReduceScatter:
  case EventKind::kSend:
  case EventKind::kRecv:
    return EventDomain::kCommunication;
  case EventKind::kRuntimeCall:
  case EventKind::kDebug:
    return EventDomain::kRuntime;
  case EventKind::kMemcpyHostToDevice:
  case EventKind::kMemcpyDeviceToHost:
  case EventKind::kMemcpyDeviceToDevice:
    return EventDomain::kMemory;
  case EventKind::kMemoryAllocation:
  case EventKind::kMemoryFree:
    return EventDomain::kMemory;
  case EventKind::kFileRead:
  case EventKind::kFileWrite:
    return EventDomain::kIO;
  case EventKind::kUnknown:
  default:
    return EventDomain::kUnknown;
  }
}

} // namespace cpp_event
