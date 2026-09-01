#include "cpp_event/event_time_assigner.hpp"

#include <algorithm>

namespace cpp_event {

std::chrono::nanoseconds
EventTimeAssigner::assign_time(const EventRecord &event) const {
  // Use placeholder logic based on event kind
  return assign_time_by_kind(event.kind);
}

std::chrono::nanoseconds
EventTimeAssigner::assign_time_by_kind(EventKind kind) const {
  auto [min_time, max_time] = get_time_range(kind);

  if (min_time == max_time) {
    return min_time;
  }

  // Generate random time in range
  std::uniform_int_distribution<std::chrono::nanoseconds::rep> dist(
      min_time.count(), max_time.count());
  return std::chrono::nanoseconds(dist(rng_));
}

std::chrono::nanoseconds EventTimeAssigner::assign_time_from_timestamp(
    const EventRecord &event, const EventRecord &previous_event) const {
  if (event.timestamp > previous_event.timestamp) {
    return event.timestamp - previous_event.timestamp;
  }
  return std::chrono::nanoseconds(0);
}

std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds>
EventTimeAssigner::get_time_range(EventKind kind) {
  using namespace std::chrono_literals;

  switch (kind) {
  case EventKind::kComputeKernel:
    return {100us, 10ms}; // 100 microseconds to 10 milliseconds

  case EventKind::kMemcpyHostToDevice:
  case EventKind::kMemcpyDeviceToHost:
  case EventKind::kMemcpyDeviceToDevice:
    return {10us, 1ms}; // 10 microseconds to 1 millisecond

  case EventKind::kSend:
  case EventKind::kRecv:
    return {50us, 5ms}; // 50 microseconds to 5 milliseconds

  case EventKind::kAllReduce:
  case EventKind::kBroadcast:
  case EventKind::kAllGather:
  case EventKind::kReduceScatter:
  case EventKind::kCollective:
    return {100us, 50ms}; // 100 microseconds to 50 milliseconds

  case EventKind::kBarrier:
    return {10us, 1ms}; // 10 microseconds to 1 millisecond

  case EventKind::kMemoryAllocation:
  case EventKind::kMemoryFree:
    return {1us, 100us}; // 1 microsecond to 100 microseconds

  case EventKind::kRuntimeCall:
    return {1us, 10us}; // 1 microsecond to 10 microseconds

  case EventKind::kFileRead:
  case EventKind::kFileWrite:
    return {1ms, 100ms}; // 1 millisecond to 100 milliseconds

  default:
    return {1us, 1ms}; // Default: 1 microsecond to 1 millisecond
  }
}

} // namespace cpp_event
