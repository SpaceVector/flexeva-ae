#pragma once

#include "cpp_event/event_schema.hpp"

#include <chrono>
#include <random>

namespace cpp_event {

/// Assigns execution times to events.
/// Currently uses placeholder logic (random or timestamp-derived).
/// Designed to be extensible for future metrics (latency, bandwidth, power).
class EventTimeAssigner final {
public:
  explicit EventTimeAssigner(std::uint64_t seed = 0)
      : rng_(seed == 0 ? std::random_device{}() : seed) {}

  /// Assign execution time to an event.
  /// Returns duration in nanoseconds.
  [[nodiscard]] std::chrono::nanoseconds
  assign_time(const EventRecord &event) const;

  /// Assign execution time based on event kind (placeholder logic).
  [[nodiscard]] std::chrono::nanoseconds
  assign_time_by_kind(EventKind kind) const;

  /// Assign execution time from timestamp difference (for NORMAL code).
  [[nodiscard]] std::chrono::nanoseconds
  assign_time_from_timestamp(const EventRecord &event,
                             const EventRecord &previous_event) const;

private:
  mutable std::mt19937 rng_;

  /// Get default time range for an event kind (in nanoseconds).
  [[nodiscard]] static std::pair<std::chrono::nanoseconds,
                                 std::chrono::nanoseconds>
  get_time_range(EventKind kind);
};

} // namespace cpp_event
