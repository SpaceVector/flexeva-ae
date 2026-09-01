#pragma once

#include "cpp_event/event_log.hpp"

#include "lowlevel/interface/wrapper_template.hpp"

namespace cpp_event {

/// Adapter that forwards lowlevel-interface recordings into an EventLog.
class EventLogRecorderAdapter final
    : public lowlevel::interface::EventRecorder {
public:
  using Clock = lowlevel::interface::EventRecorder::Clock;

  explicit EventLogRecorderAdapter(EventLog &log) : log_(log) {}

  void record_event(std::string_view api_name, EventKind kind,
                    Clock::time_point start_time,
                    const EventPayload &payload = {}) override {
    log_.append(api_name, kind, start_time, payload);
  }

  void record_event(std::string_view api_name, EventKind kind,
                    Clock::time_point start_time,
                    Clock::time_point end_time,
                    const EventPayload &payload = {}) override {
    log_.append(api_name, kind, start_time, end_time, payload);
  }

  [[nodiscard]] EventLog &log() noexcept { return log_; }
  [[nodiscard]] const EventLog &log() const noexcept { return log_; }

private:
  EventLog &log_;
};

} // namespace cpp_event
