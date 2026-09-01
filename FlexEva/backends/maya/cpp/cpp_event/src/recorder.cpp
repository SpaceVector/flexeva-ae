#include "lowlevel/interface/wrapper_template.hpp"

#include <atomic>

namespace lowlevel::interface {
namespace {

std::atomic<EventRecorder *> g_recorder{nullptr};

class NullRecorder final : public EventRecorder {
public:
  void record_event(std::string_view /*api_name*/,
                    cpp_event::EventKind /*kind*/,
                    Clock::time_point /*start_time*/,
                    const cpp_event::EventPayload & /*payload*/ = {}) override {}

  void record_event(std::string_view /*api_name*/,
                    cpp_event::EventKind /*kind*/,
                    Clock::time_point /*start_time*/,
                    Clock::time_point /*end_time*/,
                    const cpp_event::EventPayload & /*payload*/ = {}) override {}
};

EventRecorder &fallback_recorder() {
  static NullRecorder instance;
  return instance;
}

} // namespace

EventRecorder &recorder() {
  if (auto *rec = g_recorder.load(std::memory_order_acquire)) {
    return *rec;
  }
  return fallback_recorder();
}

EventRecorder *set_recorder(EventRecorder *recorder) noexcept {
  return g_recorder.exchange(recorder, std::memory_order_acq_rel);
}

EventRecorder *current_recorder() noexcept {
  return g_recorder.load(std::memory_order_acquire);
}

void reset_recorder() noexcept {
  g_recorder.store(nullptr, std::memory_order_release);
}

} // namespace lowlevel::interface
