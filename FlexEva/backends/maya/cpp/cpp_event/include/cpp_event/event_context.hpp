#pragma once

#include "cpp_event/event_schema.hpp"

#include <mutex>
#include <utility>

namespace cpp_event {

/// Snapshot of contextual metadata applied to each recorded event.
struct EventContextSnapshot final {
  RankGroup active_group{};
  Placement placement{};
  EventScope scope{EventScope::kLocal};
};

/// Long-lived context owned by CppEvent runtime components. Other subsystems
/// (e.g. pyextend) mutate the context; lowlevel-interface wrappers merely emit
/// API metadata.
class EventContext final {
public:
  EventContext() = default;

  void set_active_group(RankGroup group);
  void reset_active_group();

  void set_placement(Placement placement);
  void reset_placement();

  void set_scope(EventScope scope) noexcept;
  void reset_scope() noexcept;

  [[nodiscard]] EventContextSnapshot snapshot() const;

private:
  mutable std::mutex mutex_;
  RankGroup active_group_{};
  Placement placement_{};
  EventScope scope_{EventScope::kLocal};
};

inline void EventContext::set_active_group(RankGroup group) {
  std::lock_guard<std::mutex> guard(mutex_);
  active_group_ = std::move(group);
}

inline void EventContext::reset_active_group() {
  std::lock_guard<std::mutex> guard(mutex_);
  active_group_ = RankGroup{};
}

inline void EventContext::set_placement(Placement placement) {
  std::lock_guard<std::mutex> guard(mutex_);
  placement_ = std::move(placement);
}

inline void EventContext::reset_placement() {
  std::lock_guard<std::mutex> guard(mutex_);
  placement_ = Placement{};
}

inline void EventContext::set_scope(EventScope scope) noexcept {
  std::lock_guard<std::mutex> guard(mutex_);
  scope_ = scope;
}

inline void EventContext::reset_scope() noexcept {
  std::lock_guard<std::mutex> guard(mutex_);
  scope_ = EventScope::kLocal;
}

inline EventContextSnapshot EventContext::snapshot() const {
  std::lock_guard<std::mutex> guard(mutex_);
  EventContextSnapshot snap;
  snap.active_group = active_group_;
  snap.placement = placement_;
  snap.scope = scope_;
  return snap;
}

} // namespace cpp_event
