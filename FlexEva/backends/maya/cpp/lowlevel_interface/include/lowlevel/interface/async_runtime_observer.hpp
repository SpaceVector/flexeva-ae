#pragma once

#include "cpp_event/event_schema.hpp"

#include <string>
#include <string_view>

namespace cpp_event {
class EventLog;
}

namespace lowlevel::interface {

struct AsyncRuntimeObservation final {
  bool active{false};
  void *stream_handle{nullptr};
  void *start_event{nullptr};
  void *end_event{nullptr};
  std::string observation_id{};
};

AsyncRuntimeObservation begin_async_runtime_observation(std::string_view api_name,
                                                        void *stream_handle);

void complete_async_runtime_observation(AsyncRuntimeObservation &observation,
                                        cpp_event::EventPayload &payload,
                                        bool success);

void register_cublas_handle_for_async_runtime(void *handle) noexcept;
void unregister_cublas_handle_for_async_runtime(void *handle) noexcept;
void update_cublas_handle_stream_for_async_runtime(void *handle,
                                                   void *stream_handle) noexcept;
[[nodiscard]] void *lookup_cublas_handle_stream_for_async_runtime(void *handle) noexcept;
[[nodiscard]] bool lookup_registered_cublas_handle_stream_for_async_runtime(
    void *handle,
    void **stream_handle) noexcept;

void resolve_async_runtime_observations(cpp_event::EventLog &log);
void clear_async_runtime_observations() noexcept;

} // namespace lowlevel::interface
