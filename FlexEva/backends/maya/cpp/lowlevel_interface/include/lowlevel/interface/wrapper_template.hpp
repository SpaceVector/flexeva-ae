#pragma once

#include "cpp_event/event_schema.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace lowlevel::interface {

/// Abstract recorder interface backing the wrappers.
class EventRecorder {
public:
  using Clock = std::chrono::steady_clock;
  virtual ~EventRecorder() = default;

  virtual void record_event(std::string_view api_name,
                            cpp_event::EventKind kind,
                            Clock::time_point start_time,
                            const cpp_event::EventPayload &payload = {}) = 0;

  virtual void record_event(std::string_view api_name,
                            cpp_event::EventKind kind,
                            Clock::time_point start_time,
                            Clock::time_point end_time,
                            const cpp_event::EventPayload &payload = {}) = 0;
};

/// Returns the process-wide recorder instance. Implementation provided by
/// runtime.
EventRecorder &recorder();

/// Installs a process-wide recorder. Returns the previously registered
/// recorder.
[[nodiscard]] EventRecorder *set_recorder(EventRecorder *recorder) noexcept;

/// Returns the currently registered recorder or nullptr if unset.
[[nodiscard]] EventRecorder *current_recorder() noexcept;

/// Clears the registered recorder, falling back to a null recorder.
void reset_recorder() noexcept;

inline void set_payload_attr(cpp_event::EventPayload &payload,
                             std::string_view key, const std::string &value) {
  payload.attributes[std::string(key)] = value;
}

inline void set_payload_attr(cpp_event::EventPayload &payload,
                             std::string_view key, std::string_view value) {
  payload.attributes[std::string(key)] = std::string(value);
}

inline void set_payload_attr(cpp_event::EventPayload &payload,
                             std::string_view key, const char *value) {
  if (value != nullptr) {
    payload.attributes[std::string(key)] = std::string(value);
  }
}

template <typename T>
std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, void>
set_payload_attr(cpp_event::EventPayload &payload, std::string_view key,
                 T value) {
  payload.attributes[std::string(key)] = std::to_string(value);
}

inline void set_payload_attr(cpp_event::EventPayload &payload,
                             std::string_view key, bool value) {
  payload.attributes[std::string(key)] = value ? "1" : "0";
}

template <typename T>
std::enable_if_t<std::is_floating_point_v<T>, void>
set_payload_attr(cpp_event::EventPayload &payload, std::string_view key,
                 T value) {
  payload.attributes[std::string(key)] = std::to_string(value);
}

template <typename T>
std::enable_if_t<std::is_enum_v<T>, void>
set_payload_attr(cpp_event::EventPayload &payload, std::string_view key,
                 T value) {
  using Underlying = std::underlying_type_t<T>;
  payload.attributes[std::string(key)] =
      std::to_string(static_cast<Underlying>(value));
}

template <typename T>
std::enable_if_t<std::is_pointer_v<T>, void>
set_payload_attr(cpp_event::EventPayload &, std::string_view, T) {}

inline bool launch_boundary_visibility_env_value_enabled(const char *value) {
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return std::strcmp(value, "1") == 0 ||
         std::strcmp(value, "true") == 0 ||
         std::strcmp(value, "TRUE") == 0 ||
         std::strcmp(value, "yes") == 0 ||
         std::strcmp(value, "YES") == 0 ||
         std::strcmp(value, "on") == 0 ||
         std::strcmp(value, "ON") == 0;
}

inline bool launch_boundary_visibility_diagnostics_enabled() {
  static const bool enabled = [] {
    const char *raw =
        std::getenv("MAYA_ENABLE_LAUNCH_BOUNDARY_VISIBILITY_DIAGNOSTICS");
    if (raw == nullptr || raw[0] == '\0') {
      raw = std::getenv(
          "FLEXSIM_MAYA_ENABLE_LAUNCH_BOUNDARY_VISIBILITY_DIAGNOSTICS");
    }
    return launch_boundary_visibility_env_value_enabled(raw);
  }();
  return enabled;
}

inline bool should_emit_launch_boundary_visibility_segments(
    std::string_view api_name) {
  return launch_boundary_visibility_diagnostics_enabled() &&
         (api_name == "cudaLaunchKernel" || api_name == "cudaGetDevice" ||
          api_name == "cublasSetStream_v2" || api_name == "cudaEventRecord" ||
          api_name == "cudaStreamWaitEvent");
}

inline std::string format_segment_duration_us(double value) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(3) << std::max(value, 0.0);
  return stream.str();
}

inline std::string escape_segment_json(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size() + 8);
  for (const unsigned char ch : value) {
    switch (ch) {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      if (ch < 0x20) {
        char buf[8];
        snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(ch));
        escaped += buf;
      } else {
        escaped.push_back(static_cast<char>(ch));
      }
      break;
    }
  }
  return escaped;
}

inline double elapsed_us(EventRecorder::Clock::time_point start,
                         EventRecorder::Clock::time_point end) {
  if (end < start) {
    return 0.0;
  }
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                 .count()) /
         1000.0;
}

struct BoundaryVisibilitySegmentLabel final {
  std::string_view name;
  std::string_view visibility_kind;
  std::string_view source_file_function;
  std::string_view classification_basis;
  bool included_in_paper_visible_host_duration{false};
  bool included_in_instrumentation_only_duration{false};
};

inline std::vector<BoundaryVisibilitySegmentLabel>
launch_boundary_visibility_segment_labels(std::string_view api_name) {
  if (api_name == "cudaLaunchKernel") {
    return {
        {"pre_call_payload_build", "instrumentation_only_wrapper_unmeasured",
         "cpp/lowlevel_interface/generated_wrapper::cudaLaunchKernel",
         "pre_start_payload_work_not_mechanically_timed_for_diagnostic_only_schema",
         false, true},
        {"async_runtime_observer_setup", "mixed_or_unresolved",
         "cpp/lowlevel_interface/generated_wrapper::cudaLaunchKernel",
         "structural_label_only_segment_timing_disabled_to_preserve_host_duration",
         false, false},
        {"real_api_body", "mixed_or_unresolved",
         "cpp/lowlevel_interface/generated_wrapper::cudaLaunchKernel",
         "structural_label_only_segment_timing_disabled_to_preserve_host_duration",
         false, false},
        {"post_call_payload_build", "instrumentation_only_wrapper_unmeasured",
         "cpp/lowlevel_interface/generated_wrapper::cudaLaunchKernel",
         "post_end_payload_work_not_mechanically_timed_for_diagnostic_only_schema",
         false, true},
    };
  }
  if (api_name == "cudaGetDevice") {
    return {{"real_api_body", "mixed_or_unresolved",
             "cpp/lowlevel_interface/generated_wrapper::cudaGetDevice",
             "structural_label_only_segment_timing_disabled_to_preserve_host_duration",
             false, false}};
  }
  if (api_name == "cublasSetStream_v2") {
    return {{"real_api_body", "mixed_or_unresolved",
             "cpp/lowlevel_interface/generated_wrapper::cublasSetStream_v2",
             "structural_label_only_segment_timing_disabled_to_preserve_host_duration",
             false, false}};
  }
  if (api_name == "cudaEventRecord") {
    return {{"real_api_body", "mixed_or_unresolved",
             "cpp/lowlevel_interface/generated_wrapper::cudaEventRecord",
             "structural_label_only_segment_timing_disabled_to_preserve_host_duration",
             false, false}};
  }
  if (api_name == "cudaStreamWaitEvent") {
    return {{"real_api_body", "mixed_or_unresolved",
             "cpp/lowlevel_interface/generated_wrapper::cudaStreamWaitEvent",
             "structural_label_only_segment_timing_disabled_to_preserve_host_duration",
             false, false}};
  }
  return {};
}

inline void attach_launch_boundary_visibility_metadata(
    cpp_event::EventPayload &payload, std::string_view api_name,
    EventRecorder::Clock::time_point wrapper_entry,
    EventRecorder::Clock::time_point wrapper_exit) {
  if (!should_emit_launch_boundary_visibility_segments(api_name)) {
    return;
  }

  const std::vector<BoundaryVisibilitySegmentLabel> segments =
      launch_boundary_visibility_segment_labels(api_name);
  std::ostringstream rows;
  rows << "[";
  bool first = true;
  for (const auto &segment : segments) {
    if (!first) {
      rows << ",";
    }
    first = false;
    rows << "{";
    rows << "\"name\":\"" << escape_segment_json(segment.name) << "\"";
    rows << ",\"visibility_kind\":\""
         << escape_segment_json(segment.visibility_kind) << "\"";
    rows << ",\"start_offset_us\":null";
    rows << ",\"end_offset_us\":null";
    rows << ",\"duration_us\":null";
    rows << ",\"clock\":\"unmeasured\"";
    rows << ",\"source_file_function\":\""
         << escape_segment_json(segment.source_file_function) << "\"";
    rows << ",\"classification_basis\":\""
         << escape_segment_json(segment.classification_basis) << "\"";
    rows << ",\"included_in_paper_visible_host_duration\":"
         << (segment.included_in_paper_visible_host_duration ? "true" : "false");
    rows << ",\"included_in_instrumentation_only_duration\":"
         << (segment.included_in_instrumentation_only_duration ? "true" : "false");
    rows << "}";
  }
  rows << "]";

  const double caller_visible_elapsed_us = elapsed_us(wrapper_entry, wrapper_exit);
  payload.attributes["boundary_segment_schema_version"] =
      "launch_boundary_visibility_v1";
  payload.attributes["wrapper_segment_coverage"] =
      "structural_labels_only_unmeasured";
  payload.attributes["wrapper_segment_sum_us"] = "0.000";
  payload.attributes["wrapper_segment_unattributed_us"] =
      format_segment_duration_us(caller_visible_elapsed_us);
  payload.attributes["caller_visible_elapsed_us"] =
      format_segment_duration_us(caller_visible_elapsed_us);
  payload.attributes["boundary_origin_kind"] = "mixed_or_unresolved";
  payload.attributes["boundary_origin_classification_basis"] =
      "producer_segment_timing_disabled_to_preserve_host_duration";
  payload.attributes["boundary_visibility_segments"] = rows.str();
  if (api_name == "cudaLaunchKernel") {
    payload.attributes["actual_launch_visibility_kind"] =
        "mixed_or_unresolved";
    payload.attributes["actual_launch_unavailable_reason"] =
        "internal_segment_timing_disabled_to_preserve_host_duration";
  }
}

/// Dispatch helper that wraps an API call with event recording.
template <typename Fn, typename... Args>
decltype(auto) dispatch(std::string_view api_name, cpp_event::EventKind kind,
                        cpp_event::EventPayload payload, Fn &&target,
                        Args &&...args) {
  constexpr bool kReturnsVoid =
      std::is_void_v<std::invoke_result_t<Fn, Args...>>;
  auto start = EventRecorder::Clock::now();
  auto &rec = recorder();

  if constexpr (kReturnsVoid) {
    std::invoke(std::forward<Fn>(target), std::forward<Args>(args)...);
    auto end = EventRecorder::Clock::now();
    rec.record_event(api_name, kind, start, end, payload);
  } else {
    auto result =
        std::invoke(std::forward<Fn>(target), std::forward<Args>(args)...);
    auto end = EventRecorder::Clock::now();
    rec.record_event(api_name, kind, start, end, payload);
    return result;
  }
}

template <typename Fn, typename... Args>
decltype(auto) dispatch(std::string_view api_name, cpp_event::EventKind kind,
                        Fn &&target, Args &&...args) {
  return dispatch(api_name, kind, cpp_event::EventPayload{},
                  std::forward<Fn>(target), std::forward<Args>(args)...);
}

} // namespace lowlevel::interface
