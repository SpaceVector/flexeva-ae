#pragma once

#include "cpp_event/event_schema.hpp"

#include <string>
#include <string_view>

namespace cpp_event {
class EventLog;
}

namespace lowlevel::interface {

struct CuptiActivityMetadataObservation final {
  bool active{false};
  std::string wrapper_logical_event_id{};
  std::string external_correlation_id{};
  std::string api_name{};
  std::string api_role{};
  std::string begin_thread_id{};
  std::string status{};
  std::string unavailable_reason{};
  bool external_correlation_pushed{false};
};

CuptiActivityMetadataObservation
begin_cupti_activity_metadata_observation(std::string_view api_name);

void complete_cupti_activity_metadata_observation(
    CuptiActivityMetadataObservation &observation,
    cpp_event::EventPayload &payload,
    bool success);

void resolve_cupti_activity_metadata_observations(cpp_event::EventLog &log);
void clear_cupti_activity_metadata_observations() noexcept;

} // namespace lowlevel::interface
