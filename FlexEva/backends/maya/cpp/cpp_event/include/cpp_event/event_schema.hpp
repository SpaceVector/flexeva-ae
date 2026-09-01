#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cpp_event {

using EventId = std::uint64_t;
using RankId = std::int32_t;
using WorldSize = std::int32_t;
using DeviceId = std::string;
using StreamId = std::string;
using AttributeKey = std::string;
using AttributeValue = std::string;
using AttributeMap = std::unordered_map<AttributeKey, AttributeValue>;

using RankList = std::vector<RankId>;

/// High-level classification of an event.
enum class EventDomain {
  kUnknown,
  kCompute,
  kCommunication,
  kSynchronization,
  kRuntime,
  kMemory,
  kIO,
};

/// Concrete kind of operation represented by an event.
enum class EventKind {
  kUnknown,
  kComputeKernel,
  kCollective,
  kPointToPoint,
  kBarrier,
  kAllReduce,
  kBroadcast,
  kAllGather,
  kReduceScatter,
  kSend,
  kRecv,
  kRuntimeCall,
  kMemcpyHostToDevice,
  kMemcpyDeviceToHost,
  kMemcpyDeviceToDevice,
  kMemoryAllocation,
  kMemoryFree,
  kFileRead,
  kFileWrite,
  kDebug,
};

/// Scope indicates whether the event is rank-local or spans multiple ranks.
enum class EventScope {
  kLocal,
  kCrossRank,
};

/// Identifies a synchronization or collective group.
using GroupId = std::string;

/// Represents the current active rank group provided by pyextend/bindlayer.
struct RankGroup final {
  GroupId id{};
  RankList members{};

  [[nodiscard]] bool empty() const noexcept { return members.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return members.size(); }
  [[nodiscard]] bool contains(RankId rank) const noexcept {
    return std::find(members.begin(), members.end(), rank) != members.end();
  }
};

/// Describes the placement of an event within the cluster.
struct Placement final {
  WorldSize world_size{0};
  DeviceId device{};
  StreamId stream{};
  GroupId group{};
};

/// Unstructured payload for events that require additional contextual data.
struct EventPayload final {
  AttributeMap attributes{};
};

/// Minimal schema for an event node in the execution graph.
struct EventRecord final {
  EventId id{};
  EventDomain domain{EventDomain::kUnknown};
  EventKind kind{EventKind::kUnknown};
  EventScope scope{EventScope::kLocal};
  std::int64_t process_id{0};
  std::int64_t thread_id{0};
  RankGroup active_group{}; ///< Rank group active when the event was emitted.
  std::string api_name{}; ///< Name of the wrapped API that triggered the event.
  Placement placement{};
  std::chrono::nanoseconds timestamp{};
  std::chrono::nanoseconds end_timestamp{};
  std::chrono::nanoseconds host_duration{};
  EventPayload payload{};

  /// Number of event nodes this record should expand into when materializing a
  /// graph.
  [[nodiscard]] std::size_t node_count_hint() const noexcept {
    if (scope == EventScope::kCrossRank || active_group.empty()) {
      return 1;
    }
    return active_group.size();
  }
};

/// Directed dependency edge between two event nodes.
struct EventEdge final {
  EventId from{};
  EventId to{};
  std::string reason{}; ///< optional human-readable explanation
};

/// Container alias representing a collection of events.
using EventList = std::vector<EventRecord>;

/// Container alias representing a collection of dependency edges.
using EventEdges = std::vector<EventEdge>;

/// Lightweight view for referencing events without copying payloads.
struct EventView final {
  const EventRecord *record{nullptr};

  [[nodiscard]] const EventRecord *operator->() const noexcept {
    return record;
  }

  [[nodiscard]] const EventRecord &operator*() const noexcept {
    return *record;
  }

  explicit operator bool() const noexcept { return record != nullptr; }
};

/// Result of building a graph for a single traced program execution.
struct EventGraph final {
  EventList events{};
  EventEdges edges{};
};

} // namespace cpp_event
