#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace flexmaya {

struct RawEvent {
    std::uint64_t id = 0;
    int rank = 0;
    std::uint64_t thread_id = 0;
    int device = 0;
    std::uint64_t stream = 0;
    std::uint64_t correlation_id = 0;
    std::uint64_t timestamp_ns = 0;
    double duration_hint_us = 0.0;
    std::string api;
    std::string kind;
    std::uint64_t bytes = 0;
    std::uint64_t count = 0;
    int peer_rank = -1;
    std::uint64_t event_handle = 0;
    std::uint64_t wait_event_handle = 0;
    std::string collective_group;
    std::string code_partition;
    bool blocking = false;
};

struct Lane {
    std::uint32_t id = 0;
    int rank = 0;
    std::string kind;
    std::uint64_t stream = 0;
    std::uint64_t thread_id = 0;
};

struct DedupRankGroup {
    std::uint32_t id = 0;
    int representative_rank = 0;
    std::vector<int> ranks;
    std::uint64_t fingerprint = 0;
    std::uint64_t representative_event_count = 0;
    std::uint64_t logical_event_count = 0;
};

struct GlobalEvent {
    std::uint64_t id = 0;
    std::uint64_t raw_id = 0;
    int rank = 0;
    std::uint64_t thread_id = 0;
    int device = 0;
    std::uint64_t stream = 0;
    std::uint64_t correlation_id = 0;
    std::uint64_t timestamp_ns = 0;
    double duration_hint_us = 0.0;
    std::string api;
    std::string kind;
    std::uint64_t bytes = 0;
    std::uint64_t count = 0;
    int peer_rank = -1;
    std::uint64_t event_handle = 0;
    std::uint64_t wait_event_handle = 0;
    std::string collective_group;
    std::string code_partition;
    bool blocking = false;
    std::uint32_t lane_id = 0;
    std::uint64_t lane_pos = 0;
    std::uint32_t dedup_group_id = 0;
    std::uint32_t dedup_weight = 1;
};

struct Edge {
    std::uint64_t from = 0;
    std::uint64_t to = 0;
    std::string reason;
};

struct SyncPartition {
    std::uint64_t id = 0;
    std::string kind;
    std::string key;
    std::string code_partition;
    std::vector<std::uint64_t> event_ids;
    std::uint64_t logical_event_count = 0;
};

struct AnchorLineageEdge {
    std::string upper_partition;
    std::uint64_t lower_partition = 0;
    std::string lower_partition_kind;
    std::string edge_kind;
};

struct GlobalTrace {
    std::vector<GlobalEvent> events;
    std::vector<Lane> lanes;
    std::vector<Edge> edges;
    std::vector<SyncPartition> sync_partitions;
    std::vector<DedupRankGroup> dedup_groups;
    std::vector<AnchorLineageEdge> lineage_edges;
    std::uint64_t logical_event_count = 0;
    bool deduplicated = false;
};

class EventArena {
  public:
    std::uint64_t append(RawEvent event);
    void clear();
    std::size_t size() const;
    std::vector<RawEvent> events() const;
    GlobalTrace build_trace_ras() const;
    GlobalTrace build_rank_grouped_trace_ras(
        const std::map<int, std::vector<int>>& rank_groups) const;

  private:
    mutable std::mutex mutex_;
    std::uint64_t next_id_ = 1;
    std::vector<RawEvent> events_;
};

class SharedEventArena {
  public:
    static std::shared_ptr<SharedEventArena> create(
        const std::string& name,
        std::uint64_t capacity,
        bool unlink_existing);
    static std::shared_ptr<SharedEventArena> open(const std::string& name);
    ~SharedEventArena();

    std::uint64_t append(const RawEvent& event);
    std::vector<RawEvent> events() const;
    void write_binary(const std::string& path) const;
    std::uint64_t size() const;
    std::uint64_t capacity() const;
    const std::string& name() const;

  private:
    SharedEventArena(
        std::string name,
        int fd,
        void* mapping,
        std::uint64_t mapping_size,
        std::uint64_t capacity,
        bool owner);

    std::string name_;
    int fd_ = -1;
    void* mapping_ = nullptr;
    std::uint64_t mapping_size_ = 0;
    std::uint64_t capacity_ = 0;
    bool owner_ = false;
};

class HookRecorder {
  public:
    static HookRecorder& instance();
    std::uint64_t record(const RawEvent& event);
    std::vector<RawEvent> events() const;
    void clear_local();

  private:
    HookRecorder();
    std::shared_ptr<EventArena> local_arena_;
    std::shared_ptr<SharedEventArena> shared_arena_;
    std::string default_code_partition_;
};

GlobalTrace build_trace_ras(const std::vector<RawEvent>& raw_events);
GlobalTrace build_deduplicated_trace_ras(const std::vector<RawEvent>& raw_events);
GlobalTrace build_rank_grouped_trace_ras(
    const std::vector<RawEvent>& raw_events,
    const std::map<int, std::vector<int>>& rank_groups);
GlobalTrace build_trace_ras_from_binary(const std::vector<std::string>& paths);
GlobalTrace build_rank_grouped_trace_ras_from_binary(
    const std::vector<std::string>& paths,
    const std::map<int, std::vector<int>>& rank_groups);
GlobalTrace filter_trace_partitions(
    const GlobalTrace& trace,
    const std::vector<std::uint64_t>& partition_ids);
GlobalTrace patch_trace_code_partitions(
    const GlobalTrace& anchor,
    const GlobalTrace& replacement,
    const std::vector<std::string>& code_partitions);

std::uint64_t record_hook_api(
    const std::string& api,
    const std::string& kind,
    int rank,
    std::uint64_t thread_id,
    int device,
    std::uint64_t stream,
    std::uint64_t correlation_id,
    std::uint64_t timestamp_ns,
    double duration_hint_us,
    std::uint64_t bytes,
    std::uint64_t count,
    int peer_rank,
    std::uint64_t event_handle,
    std::uint64_t wait_event_handle,
    const std::string& collective_group,
    const std::string& code_partition,
    bool blocking);

std::uint64_t record_marker(
    int rank,
    const std::string& marker_kind,
    const std::string& code_partition,
    std::uint64_t timestamp_ns);

}  // namespace flexmaya

extern "C" {
std::uint64_t flexmaya_record_api_v1(
    const char* api,
    const char* kind,
    int rank,
    std::uint64_t thread_id,
    int device,
    std::uint64_t stream,
    std::uint64_t correlation_id,
    std::uint64_t timestamp_ns,
    double duration_hint_us,
    std::uint64_t bytes,
    std::uint64_t count,
    int peer_rank,
    std::uint64_t event_handle,
    std::uint64_t wait_event_handle,
    const char* collective_group,
    const char* code_partition,
    int blocking);

std::uint64_t flexmaya_record_marker_v1(
    int rank,
    const char* marker_kind,
    const char* code_partition,
    std::uint64_t timestamp_ns);

std::uint64_t plain_maya_hook_record_api_v2(
    const char* api,
    const char* kind,
    int rank,
    std::uint64_t thread_id,
    int device,
    std::uint64_t stream,
    std::uint64_t correlation_id,
    std::uint64_t timestamp_ns,
    double duration_hint_us,
    std::uint64_t bytes,
    std::uint64_t count,
    int peer_rank,
    std::uint64_t event_handle,
    std::uint64_t wait_event_handle,
    const char* collective_group,
    const char* code_partition,
    int blocking);
}
