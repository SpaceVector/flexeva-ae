#include "flexmaya_core.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <unordered_set>
#include <unistd.h>

namespace flexmaya {
namespace {

constexpr std::uint64_t kSharedMagic = 0x464c45584d415941ULL;
constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

struct SharedHeader {
    std::uint64_t magic;
    std::uint64_t capacity;
    std::atomic<std::uint64_t> next_id;
};

struct SharedRawEventSlot {
    std::uint64_t id;
    int rank;
    std::uint64_t thread_id;
    int device;
    std::uint64_t stream;
    std::uint64_t correlation_id;
    std::uint64_t timestamp_ns;
    double duration_hint_us;
    std::uint64_t bytes;
    std::uint64_t count;
    int peer_rank;
    std::uint64_t event_handle;
    std::uint64_t wait_event_handle;
    std::uint8_t blocking;
    char api[96];
    char kind[48];
    char collective_group[160];
    char code_partition[96];
};

SharedHeader* header(void* mapping) {
    return reinterpret_cast<SharedHeader*>(mapping);
}

SharedRawEventSlot* slots(void* mapping) {
    return reinterpret_cast<SharedRawEventSlot*>(
        reinterpret_cast<char*>(mapping) + sizeof(SharedHeader));
}

std::uint64_t mapping_size_for(std::uint64_t capacity) {
    return sizeof(SharedHeader) + capacity * sizeof(SharedRawEventSlot);
}

void copy_string(char* dst, std::size_t dst_size, const std::string& value) {
    if (dst_size == 0) {
        return;
    }
    const std::size_t n = std::min(dst_size - 1, value.size());
    std::memcpy(dst, value.data(), n);
    dst[n] = '\0';
}

std::string env_value(const char* key) {
    const char* value = std::getenv(key);
    return value == nullptr ? std::string() : std::string(value);
}

constexpr std::uint64_t kRawBinaryMagic = 0x464c455852415731ULL;

template <typename T>
void write_binary_value(std::ofstream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!stream) {
        throw std::runtime_error("failed to write compact trace");
    }
}

template <typename T>
T read_binary_value(std::ifstream& stream) {
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!stream) {
        throw std::runtime_error("truncated compact trace");
    }
    return value;
}

void write_binary_string(std::ofstream& stream, const std::string& value) {
    const auto length = static_cast<std::uint64_t>(value.size());
    write_binary_value(stream, length);
    stream.write(value.data(), static_cast<std::streamsize>(length));
    if (!stream) {
        throw std::runtime_error("failed to write compact trace string");
    }
}

std::string read_binary_string(std::ifstream& stream) {
    const auto length = read_binary_value<std::uint64_t>(stream);
    if (length > (1ULL << 20)) {
        throw std::runtime_error("compact trace string is too large");
    }
    std::string value(static_cast<std::size_t>(length), '\0');
    stream.read(value.data(), static_cast<std::streamsize>(length));
    if (!stream) {
        throw std::runtime_error("truncated compact trace string");
    }
    return value;
}

void write_raw_events_binary(const std::vector<RawEvent>& events, const std::string& path) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot open compact trace for writing: " + path);
    }
    write_binary_value(stream, kRawBinaryMagic);
    write_binary_value(stream, static_cast<std::uint64_t>(events.size()));
    for (const RawEvent& event : events) {
        write_binary_value(stream, event.id);
        write_binary_value(stream, static_cast<std::int32_t>(event.rank));
        write_binary_value(stream, event.thread_id);
        write_binary_value(stream, static_cast<std::int32_t>(event.device));
        write_binary_value(stream, event.stream);
        write_binary_value(stream, event.correlation_id);
        write_binary_value(stream, event.timestamp_ns);
        write_binary_value(stream, event.duration_hint_us);
        write_binary_value(stream, event.bytes);
        write_binary_value(stream, event.count);
        write_binary_value(stream, static_cast<std::int32_t>(event.peer_rank));
        write_binary_value(stream, event.event_handle);
        write_binary_value(stream, event.wait_event_handle);
        write_binary_value(stream, static_cast<std::uint8_t>(event.blocking ? 1 : 0));
        write_binary_string(stream, event.api);
        write_binary_string(stream, event.kind);
        write_binary_string(stream, event.collective_group);
        write_binary_string(stream, event.code_partition);
    }
}

std::vector<RawEvent> read_raw_events_binary_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open compact trace: " + path);
    }
    if (read_binary_value<std::uint64_t>(stream) != kRawBinaryMagic) {
        throw std::runtime_error("invalid compact trace header: " + path);
    }
    const auto count = read_binary_value<std::uint64_t>(stream);
    if (count > (1ULL << 32)) {
        throw std::runtime_error("compact trace has too many events");
    }
    std::vector<RawEvent> events;
    events.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
        RawEvent event;
        event.id = read_binary_value<std::uint64_t>(stream);
        event.rank = read_binary_value<std::int32_t>(stream);
        event.thread_id = read_binary_value<std::uint64_t>(stream);
        event.device = read_binary_value<std::int32_t>(stream);
        event.stream = read_binary_value<std::uint64_t>(stream);
        event.correlation_id = read_binary_value<std::uint64_t>(stream);
        event.timestamp_ns = read_binary_value<std::uint64_t>(stream);
        event.duration_hint_us = read_binary_value<double>(stream);
        event.bytes = read_binary_value<std::uint64_t>(stream);
        event.count = read_binary_value<std::uint64_t>(stream);
        event.peer_rank = read_binary_value<std::int32_t>(stream);
        event.event_handle = read_binary_value<std::uint64_t>(stream);
        event.wait_event_handle = read_binary_value<std::uint64_t>(stream);
        event.blocking = read_binary_value<std::uint8_t>(stream) != 0;
        event.api = read_binary_string(stream);
        event.kind = read_binary_string(stream);
        event.collective_group = read_binary_string(stream);
        event.code_partition = read_binary_string(stream);
        events.push_back(std::move(event));
    }
    return events;
}

RawEvent slot_to_event(const SharedRawEventSlot& slot) {
    RawEvent event;
    event.id = slot.id;
    event.rank = slot.rank;
    event.thread_id = slot.thread_id;
    event.device = slot.device;
    event.stream = slot.stream;
    event.correlation_id = slot.correlation_id;
    event.timestamp_ns = slot.timestamp_ns;
    event.duration_hint_us = slot.duration_hint_us;
    event.bytes = slot.bytes;
    event.count = slot.count;
    event.peer_rank = slot.peer_rank;
    event.event_handle = slot.event_handle;
    event.wait_event_handle = slot.wait_event_handle;
    event.blocking = slot.blocking != 0;
    event.api = slot.api;
    event.kind = slot.kind;
    event.collective_group = slot.collective_group;
    event.code_partition = slot.code_partition;
    return event;
}

void event_to_slot(const RawEvent& event, SharedRawEventSlot* slot) {
    slot->id = event.id;
    slot->rank = event.rank;
    slot->thread_id = event.thread_id;
    slot->device = event.device;
    slot->stream = event.stream;
    slot->correlation_id = event.correlation_id;
    slot->timestamp_ns = event.timestamp_ns;
    slot->duration_hint_us = event.duration_hint_us;
    slot->bytes = event.bytes;
    slot->count = event.count;
    slot->peer_rank = event.peer_rank;
    slot->event_handle = event.event_handle;
    slot->wait_event_handle = event.wait_event_handle;
    slot->blocking = event.blocking ? 1 : 0;
    copy_string(slot->api, sizeof(slot->api), event.api);
    copy_string(slot->kind, sizeof(slot->kind), event.kind);
    copy_string(slot->collective_group, sizeof(slot->collective_group), event.collective_group);
    copy_string(slot->code_partition, sizeof(slot->code_partition), event.code_partition);
}

bool is_stream_event(const RawEvent& event) {
    if (event.kind == "kernel_launch" || event.kind == "blas_compute" ||
        event.kind == "nccl_collective" || event.kind == "mem_copy") {
        return true;
    }
    if (event.kind == "stream_op" && event.stream != 0) {
        return true;
    }
    return event.stream != 0 && event.kind != "host_delay" && event.kind != "host_marker";
}

bool is_collective(const GlobalEvent& event) {
    return event.kind == "nccl_collective" || event.api.rfind("nccl", 0) == 0;
}

bool is_event_record(const GlobalEvent& event) {
    return event.api.find("EventRecord") != std::string::npos && event.event_handle != 0;
}

bool is_event_wait(const GlobalEvent& event) {
    return event.api.find("WaitEvent") != std::string::npos && event.wait_event_handle != 0;
}

bool is_stream_sync(const GlobalEvent& event) {
    return event.api.find("StreamSynchronize") != std::string::npos;
}

bool is_device_sync(const GlobalEvent& event) {
    return event.api.find("DeviceSynchronize") != std::string::npos;
}

std::string collective_base_key(const GlobalEvent& event) {
    if (!event.collective_group.empty()) {
        return event.collective_group;
    }
    std::ostringstream oss;
    oss << event.api << ":peer=" << event.peer_rank << ":bytes=" << event.bytes
        << ":count=" << event.count;
    return oss.str();
}

std::uint64_t fnv_mix(std::uint64_t state, const std::string& value) {
    for (unsigned char ch : value) {
        state ^= static_cast<std::uint64_t>(ch);
        state *= kFnvPrime;
    }
    return state;
}

std::uint64_t fnv_mix_u64(std::uint64_t state, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        state ^= (value >> (i * 8)) & 0xffULL;
        state *= kFnvPrime;
    }
    return state;
}

bool simulation_relevant(const RawEvent& event) {
    if (event.kind == "host_marker") {
        return false;
    }
    if (event.kind == "context_op" && !event.blocking) {
        return false;
    }
    return true;
}

std::uint64_t rank_fingerprint(const std::vector<RawEvent>& events) {
    std::uint64_t fp = kFnvOffset;
    for (const RawEvent& event : events) {
        if (!simulation_relevant(event)) {
            continue;
        }
        fp = fnv_mix(fp, event.api);
        fp = fnv_mix(fp, event.kind);
        fp = fnv_mix_u64(fp, event.stream == 0 ? 0 : 1);
        fp = fnv_mix_u64(fp, event.bytes);
        fp = fnv_mix_u64(fp, event.count);
        fp = fnv_mix(fp, event.collective_group.empty() ? "collective" : event.collective_group);
        fp = fnv_mix(fp, event.code_partition);
    }
    return fp;
}

class LaneTable {
  public:
    std::uint32_t lane_for(const RawEvent& raw) {
        Lane key;
        key.rank = raw.rank;
        if (is_stream_event(raw)) {
            key.kind = "stream";
            key.stream = raw.stream;
            key.thread_id = 0;
        } else {
            key.kind = "host";
            key.stream = 0;
            key.thread_id = raw.thread_id;
        }
        const std::string map_key =
            std::to_string(key.rank) + "|" + key.kind + "|" +
            std::to_string(key.stream) + "|" + std::to_string(key.thread_id);
        auto iter = ids_.find(map_key);
        if (iter != ids_.end()) {
            return iter->second;
        }
        key.id = static_cast<std::uint32_t>(lanes_.size() + 1);
        ids_[map_key] = key.id;
        lanes_.push_back(key);
        return key.id;
    }

    const std::vector<Lane>& lanes() const {
        return lanes_;
    }

  private:
    std::unordered_map<std::string, std::uint32_t> ids_;
    std::vector<Lane> lanes_;
};

std::map<int, std::vector<RawEvent>> events_by_rank(const std::vector<RawEvent>& raw_events) {
    std::map<int, std::vector<RawEvent>> rows;
    for (RawEvent event : raw_events) {
        rows[event.rank].push_back(std::move(event));
    }
    for (auto& item : rows) {
        std::sort(item.second.begin(), item.second.end(), [](const RawEvent& left, const RawEvent& right) {
            if (left.timestamp_ns != right.timestamp_ns) {
                return left.timestamp_ns < right.timestamp_ns;
            }
            return left.id < right.id;
        });
    }
    return rows;
}

std::vector<DedupRankGroup> planned_groups(
    const std::vector<RawEvent>& raw_events,
    const std::map<int, std::vector<int>>& rank_groups) {
    std::map<int, std::uint64_t> counts;
    std::set<int> all_ranks;
    for (const RawEvent& event : raw_events) {
        counts[event.rank] += 1;
        all_ranks.insert(event.rank);
    }
    std::set<int> covered;
    std::vector<DedupRankGroup> groups;
    for (const auto& item : rank_groups) {
        if (item.second.empty()) {
            continue;
        }
        DedupRankGroup group;
        group.id = static_cast<std::uint32_t>(groups.size() + 1);
        group.representative_rank = item.first;
        group.ranks = item.second;
        std::sort(group.ranks.begin(), group.ranks.end());
        group.representative_event_count = counts[group.representative_rank];
        group.fingerprint = fnv_mix_u64(kFnvOffset, static_cast<std::uint64_t>(group.representative_rank));
        for (int rank : group.ranks) {
            const std::uint64_t member_count = counts[rank] == 0
                ? group.representative_event_count
                : counts[rank];
            group.logical_event_count += member_count;
            covered.insert(rank);
            group.fingerprint = fnv_mix_u64(group.fingerprint, static_cast<std::uint64_t>(rank));
        }
        groups.push_back(group);
    }
    for (int rank : all_ranks) {
        if (covered.count(rank) != 0) {
            continue;
        }
        DedupRankGroup group;
        group.id = static_cast<std::uint32_t>(groups.size() + 1);
        group.representative_rank = rank;
        group.ranks = {rank};
        group.representative_event_count = counts[rank];
        group.logical_event_count = counts[rank];
        group.fingerprint = fnv_mix_u64(kFnvOffset, static_cast<std::uint64_t>(rank));
        groups.push_back(group);
    }
    return groups;
}

std::vector<DedupRankGroup> pattern_groups(const std::vector<RawEvent>& raw_events) {
    std::map<int, std::vector<RawEvent>> by_rank = events_by_rank(raw_events);
    std::map<std::uint64_t, std::vector<int>> by_fp;
    std::map<int, std::uint64_t> rank_count;
    for (const auto& item : by_rank) {
        by_fp[rank_fingerprint(item.second)].push_back(item.first);
        rank_count[item.first] = item.second.size();
    }
    std::vector<DedupRankGroup> groups;
    for (auto& item : by_fp) {
        std::sort(item.second.begin(), item.second.end());
        DedupRankGroup group;
        group.id = static_cast<std::uint32_t>(groups.size() + 1);
        group.representative_rank = item.second.front();
        group.ranks = item.second;
        group.fingerprint = item.first;
        group.representative_event_count = rank_count[group.representative_rank];
        for (int rank : group.ranks) {
            group.logical_event_count += rank_count[rank];
        }
        groups.push_back(group);
    }
    std::sort(groups.begin(), groups.end(), [](const DedupRankGroup& left, const DedupRankGroup& right) {
        return left.representative_rank < right.representative_rank;
    });
    for (std::size_t i = 0; i < groups.size(); ++i) {
        groups[i].id = static_cast<std::uint32_t>(i + 1);
    }
    return groups;
}

void attach_dedup(GlobalTrace* trace, const std::vector<DedupRankGroup>& groups) {
    std::unordered_map<int, DedupRankGroup> by_rep;
    for (const DedupRankGroup& group : groups) {
        by_rep[group.representative_rank] = group;
    }
    for (GlobalEvent& event : trace->events) {
        auto iter = by_rep.find(event.rank);
        if (iter == by_rep.end()) {
            continue;
        }
        event.dedup_group_id = iter->second.id;
        event.dedup_weight = static_cast<std::uint32_t>(std::max<std::size_t>(iter->second.ranks.size(), 1));
    }
    trace->dedup_groups = groups;
    trace->deduplicated = groups.size() > 0;
}

void finalize_counts(GlobalTrace* trace) {
    trace->logical_event_count = 0;
    std::unordered_map<std::uint64_t, const GlobalEvent*> by_id;
    for (const GlobalEvent& event : trace->events) {
        trace->logical_event_count += std::max<std::uint32_t>(event.dedup_weight, 1);
        by_id[event.id] = &event;
    }
    for (SyncPartition& partition : trace->sync_partitions) {
        partition.logical_event_count = 0;
        for (std::uint64_t event_id : partition.event_ids) {
            auto iter = by_id.find(event_id);
            if (iter == by_id.end()) {
                continue;
            }
            partition.logical_event_count += std::max<std::uint32_t>(iter->second->dedup_weight, 1);
        }
    }
}

std::vector<RawEvent> representative_events(
    const std::vector<RawEvent>& raw_events,
    const std::vector<DedupRankGroup>& groups) {
    std::set<int> representatives;
    for (const DedupRankGroup& group : groups) {
        representatives.insert(group.representative_rank);
    }
    std::vector<RawEvent> filtered;
    for (const RawEvent& event : raw_events) {
        if (representatives.count(event.rank) != 0) {
            filtered.push_back(event);
        }
    }
    return filtered;
}

}  // namespace

std::uint64_t EventArena::append(RawEvent event) {
    std::lock_guard<std::mutex> guard(mutex_);
    event.id = next_id_++;
    events_.push_back(std::move(event));
    return events_.back().id;
}

void EventArena::clear() {
    std::lock_guard<std::mutex> guard(mutex_);
    next_id_ = 1;
    events_.clear();
}

std::size_t EventArena::size() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return events_.size();
}

std::vector<RawEvent> EventArena::events() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return events_;
}

GlobalTrace EventArena::build_trace_ras() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return flexmaya::build_trace_ras(events_);
}

GlobalTrace EventArena::build_rank_grouped_trace_ras(
    const std::map<int, std::vector<int>>& rank_groups) const {
    std::lock_guard<std::mutex> guard(mutex_);
    return flexmaya::build_rank_grouped_trace_ras(events_, rank_groups);
}

SharedEventArena::SharedEventArena(
    std::string name,
    int fd,
    void* mapping,
    std::uint64_t mapping_size,
    std::uint64_t capacity,
    bool owner)
    : name_(std::move(name)),
      fd_(fd),
      mapping_(mapping),
      mapping_size_(mapping_size),
      capacity_(capacity),
      owner_(owner) {}

SharedEventArena::~SharedEventArena() {
    if (mapping_ != nullptr) {
        munmap(mapping_, mapping_size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
    if (owner_ && !name_.empty()) {
        shm_unlink(name_.c_str());
    }
}

std::shared_ptr<SharedEventArena> SharedEventArena::create(
    const std::string& name,
    std::uint64_t capacity,
    bool unlink_existing) {
    if (capacity == 0) {
        throw std::invalid_argument("SharedEventArena capacity must be positive");
    }
    std::string shm_name = name;
    if (shm_name.empty() || shm_name[0] != '/') {
        shm_name = "/" + shm_name;
    }
    if (unlink_existing) {
        shm_unlink(shm_name.c_str());
    }
    const int fd = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
        throw std::runtime_error("shm_open create failed for " + shm_name + ": " + std::strerror(errno));
    }
    const std::uint64_t map_size = mapping_size_for(capacity);
    if (ftruncate(fd, static_cast<off_t>(map_size)) != 0) {
        close(fd);
        shm_unlink(shm_name.c_str());
        throw std::runtime_error("ftruncate failed for " + shm_name + ": " + std::strerror(errno));
    }
    void* mapping = mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) {
        close(fd);
        shm_unlink(shm_name.c_str());
        throw std::runtime_error("mmap create failed for " + shm_name + ": " + std::strerror(errno));
    }
    auto* hdr = header(mapping);
    hdr->magic = kSharedMagic;
    hdr->capacity = capacity;
    new (&hdr->next_id) std::atomic<std::uint64_t>(1);
    return std::shared_ptr<SharedEventArena>(
        new SharedEventArena(shm_name, fd, mapping, map_size, capacity, true));
}

std::shared_ptr<SharedEventArena> SharedEventArena::open(const std::string& name) {
    std::string shm_name = name;
    if (shm_name.empty() || shm_name[0] != '/') {
        shm_name = "/" + shm_name;
    }
    const int fd = shm_open(shm_name.c_str(), O_RDWR, 0600);
    if (fd < 0) {
        throw std::runtime_error("shm_open open failed for " + shm_name + ": " + std::strerror(errno));
    }
    SharedHeader probe {};
    if (pread(fd, &probe, sizeof(probe), 0) != static_cast<ssize_t>(sizeof(probe))) {
        close(fd);
        throw std::runtime_error("failed to read shared arena header for " + shm_name);
    }
    if (probe.magic != kSharedMagic || probe.capacity == 0) {
        close(fd);
        throw std::runtime_error("invalid shared arena header for " + shm_name);
    }
    const std::uint64_t map_size = mapping_size_for(probe.capacity);
    void* mapping = mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("mmap open failed for " + shm_name + ": " + std::strerror(errno));
    }
    return std::shared_ptr<SharedEventArena>(
        new SharedEventArena(shm_name, fd, mapping, map_size, probe.capacity, false));
}

std::uint64_t SharedEventArena::append(const RawEvent& input) {
    auto* hdr = header(mapping_);
    const std::uint64_t id = hdr->next_id.fetch_add(1, std::memory_order_acq_rel);
    if (id == 0 || id > capacity_) {
        hdr->next_id.fetch_sub(1, std::memory_order_acq_rel);
        throw std::runtime_error("SharedEventArena capacity exceeded");
    }
    RawEvent event = input;
    event.id = id;
    event_to_slot(event, &slots(mapping_)[id - 1]);
    return id;
}

std::vector<RawEvent> SharedEventArena::events() const {
    const auto* hdr = header(mapping_);
    const std::uint64_t next = hdr->next_id.load(std::memory_order_acquire);
    const std::uint64_t n = next == 0 ? 0 : std::min<std::uint64_t>(next - 1, capacity_);
    std::vector<RawEvent> out;
    out.reserve(static_cast<std::size_t>(n));
    const auto* slot_rows = slots(mapping_);
    for (std::uint64_t i = 0; i < n; ++i) {
        if (slot_rows[i].id != 0) {
            out.push_back(slot_to_event(slot_rows[i]));
        }
    }
    return out;
}

void SharedEventArena::write_binary(const std::string& path) const {
    write_raw_events_binary(events(), path);
}

std::uint64_t SharedEventArena::size() const {
    const auto* hdr = header(mapping_);
    const std::uint64_t next = hdr->next_id.load(std::memory_order_acquire);
    return next == 0 ? 0 : std::min<std::uint64_t>(next - 1, capacity_);
}

std::uint64_t SharedEventArena::capacity() const {
    return capacity_;
}

const std::string& SharedEventArena::name() const {
    return name_;
}

HookRecorder& HookRecorder::instance() {
    static HookRecorder recorder;
    return recorder;
}

HookRecorder::HookRecorder() : local_arena_(std::make_shared<EventArena>()) {
    std::string shm_name = env_value("FLEXMAYA_SHM_NAME");
    if (shm_name.empty()) {
        shm_name = env_value("PLAIN_MAYA_SHM_NAME");
    }
    default_code_partition_ = env_value("FLEXMAYA_CODE_PARTITION");
    if (default_code_partition_.empty()) {
        default_code_partition_ = env_value("PLAIN_MAYA_CODE_PARTITION");
    }
    if (!shm_name.empty()) {
        try {
            shared_arena_ = SharedEventArena::open(shm_name);
        } catch (...) {
            shared_arena_.reset();
        }
    }
}

std::uint64_t HookRecorder::record(const RawEvent& event) {
    RawEvent row = event;
    if (row.code_partition.empty() && !default_code_partition_.empty()) {
        row.code_partition = default_code_partition_;
    }
    if (shared_arena_) {
        return shared_arena_->append(row);
    }
    return local_arena_->append(row);
}

std::vector<RawEvent> HookRecorder::events() const {
    if (shared_arena_) {
        return shared_arena_->events();
    }
    return local_arena_->events();
}

void HookRecorder::clear_local() {
    local_arena_->clear();
}

GlobalTrace build_trace_ras(const std::vector<RawEvent>& raw_events) {
    std::vector<RawEvent> sorted = raw_events;
    std::sort(sorted.begin(), sorted.end(), [](const RawEvent& left, const RawEvent& right) {
        if (left.rank != right.rank) {
            return left.rank < right.rank;
        }
        if (left.timestamp_ns != right.timestamp_ns) {
            return left.timestamp_ns < right.timestamp_ns;
        }
        return left.id < right.id;
    });

    GlobalTrace trace;
    LaneTable lanes;
    std::unordered_map<std::uint32_t, std::uint64_t> last_on_lane;
    std::unordered_map<std::uint32_t, std::uint64_t> lane_pos;
    std::unordered_map<std::uint32_t, std::uint64_t> trace_window_epoch;
    std::unordered_map<std::uint32_t, bool> trace_window_has_events;

    std::unordered_map<std::string, std::uint64_t> recorded_events;
    std::map<std::pair<int, std::uint32_t>, std::uint64_t> collective_occurrence;
    std::map<std::string, std::vector<std::uint64_t>> collective_members;
    std::map<std::string, SyncPartition> partition_by_key;

    for (const RawEvent& raw : sorted) {
        const std::uint32_t lane_id = lanes.lane_for(raw);
        GlobalEvent event;
        event.id = static_cast<std::uint64_t>(trace.events.size() + 1);
        event.raw_id = raw.id;
        event.rank = raw.rank;
        event.thread_id = raw.thread_id;
        event.device = raw.device;
        event.stream = raw.stream;
        event.correlation_id = raw.correlation_id;
        event.timestamp_ns = raw.timestamp_ns;
        event.duration_hint_us = raw.duration_hint_us;
        event.api = raw.api;
        event.kind = raw.kind;
        event.bytes = raw.bytes;
        event.count = raw.count;
        event.peer_rank = raw.peer_rank;
        event.event_handle = raw.event_handle;
        event.wait_event_handle = raw.wait_event_handle;
        event.collective_group = raw.collective_group;
        event.code_partition = raw.code_partition;
        event.blocking = raw.blocking;
        event.lane_id = lane_id;
        event.lane_pos = ++lane_pos[lane_id];

        auto last_iter = last_on_lane.find(lane_id);
        if (last_iter != last_on_lane.end()) {
            trace.edges.push_back({last_iter->second, event.id, "lane_order"});
        }
        last_on_lane[lane_id] = event.id;

        if (is_event_record(event)) {
            recorded_events[std::to_string(event.rank) + "|" + std::to_string(event.event_handle)] = event.id;
        }
        if (is_event_wait(event)) {
            const std::string key = std::to_string(event.rank) + "|" + std::to_string(event.wait_event_handle);
            auto iter = recorded_events.find(key);
            if (iter != recorded_events.end()) {
                trace.edges.push_back({iter->second, event.id, "event_wait"});
            }
        }

        if (is_collective(event)) {
            const auto occurrence_key = std::make_pair(event.rank, lane_id);
            const std::uint64_t occ = ++collective_occurrence[occurrence_key];
            const std::string key = collective_base_key(event) + "|occ:" + std::to_string(occ);
            event.collective_group = key;
            collective_members[key].push_back(event.id);
        }

        const bool boundary_event = is_stream_sync(event) || is_device_sync(event) || is_collective(event);
        if (boundary_event && trace_window_has_events[lane_id]) {
            ++trace_window_epoch[lane_id];
            trace_window_has_events[lane_id] = false;
        }

        if (is_stream_sync(event) || is_device_sync(event)) {
            SyncPartition partition;
            partition.id = static_cast<std::uint64_t>(partition_by_key.size() + 1);
            partition.kind = is_stream_sync(event) ? "stream_sync" : "device_sync";
            partition.key = partition.kind + ":rank=" + std::to_string(event.rank) +
                            ":stream=" + std::to_string(event.stream) +
                            ":event=" + std::to_string(event.id);
            partition.event_ids.push_back(event.id);
            partition_by_key[partition.key] = partition;
        }

        if (!event.code_partition.empty()) {
            const std::uint64_t epoch = trace_window_epoch[lane_id];
            const std::string key = "trace_window:code=" + event.code_partition +
                                    ":rank=" + std::to_string(event.rank) +
                                    ":lane=" + std::to_string(lane_id) +
                                    ":window=" + std::to_string(epoch);
            auto iter = partition_by_key.find(key);
            if (iter == partition_by_key.end()) {
                SyncPartition partition;
                partition.id = static_cast<std::uint64_t>(partition_by_key.size() + 1);
                partition.kind = "trace_window";
                partition.key = key;
                partition.code_partition = event.code_partition;
                iter = partition_by_key.emplace(key, std::move(partition)).first;
            }
            iter->second.event_ids.push_back(event.id);
            trace_window_has_events[lane_id] = true;
        }

        if (boundary_event) {
            ++trace_window_epoch[lane_id];
            trace_window_has_events[lane_id] = false;
        }

        trace.events.push_back(std::move(event));
    }

    trace.lanes = lanes.lanes();

    for (auto& item : collective_members) {
        SyncPartition partition;
        partition.id = static_cast<std::uint64_t>(partition_by_key.size() + 1);
        partition.kind = "collective";
        partition.key = item.first;
        partition.event_ids = item.second;
        partition_by_key[partition.key] = std::move(partition);
    }

    for (auto& item : partition_by_key) {
        trace.sync_partitions.push_back(std::move(item.second));
    }
    std::sort(trace.sync_partitions.begin(), trace.sync_partitions.end(), [](const auto& left, const auto& right) {
        return left.id < right.id;
    });
    std::set<std::pair<std::string, std::uint64_t>> lineage_seen;
    for (const SyncPartition& partition : trace.sync_partitions) {
        if (partition.kind != "trace_window" || partition.code_partition.empty()) {
            continue;
        }
        const auto key = std::make_pair(partition.code_partition, partition.id);
        if (lineage_seen.insert(key).second) {
            trace.lineage_edges.push_back(
                {partition.code_partition, partition.id, partition.kind, "code_to_trace_partition"});
        }
    }
    finalize_counts(&trace);
    return trace;
}

GlobalTrace build_deduplicated_trace_ras(const std::vector<RawEvent>& raw_events) {
    const std::vector<DedupRankGroup> groups = pattern_groups(raw_events);
    std::vector<RawEvent> filtered = representative_events(raw_events, groups);
    GlobalTrace trace = build_trace_ras(filtered);
    attach_dedup(&trace, groups);
    finalize_counts(&trace);
    return trace;
}

GlobalTrace build_rank_grouped_trace_ras(
    const std::vector<RawEvent>& raw_events,
    const std::map<int, std::vector<int>>& rank_groups) {
    const std::vector<DedupRankGroup> groups = planned_groups(raw_events, rank_groups);
    std::vector<RawEvent> filtered = representative_events(raw_events, groups);
    GlobalTrace trace = build_trace_ras(filtered);
    attach_dedup(&trace, groups);
    finalize_counts(&trace);
    return trace;
}

GlobalTrace build_trace_ras_from_binary(const std::vector<std::string>& paths) {
    std::vector<RawEvent> events;
    for (const std::string& path : paths) {
        std::vector<RawEvent> part = read_raw_events_binary_file(path);
        events.insert(events.end(),
                      std::make_move_iterator(part.begin()),
                      std::make_move_iterator(part.end()));
    }
    return build_trace_ras(events);
}

GlobalTrace build_rank_grouped_trace_ras_from_binary(
    const std::vector<std::string>& paths,
    const std::map<int, std::vector<int>>& rank_groups) {
    std::vector<RawEvent> events;
    for (const std::string& path : paths) {
        std::vector<RawEvent> part = read_raw_events_binary_file(path);
        events.insert(events.end(),
                      std::make_move_iterator(part.begin()),
                      std::make_move_iterator(part.end()));
    }
    return build_rank_grouped_trace_ras(events, rank_groups);
}

GlobalTrace filter_trace_partitions(
    const GlobalTrace& trace,
    const std::vector<std::uint64_t>& partition_ids) {
    std::unordered_set<std::uint64_t> selected_partitions(partition_ids.begin(), partition_ids.end());
    std::unordered_set<std::uint64_t> selected_events;
    for (const SyncPartition& partition : trace.sync_partitions) {
        if (selected_partitions.count(partition.id) == 0) {
            continue;
        }
        selected_events.insert(partition.event_ids.begin(), partition.event_ids.end());
    }

    GlobalTrace filtered;

    std::unordered_set<std::uint32_t> selected_lanes;
    std::unordered_set<std::uint32_t> selected_dedup_groups;
    for (const GlobalEvent& event : trace.events) {
        if (selected_events.count(event.id) == 0) {
            continue;
        }
        filtered.events.push_back(event);
        selected_lanes.insert(event.lane_id);
        if (event.dedup_group_id != 0) {
            selected_dedup_groups.insert(event.dedup_group_id);
        }
    }
    for (const DedupRankGroup& group : trace.dedup_groups) {
        if (selected_dedup_groups.count(group.id) != 0) {
            filtered.dedup_groups.push_back(group);
        }
    }
    filtered.deduplicated = !filtered.dedup_groups.empty();
    for (const Lane& lane : trace.lanes) {
        if (selected_lanes.count(lane.id) != 0) {
            filtered.lanes.push_back(lane);
        }
    }
    for (const Edge& edge : trace.edges) {
        if (selected_events.count(edge.from) != 0 && selected_events.count(edge.to) != 0) {
            filtered.edges.push_back(edge);
        }
    }
    for (const SyncPartition& partition : trace.sync_partitions) {
        const bool explicit_partition = selected_partitions.count(partition.id) != 0;
        std::vector<std::uint64_t> event_ids;
        for (std::uint64_t event_id : partition.event_ids) {
            if (selected_events.count(event_id) != 0) {
                event_ids.push_back(event_id);
            }
        }
        if (!explicit_partition && event_ids.empty()) {
            continue;
        }
        SyncPartition copy = partition;
        copy.event_ids = std::move(event_ids);
        filtered.sync_partitions.push_back(std::move(copy));
    }
    for (const AnchorLineageEdge& edge : trace.lineage_edges) {
        if (selected_partitions.count(edge.lower_partition) != 0) {
            filtered.lineage_edges.push_back(edge);
        }
    }
    finalize_counts(&filtered);
    return filtered;
}

GlobalTrace patch_trace_code_partitions(
    const GlobalTrace& anchor,
    const GlobalTrace& replacement,
    const std::vector<std::string>& code_partitions) {
    const std::unordered_set<std::string> selected(
        code_partitions.begin(), code_partitions.end());
    if (selected.empty()) {
        throw std::invalid_argument("patch_trace_code_partitions requires a non-empty partition set");
    }

    auto raw_from_global = [](const GlobalEvent& source) {
        RawEvent event;
        event.id = source.raw_id;
        event.rank = source.rank;
        event.thread_id = source.thread_id;
        event.device = source.device;
        event.stream = source.stream;
        event.correlation_id = source.correlation_id;
        event.timestamp_ns = source.timestamp_ns;
        event.duration_hint_us = source.duration_hint_us;
        event.api = source.api;
        event.kind = source.kind;
        event.bytes = source.bytes;
        event.count = source.count;
        event.peer_rank = source.peer_rank;
        event.event_handle = source.event_handle;
        event.wait_event_handle = source.wait_event_handle;
        event.collective_group = source.collective_group;
        event.code_partition = source.code_partition;
        event.blocking = source.blocking;
        return event;
    };
    auto by_rank = [](const GlobalTrace& trace) {
        std::map<int, std::vector<const GlobalEvent*>> result;
        for (const GlobalEvent& event : trace.events) {
            result[event.rank].push_back(&event);
        }
        return result;
    };

    const auto anchor_by_rank = by_rank(anchor);
    const auto replacement_by_rank = by_rank(replacement);
    std::vector<RawEvent> patched;
    patched.reserve(anchor.events.size() + replacement.events.size());

    for (const auto& rank_rows : anchor_by_rank) {
        const int rank = rank_rows.first;
        const auto replacement_iter = replacement_by_rank.find(rank);
        if (replacement_iter == replacement_by_rank.end()) {
            throw std::invalid_argument("replacement trace is missing rank " + std::to_string(rank));
        }

        std::map<std::string, std::vector<std::vector<const GlobalEvent*>>> replacement_chunks;
        const std::vector<const GlobalEvent*>& replacements = replacement_iter->second;
        for (std::size_t index = 0; index < replacements.size();) {
            const std::string label = replacements[index]->code_partition;
            if (selected.count(label) == 0) {
                ++index;
                continue;
            }
            std::vector<const GlobalEvent*> chunk;
            while (index < replacements.size() && replacements[index]->code_partition == label) {
                chunk.push_back(replacements[index++]);
            }
            replacement_chunks[label].push_back(std::move(chunk));
        }
        std::map<std::string, std::size_t> next_chunk;
        std::uint64_t timestamp = 1;
        const std::vector<const GlobalEvent*>& anchors = rank_rows.second;
        for (std::size_t index = 0; index < anchors.size();) {
            const std::string label = anchors[index]->code_partition;
            if (selected.count(label) == 0) {
                RawEvent event = raw_from_global(*anchors[index++]);
                event.timestamp_ns = timestamp++;
                patched.push_back(std::move(event));
                continue;
            }

            std::uint64_t anchor_host_thread = 0;
            while (index < anchors.size() && anchors[index]->code_partition == label) {
                RawEvent event = raw_from_global(*anchors[index++]);
                if (anchor_host_thread == 0 && !is_stream_event(event)) {
                    anchor_host_thread = event.thread_id;
                }
            }
            const std::size_t chunk_index = next_chunk[label]++;
            const auto chunks_iter = replacement_chunks.find(label);
            if (chunks_iter == replacement_chunks.end() || chunk_index >= chunks_iter->second.size()) {
                throw std::invalid_argument(
                    "replacement trace has too few chunks for rank " + std::to_string(rank) +
                    " partition " + label);
            }
            for (const GlobalEvent* source : chunks_iter->second[chunk_index]) {
                RawEvent event = raw_from_global(*source);
                if (anchor_host_thread != 0 && !is_stream_event(event)) {
                    event.thread_id = anchor_host_thread;
                }
                event.timestamp_ns = timestamp++;
                patched.push_back(std::move(event));
            }
        }
        for (const auto& item : replacement_chunks) {
            if (next_chunk[item.first] != item.second.size()) {
                throw std::invalid_argument(
                    "replacement trace has too many chunks for rank " + std::to_string(rank) +
                    " partition " + item.first);
            }
        }
    }
    if (replacement_by_rank.size() != anchor_by_rank.size()) {
        throw std::invalid_argument("anchor and replacement rank sets differ");
    }
    return build_trace_ras(patched);
}

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
    bool blocking) {
    RawEvent event;
    event.api = api;
    event.kind = kind;
    event.rank = rank;
    event.thread_id = thread_id;
    event.device = device;
    event.stream = stream;
    event.correlation_id = correlation_id;
    event.timestamp_ns = timestamp_ns;
    event.duration_hint_us = duration_hint_us;
    event.bytes = bytes;
    event.count = count;
    event.peer_rank = peer_rank;
    event.event_handle = event_handle;
    event.wait_event_handle = wait_event_handle;
    event.collective_group = collective_group;
    event.code_partition = code_partition;
    event.blocking = blocking;
    return HookRecorder::instance().record(event);
}

std::uint64_t record_marker(
    int rank,
    const std::string& marker_kind,
    const std::string& code_partition,
    std::uint64_t timestamp_ns) {
    return record_hook_api(
        marker_kind,
        "host_marker",
        rank,
        0,
        0,
        0,
        0,
        timestamp_ns,
        0.0,
        0,
        0,
        -1,
        0,
        0,
        "",
        code_partition,
        false);
}

}  // namespace flexmaya

namespace {

std::string cstr_or_empty(const char* value) {
    return value == nullptr ? std::string() : std::string(value);
}

}

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
    int blocking) {
    try {
        return flexmaya::record_hook_api(
            cstr_or_empty(api),
            cstr_or_empty(kind),
            rank,
            thread_id,
            device,
            stream,
            correlation_id,
            timestamp_ns,
            duration_hint_us,
            bytes,
            count,
            peer_rank,
            event_handle,
            wait_event_handle,
            cstr_or_empty(collective_group),
            cstr_or_empty(code_partition),
            blocking != 0);
    } catch (...) {
        return 0;
    }
}

std::uint64_t flexmaya_record_marker_v1(
    int rank,
    const char* marker_kind,
    const char* code_partition,
    std::uint64_t timestamp_ns) {
    try {
        return flexmaya::record_marker(rank, cstr_or_empty(marker_kind), cstr_or_empty(code_partition), timestamp_ns);
    } catch (...) {
        return 0;
    }
}

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
    int blocking) {
    return flexmaya_record_api_v1(
        api,
        kind,
        rank,
        thread_id,
        device,
        stream,
        correlation_id,
        timestamp_ns,
        duration_hint_us,
        bytes,
        count,
        peer_rank,
        event_handle,
        wait_event_handle,
        collective_group,
        code_partition,
        blocking);
}

}
