#include "flexmaya_core.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

flexmaya::RawEvent make_event(
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
    flexmaya::RawEvent event;
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
    return event;
}

py::dict raw_event_dict(const flexmaya::RawEvent& event) {
    py::dict row;
    row["id"] = event.id;
    row["rank"] = event.rank;
    row["thread_id"] = event.thread_id;
    row["device"] = event.device;
    row["stream"] = event.stream;
    row["correlation_id"] = event.correlation_id;
    row["timestamp_ns"] = event.timestamp_ns;
    row["duration_hint_us"] = event.duration_hint_us;
    row["api"] = event.api;
    row["kind"] = event.kind;
    row["bytes"] = event.bytes;
    row["count"] = event.count;
    row["peer_rank"] = event.peer_rank;
    row["event_handle"] = event.event_handle;
    row["wait_event_handle"] = event.wait_event_handle;
    row["collective_group"] = event.collective_group;
    row["code_partition"] = event.code_partition;
    row["blocking"] = event.blocking;
    return row;
}

py::dict global_event_dict(const flexmaya::GlobalEvent& event) {
    py::dict row;
    row["id"] = event.id;
    row["raw_id"] = event.raw_id;
    row["rank"] = event.rank;
    row["thread_id"] = event.thread_id;
    row["device"] = event.device;
    row["stream"] = event.stream;
    row["correlation_id"] = event.correlation_id;
    row["timestamp_ns"] = event.timestamp_ns;
    row["duration_hint_us"] = event.duration_hint_us;
    row["api"] = event.api;
    row["kind"] = event.kind;
    row["bytes"] = event.bytes;
    row["count"] = event.count;
    row["peer_rank"] = event.peer_rank;
    row["event_handle"] = event.event_handle;
    row["wait_event_handle"] = event.wait_event_handle;
    row["collective_group"] = event.collective_group;
    row["code_partition"] = event.code_partition;
    row["blocking"] = event.blocking;
    row["lane_id"] = event.lane_id;
    row["lane_pos"] = event.lane_pos;
    row["dedup_group_id"] = event.dedup_group_id;
    row["dedup_weight"] = event.dedup_weight;
    return row;
}

py::dict trace_dict(const flexmaya::GlobalTrace& trace) {
    py::dict row;
    row["events"] = trace.events;
    row["lanes"] = trace.lanes;
    row["edges"] = trace.edges;
    row["sync_partitions"] = trace.sync_partitions;
    row["dedup_groups"] = trace.dedup_groups;
    row["lineage_edges"] = trace.lineage_edges;
    row["logical_event_count"] = trace.logical_event_count;
    row["deduplicated"] = trace.deduplicated;
    return row;
}

}  // namespace

PYBIND11_MODULE(_flexmaya_ras, m) {
    m.doc() = "FlexMaya RAS C++ hook-memory trace core";

    py::class_<flexmaya::RawEvent>(m, "RawEvent")
        .def(py::init<>())
        .def_readwrite("id", &flexmaya::RawEvent::id)
        .def_readwrite("rank", &flexmaya::RawEvent::rank)
        .def_readwrite("thread_id", &flexmaya::RawEvent::thread_id)
        .def_readwrite("device", &flexmaya::RawEvent::device)
        .def_readwrite("stream", &flexmaya::RawEvent::stream)
        .def_readwrite("correlation_id", &flexmaya::RawEvent::correlation_id)
        .def_readwrite("timestamp_ns", &flexmaya::RawEvent::timestamp_ns)
        .def_readwrite("duration_hint_us", &flexmaya::RawEvent::duration_hint_us)
        .def_readwrite("api", &flexmaya::RawEvent::api)
        .def_readwrite("kind", &flexmaya::RawEvent::kind)
        .def_readwrite("bytes", &flexmaya::RawEvent::bytes)
        .def_readwrite("count", &flexmaya::RawEvent::count)
        .def_readwrite("peer_rank", &flexmaya::RawEvent::peer_rank)
        .def_readwrite("event_handle", &flexmaya::RawEvent::event_handle)
        .def_readwrite("wait_event_handle", &flexmaya::RawEvent::wait_event_handle)
        .def_readwrite("collective_group", &flexmaya::RawEvent::collective_group)
        .def_readwrite("code_partition", &flexmaya::RawEvent::code_partition)
        .def_readwrite("blocking", &flexmaya::RawEvent::blocking)
        .def("to_dict", &raw_event_dict);

    py::class_<flexmaya::Lane>(m, "Lane")
        .def_readonly("id", &flexmaya::Lane::id)
        .def_readonly("rank", &flexmaya::Lane::rank)
        .def_readonly("kind", &flexmaya::Lane::kind)
        .def_readonly("stream", &flexmaya::Lane::stream)
        .def_readonly("thread_id", &flexmaya::Lane::thread_id);

    py::class_<flexmaya::DedupRankGroup>(m, "DedupRankGroup")
        .def_readonly("id", &flexmaya::DedupRankGroup::id)
        .def_readonly("representative_rank", &flexmaya::DedupRankGroup::representative_rank)
        .def_readonly("ranks", &flexmaya::DedupRankGroup::ranks)
        .def_readonly("fingerprint", &flexmaya::DedupRankGroup::fingerprint)
        .def_readonly("representative_event_count", &flexmaya::DedupRankGroup::representative_event_count)
        .def_readonly("logical_event_count", &flexmaya::DedupRankGroup::logical_event_count);

    py::class_<flexmaya::GlobalEvent>(m, "GlobalEvent")
        .def_readonly("id", &flexmaya::GlobalEvent::id)
        .def_readonly("raw_id", &flexmaya::GlobalEvent::raw_id)
        .def_readonly("rank", &flexmaya::GlobalEvent::rank)
        .def_readonly("thread_id", &flexmaya::GlobalEvent::thread_id)
        .def_readonly("device", &flexmaya::GlobalEvent::device)
        .def_readonly("stream", &flexmaya::GlobalEvent::stream)
        .def_readonly("correlation_id", &flexmaya::GlobalEvent::correlation_id)
        .def_readonly("timestamp_ns", &flexmaya::GlobalEvent::timestamp_ns)
        .def_readonly("duration_hint_us", &flexmaya::GlobalEvent::duration_hint_us)
        .def_readonly("api", &flexmaya::GlobalEvent::api)
        .def_readonly("kind", &flexmaya::GlobalEvent::kind)
        .def_readonly("bytes", &flexmaya::GlobalEvent::bytes)
        .def_readonly("count", &flexmaya::GlobalEvent::count)
        .def_readonly("peer_rank", &flexmaya::GlobalEvent::peer_rank)
        .def_readonly("event_handle", &flexmaya::GlobalEvent::event_handle)
        .def_readonly("wait_event_handle", &flexmaya::GlobalEvent::wait_event_handle)
        .def_readonly("collective_group", &flexmaya::GlobalEvent::collective_group)
        .def_readonly("code_partition", &flexmaya::GlobalEvent::code_partition)
        .def_readonly("blocking", &flexmaya::GlobalEvent::blocking)
        .def_readonly("lane_id", &flexmaya::GlobalEvent::lane_id)
        .def_readonly("lane_pos", &flexmaya::GlobalEvent::lane_pos)
        .def_readonly("dedup_group_id", &flexmaya::GlobalEvent::dedup_group_id)
        .def_readonly("dedup_weight", &flexmaya::GlobalEvent::dedup_weight)
        .def("to_dict", &global_event_dict);

    py::class_<flexmaya::Edge>(m, "Edge")
        .def_readonly("from_id", &flexmaya::Edge::from)
        .def_readonly("to_id", &flexmaya::Edge::to)
        .def_readonly("reason", &flexmaya::Edge::reason);

    py::class_<flexmaya::SyncPartition>(m, "SyncPartition")
        .def_readonly("id", &flexmaya::SyncPartition::id)
        .def_readonly("kind", &flexmaya::SyncPartition::kind)
        .def_readonly("key", &flexmaya::SyncPartition::key)
        .def_readonly("code_partition", &flexmaya::SyncPartition::code_partition)
        .def_readonly("event_ids", &flexmaya::SyncPartition::event_ids)
        .def_readonly("logical_event_count", &flexmaya::SyncPartition::logical_event_count);

    py::class_<flexmaya::AnchorLineageEdge>(m, "AnchorLineageEdge")
        .def_readonly("upper_partition", &flexmaya::AnchorLineageEdge::upper_partition)
        .def_readonly("lower_partition", &flexmaya::AnchorLineageEdge::lower_partition)
        .def_readonly("lower_partition_kind", &flexmaya::AnchorLineageEdge::lower_partition_kind)
        .def_readonly("edge_kind", &flexmaya::AnchorLineageEdge::edge_kind);

    py::class_<flexmaya::GlobalTrace>(m, "GlobalTrace")
        .def_readonly("events", &flexmaya::GlobalTrace::events)
        .def_readonly("lanes", &flexmaya::GlobalTrace::lanes)
        .def_readonly("edges", &flexmaya::GlobalTrace::edges)
        .def_readonly("sync_partitions", &flexmaya::GlobalTrace::sync_partitions)
        .def_readonly("dedup_groups", &flexmaya::GlobalTrace::dedup_groups)
        .def_readonly("lineage_edges", &flexmaya::GlobalTrace::lineage_edges)
        .def_readonly("logical_event_count", &flexmaya::GlobalTrace::logical_event_count)
        .def_readonly("deduplicated", &flexmaya::GlobalTrace::deduplicated)
        .def("to_dict", &trace_dict);

    py::class_<flexmaya::EventArena>(m, "EventArena")
        .def(py::init<>())
        .def("append", &flexmaya::EventArena::append)
        .def("clear", &flexmaya::EventArena::clear)
        .def("size", &flexmaya::EventArena::size)
        .def("events", &flexmaya::EventArena::events)
        .def("build_trace_ras", &flexmaya::EventArena::build_trace_ras)
        .def("build_rank_grouped_trace_ras", &flexmaya::EventArena::build_rank_grouped_trace_ras);

    py::class_<flexmaya::SharedEventArena, std::shared_ptr<flexmaya::SharedEventArena>>(m, "SharedEventArena")
        .def_static("create", &flexmaya::SharedEventArena::create,
            py::arg("name"), py::arg("capacity"), py::arg("unlink_existing") = true)
        .def_static("open", &flexmaya::SharedEventArena::open)
        .def("append", &flexmaya::SharedEventArena::append)
        .def("events", &flexmaya::SharedEventArena::events)
        .def("write_binary", &flexmaya::SharedEventArena::write_binary)
        .def("size", &flexmaya::SharedEventArena::size)
        .def("capacity", &flexmaya::SharedEventArena::capacity)
        .def("name", &flexmaya::SharedEventArena::name)
        .def("build_trace_ras", [](const std::shared_ptr<flexmaya::SharedEventArena>& arena) {
            return flexmaya::build_trace_ras(arena->events());
        })
        .def("build_deduplicated_trace_ras", [](const std::shared_ptr<flexmaya::SharedEventArena>& arena) {
            return flexmaya::build_deduplicated_trace_ras(arena->events());
        })
        .def("build_rank_grouped_trace_ras",
            [](const std::shared_ptr<flexmaya::SharedEventArena>& arena,
               const std::map<int, std::vector<int>>& rank_groups) {
                return flexmaya::build_rank_grouped_trace_ras(arena->events(), rank_groups);
            });

    m.def("make_event", &make_event,
        py::arg("api"),
        py::arg("kind"),
        py::arg("rank") = 0,
        py::arg("thread_id") = 0,
        py::arg("device") = 0,
        py::arg("stream") = 0,
        py::arg("correlation_id") = 0,
        py::arg("timestamp_ns") = 0,
        py::arg("duration_hint_us") = 0.0,
        py::arg("bytes") = 0,
        py::arg("count") = 0,
        py::arg("peer_rank") = -1,
        py::arg("event_handle") = 0,
        py::arg("wait_event_handle") = 0,
        py::arg("collective_group") = "",
        py::arg("code_partition") = "",
        py::arg("blocking") = false);
    m.def("build_trace_ras", &flexmaya::build_trace_ras);
    m.def("build_deduplicated_trace_ras", &flexmaya::build_deduplicated_trace_ras);
    m.def("build_rank_grouped_trace_ras", &flexmaya::build_rank_grouped_trace_ras);
    m.def("build_trace_ras_from_binary", &flexmaya::build_trace_ras_from_binary);
    m.def("build_rank_grouped_trace_ras_from_binary", &flexmaya::build_rank_grouped_trace_ras_from_binary);
    m.def("filter_trace_partitions", &flexmaya::filter_trace_partitions);
    m.def("patch_trace_code_partitions", &flexmaya::patch_trace_code_partitions);
    m.def("record_hook_api", &flexmaya::record_hook_api,
        py::arg("api"),
        py::arg("kind"),
        py::arg("rank") = 0,
        py::arg("thread_id") = 0,
        py::arg("device") = 0,
        py::arg("stream") = 0,
        py::arg("correlation_id") = 0,
        py::arg("timestamp_ns") = 0,
        py::arg("duration_hint_us") = 0.0,
        py::arg("bytes") = 0,
        py::arg("count") = 0,
        py::arg("peer_rank") = -1,
        py::arg("event_handle") = 0,
        py::arg("wait_event_handle") = 0,
        py::arg("collective_group") = "",
        py::arg("code_partition") = "",
        py::arg("blocking") = false);
    m.def("record_marker", &flexmaya::record_marker);
    m.def("hook_events", []() { return flexmaya::HookRecorder::instance().events(); });
    m.def("clear_hook_events", []() { flexmaya::HookRecorder::instance().clear_local(); });
}
