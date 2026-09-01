#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp_event/cluster_state.hpp"
#include "cpp_event/event_context.hpp"
#include "cpp_event/event_graph_builder.hpp"
#include "cpp_event/event_log.hpp"
#include "cpp_event/event_recorder_adapter.hpp"
#include "cpp_event/event_schema.hpp"
#include "cpp_event/event_simulator.hpp"
#include "cpp_event/event_time_assigner.hpp"
#include "cpp_event/graph_serializer.hpp"
#include "cpp_event/graph_validator.hpp"
#include "cpp_event/shared_event_merger.hpp"
#include "cpp_event/spmd_execution_controller.hpp"
#include "cpp_event/timing_model.hpp"
#include "lowlevel/interface/async_runtime_observer.hpp"
#include "lowlevel/interface/cupti_activity_metadata_observer.hpp"
#include "lowlevel/interface/wrapper_template.hpp"

namespace py = pybind11;
using namespace cpp_event;

PYBIND11_MODULE(cpp_event_py, m) {
  m.doc() = "CppEvent Python bindings for event graph generation";

  // EventSchema types
  py::enum_<EventDomain>(m, "EventDomain")
      .value("Unknown", EventDomain::kUnknown)
      .value("Compute", EventDomain::kCompute)
      .value("Communication", EventDomain::kCommunication)
      .value("Synchronization", EventDomain::kSynchronization)
      .value("Runtime", EventDomain::kRuntime)
      .value("Memory", EventDomain::kMemory)
      .value("IO", EventDomain::kIO);

  py::enum_<EventKind>(m, "EventKind")
      .value("Unknown", EventKind::kUnknown)
      .value("ComputeKernel", EventKind::kComputeKernel)
      .value("Collective", EventKind::kCollective)
      .value("PointToPoint", EventKind::kPointToPoint)
      .value("Barrier", EventKind::kBarrier)
      .value("AllReduce", EventKind::kAllReduce)
      .value("Broadcast", EventKind::kBroadcast)
      .value("AllGather", EventKind::kAllGather)
      .value("ReduceScatter", EventKind::kReduceScatter)
      .value("Send", EventKind::kSend)
      .value("Recv", EventKind::kRecv)
      .value("RuntimeCall", EventKind::kRuntimeCall)
      .value("MemcpyHostToDevice", EventKind::kMemcpyHostToDevice)
      .value("MemcpyDeviceToHost", EventKind::kMemcpyDeviceToHost)
      .value("MemcpyDeviceToDevice", EventKind::kMemcpyDeviceToDevice)
      .value("MemoryAllocation", EventKind::kMemoryAllocation)
      .value("MemoryFree", EventKind::kMemoryFree)
      .value("FileRead", EventKind::kFileRead)
      .value("FileWrite", EventKind::kFileWrite)
      .value("Debug", EventKind::kDebug);

  py::enum_<EventScope>(m, "EventScope")
      .value("Local", EventScope::kLocal)
      .value("CrossRank", EventScope::kCrossRank);

  py::class_<RankGroup>(m, "RankGroup")
      .def(py::init<>())
      .def_readwrite("id", &RankGroup::id)
      .def_readwrite("members", &RankGroup::members)
      .def("empty", &RankGroup::empty)
      .def("size", &RankGroup::size)
      .def("contains", &RankGroup::contains);

  py::class_<Placement>(m, "Placement")
      .def(py::init<>())
      .def_readwrite("world_size", &Placement::world_size)
      .def_readwrite("device", &Placement::device)
      .def_readwrite("stream", &Placement::stream)
      .def_readwrite("group", &Placement::group);

  py::class_<EventPayload>(m, "EventPayload")
      .def(py::init<>())
      .def_readwrite("attributes", &EventPayload::attributes);

  py::class_<EventRecord>(m, "EventRecord")
      .def(py::init<>())
      .def_readwrite("id", &EventRecord::id)
      .def_readwrite("domain", &EventRecord::domain)
      .def_readwrite("kind", &EventRecord::kind)
      .def_readwrite("scope", &EventRecord::scope)
      .def_readwrite("process_id", &EventRecord::process_id)
      .def_readwrite("thread_id", &EventRecord::thread_id)
      .def_readwrite("active_group", &EventRecord::active_group)
      .def_readwrite("api_name", &EventRecord::api_name)
      .def_readwrite("placement", &EventRecord::placement)
      .def_readwrite("timestamp", &EventRecord::timestamp)
      .def_readwrite("end_timestamp", &EventRecord::end_timestamp)
      .def_readwrite("host_duration", &EventRecord::host_duration)
      .def_readwrite("payload", &EventRecord::payload)
      .def("node_count_hint", &EventRecord::node_count_hint);

  py::class_<EventEdge>(m, "EventEdge")
      .def(py::init<>())
      .def_readwrite("from", &EventEdge::from)
      .def_readwrite("to", &EventEdge::to)
      .def_readwrite("reason", &EventEdge::reason);

  py::class_<EventGraph>(m, "EventGraph")
      .def(py::init<>())
      .def_readwrite("events", &EventGraph::events)
      .def_readwrite("edges", &EventGraph::edges);

  // EventContext
  py::class_<EventContext>(m, "EventContext")
      .def(py::init<>())
      .def("set_active_group", &EventContext::set_active_group)
      .def("reset_active_group", &EventContext::reset_active_group)
      .def("set_placement", &EventContext::set_placement)
      .def("reset_placement", &EventContext::reset_placement)
      .def("set_scope", &EventContext::set_scope)
      .def("reset_scope", &EventContext::reset_scope)
      .def("snapshot", &EventContext::snapshot);

  // EventLog
  py::class_<EventLog>(m, "EventLog")
      .def(py::init<EventContext &>())
      .def(
          "append",
          py::overload_cast<std::string_view, EventKind, EventLog::Clock::time_point,
                            const EventPayload &>(&EventLog::append),
          py::arg("api_name"),
          py::arg("kind"),
          py::arg("start_time"),
          py::arg("payload") = EventPayload{})
      .def(
          "append_with_end",
          py::overload_cast<std::string_view, EventKind, EventLog::Clock::time_point,
                            EventLog::Clock::time_point, const EventPayload &>(
              &EventLog::append),
          py::arg("api_name"),
          py::arg("kind"),
          py::arg("start_time"),
          py::arg("end_time"),
          py::arg("payload") = EventPayload{})
      .def("snapshot", &EventLog::snapshot)
      .def("clear", &EventLog::clear);

  // EventGraphBuilder
  py::class_<EventGraphBuilder>(m, "EventGraphBuilder")
      .def(py::init<>())
      .def("build_graph", &EventGraphBuilder::build_graph)
      .def("build_program_order_graph",
           &EventGraphBuilder::build_program_order_graph);

  // GraphSerializer
  py::class_<GraphSerializer>(m, "GraphSerializer")
      .def(py::init<>())
      .def("serialize_to_json",
           [](const GraphSerializer &self, const EventGraph &graph) {
             return self.serialize_to_json_string(graph);
           })
      .def("serialize_to_graphml",
           [](const GraphSerializer &self, const EventGraph &graph) {
             return self.serialize_to_graphml_string(graph);
           });

  // GraphValidator
  py::class_<ValidationResult>(m, "ValidationResult")
      .def(py::init<>())
      .def_readwrite("is_valid", &ValidationResult::is_valid)
      .def_readwrite("errors", &ValidationResult::errors)
      .def_readwrite("warnings", &ValidationResult::warnings);

  py::class_<GraphValidator>(m, "GraphValidator")
      .def(py::init<>())
      .def("validate", &GraphValidator::validate)
      .def("has_cycles", &GraphValidator::has_cycles)
      .def("has_invalid_edges", &GraphValidator::has_invalid_edges)
      .def("find_orphaned_events", &GraphValidator::find_orphaned_events);

  // EventTimeAssigner
  py::class_<EventTimeAssigner>(m, "EventTimeAssigner")
      .def(py::init<std::uint64_t>())
      .def("assign_time", &EventTimeAssigner::assign_time)
      .def("assign_time_by_kind", &EventTimeAssigner::assign_time_by_kind);

  // SharedEventMerger
  py::class_<SharedEventMerger>(m, "SharedEventMerger")
      .def(py::init<>())
      .def("merge_shared_events", &SharedEventMerger::merge_shared_events)
      .def_static("is_shared_event", &SharedEventMerger::is_shared_event);

  // SPMDExecutionController
  py::class_<SPMDExecutionController>(m, "SPMDExecutionController")
      .def(py::init<EventContext &>())
      .def("process_event", &SPMDExecutionController::process_event)
      .def("is_rank_ready", &SPMDExecutionController::is_rank_ready)
      .def("get_next_ready_rank", &SPMDExecutionController::get_next_ready_rank)
      .def("build_final_graph", &SPMDExecutionController::build_final_graph);

  // EventLogRecorderAdapter
  py::class_<EventLogRecorderAdapter>(m, "EventLogRecorderAdapter")
      .def(py::init<EventLog &>())
      .def(
          "log",
          [](EventLogRecorderAdapter &self) -> EventLog & {
            return self.log();
          },
          py::return_value_policy::reference_internal);

  // Recorder management functions
  // Note: These require lowlevel-interface_runtime to be linked
  m.def(
      "set_recorder",
      [](EventLogRecorderAdapter *adapter) -> py::object {
        // Set the recorder
        // Note: The adapter must be kept alive by the caller (Python code)
        // The integration module keeps _adapter_ref to prevent garbage
        // collection
        auto *prev = lowlevel::interface::set_recorder(adapter);
        // Return None since we can't convert EventRecorder* to Python
        return py::none();
      },
      py::arg("adapter"),
      "Install an EventLogRecorderAdapter as the process-wide recorder");

  m.def("reset_recorder", &lowlevel::interface::reset_recorder,
        "Reset the process-wide recorder to null");
  m.def(
      "resolve_async_runtime_observations",
      [](EventLog &log) {
        lowlevel::interface::resolve_async_runtime_observations(log);
      },
      py::arg("log"),
      "Resolve deferred async runtime observations into EventLog payloads");
  m.def("clear_async_runtime_observations",
        &lowlevel::interface::clear_async_runtime_observations,
        "Clear deferred async runtime observations and auxiliary handle state");
  m.def(
      "resolve_cupti_activity_metadata_observations",
      [](EventLog &log) {
        lowlevel::interface::resolve_cupti_activity_metadata_observations(log);
      },
      py::arg("log"),
      "Resolve CUPTI activity metadata observations into sidecar-ready payloads");
  m.def("clear_cupti_activity_metadata_observations",
        &lowlevel::interface::clear_cupti_activity_metadata_observations,
        "Clear CUPTI activity metadata observer state");

  m.def(
      "current_recorder",
      []() -> py::object {
        auto *rec = lowlevel::interface::current_recorder();
        if (rec == nullptr) {
          return py::none();
        }
        // We can't safely cast to EventLogRecorderAdapter, so just return None
        // The important thing is that the recorder is set
        return py::none();
      },
      "Get the currently installed recorder (returns None, check is for "
      "internal use)");

  // Device topology types
  py::class_<Cpu>(m, "Cpu")
      .def(py::init<>())
      .def_readwrite("id", &Cpu::id)
      .def_readwrite("numa_node", &Cpu::numa_node);

  py::class_<Nic>(m, "Nic")
      .def(py::init<>())
      .def_readwrite("id", &Nic::id)
      .def_readwrite("model", &Nic::model)
      .def_readwrite("gpu_group", &Nic::gpu_group);

  py::class_<Gpu>(m, "Gpu")
      .def(py::init<>())
      .def_readwrite("id", &Gpu::id)
      .def_readwrite("model", &Gpu::model)
      .def_readwrite("cpu_index", &Gpu::cpu_index)
      .def_readwrite("nic_index", &Gpu::nic_index);

  py::class_<Machine>(m, "Machine")
      .def(py::init<>())
      .def_readwrite("id", &Machine::id)
      .def_readwrite("hostname", &Machine::hostname)
      .def_readwrite("cpus", &Machine::cpus)
      .def_readwrite("nics", &Machine::nics)
      .def_readwrite("gpus", &Machine::gpus);

  py::class_<RankBinding>(m, "RankBinding")
      .def(py::init<>())
      .def_readwrite("rank", &RankBinding::rank)
      .def_readwrite("machine_id", &RankBinding::machine_id)
      .def_readwrite("gpu_index", &RankBinding::gpu_index);

  py::class_<ClusterTopology>(m, "ClusterTopology")
      .def(py::init<>())
      .def_readwrite("machines", &ClusterTopology::machines)
      .def_readwrite("ranks", &ClusterTopology::ranks)
      .def("find_machine", &ClusterTopology::find_machine,
           py::return_value_policy::reference)
      .def("binding_for_rank", &ClusterTopology::binding_for_rank);

  // ClusterState
  py::class_<DeviceState>(m, "DeviceState")
      .def(py::init<>())
      .def_readwrite("device_id", &DeviceState::device_id)
      .def_readwrite("memory_capacity_bytes", &DeviceState::memory_capacity_bytes)
      .def_readwrite("memory_used_bytes", &DeviceState::memory_used_bytes)
      .def_readwrite("is_available", &DeviceState::is_available);

  py::class_<ClusterState>(m, "ClusterState")
      .def(py::init<const ClusterTopology &>())
      .def_static("create_default", &ClusterState::create_default)
      .def("topology", &ClusterState::topology,
           py::return_value_policy::reference_internal)
      .def("world_size", &ClusterState::world_size)
      .def("is_device_available", &ClusterState::is_device_available)
      .def("allocate_memory", &ClusterState::allocate_memory)
      .def("free_memory", &ClusterState::free_memory)
      .def("get_memory_usage", &ClusterState::get_memory_usage)
      .def("get_memory_capacity", &ClusterState::get_memory_capacity)
      .def("estimate_network_latency", &ClusterState::estimate_network_latency)
      .def("are_ranks_on_same_machine", &ClusterState::are_ranks_on_same_machine)
      .def("get_network_bandwidth", &ClusterState::get_network_bandwidth);

  m.def("create_default_topology", &create_default_topology,
        "Create a default single-node topology with specified world size");

  m.def("load_topology_from_json", &load_topology_from_json,
        "Load ClusterTopology from a JSON file");

  // Timing model
  py::class_<HardwareParams>(m, "HardwareParams")
      .def(py::init<>())
      .def_readwrite("peak_fp16_tflops", &HardwareParams::peak_fp16_tflops)
      .def_readwrite("peak_fp32_tflops", &HardwareParams::peak_fp32_tflops)
      .def_readwrite("peak_int8_tops", &HardwareParams::peak_int8_tops)
      .def_readwrite("memory_bandwidth_gbps", &HardwareParams::memory_bandwidth_gbps)
      .def_readwrite("nvlink_bandwidth_gbps", &HardwareParams::nvlink_bandwidth_gbps)
      .def_readwrite("nvlink_latency_us", &HardwareParams::nvlink_latency_us)
      .def_readwrite("network_bandwidth_gbps", &HardwareParams::network_bandwidth_gbps)
      .def_readwrite("network_latency_us", &HardwareParams::network_latency_us)
      .def_readwrite("ring_allreduce_efficiency", &HardwareParams::ring_allreduce_efficiency)
      .def_readwrite("tree_broadcast_efficiency", &HardwareParams::tree_broadcast_efficiency);

  py::class_<TimingModel, std::shared_ptr<TimingModel>>(m, "TimingModel")
      .def("estimate", &TimingModel::estimate)
      .def("name", &TimingModel::name);

  py::class_<SimpleTimingModel, TimingModel, std::shared_ptr<SimpleTimingModel>>(
      m, "SimpleTimingModel")
      .def(py::init<HardwareParams>(), py::arg("params") = HardwareParams{})
      .def("params",
           py::overload_cast<>(&SimpleTimingModel::params),
           py::return_value_policy::reference_internal);

  m.def("create_default_timing_model", &create_default_timing_model,
        "Create a timing model with default A100 parameters");

  // Simulation result types
  py::class_<RankMetrics>(m, "RankMetrics")
      .def(py::init<>())
      .def_readwrite("rank", &RankMetrics::rank)
      .def_readwrite("compute_time", &RankMetrics::compute_time)
      .def_readwrite("communication_time", &RankMetrics::communication_time)
      .def_readwrite("blocked_time", &RankMetrics::blocked_time)
      .def_readwrite("total_time", &RankMetrics::total_time)
      .def_readwrite("num_events", &RankMetrics::num_events)
      .def_readwrite("utilization", &RankMetrics::utilization);

  py::class_<SimulationResult>(m, "SimulationResult")
      .def(py::init<>())
      .def_readwrite("total_time", &SimulationResult::total_time)
      .def_readwrite("critical_path", &SimulationResult::critical_path)
      .def_readwrite("rank_metrics", &SimulationResult::rank_metrics)
      .def_readwrite("final_graph", &SimulationResult::final_graph)
      .def_readwrite("success", &SimulationResult::success)
      .def_readwrite("error_message", &SimulationResult::error_message);

  py::class_<SimulatedEvent>(m, "SimulatedEvent")
      .def(py::init<>())
      .def_readwrite("record", &SimulatedEvent::record)
      .def_readwrite("rank", &SimulatedEvent::rank)
      .def_readwrite("start_time", &SimulatedEvent::start_time)
      .def_readwrite("duration", &SimulatedEvent::duration)
      .def_readwrite("end_time", &SimulatedEvent::end_time);

  // EventSimulator
  py::class_<EventSimulator>(m, "EventSimulator")
      .def(py::init<ClusterTopology, EventContext &>())
      .def_static("create_default",
                  [](std::int32_t world_size, EventContext &ctx) {
                    return new EventSimulator(create_default_topology(world_size), ctx);
                  },
                  py::return_value_policy::take_ownership)
      // set_timing_model takes ownership via std::unique_ptr<TimingModel>;
      // exposing that setter directly is not a valid Python calling
      // convention with current pybind11. The simulator already owns a
      // default timing model, and no Python caller in this repo uses this
      // setter.
      .def("load_events",
           py::overload_cast<const std::string &>(&EventSimulator::load_events))
      .def("load_events",
           py::overload_cast<const EventGraph &>(&EventSimulator::load_events))
      .def("add_event", &EventSimulator::add_event)
      .def("run", &EventSimulator::run)
      .def("get_rank_clock", &EventSimulator::get_rank_clock)
      .def("get_total_time", &EventSimulator::get_total_time)
      .def("get_critical_path", &EventSimulator::get_critical_path)
      .def("get_rank_metrics", &EventSimulator::get_rank_metrics)
      .def("cluster_state", &EventSimulator::cluster_state,
           py::return_value_policy::reference_internal)
      .def("simulated_events", &EventSimulator::simulated_events,
           py::return_value_policy::reference_internal);
}
