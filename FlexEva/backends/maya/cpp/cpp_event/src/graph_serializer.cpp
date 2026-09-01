#include "cpp_event/graph_serializer.hpp"

#include <sstream>
#include <unordered_map>

namespace cpp_event {

void GraphSerializer::serialize_to_json(const EventGraph &graph,
                                        std::ostream &os) const {
  os << "{\n";
  os << "  \"events\": [\n";
  for (size_t i = 0; i < graph.events.size(); ++i) {
    const auto &event = graph.events[i];
    os << "    {\n";
    os << "      \"id\": " << event.id << ",\n";
    os << "      \"domain\": \"" << domain_to_string(event.domain) << "\",\n";
    os << "      \"kind\": \"" << kind_to_string(event.kind) << "\",\n";
    os << "      \"scope\": \"" << scope_to_string(event.scope) << "\",\n";
    os << "      \"api_name\": \"" << escape_json(event.api_name) << "\",\n";
    os << "      \"timestamp\": " << event.timestamp.count() << ",\n";
    os << "      \"placement\": {\n";
    os << "        \"world_size\": " << event.placement.world_size << ",\n";
    os << "        \"device\": \"" << escape_json(event.placement.device)
       << "\",\n";
    os << "        \"stream\": \"" << escape_json(event.placement.stream)
       << "\",\n";
    os << "        \"group\": \"" << escape_json(event.placement.group)
       << "\"\n";
    os << "      },\n";
    os << "      \"active_group\": {\n";
    os << "        \"id\": \"" << escape_json(event.active_group.id) << "\",\n";
    os << "        \"members\": [";
    for (size_t j = 0; j < event.active_group.members.size(); ++j) {
      os << event.active_group.members[j];
      if (j < event.active_group.members.size() - 1) {
        os << ", ";
      }
    }
    os << "]\n";
    os << "      }\n";
    os << "    }";
    if (i < graph.events.size() - 1) {
      os << ",";
    }
    os << "\n";
  }
  os << "  ],\n";
  os << "  \"edges\": [\n";
  for (size_t i = 0; i < graph.edges.size(); ++i) {
    const auto &edge = graph.edges[i];
    os << "    {\n";
    os << "      \"from\": " << edge.from << ",\n";
    os << "      \"to\": " << edge.to << ",\n";
    os << "      \"reason\": \"" << escape_json(edge.reason) << "\"\n";
    os << "    }";
    if (i < graph.edges.size() - 1) {
      os << ",";
    }
    os << "\n";
  }
  os << "  ]\n";
  os << "}\n";
}

std::string GraphSerializer::serialize_to_json_string(const EventGraph &graph) const {
  std::ostringstream oss;
  serialize_to_json(graph, oss);
  return oss.str();
}

void GraphSerializer::serialize_to_graphml(const EventGraph &graph,
                                           std::ostream &os) const {
  os << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  os << "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\"\n";
  os << "         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n";
  os << "         xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns "
        "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n";
  os << "  <key id=\"domain\" for=\"node\" attr.name=\"domain\" attr.type=\"string\"/>\n";
  os << "  <key id=\"kind\" for=\"node\" attr.name=\"kind\" attr.type=\"string\"/>\n";
  os << "  <key id=\"api_name\" for=\"node\" attr.name=\"api_name\" attr.type=\"string\"/>\n";
  os << "  <key id=\"reason\" for=\"edge\" attr.name=\"reason\" attr.type=\"string\"/>\n";
  os << "  <graph id=\"event_graph\" edgedefault=\"directed\">\n";

  // Add nodes
  for (const auto &event : graph.events) {
    os << "    <node id=\"n" << event.id << "\">\n";
    os << "      <data key=\"domain\">" << domain_to_string(event.domain) << "</data>\n";
    os << "      <data key=\"kind\">" << kind_to_string(event.kind) << "</data>\n";
    os << "      <data key=\"api_name\">" << escape_json(event.api_name) << "</data>\n";
    os << "    </node>\n";
  }

  // Add edges
  for (const auto &edge : graph.edges) {
    os << "    <edge source=\"n" << edge.from << "\" target=\"n" << edge.to << "\">\n";
    os << "      <data key=\"reason\">" << escape_json(edge.reason) << "</data>\n";
    os << "    </edge>\n";
  }

  os << "  </graph>\n";
  os << "</graphml>\n";
}

std::string GraphSerializer::serialize_to_graphml_string(const EventGraph &graph) const {
  std::ostringstream oss;
  serialize_to_graphml(graph, oss);
  return oss.str();
}

std::string GraphSerializer::escape_json(const std::string &str) {
  std::string result;
  result.reserve(str.size() * 2);
  for (char c : str) {
    switch (c) {
    case '"':
      result += "\\\"";
      break;
    case '\\':
      result += "\\\\";
      break;
    case '\b':
      result += "\\b";
      break;
    case '\f':
      result += "\\f";
      break;
    case '\n':
      result += "\\n";
      break;
    case '\r':
      result += "\\r";
      break;
    case '\t':
      result += "\\t";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        result += "\\u";
        result += "0000";
        char hex[3];
        snprintf(hex, sizeof(hex), "%02x", static_cast<unsigned char>(c));
        result += hex;
      } else {
        result += c;
      }
      break;
    }
  }
  return result;
}

std::string GraphSerializer::domain_to_string(EventDomain domain) {
  switch (domain) {
  case EventDomain::kUnknown:
    return "Unknown";
  case EventDomain::kCompute:
    return "Compute";
  case EventDomain::kCommunication:
    return "Communication";
  case EventDomain::kSynchronization:
    return "Synchronization";
  case EventDomain::kRuntime:
    return "Runtime";
  case EventDomain::kMemory:
    return "Memory";
  case EventDomain::kIO:
    return "IO";
  default:
    return "Unknown";
  }
}

std::string GraphSerializer::kind_to_string(EventKind kind) {
  switch (kind) {
  case EventKind::kUnknown:
    return "Unknown";
  case EventKind::kComputeKernel:
    return "ComputeKernel";
  case EventKind::kCollective:
    return "Collective";
  case EventKind::kPointToPoint:
    return "PointToPoint";
  case EventKind::kBarrier:
    return "Barrier";
  case EventKind::kAllReduce:
    return "AllReduce";
  case EventKind::kBroadcast:
    return "Broadcast";
  case EventKind::kAllGather:
    return "AllGather";
  case EventKind::kReduceScatter:
    return "ReduceScatter";
  case EventKind::kSend:
    return "Send";
  case EventKind::kRecv:
    return "Recv";
  case EventKind::kRuntimeCall:
    return "RuntimeCall";
  case EventKind::kMemcpyHostToDevice:
    return "MemcpyHostToDevice";
  case EventKind::kMemcpyDeviceToHost:
    return "MemcpyDeviceToHost";
  case EventKind::kMemcpyDeviceToDevice:
    return "MemcpyDeviceToDevice";
  case EventKind::kMemoryAllocation:
    return "MemoryAllocation";
  case EventKind::kMemoryFree:
    return "MemoryFree";
  case EventKind::kFileRead:
    return "FileRead";
  case EventKind::kFileWrite:
    return "FileWrite";
  default:
    return "Unknown";
  }
}

std::string GraphSerializer::scope_to_string(EventScope scope) {
  switch (scope) {
  case EventScope::kLocal:
    return "Local";
  case EventScope::kCrossRank:
    return "CrossRank";
  default:
    return "Local";
  }
}

} // namespace cpp_event
