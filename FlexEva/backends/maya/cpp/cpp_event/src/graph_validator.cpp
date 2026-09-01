#include "cpp_event/graph_validator.hpp"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cpp_event {

ValidationResult GraphValidator::validate(const EventGraph &graph) const {
  ValidationResult result;

  // Check for invalid edges
  if (has_invalid_edges(graph)) {
    result.add_error("Graph contains edges with invalid event IDs");
  }

  // Check for cycles
  if (has_cycles(graph)) {
    result.add_error("Graph contains cycles (should be a DAG)");
  }

  // Check for orphaned events
  auto orphaned = find_orphaned_events(graph);
  if (!orphaned.empty()) {
    result.add_warning("Graph contains " + std::to_string(orphaned.size()) +
                       " orphaned events (no edges)");
  }

  // Check that all event IDs are unique
  std::unordered_set<EventId> seen_ids;
  for (const auto &event : graph.events) {
    if (seen_ids.find(event.id) != seen_ids.end()) {
      result.add_error("Duplicate event ID: " + std::to_string(event.id));
    }
    seen_ids.insert(event.id);
  }

  return result;
}

bool GraphValidator::has_cycles(const EventGraph &graph) const {
  // Build adjacency list
  std::unordered_map<EventId, std::vector<EventId>> adjacency_list;
  for (const auto &edge : graph.edges) {
    adjacency_list[edge.from].push_back(edge.to);
  }

  // Build set of all nodes
  std::unordered_set<EventId> all_nodes;
  for (const auto &event : graph.events) {
    all_nodes.insert(event.id);
  }

  // DFS for cycle detection
  std::unordered_map<EventId, int> visited; // 0 = white, 1 = gray, 2 = black
  for (EventId node_id : all_nodes) {
    if (visited[node_id] == 0) {
      if (dfs_cycle_detection(node_id, adjacency_list, visited)) {
        return true;
      }
    }
  }

  return false;
}

bool GraphValidator::dfs_cycle_detection(
    EventId node_id,
    const std::unordered_map<EventId, std::vector<EventId>> &adjacency_list,
    std::unordered_map<EventId, int> &visited) const {
  visited[node_id] = 1; // Mark as gray (in current path)

  auto it = adjacency_list.find(node_id);
  if (it != adjacency_list.end()) {
    for (EventId neighbor : it->second) {
      if (visited[neighbor] == 1) {
        return true; // Cycle detected (back edge to gray node)
      }
      if (visited[neighbor] == 0) {
        if (dfs_cycle_detection(neighbor, adjacency_list, visited)) {
          return true;
        }
      }
    }
  }

  visited[node_id] = 2; // Mark as black (fully processed)
  return false;
}

bool GraphValidator::has_invalid_edges(const EventGraph &graph) const {
  // Build set of valid event IDs
  std::unordered_set<EventId> valid_ids;
  for (const auto &event : graph.events) {
    valid_ids.insert(event.id);
  }

  // Check all edges
  for (const auto &edge : graph.edges) {
    if (valid_ids.find(edge.from) == valid_ids.end() ||
        valid_ids.find(edge.to) == valid_ids.end()) {
      return true;
    }
  }

  return false;
}

std::vector<EventId>
GraphValidator::find_orphaned_events(const EventGraph &graph) const {
  std::unordered_set<EventId> connected_nodes;

  // Find all nodes connected by edges
  for (const auto &edge : graph.edges) {
    connected_nodes.insert(edge.from);
    connected_nodes.insert(edge.to);
  }

  // Find events not in connected set
  std::vector<EventId> orphaned;
  for (const auto &event : graph.events) {
    if (connected_nodes.find(event.id) == connected_nodes.end()) {
      orphaned.push_back(event.id);
    }
  }

  return orphaned;
}

} // namespace cpp_event
