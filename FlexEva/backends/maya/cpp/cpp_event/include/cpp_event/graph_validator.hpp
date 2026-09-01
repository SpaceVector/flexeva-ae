#pragma once

#include "cpp_event/event_schema.hpp"

#include <string>
#include <vector>

namespace cpp_event {

/// Validation result for graph consistency checks.
struct ValidationResult final {
  bool is_valid{true};
  std::vector<std::string> errors{};
  std::vector<std::string> warnings{};

  void add_error(const std::string &error) {
    is_valid = false;
    errors.push_back(error);
  }

  void add_warning(const std::string &warning) {
    warnings.push_back(warning);
  }
};

/// Validates event graph consistency and detects issues.
class GraphValidator final {
public:
  GraphValidator() = default;

  /// Validate the graph structure.
  [[nodiscard]] ValidationResult validate(const EventGraph &graph) const;

  /// Check for cycles in the graph (should be a DAG).
  [[nodiscard]] bool has_cycles(const EventGraph &graph) const;

  /// Check if all edges reference valid event IDs.
  [[nodiscard]] bool has_invalid_edges(const EventGraph &graph) const;

  /// Check for orphaned events (events with no edges).
  [[nodiscard]] std::vector<EventId>
  find_orphaned_events(const EventGraph &graph) const;

private:
  /// DFS helper for cycle detection.
  bool dfs_cycle_detection(EventId node_id,
                           const std::unordered_map<EventId, std::vector<EventId>> &adjacency_list,
                           std::unordered_map<EventId, int> &visited) const;
};

} // namespace cpp_event
