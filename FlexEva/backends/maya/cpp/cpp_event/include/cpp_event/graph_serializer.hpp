#pragma once

#include "cpp_event/event_schema.hpp"

#include <ostream>
#include <string>

namespace cpp_event {

/// Serializes an event graph to various formats.
class GraphSerializer final {
public:
  GraphSerializer() = default;

  /// Serialize graph to JSON format.
  void serialize_to_json(const EventGraph &graph, std::ostream &os) const;

  /// Serialize graph to JSON string.
  [[nodiscard]] std::string serialize_to_json_string(const EventGraph &graph) const;

  /// Serialize graph to GraphML format.
  void serialize_to_graphml(const EventGraph &graph, std::ostream &os) const;

  /// Serialize graph to GraphML string.
  [[nodiscard]] std::string serialize_to_graphml_string(const EventGraph &graph) const;

private:
  /// Helper to escape JSON strings.
  [[nodiscard]] static std::string escape_json(const std::string &str);

  /// Convert EventDomain to string.
  [[nodiscard]] static std::string domain_to_string(EventDomain domain);

  /// Convert EventKind to string.
  [[nodiscard]] static std::string kind_to_string(EventKind kind);

  /// Convert EventScope to string.
  [[nodiscard]] static std::string scope_to_string(EventScope scope);
};

} // namespace cpp_event
