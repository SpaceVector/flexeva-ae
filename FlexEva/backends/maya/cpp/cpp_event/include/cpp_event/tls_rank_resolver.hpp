#pragma once

#include "cpp_event/event_schema.hpp"

namespace cpp_event {

/// Resolves the active rank from Thread Local Storage (TLS).
/// Integrates with bindlayer's TLS implementation to determine current rank context.
class TLSRankResolver final {
public:
  TLSRankResolver() = default;

  /// Get the current active rank from TLS.
  /// Returns -1 if no rank is set in TLS.
  [[nodiscard]] RankId get_current_rank() const;

  /// Get the current active rank group from TLS.
  /// Returns empty group if none is set.
  [[nodiscard]] RankGroup get_current_rank_group() const;

  /// Check if TLS has rank information.
  [[nodiscard]] bool has_rank_info() const;

private:
  /// Internal helper to read from TLS.
  /// This will need to be implemented via bindlayer integration.
  [[nodiscard]] RankId read_rank_from_tls() const;
};

} // namespace cpp_event
