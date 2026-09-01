#include "cpp_event/tls_rank_resolver.hpp"
#include "cpp_event/tls_rank_context.h"

namespace cpp_event {

RankId TLSRankResolver::get_current_rank() const {
  RankGroup group = get_current_rank_group();
  if (!group.empty() && !group.members.empty()) {
    return group.members[0]; // Return first active rank
  }
  return read_rank_from_tls();
}

RankGroup TLSRankResolver::get_current_rank_group() const {
  RankGroup group;
  auto active_ranks = get_active_ranks_from_tls();
  if (!active_ranks.empty()) {
    group.members.reserve(active_ranks.size());
    for (int rank : active_ranks) {
      group.members.push_back(static_cast<RankId>(rank));
    }
    group.id = "tls_group"; // Default group ID
  }
  return group;
}

bool TLSRankResolver::has_rank_info() const {
  auto active_ranks = get_active_ranks_from_tls();
  return !active_ranks.empty();
}

RankId TLSRankResolver::read_rank_from_tls() const {
  auto active_ranks = get_active_ranks_from_tls();
  if (!active_ranks.empty()) {
    return static_cast<RankId>(active_ranks[0]);
  }
  return -1;
}

} // namespace cpp_event
