#include "cpp_event/spmd_execution_controller.hpp"

namespace cpp_event {

bool SPMDExecutionController::process_event(RankId rank_id,
                                            const EventRecord &event) {
  // Initialize rank state if needed
  if (state_manager_.get_state(rank_id) == RankExecutionState::kCompleted) {
    state_manager_.initialize_rank(rank_id);
  }

  // Check if rank is blocked
  if (state_manager_.get_state(rank_id) == RankExecutionState::kBlocked) {
    // Try to resolve dependencies
    resolver_.resolve_dependencies(rank_id);

    // Check if still blocked
    if (state_manager_.has_requirements(rank_id) ||
        !is_rank_ready(rank_id)) {
      return false; // Still blocked
    }
    // Unblock
    state_manager_.set_state(rank_id, RankExecutionState::kRunning);
  }

  // Record the event
  record_event(rank_id, event);

  // Update current event
  state_manager_.set_current_event(rank_id, event.id);

  // Check if this event creates a requirement that blocks execution
  if (creates_requirement(event.kind)) {
    register_requirements(rank_id, event);
    // Check if requirements can be satisfied immediately
    if (!is_rank_ready(rank_id)) {
      state_manager_.set_state(rank_id, RankExecutionState::kBlocked);
      return false; // Blocked
    }
  }

  return true; // Continue execution
}

bool SPMDExecutionController::is_rank_ready(RankId rank_id) const {
  if (state_manager_.get_state(rank_id) == RankExecutionState::kCompleted) {
    return false;
  }

  if (state_manager_.get_state(rank_id) == RankExecutionState::kBlocked) {
    // Check if all requirements are satisfied
    if (state_manager_.has_requirements(rank_id)) {
      // Check if requirements can be satisfied
      // This is simplified - in practice, we'd check resolver
      return false;
    }
  }

  return true;
}

RankId SPMDExecutionController::get_next_ready_rank() const {
  auto all_ranks = state_manager_.get_all_ranks();
  for (RankId rank_id : all_ranks) {
    if (is_rank_ready(rank_id)) {
      return rank_id;
    }
  }
  // Return first rank if none are ready (or -1 if no ranks)
  return all_ranks.empty() ? -1 : all_ranks[0];
}

EventGraph SPMDExecutionController::build_final_graph() {
  // Get all events from event log
  auto events = event_log_.snapshot();

  // Build initial graph with program-order edges
  auto graph = graph_builder_.build_program_order_graph(events);

  // Resolve all dependencies and add dependency edges
  auto dep_edges = resolver_.resolve_all_dependencies();
  graph.edges.insert(graph.edges.end(), dep_edges.begin(), dep_edges.end());

  // Handle bidirectional dependencies (collapse nodes)
  auto merge_map = resolver_.detect_bidirectional_dependencies(graph.events);
  // Note: Actual node merging would require more complex graph manipulation
  // For now, we just detect them - full implementation would merge nodes

  return graph;
}

void SPMDExecutionController::record_event(RankId rank_id,
                                           const EventRecord &event) {
  // Events are recorded via EventLog through lowlevel-interface hooks
  // This method handles the dependency tracking side

  // Register productions
  if (creates_production(event.kind)) {
    register_productions(rank_id, event);
  }
}

void SPMDExecutionController::register_requirements(RankId rank_id,
                                                    const EventRecord &event) {
  // Determine which ranks can satisfy this requirement
  if (event.kind == EventKind::kRecv) {
    // Recv needs a send from another rank
    // Determine target rank from event metadata
    // For now, we'll need to determine from active_group or payload
    // This is simplified - actual implementation would parse event details
    if (!event.active_group.empty()) {
      for (RankId other_rank : event.active_group.members) {
        if (other_rank != rank_id) {
          req_dict_.add_requirement(rank_id, other_rank, EventKind::kSend,
                                    event.id);
          state_manager_.add_requirement(rank_id, event.id);
        }
      }
    }
  }
  // Add other event types that create requirements
}

void SPMDExecutionController::register_productions(RankId rank_id,
                                                   const EventRecord &event) {
  if (event.kind == EventKind::kSend) {
    prod_dict_.add_production(rank_id, EventKind::kSend, event.id);
  }
  // Add other event types that create productions
}

bool SPMDExecutionController::creates_requirement(EventKind kind) const {
  switch (kind) {
  case EventKind::kRecv:
    return true;
  case EventKind::kBarrier:
    return true; // Barriers require all ranks
  default:
    return false;
  }
}

bool SPMDExecutionController::creates_production(EventKind kind) const {
  switch (kind) {
  case EventKind::kSend:
    return true;
  case EventKind::kBarrier:
    return true; // Barriers produce synchronization events
  default:
    return false;
  }
}

} // namespace cpp_event
