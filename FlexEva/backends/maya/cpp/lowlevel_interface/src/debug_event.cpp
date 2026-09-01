// debug_event.cpp
// Helper functions for accessing TLS in debug operator events
// This is the low-level API that black-box operators should use

#include "lowlevel/interface/debug_event.h"
#include "cpp_event/tls_rank_context.h"
#include "cpp_event/event_schema.hpp"
#include "lowlevel/interface/wrapper_template.hpp"
#include <iostream>
#include <vector>

// Print active ranks from TLS
void print_active_ranks_from_tls(const std::string &prefix) {
  // Record debug event
  auto &rec = lowlevel::interface::recorder();
  auto start_time = lowlevel::interface::EventRecorder::Clock::now();
  rec.record_event("print_active_ranks_from_tls", cpp_event::EventKind::kDebug,
                   start_time);

  // Directly access active_ranks from TLS
  std::vector<int> ranks = get_active_ranks_from_tls();

  if (!ranks.empty()) {
    std::cout << prefix << ": [";
    for (size_t i = 0; i < ranks.size(); ++i) {
      std::cout << ranks[i];
      if (i < ranks.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  } else {
    std::cout << prefix << ": (empty)" << std::endl;
  }
}

// Note: get_current_tls_context() and set_tls_context() have been removed.
// TLS context is now automatically propagated by interposed pthread_create()
// and fork() functions in tls_rank_context_impl.cpp. Black-box code should not need to
// manage TLS context explicitly.

