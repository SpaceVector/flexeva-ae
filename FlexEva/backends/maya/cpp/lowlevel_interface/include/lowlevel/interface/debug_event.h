// debug_event.h
// Helper functions for accessing TLS in debug operator events
// This is the low-level API that black-box operators should use

#ifndef LOWLEVEL_INTERFACE_DEBUG_EVENT_H
#define LOWLEVEL_INTERFACE_DEBUG_EVENT_H

#include <string>
#include <vector>

// Print active ranks from TLS
void print_active_ranks_from_tls(
    const std::string &prefix = "    -> C++ TLS active_ranks");

#endif // LOWLEVEL_INTERFACE_DEBUG_EVENT_H

