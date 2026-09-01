// tls_rank_context.h
// Header for TLS rank context management functions
// Provides thread-local storage for active_ranks context

#ifndef CPP_EVENT_TLS_RANK_CONTEXT_H
#define CPP_EVENT_TLS_RANK_CONTEXT_H

#include <vector>

// Get active_ranks from thread-local storage
std::vector<int> get_active_ranks_from_tls();

// Set active_ranks in thread-local storage
void set_active_ranks_in_tls(const std::vector<int> &active_ranks);

#endif // CPP_EVENT_TLS_RANK_CONTEXT_H
