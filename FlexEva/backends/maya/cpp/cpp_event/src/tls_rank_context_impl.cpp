// tls_rank_context_impl.cpp
// Implementation of TLS functions without pybind11 dependencies
// Also contains interposed pthread_create and fork functions for automatic TLS
// propagation

#include "cpp_event/tls_rank_context.h"
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#include <vector>

// Thread-local storage for active_ranks context
thread_local std::vector<int> tls_active_ranks;

// Set active_ranks in thread-local storage
void set_active_ranks_in_tls(const std::vector<int> &active_ranks) {
  tls_active_ranks = active_ranks;
}

// Get active_ranks from thread-local storage
std::vector<int> get_active_ranks_from_tls() { return tls_active_ranks; }

// Structure to pass context and original function to new thread
struct ThreadContext {
  std::vector<int> context_snapshot;
  void *(*original_start_routine)(void *);
  void *original_arg;
};

// Wrapper function for thread start routine that sets TLS context
static void *thread_wrapper(void *arg) {
  ThreadContext *ctx = static_cast<ThreadContext *>(arg);

  // Set the context snapshot in the new thread's TLS
  set_active_ranks_in_tls(ctx->context_snapshot);

  // Call the original start routine
  void *result = ctx->original_start_routine(ctx->original_arg);

  // Clean up
  delete ctx;
  return result;
}

// Get original pthread_create function
static int (*original_pthread_create)(pthread_t *, const pthread_attr_t *,
                                      void *(*)(void *), void *) = nullptr;

// Initialize original function pointer
static void init_pthread_create() {
  if (original_pthread_create == nullptr) {
    original_pthread_create = reinterpret_cast<int (*)(
        pthread_t *, const pthread_attr_t *, void *(*)(void *), void *)>(
        dlsym(RTLD_NEXT, "pthread_create"));
    if (original_pthread_create == nullptr) {
      // Fallback: try to get from libpthread
      void *handle = dlopen("libpthread.so.0", RTLD_LAZY);
      if (handle) {
        original_pthread_create = reinterpret_cast<int (*)(
            pthread_t *, const pthread_attr_t *, void *(*)(void *), void *)>(
            dlsym(handle, "pthread_create"));
      }
    }
  }
}

// Interposed pthread_create that automatically propagates context
// This function will be called instead of the system pthread_create
// when this shared library is loaded
extern "C" int pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                              void *(*start_routine)(void *), void *arg) {
  init_pthread_create();

  if (original_pthread_create == nullptr) {
    // If we can't find the original, this shouldn't happen, but handle it
    return -1;
  }

  // Always capture and propagate TLS context snapshot (even if empty)
  // This ensures consistent behavior and allows context to be set in new
  // threads
  std::vector<int> context_snapshot = get_active_ranks_from_tls();

  // Debug: Verify interposition is working
  // Uncomment to verify interposed function is being called:
  // fprintf(stderr, "[INTERPOSED pthread_create] Context snapshot size: %zu\n",
  // context_snapshot.size());

  // Always wrap the thread to propagate context
  ThreadContext *ctx = new ThreadContext{context_snapshot, start_routine, arg};
  return original_pthread_create(thread, attr, thread_wrapper, ctx);
}

// Get original fork function
static pid_t (*original_fork)() = nullptr;

// Initialize original fork function pointer
static void init_fork() {
  if (original_fork == nullptr) {
    original_fork = reinterpret_cast<pid_t (*)()>(dlsym(RTLD_NEXT, "fork"));
  }
}

// Interposed fork that automatically propagates context
extern "C" pid_t fork() {
  init_fork();

  if (original_fork == nullptr) {
    return -1; // Error
  }

  // Always capture TLS context snapshot before forking
  std::vector<int> context_snapshot = get_active_ranks_from_tls();

  pid_t pid = original_fork();

  if (pid == 0) {
    // Child process: set the context snapshot in TLS
    set_active_ranks_in_tls(context_snapshot);
  }
  // Parent process: TLS already has the context, no change needed

  return pid;
}
