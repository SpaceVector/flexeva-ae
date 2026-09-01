// tls_rank_context_python.cpp
// Python bindings for TLS rank context management
// Provides Python access to C++ thread-local storage for active_ranks context

#include "cpp_event/tls_rank_context.h"
#include <dlfcn.h>
#include <pthread.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unistd.h>
#include <vector>

namespace py = pybind11;

// Wrapper functions for explicit context propagation (still available via
// Python) These simply call the interposed pthread_create/fork which handle
// context propagation Note: We need to get function pointers to avoid direct
// symbol resolution issues
static int (*pthread_create_ptr)(pthread_t *, const pthread_attr_t *,
                                 void *(*)(void *), void *) = nullptr;
static pid_t (*fork_ptr)() = nullptr;

static void init_function_pointers() {
  if (pthread_create_ptr == nullptr) {
    pthread_create_ptr = reinterpret_cast<int (*)(
        pthread_t *, const pthread_attr_t *, void *(*)(void *), void *)>(
        dlsym(RTLD_DEFAULT, "pthread_create"));
  }
  if (fork_ptr == nullptr) {
    fork_ptr = reinterpret_cast<pid_t (*)()>(dlsym(RTLD_DEFAULT, "fork"));
  }
}

// Wrapper for pthread_create that propagates TLS context
// This calls the interposed pthread_create which handles context propagation
int pthread_create_with_context(pthread_t *thread, const pthread_attr_t *attr,
                                void *(*start_routine)(void *), void *arg) {
  init_function_pointers();
  if (pthread_create_ptr == nullptr) {
    return -1;
  }
  return pthread_create_ptr(thread, attr, start_routine, arg);
}

// Wrapper for fork that propagates TLS context
// This calls the interposed fork which handles context propagation
pid_t fork_with_context() {
  init_function_pointers();
  if (fork_ptr == nullptr) {
    return -1;
  }
  return fork_ptr();
}

// PYBIND11_MODULE creates a Python module
PYBIND11_MODULE(cpp_event_tls, m) {
  m.doc() = "CppEvent C++ thread-local storage for active_ranks context";

  m.def("set_active_ranks_in_tls", &set_active_ranks_in_tls,
        py::arg("active_ranks"),
        "Set active_ranks in C++ thread-local storage");

  m.def("get_active_ranks_from_tls", &get_active_ranks_from_tls,
        "Get active_ranks from C++ thread-local storage");

  m.def("pthread_create_with_context", &pthread_create_with_context,
        py::arg("thread"), py::arg("attr"), py::arg("start_routine"),
        py::arg("arg"),
        "Wrapper for pthread_create that propagates TLS context");

  m.def("fork_with_context", &fork_with_context,
        "Wrapper for fork that propagates TLS context");
}
