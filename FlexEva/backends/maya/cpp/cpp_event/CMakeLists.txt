cmake_minimum_required(VERSION 3.16)
project(CppEvent LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(CPPEVENT_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(CPPEVENT_LEGACY_OUTPUT_DIR "${CPPEVENT_REPO_ROOT}/CppEvent")
set(CPPEVENT_LEGACY_BUILD_DIR "${CPPEVENT_REPO_ROOT}/build/CppEvent")
set(CPPEVENT_PYTHON_OUTPUT_DIR "${CPPEVENT_REPO_ROOT}/python")

function(cppevent_set_legacy_build_output target_name)
  set_target_properties(${target_name} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CPPEVENT_LEGACY_BUILD_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CPPEVENT_LEGACY_BUILD_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CPPEVENT_LEGACY_BUILD_DIR}"
  )
endfunction()

# Handle test building
if(DEFINED COMPOSE_ENABLE_TESTS)
  set(CPPEVENT_BUILD_TESTS ${COMPOSE_ENABLE_TESTS} CACHE BOOL "Build CppEvent tests")
else()
  option(CPPEVENT_BUILD_TESTS "Build CppEvent tests" OFF)
endif()

# Handle Python bindings (optional, defaults to OFF if not defined)
if(DEFINED COMPOSE_ENABLE_PYTHON)
  set(CPPEVENT_BUILD_PYTHON ${COMPOSE_ENABLE_PYTHON} CACHE BOOL "Build CppEvent Python bindings")
else()
  option(CPPEVENT_BUILD_PYTHON "Build CppEvent Python bindings" OFF)
endif()
option(CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY
       "Compile optional real CUPTI activity metadata collection support" OFF)

# Convert from INTERFACE to STATIC library since we have source files now
add_library(cpp_event STATIC
  src/event_graph_builder.cpp
  src/rank_state.cpp
  src/requirement_dict.cpp
  src/producing_dict.cpp
  src/dependency_resolver.cpp
  src/spmd_execution_controller.cpp
  src/shared_event_merger.cpp
  src/thread_event_tracker.cpp
  src/tls_rank_resolver.cpp
  src/graph_serializer.cpp
  src/graph_validator.cpp
  src/event_time_assigner.cpp
  src/cluster_state.cpp
  src/timing_model.cpp
  src/event_simulator.cpp
)

target_include_directories(cpp_event PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Add -fPIC for static library to allow linking into shared libraries
set_target_properties(cpp_event PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

# Thread library for mutex support
find_package(Threads REQUIRED)
target_link_libraries(cpp_event PUBLIC Threads::Threads)

# TLS rank context shared library
# This shared library provides thread-local storage for active_ranks context
# and interposed pthread_create/fork functions for automatic context propagation
add_library(cpp_event_tls_shared SHARED src/tls_rank_context_impl.cpp)

target_include_directories(cpp_event_tls_shared PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Link pthread and dl for interposition functions
target_link_libraries(cpp_event_tls_shared PRIVATE Threads::Threads)
find_library(DL_LIBRARY dl)
if(DL_LIBRARY)
  target_link_libraries(cpp_event_tls_shared PRIVATE ${DL_LIBRARY})
endif()

# Set output directory so it can be found by other modules
set_target_properties(cpp_event_tls_shared PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CPPEVENT_LEGACY_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CPPEVENT_LEGACY_OUTPUT_DIR}"
  OUTPUT_NAME "cpp_event_tls_shared"
)

# Link cpp_event against cpp_event_tls_shared so TLS functions are available
# This is needed because tls_rank_resolver.cpp calls get_active_ranks_from_tls()
target_link_libraries(cpp_event PUBLIC cpp_event_tls_shared)

# Recorder runtime shared library
# This shared library provides the global recorder instance that is shared
# across all Python modules. The recorder implementation has been moved from
# lowlevel-interface to CppEvent where it belongs (records to EventLog).
# All modules that need recorder functions must link against this shared library.
add_library(cpp_event_runtime_shared SHARED
  src/recorder.cpp
  src/async_runtime_observer.cpp
  src/cupti_activity_metadata_observer.cpp
)

target_compile_features(cpp_event_runtime_shared PUBLIC cxx_std_20)

target_include_directories(cpp_event_runtime_shared PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Need access to lowlevel-interface headers for EventRecorder class definition
# Since CppEvent is added as subdirectory before lowlevel-interface, we can
# reference the source directory directly. Headers only need to exist at compile time.
target_include_directories(cpp_event_runtime_shared PRIVATE
  "${CPPEVENT_REPO_ROOT}/cpp/lowlevel_interface/include"
)

if(CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY)
  find_path(CPPEVENT_CUPTI_INCLUDE_DIR
            NAMES cupti.h
            HINTS ENV CUDA_HOME ENV CUDA_PATH
            PATH_SUFFIXES include extras/CUPTI/include
            PATHS /usr/local/cuda/include /usr/local/cuda/extras/CUPTI/include)
  find_library(CPPEVENT_CUPTI_LIBRARY
               NAMES cupti
               HINTS ENV CUDA_HOME ENV CUDA_PATH
               PATH_SUFFIXES lib64 extras/CUPTI/lib64
               PATHS /usr/local/cuda/lib64 /usr/local/cuda/extras/CUPTI/lib64)
  if(NOT CPPEVENT_CUPTI_INCLUDE_DIR OR NOT CPPEVENT_CUPTI_LIBRARY)
    message(FATAL_ERROR
            "CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY=ON requires cupti.h and libcupti")
  endif()
  target_compile_definitions(cpp_event_runtime_shared
                             PRIVATE CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY=1)
  target_include_directories(cpp_event_runtime_shared
                             PRIVATE ${CPPEVENT_CUPTI_INCLUDE_DIR})
  target_link_libraries(cpp_event_runtime_shared PRIVATE ${CPPEVENT_CUPTI_LIBRARY})
endif()

# Set output directory so it can be found by other modules
set_target_properties(cpp_event_runtime_shared PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CPPEVENT_LEGACY_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CPPEVENT_LEGACY_OUTPUT_DIR}"
  OUTPUT_NAME "cpp_event_runtime_shared"
)

if(CPPEVENT_BUILD_TESTS)
  enable_testing()

  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  # Ground-truth tests (use profiler to generate event graphs for human verification)
  add_executable(cpp_event_groundtruth_simple_sequential
    ${CMAKE_SOURCE_DIR}/tests/cpp_event/test_groundtruth_simple_sequential.cpp
  )
  target_link_libraries(cpp_event_groundtruth_simple_sequential
    PRIVATE
      cpp_event
      GTest::gtest_main
  )
  cppevent_set_legacy_build_output(cpp_event_groundtruth_simple_sequential)
  add_test(
    NAME cpp_event_groundtruth_simple_sequential
    COMMAND cpp_event_groundtruth_simple_sequential
  )

  add_executable(cpp_event_groundtruth_collective
    ${CMAKE_SOURCE_DIR}/tests/cpp_event/test_groundtruth_collective.cpp
  )
  target_link_libraries(cpp_event_groundtruth_collective
    PRIVATE
      cpp_event
      GTest::gtest_main
  )
  cppevent_set_legacy_build_output(cpp_event_groundtruth_collective)
  add_test(
    NAME cpp_event_groundtruth_collective
    COMMAND cpp_event_groundtruth_collective
  )

  add_executable(cpp_event_groundtruth_debug_events
    ${CMAKE_SOURCE_DIR}/tests/cpp_event/test_groundtruth_debug_events.cpp
  )
  target_link_libraries(cpp_event_groundtruth_debug_events
    PRIVATE
      cpp_event
      GTest::gtest_main
  )
  cppevent_set_legacy_build_output(cpp_event_groundtruth_debug_events)
  add_test(
    NAME cpp_event_groundtruth_debug_events
    COMMAND cpp_event_groundtruth_debug_events
  )
endif()

# Python bindings (optional)
if(CPPEVENT_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
  if(NOT Python3_Interpreter_FOUND)
    if(DEFINED Python3_EXECUTABLE AND EXISTS "${Python3_EXECUTABLE}")
      set(Python3_Interpreter_FOUND TRUE)
    else()
      message(FATAL_ERROR "Python3 interpreter not found. Set Python3_EXECUTABLE.")
    endif()
  endif()
  if(NOT Python3_Development.Module_FOUND)
    if(DEFINED Python3_INCLUDE_DIR AND EXISTS "${Python3_INCLUDE_DIR}/Python.h")
      set(Python3_INCLUDE_DIRS "${Python3_INCLUDE_DIR}")
    elseif(DEFINED Python3_INCLUDE_DIRS)
      # keep caller-provided include dirs
    else()
      message(FATAL_ERROR "Python3 Development.Module not found. Set Python3_INCLUDE_DIR.")
    endif()
    if(DEFINED Python3_LIBRARY AND EXISTS "${Python3_LIBRARY}")
      set(Python3_LIBRARIES "${Python3_LIBRARY}")
    elseif(DEFINED Python3_LIBRARIES)
      # keep caller-provided libraries
    else()
      message(FATAL_ERROR "Python3 Development.Module not found. Set Python3_LIBRARY.")
    endif()
    set(Python3_Development.Module_FOUND TRUE)
  endif()

  # Get pybind11 cmake path from the Python environment
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE _PYBIND11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _PYBIND11_RESULT
    ERROR_QUIET
  )

  if(_PYBIND11_RESULT EQUAL 0 AND EXISTS "${_PYBIND11_DIR}/pybind11Config.cmake")
    set(pybind11_DIR "${_PYBIND11_DIR}" CACHE PATH "pybind11 CMake dir" FORCE)
    set(PYBIND11_FINDPYTHON ON CACHE BOOL "Use FindPython mode in pybind11" FORCE)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(cpp_event_py bindings/cpp_event_python.cpp)
    target_link_libraries(cpp_event_py PRIVATE cpp_event pybind11::module)

    # Add TLS Python module
    pybind11_add_module(cpp_event_tls bindings/tls_rank_context_python.cpp)
    target_link_libraries(cpp_event_tls PRIVATE cpp_event_tls_shared pybind11::module)
    target_include_directories(cpp_event_tls PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )

    # Set output directory for TLS module
    set_target_properties(cpp_event_tls PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CPPEVENT_PYTHON_OUTPUT_DIR}"
      RUNTIME_OUTPUT_DIRECTORY "${CPPEVENT_PYTHON_OUTPUT_DIR}"
      BUILD_RPATH "${CPPEVENT_LEGACY_OUTPUT_DIR}"
      INSTALL_RPATH "\$ORIGIN/../CppEvent"
    )

    # Link against cpp_event_runtime_shared for recorder management functions
    # The recorder implementation has been moved to CppEvent as a shared library
    # so all Python modules share the same recorder instance.
    if(TARGET cpp_event_runtime_shared)
      target_link_libraries(cpp_event_py PRIVATE cpp_event_runtime_shared)
      message(STATUS "Linked cpp_event_runtime_shared to cpp_event_py")
    else()
      message(WARNING "cpp_event_runtime_shared not found. Recorder functions may not work.")
    endif()

    # Also need lowlevel-interface headers for EventRecorder class definition
    if(EXISTS "${CPPEVENT_REPO_ROOT}/cpp/lowlevel_interface/include")
      target_include_directories(cpp_event_py PRIVATE
        "${CPPEVENT_REPO_ROOT}/cpp/lowlevel_interface/include"
      )
    endif()

    # Set output directory
    set_target_properties(cpp_event_py PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CPPEVENT_PYTHON_OUTPUT_DIR}"
      RUNTIME_OUTPUT_DIRECTORY "${CPPEVENT_PYTHON_OUTPUT_DIR}"
      BUILD_RPATH "${CPPEVENT_LEGACY_OUTPUT_DIR}"
      INSTALL_RPATH "\$ORIGIN/../CppEvent"
    )
  else()
    message(WARNING "pybind11 not found in Python environment. Python bindings will not be built.")
    set(CPPEVENT_BUILD_PYTHON OFF CACHE BOOL "Build CppEvent Python bindings" FORCE)
  endif()
endif()
