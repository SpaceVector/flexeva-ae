cmake_minimum_required(VERSION 3.16)
project(lowlevel_interface LANGUAGES C CXX)

option(LLI_ENABLE_WRAPPER_BUILDS "Generate wrapper libraries and real-backend interposers" ON)
option(LLI_REFRESH_SCAN_CONFIGS "Refresh wrapper configs by scanning fake sources with libclang" OFF)
option(LLI_ENABLE_CUPTI_ACTIVITY_METADATA_WRAPPER_HOOKS
       "Generate CUPTI activity metadata observer hooks into wrapper sources" OFF)

if(LLI_REFRESH_SCAN_CONFIGS)
  find_path(
    LIBCLANG_INCLUDE_DIR
    NAMES clang-c/Index.h
    HINTS ENV LIBCLANG_PATH ENV LLVM_DIR
    PATHS /usr/lib/llvm-18/include /usr/lib/llvm-17/include /usr/lib/llvm-16/include
          /usr/lib/llvm-15/include /usr/lib/llvm-14/include
  )
  find_library(
    LIBCLANG_LIBRARY
    NAMES clang clang-18 clang-17 clang-16 clang-15 clang-14
    HINTS ENV LIBCLANG_PATH ENV LLVM_DIR
    PATHS /usr/lib/llvm-18/lib /usr/lib/llvm-17/lib /usr/lib/llvm-16/lib
          /usr/lib/llvm-15/lib /usr/lib/llvm-14/lib /usr/lib
  )

  if(NOT LIBCLANG_INCLUDE_DIR OR NOT LIBCLANG_LIBRARY)
    message(FATAL_ERROR "libclang headers or library not found. Install libclang-dev.")
  endif()

  find_package(nlohmann_json CONFIG REQUIRED)
endif()

get_filename_component(LLI_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(LLI_CPPEVENT_INCLUDE_DIR "${LLI_REPO_ROOT}/cpp/cpp_event/include")
set(LLI_CPPEVENT_LEGACY_OUTPUT_DIR "${LLI_REPO_ROOT}/CppEvent")
set(LLI_LEGACY_BUILD_DIR "${LLI_REPO_ROOT}/build/lowlevel-interface")

function(lli_set_legacy_output target_name)
  set_target_properties(${target_name} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${LLI_LEGACY_BUILD_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${LLI_LEGACY_BUILD_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${LLI_LEGACY_BUILD_DIR}"
  )
endfunction()

add_library(lowlevel_interface_runtime SHARED 
  src/debug_event.cpp
)
target_compile_features(lowlevel_interface_runtime PUBLIC cxx_std_20)
target_include_directories(lowlevel_interface_runtime
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                  ${LLI_CPPEVENT_INCLUDE_DIR})
lli_set_legacy_output(lowlevel_interface_runtime)
# Link against CppEvent runtime shared library for recorder functions
# The recorder implementation has been moved from lowlevel-interface to CppEvent
# as cpp_event_runtime_shared. This ensures all modules share the same recorder instance.
if(TARGET cpp_event_runtime_shared)
  target_link_libraries(lowlevel_interface_runtime PUBLIC cpp_event_runtime_shared)
else()
  # Fallback: try to find the shared library
  find_library(CPPEVENT_RUNTIME_SHARED
    NAMES cpp_event_runtime_shared
    PATHS ${CMAKE_BINARY_DIR}/CppEvent
          ${CMAKE_BINARY_DIR}/cpp/cpp_event
          ${LLI_CPPEVENT_LEGACY_OUTPUT_DIR}
    NO_DEFAULT_PATH
  )
  if(CPPEVENT_RUNTIME_SHARED)
    target_link_libraries(lowlevel_interface_runtime PUBLIC ${CPPEVENT_RUNTIME_SHARED})
  else()
    message(WARNING "cpp_event_runtime_shared not found. recorder functions may not work.")
  endif()
endif()
# Link against CppEvent TLS shared library for TLS context access
if(TARGET cpp_event_tls_shared)
  target_link_libraries(lowlevel_interface_runtime PRIVATE cpp_event_tls_shared)
else()
  # Fallback: try to find the shared library
  find_library(CPPEVENT_TLS_SHARED
    NAMES cpp_event_tls_shared
    PATHS ${CMAKE_BINARY_DIR}/CppEvent
          ${CMAKE_BINARY_DIR}/cpp/cpp_event
          ${LLI_CPPEVENT_LEGACY_OUTPUT_DIR}
    NO_DEFAULT_PATH
  )
  if(CPPEVENT_TLS_SHARED)
    target_link_libraries(lowlevel_interface_runtime PRIVATE ${CPPEVENT_TLS_SHARED})
  else()
    message(WARNING "cpp_event_tls_shared not found. debug_event may not work correctly.")
  endif()
endif()

if(LLI_REFRESH_SCAN_CONFIGS)
  add_executable(lli_scan_fakes tools/scan_fakes.cpp)
  target_compile_features(lli_scan_fakes PRIVATE cxx_std_20)
  target_include_directories(lli_scan_fakes PRIVATE ${LIBCLANG_INCLUDE_DIR}
                                                    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(lli_scan_fakes PRIVATE ${LIBCLANG_LIBRARY}
                                              nlohmann_json::nlohmann_json)
  lli_set_legacy_output(lli_scan_fakes)

  add_custom_target(
    lli_scan_configs
    COMMAND
      $<TARGET_FILE:lli_scan_fakes>
        --fakes-dir ${CMAKE_CURRENT_SOURCE_DIR}/fakes
        --config-dir ${CMAKE_CURRENT_SOURCE_DIR}/config
        --compile-commands ${LLI_REPO_ROOT}/compile_commands.json
    WORKING_DIRECTORY ${LLI_REPO_ROOT}
    COMMENT "Refreshing wrapper metadata configs from fake sources"
    DEPENDS lli_scan_fakes
    VERBATIM)
else()
  add_custom_target(
    lli_scan_configs
    COMMENT "Using checked-in wrapper metadata configs"
  )
endif()

file(GLOB LLI_FAKE_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/fakes/*.cpp)
file(GLOB LLI_CONFIG_FILES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/config/*_wrappers.json)

set(LLI_FAKE_PREFIXES "")
set(LLI_FAKE_TARGETS "")
set(LLI_WRAPPER_CONFIG_PATHS "")

if(LLI_FAKE_SOURCES)
  foreach(fake_src ${LLI_FAKE_SOURCES})
    get_filename_component(fake_name "${fake_src}" NAME_WE)
    add_library(${fake_name} SHARED "${fake_src}")
    target_compile_features(${fake_name} PUBLIC cxx_std_20)
    target_include_directories(${fake_name}
                               PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    lli_set_legacy_output(${fake_name})

    string(REGEX MATCH "^[^_]+" fake_prefix "${fake_name}")
    if(fake_prefix STREQUAL "")
      message(FATAL_ERROR
              "Unable to derive prefix from fake library '${fake_name}'.")
    endif()

    set(matched_config "")
    foreach(config_path ${LLI_CONFIG_FILES})
      get_filename_component(config_stem "${config_path}" NAME_WE)
      string(REGEX REPLACE "_wrappers$" "" config_prefix "${config_stem}")
      if(config_prefix STREQUAL fake_prefix)
        set(matched_config "${config_path}")
        break()
      endif()
    endforeach()

    if(matched_config STREQUAL "")
      message(FATAL_ERROR
              "No wrapper config found for fake '${fake_name}' (prefix '${fake_prefix}').")
    endif()

    list(APPEND LLI_FAKE_PREFIXES "${fake_prefix}")
    list(APPEND LLI_FAKE_TARGETS "${fake_name}")
    list(APPEND LLI_WRAPPER_CONFIG_PATHS "${matched_config}")
  endforeach()
endif()

# Also generate wrappers for configs without matching fake sources (RTLD_NEXT mode)
# These support external backends. Fake-cuda itself must be launched via frun/proot,
# not via LD_PRELOAD.
set(LLI_EXTERNAL_CONFIGS "")
foreach(config_path ${LLI_CONFIG_FILES})
  get_filename_component(config_stem "${config_path}" NAME_WE)
  string(REGEX REPLACE "_wrappers$" "" config_prefix "${config_stem}")

  # Check if this config has a matching local fake
  set(has_local_fake FALSE)
  foreach(fake_prefix ${LLI_FAKE_PREFIXES})
    if(config_prefix STREQUAL fake_prefix)
      set(has_local_fake TRUE)
      break()
    endif()
  endforeach()

  if(NOT has_local_fake)
    list(APPEND LLI_EXTERNAL_CONFIGS "${config_path}")
    message(STATUS "Found external wrapper config: ${config_stem} (RTLD_NEXT mode)")
  endif()
endforeach()

if(NOT LLI_FAKE_TARGETS AND NOT LLI_EXTERNAL_CONFIGS)
  message(FATAL_ERROR
          "No fake libraries or wrapper configs detected.")
endif()

if(COMPOSE_ENABLE_TESTS OR LLI_ENABLE_WRAPPER_BUILDS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  set(LLI_WRAPPER_LIBS "")
  set(LLI_CUPTI_ACTIVITY_METADATA_GENERATOR_ARGS "")
  if(LLI_ENABLE_CUPTI_ACTIVITY_METADATA_WRAPPER_HOOKS)
    list(APPEND LLI_CUPTI_ACTIVITY_METADATA_GENERATOR_ARGS
         --enable-cupti-activity-metadata)
  endif()

  list(LENGTH LLI_FAKE_TARGETS LLI_FAKE_COUNT)
  math(EXPR LLI_FAKE_LAST "${LLI_FAKE_COUNT} - 1")

  foreach(idx RANGE 0 ${LLI_FAKE_LAST})
    list(GET LLI_FAKE_PREFIXES ${idx} fake_prefix)
    list(GET LLI_FAKE_TARGETS ${idx} fake_target)
    list(GET LLI_WRAPPER_CONFIG_PATHS ${idx} config_path)

    string(TOUPPER "${fake_prefix}" fake_prefix_upper)
    set(wrapper_macro "LLI_FAKE_${fake_prefix_upper}_PATH")
    set(wrapper_output
        ${CMAKE_CURRENT_BINARY_DIR}/generated/${fake_prefix}_wrappers.cpp)

    add_custom_command(
      OUTPUT ${wrapper_output}
      COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_wrappers.py
        --config ${config_path}
        --wrapper-template ${CMAKE_CURRENT_SOURCE_DIR}/templates/wrapper.cpp.in
        --preamble-template
        ${CMAKE_CURRENT_SOURCE_DIR}/templates/module_preamble.cpp.in
        --backend-macro ${wrapper_macro}
        --output-root ${CMAKE_CURRENT_BINARY_DIR}
        ${LLI_CUPTI_ACTIVITY_METADATA_GENERATOR_ARGS}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_wrappers.py
              ${config_path}
              ${CMAKE_CURRENT_SOURCE_DIR}/templates/wrapper.cpp.in
              ${CMAKE_CURRENT_SOURCE_DIR}/templates/module_preamble.cpp.in
              lli_scan_configs
      COMMENT "Generating ${fake_prefix} wrapper sources"
      VERBATIM)

    set(wrapper_target "${fake_prefix}_wrappers")
    add_library(${wrapper_target} SHARED ${wrapper_output})
    target_include_directories(
      ${wrapper_target}
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
             ${LLI_CPPEVENT_INCLUDE_DIR})
    target_compile_features(${wrapper_target} PUBLIC cxx_std_20)
    target_compile_definitions(
      ${wrapper_target}
      PRIVATE ${wrapper_macro}="$<TARGET_FILE:${fake_target}>")
    add_dependencies(${wrapper_target} ${fake_target})
    target_link_libraries(${wrapper_target} PUBLIC lowlevel_interface_runtime dl)
    lli_set_legacy_output(${wrapper_target})

    list(APPEND LLI_WRAPPER_LIBS ${wrapper_target})

    # Also build real-backend interposers for the API sets used by Maya-lite
    # ground-truth capture.
    if(fake_prefix STREQUAL "cuda" OR fake_prefix STREQUAL "nccl" OR fake_prefix STREQUAL "cublas")
      set(real_wrapper_output
          ${CMAKE_CURRENT_BINARY_DIR}/generated/${fake_prefix}_real_wrappers.cpp)
      set(real_wrapper_macro "LLI_REAL_EXTERNAL_${fake_prefix_upper}_PATH")
      if(fake_prefix STREQUAL "cuda")
        set(real_backend_soname "libcudart.so.12")
      elseif(fake_prefix STREQUAL "nccl")
        set(real_backend_soname "libnccl.so.2")
      elseif(fake_prefix STREQUAL "cublas")
        set(real_backend_soname "libcublas.so.12")
      endif()

      add_custom_command(
        OUTPUT ${real_wrapper_output}
        COMMAND
          ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_wrappers.py
          --config ${config_path}
          --wrapper-template ${CMAKE_CURRENT_SOURCE_DIR}/templates/wrapper.cpp.in
          --preamble-template
          ${CMAKE_CURRENT_SOURCE_DIR}/templates/module_preamble.cpp.in
          --backend-macro ${real_wrapper_macro}
          --output-root ${CMAKE_CURRENT_BINARY_DIR}
          --output-relpath generated/${fake_prefix}_real_wrappers.cpp
          ${LLI_CUPTI_ACTIVITY_METADATA_GENERATOR_ARGS}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_wrappers.py
                ${config_path}
                ${CMAKE_CURRENT_SOURCE_DIR}/templates/wrapper.cpp.in
                ${CMAKE_CURRENT_SOURCE_DIR}/templates/module_preamble.cpp.in
                lli_scan_configs
        COMMENT "Generating ${fake_prefix} real-backend wrapper sources"
        VERBATIM)

      set(real_wrapper_target "${fake_prefix}_real_wrappers")
      add_library(${real_wrapper_target} SHARED ${real_wrapper_output})
      target_include_directories(
        ${real_wrapper_target}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
               ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_stubs
               ${LLI_CPPEVENT_INCLUDE_DIR})
      target_compile_features(${real_wrapper_target} PUBLIC cxx_std_20)
      target_compile_definitions(
        ${real_wrapper_target}
        PRIVATE ${real_wrapper_macro}="${real_backend_soname}")
      target_link_libraries(${real_wrapper_target} PUBLIC lowlevel_interface_runtime dl)
      lli_set_legacy_output(${real_wrapper_target})

      list(APPEND LLI_WRAPPER_LIBS ${real_wrapper_target})
    endif()
  endforeach()

  # Generate wrappers for external configs using RTLD_NEXT-based resolution.
  foreach(config_path ${LLI_EXTERNAL_CONFIGS})
    get_filename_component(config_stem "${config_path}" NAME_WE)
    string(REGEX REPLACE "_wrappers$" "" config_prefix "${config_stem}")
    string(TOUPPER "${config_prefix}" config_prefix_upper)

    # Use a dummy macro name - the wrapper will use RTLD_NEXT since it won't be defined
    set(wrapper_macro "LLI_EXTERNAL_${config_prefix_upper}_PATH")
    set(wrapper_output
        ${CMAKE_CURRENT_BINARY_DIR}/generated/${config_prefix}_wrappers.cpp)

    add_custom_command(
      OUTPUT ${wrapper_output}
      COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_wrappers.py
        --config ${config_path}
        --wrapper-template ${CMAKE_CURRENT_SOURCE_DIR}/templates/wrapper.cpp.in
        --preamble-template
        ${CMAKE_CURRENT_SOURCE_DIR}/templates/module_preamble.cpp.in
        --backend-macro ${wrapper_macro}
        --output-root ${CMAKE_CURRENT_BINARY_DIR}
        ${LLI_CUPTI_ACTIVITY_METADATA_GENERATOR_ARGS}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_wrappers.py
              ${config_path}
              ${CMAKE_CURRENT_SOURCE_DIR}/templates/wrapper.cpp.in
              ${CMAKE_CURRENT_SOURCE_DIR}/templates/module_preamble.cpp.in
      COMMENT "Generating ${config_prefix} wrapper sources (RTLD_NEXT mode)"
      VERBATIM)

    set(wrapper_target "${config_prefix}_wrappers")
    add_library(${wrapper_target} SHARED ${wrapper_output})
    target_include_directories(
      ${wrapper_target}
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
             ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_stubs
             ${LLI_CPPEVENT_INCLUDE_DIR})
    target_compile_features(${wrapper_target} PUBLIC cxx_std_20)
    # No macro defined - will use RTLD_NEXT fallback
    target_link_libraries(${wrapper_target} PUBLIC lowlevel_interface_runtime dl)
    lli_set_legacy_output(${wrapper_target})

    list(APPEND LLI_WRAPPER_LIBS ${wrapper_target})
  endforeach()

  if(COMPOSE_ENABLE_TESTS)
    add_executable(fake_pthread_smoke
                   ${LLI_REPO_ROOT}/tests/lowlevel_interface/fake_pthread_smoke.cpp)
    target_link_libraries(fake_pthread_smoke PRIVATE pthread_fake)
    lli_set_legacy_output(fake_pthread_smoke)
    add_test(NAME fake_pthread_smoke COMMAND fake_pthread_smoke)

    add_executable(lowlevel_interface_wrapper_smoke
                   ${LLI_REPO_ROOT}/tests/lowlevel_interface/lowlevel_interface_wrapper_smoke.cpp)
    target_include_directories(
      lowlevel_interface_wrapper_smoke
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
              ${LLI_CPPEVENT_INCLUDE_DIR})
    target_link_libraries(lowlevel_interface_wrapper_smoke PRIVATE
                                                              ${LLI_WRAPPER_LIBS}
                                                              dl)
    if(LLI_FAKE_TARGETS)
      add_dependencies(lowlevel_interface_wrapper_smoke ${LLI_FAKE_TARGETS})
    endif()
    lli_set_legacy_output(lowlevel_interface_wrapper_smoke)
    add_test(NAME lowlevel_interface_wrapper_smoke
             COMMAND lowlevel_interface_wrapper_smoke)
  endif()

endif()
