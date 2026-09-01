#include "lowlevel/interface/async_runtime_observer.hpp"

#include "cpp_event/event_log.hpp"
#include "cuda_stubs/cuda_runtime.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace lowlevel::interface {
namespace {

constexpr const char *kEnableEnvVar = "FLEXSIM_CAPTURE_REAL_ENABLE_ASYNC_RUNTIME";
constexpr const char *kCudartPathEnvVar = "FLEXSIM_LLI_CUDART_PATH";
constexpr const char *kRuntimeObservationIdKey = "async_runtime_observation_id";
constexpr const char *kRuntimeObservationSourceKey = "runtime_observation_source";
constexpr const char *kRuntimeObservationSourceValue = "capture_real_cuda_event";
constexpr const char *kObservedRuntimeUsKey = "observed_runtime_us";
constexpr const char *kWrapperRuntimeContractKey = "wrapper_runtime_contract";

using EventCreateWithFlagsFn = cudaError_t (*)(cudaEvent_t *, unsigned int);
using EventRecordFn = cudaError_t (*)(cudaEvent_t, cudaStream_t);
using EventSynchronizeFn = cudaError_t (*)(cudaEvent_t);
using EventElapsedTimeFn = cudaError_t (*)(float *, cudaEvent_t, cudaEvent_t);
using EventDestroyFn = cudaError_t (*)(cudaEvent_t);

struct CudartBackend final {
  void *handle{nullptr};
  EventCreateWithFlagsFn event_create_with_flags{nullptr};
  EventRecordFn event_record{nullptr};
  EventSynchronizeFn event_synchronize{nullptr};
  EventElapsedTimeFn event_elapsed_time{nullptr};
  EventDestroyFn event_destroy{nullptr};

  [[nodiscard]] bool available() const noexcept {
    return handle != nullptr && event_create_with_flags != nullptr &&
           event_record != nullptr && event_synchronize != nullptr &&
           event_elapsed_time != nullptr && event_destroy != nullptr;
  }
};

struct ObservationEntry final {
  cudaEvent_t start_event{nullptr};
  cudaEvent_t end_event{nullptr};
};

std::once_flag g_cudart_backend_once;
CudartBackend g_cudart_backend{};

std::mutex &observation_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<std::string, ObservationEntry> &observation_registry() {
  static auto *registry = new std::unordered_map<std::string, ObservationEntry>();
  return *registry;
}

std::unordered_map<void *, void *> &cublas_handle_stream_registry() {
  static auto *registry = new std::unordered_map<void *, void *>();
  return *registry;
}

std::atomic<unsigned long long> &observation_counter() {
  static auto *counter = new std::atomic<unsigned long long>(0);
  return *counter;
}

[[nodiscard]] std::string format_observed_runtime_us(double runtime_us) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(3) << std::max(runtime_us, 0.0);
  return stream.str();
}

[[nodiscard]] void *resolve_symbol(void *handle, const char *name) {
  if (handle == nullptr || name == nullptr) {
    return nullptr;
  }
  dlerror();
  void *symbol = dlsym(handle, name);
  if (dlerror() != nullptr) {
    return nullptr;
  }
  return symbol;
}

void initialize_cudart_backend() {
  const char *override_path = std::getenv(kCudartPathEnvVar);
  const char *candidates[] = {
      (override_path != nullptr && *override_path != '\0') ? override_path : nullptr,
      "libcudart.so.12",
      "libcudart.so",
  };
  for (const char *candidate : candidates) {
    if (candidate == nullptr) {
      continue;
    }
    void *handle = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      continue;
    }
    CudartBackend backend{};
    backend.handle = handle;
    backend.event_create_with_flags = reinterpret_cast<EventCreateWithFlagsFn>(
        resolve_symbol(handle, "cudaEventCreateWithFlags"));
    backend.event_record =
        reinterpret_cast<EventRecordFn>(resolve_symbol(handle, "cudaEventRecord"));
    backend.event_synchronize = reinterpret_cast<EventSynchronizeFn>(
        resolve_symbol(handle, "cudaEventSynchronize"));
    backend.event_elapsed_time = reinterpret_cast<EventElapsedTimeFn>(
        resolve_symbol(handle, "cudaEventElapsedTime"));
    backend.event_destroy =
        reinterpret_cast<EventDestroyFn>(resolve_symbol(handle, "cudaEventDestroy"));
    if (backend.available()) {
      g_cudart_backend = backend;
      return;
    }
    dlclose(handle);
  }
}

[[nodiscard]] CudartBackend &cudart_backend() {
  std::call_once(g_cudart_backend_once, initialize_cudart_backend);
  return g_cudart_backend;
}

[[nodiscard]] bool async_runtime_observer_enabled() noexcept {
  const char *raw = std::getenv(kEnableEnvVar);
  return raw != nullptr && raw[0] != '\0' && std::strcmp(raw, "0") != 0;
}

[[nodiscard]] std::string_view canonicalize_async_runtime_api(std::string_view api_name) {
  if (api_name == "cudaEventRecordWithFlags") {
    return "cudaEventRecord";
  }
  if (api_name == "ncclBcast") {
    return "ncclBroadcast";
  }
  return api_name;
}

[[nodiscard]] bool api_supports_async_runtime_observation(std::string_view api_name) {
  api_name = canonicalize_async_runtime_api(api_name);
  return api_name == "cudaLaunchKernel" || api_name == "cublasSgemm_v2" ||
         api_name == "cublasGemmEx" ||
         api_name == "cublasGemmStridedBatchedEx" ||
         api_name == "cublasGemmBatchedEx" || api_name == "cublasLtMatmul" ||
         api_name == "ncclAllReduce" || api_name == "ncclAllGather" ||
         api_name == "ncclAllToAll" || api_name == "ncclAllToAllv" ||
         api_name == "ncclBroadcast" || api_name == "ncclReduce" ||
         api_name == "ncclReduceScatter" || api_name == "ncclSend" ||
         api_name == "ncclRecv";
}

[[nodiscard]] std::string next_observation_id() {
  auto value =
      observation_counter().fetch_add(1, std::memory_order_acq_rel) + 1ULL;
  return std::to_string(value);
}

void destroy_event_noexcept(cudaEvent_t event) noexcept {
  if (event == nullptr) {
    return;
  }
  auto &backend = cudart_backend();
  if (backend.event_destroy != nullptr) {
    backend.event_destroy(event);
  }
}

void discard_observation(AsyncRuntimeObservation &observation) noexcept {
  destroy_event_noexcept(reinterpret_cast<cudaEvent_t>(observation.start_event));
  destroy_event_noexcept(reinterpret_cast<cudaEvent_t>(observation.end_event));
  observation = AsyncRuntimeObservation{};
}

} // namespace

AsyncRuntimeObservation begin_async_runtime_observation(std::string_view api_name,
                                                        void *stream_handle) {
  AsyncRuntimeObservation observation{};
  if (!async_runtime_observer_enabled() ||
      !api_supports_async_runtime_observation(api_name)) {
    return observation;
  }
  auto &backend = cudart_backend();
  if (!backend.available()) {
    return observation;
  }
  cudaEvent_t start_event = nullptr;
  cudaEvent_t end_event = nullptr;
  if (backend.event_create_with_flags(&start_event, 0U) != cudaSuccess) {
    return observation;
  }
  if (backend.event_create_with_flags(&end_event, 0U) != cudaSuccess) {
    destroy_event_noexcept(start_event);
    return observation;
  }
  if (backend.event_record(start_event,
                           reinterpret_cast<cudaStream_t>(stream_handle)) !=
      cudaSuccess) {
    destroy_event_noexcept(start_event);
    destroy_event_noexcept(end_event);
    return observation;
  }
  observation.active = true;
  observation.stream_handle = stream_handle;
  observation.start_event = reinterpret_cast<void *>(start_event);
  observation.end_event = reinterpret_cast<void *>(end_event);
  observation.observation_id = next_observation_id();
  return observation;
}

void complete_async_runtime_observation(AsyncRuntimeObservation &observation,
                                        cpp_event::EventPayload &payload,
                                        bool success) {
  if (!observation.active) {
    return;
  }
  auto &backend = cudart_backend();
  if (!backend.available() || !success) {
    discard_observation(observation);
    return;
  }
  cudaEvent_t end_event = reinterpret_cast<cudaEvent_t>(observation.end_event);
  if (backend.event_record(
          end_event,
          reinterpret_cast<cudaStream_t>(observation.stream_handle)) !=
      cudaSuccess) {
    discard_observation(observation);
    return;
  }
  {
    std::lock_guard<std::mutex> guard(observation_mutex());
    observation_registry().emplace(
        observation.observation_id,
        ObservationEntry{
            .start_event =
                reinterpret_cast<cudaEvent_t>(observation.start_event),
            .end_event = end_event,
        });
  }
  payload.attributes[kRuntimeObservationIdKey] = observation.observation_id;
  observation = AsyncRuntimeObservation{};
}

void register_cublas_handle_for_async_runtime(void *handle) noexcept {
  if (handle == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> guard(observation_mutex());
  cublas_handle_stream_registry().insert_or_assign(handle, nullptr);
}

void unregister_cublas_handle_for_async_runtime(void *handle) noexcept {
  if (handle == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> guard(observation_mutex());
  cublas_handle_stream_registry().erase(handle);
}

void update_cublas_handle_stream_for_async_runtime(void *handle,
                                                   void *stream_handle) noexcept {
  if (handle == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> guard(observation_mutex());
  cublas_handle_stream_registry().insert_or_assign(handle, stream_handle);
}

void *lookup_cublas_handle_stream_for_async_runtime(void *handle) noexcept {
  if (handle == nullptr) {
    return nullptr;
  }
  std::lock_guard<std::mutex> guard(observation_mutex());
  auto &registry = cublas_handle_stream_registry();
  auto it = registry.find(handle);
  if (it == registry.end()) {
    return nullptr;
  }
  return it->second;
}

bool lookup_registered_cublas_handle_stream_for_async_runtime(
    void *handle,
    void **stream_handle) noexcept {
  if (stream_handle != nullptr) {
    *stream_handle = nullptr;
  }
  if (handle == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> guard(observation_mutex());
  auto &registry = cublas_handle_stream_registry();
  auto it = registry.find(handle);
  if (it == registry.end()) {
    return false;
  }
  if (stream_handle != nullptr) {
    *stream_handle = it->second;
  }
  return true;
}

void resolve_async_runtime_observations(cpp_event::EventLog &log) {
  auto &backend = cudart_backend();
  if (!backend.available()) {
    return;
  }
  log.with_mutable_events([&](cpp_event::EventList &events) {
    for (auto &event : events) {
      auto it = event.payload.attributes.find(kRuntimeObservationIdKey);
      if (it == event.payload.attributes.end()) {
        continue;
      }
      ObservationEntry entry{};
      bool found = false;
      {
        std::lock_guard<std::mutex> guard(observation_mutex());
        auto registry_it = observation_registry().find(it->second);
        if (registry_it != observation_registry().end()) {
          entry = registry_it->second;
          observation_registry().erase(registry_it);
          found = true;
        }
      }
      event.payload.attributes.erase(kRuntimeObservationIdKey);
      if (!found) {
        continue;
      }
      float elapsed_ms = 0.0f;
      bool resolved =
          backend.event_synchronize(entry.end_event) == cudaSuccess &&
          backend.event_elapsed_time(&elapsed_ms, entry.start_event,
                                     entry.end_event) == cudaSuccess;
      destroy_event_noexcept(entry.start_event);
      destroy_event_noexcept(entry.end_event);
      if (!resolved) {
        continue;
      }
      event.payload.attributes[kRuntimeObservationSourceKey] =
          kRuntimeObservationSourceValue;
      event.payload.attributes[kWrapperRuntimeContractKey] = "async_runtime";
      event.payload.attributes[kObservedRuntimeUsKey] =
          format_observed_runtime_us(static_cast<double>(elapsed_ms) * 1000.0);
    }
  });
}

void clear_async_runtime_observations() noexcept {
  std::lock_guard<std::mutex> guard(observation_mutex());
  for (auto &[_, entry] : observation_registry()) {
    destroy_event_noexcept(entry.start_event);
    destroy_event_noexcept(entry.end_event);
  }
  observation_registry().clear();
  cublas_handle_stream_registry().clear();
}

} // namespace lowlevel::interface
