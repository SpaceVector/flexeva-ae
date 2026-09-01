#include "lowlevel/interface/cupti_activity_metadata_observer.hpp"

#include "cpp_event/event_log.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <deque>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <sys/syscall.h>
#include <unistd.h>

#if defined(CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY) && CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY
#include <cupti.h>
#endif

namespace lowlevel::interface {
namespace {

constexpr const char *kSchemaVersion = "1";
constexpr const char *kEnableEnvVar = "MAYA_ENABLE_CUPTI_ACTIVITY_METADATA";
constexpr const char *kCompatEnableEnvVar =
    "FLEXSIM_MAYA_ENABLE_CUPTI_ACTIVITY_METADATA";
constexpr std::size_t kCuptiBufferSize = 32 * 1024;

struct ActivitySummary final {
  unsigned long long external_records{0};
  unsigned long long callback_correlation_records{0};
  unsigned long long runtime_records{0};
  unsigned long long kernel_records{0};
  unsigned long long dropped_records{0};
  std::string first_runtime_start{};
  std::string first_runtime_end{};
  std::string first_kernel_start{};
  std::string first_kernel_end{};
  std::string last_kernel_start{};
  std::string last_kernel_end{};
  std::string first_kernel_stream_id{};
  std::string last_kernel_stream_id{};
  std::set<std::string> kernel_stream_ids{};
};

struct PendingActivitySummary final {
  unsigned long long runtime_records{0};
  unsigned long long kernel_records{0};
  std::string first_runtime_start{};
  std::string first_runtime_end{};
  std::string first_kernel_start{};
  std::string first_kernel_end{};
  std::string last_kernel_start{};
  std::string last_kernel_end{};
  std::string first_kernel_stream_id{};
  std::string last_kernel_stream_id{};
  std::set<std::string> kernel_stream_ids{};
};

std::mutex &activity_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<unsigned long long, ActivitySummary> &activity_by_external_id() {
  static auto *records = new std::unordered_map<unsigned long long, ActivitySummary>();
  return *records;
}

std::unordered_map<unsigned int, unsigned long long> &correlation_to_external_id() {
  static auto *records = new std::unordered_map<unsigned int, unsigned long long>();
  return *records;
}

struct ThreadWindowState final {
  unsigned long long external_id{0};
  std::string api_name{};
  bool completed{false};
};

struct CompletedCallsite final {
  unsigned long long external_id{0};
  std::string api_name{};
};

std::unordered_map<long long, ThreadWindowState> &thread_window_external_id() {
  static auto *records = new std::unordered_map<long long, ThreadWindowState>();
  return *records;
}

std::unordered_map<long long, std::unordered_map<std::string, std::deque<CompletedCallsite>>> &
completed_callsite_fifo_by_thread_api() {
  static auto *records = new std::unordered_map<
      long long, std::unordered_map<std::string, std::deque<CompletedCallsite>>>();
  return *records;
}

std::unordered_map<unsigned int, PendingActivitySummary> &pending_activity_by_correlation_id() {
  static auto *records =
      new std::unordered_map<unsigned int, PendingActivitySummary>();
  return *records;
}

std::atomic<unsigned long long> &total_external_activity_records() {
  static auto *counter = new std::atomic<unsigned long long>(0);
  return *counter;
}

std::atomic<unsigned long long> &total_callback_correlation_records() {
  static auto *counter = new std::atomic<unsigned long long>(0);
  return *counter;
}

std::atomic<unsigned long long> &total_runtime_callback_records() {
  static auto *counter = new std::atomic<unsigned long long>(0);
  return *counter;
}

std::atomic<unsigned long long> &total_runtime_callback_with_wrapper_records() {
  static auto *counter = new std::atomic<unsigned long long>(0);
  return *counter;
}

std::atomic<unsigned long long> &total_runtime_callback_with_thread_window_records() {
  static auto *counter = new std::atomic<unsigned long long>(0);
  return *counter;
}

std::atomic<unsigned long long> &total_target_runtime_callback_records() {
  static auto *counter = new std::atomic<unsigned long long>(0);
  return *counter;
}

std::atomic<unsigned long long> &total_target_runtime_callback_with_thread_window_records() {
  static auto *counter = new std::atomic<unsigned long long>(0);
  return *counter;
}

std::atomic<unsigned long long> &total_callsite_fifo_correlation_records() {
  static auto *counter = new std::atomic<unsigned long long>(0);
  return *counter;
}

std::atomic<unsigned long long> &total_callsite_fifo_candidate_records() {
  static auto *counter = new std::atomic<unsigned long long>(0);
  return *counter;
}

std::atomic<unsigned long long> &total_runtime_activity_records() {
  static auto *counter = new std::atomic<unsigned long long>(0);
  return *counter;
}

std::atomic<unsigned long long> &total_kernel_activity_records() {
  static auto *counter = new std::atomic<unsigned long long>(0);
  return *counter;
}

std::string to_string_u64(unsigned long long value) {
  return std::to_string(value);
}

template <typename SummaryT>
void add_kernel_activity(SummaryT &summary, unsigned long long start,
                         unsigned long long end, unsigned long long stream_id) {
  const std::string start_text = to_string_u64(start);
  const std::string end_text = to_string_u64(end);
  const std::string stream_text = to_string_u64(stream_id);
  if (summary.first_kernel_start.empty()) {
    summary.first_kernel_start = start_text;
    summary.first_kernel_end = end_text;
    summary.first_kernel_stream_id = stream_text;
  }
  summary.last_kernel_start = start_text;
  summary.last_kernel_end = end_text;
  summary.last_kernel_stream_id = stream_text;
  summary.kernel_stream_ids.insert(stream_text);
}

void merge_pending_activity(ActivitySummary &summary,
                            const PendingActivitySummary &pending) {
  summary.runtime_records += pending.runtime_records;
  summary.kernel_records += pending.kernel_records;
  if (summary.first_runtime_start.empty() &&
      !pending.first_runtime_start.empty()) {
    summary.first_runtime_start = pending.first_runtime_start;
    summary.first_runtime_end = pending.first_runtime_end;
  }
  if (summary.first_kernel_start.empty() &&
      !pending.first_kernel_start.empty()) {
    summary.first_kernel_start = pending.first_kernel_start;
    summary.first_kernel_end = pending.first_kernel_end;
    summary.first_kernel_stream_id = pending.first_kernel_stream_id;
  }
  if (!pending.last_kernel_start.empty()) {
    summary.last_kernel_start = pending.last_kernel_start;
    summary.last_kernel_end = pending.last_kernel_end;
    summary.last_kernel_stream_id = pending.last_kernel_stream_id;
  }
  summary.kernel_stream_ids.insert(pending.kernel_stream_ids.begin(),
                                   pending.kernel_stream_ids.end());
}

unsigned long long parse_u64(std::string_view value) {
  unsigned long long result = 0;
  for (char ch : value) {
    if (ch < '0' || ch > '9') {
      break;
    }
    result = result * 10ULL + static_cast<unsigned long long>(ch - '0');
  }
  return result;
}

std::atomic<unsigned long long> &logical_event_counter() {
  static auto *counter = new std::atomic<unsigned long long>(0);
  return *counter;
}

[[nodiscard]] bool env_flag_enabled(const char *name) noexcept {
  const char *raw = std::getenv(name);
  return raw != nullptr && raw[0] != '\0' && std::strcmp(raw, "0") != 0;
}

[[nodiscard]] bool cupti_activity_metadata_enabled() noexcept {
  return env_flag_enabled(kEnableEnvVar) || env_flag_enabled(kCompatEnableEnvVar);
}

[[nodiscard]] std::string_view canonicalize_api_name(std::string_view api_name) {
  if (api_name == "cudaEventRecordWithFlags") {
    return "cudaEventRecord";
  }
  return api_name;
}

[[nodiscard]] bool api_supports_cupti_activity_metadata(std::string_view api_name) {
  api_name = canonicalize_api_name(api_name);
  return api_name == "cudaLaunchKernel" || api_name == "cudaEventRecord" ||
         api_name == "cudaStreamWaitEvent" || api_name == "cublasGemmEx" ||
         api_name == "cublasGemmStridedBatchedEx";
}

[[nodiscard]] std::string_view api_role(std::string_view api_name) {
  api_name = canonicalize_api_name(api_name);
  if (api_name == "cudaLaunchKernel") {
    return "kernel_launch";
  }
  if (api_name == "cudaEventRecord") {
    return "cuda_event_record";
  }
  if (api_name == "cudaStreamWaitEvent") {
    return "cuda_stream_wait_event";
  }
  if (api_name == "cublasGemmEx") {
    return "cublas_gemm";
  }
  if (api_name == "cublasGemmStridedBatchedEx") {
    return "cublas_strided_batched_gemm";
  }
  return "unsupported";
}

[[nodiscard]] std::string next_logical_event_id() {
  auto value =
      logical_event_counter().fetch_add(1, std::memory_order_acq_rel) + 1ULL;
  return std::to_string(value);
}

void set_attr(cpp_event::EventPayload &payload, const char *key,
              std::string_view value) {
  payload.attributes[std::string(key)] = std::string(value);
}

long long current_thread_id() {
  return static_cast<long long>(::syscall(SYS_gettid));
}

[[nodiscard]] bool is_target_callback_function(const char *function_name) {
  if (function_name == nullptr) {
    return false;
  }
  std::string_view name(function_name);
  return name.find("cudaLaunchKernel") != std::string_view::npos ||
         name.find("cudaEventRecord") != std::string_view::npos ||
         name.find("cudaStreamWaitEvent") != std::string_view::npos;
}

[[nodiscard]] std::string_view canonical_callback_api_name(const char *function_name) {
  if (function_name == nullptr) {
    return "";
  }
  std::string_view name(function_name);
  if (name.find("cudaLaunchKernel") != std::string_view::npos) {
    return "cudaLaunchKernel";
  }
  if (name.find("cudaEventRecord") != std::string_view::npos) {
    return "cudaEventRecord";
  }
  if (name.find("cudaStreamWaitEvent") != std::string_view::npos) {
    return "cudaStreamWaitEvent";
  }
  return "";
}

[[nodiscard]] bool callback_function_matches_api(const char *function_name,
                                                 std::string_view api_name) {
  if (function_name == nullptr || api_name.empty()) {
    return false;
  }
  return std::string_view(function_name).find(api_name) != std::string_view::npos;
}

struct CallbackDebugState final {
  std::string first_callback_thread_id{};
  std::string first_callback_function{};
  std::string first_callback_site{};
  std::string first_callback_tls_external_id{};
  std::string first_callback_thread_window_external_id{};
  std::string first_window_callback_thread_id{};
  std::string first_window_callback_function{};
  std::string first_window_callback_site{};
  std::string first_target_callback_thread_id{};
  std::string first_target_callback_function{};
  std::string first_target_callback_site{};
  std::string first_target_window_callback_thread_id{};
  std::string first_target_window_callback_function{};
  std::string first_target_window_callback_site{};
  std::string first_callsite_fifo_callback_thread_id{};
  std::string first_callsite_fifo_callback_function{};
  std::string first_callsite_fifo_callback_site{};
  std::string first_callsite_fifo_external_id{};
  std::vector<std::string> callback_samples{};
};

CallbackDebugState &callback_debug_state() {
  static auto *state = new CallbackDebugState();
  return *state;
}

#if defined(CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY) && CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY

thread_local unsigned long long current_wrapper_external_id = 0;

std::once_flag &cupti_init_once() {
  static auto *flag = new std::once_flag();
  return *flag;
}

std::atomic<bool> &cupti_available() {
  static auto *available = new std::atomic<bool>(false);
  return *available;
}

std::atomic<bool> &cupti_callback_correlation_available() {
  static auto *available = new std::atomic<bool>(false);
  return *available;
}

std::atomic<unsigned long long> &cupti_dropped_records() {
  static auto *dropped = new std::atomic<unsigned long long>(0);
  return *dropped;
}

CUpti_SubscriberHandle &cupti_subscriber_handle() {
  static auto *subscriber = new CUpti_SubscriberHandle();
  return *subscriber;
}

std::mutex &cupti_buffer_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

std::vector<void *> &cupti_buffers() {
  static auto *buffers = new std::vector<void *>();
  return *buffers;
}

const char *cupti_result_name(CUptiResult result) {
  const char *name = nullptr;
  if (cuptiGetResultString(result, &name) == CUPTI_SUCCESS && name != nullptr) {
    return name;
  }
  return "unknown_cupti_error";
}

void CUPTIAPI request_activity_buffer(uint8_t **buffer, size_t *size,
                                      size_t *max_records) {
  *size = kCuptiBufferSize;
  *buffer = reinterpret_cast<uint8_t *>(std::malloc(*size));
  *max_records = 0;
  if (*buffer != nullptr) {
    std::lock_guard<std::mutex> guard(cupti_buffer_mutex());
    cupti_buffers().push_back(*buffer);
  }
}

void record_activity(CUpti_Activity *record) {
  if (record == nullptr) {
    return;
  }
  auto &correlation_to_external = correlation_to_external_id();
  auto &by_external = activity_by_external_id();
  auto &pending_by_correlation = pending_activity_by_correlation_id();
  switch (record->kind) {
  case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
    auto *external = reinterpret_cast<CUpti_ActivityExternalCorrelation *>(record);
    if (external->externalKind != CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0) {
      return;
    }
    total_external_activity_records().fetch_add(1, std::memory_order_acq_rel);
    const auto external_id =
        static_cast<unsigned long long>(external->externalId);
    correlation_to_external[external->correlationId] = external_id;
    auto &summary = by_external[external_id];
    summary.external_records++;
    auto pending_it = pending_by_correlation.find(external->correlationId);
    if (pending_it != pending_by_correlation.end()) {
      merge_pending_activity(summary, pending_it->second);
      pending_by_correlation.erase(pending_it);
    }
    break;
  }
  case CUPTI_ACTIVITY_KIND_RUNTIME: {
    auto *runtime = reinterpret_cast<CUpti_ActivityAPI *>(record);
    total_runtime_activity_records().fetch_add(1, std::memory_order_acq_rel);
    auto it = correlation_to_external.find(runtime->correlationId);
    if (it == correlation_to_external.end()) {
      auto &pending = pending_by_correlation[runtime->correlationId];
      pending.runtime_records++;
      if (pending.first_runtime_start.empty()) {
        pending.first_runtime_start = to_string_u64(runtime->start);
        pending.first_runtime_end = to_string_u64(runtime->end);
      }
      return;
    }
    auto &summary = by_external[it->second];
    summary.runtime_records++;
    if (summary.first_runtime_start.empty()) {
      summary.first_runtime_start = to_string_u64(runtime->start);
      summary.first_runtime_end = to_string_u64(runtime->end);
    }
    break;
  }
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    auto *kernel = reinterpret_cast<CUpti_ActivityKernel4 *>(record);
    total_kernel_activity_records().fetch_add(1, std::memory_order_acq_rel);
    auto it = correlation_to_external.find(kernel->correlationId);
    if (it == correlation_to_external.end()) {
      auto &pending = pending_by_correlation[kernel->correlationId];
      pending.kernel_records++;
      add_kernel_activity(pending, kernel->start, kernel->end, kernel->streamId);
      return;
    }
    auto &summary = by_external[it->second];
    summary.kernel_records++;
    add_kernel_activity(summary, kernel->start, kernel->end, kernel->streamId);
    break;
  }
  default:
    break;
  }
}

void map_callback_correlation_id(uint32_t correlation_id,
                                 unsigned long long external_id) {
  if (external_id == 0) {
    return;
  }

  auto &correlation_to_external = correlation_to_external_id();
  auto &by_external = activity_by_external_id();
  auto &pending_by_correlation = pending_activity_by_correlation_id();

  auto insert_result =
      correlation_to_external.emplace(correlation_id, external_id);
  if (!insert_result.second && insert_result.first->second == external_id) {
    return;
  }
  insert_result.first->second = external_id;

  total_callback_correlation_records().fetch_add(1, std::memory_order_acq_rel);
  auto &summary = by_external[external_id];
  summary.callback_correlation_records++;

  auto pending_it = pending_by_correlation.find(correlation_id);
  if (pending_it == pending_by_correlation.end()) {
    return;
  }
  merge_pending_activity(summary, pending_it->second);
  pending_by_correlation.erase(pending_it);
}

void CUPTIAPI runtime_callback(void *, CUpti_CallbackDomain domain,
                               CUpti_CallbackId, const void *cbdata) {
  if (domain != CUPTI_CB_DOMAIN_RUNTIME_API || cbdata == nullptr) {
    return;
  }
  auto *callback_data = static_cast<const CUpti_CallbackData *>(cbdata);
  if (callback_data->callbackSite != CUPTI_API_ENTER &&
      callback_data->callbackSite != CUPTI_API_EXIT) {
    return;
  }
  total_runtime_callback_records().fetch_add(1, std::memory_order_acq_rel);
  const long long tid = current_thread_id();
  const bool target_function =
      is_target_callback_function(callback_data->functionName);
  if (target_function) {
    total_target_runtime_callback_records().fetch_add(1,
                                                      std::memory_order_acq_rel);
  }

  unsigned long long thread_window_external_id_value = 0;
  unsigned long long callsite_fifo_external_id = 0;
  bool thread_window_matches_callback = false;
  bool thread_window_completed = false;
  bool callsite_fifo_matched = false;
  {
    std::lock_guard<std::mutex> guard(activity_mutex());
    auto &debug = callback_debug_state();
    const char *site =
        callback_data->callbackSite == CUPTI_API_ENTER ? "ENTER" : "EXIT";
    const char *function =
        callback_data->functionName == nullptr ? "" : callback_data->functionName;
    if (debug.first_callback_thread_id.empty()) {
      debug.first_callback_thread_id =
          to_string_u64(static_cast<unsigned long long>(tid));
      debug.first_callback_function = function;
      debug.first_callback_site = site;
      debug.first_callback_tls_external_id =
          to_string_u64(current_wrapper_external_id);
    }
    auto window_it = thread_window_external_id().find(tid);
    std::string window_api_name{};
    if (window_it != thread_window_external_id().end()) {
      thread_window_external_id_value = window_it->second.external_id;
      thread_window_completed = window_it->second.completed;
      window_api_name = window_it->second.api_name;
      thread_window_matches_callback =
          callback_function_matches_api(callback_data->functionName,
                                        window_it->second.api_name);
      if (debug.first_window_callback_thread_id.empty()) {
        debug.first_window_callback_thread_id =
            to_string_u64(static_cast<unsigned long long>(tid));
        debug.first_window_callback_function = function;
        debug.first_window_callback_site = site;
      }
    }
    debug.first_callback_thread_window_external_id =
        to_string_u64(thread_window_external_id_value);
    if (target_function && debug.first_target_callback_thread_id.empty()) {
      debug.first_target_callback_thread_id =
          to_string_u64(static_cast<unsigned long long>(tid));
      debug.first_target_callback_function = function;
      debug.first_target_callback_site = site;
    }
    if (target_function && thread_window_external_id_value != 0 &&
        thread_window_matches_callback &&
        debug.first_target_window_callback_thread_id.empty()) {
      debug.first_target_window_callback_thread_id =
          to_string_u64(static_cast<unsigned long long>(tid));
      debug.first_target_window_callback_function = function;
      debug.first_target_window_callback_site = site;
    }
    if (debug.callback_samples.size() < 24) {
      std::ostringstream sample;
      sample << "tid=" << tid << ",site=" << site << ",function=" << function
             << ",tls_external=" << current_wrapper_external_id
             << ",window_external=" << thread_window_external_id_value
             << ",window_api=" << window_api_name
             << ",window_completed=" << (thread_window_completed ? "true" : "false")
             << ",matches_window="
             << (thread_window_matches_callback ? "true" : "false")
             << ",target=" << (target_function ? "true" : "false");
      debug.callback_samples.push_back(sample.str());
    }
    if (callback_data->callbackSite == CUPTI_API_ENTER) {
      std::string_view callback_api =
          canonical_callback_api_name(callback_data->functionName);
      if (!callback_api.empty()) {
        auto thread_it = completed_callsite_fifo_by_thread_api().find(tid);
        if (thread_it != completed_callsite_fifo_by_thread_api().end()) {
          auto queue_it = thread_it->second.find(std::string(callback_api));
          if (queue_it != thread_it->second.end() && !queue_it->second.empty()) {
            CompletedCallsite callsite = queue_it->second.front();
            queue_it->second.pop_front();
            if (queue_it->second.empty()) {
              thread_it->second.erase(queue_it);
            }
            if (thread_it->second.empty()) {
              completed_callsite_fifo_by_thread_api().erase(thread_it);
            }
            callsite_fifo_external_id = callsite.external_id;
            callsite_fifo_matched = true;
            if (debug.first_callsite_fifo_callback_thread_id.empty()) {
              debug.first_callsite_fifo_callback_thread_id =
                  to_string_u64(static_cast<unsigned long long>(tid));
              debug.first_callsite_fifo_callback_function = function;
              debug.first_callsite_fifo_callback_site = site;
              debug.first_callsite_fifo_external_id =
                  to_string_u64(callsite.external_id);
            }
          }
        }
      }
    }
  }

  unsigned long long external_id = current_wrapper_external_id;
  if (external_id == 0 && callsite_fifo_matched) {
    total_callsite_fifo_candidate_records().fetch_add(
        1, std::memory_order_acq_rel);
  }
  if (external_id == 0 && thread_window_completed &&
      thread_window_matches_callback &&
      callback_data->callbackSite == CUPTI_API_EXIT) {
    external_id = thread_window_external_id_value;
  }
  if (external_id == 0) {
    return;
  }
  if (current_wrapper_external_id != 0) {
    total_runtime_callback_with_wrapper_records().fetch_add(
        1, std::memory_order_acq_rel);
  }
  if (thread_window_external_id_value != 0 && thread_window_completed &&
      thread_window_matches_callback &&
      callback_data->callbackSite == CUPTI_API_EXIT) {
    total_runtime_callback_with_thread_window_records().fetch_add(
        1, std::memory_order_acq_rel);
    if (target_function) {
      total_target_runtime_callback_with_thread_window_records().fetch_add(
          1, std::memory_order_acq_rel);
    }
  }
  std::lock_guard<std::mutex> guard(activity_mutex());
  map_callback_correlation_id(callback_data->correlationId, external_id);
  if (thread_window_completed && thread_window_matches_callback &&
      callback_data->callbackSite == CUPTI_API_EXIT) {
    thread_window_external_id().erase(tid);
  }
}

void CUPTIAPI complete_activity_buffer(CUcontext context, uint32_t stream_id,
                                       uint8_t *buffer, size_t size,
                                       size_t valid_size) {
  (void)size;
  CUpti_Activity *record = nullptr;
  {
    std::lock_guard<std::mutex> guard(activity_mutex());
    while (cuptiActivityGetNextRecord(buffer, valid_size, &record) == CUPTI_SUCCESS) {
      record_activity(record);
    }
  }
  size_t dropped = 0;
  if (cuptiActivityGetNumDroppedRecords(context, stream_id, &dropped) ==
      CUPTI_SUCCESS) {
    cupti_dropped_records().fetch_add(static_cast<unsigned long long>(dropped),
                                      std::memory_order_acq_rel);
  }
}

void initialize_cupti_once() {
  CUptiResult result =
      cuptiActivityRegisterCallbacks(request_activity_buffer, complete_activity_buffer);
  if (result != CUPTI_SUCCESS) {
    return;
  }
  result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
  if (result != CUPTI_SUCCESS) {
    return;
  }
  result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  if (result != CUPTI_SUCCESS) {
    return;
  }
  result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
  if (result != CUPTI_SUCCESS) {
    return;
  }
  result = cuptiSubscribe(&cupti_subscriber_handle(), runtime_callback, nullptr);
  if (result == CUPTI_SUCCESS) {
    result = cuptiEnableDomain(1, cupti_subscriber_handle(),
                               CUPTI_CB_DOMAIN_RUNTIME_API);
    if (result == CUPTI_SUCCESS) {
      cupti_callback_correlation_available().store(true,
                                                   std::memory_order_release);
    }
  }
  cupti_available().store(true, std::memory_order_release);
}

bool ensure_cupti_available() {
  std::call_once(cupti_init_once(), initialize_cupti_once);
  return cupti_available().load(std::memory_order_acquire);
}

bool push_external_id(unsigned long long external_id,
                      std::string &unavailable_reason) {
  if (!ensure_cupti_available()) {
    unavailable_reason = "cupti_activity_initialization_failed";
    return false;
  }
  CUptiResult result = cuptiActivityPushExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, external_id);
  if (result != CUPTI_SUCCESS) {
    unavailable_reason = std::string("cupti_push_external_correlation_failed:") +
                         cupti_result_name(result);
    return false;
  }
  return true;
}

void set_current_callback_external_id(unsigned long long external_id,
                                      std::string_view api_name) {
  current_wrapper_external_id = external_id;
  std::lock_guard<std::mutex> guard(activity_mutex());
  thread_window_external_id()[current_thread_id()] =
      ThreadWindowState{external_id, std::string(api_name), false};
}

bool callback_correlation_available() {
  return cupti_callback_correlation_available().load(std::memory_order_acquire);
}

void pop_external_id() {
  uint64_t popped = 0;
  (void)cuptiActivityPopExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &popped);
}

void clear_current_callback_external_id() {
  {
    std::lock_guard<std::mutex> guard(activity_mutex());
    auto window_it = thread_window_external_id().find(current_thread_id());
    if (window_it != thread_window_external_id().end()) {
      window_it->second.completed = true;
      completed_callsite_fifo_by_thread_api()[current_thread_id()]
                                             [window_it->second.api_name]
                                                 .push_back(CompletedCallsite{
                                                     window_it->second.external_id,
                                                     window_it->second.api_name});
    }
  }
  current_wrapper_external_id = 0;
}

void flush_cupti_activity() {
  if (cupti_available().load(std::memory_order_acquire)) {
    (void)cuptiActivityFlushAll(0);
  }
}

void clear_cupti_buffers() {
  std::lock_guard<std::mutex> guard(cupti_buffer_mutex());
  for (void *buffer : cupti_buffers()) {
    std::free(buffer);
  }
  cupti_buffers().clear();
}

#else

bool push_external_id(unsigned long long, std::string &unavailable_reason) {
  unavailable_reason = "cupti_activity_collector_not_compiled";
  return false;
}

void set_current_callback_external_id(unsigned long long, std::string_view) {}
bool callback_correlation_available() { return false; }
void pop_external_id() {}
void clear_current_callback_external_id() {}
void flush_cupti_activity() {}
void clear_cupti_buffers() {}

#endif

} // namespace

CuptiActivityMetadataObservation
begin_cupti_activity_metadata_observation(std::string_view api_name) {
  CuptiActivityMetadataObservation observation{};
  if (!cupti_activity_metadata_enabled() ||
      !api_supports_cupti_activity_metadata(api_name)) {
    return observation;
  }

  observation.active = true;
  observation.wrapper_logical_event_id = next_logical_event_id();
  observation.external_correlation_id = observation.wrapper_logical_event_id;
  observation.api_name = std::string(canonicalize_api_name(api_name));
  observation.api_role = std::string(api_role(api_name));
  observation.begin_thread_id =
      to_string_u64(static_cast<unsigned long long>(current_thread_id()));
  observation.status = "unavailable";
  observation.unavailable_reason = "cupti_activity_collector_not_compiled";
  std::string unavailable_reason{};
  const auto external_id = parse_u64(observation.external_correlation_id);
  if (push_external_id(parse_u64(observation.external_correlation_id),
                       unavailable_reason)) {
    observation.status = "enabled_pending_flush";
    observation.unavailable_reason.clear();
    observation.external_correlation_pushed = true;
    set_current_callback_external_id(external_id, observation.api_name);
  } else if (!unavailable_reason.empty()) {
    observation.unavailable_reason = unavailable_reason;
  }
  return observation;
}

void complete_cupti_activity_metadata_observation(
    CuptiActivityMetadataObservation &observation,
    cpp_event::EventPayload &payload,
    bool success) {
  if (!observation.active) {
    return;
  }
  if (observation.external_correlation_pushed) {
    pop_external_id();
  }
  clear_current_callback_external_id();

  set_attr(payload, "cupti_activity_metadata_schema_version", kSchemaVersion);
  set_attr(payload, "cupti_activity_metadata_status", observation.status);
  if (!observation.unavailable_reason.empty()) {
    set_attr(payload, "cupti_activity_metadata_unavailable_reason",
             observation.unavailable_reason);
  }
  set_attr(payload, "cupti_activity_metadata_api", observation.api_name);
  set_attr(payload, "cupti_activity_metadata_api_role", observation.api_role);
  set_attr(payload, "cupti_activity_wrapper_begin_thread_id",
           observation.begin_thread_id);
  set_attr(payload, "cupti_activity_wrapper_complete_thread_id",
           to_string_u64(static_cast<unsigned long long>(current_thread_id())));
  set_attr(payload, "cupti_wrapper_logical_event_id",
           observation.wrapper_logical_event_id);
  set_attr(payload, "cupti_external_correlation_id",
           observation.external_correlation_id);
  set_attr(payload, "cupti_activity_metadata_forward_call_success",
           success ? "true" : "false");
  set_attr(payload, "cupti_activity_metadata_event_stream_evidence_domain",
           "app_payload_handles_only");
  set_attr(payload, "cupti_activity_metadata_raw_event_id_source",
           "capture_real_rank_trace_serialization");
  set_attr(payload, "cupti_activity_callback_correlation_strategy",
           callback_correlation_available() ? "runtime_api_callback_correlation_id"
                                            : "unavailable");
  set_attr(payload, "cupti_activity_metadata_runtime_substitution", "false");
  set_attr(payload, "cupti_activity_metadata_endpoint_substitution", "false");
  set_attr(payload, "cupti_activity_metadata_replay_substitution", "false");
  set_attr(payload, "cupti_activity_metadata_strict_actual_wait_timing",
           "unavailable");
  observation = CuptiActivityMetadataObservation{};
}

void resolve_cupti_activity_metadata_observations(cpp_event::EventLog &log) {
  flush_cupti_activity();
  log.with_mutable_events([](cpp_event::EventList &events) {
    std::lock_guard<std::mutex> guard(activity_mutex());
    auto &summaries = activity_by_external_id();
    const auto total_external =
        total_external_activity_records().load(std::memory_order_acquire);
    const auto total_callback_correlations =
        total_callback_correlation_records().load(std::memory_order_acquire);
    const auto total_runtime_callbacks =
        total_runtime_callback_records().load(std::memory_order_acquire);
    const auto total_runtime_callbacks_with_wrapper =
        total_runtime_callback_with_wrapper_records().load(
            std::memory_order_acquire);
    const auto total_runtime_callbacks_with_thread_window =
        total_runtime_callback_with_thread_window_records().load(
            std::memory_order_acquire);
    const auto total_target_runtime_callbacks =
        total_target_runtime_callback_records().load(std::memory_order_acquire);
    const auto total_target_runtime_callbacks_with_thread_window =
        total_target_runtime_callback_with_thread_window_records().load(
            std::memory_order_acquire);
    const auto total_callsite_fifo_correlations =
        total_callsite_fifo_correlation_records().load(std::memory_order_acquire);
    const auto total_callsite_fifo_candidates =
        total_callsite_fifo_candidate_records().load(std::memory_order_acquire);
    const auto total_runtime =
        total_runtime_activity_records().load(std::memory_order_acquire);
    const auto total_kernel =
        total_kernel_activity_records().load(std::memory_order_acquire);
    const auto pending_correlations = pending_activity_by_correlation_id().size();
    for (auto &event : events) {
      auto id_it = event.payload.attributes.find("cupti_external_correlation_id");
      if (id_it == event.payload.attributes.end()) {
        continue;
      }
      event.payload.attributes["cupti_activity_total_external_record_count"] =
          to_string_u64(total_external);
      event.payload
          .attributes["cupti_activity_total_callback_correlation_record_count"] =
          to_string_u64(total_callback_correlations);
      event.payload.attributes["cupti_activity_total_runtime_callback_count"] =
          to_string_u64(total_runtime_callbacks);
      event.payload.attributes
          ["cupti_activity_total_runtime_callback_with_wrapper_count"] =
          to_string_u64(total_runtime_callbacks_with_wrapper);
      event.payload.attributes
          ["cupti_activity_total_runtime_callback_with_thread_window_count"] =
          to_string_u64(total_runtime_callbacks_with_thread_window);
      event.payload.attributes["cupti_activity_total_target_runtime_callback_count"] =
          to_string_u64(total_target_runtime_callbacks);
      event.payload.attributes
          ["cupti_activity_total_target_runtime_callback_with_thread_window_count"] =
          to_string_u64(total_target_runtime_callbacks_with_thread_window);
      event.payload.attributes
          ["cupti_activity_total_callsite_fifo_correlation_record_count"] =
          to_string_u64(total_callsite_fifo_correlations);
      event.payload.attributes
          ["cupti_activity_total_callsite_fifo_candidate_record_count"] =
          to_string_u64(total_callsite_fifo_candidates);
      event.payload.attributes["cupti_activity_total_runtime_record_count"] =
          to_string_u64(total_runtime);
      event.payload.attributes["cupti_activity_total_kernel_record_count"] =
          to_string_u64(total_kernel);
      event.payload.attributes["cupti_activity_pending_correlation_count"] =
          to_string_u64(static_cast<unsigned long long>(pending_correlations));
      unsigned long long external_id = parse_u64(id_it->second);
      auto summary_it = summaries.find(external_id);
      if (summary_it == summaries.end()) {
        event.payload.attributes["cupti_activity_metadata_real_activity_status"] =
            "no_activity_records_matched";
        auto &debug = callback_debug_state();
        event.payload.attributes["cupti_activity_first_callback_thread_id"] =
            debug.first_callback_thread_id;
        event.payload.attributes["cupti_activity_first_callback_function"] =
            debug.first_callback_function;
        event.payload.attributes["cupti_activity_first_callback_site"] =
            debug.first_callback_site;
        event.payload.attributes["cupti_activity_first_callback_tls_external_id"] =
            debug.first_callback_tls_external_id;
        event.payload.attributes
            ["cupti_activity_first_callback_thread_window_external_id"] =
            debug.first_callback_thread_window_external_id;
        event.payload.attributes["cupti_activity_first_window_callback_thread_id"] =
            debug.first_window_callback_thread_id;
        event.payload.attributes["cupti_activity_first_window_callback_function"] =
            debug.first_window_callback_function;
        event.payload.attributes["cupti_activity_first_window_callback_site"] =
            debug.first_window_callback_site;
        event.payload.attributes["cupti_activity_first_target_callback_thread_id"] =
            debug.first_target_callback_thread_id;
        event.payload.attributes["cupti_activity_first_target_callback_function"] =
            debug.first_target_callback_function;
        event.payload.attributes["cupti_activity_first_target_callback_site"] =
            debug.first_target_callback_site;
        event.payload.attributes
            ["cupti_activity_first_target_window_callback_thread_id"] =
            debug.first_target_window_callback_thread_id;
        event.payload.attributes
            ["cupti_activity_first_target_window_callback_function"] =
            debug.first_target_window_callback_function;
        event.payload.attributes
            ["cupti_activity_first_target_window_callback_site"] =
            debug.first_target_window_callback_site;
        event.payload.attributes
            ["cupti_activity_first_callsite_fifo_callback_thread_id"] =
            debug.first_callsite_fifo_callback_thread_id;
        event.payload.attributes
            ["cupti_activity_first_callsite_fifo_callback_function"] =
            debug.first_callsite_fifo_callback_function;
        event.payload.attributes["cupti_activity_first_callsite_fifo_callback_site"] =
            debug.first_callsite_fifo_callback_site;
        event.payload.attributes["cupti_activity_first_callsite_fifo_external_id"] =
            debug.first_callsite_fifo_external_id;
        std::ostringstream samples;
        for (std::size_t index = 0; index < debug.callback_samples.size(); ++index) {
          if (index != 0) {
            samples << " | ";
          }
          samples << debug.callback_samples[index];
        }
        event.payload.attributes["cupti_activity_callback_order_samples"] =
            samples.str();
        continue;
      }
      const ActivitySummary &summary = summary_it->second;
      event.payload.attributes["cupti_activity_metadata_real_activity_status"] =
          "matched";
      event.payload.attributes["cupti_activity_external_record_count"] =
          to_string_u64(summary.external_records);
      event.payload.attributes["cupti_activity_callback_correlation_record_count"] =
          to_string_u64(summary.callback_correlation_records);
      event.payload.attributes["cupti_activity_runtime_record_count"] =
          to_string_u64(summary.runtime_records);
      event.payload.attributes["cupti_activity_kernel_record_count"] =
          to_string_u64(summary.kernel_records);
      (void)summary.dropped_records;
      if (!summary.first_runtime_start.empty()) {
        event.payload.attributes["cupti_activity_first_runtime_start"] =
            summary.first_runtime_start;
        event.payload.attributes["cupti_activity_first_runtime_end"] =
            summary.first_runtime_end;
      }
      if (!summary.first_kernel_start.empty()) {
        event.payload.attributes["cupti_activity_first_kernel_start"] =
            summary.first_kernel_start;
        event.payload.attributes["cupti_activity_first_kernel_end"] =
            summary.first_kernel_end;
        event.payload.attributes["cupti_activity_last_kernel_start"] =
            summary.last_kernel_start;
        event.payload.attributes["cupti_activity_last_kernel_end"] =
            summary.last_kernel_end;
        event.payload.attributes["cupti_activity_first_kernel_stream_id"] =
            summary.first_kernel_stream_id;
        event.payload.attributes["cupti_activity_last_kernel_stream_id"] =
            summary.last_kernel_stream_id;
        event.payload.attributes["cupti_activity_kernel_stream_id_unique_count"] =
            to_string_u64(
                static_cast<unsigned long long>(summary.kernel_stream_ids.size()));
        event.payload.attributes["cupti_activity_kernel_stream_id_status"] =
            summary.kernel_stream_ids.size() == 1 ? "single_stream"
                                                  : "multiple_streams";
        event.payload.attributes["cupti_activity_device_activity_timing_status"] =
            "diagnostic_activity_timestamps";
        event.payload.attributes["cupti_activity_kernel_stream_id_basis"] =
            "cupti_activity_kernel_stream_id";
      }
      event.payload.attributes["cupti_activity_app_stream_event_basis"] =
          "wrapper_payload_handles_only";
      event.payload.attributes["cupti_activity_common_clock_status"] =
          "unreviewed";
      event.payload.attributes["cupti_activity_strict_wait_timing"] =
          "unavailable";
    }
    unsigned long long dropped = 0;
#if defined(CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY) && CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY
    dropped = cupti_dropped_records().load(std::memory_order_acquire);
#endif
    for (auto &event : events) {
      if (event.payload.attributes.find("cupti_external_correlation_id") ==
          event.payload.attributes.end()) {
        continue;
      }
      event.payload.attributes["cupti_activity_global_dropped_record_count"] =
          to_string_u64(dropped);
      event.payload.attributes["cupti_activity_dropped_record_count_basis"] =
          "global_since_last_clear";
    }
  });
}

void clear_cupti_activity_metadata_observations() noexcept {
  logical_event_counter().store(0, std::memory_order_release);
  std::lock_guard<std::mutex> guard(activity_mutex());
  activity_by_external_id().clear();
  correlation_to_external_id().clear();
  thread_window_external_id().clear();
  completed_callsite_fifo_by_thread_api().clear();
  pending_activity_by_correlation_id().clear();
  total_external_activity_records().store(0, std::memory_order_release);
  total_callback_correlation_records().store(0, std::memory_order_release);
  total_runtime_callback_records().store(0, std::memory_order_release);
  total_runtime_callback_with_wrapper_records().store(0,
                                                     std::memory_order_release);
  total_runtime_callback_with_thread_window_records().store(
      0, std::memory_order_release);
  total_target_runtime_callback_records().store(0, std::memory_order_release);
  total_target_runtime_callback_with_thread_window_records().store(
      0, std::memory_order_release);
  total_callsite_fifo_correlation_records().store(0,
                                                 std::memory_order_release);
  total_callsite_fifo_candidate_records().store(0,
                                               std::memory_order_release);
  total_runtime_activity_records().store(0, std::memory_order_release);
  total_kernel_activity_records().store(0, std::memory_order_release);
  callback_debug_state() = CallbackDebugState{};
  clear_cupti_buffers();
#if defined(CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY) && CPPEVENT_ENABLE_REAL_CUPTI_ACTIVITY
  cupti_dropped_records().store(0, std::memory_order_release);
#endif
}

} // namespace lowlevel::interface
