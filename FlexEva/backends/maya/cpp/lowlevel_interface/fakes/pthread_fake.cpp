#include <pthread.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace {

struct ThreadRecord {
  void *result{nullptr};
  bool joined{false};
};

std::atomic<std::uint64_t> g_next_id{1};
std::mutex g_records_mutex;
std::unordered_map<std::uint64_t, ThreadRecord> g_records;

std::uint64_t to_key(pthread_t value) {
  std::uint64_t key = 0;
  static_assert(sizeof(key) >= sizeof(value),
                "pthread_t larger than supported key size");
  std::memcpy(&key, &value, sizeof(value));
  return key;
}

pthread_t from_key(std::uint64_t key) {
  pthread_t value{};
  static_assert(sizeof(key) >= sizeof(value),
                "pthread_t larger than supported key size");
  std::memcpy(&value, &key, sizeof(value));
  return value;
}

} // namespace

extern "C" {

int pthread_create(pthread_t *thread, const pthread_attr_t * /*attr*/,
                   void *(*start_routine)(void *), void *arg) {
  const auto id = g_next_id.fetch_add(1, std::memory_order_relaxed);
  *thread = from_key(id);

  void *result = start_routine(arg);

  {
    std::lock_guard<std::mutex> lock(g_records_mutex);
    g_records[id] = ThreadRecord{result, false};
  }

  return 0;
}

int pthread_join(pthread_t thread, void **retval) {
  const auto key = to_key(thread);
  std::optional<ThreadRecord> record;
  {
    std::lock_guard<std::mutex> lock(g_records_mutex);
    auto it = g_records.find(key);
    if (it == g_records.end()) {
      return ESRCH;
    }
    record = it->second;
    g_records.erase(it);
  }

  if (retval != nullptr) {
    *retval = record->result;
  }
  return 0;
}

pthread_t pthread_self(void) {
  // The fake implementation runs everything on the caller thread. Assign a
  // sentinel identifier of zero to represent the main thread.
  return from_key(0);
}

int pthread_equal(pthread_t t1, pthread_t t2) {
  return to_key(t1) == to_key(t2);
}

} // extern "C"
