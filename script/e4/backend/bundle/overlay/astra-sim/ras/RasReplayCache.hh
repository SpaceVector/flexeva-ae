/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __RAS_REPLAY_CACHE_HH__
#define __RAS_REPLAY_CACHE_HH__

#include <cstdint>
#include <json/json.hpp>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace AstraSim {

class RasReplayCache {
  public:
    struct WorkloadTiming {
        uint64_t duration = 0;
        uint64_t issue_tick = 0;
        uint64_t finish_tick = 0;
        uint64_t rank = 0;
        uint64_t node_id = 0;
    };

    static RasReplayCache& get();
    static void set_current_context_fingerprint(
        const std::string& fingerprint);

    bool load_from_environment();
    bool load_from_files(const std::string& replay_cache_path,
                         const std::string& reuse_plan_path);
    bool load_from_json(const nlohmann::json& replay_cache,
                        const nlohmann::json& reuse_plan);
    void clear();

    bool enabled() const;
    bool incremental_execution_enabled() const;
    bool lookup_workload_timing(const std::string& partition_id,
                                WorkloadTiming* timing) const;
    bool lookup_workload_duration(const std::string& partition_id,
                                  uint64_t* duration) const;
    bool lookup_collective_workload_duration(
        const std::string& partition_id,
        const std::string& collective_group_key,
        uint64_t* duration) const;
    bool workload_partition_is_reusable(
        const std::string& partition_id) const;
    bool workload_partition_should_execute(
        const std::string& partition_id) const;
    bool incremental_workload_partition_can_reuse_duration(
        const std::string& partition_id,
        const std::string& collective_group_key) const;
    std::unordered_set<uint64_t> executable_workload_node_ids_for_rank(
        uint64_t rank) const;

  private:
    struct State {
        bool is_enabled = false;
        bool incremental_execution_enabled = false;
        std::string context_fingerprint;
        std::unordered_map<std::string, WorkloadTiming> workload_timings;
        std::unordered_map<std::string, uint64_t> workload_durations;
        std::unordered_set<std::string> reusable_workload_partitions;
        std::unordered_set<std::string> rerun_workload_partitions;
        std::unordered_set<std::string> refresh_workload_partitions;
        std::unordered_set<std::string> executable_workload_partitions;
        std::unordered_map<uint64_t, std::unordered_set<uint64_t>>
            executable_workload_node_ids_by_rank;
        std::unordered_map<std::string, std::string>
            collective_group_key_by_partition_id;
        std::unordered_map<std::string, std::unordered_set<uint64_t>>
            expected_ranks_by_collective_group_key;
        std::unordered_map<std::string, std::unordered_set<uint64_t>>
            cached_ranks_by_collective_group_key;
        std::unordered_map<std::string, std::unordered_set<uint64_t>>
            reusable_ranks_by_collective_group_key;
        std::unordered_set<std::string>
            fully_reusable_collective_group_keys;
    };

    RasReplayCache();

    bool build_state(const nlohmann::json& replay_cache,
                     const nlohmann::json& reuse_plan,
                     State* next_state) const;

    mutable std::mutex mutex;
    State state;
};

}  // namespace AstraSim

#endif /* __RAS_REPLAY_CACHE_HH__ */
