/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/ras/RasReplayCache.hh"

#include <json/json.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using namespace AstraSim;
using json = nlohmann::json;

namespace {

constexpr const char* kReplayCacheSchema = "ras-replay-cache-v1";
constexpr const char* kDiffSchema = "ras-diff-v1";
constexpr const char* kDefaultReusePolicy =
    "cache_minus_rerun_refresh_removed";

bool env_path_is_set(const char* path) {
    return path != nullptr && path[0] != '\0';
}

std::string& current_context_fingerprint_storage() {
    static std::string fingerprint;
    return fingerprint;
}

bool collect_partition_ids(const json& partitions,
                           std::unordered_set<std::string>* partition_ids) {
    if (partitions.is_array()) {
        for (const auto& item : partitions) {
            if (item.is_string()) {
                partition_ids->insert(item.get<std::string>());
            } else if (item.is_object()) {
                if (item.contains("partition_id") &&
                    item["partition_id"].is_string()) {
                    partition_ids->insert(
                        item["partition_id"].get<std::string>());
                } else if (item.contains("id") && item["id"].is_string()) {
                    partition_ids->insert(item["id"].get<std::string>());
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }
        return true;
    } else if (partitions.is_object()) {
        for (auto it = partitions.begin(); it != partitions.end(); ++it) {
            if (!it.value().is_boolean()) {
                return false;
            }
            if (it.value().get<bool>()) {
                partition_ids->insert(it.key());
            }
        }
        return true;
    }
    return false;
}

bool collect_excluded_compact_policy_partition_ids(
    const json& reuse_plan,
    std::unordered_set<std::string>* excluded_partitions) {
    const char* fields[] = {"rerun_partitions", "refresh_partitions",
                            "removed_baseline_partitions"};
    for (const char* field : fields) {
        if (!reuse_plan.contains(field)) {
            return false;
        }
        if (!collect_partition_ids(reuse_plan[field], excluded_partitions)) {
            return false;
        }
    }
    return true;
}

bool get_optional_string(const json& value,
                         const char* field,
                         std::string* output) {
    if (!value.is_object() || !value.contains(field)) {
        return false;
    }
    if (!value[field].is_string()) {
        return false;
    }
    *output = value[field].get<std::string>();
    return true;
}

bool get_collective_group_key(const json& partition,
                              std::string* collective_group_key) {
    if (get_optional_string(partition, "collective_group_key",
                            collective_group_key) ||
        get_optional_string(partition, "collective_key",
                            collective_group_key)) {
        return true;
    }
    if (partition.contains("collective_group") &&
        partition["collective_group"].is_object()) {
        return get_optional_string(partition["collective_group"], "key",
                                   collective_group_key);
    }
    return false;
}

bool insert_rank(const json& value, std::unordered_set<uint64_t>* ranks) {
    if (value.is_number_unsigned()) {
        ranks->insert(value.get<uint64_t>());
        return true;
    }
    if (value.is_number_integer()) {
        const int64_t rank = value.get<int64_t>();
        if (rank < 0) {
            return false;
        }
        ranks->insert(static_cast<uint64_t>(rank));
        return true;
    }
    return false;
}

bool collect_ranks_from_array(const json& value,
                              std::unordered_set<uint64_t>* ranks) {
    if (!value.is_array()) {
        return false;
    }
    for (const auto& item : value) {
        if (!insert_rank(item, ranks)) {
            return false;
        }
    }
    return !ranks->empty();
}

bool collect_expected_ranks_from_group_key(
    const std::string& collective_group_key,
    std::unordered_set<uint64_t>* ranks) {
    try {
        const json key = json::parse(collective_group_key);
        if (!key.is_object() || !key.contains("group_ranks")) {
            return false;
        }
        return collect_ranks_from_array(key["group_ranks"], ranks);
    } catch (const std::exception&) {
        return false;
    }
}

bool collect_node_id_from_group_key(const std::string& collective_group_key,
                                    uint64_t* node_id) {
    if (node_id == nullptr) {
        return false;
    }
    try {
        const json key = json::parse(collective_group_key);
        if (!key.is_object() || !key.contains("node_id")) {
            return false;
        }
        if (key["node_id"].is_number_unsigned()) {
            *node_id = key["node_id"].get<uint64_t>();
            return true;
        }
        if (key["node_id"].is_number_integer()) {
            const int64_t value = key["node_id"].get<int64_t>();
            if (value < 0) {
                return false;
            }
            *node_id = static_cast<uint64_t>(value);
            return true;
        }
    } catch (const std::exception&) {
        return false;
    }
    return false;
}

bool collect_expected_ranks_from_metadata(
    const json& partition,
    std::unordered_set<uint64_t>* ranks) {
    if (partition.contains("collective_group_ranks") &&
        collect_ranks_from_array(partition["collective_group_ranks"], ranks)) {
        return true;
    }
    if (partition.contains("collective_group") &&
        partition["collective_group"].is_object() &&
        partition["collective_group"].contains("ranks") &&
        collect_ranks_from_array(partition["collective_group"]["ranks"],
                                 ranks)) {
        return true;
    }
    return false;
}

bool collect_expected_ranks(const json& partition,
                            const std::string& collective_group_key,
                            std::unordered_set<uint64_t>* ranks) {
    return collect_expected_ranks_from_group_key(collective_group_key,
                                                 ranks) ||
           collect_expected_ranks_from_metadata(partition, ranks);
}

bool same_rank_set(const std::unordered_set<uint64_t>& lhs,
                   const std::unordered_set<uint64_t>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (const uint64_t rank : lhs) {
        if (rhs.find(rank) == rhs.end()) {
            return false;
        }
    }
    return true;
}

bool parse_uint64_from_string(const std::string& value, uint64_t* output) {
    if (output == nullptr || value.empty()) {
        return false;
    }
    size_t parsed = 0;
    try {
        const unsigned long long parsed_value =
            std::stoull(value, &parsed, 10);
        if (parsed != value.size() ||
            parsed_value > std::numeric_limits<uint64_t>::max()) {
            return false;
        }
        *output = static_cast<uint64_t>(parsed_value);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool parse_workload_partition_id(const std::string& partition_id,
                                 uint64_t* rank,
                                 uint64_t* node_id) {
    constexpr const char* kRankPrefix = "rank:";
    constexpr const char* kChakraSep = "/chakra:";
    if (partition_id.rfind(kRankPrefix, 0) != 0) {
        return false;
    }
    const size_t sep = partition_id.find(kChakraSep);
    if (sep == std::string::npos) {
        return false;
    }
    const std::string rank_text =
        partition_id.substr(std::string(kRankPrefix).size(),
                            sep - std::string(kRankPrefix).size());
    const std::string node_text =
        partition_id.substr(sep + std::string(kChakraSep).size());
    return parse_uint64_from_string(rank_text, rank) &&
           parse_uint64_from_string(node_text, node_id);
}

}  // namespace

RasReplayCache& RasReplayCache::get() {
    static RasReplayCache cache;
    return cache;
}

void RasReplayCache::set_current_context_fingerprint(
    const std::string& fingerprint) {
    current_context_fingerprint_storage() = fingerprint;
}

RasReplayCache::RasReplayCache() {
    load_from_environment();
}

bool RasReplayCache::load_from_environment() {
    const char* cache_path = std::getenv("ASTRA_SIM_RAS_REPLAY_CACHE");
    const char* plan_path = std::getenv("ASTRA_SIM_RAS_REUSE_PLAN");
    if (!env_path_is_set(cache_path) || !env_path_is_set(plan_path)) {
        clear();
        return false;
    }
    return load_from_files(cache_path, plan_path);
}

bool RasReplayCache::load_from_files(const std::string& replay_cache_path,
                                     const std::string& reuse_plan_path) {
    std::ifstream cache_file(replay_cache_path);
    std::ifstream plan_file(reuse_plan_path);
    if (!cache_file.is_open() || !plan_file.is_open()) {
        std::cerr << "ASTRA-sim RAS replay disabled: unable to open replay "
                  << "cache or reuse plan" << std::endl;
        clear();
        return false;
    }

    try {
        json cache_root;
        json plan_root;
        cache_file >> cache_root;
        plan_file >> plan_root;
        return load_from_json(cache_root, plan_root);
    } catch (const std::exception& e) {
        std::cerr << "ASTRA-sim RAS replay disabled: " << e.what()
                  << std::endl;
        clear();
        return false;
    }
}

bool RasReplayCache::load_from_json(const json& replay_cache,
                                    const json& reuse_plan) {
    State next_state;
    if (!build_state(replay_cache, reuse_plan, &next_state)) {
        clear();
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex);
    state = std::move(next_state);
    return true;
}

void RasReplayCache::clear() {
    std::lock_guard<std::mutex> lock(mutex);
    state = State();
}

bool RasReplayCache::build_state(const json& cache_root,
                                 const json& plan_root,
                                 State* next_state) const {
    if (next_state == nullptr) {
        return false;
    }

    try {
        const bool plan_schema_valid =
            !plan_root.contains("schema_version") ||
            (plan_root["schema_version"].is_string() &&
             plan_root["schema_version"] == kDiffSchema);

        if (!cache_root.is_object() || !plan_root.is_object() ||
            !cache_root.contains("schema_version") ||
            !cache_root["schema_version"].is_string() ||
            cache_root["schema_version"] != kReplayCacheSchema ||
            !cache_root.contains("workload_partitions") ||
            !cache_root["workload_partitions"].is_object() ||
            !plan_schema_valid ||
            !plan_root.contains("reuse_plan") ||
            !plan_root["reuse_plan"].is_object()) {
            std::cerr << "ASTRA-sim RAS replay disabled: malformed replay "
                      << "cache or reuse plan" << std::endl;
            return false;
        }
        const json& reuse_plan = plan_root["reuse_plan"];
        const bool has_default_reuse_policy =
            reuse_plan.contains("default_reuse_policy");
        if ((has_default_reuse_policy &&
             (!reuse_plan["default_reuse_policy"].is_string() ||
              reuse_plan["default_reuse_policy"].get<std::string>() !=
                  kDefaultReusePolicy)) ||
            (!has_default_reuse_policy &&
             !reuse_plan.contains("reusable_partitions"))) {
            std::cerr << "ASTRA-sim RAS replay disabled: malformed replay "
                      << "cache or reuse plan" << std::endl;
            return false;
        }

        if (!cache_root.contains("context_fingerprint") ||
            !cache_root["context_fingerprint"].is_string() ||
            cache_root["context_fingerprint"].get<std::string>().empty()) {
            std::cerr << "ASTRA-sim RAS replay disabled: malformed or "
                      << "missing context_fingerprint in replay cache"
                      << std::endl;
            return false;
        }
        const std::string cache_context_fingerprint =
            cache_root["context_fingerprint"].get<std::string>();
        const std::string& current_context_fingerprint =
            current_context_fingerprint_storage();
        if (current_context_fingerprint.empty()) {
            std::cerr << "ASTRA-sim RAS replay disabled: current "
                      << "evaluation context fingerprint is missing"
                      << std::endl;
            return false;
        }
        if (current_context_fingerprint != cache_context_fingerprint) {
            std::cerr << "ASTRA-sim RAS replay disabled: replay cache "
                      << "context fingerprint does not match current "
                      << "evaluation context" << std::endl;
            return false;
        }

        std::unordered_set<std::string> reusable_partitions;
        std::unordered_set<std::string> excluded_partitions;
        std::unordered_set<std::string> rerun_partitions;
        std::unordered_set<std::string> refresh_partitions;
        std::unordered_set<std::string> executable_partitions;
        std::unordered_set<std::string> removed_partitions;
        if (!reuse_plan.contains("rerun_partitions") ||
            !collect_partition_ids(reuse_plan["rerun_partitions"],
                                   &rerun_partitions) ||
            !reuse_plan.contains("refresh_partitions") ||
            !collect_partition_ids(reuse_plan["refresh_partitions"],
                                   &refresh_partitions)) {
            std::cerr << "ASTRA-sim RAS replay disabled: malformed replay "
                      << "cache or reuse plan" << std::endl;
            return false;
        }
        executable_partitions.insert(rerun_partitions.begin(),
                                     rerun_partitions.end());
        executable_partitions.insert(refresh_partitions.begin(),
                                     refresh_partitions.end());
        if (has_default_reuse_policy) {
            if (!collect_excluded_compact_policy_partition_ids(
                    reuse_plan, &excluded_partitions)) {
                std::cerr
                    << "ASTRA-sim RAS replay disabled: malformed replay "
                    << "cache or reuse plan" << std::endl;
                return false;
            }
        } else if (!collect_partition_ids(reuse_plan["reusable_partitions"],
                                          &reusable_partitions)) {
            std::cerr << "ASTRA-sim RAS replay disabled: malformed replay "
                      << "cache or reuse plan" << std::endl;
            return false;
        }
        if (reuse_plan.contains("removed_baseline_partitions") &&
            !collect_partition_ids(reuse_plan["removed_baseline_partitions"],
                                   &removed_partitions)) {
            std::cerr << "ASTRA-sim RAS replay disabled: malformed replay "
                      << "cache or reuse plan" << std::endl;
            return false;
        }

        const json& workload_partitions = cache_root["workload_partitions"];
        for (auto it = workload_partitions.begin();
             it != workload_partitions.end(); ++it) {
            if (!it.value().is_object() ||
                !it.value().contains("duration") ||
                !it.value()["duration"].is_number_unsigned() ||
                !it.value().contains("issue_tick") ||
                !it.value()["issue_tick"].is_number_unsigned() ||
                !it.value().contains("finish_tick") ||
                !it.value()["finish_tick"].is_number_unsigned() ||
                !it.value().contains("node_id") ||
                !it.value()["node_id"].is_number_unsigned() ||
                !it.value().contains("rank") ||
                !it.value()["rank"].is_number_unsigned()) {
                std::cerr << "ASTRA-sim RAS replay disabled: malformed "
                          << "workload partition in replay cache"
                          << std::endl;
                return false;
            }

            const uint64_t rank = it.value()["rank"].get<uint64_t>();
            const uint64_t node_id = it.value()["node_id"].get<uint64_t>();
            RasReplayCache::WorkloadTiming timing;
            timing.duration = it.value()["duration"].get<uint64_t>();
            timing.issue_tick = it.value()["issue_tick"].get<uint64_t>();
            timing.finish_tick = it.value()["finish_tick"].get<uint64_t>();
            timing.rank = rank;
            timing.node_id = node_id;
            next_state->workload_timings[it.key()] = timing;
            std::string collective_group_key;
            const bool has_collective_group_key =
                get_collective_group_key(it.value(), &collective_group_key);
            if (has_collective_group_key) {
                next_state->collective_group_key_by_partition_id[it.key()] =
                    collective_group_key;
                next_state
                    ->cached_ranks_by_collective_group_key[collective_group_key]
                    .insert(rank);

                std::unordered_set<uint64_t> expected_ranks;
                if (collect_expected_ranks(it.value(), collective_group_key,
                                           &expected_ranks)) {
                    auto& known_expected_ranks =
                        next_state->expected_ranks_by_collective_group_key[
                            collective_group_key];
                    if (known_expected_ranks.empty()) {
                        known_expected_ranks = expected_ranks;
                    } else if (!same_rank_set(known_expected_ranks,
                                              expected_ranks)) {
                        std::cerr
                            << "ASTRA-sim RAS replay disabled: inconsistent "
                            << "collective rank metadata in replay cache"
                            << std::endl;
                        return false;
                    }
                }
            }

            const bool partition_is_reusable =
                has_default_reuse_policy
                    ? excluded_partitions.find(it.key()) ==
                          excluded_partitions.end()
                    : reusable_partitions.find(it.key()) !=
                          reusable_partitions.end();
            if (!partition_is_reusable) {
                if (removed_partitions.find(it.key()) ==
                    removed_partitions.end()) {
                    executable_partitions.insert(it.key());
                }
                continue;
            }

            if (has_collective_group_key) {
                next_state
                    ->reusable_ranks_by_collective_group_key[
                        collective_group_key]
                    .insert(rank);
            }
            next_state->reusable_workload_partitions.insert(it.key());
            next_state->workload_durations[it.key()] =
                timing.duration;
        }
        for (const auto& item :
             next_state->expected_ranks_by_collective_group_key) {
            const auto& key = item.first;
            const auto& expected_ranks = item.second;
            const auto cached_it =
                next_state->cached_ranks_by_collective_group_key.find(key);
            const auto reusable_it =
                next_state->reusable_ranks_by_collective_group_key.find(key);
            if (cached_it !=
                    next_state->cached_ranks_by_collective_group_key.end() &&
                reusable_it !=
                    next_state->reusable_ranks_by_collective_group_key.end() &&
                same_rank_set(cached_it->second, expected_ranks) &&
                same_rank_set(reusable_it->second, expected_ranks)) {
                next_state->fully_reusable_collective_group_keys.insert(key);
            }
        }

        for (const auto& item :
             next_state->expected_ranks_by_collective_group_key) {
            const auto& key = item.first;
            if (next_state->fully_reusable_collective_group_keys.find(key) !=
                next_state->fully_reusable_collective_group_keys.end()) {
                continue;
            }

            uint64_t node_id = 0;
            if (!collect_node_id_from_group_key(key, &node_id)) {
                std::cerr << "ASTRA-sim RAS replay disabled: malformed "
                          << "collective group key in replay cache"
                          << std::endl;
                return false;
            }

            for (const uint64_t rank : item.second) {
                const std::string partition_id =
                    "rank:" + std::to_string(rank) +
                    "/chakra:" + std::to_string(node_id);
                if (removed_partitions.find(partition_id) ==
                    removed_partitions.end()) {
                    executable_partitions.insert(partition_id);
                }
            }
        }

        for (const auto& partition_id : executable_partitions) {
            next_state->reusable_workload_partitions.erase(partition_id);
            next_state->workload_durations.erase(partition_id);
        }
        next_state->executable_workload_partitions = executable_partitions;
        next_state->rerun_workload_partitions = std::move(rerun_partitions);
        next_state->refresh_workload_partitions =
            std::move(refresh_partitions);

        for (const auto& partition_id : executable_partitions) {
            uint64_t rank = 0;
            uint64_t node_id = 0;
            const auto timing_it =
                next_state->workload_timings.find(partition_id);
            if (timing_it != next_state->workload_timings.end()) {
                rank = timing_it->second.rank;
                node_id = timing_it->second.node_id;
            } else if (!parse_workload_partition_id(partition_id, &rank,
                                                    &node_id)) {
                continue;
            }
            next_state->executable_workload_node_ids_by_rank[rank].insert(
                node_id);
        }
    } catch (const std::exception& e) {
        std::cerr << "ASTRA-sim RAS replay disabled: " << e.what()
                  << std::endl;
        return false;
    }

    next_state->context_fingerprint = current_context_fingerprint_storage();
    next_state->is_enabled = true;
    next_state->incremental_execution_enabled = true;
    return true;
}

bool RasReplayCache::enabled() const {
    std::lock_guard<std::mutex> lock(mutex);
    return state.is_enabled &&
           state.context_fingerprint == current_context_fingerprint_storage();
}

bool RasReplayCache::incremental_execution_enabled() const {
    std::lock_guard<std::mutex> lock(mutex);
    return state.is_enabled && state.incremental_execution_enabled &&
           state.context_fingerprint == current_context_fingerprint_storage();
}

bool RasReplayCache::lookup_workload_timing(
    const std::string& partition_id,
    WorkloadTiming* timing) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (!state.is_enabled ||
        state.context_fingerprint != current_context_fingerprint_storage() ||
        timing == nullptr) {
        return false;
    }
    const auto it = state.workload_timings.find(partition_id);
    if (it == state.workload_timings.end()) {
        return false;
    }
    *timing = it->second;
    return true;
}

bool RasReplayCache::lookup_workload_duration(
    const std::string& partition_id,
    uint64_t* duration) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (!state.is_enabled ||
        state.context_fingerprint != current_context_fingerprint_storage() ||
        duration == nullptr) {
        return false;
    }
    const auto it = state.workload_durations.find(partition_id);
    if (it == state.workload_durations.end()) {
        return false;
    }
    *duration = it->second;
    return true;
}

bool RasReplayCache::lookup_collective_workload_duration(
    const std::string& partition_id,
    const std::string& collective_group_key,
    uint64_t* duration) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (!state.is_enabled ||
        state.context_fingerprint != current_context_fingerprint_storage() ||
        duration == nullptr ||
        collective_group_key.empty()) {
        return false;
    }
    const auto partition_key_it =
        state.collective_group_key_by_partition_id.find(partition_id);
    if (partition_key_it ==
            state.collective_group_key_by_partition_id.end() ||
        partition_key_it->second != collective_group_key) {
        return false;
    }
    if (state.fully_reusable_collective_group_keys.find(
            collective_group_key) ==
        state.fully_reusable_collective_group_keys.end()) {
        return false;
    }
    const auto duration_it = state.workload_durations.find(partition_id);
    if (duration_it == state.workload_durations.end()) {
        return false;
    }
    *duration = duration_it->second;
    return true;
}

bool RasReplayCache::workload_partition_is_reusable(
    const std::string& partition_id) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (!state.is_enabled ||
        state.context_fingerprint != current_context_fingerprint_storage()) {
        return false;
    }
    if (state.reusable_workload_partitions.find(partition_id) ==
        state.reusable_workload_partitions.end()) {
        return false;
    }
    const auto partition_key_it =
        state.collective_group_key_by_partition_id.find(partition_id);
    if (partition_key_it ==
        state.collective_group_key_by_partition_id.end()) {
        return true;
    }
    return state.fully_reusable_collective_group_keys.find(
               partition_key_it->second) !=
           state.fully_reusable_collective_group_keys.end();
}

bool RasReplayCache::workload_partition_should_execute(
    const std::string& partition_id) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (!state.is_enabled ||
        state.context_fingerprint != current_context_fingerprint_storage()) {
        return false;
    }
    return state.executable_workload_partitions.find(partition_id) !=
           state.executable_workload_partitions.end();
}

bool RasReplayCache::incremental_workload_partition_can_reuse_duration(
    const std::string& partition_id,
    const std::string& collective_group_key) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (!state.is_enabled ||
        state.context_fingerprint != current_context_fingerprint_storage()) {
        return false;
    }
    if (state.refresh_workload_partitions.find(partition_id) ==
        state.refresh_workload_partitions.end()) {
        return false;
    }
    if (state.rerun_workload_partitions.find(partition_id) !=
        state.rerun_workload_partitions.end()) {
        return false;
    }
    if (state.workload_timings.find(partition_id) ==
        state.workload_timings.end()) {
        return false;
    }

    const auto partition_key_it =
        state.collective_group_key_by_partition_id.find(partition_id);
    if (partition_key_it ==
        state.collective_group_key_by_partition_id.end()) {
        return collective_group_key.empty();
    }
    if (partition_key_it->second != collective_group_key) {
        return false;
    }
    return state.fully_reusable_collective_group_keys.find(
               collective_group_key) !=
           state.fully_reusable_collective_group_keys.end();
}

std::unordered_set<uint64_t>
RasReplayCache::executable_workload_node_ids_for_rank(uint64_t rank) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (!state.is_enabled ||
        state.context_fingerprint != current_context_fingerprint_storage()) {
        return {};
    }
    const auto it = state.executable_workload_node_ids_by_rank.find(rank);
    if (it == state.executable_workload_node_ids_by_rank.end()) {
        return {};
    }
    return it->second;
}
