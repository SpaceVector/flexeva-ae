#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include "../../../cpp/fake_cuda/include/common/utils.hpp"
#include "../../../cpp/fake_cuda/include/common/host_timing.hpp"
#include "../../../cpp/fake_cuda/include/common/trace_log.hpp"

extern "C" int fake_getDevProps(CUdevice_attribute attrib, int dev) {
    (void)dev;
    switch (attrib) {
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
            return 8;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
            return 0;
        default:
            return 0;
    }
}

namespace {

bool extract_json_number(const std::string& line, const char* field, double* value_out) {
    const std::string key = std::string("\"") + field + "\":";
    const std::size_t start = line.find(key);
    if (start == std::string::npos) {
        return false;
    }
    const std::size_t value_start = start + key.size();
    std::size_t value_end = value_start;
    while (value_end < line.size() &&
           (std::isdigit(static_cast<unsigned char>(line[value_end])) ||
            line[value_end] == '-' ||
            line[value_end] == '+' ||
            line[value_end] == '.')) {
        ++value_end;
    }
    if (value_end == value_start) {
        return false;
    }
    *value_out = std::atof(line.substr(value_start, value_end - value_start).c_str());
    return true;
}

std::vector<std::string> read_trace_lines(const char* path) {
    std::ifstream handle(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(handle, line)) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    return lines;
}

bool run_logger_entry_hook_smoke(const char* trace_path, bool expect_enabled) {
    std::remove(trace_path);
    setenv("FAKECUDA_HOST_TIMING_MODE", "measure", 1);
    setenv("FAKECUDA_TRACE_PATH", trace_path, 1);
    fakecuda::trace::trace_writer_overhead_ns() = 0LL;

    const long long pop_wall_start_ns = fakecuda::host_timing::real_now_ns();
    LOG_DEBUG(CUDART, "__cudaPopCallConfiguration() called.");
    usleep(1000);
    TracePayloadBuilder pop_payload;
    pop_payload.add_uint("grid_x", 1);
    pop_payload.add_uint("block_x", 1);
    const long long pop_ts = TraceLogger::instance().log_lightweight_with_payload(
        CUDART,
        "__cudaPopCallConfiguration",
        "other",
        pop_payload
    );
    const long long pop_wall_end_ns = fakecuda::host_timing::real_now_ns();

    const long long get_wall_start_ns = fakecuda::host_timing::real_now_ns();
    fakecuda::trace::maybe_record_wrapper_entry_api("cudaGetDevice");
    usleep(1000);
    TracePayloadBuilder get_payload;
    get_payload.add_int("device", 0);
    const long long get_ts = TraceLogger::instance().log_lightweight_with_payload(
        CUDART,
        "cudaGetDevice",
        "context_op",
        get_payload
    );
    const long long get_wall_end_ns = fakecuda::host_timing::real_now_ns();

    if (pop_ts < 0 || get_ts < 0 || get_ts < pop_ts) {
        std::cerr << "logger smoke emitted invalid timestamps: pop=" << pop_ts
                  << " get=" << get_ts << "\n";
        return false;
    }
    if (fakecuda::trace::current_trace_writer_overhead_ns() <= 0LL) {
        std::cerr << "logger smoke did not accumulate trace producer overhead\n";
        return false;
    }

    const std::vector<std::string> lines = read_trace_lines(trace_path);
    if (lines.size() != 2U) {
        std::cerr << "logger smoke expected two trace lines, got " << lines.size() << "\n";
        return false;
    }

    double pop_host_duration_us = 0.0;
    double get_host_duration_us = 0.0;
    if (!extract_json_number(lines[0], "host_duration_us", &pop_host_duration_us) ||
        !extract_json_number(lines[1], "host_duration_us", &get_host_duration_us)) {
        std::cerr << "logger smoke missing host_duration_us\n";
        return false;
    }

    const double pop_wall_us =
        static_cast<double>(pop_wall_end_ns - pop_wall_start_ns) / 1000.0;
    const double get_wall_us =
        static_cast<double>(get_wall_end_ns - get_wall_start_ns) / 1000.0;
    if (pop_host_duration_us <= 0.0 || get_host_duration_us <= 0.0) {
        std::cerr << "logger smoke host_duration_us was not positive\n";
        return false;
    }
    if (pop_host_duration_us > pop_wall_us + 1000.0 ||
        get_host_duration_us > get_wall_us + 1000.0) {
        std::cerr << "logger smoke host_duration_us exceeded its wrapper wall-clock envelope: "
                  << "pop=" << pop_host_duration_us << "/" << pop_wall_us
                  << " get=" << get_host_duration_us << "/" << get_wall_us << "\n";
        return false;
    }

    std::cout << " logger_trace_path=" << trace_path
              << " logger_expect_enabled=" << (expect_enabled ? "true" : "false")
              << " logger_pop_ts_us=" << pop_ts
              << " logger_get_ts_us=" << get_ts
              << " logger_trace_delta_us=" << (get_ts - pop_ts)
              << " logger_pop_host_duration_us=" << pop_host_duration_us
              << " logger_get_host_duration_us=" << get_host_duration_us
              << " logger_trace_writer_overhead_ns="
              << fakecuda::trace::current_trace_writer_overhead_ns()
              << "\n";
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        return 2;
    }

    unsetenv("MAYA_ENABLE_LOGICAL_TRACE_TIMESTAMP_OVERHEAD_ADJUSTMENT");
    unsetenv("FLEXSIM_MAYA_ENABLE_LOGICAL_TRACE_TIMESTAMP_OVERHEAD_ADJUSTMENT");
    unsetenv("MAYA_DISABLE_LOGICAL_TRACE_TIMESTAMP_OVERHEAD_ADJUSTMENT");
    unsetenv("FLEXSIM_MAYA_DISABLE_LOGICAL_TRACE_TIMESTAMP_OVERHEAD_ADJUSTMENT");
    unsetenv("FAKECUDA_TRACE");

    bool expect_enabled = true;
    if (std::strcmp(argv[1], "default_on") == 0) {
        expect_enabled = true;
    } else if (std::strcmp(argv[1], "disable") == 0) {
        setenv("MAYA_DISABLE_LOGICAL_TRACE_TIMESTAMP_OVERHEAD_ADJUSTMENT", "1", 1);
        expect_enabled = false;
    } else if (std::strcmp(argv[1], "flexsim_disable") == 0) {
        setenv("FLEXSIM_MAYA_DISABLE_LOGICAL_TRACE_TIMESTAMP_OVERHEAD_ADJUSTMENT", "1", 1);
        expect_enabled = false;
    } else if (std::strcmp(argv[1], "enable0") == 0) {
        setenv("MAYA_ENABLE_LOGICAL_TRACE_TIMESTAMP_OVERHEAD_ADJUSTMENT", "0", 1);
        expect_enabled = false;
    } else if (std::strcmp(argv[1], "flexsim_enable0") == 0) {
        setenv("FLEXSIM_MAYA_ENABLE_LOGICAL_TRACE_TIMESTAMP_OVERHEAD_ADJUSTMENT", "0", 1);
        expect_enabled = false;
    } else if (std::strcmp(argv[1], "enable1") == 0) {
        setenv("MAYA_ENABLE_LOGICAL_TRACE_TIMESTAMP_OVERHEAD_ADJUSTMENT", "1", 1);
        expect_enabled = true;
    } else if (std::strcmp(argv[1], "flexsim_enable1") == 0) {
        setenv("FLEXSIM_MAYA_ENABLE_LOGICAL_TRACE_TIMESTAMP_OVERHEAD_ADJUSTMENT", "1", 1);
        expect_enabled = true;
    } else {
        return 2;
    }

    setenv("FAKECUDA_TRACE", "1", 1);
    fakecuda::trace::trace_writer_overhead_ns() = 0LL;

    fakecuda::trace::maybe_record_wrapper_entry_api("cudaGetDevice");
    const long long after_api_entry_overhead_ns =
        fakecuda::trace::current_trace_writer_overhead_ns();
    if (after_api_entry_overhead_ns != 0LL) {
        std::cerr << "wrapper-entry API bookkeeping unexpectedly increased overhead: "
                  << after_api_entry_overhead_ns << "\n";
        return 1;
    }
    long long cuda_get_device_entry_ns = 0LL;
    if (!fakecuda::trace::take_wrapper_entry_start_ns(
            "cudaGetDevice",
            &cuda_get_device_entry_ns
        ) || cuda_get_device_entry_ns <= 0LL) {
        std::cerr << "wrapper-entry API bookkeeping did not record entry start\n";
        return 1;
    }

    fakecuda::trace::maybe_record_wrapper_entry("__cudaPopCallConfiguration() called.");
    const long long after_debug_entry_overhead_ns =
        fakecuda::trace::current_trace_writer_overhead_ns();
    if (after_debug_entry_overhead_ns < after_api_entry_overhead_ns) {
        std::cerr << "wrapper-entry debug parsing moved overhead backwards: "
                  << after_debug_entry_overhead_ns << "\n";
        return 1;
    }
    long long pop_config_entry_ns = 0LL;
    if (!fakecuda::trace::take_wrapper_entry_start_ns(
            "__cudaPopCallConfiguration",
            &pop_config_entry_ns
        ) || pop_config_entry_ns <= 0LL) {
        std::cerr << "wrapper-entry debug parsing did not record entry start\n";
        return 1;
    }

    fakecuda::trace::add_trace_writer_overhead_ns(5000LL);
    const long long total_overhead_ns =
        fakecuda::trace::current_trace_writer_overhead_ns();
    if (total_overhead_ns <= after_debug_entry_overhead_ns) {
        std::cerr << "trace writer overhead was not mechanically recorded\n";
        return 1;
    }

    const long long raw_now_ns = total_overhead_ns + 20000LL;
    const long long clamped_raw_now_ns = total_overhead_ns - 1LL;
    const long long adjusted_now_ns =
        fakecuda::host_timing::trace_timestamp_now_ns_for_mode(
            raw_now_ns,
            fakecuda::host_timing::Mode::kMeasure
        );
    const long long later_adjusted_now_ns =
        fakecuda::host_timing::trace_timestamp_now_ns_for_mode(
            raw_now_ns + 1000LL,
            fakecuda::host_timing::Mode::kMeasure
        );
    const long long clamped_adjusted_now_ns =
        fakecuda::host_timing::trace_timestamp_now_ns_for_mode(
            clamped_raw_now_ns,
            fakecuda::host_timing::Mode::kMeasure
        );

    const long long trace_mode_now_ns =
        fakecuda::host_timing::trace_timestamp_now_ns_for_mode(
            raw_now_ns,
            fakecuda::host_timing::Mode::kTrace
        );
    const long long sleep_mode_now_ns =
        fakecuda::host_timing::trace_timestamp_now_ns_for_mode(
            raw_now_ns,
            fakecuda::host_timing::Mode::kSleep
        );
    const long long none_mode_now_ns =
        fakecuda::host_timing::trace_timestamp_now_ns_for_mode(
            raw_now_ns,
            fakecuda::host_timing::Mode::kNone
        );

    const long long expected_measure_ns = expect_enabled ? 20000LL : raw_now_ns;
    const long long expected_clamped_ns = expect_enabled ? 0LL : clamped_raw_now_ns;
    if (adjusted_now_ns != expected_measure_ns) {
        std::cerr << "measure timestamp mismatch: got " << adjusted_now_ns
                  << " expected " << expected_measure_ns << "\n";
        return 1;
    }
    if (clamped_adjusted_now_ns != expected_clamped_ns) {
        std::cerr << "clamped timestamp mismatch: got " << clamped_adjusted_now_ns
                  << " expected " << expected_clamped_ns << "\n";
        return 1;
    }
    if (adjusted_now_ns < 0LL || clamped_adjusted_now_ns < 0LL) {
        std::cerr << "adjusted timestamp went negative\n";
        return 1;
    }
    if (later_adjusted_now_ns < adjusted_now_ns) {
        std::cerr << "adjusted timestamp went backwards: earlier "
                  << adjusted_now_ns << " later " << later_adjusted_now_ns << "\n";
        return 1;
    }
    if (trace_mode_now_ns != raw_now_ns ||
        sleep_mode_now_ns != raw_now_ns ||
        none_mode_now_ns != raw_now_ns) {
        std::cerr << "non-measure timestamp changed: trace=" << trace_mode_now_ns
                  << " sleep=" << sleep_mode_now_ns
                  << " none=" << none_mode_now_ns
                  << " expected " << raw_now_ns << "\n";
        return 1;
    }
    if (argc == 3 && !run_logger_entry_hook_smoke(argv[2], expect_enabled)) {
        return 1;
    }

    std::cout << "mode=" << argv[1]
              << " expect_enabled=" << (expect_enabled ? "true" : "false")
              << " raw_now_ns=" << raw_now_ns
              << " trace_writer_overhead_ns="
              << total_overhead_ns
              << " wrapper_api_entry_overhead_ns=" << after_api_entry_overhead_ns
              << " wrapper_debug_entry_overhead_ns=" << after_debug_entry_overhead_ns
              << " measure_timestamp_ns=" << adjusted_now_ns
              << " later_measure_timestamp_ns=" << later_adjusted_now_ns
              << " clamped_measure_timestamp_ns=" << clamped_adjusted_now_ns
              << " trace_mode_timestamp_ns=" << trace_mode_now_ns
              << " sleep_mode_timestamp_ns=" << sleep_mode_now_ns
              << " none_mode_timestamp_ns=" << none_mode_now_ns
              << "\n";
    return 0;
}
