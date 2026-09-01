#include <chrono>
#include <iostream>
#include <string>

#include "lowlevel/interface/wrapper_template.hpp"

namespace {

bool has_attr(const cpp_event::EventPayload& payload, const char* key) {
    return payload.attributes.find(key) != payload.attributes.end();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: launch_boundary_visibility_wrapper_smoke <enabled|disabled>\n";
        return 2;
    }

    const std::string mode = argv[1];
    const bool expect_enabled = mode == "enabled";
    if (!expect_enabled && mode != "disabled") {
        std::cerr << "mode must be enabled or disabled\n";
        return 2;
    }

    cpp_event::EventPayload payload;
    const auto start = lowlevel::interface::EventRecorder::Clock::now();
    const auto end = start + std::chrono::microseconds(123);
    lowlevel::interface::attach_launch_boundary_visibility_metadata(
        payload,
        "cudaLaunchKernel",
        start,
        end
    );

    const bool has_schema = has_attr(payload, "boundary_segment_schema_version");
    if (expect_enabled && !has_schema) {
        std::cerr << "expected launch-boundary metadata with opt-in enabled\n";
        return 10;
    }
    if (!expect_enabled && has_schema) {
        std::cerr << "launch-boundary metadata must be default-off\n";
        return 11;
    }
    if (!expect_enabled) {
        return 0;
    }

    if (payload.attributes["boundary_segment_schema_version"] != "launch_boundary_visibility_v1") {
        std::cerr << "unexpected schema version\n";
        return 12;
    }
    if (payload.attributes["wrapper_segment_coverage"] != "structural_labels_only_unmeasured") {
        std::cerr << "unexpected coverage\n";
        return 13;
    }
    if (payload.attributes["wrapper_segment_sum_us"] != "0.000") {
        std::cerr << "segment durations must remain unmeasured\n";
        return 14;
    }
    if (has_attr(payload, "actual_launch_control_dispatch_us") ||
        has_attr(payload, "actual_launch_api_body_us") ||
        has_attr(payload, "actual_launch_instrumentation_only_us")) {
        std::cerr << "measured launch segment timers are not allowed in this smoke\n";
        return 15;
    }
    if (payload.attributes["actual_launch_visibility_kind"] != "mixed_or_unresolved") {
        std::cerr << "actual launch visibility must remain unresolved\n";
        return 16;
    }
    return 0;
}
