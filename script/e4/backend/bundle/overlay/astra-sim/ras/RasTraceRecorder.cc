/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/ras/RasTraceRecorder.hh"

#include <cstdlib>
#include <iostream>
#include <sstream>

using namespace AstraSim;

namespace {

constexpr const char* kTraceModeEnv = "ASTRA_SIM_RAS_TRACE_MODE";

}  // namespace

RasTraceRecorder& RasTraceRecorder::get() {
    static RasTraceRecorder recorder;
    return recorder;
}

RasTraceRecorder::RasTraceRecorder()
    : is_enabled(false), trace_mode(TraceMode::Full), sequence(0) {
    const char* trace_path = std::getenv("ASTRA_SIM_RAS_TRACE");
    if (trace_path == nullptr || trace_path[0] == '\0') {
        return;
    }

    trace_mode = parse_trace_mode();
    if (trace_mode == TraceMode::Off) {
        return;
    }

    output.open(trace_path, std::ios::out | std::ios::trunc);
    if (!output.is_open()) {
        std::cerr << "ASTRA-sim RAS trace disabled: unable to open "
                  << trace_path << std::endl;
        return;
    }

    is_enabled = true;
}

bool RasTraceRecorder::enabled() const {
    return is_enabled;
}

RasTraceRecorder::TraceMode RasTraceRecorder::parse_trace_mode() {
    const char* trace_mode_env = std::getenv(kTraceModeEnv);
    if (trace_mode_env == nullptr || trace_mode_env[0] == '\0' ||
        std::string(trace_mode_env) == "full") {
        return TraceMode::Full;
    }
    const std::string trace_mode_value(trace_mode_env);
    if (trace_mode_value == "summary" || trace_mode_value == "reuse" ||
        trace_mode_value == "changed") {
        return TraceMode::Summary;
    }
    if (trace_mode_value == "off") {
        return TraceMode::Off;
    }
    std::cerr << "ASTRA-sim RAS trace mode '" << trace_mode_value
              << "' is unknown; using full trace mode" << std::endl;
    return TraceMode::Full;
}

void RasTraceRecorder::record_event(const std::string& layer,
                                    const std::string& kind,
                                    const std::string& lane,
                                    const std::string& payload) {
    if (!is_enabled || !should_record_event(layer, kind)) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex);
    std::ostringstream line;
    line << "{\"type\":\"event\",\"seq\":" << sequence++
         << ",\"layer\":\"" << escape_json(layer) << "\",\"kind\":\""
         << escape_json(kind) << "\",\"lane\":\"" << escape_json(lane)
         << "\",\"payload\":" << normalize_payload(payload) << "}";
    write_line(line.str());
}

void RasTraceRecorder::record_edge(const std::string& src_layer,
                                   const std::string& src_id,
                                   const std::string& dst_layer,
                                   const std::string& dst_id,
                                   const std::string& kind,
                                   const std::string& payload) {
    if (!is_enabled || !should_record_edge()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex);
    std::ostringstream line;
    line << "{\"type\":\"edge\",\"seq\":" << sequence++ << ",\"src_layer\":\""
         << escape_json(src_layer) << "\",\"src_id\":\""
         << escape_json(src_id) << "\",\"dst_layer\":\""
         << escape_json(dst_layer) << "\",\"dst_id\":\""
         << escape_json(dst_id) << "\",\"kind\":\"" << escape_json(kind)
         << "\",\"payload\":" << normalize_payload(payload) << "}";
    write_line(line.str());
}

std::string RasTraceRecorder::escape_json(const std::string& value) {
    std::ostringstream escaped;
    for (const char c : value) {
        const unsigned char uc = static_cast<unsigned char>(c);
        switch (c) {
        case '\\':
            escaped << "\\\\";
            break;
        case '"':
            escaped << "\\\"";
            break;
        case '\b':
            escaped << "\\b";
            break;
        case '\f':
            escaped << "\\f";
            break;
        case '\n':
            escaped << "\\n";
            break;
        case '\r':
            escaped << "\\r";
            break;
        case '\t':
            escaped << "\\t";
            break;
        default:
            if (uc < 0x20) {
                escaped << "\\u00";
                const char* digits = "0123456789abcdef";
                escaped << digits[(uc >> 4) & 0x0f] << digits[uc & 0x0f];
            } else {
                escaped << c;
            }
            break;
        }
    }
    return escaped.str();
}

const std::string& RasTraceRecorder::normalize_payload(
    const std::string& payload) {
    static const std::string empty_payload = "{}";
    if (payload.empty()) {
        return empty_payload;
    }
    return payload;
}

bool RasTraceRecorder::should_record_event(const std::string& layer,
                                           const std::string& kind) const {
    if (trace_mode == TraceMode::Full) {
        return true;
    }

    // Summary mode is intentionally a low-volume filter rather than a final
    // aggregate: keep run context plus replay reuse decisions, and skip system,
    // network, dependency, and other high-frequency lower-layer events.
    return (layer == "context" && kind == "run_context") ||
           (layer == "context" && kind == "simulation_phase") ||
           (layer == "workload" && kind == "chakra_node_reuse");
}

bool RasTraceRecorder::should_record_edge() const {
    return trace_mode == TraceMode::Full;
}

void RasTraceRecorder::write_line(const std::string& line) {
    output << line << '\n';
}
