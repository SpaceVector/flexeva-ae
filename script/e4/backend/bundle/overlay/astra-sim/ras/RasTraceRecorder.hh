/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __RAS_TRACE_RECORDER_HH__
#define __RAS_TRACE_RECORDER_HH__

#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>

namespace AstraSim {

class RasTraceRecorder {
  public:
    static RasTraceRecorder& get();

    bool enabled() const;
    void record_event(const std::string& layer,
                      const std::string& kind,
                      const std::string& lane,
                      const std::string& payload = "{}");
    void record_edge(const std::string& src_layer,
                     const std::string& src_id,
                     const std::string& dst_layer,
                     const std::string& dst_id,
                     const std::string& kind,
                     const std::string& payload = "{}");

  private:
    enum class TraceMode { Full, Summary, Off };

    RasTraceRecorder();

    static TraceMode parse_trace_mode();
    static std::string escape_json(const std::string& value);
    static const std::string& normalize_payload(const std::string& payload);
    bool should_record_event(const std::string& layer,
                             const std::string& kind) const;
    bool should_record_edge() const;
    void write_line(const std::string& line);

    mutable std::mutex mutex;
    std::ofstream output;
    bool is_enabled;
    TraceMode trace_mode;
    uint64_t sequence;
};

}  // namespace AstraSim

#endif /* __RAS_TRACE_RECORDER_HH__ */
