#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
from pathlib import Path


def main() -> int:
    trace_dir = Path(os.environ["FLEXMAYA_TRACE_DIR"])
    rank = int(os.environ.get("RANK", "0"))
    local_device_count = int(os.environ.get("FLEXMAYA_LOCAL_DEVICE_COUNT", "0") or "0")
    if local_device_count > 0:
        os.environ["LOCAL_RANK"] = str(rank % local_device_count)
    trace_dir.mkdir(parents=True, exist_ok=True)
    os.environ["FAKECUDA_TRACE"] = "1"
    os.environ["FAKECUDA_TRACE_PATH"] = str(trace_dir / f"rank_{rank}.jsonl")
    os.environ["FLEXSIM_MAYA_MARKERS_PATH"] = str(trace_dir / f"rank_{rank}_markers.jsonl")
    os.environ.setdefault("FAKECUDA_TRACE_STDIO_BUFFER_BYTES", "0")
    runpy.run_path(os.environ["MAYA_MEGATRON_SCRIPT"], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
