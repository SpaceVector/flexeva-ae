#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
from pathlib import Path


def main() -> int:
    trace_dir = Path(os.environ["FLEXMAYA_TRACE_DIR"])
    trace_dir.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    local_devices = max(int(os.environ.get("FLEXMAYA_LOCAL_DEVICE_COUNT", "1")), 1)
    os.environ["LOCAL_RANK"] = str(rank % local_devices)
    os.environ["FAKECUDA_TRACE"] = "1"
    os.environ["FAKECUDA_TRACE_PATH"] = str(trace_dir / f"rank_{rank}.jsonl")
    os.environ["FLEXSIM_MAYA_MARKERS_PATH"] = str(trace_dir / f"rank_{rank}_markers.jsonl")
    script = os.environ["TABLE4_SCRIPT"]
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
