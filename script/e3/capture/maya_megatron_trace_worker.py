#!/usr/bin/env python3
from __future__ import annotations

import ctypes
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
    hook_library = os.environ.get("PLAIN_MAYA_HOOK_LIBRARY")
    if hook_library:
        ctypes.CDLL(hook_library, mode=ctypes.RTLD_GLOBAL)
    os.environ["FAKECUDA_TRACE"] = "1"
    os.environ["FAKECUDA_TRACE_PATH"] = str(trace_dir / f"rank_{rank}.jsonl")
    os.environ["FLEXSIM_MAYA_MARKERS_PATH"] = str(trace_dir / f"rank_{rank}_markers.jsonl")
    pp = int(os.environ.get("FLEXMAYA_PP", "1"))
    os.environ["FLEXMAYA_STAGE_PARTITION"] = f"stage_{rank % max(pp, 1):03d}"
    os.environ["FLEXMAYA_CODE_PARTITION"] = os.environ["FLEXMAYA_STAGE_PARTITION"]
    os.environ.setdefault("FAKECUDA_TRACE_STDIO_BUFFER_BYTES", "0")
    runpy.run_path(os.environ["MAYA_MEGATRON_SCRIPT"], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
