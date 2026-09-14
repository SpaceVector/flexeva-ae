#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import gc
import json
import os
import runpy
from pathlib import Path


def run_workload(source):
    import torch
    from flexsim.maya_lite import markers

    phase = os.environ.get("FLEXMAYA_TABLE4_PHASE")
    if phase is None:
        return runpy.run_path(source, run_name="__main__")
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    native = ctypes.CDLL("libcudart.so.12")
    checkpoint = native.fakecudaTraceCheckpoint
    checkpoint.argtypes, checkpoint.restype = [], ctypes.c_char_p
    original = markers.emit_step_marker
    receipt = {"phase": phase, "optimizer_windows": 0, "measured_steps": 0}

    def emit(kind, *, step=None, label="training_step"):
        if kind == "step_begin":
            receipt["measured_steps"] += 1
            if receipt["measured_steps"] != 1:
                raise ValueError("Table 4 refresh requires one measured step")
        if kind == "region_begin" and label == "optimizer_step":
            receipt["optimizer_windows"] += 1
            if receipt["optimizer_windows"] != 1:
                raise ValueError("Table 4 refresh requires one optimizer window")
            receipt["entry_resources"] = checkpoint().decode()
            os.environ["FAKECUDA_RETAIN"] = "0"
        original(kind, step=step, label=label)
        if kind == "step_end":
            os.environ["FAKECUDA_RETAIN"] = "1"

    markers.emit_step_marker = emit
    # Retain mode still executes the prefix to establish runtime state; ordinary
    # trace recording starts when the optimizer marker switches it off.
    os.environ["FAKECUDA_RETAIN"] = "1" if phase == "selective" else "0"
    enabled = gc.isenabled()
    gc.disable()
    try:
        # ponytail: process-wide capture gate; only synchronous optimizer-tail
        # edits are admitted. General async emissions need the task-local adapter.
        with torch.autograd.set_multithreading_enabled(False):
            runpy.run_path(source, run_name="__main__")
        if receipt["optimizer_windows"] != 1 or receipt["measured_steps"] != 1:
            raise ValueError("workload did not traverse the selected boundary")
        path = Path(os.environ["FLEXMAYA_TRACE_DIR"]) / f"rank_{os.environ['RANK']}_refresh.json"
        path.write_text(json.dumps(receipt) + "\n")
    finally:
        markers.emit_step_marker = original
        os.environ["FAKECUDA_RETAIN"] = "0"
        if enabled:
            gc.enable()


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
    run_workload(os.environ["MAYA_MEGATRON_SCRIPT"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
