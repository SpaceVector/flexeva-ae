#!/usr/bin/env python3
"""Validate the supplied 128-GPU E1 traces and extract current inputs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import tempfile
from pathlib import Path


STAGES = ("Base", "S1", "S2", "S3", "S4")
WORLD_SIZE = 128
NCCL_DTYPE_BYTES = {0: 1, 1: 1, 2: 4, 3: 4, 4: 8, 5: 8, 6: 2, 7: 4, 8: 8, 9: 2, 10: 1, 11: 1}


def marker_duration_s(path: Path) -> float:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    begins = [row for row in records if row.get("kind") == "step_begin" and row.get("label") == "training_step"]
    ends = [row for row in records if row.get("kind") == "step_end" and row.get("label") == "training_step"]
    if len(begins) != 1 or len(ends) != 1 or begins[0].get("step") != ends[0].get("step"):
        raise ValueError(f"expected one matching training-step marker pair: {path}")
    duration = (int(ends[0]["monotonic_ns"]) - int(begins[0]["monotonic_ns"])) / 1e9
    if duration <= 0.0:
        raise ValueError(f"non-positive marker duration: {path}")
    return duration


def rank0_send_bytes(path: Path, start_ts: int, end_ts: int) -> tuple[int, int]:
    total = calls = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if "ncclSend" not in line:
                continue
            row = json.loads(line)
            if row.get("api") != "ncclSend" or row.get("type") != "nccl_collective":
                continue
            timestamp = int(row.get("ts", 0))
            if not start_ts <= timestamp <= end_ts:
                continue
            datatype = int(row.get("datatype", row.get("dtype_code", -1)))
            if datatype not in NCCL_DTYPE_BYTES:
                raise ValueError(f"unknown NCCL datatype {datatype}: {path}")
            count = int(row.get("count", row.get("numel", 0)))
            if count < 0:
                raise ValueError(f"negative NCCL count: {path}")
            total += count * NCCL_DTYPE_BYTES[datatype]
            calls += 1
    if calls == 0:
        raise ValueError(f"no measured-window ncclSend records: {path}")
    return total, calls


def extract(trace_root: Path) -> list[dict[str, object]]:
    if not trace_root.is_dir():
        raise FileNotFoundError(f"E1 trace root is unavailable: {trace_root}")
    rows: list[dict[str, object]] = []
    for round_id, stage in enumerate(STAGES):
        trace_dir = trace_root / f"round{round_id:02d}" / "gpus_128" / f"historical_sparse_moe_realtrace_20260420_round{round_id:02d}"
        manifest = json.loads((trace_dir / "capture_manifest.json").read_text(encoding="utf-8"))
        windows = manifest.get("step_windows", {})
        if int(manifest.get("original_world_size", 0)) != WORLD_SIZE:
            raise ValueError(f"trace world size differs at {stage}")
        if int(manifest.get("capture_manifest_materialized_from_partials", 0)) != 16:
            raise ValueError(f"trace manifest is not merged from 16 nodes at {stage}")
        if set(windows) != {str(rank) for rank in range(WORLD_SIZE)}:
            raise ValueError(f"trace step-window coverage differs at {stage}")
        if not all(bool(window.get("is_paper_valid_step_window")) for window in windows.values()):
            raise ValueError(f"invalid trace step window at {stage}")

        traces = [trace_dir / f"rank_{rank}.jsonl" for rank in range(WORLD_SIZE)]
        markers = [trace_dir / f"rank_{rank}.markers.jsonl" for rank in range(WORLD_SIZE)]
        missing = [str(path) for path in traces + markers if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise ValueError(f"missing or empty E1 trace files at {stage}: {missing[:4]}")
        durations = [marker_duration_s(path) for path in markers]
        rank0_window = windows["0"]
        send_bytes, send_calls = rank0_send_bytes(traces[0], int(rank0_window["start_ts"]), int(rank0_window["end_ts"]))
        rows.append(
            {
                "round": round_id,
                "stage": stage,
                "trace_dir": str(trace_dir.resolve()),
                "world_size": WORLD_SIZE,
                "trace_files": len(traces),
                "marker_files": len(markers),
                "rank0_marker_step_s": durations[0],
                "marker_step_min_s": min(durations),
                "marker_step_median_s": statistics.median(durations),
                "marker_step_max_s": max(durations),
                "rank0_nccl_send_calls": send_calls,
                "trace_rank0_a2a_bytes": send_bytes,
            }
        )
    if min(rows, key=lambda row: float(row["rank0_marker_step_s"]))["stage"] != "S4":
        raise ValueError("S4 is not the fastest supplied trace round")
    return rows


def write(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        marker = root / "rank_0.markers.jsonl"
        marker.write_text(
            '{"kind":"step_begin","label":"training_step","step":2,"monotonic_ns":1000000000}\n'
            '{"kind":"step_end","label":"training_step","step":2,"monotonic_ns":3000000000}\n',
            encoding="utf-8",
        )
        trace = root / "rank_0.jsonl"
        trace.write_text(
            '{"ts":9,"api":"ncclSend","type":"nccl_collective","datatype":7,"count":4}\n'
            '{"ts":10,"api":"ncclSend","type":"nccl_collective","datatype":7,"count":4}\n',
            encoding="utf-8",
        )
        assert marker_duration_s(marker) == 2.0
        assert rank0_send_bytes(trace, 10, 11) == (16, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("E1 trace validator self-test: PASS")
        return 0
    if args.trace_root is None or args.output is None:
        parser.error("--trace-root and --output are required")
    values = extract(args.trace_root.resolve())
    write(args.output, values)
    print("E1 supplied traces: PASS (5 rounds, 128 ranks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
