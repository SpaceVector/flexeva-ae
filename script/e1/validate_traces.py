#!/usr/bin/env python3
"""Validate the linked 128-GPU E1 traces against the historical ledger."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE_ROOT = ROOT / "large-cluster" / "e1" / "historical_sparse_moe"
DEFAULT_LEDGER = ROOT / "large-cluster" / "e1" / "trajectory.csv"
EXPECTED_STAGES = ("Base", "S1", "S2", "S3", "S4")
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


def direction(value: float) -> int:
    if math.isclose(value, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        return 0
    return 1 if value > 1.0 else -1


def validate(trace_root: Path, ledger_path: Path, tolerance: float) -> list[dict[str, object]]:
    with ledger_path.open(newline="", encoding="utf-8") as stream:
        ledger = list(csv.DictReader(stream))
    if len(ledger) != len(EXPECTED_STAGES):
        raise ValueError(f"E1 ledger requires five rows, found {len(ledger)}")
    if not trace_root.is_dir():
        raise FileNotFoundError(f"E1 trace root is unavailable: {trace_root}")

    rows: list[dict[str, object]] = []
    for round_id, (record, stage) in enumerate(zip(ledger, EXPECTED_STAGES, strict=True)):
        if (int(record["round"]), record["stage"], int(record["world_size"])) != (round_id, stage, WORLD_SIZE):
            raise ValueError(f"unexpected ledger contract at round {round_id}")
        trace_dir = (
            trace_root
            / f"round{round_id:02d}"
            / "gpus_128"
            / f"historical_sparse_moe_realtrace_20260420_round{round_id:02d}"
        )
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
        send_bytes, send_calls = rank0_send_bytes(
            traces[0], int(rank0_window["start_ts"]), int(rank0_window["end_ts"])
        )
        rows.append(
            {
                "round": round_id,
                "stage": stage,
                "trace_dir": str(trace_dir.relative_to(trace_root)),
                "world_size": WORLD_SIZE,
                "trace_files": len(traces),
                "marker_files": len(markers),
                "manifest_parts": 16,
                "paper_valid_step_windows": len(windows),
                "rank0_marker_step_s": durations[0],
                "marker_step_min_s": min(durations),
                "marker_step_median_s": statistics.median(durations),
                "marker_step_max_s": max(durations),
                "rank0_nccl_send_calls": send_calls,
                "trace_rank0_a2a_bytes": send_bytes,
                "ledger_a2a_bytes": int(float(record["estimated_a2a_bytes"])),
                "ledger_time_normalized_improvement": float(record["time_normalized_improvement"]),
                "ledger_a2a_normalized_improvement": float(record["a2a_normalized_improvement"]),
            }
        )

    baseline_time = float(rows[0]["rank0_marker_step_s"])
    baseline_a2a = int(rows[0]["trace_rank0_a2a_bytes"])
    for row in rows:
        trace_time = 1.0 + (baseline_time - float(row["rank0_marker_step_s"])) / baseline_time
        trace_a2a = 1.0 + (baseline_a2a - int(row["trace_rank0_a2a_bytes"])) / baseline_a2a
        a2a_delta = abs(trace_a2a - float(row["ledger_a2a_normalized_improvement"]))
        time_matches = direction(trace_time) == direction(float(row["ledger_time_normalized_improvement"]))
        row.update(
            {
                "trace_time_normalized_improvement": trace_time,
                "time_direction_matches_ledger": time_matches,
                "trace_a2a_normalized_improvement": trace_a2a,
                "a2a_normalized_delta": a2a_delta,
            }
        )
        if not time_matches:
            raise ValueError(f"trace time direction differs at {row['stage']}")
        if a2a_delta > tolerance:
            raise ValueError(f"trace A2A trajectory differs at {row['stage']}: {a2a_delta} > {tolerance}")
    if min(rows, key=lambda row: float(row["rank0_marker_step_s"]))["stage"] != "S4":
        raise ValueError("S4 is not the fastest linked trace round")
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
            '{"ts":10,"api":"ncclSend","type":"nccl_collective","datatype":7,"count":4}\n'
            '{"ts":11,"api":"ncclRecv","type":"nccl_collective","datatype":7,"count":4}\n',
            encoding="utf-8",
        )
        assert marker_duration_s(marker) == 2.0
        assert rank0_send_bytes(trace, 10, 11) == (16, 1)
        assert direction(1.1) == 1 and direction(0.9) == -1 and direction(1.0) == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--a2a-normalized-tolerance", type=float, default=0.02)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("E1 trace validator self-test: PASS")
        return 0
    if args.a2a_normalized_tolerance < 0.0:
        raise ValueError("--a2a-normalized-tolerance must be non-negative")
    if not args.check_only and args.output is None:
        parser.error("--output is required unless --check-only is used")
    rows = validate(args.trace_root.resolve(), args.ledger, args.a2a_normalized_tolerance)
    if not args.check_only:
        write_rows(args.output, rows)
    max_delta = max(float(row["a2a_normalized_delta"]) for row in rows)
    print(f"E1 traces and ledger: PASS (5 rounds, 128 ranks, max A2A delta {max_delta:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
