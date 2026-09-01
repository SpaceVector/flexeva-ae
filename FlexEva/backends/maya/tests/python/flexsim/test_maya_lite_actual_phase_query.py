from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_helper():
    repo = Path(__file__).resolve().parents[3]
    path = repo / "paper" / "maya_lite" / "diagnose_actual_phase_query.py"
    spec = importlib.util.spec_from_file_location("diagnose_actual_phase_query", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_rank(trace_dir: Path, *, rank: int, api: str, peer: str, count: int) -> None:
    (trace_dir / f"rank_{rank}.markers.jsonl").write_text(
        json.dumps({"kind": "step_begin", "monotonic_ns": 0}) + "\n",
        encoding="utf-8",
    )
    rows = []
    for index in range(count):
        ts = index * 1000 + rank
        rows.extend(
            [
                {"ts": ts, "api": "ncclGroupStart"},
                {
                    "ts": ts + 1,
                    "api": api,
                    "peer": str(peer),
                    "observed_runtime_us": "1.0",
                    "host_duration_us": 1,
                    "stream_id": "s",
                    "comm_id": "c",
                    "numel": "1",
                },
                {"ts": ts + 2, "api": "ncclGroupEnd"},
            ]
        )
    with (trace_dir / f"rank_{rank}.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_actual_phase_query_reports_no_sequence_overlap_for_mismatched_lengths(tmp_path: Path) -> None:
    helper = _load_helper()
    _write_rank(tmp_path, rank=8, api="ncclRecv", peer="1", count=10)
    _write_rank(tmp_path, rank=9, api="ncclSend", peer="0", count=3)
    output = tmp_path / "out.json"

    rc = helper.main(
        [
            "--real-trace-dir",
            str(tmp_path),
            "--rank-a",
            "8",
            "--api-a",
            "ncclRecv",
            "--peer-a",
            "1",
            "--rank-b",
            "9",
            "--api-b",
            "ncclSend",
            "--peer-b",
            "0",
            "--phase-min",
            "0.8",
            "--phase-max",
            "0.9",
            "--context",
            "0",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["counts"] == {"rank_a": 10, "rank_b": 3, "paired_min": 3}
    assert payload["pairing_semantics"]["target_window_has_sequence_overlap"] is False
    assert "beyond paired_min" in payload["pairing_semantics"]["no_sequence_overlap_reason"]
    assert payload["target_indices"]["context_hi"] is None
    assert payload["sequence_rows"] == []
    assert payload["nearest_rows"] == []
