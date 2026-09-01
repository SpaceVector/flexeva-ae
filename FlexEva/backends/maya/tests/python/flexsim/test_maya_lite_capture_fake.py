from __future__ import annotations

import json
from pathlib import Path

from flexsim.maya_lite.capture_fake import select_worker_trace_files


def _write_trace(path: Path, *, num_register: int, num_work: int, api: str = "ncclAllReduce") -> None:
    with path.open("w", encoding="utf-8") as handle:
        for _ in range(num_register):
            handle.write(json.dumps({"api": "__cudaRegisterFunction"}) + "\n")
        for _ in range(num_work):
            handle.write(json.dumps({"api": api}) + "\n")


def test_select_worker_trace_files_drops_launcher_like_trace(tmp_path: Path) -> None:
    launcher = tmp_path / "trace_launcher.jsonl"
    worker0 = tmp_path / "trace_worker0.jsonl"
    worker1 = tmp_path / "trace_worker1.jsonl"
    _write_trace(launcher, num_register=200, num_work=0)
    _write_trace(worker0, num_register=50, num_work=20)
    _write_trace(worker1, num_register=40, num_work=18)

    selected = select_worker_trace_files([launcher, worker0, worker1], expected_ranks=2)

    assert launcher not in selected
    assert selected == sorted([worker0, worker1])


def test_select_worker_trace_files_keeps_all_when_count_matches(tmp_path: Path) -> None:
    worker0 = tmp_path / "trace_worker0.jsonl"
    worker1 = tmp_path / "trace_worker1.jsonl"
    _write_trace(worker0, num_register=10, num_work=5, api="cudaLaunchKernel")
    _write_trace(worker1, num_register=10, num_work=5, api="cudaLaunchKernel")

    selected = select_worker_trace_files([worker0, worker1], expected_ranks=2)

    assert selected == [worker0, worker1]
