from __future__ import annotations

import json
from pathlib import Path

import pytest

from flexsim.maya_lite.io import load_trace_directory
from flexsim.maya_lite.fig13_contract import (
    inspect_fig13_step_contract,
    inspect_fig13_step_contract_bundle,
    validate_fig13_step_contract_bundle,
    validate_fig13_step_contract,
)


def _write_rank_trace(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_step_manifest(trace_dir: Path, *, start_ts: int = 10, end_ts: int = 20) -> None:
    payload = {
        "step_windows": {
            "0": {"start_ts": start_ts, "end_ts": end_ts, "source": "trace_markers", "step_count": 1},
            "1": {"start_ts": start_ts, "end_ts": end_ts, "source": "trace_markers", "step_count": 1},
        }
    }
    (trace_dir / "capture_manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_validate_fig13_step_contract_accepts_expected_nccl_ops(tmp_path: Path) -> None:
    _write_rank_trace(
        tmp_path / "rank_0.jsonl",
        [
            {"ts": 5, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclCommInitRank", "type": "control"},
            {
                "ts": 10,
                "pid": 0,
                "tid": 0,
                "mod": "libnccl.so",
                "api": "ncclGetVersion",
                "type": "control",
            },
            {"ts": 11, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclGroupStart", "type": "control"},
            {"ts": 12, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclSend", "type": "collective"},
            {"ts": 13, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclAllReduce", "type": "collective"},
            {"ts": 14, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclGroupEnd", "type": "control"},
            {
                "ts": 15,
                "pid": 0,
                "tid": 0,
                "mod": "libnccl.so",
                "api": "ncclGetUniqueId",
                "type": "control",
            },
        ],
    )
    _write_rank_trace(
        tmp_path / "rank_1.jsonl",
        [
            {"ts": 6, "pid": 1, "tid": 1, "mod": "libnccl.so", "api": "ncclCommInitRank", "type": "control"},
            {"ts": 14, "pid": 1, "tid": 1, "mod": "libnccl.so", "api": "ncclRecv", "type": "collective"},
            {
                "ts": 16,
                "pid": 1,
                "tid": 1,
                "mod": "libnccl.so",
                "api": "ncclCommInitRankConfig",
                "type": "control",
            },
            {
                "ts": 17,
                "pid": 1,
                "tid": 1,
                "mod": "libnccl.so",
                "api": "ncclCommGetAsyncError",
                "type": "control",
            },
        ],
    )
    _write_step_manifest(tmp_path)

    summary = validate_fig13_step_contract(tmp_path)

    assert summary["trace_window"] == "step"
    assert summary["paper_valid_step_window_rank_count"] == 2
    assert summary["step_window_sources"] == ["trace_markers"]
    assert summary["observed_nccl_apis"] == [
        "ncclAllReduce",
        "ncclCommGetAsyncError",
        "ncclCommInitRankConfig",
        "ncclGetUniqueId",
        "ncclGetVersion",
        "ncclGroupEnd",
        "ncclGroupStart",
        "ncclRecv",
        "ncclSend",
    ]
    assert summary["unexpected_nccl_apis"] == []


def test_validate_fig13_step_contract_rejects_unexpected_nccl_ops(tmp_path: Path) -> None:
    _write_rank_trace(
        tmp_path / "rank_0.jsonl",
        [
            {"ts": 12, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclAllGather", "type": "collective"},
        ],
    )
    _write_rank_trace(
        tmp_path / "rank_1.jsonl",
        [
            {"ts": 14, "pid": 1, "tid": 1, "mod": "libnccl.so", "api": "ncclRecv", "type": "collective"},
        ],
    )
    _write_step_manifest(tmp_path)

    with pytest.raises(ValueError, match="unexpected NCCL APIs: ncclAllGather"):
        validate_fig13_step_contract(tmp_path)


def test_inspect_fig13_step_contract_reports_full_trace_when_no_step_window(tmp_path: Path) -> None:
    _write_rank_trace(
        tmp_path / "rank_0.jsonl",
        [
            {"ts": 12, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclSend", "type": "collective"},
        ],
    )

    summary = inspect_fig13_step_contract(tmp_path)

    assert summary["trace_window"] == "full"
    assert summary["observed_nccl_apis"] == ["ncclSend"]


def test_validate_fig13_step_contract_bundle_matches_directory(tmp_path: Path) -> None:
    _write_rank_trace(
        tmp_path / "rank_0.jsonl",
        [
            {"ts": 11, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclGroupStart", "type": "control"},
            {"ts": 12, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclSend", "type": "collective"},
            {"ts": 13, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclGroupEnd", "type": "control"},
        ],
    )
    _write_rank_trace(
        tmp_path / "rank_1.jsonl",
        [
            {"ts": 14, "pid": 1, "tid": 1, "mod": "libnccl.so", "api": "ncclRecv", "type": "collective"},
        ],
    )
    _write_step_manifest(tmp_path)

    bundle = load_trace_directory(tmp_path, trace_window="step")
    assert inspect_fig13_step_contract_bundle(bundle) == inspect_fig13_step_contract(tmp_path)
    assert validate_fig13_step_contract_bundle(bundle) == validate_fig13_step_contract(tmp_path)


def test_validate_fig13_step_contract_bundle_rejects_trace_window_mode(
    tmp_path: Path,
) -> None:
    _write_rank_trace(
        tmp_path / "rank_0.jsonl",
        [
            {"ts": 11, "pid": 0, "tid": 0, "mod": "libnccl.so", "api": "ncclSend", "type": "collective"},
        ],
    )
    _write_rank_trace(
        tmp_path / "rank_1.jsonl",
        [
            {"ts": 14, "pid": 1, "tid": 1, "mod": "libnccl.so", "api": "ncclRecv", "type": "collective"},
        ],
    )
    _write_step_manifest(tmp_path, start_ts=10, end_ts=20)

    bundle = load_trace_directory(tmp_path, trace_window="trace")

    assert bundle.trace_window == "trace"
    assert bundle.step_windows == {}
    with pytest.raises(ValueError, match="requires explicit step-window traces"):
        validate_fig13_step_contract_bundle(bundle)
