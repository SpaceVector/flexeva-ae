import json
from pathlib import Path

import pytest

from flexsim.estimator import Estimator
from flexsim.maya_lite.schema import (
    CandidateEvaluation,
    ReplayResult,
    TraceSource,
)
from flexsim.maya_lite.evaluate import evaluate_candidate_set, evaluate_trace_directory


@pytest.fixture(scope="module")
def estimator():
    trace_dir = Path("paper/traces/real/e1")
    if not trace_dir.exists():
        pytest.skip(f"trace dir not available: {trace_dir}")
    return Estimator.fit_from_traces(str(trace_dir), max_files=2)


def test_evaluate_trace_directory_smoke(estimator: Estimator):
    trace_dir = Path("paper/traces/fake/e3")
    if not trace_dir.exists():
        pytest.skip(f"trace dir not available: {trace_dir}")

    result = evaluate_trace_directory(trace_dir, estimator, max_events_per_rank=128)
    assert result.total_time_us > 0
    assert result.critical_path_us == result.total_time_us
    assert result.world_size >= 1
    assert result.total_events >= result.world_size
    assert 0.0 <= result.average_utilization <= 1.0


def test_evaluate_candidate_set_returns_sorted_results(estimator: Estimator):
    candidates = {
        "fake_e1": Path("paper/traces/fake/e1"),
        "fake_e3": Path("paper/traces/fake/e3"),
    }
    missing = [name for name, path in candidates.items() if not path.exists()]
    if missing:
        pytest.skip(f"trace dirs not available: {missing}")

    results = evaluate_candidate_set(candidates, estimator, max_events_per_rank=64)
    assert len(results) == 2
    assert [result.total_time_us for result in results] == sorted(
        result.total_time_us for result in results
    )
    assert {result.candidate_name for result in results} == set(candidates)


def test_evaluate_trace_directory_can_dedup_identical_ranks(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()

    rank_payloads = {
        0: [
            {"ts": 10, "pid": 1, "tid": 1, "mod": "libcudart.so.12", "api": "cudaLaunchKernel", "type": "kernel_launch", "m": "64", "n": "64", "k": "64"},
            {"ts": 20, "pid": 1, "tid": 1, "mod": "libnccl.so.2", "api": "ncclAllReduce", "type": "nccl_collective", "count": "8", "datatype": "7", "op": "0"},
        ],
        1: [
            {"ts": 11, "pid": 2, "tid": 2, "mod": "libcudart.so.12", "api": "cudaLaunchKernel", "type": "kernel_launch", "m": "64", "n": "64", "k": "64"},
            {"ts": 21, "pid": 2, "tid": 2, "mod": "libnccl.so.2", "api": "ncclAllReduce", "type": "nccl_collective", "count": "8", "datatype": "7", "op": "0"},
        ],
        2: [
            {"ts": 12, "pid": 3, "tid": 3, "mod": "libcudart.so.12", "api": "cudaLaunchKernel", "type": "kernel_launch", "m": "32", "n": "32", "k": "32"},
            {"ts": 22, "pid": 3, "tid": 3, "mod": "libnccl.so.2", "api": "ncclAllReduce", "type": "nccl_collective", "count": "4", "datatype": "7", "op": "0"},
        ],
    }
    for rank, payload in rank_payloads.items():
        path = trace_dir / f"rank_{rank}.jsonl"
        path.write_text("\n".join(json.dumps(item) for item in payload) + "\n")

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)
    result = evaluate_trace_directory(trace_dir, estimator, dedup_identical_ranks=True)

    assert result.world_size == 3
    assert result.profiled_world_size == 2
    assert result.profiled_rank_groups == {0: (0, 1), 2: (2,)}
    assert len(result.rank_metrics) == 2
    assert result.total_time_us > 0


def test_evaluate_trace_directory_can_dedup_rank_patterns(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "original_world_size": 3,
                "communicators": {
                    "tp-a": {"members": [0, 4, 8, 12], "size": 4, "name": "tp"},
                    "tp-b": {"members": [1, 5, 9, 13], "size": 4, "name": "tp"},
                    "tp-c": {"members": [2, 3], "size": 2, "name": "tp"},
                },
            }
        ),
        encoding="utf-8",
    )
    rank_payloads = {
        0: [
            {"ts": 10, "pid": 1, "tid": 1, "mod": "libcudart.so.12", "api": "cudaLaunchKernel", "type": "kernel_launch", "m": "64", "n": "64", "k": "64"},
            {"ts": 20, "pid": 1, "tid": 1, "mod": "libnccl.so.2", "api": "ncclAllReduce", "type": "nccl_collective", "count": "8", "datatype": "7", "op": "0", "comm_id": "tp-a", "call_idx": "0"},
        ],
        1: [
            {"ts": 11, "pid": 2, "tid": 2, "mod": "libcudart.so.12", "api": "cudaLaunchKernel", "type": "kernel_launch", "m": "64", "n": "64", "k": "64"},
            {"ts": 21, "pid": 2, "tid": 2, "mod": "libnccl.so.2", "api": "ncclAllReduce", "type": "nccl_collective", "count": "8", "datatype": "7", "op": "0", "comm_id": "tp-b", "call_idx": "7"},
        ],
        2: [
            {"ts": 12, "pid": 3, "tid": 3, "mod": "libcudart.so.12", "api": "cudaLaunchKernel", "type": "kernel_launch", "m": "32", "n": "32", "k": "32"},
            {"ts": 22, "pid": 3, "tid": 3, "mod": "libnccl.so.2", "api": "ncclAllReduce", "type": "nccl_collective", "count": "8", "datatype": "7", "op": "0", "comm_id": "tp-c", "call_idx": "3"},
        ],
    }
    for rank, payload in rank_payloads.items():
        path = trace_dir / f"rank_{rank}.jsonl"
        path.write_text("\n".join(json.dumps(item) for item in payload) + "\n")

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)
    result = evaluate_trace_directory(trace_dir, estimator, dedup_pattern_ranks=True)

    assert result.world_size == 3
    assert result.profiled_world_size == 2
    assert result.profiled_rank_groups == {0: (0, 1), 2: (2,)}
    assert len(result.rank_metrics) == 2
    assert result.total_time_us > 0


def test_evaluate_trace_directory_reports_collective_group_duration_diagnostics(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "communicators": {
                    "dp0": {"members": [0, 1], "size": 2, "name": "dp"},
                },
            }
        ),
        encoding="utf-8",
    )
    rank_payloads = {
        0: [
            {
                "ts": 10,
                "pid": 1,
                "tid": 1,
                "mod": "libnccl.so.2",
                "api": "ncclAllReduce",
                "type": "nccl_collective",
                "count": "8",
                "datatype": "7",
                "op": "0",
                "comm_id": "dp0",
                "call_idx": "0",
                "nranks": "2",
            },
        ],
        1: [
            {
                "ts": 11,
                "pid": 2,
                "tid": 2,
                "mod": "libnccl.so.2",
                "api": "ncclAllReduce",
                "type": "nccl_collective",
                "count": "8",
                "datatype": "7",
                "op": "0",
                "comm_id": "dp0",
                "call_idx": "0",
                "nranks": "2",
            },
        ],
    }
    for rank, payload in rank_payloads.items():
        path = trace_dir / f"rank_{rank}.jsonl"
        path.write_text("\n".join(json.dumps(item) for item in payload) + "\n")

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)
    result = evaluate_trace_directory(trace_dir, estimator)

    assert result.annotation_diagnostics["collective_group_count"] == 1
    assert result.annotation_diagnostics["collective_group_with_duration_metadata_count"] == 1
    assert sum(result.annotation_diagnostics["collective_group_duration_basis_counts"].values()) == 1
    assert result.annotation_diagnostics["duration_source_counts"] == {
        "estimator_api_stats": 2,
    }
    assert result.annotation_diagnostics["strict_runtime_signal_duration_source_counts"] == {
        "estimator_api_stats": 2,
    }
    assert result.annotation_diagnostics["strict_runtime_signal_wrapper_timing_contract_counts"] == {
        "missing": 2,
    }
    assert result.annotation_diagnostics["strict_runtime_signal_event_count"] == 2
    assert result.annotation_diagnostics["strict_runtime_signal_event_with_wrapper_timing_field_count"] == 0
    assert result.annotation_diagnostics["strict_runtime_signal_event_with_direct_wrapper_runtime_count"] == 0
    assert result.annotation_diagnostics["strict_runtime_signal_event_with_direct_runtime_contract_count"] == 0


def test_evaluate_trace_directory_skips_simulated_event_recording_on_paper_facing_path(
    tmp_path: Path,
    monkeypatch,
):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "rank_0.jsonl").write_text(
        json.dumps(
            {
                "ts": 0,
                "pid": 1,
                "tid": 1,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "m": "64",
                "n": "64",
                "k": "64",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    estimator = Estimator()

    def fake_replay_annotated_trace(arg, **kwargs):
        assert kwargs["record_simulated_events"] is False
        return ReplayResult(
            total_time_us=5.0,
            critical_path_us=5.0,
            success=True,
            rank_metrics=(),
            simulated_events=(),
        )

    def fake_build_candidate_evaluation(collated, replay, **kwargs):
        return CandidateEvaluation(
            candidate_name="synthetic",
            trace_dir=trace_dir,
            source=TraceSource.UNKNOWN,
            world_size=1,
            total_events=1,
            total_time_us=replay.total_time_us,
            critical_path_us=replay.critical_path_us,
            average_utilization=0.0,
            rank_metrics=(),
            annotation_diagnostics={},
        )

    monkeypatch.setattr(
        "flexsim.maya_lite.evaluate.replay_annotated_trace",
        fake_replay_annotated_trace,
    )
    monkeypatch.setattr(
        "flexsim.maya_lite.evaluate.build_candidate_evaluation",
        fake_build_candidate_evaluation,
    )

    result = evaluate_trace_directory(trace_dir, estimator)

    assert result.total_time_us == 5.0


def test_evaluate_trace_directory_can_expand_profiled_rank_groups(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        '{"original_world_size": 4, "profiled_ranks": [0, 2], "profiled_rank_groups": {"0": [0, 1], "2": [2, 3]}}'
    )
    rank_payloads = {
        0: [
            {"ts": 10, "pid": 1, "tid": 1, "mod": "libcudart.so.12", "api": "cudaLaunchKernel", "type": "kernel_launch", "m": "64", "n": "64", "k": "64"},
        ],
        2: [
            {"ts": 12, "pid": 3, "tid": 3, "mod": "libcudart.so.12", "api": "cudaLaunchKernel", "type": "kernel_launch", "m": "32", "n": "32", "k": "32"},
        ],
    }
    for rank, payload in rank_payloads.items():
        path = trace_dir / f"rank_{rank}.jsonl"
        path.write_text("\n".join(json.dumps(item) for item in payload) + "\n")

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)
    result = evaluate_trace_directory(
        trace_dir,
        estimator,
        expand_profiled_rank_groups=True,
    )

    assert result.world_size == 4
    assert result.profiled_world_size == 2
    assert result.trace_window == "full"
    assert result.profiled_rank_groups == {0: (0, 1), 2: (2, 3)}
    assert [metric.rank for metric in result.rank_metrics] == [0, 1, 2, 3]


def test_evaluate_trace_directory_can_materialize_logical_ranks(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        '{"original_world_size": 4, "profiled_ranks": [0, 2], "profiled_rank_groups": {"0": [0, 1], "2": [2, 3]}}'
    )
    rank_payloads = {
        0: [
            {"ts": 10, "pid": 1, "tid": 1, "mod": "libcudart.so.12", "api": "cudaLaunchKernel", "type": "kernel_launch", "m": "64", "n": "64", "k": "64"},
        ],
        2: [
            {"ts": 12, "pid": 3, "tid": 3, "mod": "libcudart.so.12", "api": "cudaLaunchKernel", "type": "kernel_launch", "m": "32", "n": "32", "k": "32"},
        ],
    }
    for rank, payload in rank_payloads.items():
        path = trace_dir / f"rank_{rank}.jsonl"
        path.write_text("\n".join(json.dumps(item) for item in payload) + "\n")

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)
    result = evaluate_trace_directory(
        trace_dir,
        estimator,
        materialize_logical_ranks=True,
    )

    assert result.world_size == 4
    assert result.profiled_world_size == 2
    assert result.trace_window == "full"
    assert result.profiled_rank_groups == {0: (0, 1), 2: (2, 3)}
    assert result.total_events == 4
    assert [metric.rank for metric in result.rank_metrics] == [0, 1, 2, 3]


def test_evaluate_trace_directory_direct_profiled_replay_matches_materialized(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        '{"original_world_size": 4, "profiled_ranks": [0, 2], "profiled_rank_groups": {"0": [0, 1], "2": [2, 3]}}'
    )
    rank_payloads = {
        0: [
            {"ts": 10, "pid": 1, "tid": 1, "mod": "libcudart.so.12", "api": "cudaLaunchKernel", "type": "kernel_launch", "m": "64", "n": "64", "k": "64"},
            {"ts": 20, "pid": 1, "tid": 1, "mod": "libnccl.so.2", "api": "ncclAllReduce", "type": "nccl_collective", "count": "8", "datatype": "7", "op": "0", "comm_id": "dp0", "call_idx": "0", "nranks": "4"},
        ],
        2: [
            {"ts": 12, "pid": 3, "tid": 3, "mod": "libcudart.so.12", "api": "cudaLaunchKernel", "type": "kernel_launch", "m": "32", "n": "32", "k": "32"},
            {"ts": 21, "pid": 3, "tid": 3, "mod": "libnccl.so.2", "api": "ncclAllReduce", "type": "nccl_collective", "count": "8", "datatype": "7", "op": "0", "comm_id": "dp0", "call_idx": "0", "nranks": "4"},
        ],
    }
    for rank, payload in rank_payloads.items():
        path = trace_dir / f"rank_{rank}.jsonl"
        path.write_text("\n".join(json.dumps(item) for item in payload) + "\n")

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)
    direct = evaluate_trace_directory(
        trace_dir,
        estimator,
        expand_profiled_rank_groups=True,
    )
    materialized = evaluate_trace_directory(
        trace_dir,
        estimator,
        materialize_logical_ranks=True,
        expand_profiled_rank_groups=True,
    )

    assert direct.world_size == 4
    assert direct.profiled_world_size == 2
    assert direct.total_time_us == materialized.total_time_us
    assert direct.critical_path_us == materialized.critical_path_us
    assert direct.total_events < materialized.total_events
    assert [metric.rank for metric in direct.rank_metrics] == [0, 1, 2, 3]
    assert [metric.rank for metric in materialized.rank_metrics] == [0, 1, 2, 3]
    assert [
        (metric.rank, metric.total_time_us, metric.communication_time_us)
        for metric in direct.rank_metrics
    ] == [
        (metric.rank, metric.total_time_us, metric.communication_time_us)
        for metric in materialized.rank_metrics
    ]


def test_evaluate_trace_directory_defaults_to_step_window_when_manifest_present(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "step_windows": {
                    "0": {"start_ts": 15, "end_ts": 25, "source": "trace_markers"},
                }
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"ts": 10, "pid": 1, "tid": 1, "mod": "libcudart.so.12", "api": "cudaGetDevice", "type": "context_op"}),
                json.dumps({"ts": 20, "pid": 1, "tid": 1, "mod": "libcudart.so.12", "api": "cudaLaunchKernel", "type": "kernel_launch", "m": "16", "n": "16", "k": "16"}),
                json.dumps({"ts": 30, "pid": 1, "tid": 1, "mod": "libcudart.so.12", "api": "cudaLaunchKernel", "type": "kernel_launch", "m": "32", "n": "32", "k": "32"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)
    result = evaluate_trace_directory(trace_dir, estimator)

    assert result.trace_window == "step"
    assert result.paper_valid_step_window_rank_count == 1
    assert result.step_window_sources == ("trace_markers",)
    assert result.total_events == 1
