from __future__ import annotations

import json
from pathlib import Path

from flexsim.estimator import Estimator
from flexsim.maya_lite.io import load_trace_directory
from flexsim.maya_lite.schema import CandidateEvaluation, ReplayResult, TraceBundle, TraceSource
from flexsim.maya_lite.stage_timing import benchmark_trace_bundle, benchmark_trace_directory


def _write_trace(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_benchmark_trace_directory_reports_stage_breakdown(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 10,
                "tid": 1,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
            },
            {
                "ts": 10,
                "pid": 10,
                "tid": 1,
                "mod": "libcudart.so.12",
                "api": "cudaDeviceSynchronize",
                "type": "stream_op",
            },
        ],
    )

    benchmark = benchmark_trace_directory(
        trace_dir,
        Estimator(),
        emulator_seconds=1.25,
        allow_kernel_launch_heuristic_fallback=True,
    )

    payload = benchmark.to_dict()
    assert payload["candidate_name"] == "trace"
    assert payload["world_size"] == 1
    assert payload["trace_window"] == "full"
    assert payload["total_events"] == 3
    assert payload["stage_timing"]["emulator_seconds"] == 1.25
    assert payload["stage_timing"]["emulator_stage_seconds"] == 1.25
    assert payload["stage_timing"]["emulator_stage_basis"] == "active_emulator_seconds"
    assert payload["stage_timing"]["stack_seconds"] >= 1.25
    assert payload["stage_timing"]["collator_seconds"] >= 0.0
    assert payload["stage_timing"]["collator_stage_basis"] == "collate_only"
    assert payload["stage_timing"]["predictor_seconds"] >= 0.0
    assert payload["stage_timing"]["predictor_stage_basis"] == "runtime_annotation_wall_time"
    assert payload["stage_timing"]["predictor_seconds"] == (
        payload["stage_timing"]["predictor_total_annotation_seconds"]
    )
    assert payload["stage_timing"]["predictor_total_annotation_seconds"] >= (
        payload["stage_timing"]["predictor_runtime_estimation_seconds"]
    )
    assert payload["stage_timing"]["paper_processing_seconds"] == (
        payload["stage_timing"]["collator_seconds"]
        + payload["stage_timing"]["predictor_seconds"]
        + payload["stage_timing"]["simulator_seconds"]
    )
    assert payload["stage_timing"]["paper_stack_seconds"] == (
        payload["stage_timing"]["predictor_seconds"]
        + payload["stage_timing"]["collator_seconds"]
        + payload["stage_timing"]["simulator_seconds"]
        + payload["stage_timing"]["emulator_stage_seconds"]
    )
    assert payload["stage_timing"]["simulator_seconds"] >= 0.0
    assert payload["stage_timing"]["simulator_stage_basis"] == "replay_only"
    assert payload["stage_timing"]["trace_load_seconds"] >= 0.0
    assert payload["stage_timing"]["trace_load_included_in_collator"] is False
    assert payload["annotation_diagnostics"]["collective_group_count"] == 0
    assert payload["annotation_diagnostics"]["collective_group_duration_basis_counts"] == {}
    assert payload["annotation_diagnostics"]["duration_source_counts"] == {
        "estimator_global_fallback": 1,
        "heuristic_kernel_launch": 1,
        "observed_host_delay": 1,
    }
    assert payload["annotation_diagnostics"]["strict_runtime_signal_duration_source_counts"] == {
        "heuristic_kernel_launch": 1,
    }
    assert payload["annotation_diagnostics"]["strict_runtime_signal_wrapper_timing_contract_counts"] == {
        "missing": 1,
    }
    assert payload["annotation_diagnostics"]["strict_runtime_signal_event_count"] == 1
    assert payload["annotation_diagnostics"]["strict_runtime_signal_event_with_wrapper_timing_field_count"] == 0
    assert payload["annotation_diagnostics"]["strict_runtime_signal_event_with_direct_wrapper_runtime_count"] == 0
    assert payload["annotation_diagnostics"]["strict_runtime_signal_event_with_direct_runtime_contract_count"] == 0


def test_benchmark_trace_directory_excludes_trace_loading_from_collator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 10,
                "tid": 1,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
            }
        ],
    )

    timestamps = iter((1.0, 3.5, 10.0, 10.25, 20.0, 20.5, 30.0, 31.25))
    monkeypatch.setattr(
        "flexsim.maya_lite.stage_timing.perf_counter",
        lambda: next(timestamps),
    )

    benchmark = benchmark_trace_directory(
        trace_dir,
        Estimator(),
        allow_kernel_launch_heuristic_fallback=True,
    )

    assert benchmark.collator_seconds == 0.25
    assert benchmark.predictor_seconds == 0.5
    assert benchmark.predictor_total_annotation_seconds == 0.5
    assert benchmark.predictor_runtime_estimation_seconds == 0.0
    assert benchmark.simulator_seconds == 1.25
    assert benchmark.trace_load_seconds == 2.5


def test_benchmark_trace_directory_uses_manifest_capture_elapsed_seconds(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        '{"capture_elapsed_seconds": 2.5}',
        encoding="utf-8",
    )
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 10,
                "tid": 1,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
            }
        ],
    )

    benchmark = benchmark_trace_directory(
        trace_dir,
        Estimator(),
        allow_kernel_launch_heuristic_fallback=True,
    )

    assert benchmark.to_dict()["stage_timing"]["emulator_seconds"] == 2.5


def test_benchmark_trace_directory_accepts_explicit_emulator_wall_seconds(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 10,
                "tid": 1,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
            }
        ],
    )

    payload = benchmark_trace_directory(
        trace_dir,
        Estimator(),
        emulator_wall_seconds=3.5,
        allow_kernel_launch_heuristic_fallback=True,
    ).to_dict()["stage_timing"]

    assert payload["emulator_seconds"] == 3.5
    assert payload["emulator_wall_seconds"] == 3.5
    assert payload["emulator_stage_basis"] == "capture_elapsed_seconds"


def test_benchmark_trace_directory_prefers_capture_elapsed_seconds_for_paper_stage(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        '{"capture_elapsed_seconds": 9.0, "active_emulator_seconds": 4.0}',
        encoding="utf-8",
    )
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 10,
                "tid": 1,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
            }
        ],
    )

    benchmark = benchmark_trace_directory(
        trace_dir,
        Estimator(),
        allow_kernel_launch_heuristic_fallback=True,
    )
    payload = benchmark.to_dict()["stage_timing"]

    assert payload["emulator_seconds"] == 4.0
    assert payload["emulator_wall_seconds"] == 9.0
    assert payload["emulator_stage_seconds"] == 9.0
    assert payload["emulator_stage_basis"] == "capture_elapsed_seconds"
    assert payload["stack_seconds"] >= 9.0


def test_benchmark_trace_directory_can_materialize_logical_ranks(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        '{"original_world_size": 4, "profiled_ranks": [0, 2], "profiled_rank_groups": {"0": [0, 1], "2": [2, 3]}}'
    )
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 10,
                "tid": 1,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
            }
        ],
    )
    _write_trace(
        trace_dir / "rank_2.jsonl",
        [
            {
                "ts": 1,
                "pid": 20,
                "tid": 2,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
            }
        ],
    )

    benchmark = benchmark_trace_directory(
        trace_dir,
        Estimator(),
        materialize_logical_ranks=True,
        allow_kernel_launch_heuristic_fallback=True,
    )

    payload = benchmark.to_dict()
    assert payload["world_size"] == 4
    assert payload["profiled_world_size"] == 2
    assert payload["trace_window"] == "full"
    assert payload["total_events"] == 4
    assert [metric["rank"] for metric in payload["rank_metrics"]] == [0, 1, 2, 3]


def test_benchmark_trace_directory_can_dedup_rank_patterns(tmp_path: Path) -> None:
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
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 10,
                "tid": 1,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "m": "64",
                "n": "64",
                "k": "64",
            },
            {
                "ts": 10,
                "pid": 10,
                "tid": 1,
                "mod": "libnccl.so.2",
                "api": "ncclAllReduce",
                "type": "nccl_collective",
                "count": "8",
                "datatype": "7",
                "op": "0",
                "comm_id": "tp-a",
                "call_idx": "0",
            },
        ],
    )
    _write_trace(
        trace_dir / "rank_1.jsonl",
        [
            {
                "ts": 1,
                "pid": 20,
                "tid": 2,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "m": "64",
                "n": "64",
                "k": "64",
            },
            {
                "ts": 11,
                "pid": 20,
                "tid": 2,
                "mod": "libnccl.so.2",
                "api": "ncclAllReduce",
                "type": "nccl_collective",
                "count": "8",
                "datatype": "7",
                "op": "0",
                "comm_id": "tp-b",
                "call_idx": "7",
            },
        ],
    )
    _write_trace(
        trace_dir / "rank_2.jsonl",
        [
            {
                "ts": 2,
                "pid": 30,
                "tid": 3,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "m": "32",
                "n": "32",
                "k": "32",
            }
        ],
    )

    benchmark = benchmark_trace_directory(
        trace_dir,
        Estimator(),
        dedup_pattern_ranks=True,
        allow_kernel_launch_heuristic_fallback=True,
    )

    payload = benchmark.to_dict()
    assert payload["world_size"] == 3
    assert payload["profiled_world_size"] == 2
    assert payload["profiled_rank_groups"] == {"0": [0, 1], "2": [2]}


def test_benchmark_trace_directory_direct_profiled_replay_matches_materialized(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        '{"original_world_size": 4, "profiled_ranks": [0, 2], "profiled_rank_groups": {"0": [0, 1], "2": [2, 3]}}'
    )
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 10,
                "tid": 1,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "m": "64",
                "n": "64",
                "k": "64",
            },
            {
                "ts": 10,
                "pid": 10,
                "tid": 1,
                "mod": "libnccl.so.2",
                "api": "ncclAllReduce",
                "type": "nccl_collective",
                "count": "8",
                "datatype": "7",
                "op": "0",
                "comm_id": "dp0",
                "call_idx": "0",
                "nranks": "4",
            },
        ],
    )
    _write_trace(
        trace_dir / "rank_2.jsonl",
        [
            {
                "ts": 1,
                "pid": 20,
                "tid": 2,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
                "m": "32",
                "n": "32",
                "k": "32",
            },
            {
                "ts": 11,
                "pid": 20,
                "tid": 2,
                "mod": "libnccl.so.2",
                "api": "ncclAllReduce",
                "type": "nccl_collective",
                "count": "8",
                "datatype": "7",
                "op": "0",
                "comm_id": "dp0",
                "call_idx": "0",
                "nranks": "4",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)
    direct = benchmark_trace_directory(
        trace_dir,
        estimator,
        expand_profiled_rank_groups=True,
    )
    materialized = benchmark_trace_directory(
        trace_dir,
        estimator,
        materialize_logical_ranks=True,
        expand_profiled_rank_groups=True,
    )

    direct_payload = direct.to_dict()
    materialized_payload = materialized.to_dict()
    assert direct_payload["world_size"] == 4
    assert direct_payload["profiled_world_size"] == 2
    assert direct_payload["total_events"] < materialized_payload["total_events"]
    assert direct_payload["total_time_us"] == materialized_payload["total_time_us"]
    assert direct_payload["critical_path_us"] == materialized_payload["critical_path_us"]
    assert direct_payload["rank_metrics"] == materialized_payload["rank_metrics"]


def test_benchmark_trace_bundle_matches_directory_path(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        '{"capture_elapsed_seconds": 5.0, "active_emulator_seconds": 2.0}',
        encoding="utf-8",
    )
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 10,
                "tid": 1,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
            },
            {
                "ts": 10,
                "pid": 10,
                "tid": 1,
                "mod": "libnccl.so.2",
                "api": "ncclAllReduce",
                "type": "nccl_collective",
                "count": "8",
                "datatype": "7",
                "op": "0",
                "comm_id": "dp0",
                "call_idx": "0",
                "nranks": "1",
            },
        ],
    )

    estimator = Estimator.fit_from_traces(str(trace_dir), max_files=1)
    bundle = load_trace_directory(trace_dir)
    from_dir = benchmark_trace_directory(trace_dir, estimator)
    from_bundle = benchmark_trace_bundle(
        bundle,
        estimator,
        emulator_seconds=2.0,
        emulator_wall_seconds=5.0,
    )

    assert from_bundle.to_dict()["total_time_us"] == from_dir.to_dict()["total_time_us"]
    assert from_bundle.to_dict()["critical_path_us"] == from_dir.to_dict()["critical_path_us"]
    assert from_bundle.to_dict()["rank_metrics"] == from_dir.to_dict()["rank_metrics"]


def test_benchmark_trace_bundle_preserves_manifest_active_time_when_wall_is_overridden(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        '{"capture_elapsed_seconds": 5.0, "active_emulator_seconds": 2.0}',
        encoding="utf-8",
    )
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 0,
                "pid": 10,
                "tid": 1,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
            }
        ],
    )

    estimator = Estimator()
    bundle = load_trace_directory(trace_dir)
    payload = benchmark_trace_bundle(
        bundle,
        estimator,
        emulator_wall_seconds=7.0,
        allow_kernel_launch_heuristic_fallback=True,
    ).to_dict()["stage_timing"]

    assert payload["emulator_seconds"] == 2.0
    assert payload["emulator_wall_seconds"] == 7.0
    assert payload["emulator_stage_seconds"] == 7.0
    assert payload["emulator_stage_basis"] == "capture_elapsed_seconds"


def test_benchmark_trace_bundle_excludes_summary_and_evaluation_from_stage_timers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    bundle = TraceBundle(
        trace_dir=trace_dir,
        source=TraceSource.REAL,
        rank_traces=(),
    )

    call_order: list[str] = []

    def fake_collate_trace_bundle(arg):
        assert arg is bundle
        call_order.append("collate")
        return "collated"

    def fake_annotate_collated_trace(arg, estimator, **kwargs):
        assert arg == "collated"
        assert isinstance(estimator, Estimator)
        assert kwargs["use_observed_semantic_wrapper_durations"] is True
        assert kwargs["timing_recorder"] is not None
        call_order.append("annotate")
        return "annotated"

    def fake_replay_annotated_trace(arg, **kwargs):
        assert arg == "annotated"
        assert kwargs["record_simulated_events"] is True
        call_order.append("replay")
        return ReplayResult(
            total_time_us=9.0,
            critical_path_us=9.0,
            global_makespan_us=9.0,
            rank0_time_us=9.0,
            success=True,
            rank_metrics=(),
            simulated_events=(),
        )

    def fake_collective_group_duration_summary(arg):
        assert arg == "annotated"
        call_order.append("summary")
        # Deliberately consume perf_counter values to ensure this work does not
        # affect collator/predictor/simulator timings.
        from flexsim.maya_lite import stage_timing as stage_timing_module

        stage_timing_module.perf_counter()
        stage_timing_module.perf_counter()
        return {"collective_group_count": 0}

    def fake_build_candidate_evaluation(collated, replay, **kwargs):
        assert collated == "collated"
        assert replay.total_time_us == 9.0
        call_order.append("build")
        from flexsim.maya_lite import stage_timing as stage_timing_module

        stage_timing_module.perf_counter()
        stage_timing_module.perf_counter()
        return CandidateEvaluation(
            candidate_name="synthetic",
            trace_dir=trace_dir,
            source=TraceSource.REAL,
            world_size=1,
            total_events=0,
            total_time_us=9.0,
            critical_path_us=9.0,
            global_makespan_us=9.0,
            rank0_time_us=9.0,
            average_utilization=0.0,
            rank_metrics=(),
            annotation_diagnostics={},
        )

    timestamps = iter((10.0, 11.0, 20.0, 22.0, 30.0, 33.0, 100.0, 101.0, 200.0, 201.0))
    monkeypatch.setattr(
        "flexsim.maya_lite.stage_timing.perf_counter",
        lambda: next(timestamps),
    )
    monkeypatch.setattr(
        "flexsim.maya_lite.stage_timing.collate_trace_bundle",
        fake_collate_trace_bundle,
    )
    monkeypatch.setattr(
        "flexsim.maya_lite.stage_timing.annotate_collated_trace",
        fake_annotate_collated_trace,
    )
    monkeypatch.setattr(
        "flexsim.maya_lite.stage_timing.replay_annotated_trace",
        fake_replay_annotated_trace,
    )
    monkeypatch.setattr(
        "flexsim.maya_lite.stage_timing.collective_group_duration_summary",
        fake_collective_group_duration_summary,
    )
    monkeypatch.setattr(
        "flexsim.maya_lite.stage_timing.build_candidate_evaluation",
        fake_build_candidate_evaluation,
    )

    benchmark = benchmark_trace_bundle(
        bundle,
        Estimator(),
        emulator_seconds=5.0,
        emulator_wall_seconds=7.0,
    )

    assert benchmark.collator_seconds == 1.0
    assert benchmark.predictor_seconds == 2.0
    assert benchmark.predictor_total_annotation_seconds == 2.0
    assert benchmark.predictor_runtime_estimation_seconds == 0.0
    assert benchmark.simulator_seconds == 3.0
    assert call_order == ["collate", "annotate", "replay", "summary", "build"]


def test_benchmark_trace_directory_defaults_to_step_window_when_manifest_present(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        '{"step_windows": {"0": {"start_ts": 15, "end_ts": 25, "source": "trace_markers"}}}',
        encoding="utf-8",
    )
    _write_trace(
        trace_dir / "rank_0.jsonl",
        [
            {
                "ts": 10,
                "pid": 10,
                "tid": 1,
                "mod": "libcudart.so.12",
                "api": "cudaGetDevice",
                "type": "context_op",
            },
            {
                "ts": 20,
                "pid": 10,
                "tid": 1,
                "mod": "libcudart.so.12",
                "api": "cudaLaunchKernel",
                "type": "kernel_launch",
            },
        ],
    )

    benchmark = benchmark_trace_directory(
        trace_dir,
        Estimator(),
        allow_kernel_launch_heuristic_fallback=True,
    )
    payload = benchmark.to_dict()

    assert payload["trace_window"] == "step"
    assert payload["paper_valid_step_window_rank_count"] == 1
    assert payload["step_window_sources"] == ["trace_markers"]
    assert payload["total_events"] == 1
