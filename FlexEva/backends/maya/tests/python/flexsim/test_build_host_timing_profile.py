from __future__ import annotations

import json

from flexsim.maya_lite.build_host_timing_profile import (
    _canonicalize_host_timing_api,
    _collect_helper_thread_templates,
    _build_parser,
    _load_step_windows,
    build_profile_lines,
    collect_host_gap_samples,
)


def test_collect_host_gap_samples_attributes_gap_to_previous_api(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_path = trace_dir / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 10,
                        "pid": 1,
                        "tid": 7,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetDevice",
                        "type": "context_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 14,
                        "pid": 1,
                        "tid": 7,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetLastError",
                        "type": "context_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 30,
                        "pid": 1,
                        "tid": 7,
                        "mod": "libcudart.so.12",
                        "api": "cudaEventRecord",
                        "type": "stream_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 31,
                        "pid": 1,
                        "tid": 7,
                        "mod": "libcudart.so.12",
                        "api": "cudaLaunchKernel",
                        "type": "kernel_launch",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (
        pair_occurrence_samples,
        pair_samples,
        api_samples,
        type_samples,
        processed_events,
        thread_start_occurrence_samples,
        thread_start_samples,
    ) = collect_host_gap_samples(trace_dir, dispatch_scope="thread")

    assert processed_events == 4
    assert thread_start_occurrence_samples["threadstartocc:cudaGetDevice#0"] == [0.0]
    assert thread_start_samples["threadstart:cudaGetDevice"] == [0.0]
    assert pair_occurrence_samples["pairocc:cudaGetDevice->cudaGetLastError#0"] == [4.0]
    assert pair_occurrence_samples["pairocc:cudaGetLastError->cudaEventRecord#0"] == [16.0]
    assert pair_occurrence_samples["pairocc:cudaEventRecord->cudaLaunchKernel#0"] == [1.0]
    assert pair_samples["pair:cudaGetDevice->cudaGetLastError"] == [4.0]
    assert pair_samples["pair:cudaGetLastError->cudaEventRecord"] == [16.0]
    assert pair_samples["pair:cudaEventRecord->cudaLaunchKernel"] == [1.0]
    assert api_samples["cudaGetDevice"] == [4.0]
    assert api_samples["cudaGetLastError"] == [16.0]
    assert api_samples["cudaEventRecord"] == [1.0]
    assert type_samples["context_op"] == [4.0, 16.0]
    assert type_samples["stream_op"] == [1.0]


def test_canonicalize_host_timing_api_normalizes_nccl_bcast_alias() -> None:
    assert _canonicalize_host_timing_api("ncclBcast") == "ncclBroadcast"
    assert _canonicalize_host_timing_api("cudaEventRecordWithFlags") == "cudaEventRecord"


def test_collect_host_gap_samples_uses_dispatch_high_watermark(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_path = trace_dir / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 100,
                        "pid": 1,
                        "tid": 9,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetDevice",
                        "type": "context_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 120,
                        "pid": 1,
                        "tid": 9,
                        "mod": "libcudart.so.12",
                        "api": "cudaEventRecord",
                        "type": "stream_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 110,
                        "pid": 1,
                        "tid": 9,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetLastError",
                        "type": "context_op",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (
        pair_occurrence_samples,
        pair_samples,
        api_samples,
        _,
        _,
        thread_start_occurrence_samples,
        thread_start_samples,
    ) = collect_host_gap_samples(trace_dir, dispatch_scope="thread")

    assert thread_start_occurrence_samples["threadstartocc:cudaGetDevice#0"] == [0.0]
    assert thread_start_samples["threadstart:cudaGetDevice"] == [0.0]
    assert pair_occurrence_samples["pairocc:cudaGetDevice->cudaGetLastError#0"] == [20.0]
    assert pair_samples["pair:cudaGetDevice->cudaGetLastError"] == [20.0]
    assert api_samples["cudaGetDevice"] == [20.0]


def test_collect_host_gap_samples_semantic_surface_accumulates_compat_gaps(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_path = trace_dir / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 10,
                        "pid": 1,
                        "tid": 7,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetDevice",
                        "type": "context_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 14,
                        "pid": 1,
                        "tid": 7,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetLastError",
                        "type": "context_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 30,
                        "pid": 1,
                        "tid": 7,
                        "mod": "libcudart.so.12",
                        "api": "cudaEventRecord",
                        "type": "stream_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 31,
                        "pid": 1,
                        "tid": 7,
                        "mod": "libcudart.so.12",
                        "api": "cudaLaunchKernel",
                        "type": "kernel_launch",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (
        pair_occurrence_samples,
        pair_samples,
        api_samples,
        type_samples,
        processed_events,
        thread_start_occurrence_samples,
        thread_start_samples,
    ) = collect_host_gap_samples(trace_dir, profile_surface="semantic")

    assert processed_events == 4
    assert thread_start_occurrence_samples["threadstartocc:cudaEventRecord#0"] == [20.0]
    assert thread_start_samples["threadstart:cudaEventRecord"] == [20.0]
    assert pair_occurrence_samples["pairocc:cudaEventRecord->cudaLaunchKernel#0"] == [1.0]
    assert pair_samples["pair:cudaEventRecord->cudaLaunchKernel"] == [1.0]
    assert api_samples["cudaEventRecord"] == [1.0]
    assert type_samples["stream_op"] == [1.0]
    assert "pair:cudaGetDevice->cudaGetLastError" not in pair_samples


def test_build_profile_lines_emits_type_and_api_entries() -> None:
    lines = build_profile_lines(
        {
            "pairocc:cudaGetDevice->cudaGetLastError#0": [9.0, 11.0],
            "pairocc:cudaGetDevice->cudaGetLastError#9": [25.0],
        },
        {
            "pair:cudaGetDevice->cudaGetLastError": [2.0, 4.0, 6.0],
            "pair:cudaEventRecord->cudaLaunchKernel": [1.0],
        },
        {
            "cudaGetDevice": [1.0, 3.0, 5.0],
            "cudaGetLastError": [2.0, 4.0, 6.0],
        },
        {
            "context_op": [1.0, 2.0, 3.0],
        },
        {
            "threadstartocc:cudaGetDevice#0": [0.0],
            "threadstartocc:cudaGetDevice#1": [60.0],
        },
        {
            "threadstart:cudaGetDevice": [0.0, 60.0],
        },
        statistic="percentile",
        percentile=50.0,
        max_pair_occurrence_index=8,
        min_pair_occurrence_samples=1,
        pair_occurrence_min_delta_us=1.0,
        pair_occurrence_min_ratio=1.0,
        max_pair_occurrence_delay_us=0.0,
        min_pair_samples=2,
        min_api_samples=2,
    )

    assert "default=0" in lines
    assert "threadstartocc:cudaGetDevice#0=0.000000" in lines
    assert "threadstartocc:cudaGetDevice#1=60.000000" in lines
    assert "threadstart:cudaGetDevice=30.000000" in lines
    assert "pairocc:cudaGetDevice->cudaGetLastError#0=10.000000" in lines
    assert "pairocc:cudaGetDevice->cudaGetLastError#9=25.000000" not in lines
    assert "pair:cudaGetDevice->cudaGetLastError=4.000000" in lines
    assert "pair:cudaEventRecord->cudaLaunchKernel=1.000000" not in lines
    assert "type:context_op=2.000000" in lines
    assert "cudaGetDevice=3.000000" in lines
    assert "cudaGetLastError=4.000000" in lines


def test_build_profile_lines_can_use_mean_statistic() -> None:
    lines = build_profile_lines(
        {"pairocc:cudaGetDevice->cudaGetLastError#0": [1.0, 5.0]},
        {"pair:cudaGetDevice->cudaGetLastError": [1.0, 5.0]},
        {"cudaGetDevice": [1.0, 5.0]},
        {"context_op": [1.0, 5.0]},
        {"threadstartocc:cudaGetDevice#0": [1.0, 5.0]},
        {"threadstart:cudaGetDevice": [1.0, 5.0]},
        statistic="mean",
        percentile=50.0,
        max_pair_occurrence_index=8,
        min_pair_occurrence_samples=1,
        pair_occurrence_min_delta_us=1.0,
        pair_occurrence_min_ratio=1.0,
        max_pair_occurrence_delay_us=0.0,
        min_pair_samples=1,
        min_api_samples=1,
    )

    assert "threadstartocc:cudaGetDevice#0=3.000000" in lines
    assert "threadstart:cudaGetDevice=3.000000" in lines
    assert "pairocc:cudaGetDevice->cudaGetLastError#0=3.000000" in lines
    assert "pair:cudaGetDevice->cudaGetLastError=3.000000" in lines
    assert "type:context_op=3.000000" in lines
    assert "cudaGetDevice=3.000000" in lines


def test_build_profile_lines_filters_sparse_occurrence_entries() -> None:
    lines = build_profile_lines(
        {
            "pairocc:cudaGetDevice->cudaGetLastError#0": [100.0, 200.0, 300.0],
            "pairocc:cudaGetDevice->cudaGetLastError#1": [110.0, 210.0, 310.0, 410.0],
        },
        {
            "pair:cudaGetDevice->cudaGetLastError": [1.0, 2.0, 3.0, 4.0],
        },
        {
            "cudaGetDevice": [1.0, 2.0, 3.0, 4.0],
        },
        {
            "context_op": [1.0, 2.0, 3.0, 4.0],
        },
        {},
        {},
        statistic="percentile",
        percentile=50.0,
        max_pair_occurrence_index=16,
        min_pair_occurrence_samples=4,
        pair_occurrence_min_delta_us=1.0,
        pair_occurrence_min_ratio=1.0,
        max_pair_occurrence_delay_us=0.0,
        min_pair_samples=1,
        min_api_samples=1,
    )

    assert "pairocc:cudaGetDevice->cudaGetLastError#0=200.000000" not in lines
    assert "pairocc:cudaGetDevice->cudaGetLastError#1=260.000000" in lines


def test_build_profile_lines_emits_only_material_occurrence_overrides_and_caps_them() -> None:
    lines = build_profile_lines(
        {
            "pairocc:cudaGetDevice->cudaGetLastError#0": [5.0, 7.0],
            "pairocc:cudaGetDevice->cudaGetLastError#1": [1500000.0, 1700000.0],
        },
        {
            "pair:cudaGetDevice->cudaGetLastError": [4.0, 5.0, 6.0],
        },
        {
            "cudaGetDevice": [4.0, 5.0, 6.0],
        },
        {
            "context_op": [4.0, 5.0, 6.0],
        },
        {},
        {},
        statistic="percentile",
        percentile=50.0,
        max_pair_occurrence_index=0,
        min_pair_occurrence_samples=2,
        pair_occurrence_min_delta_us=1000.0,
        pair_occurrence_min_ratio=8.0,
        max_pair_occurrence_delay_us=250000.0,
        min_pair_samples=1,
        min_api_samples=1,
    )

    assert "pairocc:cudaGetDevice->cudaGetLastError#0" not in lines
    assert "pairocc:cudaGetDevice->cudaGetLastError#1=250000.000000" in lines


def test_build_host_timing_profile_defaults_preserve_occurrence_fidelity() -> None:
    args = _build_parser().parse_args(
        [
            "dummy-traces",
            "--output",
            "dummy.profile",
        ]
    )

    assert args.max_pair_occurrence_index == 0
    assert args.min_pair_occurrence_samples == 1
    assert args.pair_occurrence_min_delta_us == 0.0
    assert args.pair_occurrence_min_ratio == 0.0
    assert args.max_pair_occurrence_delay_us == 0.0
    assert args.dispatch_scope == "host_machine"
    assert args.profile_surface == "semantic"


def test_collect_host_gap_samples_uses_manifest_step_windows(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "step_windows": {
                    "0": {"start_ts": 100, "end_ts": 220},
                }
            }
        ),
        encoding="utf-8",
    )
    trace_path = trace_dir / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 10,
                        "pid": 1,
                        "tid": 7,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetDevice",
                        "type": "context_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 20,
                        "pid": 1,
                        "tid": 7,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetLastError",
                        "type": "context_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 100,
                        "pid": 1,
                        "tid": 7,
                        "mod": "libcudart.so.12",
                        "api": "ncclGroupEnd",
                        "type": "other",
                    }
                ),
                json.dumps(
                    {
                        "ts": 180,
                        "pid": 1,
                        "tid": 7,
                        "mod": "libcudart.so.12",
                        "api": "ncclAllReduce",
                        "type": "nccl_collective",
                    }
                ),
                json.dumps(
                    {
                        "ts": 220,
                        "pid": 1,
                        "tid": 7,
                        "mod": "libcudart.so.12",
                        "api": "ncclCommGetAsyncError",
                        "type": "other",
                    }
                ),
                json.dumps(
                    {
                        "ts": 260,
                        "pid": 1,
                        "tid": 7,
                        "mod": "libcudart.so.12",
                        "api": "ncclAllReduce",
                        "type": "nccl_collective",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    step_windows = _load_step_windows(trace_dir, window_mode="auto")
    (
        pair_occurrence_samples,
        pair_samples,
        api_samples,
        _,
        processed_events,
        thread_start_occurrence_samples,
        thread_start_samples,
    ) = collect_host_gap_samples(
        trace_dir,
        step_windows=step_windows,
    )

    assert step_windows == {0: (100, 220)}
    assert processed_events == 3
    assert thread_start_occurrence_samples["threadstartocc:ncclAllReduce#0"] == [80.0]
    assert thread_start_samples["threadstart:ncclAllReduce"] == [80.0]
    assert pair_occurrence_samples["pairocc:ncclAllReduce->ncclAllReduce#0"] == [40.0]
    assert pair_samples["pair:ncclAllReduce->ncclAllReduce"] == [40.0]
    assert api_samples["ncclAllReduce"] == [40.0]


def test_load_step_windows_falls_back_to_markers_for_missing_ranks(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "step_windows": {
                    "0": {"start_ts": 100, "end_ts": 220},
                }
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        '{"ts":100,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}\n',
        encoding="utf-8",
    )
    (trace_dir / "rank_1.jsonl").write_text(
        '{"ts":120,"pid":2,"tid":2,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}\n',
        encoding="utf-8",
    )
    (trace_dir / "rank_1.markers.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "kind": "step_begin",
                        "label": "training_step",
                        "pid": 2,
                        "monotonic_ns": 150_000,
                        "realtime_ns": 150_000,
                        "step": 1,
                    }
                ),
                json.dumps(
                    {
                        "kind": "step_end",
                        "label": "training_step",
                        "pid": 2,
                        "monotonic_ns": 240_000,
                        "realtime_ns": 240_000,
                        "step": 1,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    step_windows = _load_step_windows(trace_dir, window_mode="auto")

    assert step_windows == {
        0: (100, 220),
        1: (150, 240),
    }


def test_load_step_windows_ignores_non_paper_valid_fidelity_windows(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "fidelity_windows": {
                    "0": {"start_ts": 100, "end_ts": 220, "source": "boundary_fallback"},
                    "1": {"start_ts": 150, "end_ts": 240, "source": "trace_markers"},
                }
            }
        ),
        encoding="utf-8",
    )

    step_windows = _load_step_windows(trace_dir, window_mode="auto")

    assert step_windows == {
        1: (150, 240),
    }


def test_collect_host_gap_samples_can_include_all_previous_types(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_path = trace_dir / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 10,
                        "pid": 1,
                        "tid": 1,
                        "mod": "libcudart.so.12",
                        "api": "cudaLaunchKernel",
                        "type": "kernel_launch",
                    }
                ),
                json.dumps(
                    {
                        "ts": 25,
                        "pid": 1,
                        "tid": 1,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetLastError",
                        "type": "other",
                    }
                ),
                json.dumps(
                    {
                        "ts": 40,
                        "pid": 1,
                        "tid": 1,
                        "mod": "libcudart.so.12",
                        "api": "cudaEventSynchronize",
                        "type": "stream_op",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _, pair_samples, api_samples, type_samples, _, _, _ = collect_host_gap_samples(
        trace_dir,
        include_types=None,
    )

    assert pair_samples["pair:cudaLaunchKernel->cudaEventSynchronize"] == [30.0]
    assert api_samples["cudaLaunchKernel"] == [30.0]
    assert type_samples["kernel_launch"] == [30.0]


def test_collect_host_gap_samples_emits_thread_start_occurrence_offsets(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_path = trace_dir / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 100,
                        "pid": 1,
                        "tid": 11,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetDevice",
                        "type": "context_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 130,
                        "pid": 1,
                        "tid": 11,
                        "mod": "libcudart.so.12",
                        "api": "cudaLaunchKernel",
                        "type": "kernel_launch",
                    }
                ),
                json.dumps(
                    {
                        "ts": 120,
                        "pid": 1,
                        "tid": 13,
                        "mod": "libnccl.so.2",
                        "api": "ncclCommGetAsyncError",
                        "type": "other",
                    }
                ),
                json.dumps(
                    {
                        "ts": 150,
                        "pid": 1,
                        "tid": 13,
                        "mod": "libnccl.so.2",
                        "api": "ncclAllReduce",
                        "type": "nccl_collective",
                    }
                ),
                json.dumps(
                    {
                        "ts": 160,
                        "pid": 1,
                        "tid": 12,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetDevice",
                        "type": "context_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 190,
                        "pid": 1,
                        "tid": 12,
                        "mod": "libcudart.so.12",
                        "api": "cudaStreamWaitEvent",
                        "type": "stream_op",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (
        _,
        _,
        _,
        _,
        processed_events,
        thread_start_occurrence_samples,
        thread_start_samples,
    ) = collect_host_gap_samples(trace_dir, dispatch_scope="thread")

    assert processed_events == 6
    assert thread_start_occurrence_samples["threadstartocc:cudaLaunchKernel#0"] == [30.0]
    assert thread_start_occurrence_samples["threadstartocc:ncclAllReduce#0"] == [50.0]
    assert thread_start_occurrence_samples["threadstartocc:cudaStreamWaitEvent#0"] == [90.0]
    assert thread_start_samples["threadstart:cudaLaunchKernel"] == [30.0]
    assert thread_start_samples["threadstart:ncclAllReduce"] == [50.0]
    assert thread_start_samples["threadstart:cudaStreamWaitEvent"] == [90.0]


def test_collect_host_gap_samples_canonicalizes_control_plane_aliases(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_path = trace_dir / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 10,
                        "pid": 1,
                        "tid": 7,
                        "mod": "libcudart.so.12",
                        "api": "cudaEventCreate",
                        "type": "event_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 14,
                        "pid": 1,
                        "tid": 7,
                        "mod": "libcudart.so.12",
                        "api": "cudaEventRecordWithFlags",
                        "type": "stream_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 20,
                        "pid": 1,
                        "tid": 9,
                        "mod": "libcudart.so.12",
                        "api": "cudaStreamCreateWithPriority",
                        "type": "stream_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 24,
                        "pid": 1,
                        "tid": 7,
                        "mod": "libcudart.so.12",
                        "api": "cudaEventSynchronize",
                        "type": "stream_op",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (
        pair_occurrence_samples,
        pair_samples,
        api_samples,
        _,
        _,
        thread_start_occurrence_samples,
        thread_start_samples,
    ) = collect_host_gap_samples(trace_dir)

    assert _canonicalize_host_timing_api("cudaEventRecordWithFlags") == "cudaEventRecord"
    assert _canonicalize_host_timing_api("cudaStreamCreateWithPriority") == "cudaStreamCreate"
    assert pair_occurrence_samples["pairocc:cudaEventRecord->cudaStreamCreate#0"] == [10.0]
    assert pair_occurrence_samples["pairocc:cudaStreamCreate->cudaEventSynchronize#0"] == [4.0]
    assert pair_samples["pair:cudaEventRecord->cudaStreamCreate"] == [10.0]
    assert pair_samples["pair:cudaStreamCreate->cudaEventSynchronize"] == [4.0]
    assert api_samples["cudaEventRecord"] == [10.0]
    assert api_samples["cudaStreamCreate"] == [4.0]
    assert thread_start_occurrence_samples["threadstartocc:cudaEventRecord#0"] == [4.0]
    assert "threadstartocc:cudaStreamCreate#0" not in thread_start_occurrence_samples
    assert "threadstart:cudaStreamCreate" not in thread_start_samples


def test_collect_host_gap_samples_process_scope_uses_single_dispatch_queue(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_path = trace_dir / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 100,
                        "pid": 1,
                        "tid": 7,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetDevice",
                        "type": "context_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 120,
                        "pid": 1,
                        "tid": 9,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetLastError",
                        "type": "other",
                    }
                ),
                json.dumps(
                    {
                        "ts": 180,
                        "pid": 1,
                        "tid": 7,
                        "mod": "libcudart.so.12",
                        "api": "cudaLaunchKernel",
                        "type": "kernel_launch",
                    }
                ),
                json.dumps(
                    {
                        "ts": 220,
                        "pid": 1,
                        "tid": 7,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetDevice",
                        "type": "context_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 260,
                        "pid": 1,
                        "tid": 9,
                        "mod": "libcudart.so.12",
                        "api": "cudaStreamWaitEvent",
                        "type": "stream_op",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    pair_occurrence_samples, pair_samples, api_samples, type_samples, _, thread_start_occurrence_samples, thread_start_samples = collect_host_gap_samples(
        trace_dir,
        dispatch_scope="process",
    )

    assert thread_start_occurrence_samples["threadstartocc:cudaLaunchKernel#0"] == [80.0]
    assert thread_start_samples["threadstart:cudaLaunchKernel"] == [80.0]
    assert pair_occurrence_samples["pairocc:cudaLaunchKernel->cudaStreamWaitEvent#0"] == [80.0]
    assert pair_samples["pair:cudaLaunchKernel->cudaStreamWaitEvent"] == [80.0]
    assert api_samples["cudaLaunchKernel"] == [80.0]
    assert type_samples["kernel_launch"] == [80.0]


def test_collect_helper_thread_templates_detects_nccl_pollers(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_path = trace_dir / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 100,
                        "pid": 1,
                        "tid": 10,
                        "mod": "libcudart.so.12",
                        "api": "__cudaRegisterFunction",
                        "type": "context_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 120,
                        "pid": 1,
                        "tid": 10,
                        "mod": "libcudart.so.12",
                        "api": "cudaLaunchKernel",
                        "type": "kernel_launch",
                    }
                ),
                json.dumps(
                    {
                        "ts": 500,
                        "pid": 1,
                        "tid": 20,
                        "mod": "libnccl.so.2",
                        "api": "ncclGetVersion",
                        "type": "other",
                    }
                ),
                json.dumps(
                    {
                        "ts": 900,
                        "pid": 1,
                        "tid": 20,
                        "mod": "libnccl.so.2",
                        "api": "ncclCommGetAsyncError",
                        "type": "other",
                    }
                ),
                json.dumps(
                    {
                        "ts": 1200,
                        "pid": 1,
                        "tid": 20,
                        "mod": "libcudart.so.12",
                        "api": "cudaSetDevice",
                        "type": "context_op",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    templates = _collect_helper_thread_templates(trace_dir)

    assert 0 in templates
    assert len(templates[0]) == 1
    template = templates[0][0]
    assert template.first_api == "ncclGetVersion"
    assert template.dominant_api == "ncclCommGetAsyncError"
    assert template.start_offset_us == 400
    assert template.end_offset_us == 1100
    assert template.api_sequence == (
        "ncclGetVersion",
        "ncclCommGetAsyncError",
        "cudaSetDevice",
    )


def test_collect_helper_thread_templates_rejects_cublaslt_setup_threads(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_path = trace_dir / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 100,
                        "pid": 1,
                        "tid": 10,
                        "mod": "libcudart.so.12",
                        "api": "cudaLaunchKernel",
                        "type": "kernel_launch",
                    }
                ),
                json.dumps(
                    {
                        "ts": 500,
                        "pid": 1,
                        "tid": 20,
                        "mod": "libnccl.so.2",
                        "api": "ncclGetVersion",
                        "type": "other",
                    }
                ),
                json.dumps(
                    {
                        "ts": 700,
                        "pid": 1,
                        "tid": 20,
                        "mod": "libnccl.so.2",
                        "api": "ncclCommGetAsyncError",
                        "type": "other",
                    }
                ),
                json.dumps(
                    {
                        "ts": 900,
                        "pid": 1,
                        "tid": 20,
                        "mod": "libcudart.so.12",
                        "api": "cudaSetDevice",
                        "type": "context_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 1100,
                        "pid": 1,
                        "tid": 20,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetDevice",
                        "type": "context_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 1300,
                        "pid": 1,
                        "tid": 20,
                        "mod": "libcublasLt.so.12",
                        "api": "cublasLtCreate",
                        "type": "context_op",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    templates = _collect_helper_thread_templates(trace_dir)

    assert 0 in templates
    assert templates[0] == []


def test_collect_helper_thread_templates_rejects_extended_nccl_collective_threads(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_path = trace_dir / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 100,
                        "pid": 1,
                        "tid": 10,
                        "mod": "libcudart.so.12",
                        "api": "cudaLaunchKernel",
                        "type": "kernel_launch",
                    }
                ),
                json.dumps(
                    {
                        "ts": 500,
                        "pid": 1,
                        "tid": 20,
                        "mod": "libnccl.so.2",
                        "api": "ncclGetVersion",
                        "type": "other",
                    }
                ),
                json.dumps(
                    {
                        "ts": 700,
                        "pid": 1,
                        "tid": 20,
                        "mod": "libnccl.so.2",
                        "api": "ncclCommGetAsyncError",
                        "type": "other",
                    }
                ),
                json.dumps(
                    {
                        "ts": 900,
                        "pid": 1,
                        "tid": 20,
                        "mod": "libcudart.so.12",
                        "api": "cudaSetDevice",
                        "type": "context_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 1100,
                        "pid": 1,
                        "tid": 20,
                        "mod": "libnccl.so.2",
                        "api": "ncclAllToAll",
                        "type": "nccl_collective",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    templates = _collect_helper_thread_templates(trace_dir)

    assert 0 in templates
    assert templates[0] == []


def test_summary_whitelist_keeps_context_helper_templates(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_path = trace_dir / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 100,
                        "pid": 1,
                        "tid": 10,
                        "mod": "libcudart.so.12",
                        "api": "cudaLaunchKernel",
                        "type": "kernel_launch",
                    }
                ),
                json.dumps(
                    {
                        "ts": 500,
                        "pid": 1,
                        "tid": 20,
                        "mod": "libcudart.so.12",
                        "api": "cudaSetDevice",
                        "type": "context_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 900,
                        "pid": 1,
                        "tid": 20,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetDevice",
                        "type": "context_op",
                    }
                ),
                json.dumps(
                    {
                        "ts": 1200,
                        "pid": 1,
                        "tid": 20,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetLastError",
                        "type": "other",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    templates = _collect_helper_thread_templates(trace_dir)

    assert 0 in templates
    assert len(templates[0]) == 1
    template = templates[0][0]
    assert template.first_api == "cudaSetDevice"
    assert template.dominant_api == "cudaSetDevice"
    assert template.api_sequence == (
        "cudaSetDevice",
        "cudaGetDevice",
        "cudaGetLastError",
    )
