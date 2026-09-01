import json
from pathlib import Path

import pytest

from flexsim.maya_lite import (
    TraceSource,
    dedup_pattern_rank_traces,
    inspect_trace_directory,
    iter_rank_trace_events,
    load_trace_directory,
    materialize_profiled_rank_traces,
)
from flexsim.maya_lite.communicators import build_emulated_communicator_id
from flexsim.maya_lite.io import (
    estimate_rank_trace_active_seconds,
    estimate_rank_trace_window,
    infer_trace_source,
)
from flexsim.maya_lite.schema import TraceEvent


TRACE_ROOT = Path("paper/traces")


def _trace_dirs() -> list[Path]:
    if not TRACE_ROOT.exists():
        return []
    return sorted(
        path
        for path in TRACE_ROOT.glob("*/*")
        if path.is_dir() and any(path.glob("rank_*.jsonl"))
    )


@pytest.mark.parametrize("trace_dir", _trace_dirs(), ids=lambda path: path.as_posix())
def test_inspect_trace_directory_current_corpus(trace_dir: Path):
    summary = inspect_trace_directory(trace_dir, sample_events_per_rank=2)
    assert summary.rank_files
    assert summary.rank_ids
    assert "api" in summary.observed_keys
    assert "ts" in summary.observed_keys
    assert "kernel_launch" in summary.observed_types or "other" in summary.observed_types
    expected_source = TraceSource.REAL if "real" in trace_dir.parts else TraceSource.FAKE
    assert summary.source is expected_source


@pytest.mark.parametrize(
    ("trace_dir", "expected_source"),
    [
        (Path("paper/traces/real/e1"), TraceSource.REAL),
        (Path("paper/traces/fake/e3"), TraceSource.FAKE),
    ],
)
def test_load_trace_directory_sample(trace_dir: Path, expected_source: TraceSource):
    if not trace_dir.exists():
        pytest.skip(f"trace dir not available: {trace_dir}")

    bundle = load_trace_directory(trace_dir, max_events_per_rank=8)
    assert bundle.source is expected_source
    assert bundle.world_size >= 1
    assert bundle.total_events >= bundle.world_size
    assert tuple(sorted(bundle.rank_ids())) == bundle.rank_ids()

    first_rank = bundle.rank_traces[0]
    assert first_rank.events
    first_event = first_rank.events[0]
    assert first_event.rank == first_rank.rank
    assert first_event.source is expected_source
    assert first_event.api
    assert first_event.module
    assert first_event.op_type


def test_iter_rank_trace_events_preserves_rank_and_order():
    trace_file = Path("paper/traces/real/e1/rank_0.jsonl")
    if not trace_file.exists():
        pytest.skip(f"trace file not available: {trace_file}")

    events = []
    for index, event in enumerate(iter_rank_trace_events(trace_file)):
        events.append(event)
        if index >= 4:
            break

    assert events
    assert all(event.rank == 0 for event in events)
    assert [event.ordinal for event in events] == sorted(event.ordinal for event in events)
    assert all(event.api for event in events)


def test_infer_trace_source_remote_traces_are_real():
    assert infer_trace_source("paper/maya_lite/remote_traces/tc30230_simple_20260324") is TraceSource.REAL


def test_trace_event_preserves_boundary_visibility_segment_extras():
    record = {
        "ts": 1,
        "pid": 2,
        "tid": 3,
        "mod": "libcudart.so.12",
        "api": "cudaLaunchKernel",
        "type": "kernel_launch",
        "boundary_segment_schema_version": "launch_boundary_visibility_v1",
        "launch_boundary_id_unavailable_reason": (
            "fakecuda_launch_pair_id_disabled_to_preserve_host_duration"
        ),
        "boundary_visibility_segments": [
            {
                "name": "real_api_body",
                "visibility_kind": "mixed_or_unresolved",
                "duration_us": None,
                "clock": "unmeasured",
            }
        ],
    }

    event = TraceEvent.from_json_record(record, rank=0, ordinal=0, source=TraceSource.FAKE)

    assert event.extras["boundary_segment_schema_version"] == "launch_boundary_visibility_v1"
    assert event.extras["launch_boundary_id_unavailable_reason"] == (
        "fakecuda_launch_pair_id_disabled_to_preserve_host_duration"
    )
    assert event.extras["boundary_visibility_segments"][0]["name"] == "real_api_body"
    assert event.extras["boundary_visibility_segments"][0]["duration_us"] is None


def test_load_trace_directory_infers_real_source_from_manifest_when_staged_path_loses_name(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "staged"
    trace_dir.mkdir()
    (trace_dir / "rank_0.jsonl").write_text(
        '{"ts":1,"pid":1,"tid":1,"mod":"libcublas.so.12","api":"cublasGemmEx","type":"blas_compute"}\n',
        encoding="utf-8",
    )
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "host_timing_paper_alignment_line": "disabled",
                "host_timing_line_family": "disabled",
                "step_windows": {"0": {"start_ts": 1, "end_ts": 10, "source": "trace_markers"}},
            }
        ),
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir)

    assert bundle.source is TraceSource.REAL
    assert bundle.rank_traces[0].source is TraceSource.REAL


def test_load_trace_directory_infers_fake_source_from_manifest_when_staged_path_loses_name(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "staged"
    trace_dir.mkdir()
    (trace_dir / "rank_0.jsonl").write_text(
        '{"ts":1,"pid":1,"tid":1,"mod":"libcublas.so.12","api":"cublasGemmEx","type":"blas_compute"}\n',
        encoding="utf-8",
    )
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "mode": "emulated_phase1",
                "host_timing_mode": "measure",
                "step_windows": {"0": {"start_ts": 1, "end_ts": 10, "source": "trace_markers"}},
            }
        ),
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir)

    assert bundle.source is TraceSource.FAKE
    assert bundle.rank_traces[0].source is TraceSource.FAKE


def test_iter_rank_trace_events_skips_invalid_json_by_default(tmp_path: Path):
    trace_file = tmp_path / "rank_0.jsonl"
    trace_file.write_text(
        "\n".join(
            [
                '{"ts":1,"pid":1,"tid":1,"mod":"libcudart.so","api":"cudaGetDevice","type":"context_op"}',
                '{"ts":2,"pi',
                '{"ts":3,"pid":1,"tid":1,"mod":"libcudart.so","api":"cudaLaunchKernel","type":"kernel_launch"}',
            ]
        )
    )

    with pytest.warns(RuntimeWarning, match="skipping invalid JSON"):
        events = list(iter_rank_trace_events(trace_file))

    assert [event.ordinal for event in events] == [0, 2]
    assert [event.api for event in events] == ["cudaGetDevice", "cudaLaunchKernel"]


def test_iter_rank_trace_events_strict_json_raises(tmp_path: Path):
    trace_file = tmp_path / "rank_0.jsonl"
    trace_file.write_text('{"ts":1,"pi')

    with pytest.raises(ValueError, match="invalid JSON"):
        list(iter_rank_trace_events(trace_file, strict_json=True))


def test_estimate_rank_trace_window_uses_ts_extrema_not_file_order(tmp_path: Path) -> None:
    trace_file = tmp_path / "rank_0.jsonl"
    trace_file.write_text(
        "\n".join(
            [
                '{"ts":200,"pid":1,"tid":11,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
                '{"ts":50,"pid":1,"tid":12,"mod":"libcudart.so.12","api":"cudaGetDevice","type":"context_op"}',
                '{"ts":900,"pid":1,"tid":13,"mod":"libcublas.so.12","api":"cublasGemmEx","type":"blas_compute"}',
                '{"ts":400,"pid":1,"tid":14,"mod":"libcudart.so.12","api":"cudaGetLastError","type":"other"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert estimate_rank_trace_window(trace_file) == {
        "start_ts": 50,
        "end_ts": 900,
        "source": "boundary_fallback",
    }


def test_estimate_rank_trace_window_uses_end_ts_for_upper_bound_when_present(tmp_path: Path) -> None:
    trace_file = tmp_path / "rank_0.jsonl"
    trace_file.write_text(
        "\n".join(
            [
                '{"ts":200,"end_ts":800,"pid":1,"tid":11,"mod":"libcudart.so.12","api":"cudaMemcpy","type":"mem_copy"}',
                '{"ts":50,"pid":1,"tid":12,"mod":"libcudart.so.12","api":"cudaGetDevice","type":"context_op"}',
                '{"ts":500,"pid":1,"tid":13,"mod":"libcublas.so.12","api":"cublasGemmEx","type":"blas_compute"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert estimate_rank_trace_window(trace_file) == {
        "start_ts": 50,
        "end_ts": 800,
        "source": "boundary_fallback",
    }


def test_iter_rank_trace_events_accepts_raw_temp_trace_name(tmp_path: Path) -> None:
    trace_file = tmp_path / "rank_7.raw.jsonl"
    trace_file.write_text(
        "\n".join(
            [
                '{"ts":10,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaGetDevice","type":"context_op"}',
                '{"ts":20,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    events = list(iter_rank_trace_events(trace_file))

    assert [event.rank for event in events] == [7, 7]
    assert [event.ordinal for event in events] == [0, 1]


def test_estimate_rank_trace_window_accepts_raw_temp_trace_name(tmp_path: Path) -> None:
    trace_file = tmp_path / "rank_3.raw.jsonl"
    trace_file.write_text(
        "\n".join(
            [
                '{"ts":600,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaGetDevice","type":"context_op"}',
                '{"ts":120,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert estimate_rank_trace_window(trace_file) == {
        "start_ts": 120,
        "end_ts": 600,
        "source": "boundary_fallback",
    }


def test_load_trace_directory_ignores_raw_temp_traces(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "rank_0.jsonl").write_text(
        '{"ts":1,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}\n',
        encoding="utf-8",
    )
    (trace_dir / "rank_0.raw.jsonl").write_text(
        '{"ts":999,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaGetLastError","type":"other"}\n',
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir)

    assert bundle.rank_ids() == (0,)
    assert len(bundle.rank_traces[0].events) == 1
    assert bundle.rank_traces[0].events[0].api == "cudaLaunchKernel"


def test_load_trace_directory_injects_host_machine_id_from_manifest(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "rank_0.jsonl").write_text(
        '{"ts":1,"pid":17,"tid":3,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}\n',
        encoding="utf-8",
    )
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps({"rank_host_machines": {"0": "nodeA"}}),
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir)

    assert bundle.rank_host_machines == {0: "nodeA"}
    assert bundle.rank_host_dispatch_queues == {0: "nodeA:rank:0"}
    assert bundle.rank_traces[0].events[0].extras["host_machine_id"] == "nodeA"
    assert bundle.rank_traces[0].events[0].extras["host_dispatch_queue_id"] == "nodeA:rank:0"


def test_load_trace_directory_preserves_explicit_host_dispatch_queue_id(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "rank_0.jsonl").write_text(
        '{"ts":1,"pid":17,"tid":3,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}\n',
        encoding="utf-8",
    )
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "rank_host_machines": {"0": "nodeA"},
                "rank_host_dispatch_queues": {"0": "nodeA:worker0"},
            }
        ),
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir)

    assert bundle.rank_host_machines == {0: "nodeA"}
    assert bundle.rank_host_dispatch_queues == {0: "nodeA:worker0"}
    assert bundle.rank_traces[0].events[0].extras["host_machine_id"] == "nodeA"
    assert bundle.rank_traces[0].events[0].extras["host_dispatch_queue_id"] == "nodeA:worker0"


def test_trace_event_normalizes_older_cuda_launchkernel_type():
    event = TraceEvent.from_json_record(
        {
            "ts": 1,
            "pid": 2,
            "tid": 3,
            "mod": "libcudart.so.12",
            "api": "cudaLaunchKernel",
            "type": "context_op",
        },
        rank=0,
        ordinal=0,
        source=TraceSource.REAL,
    )

    assert event.op_type == "kernel_launch"


def test_trace_event_normalizes_cublas_control_plane_types_from_legacy_other():
    cases = [
        ("cublasCreate_v2", "context_op"),
        ("cublasDestroy_v2", "context_op"),
        ("cublasSetStream_v2", "stream_op"),
        ("cublasLtCreate", "context_op"),
        ("cublasLtMatmulDescCreate", "context_op"),
        ("cublasLtMatmulPreferenceCreate", "context_op"),
    ]

    for ordinal, (api, expected) in enumerate(cases):
        event = TraceEvent.from_json_record(
            {
                "ts": ordinal + 1,
                "pid": 2,
                "tid": 3,
                "mod": "libcublas.so.12",
                "api": api,
                "type": "other",
            },
            rank=0,
            ordinal=ordinal,
            source=TraceSource.FAKE,
        )

        assert event.op_type == expected


def test_trace_event_normalizes_additional_nccl_collectives_from_legacy_other():
    cases = [
        "ncclReduce",
        "ncclAllToAll",
        "ncclAllToAllv",
    ]

    for ordinal, api in enumerate(cases):
        event = TraceEvent.from_json_record(
            {
                "ts": ordinal + 1,
                "pid": 2,
                "tid": 3,
                "mod": "libnccl.so.2",
                "api": api,
                "type": "other",
            },
            rank=0,
            ordinal=ordinal,
            source=TraceSource.FAKE,
        )

        assert event.op_type == "nccl_collective"


def test_load_trace_directory_honors_capture_manifest(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        '{"original_world_size": 4, "profiled_ranks": [0, 2], "profiled_rank_groups": {"0": [0, 1], "2": [2, 3]}}'
    )
    (trace_dir / "rank_0.jsonl").write_text(
        '{"ts":1,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}\n'
    )
    (trace_dir / "rank_2.jsonl").write_text(
        '{"ts":2,"pid":2,"tid":2,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}\n'
    )

    bundle = load_trace_directory(trace_dir)

    assert bundle.world_size == 4
    assert bundle.profiled_world_size == 2
    assert bundle.rank_ids() == (0, 2)
    assert bundle.profiled_rank_groups == {0: (0, 1), 2: (2, 3)}
    assert bundle.trace_window == "full"


def test_load_trace_directory_reads_communicator_memberships_from_manifest(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "original_world_size": 4,
                "communicators": {
                    "comm-a": {"members": [0, 2], "size": 2, "name": "dp"},
                    "comm-b": {"members": [1, 3], "size": 2, "name": "dp"},
                },
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        '{"ts":1,"pid":1,"tid":1,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","comm_id":"comm-a","call_idx":"0"}\n',
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir)

    assert bundle.communicator_memberships == {
        "comm-a": (0, 2),
        "comm-b": (1, 3),
    }


def test_load_trace_directory_reuses_transparent_trace_bundle_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import flexsim.maya_lite.io as io_module
    monkeypatch.setenv("FLEXSIM_MAYA_LITE_TRACE_BUNDLE_CACHE", "1")

    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "rank_0.jsonl").write_text(
        '{"ts":1,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}\n',
        encoding="utf-8",
    )

    first = load_trace_directory(trace_dir)
    assert first.total_events == 1
    assert (trace_dir / ".maya_lite_cache").exists()

    def _unexpected_iter(*args, **kwargs):
        raise AssertionError("cache should avoid re-reading raw rank traces")

    monkeypatch.setattr(io_module, "iter_rank_trace_events", _unexpected_iter)

    second = load_trace_directory(trace_dir)

    assert second.total_events == 1
    assert second.rank_ids() == (0,)


def test_load_trace_directory_invalidates_cache_when_rank_trace_changes(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    trace_file = trace_dir / "rank_0.jsonl"
    trace_file.write_text(
        '{"ts":1,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}\n',
        encoding="utf-8",
    )

    first = load_trace_directory(trace_dir)
    assert first.total_events == 1

    trace_file.write_text(
        '{"ts":1,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}\n'
        '{"ts":2,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaDeviceSynchronize","type":"stream_op"}\n',
        encoding="utf-8",
    )

    second = load_trace_directory(trace_dir)

    assert second.total_events == 2


def test_load_trace_directory_infers_communicator_memberships_from_trace_init_events(
    tmp_path: Path,
):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                '{"ts":1,"pid":1,"tid":1,"mod":"libnccl.so.2","api":"ncclCommInitRankConfig","type":"other","comm_id":"comm-a","rank":"0","nranks":"2"}',
                '{"ts":2,"pid":1,"tid":1,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","comm_id":"comm-a","call_idx":"0"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (trace_dir / "rank_2.jsonl").write_text(
        '{"ts":3,"pid":2,"tid":2,"mod":"libnccl.so.2","api":"ncclCommInitRankConfig","type":"other","comm_id":"comm-a","rank":"1","nranks":"2"}\n',
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir)

    assert bundle.communicator_memberships == {"comm-a": (0, 2)}


def test_load_trace_directory_infers_communicator_memberships_from_nccl_comm_init_rank_events(
    tmp_path: Path,
):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "rank_0.jsonl").write_text(
        '{"ts":1,"pid":1,"tid":1,"mod":"libnccl.so.2","api":"ncclCommInitRank","type":"other","comm_id":"comm-a","rank":"0","nranks":"2"}\n',
        encoding="utf-8",
    )
    (trace_dir / "rank_2.jsonl").write_text(
        '{"ts":2,"pid":2,"tid":2,"mod":"libnccl.so.2","api":"ncclCommInitRank","type":"other","comm_id":"comm-a","rank":"1","nranks":"2"}\n',
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir)

    assert bundle.communicator_memberships == {"comm-a": (0, 2)}


def test_load_trace_directory_trace_communicators_override_missing_manifest_memberships(
    tmp_path: Path,
):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps({"original_world_size": 2}),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        '{"ts":1,"pid":1,"tid":1,"mod":"libnccl.so.2","api":"ncclCommInitRankConfig","type":"other","comm_id":"comm-a","rank":"0","nranks":"2"}\n',
        encoding="utf-8",
    )
    (trace_dir / "rank_1.jsonl").write_text(
        '{"ts":2,"pid":2,"tid":2,"mod":"libnccl.so.2","api":"ncclCommInitRankConfig","type":"other","comm_id":"comm-a","rank":"1","nranks":"2"}\n',
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir)

    assert bundle.communicator_memberships == {"comm-a": (0, 1)}


def test_load_trace_directory_rejects_conflicting_manifest_and_trace_communicators(
    tmp_path: Path,
):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "communicators": {
                    "comm-a": {"members": [0, 3], "size": 2, "name": "dp"},
                }
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        '{"ts":1,"pid":1,"tid":1,"mod":"libnccl.so.2","api":"ncclCommInitRankConfig","type":"other","comm_id":"comm-a","rank":"0","nranks":"2"}\n',
        encoding="utf-8",
    )
    (trace_dir / "rank_2.jsonl").write_text(
        '{"ts":2,"pid":2,"tid":2,"mod":"libnccl.so.2","api":"ncclCommInitRankConfig","type":"other","comm_id":"comm-a","rank":"1","nranks":"2"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="trace-derived communicator topology disagrees"):
        load_trace_directory(trace_dir)


def test_load_trace_directory_inferrs_communicators_from_full_trace_outside_step_window(
    tmp_path: Path,
):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "step_windows": {
                    "0": {"start_ts": 100, "end_ts": 120, "source": "trace_markers"},
                    "2": {"start_ts": 100, "end_ts": 120, "source": "trace_markers"},
                }
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                '{"ts":1,"pid":1,"tid":1,"mod":"libnccl.so.2","api":"ncclCommInitRankConfig","type":"other","comm_id":"comm-a","rank":"0","nranks":"2"}',
                '{"ts":110,"pid":1,"tid":1,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","comm_id":"comm-a","call_idx":"0"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (trace_dir / "rank_2.jsonl").write_text(
        "\n".join(
            [
                '{"ts":2,"pid":2,"tid":2,"mod":"libnccl.so.2","api":"ncclCommInitRankConfig","type":"other","comm_id":"comm-a","rank":"1","nranks":"2"}',
                '{"ts":111,"pid":2,"tid":2,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","comm_id":"comm-a","call_idx":"0"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir)

    assert bundle.trace_window == "step"
    assert bundle.communicator_memberships == {"comm-a": (0, 2)}
    assert [event.api for event in bundle.rank_traces[0].events] == ["ncclAllReduce"]


def test_materialize_profiled_rank_traces_expands_logical_world(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        '{"original_world_size": 4, "profiled_ranks": [0, 2], "profiled_rank_groups": {"0": [0, 1], "2": [2, 3]}}'
    )
    (trace_dir / "rank_0.jsonl").write_text(
        '{"ts":1,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}\n'
    )
    (trace_dir / "rank_2.jsonl").write_text(
        '{"ts":2,"pid":2,"tid":2,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}\n'
    )

    bundle = materialize_profiled_rank_traces(load_trace_directory(trace_dir))

    assert bundle.logical_rank_materialized is True
    assert bundle.world_size == 4
    assert bundle.profiled_world_size == 2
    assert bundle.rank_ids() == (0, 1, 2, 3)
    assert bundle.rank_traces[1].events[0].rank == 1
    assert bundle.rank_traces[3].events[0].rank == 3
    assert bundle.profiled_rank_groups == {0: (0, 1), 2: (2, 3)}
    assert bundle.trace_window == "full"


def test_materialize_profiled_rank_traces_remaps_representative_only_communicators(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "original_world_size": 4,
                "profiled_ranks": [0, 1],
                "profiled_rank_groups": {"0": [0, 2], "1": [1, 3]},
                "communicators": {
                    "pp-representative": {"members": [0, 1], "size": 2, "name": "pp"},
                    "dp-logical": {"members": [0, 2], "size": 2, "name": "dp"},
                },
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                '{"ts":1,"pid":1,"tid":1,"mod":"libnccl.so.2","api":"ncclSend","type":"nccl_collective","comm_id":"pp-representative","call_idx":"0","peer":"1"}',
                '{"ts":2,"pid":1,"tid":1,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","comm_id":"dp-logical","call_idx":"0"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (trace_dir / "rank_1.jsonl").write_text(
        '{"ts":1,"pid":2,"tid":2,"mod":"libnccl.so.2","api":"ncclRecv","type":"nccl_collective","comm_id":"pp-representative","call_idx":"0","peer":"0"}\n',
        encoding="utf-8",
    )

    bundle = materialize_profiled_rank_traces(load_trace_directory(trace_dir))

    remapped_comm_id = build_emulated_communicator_id((2, 3))
    rank2_events = bundle.rank_traces[2].events
    assert rank2_events[0].extras["comm_id"] == remapped_comm_id
    assert rank2_events[1].extras["comm_id"] == "dp-logical"
    assert bundle.communicator_memberships["pp-representative"] == (0, 1)
    assert bundle.communicator_memberships[remapped_comm_id] == (2, 3)
    assert bundle.communicator_memberships["dp-logical"] == (0, 2)


def test_load_trace_directory_recovers_real_trace_communicators_from_local_handles(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps({"original_world_size": 2}),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                '{"ts":1,"pid":10,"tid":10,"mod":"libnccl.so.2","api":"ncclCommInitRankConfig","type":"other","world_size":"2","nranks":"2","rank":"0","comm_id":"94111111111111"}',
                '{"ts":2,"pid":10,"tid":10,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","world_size":"2","comm_id":"94111111111111","call_idx":"0"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (trace_dir / "rank_1.jsonl").write_text(
        "\n".join(
            [
                '{"ts":1,"pid":11,"tid":11,"mod":"libnccl.so.2","api":"ncclCommInitRankConfig","type":"other","world_size":"2","nranks":"2","rank":"1","comm_id":"94222222222222"}',
                '{"ts":2,"pid":11,"tid":11,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","world_size":"2","comm_id":"94222222222222","call_idx":"0"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir)

    assert set(bundle.communicator_memberships.values()) == {(0, 1)}
    rank0_collective = bundle.rank_traces[0].events[1]
    rank1_collective = bundle.rank_traces[1].events[1]
    assert rank0_collective.extras["comm_id"] == rank1_collective.extras["comm_id"]
    assert rank0_collective.extras["local_comm_id"] == "94111111111111"
    assert rank1_collective.extras["local_comm_id"] == "94222222222222"


def test_load_trace_directory_partitions_same_shape_real_trace_subgroups(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps({"original_world_size": 4}),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                '{"ts":1,"pid":10,"tid":10,"mod":"libnccl.so.2","api":"ncclCommInitRankConfig","type":"other","world_size":"4","nranks":"2","rank":"0","comm_id":"94100000000001"}',
                '{"ts":2,"pid":10,"tid":10,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","world_size":"4","comm_id":"94100000000001","call_idx":"0"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (trace_dir / "rank_1.jsonl").write_text(
        "\n".join(
            [
                '{"ts":1,"pid":11,"tid":11,"mod":"libnccl.so.2","api":"ncclCommInitRankConfig","type":"other","world_size":"4","nranks":"2","rank":"1","comm_id":"94100000000002"}',
                '{"ts":2,"pid":11,"tid":11,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","world_size":"4","comm_id":"94100000000002","call_idx":"0"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (trace_dir / "rank_2.jsonl").write_text(
        "\n".join(
            [
                '{"ts":1,"pid":12,"tid":12,"mod":"libnccl.so.2","api":"ncclCommInitRankConfig","type":"other","world_size":"4","nranks":"2","rank":"0","comm_id":"94200000000001"}',
                '{"ts":2,"pid":12,"tid":12,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","world_size":"4","comm_id":"94200000000001","call_idx":"0"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (trace_dir / "rank_3.jsonl").write_text(
        "\n".join(
            [
                '{"ts":1,"pid":13,"tid":13,"mod":"libnccl.so.2","api":"ncclCommInitRankConfig","type":"other","world_size":"4","nranks":"2","rank":"1","comm_id":"94200000000002"}',
                '{"ts":2,"pid":13,"tid":13,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","world_size":"4","comm_id":"94200000000002","call_idx":"0"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir)

    assert set(bundle.communicator_memberships.values()) == {(0, 1), (2, 3)}
    rank0_comm = bundle.rank_traces[0].events[1].extras["comm_id"]
    rank1_comm = bundle.rank_traces[1].events[1].extras["comm_id"]
    rank2_comm = bundle.rank_traces[2].events[1].extras["comm_id"]
    rank3_comm = bundle.rank_traces[3].events[1].extras["comm_id"]
    assert rank0_comm == rank1_comm
    assert rank2_comm == rank3_comm
    assert rank0_comm != rank2_comm


def test_load_trace_directory_recovers_host_local_p2p_real_trace_components(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps({"original_world_size": 4}),
        encoding="utf-8",
    )
    records_by_rank = {
        0: [
            '{"ts":1,"pid":10,"tid":10,"mod":"libnccl.so.2","api":"ncclSend","type":"nccl_collective","host_machine_id":"hostA","comm_id":"941a","peer":"1","count":"8","datatype":"9"}',
        ],
        1: [
            '{"ts":1,"pid":11,"tid":11,"mod":"libnccl.so.2","api":"ncclRecv","type":"nccl_collective","host_machine_id":"hostA","comm_id":"941b","peer":"0","count":"8","datatype":"9"}',
            '{"ts":2,"pid":11,"tid":11,"mod":"libnccl.so.2","api":"ncclSend","type":"nccl_collective","host_machine_id":"hostA","comm_id":"941b","peer":"2","count":"8","datatype":"9"}',
        ],
        2: [
            '{"ts":1,"pid":12,"tid":12,"mod":"libnccl.so.2","api":"ncclRecv","type":"nccl_collective","host_machine_id":"hostA","comm_id":"941c","peer":"1","count":"8","datatype":"9"}',
            '{"ts":2,"pid":12,"tid":12,"mod":"libnccl.so.2","api":"ncclSend","type":"nccl_collective","host_machine_id":"hostA","comm_id":"941c","peer":"3","count":"8","datatype":"9"}',
        ],
        3: [
            '{"ts":1,"pid":13,"tid":13,"mod":"libnccl.so.2","api":"ncclRecv","type":"nccl_collective","host_machine_id":"hostA","comm_id":"941d","peer":"2","count":"8","datatype":"9"}',
        ],
    }
    for rank, lines in records_by_rank.items():
        (trace_dir / f"rank_{rank}.jsonl").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    bundle = load_trace_directory(trace_dir)

    assert set(bundle.communicator_memberships.values()) == {(0, 1, 2, 3)}
    canonical_comm_ids = {
        bundle.rank_traces[rank].events[0].extras["comm_id"]
        for rank in range(4)
    }
    assert len(canonical_comm_ids) == 1


def test_load_trace_directory_recovers_collective_sequence_real_trace_groups(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps({"original_world_size": 4}),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                '{"ts":1,"pid":10,"tid":10,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","comm_id":"941a","count":"64","datatype":"9","op":"0"}',
                '{"ts":2,"pid":10,"tid":10,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","comm_id":"941a","count":"4096","datatype":"9","op":"0"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (trace_dir / "rank_1.jsonl").write_text(
        "\n".join(
            [
                '{"ts":1,"pid":11,"tid":11,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","comm_id":"941b","count":"64","datatype":"9","op":"0"}',
                '{"ts":2,"pid":11,"tid":11,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","comm_id":"941b","count":"4096","datatype":"9","op":"0"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (trace_dir / "rank_2.jsonl").write_text(
        "\n".join(
            [
                '{"ts":1,"pid":12,"tid":12,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","comm_id":"942a","count":"128","datatype":"9","op":"0"}',
                '{"ts":2,"pid":12,"tid":12,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","comm_id":"942a","count":"8192","datatype":"9","op":"0"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (trace_dir / "rank_3.jsonl").write_text(
        "\n".join(
            [
                '{"ts":1,"pid":13,"tid":13,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","comm_id":"942b","count":"128","datatype":"9","op":"0"}',
                '{"ts":2,"pid":13,"tid":13,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","comm_id":"942b","count":"8192","datatype":"9","op":"0"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir)

    assert set(bundle.communicator_memberships.values()) == {(0, 1), (2, 3)}
    rank0_comm = bundle.rank_traces[0].events[0].extras["comm_id"]
    rank1_comm = bundle.rank_traces[1].events[0].extras["comm_id"]
    rank2_comm = bundle.rank_traces[2].events[0].extras["comm_id"]
    rank3_comm = bundle.rank_traces[3].events[0].extras["comm_id"]
    assert rank0_comm == rank1_comm
    assert rank2_comm == rank3_comm
    assert rank0_comm != rank2_comm


def test_materialize_profiled_rank_traces_rejects_incomplete_world(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        '{"original_world_size": 4, "profiled_ranks": [0], "profiled_rank_groups": {"0": [0, 1]}}'
    )
    (trace_dir / "rank_0.jsonl").write_text(
        '{"ts":1,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}\n'
    )

    with pytest.raises(ValueError, match="does not cover"):
        materialize_profiled_rank_traces(load_trace_directory(trace_dir))


def test_dedup_pattern_rank_traces_groups_semantically_equivalent_ranks(tmp_path: Path):
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
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                '{"ts":1,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch","m":"64","n":"64","k":"64"}',
                '{"ts":2,"pid":1,"tid":1,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","count":"8","datatype":"7","op":"0","comm_id":"tp-a","call_idx":"3"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (trace_dir / "rank_1.jsonl").write_text(
        "\n".join(
            [
                '{"ts":4,"pid":2,"tid":2,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch","m":"64","n":"64","k":"64"}',
                '{"ts":5,"pid":2,"tid":2,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","count":"8","datatype":"7","op":"0","comm_id":"tp-b","call_idx":"17"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (trace_dir / "rank_2.jsonl").write_text(
        "\n".join(
            [
                '{"ts":7,"pid":3,"tid":3,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch","m":"64","n":"64","k":"64"}',
                '{"ts":8,"pid":3,"tid":3,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","count":"8","datatype":"7","op":"0","comm_id":"tp-c","call_idx":"9"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    deduped = dedup_pattern_rank_traces(load_trace_directory(trace_dir))

    assert deduped.world_size == 3
    assert deduped.profiled_world_size == 2
    assert deduped.rank_ids() == (0, 2)
    assert deduped.profiled_rank_groups == {0: (0, 1), 2: (2,)}
    assert deduped.trace_window == "full"


def test_dedup_pattern_rank_traces_ignores_setup_and_allocator_noise(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps({"original_world_size": 2}),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                '{"ts":1,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaGetDevice","type":"context_op"}',
                '{"ts":2,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch","m":"64","n":"64","k":"64"}',
                '{"ts":3,"pid":1,"tid":1,"mod":"libcublas.so.12","api":"cublasGemmEx","type":"blas_compute","m":"64","n":"64","k":"64"}',
                '{"ts":4,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaGetLastError","type":"other"}',
                '{"ts":5,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch","m":"32","n":"32","k":"32"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (trace_dir / "rank_1.jsonl").write_text(
        "\n".join(
            [
                '{"ts":1,"pid":2,"tid":2,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch","m":"64","n":"64","k":"64"}',
                '{"ts":2,"pid":2,"tid":2,"mod":"libcublas.so.12","api":"cublasGemmEx","type":"blas_compute","m":"64","n":"64","k":"64"}',
                '{"ts":3,"pid":2,"tid":2,"mod":"libcudart.so.12","api":"cudaMalloc","type":"mem_alloc","bytes":"4096"}',
                '{"ts":4,"pid":2,"tid":2,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch","m":"32","n":"32","k":"32"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    deduped = dedup_pattern_rank_traces(load_trace_directory(trace_dir))

    assert deduped.profiled_world_size == 1
    assert deduped.rank_ids() == (0,)
    assert deduped.profiled_rank_groups == {0: (0, 1)}


def test_dedup_pattern_rank_traces_preserves_upstream_logical_group_members(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "original_world_size": 6,
                "profiled_ranks": [0, 2, 4],
                "profiled_rank_groups": {
                    "0": [0, 1],
                    "2": [2, 3],
                    "4": [4, 5],
                },
                "communicators": {
                    "tp-a": {"members": [0, 2, 4], "size": 3, "name": "tp"},
                    "tp-b": {"members": [1, 3, 5], "size": 3, "name": "tp"},
                },
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                '{"ts":1,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch","m":"64","n":"64","k":"64"}',
                '{"ts":2,"pid":1,"tid":1,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","count":"8","datatype":"7","op":"0","comm_id":"tp-a","call_idx":"3"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (trace_dir / "rank_2.jsonl").write_text(
        "\n".join(
            [
                '{"ts":4,"pid":2,"tid":2,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch","m":"64","n":"64","k":"64"}',
                '{"ts":5,"pid":2,"tid":2,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","count":"8","datatype":"7","op":"0","comm_id":"tp-b","call_idx":"17"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (trace_dir / "rank_4.jsonl").write_text(
        "\n".join(
            [
                '{"ts":7,"pid":3,"tid":3,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch","m":"32","n":"32","k":"32"}',
                '{"ts":8,"pid":3,"tid":3,"mod":"libnccl.so.2","api":"ncclAllReduce","type":"nccl_collective","count":"8","datatype":"7","op":"0","comm_id":"tp-a","call_idx":"9"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    deduped = dedup_pattern_rank_traces(load_trace_directory(trace_dir))

    assert deduped.world_size == 6
    assert deduped.profiled_world_size == 2
    assert deduped.rank_ids() == (0, 4)
    assert deduped.profiled_rank_groups == {0: (0, 1, 2, 3), 4: (4, 5)}


def test_load_trace_directory_defaults_to_manifest_step_window(tmp_path: Path):
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
                '{"ts":10,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaGetDevice","type":"context_op"}',
                '{"ts":20,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
                '{"ts":30,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir)

    assert bundle.trace_window == "step"
    assert [event.ts for event in bundle.rank_traces[0].events] == [20]
    assert bundle.fidelity_windows[0].source == "trace_markers"
    assert bundle.fidelity_windows[0].is_paper_valid_step_window is True


def test_load_trace_directory_can_force_full_window(tmp_path: Path):
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
                '{"ts":10,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaGetDevice","type":"context_op"}',
                '{"ts":20,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
                '{"ts":30,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir, trace_window="full")

    assert bundle.trace_window == "full"
    assert [event.ts for event in bundle.rank_traces[0].events] == [10, 20, 30]


def test_load_trace_directory_can_pad_step_window(tmp_path: Path):
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
                '{"ts":10,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaGetDevice","type":"context_op"}',
                '{"ts":20,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
                '{"ts":30,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir, step_pre_padding_us=5)

    assert bundle.trace_window == "step"
    assert [event.ts for event in bundle.rank_traces[0].events] == [10, 20]


def test_load_trace_directory_falls_back_to_trace_markers_when_manifest_window_missing(
    tmp_path: Path,
):
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
                '{"ts":10,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaGetDevice","type":"context_op"}',
                '{"ts":20,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
                '{"ts":30,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (trace_dir / "rank_1.jsonl").write_text(
        "\n".join(
            [
                '{"ts":90,"pid":2,"tid":2,"mod":"libcudart.so.12","api":"cudaMemcpyAsync","type":"mem_copy"}',
                '{"ts":120,"pid":2,"tid":2,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
                '{"ts":180,"pid":2,"tid":2,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
            ]
        )
        + "\n",
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
                        "monotonic_ns": 100_000,
                        "realtime_ns": 100_000,
                        "step": 1,
                    }
                ),
                json.dumps(
                    {
                        "kind": "step_end",
                        "label": "training_step",
                        "pid": 2,
                        "monotonic_ns": 150_000,
                        "realtime_ns": 150_000,
                        "step": 1,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir)

    assert bundle.trace_window == "step"
    assert [event.ts for event in bundle.rank_traces[0].events] == [20]
    assert [event.ts for event in bundle.rank_traces[1].events] == [120]
    assert bundle.fidelity_windows[1].source == "trace_markers"
    assert bundle.fidelity_windows[1].is_paper_valid_step_window is True


def test_load_trace_directory_uses_but_does_not_materialize_trace_markers(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 10,
                        "pid": 1,
                        "tid": 1,
                        "mod": "libcudart.so.12",
                        "api": "mayaStepMarker",
                        "type": "marker",
                        "kind": "step_begin",
                        "label": "training_step",
                    }
                ),
                '{"ts":20,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
                json.dumps(
                    {
                        "ts": 30,
                        "pid": 1,
                        "tid": 1,
                        "mod": "libcudart.so.12",
                        "api": "mayaStepMarker",
                        "type": "marker",
                        "kind": "step_end",
                        "label": "training_step",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir)

    assert bundle.trace_window == "step"
    assert [event.api for event in bundle.rank_traces[0].events] == ["cudaLaunchKernel"]
    assert bundle.fidelity_windows[0].source == "trace_markers"


def test_load_trace_directory_auto_keeps_full_when_only_boundary_fallback_exists(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                '{"ts":10,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaGetDevice","type":"context_op"}',
                '{"ts":20,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
                '{"ts":30,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir, trace_window="auto")

    assert bundle.trace_window == "full"
    assert [event.ts for event in bundle.rank_traces[0].events] == [10, 20, 30]
    assert bundle.fidelity_windows[0].source == "boundary_fallback"
    assert bundle.fidelity_windows[0].is_paper_valid_step_window is False
    assert bundle.step_windows == {}


def test_load_trace_directory_trace_window_uses_non_paper_manifest_fidelity_window(
    tmp_path: Path,
):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "fidelity_windows": {
                    "0": {
                        "start_ts": 15,
                        "end_ts": 25,
                        "source": "boundary_fallback",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                '{"ts":10,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaGetDevice","type":"context_op"}',
                '{"ts":20,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
                '{"ts":30,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (trace_dir / "rank_0.markers.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "kind": "step_begin",
                        "label": "training_step",
                        "pid": 1,
                        "monotonic_ns": 5_000,
                        "realtime_ns": 5_000,
                        "step": 1,
                    }
                ),
                json.dumps(
                    {
                        "kind": "step_end",
                        "label": "training_step",
                        "pid": 1,
                        "monotonic_ns": 50_000,
                        "realtime_ns": 50_000,
                        "step": 1,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir, trace_window="trace")

    assert bundle.trace_window == "trace"
    assert [event.ts for event in bundle.rank_traces[0].events] == [20]
    assert bundle.step_windows == {}
    assert bundle.fidelity_windows[0].source == "boundary_fallback"
    assert bundle.fidelity_windows[0].is_paper_valid_step_window is False


def test_load_trace_directory_trace_window_uses_boundary_fallback_without_step_windows(
    tmp_path: Path,
):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                '{"ts":10,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaGetDevice","type":"context_op"}',
                '{"ts":20,"end_ts":40,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaMemcpy","type":"mem_copy"}',
                '{"ts":60,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaGetDevice","type":"context_op"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir, trace_window="trace")

    assert bundle.trace_window == "trace"
    assert [event.api for event in bundle.rank_traces[0].events] == ["cudaMemcpy"]
    assert bundle.step_windows == {}
    assert bundle.fidelity_windows[0].source == "boundary_fallback"
    assert bundle.fidelity_windows[0].is_paper_valid_step_window is False


def test_load_trace_directory_step_rejects_boundary_fallback_only(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "rank_0.jsonl").write_text(
        '{"ts":20,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="paper-valid step windows"):
        load_trace_directory(trace_dir, trace_window="step")


def test_load_trace_directory_treats_boundary_fallback_as_non_paper_even_if_flagged(
    tmp_path: Path,
):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "fidelity_windows": {
                    "0": {
                        "start_ts": 15,
                        "end_ts": 25,
                        "source": "boundary_fallback",
                        "is_paper_valid_step_window": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                '{"ts":10,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaGetDevice","type":"context_op"}',
                '{"ts":20,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
                '{"ts":30,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir, trace_window="auto")

    assert bundle.trace_window == "full"
    assert bundle.step_windows == {}
    assert bundle.fidelity_windows[0].source == "boundary_fallback"
    assert bundle.fidelity_windows[0].is_paper_valid_step_window is False

    with pytest.raises(ValueError, match="paper-valid step windows"):
        load_trace_directory(trace_dir, trace_window="step")


def test_load_trace_directory_prefers_trace_markers_over_manifest_boundary_fallback(
    tmp_path: Path,
):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "fidelity_windows": {
                    "0": {
                        "start_ts": 5,
                        "end_ts": 50,
                        "source": "boundary_fallback",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                '{"ts":10,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaGetDevice","type":"context_op"}',
                '{"ts":20,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
                '{"ts":30,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (trace_dir / "rank_0.markers.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "kind": "step_begin",
                        "label": "training_step",
                        "pid": 1,
                        "monotonic_ns": 15_000,
                        "realtime_ns": 15_000,
                        "step": 1,
                    }
                ),
                json.dumps(
                    {
                        "kind": "step_end",
                        "label": "training_step",
                        "pid": 1,
                        "monotonic_ns": 25_000,
                        "realtime_ns": 25_000,
                        "step": 1,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir, trace_window="auto")

    assert bundle.trace_window == "step"
    assert [event.ts for event in bundle.rank_traces[0].events] == [20]
    assert bundle.fidelity_windows[0].source == "trace_markers"


def test_load_trace_directory_canonicalizes_legacy_marker_sidecar_source(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "fidelity_windows": {
                    "0": {
                        "start_ts": 15,
                        "end_ts": 25,
                        "source": "marker_sidecar",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                '{"ts":10,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaGetDevice","type":"context_op"}',
                '{"ts":20,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
                '{"ts":30,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir, trace_window="auto")

    assert bundle.trace_window == "step"
    assert [event.ts for event in bundle.rank_traces[0].events] == [20]
    assert bundle.fidelity_windows[0].source == "trace_markers"
    assert bundle.fidelity_windows[0].is_paper_valid_step_window is True


def test_load_trace_directory_legacy_helper_tail_source_stays_non_paper_valid(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "fidelity_windows": {
                    "0": {
                        "start_ts": 15,
                        "end_ts": 35,
                        "source": "markers_helper_tail_extended",
                        "is_paper_valid_step_window": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                '{"ts":10,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaGetDevice","type":"context_op"}',
                '{"ts":20,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
                '{"ts":30,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir, trace_window="auto")

    assert bundle.trace_window == "full"
    assert bundle.fidelity_windows[0].source == "trace_markers"
    assert bundle.fidelity_windows[0].is_paper_valid_step_window is False
    assert bundle.fidelity_windows[0].extras["diagnostic_extension"] == "helper_tail"

    with pytest.raises(ValueError, match="paper-valid step windows"):
        load_trace_directory(trace_dir, trace_window="step")


def test_load_trace_directory_prefers_explicit_step_window_over_diagnostic_fidelity_window(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "step_windows": {
                    "0": {
                        "start_ts": 15,
                        "end_ts": 25,
                        "source": "trace_markers",
                        "step_count": 1,
                    }
                },
                "fidelity_windows": {
                    "0": {
                        "start_ts": 15,
                        "end_ts": 35,
                        "source": "trace_markers",
                        "is_paper_valid_step_window": False,
                        "diagnostic_extension": "helper_tail",
                        "diagnostic_only": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                '{"ts":10,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaGetDevice","type":"context_op"}',
                '{"ts":20,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
                '{"ts":30,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir, trace_window="step")

    assert bundle.trace_window == "step"
    assert [event.ts for event in bundle.rank_traces[0].events] == [20]
    assert bundle.fidelity_windows[0].source == "trace_markers"
    assert bundle.fidelity_windows[0].is_paper_valid_step_window is True
    assert bundle.fidelity_windows[0].extras["diagnostic_fidelity_window"] == {
        "start_ts": 15,
        "end_ts": 35,
        "source": "trace_markers",
        "diagnostic_extension": "helper_tail",
        "diagnostic_only": True,
    }


def test_load_trace_directory_trace_window_prefers_diagnostic_fidelity_window_over_step_window(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "step_windows": {
                    "0": {
                        "start_ts": 15,
                        "end_ts": 25,
                        "source": "trace_markers",
                        "step_count": 1,
                    }
                },
                "fidelity_windows": {
                    "0": {
                        "start_ts": 15,
                        "end_ts": 35,
                        "source": "trace_markers",
                        "is_paper_valid_step_window": False,
                        "diagnostic_extension": "helper_tail",
                        "diagnostic_only": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                '{"ts":10,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaGetDevice","type":"context_op"}',
                '{"ts":20,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
                '{"ts":30,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
                '{"ts":40,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir, trace_window="trace")

    assert bundle.trace_window == "trace"
    assert [event.ts for event in bundle.rank_traces[0].events] == [20, 30]
    assert bundle.step_windows == {}
    assert bundle.fidelity_windows[0].is_paper_valid_step_window is False
    assert bundle.fidelity_windows[0].extras["diagnostic_extension"] == "helper_tail"


def test_load_trace_directory_ignores_incomplete_trace_marker_files(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "rank_0.jsonl").write_text(
        '{"ts":20,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}\n',
        encoding="utf-8",
    )
    (trace_dir / "rank_0.markers.jsonl").write_text(
        '{"kind":"step_begin","step":1}\n',
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir)

    assert bundle.rank_ids() == (0,)
    assert [event.api for event in bundle.rank_traces[0].events] == ["cudaLaunchKernel"]


def test_estimate_rank_trace_active_seconds_ignores_setup_and_teardown(tmp_path: Path):
    trace_file = tmp_path / "rank_0.jsonl"
    trace_file.write_text(
        "\n".join(
            [
                '{"ts":10,"pid":1,"tid":1,"mod":"libnccl.so","api":"ncclGetUniqueId","type":"other"}',
                '{"ts":20,"pid":1,"tid":1,"mod":"libnccl.so","api":"ncclCommInitRankConfig","type":"other"}',
                '{"ts":1000010,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
                '{"ts":3000010,"pid":1,"tid":1,"mod":"libnccl.so","api":"ncclAllReduce","type":"nccl_collective"}',
                '{"ts":5000010,"pid":1,"tid":1,"mod":"libnccl.so","api":"ncclCommDestroy","type":"other"}',
            ]
        ),
        encoding="utf-8",
    )

    active_seconds = estimate_rank_trace_active_seconds(trace_file)

    assert active_seconds == pytest.approx(2.0)


def test_estimate_rank_trace_active_seconds_uses_end_ts_when_present(tmp_path: Path):
    trace_file = tmp_path / "rank_0.jsonl"
    trace_file.write_text(
        "\n".join(
            [
                '{"ts":10,"pid":1,"tid":1,"mod":"libnccl.so","api":"ncclGetUniqueId","type":"other"}',
                '{"ts":1000010,"end_ts":4000010,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaMemcpy","type":"mem_copy"}',
                '{"ts":5000010,"pid":1,"tid":1,"mod":"libnccl.so","api":"ncclCommDestroy","type":"other"}',
            ]
        ),
        encoding="utf-8",
    )

    active_seconds = estimate_rank_trace_active_seconds(trace_file)

    assert active_seconds == pytest.approx(3.0)


def test_load_trace_directory_step_window_keeps_event_overlapping_window_via_end_ts(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps({"step_windows": {"0": {"start_ts": 100, "end_ts": 200, "source": "trace_markers"}}}),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            [
                '{"ts":90,"end_ts":120,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaMemcpy","type":"mem_copy"}',
                '{"ts":250,"pid":1,"tid":1,"mod":"libcudart.so.12","api":"cudaLaunchKernel","type":"kernel_launch"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = load_trace_directory(trace_dir)

    assert bundle.trace_window == "step"
    assert [event.api for event in bundle.rank_traces[0].events] == ["cudaMemcpy"]

def test_trace_event_canonicalizes_event_record_with_flags():
    event = TraceEvent.from_json_record(
        {
            "ts": 1,
            "pid": 2,
            "tid": 3,
            "mod": "libcudart.so.12",
            "api": "cudaEventRecordWithFlags",
            "type": "stream_op",
        },
        rank=0,
        ordinal=0,
        source=TraceSource.REAL,
    )

    assert event.api == "cudaEventRecord"
    assert event.op_type == "stream_op"


def test_load_trace_directory_skips_raw_communicator_recovery_when_manifest_is_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(
        json.dumps(
            {
                "communicators": {
                    "comm-0": [0, 1],
                },
                "communicator_aliases": {
                    "0": {"1001": "comm-0"},
                    "1": {"1002": "comm-0"},
                },
                "rank_host_machines": {"0": "host0", "1": "host0"},
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "rank_0.jsonl").write_text(
        '{"ts":10,"pid":1,"tid":1,"mod":"libnccl.so","api":"ncclAllReduce","type":"nccl_collective","comm_id":"1001"}\n',
        encoding="utf-8",
    )
    (trace_dir / "rank_1.jsonl").write_text(
        '{"ts":11,"pid":2,"tid":2,"mod":"libnccl.so","api":"ncclAllReduce","type":"nccl_collective","comm_id":"1002"}\n',
        encoding="utf-8",
    )

    def _unexpected_recovery(*args, **kwargs):
        raise AssertionError("raw communicator recovery should not be used when manifest is complete")

    monkeypatch.setattr("flexsim.maya_lite.io._trace_communicator_recovery", _unexpected_recovery)

    bundle = load_trace_directory(trace_dir)

    assert bundle.communicator_memberships == {"comm-0": (0, 1)}
    assert bundle.rank_traces[0].events[0].extras["comm_id"] == "comm-0"
    assert bundle.rank_traces[0].events[0].extras["local_comm_id"] == "1001"
    assert bundle.rank_traces[1].events[0].extras["comm_id"] == "comm-0"
    assert bundle.rank_traces[1].events[0].extras["local_comm_id"] == "1002"
