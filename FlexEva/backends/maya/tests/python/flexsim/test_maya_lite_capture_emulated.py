from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flexsim.maya_lite.augment_emulated_helper_threads import (
    record_helper_thread_augmentation_status,
)
from flexsim.maya_lite.capture_emulated import (
    DEFAULT_DYNAMIC_DEDUP_POLL_INTERVAL_MS,
    _cached_communicator_memberships,
    _cached_prefix_pattern_tokens,
    _apply_capture_bootstrap_env_defaults,
    _capture_elapsed_metadata,
    _derive_fakecuda_runtime_env,
    _dynamic_dedup_sequence_hash,
    _launch_blocking_worker_count,
    _maybe_classify_first_step_pattern,
    _maybe_escalate_dynamic_dedup_termination,
    _build_parser,
    _capture_worker_command_and_env,
    _fakecuda_artifact_fingerprint_metadata,
    _finalize_helper_thread_augmentation_contract,
    _request_dynamic_dedup_termination,
    _parse_profiled_rank_groups,
    _prefix_pattern_tokens,
    _resolve_trace_flush_policy,
    _resolve_host_timing_dispatch_scope,
    _resolve_host_timing_schedule_surface,
    _resolve_capture_step_window,
    _should_apply_default_workload_heuristic_step_window,
    _resolve_workload_heuristic_step_window,
    _step_window_pattern_tokens,
    _summarize_worker_timing_diagnostics,
    _summarize_host_timing_policy,
    _summarize_profiled_capture,
    _trim_trace_file_to_ts_window,
    _worker_step_timing_diagnostics,
    _worker_cpu_affinity_spec,
    _worker_cpu_affinity_manifest_metadata,
    _available_cpu_ids,
    _parse_cpu_set,
)
from flexsim.maya_lite.emulated_dist import (
    EmulatedDistributedEnvironment,
    EmulatedWork,
    EmulatedProcessGroup,
    _copy_tensor_contents,
)
from flexsim.maya_lite.markers import (
    TRACE_MARKER_API,
    resolve_indexed_step_window_from_marker_trace_timestamps,
    resolve_indexed_step_window_from_markers,
    resolve_indexed_step_window_from_trace_markers,
    resolve_step_window_from_marker_trace_timestamps,
    resolve_step_window_from_markers,
)
from flexsim.maya_lite.schema import TraceSource


def test_parse_profiled_rank_groups_round_trips_mapping() -> None:
    assert _parse_profiled_rank_groups("0:0,2;1:1,3") == {
        0: (0, 2),
        1: (1, 3),
    }


def test_capture_emulated_parser_defaults_to_buffered_trace_flush() -> None:
    args = _build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--logical-world-size",
            "4",
            "--profiled-rank-groups",
            "0:0,1",
            "tests/workloads/fake_cuda/maya_fig13_megatron.py",
        ]
    )

    assert args.trace_flush_mode == "buffered"
    assert args.trace_flush_every == 4096
    assert args.trace_stdio_buffer_bytes == 4 * 1024 * 1024
    assert args.max_concurrent_workers == 1
    assert args.dynamic_dedup_poll_interval_ms == DEFAULT_DYNAMIC_DEDUP_POLL_INTERVAL_MS
    assert args.host_timing_mode == "measure"


def test_capture_emulated_parser_accepts_identity_strategy() -> None:
    args = _build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--logical-world-size",
            "4",
            "--auto-profiled-strategy",
            "identity",
            "tests/workloads/fake_cuda/maya_fig13_megatron.py",
        ]
    )

    assert args.auto_profiled_strategy == "identity"


def test_capture_emulated_parser_accepts_host_timing_controls() -> None:
    args = _build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--logical-world-size",
            "4",
            "--auto-profiled-strategy",
            "identity",
            "--host-timing-mode",
            "trace",
            "--host-timing-profile",
            "/tmp/host_timing.profile",
            "--host-timing-default-us",
            "0.25",
            "tests/workloads/fake_cuda/maya_fig13_megatron.py",
        ]
    )

    assert args.host_timing_mode == "trace"
    assert str(args.host_timing_profile) == "/tmp/host_timing.profile"
    assert args.host_timing_default_us == 0.25


def test_capture_emulated_parser_accepts_host_timing_summary_dir() -> None:
    args = _build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--logical-world-size",
            "4",
            "--auto-profiled-strategy",
            "identity",
            "--host-timing-mode",
            "trace",
            "--host-timing-summary-dir",
            "/tmp/host_summaries",
            "tests/workloads/fake_cuda/maya_fig13_megatron.py",
        ]
    )

    assert str(args.host_timing_summary_dir) == "/tmp/host_summaries"


def test_capture_emulated_parser_accepts_measure_host_timing_mode() -> None:
    args = _build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--logical-world-size",
            "4",
            "--auto-profiled-strategy",
            "identity",
            "--host-timing-mode",
            "measure",
            "tests/workloads/fake_cuda/maya_fig13_megatron.py",
        ]
    )

    assert args.host_timing_mode == "measure"


def test_capture_emulated_parser_accepts_explicit_capture_step_window_selection() -> None:
    args = _build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--logical-world-size",
            "4",
            "--auto-profiled-strategy",
            "identity",
            "--capture-step-window-occurrence",
            "2",
            "--capture-step-window-step",
            "7",
            "tests/workloads/fake_cuda/maya_fig13_megatron.py",
        ]
    )

    assert args.capture_step_window_occurrence == 2
    assert args.capture_step_window_step == 7


def test_capture_emulated_resolves_dispatch_scope_by_mode() -> None:
    assert _resolve_host_timing_dispatch_scope(
        host_timing_mode="trace",
        requested_dispatch_scope=None,
    ) == "host_machine"
    assert _resolve_host_timing_dispatch_scope(
        host_timing_mode="measure",
        requested_dispatch_scope=None,
    ) == "host_machine"
    assert _resolve_host_timing_dispatch_scope(
        host_timing_mode="sleep",
        requested_dispatch_scope=None,
    ) == "host_machine"
    assert _resolve_host_timing_dispatch_scope(
        host_timing_mode="measure",
        requested_dispatch_scope="thread",
    ) == "thread"


def test_capture_emulated_resolves_schedule_surface_to_runtime_default() -> None:
    assert _resolve_host_timing_schedule_surface(
        host_timing_mode="trace",
        requested_schedule_surface=None,
    ) == "semantic"
    assert _resolve_host_timing_schedule_surface(
        host_timing_mode="measure",
        requested_schedule_surface=None,
    ) == "semantic"
    assert _resolve_host_timing_schedule_surface(
        host_timing_mode="measure",
        requested_schedule_surface="semantic",
    ) == "semantic"


def test_capture_emulated_summarizes_paper_host_timing_policy() -> None:
    summary = _summarize_host_timing_policy(
        host_timing_mode="measure",
        requested_dispatch_scope=None,
        resolved_dispatch_scope="host_machine",
        requested_schedule_surface=None,
        resolved_schedule_surface="semantic",
        host_timing_profile=None,
        host_timing_profile_dir=Path("/tmp/host_profiles"),
    )

    assert summary["host_timing_mode_resolved"] == "measure"
    assert summary["host_timing_dispatch_scope_resolved"] == "host_machine"
    assert summary["host_timing_schedule_surface_resolved"] == "semantic"
    assert summary["host_timing_profile_requested"] is True
    assert summary["host_timing_profile_backed"] is False
    assert summary["host_timing_synthetic_shaping"] is False
    assert summary["host_timing_line_contract_version"] == "phase4_v1"
    assert summary["host_timing_line_family"] == "direct_wallclock"
    assert summary["host_timing_dispatch_scope_defaulted"] is True
    assert summary["host_timing_schedule_surface_defaulted"] is True
    assert (
        summary["host_timing_paper_alignment_line"]
        == "direct_emulation_measured_host_overhead"
    )
    assert summary["host_timing_paper_alignment_ready"] is True


def test_capture_elapsed_metadata_includes_post_worker_finalize_in_capture_elapsed() -> None:
    payload = _capture_elapsed_metadata(
        capture_elapsed_seconds=181.75,
        capture_command_elapsed_seconds=181.75,
        worker_capture_elapsed_seconds=126.5,
        post_worker_finalize_seconds=55.25,
    )

    assert payload["capture_elapsed_seconds"] == pytest.approx(181.75)
    assert payload["capture_command_elapsed_seconds"] == pytest.approx(181.75)
    assert payload["worker_capture_elapsed_seconds"] == pytest.approx(126.5)
    assert payload["post_worker_finalize_seconds"] == pytest.approx(55.25)
    assert payload["post_worker_finalize_included_in_capture_elapsed"] is True


def test_worker_step_timing_diagnostics_split_bootstrap_step_and_overhang() -> None:
    diagnostics = _worker_step_timing_diagnostics(
        marker_records=[
            {"kind": "step_begin", "label": "training_step", "realtime_ns": 2_000_000_000},
            {"kind": "step_end", "label": "training_step", "realtime_ns": 5_000_000_000},
        ],
        worker_start_realtime_ns=1_000_000_000,
        worker_end_realtime_ns=7_000_000_000,
        worker_elapsed_seconds=6.0,
        active_trace_seconds=3.5,
    )

    assert diagnostics is not None
    assert diagnostics["bootstrap_before_step_seconds"] == pytest.approx(1.0)
    assert diagnostics["marker_step_seconds"] == pytest.approx(3.0)
    assert diagnostics["post_step_overhang_seconds"] == pytest.approx(2.0)
    assert diagnostics["active_trace_minus_marker_step_seconds"] == pytest.approx(0.5)
    assert diagnostics["active_trace_minus_marker_step_seconds_signed"] == pytest.approx(0.5)
    assert diagnostics["marker_step_minus_active_trace_seconds"] == pytest.approx(0.0)
    assert diagnostics["worker_unaccounted_seconds"] == pytest.approx(0.0)


def test_worker_step_timing_diagnostics_preserves_negative_trace_marker_delta() -> None:
    diagnostics = _worker_step_timing_diagnostics(
        marker_records=[
            {"kind": "step_begin", "label": "training_step", "realtime_ns": 2_000_000_000},
            {"kind": "step_end", "label": "training_step", "realtime_ns": 5_000_000_000},
        ],
        worker_start_realtime_ns=1_000_000_000,
        worker_end_realtime_ns=7_000_000_000,
        worker_elapsed_seconds=6.0,
        active_trace_seconds=2.25,
    )

    assert diagnostics is not None
    assert diagnostics["marker_step_seconds"] == pytest.approx(3.0)
    assert diagnostics["active_trace_minus_marker_step_seconds"] == pytest.approx(0.0)
    assert diagnostics["active_trace_minus_marker_step_seconds_signed"] == pytest.approx(-0.75)
    assert diagnostics["marker_step_minus_active_trace_seconds"] == pytest.approx(0.75)


def test_summarize_worker_timing_diagnostics_combines_unique_and_duplicate_workers() -> None:
    summary = _summarize_worker_timing_diagnostics(
        [
            {
                "representative_rank": 0,
                "elapsed_seconds": 10.0,
                "step_timing_diagnostics": {
                    "worker_elapsed_seconds": 10.0,
                    "bootstrap_before_step_seconds": 2.0,
                    "marker_step_seconds": 4.0,
                    "post_step_overhang_seconds": 4.0,
                    "active_trace_seconds": 4.5,
                    "active_trace_minus_marker_step_seconds": 0.5,
                    "active_trace_minus_marker_step_seconds_signed": 0.5,
                    "marker_step_minus_active_trace_seconds": 0.0,
                },
            },
            {
                "representative_rank": 1,
                "elapsed_seconds": 8.0,
                "dynamic_duplicate_of": 0,
                "first_step_classified_elapsed_seconds": 5.0,
                "termination_requested_elapsed_seconds": 5.5,
            },
        ]
    )

    assert summary is not None
    assert summary["unique_worker_count"] == 1
    assert summary["duplicate_worker_count"] == 1
    assert summary["unique_bootstrap_before_step_seconds_mean"] == pytest.approx(2.0)
    assert summary["unique_marker_step_seconds_mean"] == pytest.approx(4.0)
    assert summary["unique_active_trace_minus_marker_step_seconds_signed_mean"] == pytest.approx(0.5)
    assert summary["unique_marker_step_minus_active_trace_seconds_mean"] == pytest.approx(0.0)
    assert summary["duplicate_elapsed_seconds_mean"] == pytest.approx(8.0)
    assert summary["duplicate_first_step_classified_elapsed_seconds_mean"] == pytest.approx(5.0)
    assert summary["duplicate_termination_requested_elapsed_seconds_mean"] == pytest.approx(5.5)


def test_finalize_helper_thread_augmentation_contract_marks_not_required(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(json.dumps({}), encoding="utf-8")

    payload = _finalize_helper_thread_augmentation_contract(
        output_dir=trace_dir,
        args=SimpleNamespace(
            host_timing_mode=None,
            host_timing_summary_dir=None,
            host_timing_profile_dir=None,
            host_timing_profile=None,
        ),
    )

    assert payload["expected"] is False
    assert payload["status"] == "not_required"
    assert payload["embedded_in_emulator_artifact"] is True
    manifest = json.loads((trace_dir / "capture_manifest.json").read_text(encoding="utf-8"))
    assert manifest["helper_thread_augmentation"] == payload


def test_finalize_helper_thread_augmentation_contract_marks_missing_summary_dir_for_trace_mode(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "capture_manifest.json").write_text(json.dumps({}), encoding="utf-8")

    payload = _finalize_helper_thread_augmentation_contract(
        output_dir=trace_dir,
        args=SimpleNamespace(
            host_timing_mode="trace",
            host_timing_summary_dir=None,
            host_timing_profile_dir=None,
            host_timing_profile=None,
        ),
    )

    assert payload["expected"] is True
    assert payload["status"] == "missing_summary_dir"
    assert payload["embedded_in_emulator_artifact"] is False
    manifest = json.loads((trace_dir / "capture_manifest.json").read_text(encoding="utf-8"))
    assert manifest["helper_thread_augmentation"] == payload


def test_finalize_helper_thread_augmentation_contract_uses_summary_dir_without_host_timing_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    summary_dir = tmp_path / "summaries"
    summary_dir.mkdir()

    manifest = {
        "step_windows": {
            "0": {
                "start_ts": 1_000,
                "end_ts": 2_000,
                "source": "trace_markers",
                "step_count": 1,
            }
        },
        "fidelity_windows": {
            "0": {
                "start_ts": 1_000,
                "end_ts": 2_000,
                "source": "trace_markers",
                "step_count": 1,
            }
        },
    }
    (trace_dir / "capture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (trace_dir / "rank_0.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "ts": 2_100 + (offset * 3_000),
                    "end_ts": 2_150 + (offset * 3_000),
                    "pid": 1,
                    "tid": 1,
                    "mod": "libcublas.so.12",
                    "api": "cublasGemmEx",
                    "type": "blas_compute",
                }
            )
            for offset in range(40)
        )
        + "\n",
        encoding="utf-8",
    )

    def _fake_augment(output_dir: Path, *, summary_dir: Path):
        return record_helper_thread_augmentation_status(
            output_dir,
            expected=True,
            status="completed",
            summary_dir=summary_dir,
        )

    monkeypatch.setattr(
        "flexsim.maya_lite.capture_emulated.augment_trace_directory",
        _fake_augment,
    )

    payload = _finalize_helper_thread_augmentation_contract(
        output_dir=trace_dir,
        args=SimpleNamespace(
            host_timing_mode=None,
            host_timing_summary_dir=summary_dir,
            host_timing_profile_dir=None,
            host_timing_profile=None,
        ),
    )

    assert payload["expected"] is True
    assert payload["status"] == "completed"
    manifest = json.loads((trace_dir / "capture_manifest.json").read_text(encoding="utf-8"))
    assert manifest["step_windows"]["0"]["source"] == "workload_heuristic"
    assert manifest["step_windows"]["0"]["end_ts"] > 100_000
    assert manifest["step_windows"]["0"]["helper_thread_augmented"] is True


def test_capture_emulated_measure_mode_ignores_profile_env_in_worker_command(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    args = _build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--logical-world-size",
            "4",
            "--auto-profiled-strategy",
            "identity",
            "--host-timing-mode",
            "measure",
            "--host-timing-profile",
            str(tmp_path / "host.profile"),
            "--host-timing-default-us",
            "17.5",
            "tests/workloads/fake_cuda/maya_fig13_megatron.py",
        ]
    )

    _, env, *_ = _capture_worker_command_and_env(
        args=args,
        profiled_index=0,
        representative_rank=0,
        rank_host_machines={},
        rank_host_dispatch_queues={},
        script_args=[],
        repo_root=str(repo_root),
        python_root=str(repo_root / "python"),
        output_dir=tmp_path / "out",
    )

    assert env["FAKECUDA_HOST_TIMING_MODE"] == "measure"
    assert env["FAKECUDA_HOST_TIMING_DISPATCH_SCOPE"] == "host_machine"
    assert env["FAKECUDA_HOST_TIMING_SCHEDULE_SURFACE"] == "semantic"
    assert env["FAKECUDA_TRACE_SURFACE"] == "all"
    assert "FAKECUDA_HOST_TIMING_PROFILE" not in env
    assert "FAKECUDA_HOST_TIMING_DEFAULT_US" not in env


def test_apply_capture_bootstrap_env_defaults_preserves_existing_values() -> None:
    env = {"OMP_NUM_THREADS": "8"}

    _apply_capture_bootstrap_env_defaults(env)

    assert env["OMP_NUM_THREADS"] == "8"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"


def test_worker_cpu_affinity_spec_uses_allowed_cpu_ids() -> None:
    assert _worker_cpu_affinity_spec(
        affinity_slot=1,
        cores_per_worker=2,
        available_cpu_ids=[4, 6, 8, 10],
    ) == "8,10"


def test_parse_cpu_set_accepts_online_cpu_ranges() -> None:
    assert _parse_cpu_set("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]


def test_available_cpu_ids_intersects_affinity_with_online_cpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(128)), raising=False)
    monkeypatch.setattr(
        "flexsim.maya_lite.capture_emulated._online_cpu_ids",
        lambda: list(range(109)),
    )

    assert _available_cpu_ids() == list(range(109))


def test_worker_cpu_affinity_spec_compacts_contiguous_allowed_cpu_ids() -> None:
    assert _worker_cpu_affinity_spec(
        affinity_slot=1,
        cores_per_worker=3,
        available_cpu_ids=[2, 3, 4, 8, 9, 10],
    ) == "8-10"


def test_worker_cpu_affinity_manifest_metadata_reports_disabled_by_default() -> None:
    metadata = _worker_cpu_affinity_manifest_metadata(
        SimpleNamespace(worker_cpu_affinity_cores_per_worker=0)
    )

    assert metadata == {
        "worker_cpu_affinity_cores_per_worker": None,
        "worker_cpu_affinity_available_cpu_count": None,
        "worker_cpu_affinity_topology_basis": None,
    }


def test_worker_cpu_affinity_manifest_metadata_reports_optin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flexsim.maya_lite.capture_emulated._available_cpu_ids",
        lambda: [2, 3, 4, 8],
    )

    metadata = _worker_cpu_affinity_manifest_metadata(
        SimpleNamespace(worker_cpu_affinity_cores_per_worker=2)
    )

    assert metadata == {
        "worker_cpu_affinity_cores_per_worker": 2,
        "worker_cpu_affinity_available_cpu_count": 4,
        "worker_cpu_affinity_topology_basis": "available_logical_cpu",
    }


def test_worker_cpu_affinity_spec_rejects_oversubscribed_slot() -> None:
    with pytest.raises(RuntimeError, match="available_cpu_count=2"):
        _worker_cpu_affinity_spec(
            affinity_slot=1,
            cores_per_worker=2,
            available_cpu_ids=[4, 6],
        )


def test_capture_emulated_worker_command_sets_bootstrap_diag_path_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    args = _build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--logical-world-size",
            "4",
            "--auto-profiled-strategy",
            "identity",
            "tests/workloads/fake_cuda/maya_fig13_megatron.py",
        ]
    )
    diag_dir = tmp_path / "bootstrap-diag"
    monkeypatch.setenv("FLEXSIM_MAYA_BOOTSTRAP_DIAG_DIR", str(diag_dir))

    _, env, *_ = _capture_worker_command_and_env(
        args=args,
        profiled_index=0,
        representative_rank=3,
        rank_host_machines={},
        rank_host_dispatch_queues={},
        script_args=[],
        repo_root=str(repo_root),
        python_root=str(repo_root / "python"),
        output_dir=tmp_path / "out",
    )

    assert env["FLEXSIM_MAYA_BOOTSTRAP_DIAG_PATH"] == str(
        diag_dir / "rank_3.bootstrap.json"
    )
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"


def test_capture_worker_command_uses_direct_proot_prefix_when_layout_is_available(
    tmp_path: Path,
) -> None:
    fakecuda_root = tmp_path / "fake-cuda"
    build_lib_dir = fakecuda_root / "build" / "liboutput"
    build_lib_dir.mkdir(parents=True)
    for library_name in (
        "libcudart.so.12",
        "libcuda.so.1",
        "libnvidia-ml.so",
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libnccl.so.2",
    ):
        (build_lib_dir / library_name).write_text("", encoding="utf-8")
    (fakecuda_root / "frun").write_text("", encoding="utf-8")
    (fakecuda_root / "proot").write_text("", encoding="utf-8")

    env_root = tmp_path / "env"
    python_bin = env_root / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("", encoding="utf-8")
    site_packages = (
        env_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    for relative in (
        ("nvidia", "cuda_runtime", "lib", "libcudart.so.12"),
        ("nvidia", "cublas", "lib", "libcublas.so.12"),
        ("nvidia", "cublas", "lib", "libcublasLt.so.12"),
        ("nvidia", "nccl", "lib", "libnccl.so.2"),
    ):
        path = site_packages.joinpath(*relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[3]
    args = _build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--logical-world-size",
            "4",
            "--auto-profiled-strategy",
            "identity",
            "tests/workloads/fake_cuda/maya_fig13_megatron.py",
        ]
    )
    args.frun = fakecuda_root / "frun"
    args.python_bin = python_bin

    command, env, *_ = _capture_worker_command_and_env(
        args=args,
        profiled_index=0,
        representative_rank=0,
        rank_host_machines={},
        rank_host_dispatch_queues={},
        script_args=[],
        repo_root=str(repo_root),
        python_root=str(repo_root / "python"),
        output_dir=tmp_path / "out",
    )

    assert command[0] == str((fakecuda_root / "proot").resolve())
    assert env["FLEXSIM_MAYA_CAPTURE_LAUNCHER"] == "direct_proot"
    assert "-b" in command
    assert command[-2:] == [
        str(python_bin.absolute()),
        str(args.script.resolve()),
    ]


def test_fakecuda_artifact_fingerprint_metadata_records_required_libraries(
    tmp_path: Path,
) -> None:
    fakecuda_root = tmp_path / "fake-cuda"
    liboutput = fakecuda_root / "build" / "liboutput"
    liboutput.mkdir(parents=True)
    (fakecuda_root / "frun").write_text("#!/bin/sh\n", encoding="utf-8")
    expected_digest = None
    for library_name in (
        "libcudart.so.12",
        "libcuda.so.1",
        "libnvidia-ml.so",
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libnccl.so.2",
    ):
        payload = f"{library_name}\n".encode("utf-8")
        (liboutput / library_name).write_bytes(payload)
        if library_name == "libcublas.so.12":
            expected_digest = hashlib.sha256(payload).hexdigest()

    metadata = _fakecuda_artifact_fingerprint_metadata(fakecuda_root / "frun")

    assert metadata["contract_version"] == "fakecuda_artifact_fingerprint_v1"
    assert metadata["required_libraries_present"] is True
    cublas = metadata["libraries"]["libcublas.so.12"]
    assert cublas["exists"] is True
    assert cublas["sha256"] == expected_digest
    assert cublas["size_bytes"] == len("libcublas.so.12\n")


def test_fakecuda_artifact_fingerprint_metadata_marks_missing_libraries(
    tmp_path: Path,
) -> None:
    fakecuda_root = tmp_path / "fake-cuda"
    fakecuda_root.mkdir(parents=True)
    (fakecuda_root / "frun").write_text("#!/bin/sh\n", encoding="utf-8")

    metadata = _fakecuda_artifact_fingerprint_metadata(fakecuda_root / "frun")

    assert metadata["required_libraries_present"] is False
    assert metadata["libraries"]["libcublas.so.12"] == {
        "path": str(fakecuda_root / "build" / "liboutput" / "libcublas.so.12"),
        "exists": False,
        "size_bytes": None,
        "mtime_ns": None,
        "sha256": None,
    }


def test_capture_worker_command_falls_back_to_frun_without_direct_proot_layout(
    tmp_path: Path,
) -> None:
    fakecuda_root = tmp_path / "fake-cuda"
    fakecuda_root.mkdir(parents=True)
    (fakecuda_root / "frun").write_text("", encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[3]
    args = _build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--logical-world-size",
            "4",
            "--auto-profiled-strategy",
            "identity",
            "tests/workloads/fake_cuda/maya_fig13_megatron.py",
        ]
    )
    args.frun = fakecuda_root / "frun"

    command, env, *_ = _capture_worker_command_and_env(
        args=args,
        profiled_index=0,
        representative_rank=0,
        rank_host_machines={},
        rank_host_dispatch_queues={},
        script_args=[],
        repo_root=str(repo_root),
        python_root=str(repo_root / "python"),
        output_dir=tmp_path / "out",
    )

    assert command[0] == str((fakecuda_root / "frun").resolve())
    assert env["FLEXSIM_MAYA_CAPTURE_LAUNCHER"] == "frun"


def test_sync_all_reduce_waits_for_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _Runtime:
        def group_start(self):
            calls["group_start"] = int(calls.get("group_start", 0)) + 1

        def group_end(self):
            calls["group_end"] = int(calls.get("group_end", 0)) + 1

        def all_reduce(self, tensor, *, op, comm, stream):
            del tensor, op, comm
            calls["stream"] = stream

        def comm_get_async_error(self, comm):
            del comm

        def init_comm(self, size, local_rank, comm_id=None):
            del size, local_rank, comm_id
            return "comm"

    class _Tensor:
        is_cuda = True
        device = "cuda:0"

    current_stream = object()
    comm_stream = object()
    completion_event = object()

    monkeypatch.setattr(
        "flexsim.maya_lite.emulated_dist._stream_handle",
        lambda stream: stream,
    )
    monkeypatch.setattr(
        "torch.cuda.current_stream",
        lambda device=None: current_stream,
    )
    monkeypatch.setattr(
        "torch.cuda.stream",
        lambda stream: contextlib.nullcontext(),
    )

    group = EmulatedProcessGroup(
        ranks=(0, 1),
        local_rank=0,
        comm=None,
        runtime=_Runtime(),
        backend="nccl",
        collective_mode="trace_only",
        name="world",
    )
    monkeypatch.setattr(
        group,
        "_prepare_stream_handoff",
        lambda tensor, wait_on_comm_stream: (current_stream, comm_stream),
    )
    def _record_completion_sync(*, current_stream, comm_stream, wait_on_current_stream):
        del current_stream, comm_stream
        calls["wait_on_current_stream"] = wait_on_current_stream
        return completion_event

    monkeypatch.setattr(group, "_record_completion", _record_completion_sync)

    completion = group.all_reduce(_Tensor(), op=0, async_op=False)

    assert completion is None
    assert calls["stream"] is comm_stream
    assert calls["wait_on_current_stream"] is False
    assert calls["group_start"] == 1
    assert calls["group_end"] == 1


def test_emulated_work_wait_synchronizes_when_event_query_is_not_ready() -> None:
    calls: dict[str, int] = {"polls": 0, "synchronize": 0, "query": 0}

    class _Event:
        def synchronize(self) -> None:
            calls["synchronize"] += 1

        def query(self) -> bool:
            calls["query"] += 1
            return False

    class _ProcessGroup:
        def poll_async_error(self, comm) -> None:
            del comm
            calls["polls"] += 1

    tensor = object()
    work = EmulatedWork(
        tensor=tensor,
        event=_Event(),
        process_group=_ProcessGroup(),
        comm="comm",
    )

    assert work.wait() is tensor
    assert calls["synchronize"] == 1
    assert calls["query"] == 1
    assert calls["polls"] == 3


def test_emulated_work_wait_returns_without_synchronize_when_event_is_ready() -> None:
    calls: dict[str, int] = {"polls": 0, "synchronize": 0, "query": 0}

    class _Event:
        def synchronize(self) -> None:
            calls["synchronize"] += 1

        def query(self) -> bool:
            calls["query"] += 1
            return True

    class _ProcessGroup:
        def poll_async_error(self, comm) -> None:
            del comm
            calls["polls"] += 1

    work = EmulatedWork(
        tensor="tensor",
        event=_Event(),
        process_group=_ProcessGroup(),
        comm="comm",
    )

    assert work.wait() == "tensor"
    assert calls["synchronize"] == 0
    assert calls["query"] == 1
    assert calls["polls"] == 2


def test_async_all_reduce_uses_comm_stream_handoff(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _Runtime:
        def group_start(self):
            calls["group_start"] = int(calls.get("group_start", 0)) + 1

        def group_end(self):
            calls["group_end"] = int(calls.get("group_end", 0)) + 1

        def all_reduce(self, tensor, *, op, comm, stream):
            del tensor, op, comm
            calls["stream"] = stream

        def comm_get_async_error(self, comm):
            del comm

        def init_comm(self, size, local_rank, comm_id=None):
            del size, local_rank, comm_id
            return "comm"

    class _Tensor:
        is_cuda = True
        device = "cuda:0"

    current_stream = object()
    comm_stream = object()
    completion_event = object()

    monkeypatch.setattr(
        "flexsim.maya_lite.emulated_dist._stream_handle",
        lambda stream: stream,
    )
    monkeypatch.setattr(
        "torch.cuda.current_stream",
        lambda device=None: current_stream,
    )
    monkeypatch.setattr(
        "torch.cuda.stream",
        lambda stream: contextlib.nullcontext(),
    )

    group = EmulatedProcessGroup(
        ranks=(0, 1),
        local_rank=0,
        comm=None,
        runtime=_Runtime(),
        backend="nccl",
        collective_mode="trace_only",
        name="world",
    )

    monkeypatch.setattr(
        group,
        "_prepare_stream_handoff",
        lambda tensor, wait_on_comm_stream: (current_stream, comm_stream),
    )
    def _record_completion_async(*, current_stream, comm_stream, wait_on_current_stream):
        del current_stream, comm_stream
        calls["wait_on_current_stream"] = wait_on_current_stream
        return completion_event

    monkeypatch.setattr(group, "_record_completion", _record_completion_async)

    completion = group.all_reduce(_Tensor(), op=0, async_op=True)

    assert completion is completion_event
    assert calls["stream"] is comm_stream
    assert calls["wait_on_current_stream"] is False
    assert calls["group_start"] == 1
    assert calls["group_end"] == 1


def test_derive_fakecuda_runtime_env_prefers_explicit_env_root_layout(tmp_path: Path) -> None:
    env_root = tmp_path / "env"
    python_bin = env_root / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("", encoding="utf-8")
    site_packages = (
        env_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    (site_packages / "nvidia" / "cuda_runtime" / "lib").mkdir(parents=True)
    (site_packages / "nvidia" / "cublas" / "lib").mkdir(parents=True)
    (site_packages / "nvidia" / "nccl" / "lib").mkdir(parents=True)
    (site_packages / "nvidia" / "cuda_runtime" / "lib" / "libcudart.so.12").write_text("", encoding="utf-8")
    (site_packages / "nvidia" / "cublas" / "lib" / "libcublas.so.12").write_text("", encoding="utf-8")
    (site_packages / "nvidia" / "cublas" / "lib" / "libcublasLt.so.12").write_text("", encoding="utf-8")
    (site_packages / "nvidia" / "nccl" / "lib" / "libnccl.so.2").write_text("", encoding="utf-8")

    resolved = _derive_fakecuda_runtime_env(str(python_bin))

    assert resolved["FAKECUDA_TARGET_ENV_ROOT"] == str(env_root.absolute())
    assert resolved["FAKECUDA_SITE_PACKAGES_ROOT"] == str(site_packages.absolute())
    assert resolved["FAKECUDA_FRUN_QUIET"] == "1"
    assert resolved["FAKECUDA_SKIP_LDCONFIG"] == "1"
    assert resolved["TARGET_CUDART"].endswith("libcudart.so.12")
    assert resolved["TARGET_CUBLAS"].endswith("libcublas.so.12")
    assert resolved["TARGET_CUBLASLT"].endswith("libcublasLt.so.12")
    assert resolved["TARGET_NCCL"].endswith("libnccl.so.2")


def test_capture_emulated_parser_accepts_dynamic_first_iteration_dedup() -> None:
    args = _build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--logical-world-size",
            "4",
            "--auto-profiled-strategy",
            "identity",
            "--dynamic-first-iteration-dedup",
            "--dynamic-dedup-window-size",
            "32",
            "--dynamic-dedup-poll-interval-ms",
            "25",
            "tests/workloads/fake_cuda/maya_fig13_megatron.py",
        ]
    )

    assert args.dynamic_first_iteration_dedup is True
    assert args.dynamic_dedup_window_size == 32
    assert args.dynamic_dedup_poll_interval_ms == 25


def test_dynamic_first_iteration_dedup_promotes_buffered_trace_flush_policy() -> None:
    args = _build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--logical-world-size",
            "4",
            "--auto-profiled-strategy",
            "identity",
            "--dynamic-first-iteration-dedup",
            "--trace-flush-mode",
            "buffered",
            "--trace-flush-every",
            "4096",
            "--trace-stdio-buffer-bytes",
            str(4 * 1024 * 1024),
            "tests/workloads/fake_cuda/maya_fig13_megatron.py",
        ]
    )

    mode, flush_every, stdio_buffer_bytes = _resolve_trace_flush_policy(args)

    assert mode == "per_event"
    assert flush_every == 64
    assert stdio_buffer_bytes == 4 * 1024 * 1024


def test_dynamic_first_iteration_dedup_keeps_explicit_per_event_flush_policy() -> None:
    args = _build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--logical-world-size",
            "4",
            "--auto-profiled-strategy",
            "identity",
            "--dynamic-first-iteration-dedup",
            "--trace-flush-mode",
            "per_event",
            "--trace-flush-every",
            "4096",
            "tests/workloads/fake_cuda/maya_fig13_megatron.py",
        ]
    )

    mode, flush_every, _ = _resolve_trace_flush_policy(args)

    assert mode == "per_event"
    assert flush_every == 4096


def test_summarize_profiled_capture_distinguishes_full_world_mode() -> None:
    summary = _summarize_profiled_capture(
        logical_world_size=4,
        profiled_rank_groups={
            0: (0,),
            1: (1,),
            2: (2,),
            3: (3,),
        },
        planning_strategy="identity",
    )

    assert summary["profiled_world_size"] == 4
    assert summary["covers_full_logical_world"] is True
    assert summary["full_world_emulation"] is True
    assert summary["worker_selection_mode"] == "full_world"
    assert summary["paper_alignment_mode"] == "emulator_full_world_validation"


def test_summarize_profiled_capture_distinguishes_fig13_unique_workers() -> None:
    summary = _summarize_profiled_capture(
        logical_world_size=8,
        profiled_rank_groups={
            0: (0, 2, 4, 6),
            1: (1, 3, 5, 7),
        },
        planning_strategy="megatron_pp_stage",
    )

    assert summary["profiled_world_size"] == 2
    assert summary["covered_logical_rank_count"] == 8
    assert summary["covers_full_logical_world"] is True
    assert summary["full_world_emulation"] is False
    assert summary["worker_selection_mode"] == "selective_profiled"
    assert summary["paper_alignment_mode"] == "fig13_unique_workers"


def test_summarize_profiled_capture_keeps_nonpaper_selective_modes_distinct() -> None:
    summary = _summarize_profiled_capture(
        logical_world_size=8,
        profiled_rank_groups={
            0: (0, 1),
            2: (2, 3),
            4: (4, 5),
            6: (6, 7),
        },
        planning_strategy="pairwise",
    )

    assert summary["full_world_emulation"] is False
    assert summary["worker_selection_mode"] == "selective_profiled"
    assert summary["paper_alignment_mode"] == "selective_profiled_validation"


def test_copy_tensor_contents_handles_view_outputs_under_grad_mode() -> None:
    source = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    destination = torch.zeros_like(source, requires_grad=True)
    destination_view = destination.unbind(0)[0]
    source_view = source.unbind(0)[0]

    _copy_tensor_contents(destination_view, source_view)

    assert torch.equal(destination_view.detach(), source_view)


def test_sitecustomize_bootstrap_installs_emulated_dist_from_env() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{repo_root / 'python'}:{env.get('PYTHONPATH', '')}"
    env["FLEXSIM_MAYA_EMULATED_DIST"] = "1"
    env["FLEXSIM_MAYA_LOGICAL_RANK"] = "3"
    env["FLEXSIM_MAYA_LOGICAL_WORLD_SIZE"] = "8"

    code = """
import json
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

world = dist.init_process_group(backend="gloo", rank=3, world_size=8)
subgroup = dist.new_group([1, 3, 5], backend="nccl")
payload = {
    "world_backend": dist.get_backend(),
    "subgroup_backend": dist.get_backend(subgroup),
    "rank": dist.get_rank(),
    "world_size": dist.get_world_size(),
    "sub_rank": dist.get_rank(subgroup),
    "sub_world_size": dist.get_world_size(subgroup),
    "c10d_world_size": c10d.get_world_size(subgroup),
    "default_is_world": world.name == "world",
}
print(json.dumps(payload))
dist.destroy_process_group()
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload == {
        "world_backend": "gloo",
        "subgroup_backend": "nccl",
        "rank": 3,
        "world_size": 8,
        "sub_rank": 1,
        "sub_world_size": 3,
        "c10d_world_size": 3,
        "default_is_world": True,
    }


def test_resolve_step_window_from_markers_uses_source_specific_clock() -> None:
    marker_records = [
        {"kind": "step_begin", "label": "training_step", "realtime_ns": 1_000_000, "monotonic_ns": 2_000_000},
        {"kind": "step_end", "label": "training_step", "realtime_ns": 6_000_000, "monotonic_ns": 9_000_000},
    ]

    fake_window = resolve_step_window_from_markers(marker_records, source=TraceSource.FAKE)
    real_window = resolve_step_window_from_markers(marker_records, source=TraceSource.REAL)

    assert fake_window == {
        "start_ts": 1_000,
        "end_ts": 6_000,
        "source": "trace_markers",
        "step_count": 1,
    }
    assert real_window == {
        "start_ts": 2_000,
        "end_ts": 9_000,
        "source": "trace_markers",
        "step_count": 1,
    }


def test_resolve_step_window_from_marker_trace_timestamps_prefers_exact_trace_clock() -> None:
    marker_records = [
        {
            "kind": "step_begin",
            "label": "training_step",
            "realtime_ns": 1_000_000,
            "trace_ts": 1_200,
        },
        {
            "kind": "step_end",
            "label": "training_step",
            "realtime_ns": 6_000_000,
            "trace_ts": 2_400,
        },
    ]

    resolved = resolve_step_window_from_marker_trace_timestamps(marker_records)

    assert resolved == {
        "start_ts": 1_200,
        "end_ts": 2_400,
        "source": "trace_markers",
        "step_count": 1,
    }


def test_resolve_indexed_step_window_from_markers_uses_first_iteration() -> None:
    marker_records = [
        {"kind": "step_begin", "label": "training_step", "step": 1, "realtime_ns": 1_000_000},
        {"kind": "step_end", "label": "training_step", "step": 1, "realtime_ns": 4_000_000},
        {"kind": "step_begin", "label": "training_step", "step": 2, "realtime_ns": 7_000_000},
        {"kind": "step_end", "label": "training_step", "step": 2, "realtime_ns": 11_000_000},
    ]

    resolved = resolve_indexed_step_window_from_markers(
        marker_records,
        source=TraceSource.FAKE,
        occurrence=1,
    )

    assert resolved == {
        "start_ts": 1_000,
        "end_ts": 4_000,
        "source": "trace_markers",
        "step_count": 2,
        "occurrence": 1,
    }


def test_resolve_indexed_step_window_from_marker_trace_timestamps_uses_requested_occurrence() -> None:
    marker_records = [
        {"kind": "step_begin", "label": "training_step", "step": 1, "trace_ts": 1_200},
        {"kind": "step_end", "label": "training_step", "step": 1, "trace_ts": 2_400},
        {"kind": "step_begin", "label": "training_step", "step": 2, "trace_ts": 4_200},
        {"kind": "step_end", "label": "training_step", "step": 2, "trace_ts": 5_400},
    ]

    resolved = resolve_indexed_step_window_from_marker_trace_timestamps(
        marker_records,
        occurrence=2,
    )

    assert resolved == {
        "start_ts": 4_200,
        "end_ts": 5_400,
        "source": "trace_markers",
        "step_count": 2,
        "occurrence": 2,
    }


def test_resolve_indexed_step_window_from_trace_markers_uses_requested_occurrence(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "rank_0.trace_markers.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"ts": 1_000, "pid": 1, "tid": 1, "mod": "libcudart.so.12", "api": TRACE_MARKER_API, "type": "marker", "kind": "step_begin", "label": "training_step", "step": 1}),
                json.dumps({"ts": 4_000, "pid": 1, "tid": 1, "mod": "libcudart.so.12", "api": TRACE_MARKER_API, "type": "marker", "kind": "step_end", "label": "training_step", "step": 1}),
                json.dumps({"ts": 7_000, "pid": 1, "tid": 1, "mod": "libcudart.so.12", "api": TRACE_MARKER_API, "type": "marker", "kind": "step_begin", "label": "training_step", "step": 2}),
                json.dumps({"ts": 11_000, "pid": 1, "tid": 1, "mod": "libcudart.so.12", "api": TRACE_MARKER_API, "type": "marker", "kind": "step_end", "label": "training_step", "step": 2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    resolved = resolve_indexed_step_window_from_trace_markers(
        trace_path,
        occurrence=2,
    )

    assert resolved == {
        "start_ts": 7_000,
        "end_ts": 11_000,
        "source": "trace_markers",
        "step_count": 2,
        "occurrence": 2,
    }


def test_resolve_capture_step_window_keeps_marker_window_without_trace_host_timing(tmp_path: Path) -> None:
    trace_path = tmp_path / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"ts": 1_100, "pid": 1, "tid": 1, "mod": "lib", "api": "a", "type": "other"}),
                json.dumps({"ts": 5_000, "pid": 1, "tid": 1, "mod": "lib", "api": "b", "type": "other"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    marker_records = [
        {"kind": "step_begin", "label": "training_step", "realtime_ns": 1_000_000},
        {"kind": "step_end", "label": "training_step", "realtime_ns": 2_000_000},
    ]

    resolved = _resolve_capture_step_window(
        trace_path,
        marker_records=marker_records,
        host_timing_mode=None,
    )

    assert resolved == {
        "start_ts": 1_000,
        "end_ts": 2_000,
        "source": "trace_markers",
        "step_count": 1,
    }


@pytest.mark.parametrize("host_timing_mode", ["trace", "measure"])
def test_resolve_capture_step_window_uses_trace_markers_in_trace_clock_modes(
    tmp_path: Path,
    host_timing_mode: str,
) -> None:
    trace_path = tmp_path / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"ts": 900, "pid": 1, "tid": 1, "mod": "lib", "api": "bootstrap", "type": "other"}),
                json.dumps({"ts": 1_200, "pid": 1, "tid": 1, "mod": "libcudart.so.12", "api": TRACE_MARKER_API, "type": "marker", "kind": "step_begin", "label": "training_step"}),
                json.dumps({"ts": 1_500, "pid": 1, "tid": 1, "mod": "lib", "api": "a", "type": "other"}),
                json.dumps({"ts": 2_400, "pid": 1, "tid": 1, "mod": "libcudart.so.12", "api": TRACE_MARKER_API, "type": "marker", "kind": "step_end", "label": "training_step"}),
                json.dumps({"ts": 5_000, "pid": 1, "tid": 1, "mod": "lib", "api": "teardown", "type": "other"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    marker_records = [
        {"kind": "step_begin", "label": "training_step", "realtime_ns": 1_000_000},
        {"kind": "step_end", "label": "training_step", "realtime_ns": 2_000_000},
    ]

    resolved = _resolve_capture_step_window(
        trace_path,
        marker_records=marker_records,
        host_timing_mode=host_timing_mode,
    )

    assert resolved == {
        "start_ts": 1_200,
        "end_ts": 2_400,
        "source": "trace_markers",
        "step_count": 1,
    }


@pytest.mark.parametrize("host_timing_mode", ["trace", "measure"])
def test_resolve_capture_step_window_prefers_marker_sidecar_trace_timestamps(
    tmp_path: Path,
    host_timing_mode: str,
) -> None:
    trace_path = tmp_path / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"ts": 900, "pid": 1, "tid": 1, "mod": "lib", "api": "bootstrap", "type": "other"}),
                json.dumps({"ts": 1_500, "pid": 1, "tid": 1, "mod": "lib", "api": "a", "type": "other"}),
                json.dumps({"ts": 5_000, "pid": 1, "tid": 1, "mod": "lib", "api": "teardown", "type": "other"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    marker_records = [
        {"kind": "step_begin", "label": "training_step", "realtime_ns": 1_000_000, "trace_ts": 1_200},
        {"kind": "step_end", "label": "training_step", "realtime_ns": 2_000_000, "trace_ts": 2_400},
    ]

    resolved = _resolve_capture_step_window(
        trace_path,
        marker_records=marker_records,
        host_timing_mode=host_timing_mode,
    )

    assert resolved == {
        "start_ts": 1_200,
        "end_ts": 2_400,
        "source": "trace_markers",
        "step_count": 1,
    }


@pytest.mark.parametrize("host_timing_mode", ["trace", "measure"])
def test_resolve_capture_step_window_keeps_clean_trace_marker_window_without_tail_extension(
    tmp_path: Path,
    host_timing_mode: str,
) -> None:
    trace_path = tmp_path / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"ts": 1_200, "pid": 1, "tid": 10, "mod": "libcudart.so.12", "api": TRACE_MARKER_API, "type": "marker", "kind": "step_begin", "label": "training_step"}),
                json.dumps({"ts": 1_500, "pid": 1, "tid": 10, "mod": "lib", "api": "forward", "type": "other"}),
                json.dumps({"ts": 2_400, "pid": 1, "tid": 10, "mod": "libcudart.so.12", "api": TRACE_MARKER_API, "type": "marker", "kind": "step_end", "label": "training_step"}),
                json.dumps({"ts": 9_000, "pid": 1, "tid": 99, "mod": "lib", "api": "late_worker", "type": "other"}),
                json.dumps({"ts": 800, "pid": 1, "tid": 77, "mod": "lib", "api": "early_worker", "type": "other"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    resolved = _resolve_capture_step_window(
        trace_path,
        marker_records=[],
        host_timing_mode=host_timing_mode,
    )

    assert resolved == {
        "start_ts": 1_200,
        "end_ts": 2_400,
        "source": "trace_markers",
        "step_count": 1,
    }


@pytest.mark.parametrize("host_timing_mode", ["trace", "measure"])
def test_resolve_capture_step_window_uses_requested_indexed_trace_window_without_tail_extension(
    tmp_path: Path,
    host_timing_mode: str,
) -> None:
    trace_path = tmp_path / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"ts": 1_200, "pid": 1, "tid": 10, "mod": "libcudart.so.12", "api": TRACE_MARKER_API, "type": "marker", "kind": "step_begin", "label": "training_step", "step": 1}),
                json.dumps({"ts": 2_400, "pid": 1, "tid": 10, "mod": "libcudart.so.12", "api": TRACE_MARKER_API, "type": "marker", "kind": "step_end", "label": "training_step", "step": 1}),
                json.dumps({"ts": 4_200, "pid": 1, "tid": 10, "mod": "libcudart.so.12", "api": TRACE_MARKER_API, "type": "marker", "kind": "step_begin", "label": "training_step", "step": 2}),
                json.dumps({"ts": 5_400, "pid": 1, "tid": 10, "mod": "libcudart.so.12", "api": TRACE_MARKER_API, "type": "marker", "kind": "step_end", "label": "training_step", "step": 2}),
                json.dumps({"ts": 9_000, "pid": 1, "tid": 99, "mod": "lib", "api": "late_worker", "type": "other"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    resolved = _resolve_capture_step_window(
        trace_path,
        marker_records=[],
        host_timing_mode=host_timing_mode,
        capture_step_window_occurrence=2,
    )

    assert resolved == {
        "start_ts": 4_200,
        "end_ts": 5_400,
        "source": "trace_markers",
        "step_count": 2,
        "occurrence": 2,
    }


def test_resolve_capture_step_window_rejects_unresolvable_requested_indexed_window(tmp_path: Path) -> None:
    trace_path = tmp_path / "rank_0.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "ts": 1_200,
                "pid": 1,
                "tid": 10,
                "mod": "libcudart.so.12",
                "api": TRACE_MARKER_API,
                "type": "marker",
                "kind": "step_begin",
                "label": "training_step",
                "step": 1,
            }
        )
        + "\n"
        + json.dumps(
            {
                "ts": 2_400,
                "pid": 1,
                "tid": 10,
                "mod": "libcudart.so.12",
                "api": TRACE_MARKER_API,
                "type": "marker",
                "kind": "step_end",
                "label": "training_step",
                "step": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="requested capture step window could not be resolved"):
        _resolve_capture_step_window(
            trace_path,
            marker_records=[],
            host_timing_mode="measure",
            capture_step_window_occurrence=2,
        )


def test_default_workload_heuristic_step_window_is_disabled_for_non_trim_capture() -> None:
    assert (
        _should_apply_default_workload_heuristic_step_window(
            args=SimpleNamespace(trim_to_step_window=False),
            resolved_step_window={
                "start_ts": 1_000,
                "end_ts": 2_000,
                "source": "trace_markers",
                "step_count": 1,
            },
        )
        is False
    )


def test_default_workload_heuristic_step_window_stays_available_for_trim_capture() -> None:
    assert (
        _should_apply_default_workload_heuristic_step_window(
            args=SimpleNamespace(trim_to_step_window=True),
            resolved_step_window={
                "start_ts": 1_000,
                "end_ts": 2_000,
                "source": "trace_markers",
                "step_count": 1,
            },
        )
        is True
    )


def test_resolve_workload_heuristic_step_window_extends_marker_window_for_supported_tail(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "rank_0.jsonl"
    records = [
        {
            "ts": 1_000,
            "pid": 1,
            "tid": 1,
            "mod": "lib",
            "api": "forward",
            "type": "other",
        }
    ]
    for offset in range(40):
        records.append(
            {
                "ts": 2_100 + (offset * 100),
                "end_ts": 2_150 + (offset * 100),
                "pid": 1,
                "tid": 1,
                "mod": "libcublas.so.12",
                "api": "cublasGemmEx",
                "type": "blas_compute",
            }
        )
    trace_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    resolved = _resolve_workload_heuristic_step_window(
        trace_path,
        resolved_step_window={
            "start_ts": 1_000,
            "end_ts": 2_000,
            "source": "trace_markers",
            "step_count": 1,
        },
        host_timing_mode="trace",
        min_extension_us=100,
        min_supported_tail_events=4,
    )

    assert resolved == {
        "start_ts": 1_000,
        "end_ts": 6_050,
        "source": "workload_heuristic",
        "step_count": 1,
        "base_source": "trace_markers",
        "base_end_ts": 2_000,
        "heuristic_name": "post_step_supported_tail",
        "post_step_supported_event_count": 40,
        "post_step_semantic_event_count": 40,
    }


def test_resolve_workload_heuristic_step_window_uses_gap_cutoff_after_semantic_tail(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "rank_0.jsonl"
    records = []
    for offset in range(120):
        ts = 2_100 + (offset * 100)
        records.append(
            {
                "ts": ts,
                "end_ts": ts + 50,
                "pid": 1,
                "tid": 1,
                "mod": "libcublas.so.12",
                "api": "cublasGemmEx",
                "type": "blas_compute",
            }
        )
    records.extend(
        [
            {
                "ts": 120_000,
                "end_ts": 120_050,
                "pid": 1,
                "tid": 1,
                "mod": "libcublas.so.12",
                "api": "cublasGemmEx",
                "type": "blas_compute",
            },
            {
                "ts": 120_100,
                "end_ts": 120_150,
                "pid": 1,
                "tid": 1,
                "mod": "libcublas.so.12",
                "api": "cublasGemmEx",
                "type": "blas_compute",
            },
        ]
    )
    trace_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    resolved = _resolve_workload_heuristic_step_window(
        trace_path,
        resolved_step_window={
            "start_ts": 1_000,
            "end_ts": 2_000,
            "source": "trace_markers",
            "step_count": 1,
        },
        host_timing_mode="trace",
        min_extension_us=100,
        min_supported_tail_events=4,
        gap_cutoff_us=100_000,
        min_gap_candidate_semantic_events=100,
    )

    assert resolved == {
        "start_ts": 1_000,
        "end_ts": 14_050,
        "source": "workload_heuristic",
        "step_count": 1,
        "base_source": "trace_markers",
        "base_end_ts": 2_000,
        "heuristic_name": "post_step_supported_tail_gap_cutoff",
        "post_step_supported_event_count": 122,
        "post_step_semantic_event_count": 122,
        "gap_cutoff_us": 100_000,
        "gap_cutoff_candidate_end_ts": 14_050,
        "gap_cutoff_min_semantic_events": 100,
    }


def test_resolve_workload_heuristic_step_window_ignores_trivial_compat_tail(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 1_000,
                        "pid": 1,
                        "tid": 1,
                        "mod": "lib",
                        "api": "forward",
                        "type": "other",
                    }
                ),
                json.dumps(
                    {
                        "ts": 2_050,
                        "end_ts": 2_055,
                        "pid": 1,
                        "tid": 1,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetDevice",
                        "type": "context_op",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    resolved = _resolve_workload_heuristic_step_window(
        trace_path,
        resolved_step_window={
            "start_ts": 1_000,
            "end_ts": 2_000,
            "source": "trace_markers",
            "step_count": 1,
        },
        host_timing_mode="trace",
        min_extension_us=100,
        min_supported_tail_events=4,
    )

    assert resolved is None


def test_trim_trace_file_to_ts_window_drops_trace_markers(tmp_path: Path) -> None:
    trace_path = tmp_path / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"ts": 1_000, "pid": 1, "tid": 1, "mod": "libcudart.so.12", "api": TRACE_MARKER_API, "type": "marker", "kind": "step_begin", "label": "training_step"}),
                json.dumps({"ts": 1_100, "pid": 1, "tid": 1, "mod": "lib", "api": "a", "type": "other"}),
                json.dumps({"ts": 2_000, "pid": 1, "tid": 1, "mod": "libcudart.so.12", "api": TRACE_MARKER_API, "type": "marker", "kind": "step_end", "label": "training_step"}),
                json.dumps({"ts": 2_100, "pid": 1, "tid": 1, "mod": "lib", "api": "b", "type": "other"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = _trim_trace_file_to_ts_window(trace_path, start_ts=1_000, end_ts=2_000)

    kept = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert summary["total_events"] == 4
    assert summary["kept_events"] == 1
    assert [record["api"] for record in kept] == ["a"]


def test_trim_trace_file_to_ts_window_drops_teardown_apis(tmp_path: Path) -> None:
    trace_path = tmp_path / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"ts": 1_100, "pid": 1, "tid": 1, "mod": "lib", "api": "work_a", "type": "other"}),
                json.dumps({"ts": 1_200, "pid": 1, "tid": 1, "mod": "libnccl.so.2", "api": "ncclCommDestroy", "type": "other"}),
                json.dumps({"ts": 1_300, "pid": 1, "tid": 1, "mod": "lib", "api": "work_b", "type": "other"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = _trim_trace_file_to_ts_window(trace_path, start_ts=1_000, end_ts=2_000)

    kept = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert summary["total_events"] == 3
    assert summary["kept_events"] == 2
    assert [record["api"] for record in kept] == ["work_a", "work_b"]


def test_trim_trace_file_to_ts_window_keeps_overlap_via_end_ts(tmp_path: Path) -> None:
    trace_path = tmp_path / "rank_0.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 900,
                        "end_ts": 1_050,
                        "pid": 1,
                        "tid": 1,
                        "mod": "libcudart.so.12",
                        "api": "cudaMemcpy",
                        "type": "mem_copy",
                    }
                ),
                json.dumps(
                    {
                        "ts": 800,
                        "end_ts": 850,
                        "pid": 1,
                        "tid": 1,
                        "mod": "libcudart.so.12",
                        "api": "cudaGetDevice",
                        "type": "context_op",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = _trim_trace_file_to_ts_window(trace_path, start_ts=1_000, end_ts=1_100)

    kept = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert summary["total_events"] == 2
    assert summary["kept_events"] == 1
    assert [record["api"] for record in kept] == ["cudaMemcpy"]


def test_step_window_pattern_tokens_accepts_raw_trace_temp_name(tmp_path: Path) -> None:
    trace_path = tmp_path / "rank_0.raw.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"ts": 1_000, "pid": 1, "tid": 1, "mod": "libcudart.so.12", "api": "cudaEventRecord", "type": "stream_op"}),
                json.dumps({"ts": 1_200, "pid": 1, "tid": 1, "mod": "libcudart.so.12", "api": "cudaLaunchKernel", "type": "kernel_launch"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    communicator_path = tmp_path / "rank_0.communicators.json"
    communicator_path.write_text(json.dumps({"communicators": {}}), encoding="utf-8")

    tokens = _step_window_pattern_tokens(
        trace_path=trace_path,
        representative_rank=0,
        step_window={"start_ts": 900, "end_ts": 1_300, "source": "trace_markers"},
        communicator_path=communicator_path,
    )

    assert tokens is not None
    assert len(tokens) == 2


def test_step_window_pattern_tokens_skips_partial_raw_trace_line(tmp_path: Path) -> None:
    trace_path = tmp_path / "rank_0.raw.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 1_000,
                        "pid": 1,
                        "tid": 1,
                        "mod": "libcudart.so.12",
                        "api": "cudaEventRecord",
                        "type": "stream_op",
                    }
                ),
                '{"ts": 1200, "pid": 1, "tid": 1, "mod": "libcudart.so.12", "api": "cudaLaunchKernel',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    communicator_path = tmp_path / "rank_0.communicators.json"
    communicator_path.write_text(json.dumps({"communicators": {}}), encoding="utf-8")

    tokens = _step_window_pattern_tokens(
        trace_path=trace_path,
        representative_rank=0,
        step_window={"start_ts": 900, "end_ts": 1_100, "source": "trace_markers"},
        communicator_path=communicator_path,
    )

    assert tokens is not None
    assert len(tokens) == 1


def test_step_window_pattern_tokens_uses_end_ts_for_overlap_and_completion(tmp_path: Path) -> None:
    trace_path = tmp_path / "rank_0.raw.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 800,
                        "end_ts": 1_300,
                        "pid": 1,
                        "tid": 1,
                        "mod": "libcudart.so.12",
                        "api": "cudaMemcpy",
                        "type": "mem_copy",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    communicator_path = tmp_path / "rank_0.communicators.json"
    communicator_path.write_text(json.dumps({"communicators": {}}), encoding="utf-8")

    tokens = _step_window_pattern_tokens(
        trace_path=trace_path,
        representative_rank=0,
        step_window={"start_ts": 900, "end_ts": 1_200, "source": "trace_markers"},
        communicator_path=communicator_path,
    )

    assert tokens is not None
    assert len(tokens) == 1


def test_dynamic_first_iteration_dedup_fallback_skips_missing_raw_trace(tmp_path: Path) -> None:
    worker = SimpleNamespace(
        first_step_classified=False,
        marker_path=tmp_path / "rank_0.markers.jsonl",
        trace_temp_path=tmp_path / "rank_0.raw.jsonl",
        representative_rank=0,
        communicator_path=tmp_path / "rank_0.communicators.json",
        process=SimpleNamespace(poll=lambda: 0),
        duplicate_of=None,
        termination_reason=None,
        termination_requested_at=None,
        first_step_classified_at=None,
        cached_marker_records=None,
        cached_marker_mtime_ns=None,
        cached_communicator_memberships=None,
        cached_communicator_mtime_ns=None,
    )

    first_step_hash_to_representative: dict[str, int] = {}
    dynamic_rank_groups: dict[int, list[int]] = {}

    _maybe_classify_first_step_pattern(
        worker,
        first_step_hash_to_representative=first_step_hash_to_representative,
        dynamic_rank_groups=dynamic_rank_groups,
        allow_full_trace_fallback=True,
    )

    assert worker.first_step_classified is False
    assert first_step_hash_to_representative == {}
    assert dynamic_rank_groups == {}


def test_dynamic_first_iteration_dedup_prefers_trace_marker_window(tmp_path: Path) -> None:
    marker_path = tmp_path / "rank_0.markers.jsonl"
    marker_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "kind": "step_begin",
                        "label": "training_step",
                        "realtime_ns": 1_000_000,
                    }
                ),
                json.dumps(
                    {
                        "kind": "step_end",
                        "label": "training_step",
                        "realtime_ns": 5_000_000,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    trace_path = tmp_path / "rank_0.raw.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 1_000,
                        "pid": 1,
                        "tid": 1,
                        "mod": "libcudart.so.12",
                        "api": TRACE_MARKER_API,
                        "type": "marker",
                        "kind": "step_begin",
                        "label": "training_step",
                    }
                ),
                json.dumps(
                    {
                        "ts": 2_000,
                        "end_ts": 2_400,
                        "pid": 1,
                        "tid": 1,
                        "mod": "libcudart.so.12",
                        "api": "cudaMemcpy",
                        "type": "mem_copy",
                    }
                ),
                json.dumps(
                    {
                        "ts": 3_000,
                        "pid": 1,
                        "tid": 1,
                        "mod": "libcudart.so.12",
                        "api": TRACE_MARKER_API,
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
    communicator_path = tmp_path / "rank_0.communicators.json"
    communicator_path.write_text(json.dumps({"communicators": {}}), encoding="utf-8")

    terminated = {"value": False}

    def _terminate() -> None:
        terminated["value"] = True

    worker = SimpleNamespace(
        first_step_classified=False,
        marker_path=marker_path,
        trace_temp_path=trace_path,
        representative_rank=0,
        communicator_path=communicator_path,
        process=SimpleNamespace(poll=lambda: None, terminate=_terminate),
        duplicate_of=None,
        termination_reason=None,
        termination_requested_at=None,
        first_step_classified_at=None,
        first_step_tokens=None,
        cached_marker_records=None,
        cached_marker_mtime_ns=None,
        cached_communicator_memberships=None,
        cached_communicator_mtime_ns=None,
    )

    first_step_tokens = _step_window_pattern_tokens(
        trace_path=trace_path,
        representative_rank=7,
        step_window={"start_ts": 1_000, "end_ts": 3_000, "source": "trace_markers"},
        communicator_path=communicator_path,
    )
    assert first_step_tokens is not None
    first_step_hash, *_ = _dynamic_dedup_sequence_hash(first_step_tokens)
    first_step_hash_to_representative = {first_step_hash: 7}
    dynamic_rank_groups: dict[int, list[int]] = {7: [7]}

    _maybe_classify_first_step_pattern(
        worker,
        first_step_hash_to_representative=first_step_hash_to_representative,
        dynamic_rank_groups=dynamic_rank_groups,
    )

    assert worker.first_step_classified is True
    assert worker.duplicate_of == 7
    assert worker.termination_reason == "dynamic_first_iteration_dedup"
    assert terminated["value"] is True
    assert dynamic_rank_groups == {7: [7, 0]}


def test_dynamic_first_iteration_dedup_waits_for_complete_step_window_before_classifying(
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / "rank_0.markers.jsonl"
    marker_path.write_text(
        json.dumps(
            {
                "kind": "step_begin",
                "label": "training_step",
                "realtime_ns": 1_000_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trace_path = tmp_path / "rank_0.raw.jsonl"
    records = []
    for index in range(32):
        records.append(
            json.dumps(
                {
                    "ts": 1_000 + (index * 100),
                    "end_ts": 1_050 + (index * 100),
                    "pid": 1,
                    "tid": 1,
                    "mod": "libcudart.so.12",
                    "api": "cudaMemcpy",
                    "type": "mem_copy",
                    "size": 1024 + index,
                }
            )
        )
    trace_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    communicator_path = tmp_path / "rank_0.communicators.json"
    communicator_path.write_text(json.dumps({"communicators": {}}), encoding="utf-8")

    terminated = {"value": False}

    def _terminate() -> None:
        terminated["value"] = True

    worker = SimpleNamespace(
        first_step_classified=False,
        marker_path=marker_path,
        trace_temp_path=trace_path,
        representative_rank=0,
        communicator_path=communicator_path,
        process=SimpleNamespace(poll=lambda: None, terminate=_terminate),
        duplicate_of=None,
        termination_reason=None,
        termination_requested_at=None,
        first_step_classified_at=None,
        first_step_tokens=None,
        cached_marker_records=None,
        cached_marker_mtime_ns=None,
        cached_communicator_memberships=None,
        cached_communicator_mtime_ns=None,
    )

    prefix_tokens = _prefix_pattern_tokens(
        trace_path=trace_path,
        representative_rank=7,
        marker_path=marker_path,
        communicator_path=communicator_path,
    )
    assert prefix_tokens is not None
    first_step_hash_to_representative: dict[str, int] = {}
    dynamic_rank_groups: dict[int, list[int]] = {7: [7]}

    _maybe_classify_first_step_pattern(
        worker,
        first_step_hash_to_representative=first_step_hash_to_representative,
        dynamic_rank_groups=dynamic_rank_groups,
    )

    assert worker.first_step_classified is False
    assert worker.duplicate_of is None
    assert worker.termination_reason is None
    assert terminated["value"] is False
    assert first_step_hash_to_representative == {}
    assert dynamic_rank_groups == {7: [7]}


def test_dynamic_first_iteration_dedup_prefix_waits_for_enough_tokens(tmp_path: Path) -> None:
    marker_path = tmp_path / "rank_0.markers.jsonl"
    marker_path.write_text(
        json.dumps(
            {
                "kind": "step_begin",
                "label": "training_step",
                "realtime_ns": 1_000_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trace_path = tmp_path / "rank_0.raw.jsonl"
    records = []
    for index in range(31):
        records.append(
            json.dumps(
                {
                    "ts": 1_000 + (index * 100),
                    "end_ts": 1_050 + (index * 100),
                    "pid": 1,
                    "tid": 1,
                    "mod": "libcudart.so.12",
                    "api": "cudaMemcpy",
                    "type": "mem_copy",
                    "size": 1024 + index,
                }
            )
        )
    trace_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    communicator_path = tmp_path / "rank_0.communicators.json"
    communicator_path.write_text(json.dumps({"communicators": {}}), encoding="utf-8")

    worker = SimpleNamespace(
        first_step_classified=False,
        marker_path=marker_path,
        trace_temp_path=trace_path,
        representative_rank=0,
        communicator_path=communicator_path,
        process=SimpleNamespace(poll=lambda: None, terminate=lambda: None),
        duplicate_of=None,
        termination_reason=None,
        termination_requested_at=None,
        first_step_classified_at=None,
        first_step_tokens=None,
        cached_marker_records=None,
        cached_marker_mtime_ns=None,
        cached_communicator_memberships=None,
        cached_communicator_mtime_ns=None,
    )
    first_step_hash_to_representative: dict[str, int] = {}
    dynamic_rank_groups: dict[int, list[int]] = {}

    _maybe_classify_first_step_pattern(
        worker,
        first_step_hash_to_representative=first_step_hash_to_representative,
        dynamic_rank_groups=dynamic_rank_groups,
    )

    assert worker.first_step_classified is False
    assert worker.duplicate_of is None
    assert worker.termination_reason is None
    assert first_step_hash_to_representative == {}


def test_request_dynamic_dedup_termination_prefers_process_group_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signaled: list[tuple[int, int]] = []

    monkeypatch.setattr(
        os,
        "killpg",
        lambda pid, signum: signaled.append((pid, signum)),
    )

    worker = SimpleNamespace(
        process=SimpleNamespace(pid=321, poll=lambda: None, terminate=lambda: None),
        termination_reason=None,
        termination_requested_at=None,
        termination_escalated=False,
    )

    _request_dynamic_dedup_termination(worker)

    assert worker.termination_reason == "dynamic_first_iteration_dedup"
    assert worker.termination_requested_at is not None
    assert signaled == [(321, signal.SIGTERM)]


def test_dynamic_dedup_termination_escalates_after_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signaled: list[tuple[int, int]] = []

    monkeypatch.setattr(
        os,
        "killpg",
        lambda pid, signum: signaled.append((pid, signum)),
    )

    worker = SimpleNamespace(
        process=SimpleNamespace(pid=654, poll=lambda: None, terminate=lambda: None, kill=lambda: None),
        termination_reason="dynamic_first_iteration_dedup",
        termination_requested_at=0.0,
        termination_escalated=False,
    )

    monkeypatch.setattr("flexsim.maya_lite.capture_emulated.time.perf_counter", lambda: 2.0)

    _maybe_escalate_dynamic_dedup_termination(
        worker,
        grace_seconds=1.0,
    )

    assert worker.termination_escalated is True
    assert signaled == [(654, signal.SIGKILL)]


def test_prefix_pattern_tokens_require_step_begin_marker(tmp_path: Path) -> None:
    marker_path = tmp_path / "rank_0.markers.jsonl"
    marker_path.write_text("", encoding="utf-8")
    trace_path = tmp_path / "rank_0.raw.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": 1_000 + (index * 100),
                        "end_ts": 1_050 + (index * 100),
                        "pid": 1,
                        "tid": 1,
                        "mod": "libcudart.so.12",
                        "api": "cudaMemcpy",
                        "type": "mem_copy",
                        "size": 1024 + index,
                    }
                )
                for index in range(40)
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    communicator_path = tmp_path / "rank_0.communicators.json"
    communicator_path.write_text(json.dumps({"communicators": {}}), encoding="utf-8")

    assert (
        _prefix_pattern_tokens(
            trace_path=trace_path,
            representative_rank=0,
            marker_path=marker_path,
            communicator_path=communicator_path,
        )
        is None
    )


def test_cached_prefix_pattern_tokens_reuses_marker_and_communicator_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / "rank_0.markers.jsonl"
    marker_path.write_text(
        json.dumps(
            {
                "kind": "step_begin",
                "label": "training_step",
                "realtime_ns": 1_000_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trace_path = tmp_path / "rank_0.raw.jsonl"
    trace_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "ts": 1_000 + (index * 100),
                    "end_ts": 1_050 + (index * 100),
                    "pid": 1,
                    "tid": 1,
                    "mod": "libcudart.so.12",
                    "api": "cudaMemcpy",
                    "type": "mem_copy",
                    "size": 1024 + index,
                }
            )
            for index in range(40)
        )
        + "\n",
        encoding="utf-8",
    )
    communicator_path = tmp_path / "rank_0.communicators.json"
    communicator_path.write_text(json.dumps({"communicators": {}}), encoding="utf-8")

    worker = SimpleNamespace(
        marker_path=marker_path,
        communicator_path=communicator_path,
        trace_temp_path=trace_path,
        representative_rank=0,
        cached_marker_mtime_ns=None,
        cached_marker_records=None,
        cached_communicator_mtime_ns=None,
        cached_communicator_memberships=None,
        cached_prefix_trace_size_bytes=-1,
        cached_prefix_tokens=None,
    )

    load_counts = {"markers": 0, "communicators": 0}
    original_load_markers = load_step_markers
    original_load_communicators = _prefix_pattern_tokens.__globals__["_load_worker_communicators"]

    def _counted_load_markers(path: Path):
        load_counts["markers"] += 1
        return original_load_markers(path)

    def _counted_load_communicators(path: Path):
        load_counts["communicators"] += 1
        return original_load_communicators(path)

    monkeypatch.setattr(
        "flexsim.maya_lite.capture_emulated.load_step_markers",
        _counted_load_markers,
    )
    monkeypatch.setattr(
        "flexsim.maya_lite.capture_emulated._load_worker_communicators",
        _counted_load_communicators,
    )

    first_tokens = _cached_prefix_pattern_tokens(worker)
    second_tokens = _cached_prefix_pattern_tokens(worker)

    assert first_tokens == second_tokens
    assert first_tokens is not None
    assert load_counts == {"markers": 1, "communicators": 1}


def test_cached_communicator_memberships_tolerates_partial_json(tmp_path: Path) -> None:
    communicator_path = tmp_path / "rank_0.communicators.json"
    communicator_path.write_text("{", encoding="utf-8")
    worker = SimpleNamespace(
        communicator_path=communicator_path,
        cached_communicator_mtime_ns=None,
        cached_communicator_memberships=None,
    )

    memberships = _cached_communicator_memberships(worker)

    assert memberships == {}


def test_launch_blocking_worker_count_excludes_retired_duplicates() -> None:
    active_workers = {
        0: SimpleNamespace(
            duplicate_of=None,
            termination_reason=None,
            termination_requested_at=None,
        ),
        1: SimpleNamespace(
            duplicate_of=0,
            termination_reason="dynamic_first_iteration_dedup",
            termination_requested_at=1.0,
        ),
        2: SimpleNamespace(
            duplicate_of=0,
            termination_reason=None,
            termination_requested_at=None,
        ),
    }

    assert _launch_blocking_worker_count(active_workers) == 2


def test_trace_only_collective_mode_keeps_all_gather_and_reduce_scatter_shape_safe() -> None:
    env = EmulatedDistributedEnvironment(
        logical_rank=0,
        logical_world_size=2,
        default_backend="gloo",
        collective_mode="trace_only",
    )
    group = env.init_process_group(backend="gloo")
    tensor = torch.tensor([5.0])
    outputs = [torch.full_like(tensor, 7.0), torch.full_like(tensor, 9.0)]

    group.all_gather(outputs, tensor)
    assert all(torch.equal(output, torch.zeros_like(output)) for output in outputs)

    group.reduce_scatter(tensor, outputs, op=dist.ReduceOp.SUM)
    assert torch.equal(tensor, torch.zeros_like(tensor))


def test_trace_only_collective_mode_keeps_all_to_all_shape_safe() -> None:
    env = EmulatedDistributedEnvironment(
        logical_rank=0,
        logical_world_size=2,
        default_backend="gloo",
        collective_mode="trace_only",
    )
    group = env.init_process_group(backend="gloo")
    inputs = [torch.full((2, 3), 5.0), torch.full((2, 3), 7.0)]
    outputs = [torch.full((2, 3), 9.0), torch.full((2, 3), 11.0)]

    group.all_to_all(outputs, inputs)

    assert all(torch.equal(output, torch.zeros_like(output)) for output in outputs)


def test_trace_only_collective_mode_keeps_all_to_all_single_shape_safe() -> None:
    env = EmulatedDistributedEnvironment(
        logical_rank=0,
        logical_world_size=2,
        default_backend="gloo",
        collective_mode="trace_only",
    )
    group = env.init_process_group(backend="gloo")
    input_tensor = torch.arange(8, dtype=torch.float32)
    output_tensor = torch.full_like(input_tensor, 7.0)

    group.all_to_all_single(output_tensor, input_tensor)

    assert torch.equal(output_tensor, torch.zeros_like(output_tensor))


def test_trace_only_collective_mode_keeps_recv_shape_safe() -> None:
    env = EmulatedDistributedEnvironment(
        logical_rank=0,
        logical_world_size=2,
        default_backend="gloo",
        collective_mode="trace_only",
    )
    group = env.init_process_group(backend="gloo")
    tensor = torch.full((2, 3), 5.0)

    group.recv(tensor, peer_group_rank=1)

    assert torch.equal(tensor, torch.zeros_like(tensor))


def test_trace_only_async_group_ops_return_waitable_handles() -> None:
    env = EmulatedDistributedEnvironment(
        logical_rank=0,
        logical_world_size=2,
        default_backend="gloo",
        collective_mode="trace_only",
    )
    group = env.init_process_group(backend="gloo")
    send_tensor = torch.full((2, 3), 5.0)
    recv_tensor = torch.full((2, 3), 7.0)

    send_req = group.send_async(send_tensor, peer_group_rank=1)
    recv_req = group.recv_async(recv_tensor, peer_group_rank=1)

    assert send_req.is_completed()
    assert recv_req.is_completed()
    assert torch.equal(send_req.wait(), send_tensor)
    assert torch.equal(recv_req.wait(), torch.zeros_like(recv_tensor))


def test_sitecustomize_bootstrap_exposes_sync_group_p2p_apis() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{repo_root / 'python'}:{env.get('PYTHONPATH', '')}"
    env["FLEXSIM_MAYA_EMULATED_DIST"] = "1"
    env["FLEXSIM_MAYA_LOGICAL_RANK"] = "0"
    env["FLEXSIM_MAYA_LOGICAL_WORLD_SIZE"] = "2"
    env["FLEXSIM_MAYA_COLLECTIVE_MODE"] = "trace_only"

    code = """
import json
import torch
import torch.distributed as dist

world = dist.init_process_group(backend="gloo", rank=0, world_size=2)
group = dist.new_group([0, 1], backend="nccl")
send_tensor = torch.ones(4)
recv_tensor = torch.ones(4)
dist.send(send_tensor, group=group, group_dst=1)
dist.recv(recv_tensor, group=group, group_src=1)
payload = {
    "world_backend": dist.get_backend(world),
    "group_backend": dist.get_backend(group),
    "recv_sum": float(recv_tensor.sum().item()),
}
print(json.dumps(payload))
dist.destroy_process_group()
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["world_backend"] == "gloo"
    assert payload["group_backend"] in {"gloo", "nccl"}
    assert payload["recv_sum"] == 0.0


def test_sitecustomize_bootstrap_persists_communicator_topology_sidecar(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    communicators_path = tmp_path / "rank_0.communicators.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{repo_root / 'python'}:{env.get('PYTHONPATH', '')}"
    env["FLEXSIM_MAYA_EMULATED_DIST"] = "1"
    env["FLEXSIM_MAYA_LOGICAL_RANK"] = "0"
    env["FLEXSIM_MAYA_LOGICAL_WORLD_SIZE"] = "4"
    env["FLEXSIM_MAYA_COMMUNICATORS_PATH"] = str(communicators_path)

    code = """
import torch.distributed as dist

world = dist.init_process_group(backend='gloo', rank=0, world_size=4)
dist.new_group([0, 2], backend='nccl')
dist.new_group([0, 1], backend='nccl')
dist.destroy_process_group()
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(communicators_path.read_text(encoding="utf-8"))
    communicators = payload["communicators"]
    memberships = {comm_id: tuple(record["members"]) for comm_id, record in communicators.items()}
    assert (0, 1, 2, 3) in memberships.values()
    assert (0, 2) in memberships.values()
    assert (0, 1) in memberships.values()


def test_trim_trace_file_to_ts_window_keeps_only_step_events(tmp_path: Path) -> None:
    trace_path = tmp_path / "rank_0.jsonl"
    trace_path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in [
                {"ts": 5, "pid": 0, "tid": 0, "mod": "libcudart.so.12", "api": "cudaGetDevice", "type": "other"},
                {
                    "ts": 10,
                    "pid": 0,
                    "tid": 0,
                    "mod": "libcudart.so.12",
                    "api": "cudaLaunchKernel",
                    "type": "kernel_launch",
                },
                {
                    "ts": 15,
                    "pid": 0,
                    "tid": 0,
                    "mod": "libnccl.so.2",
                    "api": "ncclAllReduce",
                    "type": "nccl_collective",
                },
                {"ts": 25, "pid": 0, "tid": 0, "mod": "libcudart.so.12", "api": "cudaFree", "type": "other"},
            ]
        ),
        encoding="utf-8",
    )

    summary = _trim_trace_file_to_ts_window(trace_path, start_ts=10, end_ts=15)

    kept = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert summary == {
        "start_ts": 10,
        "end_ts": 15,
        "total_events": 4,
        "kept_events": 2,
    }
    assert [record["api"] for record in kept] == ["cudaLaunchKernel", "ncclAllReduce"]


def test_sitecustomize_bootstrap_exposes_async_p2p_apis() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{repo_root / 'python'}:{env.get('PYTHONPATH', '')}"
    env["FLEXSIM_MAYA_EMULATED_DIST"] = "1"
    env["FLEXSIM_MAYA_LOGICAL_RANK"] = "0"
    env["FLEXSIM_MAYA_LOGICAL_WORLD_SIZE"] = "2"
    env["FLEXSIM_MAYA_COLLECTIVE_MODE"] = "trace_only"

    code = """
import json
import torch
import torch.distributed as dist

world = dist.init_process_group(backend="gloo", rank=0, world_size=2)
group = dist.new_group([0, 1], backend="nccl")
send_tensor = torch.ones(4)
recv_tensor = torch.ones(4)
send_req = dist.isend(send_tensor, dst=1, group=group)
recv_req = dist.irecv(recv_tensor, src=1, group=group)
for req in (send_req, recv_req):
    assert req is not None
    req.wait()
ops = [
    dist.P2POp(dist.isend, send_tensor, 1, group=group),
    dist.P2POp(dist.irecv, recv_tensor, 1, group=group),
]
batch_reqs = dist.batch_isend_irecv(ops)
for req in batch_reqs:
    req.wait()
payload = {
    "world_backend": dist.get_backend(world),
    "group_backend": dist.get_backend(group),
    "batch_request_count": len(batch_reqs),
    "recv_sum": float(recv_tensor.sum().item()),
}
print(json.dumps(payload))
dist.destroy_process_group()
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["world_backend"] == "gloo"
    assert payload["group_backend"] in {"gloo", "nccl"}
    assert payload["batch_request_count"] == 2
    assert payload["recv_sum"] == 0.0
