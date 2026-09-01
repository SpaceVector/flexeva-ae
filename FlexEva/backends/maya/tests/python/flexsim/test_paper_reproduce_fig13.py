from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest


def _load_reproduce_fig13_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "paper" / "maya_lite" / "reproduce_fig13.py"
    spec = spec_from_file_location("paper_maya_lite_reproduce_fig13", module_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reproduce_fig13_defaults_enable_observed_semantic_wrapper_durations():
    module = _load_reproduce_fig13_module()
    parser = module._build_parser()

    args = parser.parse_args(["dummy-trace-dir"])

    assert args.use_observed_semantic_wrapper_durations is True


def test_reproduce_fig13_rejects_old_observed_semantic_wrapper_toggle():
    module = _load_reproduce_fig13_module()
    parser = module._build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["dummy-trace-dir", "--disable-observed-semantic-wrapper-durations"]
        )


def test_reproduce_fig13_merge_manifest_preserves_dynamic_capture_groups(tmp_path: Path) -> None:
    module = _load_reproduce_fig13_module()
    manifest = tmp_path / "capture_manifest.json"
    manifest.write_text(
        """
{
  "original_world_size": 16,
  "profiled_rank_groups": {
    "0": [0],
    "1": [1],
    "2": [2],
    "3": [3],
    "4": [4],
    "5": [5],
    "6": [6],
    "7": [7],
    "8": [8],
    "9": [9],
    "10": [10],
    "11": [11],
    "12": [12],
    "13": [13],
    "14": [14],
    "15": [15]
  },
  "launched_workers": [
    {"representative_rank": 0, "dynamic_duplicate_of": 8},
    {"representative_rank": 1},
    {"representative_rank": 2},
    {"representative_rank": 3},
    {"representative_rank": 4},
    {"representative_rank": 5},
    {"representative_rank": 6},
    {"representative_rank": 7},
    {"representative_rank": 8},
    {"representative_rank": 9, "dynamic_duplicate_of": 1},
    {"representative_rank": 10, "dynamic_duplicate_of": 2},
    {"representative_rank": 11, "dynamic_duplicate_of": 3},
    {"representative_rank": 12, "dynamic_duplicate_of": 4},
    {"representative_rank": 13, "dynamic_duplicate_of": 5},
    {"representative_rank": 14, "dynamic_duplicate_of": 6},
    {"representative_rank": 15, "dynamic_duplicate_of": 7}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    expected_identity_groups = {rank: (rank,) for rank in range(16)}
    module._merge_capture_manifest(
        tmp_path,
        logical_world_size=16,
        profiled_rank_groups=expected_identity_groups,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["profiled_rank_groups"] == {
        "1": [1, 9],
        "2": [2, 10],
        "3": [3, 11],
        "4": [4, 12],
        "5": [5, 13],
        "6": [6, 14],
        "7": [7, 15],
        "8": [8, 0],
    }
    assert payload["expected_profiled_rank_groups"]["0"] == [0]


def test_reproduce_fig13_defaults_to_strict_paper_timing_contract() -> None:
    module = _load_reproduce_fig13_module()
    parser = module._build_parser()

    args = parser.parse_args(["dummy-trace-dir"])

    assert args.allow_uncalibrated_estimator is False
    assert args.allow_paper_path_timing_violations is False
    assert args.allow_heuristic_kernel_launch_fallback is False
    assert args.allow_weak_runtime_fallback is False


def test_reproduce_fig13_rejects_old_timing_fallback_toggles() -> None:
    module = _load_reproduce_fig13_module()
    parser = module._build_parser()

    for flag in (
        "--allow-uncalibrated-estimator",
        "--allow-heuristic-kernel-launch-fallback",
        "--allow-weak-runtime-fallback",
        "--allow-paper-path-timing-violations",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(["dummy-trace-dir", flag])


def test_reproduce_fig13_paper_timing_contract_summary_flags_forbidden_sources() -> None:
    module = _load_reproduce_fig13_module()

    summary = module._paper_timing_contract_summary(
        {
            "estimator_calibrated": True,
            "estimator_mode_resolved": "hybrid",
            "annotation_diagnostics": {
                "duration_source_counts": {
                    "observed_host_delay": 3,
                    "heuristic_kernel_launch": 2,
                    "estimator_api_stats": 5,
                },
                "strict_runtime_signal_observed_wrapper_duration_dispatch_only_count": 0,
            },
        }
    )

    assert summary["paper_timing_contract_passed"] is False
    assert summary["forbidden_duration_source_counts"] == {
        "heuristic_kernel_launch": 2,
    }
    assert summary["forbidden_duration_source_total"] == 2


def test_reproduce_fig13_paper_timing_contract_summary_flags_dispatch_only_wrapper_runtime():
    module = _load_reproduce_fig13_module()

    summary = module._paper_timing_contract_summary(
        {
            "estimator_calibrated": True,
            "estimator_mode_resolved": "hybrid",
            "annotation_diagnostics": {
                "duration_source_counts": {
                    "observed_wrapper_duration": 4,
                    "estimator_api_stats": 2,
                },
                "strict_runtime_signal_observed_wrapper_duration_count": 2,
                "strict_runtime_signal_observed_wrapper_duration_direct_runtime_count": 1,
                "strict_runtime_signal_observed_wrapper_duration_dispatch_only_count": 1,
            },
        }
    )

    assert summary["paper_timing_contract_passed"] is False
    assert (
        summary["strict_runtime_signal_observed_wrapper_duration_dispatch_only_count"]
        == 1
    )


def test_reproduce_fig13_rejects_paper_timing_contract_violations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    module = _load_reproduce_fig13_module()

    class _Estimator:
        def provider_names(self):
            return ("gpu_estimator_xgboost",)

        def provider_diagnostics(self):
            return {}

        def is_calibrated(self):
            return True

        def operator_family_summary(self, limit=10):
            del limit
            return []

        def kernel_launch_metadata_summary(self):
            return {}

        def transparent_profiling_summary(self):
            return {}

        def provider_coverage_summary(self, provider_name, limit=10):
            del provider_name, limit
            return {}

    class _Benchmark:
        def to_dict(self):
            return {
                "critical_path_us": 1.0,
                "annotation_diagnostics": {
                    "duration_source_counts": {"heuristic_kernel_launch": 1},
                    "strict_runtime_signal_observed_wrapper_duration_dispatch_only_count": 0,
                },
            }

    monkeypatch.setattr(module, "_resolve_estimator", lambda args: (_Estimator(), "hybrid"))
    monkeypatch.setattr(module, "load_trace_directory", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "benchmark_trace_bundle", lambda *args, **kwargs: _Benchmark())
    monkeypatch.setattr(module, "load_capture_manifest", lambda trace_dir: None)

    output_path = tmp_path / "out.json"
    with pytest.raises(SystemExit, match="Paper timing contract violated"):
        module.main(["dummy-trace-dir", "--model", "dummy-model.json", "--output", str(output_path)])

    payload = output_path.read_text(encoding="utf-8")
    assert "heuristic_kernel_launch" in payload


def test_reproduce_fig13_loads_traces_with_strict_step_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    module = _load_reproduce_fig13_module()
    seen: dict[str, object] = {}

    def fake_resolve_estimator(args):
        class _Estimator:
            def provider_names(self):
                return []

            def provider_diagnostics(self):
                return {}

            def is_calibrated(self):
                return True

            def operator_family_summary(self, limit=10):
                del limit
                return {}

            def kernel_launch_metadata_summary(self):
                return {}

            def transparent_profiling_summary(self):
                return {}

            def provider_coverage_summary(self, provider_name, limit=10):
                del provider_name, limit
                return {}

        return _Estimator(), "loaded_model"

    def fake_load_trace_directory(trace_dir, *, max_events_per_rank=None, trace_window="auto", **kwargs):
        del trace_dir, max_events_per_rank, kwargs
        seen["trace_window"] = trace_window
        return object()

    class _Benchmark:
        def to_dict(self):
            return {"critical_path_us": 1.0}

    monkeypatch.setattr(module, "_resolve_estimator", fake_resolve_estimator)
    monkeypatch.setattr(module, "load_trace_directory", fake_load_trace_directory)
    monkeypatch.setattr(module, "benchmark_trace_bundle", lambda *args, **kwargs: _Benchmark())
    monkeypatch.setattr(module, "load_capture_manifest", lambda trace_dir: None)

    output_path = tmp_path / "out.json"
    rc = module.main(["dummy-trace-dir", "--model", "dummy-model.json", "--output", str(output_path)])

    assert rc == 0
    assert seen["trace_window"] == "step"


def test_reproduce_fig13_keeps_selective_unique_worker_replay_compact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_reproduce_fig13_module()
    seen: dict[str, object] = {}

    def fake_resolve_estimator(args):
        class _Estimator:
            def provider_names(self):
                return []

            def provider_diagnostics(self):
                return {}

            def is_calibrated(self):
                return True

            def operator_family_summary(self, limit=10):
                del limit
                return {}

            def kernel_launch_metadata_summary(self):
                return {}

            def transparent_profiling_summary(self):
                return {}

            def provider_coverage_summary(self, provider_name, limit=10):
                del provider_name, limit
                return {}

        return _Estimator(), "loaded_model"

    class _Benchmark:
        def to_dict(self):
            return {"critical_path_us": 1.0}

    monkeypatch.setattr(module, "_resolve_estimator", fake_resolve_estimator)
    monkeypatch.setattr(module, "load_trace_directory", lambda *args, **kwargs: object())

    def fake_benchmark_trace_bundle(*args, **kwargs):
        del args
        seen["materialize_logical_ranks"] = kwargs.get("materialize_logical_ranks")
        seen["expand_profiled_rank_groups"] = kwargs.get("expand_profiled_rank_groups")
        return _Benchmark()

    monkeypatch.setattr(module, "benchmark_trace_bundle", fake_benchmark_trace_bundle)
    monkeypatch.setattr(module, "load_capture_manifest", lambda trace_dir: None)

    output_path = tmp_path / "out.json"
    rc = module.main(["dummy-trace-dir", "--model", "dummy-model.json", "--output", str(output_path)])

    assert rc == 0
    assert seen["materialize_logical_ranks"] is False
    assert seen["expand_profiled_rank_groups"] is True


def test_reproduce_fig13_helper_thread_contract_summary_requires_completed_for_synthetic_host_timing():
    module = _load_reproduce_fig13_module()

    summary = module._helper_thread_augmentation_contract_summary(
        {
            "host_timing_mode_resolved": "trace",
            "helper_thread_augmentation": {
                "expected": True,
                "status": "missing_summary_dir",
                "embedded_in_emulator_artifact": False,
                "summary_dir": None,
                "total_injected_events": 0,
            },
        }
    )

    assert summary["expected"] is True
    assert summary["status"] == "missing_summary_dir"
    assert summary["helper_thread_augmentation_contract_passed"] is False


def test_reproduce_fig13_rejects_missing_helper_thread_augmentation_contract_for_synthetic_host_timing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    module = _load_reproduce_fig13_module()

    class _Estimator:
        def provider_names(self):
            return []

        def provider_diagnostics(self):
            return {}

        def is_calibrated(self):
            return True

        def operator_family_summary(self, limit=10):
            del limit
            return {}

        def kernel_launch_metadata_summary(self):
            return {}

        def transparent_profiling_summary(self):
            return {}

        def provider_coverage_summary(self, provider_name, limit=10):
            del provider_name, limit
            return {}

    class _Benchmark:
        def to_dict(self):
            return {"critical_path_us": 1.0, "annotation_diagnostics": {"duration_source_counts": {}}}

    monkeypatch.setattr(module, "_resolve_estimator", lambda args: (_Estimator(), "loaded_model"))
    monkeypatch.setattr(module, "load_trace_directory", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "benchmark_trace_bundle", lambda *args, **kwargs: _Benchmark())
    monkeypatch.setattr(
        module,
        "load_capture_manifest",
        lambda trace_dir: {"host_timing_mode_resolved": "trace"},
    )

    output_path = tmp_path / "out.json"
    with pytest.raises(SystemExit, match="Helper-thread augmentation contract violated"):
        module.main(["dummy-trace-dir", "--model", "dummy-model.json", "--output", str(output_path)])

    payload = output_path.read_text(encoding="utf-8")
    assert "helper_thread_augmentation_contract" in payload


def test_reproduce_fig13_injects_host_timing_summary_dir_into_capture_command() -> None:
    module = _load_reproduce_fig13_module()
    command = [
        "/usr/bin/python",
        "/repo/python/flexsim/maya_lite/capture_emulated.py",
        "--output-dir",
        "{trace_dir}",
        "--logical-world-size",
        "8",
        "/repo/tests/workloads/fake_cuda/maya_fig13_megatron.py",
        "--steps",
        "1",
    ]

    updated = module._inject_host_timing_summary_dir_into_capture_command(
        command,
        Path("/tmp/host_summaries"),
    )

    assert "--host-timing-summary-dir" in updated
    summary_index = updated.index("--host-timing-summary-dir")
    assert updated[summary_index + 1] == str(Path("/tmp/host_summaries").resolve())
    workload_index = updated.index("/repo/tests/workloads/fake_cuda/maya_fig13_megatron.py")
    assert summary_index < workload_index


def test_reproduce_fig13_stage_share_contract_ready_for_reuse_model_replay() -> None:
    module = _load_reproduce_fig13_module()

    summary = module._paper_stage_share_contract_summary(
        {
            "post_worker_finalize_included_in_capture_elapsed": True,
            "stage_timing": {
                "emulator_stage_basis": "capture_elapsed_seconds",
                "collator_stage_basis": "collate_only",
                "predictor_stage_basis": "runtime_annotation_wall_time",
                "predictor_seconds": 3.0,
                "predictor_runtime_estimation_seconds": 2.0,
                "predictor_total_annotation_seconds": 3.0,
                "predictor_pass_through_annotation_seconds": 1.0,
                "simulator_stage_basis": "replay_only",
                "trace_load_seconds": 1.5,
                "trace_load_included_in_collator": False,
            }
        },
        mode="reuse_model_replay",
        estimator_prepare_seconds=2.0,
        estimator_fit_seconds=0.0,
        estimator_save_seconds=0.0,
    )

    assert summary["paper_stage_share_mode"] == "reuse_model_replay"
    assert summary["paper_stage_share_ready"] is True
    assert summary["emulator_trace_collection_ready"] is True
    assert summary["predictor_accounting_ready"] is True
    assert summary["estimator_fit_included_in_predictor"] is False


def test_reproduce_fig13_stage_share_contract_not_ready_for_fit_and_replay() -> None:
    module = _load_reproduce_fig13_module()

    summary = module._paper_stage_share_contract_summary(
        {
            "post_worker_finalize_included_in_capture_elapsed": True,
            "stage_timing": {
                "emulator_stage_basis": "capture_elapsed_seconds",
                "collator_stage_basis": "collate_only",
                "predictor_stage_basis": "runtime_annotation_wall_time",
                "predictor_seconds": 3.0,
                "predictor_runtime_estimation_seconds": 2.0,
                "predictor_total_annotation_seconds": 3.0,
                "predictor_pass_through_annotation_seconds": 1.0,
                "simulator_stage_basis": "replay_only",
                "trace_load_seconds": 1.5,
                "trace_load_included_in_collator": False,
            }
        },
        mode="fit_and_replay",
        estimator_prepare_seconds=3.0,
        estimator_fit_seconds=3.0,
        estimator_save_seconds=0.5,
    )

    assert summary["paper_stage_share_mode"] == "fit_and_replay"
    assert summary["paper_stage_share_ready"] is False
    assert summary["estimator_fit_seconds"] == 3.0
    assert summary["estimator_fit_included_in_predictor"] is False


def test_reproduce_fig13_host_timing_line_contract_summary_for_measure_mode() -> None:
    module = _load_reproduce_fig13_module()

    summary = module._host_timing_line_contract_summary(
        {
            "host_timing_mode_resolved": "measure",
            "host_timing_dispatch_scope_resolved": "host_machine",
            "host_timing_schedule_surface_resolved": "semantic",
            "host_timing_paper_alignment_line": "direct_emulation_measured_host_overhead",
            "host_timing_line_family": "direct_wallclock",
            "host_timing_line_contract_version": "phase4_v1",
            "host_timing_profile_backed": False,
            "host_timing_paper_alignment_ready": True,
        }
    )

    assert summary["host_timing_line_contract_passed"] is True
    assert summary["host_timing_line_is_canonical"] is True


def test_reproduce_fig13_host_timing_line_contract_summary_flags_inconsistent_line() -> None:
    module = _load_reproduce_fig13_module()

    summary = module._host_timing_line_contract_summary(
        {
            "host_timing_mode_resolved": "trace",
            "host_timing_dispatch_scope_resolved": "thread",
            "host_timing_schedule_surface_resolved": "supported",
            "host_timing_paper_alignment_line": "direct_emulation_measured_host_overhead",
            "host_timing_line_family": "direct_wallclock",
            "host_timing_line_contract_version": "phase4_v1",
            "host_timing_profile_backed": True,
            "host_timing_paper_alignment_ready": True,
        }
    )

    assert summary["host_timing_line_contract_passed"] is False
    assert summary["host_timing_line_is_canonical"] is False


def test_reproduce_fig13_rejects_missing_host_timing_line_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_reproduce_fig13_module()

    class _Estimator:
        def provider_names(self):
            return []

        def provider_diagnostics(self):
            return {}

        def is_calibrated(self):
            return True

        def operator_family_summary(self, limit=10):
            del limit
            return {}

        def kernel_launch_metadata_summary(self):
            return {}

        def transparent_profiling_summary(self):
            return {}

        def provider_coverage_summary(self, provider_name, limit=10):
            del provider_name, limit
            return {}

    monkeypatch.setattr(module, "_resolve_estimator", lambda args: (_Estimator(), "loaded_model"))
    output_path = tmp_path / "out.json"

    with pytest.raises(SystemExit, match="Host-timing line contract violated"):
        module.main(
            [
                "dummy-trace-dir",
                "--fit-traces",
                "dummy-trace-dir",
                "--fit-only",
                "--output",
                str(output_path),
            ]
        )

    payload = output_path.read_text(encoding="utf-8")
    assert "host_timing_line_contract" in payload


def test_reproduce_fig13_network_model_contract_summary_accepts_in_tree_group_stable_topology_inputs() -> None:
    module = _load_reproduce_fig13_module()

    summary = module._network_model_contract_summary(
        {
            "annotation_diagnostics": {
                "collective_group_count": 3,
                "collective_group_with_duration_metadata_count": 3,
                "collective_group_duration_basis_counts": {
                    "group_provider:trace_signature_stats": 3,
                },
            }
        },
        capture_manifest={
            "communicators": {
                "comm-0": {"members": [0, 1, 2, 3]},
                "comm-1": {"members": [0, 1]},
            }
        },
    )

    assert summary["network_model_backend"] == "in_tree"
    assert summary["astra_sim_required"] is False
    assert summary["pluggable_model_boundary"] is True
    assert summary["topology_aware_inputs_present"] is True
    assert summary["group_stable_collective_timing"] is True
    assert summary["group_runtime_estimator_present"] is True
    assert summary["estimator_group_basis_count"] == 3
    assert summary["fallback_group_basis_count"] == 0
    assert summary["network_model_contract_passed"] is True


def test_reproduce_fig13_network_model_contract_summary_flags_missing_topology_for_collectives() -> None:
    module = _load_reproduce_fig13_module()

    summary = module._network_model_contract_summary(
        {
            "annotation_diagnostics": {
                "collective_group_count": 2,
                "collective_group_with_duration_metadata_count": 2,
                "collective_group_duration_basis_counts": {
                    "max_member_duration": 2,
                },
            }
        },
        capture_manifest={"communicators": {}},
    )

    assert summary["topology_aware_inputs_required"] is True
    assert summary["topology_aware_inputs_present"] is False
    assert summary["group_runtime_estimator_present"] is False
    assert "communicators" in summary["missing_fields"]
    assert "collective_group_runtime_estimator" in summary["missing_fields"]
    assert summary["network_model_contract_passed"] is False
