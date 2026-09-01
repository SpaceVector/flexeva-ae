from pathlib import Path

import pytest

from flexsim.maya_lite import (
    AnnotatedEvent,
    AnnotatedTrace,
    collate_trace_bundle,
    dedup_identical_rank_traces,
    load_trace_directory,
    replay_annotated_trace,
)
from flexsim.maya_lite.filters import is_collective_api


_HOST_CONTROL_DEFAULT_OFF_DIAGNOSTIC_FIELDS = {
    "host_control_envelope_counterpart_schema_version",
    "host_control_envelope_counterpart_opt_in_flag",
    "host_control_envelope_counterpart_key",
    "hostdelay_counterpart_key",
    "host_control_envelope_replay_interval_id",
    "host_control_visibility_schema_version",
    "host_control_visibility_opt_in_flag",
    "selected_occurrence_id",
    "paper_valid_window_id",
    "actual_paper_valid_window_id",
    "actual_boundary_family",
    "actual_rank",
    "actual_raw_prev_event_id",
    "actual_raw_current_event_id",
    "counterpart_join_key",
    "counterpart_join_method",
    "counterpart_join_confidence",
    "split_sum_check_status",
    "affected_interval_id",
    "interval_kind",
    "count_once_status",
    "safe_to_use_as_repair_evidence",
    "safe_to_use_as_subtraction_delta",
    "launch_neighborhood_equivalence_schema_version",
    "launch_neighborhood_equivalence_opt_in_flag",
    "launch_neighborhood_occurrence_id",
    "launch_neighborhood_normalized_signature",
    "launch_neighborhood_boundary_exclusion_reasons",
}
_HOSTDELAY_OCCURRENCE_METADATA_FIELDS = {
    "hostdelay_occurrence_metadata_schema_version",
    "hostdelay_occurrence_metadata_opt_in_flag",
    "stable_hostdelay_occurrence_id",
    "hostdelay_occurrence_interval_start_ts_us",
    "hostdelay_occurrence_interval_end_ts_us",
    "hostdelay_occurrence_duration_us",
    "hostdelay_occurrence_raw_predecessor_event_id",
    "hostdelay_occurrence_raw_successor_event_id",
    "hostdelay_occurrence_semantic_predecessor_event_id",
    "hostdelay_occurrence_semantic_successor_event_id",
    "hostdelay_occurrence_raw_boundary_family",
    "hostdelay_occurrence_semantic_boundary_family",
    "hostdelay_occurrence_count_once_group_id",
    "hostdelay_occurrence_cuda_event_wait_map_safety_status",
    "hostdelay_occurrence_repair_ready",
    "hostdelay_occurrence_safe_to_use_as_subtraction_delta",
    "safe_delta_us",
}
_HOSTDELAY_OCCURRENCE_JOIN_METADATA_FIELDS = {
    "hostdelay_occurrence_join_metadata_schema_version",
    "hostdelay_occurrence_join_metadata_opt_in_flag",
    "hostdelay_occurrence_boundary_origin_join_key",
    "hostdelay_occurrence_boundary_origin_status",
    "hostdelay_occurrence_boundary_visibility_join_key",
    "hostdelay_occurrence_boundary_visibility_status",
    "hostdelay_occurrence_count_once_join_key",
    "hostdelay_occurrence_nonoverlap_join_key",
    "hostdelay_occurrence_cuda_event_waitmap_join_key",
    "hostdelay_occurrence_collective_waitmap_join_key",
    "hostdelay_occurrence_join_repair_ready",
    "hostdelay_occurrence_join_safe_to_use_as_subtraction_delta",
}
_COLLECTIVE_EVENT_POLLING_METADATA_FIELDS = {
    "collective_event_polling_metadata_schema_version",
    "collective_event_polling_metadata_opt_in_flag",
    "collective_event_polling_occurrence_join_key",
    "collective_event_polling_raw_boundary_family",
    "collective_event_polling_semantic_boundary_family",
    "collective_event_polling_target_family",
    "collective_event_polling_collective_api",
    "collective_event_polling_collective_group_id",
    "collective_event_polling_cuda_event_handle",
    "collective_event_polling_boundary_origin_status",
    "collective_event_polling_boundary_visibility_status",
    "collective_event_polling_wait_map_release_status",
    "collective_event_polling_count_once_status",
    "collective_event_polling_nonoverlap_status",
    "collective_event_polling_repair_ready",
    "collective_event_polling_safe_to_use_as_repair_evidence",
    "collective_event_polling_safe_to_use_as_subtraction_delta",
    "collective_event_polling_safe_delta_us",
}
_EVENT_POLLING_BOUNDARY_METADATA_FIELDS = {
    "event_polling_boundary_metadata_schema_version",
    "event_polling_boundary_metadata_opt_in_flag",
    "event_polling_boundary_occurrence_id",
    "event_polling_boundary_raw_family",
    "event_polling_boundary_semantic_family",
    "event_polling_boundary_target_class",
    "event_polling_boundary_polling_class",
    "event_polling_boundary_origin_kind",
    "event_polling_boundary_origin_status",
    "event_polling_boundary_visibility_kind",
    "event_polling_boundary_visibility_status",
    "event_polling_boundary_paper_visibility_class",
    "event_polling_boundary_candidate_control_plane_subregion_status",
    "event_polling_boundary_candidate_instrumentation_only_status",
    "event_polling_boundary_already_modeled_replay_waitmap_status",
    "event_polling_boundary_count_once_status",
    "event_polling_boundary_nonoverlap_status",
    "event_polling_boundary_wait_map_safety_status",
    "event_polling_boundary_repair_ready",
    "event_polling_boundary_safe_to_use_as_repair_evidence",
    "event_polling_boundary_safe_to_use_as_subtraction_delta",
    "event_polling_boundary_safe_delta_us",
}
_BOUNDARY_ORIGIN_SUBREGION_METADATA_FIELDS = {
    "boundary_origin_subregion_metadata_schema_version",
    "boundary_origin_subregion_metadata_opt_in_flag",
    "boundary_origin_subregion_occurrence_id",
    "boundary_origin_subregion_id",
    "boundary_origin_subregion_kind",
    "boundary_origin_subregion_candidate_role",
    "boundary_origin_subregion_raw_family",
    "boundary_origin_subregion_semantic_family",
    "boundary_origin_subregion_raw_semantic_pair",
    "boundary_origin_subregion_origin_status",
    "boundary_origin_subregion_visibility_status",
    "boundary_origin_subregion_paper_visibility_class",
    "boundary_origin_subregion_strict_extent_status",
    "boundary_origin_subregion_strict_proof_status",
    "boundary_origin_subregion_count_once_status",
    "boundary_origin_subregion_nonoverlap_status",
    "boundary_origin_subregion_wait_map_safety_status",
    "boundary_origin_subregion_fresh16_fresh8_join_key_status",
    "boundary_origin_subregion_repair_ready",
    "boundary_origin_subregion_safe_to_use_as_repair_evidence",
    "boundary_origin_subregion_safe_to_use_as_subtraction_delta",
    "boundary_origin_subregion_safe_delta_us",
}
_STRICT_SUBREGION_EXTENT_METADATA_FIELDS = {
    "strict_subregion_extent_metadata_schema_version",
    "strict_subregion_extent_metadata_opt_in_flag",
    "strict_subregion_extent_occurrence_id",
    "strict_subregion_extent_boundary_id",
    "strict_subregion_extent_raw_family",
    "strict_subregion_extent_semantic_family",
    "strict_subregion_extent_raw_semantic_pair",
    "strict_subregion_extent_target_family_class",
    "strict_subregion_extent_candidate_subregion_kind",
    "strict_subregion_extent_candidate_subregion_role",
    "strict_subregion_extent_start_ts_us",
    "strict_subregion_extent_end_ts_us",
    "strict_subregion_extent_duration_us_context_only",
    "strict_subregion_extent_timestamp_basis",
    "strict_subregion_extent_timestamp_source_kind",
    "strict_subregion_extent_source_proof_status",
    "strict_subregion_extent_origin_status",
    "strict_subregion_extent_visibility_status",
    "strict_subregion_extent_paper_visibility_class",
    "strict_subregion_extent_count_once_status",
    "strict_subregion_extent_nonoverlap_status",
    "strict_subregion_extent_cuda_event_wait_map_safety_status",
    "strict_subregion_extent_collective_wait_map_safety_status",
    "strict_subregion_extent_fresh16_fresh8_join_key_status",
    "strict_subregion_extent_repair_ready",
    "strict_subregion_extent_safe_to_use_as_repair_evidence",
    "strict_subregion_extent_safe_to_use_as_subtraction_delta",
    "strict_subregion_extent_safe_delta_us",
    "strict_subregion_extent_runtime_or_endpoint_substitution_used",
    "strict_subregion_extent_hostdelay_shortening_used",
    "strict_subregion_extent_rank_workload_special_case_used",
}


def _assert_launch_config_metadata(
    event,
    *,
    adjacent_gap_us: int,
    contribution: str = "excluded_from_pending_host_gap",
    normalization_enabled: bool = True,
    normalization_status: str | None = None,
    raw_event_id: str = "r0:e0",
    raw_ts_us: int = 100,
    raw_end_ts_us: int | None = None,
) -> None:
    if raw_end_ts_us is None:
        raw_end_ts_us = raw_ts_us + 4
    assert event.api == "cudaLaunchKernel"
    assert event.extras["launch_config_metadata_basis"] == "internal_launch_config_metadata"
    assert event.extras["launch_config_metadata_reason"] == (
        "__cudaPopCallConfiguration carries internal CUDA launch configuration "
        "metadata for the following cudaLaunchKernel"
    )
    assert event.extras["launch_config_raw_event_id"] == raw_event_id
    assert event.extras["launch_config_raw_api"] == "__cudaPopCallConfiguration"
    assert event.extras["launch_config_raw_ts_us"] == raw_ts_us
    assert event.extras["launch_config_raw_end_ts_us"] == raw_end_ts_us
    assert event.extras["launch_config_raw_end_ts_source"] == "raw_end_ts"
    assert event.extras["launch_config_raw_host_duration_us"] == 4.0
    assert event.extras["launch_config_raw_duration_us"] == 4
    assert event.extras["launch_config_adjacent_host_gap_us"] == adjacent_gap_us
    assert event.extras["launch_config_adjacent_host_gap_contribution"] == contribution
    assert event.extras["launch_config_hostdelay_normalization_enabled"] is normalization_enabled
    assert event.extras["launch_config_hostdelay_normalization_env_flags"] == [
        "MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION",
        "FLEXSIM_MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION",
    ]
    assert event.extras["launch_config_hostdelay_normalization_disable_env_flags"] == [
        "MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION",
        "FLEXSIM_MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION",
    ]
    if normalization_enabled:
        expected_status = normalization_status or "enabled_default_excluded_from_hostdelay"
        assert event.extras["launch_config_hostdelay_normalization_status"] == expected_status
        assert event.extras["hostdelay_normalization_basis"] == "internal_launch_config_metadata"
    else:
        expected_status = (
            normalization_status
            or "disabled_explicit_disable_control_hostdelay_preserved"
        )
        assert event.extras["launch_config_hostdelay_normalization_status"] == expected_status
        assert "hostdelay_normalization_basis" not in event.extras
    assert event.extras["launch_config_raw_extras"]["end_ts"] == raw_end_ts_us
    assert event.extras["launch_config_raw_extras"]["host_duration_us"] == 4.0


def _assert_cublas_context_query_suffix_fold(
    event,
    *,
    folded_count: int,
    suppressed_host_gap_us: int,
    preserved_pre_suffix_gap_us: int,
) -> None:
    assert event.api == "cublasSetStream_v2"
    assert event.extras["cublas_set_stream_context_query_suffix_fold_basis"] == (
        "cudaGetDevice_context_query_suffix_before_cublasSetStream_v2"
    )
    assert event.extras["cublas_set_stream_context_query_suffix_fold_status"] == (
        "applied_default_paper_aligned_metadata_fold"
    )
    assert event.extras["cublas_set_stream_context_query_suffix_folded_count"] == folded_count
    assert event.extras["cublas_set_stream_context_query_suffix_folded_apis"] == [
        "cudaGetDevice"
    ] * folded_count
    assert (
        event.extras["cublas_set_stream_context_query_suffix_suppressed_host_gap_us"]
        == suppressed_host_gap_us
    )
    assert (
        event.extras["cublas_set_stream_context_query_suffix_preserved_pre_suffix_gap_us"]
        == preserved_pre_suffix_gap_us
    )
    assert event.extras["cublas_set_stream_context_query_suffix_host_gap_contribution"] == (
        "only_internal_suffix_gaps_excluded_from_pending_hostdelay"
    )
    assert len(event.extras["cublas_set_stream_context_query_suffix_folded_rows"]) == folded_count


def _assert_cuda_get_device_context_query_run_fold(
    event,
    *,
    event_count: int,
    internal_gap_us: int,
    terminal_gap_us: int,
    preserved_pre_run_gap_us: int,
) -> None:
    assert event.extras["cuda_get_device_context_query_run_fold_basis"] == (
        "consecutive_cudaGetDevice_context_query_run"
    )
    assert event.extras["cuda_get_device_context_query_run_fold_status"] == (
        "applied_default_paper_aligned_internal_gap_fold"
    )
    assert event.extras["cuda_get_device_context_query_run_event_count"] == event_count
    assert (
        event.extras["cuda_get_device_context_query_run_internal_gap_us"]
        == internal_gap_us
    )
    assert (
        event.extras["cuda_get_device_context_query_run_terminal_gap_us"]
        == terminal_gap_us
    )
    assert (
        event.extras["cuda_get_device_context_query_run_suppressed_host_gap_us"]
        == internal_gap_us
    )
    assert (
        event.extras["cuda_get_device_context_query_run_preserved_pre_run_gap_us"]
        == preserved_pre_run_gap_us
    )
    assert event.extras["cuda_get_device_context_query_run_host_gap_contribution"] == (
        "only_internal_run_gaps_excluded_from_pending_hostdelay"
    )
    assert len(event.extras["cuda_get_device_context_query_run_rows"]) == event_count


def _assert_semantic_predecessor_context_query_suffix_fold(
    event,
    *,
    target_api: str,
    folded_count: int,
    suppressed_host_gap_us: int,
    preserved_pre_suffix_gap_us: int,
) -> None:
    assert event.extras[
        "hostdelay_semantic_predecessor_control_query_suffix_fold_basis"
    ] == f"cudaGetDevice_context_query_suffix_before_{target_api}"
    assert event.extras[
        "hostdelay_semantic_predecessor_control_query_suffix_fold_status"
    ] == "applied_default_paper_aligned_metadata_fold"
    assert event.extras[
        "hostdelay_semantic_predecessor_control_query_suffix_target_api"
    ] == target_api
    assert event.extras[
        "hostdelay_semantic_predecessor_control_query_suffix_folded_count"
    ] == folded_count
    assert event.extras[
        "hostdelay_semantic_predecessor_control_query_suffix_folded_apis"
    ] == ["cudaGetDevice"] * folded_count
    assert (
        event.extras[
            "hostdelay_semantic_predecessor_control_query_suffix_suppressed_host_gap_us"
        ]
        == suppressed_host_gap_us
    )
    assert (
        event.extras[
            "hostdelay_semantic_predecessor_control_query_suffix_preserved_pre_suffix_gap_us"
        ]
        == preserved_pre_suffix_gap_us
    )
    assert event.extras[
        "hostdelay_semantic_predecessor_control_query_suffix_host_gap_contribution"
    ] == "only_internal_suffix_gaps_excluded_from_pending_hostdelay"


def _assert_cublas_set_stream_idempotent_fold(
    event,
    *,
    folded_count: int,
    pending_host_gap_us: int,
    handle_id: str = "h0",
    stream_id: str = "s0",
    folded_handle_ids: list[str] | None = None,
    folded_stream_ids: list[str] | None = None,
) -> None:
    if folded_handle_ids is None:
        folded_handle_ids = [handle_id]
    if folded_stream_ids is None:
        folded_stream_ids = [stream_id]
    assert event.extras["cublas_set_stream_idempotent_fold_basis"] == (
        "rank_dispatch_local_handle_stream_state"
    )
    assert event.extras["cublas_set_stream_idempotent_fold_status"] == (
        "enabled_opt_in_experimental_idempotent_state_fold"
    )
    assert event.extras["cublas_set_stream_idempotent_fold_enabled"] is True
    assert event.extras["cublas_set_stream_idempotent_fold_env_flags"] == [
        "MAYA_ENABLE_IDEMPOTENT_CUBLAS_SET_STREAM_FOLD",
        "FLEXSIM_MAYA_ENABLE_IDEMPOTENT_CUBLAS_SET_STREAM_FOLD",
    ]
    assert event.extras["cublas_set_stream_idempotent_fold_handle_id"] == handle_id
    assert event.extras["cublas_set_stream_idempotent_fold_stream_id"] == stream_id
    assert (
        event.extras["cublas_set_stream_idempotent_folded_handle_ids"]
        == folded_handle_ids
    )
    assert (
        event.extras["cublas_set_stream_idempotent_folded_stream_ids"]
        == folded_stream_ids
    )
    assert event.extras["cublas_set_stream_idempotent_folded_count"] == folded_count
    assert (
        event.extras["cublas_set_stream_idempotent_fold_pending_host_gap_us"]
        == pending_host_gap_us
    )
    assert event.extras["cublas_set_stream_idempotent_fold_host_gap_contribution"] == (
        "folded_api_gaps_preserved_in_pending_hostdelay"
    )
    assert len(event.extras["cublas_set_stream_idempotent_folded_rows"]) == folded_count


def _enable_idempotent_cublas_set_stream_fold(monkeypatch, *, compat: bool = False) -> None:
    if compat:
        monkeypatch.delenv("MAYA_ENABLE_IDEMPOTENT_CUBLAS_SET_STREAM_FOLD", raising=False)
        monkeypatch.setenv("FLEXSIM_MAYA_ENABLE_IDEMPOTENT_CUBLAS_SET_STREAM_FOLD", "1")
    else:
        monkeypatch.setenv("MAYA_ENABLE_IDEMPOTENT_CUBLAS_SET_STREAM_FOLD", "1")
        monkeypatch.delenv(
            "FLEXSIM_MAYA_ENABLE_IDEMPOTENT_CUBLAS_SET_STREAM_FOLD",
            raising=False,
        )


def _enable_context_query_suffix_event_record_fold(
    monkeypatch,
    *,
    compat: bool = False,
) -> None:
    if compat:
        monkeypatch.delenv("MAYA_ENABLE_CONTEXT_QUERY_SUFFIX_EVENT_RECORD_FOLD", raising=False)
        monkeypatch.setenv(
            "FLEXSIM_MAYA_ENABLE_CONTEXT_QUERY_SUFFIX_EVENT_RECORD_FOLD",
            "1",
        )
    else:
        monkeypatch.setenv("MAYA_ENABLE_CONTEXT_QUERY_SUFFIX_EVENT_RECORD_FOLD", "1")
        monkeypatch.delenv(
            "FLEXSIM_MAYA_ENABLE_CONTEXT_QUERY_SUFFIX_EVENT_RECORD_FOLD",
            raising=False,
        )


def _disable_context_query_suffix_event_record_fold(monkeypatch) -> None:
    monkeypatch.delenv("MAYA_ENABLE_CONTEXT_QUERY_SUFFIX_EVENT_RECORD_FOLD", raising=False)
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_CONTEXT_QUERY_SUFFIX_EVENT_RECORD_FOLD",
        raising=False,
    )


def _enable_context_query_suffix_launch_config_pop_fold(
    monkeypatch,
    *,
    compat: bool = False,
) -> None:
    if compat:
        monkeypatch.delenv("MAYA_ENABLE_CONTEXT_QUERY_SUFFIX_LAUNCH_CONFIG_POP_FOLD", raising=False)
        monkeypatch.setenv(
            "FLEXSIM_MAYA_ENABLE_CONTEXT_QUERY_SUFFIX_LAUNCH_CONFIG_POP_FOLD",
            "1",
        )
    else:
        monkeypatch.setenv("MAYA_ENABLE_CONTEXT_QUERY_SUFFIX_LAUNCH_CONFIG_POP_FOLD", "1")
        monkeypatch.delenv(
            "FLEXSIM_MAYA_ENABLE_CONTEXT_QUERY_SUFFIX_LAUNCH_CONFIG_POP_FOLD",
            raising=False,
        )


def _disable_context_query_suffix_launch_config_pop_fold(monkeypatch) -> None:
    monkeypatch.delenv(
        "MAYA_ENABLE_CONTEXT_QUERY_SUFFIX_LAUNCH_CONFIG_POP_FOLD",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_CONTEXT_QUERY_SUFFIX_LAUNCH_CONFIG_POP_FOLD",
        raising=False,
    )


def test_collate_preserves_rank_program_order():
    trace_dir = Path("paper/traces/fake/e3")
    if not trace_dir.exists():
        pytest.skip(f"trace dir not available: {trace_dir}")

    bundle = load_trace_directory(trace_dir, max_events_per_rank=64)
    collated = collate_trace_bundle(bundle)

    assert collated.total_events >= bundle.total_events
    assert collated.world_size == bundle.world_size

    for rank, events in collated.rank_events.items():
        assert events
        assert [event.ordinal for event in events] == sorted(event.ordinal for event in events)
        seen_ids = set()
        lane_heads: dict[tuple[int, int], str] = {}
        for event in events:
            dispatch_key = (event.pid, event.tid)
            if event.prev_event_id is None:
                assert dispatch_key not in lane_heads
            else:
                assert event.prev_event_id in seen_ids
            lane_heads[dispatch_key] = event.id
            seen_ids.add(event.id)
            assert event.rank == rank


def test_collate_inserts_host_delay_events_from_timestamp_gaps():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.REAL,
                ts=140,
                pid=1,
                tid=2,
                module="libnccl.so.2",
                api="ncclAllReduce",
                op_type="nccl_collective",
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank_trace,),
    )

    collated = collate_trace_bundle(bundle)
    events = collated.rank_events[0]

    assert [event.api for event in events] == [
        "cudaLaunchKernel",
        "__hostDelay__",
        "ncclAllReduce",
    ]


def test_collate_component_strict_counterpart_hostdelay_metadata_opt_in(monkeypatch):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.setenv(
        "MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
        "1",
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
        raising=False,
    )
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"stream_id": "0"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.REAL,
                ts=140,
                pid=1,
                tid=2,
                module="libnccl.so.2",
                api="ncclAllReduce",
                op_type="nccl_collective",
                extras={"stream_id": "1"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]

    assert host_delay.extras["observed_gap_us"] == 40
    assert host_delay.extras["component_strict_counterpart_schema_version"] == (
        "component_strict_counterpart_metadata_evidence_v1"
    )
    assert host_delay.extras["component_strict_counterpart_opt_in_flag"] is True
    assert host_delay.extras["source_side"] == "predicted_component_metadata"
    assert host_delay.extras["stable_predicted_component_row_id"] == (
        "rank:0:hostdelay_boundary:raw_ordinal:0->raw_ordinal:1"
    )
    assert host_delay.extras["stable_predicted_interval_row_id"] == "r0:h1"
    assert host_delay.extras["stable_predicted_edge_row_id"] is None
    assert host_delay.extras["stable_predicted_count_once_group_id"] == (
        "predicted_hostdelay_boundary:rank:0:hostdelay_boundary:raw_ordinal:0->raw_ordinal:1"
    )
    assert host_delay.extras["component_kind"] == "host_control_interval"
    assert host_delay.extras["predicted_replay_component"] == "materialized_hostDelay_boundary"
    assert host_delay.extras["api_or_kernel_family"] == "cudaLaunchKernel -> ncclAllReduce"
    assert host_delay.extras["predicted_interval_duration_us"] == 40.0
    assert host_delay.extras["predicted_interval_resource_kind"] == "host"
    assert host_delay.extras["predicted_interval_resource_id"] == "legacy_pid:1"
    assert host_delay.extras["strict_actual_timing_status"] == "unavailable"
    assert host_delay.extras["strict_actual_timing_available"] is False
    assert host_delay.extras["actual_start_us"] is None
    assert host_delay.extras["actual_end_us"] is None
    assert host_delay.extras["actual_duration_us"] is None
    assert host_delay.extras["actual_endpoint_timestamps_used_as_strict_timing"] is False
    assert host_delay.extras["actual_host_duration_used_as_strict_timing"] is False
    assert host_delay.extras["actual_runtime_direct_substitution"] is False
    assert host_delay.extras["actual_observed_runtime_used_as_prediction"] is False
    assert host_delay.extras["stream_namespace_alignment_status"] == (
        "predicted_only_actual_alignment_unavailable"
    )
    assert host_delay.extras["count_once_status"] == "unavailable"
    assert host_delay.extras["nonoverlap_status"] == "unavailable"
    assert host_delay.extras["wait_map_safety_status"] == "unavailable"
    assert host_delay.extras["producer_visibility_status"] == "unavailable"
    assert host_delay.extras["repair_ready"] is False
    assert host_delay.extras["safe_to_use_as_repair_evidence"] is False
    assert host_delay.extras["safe_to_use_as_subtraction_delta"] is False
    assert events[1].extras["observed_gap_us"] == 40
    assert events[2].prev_event_id == events[1].id


def test_collate_component_strict_actual_endpoint_fields_remain_sidecar(monkeypatch):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.setenv(
        "MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
        "1",
    )
    actual_component_fields = {
        "component_strict_counterpart_schema_version": (
            "component_strict_counterpart_actual_metadata_evidence_v1"
        ),
        "component_strict_counterpart_opt_in_flag": True,
        "source_side": "actual_counterpart_metadata",
        "actual_trace_id": "actual-trace-r0",
        "actual_rank": 0,
        "actual_paper_valid_window_id": "actual-window-0",
        "actual_paper_valid_window_unavailable_reason": None,
        "boundary_origin_kind": "paper_visible_user_framework_cpu",
        "actual_counterpart_join_status": (
            "actual_metadata_export_only_predicted_join_deferred"
        ),
        "actual_counterpart_join_basis": "not_joined_during_capture",
        "strict_actual_timing_status": "unavailable",
        "strict_actual_timing_available": False,
        "actual_start_us": None,
        "actual_end_us": None,
        "actual_duration_us": None,
        "actual_endpoint_timestamps_used_as_strict_timing": False,
        "actual_host_duration_used_as_strict_timing": False,
        "actual_runtime_direct_substitution": False,
        "actual_observed_runtime_used_as_prediction": False,
        "stream_namespace_alignment_status": (
            "actual_only_unresolved_predicted_namespace_not_joined"
        ),
        "count_once_status": "unavailable",
        "nonoverlap_status": "unavailable",
        "wait_map_safety_status": "unavailable",
        "producer_visibility_status": "unavailable",
        "repair_ready": False,
        "safe_to_use_as_repair_evidence": False,
        "safe_to_use_as_subtraction_delta": False,
    }
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"stream_id": "0", **actual_component_fields},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.REAL,
                ts=140,
                pid=1,
                tid=2,
                module="libnccl.so.2",
                api="ncclAllReduce",
                op_type="nccl_collective",
                extras={"stream_id": "1", **actual_component_fields},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]

    assert host_delay.extras["component_strict_counterpart_schema_version"] == (
        "component_strict_counterpart_metadata_evidence_v1"
    )
    assert host_delay.extras["source_side"] == "predicted_component_metadata"
    assert host_delay.extras["actual_counterpart_join_status"] == (
        "predicted_metadata_only_actual_join_deferred"
    )
    assert host_delay.extras["actual_counterpart_join_basis"] == (
        "offline_component_strict_counterpart_join_required"
    )
    assert host_delay.extras["stream_namespace_alignment_status"] == (
        "predicted_only_actual_alignment_unavailable"
    )

    prev_fields = host_delay.extras["boundary_origin_prev_fields"]
    current_fields = host_delay.extras["boundary_origin_current_fields"]
    assert prev_fields["component_strict_counterpart_schema_version"] == (
        "component_strict_counterpart_actual_metadata_evidence_v1"
    )
    assert current_fields["component_strict_counterpart_schema_version"] == (
        "component_strict_counterpart_actual_metadata_evidence_v1"
    )
    assert prev_fields["source_side"] == "actual_counterpart_metadata"
    assert current_fields["source_side"] == "actual_counterpart_metadata"
    assert prev_fields["actual_trace_id"] == "actual-trace-r0"
    assert current_fields["actual_trace_id"] == "actual-trace-r0"
    assert prev_fields["actual_rank"] == 0
    assert current_fields["actual_rank"] == 0
    assert prev_fields["actual_paper_valid_window_id"] == "actual-window-0"
    assert current_fields["actual_paper_valid_window_id"] == "actual-window-0"
    assert current_fields["actual_counterpart_join_basis"] == "not_joined_during_capture"
    assert host_delay.extras["boundary_origin_kind"] == "paper_visible_user_framework_cpu"
    assert "actual_trace_id" not in host_delay.extras
    assert "actual_rank" not in host_delay.extras
    assert "actual_paper_valid_window_id" not in host_delay.extras


def test_collate_materializes_step_window_boundary_host_delay() -> None:
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
        step_windows={0: (100, 220)},
        trace_window="step",
    )

    collated = collate_trace_bundle(bundle)
    events = collated.rank_events[0]

    assert [event.api for event in events] == [
        "__hostDelay__",
        "cudaLaunchKernel",
        "__hostDelay__",
    ]
    assert events[0].extras["observed_gap_us"] == 40
    assert events[0].extras["boundary"] == "step_start"
    assert events[1].prev_event_id == events[0].id
    assert events[2].extras["observed_gap_us"] == 80
    assert events[2].extras["boundary"] == "step_end"
    assert events[2].prev_event_id == events[1].id
    assert events[0].ts == 100
    assert events[0].ts + events[0].extras["observed_gap_us"] == 140
    assert events[2].ts == 140
    assert events[2].ts + events[2].extras["observed_gap_us"] == 220


def test_collate_places_materialized_host_delay_on_measured_interval() -> None:
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=190,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
        step_windows={0: (100, 250)},
        trace_window="step",
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delays = [event for event in events if event.api == "__hostDelay__"]

    assert [event.extras["hostdelay_source"] for event in host_delays] == [
        "leading_step_gap",
        "collate_host_gap",
        "trailing_step_gap",
    ]
    assert [(event.ts, event.extras["observed_gap_us"]) for event in host_delays] == [
        (100, 40),
        (140, 50),
        (190, 60),
    ]
    assert [event.ts + event.extras["observed_gap_us"] for event in host_delays] == [
        140,
        190,
        250,
    ]
    assert [
        event.extras["paper_valid_window_membership"]["in_paper_valid_window"]
        for event in host_delays
    ] == [True, True, True]


def test_collate_preserves_host_delay_around_ignorable_setup_apis():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.REAL,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank_trace,),
    )

    collated = collate_trace_bundle(bundle)
    events = collated.rank_events[0]

    assert [event.api for event in events] == ["__hostDelay__", "cudaLaunchKernel"]
    assert events[0].extras["observed_gap_us"] == 40
    assert "launch_config_metadata_basis" not in events[1].extras


def test_collate_suppresses_launch_config_pop_to_launch_host_gap_default_on(monkeypatch):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.delenv("MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("FLEXSIM_MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("FLEXSIM_MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="__cudaPopCallConfiguration",
                op_type="other",
                extras={"end_ts": 104, "host_duration_us": 4.0},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=150,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]

    assert [event.api for event in events] == ["cudaLaunchKernel"]
    assert events[0].prev_event_id is None
    _assert_launch_config_metadata(
        events[0],
        adjacent_gap_us=50,
        contribution="excluded_from_pending_host_gap",
        normalization_enabled=True,
        normalization_status="enabled_default_excluded_from_hostdelay",
    )


def test_collate_keeps_launch_config_pop_to_launch_host_gap_with_enable_zero_control(monkeypatch):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.setenv("MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", "0")
    monkeypatch.delenv("FLEXSIM_MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("FLEXSIM_MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="__cudaPopCallConfiguration",
                op_type="other",
                extras={"end_ts": 104, "host_duration_us": 4.0},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=150,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]

    assert [event.api for event in events] == ["__hostDelay__", "cudaLaunchKernel"]
    assert events[0].extras["observed_gap_us"] == 50
    _assert_launch_config_metadata(
        events[1],
        adjacent_gap_us=50,
        contribution="included_in_pending_host_gap_disable_control",
        normalization_enabled=False,
        normalization_status="disabled_enable_zero_control_hostdelay_preserved",
    )


def test_collate_suppresses_launch_config_pop_to_launch_host_gap_opt_in(monkeypatch):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.setenv("MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", "1")
    monkeypatch.delenv("FLEXSIM_MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("FLEXSIM_MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="__cudaPopCallConfiguration",
                op_type="other",
                extras={"end_ts": 104, "host_duration_us": 4.0},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=150,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]

    assert [event.api for event in events] == ["cudaLaunchKernel"]
    assert events[0].prev_event_id is None
    _assert_launch_config_metadata(
        events[0],
        adjacent_gap_us=50,
        contribution="excluded_from_pending_host_gap",
        normalization_enabled=True,
        normalization_status="enabled_explicit_enable_excluded_from_hostdelay",
    )


def test_collate_launch_config_default_on_matches_enable_one_and_disable_differs(monkeypatch):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    def run_with_env(enable_value: str | None, disable_value: str | None = None) -> list[str]:
        monkeypatch.delenv("FLEXSIM_MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
        monkeypatch.delenv("FLEXSIM_MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
        if enable_value is None:
            monkeypatch.delenv("MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
        else:
            monkeypatch.setenv("MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", enable_value)
        if disable_value is None:
            monkeypatch.delenv("MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
        else:
            monkeypatch.setenv("MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", disable_value)
        rank_trace = RankTrace(
            rank=0,
            path=Path("/tmp/rank_0.jsonl"),
            source=TraceSource.FAKE,
            events=(
                TraceEvent(
                    rank=0,
                    ordinal=0,
                    source=TraceSource.FAKE,
                    ts=100,
                    pid=1,
                    tid=2,
                    module="libcudart.so.12",
                    api="__cudaPopCallConfiguration",
                    op_type="other",
                    extras={"end_ts": 104, "host_duration_us": 4.0},
                ),
                TraceEvent(
                    rank=0,
                    ordinal=1,
                    source=TraceSource.FAKE,
                    ts=150,
                    pid=1,
                    tid=2,
                    module="libcudart.so.12",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                ),
            ),
        )
        bundle = TraceBundle(
            trace_dir=Path("/tmp/manual"),
            source=TraceSource.FAKE,
            rank_traces=(rank_trace,),
        )
        events = collate_trace_bundle(bundle).rank_events[0]
        launch = [event for event in events if event.api == "cudaLaunchKernel"][0]
        enabled = disable_value != "1" and enable_value != "0"
        if not enabled and disable_value == "1":
            status = "disabled_explicit_disable_control_hostdelay_preserved"
        elif not enabled:
            status = "disabled_enable_zero_control_hostdelay_preserved"
        elif enable_value == "1":
            status = "enabled_explicit_enable_excluded_from_hostdelay"
        else:
            status = "enabled_default_excluded_from_hostdelay"
        _assert_launch_config_metadata(
            launch,
            adjacent_gap_us=50,
            contribution=(
                "excluded_from_pending_host_gap"
                if enabled
                else "included_in_pending_host_gap_disable_control"
            ),
            normalization_enabled=enabled,
            normalization_status=status,
        )
        return [event.api for event in events]

    assert run_with_env(None) == ["cudaLaunchKernel"]
    assert run_with_env("1") == ["cudaLaunchKernel"]
    assert run_with_env("0") == ["__hostDelay__", "cudaLaunchKernel"]
    assert run_with_env(None, disable_value="1") == ["__hostDelay__", "cudaLaunchKernel"]
    assert run_with_env("1", disable_value="1") == ["__hostDelay__", "cudaLaunchKernel"]


def test_collate_preserves_step_start_gap_before_initial_launch_config_pop(monkeypatch):
    from flexsim.maya_lite.schema import FidelityWindow, RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.delenv("MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("FLEXSIM_MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("FLEXSIM_MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=120,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="__cudaPopCallConfiguration",
                op_type="other",
                extras={"end_ts": 124, "host_duration_us": 4.0},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=150,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"host_duration_us": 10.0},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
        trace_window="step",
        step_windows={0: (90, 200)},
        fidelity_windows={
            0: FidelityWindow(
                start_ts=90,
                end_ts=200,
                source="manifest",
                is_paper_valid_step_window=True,
            )
        },
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    leading_host_delay = events[0]
    launch = events[1]

    assert [event.api for event in events] == ["__hostDelay__", "cudaLaunchKernel", "__hostDelay__"]
    assert leading_host_delay.extras["hostdelay_source"] == "leading_step_gap"
    assert leading_host_delay.extras["observed_gap_us"] == 30
    assert leading_host_delay.extras["raw_prev_event_id"] is None
    assert leading_host_delay.extras["raw_current_event_id"] == "r0:e0"
    assert leading_host_delay.extras["raw_current_api"] == "__cudaPopCallConfiguration"
    assert leading_host_delay.extras["raw_boundary_family"] == "__cudaPopCallConfiguration"
    assert leading_host_delay.extras["current_materialized_event_id"] is None
    assert leading_host_delay.extras["current_materialized_api"] is None
    assert leading_host_delay.extras["paper_valid_window_membership"] == {
        "in_paper_valid_window": True,
        "window_id": "rank0:step_window",
        "window_source": "manifest",
        "start_ts": 90,
        "end_ts": 200,
        "is_paper_valid_step_window": True,
        "membership_basis": "collate_step_window",
        "unavailable_reason": None,
    }
    assert launch.prev_event_id == leading_host_delay.id
    _assert_launch_config_metadata(
        launch,
        adjacent_gap_us=30,
        contribution="excluded_from_pending_host_gap",
        normalization_enabled=True,
        normalization_status="enabled_default_excluded_from_hostdelay",
        raw_event_id="r0:e0",
        raw_ts_us=120,
    )


def test_collate_folds_cuda_get_device_suffix_before_cublas_set_stream_default_on():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=120,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
                extras={"end_ts": 121, "host_duration_us": 1.0},
            ),
            TraceEvent(
                rank=0,
                ordinal=2,
                source=TraceSource.FAKE,
                ts=135,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
                extras={"end_ts": 136, "host_duration_us": 1.0},
            ),
            TraceEvent(
                rank=0,
                ordinal=3,
                source=TraceSource.FAKE,
                ts=160,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"handle_id": "h0", "stream_id": 7},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]
    cublas_set_stream = [event for event in events if event.api == "cublasSetStream_v2"][0]

    assert [event.api for event in events] == [
        "cudaLaunchKernel",
        "__hostDelay__",
        "cublasSetStream_v2",
    ]
    assert host_delay.extras["observed_gap_us"] == 20
    assert host_delay.extras["raw_prev_api"] == "cudaLaunchKernel"
    assert host_delay.extras["raw_current_api"] == "cudaGetDevice"
    assert host_delay.extras["raw_boundary_family"] == "cudaLaunchKernel -> cudaGetDevice"
    assert cublas_set_stream.prev_event_id == host_delay.id
    _assert_cublas_context_query_suffix_fold(
        cublas_set_stream,
        folded_count=2,
        suppressed_host_gap_us=40,
        preserved_pre_suffix_gap_us=20,
    )
    assert (
        cublas_set_stream.extras["cublas_set_stream_context_query_suffix_internal_gap_us"]
        == 15
    )
    assert (
        cublas_set_stream.extras["cublas_set_stream_context_query_suffix_terminal_gap_us"]
        == 25
    )
    assert cublas_set_stream.extras["cublas_set_stream_context_query_suffix_folded_event_ids"] == [
        "r0:e1",
        "r0:e2",
    ]
    assert [
        row["gap_kind"]
        for row in cublas_set_stream.extras["cublas_set_stream_context_query_suffix_gap_rows"]
    ] == [
        "context_query_suffix_internal_gap",
        "context_query_suffix_terminal_gap_to_cublasSetStream_v2",
    ]


def test_collate_folds_cuda_get_device_suffix_before_cublas_set_stream_without_pre_gap():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=150,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"handle_id": "h0", "stream_id": 7},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]

    assert [event.api for event in events] == ["cublasSetStream_v2"]
    assert events[0].prev_event_id is None
    _assert_cublas_context_query_suffix_fold(
        events[0],
        folded_count=1,
        suppressed_host_gap_us=50,
        preserved_pre_suffix_gap_us=0,
    )
    assert events[0].extras["cublas_set_stream_context_query_suffix_terminal_gap_us"] == 50


def test_collate_preserves_first_dispatch_context_query_pre_suffix_step_gap():
    from flexsim.maya_lite.schema import FidelityWindow, RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=150,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"handle_id": "h0", "stream_id": 7},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
        trace_window="step",
        step_windows={0: (90, 220)},
        fidelity_windows={
            0: FidelityWindow(
                start_ts=90,
                end_ts=220,
                source="manifest",
                is_paper_valid_step_window=True,
            )
        },
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    leading_host_delay = [
        event
        for event in events
        if event.api == "__hostDelay__" and event.extras.get("hostdelay_source") == "leading_step_gap"
    ][0]
    cublas_set_stream = [event for event in events if event.api == "cublasSetStream_v2"][0]

    assert [event.api for event in events] == [
        "__hostDelay__",
        "cublasSetStream_v2",
        "__hostDelay__",
    ]
    assert leading_host_delay.extras["observed_gap_us"] == 10
    assert leading_host_delay.extras["boundary"] == "step_start"
    assert leading_host_delay.extras["raw_prev_api"] is None
    assert leading_host_delay.extras["raw_current_api"] == "cudaGetDevice"
    assert leading_host_delay.extras["raw_boundary_family"] == "cudaGetDevice"
    assert leading_host_delay.extras["current_materialized_event_id"] == "r0:e1"
    assert leading_host_delay.extras["current_materialized_api"] == "cublasSetStream_v2"
    assert cublas_set_stream.prev_event_id == leading_host_delay.id
    _assert_cublas_context_query_suffix_fold(
        cublas_set_stream,
        folded_count=1,
        suppressed_host_gap_us=50,
        preserved_pre_suffix_gap_us=10,
    )
    assert cublas_set_stream.extras["cublas_set_stream_context_query_suffix_terminal_gap_us"] == 50
    assert not [
        event
        for event in events
        if event.api == "__hostDelay__"
        and event.extras.get("raw_prev_api") == "cudaGetDevice"
        and event.extras.get("raw_current_api") == "cublasSetStream_v2"
    ]


def test_collate_folds_only_internal_cuda_get_device_run_gap_before_launch():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=120,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=2,
                source=TraceSource.FAKE,
                ts=135,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=3,
                source=TraceSource.FAKE,
                ts=160,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]
    launch = events[-1]

    assert [event.api for event in events] == [
        "cudaLaunchKernel",
        "__hostDelay__",
        "cudaLaunchKernel",
    ]
    assert host_delay.extras["observed_gap_us"] == 45
    assert host_delay.extras["semantic_predecessor_boundary_family"] == (
        "cudaLaunchKernel -> cudaLaunchKernel"
    )
    assert host_delay.extras["semantic_predecessor_boundary_namespace_basis"] == (
        "raw_immediate_boundary"
    )
    _assert_cuda_get_device_context_query_run_fold(
        launch,
        event_count=2,
        internal_gap_us=15,
        terminal_gap_us=25,
        preserved_pre_run_gap_us=20,
    )
    assert launch.extras["cuda_get_device_context_query_run_event_ids"] == [
        "r0:e1",
        "r0:e2",
    ]
    assert [
        row["gap_kind"]
        for row in launch.extras["cuda_get_device_context_query_run_gap_rows"]
    ] == [
        "context_query_suffix_internal_gap",
        "cudaGetDevice_context_query_run_terminal_gap_to_materialized_event",
    ]
    assert [
        row["host_gap_contribution"]
        for row in launch.extras["cuda_get_device_context_query_run_gap_rows"]
    ] == ["excluded_from_pending_hostdelay", "included_in_pending_hostdelay"]
    assert not any(
        key.startswith("cublas_set_stream_context_query_suffix_")
        for key in launch.extras
    )


def test_collate_keeps_single_cuda_get_device_suffix_before_launch_hostdelay_default_on():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=120,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=2,
                source=TraceSource.FAKE,
                ts=160,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]
    launch = events[-1]

    assert [event.api for event in events] == [
        "cudaLaunchKernel",
        "__hostDelay__",
        "cudaLaunchKernel",
    ]
    assert host_delay.extras["observed_gap_us"] == 60
    assert host_delay.extras["raw_boundary_family"] == "cudaLaunchKernel -> cudaLaunchKernel"
    assert host_delay.extras["semantic_predecessor_boundary_family"] == (
        "cudaLaunchKernel -> cudaLaunchKernel"
    )
    assert host_delay.extras["semantic_predecessor_boundary_namespace_basis"] == (
        "raw_immediate_boundary"
    )
    assert not any(
        key.startswith("hostdelay_semantic_predecessor_control_query_suffix_")
        for key in launch.extras
    )


def test_collate_preserves_first_dispatch_cuda_get_device_run_leading_gap():
    from flexsim.maya_lite.schema import FidelityWindow, RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=130,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=2,
                source=TraceSource.FAKE,
                ts=160,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
        trace_window="step",
        step_windows={0: (90, 220)},
        fidelity_windows={
            0: FidelityWindow(
                start_ts=90,
                end_ts=220,
                source="manifest",
                is_paper_valid_step_window=True,
            )
        },
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    leading_host_delay = [
        event
        for event in events
        if event.api == "__hostDelay__" and event.extras.get("boundary") == "step_start"
    ][0]
    terminal_host_delay = [
        event
        for event in events
        if event.api == "__hostDelay__"
        and event.extras.get("hostdelay_source") == "collate_host_gap"
    ][0]
    launch = [event for event in events if event.api == "cudaLaunchKernel"][0]

    assert [event.api for event in events] == [
        "__hostDelay__",
        "__hostDelay__",
        "cudaLaunchKernel",
        "__hostDelay__",
    ]
    assert leading_host_delay.extras["observed_gap_us"] == 10
    assert leading_host_delay.extras["raw_current_api"] == "cudaGetDevice"
    assert terminal_host_delay.extras["observed_gap_us"] == 30
    assert terminal_host_delay.extras["raw_boundary_family"] == (
        "cudaGetDevice -> cudaLaunchKernel"
    )
    assert terminal_host_delay.extras["semantic_predecessor_boundary_family"] == (
        "cudaLaunchKernel"
    )
    assert terminal_host_delay.extras["semantic_predecessor_boundary_namespace_basis"] == (
        "materialized_semantic_predecessor_after_control_query_filter"
    )
    _assert_cuda_get_device_context_query_run_fold(
        launch,
        event_count=2,
        internal_gap_us=30,
        terminal_gap_us=30,
        preserved_pre_run_gap_us=10,
    )


@pytest.mark.parametrize(
    "target_api",
    ["cudaEventRecord", "cudaEventRecordWithFlags", "cudaStreamWaitEvent"],
)
def test_collate_does_not_fold_cuda_get_device_suffix_before_event_or_wait_default_off(
    monkeypatch,
    target_api,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    _disable_context_query_suffix_event_record_fold(monkeypatch)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=150,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api=target_api,
                op_type="stream_op",
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]

    assert [event.api for event in events] == ["__hostDelay__", target_api]
    assert events[0].extras["observed_gap_us"] == 50
    assert events[0].extras["raw_boundary_family"] == f"cudaGetDevice -> {target_api}"
    assert not any(
        key.startswith("cublas_set_stream_context_query_suffix_")
        for key in events[1].extras
    )
    assert not any(
        key.startswith("hostdelay_semantic_predecessor_control_query_suffix_")
        for key in events[1].extras
    )


@pytest.mark.parametrize("target_api", ["cudaEventRecord", "cudaEventRecordWithFlags"])
def test_collate_opt_in_folds_cuda_get_device_suffix_before_event_record(
    monkeypatch,
    target_api,
):
    from flexsim.maya_lite.filters import canonicalize_trace_api
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    _enable_context_query_suffix_event_record_fold(monkeypatch)
    assert canonicalize_trace_api(target_api) == "cudaEventRecord"
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=120,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
                extras={"end_ts": 121, "host_duration_us": 1.0},
            ),
            TraceEvent(
                rank=0,
                ordinal=2,
                source=TraceSource.FAKE,
                ts=160,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api=target_api,
                op_type="stream_op",
                extras={"event_id": "evt0", "stream_id": 7},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]
    event_record = [event for event in events if event.api == target_api][0]

    assert [event.api for event in events] == [
        "cudaLaunchKernel",
        "__hostDelay__",
        target_api,
    ]
    assert host_delay.extras["observed_gap_us"] == 20
    assert host_delay.extras["raw_prev_api"] == "cudaLaunchKernel"
    assert host_delay.extras["raw_current_api"] == "cudaGetDevice"
    assert host_delay.extras["raw_boundary_family"] == "cudaLaunchKernel -> cudaGetDevice"
    assert host_delay.extras["current_materialized_api"] == target_api
    assert event_record.prev_event_id == host_delay.id
    assert event_record.extras["event_id"] == "evt0"
    assert event_record.extras["stream_id"] == 7
    _assert_semantic_predecessor_context_query_suffix_fold(
        event_record,
        target_api=target_api,
        folded_count=1,
        suppressed_host_gap_us=40,
        preserved_pre_suffix_gap_us=20,
    )
    assert (
        event_record.extras[
            "hostdelay_semantic_predecessor_control_query_suffix_terminal_gap_us"
        ]
        == 40
    )
    assert not any(
        key.startswith("cublas_set_stream_context_query_suffix_")
        for key in event_record.extras
    )
    assert not [
        event
        for event in events
        if event.api == "__hostDelay__"
        and event.extras.get("raw_prev_api") == "cudaGetDevice"
        and event.extras.get("raw_current_api") == target_api
    ]


def test_collate_opt_in_preserves_first_dispatch_event_record_pre_suffix_step_gap(
    monkeypatch,
):
    from flexsim.maya_lite.schema import FidelityWindow, RankTrace, TraceBundle, TraceEvent, TraceSource

    _enable_context_query_suffix_event_record_fold(monkeypatch, compat=True)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=150,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaEventRecord",
                op_type="stream_op",
                extras={"event_id": "evt0", "stream_id": 7},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
        trace_window="step",
        step_windows={0: (90, 220)},
        fidelity_windows={
            0: FidelityWindow(
                start_ts=90,
                end_ts=220,
                source="manifest",
                is_paper_valid_step_window=True,
            )
        },
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    leading_host_delay = [
        event
        for event in events
        if event.api == "__hostDelay__" and event.extras.get("hostdelay_source") == "leading_step_gap"
    ][0]
    event_record = [event for event in events if event.api == "cudaEventRecord"][0]

    assert [event.api for event in events] == [
        "__hostDelay__",
        "cudaEventRecord",
        "__hostDelay__",
    ]
    assert leading_host_delay.extras["observed_gap_us"] == 10
    assert leading_host_delay.extras["boundary"] == "step_start"
    assert leading_host_delay.extras["raw_prev_api"] is None
    assert leading_host_delay.extras["raw_current_api"] == "cudaGetDevice"
    assert leading_host_delay.extras["current_materialized_event_id"] == "r0:e1"
    assert leading_host_delay.extras["current_materialized_api"] == "cudaEventRecord"
    _assert_semantic_predecessor_context_query_suffix_fold(
        event_record,
        target_api="cudaEventRecord",
        folded_count=1,
        suppressed_host_gap_us=50,
        preserved_pre_suffix_gap_us=10,
    )
    assert event_record.extras[
        "hostdelay_semantic_predecessor_control_query_suffix_terminal_gap_us"
    ] == 50
    assert not [
        event
        for event in events
        if event.api == "__hostDelay__"
        and event.extras.get("raw_prev_api") == "cudaGetDevice"
        and event.extras.get("raw_current_api") == "cudaEventRecord"
    ]


def test_collate_opt_in_folds_cuda_get_device_run_before_event_record_as_suffix(
    monkeypatch,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    _enable_context_query_suffix_event_record_fold(monkeypatch)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=125,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=2,
                source=TraceSource.FAKE,
                ts=175,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaEventRecord",
                op_type="stream_op",
                extras={"event_id": "evt0", "stream_id": 7},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]

    assert [event.api for event in events] == ["cudaEventRecord"]
    event_record = events[0]
    _assert_semantic_predecessor_context_query_suffix_fold(
        event_record,
        target_api="cudaEventRecord",
        folded_count=2,
        suppressed_host_gap_us=75,
        preserved_pre_suffix_gap_us=0,
    )
    assert event_record.extras[
        "hostdelay_semantic_predecessor_control_query_suffix_internal_gap_us"
    ] == 25
    assert event_record.extras[
        "hostdelay_semantic_predecessor_control_query_suffix_terminal_gap_us"
    ] == 50
    assert [
        row["gap_kind"]
        for row in event_record.extras[
            "hostdelay_semantic_predecessor_control_query_suffix_gap_rows"
        ]
    ] == [
        "context_query_suffix_internal_gap",
        "context_query_suffix_terminal_gap_to_cudaEventRecord",
    ]
    assert not any(
        key.startswith("cuda_get_device_context_query_run_")
        for key in event_record.extras
    )


@pytest.mark.parametrize("target_api", ["cudaEventRecord", "cudaStreamWaitEvent"])
def test_collate_preserves_event_wait_terminal_gap_after_cuda_get_device_run(
    monkeypatch,
    target_api,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    _disable_context_query_suffix_event_record_fold(monkeypatch)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=125,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=2,
                source=TraceSource.FAKE,
                ts=175,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api=target_api,
                op_type="stream_op",
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]

    assert [event.api for event in events] == ["__hostDelay__", target_api]
    assert events[0].extras["observed_gap_us"] == 50
    assert events[0].extras["raw_boundary_family"] == f"cudaGetDevice -> {target_api}"
    _assert_cuda_get_device_context_query_run_fold(
        events[1],
        event_count=2,
        internal_gap_us=25,
        terminal_gap_us=50,
        preserved_pre_run_gap_us=0,
    )
    assert not any(
        key.startswith("cublas_set_stream_context_query_suffix_")
        for key in events[1].extras
    )


def test_collate_opt_in_does_not_fold_cuda_get_device_suffix_before_stream_wait(
    monkeypatch,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    _enable_context_query_suffix_event_record_fold(monkeypatch)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=150,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaStreamWaitEvent",
                op_type="stream_op",
                extras={"event_id": "evt0", "stream_id": 7},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]

    assert [event.api for event in events] == ["__hostDelay__", "cudaStreamWaitEvent"]
    assert events[0].extras["observed_gap_us"] == 50
    assert events[0].extras["raw_boundary_family"] == (
        "cudaGetDevice -> cudaStreamWaitEvent"
    )
    assert not any(
        key.startswith("hostdelay_semantic_predecessor_control_query_suffix_")
        for key in events[1].extras
    )


def test_collate_opt_in_does_not_fold_cuda_get_device_suffix_before_direct_launch(
    monkeypatch,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    _enable_context_query_suffix_event_record_fold(monkeypatch)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=150,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"kernel": "k0", "stream_id": 7},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]
    launch = [event for event in events if event.api == "cudaLaunchKernel"][0]

    assert [event.api for event in events] == ["__hostDelay__", "cudaLaunchKernel"]
    assert host_delay.extras["observed_gap_us"] == 50
    assert host_delay.extras["raw_boundary_family"] == "cudaGetDevice -> cudaLaunchKernel"
    assert launch.prev_event_id == host_delay.id
    assert not any(
        key.startswith("hostdelay_semantic_predecessor_control_query_suffix_")
        for key in launch.extras
    )


def test_collate_opt_in_does_not_fold_cuda_get_device_suffix_before_launch_config_pop(
    monkeypatch,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    _enable_context_query_suffix_event_record_fold(monkeypatch)
    monkeypatch.delenv("MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("FLEXSIM_MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("FLEXSIM_MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=120,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="__cudaPopCallConfiguration",
                op_type="other",
                extras={"end_ts": 124, "host_duration_us": 4.0},
            ),
            TraceEvent(
                rank=0,
                ordinal=2,
                source=TraceSource.FAKE,
                ts=150,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"kernel": "k0", "stream_id": 7},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]
    launch = [event for event in events if event.api == "cudaLaunchKernel"][0]

    assert [event.api for event in events] == ["__hostDelay__", "cudaLaunchKernel"]
    assert host_delay.extras["observed_gap_us"] == 20
    assert host_delay.extras["raw_boundary_family"] == (
        "cudaGetDevice -> __cudaPopCallConfiguration"
    )
    assert host_delay.extras["current_materialized_api"] == "cudaLaunchKernel"
    assert launch.prev_event_id == host_delay.id
    _assert_launch_config_metadata(
        launch,
        adjacent_gap_us=30,
        contribution="excluded_from_pending_host_gap",
        normalization_enabled=True,
        normalization_status="enabled_default_excluded_from_hostdelay",
        raw_event_id="r0:e1",
        raw_ts_us=120,
    )
    assert not any(
        key.startswith("hostdelay_semantic_predecessor_control_query_suffix_")
        for key in launch.extras
    )


@pytest.mark.parametrize(
    ("target_api", "op_type"),
    [
        ("cublasGemmEx", "blas_compute"),
        ("ncclAllReduce", "nccl_collective"),
    ],
)
def test_collate_preserves_compute_collective_terminal_gap_after_cuda_get_device_run(
    target_api,
    op_type,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=2,
                source=TraceSource.FAKE,
                ts=210,
                pid=1,
                tid=2,
                module="libcublas.so.12" if target_api.startswith("cublas") else "libnccl.so.2",
                api=target_api,
                op_type=op_type,
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]

    assert [event.api for event in events] == ["__hostDelay__", target_api]
    assert events[0].extras["observed_gap_us"] == 70
    assert events[0].extras["raw_boundary_family"] == f"cudaGetDevice -> {target_api}"
    _assert_cuda_get_device_context_query_run_fold(
        events[1],
        event_count=2,
        internal_gap_us=40,
        terminal_gap_us=70,
        preserved_pre_run_gap_us=0,
    )


def test_collate_keeps_idempotent_cublas_set_stream_materialized_by_default(monkeypatch):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.delenv("MAYA_ENABLE_IDEMPOTENT_CUBLAS_SET_STREAM_FOLD", raising=False)
    monkeypatch.delenv("FLEXSIM_MAYA_ENABLE_IDEMPOTENT_CUBLAS_SET_STREAM_FOLD", raising=False)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"handle_id": "h0", "stream_id": "s0"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=130,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"handle_id": "h0", "stream_id": "s0"},
            ),
            TraceEvent(
                rank=0,
                ordinal=2,
                source=TraceSource.FAKE,
                ts=160,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasGemmEx",
                op_type="blas_compute",
                extras={"handle_id": "h0"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]

    assert [event.api for event in events] == [
        "cublasSetStream_v2",
        "__hostDelay__",
        "cublasSetStream_v2",
        "__hostDelay__",
        "cublasGemmEx",
    ]
    assert [event.extras["observed_gap_us"] for event in events if event.api == "__hostDelay__"] == [
        30,
        30,
    ]
    assert not any(
        key.startswith("cublas_set_stream_idempotent_fold_")
        for event in events
        for key in event.extras
    )


def test_collate_folds_idempotent_cublas_set_stream_into_next_materialized_event_opt_in(
    monkeypatch,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    _enable_idempotent_cublas_set_stream_fold(monkeypatch)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"handle_id": "h0", "stream_id": "s0"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=130,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"handle_id": "h0", "stream_id": "s0"},
            ),
            TraceEvent(
                rank=0,
                ordinal=2,
                source=TraceSource.FAKE,
                ts=160,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasGemmEx",
                op_type="blas_compute",
                extras={"handle_id": "h0"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]
    gemm = [event for event in events if event.api == "cublasGemmEx"][0]

    assert [event.api for event in events] == [
        "cublasSetStream_v2",
        "__hostDelay__",
        "cublasGemmEx",
    ]
    assert host_delay.extras["observed_gap_us"] == 60
    assert host_delay.extras["raw_prev_api"] == "cublasSetStream_v2"
    assert host_delay.extras["raw_current_api"] == "cublasGemmEx"
    _assert_cublas_set_stream_idempotent_fold(
        gemm,
        folded_count=1,
        pending_host_gap_us=60,
    )
    assert gemm.extras["cublas_set_stream_idempotent_folded_event_ids"] == ["r0:e1"]
    assert [
        row["gap_kind"]
        for row in gemm.extras["cublas_set_stream_idempotent_fold_gap_rows"]
    ] == ["idempotent_cublasSetStream_v2_terminal_gap_to_next_materialized_event"]
    assert gemm.extras["cublas_set_stream_idempotent_fold_gap_rows"][0][
        "host_gap_contribution"
    ] == "included_in_pending_hostdelay"


def test_collate_aggregates_idempotent_cublas_set_stream_metadata_for_multiple_handles(
    monkeypatch,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    _enable_idempotent_cublas_set_stream_fold(monkeypatch, compat=True)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"handle_id": "h0", "stream_id": "s0"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=120,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"handle_id": "h1", "stream_id": "s1"},
            ),
            TraceEvent(
                rank=0,
                ordinal=2,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"handle_id": "h0", "stream_id": "s0"},
            ),
            TraceEvent(
                rank=0,
                ordinal=3,
                source=TraceSource.FAKE,
                ts=160,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"handle_id": "h1", "stream_id": "s1"},
            ),
            TraceEvent(
                rank=0,
                ordinal=4,
                source=TraceSource.FAKE,
                ts=180,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasGemmEx",
                op_type="blas_compute",
                extras={"handle_id": "h1"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delays = [event for event in events if event.api == "__hostDelay__"]
    gemm = [event for event in events if event.api == "cublasGemmEx"][0]

    assert [event.api for event in events] == [
        "cublasSetStream_v2",
        "__hostDelay__",
        "cublasSetStream_v2",
        "__hostDelay__",
        "cublasGemmEx",
    ]
    assert [event.extras["observed_gap_us"] for event in host_delays] == [20, 60]
    _assert_cublas_set_stream_idempotent_fold(
        gemm,
        folded_count=2,
        pending_host_gap_us=60,
        handle_id="__multiple__",
        stream_id="__multiple__",
        folded_handle_ids=["h0", "h1"],
        folded_stream_ids=["s0", "s1"],
    )
    assert gemm.extras["cublas_set_stream_idempotent_folded_event_ids"] == [
        "r0:e2",
        "r0:e3",
    ]
    assert [
        (row["raw_event_id"], row["raw_handle_id"], row["raw_stream_id"])
        for row in gemm.extras["cublas_set_stream_idempotent_folded_rows"]
    ] == [
        ("r0:e2", "h0", "s0"),
        ("r0:e3", "h1", "s1"),
    ]
    assert gemm.extras["cublas_set_stream_idempotent_folded_handle_stream_pairs"] == [
        {
            "handle_id": "h0",
            "stream_id": "s0",
            "folded_count": 1,
            "folded_event_ids": ["r0:e2"],
        },
        {
            "handle_id": "h1",
            "stream_id": "s1",
            "folded_count": 1,
            "folded_event_ids": ["r0:e3"],
        },
    ]
    assert [
        row["gap_kind"]
        for row in gemm.extras["cublas_set_stream_idempotent_fold_gap_rows"]
    ] == [
        "idempotent_cublasSetStream_v2_internal_gap",
        "idempotent_cublasSetStream_v2_terminal_gap_to_next_materialized_event",
    ]


def test_collate_flushes_terminal_folded_cublas_set_stream_gap_at_step_end(monkeypatch):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    _enable_idempotent_cublas_set_stream_fold(monkeypatch)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"handle_id": "h0", "stream_id": "s0"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=150,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"handle_id": "h0", "stream_id": "s0"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
        step_windows={0: (90, 220)},
        trace_window="step",
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    step_end_host_delay = [
        event
        for event in events
        if event.api == "__hostDelay__" and event.extras.get("boundary") == "step_end"
    ][0]

    assert [event.api for event in events] == [
        "__hostDelay__",
        "cublasSetStream_v2",
        "__hostDelay__",
    ]
    assert events[0].extras["observed_gap_us"] == 10
    assert events[0].extras["boundary"] == "step_start"
    assert step_end_host_delay.extras["observed_gap_us"] == 120
    assert step_end_host_delay.extras["raw_prev_event_id"] == "r0:e0"
    assert step_end_host_delay.extras["raw_current_event_id"] == "r0:e1"
    assert step_end_host_delay.extras["raw_boundary_family"] == (
        "cublasSetStream_v2 -> cublasSetStream_v2"
    )
    assert step_end_host_delay.prev_event_id == events[1].id


def test_collate_keeps_cublas_set_stream_change_materialized():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"handle_id": "h0", "stream_id": "s0"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=130,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"handle_id": "h0", "stream_id": "s1"},
            ),
            TraceEvent(
                rank=0,
                ordinal=2,
                source=TraceSource.FAKE,
                ts=160,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasGemmEx",
                op_type="blas_compute",
                extras={"handle_id": "h0"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]

    assert [event.api for event in events] == [
        "cublasSetStream_v2",
        "__hostDelay__",
        "cublasSetStream_v2",
        "__hostDelay__",
        "cublasGemmEx",
    ]
    assert [event.extras["observed_gap_us"] for event in events if event.api == "__hostDelay__"] == [
        30,
        30,
    ]
    assert not any(
        key.startswith("cublas_set_stream_idempotent_fold_")
        for event in events
        for key in event.extras
    )


def test_collate_keeps_first_cublas_set_stream_materialized():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"handle_id": "h0", "stream_id": "s0"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=130,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasGemmEx",
                op_type="blas_compute",
                extras={"handle_id": "h0"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]

    assert [event.api for event in events] == [
        "cublasSetStream_v2",
        "__hostDelay__",
        "cublasGemmEx",
    ]
    assert not any(
        key.startswith("cublas_set_stream_idempotent_fold_")
        for event in events
        for key in event.extras
    )


@pytest.mark.parametrize(
    ("lifecycle_events", "expected_first_gap_us"),
    [
        (("cublasCreate_v2",), 60),
        (("cublasDestroy_v2", "cublasCreate_v2"), 70),
    ],
)
def test_collate_invalidates_cublas_set_stream_idempotence_on_handle_lifecycle(
    lifecycle_events,
    expected_first_gap_us,
    monkeypatch,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    _enable_idempotent_cublas_set_stream_fold(monkeypatch)
    raw_events = [
        TraceEvent(
            rank=0,
            ordinal=0,
            source=TraceSource.FAKE,
            ts=100,
            pid=1,
            tid=2,
            module="libcublas.so.12",
            api="cublasSetStream_v2",
            op_type="stream_op",
            extras={"handle_id": "h0", "stream_id": "s0"},
        )
    ]
    next_ts = 120 if len(lifecycle_events) > 1 else 130
    for api in lifecycle_events:
        raw_events.append(
            TraceEvent(
                rank=0,
                ordinal=len(raw_events),
                source=TraceSource.FAKE,
                ts=next_ts,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api=api,
                op_type="context_op",
                extras={"handle_id": "h0"},
            )
        )
        next_ts += 20 if len(lifecycle_events) > 1 else 30
    raw_events.extend(
        [
            TraceEvent(
                rank=0,
                ordinal=len(raw_events),
                source=TraceSource.FAKE,
                ts=100 + expected_first_gap_us,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"handle_id": "h0", "stream_id": "s0"},
            ),
            TraceEvent(
                rank=0,
                ordinal=len(raw_events) + 1,
                source=TraceSource.FAKE,
                ts=130 + expected_first_gap_us,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasGemmEx",
                op_type="blas_compute",
                extras={"handle_id": "h0"},
            ),
        ]
    )
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=tuple(raw_events),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    materialized_setters = [
        event for event in events if event.api == "cublasSetStream_v2"
    ]

    assert [event.api for event in events] == [
        "cublasSetStream_v2",
        "__hostDelay__",
        "cublasSetStream_v2",
        "__hostDelay__",
        "cublasGemmEx",
    ]
    assert len(materialized_setters) == 2
    assert [event.extras["observed_gap_us"] for event in events if event.api == "__hostDelay__"] == [
        expected_first_gap_us,
        30,
    ]
    assert not any(
        key.startswith("cublas_set_stream_idempotent_fold_")
        for event in events
        for key in event.extras
    )


@pytest.mark.parametrize(
    ("target_api", "target_op_type", "target_module"),
    [
        ("cudaEventRecord", "stream_op", "libcudart.so.12"),
        ("cudaStreamWaitEvent", "stream_op", "libcudart.so.12"),
        ("cublasGemmEx", "blas_compute", "libcublas.so.12"),
        ("ncclAllReduce", "nccl_collective", "libnccl.so.2"),
    ],
)
def test_collate_idempotent_cublas_set_stream_fold_preserves_next_event_kind(
    target_api,
    target_op_type,
    target_module,
    monkeypatch,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    _enable_idempotent_cublas_set_stream_fold(monkeypatch)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"handle_id": "h0", "stream_id": "s0"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=130,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"handle_id": "h0", "stream_id": "s0"},
            ),
            TraceEvent(
                rank=0,
                ordinal=2,
                source=TraceSource.FAKE,
                ts=160,
                pid=1,
                tid=2,
                module=target_module,
                api=target_api,
                op_type=target_op_type,
                extras={"handle_id": "h0"} if target_api == "cublasGemmEx" else {},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    target = [event for event in events if event.api == target_api][0]

    assert [event.api for event in events] == [
        "cublasSetStream_v2",
        "__hostDelay__",
        target_api,
    ]
    assert target.op_type == target_op_type
    _assert_cublas_set_stream_idempotent_fold(
        target,
        folded_count=1,
        pending_host_gap_us=60,
    )


def test_collate_forwards_boundary_origin_provenance_to_host_delay():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
                extras={
                    "boundary_origin_kind": "fake_wrapper_instrumentation_only",
                    "instrumentation_only_duration_us": 7.5,
                },
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"caller_visible_elapsed_us": 40.0},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    collated = collate_trace_bundle(bundle)
    host_delay = [event for event in collated.rank_events[0] if event.api == "__hostDelay__"][0]

    assert host_delay.extras["observed_gap_us"] == 40
    assert host_delay.extras["boundary_origin_family"] == "cudaGetDevice -> cudaLaunchKernel"
    assert host_delay.extras["boundary_origin_kind"] == "fake_wrapper_instrumentation_only"
    assert host_delay.extras["instrumentation_only_duration_us"] == 7.5
    assert host_delay.extras["caller_visible_elapsed_us"] == 40.0
    assert host_delay.extras["boundary_origin_field_sources"] == {
        "boundary_origin_kind": ["prev"],
        "instrumentation_only_duration_us": ["prev"],
        "caller_visible_elapsed_us": ["current"],
    }


def test_collate_forwards_current_only_boundary_origin_provenance():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={
                    "boundary_origin_kind": "paper_visible_user_framework_cpu",
                    "paper_visible_host_duration_us": 40.0,
                },
            ),
        ),
    )
    bundle = TraceBundle(trace_dir=Path("/tmp/manual"), source=TraceSource.FAKE, rank_traces=(rank_trace,))

    host_delay = [event for event in collate_trace_bundle(bundle).rank_events[0] if event.api == "__hostDelay__"][0]

    assert host_delay.extras["observed_gap_us"] == 40
    assert host_delay.extras["boundary_origin_kind"] == "paper_visible_user_framework_cpu"
    assert host_delay.extras["paper_visible_host_duration_us"] == 40.0
    assert host_delay.extras["boundary_origin_field_sources"] == {
        "boundary_origin_kind": ["current"],
        "paper_visible_host_duration_us": ["current"],
    }


def test_collate_boundary_origin_same_value_records_both_sources():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
                extras={"boundary_origin_kind": "mixed_or_unresolved"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"boundary_origin_kind": "mixed_or_unresolved"},
            ),
        ),
    )
    bundle = TraceBundle(trace_dir=Path("/tmp/manual"), source=TraceSource.FAKE, rank_traces=(rank_trace,))

    host_delay = [event for event in collate_trace_bundle(bundle).rank_events[0] if event.api == "__hostDelay__"][0]

    assert host_delay.extras["observed_gap_us"] == 40
    assert host_delay.extras["boundary_origin_kind"] == "mixed_or_unresolved"
    assert host_delay.extras["boundary_origin_field_sources"] == {"boundary_origin_kind": ["current", "prev"]}
    assert "boundary_origin_conflicting_fields" not in host_delay.extras


def test_collate_boundary_origin_conflict_preserves_both_sides():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
                extras={
                    "boundary_origin_kind": "fake_wrapper_instrumentation_only",
                    "instrumentation_only_duration_us": 4.0,
                },
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={
                    "boundary_origin_kind": "paper_visible_user_framework_cpu",
                    "paper_visible_host_duration_us": 36.0,
                },
            ),
        ),
    )
    bundle = TraceBundle(trace_dir=Path("/tmp/manual"), source=TraceSource.FAKE, rank_traces=(rank_trace,))

    host_delay = [event for event in collate_trace_bundle(bundle).rank_events[0] if event.api == "__hostDelay__"][0]

    assert host_delay.extras["observed_gap_us"] == 40
    assert host_delay.extras["boundary_origin_kind"] == "mixed_or_unresolved"
    assert host_delay.extras["boundary_origin_conflicting_fields"] == {
        "boundary_origin_kind": {
            "prev": "fake_wrapper_instrumentation_only",
            "current": "paper_visible_user_framework_cpu",
        }
    }
    assert host_delay.extras["boundary_origin_prev_fields"] == {
        "boundary_origin_kind": "fake_wrapper_instrumentation_only",
        "instrumentation_only_duration_us": 4.0,
    }
    assert host_delay.extras["boundary_origin_current_fields"] == {
        "boundary_origin_kind": "paper_visible_user_framework_cpu",
        "paper_visible_host_duration_us": 36.0,
    }


def test_collate_derives_actual_counterpart_and_launch_metadata_without_default_off_visibility_fields():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.REAL,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={
                    "kernel": "kernel_a",
                    "grid_x": 2,
                    "grid_y": 3,
                    "grid_z": 4,
                    "block_x": 5,
                    "block_y": 6,
                    "block_z": 7,
                    "shared_mem": 128,
                    "stream_id": "9",
                },
            ),
        ),
    )
    bundle = TraceBundle(trace_dir=Path("/tmp/manual"), source=TraceSource.REAL, rank_traces=(rank_trace,))

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]
    launch = [event for event in events if event.api == "cudaLaunchKernel"][0]

    assert host_delay.extras["actual_counterpart_component_id"] == "host_inter_op_overhead"
    assert host_delay.extras["actual_inter_host_op_gap_us"] == 40.0
    assert host_delay.extras["actual_counterpart_rank"] == 0
    assert host_delay.extras["actual_counterpart_prev_event_id"] == "r0:e0"
    assert host_delay.extras["actual_counterpart_current_event_id"] == "r0:e1"
    assert host_delay.extras["actual_counterpart_boundary_family"] == "cudaGetDevice -> cudaLaunchKernel"
    assert host_delay.extras["actual_counterpart_visibility_kind"] == "mixed_or_unresolved"
    assert host_delay.extras["actual_host_dispatch_duration_us"] is None
    assert host_delay.extras["actual_launch_control_dispatch_us"] is None
    assert not (_HOST_CONTROL_DEFAULT_OFF_DIAGNOSTIC_FIELDS & host_delay.extras.keys())

    assert launch.extras["launch_kernel_name"] == "kernel_a"
    assert launch.extras["launch_grid"] == "2x3x4"
    assert launch.extras["launch_block"] == "5x6x7"
    assert launch.extras["launch_shared_mem_bytes"] == 128
    assert launch.extras["launch_stream_id"] == "9"


def test_collate_opt_in_host_control_envelope_counterpart_metadata(monkeypatch):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.setenv("MAYA_ENABLE_HOST_CONTROL_ENVELOPE_COUNTERPART_DIAGNOSTICS", "1")
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
                extras={"end_ts": 104, "host_duration_us": 4.0},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.REAL,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"stream_id": "7", "end_ts": 145, "host_duration_us": 5.0},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank_trace,),
        trace_window="step",
        step_windows={0: (90, 200)},
    )

    host_delay = [
        event for event in collate_trace_bundle(bundle).rank_events[0]
        if event.api == "__hostDelay__"
        and event.extras.get("raw_prev_api") == "cudaGetDevice"
        and event.extras.get("raw_current_api") == "cudaLaunchKernel"
    ][0]
    key = "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:1"

    assert host_delay.extras["observed_gap_us"] == 40
    assert host_delay.extras["host_control_envelope_counterpart_schema_version"] == (
        "host_control_replay_envelope_counterpart_metadata_v1"
    )
    assert host_delay.extras["host_control_envelope_counterpart_opt_in_flag"] is True
    assert host_delay.extras["host_control_envelope_counterpart_key"] == key
    assert host_delay.extras["hostdelay_counterpart_key"] == key
    assert host_delay.extras["hostdelay_interval_id"] == "r0:h1"
    assert host_delay.extras["host_control_envelope_interval_start_ts_us"] == 100
    assert host_delay.extras["host_control_envelope_interval_end_ts_us"] == 140
    assert host_delay.extras["host_control_envelope_interval_duration_us"] == 40.0
    assert host_delay.extras["host_control_envelope_visibility_kind"] == (
        "mixed_or_unresolved"
    )
    assert host_delay.extras["host_control_envelope_replay_overlap_status"] == (
        "unavailable"
    )
    assert host_delay.extras["paper_valid_window_id"] == "rank0:step_window"
    assert host_delay.extras["safe_to_use_as_repair_evidence"] is False
    assert host_delay.extras["safe_to_use_as_subtraction_delta"] is False


def test_collate_opt_in_gemm_adjacent_hostdelay_metadata_without_duration_change(
    monkeypatch,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    def make_bundle() -> TraceBundle:
        rank_trace = RankTrace(
            rank=0,
            path=Path("/tmp/rank_0.jsonl"),
            source=TraceSource.REAL,
            events=(
                TraceEvent(
                    rank=0,
                    ordinal=0,
                    source=TraceSource.REAL,
                    ts=100,
                    pid=1,
                    tid=2,
                    module="libcublas.so.12",
                    api="cublasSetStream_v2",
                    op_type="stream_op",
                    extras={"stream_id": "9", "end_ts": 104, "host_duration_us": 4.0},
                ),
                TraceEvent(
                    rank=0,
                    ordinal=1,
                    source=TraceSource.REAL,
                    ts=140,
                    pid=1,
                    tid=2,
                    module="libcublas.so.12",
                    api="cublasGemmEx",
                    op_type="blas_compute",
                    extras={
                        "stream_id": "9",
                        "end_ts": 146,
                        "host_duration_us": 6.0,
                        "m": "256",
                        "n": "128",
                        "k": "64",
                        "computeType": "68",
                        "transa": "1",
                        "transb": "0",
                        "algorithm": "23",
                    },
                ),
            ),
        )
        return TraceBundle(
            trace_dir=Path("/tmp/manual"),
            source=TraceSource.REAL,
            rank_traces=(rank_trace,),
            trace_window="step",
            step_windows={0: (90, 200)},
        )

    monkeypatch.delenv(
        "MAYA_ENABLE_GEMM_HOSTDELAY_BOUNDARY_COUNTERPART_DIAGNOSTICS",
        raising=False,
    )
    baseline_host_delay = [
        event
        for event in collate_trace_bundle(make_bundle()).rank_events[0]
        if event.api == "__hostDelay__"
        and event.extras.get("raw_prev_api") == "cublasSetStream_v2"
        and event.extras.get("raw_current_api") == "cublasGemmEx"
    ][0]

    monkeypatch.setenv(
        "MAYA_ENABLE_GEMM_HOSTDELAY_BOUNDARY_COUNTERPART_DIAGNOSTICS",
        "1",
    )
    diagnostic_host_delay = [
        event
        for event in collate_trace_bundle(make_bundle()).rank_events[0]
        if event.api == "__hostDelay__"
        and event.extras.get("raw_prev_api") == "cublasSetStream_v2"
        and event.extras.get("raw_current_api") == "cublasGemmEx"
    ][0]

    assert diagnostic_host_delay.extras["observed_gap_us"] == baseline_host_delay.extras[
        "observed_gap_us"
    ]
    assert not any(
        key.startswith("gemm_adjacent_")
        for key in baseline_host_delay.extras
    )
    assert diagnostic_host_delay.extras["gemm_adjacent_hostdelay_schema_version"] == (
        "gemm_adjacent_hostdelay_boundary_counterpart_visibility_count_once_metadata_v1"
    )
    assert diagnostic_host_delay.extras["gemm_adjacent_hostdelay_opt_in_flag"] is True
    assert diagnostic_host_delay.extras["gemm_adjacent_stable_boundary_row_id"] == (
        "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:1"
    )
    assert diagnostic_host_delay.extras["gemm_adjacent_materialized_hostdelay_event_id"] == (
        diagnostic_host_delay.id
    )
    assert diagnostic_host_delay.extras["gemm_adjacent_target_gemm_api"] == "cublasGemmEx"
    assert diagnostic_host_delay.extras["gemm_adjacent_adjacent_api"] == "cublasSetStream_v2"
    assert diagnostic_host_delay.extras["gemm_adjacent_gemm_shape_signature"] == (
        "m=256;n=128;k=64;compute_type=68;transa=1;transb=0;algorithm=23"
    )
    assert diagnostic_host_delay.extras["gemm_adjacent_actual_timing_status"] == "unavailable"
    assert (
        diagnostic_host_delay.extras[
            "gemm_adjacent_actual_endpoint_timestamps_used_as_strict_timing"
        ]
        is False
    )
    assert diagnostic_host_delay.extras["gemm_adjacent_actual_runtime_direct_substitution"] is False
    assert diagnostic_host_delay.extras["gemm_adjacent_count_once_status"] == "unavailable"
    assert diagnostic_host_delay.extras["gemm_adjacent_count_once_non_overlap_status"] == (
        "unavailable"
    )
    assert diagnostic_host_delay.extras["gemm_adjacent_wait_map_safety_status"] == (
        "unavailable"
    )
    assert diagnostic_host_delay.extras["gemm_adjacent_safe_to_use_as_repair_evidence"] is False
    assert diagnostic_host_delay.extras["gemm_adjacent_safe_to_use_as_subtraction_delta"] is False
    assert diagnostic_host_delay.extras["safe_to_use_as_repair_evidence"] is False
    assert diagnostic_host_delay.extras["safe_to_use_as_subtraction_delta"] is False


def test_collate_cuda_gemm_hostdispatch_strict_occurrence_gap_default_off_absent(
    monkeypatch,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    for key in (
        "MAYA_ENABLE_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_METADATA_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_METADATA_DIAGNOSTICS",
    ):
        monkeypatch.delenv(key, raising=False)

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=100,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"stream_id": "9"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.REAL,
                ts=140,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasGemmEx",
                op_type="blas_compute",
                extras={
                    "stream_id": "9",
                    "m": "256",
                    "n": "128",
                    "k": "64",
                    "computeType": "68",
                    "transa": "1",
                    "transb": "0",
                    "algorithm": "23",
                },
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank_trace,),
        trace_window="step",
        step_windows={0: (90, 200)},
    )

    host_delay = [
        event
        for event in collate_trace_bundle(bundle).rank_events[0]
        if event.api == "__hostDelay__"
        and event.extras.get("raw_prev_api") == "cublasSetStream_v2"
        and event.extras.get("raw_current_api") == "cublasGemmEx"
    ][0]

    assert host_delay.extras["observed_gap_us"] == 39
    forbidden_fields = {
        "cuda_gemm_hostdispatch_strict_occurrence_gap_schema_version",
        "cuda_gemm_hostdispatch_strict_occurrence_gap_opt_in_flag",
        "cuda_gemm_hostdispatch_strict_occurrence_gap_predicted_row_id",
        "strict_occurrence_common_basis_key",
        "actual_mechanical_dispatch_split_status",
        "actual_endpoint_timestamps_used_as_dispatch_split",
        "repair_ready",
        "safe_to_use_for_runtime_substitution",
    }
    assert not any(field in host_delay.extras for field in forbidden_fields)


@pytest.mark.parametrize(
    "env_key",
    [
        "MAYA_ENABLE_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_METADATA_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_METADATA_DIAGNOSTICS",
    ],
)
def test_collate_cuda_gemm_hostdispatch_strict_occurrence_gap_opt_in_gap_unchanged(
    monkeypatch,
    env_key,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    def make_bundle() -> TraceBundle:
        rank_trace = RankTrace(
            rank=0,
            path=Path("/tmp/rank_0.jsonl"),
            source=TraceSource.REAL,
            events=(
                TraceEvent(
                    rank=0,
                    ordinal=0,
                    source=TraceSource.REAL,
                    ts=100,
                    pid=1,
                    tid=2,
                    module="libcublas.so.12",
                    api="cublasSetStream_v2",
                    op_type="stream_op",
                    extras={"stream_id": "9"},
                ),
                TraceEvent(
                    rank=0,
                    ordinal=1,
                    source=TraceSource.REAL,
                    ts=140,
                    pid=1,
                    tid=2,
                    module="libcublas.so.12",
                    api="cublasGemmEx",
                    op_type="blas_compute",
                    extras={
                        "stream_id": "9",
                        "m": "256",
                        "n": "128",
                        "k": "64",
                        "computeType": "68",
                        "transa": "1",
                        "transb": "0",
                        "algorithm": "23",
                    },
                ),
            ),
        )
        return TraceBundle(
            trace_dir=Path("/tmp/manual"),
            source=TraceSource.REAL,
            rank_traces=(rank_trace,),
            trace_window="step",
            step_windows={0: (90, 200)},
        )

    for key in (
        "MAYA_ENABLE_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_METADATA_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_METADATA_DIAGNOSTICS",
    ):
        monkeypatch.delenv(key, raising=False)
    baseline_host_delay = [
        event
        for event in collate_trace_bundle(make_bundle()).rank_events[0]
        if event.api == "__hostDelay__"
        and event.extras.get("raw_prev_api") == "cublasSetStream_v2"
        and event.extras.get("raw_current_api") == "cublasGemmEx"
    ][0]

    monkeypatch.setenv(env_key, "1")
    diagnostic_host_delay = [
        event
        for event in collate_trace_bundle(make_bundle()).rank_events[0]
        if event.api == "__hostDelay__"
        and event.extras.get("raw_prev_api") == "cublasSetStream_v2"
        and event.extras.get("raw_current_api") == "cublasGemmEx"
    ][0]

    assert diagnostic_host_delay.extras["observed_gap_us"] == baseline_host_delay.extras[
        "observed_gap_us"
    ]
    assert diagnostic_host_delay.ts == baseline_host_delay.ts
    assert diagnostic_host_delay.extras[
        "cuda_gemm_hostdispatch_strict_occurrence_gap_schema_version"
    ] == "cudaLaunch_GEMM_hostdispatch_strict_occurrence_gap_metadata_v1"
    assert diagnostic_host_delay.extras[
        "cuda_gemm_hostdispatch_strict_occurrence_gap_opt_in_flag"
    ] is True
    assert diagnostic_host_delay.extras[
        "cuda_gemm_hostdispatch_strict_occurrence_gap_source_side"
    ] == "predicted_hostDelay_boundary_metadata"
    assert diagnostic_host_delay.extras["strict_occurrence_count_basis_side"] == (
        "predicted_hostDelay_boundary"
    )
    assert diagnostic_host_delay.extras[
        "cuda_gemm_hostdispatch_strict_occurrence_gap_predicted_row_id"
    ] == "rank:0:strict_occurrence_gap_hostDelay:r0:e0->r0:e1"
    assert diagnostic_host_delay.extras["api_family"] == "cublasGemmEx"
    assert diagnostic_host_delay.extras["component_role"] == "hostDelay"
    assert diagnostic_host_delay.extras["paper_valid_window_id"] == "rank0:step_window"
    assert diagnostic_host_delay.extras["api_sequence_ordinal_in_window"] == 0
    assert diagnostic_host_delay.extras["host_queue_sequence_ordinal_in_window"] == 1
    assert diagnostic_host_delay.extras["stream_sequence_ordinal_in_window"] == 1
    assert diagnostic_host_delay.extras["predicted_stream_resource_id"] == (
        "rank:0:stream:9"
    )
    assert diagnostic_host_delay.extras["material_signature"] == (
        "m=256;n=128;k=64;compute_type=68;transa=1;transb=0;algorithm=23"
    )
    assert diagnostic_host_delay.extras[
        "strict_occurrence_material_without_embedded_algo"
    ] == "m=256;n=128;k=64;compute_type=68;transa=1;transb=0;algorithm=23"
    assert diagnostic_host_delay.extras["algorithm"] == "23"
    assert diagnostic_host_delay.extras["gemm_shape_signature"] == (
        "m=256;n=128;k=64;compute_type=68;transa=1;transb=0;algorithm=23"
    )
    common_key = diagnostic_host_delay.extras["strict_occurrence_common_basis_key"]
    assert "window:" not in common_key
    assert "role:" not in common_key
    assert "api:cublasGemmEx" in common_key
    assert "api_seq:0" in common_key
    assert "host_queue_seq:1" in common_key
    assert "stream_seq:1" in common_key
    assert "algorithm:23" in common_key
    assert diagnostic_host_delay.extras["strict_occurrence_boundary_target_side"] == (
        "incoming_to_target"
    )
    assert diagnostic_host_delay.extras[
        "strict_occurrence_projection_keys_status"
    ] == "diagnostic_only_projection_not_strict_join_key"
    assert diagnostic_host_delay.extras[
        "strict_occurrence_projection_keys_repair_ready"
    ] is False
    assert diagnostic_host_delay.extras[
        "strict_occurrence_endpoint_identity_projection_key"
    ] == (
        "rank:0|queue:legacy_pid:1|api:cublasGemmEx|api_seq:0|"
        "material:m=256;n=128;k=64;compute_type=68;transa=1;transb=0;algorithm=23|algorithm:23"
    )
    assert diagnostic_host_delay.extras[
        "strict_occurrence_boundary_target_side_projection_key"
    ] == (
        "rank:0|queue:legacy_pid:1|api:cublasGemmEx|api_seq:0|"
        "material:m=256;n=128;k=64;compute_type=68;transa=1;transb=0;algorithm=23|algorithm:23|"
        "boundary_target_side:incoming_to_target"
    )
    assert diagnostic_host_delay.extras["actual_mechanical_dispatch_split_status"] == (
        "unavailable"
    )
    assert diagnostic_host_delay.extras["actual_control_dispatch_us"] is None
    assert diagnostic_host_delay.extras["actual_api_body_us"] is None
    assert diagnostic_host_delay.extras["actual_instrumentation_only_us"] is None
    assert diagnostic_host_delay.extras[
        "actual_endpoint_timestamps_used_as_dispatch_split"
    ] is False
    assert diagnostic_host_delay.extras["actual_host_duration_used_as_dispatch_split"] is False
    assert diagnostic_host_delay.extras["actual_runtime_used_as_dispatch_split"] is False
    assert diagnostic_host_delay.extras["stream_alignment_status"] == (
        "predicted_only_actual_alignment_unavailable"
    )
    assert diagnostic_host_delay.extras["count_once_status"] == (
        "metadata_only_count_once_group_not_strict_nonoverlap_proof"
    )
    assert diagnostic_host_delay.extras["nonoverlap_status"] == "unavailable"
    assert diagnostic_host_delay.extras["wait_map_safety_status"] == "unavailable"
    assert diagnostic_host_delay.extras["hostdispatch_producer_visibility_status"] == (
        "unavailable"
    )
    assert diagnostic_host_delay.extras["strict_occurrence_join_ready"] is False
    assert diagnostic_host_delay.extras[
        "strict_actual_timing_or_mechanical_split_ready"
    ] is False
    assert diagnostic_host_delay.extras["strict_apples_to_apples_delta_ready"] is False
    assert diagnostic_host_delay.extras["repair_ready"] is False
    assert diagnostic_host_delay.extras["safe_to_use_as_repair_evidence"] is False
    assert diagnostic_host_delay.extras["safe_to_use_as_subtraction_delta"] is False
    assert diagnostic_host_delay.extras["safe_to_use_for_runtime_substitution"] is False
    assert diagnostic_host_delay.extras[
        "safe_to_use_for_endpoint_timestamp_substitution"
    ] is False


def test_collate_cuda_gemm_hostdispatch_strict_occurrence_preserves_zero_algorithm(
    monkeypatch,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.setenv(
        "MAYA_ENABLE_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_METADATA_DIAGNOSTICS",
        "1",
    )

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=100,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={"stream_id": "9"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.REAL,
                ts=140,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasGemmEx",
                op_type="blas_compute",
                extras={
                    "stream_id": "9",
                    "m": "256",
                    "n": "128",
                    "k": "64",
                    "computeType": "68",
                    "transa": "1",
                    "transb": "0",
                    "algorithm": 0,
                },
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank_trace,),
        trace_window="step",
        step_windows={0: (90, 200)},
    )

    host_delay = [
        event
        for event in collate_trace_bundle(bundle).rank_events[0]
        if event.api == "__hostDelay__"
        and event.extras.get("raw_prev_api") == "cublasSetStream_v2"
        and event.extras.get("raw_current_api") == "cublasGemmEx"
    ][0]

    assert host_delay.extras["material_signature"] == (
        "m=256;n=128;k=64;compute_type=68;transa=1;transb=0;algorithm=0"
    )
    assert host_delay.extras["algorithm"] == 0
    assert "algorithm:0" in host_delay.extras["strict_occurrence_common_basis_key"]
    assert "algorithm:unavailable" not in host_delay.extras[
        "strict_occurrence_common_basis_key"
    ]
    assert host_delay.extras["strict_occurrence_endpoint_identity_projection_key"] == (
        "rank:0|queue:legacy_pid:1|api:cublasGemmEx|api_seq:0|"
        "material:m=256;n=128;k=64;compute_type=68;transa=1;transb=0;algorithm=0|algorithm:0"
    )


def test_collate_joined_gemm_stream_queue_wait_actual_metadata_sidecar_copy_through():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    joined_stream_metadata = {
        "joined_gemm_stream_queue_wait_actual_counterpart_schema_version": (
            "joined_gemm_stream_queue_wait_actual_counterpart_metadata_v1"
        ),
        "joined_gemm_stream_queue_wait_actual_counterpart_opt_in_flag": True,
        "joined_gemm_stream_queue_wait_source_side": (
            "actual_stream_order_endpoint_metadata"
        ),
        "joined_gemm_stream_queue_wait_actual_row_id": (
            "rank:0:joined_gemm_stream_queue_wait_actual:raw_ordinal:1"
        ),
        "joined_gemm_stream_queue_wait_actual_rank": 0,
        "joined_gemm_stream_queue_wait_actual_api": "cublasGemmEx",
        "joined_gemm_stream_queue_wait_actual_raw_event_id": "rank:0:raw_ordinal:1",
        "joined_gemm_stream_queue_wait_actual_raw_ordinal": 1,
        "joined_gemm_stream_queue_wait_actual_stream_id": "9",
        "joined_gemm_stream_queue_wait_actual_stream_resource_id": "rank:0:stream:9",
        "joined_gemm_stream_queue_wait_actual_stream_sequence_ordinal": 1,
        "joined_gemm_stream_queue_wait_previous_same_stream_raw_event_id": (
            "rank:0:raw_ordinal:0"
        ),
        "joined_gemm_stream_queue_wait_previous_same_stream_api": "cudaLaunchKernel",
        "joined_gemm_stream_queue_wait_previous_device_predecessor_raw_event_id": (
            "rank:0:raw_ordinal:0"
        ),
        "joined_gemm_stream_queue_wait_previous_device_predecessor_api": (
            "cudaLaunchKernel"
        ),
        "joined_gemm_stream_queue_wait_previous_device_predecessor_material_signature": (
            "kernel=void fused_kernel;grid=1x1x1;block=64x1x1;shared_mem=0"
        ),
        "joined_gemm_stream_queue_wait_previous_device_predecessor_algorithm": None,
        "joined_gemm_stream_queue_wait_previous_device_predecessor_status": (
            "available_previous_same_stream_device_predecessor_gap_unreviewed_clock"
        ),
        "joined_gemm_stream_queue_wait_previous_device_predecessor_source": (
            "rank_local_previous_same_stream_cupti_backed_device_predecessor"
        ),
        "joined_gemm_stream_queue_wait_actual_stream_order_pair_id": (
            "rank:0:stream:9:previous:rank:0:raw_ordinal:0->current:rank:0:raw_ordinal:1"
        ),
        "joined_gemm_stream_queue_wait_actual_release_timing_status": "unavailable",
        "joined_gemm_stream_queue_wait_actual_wait_timing_status": "unavailable",
        "joined_gemm_stream_queue_wait_actual_wait_start_us": None,
        "joined_gemm_stream_queue_wait_actual_release_us": None,
        "joined_gemm_stream_queue_wait_actual_waited_us": None,
        "joined_gemm_stream_queue_wait_actual_previous_kernel_start_cupti_timestamp": (
            "900910"
        ),
        "joined_gemm_stream_queue_wait_previous_device_predecessor_kernel_start_cupti_timestamp": (
            "900910"
        ),
        "joined_gemm_stream_queue_wait_previous_device_predecessor_kernel_end_cupti_timestamp": (
            "901000"
        ),
        "joined_gemm_stream_queue_wait_previous_device_predecessor_cupti_kernel_stream_id": (
            "17"
        ),
        "joined_gemm_stream_queue_wait_previous_device_predecessor_cupti_kernel_stream_id_pair_status": (
            "same_cupti_stream_id_observed"
        ),
        "joined_gemm_stream_queue_wait_previous_device_predecessor_stream_order_gap_cupti_ticks": (
            16
        ),
        "joined_gemm_stream_queue_wait_actual_previous_kernel_end_cupti_timestamp": (
            "901000"
        ),
        "joined_gemm_stream_queue_wait_actual_current_kernel_start_cupti_timestamp": (
            "901016"
        ),
        "joined_gemm_stream_queue_wait_actual_previous_cupti_kernel_stream_id": "17",
        "joined_gemm_stream_queue_wait_actual_current_cupti_kernel_stream_id": "17",
        "joined_gemm_stream_queue_wait_actual_cupti_kernel_stream_id_pair_status": (
            "same_cupti_stream_id_observed"
        ),
        "joined_gemm_stream_queue_wait_actual_stream_order_gap_cupti_ticks": 16,
        "joined_gemm_stream_queue_wait_endpoint_timestamps_used_as_wait_release": False,
        "joined_gemm_stream_queue_wait_endpoint_timestamps_used_as_strict_delta": False,
        "joined_gemm_stream_queue_wait_strict_delta_calculable": False,
        "joined_gemm_stream_queue_wait_wait_map_safety_status": "unavailable",
        "joined_gemm_stream_queue_wait_wait_map_safety_proven": False,
        "joined_gemm_stream_queue_wait_repair_ready": False,
        "joined_gemm_stream_queue_wait_safe_to_use_as_repair_evidence": False,
        "joined_gemm_stream_queue_wait_safe_to_use_as_subtraction_delta": False,
    }
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"stream_id": "9"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.REAL,
                ts=140,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasGemmEx",
                op_type="blas_compute",
                extras={
                    "stream_id": "9",
                    "m": "256",
                    "n": "128",
                    "k": "64",
                    "computeType": "68",
                    "transa": "1",
                    "transb": "0",
                    "algorithm": "23",
                    **joined_stream_metadata,
                },
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank_trace,),
        trace_window="step",
        step_windows={0: (90, 200)},
    )

    host_delay = [
        event
        for event in collate_trace_bundle(bundle).rank_events[0]
        if event.api == "__hostDelay__"
        and event.extras.get("raw_prev_api") == "cudaLaunchKernel"
        and event.extras.get("raw_current_api") == "cublasGemmEx"
    ][0]

    current_fields = host_delay.extras["boundary_origin_current_fields"]
    assert current_fields[
        "joined_gemm_stream_queue_wait_actual_counterpart_schema_version"
    ] == "joined_gemm_stream_queue_wait_actual_counterpart_metadata_v1"
    assert current_fields[
        "joined_gemm_stream_queue_wait_actual_stream_order_pair_id"
    ] == joined_stream_metadata[
        "joined_gemm_stream_queue_wait_actual_stream_order_pair_id"
    ]
    assert current_fields[
        "joined_gemm_stream_queue_wait_endpoint_timestamps_used_as_wait_release"
    ] is False
    assert current_fields["joined_gemm_stream_queue_wait_actual_wait_start_us"] is None
    assert current_fields["joined_gemm_stream_queue_wait_actual_release_us"] is None
    assert current_fields["joined_gemm_stream_queue_wait_actual_waited_us"] is None
    assert current_fields[
        "joined_gemm_stream_queue_wait_actual_previous_kernel_start_cupti_timestamp"
    ] == "900910"
    assert current_fields[
        "joined_gemm_stream_queue_wait_actual_stream_order_gap_cupti_ticks"
    ] == 16
    assert current_fields[
        "joined_gemm_stream_queue_wait_previous_device_predecessor_api"
    ] == "cudaLaunchKernel"
    assert current_fields[
        "joined_gemm_stream_queue_wait_previous_device_predecessor_stream_order_gap_cupti_ticks"
    ] == 16
    assert current_fields[
        "joined_gemm_stream_queue_wait_actual_cupti_kernel_stream_id_pair_status"
    ] == "same_cupti_stream_id_observed"
    assert current_fields[
        "joined_gemm_stream_queue_wait_strict_delta_calculable"
    ] is False
    assert current_fields["joined_gemm_stream_queue_wait_repair_ready"] is False
    assert (
        "joined_gemm_stream_queue_wait_actual_counterpart_schema_version"
        not in host_delay.extras
    )
    assert host_delay.extras["observed_gap_us"] == 40


def test_collate_opt_in_launch_neighborhood_equivalence_metadata(monkeypatch):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.setenv("MAYA_ENABLE_LAUNCH_NEIGHBORHOOD_EQUIVALENCE_DIAGNOSTICS", "1")
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
                extras={"end_ts": 104, "host_duration_us": 4.0},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.REAL,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"stream_id": "7", "end_ts": 145, "host_duration_us": 5.0},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank_trace,),
        trace_window="step",
        step_windows={0: (90, 200)},
    )

    host_delay = [
        event for event in collate_trace_bundle(bundle).rank_events[0]
        if event.api == "__hostDelay__"
        and event.extras.get("raw_prev_api") == "cudaGetDevice"
        and event.extras.get("raw_current_api") == "cudaLaunchKernel"
    ][0]

    assert host_delay.extras["launch_neighborhood_equivalence_schema_version"] == (
        "launch_neighborhood_occurrence_equivalence_diagnostics_v1"
    )
    assert host_delay.extras["launch_neighborhood_equivalence_opt_in_flag"] is True
    assert host_delay.extras["launch_neighborhood_occurrence_id"] == (
        "rank:0:launch_neighborhood:raw_ordinal:0->raw_ordinal:1"
    )
    assert host_delay.extras["launch_neighborhood_role"] == (
        "predicted_materialized_hostdelay_boundary"
    )
    assert host_delay.extras["launch_neighborhood_normalized_signature"] == (
        "unresolved_wrapper_control_cpu_work -> paper_visible_operation_boundary"
    )
    assert host_delay.extras["launch_neighborhood_prev_raw_event_id"] == "r0:e0"
    assert host_delay.extras["launch_neighborhood_current_raw_event_id"] == "r0:e1"
    assert host_delay.extras["launch_neighborhood_prev_api_visibility_label"] == (
        "unresolved_wrapper_control_cpu_work"
    )
    assert host_delay.extras["launch_neighborhood_current_api_visibility_label"] == (
        "paper_visible_operation_boundary"
    )
    assert host_delay.extras["launch_neighborhood_boundary_exclusion_reasons"] == [
        "prev:cudaGetDevice:wrapper_control_visibility_unresolved",
        "current:cudaLaunchKernel:paper_visible_operation_boundary",
    ]
    assert host_delay.extras["launch_neighborhood_paper_valid_window_id"] == (
        "rank0:step_window"
    )
    assert host_delay.extras["launch_neighborhood_host_dispatch_queue_id"] == (
        "legacy_pid:1"
    )
    assert host_delay.extras["launch_neighborhood_stream_id"] == "7"
    assert host_delay.extras["launch_neighborhood_wait_map_nonoverlap_status"] == (
        "unavailable"
    )
    assert host_delay.extras["launch_neighborhood_safe_to_use_as_repair_evidence"] is False
    assert host_delay.extras["host_control_envelope_counterpart_opt_in_flag"] is True


def test_collate_forwards_actual_counterpart_fields_to_host_delay():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
                extras={
                    "actual_counterpart_component_id": "host_inter_op_overhead",
                    "actual_counterpart_visibility_kind": "mixed_or_unresolved",
                    "actual_inter_host_op_gap_us": 40.0,
                },
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.REAL,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={
                    "actual_host_dispatch_duration_us": 3.0,
                    "actual_counterpart_component_id": "host_dispatch_overhead",
                    "actual_counterpart_visibility_kind": "mixed_or_unresolved",
                    "launch_kernel_name": "kernel_a",
                },
            ),
        ),
    )
    bundle = TraceBundle(trace_dir=Path("/tmp/manual"), source=TraceSource.REAL, rank_traces=(rank_trace,))

    host_delay = [event for event in collate_trace_bundle(bundle).rank_events[0] if event.api == "__hostDelay__"][0]

    assert host_delay.extras["actual_counterpart_component_id"] == "host_inter_op_overhead"
    assert host_delay.extras["actual_counterpart_visibility_kind"] == "mixed_or_unresolved"
    assert host_delay.extras["actual_inter_host_op_gap_us"] == 40.0
    assert host_delay.extras["actual_host_dispatch_duration_us"] == 3.0
    assert host_delay.extras["boundary_origin_field_sources"] == {
        "actual_counterpart_visibility_kind": ["current", "prev"],
        "actual_inter_host_op_gap_us": ["prev"],
        "actual_host_dispatch_duration_us": ["current"],
    }


def test_collate_folds_cublas_set_stream_context_query_boundary_counterpart_metadata():
    from flexsim.maya_lite.schema import FidelityWindow, RankTrace, TraceBundle, TraceEvent, TraceSource

    schema_version = "host_control_boundary_visibility_unblocker_v2_row_evidence_v1"
    leading_occurrence_id = "rank:0:host_control_boundary:leading->raw_ordinal:0"
    materialized_occurrence_id = "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:1"

    def writer_shaped_row(
        *,
        ordinal: int,
        ts: int,
        end_ts: int,
        api: str,
        event_type: str,
        occurrence_id: str,
        previous_raw_event_id: str | None,
        previous_api: str | None,
        family: str,
    ) -> dict[str, object]:
        raw_event_id = f"rank:0:raw_ordinal:{ordinal}"
        return {
            "ts": ts,
            "pid": 33,
            "tid": 44,
            "mod": "libcudart.so.12",
            "api": api,
            "type": event_type,
            "end_ts": end_ts,
            "host_duration_us": float(end_ts - ts),
            "host_machine_id": "host-a",
            "host_dispatch_queue_id": "host-a:rank:0",
            "host_control_boundary_counterpart_schema_version": schema_version,
            "host_control_boundary_row_id": raw_event_id,
            "host_control_boundary_occurrence_id": occurrence_id,
            "host_control_boundary_selection_status": "selected_family",
            "host_control_boundary_prev_raw_event_id": previous_raw_event_id,
            "host_control_boundary_current_raw_event_id": raw_event_id,
            "host_control_boundary_prev_api": previous_api,
            "host_control_boundary_current_api": api,
            "host_control_boundary_family": family,
            "actual_counterpart_id": occurrence_id,
            "actual_counterpart_status": (
                "actual_boundary_row_id_exported_selected_occurrence_join_not_attempted"
            ),
            "actual_counterpart_window_id": None,
            "actual_counterpart_window_unavailable_reason": (
                "paper_valid_step_window_resolved_later_by_collate_or_ledger"
            ),
            "actual_counterpart_prev_event_id": previous_raw_event_id,
            "actual_counterpart_current_event_id": raw_event_id,
            "actual_counterpart_boundary_family": family,
            "actual_counterpart_dispatch_queue_id": "host-a:rank:0",
            "actual_counterpart_visibility_kind": "mixed_or_unresolved",
            "host_control_visibility_split_status": "unavailable",
            "mechanical_visibility_split_status": "unavailable",
            "safe_to_use_as_subtraction_delta": False,
            "safe_to_use_as_repair_evidence": False,
        }

    rows = (
        writer_shaped_row(
            ordinal=0,
            ts=1_000_000,
            end_ts=1_000_003,
            api="cudaGetDevice",
            event_type="context_op",
            occurrence_id=leading_occurrence_id,
            previous_raw_event_id=None,
            previous_api=None,
            family="cudaGetDevice",
        ),
        writer_shaped_row(
            ordinal=1,
            ts=1_000_009,
            end_ts=1_000_014,
            api="cublasSetStream_v2",
            event_type="stream_op",
            occurrence_id=materialized_occurrence_id,
            previous_raw_event_id="rank:0:raw_ordinal:0",
            previous_api="cudaGetDevice",
            family="cudaGetDevice -> cublasSetStream_v2",
        ),
    )
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=tuple(
            TraceEvent.from_json_record(row, rank=0, ordinal=ordinal, source=TraceSource.REAL)
            for ordinal, row in enumerate(rows)
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank_trace,),
        trace_window="step",
        step_windows={0: (999_990, 1_000_100)},
        fidelity_windows={
            0: FidelityWindow(
                start_ts=999_990,
                end_ts=1_000_100,
                source="manifest",
                is_paper_valid_step_window=True,
            )
        },
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    cublas_set_stream = [event for event in events if event.api == "cublasSetStream_v2"][0]

    assert not [
        event
        for event in events
        if event.api == "__hostDelay__"
        and event.extras.get("raw_prev_api") == "cudaGetDevice"
        and event.extras.get("raw_current_api") == "cublasSetStream_v2"
    ]
    assert cublas_set_stream.extras["host_control_boundary_occurrence_id"] == (
        materialized_occurrence_id
    )
    assert cublas_set_stream.extras["actual_counterpart_id"] == materialized_occurrence_id
    assert cublas_set_stream.extras["actual_counterpart_window_id"] is None
    assert cublas_set_stream.extras["actual_counterpart_window_unavailable_reason"] == (
        "paper_valid_step_window_resolved_later_by_collate_or_ledger"
    )
    _assert_cublas_context_query_suffix_fold(
        cublas_set_stream,
        folded_count=1,
        suppressed_host_gap_us=9,
        preserved_pre_suffix_gap_us=10,
    )
    assert (
        cublas_set_stream.extras["cublas_set_stream_context_query_suffix_terminal_gap_us"]
        == 9
    )
    folded_row = cublas_set_stream.extras["cublas_set_stream_context_query_suffix_folded_rows"][0]
    assert folded_row["raw_event_id"] == "r0:e0"
    assert folded_row["raw_api"] == "cudaGetDevice"
    assert folded_row["raw_extras"]["host_control_boundary_occurrence_id"] == (
        leading_occurrence_id
    )
    assert folded_row["raw_extras"]["actual_counterpart_id"] == leading_occurrence_id
    terminal_gap = cublas_set_stream.extras["cublas_set_stream_context_query_suffix_gap_rows"][0]
    assert terminal_gap["gap_kind"] == "context_query_suffix_terminal_gap_to_cublasSetStream_v2"
    assert terminal_gap["raw_prev_api"] == "cudaGetDevice"
    assert terminal_gap["raw_current_api"] == "cublasSetStream_v2"
    assert terminal_gap["host_gap_contribution"] == "excluded_from_pending_hostdelay"
    assert cublas_set_stream.extras["safe_to_use_as_subtraction_delta"] is False
    assert cublas_set_stream.extras["safe_to_use_as_repair_evidence"] is False


def test_collate_preserves_prepop_pending_gap_when_launch_config_edge_is_suppressed_default_on(
    monkeypatch,
):
    from flexsim.maya_lite.schema import FidelityWindow, RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.delenv("MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("FLEXSIM_MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("FLEXSIM_MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=120,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="__cudaPopCallConfiguration",
                op_type="other",
                extras={"end_ts": 124, "host_duration_us": 4.0},
            ),
            TraceEvent(
                rank=0,
                ordinal=2,
                source=TraceSource.FAKE,
                ts=150,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"actual_launch_api_body_us": 3.5, "host_duration_us": 10.0},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
        trace_window="step",
        step_windows={0: (90, 200)},
        fidelity_windows={
            0: FidelityWindow(
                start_ts=90,
                end_ts=200,
                source="manifest",
                is_paper_valid_step_window=True,
            )
        },
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [
        event
        for event in events
        if event.api == "__hostDelay__" and event.extras.get("hostdelay_source") == "collate_host_gap"
    ][0]
    launch = [event for event in events if event.api == "cudaLaunchKernel"][0]

    assert [event.api for event in events] == ["__hostDelay__", "cudaLaunchKernel", "__hostDelay__"]
    assert host_delay.extras["observed_gap_us"] == 20
    assert host_delay.extras["raw_prev_event_id"] == "r0:e0"
    assert host_delay.extras["raw_prev_api"] == "cudaGetDevice"
    assert host_delay.extras["raw_prev_ts_us"] == 100
    assert host_delay.extras["raw_prev_end_ts_us"] is None
    assert host_delay.extras["raw_prev_end_ts_source"] == "unavailable"
    assert host_delay.extras["raw_prev_host_duration_us"] is None
    assert host_delay.extras["raw_current_event_id"] == "r0:e1"
    assert host_delay.extras["raw_current_api"] == "__cudaPopCallConfiguration"
    assert host_delay.extras["raw_current_ts_us"] == 120
    assert host_delay.extras["raw_current_end_ts_us"] == 124.0
    assert host_delay.extras["raw_current_end_ts_source"] == "raw_end_ts"
    assert host_delay.extras["raw_current_host_duration_us"] == 4.0
    assert host_delay.extras["hostdelay_source"] == "collate_host_gap"
    assert host_delay.extras["raw_boundary_family"] == (
        "cudaGetDevice -> __cudaPopCallConfiguration"
    )
    assert host_delay.extras["previous_materialized_event_id"] is None
    assert host_delay.extras["previous_materialized_api"] is None
    assert host_delay.extras["current_materialized_event_id"] == "r0:e2"
    assert host_delay.extras["current_materialized_api"] == "cudaLaunchKernel"
    assert host_delay.extras["materialized_boundary_family"] == "cudaLaunchKernel"
    assert host_delay.extras["boundary_visibility_kind"] == "unavailable"
    assert host_delay.extras["paper_valid_window_membership"] == {
        "in_paper_valid_window": True,
        "window_id": "rank0:step_window",
        "window_source": "manifest",
        "start_ts": 90,
        "end_ts": 200,
        "is_paper_valid_step_window": True,
        "membership_basis": "collate_step_window",
        "unavailable_reason": None,
    }
    assert "safe_to_use_as_subtraction_delta" not in host_delay.extras
    assert "safe_to_use_as_repair_evidence" not in host_delay.extras
    assert "wait_map_safety_status" not in host_delay.extras
    assert host_delay.extras["actual_launch_api_body_us"] is None
    assert launch.extras["actual_launch_api_body_us"] == 3.5
    _assert_launch_config_metadata(
        launch,
        adjacent_gap_us=30,
        contribution="excluded_from_pending_host_gap",
        normalization_enabled=True,
        normalization_status="enabled_default_excluded_from_hostdelay",
        raw_event_id="r0:e1",
        raw_ts_us=120,
    )


def test_collate_opt_in_folds_context_query_suffix_before_launch_config_pop(
    monkeypatch,
):
    from flexsim.maya_lite.schema import FidelityWindow, RankTrace, TraceBundle, TraceEvent, TraceSource

    _enable_context_query_suffix_launch_config_pop_fold(monkeypatch)
    _disable_context_query_suffix_event_record_fold(monkeypatch)
    monkeypatch.delenv("MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("FLEXSIM_MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("FLEXSIM_MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=120,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="__cudaPopCallConfiguration",
                op_type="other",
                extras={"end_ts": 124, "host_duration_us": 4.0},
            ),
            TraceEvent(
                rank=0,
                ordinal=2,
                source=TraceSource.FAKE,
                ts=150,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"actual_launch_api_body_us": 3.5, "host_duration_us": 10.0},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
        trace_window="step",
        step_windows={0: (90, 200)},
        fidelity_windows={
            0: FidelityWindow(
                start_ts=90,
                end_ts=200,
                source="manifest",
                is_paper_valid_step_window=True,
            )
        },
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delays = [event for event in events if event.api == "__hostDelay__"]
    launch = [event for event in events if event.api == "cudaLaunchKernel"][0]

    assert [event.api for event in events] == ["__hostDelay__", "cudaLaunchKernel", "__hostDelay__"]
    assert [event.extras["observed_gap_us"] for event in host_delays] == [10, 50]
    assert not any(
        event.extras.get("raw_boundary_family") == "cudaGetDevice -> __cudaPopCallConfiguration"
        for event in host_delays
    )
    _assert_launch_config_metadata(
        launch,
        adjacent_gap_us=30,
        contribution="excluded_from_pending_host_gap",
        normalization_enabled=True,
        normalization_status="enabled_default_excluded_from_hostdelay",
        raw_event_id="r0:e1",
        raw_ts_us=120,
    )
    assert launch.extras[
        "hostdelay_semantic_predecessor_control_query_suffix_fold_basis"
    ] == "cudaGetDevice_context_query_suffix_before___cudaPopCallConfiguration"
    assert launch.extras[
        "hostdelay_semantic_predecessor_control_query_suffix_fold_status"
    ] == "applied_opt_in_launch_config_pop_internal_metadata_fold"
    assert launch.extras[
        "hostdelay_semantic_predecessor_control_query_suffix_target_api"
    ] == "__cudaPopCallConfiguration"
    assert launch.extras[
        "hostdelay_semantic_predecessor_control_query_suffix_target_role"
    ] == "internal_launch_config_pop"
    assert launch.extras[
        "hostdelay_semantic_predecessor_control_query_suffix_materialized_target_api"
    ] == "cudaLaunchKernel"
    assert launch.extras[
        "hostdelay_semantic_predecessor_control_query_suffix_folded_event_ids"
    ] == ["r0:e0"]
    assert launch.extras[
        "hostdelay_semantic_predecessor_control_query_suffix_terminal_gap_us"
    ] == 20
    assert launch.extras[
        "hostdelay_semantic_predecessor_control_query_suffix_suppressed_host_gap_us"
    ] == 20
    assert launch.extras[
        "hostdelay_semantic_predecessor_control_query_suffix_preserved_pre_suffix_gap_us"
    ] == 10
    assert [
        row["gap_kind"]
        for row in launch.extras[
            "hostdelay_semantic_predecessor_control_query_suffix_gap_rows"
        ]
    ] == ["context_query_suffix_terminal_gap_to___cudaPopCallConfiguration"]


def test_collate_opt_in_launch_config_pop_fold_does_not_fold_direct_launch(
    monkeypatch,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    _enable_context_query_suffix_launch_config_pop_fold(monkeypatch)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=150,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"kernel": "direct", "stream_id": 7},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]
    launch = [event for event in events if event.api == "cudaLaunchKernel"][0]

    assert [event.api for event in events] == ["__hostDelay__", "cudaLaunchKernel"]
    assert host_delay.extras["observed_gap_us"] == 50
    assert host_delay.extras["raw_boundary_family"] == "cudaGetDevice -> cudaLaunchKernel"
    assert not any(
        key.startswith("hostdelay_semantic_predecessor_control_query_suffix_")
        for key in launch.extras
    )


def test_collate_opt_in_launch_config_pop_fold_requires_following_launch(
    monkeypatch,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    _enable_context_query_suffix_launch_config_pop_fold(monkeypatch)
    _disable_context_query_suffix_event_record_fold(monkeypatch)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=120,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="__cudaPopCallConfiguration",
                op_type="other",
                extras={"end_ts": 124, "host_duration_us": 4.0},
            ),
            TraceEvent(
                rank=0,
                ordinal=2,
                source=TraceSource.FAKE,
                ts=150,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaEventRecord",
                op_type="event_record",
                extras={"event_id": "ev0", "stream_id": 3},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]
    event_record = [event for event in events if event.api == "cudaEventRecord"][0]

    assert [event.api for event in events] == ["__hostDelay__", "cudaEventRecord"]
    assert host_delay.extras["observed_gap_us"] == 50
    assert host_delay.extras["raw_boundary_family"] == "cudaGetDevice -> cudaEventRecord"
    assert not any(
        key.startswith("hostdelay_semantic_predecessor_control_query_suffix_")
        for key in event_record.extras
    )


def test_collate_hostdelay_occurrence_metadata_default_off_absent(monkeypatch):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.delenv("MAYA_ENABLE_HOSTDELAY_OCCURRENCE_METADATA_EXPORT", raising=False)
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_HOSTDELAY_OCCURRENCE_METADATA_EXPORT",
        raising=False,
    )
    monkeypatch.delenv("MAYA_ENABLE_HOSTDELAY_OCCURRENCE_JOIN_METADATA_EXPORT", raising=False)
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_HOSTDELAY_OCCURRENCE_JOIN_METADATA_EXPORT",
        raising=False,
    )
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"stream_id": "7"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaEventRecord",
                op_type="event_record",
                extras={"event_id": "ev0", "stream_id": "7"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]

    assert [event.api for event in events] == [
        "cudaLaunchKernel",
        "__hostDelay__",
        "cudaEventRecord",
    ]
    assert host_delay.extras["observed_gap_us"] == 40
    assert not (_HOSTDELAY_OCCURRENCE_METADATA_FIELDS & host_delay.extras.keys())
    assert host_delay.extras["raw_boundary_family"] == (
        "cudaLaunchKernel -> cudaEventRecord"
    )


def test_collate_hostdelay_occurrence_metadata_opt_in_additive(monkeypatch):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.setenv("MAYA_ENABLE_HOSTDELAY_OCCURRENCE_METADATA_EXPORT", "1")
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"stream_id": "7"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaEventRecord",
                op_type="event_record",
                extras={"event_id": "ev0", "stream_id": "7"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]
    event_record = [event for event in events if event.api == "cudaEventRecord"][0]

    assert [event.api for event in events] == [
        "cudaLaunchKernel",
        "__hostDelay__",
        "cudaEventRecord",
    ]
    assert host_delay.ts == 100
    assert host_delay.extras["observed_gap_us"] == 40
    assert event_record.prev_event_id == host_delay.id
    assert host_delay.extras["hostdelay_occurrence_metadata_schema_version"] == (
        "hostdelay_occurrence_metadata_export_v1"
    )
    assert host_delay.extras["hostdelay_occurrence_metadata_opt_in_flag"] is True
    assert host_delay.extras["stable_hostdelay_occurrence_id"] == (
        "rank:0:hostdelay_occurrence:raw_ordinal:0->raw_ordinal:1"
    )
    assert host_delay.extras["hostdelay_occurrence_rank"] == 0
    assert host_delay.extras["hostdelay_occurrence_pid"] == 1
    assert host_delay.extras["hostdelay_occurrence_tid"] == 2
    assert host_delay.extras["hostdelay_occurrence_interval_start_ts_us"] == 100
    assert host_delay.extras["hostdelay_occurrence_interval_end_ts_us"] == 140
    assert host_delay.extras["hostdelay_occurrence_duration_us"] == 40
    assert host_delay.extras["hostdelay_occurrence_raw_predecessor_event_id"] == "r0:e0"
    assert host_delay.extras["hostdelay_occurrence_raw_predecessor_api"] == (
        "cudaLaunchKernel"
    )
    assert host_delay.extras["hostdelay_occurrence_raw_successor_event_id"] == "r0:e1"
    assert host_delay.extras["hostdelay_occurrence_raw_successor_api"] == (
        "cudaEventRecord"
    )
    assert host_delay.extras["hostdelay_occurrence_semantic_predecessor_event_id"] == (
        "r0:e0"
    )
    assert host_delay.extras["hostdelay_occurrence_semantic_predecessor_api"] == (
        "cudaLaunchKernel"
    )
    assert host_delay.extras["hostdelay_occurrence_semantic_successor_event_id"] == (
        "r0:e1"
    )
    assert host_delay.extras["hostdelay_occurrence_semantic_successor_api"] == (
        "cudaEventRecord"
    )
    assert host_delay.extras["hostdelay_occurrence_raw_boundary_family"] == (
        "cudaLaunchKernel -> cudaEventRecord"
    )
    assert host_delay.extras["hostdelay_occurrence_semantic_boundary_family"] == (
        "cudaLaunchKernel -> cudaEventRecord"
    )
    assert host_delay.extras["hostdelay_occurrence_paper_visibility_class"] == (
        "paper_visible_by_default_or_unresolved"
    )
    assert host_delay.extras["hostdelay_occurrence_count_once_status"] == "unavailable"
    assert host_delay.extras[
        "hostdelay_occurrence_cuda_event_wait_map_safety_status"
    ] == "unavailable"
    assert host_delay.extras["hostdelay_occurrence_repair_ready"] is False
    assert host_delay.extras["hostdelay_occurrence_safe_to_use_as_repair_evidence"] is False
    assert host_delay.extras["hostdelay_occurrence_safe_to_use_as_subtraction_delta"] is False
    assert host_delay.extras["hostdelay_occurrence_safe_delta_us"] is None
    assert host_delay.extras["repair_ready"] is False
    assert host_delay.extras["safe_to_use_as_repair_evidence"] is False
    assert host_delay.extras["safe_to_use_as_subtraction_delta"] is False
    assert host_delay.extras["safe_delta_us"] is None


def test_collate_hostdelay_occurrence_join_metadata_default_off_absent_when_base_opted_in(
    monkeypatch,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.setenv("MAYA_ENABLE_HOSTDELAY_OCCURRENCE_METADATA_EXPORT", "1")
    monkeypatch.delenv("MAYA_ENABLE_HOSTDELAY_OCCURRENCE_JOIN_METADATA_EXPORT", raising=False)
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_HOSTDELAY_OCCURRENCE_JOIN_METADATA_EXPORT",
        raising=False,
    )
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"stream_id": "7"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaEventRecord",
                op_type="event_record",
                extras={"event_id": "ev0", "stream_id": "7"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]

    assert host_delay.extras["hostdelay_occurrence_metadata_opt_in_flag"] is True
    assert not (_HOSTDELAY_OCCURRENCE_JOIN_METADATA_FIELDS & host_delay.extras.keys())


@pytest.mark.parametrize(
    ("previous_api", "previous_op_type", "current_api", "current_op_type"),
    [
        ("cudaLaunchKernel", "kernel_launch", "cudaEventQuery", "stream_op"),
        ("cublasSetStream_v2", "stream_op", "cublasGemmEx", "blas_compute"),
        ("cudaLaunchKernel", "kernel_launch", "cudaEventRecord", "event_record"),
        ("ncclAllReduce", "nccl_collective", "cudaEventRecord", "event_record"),
    ],
)
def test_collate_hostdelay_occurrence_join_metadata_opt_in_additive_cases(
    monkeypatch,
    previous_api,
    previous_op_type,
    current_api,
    current_op_type,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.delenv("MAYA_ENABLE_HOSTDELAY_OCCURRENCE_METADATA_EXPORT", raising=False)
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_HOSTDELAY_OCCURRENCE_METADATA_EXPORT",
        raising=False,
    )
    monkeypatch.setenv("MAYA_ENABLE_HOSTDELAY_OCCURRENCE_JOIN_METADATA_EXPORT", "1")
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcuda.so",
                api=previous_api,
                op_type=previous_op_type,
                extras={"stream_id": "7"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcuda.so",
                api=current_api,
                op_type=current_op_type,
                extras={"event_id": "ev0", "stream_id": "7"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]
    target = [event for event in events if event.api == current_api][0]

    assert [event.api for event in events] == [
        previous_api,
        "__hostDelay__",
        current_api,
    ]
    assert host_delay.ts == 100
    assert host_delay.extras["observed_gap_us"] == 40
    assert target.prev_event_id == host_delay.id
    assert host_delay.extras["hostdelay_occurrence_metadata_opt_in_flag"] is True
    assert host_delay.extras["hostdelay_occurrence_join_metadata_schema_version"] == (
        "hostdelay_occurrence_join_metadata_export_v1"
    )
    assert host_delay.extras["hostdelay_occurrence_join_metadata_opt_in_flag"] is True
    assert host_delay.extras["hostdelay_occurrence_boundary_origin_status"] == (
        "unresolved"
    )
    assert host_delay.extras["hostdelay_occurrence_boundary_visibility_status"] == (
        "unresolved"
    )
    assert host_delay.extras["hostdelay_occurrence_count_once_status"] == "unavailable"
    assert host_delay.extras["hostdelay_occurrence_nonoverlap_status"] == "unavailable"
    assert host_delay.extras[
        "hostdelay_occurrence_cuda_event_wait_map_safety_status"
    ] == "unavailable"
    assert host_delay.extras[
        "hostdelay_occurrence_collective_wait_map_safety_status"
    ] == "unavailable"
    assert host_delay.extras["hostdelay_occurrence_join_repair_ready"] is False
    assert host_delay.extras[
        "hostdelay_occurrence_join_safe_to_use_as_repair_evidence"
    ] is False
    assert host_delay.extras[
        "hostdelay_occurrence_join_safe_to_use_as_subtraction_delta"
    ] is False
    assert host_delay.extras["hostdelay_occurrence_join_safe_delta_us"] is None
    assert host_delay.extras["hostdelay_occurrence_repair_ready"] is False
    assert host_delay.extras["hostdelay_occurrence_safe_to_use_as_subtraction_delta"] is False
    assert host_delay.extras["hostdelay_occurrence_safe_delta_us"] is None


def test_collate_collective_event_polling_metadata_default_off_absent_when_base_opted_in(
    monkeypatch,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.setenv("MAYA_ENABLE_HOSTDELAY_OCCURRENCE_METADATA_EXPORT", "1")
    monkeypatch.delenv(
        "MAYA_ENABLE_COLLECTIVE_EVENT_POLLING_BOUNDARY_METADATA_EXPORT",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_COLLECTIVE_EVENT_POLLING_BOUNDARY_METADATA_EXPORT",
        raising=False,
    )
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libnccl.so.2",
                api="ncclAllReduce",
                op_type="nccl_collective",
                extras={"stream_id": "7", "collective_group_id": "cg0"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaEventRecord",
                op_type="event_record",
                extras={"event_id": "ev0", "event_version": 1, "stream_id": "7"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]

    assert [event.api for event in events] == [
        "ncclAllReduce",
        "__hostDelay__",
        "cudaEventRecord",
    ]
    assert host_delay.extras["observed_gap_us"] == 40
    assert host_delay.extras["hostdelay_occurrence_metadata_opt_in_flag"] is True
    assert not (_COLLECTIVE_EVENT_POLLING_METADATA_FIELDS & host_delay.extras.keys())


@pytest.mark.parametrize(
    (
        "previous_api",
        "previous_op_type",
        "current_api",
        "current_op_type",
        "expected_target_family",
        "expected_collective_api",
        "expected_semantic_boundary_family",
    ),
    [
        (
            "ncclCommGetAsyncError",
            "nccl_status",
            "cudaEventRecord",
            "event_record",
            "mixed_collective_event",
            "ncclCommGetAsyncError",
            "cudaEventRecord",
        ),
        (
            "ncclAllReduce",
            "nccl_collective",
            "cudaEventRecord",
            "event_record",
            "ncclAllReduce",
            "ncclAllReduce",
            "ncclAllReduce -> cudaEventRecord",
        ),
        (
            "cudaEventQuery",
            "stream_op",
            "ncclAllReduce",
            "nccl_collective",
            "ncclAllReduce",
            "ncclAllReduce",
            "cudaEventQuery -> ncclAllReduce",
        ),
        (
            "cudaStreamWaitEvent",
            "stream_op",
            "ncclSend",
            "nccl_collective",
            "ncclSend",
            "ncclSend",
            "cudaStreamWaitEvent -> ncclSend",
        ),
        (
            "cudaStreamWaitEvent",
            "stream_op",
            "ncclRecv",
            "nccl_collective",
            "ncclRecv",
            "ncclRecv",
            "cudaStreamWaitEvent -> ncclRecv",
        ),
    ],
)
def test_collate_collective_event_polling_metadata_opt_in_additive_cases(
    monkeypatch,
    previous_api,
    previous_op_type,
    current_api,
    current_op_type,
    expected_target_family,
    expected_collective_api,
    expected_semantic_boundary_family,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.delenv("MAYA_ENABLE_HOSTDELAY_OCCURRENCE_METADATA_EXPORT", raising=False)
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_HOSTDELAY_OCCURRENCE_METADATA_EXPORT",
        raising=False,
    )
    monkeypatch.delenv("MAYA_ENABLE_HOSTDELAY_OCCURRENCE_JOIN_METADATA_EXPORT", raising=False)
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_HOSTDELAY_OCCURRENCE_JOIN_METADATA_EXPORT",
        raising=False,
    )
    monkeypatch.setenv(
        "MAYA_ENABLE_COLLECTIVE_EVENT_POLLING_BOUNDARY_METADATA_EXPORT",
        "1",
    )
    previous_extras = {
        "stream_id": "7",
        "event_id": "ev0",
        "event_version": 3,
        "collective_group_id": "cg0",
        "collective_member_id": "rank0",
        "comm_id": "comm0",
        "nranks": 16,
        "call_idx": 4,
    }
    current_extras = {
        "stream_id": "7",
        "event_id": "ev0",
        "event_version": 3,
        "collective_group_id": "cg0",
        "collective_member_id": "rank0",
        "comm_id": "comm0",
        "nranks": 16,
        "call_idx": 5,
    }
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcuda.so",
                api=previous_api,
                op_type=previous_op_type,
                extras=previous_extras,
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcuda.so",
                api=current_api,
                op_type=current_op_type,
                extras=current_extras,
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]
    target = [event for event in events if event.api == current_api][0]

    if previous_api == "ncclCommGetAsyncError":
        assert [event.api for event in events] == ["__hostDelay__", current_api]
    else:
        assert [event.api for event in events] == [
            previous_api,
            "__hostDelay__",
            current_api,
        ]
    assert host_delay.ts == 100
    assert host_delay.extras["observed_gap_us"] == 40
    assert target.prev_event_id == host_delay.id
    assert host_delay.extras["hostdelay_occurrence_metadata_opt_in_flag"] is True
    assert host_delay.extras["hostdelay_occurrence_duration_us"] == 40
    assert host_delay.extras["collective_event_polling_metadata_schema_version"] == (
        "collective_event_polling_boundary_metadata_v1"
    )
    assert host_delay.extras["collective_event_polling_metadata_opt_in_flag"] is True
    assert host_delay.extras["collective_event_polling_raw_boundary_family"] == (
        f"{previous_api} -> {current_api}"
    )
    assert host_delay.extras["collective_event_polling_semantic_boundary_family"] == (
        expected_semantic_boundary_family
    )
    assert host_delay.extras["collective_event_polling_target_family"] == (
        expected_target_family
    )
    assert host_delay.extras["collective_event_polling_collective_api"] == (
        expected_collective_api
    )
    assert host_delay.extras["collective_event_polling_collective_group_id"] == "cg0"
    assert host_delay.extras["collective_event_polling_communicator_id"] == "comm0"
    assert host_delay.extras["collective_event_polling_participant_count"] == 16
    assert host_delay.extras["collective_event_polling_cuda_event_handle"] == "ev0"
    assert host_delay.extras["collective_event_polling_cuda_event_version"] == 3
    assert host_delay.extras["collective_event_polling_stream_resource_id"] == (
        "rank:0:stream:7"
    )
    assert host_delay.extras["collective_event_polling_boundary_origin_status"] == (
        "unresolved"
    )
    assert host_delay.extras["collective_event_polling_boundary_visibility_status"] == (
        "unresolved"
    )
    assert host_delay.extras["collective_event_polling_wait_map_release_status"] == (
        "unavailable"
    )
    assert host_delay.extras["collective_event_polling_count_once_status"] == (
        "unavailable"
    )
    assert host_delay.extras["collective_event_polling_nonoverlap_status"] == (
        "unavailable"
    )
    assert host_delay.extras["collective_event_polling_repair_ready"] is False
    assert host_delay.extras[
        "collective_event_polling_safe_to_use_as_repair_evidence"
    ] is False
    assert host_delay.extras[
        "collective_event_polling_safe_to_use_as_subtraction_delta"
    ] is False
    assert host_delay.extras["collective_event_polling_safe_delta_us"] is None


def test_collate_collective_event_polling_metadata_opt_in_skips_unrelated_boundary(
    monkeypatch,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.setenv(
        "MAYA_ENABLE_COLLECTIVE_EVENT_POLLING_BOUNDARY_METADATA_EXPORT",
        "1",
    )
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"stream_id": "7"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasGemmEx",
                op_type="blas_compute",
                extras={"stream_id": "7"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]

    assert [event.api for event in events] == [
        "cudaLaunchKernel",
        "__hostDelay__",
        "cublasGemmEx",
    ]
    assert host_delay.extras["observed_gap_us"] == 40
    assert host_delay.extras["hostdelay_occurrence_metadata_opt_in_flag"] is True
    assert not (_COLLECTIVE_EVENT_POLLING_METADATA_FIELDS & host_delay.extras.keys())


def test_collate_event_polling_boundary_metadata_default_off_absent_when_base_opted_in(
    monkeypatch,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.setenv("MAYA_ENABLE_HOSTDELAY_OCCURRENCE_METADATA_EXPORT", "1")
    monkeypatch.delenv(
        "MAYA_ENABLE_EVENT_POLLING_BOUNDARY_ORIGIN_VISIBILITY_METADATA_EXPORT",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_EVENT_POLLING_BOUNDARY_ORIGIN_VISIBILITY_METADATA_EXPORT",
        raising=False,
    )
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaEventQuery",
                op_type="event_query",
                extras={"event_id": "ev0", "event_version": 1},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaEventQuery",
                op_type="event_query",
                extras={"event_id": "ev0", "event_version": 1},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]

    assert [event.api for event in events] == [
        "cudaEventQuery",
        "__hostDelay__",
        "cudaEventQuery",
    ]
    assert host_delay.extras["observed_gap_us"] == 40
    assert host_delay.extras["hostdelay_occurrence_metadata_opt_in_flag"] is True
    assert not (_EVENT_POLLING_BOUNDARY_METADATA_FIELDS & host_delay.extras.keys())


def test_collate_event_polling_boundary_metadata_opt_in_skips_non_event_boundary(
    monkeypatch,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.setenv(
        "MAYA_ENABLE_EVENT_POLLING_BOUNDARY_ORIGIN_VISIBILITY_METADATA_EXPORT",
        "1",
    )
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"stream_id": "7"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasGemmEx",
                op_type="blas_compute",
                extras={"stream_id": "7"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]

    assert [event.api for event in events] == [
        "cudaLaunchKernel",
        "__hostDelay__",
        "cublasGemmEx",
    ]
    assert host_delay.extras["observed_gap_us"] == 40
    assert host_delay.extras["hostdelay_occurrence_metadata_opt_in_flag"] is True
    assert not (_EVENT_POLLING_BOUNDARY_METADATA_FIELDS & host_delay.extras.keys())


@pytest.mark.parametrize(
    (
        "previous_api",
        "current_api",
        "expected_target_class",
        "expected_polling_class",
        "expected_origin_status",
        "expected_visibility_status",
        "expected_paper_visibility_class",
        "expected_control_status",
    ),
    [
        (
            "cudaEventQuery",
            "cudaEventQuery",
            "nonblocking_cudaEventQuery_polling_pressure",
            "nonblocking_cudaEventQuery_polling",
            "classified_paper_visible_by_default",
            "paper_visible_by_default",
            "paper_visible_by_default",
            "not_proven",
        ),
        (
            "cudaLaunchKernel",
            "cudaEventQuery",
            "nonblocking_cudaEventQuery_polling_pressure",
            "event_polling_boundary_mixed",
            "unresolved",
            "unresolved",
            "unresolved_mixed",
            "candidate_needs_strict_boundary_origin_proof",
        ),
        (
            "cudaEventQuery",
            "cudaLaunchKernel",
            "nonblocking_cudaEventQuery_polling_pressure",
            "event_polling_boundary_mixed",
            "unresolved",
            "unresolved",
            "unresolved_mixed",
            "candidate_needs_strict_boundary_origin_proof",
        ),
        (
            "cudaEventQuery",
            "cublasSetStream_v2",
            "nonblocking_cudaEventQuery_polling_pressure",
            "event_polling_boundary_mixed",
            "unresolved",
            "unresolved",
            "unresolved_mixed",
            "candidate_needs_strict_boundary_origin_proof",
        ),
        (
            "cudaEventRecord",
            "cudaLaunchKernel",
            "event_record_launch_boundary",
            "event_polling_boundary_mixed",
            "unresolved",
            "unresolved",
            "unresolved_mixed",
            "candidate_needs_strict_boundary_origin_proof",
        ),
    ],
)
def test_collate_event_polling_boundary_metadata_opt_in_classifies_conservatively(
    monkeypatch,
    previous_api,
    current_api,
    expected_target_class,
    expected_polling_class,
    expected_origin_status,
    expected_visibility_status,
    expected_paper_visibility_class,
    expected_control_status,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.setenv(
        "MAYA_ENABLE_EVENT_POLLING_BOUNDARY_ORIGIN_VISIBILITY_METADATA_EXPORT",
        "1",
    )
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api=previous_api,
                op_type="event_or_launch",
                extras={"event_id": "ev0", "event_version": 1, "stream_id": "7"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api=current_api,
                op_type="event_or_launch",
                extras={"event_id": "ev0", "event_version": 1, "stream_id": "7"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]
    target = [event for event in events if event.api == current_api][-1]

    assert host_delay.extras["observed_gap_us"] == 40
    assert target.prev_event_id == host_delay.id
    assert _EVENT_POLLING_BOUNDARY_METADATA_FIELDS <= host_delay.extras.keys()
    assert host_delay.extras["event_polling_boundary_metadata_schema_version"] == (
        "event_polling_boundary_origin_visibility_trace_processing_metadata_v1"
    )
    assert host_delay.extras["event_polling_boundary_metadata_opt_in_flag"] is True
    assert host_delay.extras["event_polling_boundary_raw_family"] == (
        f"{previous_api} -> {current_api}"
    )
    assert host_delay.extras["event_polling_boundary_semantic_family"] == (
        f"{previous_api} -> {current_api}"
    )
    assert host_delay.extras["event_polling_boundary_target_class"] == (
        expected_target_class
    )
    assert host_delay.extras["event_polling_boundary_polling_class"] == (
        expected_polling_class
    )
    assert host_delay.extras["event_polling_boundary_origin_status"] == (
        expected_origin_status
    )
    assert host_delay.extras["event_polling_boundary_visibility_status"] == (
        expected_visibility_status
    )
    assert host_delay.extras["event_polling_boundary_paper_visibility_class"] == (
        expected_paper_visibility_class
    )
    assert host_delay.extras[
        "event_polling_boundary_candidate_control_plane_subregion_status"
    ] == expected_control_status
    assert host_delay.extras["event_polling_boundary_count_once_status"] == (
        "unavailable"
    )
    assert host_delay.extras["event_polling_boundary_nonoverlap_status"] == (
        "unavailable"
    )
    assert host_delay.extras["event_polling_boundary_wait_map_safety_status"] == (
        "unavailable"
    )
    assert host_delay.extras["event_polling_boundary_repair_ready"] is False
    assert host_delay.extras[
        "event_polling_boundary_safe_to_use_as_repair_evidence"
    ] is False
    assert host_delay.extras[
        "event_polling_boundary_safe_to_use_as_subtraction_delta"
    ] is False
    assert host_delay.extras["event_polling_boundary_safe_delta_us"] is None


def test_collate_boundary_origin_subregion_metadata_default_off_absent_when_base_opted_in(
    monkeypatch,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.setenv("MAYA_ENABLE_HOSTDELAY_OCCURRENCE_METADATA_EXPORT", "1")
    monkeypatch.delenv(
        "MAYA_ENABLE_BOUNDARY_ORIGIN_SUBREGION_PROOF_METADATA_EXPORT",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_BOUNDARY_ORIGIN_SUBREGION_PROOF_METADATA_EXPORT",
        raising=False,
    )
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"stream_id": "7"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaEventQuery",
                op_type="event_query",
                extras={"event_id": "ev0", "event_version": 1, "stream_id": "7"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]

    assert host_delay.extras["observed_gap_us"] == 40
    assert host_delay.extras["hostdelay_occurrence_metadata_opt_in_flag"] is True
    assert not (
        _BOUNDARY_ORIGIN_SUBREGION_METADATA_FIELDS & host_delay.extras.keys()
    )


@pytest.mark.parametrize(
    (
        "previous_api",
        "current_api",
        "expected_kind",
        "expected_candidate_role",
        "expected_origin_status",
        "expected_visibility_status",
        "expected_paper_visibility_class",
        "expected_strict_proof_status",
    ),
    [
        (
            "cudaLaunchKernel",
            "cudaEventQuery",
            "candidate_internal_event_launch_library_control_gap",
            "candidate_internal_subregion_needs_strict_proof",
            "candidate_needs_strict_boundary_origin_proof",
            "candidate_needs_strict_visibility_proof",
            "unresolved_mixed",
            "unavailable_or_unproven",
        ),
        (
            "cudaEventQuery",
            "cudaLaunchKernel",
            "candidate_internal_event_launch_library_control_gap",
            "candidate_internal_subregion_needs_strict_proof",
            "candidate_needs_strict_boundary_origin_proof",
            "candidate_needs_strict_visibility_proof",
            "unresolved_mixed",
            "unavailable_or_unproven",
        ),
        (
            "cudaEventRecord",
            "cudaEventRecord",
            "candidate_internal_event_launch_library_control_gap",
            "candidate_internal_subregion_needs_strict_proof",
            "candidate_needs_strict_boundary_origin_proof",
            "candidate_needs_strict_visibility_proof",
            "unresolved_mixed",
            "unavailable_or_unproven",
        ),
        (
            "cudaEventQuery",
            "cublasSetStream_v2",
            "candidate_internal_event_launch_library_control_gap",
            "candidate_internal_subregion_needs_strict_proof",
            "candidate_needs_strict_boundary_origin_proof",
            "candidate_needs_strict_visibility_proof",
            "unresolved_mixed",
            "unavailable_or_unproven",
        ),
        (
            "cudaEventQuery",
            "cudaEventQuery",
            "not_applicable_pure_polling",
            "paper_visible_polling_not_targeted_for_removal",
            "classified_paper_visible_by_default",
            "paper_visible_by_default",
            "paper_visible_by_default",
            "not_applicable",
        ),
        (
            "cublasGemmEx",
            "cudaEventQuery",
            "not_applicable_pure_polling",
            "paper_visible_polling_not_targeted_for_removal",
            "classified_paper_visible_by_default",
            "paper_visible_by_default",
            "paper_visible_by_default",
            "not_applicable",
        ),
        (
            "cublasGemmStridedBatchedEx",
            "cudaEventQuery",
            "not_applicable_pure_polling",
            "paper_visible_polling_not_targeted_for_removal",
            "classified_paper_visible_by_default",
            "paper_visible_by_default",
            "paper_visible_by_default",
            "not_applicable",
        ),
    ],
)
def test_collate_boundary_origin_subregion_metadata_opt_in_classifies_conservatively(
    monkeypatch,
    previous_api,
    current_api,
    expected_kind,
    expected_candidate_role,
    expected_origin_status,
    expected_visibility_status,
    expected_paper_visibility_class,
    expected_strict_proof_status,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.setenv(
        "MAYA_ENABLE_BOUNDARY_ORIGIN_SUBREGION_PROOF_METADATA_EXPORT",
        "1",
    )
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api=previous_api,
                op_type="event_or_launch",
                extras={"event_id": "ev0", "event_version": 1, "stream_id": "7"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api=current_api,
                op_type="event_or_launch",
                extras={"event_id": "ev0", "event_version": 1, "stream_id": "7"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]
    target = [event for event in events if event.api == current_api][-1]

    assert host_delay.extras["observed_gap_us"] == 40
    assert target.prev_event_id == host_delay.id
    assert _BOUNDARY_ORIGIN_SUBREGION_METADATA_FIELDS <= host_delay.extras.keys()
    assert host_delay.extras["boundary_origin_subregion_metadata_schema_version"] == (
        "boundary_origin_subregion_proof_metadata_v1"
    )
    assert host_delay.extras["boundary_origin_subregion_metadata_opt_in_flag"] is True
    assert host_delay.extras["boundary_origin_subregion_raw_family"] == (
        f"{previous_api} -> {current_api}"
    )
    assert host_delay.extras["boundary_origin_subregion_semantic_family"] == (
        f"{previous_api} -> {current_api}"
    )
    assert host_delay.extras["boundary_origin_subregion_kind"] == expected_kind
    assert host_delay.extras["boundary_origin_subregion_candidate_role"] == (
        expected_candidate_role
    )
    assert host_delay.extras["boundary_origin_subregion_origin_status"] == (
        expected_origin_status
    )
    assert host_delay.extras["boundary_origin_subregion_visibility_status"] == (
        expected_visibility_status
    )
    assert host_delay.extras[
        "boundary_origin_subregion_paper_visibility_class"
    ] == expected_paper_visibility_class
    assert host_delay.extras["boundary_origin_subregion_strict_extent_status"] == (
        "unavailable_or_unproven"
    )
    assert host_delay.extras["boundary_origin_subregion_strict_proof_status"] == (
        expected_strict_proof_status
    )
    assert host_delay.extras["boundary_origin_subregion_count_once_status"] == (
        "unavailable"
    )
    assert host_delay.extras["boundary_origin_subregion_nonoverlap_status"] == (
        "unavailable"
    )
    assert host_delay.extras["boundary_origin_subregion_wait_map_safety_status"] == (
        "unavailable"
    )
    assert host_delay.extras["boundary_origin_subregion_repair_ready"] is False
    assert host_delay.extras[
        "boundary_origin_subregion_safe_to_use_as_repair_evidence"
    ] is False
    assert host_delay.extras[
        "boundary_origin_subregion_safe_to_use_as_subtraction_delta"
    ] is False
    assert host_delay.extras["boundary_origin_subregion_safe_delta_us"] is None


def test_collate_strict_subregion_extent_metadata_default_off_absent_when_related_opted_in(
    monkeypatch,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.setenv(
        "MAYA_ENABLE_BOUNDARY_ORIGIN_SUBREGION_PROOF_METADATA_EXPORT",
        "1",
    )
    monkeypatch.delenv(
        "MAYA_ENABLE_STRICT_SUBREGION_EXTENT_SOURCE_METADATA_EXPORT",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_STRICT_SUBREGION_EXTENT_SOURCE_METADATA_EXPORT",
        raising=False,
    )
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"stream_id": "7"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaEventQuery",
                op_type="event_query",
                extras={"event_id": "ev0", "event_version": 1, "stream_id": "7"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]

    assert host_delay.extras["observed_gap_us"] == 40
    assert _BOUNDARY_ORIGIN_SUBREGION_METADATA_FIELDS <= host_delay.extras.keys()
    assert not (_STRICT_SUBREGION_EXTENT_METADATA_FIELDS & host_delay.extras.keys())


@pytest.mark.parametrize(
    (
        "previous_api",
        "current_api",
        "expected_target_family_class",
        "expected_candidate_kind",
        "expected_candidate_role",
        "expected_source_proof_status",
        "expected_origin_status",
        "expected_visibility_status",
        "expected_paper_visibility_class",
    ),
    [
        (
            "cudaLaunchKernel",
            "cudaEventQuery",
            "unresolved_launch_event_library_control_boundary",
            "candidate_internal_event_launch_library_control_gap",
            "requires_strict_subregion_extent_source_proof",
            "unavailable_or_unproven",
            "candidate_needs_strict_boundary_origin_proof",
            "candidate_needs_strict_visibility_proof",
            "unresolved_mixed",
        ),
        (
            "cudaEventQuery",
            "cudaLaunchKernel",
            "unresolved_launch_event_library_control_boundary",
            "candidate_internal_event_launch_library_control_gap",
            "requires_strict_subregion_extent_source_proof",
            "unavailable_or_unproven",
            "candidate_needs_strict_boundary_origin_proof",
            "candidate_needs_strict_visibility_proof",
            "unresolved_mixed",
        ),
        (
            "cudaEventRecord",
            "cudaEventRecord",
            "unresolved_launch_event_library_control_boundary",
            "candidate_internal_event_launch_library_control_gap",
            "requires_strict_subregion_extent_source_proof",
            "unavailable_or_unproven",
            "candidate_needs_strict_boundary_origin_proof",
            "candidate_needs_strict_visibility_proof",
            "unresolved_mixed",
        ),
        (
            "cudaEventQuery",
            "cublasSetStream_v2",
            "unresolved_launch_event_library_control_boundary",
            "candidate_internal_event_launch_library_control_gap",
            "requires_strict_subregion_extent_source_proof",
            "unavailable_or_unproven",
            "candidate_needs_strict_boundary_origin_proof",
            "candidate_needs_strict_visibility_proof",
            "unresolved_mixed",
        ),
        (
            "cudaEventQuery",
            "cudaEventQuery",
            "paper_visible_polling_not_targeted_for_removal",
            "not_applicable_paper_visible_polling",
            "not_targeted_for_removal",
            "not_applicable",
            "classified_paper_visible_by_default",
            "paper_visible_by_default",
            "paper_visible_by_default",
        ),
        (
            "cublasGemmEx",
            "cudaEventQuery",
            "paper_visible_polling_not_targeted_for_removal",
            "not_applicable_paper_visible_polling",
            "not_targeted_for_removal",
            "not_applicable",
            "classified_paper_visible_by_default",
            "paper_visible_by_default",
            "paper_visible_by_default",
        ),
        (
            "cublasGemmStridedBatchedEx",
            "cudaEventQuery",
            "paper_visible_polling_not_targeted_for_removal",
            "not_applicable_paper_visible_polling",
            "not_targeted_for_removal",
            "not_applicable",
            "classified_paper_visible_by_default",
            "paper_visible_by_default",
            "paper_visible_by_default",
        ),
    ],
)
def test_collate_strict_subregion_extent_metadata_opt_in_classifies_conservatively(
    monkeypatch,
    previous_api,
    current_api,
    expected_target_family_class,
    expected_candidate_kind,
    expected_candidate_role,
    expected_source_proof_status,
    expected_origin_status,
    expected_visibility_status,
    expected_paper_visibility_class,
):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.setenv(
        "MAYA_ENABLE_STRICT_SUBREGION_EXTENT_SOURCE_METADATA_EXPORT",
        "1",
    )
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api=previous_api,
                op_type="event_or_launch",
                extras={"event_id": "ev0", "event_version": 1, "stream_id": "7"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api=current_api,
                op_type="event_or_launch",
                extras={"event_id": "ev0", "event_version": 1, "stream_id": "7"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [event for event in events if event.api == "__hostDelay__"][0]
    target = [event for event in events if event.api == current_api][-1]

    assert host_delay.extras["observed_gap_us"] == 40
    assert target.prev_event_id == host_delay.id
    assert _STRICT_SUBREGION_EXTENT_METADATA_FIELDS <= host_delay.extras.keys()
    assert host_delay.extras["strict_subregion_extent_metadata_schema_version"] == (
        "strict_subregion_extent_source_metadata_v1"
    )
    assert host_delay.extras["strict_subregion_extent_metadata_opt_in_flag"] is True
    assert host_delay.extras["strict_subregion_extent_raw_family"] == (
        f"{previous_api} -> {current_api}"
    )
    assert host_delay.extras["strict_subregion_extent_semantic_family"] == (
        f"{previous_api} -> {current_api}"
    )
    assert host_delay.extras["strict_subregion_extent_target_family_class"] == (
        expected_target_family_class
    )
    assert host_delay.extras["strict_subregion_extent_candidate_subregion_kind"] == (
        expected_candidate_kind
    )
    assert host_delay.extras["strict_subregion_extent_candidate_subregion_role"] == (
        expected_candidate_role
    )
    assert host_delay.extras["strict_subregion_extent_start_ts_us"] is None
    assert host_delay.extras["strict_subregion_extent_end_ts_us"] is None
    assert host_delay.extras[
        "strict_subregion_extent_duration_us_context_only"
    ] == 40
    assert host_delay.extras["strict_subregion_extent_timestamp_source_kind"] == (
        "none_strict_source_unavailable"
    )
    assert host_delay.extras["strict_subregion_extent_source_is_non_perturbing"] is False
    assert host_delay.extras[
        "strict_subregion_extent_source_uses_runtime_endpoint_substitution"
    ] is False
    assert host_delay.extras[
        "strict_subregion_extent_source_uses_measured_actual_runtime"
    ] is False
    assert host_delay.extras[
        "strict_subregion_extent_source_uses_hostdelay_shortening"
    ] is False
    assert host_delay.extras["strict_subregion_extent_source_proof_status"] == (
        expected_source_proof_status
    )
    assert host_delay.extras["strict_subregion_extent_origin_status"] == (
        expected_origin_status
    )
    assert host_delay.extras["strict_subregion_extent_visibility_status"] == (
        expected_visibility_status
    )
    assert host_delay.extras["strict_subregion_extent_paper_visibility_class"] == (
        expected_paper_visibility_class
    )
    assert host_delay.extras["strict_subregion_extent_count_once_status"] == (
        "unavailable"
    )
    assert host_delay.extras["strict_subregion_extent_nonoverlap_status"] == (
        "unavailable"
    )
    assert host_delay.extras[
        "strict_subregion_extent_cuda_event_wait_map_safety_status"
    ] == "unavailable"
    assert host_delay.extras[
        "strict_subregion_extent_collective_wait_map_safety_status"
    ] == "unavailable"
    assert host_delay.extras["strict_subregion_extent_repair_ready"] is False
    assert host_delay.extras[
        "strict_subregion_extent_safe_to_use_as_repair_evidence"
    ] is False
    assert host_delay.extras[
        "strict_subregion_extent_safe_to_use_as_subtraction_delta"
    ] is False
    assert host_delay.extras["strict_subregion_extent_safe_delta_us"] is None
    assert host_delay.extras[
        "strict_subregion_extent_runtime_or_endpoint_substitution_used"
    ] is False
    assert host_delay.extras["strict_subregion_extent_hostdelay_shortening_used"] is False
    assert host_delay.extras[
        "strict_subregion_extent_rank_workload_special_case_used"
    ] is False


def test_collate_opt_in_launch_config_pop_entry_boundary_excludes_only_pop_body(
    monkeypatch,
):
    from flexsim.maya_lite.schema import FidelityWindow, RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.setenv("MAYA_ENABLE_LAUNCH_CONFIG_POP_ENTRY_HOSTDELAY_BOUNDARY", "1")
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_LAUNCH_CONFIG_POP_ENTRY_HOSTDELAY_BOUNDARY",
        raising=False,
    )
    monkeypatch.delenv("MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("FLEXSIM_MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("FLEXSIM_MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=120,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="__cudaPopCallConfiguration",
                op_type="other",
                extras={
                    "end_ts": 120,
                    "host_duration_us": 4.0,
                    "host_timing_mode": "measure",
                    "host_timing_source": "direct_wallclock",
                },
            ),
            TraceEvent(
                rank=0,
                ordinal=2,
                source=TraceSource.FAKE,
                ts=150,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"actual_launch_api_body_us": 3.5, "host_duration_us": 10.0},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
        trace_window="step",
        step_windows={0: (90, 200)},
        fidelity_windows={
            0: FidelityWindow(
                start_ts=90,
                end_ts=200,
                source="manifest",
                is_paper_valid_step_window=True,
            )
        },
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delay = [
        event
        for event in events
        if event.api == "__hostDelay__"
        and event.extras.get("hostdelay_source") == "collate_host_gap"
    ][0]
    launch = [event for event in events if event.api == "cudaLaunchKernel"][0]

    assert [event.api for event in events] == ["__hostDelay__", "cudaLaunchKernel", "__hostDelay__"]
    assert host_delay.extras["observed_gap_us"] == 16
    assert host_delay.extras["raw_boundary_family"] == (
        "cudaGetDevice -> __cudaPopCallConfiguration"
    )
    assert host_delay.extras["launch_config_pop_entry_hostdelay_boundary_enabled"] is True
    assert host_delay.extras[
        "launch_config_pop_entry_excluded_from_prepop_hostdelay_us"
    ] == 4
    assert host_delay.extras[
        "launch_config_pop_entry_hostdelay_boundary_status"
    ] == "enabled_opt_in_pop_wrapper_body_not_counted_as_prepop_hostdelay"
    _assert_launch_config_metadata(
        launch,
        adjacent_gap_us=30,
        contribution="excluded_from_pending_host_gap",
        normalization_enabled=True,
        normalization_status="enabled_default_excluded_from_hostdelay",
        raw_event_id="r0:e1",
        raw_ts_us=120,
        raw_end_ts_us=120,
    )


def test_launch_boundary_visibility_fields_do_not_change_collate_or_replay_timing(monkeypatch):
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    monkeypatch.setenv("MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", "1")
    monkeypatch.delenv("FLEXSIM_MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    monkeypatch.delenv("FLEXSIM_MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION", raising=False)
    diagnostic_extras = {
        "boundary_segment_schema_version": "launch_boundary_visibility_v1",
        "launch_boundary_id_unavailable_reason": (
            "fakecuda_launch_pair_id_disabled_to_preserve_host_duration"
        ),
        "wrapper_segment_coverage": "structural_labels_only_unmeasured",
        "wrapper_segment_sum_us": 0.0,
        "wrapper_segment_unattributed_us": 15.0,
        "paper_visible_host_duration_us": None,
        "instrumentation_only_duration_us": None,
        "caller_visible_elapsed_us": 15.0,
        "boundary_origin_kind": "mixed_or_unresolved",
        "boundary_origin_classification_basis": (
            "producer_segment_timing_disabled_to_preserve_host_duration"
        ),
        "boundary_visibility_segments": [
            {
                "name": "real_api_body",
                "visibility_kind": "mixed_or_unresolved",
                "duration_us": None,
                "clock": "unmeasured",
                "included_in_paper_visible_host_duration": False,
                "included_in_instrumentation_only_duration": False,
            }
        ],
        "host_control_boundary_counterpart_schema_version": (
            "host_control_boundary_visibility_unblocker_v2_row_evidence_v1"
        ),
        "host_control_visibility_schema_version": (
            "host_control_launch_neighborhood_visibility_counterpart_isolation_v1"
        ),
        "host_control_visibility_opt_in_flag": True,
        "host_control_boundary_occurrence_id": (
            "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:1"
        ),
        "selected_occurrence_id": (
            "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:1"
        ),
        "actual_counterpart_id": (
            "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:1"
        ),
        "actual_counterpart_status": (
            "actual_boundary_row_id_exported_selected_occurrence_join_not_attempted"
        ),
        "host_control_visibility_split_status": "unavailable",
        "host_control_visibility_split_unavailable_reason": (
            "mechanical_visibility_split_not_measured_to_preserve_wrapper_host_timing"
        ),
        "mechanical_visibility_split_status": "unavailable",
        "mechanical_visibility_split_unavailable_reason": (
            "mechanical_visibility_split_not_measured_to_preserve_wrapper_host_timing"
        ),
        "host_control_producer_visibility_schema_version": (
            "host_control_producer_visibility_nonoverlap_v1"
        ),
        "host_control_producer_visibility_status": "structural_unavailable",
        "host_control_producer_visibility_unavailable_reason": (
            "structural_labels_only_internal_clocks_disabled_to_preserve_start_time_end_time"
        ),
        "host_control_producer_visibility_basis": (
            "real_wrapper_default_off_structural_metadata_no_internal_wrapper_clocks"
        ),
        "host_control_producer_visibility_segments": [
            {
                "name": "real_api_call_envelope",
                "visibility_kind": "mixed_or_unresolved",
                "duration_us": None,
                "clock": "unmeasured",
            }
        ],
        "host_control_producer_numeric_split_status": "unavailable",
        "host_control_producer_nonoverlap_status": "unavailable",
        "host_control_producer_wait_map_nonoverlap_status": "unavailable",
        "host_control_producer_double_counting_nonoverlap_status": "unavailable",
        "host_control_compat_launch_pop_coverage_status": (
            "unavailable_not_exported_by_current_real_wrapper_producer"
        ),
        "host_control_compat_launch_pop_coverage_unavailable_reason": (
            "__cudaPopCallConfiguration_interposition_not_proven_for_real_libcudart;"
            "do_not_synthesize_compat_launch_family_from_cudaLaunchKernel"
        ),
        "actual_raw_prev_event_id": "rank:0:raw_ordinal:0",
        "actual_raw_current_event_id": "rank:0:raw_ordinal:1",
        "actual_boundary_family": "__cudaPopCallConfiguration -> cudaLaunchKernel",
        "counterpart_join_key": (
            "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:1"
        ),
        "counterpart_join_method": (
            "actual_row_id_only_emulated_selected_occurrence_join_not_attempted"
        ),
        "counterpart_join_confidence": "unavailable",
        "comparable_actual_context_only": True,
        "split_sum_check_status": "unavailable",
        "classification_unavailable_reason": (
            "mechanical_visibility_split_not_measured_to_preserve_wrapper_host_timing"
        ),
        "count_once_status": "unavailable",
        "double_counting_overlap_unavailable_reason": (
            "requires_figure6_count_once_interval_ledger_not_available_during_capture"
        ),
    }

    def make_bundle(with_diagnostics: bool) -> TraceBundle:
        current_extras = {"end_ts": 150, "host_duration_us": 15.0}
        if with_diagnostics:
            current_extras.update(diagnostic_extras)
        rank_trace = RankTrace(
            rank=0,
            path=Path("/tmp/rank_0.jsonl"),
            source=TraceSource.FAKE,
            events=(
                TraceEvent(
                    rank=0,
                    ordinal=0,
                    source=TraceSource.FAKE,
                    ts=100,
                    pid=1,
                    tid=2,
                    module="libcudart.so.12",
                    api="__cudaPopCallConfiguration",
                    op_type="other",
                    extras={"end_ts": 100, "host_duration_us": 4.0},
                ),
                TraceEvent(
                    rank=0,
                    ordinal=1,
                    source=TraceSource.FAKE,
                    ts=150,
                    pid=1,
                    tid=2,
                    module="libcudart.so.12",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    extras=current_extras,
                ),
            ),
        )
        return TraceBundle(
            trace_dir=Path("/tmp/manual"),
            source=TraceSource.FAKE,
            rank_traces=(rank_trace,),
            trace_window="step",
            step_windows={0: (90, 200)},
        )

    def annotated_from_collated(bundle: TraceBundle) -> AnnotatedTrace:
        collated = collate_trace_bundle(bundle)
        rank_events: dict[int, tuple[AnnotatedEvent, ...]] = {}
        for rank, events in collated.rank_events.items():
            annotated_events = []
            for event in events:
                if event.api == "__hostDelay__":
                    duration_us = float(event.extras["observed_gap_us"])
                else:
                    duration_us = float(event.extras.get("host_duration_us", 0.0))
                annotated_events.append(
                    AnnotatedEvent(
                        id=event.id,
                        rank=event.rank,
                        ordinal=event.ordinal,
                        source=event.source,
                        ts=event.ts,
                        pid=event.pid,
                        tid=event.tid,
                        module=event.module,
                        api=event.api,
                        op_type=event.op_type,
                        extras=dict(event.extras),
                        prev_event_id=event.prev_event_id,
                        collective_group_id=event.collective_group_id,
                        duration_us=duration_us,
                        duration_source="unit_test",
                    )
                )
            rank_events[rank] = tuple(annotated_events)
        global_events = tuple(event for events in rank_events.values() for event in events)
        return AnnotatedTrace(
            trace_dir=collated.trace_dir,
            source=collated.source,
            rank_events=rank_events,
            global_events=global_events,
            collective_groups=collated.collective_groups,
            trace_window=collated.trace_window,
        )

    baseline = annotated_from_collated(make_bundle(False))
    diagnostic = annotated_from_collated(make_bundle(True))
    baseline_events = baseline.rank_events[0]
    diagnostic_events = diagnostic.rank_events[0]
    baseline_host_delays = [event for event in baseline_events if event.api == "__hostDelay__"]
    diagnostic_host_delays = [event for event in diagnostic_events if event.api == "__hostDelay__"]

    assert [
        (event.ts, event.extras.get("end_ts"), event.extras.get("host_duration_us"))
        for event in baseline_events
        if event.api != "__hostDelay__"
    ] == [
        (event.ts, event.extras.get("end_ts"), event.extras.get("host_duration_us"))
        for event in diagnostic_events
        if event.api != "__hostDelay__"
    ]
    assert len(baseline_host_delays) == len(diagnostic_host_delays) == 2
    assert [event.duration_us for event in baseline_host_delays] == [
        event.duration_us for event in diagnostic_host_delays
    ]
    assert [event.extras["hostdelay_source"] for event in baseline_host_delays] == [
        "leading_step_gap",
        "trailing_step_gap",
    ]
    assert [event.extras["hostdelay_source"] for event in diagnostic_host_delays] == [
        "leading_step_gap",
        "trailing_step_gap",
    ]
    baseline_launch = [event for event in baseline_events if event.api == "cudaLaunchKernel"][0]
    diagnostic_launch = [event for event in diagnostic_events if event.api == "cudaLaunchKernel"][0]
    assert not (_HOST_CONTROL_DEFAULT_OFF_DIAGNOSTIC_FIELDS & baseline_launch.extras.keys())
    assert diagnostic_launch.extras["boundary_visibility_segments"][0]["name"] == "real_api_body"
    assert diagnostic_launch.extras["boundary_visibility_segments"][0]["duration_us"] is None
    assert diagnostic_launch.extras["host_control_boundary_occurrence_id"] == (
        "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:1"
    )
    assert diagnostic_launch.extras["host_control_visibility_schema_version"] == (
        "host_control_launch_neighborhood_visibility_counterpart_isolation_v1"
    )
    assert diagnostic_launch.extras["selected_occurrence_id"] == (
        "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:1"
    )
    assert "affected_interval_id" not in diagnostic_launch.extras
    assert diagnostic_launch.extras["count_once_status"] == "unavailable"
    assert diagnostic_launch.extras["host_control_visibility_split_status"] == "unavailable"
    assert diagnostic_launch.extras["host_control_producer_visibility_status"] == (
        "structural_unavailable"
    )
    assert diagnostic_launch.extras["host_control_producer_visibility_segments"][0]["duration_us"] is None
    assert diagnostic_launch.extras["host_control_producer_numeric_split_status"] == (
        "unavailable"
    )
    assert diagnostic_launch.extras["host_control_producer_nonoverlap_status"] == (
        "unavailable"
    )
    assert diagnostic_launch.extras["host_control_compat_launch_pop_coverage_status"] == (
        "unavailable_not_exported_by_current_real_wrapper_producer"
    )
    assert diagnostic_launch.extras["actual_counterpart_id"] == (
        "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:1"
    )
    _assert_launch_config_metadata(
        diagnostic_launch,
        adjacent_gap_us=50,
        contribution="excluded_from_pending_host_gap",
        normalization_enabled=True,
        normalization_status="enabled_explicit_enable_excluded_from_hostdelay",
        raw_event_id="r0:e0",
        raw_ts_us=100,
        raw_end_ts_us=100,
    )

    baseline_replay = replay_annotated_trace(baseline)
    diagnostic_replay = replay_annotated_trace(diagnostic)

    assert baseline_replay.success
    assert diagnostic_replay.success
    assert diagnostic_replay.critical_path_us == baseline_replay.critical_path_us
    assert diagnostic_replay.total_time_us == baseline_replay.total_time_us


def test_collate_preserves_host_delay_between_two_setup_apis():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.REAL,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank_trace,),
    )

    collated = collate_trace_bundle(bundle)
    events = collated.rank_events[0]

    assert not events


def test_collate_host_delay_uses_dispatch_timestamp_high_watermark():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.REAL,
                ts=200,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
            TraceEvent(
                rank=0,
                ordinal=2,
                source=TraceSource.REAL,
                ts=150,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=3,
                source=TraceSource.REAL,
                ts=240,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank_trace,),
    )

    collated = collate_trace_bundle(bundle)
    host_delays = [event for event in collated.rank_events[0] if event.api == "__hostDelay__"]
    launches = [event for event in collated.rank_events[0] if event.api == "cudaLaunchKernel"]

    assert len(host_delays) == 2
    assert [event.extras["observed_gap_us"] for event in host_delays] == [50, 40]
    _assert_cuda_get_device_context_query_run_fold(
        launches[0],
        event_count=2,
        internal_gap_us=50,
        terminal_gap_us=50,
        preserved_pre_run_gap_us=0,
    )


def test_collate_uses_single_host_machine_dispatch_queue_by_default():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.REAL,
                ts=110,
                pid=1,
                tid=3,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
            TraceEvent(
                rank=0,
                ordinal=2,
                source=TraceSource.REAL,
                ts=145,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank_trace,),
    )

    collated = collate_trace_bundle(bundle)
    events = collated.rank_events[0]

    assert [event.api for event in events] == [
        "cudaLaunchKernel",
        "__hostDelay__",
        "cudaLaunchKernel",
    ]
    host_delays = [event for event in events if event.api == "__hostDelay__"]
    assert len(host_delays) == 1
    assert [event.extras["observed_gap_us"] for event in host_delays] == [45]
    assert host_delays[0].extras["host_timing_dispatch_scope"] == "host_machine"
    assert host_delays[0].extras["host_machine_id"] == "legacy_pid:1"
    assert host_delays[0].extras["host_dispatch_queue_id"] == "legacy_pid:1"
    assert events[0].extras["host_machine_id"] == "legacy_pid:1"
    assert events[0].extras["host_dispatch_queue_id"] == "legacy_pid:1"
    assert events[1].extras["host_dispatch_model"] == "single_dispatch_queue_per_host_execution_context"


def test_collate_keeps_process_scoped_direct_measure_without_contract_unprojected():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={
                    "host_timing_mode": "measure",
                    "host_timing_source": "direct_wallclock",
                    "host_timing_dispatch_scope": "process",
                    "host_duration_us": 25.0,
                    "end_ts": 125,
                },
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=110,
                pid=1,
                tid=3,
                module="libnccl.so.2",
                api="ncclAllReduce",
                op_type="nccl_collective",
                extras={
                    "host_timing_mode": "measure",
                    "host_timing_source": "direct_wallclock",
                    "host_timing_dispatch_scope": "process",
                    "host_duration_us": 7.0,
                    "end_ts": 117,
                },
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    collated = collate_trace_bundle(bundle)
    events = collated.rank_events[0]

    assert [event.api for event in events] == [
        "cudaLaunchKernel",
        "__hostDelay__",
        "ncclAllReduce",
    ]
    assert events[1].extras["observed_gap_us"] == 10
    assert events[1].extras["host_timing_dispatch_scope"] == "process"
    assert events[2].ts == 110


def test_collate_keeps_host_machine_direct_measure_without_contract_unprojected():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={
                    "host_timing_mode": "measure",
                    "host_timing_source": "direct_wallclock",
                    "host_timing_dispatch_scope": "host_machine",
                    "host_duration_us": 25.0,
                    "end_ts": 125,
                },
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=110,
                pid=1,
                tid=3,
                module="libnccl.so.2",
                api="ncclAllReduce",
                op_type="nccl_collective",
                extras={
                    "host_timing_mode": "measure",
                    "host_timing_source": "direct_wallclock",
                    "host_timing_dispatch_scope": "host_machine",
                    "host_duration_us": 7.0,
                    "end_ts": 117,
                },
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    collated = collate_trace_bundle(bundle)
    events = collated.rank_events[0]

    assert [event.api for event in events] == [
        "cudaLaunchKernel",
        "__hostDelay__",
        "ncclAllReduce",
    ]
    assert events[1].extras["observed_gap_us"] == 10
    assert events[1].extras["host_timing_dispatch_scope"] == "host_machine"
    assert events[2].ts == 110


def test_collate_direct_measure_host_delay_uses_serialized_api_entry_delta_without_dispatch_contract():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={
                    "host_timing_mode": "measure",
                    "host_timing_source": "direct_wallclock",
                    "host_timing_dispatch_scope": "host_machine",
                    "host_duration_us": 25.0,
                    "end_ts": 125,
                },
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=3,
                module="libnccl.so.2",
                api="ncclAllReduce",
                op_type="nccl_collective",
                extras={
                    "host_timing_mode": "measure",
                    "host_timing_source": "direct_wallclock",
                    "host_timing_dispatch_scope": "host_machine",
                    "host_duration_us": 7.0,
                    "end_ts": 147,
                },
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    collated = collate_trace_bundle(bundle)
    events = collated.rank_events[0]

    assert [event.api for event in events] == [
        "cudaLaunchKernel",
        "__hostDelay__",
        "ncclAllReduce",
    ]
    assert events[1].extras["observed_gap_us"] == 40
    assert events[1].ts + events[1].extras["observed_gap_us"] == events[2].ts
    assert events[2].ts == 140


def test_collate_dispatch_only_direct_measure_host_delay_starts_after_wrapper_body():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={
                    "host_timing_mode": "measure",
                    "host_timing_source": "direct_wallclock",
                    "host_timing_dispatch_scope": "host_machine",
                    "wrapper_runtime_contract": "dispatch_only",
                    "host_duration_us": 25.0,
                    "end_ts": 125,
                },
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=140,
                pid=1,
                tid=3,
                module="libnccl.so.2",
                api="ncclAllReduce",
                op_type="nccl_collective",
                extras={
                    "host_timing_mode": "measure",
                    "host_timing_source": "direct_wallclock",
                    "host_timing_dispatch_scope": "host_machine",
                    "wrapper_runtime_contract": "dispatch_only",
                    "host_duration_us": 7.0,
                    "end_ts": 147,
                },
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]

    assert [event.api for event in events] == [
        "cudaLaunchKernel",
        "__hostDelay__",
        "ncclAllReduce",
    ]
    assert events[1].ts == 125
    assert events[1].extras["observed_gap_us"] == 15
    assert events[1].ts + events[1].extras["observed_gap_us"] == events[2].ts
    assert events[2].ts == 140


def test_collate_legacy_direct_measure_compat_row_does_not_advance_dispatch_queue():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
                extras={
                    "host_timing_mode": "measure",
                    "host_timing_source": "direct_wallclock",
                    "host_timing_dispatch_scope": "host_machine",
                    "host_duration_us": 80.0,
                    "end_ts": 180,
                },
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=120,
                pid=1,
                tid=3,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={
                    "host_timing_mode": "measure",
                    "host_timing_source": "direct_wallclock",
                    "host_timing_dispatch_scope": "host_machine",
                    "wrapper_runtime_contract": "dispatch_only",
                    "host_duration_us": 5.0,
                    "end_ts": 125,
                    "stream_id": "0",
                },
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]

    assert [event.api for event in events] == ["__hostDelay__", "cudaLaunchKernel"]
    assert events[0].ts == 100
    assert events[0].extras["observed_gap_us"] == 20
    assert events[0].extras["raw_prev_api"] == "cudaGetDevice"
    assert events[0].extras["raw_current_api"] == "cudaLaunchKernel"
    assert events[1].ts == 120


def test_collate_to_replay_dispatch_only_direct_measure_does_not_double_count_wrapper_body():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=0,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={
                    "host_timing_mode": "measure",
                    "host_timing_source": "direct_wallclock",
                    "host_timing_dispatch_scope": "host_machine",
                    "wrapper_runtime_contract": "dispatch_only",
                    "host_duration_us": 25.0,
                    "end_ts": 25,
                    "stream_id": "0",
                },
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=40,
                pid=1,
                tid=3,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={
                    "host_timing_mode": "measure",
                    "host_timing_source": "direct_wallclock",
                    "host_timing_dispatch_scope": "host_machine",
                    "wrapper_runtime_contract": "dispatch_only",
                    "host_duration_us": 7.0,
                    "end_ts": 47,
                    "stream_id": "0",
                },
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )
    collated = collate_trace_bundle(bundle)
    annotated_events = []
    for event in collated.rank_events[0]:
        duration_us = (
            float(event.extras["observed_gap_us"])
            if event.api == "__hostDelay__"
            else 0.0
        )
        annotated_events.append(
            AnnotatedEvent(
                id=event.id,
                rank=event.rank,
                ordinal=event.ordinal,
                source=event.source,
                ts=event.ts,
                pid=event.pid,
                tid=event.tid,
                module=event.module,
                api=event.api,
                op_type=event.op_type,
                extras=dict(event.extras),
                prev_event_id=event.prev_event_id,
                collective_group_id=event.collective_group_id,
                duration_us=duration_us,
                duration_source="unit_test",
            )
        )
    annotated = AnnotatedTrace(
        trace_dir=collated.trace_dir,
        source=collated.source,
        rank_events={0: tuple(annotated_events)},
        global_events=tuple(annotated_events),
        collective_groups=collated.collective_groups,
        trace_window=collated.trace_window,
    )

    result = replay_annotated_trace(annotated)
    host_dispatch = [
        event for event in result.simulated_events if event.api.endswith(":host_dispatch")
    ]
    host_delay = [
        event for event in result.simulated_events if event.api == "__hostDelay__"
    ]

    assert [event.duration_us for event in host_dispatch] == [25.0, 7.0]
    assert [(event.start_us, event.duration_us) for event in host_delay] == [(25.0, 15.0)]
    assert result.total_time_us == 47.0


def test_collate_dispatch_only_direct_measure_overlap_does_not_emit_host_delay():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={
                    "host_timing_mode": "measure",
                    "host_timing_source": "direct_wallclock",
                    "host_timing_dispatch_scope": "host_machine",
                    "wrapper_runtime_contract": "dispatch_only",
                    "host_duration_us": 25.0,
                    "end_ts": 125,
                },
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=110,
                pid=1,
                tid=3,
                module="libnccl.so.2",
                api="ncclAllReduce",
                op_type="nccl_collective",
                extras={
                    "host_timing_mode": "measure",
                    "host_timing_source": "direct_wallclock",
                    "host_timing_dispatch_scope": "host_machine",
                    "wrapper_runtime_contract": "dispatch_only",
                    "host_duration_us": 7.0,
                    "end_ts": 117,
                },
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    events = collate_trace_bundle(bundle).rank_events[0]

    assert [event.api for event in events] == ["cudaLaunchKernel", "ncclAllReduce"]
    assert events[1].ts == 125
    assert not [event for event in events if event.api == "__hostDelay__"]


def test_collate_dispatch_only_direct_measure_step_end_starts_after_wrapper_body():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={
                    "host_timing_mode": "measure",
                    "host_timing_source": "direct_wallclock",
                    "host_timing_dispatch_scope": "host_machine",
                    "wrapper_runtime_contract": "dispatch_only",
                    "host_duration_us": 25.0,
                    "end_ts": 125,
                },
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
        step_windows={0: (90, 150)},
        trace_window="step",
    )

    events = collate_trace_bundle(bundle).rank_events[0]
    host_delays = [event for event in events if event.api == "__hostDelay__"]

    assert [event.extras["hostdelay_source"] for event in host_delays] == [
        "leading_step_gap",
        "trailing_step_gap",
    ]
    assert [(event.ts, event.extras["observed_gap_us"]) for event in host_delays] == [
        (90, 10),
        (125, 25),
    ]
    assert host_delays[1].ts + host_delays[1].extras["observed_gap_us"] == 150


def test_collate_respects_thread_dispatch_scope_for_host_gap_lanes():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"host_machine_id": "host0"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=145,
                pid=1,
                tid=3,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={"host_machine_id": "host0"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
        host_timing_dispatch_scope_resolved="thread",
    )

    collated = collate_trace_bundle(bundle)
    events = collated.rank_events[0]

    assert [event.api for event in events] == [
        "cudaLaunchKernel",
        "cudaLaunchKernel",
    ]
    assert [event.prev_event_id for event in events] == [None, None]
    assert [event.extras["host_dispatch_model"] for event in events] == [
        "dispatch_queue_per_host_thread",
        "dispatch_queue_per_host_thread",
    ]
    assert collated.host_timing_dispatch_scope_resolved == "thread"


def test_collate_keeps_direct_measure_thread_scope_unprojected():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={
                    "host_timing_mode": "measure",
                    "host_timing_source": "direct_wallclock",
                    "host_timing_dispatch_scope": "thread",
                    "host_duration_us": 25.0,
                    "end_ts": 125,
                },
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.FAKE,
                ts=110,
                pid=1,
                tid=3,
                module="libnccl.so.2",
                api="ncclAllReduce",
                op_type="nccl_collective",
                extras={
                    "host_timing_mode": "measure",
                    "host_timing_source": "direct_wallclock",
                    "host_timing_dispatch_scope": "thread",
                    "host_duration_us": 7.0,
                    "end_ts": 117,
                },
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank_trace,),
    )

    collated = collate_trace_bundle(bundle)
    events = collated.rank_events[0]

    assert [event.api for event in events] == [
        "cudaLaunchKernel",
        "ncclAllReduce",
    ]
    assert events[0].extras["host_timing_dispatch_scope"] == "thread"
    assert events[1].extras["host_timing_dispatch_scope"] == "thread"
    assert events[1].ts == 110


def test_collate_projects_real_host_runtime_for_positive_host_api_duration():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaMalloc",
                op_type="mem_alloc",
                extras={
                    "host_duration_us": 25.0,
                    "end_ts": 125,
                },
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.REAL,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank_trace,),
    )

    collated = collate_trace_bundle(bundle)
    events = collated.rank_events[0]

    assert [event.api for event in events] == [
        "cudaMalloc",
        "__hostDelay__",
        "cudaLaunchKernel",
    ]
    assert events[1].extras["observed_gap_us"] == 15
    assert events[2].ts == 140


def test_collate_keeps_real_compat_runtime_inside_host_delay_envelope():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=100,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
                extras={
                    "host_duration_us": 25.0,
                    "end_ts": 125,
                },
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.REAL,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank_trace,),
    )

    collated = collate_trace_bundle(bundle)
    events = collated.rank_events[0]

    assert [event.api for event in events] == [
        "__hostDelay__",
        "cudaLaunchKernel",
    ]
    assert events[0].extras["observed_gap_us"] == 40
    assert events[0].extras["raw_prev_api"] == "cudaGetDevice"
    assert events[0].extras["raw_current_api"] == "cudaLaunchKernel"
    assert events[0].extras["raw_boundary_family"] == "cudaGetDevice -> cudaLaunchKernel"
    assert not any(
        key.startswith("cublas_set_stream_context_query_suffix_")
        for key in events[1].extras
    )


def test_collate_does_not_double_count_real_async_runtime_observation_as_host_delay():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=100,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasGemmEx",
                op_type="blas_compute",
                extras={
                    "host_duration_us": 25.0,
                    "end_ts": 125,
                    "wrapper_runtime_contract": "async_runtime",
                    "observed_runtime_us": "25.0",
                    "runtime_observation_source": "capture_real_cuda_event",
                },
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.REAL,
                ts=130,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaGetDevice",
                op_type="context_op",
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank_trace,),
    )

    collated = collate_trace_bundle(bundle)
    events = collated.rank_events[0]

    assert [event.api for event in events] == [
        "cublasGemmEx",
    ]
    assert not [event for event in events if event.api == "__hostDelay__"]


def test_collate_projects_real_low_overhead_host_api_with_min_runtime():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=100,
                pid=1,
                tid=2,
                module="libcublas.so.12",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={
                    "handle_id": "h0",
                    "stream_id": "s0",
                },
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.REAL,
                ts=140,
                pid=1,
                tid=2,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank_trace,),
    )

    collated = collate_trace_bundle(bundle)
    events = collated.rank_events[0]

    assert [event.api for event in events] == [
        "cublasSetStream_v2",
        "__hostDelay__",
        "cudaLaunchKernel",
    ]
    assert events[1].extras["observed_gap_us"] == 39


def test_collate_global_order_is_deterministic():
    trace_dir = Path("paper/traces/fake/e3")
    if not trace_dir.exists():
        pytest.skip(f"trace dir not available: {trace_dir}")

    bundle = load_trace_directory(trace_dir, max_events_per_rank=32)
    first = collate_trace_bundle(bundle)
    second = collate_trace_bundle(bundle)

    assert [event.id for event in first.global_events] == [event.id for event in second.global_events]
    assert [event.ts for event in first.global_events] == sorted(event.ts for event in first.global_events)


def test_collate_groups_collectives_without_semantic_fields():
    trace_dir = Path("paper/traces/real/e1")
    if not trace_dir.exists():
        pytest.skip(f"trace dir not available: {trace_dir}")

    bundle = load_trace_directory(trace_dir, max_events_per_rank=31000)
    collated = collate_trace_bundle(bundle)

    assert collated.collective_groups
    first_group = next(iter(collated.collective_groups.values()))
    assert first_group.api.lower().startswith("nccl")
    assert first_group.ranks
    assert first_group.event_ids

    for event_id in first_group.event_ids:
        event = next(event for event in collated.global_events if event.id == event_id)
        assert event.collective_group_id == first_group.id
        assert "logical_op_id" not in event.extras
        assert "semantic_role" not in event.extras


def test_collective_detection_excludes_nccl_control_apis():
    assert is_collective_api("ncclAllReduce", "nccl_collective")
    assert is_collective_api("ncclBroadcast", "nccl_collective")
    assert is_collective_api("ncclSend", "nccl_collective")
    assert is_collective_api("ncclRecv", "nccl_collective")
    assert not is_collective_api("ncclCommGetAsyncError", "nccl_collective")
    assert not is_collective_api("ncclCommInitRankConfig", "nccl_collective")


def test_collate_groups_nccl_send_recv_by_comm_and_call_index():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank0 = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=10,
                pid=1,
                tid=1,
                module="libnccl.so.2",
                api="ncclSend",
                op_type="nccl_collective",
                extras={"comm_id": "comm-0", "call_idx": "7", "peer": "1", "nranks": "8"},
            ),
        ),
    )
    rank1 = RankTrace(
        rank=1,
        path=Path("/tmp/rank_1.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=1,
                ordinal=0,
                source=TraceSource.REAL,
                ts=11,
                pid=2,
                tid=2,
                module="libnccl.so.2",
                api="ncclRecv",
                op_type="nccl_collective",
                extras={"comm_id": "comm-0", "call_idx": "7", "peer": "0", "nranks": "8"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank0, rank1),
    )

    collated = collate_trace_bundle(bundle)

    send_groups = [event.collective_group_id for event in collated.rank_events[0]]
    recv_groups = [event.collective_group_id for event in collated.rank_events[1]]
    assert send_groups == ["ncclP2P|comm:comm-0|members:0-1|pair_seq:0"]
    assert recv_groups == ["ncclP2P|comm:comm-0|members:0-1|pair_seq:0"]
    assert set(collated.collective_groups) == {"ncclP2P|comm:comm-0|members:0-1|pair_seq:0"}
    group = collated.collective_groups["ncclP2P|comm:comm-0|members:0-1|pair_seq:0"]
    assert group.api == "ncclP2P"
    assert group.match_basis == "communicator_pair_sequence"
    assert group.communicator_id == "comm-0"
    assert group.sequence_number == 0
    assert group.communicator_size == 2
    assert group.participant_count == 2
    send = collated.rank_events[0][0]
    recv = collated.rank_events[1][0]
    assert send.extras["collective_api"] == "ncclP2P"
    assert recv.extras["collective_api"] == "ncclP2P"
    assert send.extras["collective_match_basis"] == "communicator_pair_sequence"
    assert send.extras["collective_communicator_id"] == "comm-0"
    assert send.extras["collective_sequence_number"] == 0
    assert send.extras["communicator_size"] == 2
    assert send.extras["participant_count"] == 2


def test_collective_grouping_uses_communicator_membership_to_resolve_p2p_global_peers():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank8 = RankTrace(
        rank=8,
        path=Path("/tmp/rank_8.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=8,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=10,
                pid=8,
                tid=8,
                module="libnccl.so.2",
                api="ncclSend",
                op_type="nccl_collective",
                extras={"comm_id": "pp-1", "call_idx": "3", "peer": "1", "nranks": "2"},
            ),
        ),
    )
    rank9 = RankTrace(
        rank=9,
        path=Path("/tmp/rank_9.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=9,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=11,
                pid=9,
                tid=9,
                module="libnccl.so.2",
                api="ncclRecv",
                op_type="nccl_collective",
                extras={"comm_id": "pp-1", "call_idx": "3", "peer": "0", "nranks": "2"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank8, rank9),
        communicator_memberships={"pp-1": (8, 9)},
    )

    collated = collate_trace_bundle(bundle)

    assert [event.collective_group_id for event in collated.rank_events[8]] == ["ncclP2P|comm:pp-1|members:8-9|pair_seq:0"]
    assert [event.collective_group_id for event in collated.rank_events[9]] == ["ncclP2P|comm:pp-1|members:8-9|pair_seq:0"]
    group = collated.collective_groups["ncclP2P|comm:pp-1|members:8-9|pair_seq:0"]
    assert group.api == "ncclP2P"
    assert group.ranks == (8, 9)
    assert group.communicator_id == "pp-1"
    assert group.sequence_number == 0
    assert group.participant_count == 2


def test_collective_grouping_matches_p2p_by_pair_sequence_when_raw_call_idx_differs():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank1 = RankTrace(
        rank=1,
        path=Path("/tmp/rank_1.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=1,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=10,
                pid=1,
                tid=1,
                module="libnccl.so.2",
                api="ncclSend",
                op_type="nccl_collective",
                extras={"comm_id": "pp-0", "call_idx": "11", "peer": "2", "nranks": "4"},
            ),
        ),
    )
    rank2 = RankTrace(
        rank=2,
        path=Path("/tmp/rank_2.jsonl"),
        source=TraceSource.FAKE,
        events=(
            TraceEvent(
                rank=2,
                ordinal=0,
                source=TraceSource.FAKE,
                ts=11,
                pid=2,
                tid=2,
                module="libnccl.so.2",
                api="ncclRecv",
                op_type="nccl_collective",
                extras={"comm_id": "pp-0", "call_idx": "10", "peer": "1", "nranks": "4"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank1, rank2),
        communicator_memberships={"pp-0": (0, 1, 2, 3)},
    )

    collated = collate_trace_bundle(bundle)

    group = collated.collective_groups["ncclP2P|comm:pp-0|members:1-2|pair_seq:0"]
    assert group.ranks == (1, 2)
    assert group.sequence_number == 0
    assert group.match_basis == "communicator_pair_sequence"


def test_collective_grouping_matches_p2p_by_pair_payload_without_call_idx():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank8 = RankTrace(
        rank=8,
        path=Path("/tmp/rank_8.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=8,
                ordinal=0,
                source=TraceSource.REAL,
                ts=10,
                pid=8,
                tid=8,
                module="libnccl.so.2",
                api="ncclSend",
                op_type="nccl_collective",
                extras={"comm_id": "pp-host", "peer": "1", "nranks": "4", "count": "1024", "datatype": "7"},
            ),
        ),
    )
    rank9 = RankTrace(
        rank=9,
        path=Path("/tmp/rank_9.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=9,
                ordinal=0,
                source=TraceSource.REAL,
                ts=11,
                pid=9,
                tid=9,
                module="libnccl.so.2",
                api="ncclRecv",
                op_type="nccl_collective",
                extras={"comm_id": "pp-host", "peer": "0", "nranks": "4", "count": "1024", "datatype": "7"},
            ),
        ),
    )
    rank10 = RankTrace(
        rank=10,
        path=Path("/tmp/rank_10.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=10,
                ordinal=0,
                source=TraceSource.REAL,
                ts=12,
                pid=10,
                tid=10,
                module="libnccl.so.2",
                api="ncclSend",
                op_type="nccl_collective",
                extras={"comm_id": "pp-host", "peer": "3", "nranks": "4", "count": "1024", "datatype": "7"},
            ),
        ),
    )
    rank11 = RankTrace(
        rank=11,
        path=Path("/tmp/rank_11.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=11,
                ordinal=0,
                source=TraceSource.REAL,
                ts=13,
                pid=11,
                tid=11,
                module="libnccl.so.2",
                api="ncclRecv",
                op_type="nccl_collective",
                extras={"comm_id": "pp-host", "peer": "2", "nranks": "4", "count": "1024", "datatype": "7"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank8, rank9, rank10, rank11),
        communicator_memberships={"pp-host": (8, 9, 10, 11)},
    )

    collated = collate_trace_bundle(bundle)

    expected_group_ids = {
        "ncclP2P|comm:pp-host|members:8-9|count:1024|dtype:7|op:|pair_occ:0",
        "ncclP2P|comm:pp-host|members:10-11|count:1024|dtype:7|op:|pair_occ:0",
    }
    assert set(collated.collective_groups) == expected_group_ids
    first_group = collated.collective_groups["ncclP2P|comm:pp-host|members:8-9|count:1024|dtype:7|op:|pair_occ:0"]
    second_group = collated.collective_groups["ncclP2P|comm:pp-host|members:10-11|count:1024|dtype:7|op:|pair_occ:0"]
    assert first_group.ranks == (8, 9)
    assert second_group.ranks == (10, 11)
    assert first_group.match_basis == "communicator_pair_payload"
    assert second_group.match_basis == "communicator_pair_payload"


def test_collective_group_event_ids_follow_global_event_order():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank1 = RankTrace(
        rank=1,
        path=Path("/tmp/rank_1.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=1,
                ordinal=0,
                source=TraceSource.REAL,
                ts=11,
                pid=2,
                tid=2,
                module="libnccl.so.2",
                api="ncclAllReduce",
                op_type="nccl_collective",
                extras={"comm_id": "comm-0", "call_idx": "1", "nranks": "2"},
            ),
        ),
    )
    rank0 = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=10,
                pid=1,
                tid=1,
                module="libnccl.so.2",
                api="ncclAllReduce",
                op_type="nccl_collective",
                extras={"comm_id": "comm-0", "call_idx": "1", "nranks": "2"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank1, rank0),
    )

    collated = collate_trace_bundle(bundle)
    group = collated.collective_groups["ncclAllReduce|comm:comm-0|call:1"]

    assert [event.id for event in collated.global_events if event.collective_group_id == group.id] == list(
        group.event_ids
    )
    assert group.event_ids == ("r0:e0", "r1:e0")
    assert group.match_basis == "communicator_sequence"
    assert group.communicator_id == "comm-0"
    assert group.sequence_number == 1
    assert group.participant_count == 2


def test_collate_attaches_collective_group_metadata_to_event_extras():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank0 = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=10,
                pid=1,
                tid=1,
                module="libnccl.so.2",
                api="ncclAllReduce",
                op_type="nccl_collective",
                extras={"comm_id": "comm-0", "call_idx": "5", "nranks": "2"},
            ),
        ),
    )
    rank1 = RankTrace(
        rank=1,
        path=Path("/tmp/rank_1.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=1,
                ordinal=0,
                source=TraceSource.REAL,
                ts=11,
                pid=2,
                tid=2,
                module="libnccl.so.2",
                api="ncclAllReduce",
                op_type="nccl_collective",
                extras={"comm_id": "comm-0", "call_idx": "5", "nranks": "2"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank0, rank1),
    )

    collated = collate_trace_bundle(bundle)
    event = collated.rank_events[0][0]

    assert event.extras["collective_api"] == "ncclAllReduce"
    assert event.extras["collective_match_basis"] == "communicator_sequence"
    assert event.extras["collective_communicator_id"] == "comm-0"
    assert event.extras["collective_sequence_number"] == 5
    assert event.extras["communicator_size"] == 2
    assert event.extras["participant_count"] == 2


def test_collate_prefers_recovered_communicator_size_over_raw_world_size():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    def _allreduce(rank: int, ts: int) -> RankTrace:
        return RankTrace(
            rank=rank,
            path=Path(f"/tmp/rank_{rank}.jsonl"),
            source=TraceSource.REAL,
            events=(
                TraceEvent(
                    rank=rank,
                    ordinal=0,
                    source=TraceSource.REAL,
                    ts=ts,
                    pid=rank + 1,
                    tid=rank + 1,
                    module="libnccl.so.2",
                    api="ncclAllReduce",
                    op_type="nccl_collective",
                    extras={
                        "comm_id": "tp-0",
                        "call_idx": "7",
                        "world_size": "16",
                    },
                ),
            ),
        )

    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(_allreduce(0, 10), _allreduce(8, 11)),
        communicator_memberships={"tp-0": (0, 8)},
    )

    collated = collate_trace_bundle(bundle)
    group = collated.collective_groups["ncclAllReduce|comm:tp-0|call:7"]

    assert group.communicator_size == 2
    assert group.participant_count == 2
    assert collated.rank_events[0][0].extras["communicator_size"] == 2


def test_collate_preserves_global_event_order_when_attaching_collective_metadata():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank0 = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=10,
                pid=1,
                tid=1,
                module="libnccl.so.2",
                api="ncclAllReduce",
                op_type="nccl_collective",
                extras={"comm_id": "comm-0", "call_idx": "5", "nranks": "2"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.REAL,
                ts=30,
                pid=1,
                tid=1,
                module="libcudart.so.12",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
            ),
        ),
    )
    rank1 = RankTrace(
        rank=1,
        path=Path("/tmp/rank_1.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=1,
                ordinal=0,
                source=TraceSource.REAL,
                ts=20,
                pid=2,
                tid=2,
                module="libnccl.so.2",
                api="ncclAllReduce",
                op_type="nccl_collective",
                extras={"comm_id": "comm-0", "call_idx": "5", "nranks": "2"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank0, rank1),
    )

    collated = collate_trace_bundle(bundle)

    assert [event.id for event in collated.global_events] == ["r0:e0", "r0:h1", "r1:e0", "r0:e1"]
    assert [event.id for event in collated.global_events if event.api != "__hostDelay__"] == [
        "r0:e0",
        "r1:e0",
        "r0:e1",
    ]
    non_host_delay_events = [
        event for event in collated.global_events if event.api != "__hostDelay__"
    ]
    assert non_host_delay_events[0].extras["collective_api"] == "ncclAllReduce"
    assert non_host_delay_events[1].extras["collective_api"] == "ncclAllReduce"


def test_collective_grouping_uses_payload_signature_not_only_api():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank0 = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=0,
                ordinal=0,
                source=TraceSource.REAL,
                ts=10,
                pid=1,
                tid=1,
                module="libnccl.so.2",
                api="ncclAllReduce",
                op_type="nccl_collective",
                extras={"count": "8", "datatype": "7", "op": "0"},
            ),
            TraceEvent(
                rank=0,
                ordinal=1,
                source=TraceSource.REAL,
                ts=20,
                pid=1,
                tid=1,
                module="libnccl.so.2",
                api="ncclAllReduce",
                op_type="nccl_collective",
                extras={"count": "1", "datatype": "7", "op": "0"},
            ),
        ),
    )
    rank1 = RankTrace(
        rank=1,
        path=Path("/tmp/rank_1.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(
                rank=1,
                ordinal=0,
                source=TraceSource.REAL,
                ts=11,
                pid=2,
                tid=2,
                module="libnccl.so.2",
                api="ncclAllReduce",
                op_type="nccl_collective",
                extras={"count": "8", "datatype": "7", "op": "0"},
            ),
            TraceEvent(
                rank=1,
                ordinal=1,
                source=TraceSource.REAL,
                ts=21,
                pid=2,
                tid=2,
                module="libnccl.so.2",
                api="ncclAllReduce",
                op_type="nccl_collective",
                extras={"count": "1", "datatype": "7", "op": "0"},
            ),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(rank0, rank1),
    )

    collated = collate_trace_bundle(bundle)
    groups = list(collated.collective_groups.values())

    assert len(groups) == 2
    assert len(set(group.id for group in groups)) == 2
    for group in groups:
        assert group.ranks == (0, 1)
        assert group.match_basis == "payload_signature"
        assert group.participant_count == 2


def test_collective_grouping_prefers_comm_and_call_metadata_when_available():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    def _event(rank: int, ordinal: int, ts: int, *, comm_id: str, call_idx: str) -> TraceEvent:
        return TraceEvent(
            rank=rank,
            ordinal=ordinal,
            source=TraceSource.FAKE,
            ts=ts,
            pid=rank + 1,
            tid=rank + 1,
            module="libnccl.so.2",
            api="ncclAllReduce",
            op_type="nccl_collective",
            extras={
                "collective": "all_reduce",
                "comm_id": comm_id,
                "call_idx": call_idx,
                "nranks": "2",
                "count": "8",
                "datatype": "7",
                "op": "0",
            },
        )

    rank0 = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.FAKE,
        events=(
            _event(0, 0, 10, comm_id="11", call_idx="1"),
            _event(0, 1, 20, comm_id="22", call_idx="1"),
        ),
    )
    rank1 = RankTrace(
        rank=1,
        path=Path("/tmp/rank_1.jsonl"),
        source=TraceSource.FAKE,
        events=(
            _event(1, 0, 11, comm_id="22", call_idx="1"),
            _event(1, 1, 21, comm_id="11", call_idx="1"),
        ),
    )
    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_traces=(rank0, rank1),
    )

    collated = collate_trace_bundle(bundle)

    assert len(collated.collective_groups) == 2
    grouped_ranks = {group.id: group.ranks for group in collated.collective_groups.values()}
    assert all(ranks == (0, 1) for ranks in grouped_ranks.values())
    assert any("comm:11" in group_id for group_id in grouped_ranks)
    assert any("comm:22" in group_id for group_id in grouped_ranks)
    for group in collated.collective_groups.values():
        assert group.match_basis == "communicator_sequence"
        assert group.sequence_number == 1
        assert group.participant_count == 2


def test_dedup_identical_rank_traces_preserves_original_world_size():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    def _trace(rank: int, *, top_k: int) -> RankTrace:
        return RankTrace(
            rank=rank,
            path=Path(f"/tmp/rank_{rank}.jsonl"),
            source=TraceSource.REAL,
            events=(
                TraceEvent(
                    rank=rank,
                    ordinal=0,
                    source=TraceSource.REAL,
                    ts=10 + rank,
                    pid=rank + 1,
                    tid=rank + 1,
                    module="libcudart.so.12",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    extras={"m": "64", "n": "64", "k": "64", "top_k": str(top_k)},
                ),
                TraceEvent(
                    rank=rank,
                    ordinal=1,
                    source=TraceSource.REAL,
                    ts=20 + rank,
                    pid=rank + 1,
                    tid=rank + 1,
                    module="libnccl.so.2",
                    api="ncclAllReduce",
                    op_type="nccl_collective",
                    extras={"count": "8", "datatype": "7", "op": "0"},
                ),
            ),
        )

    bundle = TraceBundle(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_traces=(
            _trace(0, top_k=2),
            _trace(1, top_k=2),
            _trace(2, top_k=1),
        ),
    )

    deduped = dedup_identical_rank_traces(bundle)

    assert deduped.world_size == 3
    assert deduped.profiled_world_size == 2
    assert deduped.rank_ids() == (0, 2)
    assert deduped.profiled_rank_groups == {0: (0, 1), 2: (2,)}

    collated = collate_trace_bundle(deduped)
    assert collated.world_size == 3
    assert collated.profiled_world_size == 2
    assert collated.profiled_rank_groups == {0: (0, 1), 2: (2,)}

def test_collate_collapses_setup_gap_into_next_semantic_host_delay():
    from flexsim.maya_lite.schema import RankTrace, TraceBundle, TraceEvent, TraceSource

    rank_trace = RankTrace(
        rank=0,
        path=Path("/tmp/rank_0.jsonl"),
        source=TraceSource.REAL,
        events=(
            TraceEvent(rank=0, ordinal=0, source=TraceSource.REAL, ts=100, pid=1, tid=2, module="libcudart.so.12", api="cudaGetDevice", op_type="context_op"),
            TraceEvent(rank=0, ordinal=1, source=TraceSource.REAL, ts=120, pid=1, tid=2, module="libcudart.so.12", api="cudaGetLastError", op_type="other"),
            TraceEvent(rank=0, ordinal=2, source=TraceSource.REAL, ts=140, pid=1, tid=2, module="libcudart.so.12", api="cudaLaunchKernel", op_type="kernel_launch"),
        ),
    )
    bundle = TraceBundle(trace_dir=Path("/tmp/manual"), source=TraceSource.REAL, rank_traces=(rank_trace,))

    collated = collate_trace_bundle(bundle)
    events = collated.rank_events[0]

    assert [event.api for event in events] == [
        "__hostDelay__",
        "cudaLaunchKernel",
    ]
    assert events[0].extras["observed_gap_us"] == 40
    assert events[0].extras["raw_prev_api"] == "cudaGetDevice"
    assert events[0].extras["raw_current_api"] == "cudaLaunchKernel"
