"""
Low-level trace collation for Maya-lite.

This module intentionally avoids SPSD or logical-op information. It only:

- preserves per-rank program order
- creates a deterministic global event order
- groups collectives using communicator IDs and sequence numbers when available
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Mapping

from .communicators import parse_emulated_communicator_id
from .filters import (
    is_collective_api,
    is_compat_only_api,
    is_host_timing_traced_api,
    is_ignorable_setup_api,
    is_low_overhead_api,
    occupies_host_dispatch_resource,
)
from .launch_neighborhood import (
    LAUNCH_NEIGHBORHOOD_EQUIVALENCE_ENV_KEYS,
    LAUNCH_NEIGHBORHOOD_EQUIVALENCE_EXPORT_FIELDS,
    build_launch_neighborhood_equivalence_metadata,
)
from .material_signature import canonical_gemm_material_signature, is_gemm_material_api
from .schema import CollatedEvent, CollatedTrace, CollectiveGroup, TraceBundle, TraceEvent, TraceSource


_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION_ENV_KEYS = (
    "MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION",
    "FLEXSIM_MAYA_ENABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION",
)
_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION_DISABLE_ENV_KEYS = (
    "MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION",
    "FLEXSIM_MAYA_DISABLE_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION",
)
_LAUNCH_CONFIG_POP_ENTRY_HOSTDELAY_BOUNDARY_ENV_KEYS = (
    "MAYA_ENABLE_LAUNCH_CONFIG_POP_ENTRY_HOSTDELAY_BOUNDARY",
    "FLEXSIM_MAYA_ENABLE_LAUNCH_CONFIG_POP_ENTRY_HOSTDELAY_BOUNDARY",
)

_IDEMPOTENT_CUBLAS_SET_STREAM_FOLD_ENV_KEYS = (
    "MAYA_ENABLE_IDEMPOTENT_CUBLAS_SET_STREAM_FOLD",
    "FLEXSIM_MAYA_ENABLE_IDEMPOTENT_CUBLAS_SET_STREAM_FOLD",
)
_CONTEXT_QUERY_SUFFIX_EVENT_RECORD_FOLD_ENV_KEYS = (
    "MAYA_ENABLE_CONTEXT_QUERY_SUFFIX_EVENT_RECORD_FOLD",
    "FLEXSIM_MAYA_ENABLE_CONTEXT_QUERY_SUFFIX_EVENT_RECORD_FOLD",
)
_CONTEXT_QUERY_SUFFIX_LAUNCH_CONFIG_POP_FOLD_ENV_KEYS = (
    "MAYA_ENABLE_CONTEXT_QUERY_SUFFIX_LAUNCH_CONFIG_POP_FOLD",
    "FLEXSIM_MAYA_ENABLE_CONTEXT_QUERY_SUFFIX_LAUNCH_CONFIG_POP_FOLD",
)
_HOSTDELAY_OCCURRENCE_METADATA_EXPORT_ENV_KEYS = (
    "MAYA_ENABLE_HOSTDELAY_OCCURRENCE_METADATA_EXPORT",
    "FLEXSIM_MAYA_ENABLE_HOSTDELAY_OCCURRENCE_METADATA_EXPORT",
)
_HOSTDELAY_OCCURRENCE_METADATA_SCHEMA_VERSION = (
    "hostdelay_occurrence_metadata_export_v1"
)
_HOSTDELAY_OCCURRENCE_JOIN_METADATA_EXPORT_ENV_KEYS = (
    "MAYA_ENABLE_HOSTDELAY_OCCURRENCE_JOIN_METADATA_EXPORT",
    "FLEXSIM_MAYA_ENABLE_HOSTDELAY_OCCURRENCE_JOIN_METADATA_EXPORT",
)
_HOSTDELAY_OCCURRENCE_JOIN_METADATA_SCHEMA_VERSION = (
    "hostdelay_occurrence_join_metadata_export_v1"
)
_COLLECTIVE_EVENT_POLLING_BOUNDARY_METADATA_EXPORT_ENV_KEYS = (
    "MAYA_ENABLE_COLLECTIVE_EVENT_POLLING_BOUNDARY_METADATA_EXPORT",
    "FLEXSIM_MAYA_ENABLE_COLLECTIVE_EVENT_POLLING_BOUNDARY_METADATA_EXPORT",
)
_COLLECTIVE_EVENT_POLLING_BOUNDARY_METADATA_SCHEMA_VERSION = (
    "collective_event_polling_boundary_metadata_v1"
)
_COLLECTIVE_EVENT_POLLING_BOUNDARY_APIS = frozenset(
    {
        "ncclAllReduce",
        "ncclSend",
        "ncclRecv",
        "ncclCommGetAsyncError",
        "cudaEventRecord",
        "cudaEventQuery",
        "cudaStreamWaitEvent",
    }
)
_EVENT_POLLING_BOUNDARY_METADATA_EXPORT_ENV_KEYS = (
    "MAYA_ENABLE_EVENT_POLLING_BOUNDARY_ORIGIN_VISIBILITY_METADATA_EXPORT",
    "FLEXSIM_MAYA_ENABLE_EVENT_POLLING_BOUNDARY_ORIGIN_VISIBILITY_METADATA_EXPORT",
)
_EVENT_POLLING_BOUNDARY_METADATA_SCHEMA_VERSION = (
    "event_polling_boundary_origin_visibility_trace_processing_metadata_v1"
)
_EVENT_POLLING_BOUNDARY_APIS = frozenset(
    {
        "cudaEventQuery",
        "cudaEventRecord",
        "cudaEventRecordWithFlags",
        "cudaStreamWaitEvent",
        "cudaLaunchKernel",
        "cublasSetStream_v2",
        "cublasGemmEx",
        "cublasGemmStridedBatchedEx",
    }
)
_EVENT_POLLING_BOUNDARY_CORE_APIS = frozenset(
    {
        "cudaEventQuery",
        "cudaEventRecord",
        "cudaEventRecordWithFlags",
        "cudaStreamWaitEvent",
    }
)
_BOUNDARY_ORIGIN_SUBREGION_PROOF_METADATA_EXPORT_ENV_KEYS = (
    "MAYA_ENABLE_BOUNDARY_ORIGIN_SUBREGION_PROOF_METADATA_EXPORT",
    "FLEXSIM_MAYA_ENABLE_BOUNDARY_ORIGIN_SUBREGION_PROOF_METADATA_EXPORT",
)
_BOUNDARY_ORIGIN_SUBREGION_PROOF_METADATA_SCHEMA_VERSION = (
    "boundary_origin_subregion_proof_metadata_v1"
)
_BOUNDARY_ORIGIN_SUBREGION_PROOF_TARGET_APIS = frozenset(
    {
        "cudaEventQuery",
        "cudaEventRecord",
        "cudaEventRecordWithFlags",
        "cudaLaunchKernel",
        "cublasSetStream_v2",
        "cublasGemmEx",
        "cublasGemmStridedBatchedEx",
        "cudaGetDevice",
    }
)
_BOUNDARY_ORIGIN_SUBREGION_PROOF_EVENT_APIS = frozenset(
    {
        "cudaEventQuery",
        "cudaEventRecord",
        "cudaEventRecordWithFlags",
    }
)
_BOUNDARY_ORIGIN_SUBREGION_PROOF_GEMM_APIS = frozenset(
    {
        "cublasGemmEx",
        "cublasGemmStridedBatchedEx",
    }
)
_STRICT_SUBREGION_EXTENT_SOURCE_METADATA_EXPORT_ENV_KEYS = (
    "MAYA_ENABLE_STRICT_SUBREGION_EXTENT_SOURCE_METADATA_EXPORT",
    "FLEXSIM_MAYA_ENABLE_STRICT_SUBREGION_EXTENT_SOURCE_METADATA_EXPORT",
)
_STRICT_SUBREGION_EXTENT_SOURCE_METADATA_SCHEMA_VERSION = (
    "strict_subregion_extent_source_metadata_v1"
)
_STRICT_SUBREGION_EXTENT_TARGET_APIS = _BOUNDARY_ORIGIN_SUBREGION_PROOF_TARGET_APIS
_STRICT_SUBREGION_EXTENT_EVENT_APIS = _BOUNDARY_ORIGIN_SUBREGION_PROOF_EVENT_APIS
_STRICT_SUBREGION_EXTENT_GEMM_APIS = _BOUNDARY_ORIGIN_SUBREGION_PROOF_GEMM_APIS

_HOST_CONTROL_ENVELOPE_COUNTERPART_ENV_KEYS = (
    "MAYA_ENABLE_HOST_CONTROL_ENVELOPE_COUNTERPART_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_HOST_CONTROL_ENVELOPE_COUNTERPART_DIAGNOSTICS",
)

_HOST_CONTROL_ENVELOPE_COUNTERPART_SCHEMA_VERSION = (
    "host_control_replay_envelope_counterpart_metadata_v1"
)
_GEMM_ADJACENT_HOSTDELAY_BOUNDARY_METADATA_ENV_KEYS = (
    "MAYA_ENABLE_GEMM_ADJACENT_HOSTDELAY_BOUNDARY_METADATA_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_GEMM_ADJACENT_HOSTDELAY_BOUNDARY_METADATA_DIAGNOSTICS",
    "MAYA_ENABLE_GEMM_HOSTDELAY_BOUNDARY_COUNTERPART_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_GEMM_HOSTDELAY_BOUNDARY_COUNTERPART_DIAGNOSTICS",
)
_GEMM_ADJACENT_HOSTDELAY_BOUNDARY_SCHEMA_VERSION = (
    "gemm_adjacent_hostdelay_boundary_counterpart_visibility_count_once_metadata_v1"
)
_COMPONENT_STRICT_COUNTERPART_METADATA_ENV_KEYS = (
    "MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
)
_COMPONENT_STRICT_COUNTERPART_SCHEMA_VERSION = (
    "component_strict_counterpart_metadata_evidence_v1"
)
_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_ENV_KEYS = (
    "MAYA_ENABLE_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_METADATA_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_METADATA_DIAGNOSTICS",
)
_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_SCHEMA_VERSION = (
    "cudaLaunch_GEMM_hostdispatch_strict_occurrence_gap_metadata_v1"
)
_JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_ENV_KEYS = (
    "MAYA_ENABLE_JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_METADATA_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_METADATA_DIAGNOSTICS",
)
_JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_SCHEMA_VERSION = (
    "joined_gemm_stream_queue_wait_actual_counterpart_metadata_v1"
)
_GEMM_ADJACENT_HOSTDELAY_GEMM_APIS = {
    "cublasGemmEx",
    "cublasGemmStridedBatchedEx",
}
_GEMM_ADJACENT_HOSTDELAY_ADJACENT_APIS = {
    "cublasSetStream_v2",
    "cudaLaunchKernel",
}
_CUDALAUNCH_GEMM_HOSTDISPATCH_TARGET_APIS = {
    "cudaLaunchKernel",
    "cublasGemmEx",
    "cublasGemmStridedBatchedEx",
}
_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_FIELDS = {
    "cuda_gemm_hostdispatch_strict_occurrence_gap_schema_version",
    "cuda_gemm_hostdispatch_strict_occurrence_gap_opt_in_flag",
    "cuda_gemm_hostdispatch_strict_occurrence_gap_source_side",
    "cuda_gemm_hostdispatch_strict_occurrence_gap_actual_row_id",
    "cuda_gemm_hostdispatch_strict_occurrence_gap_actual_endpoint_context_only",
    "cuda_gemm_hostdispatch_strict_occurrence_gap_predicted_row_id",
    "strict_occurrence_common_basis_key",
    "strict_occurrence_key_parts",
    "strict_occurrence_count_basis_side",
    "api_family",
    "component_role",
    "api_sequence_ordinal_in_window",
    "host_queue_sequence_ordinal_in_window",
    "stream_sequence_ordinal_in_window",
    "key_completeness_status",
    "actual_mechanical_dispatch_split_status",
    "actual_mechanical_dispatch_split_basis",
    "actual_control_dispatch_us",
    "actual_api_body_us",
    "actual_instrumentation_only_us",
    "actual_endpoint_timestamps_used_as_dispatch_split",
    "actual_host_duration_used_as_dispatch_split",
    "actual_runtime_used_as_dispatch_split",
    "actual_mechanical_dispatch_split_unavailable_reason",
    "stream_namespace_basis",
    "stream_alignment_status",
    "default_stream_equivalence_reviewed",
    "provider_runtime_overlap_status",
    "host_dispatch_overlap_status",
    "hostDelay_overlap_status",
    "stream_wait_overlap_status",
    "wait_map_blocking_overlap_status",
    "hostdispatch_producer_visibility_status",
    "hostdispatch_producer_visibility_basis",
    "paper_visible_host_dispatch_us",
    "instrumentation_only_host_dispatch_us",
    "unresolved_mixed_host_dispatch_us",
    "producer_side_safe_for_repair_design",
    "predicted_wait_map_edge_ids",
    "actual_wait_release_source_status",
    "dependency_release_timing_preserved",
    "cuda_event_wait_safety_status",
    "collective_wait_safety_status",
    "stream_queue_wait_safety_status",
    "strict_occurrence_join_ready",
    "strict_actual_timing_or_mechanical_split_ready",
    "strict_apples_to_apples_delta_ready",
    "safe_to_use_for_runtime_substitution",
    "safe_to_use_for_endpoint_timestamp_substitution",
}
_JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_FIELDS = {
    "joined_gemm_stream_queue_wait_actual_counterpart_schema_version",
    "joined_gemm_stream_queue_wait_actual_counterpart_opt_in_flag",
    "joined_gemm_stream_queue_wait_source_side",
    "joined_gemm_stream_queue_wait_actual_row_id",
    "joined_gemm_stream_queue_wait_actual_rank",
    "joined_gemm_stream_queue_wait_actual_api",
    "joined_gemm_stream_queue_wait_actual_raw_event_id",
    "joined_gemm_stream_queue_wait_actual_raw_ordinal",
    "joined_gemm_stream_queue_wait_actual_host_machine_id",
    "joined_gemm_stream_queue_wait_actual_host_dispatch_queue_id",
    "joined_gemm_stream_queue_wait_actual_stream_id",
    "joined_gemm_stream_queue_wait_actual_stream_resource_id",
    "joined_gemm_stream_queue_wait_actual_stream_namespace_basis",
    "joined_gemm_stream_queue_wait_actual_stream_sequence_ordinal",
    "joined_gemm_stream_queue_wait_previous_same_stream_raw_event_id",
    "joined_gemm_stream_queue_wait_previous_same_stream_raw_ordinal",
    "joined_gemm_stream_queue_wait_previous_same_stream_api",
    "joined_gemm_stream_queue_wait_previous_same_stream_material_signature",
    "joined_gemm_stream_queue_wait_previous_same_stream_algorithm",
    "joined_gemm_stream_queue_wait_previous_same_stream_sequence_ordinal",
    "joined_gemm_stream_queue_wait_previous_device_predecessor_raw_event_id",
    "joined_gemm_stream_queue_wait_previous_device_predecessor_raw_ordinal",
    "joined_gemm_stream_queue_wait_previous_device_predecessor_api",
    "joined_gemm_stream_queue_wait_previous_device_predecessor_material_signature",
    "joined_gemm_stream_queue_wait_previous_device_predecessor_algorithm",
    "joined_gemm_stream_queue_wait_previous_device_predecessor_sequence_ordinal",
    "joined_gemm_stream_queue_wait_previous_device_predecessor_status",
    "joined_gemm_stream_queue_wait_previous_device_predecessor_source",
    "joined_gemm_stream_queue_wait_previous_device_predecessor_kernel_start_cupti_timestamp",
    "joined_gemm_stream_queue_wait_previous_device_predecessor_kernel_end_cupti_timestamp",
    "joined_gemm_stream_queue_wait_previous_device_predecessor_cupti_kernel_stream_id",
    "joined_gemm_stream_queue_wait_previous_device_predecessor_cupti_kernel_stream_id_pair_status",
    "joined_gemm_stream_queue_wait_previous_device_predecessor_stream_order_gap_cupti_ticks",
    "joined_gemm_stream_queue_wait_actual_material_signature",
    "joined_gemm_stream_queue_wait_actual_algorithm",
    "joined_gemm_stream_queue_wait_actual_gemm_shape_signature",
    "joined_gemm_stream_queue_wait_actual_stream_order_pair_id",
    "joined_gemm_stream_queue_wait_actual_stream_order_pair_basis",
    "joined_gemm_stream_queue_wait_actual_counterpart_join_status",
    "joined_gemm_stream_queue_wait_actual_counterpart_join_basis",
    "joined_gemm_stream_queue_wait_actual_release_timing_status",
    "joined_gemm_stream_queue_wait_actual_wait_timing_status",
    "joined_gemm_stream_queue_wait_actual_wait_release_timing_unavailable_reason",
    "joined_gemm_stream_queue_wait_actual_wait_start_us",
    "joined_gemm_stream_queue_wait_actual_release_us",
    "joined_gemm_stream_queue_wait_actual_waited_us",
    "joined_gemm_stream_queue_wait_actual_device_timing_source",
    "joined_gemm_stream_queue_wait_actual_previous_kernel_start_cupti_timestamp",
    "joined_gemm_stream_queue_wait_actual_previous_kernel_end_cupti_timestamp",
    "joined_gemm_stream_queue_wait_actual_current_kernel_start_cupti_timestamp",
    "joined_gemm_stream_queue_wait_actual_current_kernel_end_cupti_timestamp",
    "joined_gemm_stream_queue_wait_actual_current_cupti_kernel_stream_id",
    "joined_gemm_stream_queue_wait_actual_previous_cupti_kernel_stream_id",
    "joined_gemm_stream_queue_wait_actual_cupti_kernel_stream_id_pair_status",
    "joined_gemm_stream_queue_wait_actual_cupti_common_clock_status",
    "joined_gemm_stream_queue_wait_actual_stream_order_gap_cupti_ticks",
    "joined_gemm_stream_queue_wait_actual_endpoint_ts_us",
    "joined_gemm_stream_queue_wait_actual_endpoint_end_ts_us",
    "joined_gemm_stream_queue_wait_actual_endpoint_host_duration_us",
    "joined_gemm_stream_queue_wait_actual_observed_runtime_us",
    "joined_gemm_stream_queue_wait_actual_endpoint_context_only",
    "joined_gemm_stream_queue_wait_endpoint_timestamps_used_as_wait_release",
    "joined_gemm_stream_queue_wait_endpoint_timestamps_used_as_strict_delta",
    "joined_gemm_stream_queue_wait_actual_runtime_direct_substitution",
    "joined_gemm_stream_queue_wait_actual_observed_runtime_used_as_prediction",
    "joined_gemm_stream_queue_wait_strict_actual_timing_available",
    "joined_gemm_stream_queue_wait_strict_delta_calculable",
    "joined_gemm_stream_queue_wait_count_once_status",
    "joined_gemm_stream_queue_wait_count_once_nonoverlap_status",
    "joined_gemm_stream_queue_wait_nonoverlap_status",
    "joined_gemm_stream_queue_wait_wait_map_safety_status",
    "joined_gemm_stream_queue_wait_wait_map_safety_proven",
    "joined_gemm_stream_queue_wait_producer_visibility_status",
    "joined_gemm_stream_queue_wait_repair_ready",
    "joined_gemm_stream_queue_wait_safe_to_use_as_repair_evidence",
    "joined_gemm_stream_queue_wait_safe_to_use_as_subtraction_delta",
    "joined_gemm_stream_queue_wait_safe_to_use_for_runtime_substitution",
    "joined_gemm_stream_queue_wait_safe_to_use_for_endpoint_timestamp_substitution",
}
_GEMM_ADJACENT_HOSTDELAY_BOUNDARY_FIELDS = {
    "gemm_adjacent_hostdelay_schema_version",
    "gemm_adjacent_hostdelay_opt_in_flag",
    "gemm_adjacent_source_side",
    "gemm_adjacent_stable_boundary_row_id",
    "gemm_adjacent_rank",
    "gemm_adjacent_paper_valid_window_id",
    "gemm_adjacent_paper_valid_window_membership_status",
    "gemm_adjacent_materialized_hostdelay_event_id",
    "gemm_adjacent_hostdelay_event_id",
    "gemm_adjacent_hostdelay_source",
    "gemm_adjacent_observed_gap_us",
    "gemm_adjacent_hostdelay_duration_us",
    "gemm_adjacent_host_dispatch_queue_id",
    "gemm_adjacent_host_machine_id",
    "gemm_adjacent_boundary_direction",
    "gemm_adjacent_target_gemm_api",
    "gemm_adjacent_adjacent_api",
    "gemm_adjacent_boundary_family_in_design_scope",
    "gemm_adjacent_previous_stream_id",
    "gemm_adjacent_previous_stream_resource_id",
    "gemm_adjacent_current_stream_id",
    "gemm_adjacent_current_stream_resource_id",
    "gemm_adjacent_stream_namespace_basis",
    "gemm_adjacent_stream_namespace_alignment_status",
    "gemm_adjacent_material_signature",
    "gemm_adjacent_algorithm",
    "gemm_adjacent_gemm_shape_signature",
    "gemm_adjacent_gemm_material_metadata_status",
    "gemm_adjacent_actual_counterpart_schema_version",
    "gemm_adjacent_actual_counterpart_opt_in_flag",
    "gemm_adjacent_actual_counterpart_row_id",
    "gemm_adjacent_actual_counterpart_candidate_kind",
    "gemm_adjacent_actual_rank",
    "gemm_adjacent_actual_api",
    "gemm_adjacent_actual_type",
    "gemm_adjacent_actual_raw_event_id",
    "gemm_adjacent_actual_raw_ordinal",
    "gemm_adjacent_actual_trace_id",
    "gemm_adjacent_actual_host_machine_id",
    "gemm_adjacent_actual_host_dispatch_queue_id",
    "gemm_adjacent_actual_paper_valid_window_id",
    "gemm_adjacent_actual_in_paper_valid_window",
    "gemm_adjacent_actual_prev_raw_event_id",
    "gemm_adjacent_actual_prev_api",
    "gemm_adjacent_actual_prev_ts_us",
    "gemm_adjacent_actual_prev_end_ts_us",
    "gemm_adjacent_actual_next_raw_event_id",
    "gemm_adjacent_actual_next_api",
    "gemm_adjacent_actual_next_ts_us",
    "gemm_adjacent_actual_next_end_ts_us",
    "gemm_adjacent_actual_raw_boundary_family_prev_to_current",
    "gemm_adjacent_actual_raw_boundary_family_current_to_next",
    "gemm_adjacent_actual_boundary_family_in_design_scope",
    "gemm_adjacent_actual_endpoint_ts_us",
    "gemm_adjacent_actual_endpoint_end_ts_us",
    "gemm_adjacent_actual_endpoint_host_duration_us",
    "gemm_adjacent_actual_observed_runtime_us",
    "gemm_adjacent_actual_wrapper_runtime_contract",
    "gemm_adjacent_actual_timing_status",
    "gemm_adjacent_actual_timing_basis",
    "gemm_adjacent_actual_timing_unavailable_reason",
    "gemm_adjacent_actual_endpoint_context_only",
    "gemm_adjacent_actual_endpoint_timestamps_used_as_strict_timing",
    "gemm_adjacent_actual_runtime_direct_substitution",
    "gemm_adjacent_actual_stream_id",
    "gemm_adjacent_actual_raw_stream_id",
    "gemm_adjacent_actual_stream_resource_id",
    "gemm_adjacent_actual_stream_namespace_basis",
    "gemm_adjacent_actual_material_signature",
    "gemm_adjacent_actual_algorithm",
    "gemm_adjacent_actual_gemm_shape_signature",
    "gemm_adjacent_actual_gemm_metadata_status",
    "gemm_adjacent_predicted_stable_boundary_row_id",
    "gemm_adjacent_predicted_materialized_hostdelay_event_id",
    "gemm_adjacent_predicted_count_once_group_id",
    "gemm_adjacent_predicted_count_once_interval_id",
    "gemm_adjacent_predicted_replay_interval_id",
    "gemm_adjacent_predicted_replay_component_kind",
    "gemm_adjacent_predicted_replay_resource_kind",
    "gemm_adjacent_predicted_replay_resource_id",
    "gemm_adjacent_predicted_start_us",
    "gemm_adjacent_predicted_end_us",
    "gemm_adjacent_predicted_duration_us",
    "gemm_adjacent_counterpart_join_key",
    "gemm_adjacent_counterpart_join_basis",
    "gemm_adjacent_counterpart_join_attempted_during_capture",
    "gemm_adjacent_counterpart_join_status",
    "gemm_adjacent_counterpart_join_confidence",
    "gemm_adjacent_counterpart_unavailable_reason",
    "gemm_adjacent_actual_counterpart_join_key",
    "gemm_adjacent_actual_counterpart_join_status",
    "gemm_adjacent_actual_counterpart_join_basis",
    "gemm_adjacent_actual_counterpart_join_confidence",
    "gemm_adjacent_actual_counterpart_unavailable_reason",
    "gemm_adjacent_actual_start_us",
    "gemm_adjacent_actual_end_us",
    "gemm_adjacent_actual_duration_us",
    "gemm_adjacent_producer_visibility_schema_version",
    "gemm_adjacent_producer_visibility_status",
    "gemm_adjacent_producer_visibility_basis",
    "gemm_adjacent_producer_visibility_unavailable_reason",
    "gemm_adjacent_boundary_origin_kind",
    "gemm_adjacent_boundary_visibility_kind",
    "gemm_adjacent_classification_basis",
    "gemm_adjacent_paper_visible_host_duration_us",
    "gemm_adjacent_instrumentation_only_duration_us",
    "gemm_adjacent_unresolved_mixed_duration_us",
    "gemm_adjacent_actual_control_dispatch_us",
    "gemm_adjacent_actual_api_body_us",
    "gemm_adjacent_actual_instrumentation_only_us",
    "gemm_adjacent_wrapper_segment_sum_us",
    "gemm_adjacent_wrapper_segment_unattributed_us",
    "gemm_adjacent_producer_visibility_segments",
    "gemm_adjacent_boundary_visibility_segments",
    "gemm_adjacent_split_sum_check_status",
    "gemm_adjacent_split_sum_check_delta_us",
    "gemm_adjacent_actual_count_once_group_id",
    "gemm_adjacent_actual_count_once_interval_id",
    "gemm_adjacent_count_once_status",
    "gemm_adjacent_count_once_non_overlap_status",
    "gemm_adjacent_count_once_non_overlap_unavailable_reason",
    "gemm_adjacent_double_counting_overlap_status",
    "gemm_adjacent_double_counting_overlap_unavailable_reason",
    "gemm_adjacent_wait_map_safety_status",
    "gemm_adjacent_wait_map_safety_unavailable_reason",
    "gemm_adjacent_stream_fifo_nonoverlap_status",
    "gemm_adjacent_host_queue_nonoverlap_status",
    "gemm_adjacent_rank_envelope_nonoverlap_status",
    "gemm_adjacent_strict_nonoverlap_proof_basis",
    "gemm_adjacent_repair_ready",
    "gemm_adjacent_safe_to_use_as_repair_evidence",
    "gemm_adjacent_safe_to_use_as_subtraction_delta",
}
_COMPONENT_STRICT_COUNTERPART_FIELDS = {
    "component_strict_counterpart_schema_version",
    "component_strict_counterpart_opt_in_flag",
    "source_side",
    "stable_predicted_component_row_id",
    "stable_predicted_interval_row_id",
    "stable_predicted_edge_row_id",
    "stable_predicted_count_once_group_id",
    "predicted_row_identity_basis",
    "component_kind",
    "predicted_replay_component",
    "api_or_kernel_family",
    "api_or_kernel_family_role",
    "paper_valid_window_id",
    "paper_valid_window_membership_status",
    "predicted_interval_start_us",
    "predicted_interval_end_us",
    "predicted_interval_duration_us",
    "predicted_interval_duration_basis",
    "predicted_interval_resource_kind",
    "predicted_interval_resource_id",
    "predicted_interval_origin_status",
    "predicted_interval_visibility_status",
    "actual_counterpart_join_status",
    "actual_counterpart_join_basis",
    "strict_actual_timing_status",
    "strict_actual_timing_available",
    "actual_start_us",
    "actual_end_us",
    "actual_duration_us",
    "actual_endpoint_timestamps_used_as_strict_timing",
    "actual_host_duration_used_as_strict_timing",
    "actual_runtime_direct_substitution",
    "actual_observed_runtime_used_as_prediction",
    "stream_namespace_alignment_status",
    "exact_stream_identity_proven",
    "count_once_status",
    "nonoverlap_status",
    "wait_map_safety_status",
    "producer_visibility_status",
    "paper_maya_tags",
    "repair_ready",
    "safe_to_use_as_repair_evidence",
    "safe_to_use_as_subtraction_delta",
}
_COMPONENT_STRICT_ACTUAL_ENDPOINT_SIDECAR_ONLY_FIELDS = {
    "actual_trace_id",
    "actual_trace_id_unavailable_reason",
    "actual_rank",
    "actual_paper_valid_window_id",
    "actual_paper_valid_window_unavailable_reason",
}


def _env_flag_enabled(*keys: str) -> bool:
    for key in keys:
        if os.environ.get(key) == "1":
            return True
    return False


def _env_flag_truthy(*keys: str) -> bool:
    truthy = {"1", "true", "yes", "on"}
    for key in keys:
        value = os.environ.get(key)
        if value is not None and value.strip().lower() in truthy:
            return True
    return False


def _env_flag_falsey(*keys: str) -> bool:
    falsey = {"0", "false", "no", "off"}
    for key in keys:
        value = os.environ.get(key)
        if value is not None and value.strip().lower() in falsey:
            return True
    return False


def _launch_config_hostdelay_normalization_policy() -> tuple[bool, str]:
    if _env_flag_truthy(*_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION_DISABLE_ENV_KEYS):
        return False, "disabled_explicit_disable_control_hostdelay_preserved"
    if _env_flag_falsey(*_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION_ENV_KEYS):
        return False, "disabled_enable_zero_control_hostdelay_preserved"
    if _env_flag_truthy(*_LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION_ENV_KEYS):
        return True, "enabled_explicit_enable_excluded_from_hostdelay"
    return True, "enabled_default_excluded_from_hostdelay"


def _launch_config_hostdelay_normalization_enabled() -> bool:
    enabled, _ = _launch_config_hostdelay_normalization_policy()
    return enabled


def _launch_config_pop_entry_hostdelay_boundary_enabled() -> bool:
    return _env_flag_truthy(*_LAUNCH_CONFIG_POP_ENTRY_HOSTDELAY_BOUNDARY_ENV_KEYS)


def _idempotent_cublas_set_stream_fold_enabled() -> bool:
    return _env_flag_enabled(*_IDEMPOTENT_CUBLAS_SET_STREAM_FOLD_ENV_KEYS)


def _context_query_suffix_event_record_fold_enabled() -> bool:
    return _env_flag_truthy(*_CONTEXT_QUERY_SUFFIX_EVENT_RECORD_FOLD_ENV_KEYS)


def _context_query_suffix_launch_config_pop_fold_enabled() -> bool:
    return _env_flag_truthy(*_CONTEXT_QUERY_SUFFIX_LAUNCH_CONFIG_POP_FOLD_ENV_KEYS)


def _hostdelay_occurrence_metadata_export_enabled() -> bool:
    return _env_flag_truthy(*_HOSTDELAY_OCCURRENCE_METADATA_EXPORT_ENV_KEYS)


def _hostdelay_occurrence_join_metadata_export_enabled() -> bool:
    return _env_flag_truthy(*_HOSTDELAY_OCCURRENCE_JOIN_METADATA_EXPORT_ENV_KEYS)


def _collective_event_polling_boundary_metadata_export_enabled() -> bool:
    return _env_flag_truthy(
        *_COLLECTIVE_EVENT_POLLING_BOUNDARY_METADATA_EXPORT_ENV_KEYS
    )


def _event_polling_boundary_metadata_export_enabled() -> bool:
    return _env_flag_truthy(*_EVENT_POLLING_BOUNDARY_METADATA_EXPORT_ENV_KEYS)


def _boundary_origin_subregion_proof_metadata_export_enabled() -> bool:
    return _env_flag_truthy(
        *_BOUNDARY_ORIGIN_SUBREGION_PROOF_METADATA_EXPORT_ENV_KEYS
    )


def _strict_subregion_extent_source_metadata_export_enabled() -> bool:
    return _env_flag_truthy(
        *_STRICT_SUBREGION_EXTENT_SOURCE_METADATA_EXPORT_ENV_KEYS
    )


def _host_control_envelope_counterpart_diagnostics_enabled() -> bool:
    return _env_flag_enabled(*_HOST_CONTROL_ENVELOPE_COUNTERPART_ENV_KEYS)


def _gemm_adjacent_hostdelay_boundary_metadata_diagnostics_enabled() -> bool:
    return _env_flag_truthy(*_GEMM_ADJACENT_HOSTDELAY_BOUNDARY_METADATA_ENV_KEYS)


def _component_strict_counterpart_metadata_diagnostics_enabled() -> bool:
    return _env_flag_truthy(*_COMPONENT_STRICT_COUNTERPART_METADATA_ENV_KEYS)


def _cuda_gemm_hostdispatch_strict_occurrence_gap_diagnostics_enabled() -> bool:
    return _env_flag_truthy(
        *_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_ENV_KEYS
    )


def _launch_neighborhood_equivalence_diagnostics_enabled() -> bool:
    return _env_flag_enabled(*LAUNCH_NEIGHBORHOOD_EQUIVALENCE_ENV_KEYS)


def _event_id(event: TraceEvent) -> str:
    return f"r{event.rank}:e{event.ordinal}"


def _host_delay_id(rank: int, before_ordinal: int) -> str:
    return f"r{rank}:h{before_ordinal}"


def _boundary_host_delay_id(rank: int, anchor_ordinal: int, boundary: str) -> str:
    return f"r{rank}:h{anchor_ordinal}:{boundary}"


def _host_machine_id(event: TraceEvent) -> str:
    host_machine_id = _normalized_text(event.extras.get("host_machine_id"))
    if host_machine_id is not None:
        return host_machine_id
    return f"legacy_pid:{int(event.pid)}"


def _host_dispatch_queue_id(event: TraceEvent) -> str:
    dispatch_queue_id = _normalized_text(event.extras.get("host_dispatch_queue_id"))
    if dispatch_queue_id is not None:
        return dispatch_queue_id
    return _host_machine_id(event)


def _host_dispatch_model(dispatch_scope: str) -> str:
    if dispatch_scope == "thread":
        return "dispatch_queue_per_host_thread"
    if dispatch_scope == "process":
        return "single_dispatch_queue_per_process"
    return "single_dispatch_queue_per_host_execution_context"


def _host_dispatch_scope(
    event: TraceEvent,
    *,
    default_scope: str | None = None,
) -> str:
    raw_dispatch_scope = _normalized_text(event.extras.get("host_timing_dispatch_scope"))
    if raw_dispatch_scope in {"thread", "process", "host_machine"}:
        return raw_dispatch_scope
    if default_scope in {"thread", "process", "host_machine"}:
        return str(default_scope)
    if raw_dispatch_scope is not None and _uses_process_scoped_direct_wallclock_queue(event):
        return raw_dispatch_scope
    return "host_machine"


def _host_dispatch_key(
    event: TraceEvent,
    *,
    default_scope: str | None = None,
) -> tuple[object, ...]:
    dispatch_scope = _host_dispatch_scope(event, default_scope=default_scope)
    if dispatch_scope == "thread":
        host_dispatch_queue_id = _host_dispatch_queue_id(event)
        return (host_dispatch_queue_id, int(event.pid), int(event.tid))
    if dispatch_scope == "process":
        host_dispatch_queue_id = _host_dispatch_queue_id(event)
        return (host_dispatch_queue_id, int(event.pid))
    return (_host_dispatch_queue_id(event),)


def _is_collective(event: TraceEvent) -> bool:
    return is_collective_api(event.api, event.op_type)


def _p2p_group_member(peer: object) -> int | str:
    try:
        return int(str(peer))
    except (TypeError, ValueError):
        return str(peer)


def _normalized_text(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _normalized_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _normalized_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _event_host_duration_us(event: TraceEvent) -> int:
    observed = _normalized_float(event.extras.get("host_duration_us"))
    if observed is not None:
        return max(int(round(observed)), 0)
    end_ts = _normalized_float(event.extras.get("end_ts"))
    if end_ts is None:
        return 0
    return max(int(round(end_ts - int(event.ts))), 0)


def _event_end_ts_context(event: TraceEvent) -> tuple[float | None, str]:
    end_ts = _normalized_float(event.extras.get("end_ts"))
    if end_ts is not None:
        return end_ts, "raw_end_ts"
    host_duration_us = _normalized_float(event.extras.get("host_duration_us"))
    if host_duration_us is not None:
        return float(int(event.ts)) + max(host_duration_us, 0.0), "computed_from_host_duration_us"
    return None, "unavailable"


def _launch_config_pop_entry_hostdelay_adjustment_us(event: TraceEvent) -> int:
    if not _launch_config_pop_entry_hostdelay_boundary_enabled():
        return 0
    if event.api != "__cudaPopCallConfiguration":
        return 0
    mode = _normalized_text(event.extras.get("host_timing_mode"))
    source = _normalized_text(event.extras.get("host_timing_source"))
    if mode != "measure" or source != "direct_wallclock":
        return 0
    return _event_host_duration_us(event)


def _uses_process_scoped_direct_wallclock_queue(event: TraceEvent) -> bool:
    mode = _normalized_text(event.extras.get("host_timing_mode"))
    source = _normalized_text(event.extras.get("host_timing_source"))
    dispatch_scope = _normalized_text(event.extras.get("host_timing_dispatch_scope"))
    if mode != "measure" or source != "direct_wallclock":
        return False
    if dispatch_scope in (None, "thread", "process", "host_machine"):
        return is_host_timing_traced_api(event.api, event.op_type)
    return False


_REAL_HOST_RUNTIME_PROJECTED_APIS = {
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cudaMalloc",
    "cudaMallocAsync",
    "cudaFree",
    "cudaFreeAsync",
    "cudaMemGetInfo",
    "ncclCommCount",
    "ncclCommInitRank",
    "ncclCommInitRankConfig",
    "ncclCommUserRank",
}


def _uses_real_host_runtime_projection(event: TraceEvent) -> bool:
    if event.source is not TraceSource.REAL:
        return False
    if not occupies_host_dispatch_resource(event.api, event.op_type):
        return False
    if is_ignorable_setup_api(event.api):
        return False
    if is_low_overhead_api(event.api):
        return True
    return event.api in _REAL_HOST_RUNTIME_PROJECTED_APIS


def _projected_host_dispatch_duration_us(event: TraceEvent) -> int:
    if _uses_process_scoped_direct_wallclock_queue(event):
        return _event_host_duration_us(event)
    if not _uses_real_host_runtime_projection(event):
        return 0
    if is_low_overhead_api(event.api):
        return 1
    return _event_host_duration_us(event)


def _uses_direct_wallclock_dispatch_only_projection(event: TraceEvent) -> bool:
    if not _uses_process_scoped_direct_wallclock_queue(event):
        return False
    contract = str(event.extras.get("wrapper_runtime_contract") or "").strip().lower()
    return contract == "dispatch_only"


def _advances_projected_host_dispatch_queue(
    event: TraceEvent,
    *,
    projected_host_dispatch_duration_us: int,
) -> bool:
    if projected_host_dispatch_duration_us <= 0:
        return False
    return (
        _uses_real_host_runtime_projection(event)
        or _uses_direct_wallclock_dispatch_only_projection(event)
    )


def _uses_real_async_runtime_observation_high_watermark(event: TraceEvent) -> bool:
    if event.source is not TraceSource.REAL:
        return False
    if event.api == "cudaLaunchKernel":
        # capture_real CUDA-event timing on raw kernel-launch wrappers is much
        # finer grained than Maya's host-dispatch envelope. Advancing the host
        # dispatch high watermark on every driver-level kernel launch can erase
        # legitimate host_delay gaps on PP-heavy traces. Keep the observed
        # runtime on the accelerator stream, but do not let raw launch wrappers
        # shift the host-gap boundary.
        return False
    if event.op_type not in {"kernel_launch", "blas_compute", "nccl_collective"} and event.api != "cudaLaunchKernel":
        return False
    if str(event.extras.get("wrapper_runtime_contract") or "").strip().lower() != "async_runtime":
        return False
    if str(event.extras.get("runtime_observation_source") or "").strip() != "capture_real_cuda_event":
        return False
    observed_runtime = _normalized_float(event.extras.get("observed_runtime_us"))
    if observed_runtime is None or observed_runtime <= 0.0:
        return False
    return True


def _dispatch_gap_high_watermark_ts(
    event: TraceEvent,
    *,
    effective_event_ts: int,
    projected_host_dispatch_duration_us: int,
) -> int:
    # Real traces can expose blocking host-side runtimes for sync/allocation
    # APIs; those should advance the dispatch high watermark so following
    # host_delay captures only time after the blocking call completes.  Emulated
    # direct-wallclock wrapper duration must do the same only when annotation /
    # replay will consume that wrapper duration as dispatch_only host-queue
    # occupancy; legacy direct-wallclock rows without that contract keep their
    # wall-clock host overhead in the hostDelay interval.
    if _advances_projected_host_dispatch_queue(
        event,
        projected_host_dispatch_duration_us=projected_host_dispatch_duration_us,
    ):
        return int(effective_event_ts) + int(projected_host_dispatch_duration_us)
    if _uses_real_async_runtime_observation_high_watermark(event):
        # Real traces may attach authoritative async runtime observations to
        # launch / collective wrappers via cuda-event timing. Those wrappers are
        # not host_delay gaps; their observed end_ts/host_duration_us already
        # spans the measured runtime envelope. Advance the high watermark so we
        # do not materialize the same observed runtime again as a following
        # host_delay.
        return int(effective_event_ts) + _event_host_duration_us(event)
    return int(effective_event_ts)


def _extract_first_int(event: TraceEvent, *keys: str) -> int | None:
    for key in keys:
        resolved = _normalized_int(event.extras.get(key))
        if resolved is not None:
            return resolved
    return None


def _resolve_communicator_members(
    event: TraceEvent,
    communicator_memberships: dict[str, tuple[int, ...]],
) -> tuple[int, ...] | None:
    comm_id = _normalized_text(event.extras.get("comm_id"))
    if comm_id is None:
        return None
    members = communicator_memberships.get(comm_id)
    if members:
        return members
    return parse_emulated_communicator_id(comm_id)


def _p2p_pair_sequence_number(group_id: str) -> int | None:
    marker = "|pair_seq:"
    if marker not in group_id:
        return None
    try:
        return int(group_id.rsplit(marker, 1)[1])
    except ValueError:
        return None


def _resolve_p2p_members(
    event: TraceEvent,
    communicator_members: tuple[int, ...] | None,
) -> tuple[int | str, int | str] | None:
    if event.api not in {"ncclSend", "ncclRecv"} or event.extras.get("peer") in (None, ""):
        return None
    peer_local_rank = _normalized_int(event.extras.get("peer"))
    peer: int | str
    if (
        communicator_members is not None
        and peer_local_rank is not None
        and 0 <= peer_local_rank < len(communicator_members)
    ):
        peer = int(communicator_members[peer_local_rank])
    else:
        # Without communicator membership metadata, peer is only a stable
        # per-communicator hint and may still be a group-local rank.
        peer = _p2p_group_member(event.extras.get("peer"))
    return tuple(
        sorted(
            (event.rank, peer),
            key=lambda value: (isinstance(value, str), value if isinstance(value, int) else str(value)),
        )
    )


@dataclass(frozen=True)
class _CollectiveDescriptor:
    key: tuple[object, ...]
    group_id_base: str
    match_basis: str
    communicator_id: str | None = None
    sequence_number: int | None = None
    communicator_size: int | None = None
    participant_count: int | None = None
    root: int | None = None


def _collective_signature(event: TraceEvent) -> str:
    comm_id = event.extras.get("comm_id")
    call_idx = event.extras.get("call_idx")
    if comm_id not in (None, "", "0", 0) and call_idx not in (None, "", "0", 0):
        if event.api in {"ncclSend", "ncclRecv"}:
            peer = _p2p_group_member(event.extras.get("peer"))
            members = tuple(
                sorted(
                    (event.rank, peer),
                    key=lambda value: (isinstance(value, str), value if isinstance(value, int) else str(value)),
                )
            )
            parts = [
                "ncclP2P",
                f"comm:{comm_id}",
                f"call:{call_idx}",
                f"members:{members[0]}-{members[1]}",
            ]
            return "|".join(parts)
        parts = [
            event.api,
            f"comm:{comm_id}",
            f"call:{call_idx}",
            f"peer:{event.extras.get('peer', '')}",
            f"root:{event.extras.get('root', '')}",
        ]
        return "|".join(parts)
    parts = [
        event.api,
        str(event.extras.get("collective", "")),
        str(event.extras.get("comm_id", "")),
        str(event.extras.get("count", event.extras.get("numel", ""))),
        str(event.extras.get("datatype", event.extras.get("dtype_code", ""))),
        str(event.extras.get("op", "")),
    ]
    return "|".join(parts)


def _collective_descriptor(
    event: TraceEvent,
    communicator_memberships: dict[str, tuple[int, ...]],
) -> _CollectiveDescriptor:
    communicator_id = _normalized_text(event.extras.get("comm_id"))
    sequence_number = _normalized_int(event.extras.get("call_idx"))
    communicator_members = _resolve_communicator_members(event, communicator_memberships)
    p2p_members = _resolve_p2p_members(event, communicator_members)
    if p2p_members is not None:
        # NCCL send/recv forms a pair-level wait-map group even when the
        # backing communicator covers a larger pipeline/data-parallel cohort.
        communicator_size = len(p2p_members)
    elif communicator_members is not None:
        # Communicator topology is authoritative: raw real-trace world_size is
        # often the process group size, not the active NCCL communicator size.
        communicator_size = len(communicator_members)
    else:
        communicator_size = _extract_first_int(event, "nranks", "num_ranks", "nRanks", "world_size")
    root = _normalized_int(event.extras.get("root"))
    if root is not None and communicator_members is not None and 0 <= root < len(communicator_members):
        root = int(communicator_members[root])
    if p2p_members is not None:
        participant_count = 2
    else:
        participant_count = communicator_size

    if communicator_id is not None:
        if sequence_number is not None and p2p_members is not None:
            return _CollectiveDescriptor(
                key=("communicator_pair_sequence", "p2p", communicator_id, p2p_members),
                group_id_base=f"ncclP2P|comm:{communicator_id}|members:{p2p_members[0]}-{p2p_members[1]}",
                match_basis="communicator_pair_sequence",
                communicator_id=communicator_id,
                sequence_number=sequence_number,
                communicator_size=communicator_size,
                participant_count=len(p2p_members),
                root=root,
            )
        if sequence_number is not None:
            return _CollectiveDescriptor(
                key=("communicator_sequence", communicator_id, sequence_number),
                group_id_base=f"{event.api}|comm:{communicator_id}|call:{sequence_number}",
                match_basis="communicator_sequence",
                communicator_id=communicator_id,
                sequence_number=sequence_number,
                communicator_size=communicator_size,
                participant_count=participant_count,
                root=root,
            )
        if p2p_members is not None:
            count = str(event.extras.get("count", event.extras.get("numel", "")))
            datatype = str(event.extras.get("datatype", event.extras.get("dtype_code", "")))
            op = str(event.extras.get("op", ""))
            return _CollectiveDescriptor(
                key=("communicator_pair_payload", "p2p", communicator_id, p2p_members, count, datatype, op),
                group_id_base=(
                    f"ncclP2P|comm:{communicator_id}|members:{p2p_members[0]}-{p2p_members[1]}"
                    f"|count:{count}|dtype:{datatype}|op:{op}"
                ),
                match_basis="communicator_pair_payload",
                communicator_id=communicator_id,
                sequence_number=None,
                communicator_size=communicator_size,
                participant_count=len(p2p_members),
                root=root,
            )

    signature = _collective_signature(event)
    return _CollectiveDescriptor(
        key=("payload_signature", signature),
        group_id_base=signature,
        match_basis="payload_signature",
        communicator_id=communicator_id,
        sequence_number=sequence_number,
        communicator_size=communicator_size,
        participant_count=participant_count,
        root=root,
    )


def _should_insert_host_delay(previous_event: TraceEvent, event: TraceEvent) -> bool:
    # Semantic conformance is checked on the stricter semantic-traced surface,
    # but host-delay must preserve observed CPU-side wall-clock gaps around both
    # semantic and compat/control-plane APIs. Unsupported APIs should still fail
    # loudly upstream instead of silently participating here.
    return is_host_timing_traced_api(event.api, event.op_type)


def _should_materialize_replay_event(event: TraceEvent) -> bool:
    """Return True when an API should remain explicit in the job-level trace.

    Compat-only calls keep PyTorch/framework execution alive, but Maya's
    paper-facing replay semantics are carried by semantic_traced events plus
    hostDelay.  Collapsing compat-only APIs here preserves their observed
    wall-clock envelope through hostDelay without letting fake-only setup or
    control-plane chatter perturb replay ordering.
    """

    return not is_compat_only_api(event.api, event.op_type)


def _collective_group_event_metadata(group: CollectiveGroup) -> dict[str, object]:
    metadata: dict[str, object] = {
        "collective_api": group.api,
        "collective_match_basis": group.match_basis,
    }
    if group.communicator_id is not None:
        metadata["collective_communicator_id"] = group.communicator_id
    if group.sequence_number is not None:
        metadata["collective_sequence_number"] = int(group.sequence_number)
    if group.communicator_size is not None:
        metadata["communicator_size"] = int(group.communicator_size)
    if group.participant_count is not None:
        metadata["participant_count"] = int(group.participant_count)
    if group.root is not None:
        metadata["collective_root"] = int(group.root)
    return metadata


def _hostdelay_occurrence_boundary_metadata(
    previous_event: TraceEvent | None,
    current_event: TraceEvent,
    *,
    observed_gap_us: int | None,
    host_dispatch_queue_id: str | None,
    previous_materialized_event_id: str | None,
    previous_materialized_api: str | None,
    current_materialized_event_id: str | None,
    current_materialized_api: str | None,
    paper_valid_window_membership: Mapping[str, object] | None,
) -> dict[str, object]:
    join_metadata_enabled = _hostdelay_occurrence_join_metadata_export_enabled()
    previous_ordinal = int(previous_event.ordinal) if previous_event else None
    current_ordinal = int(current_event.ordinal)
    previous_key = (
        "leading" if previous_ordinal is None else f"raw_ordinal:{previous_ordinal}"
    )
    raw_boundary_family = _boundary_family(
        previous_event.api if previous_event else None,
        current_event.api,
    )
    semantic_boundary_family = _boundary_family(
        previous_materialized_api,
        current_materialized_api,
    )
    stable_boundary_key = (
        f"rank:{int(current_event.rank)}:hostdelay_occurrence:"
        f"{previous_key}->raw_ordinal:{current_ordinal}"
    )
    comparable_key = (
        "hostdelay_occurrence:"
        f"raw:{raw_boundary_family}|semantic:{semantic_boundary_family}|"
        f"prev:{previous_key}|current:raw_ordinal:{current_ordinal}"
    )
    paper_window = dict(paper_valid_window_membership or {})
    metadata: dict[str, object] = {
        "hostdelay_occurrence_metadata_schema_version": (
            _HOSTDELAY_OCCURRENCE_METADATA_SCHEMA_VERSION
        ),
        "hostdelay_occurrence_metadata_opt_in_flag": True,
        "hostdelay_occurrence_metadata_env_flags": list(
            _HOSTDELAY_OCCURRENCE_METADATA_EXPORT_ENV_KEYS
        ),
        "stable_hostdelay_occurrence_id": stable_boundary_key,
        "hostdelay_occurrence_id_basis": (
            "rank_and_raw_predecessor_successor_ordinals"
        ),
        "hostdelay_occurrence_rank": int(current_event.rank),
        "hostdelay_occurrence_pid": int(current_event.pid),
        "hostdelay_occurrence_tid": int(current_event.tid),
        "hostdelay_occurrence_thread_id": int(current_event.tid),
        "hostdelay_occurrence_dispatch_key": host_dispatch_queue_id,
        "hostdelay_occurrence_host_dispatch_queue_id": host_dispatch_queue_id,
        "hostdelay_occurrence_stream_id": current_event.extras.get("stream_id"),
        "hostdelay_occurrence_paper_valid_window_id": paper_window.get("window_id"),
        "hostdelay_occurrence_raw_predecessor_event_id": (
            _event_id(previous_event) if previous_event else None
        ),
        "hostdelay_occurrence_raw_predecessor_api": (
            previous_event.api if previous_event else None
        ),
        "hostdelay_occurrence_raw_predecessor_op_type": (
            previous_event.op_type if previous_event else None
        ),
        "hostdelay_occurrence_raw_predecessor_ts_us": (
            int(previous_event.ts) if previous_event else None
        ),
        "hostdelay_occurrence_raw_predecessor_ordinal": previous_ordinal,
        "hostdelay_occurrence_raw_successor_event_id": _event_id(current_event),
        "hostdelay_occurrence_raw_successor_api": current_event.api,
        "hostdelay_occurrence_raw_successor_op_type": current_event.op_type,
        "hostdelay_occurrence_raw_successor_ts_us": int(current_event.ts),
        "hostdelay_occurrence_raw_successor_ordinal": current_ordinal,
        "hostdelay_occurrence_semantic_predecessor_event_id": previous_materialized_event_id,
        "hostdelay_occurrence_semantic_predecessor_api": previous_materialized_api,
        "hostdelay_occurrence_semantic_predecessor_materialized": (
            previous_materialized_event_id is not None
        ),
        "hostdelay_occurrence_semantic_successor_event_id": current_materialized_event_id,
        "hostdelay_occurrence_semantic_successor_api": current_materialized_api,
        "hostdelay_occurrence_semantic_successor_materialized": (
            current_materialized_event_id is not None
        ),
        "hostdelay_occurrence_raw_boundary_family": raw_boundary_family,
        "hostdelay_occurrence_semantic_boundary_family": semantic_boundary_family,
        "hostdelay_occurrence_boundary_origin_kind": "unavailable",
        "hostdelay_occurrence_boundary_origin_evidence": None,
        "hostdelay_occurrence_boundary_visibility_kind": "unavailable",
        "hostdelay_occurrence_boundary_visibility_evidence": None,
        "hostdelay_occurrence_paper_visibility_class": (
            "paper_visible_by_default_or_unresolved"
        ),
        "hostdelay_occurrence_paper_visibility_reason": (
            "default_off_diagnostic_metadata_does_not_prove_paper_invisible_host_work"
        ),
        "hostdelay_occurrence_instrumentation_only_evidence": None,
        "hostdelay_occurrence_control_plane_only_evidence": None,
        "hostdelay_occurrence_already_counted_elsewhere_evidence": None,
        "hostdelay_occurrence_host_dispatch_interval_id": None,
        "hostdelay_occurrence_host_dispatch_overlap_status": "unavailable",
        "hostdelay_occurrence_provider_runtime_interval_id": None,
        "hostdelay_occurrence_provider_runtime_overlap_status": "unavailable",
        "hostdelay_occurrence_stream_queue_wait_interval_id": None,
        "hostdelay_occurrence_stream_queue_wait_overlap_status": "unavailable",
        "hostdelay_occurrence_count_once_group_id": (
            f"hostdelay_occurrence_count_once:{stable_boundary_key}"
        ),
        "hostdelay_occurrence_count_once_status": "unavailable",
        "hostdelay_occurrence_nonoverlap_status": "unavailable",
        "hostdelay_occurrence_cuda_event_record_id": None,
        "hostdelay_occurrence_cuda_event_wait_id": None,
        "hostdelay_occurrence_cuda_event_pair_id": None,
        "hostdelay_occurrence_cuda_event_wait_map_safety_status": "unavailable",
        "hostdelay_occurrence_collective_group_id": None,
        "hostdelay_occurrence_collective_member_id": None,
        "hostdelay_occurrence_collective_wait_edge_id": None,
        "hostdelay_occurrence_collective_wait_map_safety_status": "unavailable",
        "hostdelay_occurrence_stream_fifo_safety_status": "unavailable",
        "hostdelay_occurrence_replay_ordering_safety_status": "unavailable",
        "hostdelay_occurrence_fresh16_fresh8_comparable_join_key": comparable_key,
        "hostdelay_occurrence_fresh16_fresh8_join_key_basis": (
            "raw_and_semantic_boundary_family_plus_rank_local_raw_ordinals"
        ),
        "hostdelay_occurrence_fresh16_fresh8_join_key_status": (
            "diagnostic_only_not_validated_as_strict_counterpart"
        ),
        "hostdelay_occurrence_rank_workload_special_case_used": False,
        "hostdelay_occurrence_repair_ready": False,
        "hostdelay_occurrence_repair_ready_reason": (
            "requires_paper_visibility_count_once_nonoverlap_and_wait_map_review"
        ),
        "hostdelay_occurrence_safe_to_use_as_repair_evidence": False,
        "hostdelay_occurrence_safe_to_use_as_subtraction_delta": False,
        "hostdelay_occurrence_safe_delta_us": None,
        "hostdelay_occurrence_safe_delta_basis": None,
        "hostdelay_occurrence_observed_gap_us_context_only": (
            int(observed_gap_us) if observed_gap_us is not None else None
        ),
    }
    if join_metadata_enabled:
        metadata.update(
            {
                "hostdelay_occurrence_join_metadata_schema_version": (
                    _HOSTDELAY_OCCURRENCE_JOIN_METADATA_SCHEMA_VERSION
                ),
                "hostdelay_occurrence_join_metadata_opt_in_flag": True,
                "hostdelay_occurrence_join_metadata_env_flags": list(
                    _HOSTDELAY_OCCURRENCE_JOIN_METADATA_EXPORT_ENV_KEYS
                ),
                "hostdelay_occurrence_join_metadata_scope": (
                    "boundary_origin_visibility_count_once_nonoverlap_waitmap"
                ),
                "hostdelay_occurrence_join_metadata_behavior_effect": (
                    "diagnostic_only_no_duration_materialization_or_replay_change"
                ),
                "hostdelay_occurrence_boundary_origin_join_key": stable_boundary_key,
                "hostdelay_occurrence_boundary_origin_join_key_basis": (
                    "stable_hostdelay_occurrence_id_plus_raw_and_semantic_boundary_family"
                ),
                "hostdelay_occurrence_boundary_origin_kind": "unresolved",
                "hostdelay_occurrence_boundary_origin_status": "unresolved",
                "hostdelay_occurrence_boundary_origin_rule_id": None,
                "hostdelay_occurrence_boundary_origin_evidence": {
                    "raw_boundary_family": raw_boundary_family,
                    "semantic_boundary_family": semantic_boundary_family,
                    "raw_predecessor_api": previous_event.api if previous_event else None,
                    "raw_successor_api": current_event.api,
                    "semantic_predecessor_api": previous_materialized_api,
                    "semantic_successor_api": current_materialized_api,
                    "evidence_status": "classification_not_proven",
                },
                "hostdelay_occurrence_boundary_visibility_join_key": stable_boundary_key,
                "hostdelay_occurrence_boundary_visibility_join_key_basis": (
                    "stable_hostdelay_occurrence_id_plus_materialized_predecessor_successor_ids"
                ),
                "hostdelay_occurrence_boundary_visibility_kind": "unresolved",
                "hostdelay_occurrence_boundary_visibility_status": "unresolved",
                "hostdelay_occurrence_boundary_visibility_rule_id": None,
                "hostdelay_occurrence_boundary_visibility_evidence": {
                    "paper_visibility_class": "paper_visible_by_default_or_unresolved",
                    "materialized_predecessor_event_id": previous_materialized_event_id,
                    "materialized_successor_event_id": current_materialized_event_id,
                    "evidence_status": "paper_invisible_or_internal_boundary_not_proven",
                },
                "hostdelay_occurrence_paper_visibility_class": (
                    "paper_visible_by_default_or_unresolved"
                ),
                "hostdelay_occurrence_count_once_join_key": (
                    f"hostdelay_occurrence_count_once:{stable_boundary_key}"
                ),
                "hostdelay_occurrence_count_once_join_key_basis": (
                    "stable_hostdelay_occurrence_id"
                ),
                "hostdelay_occurrence_count_once_evidence": None,
                "hostdelay_occurrence_nonoverlap_join_key": stable_boundary_key,
                "hostdelay_occurrence_nonoverlap_join_key_basis": (
                    "stable_hostdelay_occurrence_id_plus_predicted_interval"
                ),
                "hostdelay_occurrence_nonoverlap_evidence": None,
                "hostdelay_occurrence_host_dispatch_interval_join_status": "unavailable",
                "hostdelay_occurrence_provider_runtime_interval_join_status": "unavailable",
                "hostdelay_occurrence_stream_queue_wait_interval_join_status": "unavailable",
                "hostdelay_occurrence_cuda_event_waitmap_join_key": stable_boundary_key,
                "hostdelay_occurrence_cuda_event_waitmap_join_key_basis": (
                    "stable_hostdelay_occurrence_id_plus_event_api_adjacency"
                ),
                "hostdelay_occurrence_cuda_event_wait_map_edge_id": None,
                "hostdelay_occurrence_cuda_event_wait_map_evidence": None,
                "hostdelay_occurrence_collective_waitmap_join_key": stable_boundary_key,
                "hostdelay_occurrence_collective_waitmap_join_key_basis": (
                    "stable_hostdelay_occurrence_id_plus_collective_api_adjacency"
                ),
                "hostdelay_occurrence_collective_provider_interval_id": None,
                "hostdelay_occurrence_collective_wait_map_evidence": None,
                "hostdelay_occurrence_join_repair_ready": False,
                "hostdelay_occurrence_join_safe_to_use_as_repair_evidence": False,
                "hostdelay_occurrence_join_safe_to_use_as_subtraction_delta": False,
                "hostdelay_occurrence_join_safe_delta_us": None,
                "hostdelay_occurrence_join_safe_delta_basis": None,
                "hostdelay_occurrence_join_repair_ready_reason": (
                    "join metadata is diagnostic only until boundary origin, visibility, "
                    "count-once, nonoverlap, and wait-map gates are proven"
                ),
            }
        )
    if _collective_event_polling_boundary_metadata_export_enabled():
        metadata.update(
            _collective_event_polling_boundary_metadata(
                previous_event,
                current_event,
                stable_boundary_key=stable_boundary_key,
                raw_boundary_family=raw_boundary_family,
                semantic_boundary_family=semantic_boundary_family,
                host_dispatch_queue_id=host_dispatch_queue_id,
                previous_materialized_event_id=previous_materialized_event_id,
                previous_materialized_api=previous_materialized_api,
                current_materialized_event_id=current_materialized_event_id,
                current_materialized_api=current_materialized_api,
            )
        )
    if _event_polling_boundary_metadata_export_enabled():
        metadata.update(
            _event_polling_boundary_metadata(
                previous_event,
                current_event,
                stable_boundary_key=stable_boundary_key,
                raw_boundary_family=raw_boundary_family,
                semantic_boundary_family=semantic_boundary_family,
                host_dispatch_queue_id=host_dispatch_queue_id,
                observed_gap_us=observed_gap_us,
                previous_materialized_event_id=previous_materialized_event_id,
                previous_materialized_api=previous_materialized_api,
                current_materialized_event_id=current_materialized_event_id,
                current_materialized_api=current_materialized_api,
            )
        )
    if _boundary_origin_subregion_proof_metadata_export_enabled():
        metadata.update(
            _boundary_origin_subregion_proof_metadata(
                previous_event,
                current_event,
                stable_boundary_key=stable_boundary_key,
                raw_boundary_family=raw_boundary_family,
                semantic_boundary_family=semantic_boundary_family,
                host_dispatch_queue_id=host_dispatch_queue_id,
                observed_gap_us=observed_gap_us,
                previous_materialized_event_id=previous_materialized_event_id,
                previous_materialized_api=previous_materialized_api,
                current_materialized_event_id=current_materialized_event_id,
                current_materialized_api=current_materialized_api,
            )
        )
    if _strict_subregion_extent_source_metadata_export_enabled():
        metadata.update(
            _strict_subregion_extent_source_metadata(
                previous_event,
                current_event,
                stable_boundary_key=stable_boundary_key,
                raw_boundary_family=raw_boundary_family,
                semantic_boundary_family=semantic_boundary_family,
                host_dispatch_queue_id=host_dispatch_queue_id,
                observed_gap_us=observed_gap_us,
                previous_materialized_event_id=previous_materialized_event_id,
                previous_materialized_api=previous_materialized_api,
                current_materialized_event_id=current_materialized_event_id,
                current_materialized_api=current_materialized_api,
            )
        )
    return metadata


def _first_nonempty_extra(
    *events: TraceEvent | None,
    keys: tuple[str, ...],
) -> object | None:
    for event in events:
        if event is None:
            continue
        for key in keys:
            value = event.extras.get(key)
            if value not in (None, ""):
                return value
    return None


def _collective_event_polling_target_family(apis: set[str]) -> str:
    collective_apis = {api for api in apis if api.startswith("nccl")}
    event_apis = apis & {"cudaEventRecord", "cudaEventQuery", "cudaStreamWaitEvent"}
    if "ncclAllReduce" in collective_apis:
        return "ncclAllReduce"
    if "ncclSend" in collective_apis:
        return "ncclSend"
    if "ncclRecv" in collective_apis:
        return "ncclRecv"
    if event_apis and collective_apis:
        return "mixed_collective_event"
    if event_apis:
        return "event_polling"
    if collective_apis:
        return "collective_status_polling"
    return "unavailable"


def _collective_event_polling_boundary_metadata(
    previous_event: TraceEvent | None,
    current_event: TraceEvent,
    *,
    stable_boundary_key: str,
    raw_boundary_family: str,
    semantic_boundary_family: str,
    host_dispatch_queue_id: str | None,
    previous_materialized_event_id: str | None,
    previous_materialized_api: str | None,
    current_materialized_event_id: str | None,
    current_materialized_api: str | None,
) -> dict[str, object]:
    raw_apis = {
        api
        for api in (
            previous_event.api if previous_event else None,
            current_event.api,
        )
        if api
    }
    semantic_apis = {
        api
        for api in (previous_materialized_api, current_materialized_api)
        if api
    }
    all_apis = raw_apis | semantic_apis
    if not (all_apis & _COLLECTIVE_EVENT_POLLING_BOUNDARY_APIS):
        return {}
    stream_id = _first_nonempty_extra(
        current_event,
        previous_event,
        keys=("stream_id", "stream", "cuda_stream_id"),
    )
    event_handle = _first_nonempty_extra(
        current_event,
        previous_event,
        keys=("event_id", "event", "event_handle", "cuda_event_handle"),
    )
    event_version = _first_nonempty_extra(
        current_event,
        previous_event,
        keys=("event_version", "cuda_event_version"),
    )
    communicator_id = _first_nonempty_extra(
        current_event,
        previous_event,
        keys=("comm_id", "comm_hash", "communicator_id", "communicator_hash"),
    )
    collective_group_id = _first_nonempty_extra(
        current_event,
        previous_event,
        keys=("collective_group_id", "group_id", "collective_id"),
    )
    collective_api = next((api for api in semantic_apis if api.startswith("nccl")), None)
    if collective_api is None:
        collective_api = next((api for api in raw_apis if api.startswith("nccl")), None)
    return {
        "collective_event_polling_metadata_schema_version": (
            _COLLECTIVE_EVENT_POLLING_BOUNDARY_METADATA_SCHEMA_VERSION
        ),
        "collective_event_polling_metadata_opt_in_flag": True,
        "collective_event_polling_metadata_env_flags": list(
            _COLLECTIVE_EVENT_POLLING_BOUNDARY_METADATA_EXPORT_ENV_KEYS
        ),
        "collective_event_polling_metadata_behavior_effect": (
            "diagnostic_only_no_duration_materialization_or_replay_change"
        ),
        "collective_event_polling_occurrence_join_key": stable_boundary_key,
        "collective_event_polling_occurrence_join_key_basis": (
            "stable_hostdelay_occurrence_id"
        ),
        "collective_event_polling_raw_boundary_family": raw_boundary_family,
        "collective_event_polling_semantic_boundary_family": semantic_boundary_family,
        "collective_event_polling_raw_predecessor_event_id": (
            _event_id(previous_event) if previous_event else None
        ),
        "collective_event_polling_raw_successor_event_id": _event_id(current_event),
        "collective_event_polling_semantic_predecessor_event_id": (
            previous_materialized_event_id
        ),
        "collective_event_polling_semantic_successor_event_id": current_materialized_event_id,
        "collective_event_polling_target_family": (
            _collective_event_polling_target_family(all_apis)
        ),
        "collective_event_polling_rank": int(current_event.rank),
        "collective_event_polling_pid": int(current_event.pid),
        "collective_event_polling_tid": int(current_event.tid),
        "collective_event_polling_dispatch_key": host_dispatch_queue_id,
        "collective_event_polling_stream_id": stream_id,
        "collective_event_polling_stream_resource_id": (
            f"rank:{int(current_event.rank)}:stream:{stream_id}"
            if stream_id not in (None, "")
            else None
        ),
        "collective_event_polling_stream_namespace_basis": (
            "predicted_collate_raw_stream_id_rank_local"
            if stream_id not in (None, "")
            else "unavailable"
        ),
        "collective_event_polling_collective_group_id": collective_group_id,
        "collective_event_polling_collective_member_id": (
            previous_materialized_event_id
            if previous_materialized_api and previous_materialized_api.startswith("nccl")
            else current_materialized_event_id
            if current_materialized_api and current_materialized_api.startswith("nccl")
            else None
        ),
        "collective_event_polling_collective_api": collective_api,
        "collective_event_polling_collective_call_order": _first_nonempty_extra(
            current_event,
            previous_event,
            keys=("call_idx", "call_index", "collective_call_order", "sequence_number"),
        ),
        "collective_event_polling_communicator_id": communicator_id,
        "collective_event_polling_communicator_size": _first_nonempty_extra(
            current_event,
            previous_event,
            keys=("nranks", "communicator_size", "participant_count"),
        ),
        "collective_event_polling_participant_count": _first_nonempty_extra(
            current_event,
            previous_event,
            keys=("participant_count", "nranks", "communicator_size"),
        ),
        "collective_event_polling_provider_interval_id": None,
        "collective_event_polling_cuda_event_handle": event_handle,
        "collective_event_polling_cuda_event_version": event_version,
        "collective_event_polling_cuda_event_record_id": (
            current_materialized_event_id
            if current_materialized_api == "cudaEventRecord"
            else previous_materialized_event_id
            if previous_materialized_api == "cudaEventRecord"
            else None
        ),
        "collective_event_polling_cuda_event_query_id": (
            current_materialized_event_id
            if current_materialized_api == "cudaEventQuery"
            else previous_materialized_event_id
            if previous_materialized_api == "cudaEventQuery"
            else None
        ),
        "collective_event_polling_cuda_stream_wait_event_id": (
            current_materialized_event_id
            if current_materialized_api == "cudaStreamWaitEvent"
            else previous_materialized_event_id
            if previous_materialized_api == "cudaStreamWaitEvent"
            else None
        ),
        "collective_event_polling_cuda_event_pair_id": None,
        "collective_event_polling_boundary_origin_kind": "unresolved",
        "collective_event_polling_boundary_origin_status": "unresolved",
        "collective_event_polling_boundary_origin_evidence": {
            "raw_apis": sorted(raw_apis),
            "semantic_apis": sorted(semantic_apis),
            "evidence_status": "classification_not_proven",
        },
        "collective_event_polling_boundary_visibility_kind": "unresolved",
        "collective_event_polling_boundary_visibility_status": "unresolved",
        "collective_event_polling_boundary_visibility_evidence": None,
        "collective_event_polling_wait_map_edge_id": None,
        "collective_event_polling_wait_map_edge_kind": "unavailable",
        "collective_event_polling_release_source_id": None,
        "collective_event_polling_release_source_kind": "unavailable",
        "collective_event_polling_release_timing_basis": "unavailable",
        "collective_event_polling_wait_map_release_status": "unavailable",
        "collective_event_polling_count_once_group_id": (
            f"collective_event_polling_count_once:{stable_boundary_key}"
        ),
        "collective_event_polling_count_once_status": "unavailable",
        "collective_event_polling_nonoverlap_status": "unavailable",
        "collective_event_polling_nonoverlap_evidence": None,
        "collective_event_polling_repair_ready": False,
        "collective_event_polling_safe_to_use_as_repair_evidence": False,
        "collective_event_polling_safe_to_use_as_subtraction_delta": False,
        "collective_event_polling_safe_delta_us": None,
        "collective_event_polling_repair_ready_reason": (
            "requires proven origin, visibility, count-once, non-overlap, and wait-map release"
        ),
    }


def _event_polling_boundary_target_class(apis: set[str]) -> str:
    if "cudaEventQuery" in apis:
        return "nonblocking_cudaEventQuery_polling_pressure"
    if "cudaEventRecord" in apis or "cudaEventRecordWithFlags" in apis:
        if "cudaLaunchKernel" in apis:
            return "event_record_launch_boundary"
        return "event_record_boundary"
    if "cudaStreamWaitEvent" in apis:
        return "cudaStreamWaitEvent_context"
    return "event_polling_related_boundary"


def _event_polling_boundary_classification(
    *,
    raw_apis: set[str],
    semantic_apis: set[str],
) -> dict[str, object]:
    all_apis = raw_apis | semantic_apis
    mixed_boundary_apis = {
        "cudaEventRecord",
        "cudaEventRecordWithFlags",
        "cudaLaunchKernel",
        "cublasSetStream_v2",
    }
    if all_apis & mixed_boundary_apis:
        return {
            "event_polling_boundary_polling_class": "event_polling_boundary_mixed",
            "event_polling_boundary_origin_kind": (
                "mixed_host_runtime_control_and_visible_work_unresolved"
            ),
            "event_polling_boundary_origin_status": "unresolved",
            "event_polling_boundary_origin_rule_id": "event_launch_library_mixed_gap_v1",
            "event_polling_boundary_visibility_kind": "unresolved_mixed",
            "event_polling_boundary_visibility_status": "unresolved",
            "event_polling_boundary_visibility_rule_id": "event_launch_visibility_unresolved_v1",
            "event_polling_boundary_paper_visibility_class": "unresolved_mixed",
            "event_polling_boundary_paper_visibility_reason": (
                "event/launch/library boundary may include paper-visible host work "
                "and internal control-plane work; strict origin and visibility proof required"
            ),
            "event_polling_boundary_candidate_control_plane_subregion_status": (
                "candidate_needs_strict_boundary_origin_proof"
            ),
            "event_polling_boundary_candidate_instrumentation_only_status": (
                "candidate_needs_trace_processing_visibility_proof"
            ),
            "event_polling_boundary_already_modeled_replay_waitmap_status": (
                "unavailable_or_not_applicable"
            ),
        }
    query_current_or_successor = "cudaEventQuery" in all_apis
    if query_current_or_successor:
        return {
            "event_polling_boundary_polling_class": "nonblocking_cudaEventQuery_polling",
            "event_polling_boundary_origin_kind": "host_runtime_polling_or_application_polling",
            "event_polling_boundary_origin_status": "classified_paper_visible_by_default",
            "event_polling_boundary_origin_rule_id": "cudaEventQuery_nonblocking_polling_v1",
            "event_polling_boundary_visibility_kind": "paper_visible_host_work",
            "event_polling_boundary_visibility_status": "paper_visible_by_default",
            "event_polling_boundary_visibility_rule_id": "cudaEventQuery_polling_visible_v1",
            "event_polling_boundary_paper_visibility_class": "paper_visible_by_default",
            "event_polling_boundary_paper_visibility_reason": (
                "cudaEventQuery is nonblocking polling/control-plane host work; "
                "duration is not a replay wait-map release or safe delta"
            ),
            "event_polling_boundary_candidate_control_plane_subregion_status": (
                "not_proven"
            ),
            "event_polling_boundary_candidate_instrumentation_only_status": (
                "not_proven"
            ),
            "event_polling_boundary_already_modeled_replay_waitmap_status": (
                "not_waitmap_release"
            ),
        }
    return {
        "event_polling_boundary_polling_class": "event_polling_boundary_context",
        "event_polling_boundary_origin_kind": "unresolved",
        "event_polling_boundary_origin_status": "unresolved",
        "event_polling_boundary_origin_rule_id": None,
        "event_polling_boundary_visibility_kind": "unresolved",
        "event_polling_boundary_visibility_status": "unresolved",
        "event_polling_boundary_visibility_rule_id": None,
        "event_polling_boundary_paper_visibility_class": "unresolved_mixed",
        "event_polling_boundary_paper_visibility_reason": (
            "target API family present but paper visibility not proven"
        ),
        "event_polling_boundary_candidate_control_plane_subregion_status": "not_proven",
        "event_polling_boundary_candidate_instrumentation_only_status": "not_proven",
        "event_polling_boundary_already_modeled_replay_waitmap_status": "unavailable",
    }


def _event_polling_boundary_metadata(
    previous_event: TraceEvent | None,
    current_event: TraceEvent,
    *,
    stable_boundary_key: str,
    raw_boundary_family: str,
    semantic_boundary_family: str,
    host_dispatch_queue_id: str | None,
    observed_gap_us: int | None,
    previous_materialized_event_id: str | None,
    previous_materialized_api: str | None,
    current_materialized_event_id: str | None,
    current_materialized_api: str | None,
) -> dict[str, object]:
    raw_apis = {
        api
        for api in (
            previous_event.api if previous_event else None,
            current_event.api,
        )
        if api
    }
    semantic_apis = {
        api
        for api in (previous_materialized_api, current_materialized_api)
        if api
    }
    all_apis = raw_apis | semantic_apis
    if not (all_apis & _EVENT_POLLING_BOUNDARY_APIS):
        return {}
    if not (all_apis & _EVENT_POLLING_BOUNDARY_CORE_APIS):
        return {}
    classification = _event_polling_boundary_classification(
        raw_apis=raw_apis,
        semantic_apis=semantic_apis,
    )
    count_once_group_id = f"event_polling_boundary_count_once:{stable_boundary_key}"
    return {
        "event_polling_boundary_metadata_schema_version": (
            _EVENT_POLLING_BOUNDARY_METADATA_SCHEMA_VERSION
        ),
        "event_polling_boundary_metadata_opt_in_flag": True,
        "event_polling_boundary_metadata_env_flags": list(
            _EVENT_POLLING_BOUNDARY_METADATA_EXPORT_ENV_KEYS
        ),
        "event_polling_boundary_behavior_effect": (
            "diagnostic_only_no_duration_materialization_or_replay_change"
        ),
        "event_polling_boundary_occurrence_id": stable_boundary_key,
        "event_polling_boundary_occurrence_id_basis": "stable_hostdelay_occurrence_id",
        "event_polling_boundary_materialized_hostdelay_event_id": None,
        "event_polling_boundary_rank": int(current_event.rank),
        "event_polling_boundary_pid": int(current_event.pid),
        "event_polling_boundary_tid": int(current_event.tid),
        "event_polling_boundary_thread_id": int(current_event.tid),
        "event_polling_boundary_dispatch_key": host_dispatch_queue_id,
        "event_polling_boundary_interval_start_ts_us": (
            int(previous_event.ts) if previous_event else None
        ),
        "event_polling_boundary_interval_end_ts_us": int(current_event.ts),
        "event_polling_boundary_duration_us_context_only": (
            int(observed_gap_us) if observed_gap_us is not None else None
        ),
        "event_polling_boundary_timestamp_basis": "raw_trace_adjacent_api_timestamps",
        "event_polling_boundary_timestamp_source": "collate_hostdelay_boundary",
        "event_polling_boundary_clock_domain": "rank_local_trace_clock",
        "event_polling_boundary_interval_basis_status": "diagnostic_context_only",
        "event_polling_boundary_raw_predecessor_event_id": (
            _event_id(previous_event) if previous_event else None
        ),
        "event_polling_boundary_raw_predecessor_api": (
            previous_event.api if previous_event else None
        ),
        "event_polling_boundary_raw_predecessor_ordinal": (
            int(previous_event.ordinal) if previous_event else None
        ),
        "event_polling_boundary_raw_successor_event_id": _event_id(current_event),
        "event_polling_boundary_raw_successor_api": current_event.api,
        "event_polling_boundary_raw_successor_ordinal": int(current_event.ordinal),
        "event_polling_boundary_semantic_predecessor_event_id": (
            previous_materialized_event_id
        ),
        "event_polling_boundary_semantic_predecessor_api": previous_materialized_api,
        "event_polling_boundary_semantic_predecessor_materialized": (
            previous_materialized_event_id is not None
        ),
        "event_polling_boundary_semantic_successor_event_id": (
            current_materialized_event_id
        ),
        "event_polling_boundary_semantic_successor_api": current_materialized_api,
        "event_polling_boundary_semantic_successor_materialized": (
            current_materialized_event_id is not None
        ),
        "event_polling_boundary_raw_family": raw_boundary_family,
        "event_polling_boundary_semantic_family": semantic_boundary_family,
        "event_polling_boundary_raw_semantic_pair": (
            f"{raw_boundary_family} || {semantic_boundary_family}"
        ),
        "event_polling_boundary_target_class": _event_polling_boundary_target_class(
            all_apis
        ),
        **classification,
        "event_polling_boundary_origin_evidence": {
            "raw_apis": sorted(raw_apis),
            "semantic_apis": sorted(semantic_apis),
            "evidence_status": "classification_rule_only_not_repair_proof",
        },
        "event_polling_boundary_visibility_evidence": {
            "semantic_predecessor_materialized": previous_materialized_event_id is not None,
            "semantic_successor_materialized": current_materialized_event_id is not None,
            "evidence_status": "visibility_classification_not_safe_delta",
        },
        "event_polling_boundary_already_counted_elsewhere_status": "unavailable",
        "event_polling_boundary_unsafe_removable_status": "unsafe_removable_false",
        "event_polling_boundary_hostdelay_occurrence_id": stable_boundary_key,
        "event_polling_boundary_replay_waitmap_edge_id": None,
        "event_polling_boundary_replay_waitmap_edge_kind": None,
        "event_polling_boundary_collective_group_id": _first_nonempty_extra(
            current_event,
            previous_event,
            keys=("collective_group_id", "group_id", "collective_id"),
        ),
        "event_polling_boundary_cuda_event_id": _first_nonempty_extra(
            current_event,
            previous_event,
            keys=("event_id", "event", "event_handle", "cuda_event_handle"),
        ),
        "event_polling_boundary_cuda_event_version": _first_nonempty_extra(
            current_event,
            previous_event,
            keys=("event_version", "cuda_event_version"),
        ),
        "event_polling_boundary_count_once_group_id": count_once_group_id,
        "event_polling_boundary_count_once_status": "unavailable",
        "event_polling_boundary_nonoverlap_status": "unavailable",
        "event_polling_boundary_wait_map_safety_status": "unavailable",
        "event_polling_boundary_fresh16_fresh8_comparable_join_key": (
            f"event_polling_boundary:{raw_boundary_family}|{semantic_boundary_family}"
        ),
        "event_polling_boundary_fresh16_fresh8_join_key_status": (
            "diagnostic_context_only_not_strict_counterpart"
        ),
        "event_polling_boundary_repair_ready": False,
        "event_polling_boundary_repair_ready_reason": (
            "requires boundary-origin, paper-visibility, count-once, nonoverlap, "
            "wait-map safety, and fresh8 preservation proof"
        ),
        "event_polling_boundary_safe_to_use_as_repair_evidence": False,
        "event_polling_boundary_safe_to_use_as_subtraction_delta": False,
        "event_polling_boundary_safe_delta_us": None,
        "event_polling_boundary_safe_delta_basis": None,
        "event_polling_boundary_runtime_or_endpoint_substitution_used": False,
        "event_polling_boundary_hostdelay_shortening_used": False,
        "event_polling_boundary_rank_workload_special_case_used": False,
    }


def _boundary_origin_subregion_classification(
    *,
    raw_apis: set[str],
    semantic_apis: set[str],
) -> dict[str, object]:
    all_apis = raw_apis | semantic_apis
    gemm_to_query_polling = (
        "cudaEventQuery" in all_apis
        and bool(all_apis & _BOUNDARY_ORIGIN_SUBREGION_PROOF_GEMM_APIS)
        and len(all_apis - (_BOUNDARY_ORIGIN_SUBREGION_PROOF_GEMM_APIS | {"cudaEventQuery"}))
        == 0
    )
    if all_apis == {"cudaEventQuery"} or gemm_to_query_polling:
        return {
            "boundary_origin_subregion_kind": "not_applicable_pure_polling",
            "boundary_origin_subregion_candidate_role": (
                "paper_visible_polling_not_targeted_for_removal"
            ),
            "boundary_origin_subregion_origin_kind": (
                "host_runtime_polling_or_application_polling"
            ),
            "boundary_origin_subregion_origin_status": (
                "classified_paper_visible_by_default"
            ),
            "boundary_origin_subregion_origin_rule_id": (
                "cudaEventQuery_nonblocking_polling_not_subregion_v1"
            ),
            "boundary_origin_subregion_visibility_kind": "paper_visible_host_work",
            "boundary_origin_subregion_visibility_status": "paper_visible_by_default",
            "boundary_origin_subregion_visibility_rule_id": (
                "cudaEventQuery_polling_visible_not_subregion_v1"
            ),
            "boundary_origin_subregion_paper_visibility_class": (
                "paper_visible_by_default"
            ),
            "boundary_origin_subregion_paper_visibility_reason": (
                "cudaEventQuery polling, including GEMM-family -> cudaEventQuery "
                "polling boundaries, remains paper-visible host work and is not a "
                "removable internal subregion"
            ),
            "boundary_origin_subregion_strict_proof_status": "not_applicable",
            "boundary_origin_subregion_strict_proof_source": None,
        }
    if all_apis & _BOUNDARY_ORIGIN_SUBREGION_PROOF_EVENT_APIS:
        return {
            "boundary_origin_subregion_kind": (
                "candidate_internal_event_launch_library_control_gap"
            ),
            "boundary_origin_subregion_candidate_role": (
                "candidate_internal_subregion_needs_strict_proof"
            ),
            "boundary_origin_subregion_origin_kind": "unresolved_mixed",
            "boundary_origin_subregion_origin_status": (
                "candidate_needs_strict_boundary_origin_proof"
            ),
            "boundary_origin_subregion_origin_rule_id": (
                "event_launch_library_control_subregion_candidate_v1"
            ),
            "boundary_origin_subregion_visibility_kind": "unresolved_mixed",
            "boundary_origin_subregion_visibility_status": (
                "candidate_needs_strict_visibility_proof"
            ),
            "boundary_origin_subregion_visibility_rule_id": (
                "event_launch_library_control_visibility_unresolved_v1"
            ),
            "boundary_origin_subregion_paper_visibility_class": "unresolved_mixed",
            "boundary_origin_subregion_paper_visibility_reason": (
                "event/launch/library/control boundary may contain paper-visible host "
                "work and internal subregions; strict origin, visibility, count-once, "
                "nonoverlap, wait-map, and fresh8 proof required"
            ),
            "boundary_origin_subregion_strict_proof_status": "unavailable_or_unproven",
            "boundary_origin_subregion_strict_proof_source": None,
        }
    return {
        "boundary_origin_subregion_kind": "not_in_subregion_proof_scope",
        "boundary_origin_subregion_candidate_role": "not_targeted",
        "boundary_origin_subregion_origin_kind": "unavailable",
        "boundary_origin_subregion_origin_status": "not_in_scope",
        "boundary_origin_subregion_origin_rule_id": None,
        "boundary_origin_subregion_visibility_kind": "unavailable",
        "boundary_origin_subregion_visibility_status": "not_in_scope",
        "boundary_origin_subregion_visibility_rule_id": None,
        "boundary_origin_subregion_paper_visibility_class": "not_in_scope",
        "boundary_origin_subregion_paper_visibility_reason": (
            "boundary lacks event/launch/library/control subregion proof target APIs"
        ),
        "boundary_origin_subregion_strict_proof_status": "not_applicable",
        "boundary_origin_subregion_strict_proof_source": None,
    }


def _boundary_origin_subregion_proof_metadata(
    previous_event: TraceEvent | None,
    current_event: TraceEvent,
    *,
    stable_boundary_key: str,
    raw_boundary_family: str,
    semantic_boundary_family: str,
    host_dispatch_queue_id: str | None,
    observed_gap_us: int | None,
    previous_materialized_event_id: str | None,
    previous_materialized_api: str | None,
    current_materialized_event_id: str | None,
    current_materialized_api: str | None,
) -> dict[str, object]:
    raw_apis = {
        api
        for api in (
            previous_event.api if previous_event else None,
            current_event.api,
        )
        if api
    }
    semantic_apis = {
        api
        for api in (previous_materialized_api, current_materialized_api)
        if api
    }
    all_apis = raw_apis | semantic_apis
    if not (all_apis & _BOUNDARY_ORIGIN_SUBREGION_PROOF_TARGET_APIS):
        return {}
    if not (all_apis & _BOUNDARY_ORIGIN_SUBREGION_PROOF_EVENT_APIS):
        return {}
    classification = _boundary_origin_subregion_classification(
        raw_apis=raw_apis,
        semantic_apis=semantic_apis,
    )
    count_once_group_id = f"boundary_origin_subregion_count_once:{stable_boundary_key}"
    raw_semantic_pair = f"{raw_boundary_family} || {semantic_boundary_family}"
    return {
        "boundary_origin_subregion_metadata_schema_version": (
            _BOUNDARY_ORIGIN_SUBREGION_PROOF_METADATA_SCHEMA_VERSION
        ),
        "boundary_origin_subregion_metadata_opt_in_flag": True,
        "boundary_origin_subregion_metadata_env_flags": list(
            _BOUNDARY_ORIGIN_SUBREGION_PROOF_METADATA_EXPORT_ENV_KEYS
        ),
        "boundary_origin_subregion_behavior_effect": (
            "diagnostic_only_no_duration_materialization_or_replay_change"
        ),
        "boundary_origin_subregion_occurrence_id": stable_boundary_key,
        "boundary_origin_subregion_occurrence_id_basis": (
            "stable_hostdelay_occurrence_id"
        ),
        "boundary_origin_subregion_rank": int(current_event.rank),
        "boundary_origin_subregion_pid": int(current_event.pid),
        "boundary_origin_subregion_tid": int(current_event.tid),
        "boundary_origin_subregion_dispatch_key": host_dispatch_queue_id,
        "boundary_origin_subregion_materialized_hostdelay_event_id": None,
        "boundary_origin_subregion_id": (
            f"boundary_origin_subregion:{stable_boundary_key}"
        ),
        "boundary_origin_subregion_id_basis": (
            "stable_hostdelay_occurrence_id_context_only"
        ),
        **classification,
        "boundary_origin_subregion_start_ts_us": None,
        "boundary_origin_subregion_end_ts_us": None,
        "boundary_origin_subregion_duration_us_context_only": (
            int(observed_gap_us) if observed_gap_us is not None else None
        ),
        "boundary_origin_subregion_timestamp_basis": (
            "raw_trace_adjacent_api_timestamps_context_only"
        ),
        "boundary_origin_subregion_clock_domain": "rank_local_trace_clock",
        "boundary_origin_subregion_strict_extent_status": "unavailable_or_unproven",
        "boundary_origin_subregion_strict_extent_source": None,
        "boundary_origin_subregion_strict_extent_evidence": {
            "raw_apis": sorted(raw_apis),
            "semantic_apis": sorted(semantic_apis),
            "evidence_status": "context_only_not_strict_subregion_extent",
        },
        "boundary_origin_subregion_raw_predecessor_event_id": (
            _event_id(previous_event) if previous_event else None
        ),
        "boundary_origin_subregion_raw_predecessor_api": (
            previous_event.api if previous_event else None
        ),
        "boundary_origin_subregion_raw_predecessor_role": "predecessor_api",
        "boundary_origin_subregion_raw_successor_event_id": _event_id(current_event),
        "boundary_origin_subregion_raw_successor_api": current_event.api,
        "boundary_origin_subregion_raw_successor_role": "successor_api",
        "boundary_origin_subregion_semantic_predecessor_event_id": (
            previous_materialized_event_id
        ),
        "boundary_origin_subregion_semantic_predecessor_api": (
            previous_materialized_api
        ),
        "boundary_origin_subregion_semantic_predecessor_materialized": (
            previous_materialized_event_id is not None
        ),
        "boundary_origin_subregion_semantic_successor_event_id": (
            current_materialized_event_id
        ),
        "boundary_origin_subregion_semantic_successor_api": current_materialized_api,
        "boundary_origin_subregion_semantic_successor_materialized": (
            current_materialized_event_id is not None
        ),
        "boundary_origin_subregion_raw_family": raw_boundary_family,
        "boundary_origin_subregion_semantic_family": semantic_boundary_family,
        "boundary_origin_subregion_raw_semantic_pair": raw_semantic_pair,
        "boundary_origin_subregion_origin_evidence": {
            "raw_boundary_family": raw_boundary_family,
            "semantic_boundary_family": semantic_boundary_family,
            "raw_apis": sorted(raw_apis),
            "semantic_apis": sorted(semantic_apis),
            "evidence_status": "strict_boundary_origin_not_proven",
        },
        "boundary_origin_subregion_visibility_evidence": {
            "semantic_predecessor_materialized": previous_materialized_event_id
            is not None,
            "semantic_successor_materialized": current_materialized_event_id
            is not None,
            "evidence_status": "paper_invisible_subregion_not_proven",
        },
        "boundary_origin_subregion_strict_proof_evidence": None,
        "boundary_origin_subregion_count_once_group_id": count_once_group_id,
        "boundary_origin_subregion_count_once_status": "unavailable",
        "boundary_origin_subregion_count_once_evidence": None,
        "boundary_origin_subregion_nonoverlap_group_id": stable_boundary_key,
        "boundary_origin_subregion_nonoverlap_status": "unavailable",
        "boundary_origin_subregion_nonoverlap_evidence": None,
        "boundary_origin_subregion_wait_map_safety_status": "unavailable",
        "boundary_origin_subregion_wait_map_edge_id": None,
        "boundary_origin_subregion_replay_ordering_safety_status": "unavailable",
        "boundary_origin_subregion_stream_fifo_safety_status": "unavailable",
        "boundary_origin_subregion_collective_wait_safety_status": "unavailable",
        "boundary_origin_subregion_fresh16_fresh8_comparable_join_key": (
            f"boundary_origin_subregion:{raw_semantic_pair}"
        ),
        "boundary_origin_subregion_fresh16_fresh8_join_key_basis": (
            "raw_and_semantic_boundary_family_context_only"
        ),
        "boundary_origin_subregion_fresh16_fresh8_join_key_status": (
            "diagnostic_context_only_not_strict_counterpart"
        ),
        "boundary_origin_subregion_fresh8_preservation_risk_status": "unreviewed",
        "boundary_origin_subregion_fresh8_preservation_evidence": None,
        "boundary_origin_subregion_repair_ready": False,
        "boundary_origin_subregion_repair_ready_reason": (
            "requires strict origin, visibility, count-once, nonoverlap, wait-map "
            "safety, and fresh8 preservation proof"
        ),
        "boundary_origin_subregion_safe_to_use_as_repair_evidence": False,
        "boundary_origin_subregion_safe_to_use_as_subtraction_delta": False,
        "boundary_origin_subregion_safe_delta_us": None,
        "boundary_origin_subregion_safe_delta_basis": None,
        "boundary_origin_subregion_runtime_or_endpoint_substitution_used": False,
        "boundary_origin_subregion_hostdelay_shortening_used": False,
        "boundary_origin_subregion_rank_workload_special_case_used": False,
    }


def _strict_subregion_extent_classification(
    *,
    raw_apis: set[str],
    semantic_apis: set[str],
) -> dict[str, object]:
    all_apis = raw_apis | semantic_apis
    gemm_to_query_polling = (
        "cudaEventQuery" in all_apis
        and bool(all_apis & _STRICT_SUBREGION_EXTENT_GEMM_APIS)
        and len(all_apis - (_STRICT_SUBREGION_EXTENT_GEMM_APIS | {"cudaEventQuery"}))
        == 0
    )
    if all_apis == {"cudaEventQuery"} or gemm_to_query_polling:
        return {
            "strict_subregion_extent_target_family_class": (
                "paper_visible_polling_not_targeted_for_removal"
            ),
            "strict_subregion_extent_candidate_subregion_kind": (
                "not_applicable_paper_visible_polling"
            ),
            "strict_subregion_extent_candidate_subregion_role": (
                "not_targeted_for_removal"
            ),
            "strict_subregion_extent_source_proof_status": "not_applicable",
            "strict_subregion_extent_source_proof_reason": (
                "cudaEventQuery polling, including GEMM-family -> cudaEventQuery, "
                "is paper-visible host work by default"
            ),
            "strict_subregion_extent_origin_kind": (
                "host_runtime_polling_or_application_polling"
            ),
            "strict_subregion_extent_origin_status": (
                "classified_paper_visible_by_default"
            ),
            "strict_subregion_extent_origin_rule_id": (
                "cudaEventQuery_nonblocking_polling_not_strict_subregion_v1"
            ),
            "strict_subregion_extent_visibility_kind": "paper_visible_host_work",
            "strict_subregion_extent_visibility_status": "paper_visible_by_default",
            "strict_subregion_extent_visibility_rule_id": (
                "cudaEventQuery_polling_visible_not_strict_subregion_v1"
            ),
            "strict_subregion_extent_paper_visibility_class": (
                "paper_visible_by_default"
            ),
            "strict_subregion_extent_paper_visibility_reason": (
                "nonblocking cudaEventQuery polling remains visible host work; no "
                "strict removable subregion is claimed"
            ),
            "strict_subregion_extent_instrumentation_only_status": "not_applicable",
            "strict_subregion_extent_control_plane_only_status": "not_applicable",
        }
    if all_apis & _STRICT_SUBREGION_EXTENT_EVENT_APIS:
        return {
            "strict_subregion_extent_target_family_class": (
                "unresolved_launch_event_library_control_boundary"
            ),
            "strict_subregion_extent_candidate_subregion_kind": (
                "candidate_internal_event_launch_library_control_gap"
            ),
            "strict_subregion_extent_candidate_subregion_role": (
                "requires_strict_subregion_extent_source_proof"
            ),
            "strict_subregion_extent_source_proof_status": (
                "unavailable_or_unproven"
            ),
            "strict_subregion_extent_source_proof_reason": (
                "no strict non-perturbing internal subregion extent/source proof is "
                "available; adjacent API gap remains unresolved"
            ),
            "strict_subregion_extent_origin_kind": "unresolved_mixed",
            "strict_subregion_extent_origin_status": (
                "candidate_needs_strict_boundary_origin_proof"
            ),
            "strict_subregion_extent_origin_rule_id": (
                "event_launch_library_control_strict_extent_candidate_v1"
            ),
            "strict_subregion_extent_visibility_kind": "unresolved_mixed",
            "strict_subregion_extent_visibility_status": (
                "candidate_needs_strict_visibility_proof"
            ),
            "strict_subregion_extent_visibility_rule_id": (
                "event_launch_library_control_visibility_unresolved_v1"
            ),
            "strict_subregion_extent_paper_visibility_class": "unresolved_mixed",
            "strict_subregion_extent_paper_visibility_reason": (
                "boundary may contain paper-visible host work and internal control "
                "subregions; strict extent/source, count-once, nonoverlap, wait-map, "
                "and fresh8 preservation proof required"
            ),
            "strict_subregion_extent_instrumentation_only_status": "unproven",
            "strict_subregion_extent_control_plane_only_status": "unproven",
        }
    return {
        "strict_subregion_extent_target_family_class": "not_in_scope",
        "strict_subregion_extent_candidate_subregion_kind": "not_in_scope",
        "strict_subregion_extent_candidate_subregion_role": "not_targeted",
        "strict_subregion_extent_source_proof_status": "not_applicable",
        "strict_subregion_extent_source_proof_reason": (
            "boundary lacks event/launch/library/control strict-subregion target APIs"
        ),
        "strict_subregion_extent_origin_kind": "unavailable",
        "strict_subregion_extent_origin_status": "not_in_scope",
        "strict_subregion_extent_origin_rule_id": None,
        "strict_subregion_extent_visibility_kind": "unavailable",
        "strict_subregion_extent_visibility_status": "not_in_scope",
        "strict_subregion_extent_visibility_rule_id": None,
        "strict_subregion_extent_paper_visibility_class": "not_in_scope",
        "strict_subregion_extent_paper_visibility_reason": (
            "boundary is not part of strict subregion extent/source target surface"
        ),
        "strict_subregion_extent_instrumentation_only_status": "not_applicable",
        "strict_subregion_extent_control_plane_only_status": "not_applicable",
    }


def _strict_subregion_extent_source_metadata(
    previous_event: TraceEvent | None,
    current_event: TraceEvent,
    *,
    stable_boundary_key: str,
    raw_boundary_family: str,
    semantic_boundary_family: str,
    host_dispatch_queue_id: str | None,
    observed_gap_us: int | None,
    previous_materialized_event_id: str | None,
    previous_materialized_api: str | None,
    current_materialized_event_id: str | None,
    current_materialized_api: str | None,
) -> dict[str, object]:
    raw_apis = {
        api
        for api in (
            previous_event.api if previous_event else None,
            current_event.api,
        )
        if api
    }
    semantic_apis = {
        api
        for api in (previous_materialized_api, current_materialized_api)
        if api
    }
    all_apis = raw_apis | semantic_apis
    if not (all_apis & _STRICT_SUBREGION_EXTENT_TARGET_APIS):
        return {}
    if not (all_apis & _STRICT_SUBREGION_EXTENT_EVENT_APIS):
        return {}

    classification = _strict_subregion_extent_classification(
        raw_apis=raw_apis,
        semantic_apis=semantic_apis,
    )
    raw_semantic_pair = f"{raw_boundary_family} || {semantic_boundary_family}"
    candidate_id = f"strict_subregion_extent:{stable_boundary_key}"
    return {
        "strict_subregion_extent_metadata_schema_version": (
            _STRICT_SUBREGION_EXTENT_SOURCE_METADATA_SCHEMA_VERSION
        ),
        "strict_subregion_extent_metadata_opt_in_flag": True,
        "strict_subregion_extent_metadata_env_flags": list(
            _STRICT_SUBREGION_EXTENT_SOURCE_METADATA_EXPORT_ENV_KEYS
        ),
        "strict_subregion_extent_behavior_effect": (
            "diagnostic_only_no_duration_materialization_or_replay_change"
        ),
        "strict_subregion_extent_occurrence_id": stable_boundary_key,
        "strict_subregion_extent_occurrence_id_basis": (
            "stable_hostdelay_occurrence_id"
        ),
        "strict_subregion_extent_boundary_id": candidate_id,
        "strict_subregion_extent_boundary_id_basis": (
            "stable_hostdelay_occurrence_id_context_only"
        ),
        "strict_subregion_extent_rank": int(current_event.rank),
        "strict_subregion_extent_pid": int(current_event.pid),
        "strict_subregion_extent_tid": int(current_event.tid),
        "strict_subregion_extent_dispatch_key": host_dispatch_queue_id,
        "strict_subregion_extent_materialized_hostdelay_event_id": None,
        "strict_subregion_extent_raw_predecessor_event_id": (
            _event_id(previous_event) if previous_event else None
        ),
        "strict_subregion_extent_raw_predecessor_api": (
            previous_event.api if previous_event else None
        ),
        "strict_subregion_extent_raw_successor_event_id": _event_id(current_event),
        "strict_subregion_extent_raw_successor_api": current_event.api,
        "strict_subregion_extent_semantic_predecessor_event_id": (
            previous_materialized_event_id
        ),
        "strict_subregion_extent_semantic_predecessor_api": (
            previous_materialized_api
        ),
        "strict_subregion_extent_semantic_predecessor_materialized": (
            previous_materialized_event_id is not None
        ),
        "strict_subregion_extent_semantic_successor_event_id": (
            current_materialized_event_id
        ),
        "strict_subregion_extent_semantic_successor_api": current_materialized_api,
        "strict_subregion_extent_semantic_successor_materialized": (
            current_materialized_event_id is not None
        ),
        "strict_subregion_extent_raw_family": raw_boundary_family,
        "strict_subregion_extent_semantic_family": semantic_boundary_family,
        "strict_subregion_extent_raw_semantic_pair": raw_semantic_pair,
        **classification,
        "strict_subregion_extent_candidate_subregion_id": candidate_id,
        "strict_subregion_extent_start_ts_us": None,
        "strict_subregion_extent_end_ts_us": None,
        "strict_subregion_extent_duration_us_context_only": (
            int(observed_gap_us) if observed_gap_us is not None else None
        ),
        "strict_subregion_extent_clock_domain": "rank_local_trace_clock",
        "strict_subregion_extent_timestamp_basis": (
            "raw_trace_adjacent_api_timestamps_context_only"
        ),
        "strict_subregion_extent_timestamp_source_kind": (
            "none_strict_source_unavailable"
        ),
        "strict_subregion_extent_timestamp_source_id": None,
        "strict_subregion_extent_source_surface": "collate_trace_processing",
        "strict_subregion_extent_source_producer": "maya_lite.collate",
        "strict_subregion_extent_source_capture_phase": (
            "post_trace_capture_collation"
        ),
        "strict_subregion_extent_source_is_non_perturbing": False,
        "strict_subregion_extent_source_perturbation_risk": (
            "no_strict_source_available_context_only"
        ),
        "strict_subregion_extent_source_uses_runtime_endpoint_substitution": False,
        "strict_subregion_extent_source_uses_measured_actual_runtime": False,
        "strict_subregion_extent_source_uses_hostdelay_shortening": False,
        "strict_subregion_extent_source_evidence": {
            "raw_apis": sorted(raw_apis),
            "semantic_apis": sorted(semantic_apis),
            "evidence_status": "context_only_not_strict_subregion_extent",
        },
        "strict_subregion_extent_origin_evidence": {
            "raw_boundary_family": raw_boundary_family,
            "semantic_boundary_family": semantic_boundary_family,
            "evidence_status": "strict_boundary_origin_not_proven",
        },
        "strict_subregion_extent_visibility_evidence": {
            "semantic_predecessor_materialized": previous_materialized_event_id
            is not None,
            "semantic_successor_materialized": current_materialized_event_id
            is not None,
            "evidence_status": "paper_invisible_subregion_not_proven",
        },
        "strict_subregion_extent_already_modeled_replay_status": "unavailable",
        "strict_subregion_extent_count_once_group_id": (
            f"strict_subregion_extent_count_once:{stable_boundary_key}"
        ),
        "strict_subregion_extent_count_once_status": "unavailable",
        "strict_subregion_extent_count_once_evidence": None,
        "strict_subregion_extent_nonoverlap_group_id": stable_boundary_key,
        "strict_subregion_extent_nonoverlap_status": "unavailable",
        "strict_subregion_extent_nonoverlap_evidence": None,
        "strict_subregion_extent_host_dispatch_overlap_status": "unavailable",
        "strict_subregion_extent_provider_runtime_overlap_status": "unavailable",
        "strict_subregion_extent_stream_queue_wait_overlap_status": "unavailable",
        "strict_subregion_extent_cuda_event_wait_map_edge_id": None,
        "strict_subregion_extent_cuda_event_wait_map_safety_status": "unavailable",
        "strict_subregion_extent_collective_wait_map_edge_id": None,
        "strict_subregion_extent_collective_wait_map_safety_status": "unavailable",
        "strict_subregion_extent_replay_ordering_safety_status": "unavailable",
        "strict_subregion_extent_stream_fifo_safety_status": "unavailable",
        "strict_subregion_extent_fresh16_fresh8_comparable_join_key": (
            f"strict_subregion_extent:{raw_semantic_pair}"
        ),
        "strict_subregion_extent_fresh16_fresh8_join_key_basis": (
            "raw_and_semantic_boundary_family_context_only"
        ),
        "strict_subregion_extent_fresh16_fresh8_join_key_status": (
            "diagnostic_context_only_not_strict_counterpart"
        ),
        "strict_subregion_extent_fresh8_preservation_risk_status": "unreviewed",
        "strict_subregion_extent_fresh8_preservation_evidence": None,
        "strict_subregion_extent_repair_ready": False,
        "strict_subregion_extent_repair_ready_reason": (
            "requires strict extent/source, origin, visibility, count-once, "
            "nonoverlap, wait-map safety, and fresh8 preservation proof"
        ),
        "strict_subregion_extent_safe_to_use_as_repair_evidence": False,
        "strict_subregion_extent_safe_to_use_as_subtraction_delta": False,
        "strict_subregion_extent_safe_delta_us": None,
        "strict_subregion_extent_safe_delta_basis": None,
        "strict_subregion_extent_runtime_or_endpoint_substitution_used": False,
        "strict_subregion_extent_hostdelay_shortening_used": False,
        "strict_subregion_extent_rank_workload_special_case_used": False,
    }


def _make_host_delay_event(
    *,
    event_id: str,
    rank: int,
    ordinal: int,
    source,
    ts: int,
    pid: int,
    tid: int,
    observed_gap_us: int,
    prev_event_id: str | None,
    dispatch_scope: str,
    host_machine_id: str,
    host_dispatch_queue_id: str,
    boundary: str | None = None,
    boundary_origin_provenance: Mapping[str, object] | None = None,
) -> CollatedEvent:
    if boundary == "step_start":
        hostdelay_source = "leading_step_gap"
    elif boundary == "step_end":
        hostdelay_source = "trailing_step_gap"
    else:
        hostdelay_source = "collate_host_gap"
    extras: dict[str, object] = {
        "hostdelay_source": hostdelay_source,
        "observed_gap_us": int(observed_gap_us),
        "host_timing_dispatch_scope": str(dispatch_scope),
        "host_machine_id": host_machine_id,
        "host_dispatch_queue_id": host_dispatch_queue_id,
        "host_dispatch_model": _host_dispatch_model(dispatch_scope),
    }
    if boundary is not None:
        extras["boundary"] = boundary
    if boundary_origin_provenance:
        extras.update(boundary_origin_provenance)
    if extras.get("hostdelay_occurrence_metadata_opt_in_flag") is True:
        interval_end_ts = int(ts) + int(max(observed_gap_us, 0))
        extras.update(
            {
                "hostdelay_occurrence_materialized_hostdelay_event_id": event_id,
                "hostdelay_occurrence_source": hostdelay_source,
                "hostdelay_occurrence_interval_start_ts_us": int(ts),
                "hostdelay_occurrence_interval_end_ts_us": interval_end_ts,
                "hostdelay_occurrence_duration_us": int(max(observed_gap_us, 0)),
                "hostdelay_occurrence_timestamp_basis": (
                    "materialized_hostDelay_ts_plus_observed_gap_us"
                ),
                "hostdelay_occurrence_timestamp_source": "collate_host_gap",
                "hostdelay_occurrence_timestamp_clock_domain": "trace_host_us",
                "hostdelay_occurrence_interval_basis_status": (
                    "diagnostic_metadata_only_duration_unchanged"
                ),
                "repair_ready": False,
                "safe_to_use_as_repair_evidence": False,
                "safe_to_use_as_subtraction_delta": False,
                "safe_delta_us": None,
            }
        )
    if extras.get("component_strict_counterpart_opt_in_flag") is True:
        component_defaults = {
            "stable_predicted_interval_row_id": event_id,
            "stable_predicted_edge_row_id": None,
            "stable_predicted_count_once_group_id": f"replay_interval:{event_id}",
            "predicted_interval_start_us": int(ts),
            "predicted_interval_end_us": int(ts) + int(max(observed_gap_us, 0)),
            "predicted_interval_duration_us": float(max(int(observed_gap_us), 0)),
            "predicted_interval_duration_basis": "materialized_hostDelay_observed_gap_us",
        }
        for key, value in component_defaults.items():
            if extras.get(key) in (None, ""):
                extras[key] = value
    if extras.get("gemm_adjacent_hostdelay_opt_in_flag") is True:
        extras.setdefault("gemm_adjacent_materialized_hostdelay_event_id", event_id)
        extras.setdefault("gemm_adjacent_hostdelay_event_id", event_id)
        extras.setdefault("gemm_adjacent_hostdelay_source", hostdelay_source)
        extras.setdefault("gemm_adjacent_predicted_count_once_interval_id", event_id)
    if extras.get("host_control_envelope_counterpart_opt_in_flag") is True:
        extras.setdefault("host_control_envelope_hostdelay_interval_id", event_id)
        extras.setdefault("hostdelay_interval_id", event_id)
        extras.setdefault("host_control_envelope_interval_start_ts_us", int(ts))
        extras.setdefault(
            "host_control_envelope_interval_end_ts_us",
            int(ts) + int(max(observed_gap_us, 0)),
        )
        extras.setdefault(
            "host_control_envelope_interval_duration_us",
            float(max(int(observed_gap_us), 0)),
        )
        extras.setdefault(
            "host_control_envelope_interval_time_basis",
            "materialized_hostDelay_ts_plus_observed_gap_us",
        )
    return CollatedEvent(
        id=event_id,
        rank=rank,
        ordinal=ordinal,
        source=source,
        ts=int(ts),
        pid=pid,
        tid=tid,
        module="host.dispatch",
        api="__hostDelay__",
        op_type="host_delay",
        extras=extras,
        prev_event_id=prev_event_id,
        collective_group_id=None,
    )


def _safe_actual_boundary_metadata(
    previous_event: TraceEvent | None,
    current_event: TraceEvent,
    *,
    observed_gap_us: int,
    host_dispatch_queue_id: str,
    actual_counterpart_window_id: str | None = None,
    include_visibility_aliases: bool = False,
) -> dict[str, object]:
    """Derive ledger-only host boundary metadata from existing trace context."""

    boundary_apis: list[str] = []
    if previous_event is not None:
        boundary_apis.append(previous_event.api)
    boundary_apis.append(current_event.api)
    metadata: dict[str, object] = {
        "actual_counterpart_component_id": "host_inter_op_overhead",
        "actual_inter_host_op_gap_us": float(max(int(observed_gap_us), 0)),
        "actual_counterpart_rank": int(current_event.rank),
        "actual_counterpart_current_event_id": _event_id(current_event),
        "actual_counterpart_boundary_family": " -> ".join(boundary_apis),
        "actual_counterpart_dispatch_queue_id": host_dispatch_queue_id,
        "actual_counterpart_visibility_kind": "mixed_or_unresolved",
    }
    if include_visibility_aliases:
        metadata["actual_rank"] = int(current_event.rank)
        metadata["actual_boundary_family"] = " -> ".join(boundary_apis)
    if previous_event is not None:
        metadata["actual_counterpart_prev_event_id"] = _event_id(previous_event)
    if actual_counterpart_window_id is not None:
        metadata["actual_counterpart_window_id"] = actual_counterpart_window_id
        if include_visibility_aliases:
            metadata["actual_paper_valid_window_id"] = actual_counterpart_window_id
    return metadata


def _safe_launch_metadata(event: TraceEvent) -> dict[str, object]:
    """Return additive launch signature aliases from existing payload fields."""

    if event.api != "cudaLaunchKernel":
        return {}
    metadata: dict[str, object] = {}
    kernel_name = event.extras.get("kernel")
    if kernel_name not in (None, ""):
        metadata["launch_kernel_name"] = kernel_name
    grid = tuple(event.extras.get(key) for key in ("grid_x", "grid_y", "grid_z"))
    if any(value is not None for value in grid):
        metadata["launch_grid"] = "x".join("" if value is None else str(value) for value in grid)
    block = tuple(event.extras.get(key) for key in ("block_x", "block_y", "block_z"))
    if any(value is not None for value in block):
        metadata["launch_block"] = "x".join("" if value is None else str(value) for value in block)
    if "shared_mem" in event.extras:
        metadata["launch_shared_mem_bytes"] = event.extras["shared_mem"]
    if "stream_id" in event.extras:
        metadata["launch_stream_id"] = event.extras["stream_id"]
    return metadata


def _is_internal_launch_config_metadata_edge(
    previous_event: TraceEvent | None,
    current_event: TraceEvent,
) -> bool:
    return (
        previous_event is not None
        and previous_event.api == "__cudaPopCallConfiguration"
        and current_event.api == "cudaLaunchKernel"
    )


_CUBLAS_SET_STREAM_CONTEXT_QUERY_SUFFIX_APIS = {
    "cudaGetDevice",
}
_CONTEXT_QUERY_SUFFIX_EVENT_RECORD_TARGET_APIS = {
    "cudaEventRecord",
    "cudaEventRecordWithFlags",
}


@dataclass
class _ContextQuerySuffixState:
    events: list[TraceEvent]
    internal_gap_us: int = 0
    internal_gap_rows: list[dict[str, object]] | None = None
    leading_pre_suffix_gap_us: int = 0
    leading_pre_suffix_dispatch_start_ts: int | None = None


@dataclass
class _PendingLaunchConfigPopContextQuerySuffixFold:
    state: _ContextQuerySuffixState
    pop_event: TraceEvent
    terminal_gap_us: int
    preserved_pre_suffix_gap_us: int


@dataclass
class _IdempotentCublasSetStreamFoldState:
    events: list[TraceEvent]
    internal_gap_rows: list[dict[str, object]]


def _is_cublas_set_stream_context_query(event: TraceEvent) -> bool:
    return (
        event.api in _CUBLAS_SET_STREAM_CONTEXT_QUERY_SUFFIX_APIS
        and is_compat_only_api(event.api, event.op_type)
    )


def _is_cublas_set_stream_context_query_suffix_target(event: TraceEvent) -> bool:
    return _is_context_query_suffix_target(event)


def _is_context_query_suffix_target(event: TraceEvent) -> bool:
    if event.api == "cublasSetStream_v2":
        return True
    if event.api in _CONTEXT_QUERY_SUFFIX_EVENT_RECORD_TARGET_APIS:
        return _context_query_suffix_event_record_fold_enabled()
    return False


def _is_context_query_suffix_launch_config_pop_target(event: TraceEvent) -> bool:
    return (
        event.api == "__cudaPopCallConfiguration"
        and _context_query_suffix_launch_config_pop_fold_enabled()
    )


def _cublas_handle_id(event: TraceEvent) -> str | None:
    handle_id = _normalized_text(event.extras.get("handle_id"))
    if handle_id in (None, "0", "0x0"):
        return None
    return handle_id


def _cublas_stream_id(event: TraceEvent) -> str | None:
    if "stream_id" not in event.extras:
        return None
    stream_id = event.extras.get("stream_id")
    if stream_id in (None, "", "0", "0x0"):
        return "__default_stream__"
    return str(stream_id)


def _cublas_handle_stream_state_key(
    event: TraceEvent,
    dispatch_key: tuple[object, ...],
) -> tuple[int, tuple[object, ...], str] | None:
    if event.api != "cublasSetStream_v2":
        return None
    handle_id = _cublas_handle_id(event)
    stream_id = _cublas_stream_id(event)
    if handle_id is None or stream_id is None:
        return None
    return (int(event.rank), dispatch_key, handle_id)


def _cublas_handle_lifecycle_state_key(
    event: TraceEvent,
    dispatch_key: tuple[object, ...],
) -> tuple[int, tuple[object, ...], str] | None:
    if event.api not in {"cublasCreate_v2", "cublasDestroy_v2"}:
        return None
    handle_id = _cublas_handle_id(event)
    if handle_id is None:
        return None
    return (int(event.rank), dispatch_key, handle_id)


def _context_query_suffix_gap_row(
    previous_event: TraceEvent,
    current_event: TraceEvent,
    *,
    observed_gap_us: int,
    gap_kind: str,
) -> dict[str, object]:
    return {
        "gap_kind": gap_kind,
        "raw_prev_event_id": _event_id(previous_event),
        "raw_prev_api": previous_event.api,
        "raw_prev_ts_us": int(previous_event.ts),
        "raw_current_event_id": _event_id(current_event),
        "raw_current_api": current_event.api,
        "raw_current_ts_us": int(current_event.ts),
        "observed_gap_us": int(max(observed_gap_us, 0)),
        "host_gap_contribution": "excluded_from_pending_hostdelay",
    }


def _context_query_run_preserved_gap_row(
    previous_event: TraceEvent,
    current_event: TraceEvent,
    *,
    observed_gap_us: int,
    gap_kind: str,
) -> dict[str, object]:
    return {
        "gap_kind": gap_kind,
        "raw_prev_event_id": _event_id(previous_event),
        "raw_prev_api": previous_event.api,
        "raw_prev_ts_us": int(previous_event.ts),
        "raw_current_event_id": _event_id(current_event),
        "raw_current_api": current_event.api,
        "raw_current_ts_us": int(current_event.ts),
        "observed_gap_us": int(max(observed_gap_us, 0)),
        "host_gap_contribution": "included_in_pending_hostdelay",
    }


def _idempotent_cublas_set_stream_gap_row(
    previous_event: TraceEvent,
    current_event: TraceEvent,
    *,
    observed_gap_us: int,
    gap_kind: str,
) -> dict[str, object]:
    return {
        "gap_kind": gap_kind,
        "raw_prev_event_id": _event_id(previous_event),
        "raw_prev_api": previous_event.api,
        "raw_prev_ts_us": int(previous_event.ts),
        "raw_current_event_id": _event_id(current_event),
        "raw_current_api": current_event.api,
        "raw_current_ts_us": int(current_event.ts),
        "observed_gap_us": int(max(observed_gap_us, 0)),
        "host_gap_contribution": "included_in_pending_hostdelay",
    }


def _context_query_run_fold_metadata(
    state: _ContextQuerySuffixState,
    target_event: TraceEvent,
    *,
    terminal_gap_us: int,
    preserved_pre_run_gap_us: int,
) -> dict[str, object]:
    folded_rows = []
    for folded_event in state.events:
        raw_end_ts_us, raw_end_ts_source = _event_end_ts_context(folded_event)
        folded_rows.append(
            {
                "raw_event_id": _event_id(folded_event),
                "raw_api": folded_event.api,
                "raw_ts_us": int(folded_event.ts),
                "raw_end_ts_us": raw_end_ts_us,
                "raw_end_ts_source": raw_end_ts_source,
                "raw_host_duration_us": _normalized_float(
                    folded_event.extras.get("host_duration_us")
                ),
                "raw_duration_us": _event_host_duration_us(folded_event),
                "raw_extras": dict(folded_event.extras),
            }
        )
    gap_rows = list(state.internal_gap_rows or [])
    if state.events:
        gap_rows.append(
            _context_query_run_preserved_gap_row(
                state.events[-1],
                target_event,
                observed_gap_us=terminal_gap_us,
                gap_kind="cudaGetDevice_context_query_run_terminal_gap_to_materialized_event",
            )
        )
    return {
        "cuda_get_device_context_query_run_fold_basis": (
            "consecutive_cudaGetDevice_context_query_run"
        ),
        "cuda_get_device_context_query_run_fold_status": (
            "applied_default_paper_aligned_internal_gap_fold"
        ),
        "cuda_get_device_context_query_run_fold_reason": (
            "repeated cudaGetDevice compat-only context queries inside one "
            "consecutive run are internal query churn; leading and terminal "
            "gaps around the run remain host-visible"
        ),
        "cuda_get_device_context_query_run_event_count": len(state.events),
        "cuda_get_device_context_query_run_event_ids": [
            _event_id(folded_event) for folded_event in state.events
        ],
        "cuda_get_device_context_query_run_rows": folded_rows,
        "cuda_get_device_context_query_run_gap_rows": gap_rows,
        "cuda_get_device_context_query_run_internal_gap_us": int(
            max(state.internal_gap_us, 0)
        ),
        "cuda_get_device_context_query_run_terminal_gap_us": int(
            max(terminal_gap_us, 0)
        ),
        "cuda_get_device_context_query_run_suppressed_host_gap_us": int(
            max(state.internal_gap_us, 0)
        ),
        "cuda_get_device_context_query_run_preserved_pre_run_gap_us": int(
            max(preserved_pre_run_gap_us, 0)
        ),
        "cuda_get_device_context_query_run_host_gap_contribution": (
            "only_internal_run_gaps_excluded_from_pending_hostdelay"
        ),
        "cuda_get_device_context_query_run_target_event_id": _event_id(target_event),
        "cuda_get_device_context_query_run_target_api": target_event.api,
    }


def _context_query_suffix_metadata(
    state: _ContextQuerySuffixState,
    target_event: TraceEvent,
    *,
    terminal_gap_us: int,
    preserved_pre_suffix_gap_us: int,
) -> dict[str, object]:
    folded_rows = []
    for folded_event in state.events:
        raw_end_ts_us, raw_end_ts_source = _event_end_ts_context(folded_event)
        folded_rows.append(
            {
                "raw_event_id": _event_id(folded_event),
                "raw_api": folded_event.api,
                "raw_ts_us": int(folded_event.ts),
                "raw_end_ts_us": raw_end_ts_us,
                "raw_end_ts_source": raw_end_ts_source,
                "raw_host_duration_us": _normalized_float(
                    folded_event.extras.get("host_duration_us")
                ),
                "raw_duration_us": _event_host_duration_us(folded_event),
                "raw_extras": dict(folded_event.extras),
            }
        )
    gap_rows = list(state.internal_gap_rows or [])
    if state.events:
        gap_rows.append(
            _context_query_suffix_gap_row(
                state.events[-1],
                target_event,
                observed_gap_us=terminal_gap_us,
                gap_kind=f"context_query_suffix_terminal_gap_to_{target_event.api}",
            )
        )
    suppressed_gap_us = int(max(state.internal_gap_us, 0)) + int(max(terminal_gap_us, 0))
    folded_apis = [folded_event.api for folded_event in state.events]
    folded_event_ids = [_event_id(folded_event) for folded_event in state.events]
    target_api = target_event.api
    metadata: dict[str, object] = {
        "hostdelay_semantic_predecessor_control_query_suffix_fold_basis": (
            f"cudaGetDevice_context_query_suffix_before_{target_api}"
        ),
        "hostdelay_semantic_predecessor_control_query_suffix_fold_status": (
            "applied_default_paper_aligned_metadata_fold"
        ),
        "hostdelay_semantic_predecessor_control_query_suffix_fold_reason": (
            "cudaGetDevice is a compat-only context query; the following "
            "paper-visible semantic operation remains the materialized boundary target"
        ),
        "hostdelay_semantic_predecessor_control_query_suffix_target_api": target_api,
        "hostdelay_semantic_predecessor_control_query_suffix_folded_count": len(
            state.events
        ),
        "hostdelay_semantic_predecessor_control_query_suffix_folded_apis": folded_apis,
        "hostdelay_semantic_predecessor_control_query_suffix_folded_event_ids": (
            folded_event_ids
        ),
        "hostdelay_semantic_predecessor_control_query_suffix_folded_rows": folded_rows,
        "hostdelay_semantic_predecessor_control_query_suffix_gap_rows": gap_rows,
        "hostdelay_semantic_predecessor_control_query_suffix_internal_gap_us": int(
            max(state.internal_gap_us, 0)
        ),
        "hostdelay_semantic_predecessor_control_query_suffix_terminal_gap_us": int(
            max(terminal_gap_us, 0)
        ),
        "hostdelay_semantic_predecessor_control_query_suffix_suppressed_host_gap_us": (
            suppressed_gap_us
        ),
        "hostdelay_semantic_predecessor_control_query_suffix_preserved_pre_suffix_gap_us": int(
            max(preserved_pre_suffix_gap_us, 0)
        ),
        "hostdelay_semantic_predecessor_control_query_suffix_host_gap_contribution": (
            "only_internal_suffix_gaps_excluded_from_pending_hostdelay"
        ),
    }
    if target_event.api == "cublasSetStream_v2":
        metadata.update(
            {
                "cublas_set_stream_context_query_suffix_fold_basis": (
                    "cudaGetDevice_context_query_suffix_before_cublasSetStream_v2"
                ),
                "cublas_set_stream_context_query_suffix_fold_status": (
                    metadata[
                        "hostdelay_semantic_predecessor_control_query_suffix_fold_status"
                    ]
                ),
                "cublas_set_stream_context_query_suffix_fold_reason": (
                    "cudaGetDevice is a compat-only context query; cublasSetStream_v2 "
                    "remains the materialized semantic handle-stream state operation"
                ),
                "cublas_set_stream_context_query_suffix_folded_count": len(state.events),
                "cublas_set_stream_context_query_suffix_folded_apis": folded_apis,
                "cublas_set_stream_context_query_suffix_folded_event_ids": (
                    folded_event_ids
                ),
                "cublas_set_stream_context_query_suffix_folded_rows": folded_rows,
                "cublas_set_stream_context_query_suffix_gap_rows": gap_rows,
                "cublas_set_stream_context_query_suffix_internal_gap_us": int(
                    max(state.internal_gap_us, 0)
                ),
                "cublas_set_stream_context_query_suffix_terminal_gap_us": int(
                    max(terminal_gap_us, 0)
                ),
                "cublas_set_stream_context_query_suffix_suppressed_host_gap_us": (
                    suppressed_gap_us
                ),
                "cublas_set_stream_context_query_suffix_preserved_pre_suffix_gap_us": int(
                    max(preserved_pre_suffix_gap_us, 0)
                ),
                "cublas_set_stream_context_query_suffix_host_gap_contribution": (
                    "only_internal_suffix_gaps_excluded_from_pending_hostdelay"
                ),
            }
        )
    return metadata


def _context_query_suffix_launch_config_pop_metadata(
    pending: _PendingLaunchConfigPopContextQuerySuffixFold,
    launch_event: TraceEvent,
) -> dict[str, object]:
    metadata = _context_query_suffix_metadata(
        pending.state,
        pending.pop_event,
        terminal_gap_us=pending.terminal_gap_us,
        preserved_pre_suffix_gap_us=pending.preserved_pre_suffix_gap_us,
    )
    metadata.update(
        {
            "hostdelay_semantic_predecessor_control_query_suffix_fold_status": (
                "applied_opt_in_launch_config_pop_internal_metadata_fold"
            ),
            "hostdelay_semantic_predecessor_control_query_suffix_fold_reason": (
                "cudaGetDevice is a compat-only context query immediately before "
                "internal __cudaPopCallConfiguration launch-configuration metadata; "
                "the following cudaLaunchKernel remains materialized"
            ),
            "hostdelay_semantic_predecessor_control_query_suffix_target_role": (
                "internal_launch_config_pop"
            ),
            "hostdelay_semantic_predecessor_control_query_suffix_materialized_target_event_id": (
                _event_id(launch_event)
            ),
            "hostdelay_semantic_predecessor_control_query_suffix_materialized_target_api": (
                launch_event.api
            ),
            "hostdelay_semantic_predecessor_control_query_suffix_opt_in_env_flags": list(
                _CONTEXT_QUERY_SUFFIX_LAUNCH_CONFIG_POP_FOLD_ENV_KEYS
            ),
        }
    )
    return metadata


def _idempotent_cublas_set_stream_fold_metadata(
    state: _IdempotentCublasSetStreamFoldState,
    target_event: TraceEvent,
    *,
    pending_host_gap_us: int,
    terminal_gap_us: int | None,
) -> dict[str, object]:
    folded_rows = []
    handle_stream_pairs: list[dict[str, object]] = []
    handle_stream_pair_by_key: dict[tuple[str | None, str | None], dict[str, object]] = {}
    for folded_event in state.events:
        handle_id = _cublas_handle_id(folded_event)
        stream_id = _cublas_stream_id(folded_event)
        handle_stream_key = (handle_id, stream_id)
        handle_stream_pair = handle_stream_pair_by_key.get(handle_stream_key)
        if handle_stream_pair is None:
            handle_stream_pair = {
                "handle_id": handle_id,
                "stream_id": stream_id,
                "folded_count": 0,
                "folded_event_ids": [],
            }
            handle_stream_pair_by_key[handle_stream_key] = handle_stream_pair
            handle_stream_pairs.append(handle_stream_pair)
        handle_stream_pair["folded_count"] = int(handle_stream_pair["folded_count"]) + 1
        handle_stream_pair["folded_event_ids"].append(_event_id(folded_event))
        raw_end_ts_us, raw_end_ts_source = _event_end_ts_context(folded_event)
        folded_rows.append(
            {
                "raw_event_id": _event_id(folded_event),
                "raw_api": folded_event.api,
                "raw_ts_us": int(folded_event.ts),
                "raw_end_ts_us": raw_end_ts_us,
                "raw_end_ts_source": raw_end_ts_source,
                "raw_host_duration_us": _normalized_float(
                    folded_event.extras.get("host_duration_us")
                ),
                "raw_duration_us": _event_host_duration_us(folded_event),
                "raw_handle_id": handle_id,
                "raw_stream_id": stream_id,
                "raw_extras": dict(folded_event.extras),
            }
        )
    gap_rows = list(state.internal_gap_rows)
    if terminal_gap_us is not None and state.events:
        gap_rows.append(
            _idempotent_cublas_set_stream_gap_row(
                state.events[-1],
                target_event,
                observed_gap_us=terminal_gap_us,
                gap_kind="idempotent_cublasSetStream_v2_terminal_gap_to_next_materialized_event",
            )
        )
    return {
        "cublas_set_stream_idempotent_fold_basis": (
            "rank_dispatch_local_handle_stream_state"
        ),
        "cublas_set_stream_idempotent_fold_status": (
            "enabled_opt_in_experimental_idempotent_state_fold"
        ),
        "cublas_set_stream_idempotent_fold_enabled": True,
        "cublas_set_stream_idempotent_fold_env_flags": list(
            _IDEMPOTENT_CUBLAS_SET_STREAM_FOLD_ENV_KEYS
        ),
        "cublas_set_stream_idempotent_fold_reason": (
            "repeated cublasSetStream_v2 with unchanged rank/dispatch-local "
            "handle stream state is a no-op state transition; the prior "
            "materialized state remains sufficient for later GEMM placement"
        ),
        "cublas_set_stream_idempotent_fold_handle_id": (
            handle_stream_pairs[0]["handle_id"]
            if len(handle_stream_pairs) == 1
            else "__multiple__"
        ),
        "cublas_set_stream_idempotent_fold_stream_id": (
            handle_stream_pairs[0]["stream_id"]
            if len(handle_stream_pairs) == 1
            else "__multiple__"
        ),
        "cublas_set_stream_idempotent_folded_handle_ids": [
            pair["handle_id"] for pair in handle_stream_pairs
        ],
        "cublas_set_stream_idempotent_folded_stream_ids": [
            pair["stream_id"] for pair in handle_stream_pairs
        ],
        "cublas_set_stream_idempotent_folded_handle_stream_pairs": handle_stream_pairs,
        "cublas_set_stream_idempotent_folded_count": len(state.events),
        "cublas_set_stream_idempotent_folded_event_ids": [
            _event_id(folded_event) for folded_event in state.events
        ],
        "cublas_set_stream_idempotent_folded_rows": folded_rows,
        "cublas_set_stream_idempotent_fold_gap_rows": gap_rows,
        "cublas_set_stream_idempotent_fold_pending_host_gap_us": int(
            max(pending_host_gap_us, 0)
        ),
        "cublas_set_stream_idempotent_fold_host_gap_contribution": (
            "folded_api_gaps_preserved_in_pending_hostdelay"
        ),
        "cublas_set_stream_idempotent_fold_target_event_id": _event_id(target_event),
        "cublas_set_stream_idempotent_fold_target_api": target_event.api,
    }


def _launch_config_metadata_from_pop(
    previous_event: TraceEvent | None,
    *,
    adjacent_gap_us: int,
    suppress_hostdelay_gap: bool,
    normalization_status: str,
) -> dict[str, object]:
    """Preserve collapsed CUDA launch-config provenance on the launch event."""

    if previous_event is None:
        return {}
    raw_end_ts_us, raw_end_ts_source = _event_end_ts_context(previous_event)
    contribution = (
        "excluded_from_pending_host_gap"
        if suppress_hostdelay_gap
        else "included_in_pending_host_gap_disable_control"
    )
    return {
        "launch_config_metadata_basis": "internal_launch_config_metadata",
        "launch_config_metadata_reason": (
            "__cudaPopCallConfiguration carries internal CUDA launch configuration "
            "metadata for the following cudaLaunchKernel"
        ),
        "launch_config_raw_event_id": _event_id(previous_event),
        "launch_config_raw_api": previous_event.api,
        "launch_config_raw_ts_us": int(previous_event.ts),
        "launch_config_raw_end_ts_us": raw_end_ts_us,
        "launch_config_raw_end_ts_source": raw_end_ts_source,
        "launch_config_raw_host_duration_us": _normalized_float(
            previous_event.extras.get("host_duration_us")
        ),
        "launch_config_raw_duration_us": _event_host_duration_us(previous_event),
        "launch_config_raw_extras": dict(previous_event.extras),
        "launch_config_adjacent_host_gap_us": int(max(adjacent_gap_us, 0)),
        "launch_config_adjacent_host_gap_contribution": contribution,
        "launch_config_hostdelay_normalization_enabled": bool(suppress_hostdelay_gap),
        "launch_config_hostdelay_normalization_status": normalization_status,
        "launch_config_hostdelay_normalization_env_flags": list(
            _LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION_ENV_KEYS
        ),
        "launch_config_hostdelay_normalization_disable_env_flags": list(
            _LAUNCH_CONFIG_HOSTDELAY_NORMALIZATION_DISABLE_ENV_KEYS
        ),
        **(
            {"hostdelay_normalization_basis": "internal_launch_config_metadata"}
            if suppress_hostdelay_gap
            else {}
        ),
    }


_BOUNDARY_ORIGIN_FIELDS = {
    *_COMPONENT_STRICT_COUNTERPART_FIELDS,
    *_COMPONENT_STRICT_ACTUAL_ENDPOINT_SIDECAR_ONLY_FIELDS,
    *_GEMM_ADJACENT_HOSTDELAY_BOUNDARY_FIELDS,
    *_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_FIELDS,
    *_JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_FIELDS,
    "host_control_envelope_counterpart_schema_version",
    "host_control_envelope_counterpart_opt_in_flag",
    "host_control_envelope_counterpart_key",
    "hostdelay_counterpart_key",
    "host_control_envelope_materialized_interval_id",
    "host_control_envelope_counterpart_interval_id",
    "host_control_envelope_hostdelay_interval_id",
    "hostdelay_interval_id",
    "host_control_envelope_actual_row_id",
    "host_control_envelope_actual_interval_id",
    "host_control_envelope_actual_interval_kind",
    "host_control_envelope_prev_raw_event_id",
    "host_control_envelope_current_raw_event_id",
    "host_control_envelope_prev_api",
    "host_control_envelope_current_api",
    "host_control_envelope_rank",
    "host_control_envelope_stream_id",
    "host_control_envelope_host_dispatch_queue_id",
    "host_control_envelope_paper_valid_window_id",
    "host_control_envelope_prev_raw_ordinal",
    "host_control_envelope_current_raw_ordinal",
    "host_control_envelope_timestamp_basis",
    "host_control_envelope_interval_start_ts_us",
    "host_control_envelope_interval_end_ts_us",
    "host_control_envelope_interval_duration_us",
    "host_control_envelope_interval_time_basis",
    "host_control_envelope_visibility_basis_status",
    "host_control_envelope_visibility_kind",
    "host_control_envelope_replay_overlap_status",
    "host_control_envelope_replay_overlap_unavailable_reason",
    "boundary_origin_kind",
    "boundary_segment_schema_version",
    "launch_boundary_id",
    "launch_boundary_id_unavailable_reason",
    "wrapper_segment_coverage",
    "wrapper_segment_sum_us",
    "wrapper_segment_unattributed_us",
    "boundary_origin_classification_basis",
    "boundary_visibility_segments",
    "paper_visible_host_duration_us",
    "wrapper_internal_duration_us",
    "instrumentation_only_duration_us",
    "caller_visible_elapsed_us",
    "fake_api_body_duration_us",
    "unresolved_mixed_duration_us",
    "actual_counterpart_component_id",
    "actual_counterpart_id",
    "actual_counterpart_status",
    "actual_counterpart_unavailable_reason",
    "actual_host_dispatch_duration_us",
    "actual_inter_host_op_gap_us",
    "actual_inter_host_op_gap_unavailable_reason",
    "actual_counterpart_rank",
    "actual_counterpart_window_id",
    "actual_counterpart_window_unavailable_reason",
    "actual_counterpart_prev_event_id",
    "actual_counterpart_current_event_id",
    "actual_counterpart_boundary_family",
    "actual_counterpart_dispatch_queue_id",
    "actual_counterpart_replay_rank_total_anchor",
    "actual_counterpart_visibility_kind",
    "host_control_boundary_counterpart_schema_version",
    "host_control_boundary_row_id",
    "host_control_boundary_occurrence_id",
    "host_control_boundary_selection_status",
    "host_control_boundary_prev_raw_event_id",
    "host_control_boundary_current_raw_event_id",
    "host_control_boundary_prev_api",
    "host_control_boundary_current_api",
    "host_control_boundary_family",
    "host_control_boundary_prev_ts_us",
    "host_control_boundary_prev_end_ts_us",
    "host_control_boundary_prev_host_duration_us",
    "host_control_boundary_current_ts_us",
    "host_control_boundary_current_end_ts_us",
    "host_control_boundary_current_host_duration_us",
    "host_control_boundary_host_machine_id",
    "host_control_boundary_dispatch_queue_id",
    "host_control_visibility_split_status",
    "host_control_visibility_split_unavailable_reason",
    "host_control_visibility_split_basis",
    "mechanical_visibility_split_status",
    "mechanical_visibility_split_unavailable_reason",
    "host_control_producer_visibility_schema_version",
    "host_control_producer_visibility_status",
    "host_control_producer_visibility_unavailable_reason",
    "host_control_producer_visibility_basis",
    "host_control_producer_visibility_segments",
    "host_control_producer_numeric_split_status",
    "host_control_producer_numeric_split_unavailable_reason",
    "host_control_producer_nonoverlap_status",
    "host_control_producer_nonoverlap_unavailable_reason",
    "host_control_producer_wait_map_nonoverlap_status",
    "host_control_producer_wait_map_nonoverlap_unavailable_reason",
    "host_control_producer_double_counting_nonoverlap_status",
    "host_control_producer_double_counting_nonoverlap_unavailable_reason",
    "host_control_compat_launch_pop_coverage_status",
    "host_control_compat_launch_pop_coverage_unavailable_reason",
    "actual_launch_control_dispatch_us",
    "actual_launch_api_body_us",
    "actual_launch_instrumentation_only_us",
    "actual_launch_visibility_kind",
    "actual_launch_unavailable_reason",
    "host_control_visibility_schema_version",
    "host_control_visibility_opt_in_flag",
    "selected_occurrence_id",
    "paper_valid_window_id",
    "paper_valid_window_unavailable_reason",
    "split_sum_check_status",
    "split_tolerance_us",
    "classification_basis",
    "classification_unavailable_reason",
    "runtime_or_framework_duration_us",
    "payload_enrichment_duration_us",
    "trace_serialization_duration_us",
    "mis_materialized_duration_us",
    "emulated_occurrence_id",
    "emulated_occurrence_id_unavailable_reason",
    "actual_raw_prev_event_id",
    "actual_raw_current_event_id",
    "actual_boundary_family",
    "counterpart_join_key",
    "counterpart_join_method",
    "counterpart_join_confidence",
    "counterpart_unavailable_reason",
    "comparable_actual_context_only",
    "affected_interval_id",
    "affected_interval_unavailable_reason",
    "candidate_subinterval_id",
    "candidate_subinterval_unavailable_reason",
    "interval_kind",
    "start_ts_us",
    "end_ts_us",
    "duration_us",
    "host_dispatch_interval_ids",
    "stream_order_interval_ids",
    "cuda_event_wait_edge_ids",
    "collective_wait_edge_ids",
    "host_sync_interval_ids",
    "rank_completion_context_id",
    "global_completion_context_id",
    "count_once_status",
    "exact_counterpart_status",
    "exact_counterpart_unavailable_reason",
    "double_counting_overlap_unavailable_reason",
}
_BOUNDARY_ORIGIN_TOP_LEVEL_PROMOTION_FIELDS = (
    _BOUNDARY_ORIGIN_FIELDS
    - _COMPONENT_STRICT_COUNTERPART_FIELDS
    - _COMPONENT_STRICT_ACTUAL_ENDPOINT_SIDECAR_ONLY_FIELDS
    - _CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_FIELDS
    - _JOINED_GEMM_STREAM_QUEUE_WAIT_ACTUAL_COUNTERPART_FIELDS
)

_CURRENT_BOUNDARY_ID_FIELDS = {
    "host_control_envelope_counterpart_key",
    "hostdelay_counterpart_key",
    "host_control_envelope_actual_row_id",
    "host_control_envelope_actual_interval_id",
    "host_control_envelope_actual_interval_kind",
    "host_control_envelope_prev_raw_event_id",
    "host_control_envelope_current_raw_event_id",
    "host_control_envelope_prev_api",
    "host_control_envelope_current_api",
    "host_control_envelope_rank",
    "host_control_envelope_stream_id",
    "host_control_envelope_host_dispatch_queue_id",
    "host_control_envelope_prev_raw_ordinal",
    "host_control_envelope_current_raw_ordinal",
    "host_control_boundary_occurrence_id",
    "actual_counterpart_id",
    "selected_occurrence_id",
    "actual_raw_prev_event_id",
    "actual_raw_current_event_id",
    "actual_boundary_family",
    "counterpart_join_key",
    "counterpart_join_method",
    "counterpart_join_confidence",
    "comparable_actual_context_only",
}

_HOST_CONTROL_VISIBILITY_OPT_IN_MARKERS = {
    "gemm_adjacent_hostdelay_schema_version",
    "gemm_adjacent_actual_counterpart_schema_version",
    "joined_gemm_stream_queue_wait_actual_counterpart_schema_version",
    "host_control_envelope_counterpart_schema_version",
    "host_control_visibility_schema_version",
    "host_control_boundary_counterpart_schema_version",
    "host_control_producer_visibility_schema_version",
    "launch_neighborhood_equivalence_schema_version",
}
_LAUNCH_NEIGHBORHOOD_EQUIVALENCE_EXPORT_FIELD_SET = set(
    LAUNCH_NEIGHBORHOOD_EQUIVALENCE_EXPORT_FIELDS
)


def _boundary_family(previous_api: str | None, current_api: str | None) -> str:
    apis = [api for api in (previous_api, current_api) if api is not None]
    return " -> ".join(apis)


def _gemm_adjacent_boundary_scope(
    *apis: str | None,
) -> tuple[bool, str | None, str | None]:
    present = [api for api in apis if api]
    target_gemm_api = next(
        (api for api in present if api in _GEMM_ADJACENT_HOSTDELAY_GEMM_APIS),
        None,
    )
    adjacent_api = next(
        (api for api in present if api in _GEMM_ADJACENT_HOSTDELAY_ADJACENT_APIS),
        None,
    )
    in_scope = target_gemm_api is not None and adjacent_api is not None
    return in_scope, target_gemm_api, adjacent_api


def _gemm_adjacent_shape_signature(event: TraceEvent | None) -> str | None:
    if event is None:
        return None
    return canonical_gemm_material_signature(event.extras)


def _gemm_adjacent_stream_resource_id(event: TraceEvent | None) -> str | None:
    if event is None:
        return None
    stream_id = event.extras.get("stream_id")
    if stream_id in (None, ""):
        return None
    return f"rank:{int(event.rank)}:stream:{stream_id}"


def _cuda_gemm_hostdispatch_target_api(
    *apis: str | None,
) -> str | None:
    for api in apis:
        if api in _CUDALAUNCH_GEMM_HOSTDISPATCH_TARGET_APIS:
            return api
    return None


def _cuda_gemm_hostdispatch_algorithm(event: TraceEvent | None) -> object | None:
    if event is None:
        return None
    if event.extras.get("algorithm") not in (None, ""):
        return event.extras.get("algorithm")
    if event.extras.get("algo") not in (None, ""):
        return event.extras.get("algo")
    return None


def _cuda_gemm_hostdispatch_material_signature(event: TraceEvent | None) -> object | None:
    if event is None:
        return None
    if is_gemm_material_api(event.api):
        return (
            canonical_gemm_material_signature(event.extras)
            or event.extras.get("material_signature")
        )
    return event.extras.get("material_signature") or _gemm_adjacent_shape_signature(event)


def _cuda_gemm_hostdispatch_strict_occurrence_key(
    *,
    rank: int,
    paper_valid_window_id: object | None,
    host_dispatch_queue_id: str | None,
    api_name: str,
    component_role: str,
    api_sequence_ordinal: object | None,
    host_queue_sequence_ordinal: object | None,
    stream_sequence_ordinal: object | None,
    material_signature: object | None,
    algorithm: object | None,
    boundary_family: str | None,
) -> str:
    material_text = (
        "unavailable" if material_signature in (None, "") else str(material_signature)
    )
    algorithm_text = "unavailable" if algorithm in (None, "") else str(algorithm)
    return "|".join(
        [
            f"rank:{int(rank)}",
            f"queue:{host_dispatch_queue_id or 'unavailable'}",
            f"api:{api_name or 'unknown'}",
            f"api_seq:{api_sequence_ordinal if api_sequence_ordinal is not None else 'unavailable'}",
            (
                "host_queue_seq:"
                f"{host_queue_sequence_ordinal if host_queue_sequence_ordinal is not None else 'unavailable'}"
            ),
            (
                "stream_seq:"
                f"{stream_sequence_ordinal if stream_sequence_ordinal is not None else 'unavailable'}"
            ),
            f"material:{material_text}",
            f"algorithm:{algorithm_text}",
            f"boundary:{boundary_family or 'unavailable'}",
        ]
    )


def _cuda_gemm_hostdispatch_material_without_embedded_algo(
    material_signature: object | None,
) -> object | None:
    if material_signature in (None, ""):
        return material_signature
    parts = [
        part
        for part in str(material_signature).split("|")
        if not part.startswith("algo=")
    ]
    return "|".join(parts) if parts else None


def _cuda_gemm_hostdispatch_boundary_target_side(
    boundary_family: str | None,
) -> str | None:
    if boundary_family in (None, ""):
        return None
    pieces = str(boundary_family).split(" -> ")
    targets = _CUDALAUNCH_GEMM_HOSTDISPATCH_TARGET_APIS
    if len(pieces) == 2:
        previous_api, current_api = pieces
        if previous_api in targets and current_api in targets:
            return "target_to_target"
        if current_api in targets:
            return "incoming_to_target"
        if previous_api in targets:
            return "target_to_outgoing"
    return str(boundary_family)


def _cuda_gemm_hostdispatch_projection_key(
    *,
    rank: int,
    host_dispatch_queue_id: str | None,
    api_name: str,
    api_sequence_ordinal: object | None,
    material_signature: object | None,
    algorithm: object | None,
    boundary_target_side: str | None = None,
) -> str:
    material_text = (
        "unavailable" if material_signature in (None, "") else str(material_signature)
    )
    algorithm_text = "unavailable" if algorithm in (None, "") else str(algorithm)
    parts = [
        f"rank:{int(rank)}",
        f"queue:{host_dispatch_queue_id or 'unavailable'}",
        f"api:{api_name or 'unknown'}",
        f"api_seq:{api_sequence_ordinal if api_sequence_ordinal is not None else 'unavailable'}",
        f"material:{material_text}",
        f"algorithm:{algorithm_text}",
    ]
    if boundary_target_side is not None:
        parts.append(f"boundary_target_side:{boundary_target_side}")
    return "|".join(parts)


def _cuda_gemm_hostdispatch_strict_occurrence_common_fields(
    *,
    rank: int,
    source_side: str,
    count_basis_side: str,
    api_name: str,
    component_role: str,
    paper_valid_window_id: object | None,
    host_dispatch_queue_id: str | None,
    stream_id: object | None,
    api_sequence_ordinal: object | None,
    host_queue_sequence_ordinal: object | None,
    stream_sequence_ordinal: object | None,
    material_signature: object | None,
    algorithm: object | None,
    gemm_shape_signature: object | None,
    boundary_family: str | None,
    predicted_count_once_group_id: object | None,
    actual_count_once_group_id: object | None,
    key_completeness_status: str,
) -> dict[str, object]:
    predicted_stream_resource_id = (
        None if stream_id in (None, "") else f"rank:{int(rank)}:stream:{stream_id}"
    )
    common_key = _cuda_gemm_hostdispatch_strict_occurrence_key(
        rank=rank,
        paper_valid_window_id=paper_valid_window_id,
        host_dispatch_queue_id=host_dispatch_queue_id,
        api_name=api_name,
        component_role=component_role,
        api_sequence_ordinal=api_sequence_ordinal,
        host_queue_sequence_ordinal=host_queue_sequence_ordinal,
        stream_sequence_ordinal=stream_sequence_ordinal,
        material_signature=material_signature,
        algorithm=algorithm,
        boundary_family=boundary_family,
    )
    material_without_algo = _cuda_gemm_hostdispatch_material_without_embedded_algo(
        material_signature
    )
    boundary_target_side = _cuda_gemm_hostdispatch_boundary_target_side(boundary_family)
    endpoint_identity_projection_key = _cuda_gemm_hostdispatch_projection_key(
        rank=rank,
        host_dispatch_queue_id=host_dispatch_queue_id,
        api_name=api_name,
        api_sequence_ordinal=api_sequence_ordinal,
        material_signature=material_without_algo,
        algorithm=algorithm,
    )
    boundary_target_side_projection_key = _cuda_gemm_hostdispatch_projection_key(
        rank=rank,
        host_dispatch_queue_id=host_dispatch_queue_id,
        api_name=api_name,
        api_sequence_ordinal=api_sequence_ordinal,
        material_signature=material_without_algo,
        algorithm=algorithm,
        boundary_target_side=boundary_target_side,
    )
    return {
        "cuda_gemm_hostdispatch_strict_occurrence_gap_schema_version": (
            _CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_SCHEMA_VERSION
        ),
        "cuda_gemm_hostdispatch_strict_occurrence_gap_opt_in_flag": True,
        "cuda_gemm_hostdispatch_strict_occurrence_gap_source_side": source_side,
        "strict_occurrence_common_basis_key": common_key,
        "strict_occurrence_material_without_embedded_algo": material_without_algo,
        "strict_occurrence_boundary_target_side": boundary_target_side,
        "strict_occurrence_endpoint_identity_projection_key": (
            endpoint_identity_projection_key
        ),
        "strict_occurrence_boundary_target_side_projection_key": (
            boundary_target_side_projection_key
        ),
        "strict_occurrence_projection_keys_status": (
            "diagnostic_only_projection_not_strict_join_key"
        ),
        "strict_occurrence_projection_keys_basis": (
            "field_ablation_followup_endpoint_identity_and_boundary_target_side;"
            "drops_host_queue_seq_stream_seq_and_exact_boundary_for_blocker_review"
        ),
        "strict_occurrence_projection_keys_repair_ready": False,
        "strict_occurrence_key_parts": {
            "rank": int(rank),
            "paper_valid_window_id": paper_valid_window_id,
            "host_dispatch_queue_id": host_dispatch_queue_id,
            "api_family": api_name,
            "component_role": component_role,
            "api_sequence_ordinal_in_window": api_sequence_ordinal,
            "host_queue_sequence_ordinal_in_window": host_queue_sequence_ordinal,
            "stream_sequence_ordinal_in_window": stream_sequence_ordinal,
            "material_signature": material_signature,
            "algorithm": algorithm,
            "gemm_shape_signature": gemm_shape_signature,
            "boundary_family": boundary_family,
        },
        "strict_occurrence_count_basis_side": count_basis_side,
        "paper_valid_window_id": paper_valid_window_id,
        "rank": int(rank),
        "host_dispatch_queue_id": host_dispatch_queue_id,
        "api_family": api_name,
        "component_role": component_role,
        "api_sequence_ordinal_in_window": api_sequence_ordinal,
        "host_queue_sequence_ordinal_in_window": host_queue_sequence_ordinal,
        "stream_sequence_ordinal_in_window": stream_sequence_ordinal,
        "material_signature": material_signature,
        "algorithm": algorithm,
        "gemm_shape_signature": gemm_shape_signature,
        "boundary_family": boundary_family,
        "key_completeness_status": key_completeness_status,
        "actual_mechanical_dispatch_split_status": "unavailable",
        "actual_mechanical_dispatch_split_basis": (
            "metadata_only_no_non_perturbing_actual_mechanical_dispatch_split"
        ),
        "actual_control_dispatch_us": None,
        "actual_api_body_us": None,
        "actual_instrumentation_only_us": None,
        "actual_endpoint_timestamps_used_as_dispatch_split": False,
        "actual_host_duration_used_as_dispatch_split": False,
        "actual_runtime_used_as_dispatch_split": False,
        "actual_mechanical_dispatch_split_unavailable_reason": (
            "endpoint timestamps, host_duration_us, and observed runtime are "
            "context only and are not a strict mechanical dispatch split"
        ),
        "predicted_stream_resource_id": predicted_stream_resource_id,
        "actual_stream_resource_id": None,
        "stream_namespace_basis": (
            "rank_scoped_predicted_raw_stream_id"
            if predicted_stream_resource_id is not None
            else "unavailable_missing_stream_id"
        ),
        "stream_alignment_status": "predicted_only_actual_alignment_unavailable",
        "exact_stream_identity_proven": False,
        "default_stream_equivalence_reviewed": False,
        "predicted_count_once_group_id": predicted_count_once_group_id,
        "actual_count_once_group_id": actual_count_once_group_id,
        "count_once_status": (
            "metadata_only_count_once_group_not_strict_nonoverlap_proof"
            if predicted_count_once_group_id is not None
            or actual_count_once_group_id is not None
            else "unavailable"
        ),
        "nonoverlap_status": "unavailable",
        "provider_runtime_overlap_status": "unavailable",
        "host_dispatch_overlap_status": "unavailable",
        "hostDelay_overlap_status": "unavailable",
        "stream_wait_overlap_status": "unavailable",
        "wait_map_blocking_overlap_status": "unavailable",
        "hostdispatch_producer_visibility_status": "unavailable",
        "hostdispatch_producer_visibility_basis": (
            "no_non_perturbing_producer_visibility_split_available"
        ),
        "paper_visible_host_dispatch_us": None,
        "instrumentation_only_host_dispatch_us": None,
        "unresolved_mixed_host_dispatch_us": None,
        "producer_side_safe_for_repair_design": False,
        "wait_map_safety_status": "unavailable",
        "predicted_wait_map_edge_ids": [],
        "actual_wait_release_source_status": "unavailable",
        "dependency_release_timing_preserved": False,
        "cuda_event_wait_safety_status": "unavailable",
        "collective_wait_safety_status": "unavailable",
        "stream_queue_wait_safety_status": "unavailable",
        "strict_occurrence_join_ready": False,
        "strict_actual_timing_or_mechanical_split_ready": False,
        "strict_apples_to_apples_delta_ready": False,
        "repair_ready": False,
        "safe_to_use_as_repair_evidence": False,
        "safe_to_use_as_subtraction_delta": False,
        "safe_to_use_for_runtime_substitution": False,
        "safe_to_use_for_endpoint_timestamp_substitution": False,
    }


def _has_host_control_visibility_opt_in(*events: TraceEvent | None) -> bool:
    for event in events:
        if event is None:
            continue
        extras = event.extras
        if extras.get("host_control_visibility_opt_in_flag") is True:
            return True
        if _metadata_has_launch_neighborhood_equivalence(extras):
            return True
        marker_fields = _HOST_CONTROL_VISIBILITY_OPT_IN_MARKERS.intersection(extras)
        if any(extras.get(field) not in (None, "") for field in marker_fields):
            return True
    return False


def _metadata_has_launch_neighborhood_equivalence(extras: Mapping[str, object]) -> bool:
    if extras.get("launch_neighborhood_equivalence_opt_in_flag") is True:
        return True
    fields = _LAUNCH_NEIGHBORHOOD_EQUIVALENCE_EXPORT_FIELD_SET.intersection(extras)
    return any(extras.get(field) not in (None, "") for field in fields)


def _paper_valid_window_membership(
    *,
    rank: int,
    boundary_ts: int,
    step_window: tuple[int, int] | None,
    fidelity_window,
) -> dict[str, object]:
    if step_window is None:
        return {
            "in_paper_valid_window": False,
            "window_id": None,
            "window_source": None,
            "start_ts": None,
            "end_ts": None,
            "is_paper_valid_step_window": False,
            "membership_basis": "collate_step_window",
            "unavailable_reason": "no_paper_valid_step_window_for_rank",
        }
    start_ts, end_ts = int(step_window[0]), int(step_window[1])
    in_window = start_ts <= int(boundary_ts) <= end_ts
    window_source = getattr(fidelity_window, "source", None) or "step_windows"
    is_paper_valid = bool(getattr(fidelity_window, "is_paper_valid_step_window", True))
    return {
        "in_paper_valid_window": bool(in_window and is_paper_valid),
        "window_id": f"rank{int(rank)}:step_window",
        "window_source": str(window_source),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "is_paper_valid_step_window": bool(is_paper_valid),
        "membership_basis": "collate_step_window",
        "unavailable_reason": None if in_window and is_paper_valid else "boundary_outside_paper_valid_window",
    }


def _host_delay_boundary_origin_provenance(
    previous_event: TraceEvent | None,
    current_event: TraceEvent,
    *,
    observed_gap_us: int | None = None,
    host_dispatch_queue_id: str | None = None,
    previous_materialized_event_id: str | None = None,
    previous_materialized_api: str | None = None,
    current_materialized_event_id: str | None = None,
    current_materialized_api: str | None = None,
    paper_valid_window_membership: Mapping[str, object] | None = None,
    strict_occurrence_sequence_ordinals: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Forward additive boundary-origin evidence from raw traces.

    This does not change hostDelay materialization or duration semantics. Old
    traces simply lack these optional fields; future capture instrumentation may
    emit them so downstream audits can distinguish paper-visible CPU/framework
    work from fake/wrapper/instrumentation-only elapsed time. If both sides of a
    boundary provide contradictory evidence, keep source-specific values and mark
    the flattened boundary origin as mixed/unresolved rather than silently
    preferring one side.
    """

    boundary_apis: list[str] = []
    if previous_event is not None:
        boundary_apis.append(previous_event.api)
    boundary_apis.append(current_event.api)
    raw_prev_api = previous_event.api if previous_event else None
    raw_current_api = current_event.api
    semantic_predecessor_boundary_family = _boundary_family(
        previous_materialized_api,
        current_materialized_api,
    )
    semantic_predecessor_namespace_basis = (
        "materialized_semantic_predecessor_after_control_query_filter"
        if semantic_predecessor_boundary_family
        and semantic_predecessor_boundary_family != _boundary_family(raw_prev_api, raw_current_api)
        else "raw_immediate_boundary"
    )
    paper_window_id = None
    if paper_valid_window_membership:
        paper_window_id = paper_valid_window_membership.get("window_id")
    gemm_adjacent_in_scope, target_gemm_api, adjacent_api = _gemm_adjacent_boundary_scope(
        raw_prev_api,
        raw_current_api,
        previous_materialized_api,
        current_materialized_api,
    )
    gemm_adjacent_opt_in = (
        _gemm_adjacent_hostdelay_boundary_metadata_diagnostics_enabled()
        and gemm_adjacent_in_scope
    )
    component_strict_opt_in = _component_strict_counterpart_metadata_diagnostics_enabled()
    cuda_gemm_strict_occurrence_gap_opt_in = (
        _cuda_gemm_hostdispatch_strict_occurrence_gap_diagnostics_enabled()
    )
    launch_neighborhood_equivalence_opt_in = (
        _launch_neighborhood_equivalence_diagnostics_enabled()
        or _metadata_has_launch_neighborhood_equivalence(current_event.extras)
        or (
            previous_event is not None
            and _metadata_has_launch_neighborhood_equivalence(previous_event.extras)
        )
    )
    host_control_visibility_opt_in = (
        _host_control_envelope_counterpart_diagnostics_enabled()
        or launch_neighborhood_equivalence_opt_in
        or gemm_adjacent_opt_in
        or _has_host_control_visibility_opt_in(
            previous_event,
            current_event,
        )
    )
    raw_prev_end_ts_us, raw_prev_end_ts_source = (
        _event_end_ts_context(previous_event) if previous_event else (None, "unavailable")
    )
    raw_current_end_ts_us, raw_current_end_ts_source = _event_end_ts_context(current_event)
    provenance: dict[str, object] = {
        "raw_prev_event_id": _event_id(previous_event) if previous_event else None,
        "raw_prev_api": raw_prev_api,
        "raw_prev_ts_us": int(previous_event.ts) if previous_event else None,
        "raw_prev_end_ts_us": raw_prev_end_ts_us,
        "raw_prev_end_ts_source": raw_prev_end_ts_source,
        "raw_prev_host_duration_us": (
            _normalized_float(previous_event.extras.get("host_duration_us")) if previous_event else None
        ),
        "raw_current_event_id": _event_id(current_event),
        "raw_current_api": raw_current_api,
        "raw_current_ts_us": int(current_event.ts),
        "raw_current_end_ts_us": raw_current_end_ts_us,
        "raw_current_end_ts_source": raw_current_end_ts_source,
        "raw_current_host_duration_us": _normalized_float(current_event.extras.get("host_duration_us")),
        "previous_materialized_event_id": previous_materialized_event_id,
        "previous_materialized_api": previous_materialized_api,
        "current_materialized_event_id": current_materialized_event_id,
        "current_materialized_api": current_materialized_api,
        "raw_boundary_family": _boundary_family(raw_prev_api, raw_current_api),
        "materialized_boundary_family": _boundary_family(
            previous_materialized_api,
            current_materialized_api,
        ),
        "semantic_predecessor_boundary_family": semantic_predecessor_boundary_family,
        "semantic_predecessor_boundary_namespace_basis": (
            semantic_predecessor_namespace_basis
        ),
        "semantic_predecessor_previous_api": previous_materialized_api,
        "semantic_predecessor_current_api": current_materialized_api,
        "boundary_origin_prev_api": previous_event.api if previous_event else None,
        "boundary_origin_current_api": current_event.api,
        "boundary_origin_family": " -> ".join(boundary_apis),
        "paper_valid_window_membership": dict(paper_valid_window_membership or {}),
    }
    launch_config_pop_entry_adjustment_us = (
        _launch_config_pop_entry_hostdelay_adjustment_us(current_event)
    )
    if launch_config_pop_entry_adjustment_us > 0:
        bounded_adjustment_us = min(
            int(max(launch_config_pop_entry_adjustment_us, 0)),
            int(max(observed_gap_us, 0)),
        )
        provenance.update(
            {
                "launch_config_pop_entry_hostdelay_boundary_enabled": True,
                "launch_config_pop_entry_hostdelay_boundary_basis": (
                    "measure_mode_wrapper_entry_boundary_for_internal_launch_config_pop"
                ),
                "launch_config_pop_entry_hostdelay_boundary_env_flags": list(
                    _LAUNCH_CONFIG_POP_ENTRY_HOSTDELAY_BOUNDARY_ENV_KEYS
                ),
                "launch_config_pop_entry_raw_trace_ts_us": int(current_event.ts),
                "launch_config_pop_entry_raw_host_duration_us": (
                    _normalized_float(current_event.extras.get("host_duration_us"))
                ),
                "launch_config_pop_entry_excluded_from_prepop_hostdelay_us": (
                    bounded_adjustment_us
                ),
                "launch_config_pop_entry_hostdelay_boundary_status": (
                    "enabled_opt_in_pop_wrapper_body_not_counted_as_prepop_hostdelay"
                ),
            }
        )
    if (
        _hostdelay_occurrence_metadata_export_enabled()
        or _hostdelay_occurrence_join_metadata_export_enabled()
        or _collective_event_polling_boundary_metadata_export_enabled()
        or _event_polling_boundary_metadata_export_enabled()
        or _boundary_origin_subregion_proof_metadata_export_enabled()
        or _strict_subregion_extent_source_metadata_export_enabled()
    ):
        provenance.update(
            _hostdelay_occurrence_boundary_metadata(
                previous_event,
                current_event,
                observed_gap_us=observed_gap_us,
                host_dispatch_queue_id=host_dispatch_queue_id,
                previous_materialized_event_id=previous_materialized_event_id,
                previous_materialized_api=previous_materialized_api,
                current_materialized_event_id=current_materialized_event_id,
                current_materialized_api=current_materialized_api,
                paper_valid_window_membership=paper_valid_window_membership,
            )
        )
    if component_strict_opt_in:
        previous_ordinal = int(previous_event.ordinal) if previous_event else None
        current_ordinal = int(current_event.ordinal)
        previous_key = (
            "leading" if previous_ordinal is None else f"raw_ordinal:{previous_ordinal}"
        )
        stable_boundary_key = (
            f"rank:{int(current_event.rank)}:hostdelay_boundary:"
            f"{previous_key}->raw_ordinal:{current_ordinal}"
        )
        paper_valid = dict(paper_valid_window_membership or {})
        stream_id = current_event.extras.get("stream_id")
        stream_resource_id = (
            f"rank:{int(current_event.rank)}:stream:{stream_id}"
            if stream_id not in (None, "")
            else None
        )
        provenance.update(
            {
                "component_strict_counterpart_schema_version": (
                    _COMPONENT_STRICT_COUNTERPART_SCHEMA_VERSION
                ),
                "component_strict_counterpart_opt_in_flag": True,
                "source_side": "predicted_component_metadata",
                "stable_predicted_component_row_id": stable_boundary_key,
                "stable_predicted_interval_row_id": None,
                "stable_predicted_edge_row_id": None,
                "stable_predicted_count_once_group_id": (
                    f"predicted_hostdelay_boundary:{stable_boundary_key}"
                ),
                "predicted_row_identity_basis": (
                    "rank-local hostDelay boundary previous/current raw ordinals"
                ),
                "component_kind": "host_control_interval",
                "predicted_replay_component": "materialized_hostDelay_boundary",
                "api_or_kernel_family": _boundary_family(raw_prev_api, raw_current_api),
                "api_or_kernel_family_role": "hostDelay_boundary",
                "paper_valid_window_id": paper_valid.get("window_id"),
                "paper_valid_window_membership_status": (
                    "available"
                    if paper_valid.get("in_paper_valid_window") is True
                    else "unavailable"
                ),
                "predicted_interval_duration_basis": (
                    "collate materialized hostDelay observed_gap_us"
                ),
                "predicted_interval_resource_kind": "host",
                "predicted_interval_resource_id": host_dispatch_queue_id,
                "predicted_interval_origin_status": (
                    "metadata_only_boundary_origin_not_strict_actual_counterpart"
                ),
                "predicted_interval_visibility_status": "producer_visibility_unavailable",
                "predicted_stream_id": stream_id,
                "predicted_stream_resource_id": stream_resource_id,
                "predicted_stream_namespace_basis": (
                    "rank_scoped_predicted_raw_stream_id"
                    if stream_resource_id is not None
                    else "not_applicable_or_unavailable"
                ),
                "actual_counterpart_join_status": (
                    "predicted_metadata_only_actual_join_deferred"
                ),
                "actual_counterpart_join_basis": (
                    "offline_component_strict_counterpart_join_required"
                ),
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
                    "predicted_only_actual_alignment_unavailable"
                ),
                "exact_stream_identity_proven": False,
                "count_once_status": "unavailable",
                "nonoverlap_status": "unavailable",
                "wait_map_safety_status": "unavailable",
                "producer_visibility_status": "unavailable",
                "paper_maya_tags": [
                    "predicted_hostdelay_boundary",
                    "metadata_only",
                    "no_repair",
                    "no_runtime_substitution",
                ],
                "repair_ready": False,
                "safe_to_use_as_repair_evidence": False,
                "safe_to_use_as_subtraction_delta": False,
            }
        )
    if cuda_gemm_strict_occurrence_gap_opt_in:
        target_api = _cuda_gemm_hostdispatch_target_api(
            raw_current_api,
            raw_prev_api,
            current_materialized_api,
            previous_materialized_api,
        )
        target_event = None
        if current_event.api == target_api:
            target_event = current_event
        elif previous_event is not None and previous_event.api == target_api:
            target_event = previous_event
        if target_api is not None and target_event is not None:
            sequence_ordinals = dict(
                (strict_occurrence_sequence_ordinals or {}).get(
                    _event_id(target_event),
                    {},
                )
            )
            paper_valid = dict(paper_valid_window_membership or {})
            paper_valid_window_id = paper_valid.get("window_id")
            predicted_row_id = (
                f"rank:{int(current_event.rank)}:strict_occurrence_gap_hostDelay:"
                f"{_event_id(previous_event) if previous_event else 'leading'}"
                f"->{_event_id(current_event)}"
            )
            material_signature = _cuda_gemm_hostdispatch_material_signature(target_event)
            algorithm = _cuda_gemm_hostdispatch_algorithm(target_event)
            provenance.update(
                {
                    **_cuda_gemm_hostdispatch_strict_occurrence_common_fields(
                        rank=current_event.rank,
                        source_side="predicted_hostDelay_boundary_metadata",
                        count_basis_side="predicted_hostDelay_boundary",
                        api_name=target_api,
                        component_role="hostDelay",
                        paper_valid_window_id=paper_valid_window_id,
                        host_dispatch_queue_id=host_dispatch_queue_id,
                        stream_id=target_event.extras.get("stream_id"),
                        api_sequence_ordinal=sequence_ordinals.get(
                            "api_sequence_ordinal_in_window"
                        ),
                        host_queue_sequence_ordinal=sequence_ordinals.get(
                            "host_queue_sequence_ordinal_in_window"
                        ),
                        stream_sequence_ordinal=sequence_ordinals.get(
                            "stream_sequence_ordinal_in_window"
                        ),
                        material_signature=material_signature,
                        algorithm=algorithm,
                        gemm_shape_signature=_gemm_adjacent_shape_signature(target_event),
                        boundary_family=_boundary_family(raw_prev_api, raw_current_api),
                        predicted_count_once_group_id=(
                            f"predicted_hostDelay_boundary:{predicted_row_id}"
                        ),
                        actual_count_once_group_id=None,
                        key_completeness_status=(
                            "predicted_hostDelay_key_parts_available_actual_join_deferred"
                        ),
                    ),
                    "cuda_gemm_hostdispatch_strict_occurrence_gap_predicted_row_id": (
                        predicted_row_id
                    ),
                }
            )
    if host_control_visibility_opt_in:
        previous_ordinal = int(previous_event.ordinal) if previous_event else None
        current_ordinal = int(current_event.ordinal)
        previous_key = (
            "leading" if previous_ordinal is None else f"raw_ordinal:{previous_ordinal}"
        )
        stable_counterpart_key = (
            f"rank:{int(current_event.rank)}:host_control_boundary:"
            f"{previous_key}->raw_ordinal:{current_ordinal}"
        )
        raw_prev_event_id = _event_id(previous_event) if previous_event else None
        raw_current_event_id = _event_id(current_event)
        provenance.update(
            {
                "host_control_envelope_counterpart_schema_version": (
                    _HOST_CONTROL_ENVELOPE_COUNTERPART_SCHEMA_VERSION
                ),
                "host_control_envelope_counterpart_opt_in_flag": True,
                "host_control_envelope_counterpart_key": stable_counterpart_key,
                "hostdelay_counterpart_key": stable_counterpart_key,
                "selected_occurrence_id": stable_counterpart_key,
                "host_control_boundary_occurrence_id": stable_counterpart_key,
                "host_control_envelope_materialized_interval_id": (
                    f"{stable_counterpart_key}:materialized_hostDelay"
                ),
                "host_control_envelope_prev_raw_event_id": raw_prev_event_id,
                "host_control_envelope_current_raw_event_id": raw_current_event_id,
                "host_control_envelope_prev_api": raw_prev_api,
                "host_control_envelope_current_api": raw_current_api,
                "host_control_envelope_rank": int(current_event.rank),
                "host_control_envelope_stream_id": current_event.extras.get("stream_id"),
                "host_control_envelope_host_dispatch_queue_id": host_dispatch_queue_id,
                "host_control_envelope_paper_valid_window_id": (
                    str(paper_window_id) if paper_window_id else None
                ),
                "host_control_envelope_prev_raw_ordinal": previous_ordinal,
                "host_control_envelope_current_raw_ordinal": current_ordinal,
                "host_control_envelope_timestamp_basis": (
                    "raw_prev_end_ts_to_raw_current_ts_materialized_by_collate"
                ),
                "host_control_envelope_visibility_basis_status": (
                    "structural_metadata_only_no_mechanical_visibility_split"
                ),
                "host_control_envelope_visibility_kind": "mixed_or_unresolved",
                "host_control_envelope_replay_overlap_status": "unavailable",
                "host_control_envelope_replay_overlap_unavailable_reason": (
                    "requires_replay_interval_export_with_wait_map_and_count_once_context"
                ),
                "safe_to_use_as_subtraction_delta": False,
                "safe_to_use_as_repair_evidence": False,
                "safe_to_use_as_subtraction_delta_reason": "requires_aggregate_exact_counterpart_join",
                "safe_to_use_as_repair_evidence_reason": (
                    "requires_mechanically_measured_paper_invisible_subregion_and_wait_map_safety"
                ),
                "double_counting_overlap_status": "unavailable",
                "wait_map_safety_status": "unavailable",
                "wait_map_non_overlap_unavailable_reason": (
                    "requires_replay_wait_edge_non_overlap_ledger_not_exported_by_collate"
                ),
                "replay_wait_edge_id": None,
                "predecessor_event_id": None,
                "predecessor_api": None,
                "release_us": None,
                "release_reason": None,
            }
        )
        if launch_neighborhood_equivalence_opt_in:
            provenance.update(
                build_launch_neighborhood_equivalence_metadata(
                    rank=int(current_event.rank),
                    previous_api=raw_prev_api,
                    current_api=raw_current_api,
                    previous_raw_event_id=raw_prev_event_id,
                    current_raw_event_id=raw_current_event_id,
                    previous_raw_ordinal=previous_ordinal,
                    current_raw_ordinal=current_ordinal,
                    host_dispatch_queue_id=host_dispatch_queue_id,
                    stream_id=current_event.extras.get("stream_id"),
                    paper_valid_window_id=str(paper_window_id) if paper_window_id else None,
                    role="predicted_materialized_hostdelay_boundary",
                )
            )
        if gemm_adjacent_opt_in:
            previous_stream_id = previous_event.extras.get("stream_id") if previous_event else None
            current_stream_id = current_event.extras.get("stream_id")
            material_event = (
                current_event
                if current_event.api in _GEMM_ADJACENT_HOSTDELAY_GEMM_APIS
                else previous_event
                if previous_event is not None
                and previous_event.api in _GEMM_ADJACENT_HOSTDELAY_GEMM_APIS
                else None
            )
            provenance.update(
                {
                    "gemm_adjacent_hostdelay_schema_version": (
                        _GEMM_ADJACENT_HOSTDELAY_BOUNDARY_SCHEMA_VERSION
                    ),
                    "gemm_adjacent_hostdelay_opt_in_flag": True,
                    "gemm_adjacent_source_side": (
                        "predicted_materialized_hostdelay_boundary"
                    ),
                    "gemm_adjacent_stable_boundary_row_id": stable_counterpart_key,
                    "gemm_adjacent_rank": int(current_event.rank),
                    "gemm_adjacent_paper_valid_window_id": (
                        str(paper_window_id) if paper_window_id else None
                    ),
                    "gemm_adjacent_paper_valid_window_membership_status": (
                        "available"
                        if paper_valid_window_membership
                        else "unavailable"
                    ),
                    "gemm_adjacent_hostdelay_source": None,
                    "gemm_adjacent_observed_gap_us": (
                        None
                        if observed_gap_us is None
                        else float(max(int(observed_gap_us), 0))
                    ),
                    "gemm_adjacent_hostdelay_duration_us": (
                        None
                        if observed_gap_us is None
                        else float(max(int(observed_gap_us), 0))
                    ),
                    "gemm_adjacent_host_dispatch_queue_id": host_dispatch_queue_id,
                    "gemm_adjacent_host_machine_id": _host_machine_id(current_event),
                    "gemm_adjacent_boundary_direction": (
                        "raw_or_materialized_adjacent_boundary"
                    ),
                    "gemm_adjacent_target_gemm_api": target_gemm_api,
                    "gemm_adjacent_adjacent_api": adjacent_api,
                    "gemm_adjacent_boundary_family_in_design_scope": True,
                    "gemm_adjacent_previous_stream_id": previous_stream_id,
                    "gemm_adjacent_previous_stream_resource_id": (
                        _gemm_adjacent_stream_resource_id(previous_event)
                    ),
                    "gemm_adjacent_current_stream_id": current_stream_id,
                    "gemm_adjacent_current_stream_resource_id": (
                        _gemm_adjacent_stream_resource_id(current_event)
                    ),
                    "gemm_adjacent_stream_namespace_basis": (
                        "predicted_collate_raw_stream_id_rank_local"
                    ),
                    "gemm_adjacent_stream_namespace_alignment_status": (
                        "predicted_only_actual_alignment_unavailable"
                    ),
                    "gemm_adjacent_material_signature": (
                        _cuda_gemm_hostdispatch_material_signature(material_event)
                    ),
                    "gemm_adjacent_algorithm": (
                        None
                        if material_event is None
                        else material_event.extras.get("algorithm")
                        if material_event.extras.get("algorithm") not in (None, "")
                        else material_event.extras.get("algo")
                    ),
                    "gemm_adjacent_gemm_shape_signature": (
                        _gemm_adjacent_shape_signature(material_event)
                    ),
                    "gemm_adjacent_gemm_material_metadata_status": (
                        "available"
                        if _gemm_adjacent_shape_signature(material_event) is not None
                        else "unavailable"
                    ),
                    "gemm_adjacent_actual_counterpart_row_id": None,
                    "gemm_adjacent_actual_counterpart_join_key": None,
                    "gemm_adjacent_actual_counterpart_join_status": (
                        "predicted_metadata_only_actual_join_deferred"
                    ),
                    "gemm_adjacent_actual_counterpart_join_basis": (
                        "offline_actual_endpoint_provenance_join_required"
                    ),
                    "gemm_adjacent_actual_counterpart_join_confidence": (
                        "unavailable"
                    ),
                    "gemm_adjacent_actual_counterpart_unavailable_reason": (
                        "actual GEMM-adjacent endpoint counterpart not joined by collate"
                    ),
                    "gemm_adjacent_actual_timing_status": "unavailable",
                    "gemm_adjacent_actual_start_us": None,
                    "gemm_adjacent_actual_end_us": None,
                    "gemm_adjacent_actual_duration_us": None,
                    "gemm_adjacent_actual_timing_basis": (
                        "unavailable_no_strict_actual_counterpart_timing"
                    ),
                    "gemm_adjacent_actual_endpoint_timestamps_used_as_strict_timing": False,
                    "gemm_adjacent_actual_runtime_direct_substitution": False,
                    "gemm_adjacent_producer_visibility_schema_version": (
                        "gemm_adjacent_hostdelay_producer_visibility_v1"
                    ),
                    "gemm_adjacent_producer_visibility_status": (
                        "structural_unavailable"
                    ),
                    "gemm_adjacent_producer_visibility_basis": (
                        "collate_predicted_boundary_metadata_reuses_host_control_counterpart_surface"
                    ),
                    "gemm_adjacent_producer_visibility_unavailable_reason": (
                        "requires opt-in actual producer structural visibility rows and "
                        "future non-perturbing mechanical split"
                    ),
                    "gemm_adjacent_paper_visible_host_duration_us": None,
                    "gemm_adjacent_instrumentation_only_duration_us": None,
                    "gemm_adjacent_unresolved_mixed_duration_us": None,
                    "gemm_adjacent_split_sum_check_status": "unavailable",
                    "gemm_adjacent_predicted_count_once_group_id": (
                        f"predicted_hostdelay_boundary:{stable_counterpart_key}"
                    ),
                    "gemm_adjacent_predicted_count_once_interval_id": None,
                    "gemm_adjacent_actual_count_once_group_id": None,
                    "gemm_adjacent_actual_count_once_interval_id": None,
                    "gemm_adjacent_count_once_status": "unavailable",
                    "gemm_adjacent_count_once_non_overlap_status": "unavailable",
                    "gemm_adjacent_count_once_non_overlap_unavailable_reason": (
                        "requires replay wait-map, host queue, stream FIFO, and rank envelope ledger"
                    ),
                    "gemm_adjacent_double_counting_overlap_status": "unavailable",
                    "gemm_adjacent_double_counting_overlap_unavailable_reason": (
                        "requires future count-once non-overlap ledger"
                    ),
                    "gemm_adjacent_wait_map_safety_status": "unavailable",
                    "gemm_adjacent_wait_map_safety_unavailable_reason": (
                        "requires future wait-map non-overlap ledger; endpoint timestamps are not wait releases"
                    ),
                    "gemm_adjacent_stream_fifo_nonoverlap_status": "unavailable",
                    "gemm_adjacent_host_queue_nonoverlap_status": "unavailable",
                    "gemm_adjacent_rank_envelope_nonoverlap_status": "unavailable",
                    "gemm_adjacent_strict_nonoverlap_proof_basis": None,
                    "gemm_adjacent_repair_ready": False,
                    "gemm_adjacent_safe_to_use_as_repair_evidence": False,
                    "gemm_adjacent_safe_to_use_as_subtraction_delta": False,
                }
            )
    if observed_gap_us is not None and host_dispatch_queue_id is not None:
        provenance.update(
            _safe_actual_boundary_metadata(
                previous_event,
                current_event,
                observed_gap_us=observed_gap_us,
                host_dispatch_queue_id=host_dispatch_queue_id,
                actual_counterpart_window_id=str(paper_window_id) if paper_window_id else None,
                include_visibility_aliases=host_control_visibility_opt_in,
            )
        )
    side_fields: dict[str, dict[str, object]] = {}
    for source_name, event in (("prev", previous_event), ("current", current_event)):
        if event is None:
            continue
        matching_fields = _BOUNDARY_ORIGIN_FIELDS.intersection(event.extras)
        values = {field: event.extras[field] for field in matching_fields}
        if values:
            side_fields[source_name] = values
    if side_fields:
        if "prev" in side_fields:
            provenance["boundary_origin_prev_fields"] = side_fields["prev"]
        if "current" in side_fields:
            provenance["boundary_origin_current_fields"] = side_fields["current"]

    field_sources: dict[str, list[str]] = {}
    conflicting_fields: dict[str, dict[str, object]] = {}
    if side_fields:
        promotion_fields: set[str] = set()
        for values in side_fields.values():
            promotion_fields.update(
                _BOUNDARY_ORIGIN_TOP_LEVEL_PROMOTION_FIELDS.intersection(values)
            )
        for field in promotion_fields:
            present = {side: values[field] for side, values in side_fields.items() if field in values}
            if not present:
                continue
            if field in _CURRENT_BOUNDARY_ID_FIELDS and "current" in present:
                current_value = present["current"]
                provenance[field] = current_value
                if all(value == current_value for value in present.values()):
                    field_sources[field] = sorted(present)
                else:
                    field_sources[field] = ["current"]
                    if len(present) > 1:
                        conflicting_fields[field] = present
                continue
            if len(present) == 1:
                side, value = next(iter(present.items()))
                provenance[field] = value
                field_sources[field] = [side]
                continue
            values = list(present.values())
            if values[0] == values[1]:
                provenance[field] = values[0]
                field_sources[field] = sorted(present)
                continue
            conflicting_fields[field] = present
            if field == "boundary_origin_kind":
                provenance[field] = "mixed_or_unresolved"
                field_sources[field] = sorted(present)
    if paper_window_id:
        provenance["actual_counterpart_window_id"] = str(paper_window_id)
        provenance["actual_counterpart_window_unavailable_reason"] = None
        field_sources["actual_counterpart_window_id"] = ["collate"]
        field_sources["actual_counterpart_window_unavailable_reason"] = ["collate"]
        if host_control_visibility_opt_in:
            provenance["paper_valid_window_id"] = str(paper_window_id)
            provenance["paper_valid_window_unavailable_reason"] = None
            provenance["actual_paper_valid_window_id"] = str(paper_window_id)
            provenance["actual_paper_valid_window_unavailable_reason"] = None
            field_sources["paper_valid_window_id"] = ["collate"]
            field_sources["paper_valid_window_unavailable_reason"] = ["collate"]
            field_sources["actual_paper_valid_window_id"] = ["collate"]
            field_sources["actual_paper_valid_window_unavailable_reason"] = ["collate"]
    if host_control_visibility_opt_in:
        selected_occurrence_id = provenance.get("selected_occurrence_id") or provenance.get(
            "host_control_boundary_occurrence_id"
        )
        if selected_occurrence_id not in (None, ""):
            provenance["selected_occurrence_id"] = str(selected_occurrence_id)
            provenance.setdefault("counterpart_join_key", str(selected_occurrence_id))
        if observed_gap_us is not None:
            interval_id = provenance.get("affected_interval_id")
            if interval_id in (None, "") and selected_occurrence_id not in (None, ""):
                interval_id = f"{selected_occurrence_id}:endpoint_gap"
                provenance["affected_interval_id"] = interval_id
                provenance["affected_interval_unavailable_reason"] = None
                field_sources["affected_interval_id"] = ["collate"]
                field_sources["affected_interval_unavailable_reason"] = ["collate"]
            if provenance.get("start_ts_us") in (None, ""):
                provenance["start_ts_us"] = raw_prev_end_ts_us
            if provenance.get("end_ts_us") in (None, ""):
                provenance["end_ts_us"] = int(current_event.ts)
            if provenance.get("duration_us") in (None, ""):
                provenance["duration_us"] = float(max(int(observed_gap_us), 0))
            provenance.setdefault(
                "host_control_envelope_counterpart_interval_id",
                interval_id,
            )
        provenance.setdefault("interval_kind", "actual_endpoint_gap_context_only")
        provenance.setdefault("candidate_subinterval_id", None)
        provenance.setdefault(
            "candidate_subinterval_unavailable_reason",
            "mechanical_visibility_split_not_measured_to_preserve_wrapper_host_timing",
        )
        provenance.setdefault("host_dispatch_interval_ids", [])
        provenance.setdefault("stream_order_interval_ids", [])
        provenance.setdefault("cuda_event_wait_edge_ids", [])
        provenance.setdefault("collective_wait_edge_ids", [])
        provenance.setdefault("host_sync_interval_ids", [])
        provenance.setdefault("rank_completion_context_id", None)
        provenance.setdefault("global_completion_context_id", None)
        provenance.setdefault("count_once_status", "unavailable")
    if field_sources:
        provenance["boundary_origin_field_sources"] = field_sources
    if conflicting_fields:
        provenance["boundary_origin_conflicting_fields"] = conflicting_fields
    provenance["boundary_visibility_kind"] = str(provenance.get("boundary_origin_kind") or "unavailable")
    for field in (
        "actual_host_dispatch_duration_us",
        "actual_launch_control_dispatch_us",
        "actual_launch_api_body_us",
        "actual_launch_instrumentation_only_us",
    ):
        provenance.setdefault(field, None)
    provenance.setdefault("actual_launch_visibility_kind", "mixed_or_unresolved")
    actual_counterpart_fields = (
        "actual_host_dispatch_duration_us",
        "actual_launch_control_dispatch_us",
        "actual_launch_api_body_us",
        "actual_launch_instrumentation_only_us",
    )
    if all(provenance.get(field) in (None, "") for field in actual_counterpart_fields):
        provenance.setdefault(
            "actual_dispatch_counterpart_unavailable_reason",
            "raw_events_do_not_export_mechanically_separated_actual_dispatch_counterpart",
        )
    return provenance


def collate_trace_bundle(bundle: TraceBundle) -> CollatedTrace:
    """Collate per-rank traces into a deterministic low-level trace."""
    rank_events: dict[int, list[CollatedEvent]] = {}
    communicator_memberships = dict(bundle.communicator_memberships)
    collective_descriptor_cache: dict[tuple[int, int], _CollectiveDescriptor] = {}
    default_dispatch_scope = bundle.host_timing_dispatch_scope_resolved

    rank_step_windows: dict[int, tuple[int, int]] = {}
    all_events: list[TraceEvent] = []
    dispatch_window_start_ts: dict[tuple[object, ...], int] = {}
    dispatch_window_end_ts: dict[tuple[object, ...], int] = {}
    for rank_trace in bundle.rank_traces:
        rank_events[rank_trace.rank] = []
        if bundle.trace_window == "step" and rank_trace.rank in bundle.step_windows:
            rank_step_windows[rank_trace.rank] = bundle.step_windows[rank_trace.rank]
        for event in rank_trace.events:
            all_events.append(event)
            rank_step_window = rank_step_windows.get(event.rank)
            if rank_step_window is None:
                continue
            dispatch_key = _host_dispatch_key(
                event,
                default_scope=default_dispatch_scope,
            )
            dispatch_window_start_ts[dispatch_key] = min(
                dispatch_window_start_ts.get(dispatch_key, int(rank_step_window[0])),
                int(rank_step_window[0]),
            )
            dispatch_window_end_ts[dispatch_key] = max(
                dispatch_window_end_ts.get(dispatch_key, int(rank_step_window[1])),
                int(rank_step_window[1]),
            )

    collective_count_by_rank_key: dict[int, dict[tuple[object, ...], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    previous_id_by_dispatch: dict[tuple[object, ...], str | None] = {}
    previous_materialized_id_by_dispatch: dict[tuple[object, ...], str | None] = {}
    previous_materialized_api_by_dispatch: dict[tuple[object, ...], str | None] = {}
    previous_event_by_dispatch: dict[tuple[object, ...], TraceEvent] = {}
    previous_effective_ts_by_dispatch: dict[tuple[object, ...], int] = {}
    dispatch_high_watermark_ts: dict[tuple[object, ...], int] = {}
    pending_host_gap_by_dispatch: dict[tuple[object, ...], int] = defaultdict(int)
    pending_host_gap_prev_event_by_dispatch: dict[tuple[object, ...], TraceEvent] = {}
    pending_host_gap_current_event_by_dispatch: dict[tuple[object, ...], TraceEvent] = {}
    context_query_suffix_by_dispatch: dict[tuple[object, ...], _ContextQuerySuffixState] = {}
    pending_launch_config_pop_context_query_suffix_by_dispatch: dict[
        tuple[object, ...], _PendingLaunchConfigPopContextQuerySuffixFold
    ] = {}
    cublas_stream_by_rank_dispatch_handle: dict[
        tuple[int, tuple[object, ...], str], str
    ] = {}
    idempotent_cublas_set_stream_folds_by_dispatch: dict[
        tuple[object, ...], _IdempotentCublasSetStreamFoldState
    ] = {}
    next_available_ts_by_dispatch: dict[tuple[object, ...], int] = {}
    (
        suppress_launch_config_hostdelay_gap,
        launch_config_hostdelay_normalization_status,
    ) = _launch_config_hostdelay_normalization_policy()
    enable_idempotent_cublas_set_stream_fold = (
        _idempotent_cublas_set_stream_fold_enabled()
    )
    strict_occurrence_sequence_ordinals: dict[str, dict[str, object]] = {}
    if _cuda_gemm_hostdispatch_strict_occurrence_gap_diagnostics_enabled():
        api_sequence_counts: dict[tuple[int, str], int] = defaultdict(int)
        host_queue_sequence_counts: dict[tuple[object, ...], int] = defaultdict(int)
        stream_sequence_counts: dict[tuple[int, object], int] = defaultdict(int)
        for occurrence_event in sorted(
            all_events,
            key=lambda item: (int(item.ts), item.rank, item.ordinal),
        ):
            occurrence_dispatch_key = _host_dispatch_key(
                occurrence_event,
                default_scope=default_dispatch_scope,
            )
            api_key = (int(occurrence_event.rank), occurrence_event.api)
            stream_id = occurrence_event.extras.get("stream_id")
            stream_key = (int(occurrence_event.rank), stream_id)
            stream_ordinal = None
            if stream_id not in (None, ""):
                stream_ordinal = stream_sequence_counts[stream_key]
                stream_sequence_counts[stream_key] += 1
            event_id = _event_id(occurrence_event)
            strict_occurrence_sequence_ordinals[event_id] = {
                "api_sequence_ordinal_in_window": api_sequence_counts[api_key],
                "host_queue_sequence_ordinal_in_window": (
                    host_queue_sequence_counts[occurrence_dispatch_key]
                ),
                "stream_sequence_ordinal_in_window": stream_ordinal,
            }
            api_sequence_counts[api_key] += 1
            host_queue_sequence_counts[occurrence_dispatch_key] += 1

    for event in sorted(all_events, key=lambda item: (int(item.ts), item.rank, item.ordinal)):
        event_key = (event.rank, event.ordinal)
        dispatch_scope = _host_dispatch_scope(
            event,
            default_scope=default_dispatch_scope,
        )
        dispatch_key = _host_dispatch_key(
            event,
            default_scope=default_dispatch_scope,
        )
        host_machine_id = _host_machine_id(event)
        host_dispatch_queue_id = _host_dispatch_queue_id(event)
        effective_event_ts = int(event.ts)
        projected_host_dispatch_duration_us = _projected_host_dispatch_duration_us(event)
        projected_host_dispatch_advances_queue = _advances_projected_host_dispatch_queue(
            event,
            projected_host_dispatch_duration_us=projected_host_dispatch_duration_us,
        )
        if projected_host_dispatch_advances_queue:
            effective_event_ts = max(
                effective_event_ts,
                next_available_ts_by_dispatch.get(dispatch_key, effective_event_ts),
            )
        launch_config_pop_entry_hostdelay_adjustment_us = (
            _launch_config_pop_entry_hostdelay_adjustment_us(event)
        )
        hostdelay_boundary_event_ts = max(
            0,
            int(effective_event_ts)
            - int(max(launch_config_pop_entry_hostdelay_adjustment_us, 0)),
        )
        previous_event = previous_event_by_dispatch.get(dispatch_key)
        previous_id = previous_id_by_dispatch.get(dispatch_key)
        previous_materialized_id = previous_materialized_id_by_dispatch.get(dispatch_key)
        previous_materialized_api = previous_materialized_api_by_dispatch.get(dispatch_key)
        cublas_state_key = _cublas_handle_stream_state_key(event, dispatch_key)
        cublas_stream_id = _cublas_stream_id(event)
        fold_idempotent_cublas_set_stream = (
            enable_idempotent_cublas_set_stream_fold
            and cublas_state_key is not None
            and cublas_stream_id is not None
            and cublas_stream_by_rank_dispatch_handle.get(cublas_state_key) == cublas_stream_id
        )
        materialize_event = (
            _should_materialize_replay_event(event)
            and not fold_idempotent_cublas_set_stream
        )
        launch_config_metadata: dict[str, object] = {}
        context_query_suffix_metadata: dict[str, object] = {}
        context_query_run_fold_metadata: dict[str, object] = {}
        idempotent_cublas_set_stream_metadata: dict[str, object] = {}
        incremental_host_gap_us = 0
        context_query_leading_pre_suffix_gap_us = 0
        context_query_leading_pre_suffix_dispatch_start_ts: int | None = None
        folded_context_query_suffix = False
        dispatch_start_ts = dispatch_window_start_ts.get(dispatch_key)
        if previous_event is None and dispatch_start_ts is not None:
            leading_gap_us = max(0, int(hostdelay_boundary_event_ts) - int(dispatch_start_ts))
            if (
                leading_gap_us > 0
                and materialize_event
                and _should_insert_host_delay(event, event)
            ):
                leading_host_delay_start_ts = int(dispatch_start_ts)
                host_delay = _make_host_delay_event(
                    event_id=_boundary_host_delay_id(
                        event.rank,
                        event.ordinal,
                        "step_start",
                    ),
                    rank=event.rank,
                    ordinal=event.ordinal,
                    source=event.source,
                    ts=leading_host_delay_start_ts,
                    pid=event.pid,
                    tid=event.tid,
                    observed_gap_us=leading_gap_us,
                    prev_event_id=None,
                    dispatch_scope=dispatch_scope,
                    host_machine_id=host_machine_id,
                    host_dispatch_queue_id=host_dispatch_queue_id,
                    boundary="step_start",
                    boundary_origin_provenance=_host_delay_boundary_origin_provenance(
                        None,
                        event,
                        observed_gap_us=leading_gap_us,
                        host_dispatch_queue_id=host_dispatch_queue_id,
                        previous_materialized_event_id=None,
                        previous_materialized_api=None,
                        current_materialized_event_id=_event_id(event),
                        current_materialized_api=event.api,
                        paper_valid_window_membership=_paper_valid_window_membership(
                            rank=event.rank,
                            boundary_ts=leading_host_delay_start_ts,
                            step_window=rank_step_windows.get(event.rank),
                            fidelity_window=bundle.fidelity_windows.get(event.rank),
                        ),
                        strict_occurrence_sequence_ordinals=(
                            strict_occurrence_sequence_ordinals
                        ),
                    ),
                )
                rank_events[event.rank].append(host_delay)
                previous_id = host_delay.id
            elif (
                leading_gap_us > 0
                and event.api == "__cudaPopCallConfiguration"
                and _should_insert_host_delay(event, event)
            ):
                leading_host_delay_start_ts = int(dispatch_start_ts)
                host_delay = _make_host_delay_event(
                    event_id=_boundary_host_delay_id(
                        event.rank,
                        event.ordinal,
                        "step_start",
                    ),
                    rank=event.rank,
                    ordinal=event.ordinal,
                    source=event.source,
                    ts=leading_host_delay_start_ts,
                    pid=event.pid,
                    tid=event.tid,
                    observed_gap_us=leading_gap_us,
                    prev_event_id=None,
                    dispatch_scope=dispatch_scope,
                    host_machine_id=host_machine_id,
                    host_dispatch_queue_id=host_dispatch_queue_id,
                    boundary="step_start",
                    boundary_origin_provenance=_host_delay_boundary_origin_provenance(
                        None,
                        event,
                        observed_gap_us=leading_gap_us,
                        host_dispatch_queue_id=host_dispatch_queue_id,
                        previous_materialized_event_id=None,
                        previous_materialized_api=None,
                        current_materialized_event_id=None,
                        current_materialized_api=None,
                        paper_valid_window_membership=_paper_valid_window_membership(
                            rank=event.rank,
                            boundary_ts=leading_host_delay_start_ts,
                            step_window=rank_step_windows.get(event.rank),
                            fidelity_window=bundle.fidelity_windows.get(event.rank),
                        ),
                        strict_occurrence_sequence_ordinals=(
                            strict_occurrence_sequence_ordinals
                        ),
                    ),
                )
                rank_events[event.rank].append(host_delay)
                previous_id = host_delay.id
            elif (
                leading_gap_us > 0
                and _is_cublas_set_stream_context_query(event)
                and _should_insert_host_delay(event, event)
            ):
                context_query_leading_pre_suffix_gap_us = leading_gap_us
                context_query_leading_pre_suffix_dispatch_start_ts = int(dispatch_start_ts)
        if previous_event is not None:
            previous_high_watermark_ts = dispatch_high_watermark_ts.get(
                dispatch_key,
                previous_effective_ts_by_dispatch.get(dispatch_key, int(previous_event.ts)),
            )
            incremental_host_gap_us = max(
                0,
                int(hostdelay_boundary_event_ts) - int(previous_high_watermark_ts),
            )
            context_query_suffix_state = context_query_suffix_by_dispatch.get(dispatch_key)
            is_context_query_suffix_target = (
                _is_cublas_set_stream_context_query_suffix_target(event)
                and not fold_idempotent_cublas_set_stream
                and context_query_suffix_state is not None
                and bool(context_query_suffix_state.events)
                and context_query_suffix_state.events[-1] is previous_event
            )
            is_context_query_launch_config_pop_target = (
                _is_context_query_suffix_launch_config_pop_target(event)
                and context_query_suffix_state is not None
                and bool(context_query_suffix_state.events)
                and context_query_suffix_state.events[-1] is previous_event
            )
            is_context_query_run_fold_target = (
                materialize_event
                and not is_context_query_suffix_target
                and not is_context_query_launch_config_pop_target
                and context_query_suffix_state is not None
                and len(context_query_suffix_state.events) >= 2
                and context_query_suffix_state.events[-1] is previous_event
                and int(max(context_query_suffix_state.internal_gap_us, 0)) > 0
            )
            is_launch_config_metadata_edge = _is_internal_launch_config_metadata_edge(
                previous_event,
                event,
            )
            pending_launch_config_pop_context_query_suffix = (
                pending_launch_config_pop_context_query_suffix_by_dispatch.get(
                    dispatch_key
                )
            )
            use_pending_launch_config_pop_context_query_suffix = (
                pending_launch_config_pop_context_query_suffix is not None
                and is_launch_config_metadata_edge
                and pending_launch_config_pop_context_query_suffix.pop_event is previous_event
            )
            if (
                pending_launch_config_pop_context_query_suffix is not None
                and not use_pending_launch_config_pop_context_query_suffix
            ):
                pending_host_gap_by_dispatch[dispatch_key] += int(
                    max(
                        pending_launch_config_pop_context_query_suffix.terminal_gap_us,
                        0,
                    )
                )
                if pending_launch_config_pop_context_query_suffix.terminal_gap_us > 0:
                    pending_host_gap_prev_event_by_dispatch.setdefault(
                        dispatch_key,
                        pending_launch_config_pop_context_query_suffix.state.events[-1],
                    )
                    pending_host_gap_current_event_by_dispatch[dispatch_key] = (
                        pending_launch_config_pop_context_query_suffix.pop_event
                    )
                pending_launch_config_pop_context_query_suffix_by_dispatch.pop(
                    dispatch_key,
                    None,
                )
            if is_launch_config_metadata_edge:
                launch_config_metadata = _launch_config_metadata_from_pop(
                    previous_event,
                    adjacent_gap_us=incremental_host_gap_us,
                    suppress_hostdelay_gap=suppress_launch_config_hostdelay_gap,
                    normalization_status=launch_config_hostdelay_normalization_status,
                )
                if use_pending_launch_config_pop_context_query_suffix:
                    context_query_suffix_metadata = (
                        _context_query_suffix_launch_config_pop_metadata(
                            pending_launch_config_pop_context_query_suffix,
                            event,
                        )
                    )
                    pending_launch_config_pop_context_query_suffix_by_dispatch.pop(
                        dispatch_key,
                        None,
                    )
            if is_context_query_suffix_target:
                pending_before_suffix_fold_us = pending_host_gap_by_dispatch[dispatch_key]
                suffix_internal_gap_us = int(max(context_query_suffix_state.internal_gap_us, 0))
                leading_pre_suffix_gap_us = int(
                    max(context_query_suffix_state.leading_pre_suffix_gap_us, 0)
                )
                preserved_pre_suffix_gap_us = max(
                    0,
                    leading_pre_suffix_gap_us
                    + int(pending_before_suffix_fold_us)
                    - suffix_internal_gap_us,
                )
                if leading_pre_suffix_gap_us > 0:
                    first_suffix_event = context_query_suffix_state.events[0]
                    leading_dispatch_start_ts = (
                        context_query_suffix_state.leading_pre_suffix_dispatch_start_ts
                    )
                    if leading_dispatch_start_ts is None:
                        leading_dispatch_start_ts = dispatch_window_start_ts.get(
                            dispatch_key,
                            int(first_suffix_event.ts),
                        )
                    leading_host_delay_start_ts = int(leading_dispatch_start_ts)
                    leading_host_delay = _make_host_delay_event(
                        event_id=_boundary_host_delay_id(
                            first_suffix_event.rank,
                            first_suffix_event.ordinal,
                            "step_start",
                        ),
                        rank=first_suffix_event.rank,
                        ordinal=first_suffix_event.ordinal,
                        source=first_suffix_event.source,
                        ts=leading_host_delay_start_ts,
                        pid=first_suffix_event.pid,
                        tid=first_suffix_event.tid,
                        observed_gap_us=leading_pre_suffix_gap_us,
                        prev_event_id=None,
                        dispatch_scope=dispatch_scope,
                        host_machine_id=host_machine_id,
                        host_dispatch_queue_id=host_dispatch_queue_id,
                        boundary="step_start",
                        boundary_origin_provenance=_host_delay_boundary_origin_provenance(
                            None,
                            first_suffix_event,
                            observed_gap_us=leading_pre_suffix_gap_us,
                            host_dispatch_queue_id=host_dispatch_queue_id,
                            previous_materialized_event_id=None,
                            previous_materialized_api=None,
                            current_materialized_event_id=_event_id(event),
                            current_materialized_api=event.api,
                            paper_valid_window_membership=_paper_valid_window_membership(
                                rank=first_suffix_event.rank,
                                boundary_ts=leading_host_delay_start_ts,
                                step_window=rank_step_windows.get(first_suffix_event.rank),
                                fidelity_window=bundle.fidelity_windows.get(first_suffix_event.rank),
                            ),
                            strict_occurrence_sequence_ordinals=(
                                strict_occurrence_sequence_ordinals
                            ),
                        ),
                    )
                    rank_events[first_suffix_event.rank].append(leading_host_delay)
                    previous_id = leading_host_delay.id
                pending_gap_after_leading_us = max(
                    0,
                    preserved_pre_suffix_gap_us - leading_pre_suffix_gap_us,
                )
                pending_host_gap_by_dispatch[dispatch_key] = pending_gap_after_leading_us
                if pending_gap_after_leading_us > 0:
                    pending_host_gap_current_event_by_dispatch[dispatch_key] = (
                        context_query_suffix_state.events[0]
                    )
                else:
                    pending_host_gap_prev_event_by_dispatch.pop(dispatch_key, None)
                    pending_host_gap_current_event_by_dispatch.pop(dispatch_key, None)
                context_query_suffix_metadata = _context_query_suffix_metadata(
                    context_query_suffix_state,
                    event,
                    terminal_gap_us=incremental_host_gap_us,
                    preserved_pre_suffix_gap_us=preserved_pre_suffix_gap_us,
                )
                folded_context_query_suffix = True
            elif is_context_query_launch_config_pop_target:
                pending_before_suffix_fold_us = pending_host_gap_by_dispatch[dispatch_key]
                suffix_internal_gap_us = int(max(context_query_suffix_state.internal_gap_us, 0))
                leading_pre_suffix_gap_us = int(
                    max(context_query_suffix_state.leading_pre_suffix_gap_us, 0)
                )
                preserved_pre_suffix_gap_us = max(
                    0,
                    leading_pre_suffix_gap_us
                    + int(pending_before_suffix_fold_us)
                    - suffix_internal_gap_us,
                )
                if leading_pre_suffix_gap_us > 0:
                    first_suffix_event = context_query_suffix_state.events[0]
                    leading_dispatch_start_ts = (
                        context_query_suffix_state.leading_pre_suffix_dispatch_start_ts
                    )
                    if leading_dispatch_start_ts is None:
                        leading_dispatch_start_ts = dispatch_window_start_ts.get(
                            dispatch_key,
                            int(first_suffix_event.ts),
                        )
                    leading_host_delay_start_ts = int(leading_dispatch_start_ts)
                    leading_host_delay = _make_host_delay_event(
                        event_id=_boundary_host_delay_id(
                            first_suffix_event.rank,
                            first_suffix_event.ordinal,
                            "step_start",
                        ),
                        rank=first_suffix_event.rank,
                        ordinal=first_suffix_event.ordinal,
                        source=first_suffix_event.source,
                        ts=leading_host_delay_start_ts,
                        pid=first_suffix_event.pid,
                        tid=first_suffix_event.tid,
                        observed_gap_us=leading_pre_suffix_gap_us,
                        prev_event_id=None,
                        dispatch_scope=dispatch_scope,
                        host_machine_id=host_machine_id,
                        host_dispatch_queue_id=host_dispatch_queue_id,
                        boundary="step_start",
                        boundary_origin_provenance=_host_delay_boundary_origin_provenance(
                            None,
                            first_suffix_event,
                            observed_gap_us=leading_pre_suffix_gap_us,
                            host_dispatch_queue_id=host_dispatch_queue_id,
                            previous_materialized_event_id=None,
                            previous_materialized_api=None,
                            current_materialized_event_id=_event_id(event),
                            current_materialized_api=event.api,
                            paper_valid_window_membership=_paper_valid_window_membership(
                                rank=first_suffix_event.rank,
                                boundary_ts=leading_host_delay_start_ts,
                                step_window=rank_step_windows.get(first_suffix_event.rank),
                                fidelity_window=bundle.fidelity_windows.get(first_suffix_event.rank),
                            ),
                            strict_occurrence_sequence_ordinals=(
                                strict_occurrence_sequence_ordinals
                            ),
                        ),
                    )
                    rank_events[first_suffix_event.rank].append(leading_host_delay)
                    previous_id = leading_host_delay.id
                pending_gap_after_leading_us = max(
                    0,
                    preserved_pre_suffix_gap_us - leading_pre_suffix_gap_us,
                )
                pending_host_gap_by_dispatch[dispatch_key] = pending_gap_after_leading_us
                if pending_gap_after_leading_us > 0:
                    pending_host_gap_current_event_by_dispatch[dispatch_key] = (
                        context_query_suffix_state.events[0]
                    )
                else:
                    pending_host_gap_prev_event_by_dispatch.pop(dispatch_key, None)
                    pending_host_gap_current_event_by_dispatch.pop(dispatch_key, None)
                pending_launch_config_pop_context_query_suffix_by_dispatch[dispatch_key] = (
                    _PendingLaunchConfigPopContextQuerySuffixFold(
                        state=context_query_suffix_state,
                        pop_event=event,
                        terminal_gap_us=incremental_host_gap_us,
                        preserved_pre_suffix_gap_us=preserved_pre_suffix_gap_us,
                    )
                )
                folded_context_query_suffix = True
            elif is_context_query_run_fold_target:
                pending_before_run_fold_us = pending_host_gap_by_dispatch[dispatch_key]
                run_internal_gap_us = int(
                    max(context_query_suffix_state.internal_gap_us, 0)
                )
                leading_pre_run_gap_us = int(
                    max(context_query_suffix_state.leading_pre_suffix_gap_us, 0)
                )
                preserved_pre_run_gap_us = max(
                    0,
                    leading_pre_run_gap_us
                    + int(pending_before_run_fold_us)
                    - run_internal_gap_us,
                )
                if leading_pre_run_gap_us > 0:
                    first_run_event = context_query_suffix_state.events[0]
                    leading_dispatch_start_ts = (
                        context_query_suffix_state.leading_pre_suffix_dispatch_start_ts
                    )
                    if leading_dispatch_start_ts is None:
                        leading_dispatch_start_ts = dispatch_window_start_ts.get(
                            dispatch_key,
                            int(first_run_event.ts),
                        )
                    leading_host_delay_start_ts = int(leading_dispatch_start_ts)
                    leading_host_delay = _make_host_delay_event(
                        event_id=_boundary_host_delay_id(
                            first_run_event.rank,
                            first_run_event.ordinal,
                            "step_start",
                        ),
                        rank=first_run_event.rank,
                        ordinal=first_run_event.ordinal,
                        source=first_run_event.source,
                        ts=leading_host_delay_start_ts,
                        pid=first_run_event.pid,
                        tid=first_run_event.tid,
                        observed_gap_us=leading_pre_run_gap_us,
                        prev_event_id=None,
                        dispatch_scope=dispatch_scope,
                        host_machine_id=host_machine_id,
                        host_dispatch_queue_id=host_dispatch_queue_id,
                        boundary="step_start",
                        boundary_origin_provenance=_host_delay_boundary_origin_provenance(
                            None,
                            first_run_event,
                            observed_gap_us=leading_pre_run_gap_us,
                            host_dispatch_queue_id=host_dispatch_queue_id,
                            previous_materialized_event_id=None,
                            previous_materialized_api=None,
                            current_materialized_event_id=_event_id(event),
                            current_materialized_api=event.api,
                            paper_valid_window_membership=_paper_valid_window_membership(
                                rank=first_run_event.rank,
                                boundary_ts=leading_host_delay_start_ts,
                                step_window=rank_step_windows.get(first_run_event.rank),
                                fidelity_window=bundle.fidelity_windows.get(first_run_event.rank),
                            ),
                            strict_occurrence_sequence_ordinals=(
                                strict_occurrence_sequence_ordinals
                            ),
                        ),
                    )
                    rank_events[first_run_event.rank].append(leading_host_delay)
                    previous_id = leading_host_delay.id
                pending_gap_after_leading_us = max(
                    0,
                    preserved_pre_run_gap_us - leading_pre_run_gap_us,
                )
                pending_gap_after_run_fold_us = (
                    pending_gap_after_leading_us + int(max(incremental_host_gap_us, 0))
                )
                pending_host_gap_by_dispatch[dispatch_key] = pending_gap_after_run_fold_us
                if pending_gap_after_run_fold_us > 0:
                    if pending_gap_after_leading_us <= 0 and incremental_host_gap_us > 0:
                        pending_host_gap_prev_event_by_dispatch[dispatch_key] = (
                            context_query_suffix_state.events[-1]
                        )
                    else:
                        pending_host_gap_prev_event_by_dispatch.setdefault(
                            dispatch_key,
                            context_query_suffix_state.events[0],
                        )
                    pending_host_gap_current_event_by_dispatch[dispatch_key] = event
                else:
                    pending_host_gap_prev_event_by_dispatch.pop(dispatch_key, None)
                    pending_host_gap_current_event_by_dispatch.pop(dispatch_key, None)
                context_query_run_fold_metadata = _context_query_run_fold_metadata(
                    context_query_suffix_state,
                    event,
                    terminal_gap_us=incremental_host_gap_us,
                    preserved_pre_run_gap_us=preserved_pre_run_gap_us,
                )
            elif not is_launch_config_metadata_edge or not suppress_launch_config_hostdelay_gap:
                pending_host_gap_by_dispatch[dispatch_key] += incremental_host_gap_us
                if incremental_host_gap_us > 0:
                    pending_host_gap_prev_event_by_dispatch.setdefault(
                        dispatch_key,
                        previous_event,
                    )
                    pending_host_gap_current_event_by_dispatch[dispatch_key] = event
            pending_idempotent_cublas_set_stream_fold = (
                idempotent_cublas_set_stream_folds_by_dispatch.get(dispatch_key)
            )
            if (
                materialize_event
                and pending_idempotent_cublas_set_stream_fold is not None
                and pending_idempotent_cublas_set_stream_fold.events
            ):
                terminal_gap_us = (
                    incremental_host_gap_us
                    if pending_idempotent_cublas_set_stream_fold.events[-1] is previous_event
                    else None
                )
                idempotent_cublas_set_stream_metadata = (
                    _idempotent_cublas_set_stream_fold_metadata(
                        pending_idempotent_cublas_set_stream_fold,
                        event,
                        pending_host_gap_us=pending_host_gap_by_dispatch[dispatch_key],
                        terminal_gap_us=terminal_gap_us,
                    )
                )
            host_gap_us = pending_host_gap_by_dispatch[dispatch_key]
            if (
                host_gap_us > 0
                and materialize_event
                and _should_insert_host_delay(previous_event, event)
            ):
                provenance_previous_event = pending_host_gap_prev_event_by_dispatch.get(
                    dispatch_key,
                    previous_event,
                )
                provenance_current_event = pending_host_gap_current_event_by_dispatch.get(
                    dispatch_key,
                    event,
                )
                host_delay_start_ts = int(effective_event_ts) - int(host_gap_us)
                host_delay = _make_host_delay_event(
                    event_id=_host_delay_id(event.rank, event.ordinal),
                    rank=event.rank,
                    ordinal=event.ordinal,
                    source=event.source,
                    ts=host_delay_start_ts,
                    pid=event.pid,
                    tid=event.tid,
                    observed_gap_us=host_gap_us,
                    prev_event_id=previous_id,
                    dispatch_scope=dispatch_scope,
                    host_machine_id=host_machine_id,
                    host_dispatch_queue_id=host_dispatch_queue_id,
                    boundary_origin_provenance=_host_delay_boundary_origin_provenance(
                        provenance_previous_event,
                        provenance_current_event,
                        observed_gap_us=host_gap_us,
                        host_dispatch_queue_id=host_dispatch_queue_id,
                        previous_materialized_event_id=previous_materialized_id,
                        previous_materialized_api=previous_materialized_api,
                        current_materialized_event_id=_event_id(event),
                        current_materialized_api=event.api,
                        paper_valid_window_membership=_paper_valid_window_membership(
                            rank=event.rank,
                            boundary_ts=host_delay_start_ts,
                            step_window=rank_step_windows.get(event.rank),
                            fidelity_window=bundle.fidelity_windows.get(event.rank),
                        ),
                        strict_occurrence_sequence_ordinals=(
                            strict_occurrence_sequence_ordinals
                        ),
                    ),
                )
                rank_events[event.rank].append(host_delay)
                previous_id = host_delay.id
                pending_host_gap_by_dispatch[dispatch_key] = 0
                pending_host_gap_prev_event_by_dispatch.pop(dispatch_key, None)
                pending_host_gap_current_event_by_dispatch.pop(dispatch_key, None)

        collective_group_id = None
        if _is_collective(event):
            descriptor = collective_descriptor_cache.get(event_key)
            if descriptor is None:
                descriptor = _collective_descriptor(event, communicator_memberships)
                collective_descriptor_cache[event_key] = descriptor
            collective_group_id = descriptor.group_id_base
            if descriptor.match_basis in {
                "payload_signature",
                "communicator_pair_sequence",
                "communicator_pair_payload",
            }:
                collective_count_by_key = collective_count_by_rank_key[event.rank]
                signature_occurrence = collective_count_by_key[descriptor.key]
                collective_count_by_key[descriptor.key] += 1
                suffix = (
                    f"|pair_seq:{signature_occurrence}"
                    if descriptor.match_basis == "communicator_pair_sequence"
                    else (
                        f"|pair_occ:{signature_occurrence}"
                        if descriptor.match_basis == "communicator_pair_payload"
                        else f"#{signature_occurrence}"
                    )
                )
                collective_group_id = f"{collective_group_id}{suffix}"

        if materialize_event:
            collated = CollatedEvent(
                id=_event_id(event),
                rank=event.rank,
                ordinal=event.ordinal,
                source=event.source,
                ts=effective_event_ts,
                pid=event.pid,
                tid=event.tid,
                module=event.module,
                api=event.api,
                op_type=event.op_type,
                extras={
                    **dict(event.extras),
                    **_safe_launch_metadata(event),
                    **launch_config_metadata,
                    **context_query_suffix_metadata,
                    **context_query_run_fold_metadata,
                    **idempotent_cublas_set_stream_metadata,
                    "host_timing_dispatch_scope": dispatch_scope,
                    "host_machine_id": host_machine_id,
                    "host_dispatch_queue_id": host_dispatch_queue_id,
                    "host_dispatch_model": _host_dispatch_model(dispatch_scope),
                },
                prev_event_id=previous_id,
                collective_group_id=collective_group_id,
            )
            rank_events[event.rank].append(collated)
            previous_id_by_dispatch[dispatch_key] = collated.id
            previous_materialized_id_by_dispatch[dispatch_key] = collated.id
            previous_materialized_api_by_dispatch[dispatch_key] = collated.api
            idempotent_cublas_set_stream_folds_by_dispatch.pop(dispatch_key, None)
        else:
            previous_id_by_dispatch[dispatch_key] = previous_id
        if fold_idempotent_cublas_set_stream:
            handle_id = _cublas_handle_id(event)
            stream_id = _cublas_stream_id(event)
            if handle_id is not None and stream_id is not None:
                fold_state = idempotent_cublas_set_stream_folds_by_dispatch.get(
                    dispatch_key
                )
                if (
                    fold_state is not None
                    and previous_event is not None
                    and fold_state.events
                    and fold_state.events[-1] is previous_event
                ):
                    fold_state.internal_gap_rows.append(
                        _idempotent_cublas_set_stream_gap_row(
                            previous_event,
                            event,
                            observed_gap_us=incremental_host_gap_us,
                            gap_kind="idempotent_cublasSetStream_v2_internal_gap",
                        )
                    )
                    fold_state.events.append(event)
                elif fold_state is not None:
                    fold_state.events.append(event)
                else:
                    idempotent_cublas_set_stream_folds_by_dispatch[dispatch_key] = (
                        _IdempotentCublasSetStreamFoldState(
                            events=[event],
                            internal_gap_rows=[],
                        )
                    )
        elif materialize_event:
            idempotent_cublas_set_stream_folds_by_dispatch.pop(dispatch_key, None)
        if _is_cublas_set_stream_context_query(event):
            suffix_state = context_query_suffix_by_dispatch.get(dispatch_key)
            if (
                suffix_state is not None
                and previous_event is not None
                and suffix_state.events
                and suffix_state.events[-1] is previous_event
            ):
                suffix_state.internal_gap_us += incremental_host_gap_us
                if suffix_state.internal_gap_rows is None:
                    suffix_state.internal_gap_rows = []
                suffix_state.internal_gap_rows.append(
                    _context_query_suffix_gap_row(
                        previous_event,
                        event,
                        observed_gap_us=incremental_host_gap_us,
                        gap_kind="context_query_suffix_internal_gap",
                    )
                )
                suffix_state.events.append(event)
            else:
                context_query_suffix_by_dispatch[dispatch_key] = _ContextQuerySuffixState(
                    events=[event],
                    internal_gap_rows=[],
                    leading_pre_suffix_gap_us=context_query_leading_pre_suffix_gap_us,
                    leading_pre_suffix_dispatch_start_ts=(
                        context_query_leading_pre_suffix_dispatch_start_ts
                    ),
                )
        elif folded_context_query_suffix:
            context_query_suffix_by_dispatch.pop(dispatch_key, None)
        else:
            context_query_suffix_by_dispatch.pop(dispatch_key, None)
        if (
            event.api == "cublasSetStream_v2"
            and cublas_state_key is not None
            and cublas_stream_id is not None
        ):
            cublas_stream_by_rank_dispatch_handle[cublas_state_key] = cublas_stream_id
        cublas_lifecycle_state_key = _cublas_handle_lifecycle_state_key(
            event,
            dispatch_key,
        )
        if cublas_lifecycle_state_key is not None:
            cublas_stream_by_rank_dispatch_handle.pop(cublas_lifecycle_state_key, None)
            pending_fold = idempotent_cublas_set_stream_folds_by_dispatch.get(
                dispatch_key
            )
            if pending_fold is not None:
                invalidated_handle_id = cublas_lifecycle_state_key[2]
                remaining_event_ids = {
                    _event_id(folded_event)
                    for folded_event in pending_fold.events
                    if _cublas_handle_id(folded_event) != invalidated_handle_id
                }
                if not remaining_event_ids:
                    idempotent_cublas_set_stream_folds_by_dispatch.pop(
                        dispatch_key,
                        None,
                    )
                elif len(remaining_event_ids) != len(pending_fold.events):
                    pending_fold.events = [
                        folded_event
                        for folded_event in pending_fold.events
                        if _event_id(folded_event) in remaining_event_ids
                    ]
                    pending_fold.internal_gap_rows = [
                        row
                        for row in pending_fold.internal_gap_rows
                        if row.get("raw_prev_event_id") in remaining_event_ids
                        and row.get("raw_current_event_id") in remaining_event_ids
                    ]
        previous_event_by_dispatch[dispatch_key] = event
        previous_effective_ts_by_dispatch[dispatch_key] = int(effective_event_ts)
        dispatch_high_watermark_ts[dispatch_key] = max(
            dispatch_high_watermark_ts.get(dispatch_key, int(effective_event_ts)),
            _dispatch_gap_high_watermark_ts(
                event,
                effective_event_ts=int(effective_event_ts),
                projected_host_dispatch_duration_us=projected_host_dispatch_duration_us,
            ),
        )
        if projected_host_dispatch_advances_queue:
            next_available_ts_by_dispatch[dispatch_key] = max(
                next_available_ts_by_dispatch.get(dispatch_key, int(effective_event_ts)),
                int(effective_event_ts) + projected_host_dispatch_duration_us,
            )

    for (
        dispatch_key,
        pending_launch_config_pop_context_query_suffix,
    ) in list(pending_launch_config_pop_context_query_suffix_by_dispatch.items()):
        pending_host_gap_by_dispatch[dispatch_key] += int(
            max(pending_launch_config_pop_context_query_suffix.terminal_gap_us, 0)
        )
        if pending_launch_config_pop_context_query_suffix.terminal_gap_us > 0:
            pending_host_gap_prev_event_by_dispatch.setdefault(
                dispatch_key,
                pending_launch_config_pop_context_query_suffix.state.events[-1],
            )
            pending_host_gap_current_event_by_dispatch[dispatch_key] = (
                pending_launch_config_pop_context_query_suffix.pop_event
            )
        pending_launch_config_pop_context_query_suffix_by_dispatch.pop(
            dispatch_key,
            None,
        )

    for dispatch_key, previous_event in sorted(previous_event_by_dispatch.items()):
        dispatch_end_ts = dispatch_window_end_ts.get(dispatch_key)
        if dispatch_end_ts is None:
            continue
        pending_host_gap_us = max(0, pending_host_gap_by_dispatch.get(dispatch_key, 0))
        trailing_gap_us = max(
            0,
            int(dispatch_end_ts)
            - int(
                dispatch_high_watermark_ts.get(
                    dispatch_key,
                    previous_effective_ts_by_dispatch.get(dispatch_key, int(previous_event.ts)),
                )
            ),
        )
        step_end_gap_us = pending_host_gap_us + trailing_gap_us
        if step_end_gap_us <= 0 or not _should_insert_host_delay(previous_event, previous_event):
            continue
        provenance_previous_event = pending_host_gap_prev_event_by_dispatch.get(
            dispatch_key,
            previous_event,
        )
        provenance_current_event = pending_host_gap_current_event_by_dispatch.get(
            dispatch_key,
            previous_event,
        )
        step_end_host_delay_start_ts = int(dispatch_end_ts) - int(step_end_gap_us)
        host_delay = _make_host_delay_event(
            event_id=_boundary_host_delay_id(
                previous_event.rank,
                previous_event.ordinal + 1,
                "step_end",
            ),
            rank=previous_event.rank,
            ordinal=previous_event.ordinal + 1,
            source=previous_event.source,
            ts=step_end_host_delay_start_ts,
            pid=previous_event.pid,
            tid=previous_event.tid,
            observed_gap_us=step_end_gap_us,
            prev_event_id=previous_id_by_dispatch.get(dispatch_key),
            dispatch_scope=_host_dispatch_scope(
                previous_event,
                default_scope=default_dispatch_scope,
            ),
            host_machine_id=_host_machine_id(previous_event),
            host_dispatch_queue_id=_host_dispatch_queue_id(previous_event),
            boundary="step_end",
            boundary_origin_provenance=_host_delay_boundary_origin_provenance(
                provenance_previous_event,
                provenance_current_event,
                observed_gap_us=step_end_gap_us,
                host_dispatch_queue_id=_host_dispatch_queue_id(previous_event),
                previous_materialized_event_id=previous_materialized_id_by_dispatch.get(dispatch_key),
                previous_materialized_api=previous_materialized_api_by_dispatch.get(dispatch_key),
                current_materialized_event_id=None,
                current_materialized_api="step_end",
                paper_valid_window_membership=_paper_valid_window_membership(
                    rank=previous_event.rank,
                    boundary_ts=step_end_host_delay_start_ts,
                    step_window=rank_step_windows.get(previous_event.rank),
                    fidelity_window=bundle.fidelity_windows.get(previous_event.rank),
                ),
                strict_occurrence_sequence_ordinals=(
                    strict_occurrence_sequence_ordinals
                ),
            ),
        )
        rank_events[previous_event.rank].append(host_delay)

    global_events = tuple(
        sorted(
            (event for events in rank_events.values() for event in events),
            key=lambda event: (event.ts, event.rank, event.ordinal),
        )
    )

    grouped_events_by_group: dict[str, list[CollatedEvent]] = defaultdict(list)
    for event in global_events:
        if event.collective_group_id is not None:
            grouped_events_by_group[event.collective_group_id].append(event)

    collective_groups: dict[str, CollectiveGroup] = {}
    for group_id, grouped_events in sorted(grouped_events_by_group.items()):
        ranks = tuple(sorted(event.rank for event in grouped_events))
        first_event = grouped_events[0]
        first_descriptor_key = (first_event.rank, first_event.ordinal)
        descriptor = collective_descriptor_cache.get(first_descriptor_key)
        if descriptor is None:
            descriptor = _collective_descriptor(first_event, communicator_memberships)
            collective_descriptor_cache[first_descriptor_key] = descriptor
        group_apis = {event.api for event in grouped_events}
        group_api = "ncclP2P" if group_apis <= {"ncclSend", "ncclRecv"} else first_event.api
        if descriptor.match_basis == "communicator_sequence" and first_event.api not in {"ncclSend", "ncclRecv"}:
            if len(group_apis) != 1:
                raise ValueError(
                    "communicator/sequence grouped incompatible collective APIs: "
                    f"{group_id} -> {sorted(group_apis)}"
                )
        grouped_descriptors: list[_CollectiveDescriptor] = []
        for event in grouped_events:
            descriptor_key = (event.rank, event.ordinal)
            grouped_descriptor = collective_descriptor_cache.get(descriptor_key)
            if grouped_descriptor is None:
                grouped_descriptor = _collective_descriptor(event, communicator_memberships)
                collective_descriptor_cache[descriptor_key] = grouped_descriptor
            grouped_descriptors.append(grouped_descriptor)
        communicator_sizes = [
            grouped_descriptor.communicator_size
            for grouped_descriptor in grouped_descriptors
            if grouped_descriptor.communicator_size is not None
        ]
        participant_counts = [
            grouped_descriptor.participant_count
            for grouped_descriptor in grouped_descriptors
            if grouped_descriptor.participant_count is not None
        ]
        communicator_size = max(communicator_sizes) if communicator_sizes else None
        participant_count = max(participant_counts) if participant_counts else None
        if participant_count is None:
            participant_count = len(ranks)
        sequence_number = descriptor.sequence_number
        if descriptor.match_basis == "communicator_pair_sequence":
            sequence_number = _p2p_pair_sequence_number(group_id)
        collective_groups[group_id] = CollectiveGroup(
            id=group_id,
            api=group_api,
            op_type=first_event.op_type,
            ranks=ranks,
            event_ids=tuple(event.id for event in grouped_events),
            communicator_id=descriptor.communicator_id,
            sequence_number=sequence_number,
            communicator_size=communicator_size,
            participant_count=participant_count,
            root=descriptor.root,
            match_basis=descriptor.match_basis,
        )

    if collective_groups:
        collective_event_metadata = {
            group_id: _collective_group_event_metadata(group)
            for group_id, group in collective_groups.items()
        }
        updated_collective_events_by_id: dict[str, CollatedEvent] = {}
        updated_rank_events: dict[int, tuple[CollatedEvent, ...]] = {}
        for rank, events in sorted(rank_events.items()):
            updated_events: list[CollatedEvent] = []
            for event in events:
                if event.collective_group_id is None:
                    updated_events.append(event)
                    continue
                updated_event = replace(
                    event,
                    extras={
                        **event.extras,
                        **collective_event_metadata[event.collective_group_id],
                    },
                )
                updated_events.append(updated_event)
                updated_collective_events_by_id[event.id] = updated_event
            updated_rank_events[rank] = tuple(updated_events)
        rank_events = updated_rank_events
        global_events = tuple(
            updated_collective_events_by_id.get(event.id, event)
            for event in global_events
        )

    return CollatedTrace(
        trace_dir=bundle.trace_dir,
        source=bundle.source,
        rank_events={rank: tuple(events) for rank, events in sorted(rank_events.items())},
        global_events=global_events,
        collective_groups=collective_groups,
        original_world_size=bundle.world_size,
        captured_world_size=bundle.profiled_world_size,
        profiled_rank_groups=(
            dict(bundle.profiled_rank_groups)
            if bundle.profiled_rank_groups
            else {rank: (rank,) for rank in sorted(rank_events)}
        ),
        rank_host_machines=dict(bundle.rank_host_machines),
        rank_host_dispatch_queues=dict(bundle.rank_host_dispatch_queues),
        communicator_memberships=communicator_memberships,
        host_timing_dispatch_scope_resolved=bundle.host_timing_dispatch_scope_resolved,
        logical_rank_materialized=bundle.logical_rank_materialized,
        trace_window=bundle.trace_window,
        fidelity_windows=dict(bundle.fidelity_windows),
    )
