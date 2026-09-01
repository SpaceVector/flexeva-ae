"""
Paper-aligned discrete-event replay simulation for Maya-lite.

This replay follows the core structure described in Maya section 4.3 and
Appendix A:

- host-op arrivals are inserted into a top-level priority queue
- a scheduler manages host-thread and accelerator-stream resources
- operations that cannot proceed block on global wait maps
- completion of each operation emits an EndEvent and triggers a scheduler tick

The implementation remains intentionally lightweight, but it now mirrors the
paper's method much more closely than the previous clock-based replay.
"""

from __future__ import annotations

from bisect import insort
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from heapq import heappop, heappush
import hashlib
import os
import re
from typing import Any, Callable

from .filters import targets_stream_resource
from .schema import (
    AnnotatedEvent,
    AnnotatedTrace,
    RankReplayMetrics,
    ReplayResult,
    SimulatedEvent,
)
from .launch_neighborhood import (
    LAUNCH_NEIGHBORHOOD_EQUIVALENCE_EXPORT_FIELDS,
    metadata_has_launch_neighborhood_equivalence,
)


_COMPUTE_TYPES = {"kernel_launch", "blas_compute"}
_COMM_TYPES = {"nccl_collective"}
_MEMORY_TYPES = {"mem_copy", "mem_alloc"}
_HOST_TYPES = {"host_delay"}
_STREAM_DEVICE_TYPES = _COMPUTE_TYPES | _COMM_TYPES | {"mem_copy"}
_CUDA_EVENT_HOST_WAIT_APIS = {"cudaEventSynchronize"}
_THREAD_SYNC_APIS = _CUDA_EVENT_HOST_WAIT_APIS | {"cudaStreamSynchronize", "cudaDeviceSynchronize"}
_CUDA_EVENT_RECORD_APIS = {"cudaEventRecord", "cudaEventRecordWithFlags"}

_CANONICAL_HOST_QUEUE_PID = -1
_CANONICAL_HOST_QUEUE_TID = -1

_HostKey = tuple[str, int, int]
_StreamKey = tuple[int, str]
_CudaWaitKey = tuple[int, str, int]
CollectiveAblationPredicate = Callable[[str, tuple["_PendingOp", ...]], bool]
StreamSerializationAblationPredicate = Callable[["_PendingOp"], bool]
StreamStartTimeOverride = Callable[["_PendingOp", float, "_ReplayScheduler"], float]
CudaEventWaitAblationPredicate = Callable[["_PendingOp"], bool]


_P2P_PAIR_SEQ_RE = re.compile(r"(?:^|\|)pair_seq:(\d+)(?:\||$)")
_GROUP_CALL_ORDINAL_RE = re.compile(r"(?:^|\|)call:(\d+)(?:\||$)")


_EDGE_TIME_TOLERANCE_US = 1e-6

_APPENDIX_AB_P2P_DIAGNOSTIC_ENV_KEYS = (
    "MAYA_ENABLE_P2P_PER_BLOCK_COMPONENT_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_P2P_PER_BLOCK_COMPONENT_DIAGNOSTICS",
)
_APPENDIX_AB_P2P_DIAGNOSTIC_SCHEMA_VERSION = (
    "appendix_ab_selected_p2p_kernel1_per_block_component_diagnostics_v1"
)
_APPENDIX_AB_ALLREDUCE_DIAGNOSTIC_ENV_KEYS = (
    "MAYA_ENABLE_APPENDIX_AB_SELECTED_NCCL_ALLREDUCE_PER_BLOCK_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_APPENDIX_AB_SELECTED_NCCL_ALLREDUCE_PER_BLOCK_DIAGNOSTICS",
)
_APPENDIX_AB_ALLREDUCE_DIAGNOSTIC_SCHEMA_VERSION = (
    "appendix_ab_selected_nccl_allreduce_per_block_component_diagnostics_v1"
)
_APPENDIX_AB_SELECTED_ALLREDUCE_MOTIF_KEY = (
    "boundary=ncclAllReduce|kernel=4-8|gemm=2-3|strided=2-3|send=0|recv=0|allreduce=1"
)
_APPENDIX_AB_SELECTED_ALLREDUCE_FAMILY_LABEL = (
    "nccl_allreduce_kernel4_8_gemm2_3_strided2_3"
)
_APPENDIX_AB_P2P_SELECTED_APIS = {"ncclSend", "ncclRecv"}
_APPENDIX_AB_P2P_COMPONENT_KEYS = (
    "provider_runtime",
    "hostDelay",
    "host_dispatch",
    "stream_queue_wait",
    "collective_wait",
    "cuda_event_wait",
    "host_sync_wait",
    "residual_unattributed",
)
_SHARED_PHASE_ANCHOR_CAUSAL_EDGE_ENV_KEYS = (
    "MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
)
_SHARED_PHASE_ANCHOR_SCHEMA_VERSION = (
    "shared_all_rank_phase_anchor_counterpart_replay_causal_edge_metadata_v1"
)
_SHARED_PHASE_ANCHOR_COMMON_BASIS_ENV_KEYS = (
    "MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
)
_SHARED_PHASE_ANCHOR_COMMON_BASIS_SCHEMA_VERSION = (
    "shared_phase_anchor_common_basis_key_fields_v1"
)
_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ENV_KEYS = (
    "MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_PHASE1_METADATA_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_PHASE1_METADATA_DIAGNOSTICS",
)
_GENERIC_REPLAY_PLACEMENT_ENVELOPE_SCHEMA_VERSION = (
    "generic_replay_placement_envelope_phase1_metadata_v1"
)
_COMPONENT_STRICT_COUNTERPART_ENV_KEYS = (
    "MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
)
_COMPONENT_STRICT_COUNTERPART_SCHEMA_VERSION = (
    "component_strict_counterpart_metadata_evidence_v1"
)
_CRITICAL_PATH_HOSTDELAY_BOUNDARY_SEMANTIC_CLASSIFICATION_ENV_KEYS = (
    "MAYA_ENABLE_CRITICAL_PATH_HOSTDELAY_BOUNDARY_SEMANTIC_CLASSIFICATION_METADATA_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_CRITICAL_PATH_HOSTDELAY_BOUNDARY_SEMANTIC_CLASSIFICATION_METADATA_DIAGNOSTICS",
)
_CRITICAL_PATH_HOSTDELAY_BOUNDARY_SEMANTIC_CLASSIFICATION_SCHEMA_VERSION = (
    "critical_path_hostDelay_boundary_semantic_classification_metadata_v1"
)
_CRITICAL_PATH_HOSTDELAY_BOUNDARY_SEMANTIC_CLASSIFICATION_FIELDS = (
    "hostdelay_semantic_classification_schema_version",
    "hostdelay_semantic_classification_opt_in_flag",
    "hostdelay_boundary_semantic_source_side",
    "hostdelay_boundary_family",
    "hostdelay_boundary_target_api",
    "hostdelay_boundary_rank_stream_queue_key",
    "hostdelay_origin_semantic_class",
    "hostdelay_origin_semantic_basis",
    "hostdelay_paper_visible_status",
    "hostdelay_classification_confidence",
    "hostdelay_paper_visible_host_overhead_us",
    "hostdelay_instrumentation_only_us",
    "hostdelay_already_counted_host_dispatch_us",
    "hostdelay_already_counted_provider_runtime_us",
    "hostdelay_unresolved_mixed_us",
    "hostdelay_same_occurrence_key",
    "hostdelay_same_occurrence_key_status",
    "hostdelay_predicted_count_once_group_id",
    "hostdelay_actual_count_once_group_id",
    "hostdelay_count_once_status",
    "hostdelay_nonoverlap_status",
    "hostdelay_wait_map_safety_status",
    "hostdelay_host_dispatch_overlap_status",
    "hostdelay_provider_runtime_overlap_status",
    "hostdelay_stream_queue_wait_overlap_status",
    "hostdelay_endpoint_duration_used_as_subtraction_delta",
    "hostdelay_runtime_used_as_substitution",
    "hostdelay_projection_overlap_used_as_subtraction_delta",
    "hostdelay_safe_to_use_as_repair_evidence",
    "hostdelay_safe_to_use_as_subtraction_delta",
    "hostdelay_repair_ready",
    "hostdelay_repair_ready_unavailable_reason",
)
_HOSTDELAY_OCCURRENCE_METADATA_EXPORT_FIELDS = (
    "hostdelay_occurrence_metadata_schema_version",
    "hostdelay_occurrence_metadata_opt_in_flag",
    "hostdelay_occurrence_metadata_env_flags",
    "stable_hostdelay_occurrence_id",
    "hostdelay_occurrence_id_basis",
    "hostdelay_occurrence_rank",
    "hostdelay_occurrence_pid",
    "hostdelay_occurrence_tid",
    "hostdelay_occurrence_thread_id",
    "hostdelay_occurrence_dispatch_key",
    "hostdelay_occurrence_host_dispatch_queue_id",
    "hostdelay_occurrence_stream_id",
    "hostdelay_occurrence_paper_valid_window_id",
    "hostdelay_occurrence_materialized_hostdelay_event_id",
    "hostdelay_occurrence_source",
    "hostdelay_occurrence_interval_start_ts_us",
    "hostdelay_occurrence_interval_end_ts_us",
    "hostdelay_occurrence_duration_us",
    "hostdelay_occurrence_timestamp_basis",
    "hostdelay_occurrence_timestamp_source",
    "hostdelay_occurrence_timestamp_clock_domain",
    "hostdelay_occurrence_interval_basis_status",
    "hostdelay_occurrence_raw_predecessor_event_id",
    "hostdelay_occurrence_raw_predecessor_api",
    "hostdelay_occurrence_raw_predecessor_op_type",
    "hostdelay_occurrence_raw_predecessor_ts_us",
    "hostdelay_occurrence_raw_predecessor_ordinal",
    "hostdelay_occurrence_raw_successor_event_id",
    "hostdelay_occurrence_raw_successor_api",
    "hostdelay_occurrence_raw_successor_op_type",
    "hostdelay_occurrence_raw_successor_ts_us",
    "hostdelay_occurrence_raw_successor_ordinal",
    "hostdelay_occurrence_semantic_predecessor_event_id",
    "hostdelay_occurrence_semantic_predecessor_api",
    "hostdelay_occurrence_semantic_predecessor_materialized",
    "hostdelay_occurrence_semantic_successor_event_id",
    "hostdelay_occurrence_semantic_successor_api",
    "hostdelay_occurrence_semantic_successor_materialized",
    "hostdelay_occurrence_raw_boundary_family",
    "hostdelay_occurrence_semantic_boundary_family",
    "hostdelay_occurrence_boundary_origin_kind",
    "hostdelay_occurrence_boundary_origin_evidence",
    "hostdelay_occurrence_boundary_visibility_kind",
    "hostdelay_occurrence_boundary_visibility_evidence",
    "hostdelay_occurrence_paper_visibility_class",
    "hostdelay_occurrence_paper_visibility_reason",
    "hostdelay_occurrence_instrumentation_only_evidence",
    "hostdelay_occurrence_control_plane_only_evidence",
    "hostdelay_occurrence_already_counted_elsewhere_evidence",
    "hostdelay_occurrence_host_dispatch_interval_id",
    "hostdelay_occurrence_host_dispatch_overlap_status",
    "hostdelay_occurrence_provider_runtime_interval_id",
    "hostdelay_occurrence_provider_runtime_overlap_status",
    "hostdelay_occurrence_stream_queue_wait_interval_id",
    "hostdelay_occurrence_stream_queue_wait_overlap_status",
    "hostdelay_occurrence_count_once_group_id",
    "hostdelay_occurrence_count_once_status",
    "hostdelay_occurrence_nonoverlap_status",
    "hostdelay_occurrence_cuda_event_record_id",
    "hostdelay_occurrence_cuda_event_wait_id",
    "hostdelay_occurrence_cuda_event_pair_id",
    "hostdelay_occurrence_cuda_event_wait_map_safety_status",
    "hostdelay_occurrence_collective_group_id",
    "hostdelay_occurrence_collective_member_id",
    "hostdelay_occurrence_collective_wait_edge_id",
    "hostdelay_occurrence_collective_wait_map_safety_status",
    "hostdelay_occurrence_stream_fifo_safety_status",
    "hostdelay_occurrence_replay_ordering_safety_status",
    "hostdelay_occurrence_fresh16_fresh8_comparable_join_key",
    "hostdelay_occurrence_fresh16_fresh8_join_key_basis",
    "hostdelay_occurrence_fresh16_fresh8_join_key_status",
    "hostdelay_occurrence_rank_workload_special_case_used",
    "hostdelay_occurrence_repair_ready",
    "hostdelay_occurrence_repair_ready_reason",
    "hostdelay_occurrence_safe_to_use_as_repair_evidence",
    "hostdelay_occurrence_safe_to_use_as_subtraction_delta",
    "hostdelay_occurrence_safe_delta_us",
    "hostdelay_occurrence_safe_delta_basis",
    "hostdelay_occurrence_observed_gap_us_context_only",
    "hostdelay_occurrence_join_metadata_schema_version",
    "hostdelay_occurrence_join_metadata_opt_in_flag",
    "hostdelay_occurrence_join_metadata_env_flags",
    "hostdelay_occurrence_join_metadata_scope",
    "hostdelay_occurrence_join_metadata_behavior_effect",
    "hostdelay_occurrence_boundary_origin_join_key",
    "hostdelay_occurrence_boundary_origin_join_key_basis",
    "hostdelay_occurrence_boundary_origin_status",
    "hostdelay_occurrence_boundary_origin_rule_id",
    "hostdelay_occurrence_boundary_visibility_join_key",
    "hostdelay_occurrence_boundary_visibility_join_key_basis",
    "hostdelay_occurrence_boundary_visibility_status",
    "hostdelay_occurrence_boundary_visibility_rule_id",
    "hostdelay_occurrence_count_once_join_key",
    "hostdelay_occurrence_count_once_join_key_basis",
    "hostdelay_occurrence_count_once_evidence",
    "hostdelay_occurrence_nonoverlap_join_key",
    "hostdelay_occurrence_nonoverlap_join_key_basis",
    "hostdelay_occurrence_nonoverlap_evidence",
    "hostdelay_occurrence_host_dispatch_interval_join_status",
    "hostdelay_occurrence_provider_runtime_interval_join_status",
    "hostdelay_occurrence_stream_queue_wait_interval_join_status",
    "hostdelay_occurrence_cuda_event_waitmap_join_key",
    "hostdelay_occurrence_cuda_event_waitmap_join_key_basis",
    "hostdelay_occurrence_cuda_event_wait_map_edge_id",
    "hostdelay_occurrence_cuda_event_wait_map_evidence",
    "hostdelay_occurrence_collective_waitmap_join_key",
    "hostdelay_occurrence_collective_waitmap_join_key_basis",
    "hostdelay_occurrence_collective_provider_interval_id",
    "hostdelay_occurrence_collective_wait_map_evidence",
    "hostdelay_occurrence_join_repair_ready",
    "hostdelay_occurrence_join_safe_to_use_as_repair_evidence",
    "hostdelay_occurrence_join_safe_to_use_as_subtraction_delta",
    "hostdelay_occurrence_join_safe_delta_us",
    "hostdelay_occurrence_join_safe_delta_basis",
    "hostdelay_occurrence_join_repair_ready_reason",
)
_COLLECTIVE_EVENT_POLLING_METADATA_EXPORT_FIELDS = (
    "collective_event_polling_metadata_schema_version",
    "collective_event_polling_metadata_opt_in_flag",
    "collective_event_polling_metadata_env_flags",
    "collective_event_polling_metadata_behavior_effect",
    "collective_event_polling_occurrence_join_key",
    "collective_event_polling_occurrence_join_key_basis",
    "collective_event_polling_raw_boundary_family",
    "collective_event_polling_semantic_boundary_family",
    "collective_event_polling_raw_predecessor_event_id",
    "collective_event_polling_raw_successor_event_id",
    "collective_event_polling_semantic_predecessor_event_id",
    "collective_event_polling_semantic_successor_event_id",
    "collective_event_polling_target_family",
    "collective_event_polling_rank",
    "collective_event_polling_pid",
    "collective_event_polling_tid",
    "collective_event_polling_dispatch_key",
    "collective_event_polling_stream_id",
    "collective_event_polling_stream_resource_id",
    "collective_event_polling_stream_namespace_basis",
    "collective_event_polling_collective_group_id",
    "collective_event_polling_collective_member_id",
    "collective_event_polling_collective_api",
    "collective_event_polling_collective_call_order",
    "collective_event_polling_communicator_id",
    "collective_event_polling_communicator_size",
    "collective_event_polling_participant_count",
    "collective_event_polling_provider_interval_id",
    "collective_event_polling_cuda_event_handle",
    "collective_event_polling_cuda_event_version",
    "collective_event_polling_cuda_event_record_id",
    "collective_event_polling_cuda_event_query_id",
    "collective_event_polling_cuda_stream_wait_event_id",
    "collective_event_polling_cuda_event_pair_id",
    "collective_event_polling_boundary_origin_kind",
    "collective_event_polling_boundary_origin_status",
    "collective_event_polling_boundary_origin_evidence",
    "collective_event_polling_boundary_visibility_kind",
    "collective_event_polling_boundary_visibility_status",
    "collective_event_polling_boundary_visibility_evidence",
    "collective_event_polling_wait_map_edge_id",
    "collective_event_polling_wait_map_edge_kind",
    "collective_event_polling_release_source_id",
    "collective_event_polling_release_source_kind",
    "collective_event_polling_release_timing_basis",
    "collective_event_polling_wait_map_release_status",
    "collective_event_polling_count_once_group_id",
    "collective_event_polling_count_once_status",
    "collective_event_polling_nonoverlap_status",
    "collective_event_polling_nonoverlap_evidence",
    "collective_event_polling_repair_ready",
    "collective_event_polling_safe_to_use_as_repair_evidence",
    "collective_event_polling_safe_to_use_as_subtraction_delta",
    "collective_event_polling_safe_delta_us",
    "collective_event_polling_repair_ready_reason",
)
_COLLECTIVE_EVENT_POLLING_REPLAY_WAITMAP_RELEASE_METADATA_ENV_KEYS = (
    "MAYA_ENABLE_COLLECTIVE_EVENT_POLLING_REPLAY_WAITMAP_RELEASE_METADATA_EXPORT",
    "FLEXSIM_MAYA_ENABLE_COLLECTIVE_EVENT_POLLING_REPLAY_WAITMAP_RELEASE_METADATA_EXPORT",
)
_COLLECTIVE_EVENT_POLLING_REPLAY_WAITMAP_RELEASE_METADATA_SCHEMA_VERSION = (
    "collective_event_polling_replay_waitmap_release_metadata_v1"
)
_COLLECTIVE_EVENT_POLLING_REPLAY_WAITMAP_TARGET_APIS = frozenset(
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
_EVENT_POLLING_BOUNDARY_METADATA_EXPORT_FIELDS = (
    "event_polling_boundary_metadata_schema_version",
    "event_polling_boundary_metadata_opt_in_flag",
    "event_polling_boundary_metadata_env_flags",
    "event_polling_boundary_behavior_effect",
    "event_polling_boundary_occurrence_id",
    "event_polling_boundary_occurrence_id_basis",
    "event_polling_boundary_materialized_hostdelay_event_id",
    "event_polling_boundary_rank",
    "event_polling_boundary_pid",
    "event_polling_boundary_tid",
    "event_polling_boundary_thread_id",
    "event_polling_boundary_dispatch_key",
    "event_polling_boundary_interval_start_ts_us",
    "event_polling_boundary_interval_end_ts_us",
    "event_polling_boundary_duration_us_context_only",
    "event_polling_boundary_timestamp_basis",
    "event_polling_boundary_timestamp_source",
    "event_polling_boundary_clock_domain",
    "event_polling_boundary_interval_basis_status",
    "event_polling_boundary_raw_predecessor_event_id",
    "event_polling_boundary_raw_predecessor_api",
    "event_polling_boundary_raw_predecessor_ordinal",
    "event_polling_boundary_raw_successor_event_id",
    "event_polling_boundary_raw_successor_api",
    "event_polling_boundary_raw_successor_ordinal",
    "event_polling_boundary_semantic_predecessor_event_id",
    "event_polling_boundary_semantic_predecessor_api",
    "event_polling_boundary_semantic_predecessor_materialized",
    "event_polling_boundary_semantic_successor_event_id",
    "event_polling_boundary_semantic_successor_api",
    "event_polling_boundary_semantic_successor_materialized",
    "event_polling_boundary_raw_family",
    "event_polling_boundary_semantic_family",
    "event_polling_boundary_raw_semantic_pair",
    "event_polling_boundary_target_class",
    "event_polling_boundary_polling_class",
    "event_polling_boundary_origin_kind",
    "event_polling_boundary_origin_status",
    "event_polling_boundary_origin_rule_id",
    "event_polling_boundary_origin_evidence",
    "event_polling_boundary_visibility_kind",
    "event_polling_boundary_visibility_status",
    "event_polling_boundary_visibility_rule_id",
    "event_polling_boundary_visibility_evidence",
    "event_polling_boundary_paper_visibility_class",
    "event_polling_boundary_paper_visibility_reason",
    "event_polling_boundary_candidate_control_plane_subregion_status",
    "event_polling_boundary_candidate_instrumentation_only_status",
    "event_polling_boundary_already_modeled_replay_waitmap_status",
    "event_polling_boundary_already_counted_elsewhere_status",
    "event_polling_boundary_unsafe_removable_status",
    "event_polling_boundary_hostdelay_occurrence_id",
    "event_polling_boundary_replay_waitmap_edge_id",
    "event_polling_boundary_replay_waitmap_edge_kind",
    "event_polling_boundary_collective_group_id",
    "event_polling_boundary_cuda_event_id",
    "event_polling_boundary_cuda_event_version",
    "event_polling_boundary_count_once_group_id",
    "event_polling_boundary_count_once_status",
    "event_polling_boundary_nonoverlap_status",
    "event_polling_boundary_wait_map_safety_status",
    "event_polling_boundary_fresh16_fresh8_comparable_join_key",
    "event_polling_boundary_fresh16_fresh8_join_key_status",
    "event_polling_boundary_repair_ready",
    "event_polling_boundary_repair_ready_reason",
    "event_polling_boundary_safe_to_use_as_repair_evidence",
    "event_polling_boundary_safe_to_use_as_subtraction_delta",
    "event_polling_boundary_safe_delta_us",
    "event_polling_boundary_safe_delta_basis",
    "event_polling_boundary_runtime_or_endpoint_substitution_used",
    "event_polling_boundary_hostdelay_shortening_used",
    "event_polling_boundary_rank_workload_special_case_used",
)
_BOUNDARY_ORIGIN_SUBREGION_PROOF_METADATA_EXPORT_ENV_KEYS = (
    "MAYA_ENABLE_BOUNDARY_ORIGIN_SUBREGION_PROOF_METADATA_EXPORT",
    "FLEXSIM_MAYA_ENABLE_BOUNDARY_ORIGIN_SUBREGION_PROOF_METADATA_EXPORT",
)
_BOUNDARY_ORIGIN_SUBREGION_PROOF_METADATA_EXPORT_FIELDS = (
    "boundary_origin_subregion_metadata_schema_version",
    "boundary_origin_subregion_metadata_opt_in_flag",
    "boundary_origin_subregion_metadata_env_flags",
    "boundary_origin_subregion_behavior_effect",
    "boundary_origin_subregion_occurrence_id",
    "boundary_origin_subregion_occurrence_id_basis",
    "boundary_origin_subregion_rank",
    "boundary_origin_subregion_pid",
    "boundary_origin_subregion_tid",
    "boundary_origin_subregion_dispatch_key",
    "boundary_origin_subregion_materialized_hostdelay_event_id",
    "boundary_origin_subregion_id",
    "boundary_origin_subregion_id_basis",
    "boundary_origin_subregion_kind",
    "boundary_origin_subregion_candidate_role",
    "boundary_origin_subregion_start_ts_us",
    "boundary_origin_subregion_end_ts_us",
    "boundary_origin_subregion_duration_us_context_only",
    "boundary_origin_subregion_timestamp_basis",
    "boundary_origin_subregion_clock_domain",
    "boundary_origin_subregion_strict_extent_status",
    "boundary_origin_subregion_strict_extent_source",
    "boundary_origin_subregion_strict_extent_evidence",
    "boundary_origin_subregion_raw_predecessor_event_id",
    "boundary_origin_subregion_raw_predecessor_api",
    "boundary_origin_subregion_raw_predecessor_role",
    "boundary_origin_subregion_raw_successor_event_id",
    "boundary_origin_subregion_raw_successor_api",
    "boundary_origin_subregion_raw_successor_role",
    "boundary_origin_subregion_semantic_predecessor_event_id",
    "boundary_origin_subregion_semantic_predecessor_api",
    "boundary_origin_subregion_semantic_predecessor_materialized",
    "boundary_origin_subregion_semantic_successor_event_id",
    "boundary_origin_subregion_semantic_successor_api",
    "boundary_origin_subregion_semantic_successor_materialized",
    "boundary_origin_subregion_raw_family",
    "boundary_origin_subregion_semantic_family",
    "boundary_origin_subregion_raw_semantic_pair",
    "boundary_origin_subregion_origin_kind",
    "boundary_origin_subregion_origin_status",
    "boundary_origin_subregion_origin_rule_id",
    "boundary_origin_subregion_origin_evidence",
    "boundary_origin_subregion_visibility_kind",
    "boundary_origin_subregion_visibility_status",
    "boundary_origin_subregion_visibility_rule_id",
    "boundary_origin_subregion_paper_visibility_class",
    "boundary_origin_subregion_paper_visibility_reason",
    "boundary_origin_subregion_strict_proof_source",
    "boundary_origin_subregion_strict_proof_status",
    "boundary_origin_subregion_strict_proof_evidence",
    "boundary_origin_subregion_count_once_group_id",
    "boundary_origin_subregion_count_once_status",
    "boundary_origin_subregion_count_once_evidence",
    "boundary_origin_subregion_nonoverlap_group_id",
    "boundary_origin_subregion_nonoverlap_status",
    "boundary_origin_subregion_nonoverlap_evidence",
    "boundary_origin_subregion_wait_map_safety_status",
    "boundary_origin_subregion_wait_map_edge_id",
    "boundary_origin_subregion_replay_ordering_safety_status",
    "boundary_origin_subregion_stream_fifo_safety_status",
    "boundary_origin_subregion_collective_wait_safety_status",
    "boundary_origin_subregion_fresh16_fresh8_comparable_join_key",
    "boundary_origin_subregion_fresh16_fresh8_join_key_basis",
    "boundary_origin_subregion_fresh16_fresh8_join_key_status",
    "boundary_origin_subregion_fresh8_preservation_risk_status",
    "boundary_origin_subregion_fresh8_preservation_evidence",
    "boundary_origin_subregion_repair_ready",
    "boundary_origin_subregion_repair_ready_reason",
    "boundary_origin_subregion_safe_to_use_as_repair_evidence",
    "boundary_origin_subregion_safe_to_use_as_subtraction_delta",
    "boundary_origin_subregion_safe_delta_us",
    "boundary_origin_subregion_safe_delta_basis",
    "boundary_origin_subregion_runtime_or_endpoint_substitution_used",
    "boundary_origin_subregion_hostdelay_shortening_used",
    "boundary_origin_subregion_rank_workload_special_case_used",
)
_STRICT_SUBREGION_EXTENT_SOURCE_METADATA_EXPORT_ENV_KEYS = (
    "MAYA_ENABLE_STRICT_SUBREGION_EXTENT_SOURCE_METADATA_EXPORT",
    "FLEXSIM_MAYA_ENABLE_STRICT_SUBREGION_EXTENT_SOURCE_METADATA_EXPORT",
)
_STRICT_SUBREGION_EXTENT_SOURCE_METADATA_EXPORT_FIELDS = (
    "strict_subregion_extent_metadata_schema_version",
    "strict_subregion_extent_metadata_opt_in_flag",
    "strict_subregion_extent_metadata_env_flags",
    "strict_subregion_extent_behavior_effect",
    "strict_subregion_extent_occurrence_id",
    "strict_subregion_extent_occurrence_id_basis",
    "strict_subregion_extent_boundary_id",
    "strict_subregion_extent_boundary_id_basis",
    "strict_subregion_extent_rank",
    "strict_subregion_extent_pid",
    "strict_subregion_extent_tid",
    "strict_subregion_extent_dispatch_key",
    "strict_subregion_extent_materialized_hostdelay_event_id",
    "strict_subregion_extent_raw_predecessor_event_id",
    "strict_subregion_extent_raw_predecessor_api",
    "strict_subregion_extent_raw_successor_event_id",
    "strict_subregion_extent_raw_successor_api",
    "strict_subregion_extent_semantic_predecessor_event_id",
    "strict_subregion_extent_semantic_predecessor_api",
    "strict_subregion_extent_semantic_predecessor_materialized",
    "strict_subregion_extent_semantic_successor_event_id",
    "strict_subregion_extent_semantic_successor_api",
    "strict_subregion_extent_semantic_successor_materialized",
    "strict_subregion_extent_raw_family",
    "strict_subregion_extent_semantic_family",
    "strict_subregion_extent_raw_semantic_pair",
    "strict_subregion_extent_target_family_class",
    "strict_subregion_extent_candidate_subregion_id",
    "strict_subregion_extent_candidate_subregion_kind",
    "strict_subregion_extent_candidate_subregion_role",
    "strict_subregion_extent_start_ts_us",
    "strict_subregion_extent_end_ts_us",
    "strict_subregion_extent_duration_us_context_only",
    "strict_subregion_extent_clock_domain",
    "strict_subregion_extent_timestamp_basis",
    "strict_subregion_extent_timestamp_source_kind",
    "strict_subregion_extent_timestamp_source_id",
    "strict_subregion_extent_source_surface",
    "strict_subregion_extent_source_producer",
    "strict_subregion_extent_source_capture_phase",
    "strict_subregion_extent_source_is_non_perturbing",
    "strict_subregion_extent_source_perturbation_risk",
    "strict_subregion_extent_source_uses_runtime_endpoint_substitution",
    "strict_subregion_extent_source_uses_measured_actual_runtime",
    "strict_subregion_extent_source_uses_hostdelay_shortening",
    "strict_subregion_extent_source_proof_status",
    "strict_subregion_extent_source_proof_reason",
    "strict_subregion_extent_source_evidence",
    "strict_subregion_extent_origin_kind",
    "strict_subregion_extent_origin_status",
    "strict_subregion_extent_origin_rule_id",
    "strict_subregion_extent_origin_evidence",
    "strict_subregion_extent_visibility_kind",
    "strict_subregion_extent_visibility_status",
    "strict_subregion_extent_visibility_rule_id",
    "strict_subregion_extent_paper_visibility_class",
    "strict_subregion_extent_paper_visibility_reason",
    "strict_subregion_extent_already_modeled_replay_status",
    "strict_subregion_extent_instrumentation_only_status",
    "strict_subregion_extent_control_plane_only_status",
    "strict_subregion_extent_count_once_group_id",
    "strict_subregion_extent_count_once_status",
    "strict_subregion_extent_count_once_evidence",
    "strict_subregion_extent_nonoverlap_group_id",
    "strict_subregion_extent_nonoverlap_status",
    "strict_subregion_extent_nonoverlap_evidence",
    "strict_subregion_extent_host_dispatch_overlap_status",
    "strict_subregion_extent_provider_runtime_overlap_status",
    "strict_subregion_extent_stream_queue_wait_overlap_status",
    "strict_subregion_extent_cuda_event_wait_map_edge_id",
    "strict_subregion_extent_cuda_event_wait_map_safety_status",
    "strict_subregion_extent_collective_wait_map_edge_id",
    "strict_subregion_extent_collective_wait_map_safety_status",
    "strict_subregion_extent_replay_ordering_safety_status",
    "strict_subregion_extent_stream_fifo_safety_status",
    "strict_subregion_extent_fresh16_fresh8_comparable_join_key",
    "strict_subregion_extent_fresh16_fresh8_join_key_basis",
    "strict_subregion_extent_fresh16_fresh8_join_key_status",
    "strict_subregion_extent_fresh8_preservation_risk_status",
    "strict_subregion_extent_fresh8_preservation_evidence",
    "strict_subregion_extent_repair_ready",
    "strict_subregion_extent_repair_ready_reason",
    "strict_subregion_extent_safe_to_use_as_repair_evidence",
    "strict_subregion_extent_safe_to_use_as_subtraction_delta",
    "strict_subregion_extent_safe_delta_us",
    "strict_subregion_extent_safe_delta_basis",
    "strict_subregion_extent_runtime_or_endpoint_substitution_used",
    "strict_subregion_extent_hostdelay_shortening_used",
    "strict_subregion_extent_rank_workload_special_case_used",
)
_CRITICAL_PATH_HOSTDELAY_BOUNDARY_SEMANTIC_TARGET_APIS = {
    "cudaLaunchKernel",
    "cublasGemmEx",
    "cublasGemmStridedBatchedEx",
}
_CRITICAL_PATH_HOSTDELAY_BOUNDARY_CONTROL_OR_POLLING_APIS = {
    "cudaEventQuery",
    "cudaGetLastError",
    "cudaPeekAtLastError",
}
_GEMM_ADJACENT_HOSTDELAY_EXPORT_FIELDS = (
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
    "gemm_adjacent_actual_timing_unavailable_reason",
    "gemm_adjacent_actual_endpoint_context_only",
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
    "gemm_adjacent_counterpart_join_key",
    "gemm_adjacent_counterpart_join_basis",
    "gemm_adjacent_counterpart_join_attempted_during_capture",
    "gemm_adjacent_counterpart_join_status",
    "gemm_adjacent_counterpart_join_confidence",
    "gemm_adjacent_counterpart_unavailable_reason",
    "gemm_adjacent_predicted_replay_interval_id",
    "gemm_adjacent_predicted_replay_component_kind",
    "gemm_adjacent_predicted_replay_resource_kind",
    "gemm_adjacent_predicted_replay_resource_id",
    "gemm_adjacent_predicted_start_us",
    "gemm_adjacent_predicted_end_us",
    "gemm_adjacent_predicted_duration_us",
    "gemm_adjacent_actual_counterpart_row_id",
    "gemm_adjacent_actual_counterpart_join_key",
    "gemm_adjacent_actual_counterpart_join_status",
    "gemm_adjacent_actual_counterpart_join_basis",
    "gemm_adjacent_actual_counterpart_join_confidence",
    "gemm_adjacent_actual_counterpart_unavailable_reason",
    "gemm_adjacent_actual_timing_status",
    "gemm_adjacent_actual_start_us",
    "gemm_adjacent_actual_end_us",
    "gemm_adjacent_actual_duration_us",
    "gemm_adjacent_actual_timing_basis",
    "gemm_adjacent_actual_endpoint_timestamps_used_as_strict_timing",
    "gemm_adjacent_actual_runtime_direct_substitution",
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
    "gemm_adjacent_predicted_count_once_group_id",
    "gemm_adjacent_predicted_count_once_interval_id",
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
)
_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_EXPORT_FIELDS = (
    "cuda_gemm_hostdispatch_strict_occurrence_gap_schema_version",
    "cuda_gemm_hostdispatch_strict_occurrence_gap_opt_in_flag",
    "cuda_gemm_hostdispatch_strict_occurrence_gap_source_side",
    "cuda_gemm_hostdispatch_strict_occurrence_gap_actual_row_id",
    "cuda_gemm_hostdispatch_strict_occurrence_gap_actual_endpoint_context_only",
    "cuda_gemm_hostdispatch_strict_occurrence_gap_predicted_row_id",
    "strict_occurrence_common_basis_key",
    "strict_occurrence_material_without_embedded_algo",
    "strict_occurrence_boundary_target_side",
    "strict_occurrence_endpoint_identity_projection_key",
    "strict_occurrence_boundary_target_side_projection_key",
    "strict_occurrence_projection_keys_status",
    "strict_occurrence_projection_keys_basis",
    "strict_occurrence_projection_keys_repair_ready",
    "strict_occurrence_key_parts",
    "strict_occurrence_count_basis_side",
    "paper_valid_window_id",
    "rank",
    "host_dispatch_queue_id",
    "api_family",
    "component_role",
    "api_sequence_ordinal_in_window",
    "host_queue_sequence_ordinal_in_window",
    "stream_sequence_ordinal_in_window",
    "material_signature",
    "algorithm",
    "gemm_shape_signature",
    "boundary_family",
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
    "predicted_stream_resource_id",
    "actual_stream_resource_id",
    "stream_namespace_basis",
    "stream_alignment_status",
    "exact_stream_identity_proven",
    "default_stream_equivalence_reviewed",
    "predicted_count_once_group_id",
    "actual_count_once_group_id",
    "count_once_status",
    "nonoverlap_status",
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
    "wait_map_safety_status",
    "predicted_wait_map_edge_ids",
    "actual_wait_release_source_status",
    "dependency_release_timing_preserved",
    "cuda_event_wait_safety_status",
    "collective_wait_safety_status",
    "stream_queue_wait_safety_status",
    "strict_occurrence_join_ready",
    "strict_actual_timing_or_mechanical_split_ready",
    "strict_apples_to_apples_delta_ready",
    "repair_ready",
    "safe_to_use_as_repair_evidence",
    "safe_to_use_as_subtraction_delta",
    "safe_to_use_for_runtime_substitution",
    "safe_to_use_for_endpoint_timestamp_substitution",
)
_SHARED_PHASE_ANCHOR_SELECTED_APIS = {"ncclSend", "ncclRecv", "ncclAllReduce"}
_SHARED_PHASE_ANCHOR_EDGE_KIND_MAP = {
    "host_order": "host_order",
    "stream_order": "stream_order",
    "stream_queue_wait_release": "stream_queue_wait",
    "cuda_event_wait": "cuda_event_wait",
    "collective_wait": "collective_wait",
    "host_sync_wait": "host_sync_wait",
    "device_sync_wait": "device_sync_wait",
    "host_dispatch": "host_dispatch",
    "rank_completion_aggregation_context": "rank_completion_aggregation_context",
}

_HOST_CONTROL_PAPER_WINDOW_DEFAULT = {
    "window_id": None,
    "in_paper_valid_window": False,
    "window_source": None,
    "start_ts": None,
    "end_ts": None,
    "is_paper_valid_step_window": False,
    "membership_basis": "unavailable",
    "unavailable_reason": "paper_valid_window_membership_not_exported",
}


def pair_seq_collective_ablation_predicate(
    pair_sequences: set[int] | frozenset[int],
) -> CollectiveAblationPredicate:
    """Return a diagnostic predicate for ablating selected NCCL P2P pair sequences.

    This is an engineering diagnostic hook, not a production repair: matched
    collectives bypass the cross-rank participant barrier with participant-local
    stream execution at the current scheduler time.  This preserves causal,
    monotonic replay time and measures an upper-bound effect of removing the
    selected P2P barrier.
    """

    selected = frozenset(int(seq) for seq in pair_sequences)

    def _predicate(group_id: str, _ready_waiters: tuple["_PendingOp", ...]) -> bool:
        if "ncclP2P" not in group_id:
            return False
        match = _P2P_PAIR_SEQ_RE.search(group_id)
        if match is None:
            return False
        return int(match.group(1)) in selected

    return _predicate


def _base_simulated_event_id(event_id: str | None) -> str | None:
    if event_id is None:
        return None
    if event_id.endswith(":host_dispatch"):
        return event_id[: -len(":host_dispatch")]
    return event_id


def _diagnostic_edge_id(edge_kind: str, predecessor_id: str | None, successor_id: str, source: str) -> str:
    import hashlib

    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"{edge_kind}:{predecessor_id or 'none'}->{successor_id}:{digest}"


def _as_simulated_dict(event: SimulatedEvent) -> dict[str, Any]:
    return asdict(event)


def _is_missing_metadata(value: object) -> bool:
    return value is None or value == ""


def _event_source_value(event: AnnotatedEvent) -> str:
    return str(getattr(event.source, "value", event.source))


def _hostdelay_source_value(event: AnnotatedEvent) -> str:
    extras = event.extras
    for key in ("hostdelay_source", "hostdelay_materialization_source", "materialization_source"):
        value = extras.get(key)
        if not _is_missing_metadata(value):
            return str(value)
    boundary = extras.get("boundary")
    if boundary == "step_start":
        return "leading_step_gap"
    if boundary == "step_end":
        return "trailing_step_gap"
    if event.api != "__hostDelay__":
        return "replay_export"
    if any(not _is_missing_metadata(extras.get(key)) for key in ("observed_gap_us", "raw_current_event_id")):
        return "collate_host_gap"
    return "unavailable"


def _metadata_or_default(extras: dict[str, Any], key: str, default: Any) -> Any:
    value = extras.get(key)
    return default if _is_missing_metadata(value) else value


def _first_metadata_value(*values: object, default: Any = None) -> Any:
    for value in values:
        if not _is_missing_metadata(value):
            return value
    return default


def _metadata_map(value: object, default: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = dict(default or {})
    if isinstance(value, dict):
        resolved.update(value)
    return resolved


def _metadata_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _env_flag_enabled(*keys: str) -> bool:
    for key in keys:
        if os.environ.get(key) == "1":
            return True
    return False


def _appendix_ab_p2p_component_diagnostics_enabled() -> bool:
    return _env_flag_enabled(*_APPENDIX_AB_P2P_DIAGNOSTIC_ENV_KEYS)


def _appendix_ab_allreduce_component_diagnostics_enabled() -> bool:
    return _env_flag_enabled(*_APPENDIX_AB_ALLREDUCE_DIAGNOSTIC_ENV_KEYS)


def _shared_phase_anchor_diagnostics_enabled() -> bool:
    return _env_flag_truthy(*_SHARED_PHASE_ANCHOR_CAUSAL_EDGE_ENV_KEYS)


def _env_flag_truthy(*keys: str) -> bool:
    truthy = {"1", "true", "yes", "on"}
    for key in keys:
        value = os.environ.get(key)
        if value is not None and value.strip().lower() in truthy:
            return True
    return False


def _shared_phase_anchor_common_basis_diagnostics_enabled() -> bool:
    return _env_flag_truthy(*_SHARED_PHASE_ANCHOR_COMMON_BASIS_ENV_KEYS)


def _generic_replay_placement_envelope_diagnostics_enabled() -> bool:
    return _env_flag_truthy(*_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ENV_KEYS)


def _component_strict_counterpart_diagnostics_enabled() -> bool:
    return _env_flag_truthy(*_COMPONENT_STRICT_COUNTERPART_ENV_KEYS)


def _critical_path_hostdelay_semantic_classification_enabled() -> bool:
    return _env_flag_truthy(
        *_CRITICAL_PATH_HOSTDELAY_BOUNDARY_SEMANTIC_CLASSIFICATION_ENV_KEYS
    )


def _collective_event_polling_replay_waitmap_release_metadata_enabled() -> bool:
    return _env_flag_truthy(
        *_COLLECTIVE_EVENT_POLLING_REPLAY_WAITMAP_RELEASE_METADATA_ENV_KEYS
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


def _boundary_family_parts(family: object) -> tuple[str, ...]:
    if _is_missing_metadata(family):
        return ()
    return tuple(part.strip() for part in str(family).split("->") if part.strip())


def _critical_path_hostdelay_boundary_family(metadata: dict[str, Any]) -> str:
    return str(
        _first_metadata_value(
            metadata.get("materialized_boundary_family"),
            metadata.get("raw_boundary_family"),
            metadata.get("host_control_boundary_family"),
            metadata.get("actual_boundary_family"),
            default="unavailable",
        )
    )


def _critical_path_hostdelay_boundary_target_api(
    metadata: dict[str, Any],
    family_parts: tuple[str, ...],
) -> str:
    for key in (
        "current_materialized_api",
        "raw_current_api",
        "host_control_envelope_current_api",
        "host_control_boundary_current_api",
    ):
        value = metadata.get(key)
        if not _is_missing_metadata(value):
            return str(value)
    for api in reversed(family_parts):
        if api in _CRITICAL_PATH_HOSTDELAY_BOUNDARY_SEMANTIC_TARGET_APIS:
            return api
    return family_parts[-1] if family_parts else "unavailable"


def _critical_path_hostdelay_stream_value(metadata: dict[str, Any]) -> str:
    for key in (
        "host_control_envelope_stream_id",
        "gemm_adjacent_current_stream_resource_id",
        "gemm_adjacent_current_stream_id",
        "predicted_stream_resource_id",
        "stream_id",
    ):
        value = metadata.get(key)
        if not _is_missing_metadata(value):
            return str(value)
    return "unavailable"


def _critical_path_hostdelay_rank_stream_queue_key(
    metadata: dict[str, Any],
    *,
    event: AnnotatedEvent,
) -> str:
    queue = _metadata_or_default(metadata, "host_dispatch_queue_id", "unavailable")
    stream = _critical_path_hostdelay_stream_value(metadata)
    return f"rank:{int(event.rank)}|queue:{queue}|stream:{stream}"


def _hostdelay_semantic_classification_target(
    event: AnnotatedEvent,
    metadata: dict[str, Any],
    family_parts: tuple[str, ...],
    target_api: str,
) -> bool:
    if event.api != "__hostDelay__":
        return False
    if target_api in _CRITICAL_PATH_HOSTDELAY_BOUNDARY_SEMANTIC_TARGET_APIS:
        return True
    return any(
        part in _CRITICAL_PATH_HOSTDELAY_BOUNDARY_SEMANTIC_TARGET_APIS
        for part in family_parts
    )


def _hostdelay_explicit_split_us(metadata: dict[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    if _is_missing_metadata(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _hostdelay_has_explicit_reviewed_split(
    metadata: dict[str, Any],
    semantic_class: str,
) -> bool:
    if semantic_class in {
        "instrumentation_only",
        "already_counted_host_dispatch",
        "already_counted_provider_runtime",
    }:
        return True
    return any(
        _hostdelay_explicit_split_us(metadata, key) is not None
        for key in (
            "paper_visible_host_duration_us",
            "instrumentation_only_duration_us",
        )
    )


def _hostdelay_semantic_classification(
    metadata: dict[str, Any],
    family_parts: tuple[str, ...],
) -> tuple[str, str, str, str]:
    visibility_kind = metadata.get("boundary_visibility_kind")
    origin_kind = metadata.get("boundary_origin_kind")
    if (
        visibility_kind == "instrumentation_only"
        and _hostdelay_explicit_split_us(metadata, "instrumentation_only_duration_us") is not None
    ):
        return (
            "instrumentation_only",
            "explicit_segment_instrumentation_only",
            "proven_paper_invisible_instrumentation",
            "proven",
        )
    if (
        metadata.get("host_dispatch_overlap_status") == "proven_counted_once"
        and metadata.get("count_once_status") == "proven"
    ):
        return (
            "already_counted_host_dispatch",
            "explicit_host_dispatch_overlap_count_once_proof",
            "already_counted_elsewhere",
            "proven",
        )
    if (
        metadata.get("provider_runtime_overlap_status") == "proven_counted_once"
        and metadata.get("count_once_status") == "proven"
    ):
        return (
            "already_counted_provider_runtime",
            "explicit_provider_runtime_overlap_count_once_proof",
            "already_counted_elsewhere",
            "proven",
        )
    if (
        any(part in _CRITICAL_PATH_HOSTDELAY_BOUNDARY_CONTROL_OR_POLLING_APIS for part in family_parts)
        or origin_kind == "control_plane_or_polling"
    ):
        return (
            "control_plane_or_polling_unresolved",
            "control_polling_api_unresolved",
            "unresolved",
            "metadata_only_context",
        )
    if any(part in _CRITICAL_PATH_HOSTDELAY_BOUNDARY_SEMANTIC_TARGET_APIS for part in family_parts):
        return (
            "paper_visible_host_overhead",
            "figure6_inter_api_host_overhead",
            "paper_visible_by_default",
            "metadata_only_context",
        )
    return (
        "unresolved_mixed",
        "insufficient_boundary_visibility_provenance",
        "unresolved",
        "unavailable",
    )


def _critical_path_hostdelay_semantic_classification_metadata(
    event: AnnotatedEvent,
    metadata: dict[str, Any],
    observed_gap_us: object,
) -> dict[str, Any]:
    family = _critical_path_hostdelay_boundary_family(metadata)
    family_parts = _boundary_family_parts(family)
    target_api = _critical_path_hostdelay_boundary_target_api(metadata, family_parts)
    if not _hostdelay_semantic_classification_target(event, metadata, family_parts, target_api):
        return {}
    (
        semantic_class,
        semantic_basis,
        paper_visible_status,
        confidence,
    ) = _hostdelay_semantic_classification(metadata, family_parts)
    unresolved_mixed_us = _hostdelay_explicit_split_us(
        metadata,
        "unresolved_mixed_duration_us",
    )
    if unresolved_mixed_us is None and not _hostdelay_has_explicit_reviewed_split(
        metadata,
        semantic_class,
    ):
        unresolved_mixed_us = _hostdelay_explicit_split_us(
            {"observed_gap_us": observed_gap_us},
            "observed_gap_us",
        )
    same_occurrence_key = _first_metadata_value(
        metadata.get("selected_occurrence_id"),
        metadata.get("host_control_boundary_occurrence_id"),
        metadata.get("hostdelay_counterpart_key"),
        metadata.get("counterpart_join_key"),
    )
    return {
        "hostdelay_semantic_classification_schema_version": (
            _CRITICAL_PATH_HOSTDELAY_BOUNDARY_SEMANTIC_CLASSIFICATION_SCHEMA_VERSION
        ),
        "hostdelay_semantic_classification_opt_in_flag": True,
        "hostdelay_boundary_semantic_source_side": "predicted_hostDelay_boundary_metadata",
        "hostdelay_boundary_family": family,
        "hostdelay_boundary_target_api": target_api,
        "hostdelay_boundary_rank_stream_queue_key": (
            _critical_path_hostdelay_rank_stream_queue_key(metadata, event=event)
        ),
        "hostdelay_origin_semantic_class": semantic_class,
        "hostdelay_origin_semantic_basis": semantic_basis,
        "hostdelay_paper_visible_status": paper_visible_status,
        "hostdelay_classification_confidence": confidence,
        "hostdelay_paper_visible_host_overhead_us": _hostdelay_explicit_split_us(
            metadata,
            "paper_visible_host_duration_us",
        ),
        "hostdelay_instrumentation_only_us": _hostdelay_explicit_split_us(
            metadata,
            "instrumentation_only_duration_us",
        ),
        "hostdelay_already_counted_host_dispatch_us": None,
        "hostdelay_already_counted_provider_runtime_us": None,
        "hostdelay_unresolved_mixed_us": unresolved_mixed_us,
        "hostdelay_same_occurrence_key": same_occurrence_key,
        "hostdelay_same_occurrence_key_status": (
            "available_metadata_only_not_strict_counterpart"
            if not _is_missing_metadata(same_occurrence_key)
            else "unavailable"
        ),
        "hostdelay_predicted_count_once_group_id": _first_metadata_value(
            metadata.get("predicted_count_once_group_id"),
            event.extras.get("predicted_count_once_group_id"),
            metadata.get("gemm_adjacent_predicted_count_once_group_id"),
            event.extras.get("gemm_adjacent_predicted_count_once_group_id"),
        ),
        "hostdelay_actual_count_once_group_id": _first_metadata_value(
            metadata.get("actual_count_once_group_id"),
            event.extras.get("actual_count_once_group_id"),
            metadata.get("gemm_adjacent_actual_count_once_group_id"),
            event.extras.get("gemm_adjacent_actual_count_once_group_id"),
        ),
        "hostdelay_count_once_status": _metadata_or_default(
            metadata,
            "count_once_status",
            "unavailable",
        ),
        "hostdelay_nonoverlap_status": _metadata_or_default(
            metadata,
            "nonoverlap_status",
            _metadata_or_default(metadata, "double_counting_overlap_status", "unavailable"),
        ),
        "hostdelay_wait_map_safety_status": _metadata_or_default(
            metadata,
            "wait_map_safety_status",
            "unavailable",
        ),
        "hostdelay_host_dispatch_overlap_status": _metadata_or_default(
            metadata,
            "host_dispatch_overlap_status",
            "unavailable",
        ),
        "hostdelay_provider_runtime_overlap_status": _metadata_or_default(
            metadata,
            "provider_runtime_overlap_status",
            "unavailable",
        ),
        "hostdelay_stream_queue_wait_overlap_status": _metadata_or_default(
            metadata,
            "stream_wait_overlap_status",
            "unavailable",
        ),
        "hostdelay_endpoint_duration_used_as_subtraction_delta": False,
        "hostdelay_runtime_used_as_substitution": False,
        "hostdelay_projection_overlap_used_as_subtraction_delta": False,
        "hostdelay_safe_to_use_as_repair_evidence": False,
        "hostdelay_safe_to_use_as_subtraction_delta": False,
        "hostdelay_repair_ready": False,
        "hostdelay_repair_ready_unavailable_reason": (
            "semantic_classification_metadata_only_requires_strict_counterpart_"
            "count_once_nonoverlap_wait_map_review"
        ),
    }


def _p2p_kernel_bucket_from_metadata(event: AnnotatedEvent) -> str | None:
    extras = event.extras
    for key in (
        "appendix_ab_p2p_kernel_bucket",
        "kernel_bucket",
        "kernel_count_bucket",
        "selected_kernel_bucket",
        "semantic_kernel_bucket",
    ):
        value = extras.get(key)
        if not _is_missing_metadata(value):
            return str(value)
    for key in (
        "appendix_ab_p2p_motif_key",
        "motif_key",
        "predicted_motif_key",
        "family_key",
        "exact_representation",
    ):
        value = extras.get(key)
        if _is_missing_metadata(value):
            continue
        for part in str(value).split("|"):
            if part.startswith("kernel="):
                return part.split("=", 1)[1]
    return None


def _appendix_ab_selected_p2p_motif_key(event: AnnotatedEvent) -> str | None:
    if event.api not in _APPENDIX_AB_P2P_SELECTED_APIS:
        return None
    kernel_bucket = _p2p_kernel_bucket_from_metadata(event)
    if kernel_bucket != "1":
        return None
    for key in (
        "appendix_ab_p2p_motif_key",
        "motif_key",
        "predicted_motif_key",
        "family_key",
        "exact_representation",
    ):
        value = event.extras.get(key)
        if not _is_missing_metadata(value):
            return str(value)
    if event.api == "ncclSend":
        return "boundary=ncclSend|kernel=1|gemm=0|strided=0|send=1|recv=0|allreduce=0"
    return "boundary=ncclRecv|kernel=1|gemm=0|strided=0|send=0|recv=1|allreduce=0"


def _appendix_ab_selected_allreduce_motif_key(event: AnnotatedEvent) -> str | None:
    if event.api != "ncclAllReduce":
        return None
    for key in (
        "appendix_ab_allreduce_motif_key",
        "motif_key",
        "predicted_motif_key",
        "family_key",
        "exact_representation",
    ):
        value = event.extras.get(key)
        if _is_missing_metadata(value):
            continue
        text = str(value)
        if text == _APPENDIX_AB_SELECTED_ALLREDUCE_MOTIF_KEY:
            return text
        if _APPENDIX_AB_SELECTED_ALLREDUCE_MOTIF_KEY in text:
            return _APPENDIX_AB_SELECTED_ALLREDUCE_MOTIF_KEY
    for key in (
        "appendix_ab_allreduce_family_label",
        "appendix_b_family",
        "family_label",
        "api_or_family",
    ):
        value = event.extras.get(key)
        if _is_missing_metadata(value):
            continue
        if str(value) == _APPENDIX_AB_SELECTED_ALLREDUCE_FAMILY_LABEL:
            return _APPENDIX_AB_SELECTED_ALLREDUCE_MOTIF_KEY
    return None


def _stable_appendix_ab_p2p_block_id(event: AnnotatedEvent, simulated_event_id: str) -> str:
    seed = "|".join(
        [
            str(event.rank),
            str(event.ordinal),
            event.id,
            simulated_event_id,
            event.api,
            str(event.collective_group_id or ""),
        ]
    )
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"appendix_ab_p2p_kernel1:{event.rank}:{event.api}:{digest}"


def _stable_appendix_ab_allreduce_block_id(event: AnnotatedEvent, simulated_event_id: str) -> str:
    seed = "|".join(
        [
            str(event.rank),
            str(event.ordinal),
            event.id,
            simulated_event_id,
            event.api,
            str(event.collective_group_id or ""),
            _APPENDIX_AB_SELECTED_ALLREDUCE_FAMILY_LABEL,
        ]
    )
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"appendix_ab_nccl_allreduce_k4_8_g2_3_s2_3:{event.rank}:{event.api}:{digest}"


def _pair_seq_from_group_id(group_id: str | None) -> int | None:
    if group_id is None:
        return None
    match = _P2P_PAIR_SEQ_RE.search(group_id)
    return None if match is None else int(match.group(1))


def _group_call_ordinal_from_group_id(group_id: str | None) -> int | None:
    if group_id is None:
        return None
    match = _GROUP_CALL_ORDINAL_RE.search(group_id)
    return None if match is None else int(match.group(1))


def _safe_float_or_none(value: object) -> float | None:
    if _is_missing_metadata(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _appendix_ab_event_end_ts(event: AnnotatedEvent) -> int:
    event_ts = int(event.ts)
    raw_end_ts = event.extras.get("end_ts")
    if _is_missing_metadata(raw_end_ts):
        return event_ts
    try:
        parsed_end_ts = int(float(raw_end_ts))
    except (TypeError, ValueError):
        return event_ts
    return max(parsed_end_ts, event_ts)


def _appendix_ab_paper_valid_window_membership(
    trace: AnnotatedTrace,
    event: AnnotatedEvent,
) -> dict[str, Any]:
    window = trace.fidelity_windows.get(int(event.rank))
    if window is None:
        return {
            "window_id": None,
            "in_paper_valid_window": False,
            "window_source": None,
            "start_ts": None,
            "end_ts": None,
            "is_paper_valid_step_window": False,
            "membership_basis": "existing_trace_fidelity_window",
            "unavailable_reason": "no_paper_valid_step_window_for_rank",
        }
    event_ts = int(event.ts)
    event_end_ts = _appendix_ab_event_end_ts(event)
    in_window = event_ts <= int(window.end_ts) and event_end_ts >= int(window.start_ts)
    is_paper_valid = bool(window.is_paper_valid_step_window)
    return {
        "window_id": f"rank{int(event.rank)}:step_window",
        "in_paper_valid_window": bool(in_window and is_paper_valid),
        "window_source": str(window.source),
        "start_ts": int(window.start_ts),
        "end_ts": int(window.end_ts),
        "is_paper_valid_step_window": is_paper_valid,
        "membership_basis": "existing_trace_fidelity_window_ts_end_ts_overlap",
        "unavailable_reason": None if in_window and is_paper_valid else "event_outside_paper_valid_step_window",
    }


def _appendix_ab_trace_window_policy(trace: AnnotatedTrace) -> dict[str, Any]:
    paper_valid_windows = {
        int(rank): window
        for rank, window in trace.fidelity_windows.items()
        if window.is_paper_valid_step_window
    }
    return {
        "paper_valid_step_windows_authoritative": True,
        "step_window_assignment_bypassed": False,
        "paper_error_basis_changed": False,
        "paper_valid_window_count": len(paper_valid_windows),
        "paper_valid_window_sources": sorted({str(window.source) for window in paper_valid_windows.values()}),
        "actual_terms_require_in_window_aligned_counterpart": True,
        "outside_window_actual_timings_included": False,
        "actual_api_end_ts_used_as_wait_map_release": False,
        "actual_side_unavailable_policy": (
            "Actual Appendix B terms remain null unless a strict actual counterpart is aligned "
            "to the existing paper-valid step window; outside-window or unaligned actual timings "
            "are not included."
        ),
    }


_HOST_CONTROL_ENVELOPE_EXPORT_FIELDS = (
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
    "host_control_envelope_replay_interval_id",
    "host_control_envelope_count_once_interval_id",
    "host_control_envelope_replay_resource_kind",
    "host_control_envelope_replay_resource_id",
    "host_control_envelope_replay_predecessor_successor_status",
)

_HOST_CONTROL_VISIBILITY_OPT_IN_MARKERS = {
    "host_control_envelope_counterpart_schema_version",
    "host_control_visibility_schema_version",
    "host_control_boundary_counterpart_schema_version",
    "host_control_producer_visibility_schema_version",
    "launch_neighborhood_equivalence_schema_version",
}

_HOST_CONTROL_VISIBILITY_EXPORT_FIELDS = {
    *_HOST_CONTROL_ENVELOPE_EXPORT_FIELDS,
    *LAUNCH_NEIGHBORHOOD_EQUIVALENCE_EXPORT_FIELDS,
    "host_control_visibility_schema_version",
    "host_control_visibility_opt_in_flag",
    "selected_occurrence_id",
    "paper_valid_window_id",
    "paper_valid_window_unavailable_reason",
    "runtime_or_framework_duration_us",
    "payload_enrichment_duration_us",
    "trace_serialization_duration_us",
    "mis_materialized_duration_us",
    "split_sum_check_status",
    "split_tolerance_us",
    "classification_basis",
    "classification_unavailable_reason",
    "host_control_boundary_counterpart_schema_version",
    "host_control_boundary_row_id",
    "host_control_boundary_occurrence_id",
    "host_control_boundary_selection_status",
    "host_control_boundary_prev_raw_event_id",
    "host_control_boundary_current_raw_event_id",
    "host_control_boundary_family",
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
    "emulated_occurrence_id",
    "emulated_occurrence_id_unavailable_reason",
    "actual_trace_id",
    "actual_trace_id_unavailable_reason",
    "actual_rank",
    "actual_paper_valid_window_id",
    "actual_paper_valid_window_unavailable_reason",
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
    "wait_map_safety_status",
    "double_counting_overlap_status",
    "double_counting_overlap_unavailable_reason",
    "wait_map_non_overlap_unavailable_reason",
    "safe_to_use_as_repair_evidence",
    "safe_to_use_as_subtraction_delta",
    "safe_to_use_as_repair_evidence_reason",
    "safe_to_use_as_subtraction_delta_reason",
}


def _host_control_visibility_opted_in(extras: dict[str, Any]) -> bool:
    if extras.get("host_control_visibility_opt_in_flag") is True:
        return True
    if metadata_has_launch_neighborhood_equivalence(extras):
        return True
    return any(
        not _is_missing_metadata(extras.get(field))
        for field in _HOST_CONTROL_VISIBILITY_OPT_IN_MARKERS
    )


def _host_control_boundary_metadata(event: AnnotatedEvent) -> dict[str, Any]:
    extras = event.extras
    observed_gap_us = extras.get("observed_gap_us")
    if _is_missing_metadata(observed_gap_us) and event.api == "__hostDelay__":
        observed_gap_us = float(event.duration_us)
    boundary_origin_kind = _metadata_or_default(extras, "boundary_origin_kind", "unavailable")
    boundary_visibility_kind = _metadata_or_default(
        extras,
        "boundary_visibility_kind",
        "unavailable" if boundary_origin_kind == "unavailable" else boundary_origin_kind,
    )
    exact_counterpart_status = _metadata_or_default(extras, "exact_counterpart_status", "unavailable")
    selected_occurrence_id = _first_metadata_value(
        extras.get("selected_occurrence_id"),
        extras.get("host_control_boundary_occurrence_id"),
        extras.get("actual_counterpart_id"),
    )
    actual_launch_unavailable_reason = _metadata_or_default(
        extras,
        "actual_launch_unavailable_reason",
        "actual_launch_split_not_exported",
    )
    hostdelay_source = _hostdelay_source_value(event)
    metadata = {
        "hostdelay_event_id": event.id,
        "rank": int(event.rank),
        "ordinal": int(event.ordinal),
        "source": hostdelay_source,
        "hostdelay_source": hostdelay_source,
        "trace_source": _event_source_value(event),
        "observed_gap_us": observed_gap_us,
        "raw_prev_event_id": _metadata_or_default(extras, "raw_prev_event_id", None),
        "raw_prev_api": _metadata_or_default(extras, "raw_prev_api", None),
        "raw_prev_ts_us": _metadata_or_default(extras, "raw_prev_ts_us", None),
        "raw_prev_end_ts_us": _metadata_or_default(extras, "raw_prev_end_ts_us", None),
        "raw_prev_end_ts_source": _metadata_or_default(extras, "raw_prev_end_ts_source", "unavailable"),
        "raw_prev_host_duration_us": _metadata_or_default(extras, "raw_prev_host_duration_us", None),
        "raw_current_event_id": _metadata_or_default(extras, "raw_current_event_id", None),
        "raw_current_api": _metadata_or_default(extras, "raw_current_api", None),
        "raw_current_ts_us": _metadata_or_default(extras, "raw_current_ts_us", None),
        "raw_current_end_ts_us": _metadata_or_default(extras, "raw_current_end_ts_us", None),
        "raw_current_end_ts_source": _metadata_or_default(extras, "raw_current_end_ts_source", "unavailable"),
        "raw_current_host_duration_us": _metadata_or_default(extras, "raw_current_host_duration_us", None),
        "raw_boundary_family": _metadata_or_default(extras, "raw_boundary_family", "unavailable"),
        "previous_materialized_event_id": _metadata_or_default(extras, "previous_materialized_event_id", None),
        "previous_materialized_api": _metadata_or_default(extras, "previous_materialized_api", None),
        "current_materialized_event_id": _metadata_or_default(extras, "current_materialized_event_id", None),
        "current_materialized_api": _metadata_or_default(extras, "current_materialized_api", None),
        "materialized_boundary_family": _metadata_or_default(extras, "materialized_boundary_family", "unavailable"),
        "host_dispatch_queue_id": _metadata_or_default(extras, "host_dispatch_queue_id", "unavailable"),
        "host_machine_id": _metadata_or_default(extras, "host_machine_id", None),
        "host_timing_dispatch_scope": _metadata_or_default(extras, "host_timing_dispatch_scope", "unavailable"),
        "host_dispatch_model": _metadata_or_default(extras, "host_dispatch_model", "unavailable"),
        "paper_valid_window_membership": _metadata_map(
            extras.get("paper_valid_window_membership"),
            _HOST_CONTROL_PAPER_WINDOW_DEFAULT,
        ),
        "boundary_origin_kind": boundary_origin_kind,
        "boundary_visibility_kind": boundary_visibility_kind,
        "boundary_origin_field_sources": _metadata_map(extras.get("boundary_origin_field_sources")),
        "boundary_origin_conflicting_fields": _metadata_map(extras.get("boundary_origin_conflicting_fields")),
        "paper_visible_host_duration_us": _metadata_or_default(extras, "paper_visible_host_duration_us", None),
        "instrumentation_only_duration_us": _metadata_or_default(extras, "instrumentation_only_duration_us", None),
        "wrapper_internal_duration_us": _metadata_or_default(extras, "wrapper_internal_duration_us", None),
        "caller_visible_elapsed_us": _metadata_or_default(extras, "caller_visible_elapsed_us", None),
        "fake_api_body_duration_us": _metadata_or_default(extras, "fake_api_body_duration_us", None),
        "unresolved_mixed_duration_us": _metadata_or_default(extras, "unresolved_mixed_duration_us", None),
        "boundary_segment_schema_version": _metadata_or_default(extras, "boundary_segment_schema_version", None),
        "wrapper_segment_coverage": _metadata_or_default(extras, "wrapper_segment_coverage", "unavailable"),
        "boundary_visibility_segments": _metadata_list(extras.get("boundary_visibility_segments")),
        "host_control_visibility_schema_version": _metadata_or_default(
            extras,
            "host_control_visibility_schema_version",
            None,
        ),
        "host_control_visibility_opt_in_flag": bool(
            extras.get("host_control_visibility_opt_in_flag") is True
        ),
        **{
            key: _metadata_or_default(extras, key, None)
            for key in _HOST_CONTROL_ENVELOPE_EXPORT_FIELDS
        },
        **{
            key: _metadata_or_default(extras, key, None)
            for key in LAUNCH_NEIGHBORHOOD_EQUIVALENCE_EXPORT_FIELDS
        },
        "selected_occurrence_id": selected_occurrence_id,
        "paper_valid_window_id": _metadata_or_default(extras, "paper_valid_window_id", None),
        "paper_valid_window_unavailable_reason": _metadata_or_default(
            extras,
            "paper_valid_window_unavailable_reason",
            None,
        ),
        "runtime_or_framework_duration_us": _metadata_or_default(
            extras,
            "runtime_or_framework_duration_us",
            None,
        ),
        "payload_enrichment_duration_us": _metadata_or_default(
            extras,
            "payload_enrichment_duration_us",
            None,
        ),
        "trace_serialization_duration_us": _metadata_or_default(
            extras,
            "trace_serialization_duration_us",
            None,
        ),
        "mis_materialized_duration_us": _metadata_or_default(
            extras,
            "mis_materialized_duration_us",
            None,
        ),
        "split_sum_check_status": _metadata_or_default(
            extras,
            "split_sum_check_status",
            "unavailable",
        ),
        "split_tolerance_us": _metadata_or_default(extras, "split_tolerance_us", None),
        "classification_basis": _metadata_or_default(
            extras,
            "classification_basis",
            "unavailable",
        ),
        "classification_unavailable_reason": _metadata_or_default(
            extras,
            "classification_unavailable_reason",
            "mechanical_visibility_split_not_exported",
        ),
        "actual_host_dispatch_duration_us": _metadata_or_default(extras, "actual_host_dispatch_duration_us", None),
        "actual_launch_control_dispatch_us": _metadata_or_default(extras, "actual_launch_control_dispatch_us", None),
        "actual_launch_api_body_us": _metadata_or_default(extras, "actual_launch_api_body_us", None),
        "actual_launch_instrumentation_only_us": _metadata_or_default(
            extras,
            "actual_launch_instrumentation_only_us",
            None,
        ),
        "actual_launch_visibility_kind": _metadata_or_default(
            extras,
            "actual_launch_visibility_kind",
            "mixed_or_unresolved",
        ),
        "actual_launch_unavailable_reason": actual_launch_unavailable_reason,
        "host_control_boundary_counterpart_schema_version": _metadata_or_default(
            extras,
            "host_control_boundary_counterpart_schema_version",
            None,
        ),
        "host_control_boundary_row_id": _metadata_or_default(
            extras,
            "host_control_boundary_row_id",
            None,
        ),
        "host_control_boundary_occurrence_id": _metadata_or_default(
            extras,
            "host_control_boundary_occurrence_id",
            None,
        ),
        "host_control_boundary_selection_status": _metadata_or_default(
            extras,
            "host_control_boundary_selection_status",
            "unavailable",
        ),
        "host_control_boundary_prev_raw_event_id": _metadata_or_default(
            extras,
            "host_control_boundary_prev_raw_event_id",
            None,
        ),
        "host_control_boundary_current_raw_event_id": _metadata_or_default(
            extras,
            "host_control_boundary_current_raw_event_id",
            None,
        ),
        "host_control_boundary_family": _metadata_or_default(
            extras,
            "host_control_boundary_family",
            "unavailable",
        ),
        "host_control_visibility_split_status": _metadata_or_default(
            extras,
            "host_control_visibility_split_status",
            "unavailable",
        ),
        "host_control_visibility_split_unavailable_reason": _metadata_or_default(
            extras,
            "host_control_visibility_split_unavailable_reason",
            "host_control_boundary_counterpart_diagnostics_not_exported",
        ),
        "host_control_visibility_split_basis": _metadata_or_default(
            extras,
            "host_control_visibility_split_basis",
            "unavailable",
        ),
        "mechanical_visibility_split_status": _metadata_or_default(
            extras,
            "mechanical_visibility_split_status",
            "unavailable",
        ),
        "mechanical_visibility_split_unavailable_reason": _metadata_or_default(
            extras,
            "mechanical_visibility_split_unavailable_reason",
            "mechanical_visibility_split_not_exported",
        ),
        "host_control_producer_visibility_schema_version": _metadata_or_default(
            extras,
            "host_control_producer_visibility_schema_version",
            None,
        ),
        "host_control_producer_visibility_status": _metadata_or_default(
            extras,
            "host_control_producer_visibility_status",
            "unavailable",
        ),
        "host_control_producer_visibility_unavailable_reason": _metadata_or_default(
            extras,
            "host_control_producer_visibility_unavailable_reason",
            "host_control_producer_visibility_not_exported",
        ),
        "host_control_producer_visibility_basis": _metadata_or_default(
            extras,
            "host_control_producer_visibility_basis",
            "unavailable",
        ),
        "host_control_producer_visibility_segments": _metadata_list(
            extras.get("host_control_producer_visibility_segments")
        ),
        "host_control_producer_numeric_split_status": _metadata_or_default(
            extras,
            "host_control_producer_numeric_split_status",
            "unavailable",
        ),
        "host_control_producer_numeric_split_unavailable_reason": _metadata_or_default(
            extras,
            "host_control_producer_numeric_split_unavailable_reason",
            "real_api_body_or_instrumentation_split_not_exported",
        ),
        "host_control_producer_nonoverlap_status": _metadata_or_default(
            extras,
            "host_control_producer_nonoverlap_status",
            "unavailable",
        ),
        "host_control_producer_nonoverlap_unavailable_reason": _metadata_or_default(
            extras,
            "host_control_producer_nonoverlap_unavailable_reason",
            "producer_nonoverlap_evidence_not_exported",
        ),
        "host_control_producer_wait_map_nonoverlap_status": _metadata_or_default(
            extras,
            "host_control_producer_wait_map_nonoverlap_status",
            "unavailable",
        ),
        "host_control_producer_wait_map_nonoverlap_unavailable_reason": _metadata_or_default(
            extras,
            "host_control_producer_wait_map_nonoverlap_unavailable_reason",
            "producer_wait_map_nonoverlap_evidence_not_exported",
        ),
        "host_control_producer_double_counting_nonoverlap_status": _metadata_or_default(
            extras,
            "host_control_producer_double_counting_nonoverlap_status",
            "unavailable",
        ),
        "host_control_producer_double_counting_nonoverlap_unavailable_reason": _metadata_or_default(
            extras,
            "host_control_producer_double_counting_nonoverlap_unavailable_reason",
            "producer_double_counting_nonoverlap_evidence_not_exported",
        ),
        "host_control_compat_launch_pop_coverage_status": _metadata_or_default(
            extras,
            "host_control_compat_launch_pop_coverage_status",
            "unavailable",
        ),
        "host_control_compat_launch_pop_coverage_unavailable_reason": _metadata_or_default(
            extras,
            "host_control_compat_launch_pop_coverage_unavailable_reason",
            "not_exported_or_not_applicable",
        ),
        "actual_counterpart_id": _metadata_or_default(extras, "actual_counterpart_id", None),
        "actual_counterpart_status": _metadata_or_default(
            extras,
            "actual_counterpart_status",
            "unavailable",
        ),
        "actual_counterpart_unavailable_reason": _metadata_or_default(
            extras,
            "actual_counterpart_unavailable_reason",
            "offline_exact_counterpart_join_not_run_by_replay_export",
        ),
        "emulated_occurrence_id": _metadata_or_default(extras, "emulated_occurrence_id", None),
        "emulated_occurrence_id_unavailable_reason": _metadata_or_default(
            extras,
            "emulated_occurrence_id_unavailable_reason",
            "emulated_selected_occurrence_join_not_run_by_replay_export",
        ),
        "actual_trace_id": _metadata_or_default(extras, "actual_trace_id", None),
        "actual_trace_id_unavailable_reason": _metadata_or_default(
            extras,
            "actual_trace_id_unavailable_reason",
            "trace_identifier_not_exported",
        ),
        "actual_rank": _metadata_or_default(extras, "actual_rank", event.rank),
        "actual_paper_valid_window_id": _metadata_or_default(
            extras,
            "actual_paper_valid_window_id",
            _metadata_or_default(extras, "actual_counterpart_window_id", None),
        ),
        "actual_paper_valid_window_unavailable_reason": _metadata_or_default(
            extras,
            "actual_paper_valid_window_unavailable_reason",
            _metadata_or_default(extras, "actual_counterpart_window_unavailable_reason", None),
        ),
        "actual_raw_prev_event_id": _metadata_or_default(
            extras,
            "actual_raw_prev_event_id",
            _metadata_or_default(extras, "host_control_boundary_prev_raw_event_id", None),
        ),
        "actual_raw_current_event_id": _metadata_or_default(
            extras,
            "actual_raw_current_event_id",
            _metadata_or_default(extras, "host_control_boundary_current_raw_event_id", None),
        ),
        "actual_boundary_family": _metadata_or_default(
            extras,
            "actual_boundary_family",
            _metadata_or_default(extras, "host_control_boundary_family", "unavailable"),
        ),
        "counterpart_join_key": _metadata_or_default(
            extras,
            "counterpart_join_key",
            selected_occurrence_id,
        ),
        "counterpart_join_method": _metadata_or_default(
            extras,
            "counterpart_join_method",
            "unavailable",
        ),
        "counterpart_join_confidence": _metadata_or_default(
            extras,
            "counterpart_join_confidence",
            "unavailable",
        ),
        "counterpart_unavailable_reason": _metadata_or_default(
            extras,
            "counterpart_unavailable_reason",
            "offline_exact_counterpart_join_not_run_by_replay_export",
        ),
        "comparable_actual_context_only": bool(
            extras.get("comparable_actual_context_only") is True
        ),
        "actual_inter_host_op_gap_unavailable_reason": _metadata_or_default(
            extras,
            "actual_inter_host_op_gap_unavailable_reason",
            None,
        ),
        "actual_counterpart_window_unavailable_reason": _metadata_or_default(
            extras,
            "actual_counterpart_window_unavailable_reason",
            None,
        ),
        "exact_counterpart_status": exact_counterpart_status,
        "exact_counterpart_unavailable_reason": _metadata_or_default(
            extras,
            "exact_counterpart_unavailable_reason",
            "offline_exact_counterpart_join_not_run_by_replay_export",
        ),
        "double_counting_overlap_status": _metadata_or_default(
            extras,
            "double_counting_overlap_status",
            "unavailable",
        ),
        "wait_map_safety_status": _metadata_or_default(extras, "wait_map_safety_status", "unavailable"),
        "wait_map_non_overlap_unavailable_reason": _metadata_or_default(
            extras,
            "wait_map_non_overlap_unavailable_reason",
            "requires_replay_wait_edge_non_overlap_review",
        ),
        "affected_interval_id": _metadata_or_default(extras, "affected_interval_id", None),
        "affected_interval_unavailable_reason": _metadata_or_default(
            extras,
            "affected_interval_unavailable_reason",
            None,
        ),
        "candidate_subinterval_id": _metadata_or_default(
            extras,
            "candidate_subinterval_id",
            None,
        ),
        "candidate_subinterval_unavailable_reason": _metadata_or_default(
            extras,
            "candidate_subinterval_unavailable_reason",
            "mechanical_visibility_split_not_exported",
        ),
        "interval_kind": _metadata_or_default(
            extras,
            "interval_kind",
            "actual_endpoint_gap_context_only",
        ),
        "start_ts_us": _metadata_or_default(extras, "start_ts_us", None),
        "end_ts_us": _metadata_or_default(extras, "end_ts_us", None),
        "duration_us": _metadata_or_default(extras, "duration_us", observed_gap_us),
        "host_dispatch_interval_ids": _metadata_list(extras.get("host_dispatch_interval_ids")),
        "stream_order_interval_ids": _metadata_list(extras.get("stream_order_interval_ids")),
        "cuda_event_wait_edge_ids": _metadata_list(extras.get("cuda_event_wait_edge_ids")),
        "collective_wait_edge_ids": _metadata_list(extras.get("collective_wait_edge_ids")),
        "host_sync_interval_ids": _metadata_list(extras.get("host_sync_interval_ids")),
        "rank_completion_context_id": _metadata_or_default(
            extras,
            "rank_completion_context_id",
            None,
        ),
        "global_completion_context_id": _metadata_or_default(
            extras,
            "global_completion_context_id",
            None,
        ),
        "count_once_status": _metadata_or_default(extras, "count_once_status", "unavailable"),
        "nonoverlap_status": _metadata_or_default(extras, "nonoverlap_status", "unavailable"),
        "host_dispatch_overlap_status": _metadata_or_default(
            extras,
            "host_dispatch_overlap_status",
            "unavailable",
        ),
        "provider_runtime_overlap_status": _metadata_or_default(
            extras,
            "provider_runtime_overlap_status",
            "unavailable",
        ),
        "stream_wait_overlap_status": _metadata_or_default(
            extras,
            "stream_wait_overlap_status",
            "unavailable",
        ),
        "safe_to_use_as_repair_evidence": bool(extras.get("safe_to_use_as_repair_evidence") is True),
        "safe_to_use_as_subtraction_delta": bool(extras.get("safe_to_use_as_subtraction_delta") is True),
        "safe_to_use_as_repair_evidence_reason": _metadata_or_default(
            extras,
            "safe_to_use_as_repair_evidence_reason",
            "requires_paper_invisible_exact_counterpart_and_wait_map_safety_review",
        ),
        "safe_to_use_as_subtraction_delta_reason": _metadata_or_default(
            extras,
            "safe_to_use_as_subtraction_delta_reason",
            "diagnostic_export_never_authorizes_runtime_substitution",
        ),
    }
    if extras.get("hostdelay_occurrence_metadata_opt_in_flag") is True:
        metadata.update(
            {
                key: _metadata_or_default(extras, key, None)
                for key in _HOSTDELAY_OCCURRENCE_METADATA_EXPORT_FIELDS
            }
        )
    if extras.get("collective_event_polling_metadata_opt_in_flag") is True or any(
        key in extras for key in _COLLECTIVE_EVENT_POLLING_METADATA_EXPORT_FIELDS
    ):
        metadata.update(
            {
                key: _metadata_or_default(extras, key, None)
                for key in _COLLECTIVE_EVENT_POLLING_METADATA_EXPORT_FIELDS
                if key in extras
            }
        )
    if _event_polling_boundary_metadata_export_enabled():
        metadata.update(
            {
                key: _metadata_or_default(extras, key, None)
                for key in _EVENT_POLLING_BOUNDARY_METADATA_EXPORT_FIELDS
                if key in extras
            }
        )
    if _boundary_origin_subregion_proof_metadata_export_enabled():
        metadata.update(
            {
                key: _metadata_or_default(extras, key, None)
                for key in _BOUNDARY_ORIGIN_SUBREGION_PROOF_METADATA_EXPORT_FIELDS
                if key in extras
            }
        )
    if _strict_subregion_extent_source_metadata_export_enabled():
        metadata.update(
            {
                key: _metadata_or_default(extras, key, None)
                for key in _STRICT_SUBREGION_EXTENT_SOURCE_METADATA_EXPORT_FIELDS
                if key in extras
            }
        )
    if _critical_path_hostdelay_semantic_classification_enabled():
        metadata.update(
            _critical_path_hostdelay_semantic_classification_metadata(
                event,
                metadata,
                observed_gap_us,
            )
        )
    if not _host_control_visibility_opted_in(extras):
        for field in _HOST_CONTROL_VISIBILITY_EXPORT_FIELDS:
            metadata.pop(field, None)
    return metadata


def _event_export_metadata(event: AnnotatedEvent | None) -> dict[str, Any]:
    if event is None:
        return {}
    extras = event.extras
    keys = (
        "raw_prev_event_id",
        "raw_prev_api",
        "raw_prev_ts_us",
        "raw_prev_end_ts_us",
        "raw_prev_end_ts_source",
        "raw_prev_host_duration_us",
        "raw_current_event_id",
        "raw_current_api",
        "raw_current_ts_us",
        "raw_current_end_ts_us",
        "raw_current_end_ts_source",
        "raw_current_host_duration_us",
        "previous_materialized_event_id",
        "previous_materialized_api",
        "current_materialized_event_id",
        "current_materialized_api",
        "raw_boundary_family",
        "materialized_boundary_family",
        "boundary_origin_family",
        "boundary_origin_kind",
        "boundary_visibility_kind",
        "boundary_origin_field_sources",
        "boundary_origin_conflicting_fields",
        "paper_visible_host_duration_us",
        "instrumentation_only_duration_us",
        "wrapper_internal_duration_us",
        "caller_visible_elapsed_us",
        "fake_api_body_duration_us",
        "unresolved_mixed_duration_us",
        "boundary_segment_schema_version",
        "launch_boundary_id",
        "launch_boundary_id_unavailable_reason",
        "wrapper_segment_coverage",
        "wrapper_segment_sum_us",
        "wrapper_segment_unattributed_us",
        "boundary_origin_classification_basis",
        "boundary_visibility_segments",
        "host_control_visibility_schema_version",
        "host_control_visibility_opt_in_flag",
        *_HOST_CONTROL_ENVELOPE_EXPORT_FIELDS,
        *LAUNCH_NEIGHBORHOOD_EQUIVALENCE_EXPORT_FIELDS,
        "selected_occurrence_id",
        "paper_valid_window_id",
        "paper_valid_window_unavailable_reason",
        "runtime_or_framework_duration_us",
        "payload_enrichment_duration_us",
        "trace_serialization_duration_us",
        "mis_materialized_duration_us",
        "split_sum_check_status",
        "split_tolerance_us",
        "classification_basis",
        "classification_unavailable_reason",
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
        "actual_counterpart_id",
        "actual_counterpart_status",
        "actual_counterpart_unavailable_reason",
        "emulated_occurrence_id",
        "emulated_occurrence_id_unavailable_reason",
        "actual_trace_id",
        "actual_trace_id_unavailable_reason",
        "actual_rank",
        "actual_paper_valid_window_id",
        "actual_paper_valid_window_unavailable_reason",
        "actual_raw_prev_event_id",
        "actual_raw_current_event_id",
        "actual_boundary_family",
        "counterpart_join_key",
        "counterpart_join_method",
        "counterpart_join_confidence",
        "counterpart_unavailable_reason",
        "comparable_actual_context_only",
        "actual_host_dispatch_duration_us",
        "actual_inter_host_op_gap_us",
        "actual_inter_host_op_gap_unavailable_reason",
        "actual_counterpart_component_id",
        "actual_counterpart_rank",
        "actual_counterpart_window_id",
        "actual_counterpart_window_unavailable_reason",
        "actual_counterpart_prev_event_id",
        "actual_counterpart_current_event_id",
        "actual_counterpart_boundary_family",
        "actual_counterpart_dispatch_queue_id",
        "actual_counterpart_visibility_kind",
        "actual_launch_control_dispatch_us",
        "actual_launch_api_body_us",
        "actual_launch_instrumentation_only_us",
        "actual_launch_visibility_kind",
        "actual_launch_unavailable_reason",
        "exact_counterpart_status",
        "exact_counterpart_unavailable_reason",
        "wait_map_safety_status",
        "double_counting_overlap_status",
        "double_counting_overlap_unavailable_reason",
        "wait_map_non_overlap_unavailable_reason",
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
        "safe_to_use_as_repair_evidence",
        "safe_to_use_as_subtraction_delta",
        "event_id",
        "event_version",
        "stream_id",
        "comm_id",
        "comm_hash",
        "call_idx",
        "collective",
        "nranks",
        "wrapper_runtime_contract",
        "host_duration_us",
        "direct_runtime_us",
        "host_machine_id",
        "host_dispatch_queue_id",
        "host_timing_dispatch_scope",
        "host_dispatch_model",
        "paper_valid_window_membership",
        "hostdelay_source",
        "trace_source",
        *_HOSTDELAY_OCCURRENCE_METADATA_EXPORT_FIELDS,
        *_GEMM_ADJACENT_HOSTDELAY_EXPORT_FIELDS,
        *_CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_EXPORT_FIELDS,
        *_CRITICAL_PATH_HOSTDELAY_BOUNDARY_SEMANTIC_CLASSIFICATION_FIELDS,
    )
    metadata = {key: extras.get(key) for key in keys if extras.get(key) not in (None, "")}
    for field in _CRITICAL_PATH_HOSTDELAY_BOUNDARY_SEMANTIC_CLASSIFICATION_FIELDS:
        metadata.pop(field, None)
    if event.api == "__hostDelay__" or "observed_gap_us" in extras or "raw_current_event_id" in extras:
        metadata.update(_host_control_boundary_metadata(event))
    if not _host_control_visibility_opted_in(extras):
        for field in _HOST_CONTROL_VISIBILITY_EXPORT_FIELDS:
            metadata.pop(field, None)
    if not _critical_path_hostdelay_semantic_classification_enabled():
        for field in _CRITICAL_PATH_HOSTDELAY_BOUNDARY_SEMANTIC_CLASSIFICATION_FIELDS:
            metadata.pop(field, None)
    if extras.get("cuda_gemm_hostdispatch_strict_occurrence_gap_opt_in_flag") is True:
        for key in _CUDALAUNCH_GEMM_HOSTDISPATCH_STRICT_OCCURRENCE_GAP_EXPORT_FIELDS:
            if key in extras:
                metadata[key] = extras.get(key)
    return metadata


def _collective_event_polling_replay_waitmap_ids(
    link_fields: dict[str, Any],
    materialized_event_id: str,
) -> tuple[list[str], list[str]]:
    successors: list[str] = []
    predecessors: list[str] = []
    for key in (
        "collective_event_polling_semantic_successor_event_id",
        "collective_event_polling_raw_successor_event_id",
        "hostdelay_occurrence_semantic_successor_event_id",
        "hostdelay_occurrence_raw_successor_event_id",
        "current_materialized_event_id",
        "raw_current_event_id",
    ):
        value = link_fields.get(key)
        if value not in (None, ""):
            successors.append(str(value))
    for key in (
        "collective_event_polling_semantic_predecessor_event_id",
        "collective_event_polling_raw_predecessor_event_id",
        "hostdelay_occurrence_semantic_predecessor_event_id",
        "hostdelay_occurrence_raw_predecessor_event_id",
        "previous_materialized_event_id",
        "raw_prev_event_id",
    ):
        value = link_fields.get(key)
        if value not in (None, ""):
            predecessors.append(str(value))
    if materialized_event_id not in successors:
        successors.append(materialized_event_id)
    return successors, predecessors


def _collective_event_polling_replay_waitmap_api_text(
    link_fields: dict[str, Any],
    event_by_id: dict[str, AnnotatedEvent],
    event_ids: list[str],
) -> str:
    parts: list[str] = []
    for key in (
        "collective_event_polling_raw_boundary_family",
        "collective_event_polling_semantic_boundary_family",
        "hostdelay_occurrence_raw_boundary_family",
        "hostdelay_occurrence_semantic_boundary_family",
        "raw_boundary_family",
        "materialized_boundary_family",
        "boundary_origin_family",
        "collective_event_polling_target_family",
    ):
        value = link_fields.get(key)
        if value not in (None, ""):
            parts.append(str(value))
    for event_id in event_ids:
        event = event_by_id.get(event_id)
        if event is not None:
            parts.append(event.api)
    return " ".join(parts)


def _collective_event_polling_replay_waitmap_base_metadata(
    link_fields: dict[str, Any],
    materialized_event_id: str,
    *,
    target_api_class: str,
    join_status: str,
) -> dict[str, Any]:
    return {
        "collective_event_polling_replay_waitmap_metadata_schema_version": (
            _COLLECTIVE_EVENT_POLLING_REPLAY_WAITMAP_RELEASE_METADATA_SCHEMA_VERSION
        ),
        "collective_event_polling_replay_waitmap_opt_in_flag": True,
        "collective_event_polling_replay_waitmap_env_flags": list(
            _COLLECTIVE_EVENT_POLLING_REPLAY_WAITMAP_RELEASE_METADATA_ENV_KEYS
        ),
        "collective_event_polling_replay_waitmap_behavior_effect": (
            "diagnostic_export_only_no_replay_scheduling_or_timing_behavior_change"
        ),
        "collective_event_polling_replay_waitmap_scope": "replay_export_link_fields_only",
        "collective_event_polling_replay_waitmap_source": (
            "predicted_replay_scheduler_diagnostics"
        ),
        "collective_event_polling_replay_waitmap_occurrence_id": (
            link_fields.get("stable_hostdelay_occurrence_id")
            or link_fields.get("collective_event_polling_occurrence_join_key")
        ),
        "collective_event_polling_replay_waitmap_materialized_event_id": (
            materialized_event_id
        ),
        "collective_event_polling_replay_waitmap_raw_boundary_family": (
            link_fields.get("collective_event_polling_raw_boundary_family")
            or link_fields.get("hostdelay_occurrence_raw_boundary_family")
            or link_fields.get("raw_boundary_family")
        ),
        "collective_event_polling_replay_waitmap_semantic_boundary_family": (
            link_fields.get("collective_event_polling_semantic_boundary_family")
            or link_fields.get("hostdelay_occurrence_semantic_boundary_family")
            or link_fields.get("materialized_boundary_family")
        ),
        "collective_event_polling_replay_waitmap_target_api_class": target_api_class,
        "collective_event_polling_replay_waitmap_join_status": join_status,
        "collective_event_polling_replay_waitmap_join_key": None,
        "collective_event_polling_replay_waitmap_join_key_basis": None,
        "collective_event_polling_replay_waitmap_edge_id": None,
        "collective_event_polling_replay_waitmap_edge_kind": None,
        "collective_event_polling_replay_waitmap_edge_source": None,
        "collective_event_polling_replay_waitmap_wait_reason": None,
        "collective_event_polling_replay_waitmap_release_reason": None,
        "collective_event_polling_replay_waitmap_wait_key": None,
        "collective_event_polling_replay_waitmap_cuda_event_id": None,
        "collective_event_polling_replay_waitmap_cuda_event_version": None,
        "collective_event_polling_replay_waitmap_record_event_id": None,
        "collective_event_polling_replay_waitmap_wait_event_id": None,
        "collective_event_polling_replay_waitmap_predecessor_simulated_event_id": None,
        "collective_event_polling_replay_waitmap_successor_simulated_event_id": None,
        "collective_event_polling_replay_waitmap_released_by_event_id": None,
        "collective_event_polling_replay_waitmap_wait_start_us": None,
        "collective_event_polling_replay_waitmap_release_us": None,
        "collective_event_polling_replay_waitmap_waited_us_context_only": None,
        "collective_event_polling_replay_waitmap_nonblocking_poll_status": None,
        "collective_event_polling_replay_waitmap_no_wait_edge_expected": False,
        "collective_event_polling_replay_waitmap_polling_classification_basis": None,
        "collective_event_polling_replay_waitmap_collective_group_id": None,
        "collective_event_polling_replay_waitmap_collective_edge_id": None,
        "collective_event_polling_replay_waitmap_collective_member_event_id": None,
        "collective_event_polling_replay_waitmap_collective_released_by_event_id": None,
        "collective_event_polling_replay_waitmap_collective_ready_count": None,
        "collective_event_polling_replay_waitmap_collective_expected_participants": None,
        "collective_event_polling_replay_waitmap_collective_participant_event_ids": None,
        "collective_event_polling_replay_waitmap_collective_wait_start_us": None,
        "collective_event_polling_replay_waitmap_collective_release_us": None,
        "collective_event_polling_replay_waitmap_collective_waited_us_context_only": None,
        "collective_event_polling_replay_waitmap_release_status": (
            "unavailable_or_not_applicable"
        ),
        "collective_event_polling_replay_waitmap_release_status_basis": None,
        "collective_event_polling_replay_waitmap_count_once_group_id": None,
        "collective_event_polling_replay_waitmap_count_once_status": "review_pending",
        "collective_event_polling_replay_waitmap_count_once_basis": (
            "predicted_replay_edge_context_only"
        ),
        "collective_event_polling_replay_waitmap_nonoverlap_status": "review_pending",
        "collective_event_polling_replay_waitmap_nonoverlap_basis": (
            "predicted_replay_edge_context_only"
        ),
        "collective_event_polling_replay_waitmap_wait_map_safety_status": (
            "review_pending"
        ),
        "collective_event_polling_replay_waitmap_wait_map_safety_basis": (
            "predicted_replay_edge_context_only_not_repair_evidence"
        ),
        "collective_event_polling_replay_waitmap_origin_visibility_required_for_repair": True,
        "collective_event_polling_replay_waitmap_repair_ready": False,
        "collective_event_polling_replay_waitmap_repair_ready_reason": (
            "requires_later_origin_visibility_count_once_nonoverlap_waitmap_and_fresh8_review"
        ),
        "collective_event_polling_replay_waitmap_safe_to_use_as_repair_evidence": False,
        "collective_event_polling_replay_waitmap_safe_to_use_as_subtraction_delta": False,
        "collective_event_polling_replay_waitmap_safe_delta_us": None,
        "collective_event_polling_replay_waitmap_runtime_or_endpoint_substitution_used": False,
        "collective_event_polling_replay_waitmap_hostdelay_shortening_used": False,
        "collective_event_polling_replay_waitmap_rank_workload_special_case_used": False,
    }


def _collective_event_polling_replay_waitmap_apply_edge(
    metadata: dict[str, Any],
    edge: dict[str, Any],
    *,
    join_status: str,
) -> None:
    metadata.update(
        {
            "collective_event_polling_replay_waitmap_join_status": join_status,
            "collective_event_polling_replay_waitmap_join_key": edge.get("edge_id"),
            "collective_event_polling_replay_waitmap_join_key_basis": (
                "predicted_replay_scheduler_wait_release_edge"
            ),
            "collective_event_polling_replay_waitmap_edge_id": edge.get("edge_id"),
            "collective_event_polling_replay_waitmap_edge_kind": edge.get("edge_kind"),
            "collective_event_polling_replay_waitmap_edge_source": edge.get("source"),
            "collective_event_polling_replay_waitmap_wait_reason": edge.get("wait_reason"),
            "collective_event_polling_replay_waitmap_release_reason": edge.get(
                "release_reason"
            ),
            "collective_event_polling_replay_waitmap_wait_key": edge.get("wait_key"),
            "collective_event_polling_replay_waitmap_cuda_event_id": edge.get(
                "cuda_event_id"
            ),
            "collective_event_polling_replay_waitmap_cuda_event_version": edge.get(
                "cuda_event_version"
            ),
            "collective_event_polling_replay_waitmap_record_event_id": (
                edge.get("record_event_id") or edge.get("predecessor_event_id")
            ),
            "collective_event_polling_replay_waitmap_wait_event_id": (
                edge.get("wait_event_id") or edge.get("successor_event_id")
            ),
            "collective_event_polling_replay_waitmap_predecessor_simulated_event_id": edge.get(
                "predecessor_simulated_event_id"
            ),
            "collective_event_polling_replay_waitmap_successor_simulated_event_id": edge.get(
                "successor_simulated_event_id"
            ),
            "collective_event_polling_replay_waitmap_released_by_event_id": edge.get(
                "released_by_event_id"
            ),
            "collective_event_polling_replay_waitmap_wait_start_us": edge.get(
                "wait_start_us"
            ),
            "collective_event_polling_replay_waitmap_release_us": edge.get("release_us"),
            "collective_event_polling_replay_waitmap_waited_us_context_only": edge.get(
                "waited_us"
            ),
            "collective_event_polling_replay_waitmap_release_status": (
                "predicted_replay_wait_release_edge_available"
            ),
            "collective_event_polling_replay_waitmap_release_status_basis": (
                "scheduler_wait_release_diagnostic"
            ),
            "collective_event_polling_replay_waitmap_count_once_group_id": edge.get(
                "edge_id"
            ),
        }
    )


def _collective_event_polling_replay_waitmap_metadata(
    *,
    link_fields: dict[str, Any],
    materialized_event_id: str,
    event_by_id: dict[str, AnnotatedEvent],
    wait_edges_by_successor: dict[str, list[dict[str, Any]]],
    collective_edges_by_successor: dict[str, list[dict[str, Any]]],
    collective_edges_by_group: dict[str, list[dict[str, Any]]],
    diagnostic_events_available: bool,
) -> dict[str, Any]:
    successors, predecessors = _collective_event_polling_replay_waitmap_ids(
        link_fields,
        materialized_event_id,
    )
    all_ids = successors + predecessors
    api_text = _collective_event_polling_replay_waitmap_api_text(
        link_fields,
        event_by_id,
        all_ids,
    )
    if not any(
        api in api_text for api in _COLLECTIVE_EVENT_POLLING_REPLAY_WAITMAP_TARGET_APIS
    ):
        return {}

    successor_api = next(
        (
            event_by_id[event_id].api
            for event_id in successors
            if event_id in event_by_id and event_by_id[event_id].api != "__hostDelay__"
        ),
        None,
    )
    if successor_api is None:
        successor_api = str(
            _first_metadata_value(
                link_fields.get("hostdelay_occurrence_raw_successor_api"),
                link_fields.get("raw_current_api"),
                link_fields.get("collective_event_polling_target_family"),
                default="unknown",
            )
        )

    if successor_api == "cudaEventQuery":
        metadata = _collective_event_polling_replay_waitmap_base_metadata(
            link_fields,
            materialized_event_id,
            target_api_class="cudaEventQuery",
            join_status="not_applicable_nonblocking_poll",
        )
        metadata.update(
            {
                "collective_event_polling_replay_waitmap_nonblocking_poll_status": (
                    "nonblocking_poll_no_replay_wait_edge_expected"
                ),
                "collective_event_polling_replay_waitmap_no_wait_edge_expected": True,
                "collective_event_polling_replay_waitmap_polling_classification_basis": (
                    "replay_cudaEventQuery_nonblocking_host_op"
                ),
                "collective_event_polling_replay_waitmap_release_status": (
                    "not_applicable_nonblocking_poll"
                ),
                "collective_event_polling_replay_waitmap_release_status_basis": (
                    "cudaEventQuery_does_not_block_on_replay_wait_map"
                ),
                "collective_event_polling_replay_waitmap_count_once_status": (
                    "not_applicable_no_wait_edge"
                ),
                "collective_event_polling_replay_waitmap_nonoverlap_status": (
                    "not_applicable_no_wait_edge"
                ),
                "collective_event_polling_replay_waitmap_wait_map_safety_status": (
                    "not_applicable_no_wait_edge"
                ),
            }
        )
        return metadata

    if successor_api == "cudaStreamWaitEvent" or "cudaStreamWaitEvent" in api_text:
        status = (
            "unavailable_no_diagnostic_events"
            if not diagnostic_events_available
            else "no_predicted_cuda_event_wait_edge_pre_ready_or_out_of_window"
        )
        metadata = _collective_event_polling_replay_waitmap_base_metadata(
            link_fields,
            materialized_event_id,
            target_api_class="cudaStreamWaitEvent",
            join_status=status,
        )
        candidates: list[dict[str, Any]] = []
        for event_id in successors + [materialized_event_id]:
            candidates.extend(wait_edges_by_successor.get(event_id, ()))
        unique = {str(edge.get("edge_id")): edge for edge in candidates if edge.get("edge_id")}
        if len(unique) == 1:
            _collective_event_polling_replay_waitmap_apply_edge(
                metadata,
                next(iter(unique.values())),
                join_status="joined_predicted_cuda_event_wait_edge",
            )
        elif len(unique) > 1:
            metadata["collective_event_polling_replay_waitmap_join_status"] = (
                "ambiguous_multiple_predicted_edges"
            )
        return metadata

    collective_ids = [
        event_id
        for event_id in all_ids
        if event_id in event_by_id and event_by_id[event_id].api.startswith("nccl")
    ]
    collective_group_ids = [
        str(value)
        for value in (
            link_fields.get("collective_event_polling_collective_group_id"),
            link_fields.get("hostdelay_occurrence_collective_group_id"),
        )
        if value not in (None, "")
    ]
    for event_id in collective_ids:
        group_id = event_by_id[event_id].collective_group_id
        if group_id not in (None, ""):
            collective_group_ids.append(str(group_id))
    if collective_ids or any(
        api in api_text for api in ("ncclAllReduce", "ncclSend", "ncclRecv")
    ):
        status = (
            "unavailable_no_diagnostic_events"
            if not diagnostic_events_available
            else "no_predicted_collective_wait_edge"
        )
        metadata = _collective_event_polling_replay_waitmap_base_metadata(
            link_fields,
            materialized_event_id,
            target_api_class="nccl_collective",
            join_status=status,
        )
        candidates: list[dict[str, Any]] = []
        for event_id in collective_ids:
            candidates.extend(collective_edges_by_successor.get(event_id, ()))
        if not candidates:
            for group_id in collective_group_ids:
                candidates.extend(collective_edges_by_group.get(group_id, ()))
        unique = {str(edge.get("edge_id")): edge for edge in candidates if edge.get("edge_id")}
        if len(unique) == 1:
            edge = next(iter(unique.values()))
            _collective_event_polling_replay_waitmap_apply_edge(
                metadata,
                edge,
                join_status="joined_predicted_collective_wait_edge",
            )
            metadata.update(
                {
                    "collective_event_polling_replay_waitmap_collective_group_id": edge.get(
                        "collective_group_id"
                    ),
                    "collective_event_polling_replay_waitmap_collective_edge_id": edge.get(
                        "edge_id"
                    ),
                    "collective_event_polling_replay_waitmap_collective_member_event_id": edge.get(
                        "successor_event_id"
                    ),
                    "collective_event_polling_replay_waitmap_collective_released_by_event_id": edge.get(
                        "released_by_event_id"
                    ),
                    "collective_event_polling_replay_waitmap_collective_wait_start_us": edge.get(
                        "wait_start_us"
                    ),
                    "collective_event_polling_replay_waitmap_collective_release_us": edge.get(
                        "release_us"
                    ),
                    "collective_event_polling_replay_waitmap_collective_waited_us_context_only": edge.get(
                        "waited_us"
                    ),
                }
            )
        elif len(unique) > 1:
            metadata["collective_event_polling_replay_waitmap_join_status"] = (
                "ambiguous_multiple_predicted_edges"
            )
        return metadata

    return {}


def _compact_export_event(event: AnnotatedEvent | None) -> dict[str, Any] | None:
    if event is None:
        return None
    return {
        "event_id": event.id,
        "rank": int(event.rank),
        "ordinal": int(event.ordinal),
        "api": event.api,
        "op_type": event.op_type,
        "duration_us": float(event.duration_us),
        "duration_source": event.duration_source,
        "collective_group_id": event.collective_group_id,
        "metadata": _event_export_metadata(event),
    }


def _find_prior_simulated(
    rows_by_resource: dict[Any, list[dict[str, Any]]],
    end_times_by_resource: dict[Any, list[float]],
    resource_key: Any,
    release_us: float,
    successor_simulated_event_id: str,
) -> dict[str, Any] | None:
    from bisect import bisect_right

    rows = rows_by_resource.get(resource_key) or []
    end_times = end_times_by_resource.get(resource_key) or []
    index = bisect_right(end_times, release_us + _EDGE_TIME_TOLERANCE_US) - 1
    while index >= 0:
        candidate = rows[index]
        if str(candidate.get("event_id")) != successor_simulated_event_id:
            return candidate
        index -= 1
    return None


def _parse_diagnostic_tuple(value: object | None) -> tuple[Any, ...] | None:
    if value in (None, ""):
        return None
    import ast

    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, tuple) else None


def _diagnostic_wait_kind(wait_reason: str | None, release_reason: str | None) -> str:
    text = f"{wait_reason or ''} {release_reason or ''}"
    if "cuda_event" in text:
        return "cuda_event_wait"
    if "collective" in text:
        return "collective_wait"
    if "stream_queue" in text:
        return "stream_queue_wait_release"
    if "host_queue" in text:
        return "host_order"
    if "host_device" in text or "host_stream" in text or "quiescent" in text:
        return "host_sync_wait"
    return "other_wait_release"


def _appendix_ab_component_split_for_event(
    *,
    event: AnnotatedEvent,
    interval: dict[str, Any],
    host_dispatch_interval: dict[str, Any] | None,
    predecessor_edges: list[dict[str, Any]],
    selected_block_label: str = "selected P2P blocks",
) -> tuple[dict[str, float | None], dict[str, str]]:
    split: dict[str, float | None] = {key: 0.0 for key in _APPENDIX_AB_P2P_COMPONENT_KEYS}
    unavailable: dict[str, str] = {}
    split["provider_runtime"] = float(interval["duration_us"])
    if host_dispatch_interval is not None:
        split["host_dispatch"] = float(host_dispatch_interval["duration_us"])
    collective_waits = [
        float(edge.get("waited_us") or 0.0)
        for edge in predecessor_edges
        if edge.get("edge_kind") == "collective_wait"
    ]
    stream_waits = [
        float(edge.get("waited_us") or 0.0)
        for edge in predecessor_edges
        if edge.get("edge_kind") == "stream_queue_wait_release"
    ]
    cuda_event_waits = [
        float(edge.get("waited_us") or 0.0)
        for edge in predecessor_edges
        if edge.get("edge_kind") == "cuda_event_wait"
    ]
    host_sync_waits = [
        float(edge.get("waited_us") or 0.0)
        for edge in predecessor_edges
        if edge.get("edge_kind") == "host_sync_wait"
    ]
    if collective_waits:
        split["collective_wait"] = max(collective_waits)
    if stream_waits:
        split["stream_queue_wait"] = max(stream_waits)
    if cuda_event_waits:
        split["cuda_event_wait"] = max(cuda_event_waits)
    if host_sync_waits:
        split["host_sync_wait"] = max(host_sync_waits)

    split["hostDelay"] = None
    unavailable["hostDelay"] = (
        f"hostDelay predecessor attribution is not attached to {selected_block_label} "
        "by the replay edge exporter"
    )
    split["residual_unattributed"] = None
    unavailable["residual_unattributed"] = (
        "Residual_unknown_b requires apples-to-apples actual block terms and "
        "complete non-overlapping component attribution"
    )
    if host_dispatch_interval is None:
        unavailable["host_dispatch"] = "no modeled host_dispatch interval for this selected event"
    if not stream_waits:
        unavailable["stream_queue_wait"] = "no exported stream_queue_wait_release edge for this selected event"
    if not collective_waits:
        unavailable["collective_wait"] = "no exported collective_wait edge for this selected event"
    if not cuda_event_waits:
        unavailable["cuda_event_wait"] = "no exported cuda_event_wait edge for this selected event"
    if not host_sync_waits:
        unavailable["host_sync_wait"] = "no exported host_sync_wait edge for this selected event"
    return split, unavailable


def _appendix_ab_selected_p2p_diagnostic_rows(
    *,
    trace: AnnotatedTrace,
    event_by_id: dict[str, AnnotatedEvent],
    interval_rows: list[dict[str, Any]],
    host_dispatch_sim_by_base_id: dict[str, dict[str, Any]],
    predecessor_edges_by_successor: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for interval in interval_rows:
        simulated_event_id = str(interval["simulated_event_id"])
        event_id = str(interval["materialized_event_id"])
        event = event_by_id.get(event_id)
        if event is None:
            continue
        motif_key = _appendix_ab_selected_p2p_motif_key(event)
        if motif_key is None:
            continue
        group = trace.collective_groups.get(event.collective_group_id or "")
        paper_window = _appendix_ab_paper_valid_window_membership(trace, event)
        host_dispatch_interval = host_dispatch_sim_by_base_id.get(event_id)
        edges = predecessor_edges_by_successor.get(simulated_event_id) or predecessor_edges_by_successor.get(event_id) or []
        component_split, component_unavailable = _appendix_ab_component_split_for_event(
            event=event,
            interval=interval,
            host_dispatch_interval=host_dispatch_interval,
            predecessor_edges=edges,
        )
        members = None if group is None else [int(rank) for rank in group.ranks]
        pair_seq = None
        if group is not None and group.sequence_number is not None:
            pair_seq = int(group.sequence_number)
        else:
            pair_seq = _pair_seq_from_group_id(event.collective_group_id)
        first_edge = edges[0] if edges else {}
        collective_wait_edges = [edge for edge in edges if edge.get("edge_kind") == "collective_wait"]
        wait_edge = collective_wait_edges[0] if collective_wait_edges else first_edge
        row = {
            "schema_version": _APPENDIX_AB_P2P_DIAGNOSTIC_SCHEMA_VERSION,
            "diagnostic_only": True,
            "stable_block_id": _stable_appendix_ab_p2p_block_id(event, simulated_event_id),
            "rank": int(event.rank),
            "rank_class": _metadata_or_default(event.extras, "rank_class", None),
            "motif_key": motif_key,
            "api": event.api,
            "event_ids": [event.id],
            "actual_counterpart_event_ids": [],
            "collective_group_id": event.collective_group_id,
            "group_api": None if group is None else group.api,
            "members": members,
            "peer": _metadata_or_default(event.extras, "peer", None),
            "pair_seq": pair_seq,
            "participant_count": None if group is None or group.participant_count is None else int(group.participant_count),
            "communicator_id": None if group is None else group.communicator_id,
            "canonical_comm_id": _metadata_or_default(
                event.extras,
                "canonical_comm_id",
                None if group is None else group.communicator_id,
            ),
            "local_comm_id": _metadata_or_default(event.extras, "comm_id", None),
            "communicator_size": None if group is None or group.communicator_size is None else int(group.communicator_size),
            "communicator_namespace_status": (
                "resolved_from_collated_collective_group"
                if group is not None and group.communicator_id is not None
                else "unavailable"
            ),
            "stream_id": interval.get("stream_id"),
            "stream_resource_id": interval.get("stream_resource_id"),
            "host_dispatch_queue_id": _first_metadata_value(
                event.extras.get("host_dispatch_queue_id"),
                interval.get("host_dispatch_queue_id"),
            ),
            "paper_valid_window_membership": paper_window,
            "paper_valid_window_id": paper_window["window_id"] if paper_window["in_paper_valid_window"] else None,
            "paper_valid_window_unavailable_reason": paper_window["unavailable_reason"],
            "schedule_resource_kind": interval.get("resource_kind"),
            "predicted_block_start_us": float(interval["start_us"]),
            "predicted_block_end_us": float(interval["end_us"]),
            "predicted_block_duration_us": float(interval["duration_us"]),
            "actual_block_start_us": None,
            "actual_block_end_us": None,
            "actual_block_duration_us": None,
            "actual_block_unavailable_reason": (
                "strict in-window/aligned actual per-block start/end counterpart is not exported; "
                "outside-window actual timings are not included; actual API end_ts is not a "
                "wait-map release or block counterpart"
            ),
            "actual_paper_valid_window_id": None,
            "actual_counterpart_window_status": "unavailable",
            "actual_counterpart_window_unavailable_reason": (
                "no strict actual counterpart aligned to the existing paper-valid step window; "
                "outside-window or unaligned actual timings are intentionally excluded"
            ),
            "component_duration_split_us": component_split,
            "component_duration_split_unavailable_reasons": component_unavailable,
            "collective_group_duration_us": _safe_float_or_none(
                event.extras.get("collective_group_duration_us")
            ),
            "collective_group_duration_basis": _metadata_or_default(
                event.extras,
                "collective_group_duration_basis",
                None,
            ),
            "duration_source": event.duration_source,
            "provider_name": _metadata_or_default(event.extras, "provider_name", None),
            "provider_tier": _metadata_or_default(event.extras, "provider_tier", None),
            "provider_duration_source_expected": _metadata_or_default(
                event.extras,
                "provider_duration_source_expected",
                None,
            ),
            "material_signature": _metadata_or_default(event.extras, "material_signature", None),
            "material_signature_inputs": _metadata_map(event.extras.get("material_signature_inputs")),
            "predecessor_edge_ids": [str(edge["edge_id"]) for edge in edges if edge.get("edge_id")],
            "predecessor_event_ids": [
                str(edge["predecessor_event_id"])
                for edge in edges
                if edge.get("predecessor_event_id") not in (None, "")
            ],
            "phase_anchor_id": None,
            "phase_anchor_unavailable_reason": "phase_anchor_id is not computed by replay diagnostics",
            "wait_start_us": wait_edge.get("wait_start_us"),
            "release_us": wait_edge.get("release_us"),
            "waited_us": wait_edge.get("waited_us"),
            "wait_reason": wait_edge.get("wait_reason"),
            "release_reason": wait_edge.get("release_reason"),
            "released_by_event_id": wait_edge.get("released_by_event_id"),
            "actual_wait_start_us": None,
            "actual_release_us": None,
            "actual_release_reason": None,
            "actual_released_by_event_id": None,
            "actual_release_unavailable_reason": (
                "strict in-window/aligned actual wait-map release/source timing is unavailable; "
                "outside-window actual timings are not included; actual API end_ts is intentionally not used"
            ),
            "stream_namespace_alignment": "unavailable",
            "double_counting_overlap_status": "unavailable",
            "wait_map_safety_status": "unavailable",
            "repair_authorization_status": "diagnostic_only_no_repair_authorized",
            "actual_api_end_ts_used_as_wait_map_release": False,
        }
        rows.append(row)
    rows.sort(key=lambda row: (int(row["rank"]), str(row["api"]), str(row["stable_block_id"])))
    return rows


def _appendix_ab_selected_allreduce_diagnostic_rows(
    *,
    trace: AnnotatedTrace,
    event_by_id: dict[str, AnnotatedEvent],
    interval_rows: list[dict[str, Any]],
    host_dispatch_sim_by_base_id: dict[str, dict[str, Any]],
    predecessor_edges_by_successor: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for interval in interval_rows:
        simulated_event_id = str(interval["simulated_event_id"])
        if simulated_event_id.endswith(":host_dispatch"):
            continue
        event_id = str(interval["materialized_event_id"])
        event = event_by_id.get(event_id)
        if event is None:
            continue
        motif_key = _appendix_ab_selected_allreduce_motif_key(event)
        if motif_key is None:
            continue
        group = trace.collective_groups.get(event.collective_group_id or "")
        paper_window = _appendix_ab_paper_valid_window_membership(trace, event)
        host_dispatch_interval = host_dispatch_sim_by_base_id.get(event_id)
        edges = predecessor_edges_by_successor.get(simulated_event_id) or predecessor_edges_by_successor.get(event_id) or []
        component_split, component_unavailable = _appendix_ab_component_split_for_event(
            event=event,
            interval=interval,
            host_dispatch_interval=host_dispatch_interval,
            predecessor_edges=edges,
            selected_block_label="selected AllReduce blocks",
        )
        members = None if group is None else [int(rank) for rank in group.ranks]
        if group is not None and group.sequence_number is not None:
            group_call_ordinal = int(group.sequence_number)
        else:
            group_call_ordinal = _group_call_ordinal_from_group_id(event.collective_group_id)
        first_edge = edges[0] if edges else {}
        collective_wait_edges = [edge for edge in edges if edge.get("edge_kind") == "collective_wait"]
        wait_edge = collective_wait_edges[0] if collective_wait_edges else first_edge
        row = {
            "schema_version": _APPENDIX_AB_ALLREDUCE_DIAGNOSTIC_SCHEMA_VERSION,
            "diagnostic_only": True,
            "stable_block_id": _stable_appendix_ab_allreduce_block_id(event, simulated_event_id),
            "rank": int(event.rank),
            "rank_class": _metadata_or_default(event.extras, "rank_class", None),
            "family_label": _metadata_or_default(
                event.extras,
                "appendix_ab_allreduce_family_label",
                _APPENDIX_AB_SELECTED_ALLREDUCE_FAMILY_LABEL,
            ),
            "motif_key": motif_key,
            "api": event.api,
            "event_ids": [event.id],
            "actual_counterpart_event_ids": [],
            "collective_group_id": event.collective_group_id,
            "group_api": None if group is None else group.api,
            "members": members,
            "group_call_ordinal": group_call_ordinal,
            "participant_count": None if group is None or group.participant_count is None else int(group.participant_count),
            "communicator_id": None if group is None else group.communicator_id,
            "canonical_comm_id": _metadata_or_default(
                event.extras,
                "canonical_comm_id",
                None if group is None else group.communicator_id,
            ),
            "local_comm_id": _metadata_or_default(event.extras, "comm_id", None),
            "communicator_size": None if group is None or group.communicator_size is None else int(group.communicator_size),
            "communicator_namespace_status": (
                "resolved_from_collated_collective_group"
                if group is not None and group.communicator_id is not None
                else "unavailable"
            ),
            "stream_id": interval.get("stream_id"),
            "stream_resource_id": interval.get("stream_resource_id"),
            "host_dispatch_queue_id": _first_metadata_value(
                event.extras.get("host_dispatch_queue_id"),
                interval.get("host_dispatch_queue_id"),
            ),
            "paper_valid_window_membership": paper_window,
            "paper_valid_window_id": paper_window["window_id"] if paper_window["in_paper_valid_window"] else None,
            "paper_valid_window_unavailable_reason": paper_window["unavailable_reason"],
            "schedule_resource_kind": interval.get("resource_kind"),
            "predicted_block_start_us": float(interval["start_us"]),
            "predicted_block_end_us": float(interval["end_us"]),
            "predicted_block_duration_us": float(interval["duration_us"]),
            "actual_block_start_us": None,
            "actual_block_end_us": None,
            "actual_block_duration_us": None,
            "actual_block_unavailable_reason": (
                "strict in-window/aligned actual per-block start/end counterpart is not exported; "
                "outside-window actual timings are not included; actual API end_ts is not a "
                "wait-map release or block counterpart"
            ),
            "actual_paper_valid_window_id": None,
            "actual_counterpart_window_status": "unavailable",
            "actual_counterpart_window_unavailable_reason": (
                "no strict actual counterpart aligned to the existing paper-valid step window; "
                "outside-window or unaligned actual timings are intentionally excluded"
            ),
            "component_duration_split_us": component_split,
            "component_duration_split_unavailable_reasons": component_unavailable,
            "collective_group_duration_us": _safe_float_or_none(
                event.extras.get("collective_group_duration_us")
            ),
            "collective_group_duration_basis": _metadata_or_default(
                event.extras,
                "collective_group_duration_basis",
                None,
            ),
            "duration_source": event.duration_source,
            "provider_name": _metadata_or_default(event.extras, "provider_name", None),
            "provider_tier": _metadata_or_default(event.extras, "provider_tier", None),
            "provider_duration_source_expected": _metadata_or_default(
                event.extras,
                "provider_duration_source_expected",
                None,
            ),
            "material_signature": _metadata_or_default(event.extras, "material_signature", None),
            "material_signature_inputs": _metadata_map(event.extras.get("material_signature_inputs")),
            "predecessor_edge_ids": [str(edge["edge_id"]) for edge in edges if edge.get("edge_id")],
            "predecessor_event_ids": [
                str(edge["predecessor_event_id"])
                for edge in edges
                if edge.get("predecessor_event_id") not in (None, "")
            ],
            "phase_anchor_id": None,
            "phase_anchor_unavailable_reason": "phase_anchor_id is not computed by replay diagnostics",
            "wait_start_us": wait_edge.get("wait_start_us"),
            "release_us": wait_edge.get("release_us"),
            "waited_us": wait_edge.get("waited_us"),
            "wait_reason": wait_edge.get("wait_reason"),
            "release_reason": wait_edge.get("release_reason"),
            "released_by_event_id": wait_edge.get("released_by_event_id"),
            "actual_wait_start_us": None,
            "actual_release_us": None,
            "actual_release_reason": None,
            "actual_released_by_event_id": None,
            "actual_release_unavailable_reason": (
                "strict in-window/aligned actual wait-map release/source timing is unavailable; "
                "outside-window actual timings are not included; actual API end_ts is intentionally not used"
            ),
            "stream_namespace_alignment": "unavailable",
            "double_counting_overlap_status": "unavailable",
            "wait_map_safety_status": "unavailable",
            "repair_authorization_status": "diagnostic_only_no_repair_authorized",
            "actual_api_end_ts_used_as_wait_map_release": False,
            "actual_api_end_ts_used_as_block_timing": False,
        }
        rows.append(row)
    rows.sort(key=lambda row: (int(row["rank"]), str(row["api"]), str(row["stable_block_id"])))
    return rows


def _shared_phase_anchor_normalized_edge_kind(edge_kind: object) -> str:
    raw = str(edge_kind or "")
    return _SHARED_PHASE_ANCHOR_EDGE_KIND_MAP.get(raw, raw or "unavailable")


def _shared_phase_anchor_semantic_family(event: AnnotatedEvent, group: Any | None) -> str:
    if group is not None and getattr(group, "api", None):
        return str(group.api)
    if event.api in {"ncclSend", "ncclRecv"}:
        return "ncclP2P"
    if event.api == "ncclAllReduce":
        return "ncclAllReduce"
    return event.api


def _shared_phase_anchor_shape_signature(event: AnnotatedEvent, group: Any | None) -> str | None:
    material = event.extras.get("material_signature")
    if material not in (None, ""):
        return str(material)
    group_api = str(getattr(group, "api", "") or event.api)
    collective = str(event.extras.get("collective") or event.api)
    count = _first_metadata_value(
        event.extras.get("numel"),
        event.extras.get("count"),
        event.extras.get("sendcount"),
        event.extras.get("recvcount"),
        default="",
    )
    datatype = _first_metadata_value(
        event.extras.get("dtype_code"),
        event.extras.get("datatype"),
        default="",
    )
    if group_api == event.api and count == "" and datatype == "":
        return None
    return f"group_api={group_api};collective={collective};count={count};datatype={datatype}"


def _shared_phase_anchor_common_collective_kind(
    event: AnnotatedEvent,
    semantic_family: str,
) -> str:
    if semantic_family == "ncclP2P" or event.api in {"ncclSend", "ncclRecv"}:
        return "p2p"
    if event.api == "ncclAllReduce" or semantic_family == "ncclAllReduce":
        return "allreduce"
    return str(event.extras.get("collective") or semantic_family or event.api)


def _shared_phase_anchor_common_api_direction(event: AnnotatedEvent) -> str | None:
    if event.api == "ncclSend":
        return "send"
    if event.api == "ncclRecv":
        return "recv"
    return None


def _shared_phase_anchor_common_metadata(
    *,
    event: AnnotatedEvent,
    group: Any | None,
    semantic_family: str,
    members: list[int] | None,
    normalized_call_order: int,
) -> dict[str, Any]:
    group_id = event.collective_group_id
    pair_seq = _pair_seq_from_group_id(group_id)
    group_call_index = _group_call_ordinal_from_group_id(group_id)
    if event.api in {"ncclSend", "ncclRecv"}:
        call_order_basis = (
            "group_id_pair_seq"
            if pair_seq is not None
            else "unavailable_missing_pair_seq"
        )
        call_order_index = pair_seq
    elif event.api == "ncclAllReduce":
        call_order_basis = (
            "group_id_call_ordinal"
            if group_call_index is not None
            else "unavailable_missing_group_id_call_ordinal"
        )
        call_order_index = group_call_index
    else:
        call_order_basis = "unavailable_unsupported_api"
        call_order_index = None

    collective_kind = _shared_phase_anchor_common_collective_kind(event, semantic_family)
    count = _first_metadata_value(
        event.extras.get("numel"),
        event.extras.get("count"),
        event.extras.get("sendcount"),
        event.extras.get("recvcount"),
        event.extras.get("material_signature_inputs", {}).get("count")
        if isinstance(event.extras.get("material_signature_inputs"), dict)
        else None,
        default=None,
    )
    datatype = _first_metadata_value(
        event.extras.get("dtype_code"),
        event.extras.get("datatype"),
        event.extras.get("material_signature_inputs", {}).get("datatype")
        if isinstance(event.extras.get("material_signature_inputs"), dict)
        else None,
        default=None,
    )
    reduction_op = _first_metadata_value(
        event.extras.get("op"),
        event.extras.get("reduction_op"),
        default=None,
    )
    member_signature = (
        "members:" + "-".join(str(int(member)) for member in sorted(members))
        if members
        else "members:unavailable"
    )
    pair_members = list(members) if collective_kind == "p2p" and members else None
    payload_inputs = {
        "api": event.api,
        "group_api": semantic_family,
        "collective_kind": collective_kind,
        "count": count,
        "datatype": datatype,
        "op": reduction_op,
        "membership_signature": member_signature,
        "pair_members": pair_members,
    }
    payload_signature = (
        f"group_api={semantic_family};kind={collective_kind};api={event.api};"
        f"members={member_signature};count={'' if count is None else count};"
        f"datatype={'' if datatype is None else datatype};"
        f"op={'null' if reduction_op is None else reduction_op}"
    )
    unavailable_reasons: list[str] = []
    if call_order_index is None:
        unavailable_reasons.append("common_call_order_index_unavailable")
    if members is None:
        unavailable_reasons.append("common_membership_unavailable")
    return {
        "common_basis_schema_version": _SHARED_PHASE_ANCHOR_COMMON_BASIS_SCHEMA_VERSION,
        "common_call_order_basis": call_order_basis,
        "common_call_order_index": call_order_index,
        "common_group_id_call_index": group_call_index,
        "common_pair_seq": pair_seq,
        "common_rank_window_index": int(normalized_call_order),
        "common_payload_signature": payload_signature,
        "common_payload_signature_inputs": payload_inputs,
        "payload_basis": "raw_operation_semantics_not_stream_only_key",
        "common_api": event.api,
        "common_group_api": semantic_family,
        "common_api_direction": _shared_phase_anchor_common_api_direction(event),
        "common_collective_kind": collective_kind,
        "common_count": count,
        "common_datatype": datatype,
        "common_reduction_op": reduction_op,
        "common_membership_signature": member_signature,
        "common_pair_members": pair_members,
        "common_tensor_or_count_shape": count,
        "common_key_unavailable_reason": (
            ";".join(unavailable_reasons) if unavailable_reasons else None
        ),
    }


def _stable_shared_phase_anchor_block_id(
    event: AnnotatedEvent,
    simulated_event_id: str,
    normalized_call_order: int,
) -> str:
    source = "|".join(
        [
            str(event.rank),
            event.api,
            str(event.collective_group_id or ""),
            str(normalized_call_order),
            simulated_event_id,
        ]
    )
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return (
        f"shared_phase_anchor_block:rank:{int(event.rank)}:"
        f"api:{event.api}:call:{int(normalized_call_order)}:{digest}"
    )


def _shared_phase_anchor_phase_id(event: AnnotatedEvent, group: Any | None) -> str:
    semantic_family = _shared_phase_anchor_semantic_family(event, group)
    group_id = event.collective_group_id or "ungrouped"
    return f"shared_all_rank_phase_anchor:{semantic_family}:{group_id}"


def _shared_phase_anchor_block_rows(
    *,
    trace: AnnotatedTrace,
    event_by_id: dict[str, AnnotatedEvent],
    interval_rows: list[dict[str, Any]],
    host_dispatch_sim_by_base_id: dict[str, dict[str, Any]],
    predecessor_edges_by_successor: dict[str, list[dict[str, Any]]],
    critical_rank: int | None,
    terminal_simulated_event_ids: set[str],
    include_common_basis: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    block_by_event_id: dict[str, dict[str, Any]] = {}
    call_orders: dict[tuple[object, ...], int] = defaultdict(int)

    for interval in interval_rows:
        simulated_event_id = str(interval["simulated_event_id"])
        if simulated_event_id.endswith(":host_dispatch"):
            continue
        event_id = str(interval["materialized_event_id"])
        event = event_by_id.get(event_id)
        if event is None or event.api not in _SHARED_PHASE_ANCHOR_SELECTED_APIS:
            continue
        group = trace.collective_groups.get(event.collective_group_id or "")
        members = None if group is None else [int(rank) for rank in group.ranks]
        semantic_family = _shared_phase_anchor_semantic_family(event, group)
        shape_signature = _shared_phase_anchor_shape_signature(event, group)
        call_key = (
            semantic_family,
            group.communicator_id if group is not None else event.extras.get("comm_id"),
            tuple(members or ()),
            shape_signature,
            event.api,
        )
        normalized_call_order = call_orders[call_key]
        call_orders[call_key] = normalized_call_order + 1
        stable_block_id = _stable_shared_phase_anchor_block_id(
            event,
            simulated_event_id,
            normalized_call_order,
        )
        phase_anchor_id = _shared_phase_anchor_phase_id(event, group)
        edges = predecessor_edges_by_successor.get(simulated_event_id) or predecessor_edges_by_successor.get(event_id) or []
        split, _unavailable = _appendix_ab_component_split_for_event(
            event=event,
            interval=interval,
            host_dispatch_interval=host_dispatch_sim_by_base_id.get(event_id),
            predecessor_edges=edges,
        )
        row = {
            "stable_block_id": stable_block_id,
            "rank": int(event.rank),
            "api": event.api,
            "semantic_family": semantic_family,
            "normalized_call_order": int(normalized_call_order),
            "phase_anchor_id": phase_anchor_id,
            "phase_anchor_type": "collective_group_phase_anchor",
            "phase_anchor_scope": (
                "all_rank_collective_group"
                if group is not None and len(group.ranks) > 2
                else "rank_pair_or_local_collective_group"
            ),
            "block_start_event_id": event.id,
            "block_end_event_id": event.id,
            "block_event_ids": [event.id],
            "predecessor_chain_depth": None,
            "terminal_rank_candidate": simulated_event_id in terminal_simulated_event_ids,
            "critical_rank_candidate": critical_rank is not None and int(event.rank) == int(critical_rank),
            "group_id": event.collective_group_id,
            "wait_key": None,
            "component_split_schema_version": _SHARED_PHASE_ANCHOR_SCHEMA_VERSION,
            "provider_runtime_us": split.get("provider_runtime"),
            "host_dispatch_or_hostdelay_us": split.get("host_dispatch"),
            "stream_queue_wait_us": split.get("stream_queue_wait"),
            "collective_wait_us": split.get("collective_wait"),
            "cuda_event_wait_us": split.get("cuda_event_wait"),
            "residual_placement_or_unknown_us": split.get("residual_unattributed"),
            "communicator_id": None if group is None else group.communicator_id,
            "participant_rank_ids": members,
            "participant_event_ids": None if group is None else list(group.event_ids),
            "shape_signature": shape_signature,
            "repair_authorization_status": "diagnostic_only_no_repair_authorized",
        }
        if include_common_basis:
            row.update(
                _shared_phase_anchor_common_metadata(
                    event=event,
                    group=group,
                    semantic_family=semantic_family,
                    members=members,
                    normalized_call_order=normalized_call_order,
                )
            )
        rows.append(row)
        block_by_event_id[event.id] = row

    rows.sort(key=lambda row: (int(row["rank"]), str(row["api"]), int(row["normalized_call_order"])))
    return rows, block_by_event_id


def _shared_phase_anchor_causal_edge_rows(
    *,
    trace: AnnotatedTrace,
    event_by_id: dict[str, AnnotatedEvent],
    edges: list[dict[str, Any]],
    predecessor_edges_by_successor: dict[str, list[dict[str, Any]]],
    block_by_event_id: dict[str, dict[str, Any]],
    critical_rank: int | None,
    terminal_simulated_event_ids: set[str],
    simulated_by_id: dict[str, dict[str, Any]],
    canonical_sim_by_base_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for edge in edges:
        successor_event_id = edge.get("successor_event_id")
        predecessor_event_id = edge.get("predecessor_event_id")
        event = event_by_id.get(str(successor_event_id)) if successor_event_id not in (None, "") else None
        predecessor_event = (
            event_by_id.get(str(predecessor_event_id))
            if predecessor_event_id not in (None, "")
            else None
        )
        successor_simulated_id = edge.get("successor_simulated_event_id")
        sim = (
            simulated_by_id.get(str(successor_simulated_id))
            if successor_simulated_id not in (None, "")
            else None
        )
        if sim is None and successor_event_id not in (None, ""):
            sim = canonical_sim_by_base_id.get(str(successor_event_id))
        group = trace.collective_groups.get((event.collective_group_id if event is not None else None) or "")
        block = block_by_event_id.get(event.id) if event is not None else None
        edge_kind = _shared_phase_anchor_normalized_edge_kind(edge.get("edge_kind"))
        predecessor_event_ids = (
            [str(predecessor_event_id)]
            if predecessor_event_id not in (None, "")
            else []
        )
        successor_event_ids = (
            [str(successor_event_id)]
            if successor_event_id not in (None, "")
            else []
        )
        predecessor_stable_edge_ids: list[str] = []
        for predecessor in predecessor_event_ids:
            predecessor_stable_edge_ids.extend(
                str(item["edge_id"])
                for item in predecessor_edges_by_successor.get(predecessor, [])
                if item.get("edge_id") not in (None, "")
            )
        rows.append(
            {
                "shared_anchor_causal_edge_schema_version": _SHARED_PHASE_ANCHOR_SCHEMA_VERSION,
                "diagnostic_opt_in_flag": True,
                "source_side": "predicted_replay",
                "stable_replay_edge_id": str(edge.get("edge_id")),
                "event_id": None if successor_event_id in (None, "") else str(successor_event_id),
                "rank": (
                    int(event.rank)
                    if event is not None
                    else (int(sim["rank"]) if sim is not None and sim.get("rank") is not None else None)
                ),
                "api": None if event is None else event.api,
                "resource_type": edge.get("resource_kind"),
                "stream_id": (
                    edge.get("stream_resource_id")
                    or (None if sim is None else sim.get("stream_id"))
                    or (None if event is None else event.extras.get("stream_id"))
                ),
                "canonical_stream_resource_id": edge.get("stream_resource_id"),
                "host_resource_id": edge.get("host_resource_id"),
                "predecessor_event_ids": predecessor_event_ids,
                "predecessor_stable_edge_ids": predecessor_stable_edge_ids,
                "successor_event_ids": successor_event_ids,
                "edge_kind": edge_kind,
                "edge_kind_detail": edge.get("edge_kind"),
                "edge_source_reason": edge.get("release_reason") or edge.get("source"),
                "wait_key": edge.get("wait_key"),
                "wait_key_version": edge.get("cuda_event_version"),
                "group_id": edge.get("collective_group_id") or (event.collective_group_id if event is not None else None),
                "communicator_id": None if group is None else group.communicator_id,
                "participant_rank_ids": None if group is None else [int(rank) for rank in group.ranks],
                "participant_event_ids": None if group is None else list(group.event_ids),
                "record_event_id": (
                    str(predecessor_event_id)
                    if edge_kind == "cuda_event_wait" and predecessor_event_id not in (None, "")
                    else None
                ),
                "wait_event_id": (
                    str(successor_event_id)
                    if edge_kind == "cuda_event_wait" and successor_event_id not in (None, "")
                    else None
                ),
                "start_us": edge.get("affected_interval_start_us"),
                "ready_us": edge.get("release_us"),
                "end_us": edge.get("affected_interval_end_us"),
                "completion_us": None if sim is None else sim.get("end_us"),
                "wait_start_us": edge.get("wait_start_us"),
                "release_us": edge.get("release_us"),
                "waited_us": edge.get("waited_us"),
                "release_reason": edge.get("release_reason"),
                "released_by_event_id": edge.get("released_by_event_id") or (
                    predecessor_event.id if predecessor_event is not None else None
                ),
                "phase_anchor_id": None if block is None else block.get("phase_anchor_id"),
                "phase_anchor_type": None if block is None else block.get("phase_anchor_type"),
                "stable_block_id": None if block is None else block.get("stable_block_id"),
                "predecessor_chain_depth": None,
                "terminal_rank_candidate": (
                    str(successor_simulated_id) in terminal_simulated_event_ids
                    if successor_simulated_id not in (None, "")
                    else False
                ),
                "critical_rank_candidate": (
                    critical_rank is not None
                    and event is not None
                    and int(event.rank) == int(critical_rank)
                ),
            }
        )
    rows.sort(key=lambda row: str(row["stable_replay_edge_id"]))
    return rows


def _generic_replay_component_kind(
    interval: dict[str, Any],
    event: AnnotatedEvent | None,
) -> str:
    simulated_event_id = str(interval.get("simulated_event_id") or "")
    if simulated_event_id.endswith(":host_dispatch"):
        return "host_dispatch_interval"
    if event is not None and event.api == "__hostDelay__":
        return "host_control_interval"
    resource_kind = interval.get("resource_kind")
    if resource_kind == "host":
        return "host_interval"
    if resource_kind == "stream":
        return "stream_operation_interval"
    return "replay_interval"


def _generic_replay_resource_id(interval: dict[str, Any]) -> str | None:
    resource_kind = interval.get("resource_kind")
    if resource_kind == "host":
        return (
            str(interval.get("host_resource_id") or interval.get("host_dispatch_queue_id"))
            if interval.get("host_resource_id") is not None
            or interval.get("host_dispatch_queue_id") is not None
            else None
        )
    if resource_kind == "stream":
        stream_resource_id = interval.get("stream_resource_id")
        if stream_resource_id not in (None, ""):
            return str(stream_resource_id)
        stream_id = interval.get("stream_id")
        if stream_id not in (None, "") and interval.get("rank") is not None:
            return f"rank:{int(interval['rank'])}:stream:{stream_id}"
    return None


def _stable_generic_replay_component_row_id(
    interval: dict[str, Any],
    *,
    component_kind: str,
    resource_id: str | None,
) -> str:
    source = "|".join(
        [
            str(interval.get("simulated_event_id") or ""),
            str(interval.get("materialized_event_id") or ""),
            component_kind,
            str(interval.get("resource_kind") or ""),
            str(resource_id or ""),
        ]
    )
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return f"generic_replay_placement_envelope:{component_kind}:{digest}"


def _generic_replay_edge_tag(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        "stable_replay_edge_id": edge.get("edge_id"),
        "edge_kind": edge.get("edge_kind"),
        "predecessor_simulated_event_id": edge.get("predecessor_simulated_event_id"),
        "predecessor_event_id": edge.get("predecessor_event_id"),
        "predecessor_api": edge.get("predecessor_api"),
        "successor_simulated_event_id": edge.get("successor_simulated_event_id"),
        "successor_event_id": edge.get("successor_event_id"),
        "successor_api": edge.get("successor_api"),
        "resource_kind": edge.get("resource_kind"),
        "resource_id": edge.get("resource_id"),
        "wait_reason": edge.get("wait_reason"),
        "release_reason": edge.get("release_reason"),
        "source": edge.get("source"),
    }


def _unique_generic_replay_edge_tags(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tags: list[dict[str, Any]] = []
    seen: set[str] = set()
    for edge in edges:
        edge_id = str(edge.get("edge_id") or "")
        if edge_id in seen:
            continue
        seen.add(edge_id)
        tags.append(_generic_replay_edge_tag(edge))
    return tags


def _generic_replay_stream_alignment_context(interval: dict[str, Any]) -> dict[str, Any]:
    stream_resource_id = interval.get("stream_resource_id")
    stream_id = interval.get("stream_id")
    if stream_resource_id not in (None, "") or stream_id not in (None, ""):
        return {
            "predicted_stream_id": stream_id,
            "predicted_stream_resource_id": stream_resource_id,
            "stream_namespace_alignment_status": (
                "predicted_replay_stream_namespace_only_actual_alignment_unavailable"
            ),
            "stream_namespace_alignment_unavailable_reason": (
                "actual_stream_namespace_alignment_not_exported_by_replay_edge_diagnostics"
            ),
        }
    return {
        "predicted_stream_id": None,
        "predicted_stream_resource_id": None,
        "stream_namespace_alignment_status": "not_applicable_no_predicted_stream_resource",
        "stream_namespace_alignment_unavailable_reason": None,
    }


def _generic_replay_critical_placement_context(
    *,
    interval: dict[str, Any],
    critical_rank: int | None,
    terminal_simulated_event_ids: set[str],
    critical_path_simulated_event_ids: set[str],
    chain_reconstruction_status: str,
) -> dict[str, Any]:
    simulated_event_id = str(interval.get("simulated_event_id") or "")
    rank_value = interval.get("rank")
    tags: list[str] = []
    if critical_rank is not None and rank_value is not None and int(rank_value) == int(critical_rank):
        tags.append("critical_rank")
    if simulated_event_id in terminal_simulated_event_ids:
        tags.append("critical_terminal_event")
    if simulated_event_id in critical_path_simulated_event_ids:
        tags.append("critical_path_chain_member")
    if chain_reconstruction_status != "reconstructed_from_exported_predecessor_edges":
        status = "unavailable"
        reason = "critical_path_chain_unavailable_no_terminal_simulated_event_match"
    elif tags:
        status = "available"
        reason = None
    else:
        status = "available_not_on_exported_critical_path"
        reason = None
    return {
        "critical_placement_status": status,
        "critical_placement_tags": tags,
        "critical_placement_unavailable_reason": reason,
        "critical_rank": critical_rank,
        "critical_path_chain_reconstruction_status": chain_reconstruction_status,
    }


def _generic_replay_component_kind_for_edge(edge: dict[str, Any]) -> str:
    edge_kind = str(edge.get("edge_kind") or "unknown")
    if edge_kind == "stream_queue_wait_release":
        return "stream_queue_wait"
    if edge_kind in {
        "host_order",
        "stream_order",
        "cuda_event_wait",
        "collective_wait",
        "host_sync_wait",
        "device_sync_wait",
    }:
        return edge_kind
    return f"replay_edge:{edge_kind}"


def _stable_generic_replay_edge_component_row_id(
    edge: dict[str, Any],
    *,
    component_kind: str,
) -> str:
    source = "|".join(
        [
            str(edge.get("edge_id") or ""),
            component_kind,
            str(edge.get("resource_kind") or ""),
            str(edge.get("resource_id") or ""),
            str(edge.get("successor_simulated_event_id") or edge.get("successor_event_id") or ""),
        ]
    )
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return f"generic_replay_placement_envelope:{component_kind}:{digest}"


def _stable_generic_replay_envelope_component_row_id(
    component_kind: str,
    resource_id: str,
) -> str:
    source = f"{component_kind}|{resource_id}"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return f"generic_replay_placement_envelope:{component_kind}:{digest}"


def _generic_replay_actual_timing_unavailable_fields() -> dict[str, Any]:
    return {
        "actual_timing_status": "unavailable",
        "actual_timing_unavailable_reason": (
            "generic_phase1_metadata_export_has_no_strict_actual_counterpart_timing;"
            "endpoint_timestamps_not_used"
        ),
        "actual_start_us": None,
        "actual_end_us": None,
        "actual_duration_us": None,
        "actual_wait_start_us": None,
        "actual_release_us": None,
        "actual_runtime_us": None,
        "actual_endpoint_timestamps_used": False,
        "actual_runtime_direct_substitution": False,
        "safe_to_use_as_repair_evidence": False,
        "safe_to_use_as_subtraction_delta": False,
    }


def _component_strict_actual_timing_unavailable_fields() -> dict[str, Any]:
    return {
        "strict_actual_timing_status": "unavailable",
        "strict_actual_timing_available": False,
        "actual_start_us": None,
        "actual_end_us": None,
        "actual_duration_us": None,
        "actual_timing_basis": "unavailable_no_strict_actual_counterpart_timing",
        "actual_timing_clock_domain": None,
        "actual_timing_common_clock_review_status": "unavailable",
        "actual_timing_source_event_ids": [],
        "actual_timing_unavailable_reason": (
            "component strict-counterpart export is predicted metadata only; "
            "endpoint, host-duration, and observed-runtime fields are not strict timing"
        ),
        "actual_endpoint_timestamps_used_as_strict_timing": False,
        "actual_host_duration_used_as_strict_timing": False,
        "actual_runtime_direct_substitution": False,
        "actual_observed_runtime_used_as_prediction": False,
    }


def _component_strict_unavailable_gate_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "actual_counterpart_join_status": "predicted_metadata_only_actual_join_deferred",
        "actual_counterpart_join_basis": "offline_component_strict_counterpart_join_required",
        "actual_counterpart_unavailable_reason": (
            "actual counterpart metadata is not joined by replay export"
        ),
        "stream_namespace_alignment_status": row.get(
            "stream_namespace_alignment_status",
            "unresolved",
        ),
        "stream_namespace_alignment_basis": row.get(
            "stream_namespace_alignment_unavailable_reason"
        ),
        "stream_namespace_alignment_evidence": None,
        "stream_namespace_mismatch_reason": row.get(
            "stream_namespace_alignment_unavailable_reason"
        ),
        "exact_stream_identity_proven": False,
        "default_stream_equivalence_reviewed": False,
        "cross_trace_stream_namespace_review_status": "unavailable",
        "actual_count_once_group_id": None,
        "count_once_group_basis": row.get("count_once_status"),
        "count_once_status": row.get("count_once_status", "unavailable"),
        "nonoverlap_status": row.get("count_once_non_overlap_status", "unavailable"),
        "nonoverlap_proof_basis": None,
        "double_counting_overlap_status": "unavailable",
        "wait_map_safety_status": row.get("wait_map_safety_status", "unavailable"),
        "wait_map_safety_basis": None,
        "producer_visibility_status": "unavailable",
        "producer_visibility_basis": "producer_visibility_not_exported_by_replay",
        "paper_maya_semantic_component_status": (
            "predicted_component_metadata_only_no_actual_semantic_counterpart_review"
        ),
        "paper_maya_semantic_component_basis": "Maya replay component provenance",
        "repair_ready": False,
        "safe_to_use_as_repair_evidence": False,
        "safe_to_use_as_subtraction_delta": False,
        "safe_to_use_as_repair_evidence_reason": (
            "strict actual timing, count-once/non-overlap, wait-map safety, "
            "stream alignment, and producer visibility are unavailable"
        ),
        "safe_to_use_as_subtraction_delta_reason": (
            "metadata export is not a residual allocation or runtime substitution"
        ),
    }


def _component_strict_predicted_rows(
    *,
    generic_rows: list[dict[str, Any]],
    event_by_id: dict[str, AnnotatedEvent],
    world_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in generic_rows:
        materialized_event_id = row.get("materialized_event_id")
        event = (
            event_by_id.get(str(materialized_event_id))
            if materialized_event_id not in (None, "")
            else None
        )
        extras = {} if event is None else event.extras
        component_row_type = str(row.get("component_row_type") or "component")
        stable_component_row_id = str(row.get("stable_component_row_id") or "")
        stable_edge_id = row.get("stable_replay_edge_id")
        simulated_event_id = row.get("simulated_event_id")
        count_once_group_id = row.get("count_once_group_id")
        component_kind = row.get("component_kind")
        api_or_kernel = row.get("api") or (event.api if event is not None else None)
        resource_kind = row.get("resource_kind")
        resource_id = row.get("resource_id")
        predicted_stream_resource_id = row.get("predicted_stream_resource_id") or row.get(
            "stream_resource_id"
        )
        critical_tags = list(row.get("critical_placement_tags") or [])
        component_row = {
            "component_strict_counterpart_schema_version": (
                _COMPONENT_STRICT_COUNTERPART_SCHEMA_VERSION
            ),
            "component_strict_counterpart_opt_in_flag": True,
            "source_side": "predicted_component_metadata",
            "stable_predicted_component_row_id": stable_component_row_id,
            "stable_predicted_interval_row_id": (
                simulated_event_id if component_row_type == "interval" else None
            ),
            "stable_predicted_edge_row_id": (
                stable_edge_id if component_row_type == "edge" else None
            ),
            "stable_predicted_count_once_group_id": count_once_group_id,
            "predicted_row_identity_basis": (
                "stable replay diagnostic row id from simulated event/edge/resource"
            ),
            "rank": row.get("rank"),
            "world_size": int(world_size),
            "paper_valid_window_id": extras.get("paper_valid_window_id")
            or (extras.get("paper_valid_window_membership") or {}).get("window_id")
            if isinstance(extras.get("paper_valid_window_membership"), dict)
            else extras.get("paper_valid_window_id"),
            "paper_valid_window_membership_status": (
                "available"
                if isinstance(extras.get("paper_valid_window_membership"), dict)
                and extras["paper_valid_window_membership"].get("in_paper_valid_window") is True
                else "unavailable"
            ),
            "component_kind": component_kind,
            "predicted_replay_component": component_row_type,
            "api_or_kernel_family": api_or_kernel,
            "api_or_kernel_family_role": "successor_or_interval_api",
            "material_signature": extras.get("material_signature"),
            "material_signature_status": (
                "available" if extras.get("material_signature") not in (None, "") else "unavailable"
            ),
            "raw_event_id": extras.get("raw_current_event_id") or materialized_event_id,
            "raw_event_ordinal": None if event is None else int(event.ordinal),
            "materialized_event_id": materialized_event_id,
            "host_dispatch_queue_id": row.get("host_resource_id")
            or extras.get("host_dispatch_queue_id"),
            "host_machine_id": extras.get("host_machine_id"),
            "stream_id": row.get("predicted_stream_id") or extras.get("stream_id"),
            "stream_resource_id": predicted_stream_resource_id,
            "collective_group_id": row.get("collective_group_id")
            or (event.collective_group_id if event is not None else None),
            "cuda_event_id": row.get("cuda_event_id") or extras.get("event_id"),
            "predicted_interval_start_us": row.get("predicted_start_us"),
            "predicted_interval_end_us": row.get("predicted_end_us"),
            "predicted_interval_duration_us": row.get("predicted_duration_us"),
            "predicted_interval_duration_basis": (
                "replay_export_interval_or_edge_affected_interval"
            ),
            "predicted_interval_resource_kind": resource_kind,
            "predicted_interval_resource_id": resource_id,
            "predicted_interval_previous_row_id": None,
            "predicted_interval_next_row_id": None,
            "predicted_interval_origin_status": (
                "metadata_only_origin_not_strict_actual_counterpart"
            ),
            "predicted_interval_visibility_status": "producer_visibility_unavailable",
            "predicted_edge_kind": row.get("edge_kind"),
            "predicted_edge_start_row_id": None,
            "predicted_edge_release_row_id": row.get("released_by_event_id"),
            "predicted_edge_waited_us": row.get("waited_us"),
            "predicted_edge_wait_start_us": row.get("wait_start_us"),
            "predicted_edge_release_us": row.get("release_us"),
            "predicted_edge_resource_kind": resource_kind,
            "predicted_edge_resource_id": resource_id,
            "predicted_edge_predecessor_api": (
                (row.get("predecessor_tags") or [{}])[0].get("predecessor_api")
                if row.get("predecessor_tags")
                else None
            ),
            "predicted_edge_successor_api": (
                (row.get("successor_tags") or [{}])[0].get("successor_api")
                if row.get("successor_tags")
                else api_or_kernel
            ),
            "predicted_stream_id": row.get("predicted_stream_id"),
            "predicted_stream_resource_id": predicted_stream_resource_id,
            "predicted_stream_namespace_basis": (
                "rank_scoped_replay_stream_resource_id"
                if predicted_stream_resource_id not in (None, "")
                else "not_applicable_or_unavailable"
            ),
            "actual_stream_id": None,
            "actual_stream_resource_id": None,
            "actual_stream_namespace_basis": None,
            "predicted_count_once_group_id": count_once_group_id,
            "predicted_count_once_interval_id": row.get("count_once_interval_id"),
            "critical_placement_status": row.get("critical_placement_status"),
            "critical_placement_tags": critical_tags,
            "critical_rank": row.get("critical_rank"),
            "critical_path_chain_reconstruction_status": row.get(
                "critical_path_chain_reconstruction_status"
            ),
            "paper_maya_tags": [
                "metadata_only",
                "no_repair",
                "no_runtime_substitution",
                "no_endpoint_timestamp_substitution",
            ],
            **_component_strict_actual_timing_unavailable_fields(),
            **_component_strict_unavailable_gate_fields(row),
        }
        rows.append(component_row)
    rows.sort(key=lambda item: str(item["stable_predicted_component_row_id"]))
    return rows


def _generic_replay_stream_alignment_context_from_edge(edge: dict[str, Any]) -> dict[str, Any]:
    stream_resource_id = edge.get("stream_resource_id")
    if stream_resource_id not in (None, ""):
        return {
            "predicted_stream_id": None,
            "predicted_stream_resource_id": stream_resource_id,
            "stream_namespace_alignment_status": (
                "predicted_replay_stream_namespace_only_actual_alignment_unavailable"
            ),
            "stream_namespace_alignment_unavailable_reason": (
                "actual_stream_namespace_alignment_not_exported_by_replay_edge_diagnostics"
            ),
        }
    return {
        "predicted_stream_id": None,
        "predicted_stream_resource_id": None,
        "stream_namespace_alignment_status": "not_applicable_no_predicted_stream_resource",
        "stream_namespace_alignment_unavailable_reason": None,
    }


def _generic_replay_placement_envelope_rows(
    *,
    event_by_id: dict[str, AnnotatedEvent],
    interval_rows: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    predecessor_edges_by_successor: dict[str, list[dict[str, Any]]],
    critical_rank: int | None,
    terminal_simulated_event_ids: set[str],
    chains: list[list[str]],
    chain_reconstruction_status: str,
    rank_metrics: list[dict[str, Any]],
    global_makespan_us: float,
    critical_path_us: float,
) -> list[dict[str, Any]]:
    successor_edges_by_predecessor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        for key in (
            edge.get("predecessor_simulated_event_id"),
            edge.get("predecessor_event_id"),
        ):
            if key not in (None, ""):
                successor_edges_by_predecessor[str(key)].append(edge)
    critical_path_simulated_event_ids = {event_id for chain in chains for event_id in chain}
    rows: list[dict[str, Any]] = []
    for interval in interval_rows:
        simulated_event_id = str(interval["simulated_event_id"])
        materialized_event_id = str(interval["materialized_event_id"])
        event = event_by_id.get(materialized_event_id)
        component_kind = _generic_replay_component_kind(interval, event)
        resource_id = _generic_replay_resource_id(interval)
        predecessor_edges = list(predecessor_edges_by_successor.get(simulated_event_id, ()))
        if materialized_event_id != simulated_event_id:
            predecessor_edges.extend(predecessor_edges_by_successor.get(materialized_event_id, ()))
        successor_edges = list(successor_edges_by_predecessor.get(simulated_event_id, ()))
        if materialized_event_id != simulated_event_id:
            successor_edges.extend(successor_edges_by_predecessor.get(materialized_event_id, ()))
        predecessor_tags = _unique_generic_replay_edge_tags(predecessor_edges)
        successor_tags = _unique_generic_replay_edge_tags(successor_edges)
        row = {
            "schema_version": _GENERIC_REPLAY_PLACEMENT_ENVELOPE_SCHEMA_VERSION,
            "source_side": "predicted_replay",
            "stable_component_row_id": _stable_generic_replay_component_row_id(
                interval,
                component_kind=component_kind,
                resource_id=resource_id,
            ),
            "component_row_type": "interval",
            "component_kind": component_kind,
            "simulated_event_id": simulated_event_id,
            "materialized_event_id": materialized_event_id,
            "rank": int(interval["rank"]),
            "api": None if event is None else event.api,
            "op_type": interval.get("op_type"),
            "count_once_group_id": f"replay_interval:{simulated_event_id}",
            "count_once_interval_id": simulated_event_id,
            "count_once_status": "metadata_only_count_once_group_not_strict_non_overlap_proof",
            "count_once_non_overlap_status": "unavailable",
            "count_once_non_overlap_unavailable_reason": (
                "generic_phase1_export_groups_predicted_replay_components_but_does_not_prove_"
                "cross_component_non_overlap"
            ),
            "resource_kind": interval.get("resource_kind"),
            "resource_id": resource_id,
            "host_resource_id": interval.get("host_resource_id"),
            "stream_resource_id": interval.get("stream_resource_id"),
            "predicted_start_us": float(interval["start_us"]),
            "predicted_end_us": float(interval["end_us"]),
            "predicted_duration_us": float(interval["duration_us"]),
            "predecessor_successor_status": (
                "available_in_replay_edge_export"
                if predecessor_tags or successor_tags
                else "unavailable_no_replay_predecessor_successor_edges"
            ),
            "predecessor_tags": predecessor_tags,
            "successor_tags": successor_tags,
            **_generic_replay_stream_alignment_context(interval),
            **_generic_replay_critical_placement_context(
                interval=interval,
                critical_rank=critical_rank,
                terminal_simulated_event_ids=terminal_simulated_event_ids,
                critical_path_simulated_event_ids=critical_path_simulated_event_ids,
                chain_reconstruction_status=chain_reconstruction_status,
            ),
            **_generic_replay_actual_timing_unavailable_fields(),
        }
        rows.append(row)
    for edge in edges:
        component_kind = _generic_replay_component_kind_for_edge(edge)
        successor_event_id = edge.get("successor_event_id")
        successor_event = event_by_id.get(str(successor_event_id)) if successor_event_id not in (None, "") else None
        successor_simulated_event_id = edge.get("successor_simulated_event_id")
        edge_rank = int(successor_event.rank) if successor_event is not None else None
        edge_interval = {
            "simulated_event_id": successor_simulated_event_id,
            "rank": edge_rank,
        }
        resource_id = None if edge.get("resource_id") in (None, "") else str(edge["resource_id"])
        rows.append(
            {
                "schema_version": _GENERIC_REPLAY_PLACEMENT_ENVELOPE_SCHEMA_VERSION,
                "source_side": "predicted_replay",
                "stable_component_row_id": _stable_generic_replay_edge_component_row_id(
                    edge,
                    component_kind=component_kind,
                ),
                "component_row_type": "edge",
                "component_kind": component_kind,
                "simulated_event_id": successor_simulated_event_id,
                "materialized_event_id": successor_event_id,
                "rank": edge_rank,
                "api": edge.get("successor_api"),
                "op_type": None,
                "stable_replay_edge_id": edge.get("edge_id"),
                "edge_kind": edge.get("edge_kind"),
                "wait_reason": edge.get("wait_reason"),
                "release_reason": edge.get("release_reason"),
                "wait_key": edge.get("wait_key"),
                "cuda_event_id": edge.get("cuda_event_id"),
                "cuda_event_version": edge.get("cuda_event_version"),
                "collective_group_id": edge.get("collective_group_id"),
                "released_by_event_id": edge.get("released_by_event_id"),
                "count_once_group_id": f"replay_edge:{edge.get('edge_id')}",
                "count_once_interval_id": edge.get("edge_id"),
                "count_once_status": "metadata_only_count_once_group_not_strict_non_overlap_proof",
                "count_once_non_overlap_status": "unavailable",
                "count_once_non_overlap_unavailable_reason": (
                    "generic_phase1_export_groups_predicted_replay_edges_but_does_not_prove_"
                    "cross_component_non_overlap"
                ),
                "resource_kind": edge.get("resource_kind"),
                "resource_id": resource_id,
                "host_resource_id": edge.get("host_resource_id"),
                "stream_resource_id": edge.get("stream_resource_id"),
                "predicted_start_us": edge.get("affected_interval_start_us"),
                "predicted_end_us": edge.get("affected_interval_end_us"),
                "predicted_duration_us": edge.get("affected_interval_duration_us"),
                "predecessor_successor_status": "available_in_replay_edge_export",
                "predecessor_tags": [_generic_replay_edge_tag(edge)],
                "successor_tags": [_generic_replay_edge_tag(edge)],
                **_generic_replay_stream_alignment_context_from_edge(edge),
                **_generic_replay_critical_placement_context(
                    interval=edge_interval,
                    critical_rank=critical_rank,
                    terminal_simulated_event_ids=terminal_simulated_event_ids,
                    critical_path_simulated_event_ids=critical_path_simulated_event_ids,
                    chain_reconstruction_status=chain_reconstruction_status,
                ),
                **_generic_replay_actual_timing_unavailable_fields(),
            }
        )
    for metric in rank_metrics:
        rank = int(metric["rank"])
        resource_id = f"rank:{rank}"
        start_us = float(metric.get("start_offset_us") or 0.0)
        end_us = float(metric.get("end_time_us") or metric.get("total_time_us") or 0.0)
        interval = {"simulated_event_id": f"rank_completion:{rank}", "rank": rank}
        rows.append(
            {
                "schema_version": _GENERIC_REPLAY_PLACEMENT_ENVELOPE_SCHEMA_VERSION,
                "source_side": "predicted_replay",
                "stable_component_row_id": _stable_generic_replay_envelope_component_row_id(
                    "rank_completion_envelope",
                    resource_id,
                ),
                "component_row_type": "envelope",
                "component_kind": "rank_completion_envelope",
                "simulated_event_id": None,
                "materialized_event_id": None,
                "rank": rank,
                "api": None,
                "op_type": "rank_completion",
                "count_once_group_id": f"rank_completion:{rank}",
                "count_once_interval_id": f"rank_completion:{rank}",
                "count_once_status": "metadata_only_count_once_group_not_strict_non_overlap_proof",
                "count_once_non_overlap_status": "unavailable",
                "count_once_non_overlap_unavailable_reason": (
                    "rank_completion_envelope_is_context_for_phase1_not_non_overlap_proof"
                ),
                "resource_kind": "rank",
                "resource_id": resource_id,
                "host_resource_id": None,
                "stream_resource_id": None,
                "predicted_start_us": start_us,
                "predicted_end_us": end_us,
                "predicted_duration_us": float(metric.get("total_time_us") or max(end_us - start_us, 0.0)),
                "predecessor_successor_status": "unavailable_rank_completion_context_only",
                "predecessor_tags": [],
                "successor_tags": [],
                "predicted_stream_id": None,
                "predicted_stream_resource_id": None,
                "stream_namespace_alignment_status": "not_applicable_rank_envelope",
                "stream_namespace_alignment_unavailable_reason": None,
                **_generic_replay_critical_placement_context(
                    interval=interval,
                    critical_rank=critical_rank,
                    terminal_simulated_event_ids=terminal_simulated_event_ids,
                    critical_path_simulated_event_ids=critical_path_simulated_event_ids,
                    chain_reconstruction_status=chain_reconstruction_status,
                ),
                **_generic_replay_actual_timing_unavailable_fields(),
            }
        )
    global_resource_id = "global:makespan"
    rows.append(
        {
            "schema_version": _GENERIC_REPLAY_PLACEMENT_ENVELOPE_SCHEMA_VERSION,
            "source_side": "predicted_replay",
            "stable_component_row_id": _stable_generic_replay_envelope_component_row_id(
                "global_completion_envelope",
                global_resource_id,
            ),
            "component_row_type": "envelope",
            "component_kind": "global_completion_envelope",
            "simulated_event_id": None,
            "materialized_event_id": None,
            "rank": None,
            "api": None,
            "op_type": "global_completion",
            "count_once_group_id": "global_completion",
            "count_once_interval_id": "global_completion",
            "count_once_status": "metadata_only_count_once_group_not_strict_non_overlap_proof",
            "count_once_non_overlap_status": "unavailable",
            "count_once_non_overlap_unavailable_reason": (
                "global_completion_envelope_is_context_for_phase1_not_non_overlap_proof"
            ),
            "resource_kind": "global",
            "resource_id": global_resource_id,
            "host_resource_id": None,
            "stream_resource_id": None,
            "predicted_start_us": 0.0,
            "predicted_end_us": float(global_makespan_us),
            "predicted_duration_us": float(critical_path_us),
            "predecessor_successor_status": "unavailable_global_completion_context_only",
            "predecessor_tags": [],
            "successor_tags": [],
            "predicted_stream_id": None,
            "predicted_stream_resource_id": None,
            "stream_namespace_alignment_status": "not_applicable_global_envelope",
            "stream_namespace_alignment_unavailable_reason": None,
            "critical_placement_status": (
                "available" if critical_rank is not None else "unavailable"
            ),
            "critical_placement_tags": ["global_completion_envelope"],
            "critical_placement_unavailable_reason": (
                None if critical_rank is not None else "critical_rank_unavailable"
            ),
            "critical_rank": critical_rank,
            "critical_path_chain_reconstruction_status": chain_reconstruction_status,
            **_generic_replay_actual_timing_unavailable_fields(),
        }
    )
    rows.sort(key=lambda row: str(row["stable_component_row_id"]))
    return rows


def export_replay_edge_diagnostics(
    trace: AnnotatedTrace,
    replay: ReplayResult,
    *,
    diagnostic_events: list[dict[str, object]] | tuple[dict[str, object], ...] = (),
    target_event_ids: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Export replay intervals and predecessor edges for diagnostics.

    This function is intentionally read-only: it consumes a completed replay and
    optional scheduler diagnostic rows, then reconstructs dependency edges for
    artifact review.  It does not affect scheduling, durations, provider
    selection, or replay metrics.
    """

    selected_ids = None if target_event_ids is None else {str(item) for item in target_event_ids}
    event_by_id = {event.id: event for events in trace.rank_events.values() for event in events}
    wait_key_by_event, record_keys = _build_cuda_event_maps(trace)
    record_event_by_key: dict[_CudaWaitKey, str] = {}
    for event_id, wait_key in wait_key_by_event.items():
        event = event_by_id.get(event_id)
        if event is not None and event.api in _CUDA_EVENT_RECORD_APIS and wait_key in record_keys:
            record_event_by_key[wait_key] = event_id

    simulated_rows = [_as_simulated_dict(event) for event in replay.simulated_events]
    simulated_by_id = {str(row["event_id"]): row for row in simulated_rows}
    canonical_sim_by_base_id: dict[str, dict[str, Any]] = {}
    host_dispatch_sim_by_base_id: dict[str, dict[str, Any]] = {}
    by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_stream: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    interval_rows: list[dict[str, Any]] = []
    for row in simulated_rows:
        simulated_event_id = str(row["event_id"])
        base_id = _base_simulated_event_id(simulated_event_id)
        if base_id is None:
            continue
        if simulated_event_id.endswith(":host_dispatch"):
            host_dispatch_sim_by_base_id[base_id] = row
        else:
            canonical_sim_by_base_id[base_id] = row
        event = event_by_id.get(base_id)
        link_fields = _event_export_metadata(event)
        if link_fields.get("host_control_envelope_counterpart_opt_in_flag") is True:
            resource_id = (
                row.get("host_resource_id")
                or row.get("host_dispatch_queue_id")
                or row.get("stream_resource_id")
                or row.get("stream_id")
            )
            replay_defaults = {
                "host_control_envelope_replay_interval_id": simulated_event_id,
                "host_control_envelope_count_once_interval_id": simulated_event_id,
                "host_control_envelope_replay_resource_kind": row.get("resource_kind"),
                "host_control_envelope_replay_resource_id": resource_id,
                "host_control_envelope_replay_predecessor_successor_status": (
                    "available_in_replay_edge_export"
                ),
            }
            for key, value in replay_defaults.items():
                if link_fields.get(key) in (None, ""):
                    link_fields[key] = value
        if link_fields.get("launch_neighborhood_equivalence_opt_in_flag") is True:
            if link_fields.get("launch_neighborhood_replay_interval_id") in (None, ""):
                link_fields["launch_neighborhood_replay_interval_id"] = simulated_event_id
            if link_fields.get("launch_neighborhood_replay_interval_unavailable_reason") in (None, ""):
                link_fields["launch_neighborhood_replay_interval_unavailable_reason"] = None
        if link_fields.get("gemm_adjacent_hostdelay_opt_in_flag") is True:
            resource_id = (
                row.get("host_resource_id")
                or row.get("host_dispatch_queue_id")
                or row.get("stream_resource_id")
                or row.get("stream_id")
            )
            gemm_replay_defaults = {
                "gemm_adjacent_predicted_replay_interval_id": simulated_event_id,
                "gemm_adjacent_predicted_replay_component_kind": (
                    "host_control_interval"
                    if event is not None and event.api == "__hostDelay__"
                    else row.get("op_type")
                ),
                "gemm_adjacent_predicted_replay_resource_kind": row.get("resource_kind"),
                "gemm_adjacent_predicted_replay_resource_id": resource_id,
                "gemm_adjacent_predicted_start_us": row.get("start_us"),
                "gemm_adjacent_predicted_end_us": row.get("end_us"),
                "gemm_adjacent_predicted_duration_us": row.get("duration_us"),
                "gemm_adjacent_predicted_count_once_interval_id": simulated_event_id,
                "gemm_adjacent_actual_timing_status": "unavailable",
                "gemm_adjacent_actual_endpoint_timestamps_used_as_strict_timing": False,
                "gemm_adjacent_actual_runtime_direct_substitution": False,
                "gemm_adjacent_count_once_non_overlap_status": "unavailable",
                "gemm_adjacent_double_counting_overlap_status": "unavailable",
                "gemm_adjacent_wait_map_safety_status": "unavailable",
                "gemm_adjacent_repair_ready": False,
                "gemm_adjacent_safe_to_use_as_repair_evidence": False,
                "gemm_adjacent_safe_to_use_as_subtraction_delta": False,
            }
            for key, value in gemm_replay_defaults.items():
                if link_fields.get(key) in (None, ""):
                    link_fields[key] = value
        export_row = dict(row)
        export_row.update(
            {
                "simulated_event_id": simulated_event_id,
                "materialized_event_id": base_id,
                "event": _compact_export_event(event),
                "link_fields": link_fields,
            }
        )
        interval_rows.append(export_row)
        if row.get("resource_kind") == "host" and row.get("host_dispatch_queue_id") is not None:
            by_host[str(row["host_dispatch_queue_id"])].append(row)
        if row.get("resource_kind") == "stream" and row.get("stream_id") is not None:
            by_stream[(int(row["rank"]), str(row["stream_id"]))].append(row)
    for rows in by_host.values():
        rows.sort(key=lambda item: (float(item["start_us"]), float(item["end_us"]), str(item["event_id"])))
    for rows in by_stream.values():
        rows.sort(key=lambda item: (float(item["start_us"]), float(item["end_us"]), str(item["event_id"])))
    host_position_by_simulated_id: dict[str, int] = {}
    stream_position_by_simulated_id: dict[str, int] = {}
    for rows in by_host.values():
        for position, row in enumerate(rows):
            host_position_by_simulated_id[str(row["event_id"])] = position
    for rows in by_stream.values():
        for position, row in enumerate(rows):
            stream_position_by_simulated_id[str(row["event_id"])] = position
    for row in interval_rows:
        simulated_event_id = str(row["simulated_event_id"])
        stream_id = row.get("stream_id")
        row["host_resource_id"] = (
            str(row["host_dispatch_queue_id"])
            if row.get("resource_kind") == "host" and row.get("host_dispatch_queue_id") is not None
            else None
        )
        row["stream_resource_id"] = (
            f"rank:{int(row['rank'])}:stream:{stream_id}"
            if row.get("resource_kind") == "stream" and stream_id is not None
            else None
        )
        row["host_queue_position"] = host_position_by_simulated_id.get(simulated_event_id)
        row["stream_queue_position"] = stream_position_by_simulated_id.get(simulated_event_id)
    by_host_end_times = {key: [float(row["end_us"]) for row in rows] for key, rows in by_host.items()}
    by_stream_end_times = {key: [float(row["end_us"]) for row in rows] for key, rows in by_stream.items()}

    edges: list[dict[str, Any]] = []

    def include_successor(successor_id: str) -> bool:
        if selected_ids is None:
            return True
        return successor_id in selected_ids or _base_simulated_event_id(successor_id) in selected_ids

    def event_api(event_id: str | None) -> str | None:
        if event_id is None:
            return None
        event = event_by_id.get(event_id)
        return None if event is None else event.api

    def edge_endpoint_context(
        *,
        predecessor_simulated_event_id: str | None,
        predecessor_event_id: str | None,
        successor_simulated_event_id: str | None,
        successor_event_id: str | None,
    ) -> dict[str, Any]:
        predecessor_materialized_event_id = predecessor_event_id or _base_simulated_event_id(
            predecessor_simulated_event_id
        )
        successor_materialized_event_id = successor_event_id or _base_simulated_event_id(
            successor_simulated_event_id
        )
        return {
            "predecessor_materialized_event_id": predecessor_materialized_event_id,
            "predecessor_api": event_api(predecessor_materialized_event_id),
            "successor_materialized_event_id": successor_materialized_event_id,
            "successor_api": event_api(successor_materialized_event_id),
        }

    def simulated_interval_context(
        simulated: dict[str, Any] | None,
        *,
        start_us: float | None = None,
        end_us: float | None = None,
    ) -> dict[str, Any]:
        if start_us is not None and end_us is not None:
            return {
                "affected_interval_start_us": float(start_us),
                "affected_interval_end_us": float(end_us),
                "affected_interval_duration_us": max(float(end_us) - float(start_us), 0.0),
                "simulated_interval_ids": [] if simulated is None else [str(simulated["event_id"])],
            }
        if simulated is None:
            return {
                "affected_interval_start_us": None,
                "affected_interval_end_us": None,
                "affected_interval_duration_us": None,
                "simulated_interval_ids": [],
            }
        return {
            "affected_interval_start_us": float(simulated["start_us"]),
            "affected_interval_end_us": float(simulated["end_us"]),
            "affected_interval_duration_us": float(simulated["duration_us"]),
            "simulated_interval_ids": [str(simulated["event_id"])],
        }

    def resource_position_context(
        *,
        predecessor_simulated_event_id: str | None,
        successor_simulated_event_id: str | None,
        host_resource_id: str | None = None,
        stream_resource_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "host_resource_id": host_resource_id,
            "stream_resource_id": stream_resource_id,
            "host_queue_position": (
                host_position_by_simulated_id.get(successor_simulated_event_id)
                if successor_simulated_event_id is not None
                else None
            ),
            "stream_queue_position": (
                stream_position_by_simulated_id.get(successor_simulated_event_id)
                if successor_simulated_event_id is not None
                else None
            ),
            "predecessor_host_queue_position": (
                host_position_by_simulated_id.get(predecessor_simulated_event_id)
                if predecessor_simulated_event_id is not None
                else None
            ),
            "predecessor_stream_queue_position": (
                stream_position_by_simulated_id.get(predecessor_simulated_event_id)
                if predecessor_simulated_event_id is not None
                else None
            ),
            "successor_host_queue_position": (
                host_position_by_simulated_id.get(successor_simulated_event_id)
                if successor_simulated_event_id is not None
                else None
            ),
            "successor_stream_queue_position": (
                stream_position_by_simulated_id.get(successor_simulated_event_id)
                if successor_simulated_event_id is not None
                else None
            ),
        }

    for host_queue_id, rows in sorted(by_host.items()):
        predecessor: dict[str, Any] | None = None
        for row in rows:
            successor_id = str(row["event_id"])
            if include_successor(successor_id):
                predecessor_id = None if predecessor is None else str(predecessor["event_id"])
                source = f"host|{host_queue_id}|{predecessor_id}->{successor_id}|{row['start_us']}"
                predecessor_event_id = _base_simulated_event_id(predecessor_id)
                successor_event_id = _base_simulated_event_id(successor_id)
                edges.append(
                    {
                        "edge_id": _diagnostic_edge_id("host_order", predecessor_id, successor_id, source),
                        "edge_kind": "host_order",
                        "predecessor_simulated_event_id": predecessor_id,
                        "predecessor_event_id": predecessor_event_id,
                        "successor_simulated_event_id": successor_id,
                        "successor_event_id": successor_event_id,
                        "resource_kind": "host",
                        "resource_id": host_queue_id,
                        **edge_endpoint_context(
                            predecessor_simulated_event_id=predecessor_id,
                            predecessor_event_id=predecessor_event_id,
                            successor_simulated_event_id=successor_id,
                            successor_event_id=successor_event_id,
                        ),
                        **simulated_interval_context(row),
                        **resource_position_context(
                            predecessor_simulated_event_id=predecessor_id,
                            successor_simulated_event_id=successor_id,
                            host_resource_id=host_queue_id,
                        ),
                        "wait_start_us": None,
                        "release_us": float(row["start_us"]),
                        "waited_us": max(float(row["start_us"]) - float(predecessor["end_us"]), 0.0) if predecessor else 0.0,
                        "wait_reason": None,
                        "release_reason": "host_queue_order",
                        "wait_key": None,
                        "cuda_event_id": None,
                        "cuda_event_version": None,
                        "collective_group_id": None,
                        "released_by_event_id": predecessor_event_id,
                        "double_counting_overlap_status": "unavailable",
                        "wait_map_safety_status": "unavailable",
                        "source": "simulated_host_queue_order",
                    }
                )
            predecessor = row

    for stream_key, rows in sorted(by_stream.items()):
        predecessor = None
        for row in rows:
            successor_id = str(row["event_id"])
            if include_successor(successor_id):
                predecessor_id = None if predecessor is None else str(predecessor["event_id"])
                source = f"stream|{stream_key}|{predecessor_id}->{successor_id}|{row['start_us']}"
                predecessor_event_id = _base_simulated_event_id(predecessor_id)
                successor_event_id = _base_simulated_event_id(successor_id)
                stream_resource_id = f"rank:{stream_key[0]}:stream:{stream_key[1]}"
                edges.append(
                    {
                        "edge_id": _diagnostic_edge_id("stream_order", predecessor_id, successor_id, source),
                        "edge_kind": "stream_order",
                        "predecessor_simulated_event_id": predecessor_id,
                        "predecessor_event_id": predecessor_event_id,
                        "successor_simulated_event_id": successor_id,
                        "successor_event_id": successor_event_id,
                        "resource_kind": "stream",
                        "resource_id": stream_resource_id,
                        **edge_endpoint_context(
                            predecessor_simulated_event_id=predecessor_id,
                            predecessor_event_id=predecessor_event_id,
                            successor_simulated_event_id=successor_id,
                            successor_event_id=successor_event_id,
                        ),
                        **simulated_interval_context(row),
                        **resource_position_context(
                            predecessor_simulated_event_id=predecessor_id,
                            successor_simulated_event_id=successor_id,
                            stream_resource_id=stream_resource_id,
                        ),
                        "wait_start_us": None,
                        "release_us": float(row["start_us"]),
                        "waited_us": max(float(row["start_us"]) - float(predecessor["end_us"]), 0.0) if predecessor else 0.0,
                        "wait_reason": None,
                        "release_reason": "stream_fifo_order",
                        "wait_key": None,
                        "cuda_event_id": None,
                        "cuda_event_version": None,
                        "collective_group_id": row.get("collective_group_id"),
                        "released_by_event_id": predecessor_event_id,
                        "double_counting_overlap_status": "unavailable",
                        "wait_map_safety_status": "unavailable",
                        "source": "simulated_stream_order",
                    }
                )
            predecessor = row

    wait_starts: dict[tuple[str, str], deque[dict[str, object]]] = defaultdict(deque)
    for diagnostic in diagnostic_events:
        kind = str(diagnostic.get("kind") or "")
        event_id = str(diagnostic.get("event_id") or "")
        wait_reason = str(diagnostic.get("wait_reason") or "")
        if kind == "wait_start":
            wait_starts[(event_id, wait_reason)].append(dict(diagnostic))
            continue
        if kind != "wait_release" or not include_successor(event_id):
            continue
        start = wait_starts[(event_id, wait_reason)].popleft() if wait_starts[(event_id, wait_reason)] else {}
        release_reason = str(diagnostic.get("release_reason") or "")
        edge_kind = _diagnostic_wait_kind(wait_reason, release_reason)
        release_us = float(_first_metadata_value(diagnostic.get("time_us"), diagnostic.get("release_us"), default=0.0))
        wait_start_us = float(
            _first_metadata_value(start.get("time_us"), diagnostic.get("wait_start_us"), default=release_us)
        )
        waited_us = float(
            _first_metadata_value(diagnostic.get("waited_us"), default=max(release_us - wait_start_us, 0.0))
        )
        successor_sim = canonical_sim_by_base_id.get(event_id) or host_dispatch_sim_by_base_id.get(event_id)
        predecessor_id: str | None = None
        predecessor_sim: dict[str, Any] | None = None
        wait_key = wait_key_by_event.get(event_id)
        normalized_stream_key: tuple[int, str] | None = None
        if edge_kind == "stream_queue_wait_release":
            stream_key = _parse_diagnostic_tuple(diagnostic.get("stream_key"))
            if stream_key is not None and len(stream_key) >= 2:
                normalized_stream_key = (int(stream_key[0]), str(stream_key[1]))
            elif successor_sim is not None and successor_sim.get("stream_id") is not None:
                normalized_stream_key = (int(successor_sim["rank"]), str(successor_sim["stream_id"]))
            else:
                normalized_stream_key = None
            predecessor_sim = _find_prior_simulated(
                by_stream,
                by_stream_end_times,
                normalized_stream_key,
                release_us,
                str((successor_sim or {}).get("event_id") or event_id),
            )
            predecessor_id = None if predecessor_sim is None else _base_simulated_event_id(str(predecessor_sim["event_id"]))
        elif edge_kind == "cuda_event_wait":
            predecessor_id = record_event_by_key.get(wait_key) if wait_key is not None else None
            predecessor_sim = canonical_sim_by_base_id.get(predecessor_id) if predecessor_id else None
        elif edge_kind in {"collective_wait", "host_sync_wait"}:
            released_by = diagnostic.get("released_by_event_id")
            predecessor_id = str(released_by) if released_by not in (None, "") else None
            predecessor_sim = canonical_sim_by_base_id.get(predecessor_id) if predecessor_id else None
        source = "|".join(
            [
                "wait",
                edge_kind,
                f"event={event_id}",
                f"wait={wait_reason}",
                f"release={release_reason}",
                f"time={release_us}",
            ]
        )
        predecessor_simulated_event_id = None if predecessor_sim is None else str(predecessor_sim["event_id"])
        successor_simulated_event_id = None if successor_sim is None else str(successor_sim["event_id"])
        wait_key_text = str(
            wait_key
            if wait_key is not None
            else diagnostic.get("wait_key") or diagnostic.get("stream_key") or diagnostic.get("group_id") or ""
        )
        stream_resource_id = None
        if normalized_stream_key is not None:
            stream_resource_id = f"rank:{normalized_stream_key[0]}:stream:{normalized_stream_key[1]}"
        elif successor_sim is not None and successor_sim.get("stream_id") is not None:
            stream_resource_id = f"rank:{int(successor_sim['rank'])}:stream:{successor_sim['stream_id']}"
        host_resource_id = (
            str(successor_sim.get("host_dispatch_queue_id"))
            if successor_sim is not None and successor_sim.get("host_dispatch_queue_id") is not None
            else None
        )
        collective_group_id = diagnostic.get("group_id")
        if collective_group_id in (None, ""):
            successor_event = event_by_id.get(event_id)
            collective_group_id = None if successor_event is None else successor_event.collective_group_id
        cuda_event_id = wait_key[1] if wait_key is not None else None
        cuda_event_version = wait_key[2] if wait_key is not None else None
        edges.append(
            {
                "edge_id": _diagnostic_edge_id(edge_kind, predecessor_id, event_id, source),
                "edge_kind": edge_kind,
                "predecessor_simulated_event_id": predecessor_simulated_event_id,
                "predecessor_event_id": predecessor_id,
                "successor_simulated_event_id": successor_simulated_event_id,
                "successor_event_id": event_id,
                "resource_kind": "wait_map",
                "resource_id": wait_key_text,
                **edge_endpoint_context(
                    predecessor_simulated_event_id=predecessor_simulated_event_id,
                    predecessor_event_id=predecessor_id,
                    successor_simulated_event_id=successor_simulated_event_id,
                    successor_event_id=event_id,
                ),
                **simulated_interval_context(successor_sim, start_us=wait_start_us, end_us=release_us),
                **resource_position_context(
                    predecessor_simulated_event_id=predecessor_simulated_event_id,
                    successor_simulated_event_id=successor_simulated_event_id,
                    host_resource_id=host_resource_id,
                    stream_resource_id=stream_resource_id,
                ),
                "wait_reason": wait_reason,
                "release_reason": release_reason,
                "wait_key": wait_key_text or None,
                "cuda_event_id": cuda_event_id,
                "cuda_event_version": cuda_event_version,
                "collective_group_id": collective_group_id,
                "wait_start_us": wait_start_us,
                "release_us": release_us,
                "waited_us": max(waited_us, 0.0),
                "released_by_event_id": diagnostic.get("released_by_event_id"),
                "double_counting_overlap_status": "unavailable",
                "wait_map_safety_status": "unavailable",
                "source": "scheduler_wait_release_diagnostic",
            }
        )

    rank_metrics = [asdict(metric) for metric in replay.rank_metrics]
    critical_metric = max(rank_metrics, key=lambda item: float(item.get("total_time_us") or 0.0), default=None)
    critical_rank = None if critical_metric is None else int(critical_metric["rank"])
    terminal_events = [
        row
        for row in simulated_rows
        if critical_rank is not None
        and int(row.get("rank", -1)) == critical_rank
        and abs(float(row.get("end_us", 0.0)) - float(critical_metric["total_time_us"])) <= _EDGE_TIME_TOLERANCE_US
    ]
    predecessor_edges_by_successor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        successor_id = edge.get("successor_simulated_event_id") or edge.get("successor_event_id")
        if successor_id is not None:
            predecessor_edges_by_successor[str(successor_id)].append(edge)
        successor_event_id = edge.get("successor_event_id")
        if successor_event_id is not None and str(successor_event_id) != str(successor_id):
            predecessor_edges_by_successor[str(successor_event_id)].append(edge)
    if _collective_event_polling_replay_waitmap_release_metadata_enabled():
        wait_edges_by_successor: dict[str, list[dict[str, Any]]] = defaultdict(list)
        collective_edges_by_successor: dict[str, list[dict[str, Any]]] = defaultdict(list)
        collective_edges_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            if edge.get("source") != "scheduler_wait_release_diagnostic":
                continue
            edge_kind = edge.get("edge_kind")
            target: dict[str, list[dict[str, Any]]] | None = None
            if edge_kind == "cuda_event_wait":
                target = wait_edges_by_successor
            elif edge_kind == "collective_wait":
                target = collective_edges_by_successor
                group_id = edge.get("collective_group_id")
                if group_id not in (None, ""):
                    collective_edges_by_group[str(group_id)].append(edge)
            if target is None:
                continue
            for key in (
                edge.get("successor_event_id"),
                edge.get("successor_simulated_event_id"),
            ):
                if key not in (None, ""):
                    target[str(key)].append(edge)
        diagnostic_events_available = bool(diagnostic_events)
        for row in interval_rows:
            materialized_event_id = str(row.get("materialized_event_id") or "")
            link_fields = row.get("link_fields")
            if not materialized_event_id or not isinstance(link_fields, dict):
                continue
            link_fields.update(
                _collective_event_polling_replay_waitmap_metadata(
                    link_fields=link_fields,
                    materialized_event_id=materialized_event_id,
                    event_by_id=event_by_id,
                    wait_edges_by_successor=wait_edges_by_successor,
                    collective_edges_by_successor=collective_edges_by_successor,
                    collective_edges_by_group=collective_edges_by_group,
                    diagnostic_events_available=diagnostic_events_available,
                )
            )
    terminal_simulated_event_ids = {str(row["event_id"]) for row in terminal_events}

    def best_predecessor(successor_id: str) -> str | None:
        candidates = predecessor_edges_by_successor.get(successor_id, [])
        best: tuple[float, str] | None = None
        for edge in candidates:
            pred_sim_id = edge.get("predecessor_simulated_event_id")
            pred_event_id = edge.get("predecessor_event_id")
            candidate_id = str(pred_sim_id or pred_event_id or "")
            if not candidate_id:
                continue
            sim = simulated_by_id.get(candidate_id) or canonical_sim_by_base_id.get(candidate_id)
            end_us = float((sim or {}).get("end_us") or edge.get("release_us") or 0.0)
            if best is None or end_us > best[0]:
                best = (end_us, candidate_id)
        return None if best is None else best[1]

    chains: list[list[str]] = []
    for row in sorted(terminal_events, key=lambda item: str(item["event_id"])):
        chain: list[str] = []
        seen: set[str] = set()
        current = str(row["event_id"])
        while current and current not in seen:
            seen.add(current)
            chain.append(current)
            predecessor = best_predecessor(current)
            if predecessor is None:
                break
            current = predecessor
        chains.append(chain)
    chain_reconstruction_status = (
        "reconstructed_from_exported_predecessor_edges"
        if chains
        else "unavailable_no_terminal_simulated_event_match"
    )

    payload = {
        "schema_version": 1,
        "artifact_type": "replay_edge_diagnostic_export",
        "scope": "diagnostic_only_no_replay_behavior_change",
        "metrics": {
            "critical_path_us": float(replay.critical_path_us),
            "global_makespan_us": float(replay.global_makespan_us),
            "critical_rank": critical_rank,
            "rank_metrics": rank_metrics,
        },
        "field_coverage": {
            "simulated_intervals": len(interval_rows),
            "simulated_intervals_with_host_queue_position": sum(
                1 for row in interval_rows if row.get("host_queue_position") is not None
            ),
            "simulated_intervals_with_stream_queue_position": sum(
                1 for row in interval_rows if row.get("stream_queue_position") is not None
            ),
            "host_control_boundary_metadata_rows": sum(
                1 for row in interval_rows if row.get("link_fields", {}).get("hostdelay_event_id") is not None
            ),
            "predecessor_edges": len(edges),
            "host_order_edges": sum(1 for edge in edges if edge["edge_kind"] == "host_order"),
            "stream_order_edges": sum(1 for edge in edges if edge["edge_kind"] == "stream_order"),
            "predecessor_edges_with_endpoint_api": sum(
                1 for edge in edges if edge.get("successor_api") is not None
            ),
            "predecessor_edges_with_resource_positions": sum(
                1
                for edge in edges
                if edge.get("host_queue_position") is not None
                or edge.get("stream_queue_position") is not None
            ),
            "wait_release_edges": sum(1 for edge in edges if edge["source"] == "scheduler_wait_release_diagnostic"),
            "cuda_event_edges": sum(1 for edge in edges if edge["edge_kind"] == "cuda_event_wait"),
            "collective_edges": sum(1 for edge in edges if edge["edge_kind"] == "collective_wait"),
            "critical_terminal_events": len(terminal_events),
            "critical_terminal_predecessor_chains": len(chains),
        },
        "simulated_events": interval_rows,
        "predecessor_edges": edges,
        "critical_path": {
            "critical_rank": critical_rank,
            "terminal_simulated_event_ids": [str(row["event_id"]) for row in terminal_events],
            "terminal_predecessor_chains_simulated_event_ids": chains,
            "chain_reconstruction_status": chain_reconstruction_status,
        },
    }
    generic_replay_placement_enabled = _generic_replay_placement_envelope_diagnostics_enabled()
    component_strict_enabled = _component_strict_counterpart_diagnostics_enabled()
    generic_rows: list[dict[str, Any]] | None = None
    if generic_replay_placement_enabled or component_strict_enabled:
        generic_rows = _generic_replay_placement_envelope_rows(
            event_by_id=event_by_id,
            interval_rows=interval_rows,
            edges=edges,
            predecessor_edges_by_successor=predecessor_edges_by_successor,
            critical_rank=critical_rank,
            terminal_simulated_event_ids=terminal_simulated_event_ids,
            chains=chains,
            chain_reconstruction_status=chain_reconstruction_status,
            rank_metrics=rank_metrics,
            global_makespan_us=float(replay.global_makespan_us),
            critical_path_us=float(replay.critical_path_us),
        )
    if generic_replay_placement_enabled and generic_rows is not None:
        payload["generic_replay_placement_envelope_phase1_metadata"] = {
            "schema_version": _GENERIC_REPLAY_PLACEMENT_ENVELOPE_SCHEMA_VERSION,
            "diagnostic_only": True,
            "enabled_by_env_flags": list(_GENERIC_REPLAY_PLACEMENT_ENVELOPE_ENV_KEYS),
            "source_side": "predicted_replay",
            "scope": "generic_replay_placement_envelope_phase1_metadata_no_repair",
            "row_count": len(generic_rows),
            "rows": generic_rows,
            "actual_timing_policy": {
                "actual_timing_fields_exported": True,
                "actual_timing_status": "unavailable",
                "endpoint_timestamps_used": False,
                "actual_runtime_direct_substitution": False,
                "unavailable_reason": (
                    "phase1_export_is_predicted_replay_metadata_only_no_actual_timing_counterpart"
                ),
            },
            "safety": {
                "default_off": True,
                "behavior_changing_replay_or_provider_edit": False,
                "native_compare_run": False,
                "live_capture_run": False,
                "actual_runtime_direct_substitution": False,
                "actual_endpoint_timestamp_substitution": False,
                "hostDelay_cap_or_shortening": False,
                "paper_facing_closure_claimed": False,
                "safe_to_use_as_repair_evidence": False,
                "safe_to_use_as_subtraction_delta": False,
            },
        }
        payload["field_coverage"]["generic_replay_placement_envelope_phase1_rows"] = len(
            generic_rows
        )
    if component_strict_enabled and generic_rows is not None:
        component_rows = _component_strict_predicted_rows(
            generic_rows=generic_rows,
            event_by_id=event_by_id,
            world_size=trace.world_size,
        )
        payload["component_strict_counterpart_metadata_evidence"] = {
            "schema_version": _COMPONENT_STRICT_COUNTERPART_SCHEMA_VERSION,
            "diagnostic_only": True,
            "enabled_by_env_flags": list(_COMPONENT_STRICT_COUNTERPART_ENV_KEYS),
            "source_side": "predicted_component_metadata",
            "scope": "component_strict_counterpart_predicted_metadata_no_repair",
            "row_count": len(component_rows),
            "rows": component_rows,
            "actual_timing_policy": {
                "strict_actual_timing_status": "unavailable",
                "actual_start_end_duration_exported_as_null": True,
                "endpoint_timestamps_used_as_strict_timing": False,
                "host_duration_used_as_strict_timing": False,
                "actual_runtime_direct_substitution": False,
                "observed_runtime_used_as_prediction": False,
            },
            "safety": {
                "default_off": True,
                "additive_metadata_only": True,
                "behavior_changing_replay_or_provider_edit": False,
                "native_compare_run": False,
                "live_capture_run": False,
                "actual_runtime_direct_substitution": False,
                "actual_endpoint_timestamp_substitution": False,
                "hostDelay_cap_or_shortening": False,
                "residual_fitting_or_subtraction_delta": False,
                "paper_facing_closure_claimed": False,
                "repair_ready": False,
                "safe_to_use_as_repair_evidence": False,
                "safe_to_use_as_subtraction_delta": False,
            },
        }
        payload["field_coverage"]["component_strict_counterpart_metadata_rows"] = len(
            component_rows
        )
    if _appendix_ab_p2p_component_diagnostics_enabled():
        appendix_rows = _appendix_ab_selected_p2p_diagnostic_rows(
            trace=trace,
            event_by_id=event_by_id,
            interval_rows=interval_rows,
            host_dispatch_sim_by_base_id=host_dispatch_sim_by_base_id,
            predecessor_edges_by_successor=predecessor_edges_by_successor,
        )
        payload["appendix_ab_selected_p2p_per_block_diagnostics"] = {
            "schema_version": _APPENDIX_AB_P2P_DIAGNOSTIC_SCHEMA_VERSION,
            "diagnostic_only": True,
            "enabled_by_env_flags": list(_APPENDIX_AB_P2P_DIAGNOSTIC_ENV_KEYS),
            "scope": "selected_ncclSend_ncclRecv_kernel1_blocks_no_repair",
            "trace_window_compatibility": _appendix_ab_trace_window_policy(trace),
            "row_count": len(appendix_rows),
            "rows": appendix_rows,
            "safety": {
                "default_off": True,
                "behavior_changing_replay_or_provider_edit": False,
                "actual_runtime_direct_substitution": False,
                "actual_api_end_ts_used_as_wait_map_release": False,
                "outside_window_actual_timings_included": False,
                "step_window_assignment_bypassed": False,
                "paper_facing_closure_claimed": False,
            },
        }
        payload["field_coverage"]["appendix_ab_selected_p2p_per_block_rows"] = len(appendix_rows)
    if _appendix_ab_allreduce_component_diagnostics_enabled():
        allreduce_rows = _appendix_ab_selected_allreduce_diagnostic_rows(
            trace=trace,
            event_by_id=event_by_id,
            interval_rows=interval_rows,
            host_dispatch_sim_by_base_id=host_dispatch_sim_by_base_id,
            predecessor_edges_by_successor=predecessor_edges_by_successor,
        )
        payload["appendix_ab_selected_nccl_allreduce_per_block_diagnostics"] = {
            "schema_version": _APPENDIX_AB_ALLREDUCE_DIAGNOSTIC_SCHEMA_VERSION,
            "diagnostic_only": True,
            "enabled_by_env_flags": list(_APPENDIX_AB_ALLREDUCE_DIAGNOSTIC_ENV_KEYS),
            "scope": "selected_ncclAllReduce_kernel4_8_gemm2_3_strided2_3_blocks_no_repair",
            "selected_family_label": _APPENDIX_AB_SELECTED_ALLREDUCE_FAMILY_LABEL,
            "selected_motif_key": _APPENDIX_AB_SELECTED_ALLREDUCE_MOTIF_KEY,
            "trace_window_compatibility": _appendix_ab_trace_window_policy(trace),
            "row_count": len(allreduce_rows),
            "rows": allreduce_rows,
            "safety": {
                "default_off": True,
                "behavior_changing_replay_or_provider_edit": False,
                "actual_runtime_direct_substitution": False,
                "actual_api_end_ts_used_as_wait_map_release": False,
                "actual_api_end_ts_used_as_block_timing": False,
                "outside_window_actual_timings_included": False,
                "step_window_assignment_bypassed": False,
                "paper_facing_closure_claimed": False,
                "safe_to_use_as_repair_evidence": False,
                "safe_to_use_as_subtraction_delta": False,
            },
        }
        payload["field_coverage"]["appendix_ab_selected_nccl_allreduce_per_block_rows"] = len(
            allreduce_rows
        )
    shared_phase_anchor_enabled = _shared_phase_anchor_diagnostics_enabled()
    common_basis_enabled = _shared_phase_anchor_common_basis_diagnostics_enabled()
    if shared_phase_anchor_enabled or common_basis_enabled:
        block_rows, block_by_event_id = _shared_phase_anchor_block_rows(
            trace=trace,
            event_by_id=event_by_id,
            interval_rows=interval_rows,
            host_dispatch_sim_by_base_id=host_dispatch_sim_by_base_id,
            predecessor_edges_by_successor=predecessor_edges_by_successor,
            critical_rank=critical_rank,
            terminal_simulated_event_ids=terminal_simulated_event_ids,
            include_common_basis=common_basis_enabled,
        )
        causal_edge_rows = _shared_phase_anchor_causal_edge_rows(
            trace=trace,
            event_by_id=event_by_id,
            edges=edges,
            predecessor_edges_by_successor=predecessor_edges_by_successor,
            block_by_event_id=block_by_event_id,
            critical_rank=critical_rank,
            terminal_simulated_event_ids=terminal_simulated_event_ids,
            simulated_by_id=simulated_by_id,
            canonical_sim_by_base_id=canonical_sim_by_base_id,
        )
        payload["shared_all_rank_phase_anchor_causal_edge_metadata"] = {
            "schema_version": _SHARED_PHASE_ANCHOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "enabled_by_env_flags": (
                list(_SHARED_PHASE_ANCHOR_CAUSAL_EDGE_ENV_KEYS)
                + list(_SHARED_PHASE_ANCHOR_COMMON_BASIS_ENV_KEYS)
            ),
            "source_side": "predicted_replay",
            "scope": "shared_p2p_allreduce_phase_anchor_replay_causal_edges_no_repair",
            "common_basis_key_fields_enabled": common_basis_enabled,
            "common_basis_schema_version": (
                _SHARED_PHASE_ANCHOR_COMMON_BASIS_SCHEMA_VERSION
                if common_basis_enabled
                else None
            ),
            "causal_edge_row_count": len(causal_edge_rows),
            "phase_anchor_block_row_count": len(block_rows),
            "causal_edge_rows": causal_edge_rows,
            "phase_anchor_block_rows": block_rows,
            "safety": {
                "default_off": True,
                "behavior_changing_replay_or_provider_edit": False,
                "actual_runtime_direct_substitution": False,
                "actual_endpoint_end_ts_used_as_release": False,
                "actual_endpoint_end_ts_used_as_block_end": False,
                "hostDelay_cap_or_shortening": False,
                "suffix_shift_repair": False,
                "paper_facing_closure_claimed": False,
                "safe_to_use_as_repair_evidence": False,
                "safe_to_use_as_subtraction_delta": False,
            },
        }
        payload["field_coverage"]["shared_phase_anchor_causal_edge_rows"] = len(causal_edge_rows)
        payload["field_coverage"]["shared_phase_anchor_block_rows"] = len(block_rows)
        if common_basis_enabled:
            payload["field_coverage"]["shared_phase_anchor_common_basis_block_rows"] = len(
                block_rows
            )
    return payload


def _categorize_duration(
    event: AnnotatedEvent,
    duration_us: float | None = None,
) -> tuple[float, float, float, float]:
    accounted_duration_us = event.duration_us if duration_us is None else duration_us
    if event.op_type in _COMPUTE_TYPES:
        return accounted_duration_us, 0.0, 0.0, 0.0
    if event.op_type in _COMM_TYPES:
        return 0.0, accounted_duration_us, 0.0, 0.0
    if event.op_type in _MEMORY_TYPES:
        return 0.0, 0.0, accounted_duration_us, 0.0
    return 0.0, 0.0, 0.0, accounted_duration_us


def _normalize_stream_id(stream_id: object | None) -> str:
    if stream_id in (None, "", "0", "0x0"):
        return "__default_stream__"
    return str(stream_id)


def _normalized_dispatch_scope(value: object | None) -> str | None:
    if value in (None, ""):
        return None
    resolved = str(value).strip().lower()
    if resolved in {"thread", "process", "host_machine"}:
        return resolved
    return None


def _host_dispatch_scope(trace: AnnotatedTrace, event: AnnotatedEvent) -> str:
    raw_dispatch_scope = _normalized_dispatch_scope(event.extras.get("host_timing_dispatch_scope"))
    if raw_dispatch_scope is not None:
        return raw_dispatch_scope
    resolved_dispatch_scope = _normalized_dispatch_scope(trace.host_timing_dispatch_scope_resolved)
    if resolved_dispatch_scope is not None:
        return resolved_dispatch_scope
    return "host_machine"


def _host_lane_pid(trace: AnnotatedTrace, event: AnnotatedEvent) -> int:
    dispatch_scope = _host_dispatch_scope(trace, event)
    if dispatch_scope in {"thread", "process"}:
        return int(event.pid)
    return _CANONICAL_HOST_QUEUE_PID


def _host_lane_tid(trace: AnnotatedTrace, event: AnnotatedEvent) -> int:
    dispatch_scope = _host_dispatch_scope(trace, event)
    if dispatch_scope == "thread":
        return int(event.tid)
    return _CANONICAL_HOST_QUEUE_TID


def _host_machine_id(event: AnnotatedEvent) -> str:
    resolved = str(event.extras.get("host_machine_id") or "").strip()
    if resolved:
        return resolved
    return f"legacy_pid:{int(event.pid)}"


def _host_dispatch_queue_id(event: AnnotatedEvent) -> str:
    resolved = str(event.extras.get("host_dispatch_queue_id") or "").strip()
    if resolved:
        return resolved
    return _host_machine_id(event)


def _host_key(trace: AnnotatedTrace, event: AnnotatedEvent) -> _HostKey:
    dispatch_scope = _host_dispatch_scope(trace, event)
    if dispatch_scope == "host_machine":
        return (
            _host_dispatch_queue_id(event),
            _CANONICAL_HOST_QUEUE_PID,
            _CANONICAL_HOST_QUEUE_TID,
        )
    return (
        _host_dispatch_queue_id(event),
        _host_lane_pid(trace, event),
        _host_lane_tid(trace, event),
    )


def _ordered_host_keys_for_trace(trace: AnnotatedTrace) -> tuple[_HostKey, ...]:
    dispatch_scope = _normalized_dispatch_scope(trace.host_timing_dispatch_scope_resolved) or "host_machine"
    if dispatch_scope == "host_machine" and trace.rank_host_machines:
        dispatch_queue_ids = tuple(
            sorted(
                {
                    str(
                        trace.rank_host_dispatch_queues.get(rank)
                        or host_machine_id
                    ).strip()
                    for rank, host_machine_id in trace.rank_host_machines.items()
                    if str(
                        trace.rank_host_dispatch_queues.get(rank)
                        or host_machine_id
                    ).strip()
                }
            )
        )
        if dispatch_queue_ids:
            return tuple(
                (dispatch_queue_id, _CANONICAL_HOST_QUEUE_PID, _CANONICAL_HOST_QUEUE_TID)
                for dispatch_queue_id in dispatch_queue_ids
            )
    return tuple(
        sorted(
            {
                _host_key(trace, event)
                for rank_events in trace.rank_events.values()
                for event in rank_events
            }
        )
    )


def _record_simulated_event(
    simulated_events: list[SimulatedEvent] | None,
    rank_totals: dict[int, dict[str, float]],
    event: AnnotatedEvent,
    *,
    start_us: float,
    end_us: float,
    collective_group_id: str | None = None,
    resource_kind: str,
    host_key: _HostKey | None = None,
    stream_key: _StreamKey | None = None,
) -> None:
    compute_us, comm_us, mem_us, other_us = _categorize_duration(event, end_us - start_us)
    totals = rank_totals[event.rank]
    totals["compute"] += compute_us
    totals["comm"] += comm_us
    totals["mem"] += mem_us
    totals["other"] += other_us
    totals["count"] += 1
    if simulated_events is not None:
        simulated_events.append(
            SimulatedEvent(
                event_id=event.id,
                rank=event.rank,
                api=event.api,
                op_type=event.op_type,
                start_us=start_us,
                end_us=end_us,
                duration_us=end_us - start_us,
                duration_source=event.duration_source,
                collective_group_id=collective_group_id,
                resource_kind=resource_kind,
                host_machine_id=(_host_machine_id(event) if host_key is not None else None),
                host_dispatch_queue_id=(host_key[0] if host_key is not None else None),
                host_pid=(
                    None
                    if host_key is None or host_key[1] == _CANONICAL_HOST_QUEUE_PID
                    else host_key[1]
                ),
                host_tid=(
                    None
                    if host_key is None or host_key[2] == _CANONICAL_HOST_QUEUE_TID
                    else host_key[2]
                ),
                stream_id=(stream_key[1] if stream_key is not None else None),
            )
        )


def _record_host_dispatch_event(
    simulated_events: list[SimulatedEvent] | None,
    rank_totals: dict[int, dict[str, float]],
    event: AnnotatedEvent,
    *,
    start_us: float,
    end_us: float,
    host_key: _HostKey,
) -> None:
    """Record host-side launch/dispatch overhead separately from device time."""

    totals = rank_totals[event.rank]
    totals["other"] += max(end_us - start_us, 0.0)
    totals["count"] += 1
    if simulated_events is not None:
        simulated_events.append(
            SimulatedEvent(
                event_id=f"{event.id}:host_dispatch",
                rank=event.rank,
                api=f"{event.api}:host_dispatch",
                op_type="host_delay",
                start_us=start_us,
                end_us=end_us,
                duration_us=end_us - start_us,
                duration_source="observed_host_dispatch_overhead",
                resource_kind="host",
                host_machine_id=_host_machine_id(event),
                host_dispatch_queue_id=host_key[0],
                host_pid=(
                    None
                    if host_key[1] == _CANONICAL_HOST_QUEUE_PID
                    else host_key[1]
                ),
                host_tid=(
                    None
                    if host_key[2] == _CANONICAL_HOST_QUEUE_TID
                    else host_key[2]
                ),
            )
        )


def _collective_group_duration_us(ready_waiters: list["_PendingOp"]) -> float:
    metadata_duration_us = 0.0
    for waiter in ready_waiters:
        raw_duration_us = waiter.event.extras.get("collective_group_duration_us")
        if raw_duration_us in (None, ""):
            continue
        try:
            metadata_duration_us = max(metadata_duration_us, float(raw_duration_us))
        except (TypeError, ValueError):
            continue
    if metadata_duration_us > 0.0:
        return metadata_duration_us
    return max(waiter.event.duration_us for waiter in ready_waiters)


def _build_collective_group_durations(trace: AnnotatedTrace) -> dict[str, float]:
    duration_by_group: dict[str, float] = {}
    metadata_duration_by_group: dict[str, float] = {}
    events = trace.global_events or tuple(
        event
        for rank in sorted(trace.rank_events)
        for event in trace.rank_events[rank]
    )
    for event in events:
        group_id = event.collective_group_id
        if group_id in (None, ""):
            continue
        duration_by_group[group_id] = max(duration_by_group.get(group_id, 0.0), event.duration_us)
        raw_duration_us = event.extras.get("collective_group_duration_us")
        if raw_duration_us in (None, ""):
            continue
        try:
            metadata_duration_by_group[group_id] = max(
                metadata_duration_by_group.get(group_id, 0.0),
                float(raw_duration_us),
            )
        except (TypeError, ValueError):
            continue
    return {
        group_id: (metadata_duration_by_group.get(group_id, 0.0) or duration_by_group.get(group_id, 0.0))
        for group_id in set(duration_by_group) | set(metadata_duration_by_group)
    }


def _expand_rank_metrics(
    rank_metrics: list[RankReplayMetrics],
    profiled_rank_groups: dict[int, tuple[int, ...]],
) -> tuple[RankReplayMetrics, ...]:
    metrics_by_rank = {metric.rank: metric for metric in rank_metrics}
    expanded: list[RankReplayMetrics] = []
    for representative, members in sorted(profiled_rank_groups.items()):
        metric = metrics_by_rank.get(representative)
        if metric is None:
            continue
        for rank in members:
            expanded.append(
                RankReplayMetrics(
                    rank=rank,
                    compute_time_us=metric.compute_time_us,
                    communication_time_us=metric.communication_time_us,
                    memory_time_us=metric.memory_time_us,
                    other_time_us=metric.other_time_us,
                    total_time_us=metric.total_time_us,
                    num_events=metric.num_events,
                    utilization=metric.utilization,
                    start_offset_us=metric.start_offset_us,
                    end_time_us=metric.end_time_us,
                )
            )
    return tuple(sorted(expanded, key=lambda item: item.rank))


def _stream_id_for_event(
    event: AnnotatedEvent,
    handle_streams: dict[int, dict[str, str]],
) -> str:
    stream_id = event.extras.get("stream_id")
    if stream_id not in (None, "", "0", "0x0"):
        return _normalize_stream_id(stream_id)
    handle_id = event.extras.get("handle_id")
    if handle_id not in (None, "", "0", "0x0"):
        return handle_streams[event.rank].get(str(handle_id), "__default_stream__")
    return "__default_stream__"


def _event_stream_key(
    event: AnnotatedEvent,
    handle_streams: dict[int, dict[str, str]],
) -> _StreamKey:
    return (event.rank, _stream_id_for_event(event, handle_streams))


def _event_targets_stream(
    event: AnnotatedEvent,
) -> bool:
    if event.collective_group_id is not None:
        return True
    return targets_stream_resource(event.api, event.op_type)


def _host_dispatch_duration_us(event: AnnotatedEvent) -> float:
    """Return direct emulation launch overhead that should occupy the host queue.

    Dispatch-only wrapper timing is not a device runtime.  Maya models this
    host-side launch overhead as a blocking dispatch-queue operation while the
    annotated event duration remains on the accelerator stream.
    """

    contract = str(event.extras.get("wrapper_runtime_contract") or "").strip().lower()
    if contract != "dispatch_only":
        return 0.0
    raw_duration_us = event.extras.get("host_duration_us")
    if raw_duration_us in (None, ""):
        return 0.0
    try:
        return max(float(raw_duration_us), 0.0)
    except (TypeError, ValueError):
        return 0.0


def _host_op_duration_us(event: AnnotatedEvent) -> float:
    dispatch_duration_us = _host_dispatch_duration_us(event)
    if dispatch_duration_us > 0.0:
        return dispatch_duration_us
    return max(event.duration_us, 0.0)


def _event_wait_key(
    event: AnnotatedEvent,
    event_wait_keys: dict[str, _CudaWaitKey],
) -> _CudaWaitKey | None:
    return event_wait_keys.get(event.id)


def _build_cuda_event_maps(trace: AnnotatedTrace) -> tuple[dict[str, _CudaWaitKey], set[_CudaWaitKey]]:
    wait_keys: dict[str, _CudaWaitKey] = {}
    record_keys: set[_CudaWaitKey] = set()
    current_versions: dict[tuple[int, str], int] = {}
    seen_events_between_records: set[tuple[int, str]] = set()
    for rank, events in sorted(trace.rank_events.items()):
        for event in events:
            event_id = event.extras.get("event_id")
            if event_id in (None, "", "0", "0x0"):
                continue
            handle = str(event_id)
            handle_key = (rank, handle)
            if event.api in _CUDA_EVENT_RECORD_APIS:
                version = current_versions.get(handle_key, 0) + 1
                current_versions[handle_key] = version
                seen_events_between_records.discard(handle_key)
                resolved_key = (rank, handle, version)
                wait_keys[event.id] = resolved_key
                record_keys.add(resolved_key)
                continue
            if event.api in {"cudaStreamWaitEvent", *_CUDA_EVENT_HOST_WAIT_APIS}:
                current_version = current_versions.get(handle_key, 0)
                # A step-window trace may start after the record that made this
                # event ready.  Do not bind a wait observed before the first
                # in-window record to a future reuse of the same event handle.
                version = current_version if current_version > 0 else 0
                # The fake runtime aggressively reuses numeric CUDA event
                # handles. Repeated polls/waits for the same recorded handle may
                # appear after the first wait has already observed it as ready.
                # Binding later waits to that same completed version can leave
                # them permanently queued because the scheduler releases waiters
                # only when the record event completes. Treat repeated wait/poll
                # observations between two records of the same handle as
                # already-ready/pre-window waits instead of stale waits.
                if version > 0 and handle_key in seen_events_between_records:
                    wait_keys[event.id] = (rank, handle, 0)
                else:
                    wait_keys[event.id] = (rank, handle, version)
                    seen_events_between_records.add(handle_key)
    return wait_keys, record_keys


@dataclass
class _PendingOp:
    event: AnnotatedEvent
    host_key: _HostKey
    arrival_us: float
    stream_key: _StreamKey | None = None
    wait_key: _CudaWaitKey | None = None
    targets_stream: bool = False
    dispatch_stream_key: _StreamKey | None = None
    host_dispatch_modeled: bool = False
    diagnostic_wait_reason: str | None = None
    diagnostic_wait_start_us: float | None = None
    diagnostic_release_reason: str | None = None
    diagnostic_released_by_event_id: str | None = None
    diagnostic_release_us: float | None = None
    diagnostic_stream_enqueue_us: float | None = None


@dataclass(frozen=True)
class _RunningOp:
    pending: _PendingOp
    start_us: float
    end_us: float
    completion_kind: str = "event"
    stream_busy_claimed: bool = True


@dataclass
class _HostState:
    busy: bool = False
    blocked: bool = False
    last_end_us: float = 0.0


@dataclass
class _StreamState:
    busy: bool = False
    blocked: bool = False
    last_end_us: float = 0.0


@dataclass(order=True)
class _QueueEvent:
    time_us: float
    sequence: int
    kind: str = field(compare=False)
    payload: object = field(compare=False)


class _ReplayScheduler:
    def __init__(
        self,
        trace: AnnotatedTrace,
        *,
        record_simulated_events: bool = True,
        diagnostic_events: list[dict[str, object]] | None = None,
        collective_ablation_predicate: CollectiveAblationPredicate | None = None,
        stream_serialization_ablation_predicate: StreamSerializationAblationPredicate | None = None,
        stream_start_time_override: StreamStartTimeOverride | None = None,
        cuda_event_wait_ablation_predicate: CudaEventWaitAblationPredicate | None = None,
    ) -> None:
        self.trace = trace
        self.rank_ids = sorted(trace.rank_events)
        self.ordered_host_keys: tuple[_HostKey, ...] = _ordered_host_keys_for_trace(trace)
        self.handle_streams: dict[int, dict[str, str]] = {rank: {} for rank in self.rank_ids}
        self.rank_totals = {
            rank: {"compute": 0.0, "comm": 0.0, "mem": 0.0, "other": 0.0, "count": 0}
            for rank in self.rank_ids
        }
        self.simulated_events: list[SimulatedEvent] | None = [] if record_simulated_events else None
        self.diagnostic_events = diagnostic_events
        self.collective_ablation_predicate = collective_ablation_predicate
        self.stream_serialization_ablation_predicate = stream_serialization_ablation_predicate
        self.stream_start_time_override = stream_start_time_override
        self.cuda_event_wait_ablation_predicate = cuda_event_wait_ablation_predicate
        self.host_states: dict[_HostKey, _HostState] = {
            host_key: _HostState() for host_key in self.ordered_host_keys
        }
        self.stream_states: dict[_StreamKey, _StreamState] = {}
        self.pending_host_arrivals: dict[_HostKey, deque[_PendingOp]] = defaultdict(deque)
        self.pending_stream_ops: dict[_StreamKey, deque[_PendingOp]] = defaultdict(deque)
        self.cuda_event_waiters: dict[_CudaWaitKey, list[_PendingOp]] = defaultdict(list)
        self.collective_waiters: dict[str, list[_PendingOp]] = defaultdict(list)
        self.stream_waiters: dict[_StreamKey, list[_PendingOp]] = defaultdict(list)
        self.device_waiters: dict[int, list[_PendingOp]] = defaultdict(list)
        self.cuda_event_ready_times: dict[_CudaWaitKey, float] = {}
        self.rank_host_end_us: dict[int, float] = {rank: 0.0 for rank in self.rank_ids}
        self.rank_stream_end_us: dict[int, float] = {rank: 0.0 for rank in self.rank_ids}
        self.collective_group_durations_us = _build_collective_group_durations(trace)
        self.event_wait_keys, self.cuda_event_record_keys = _build_cuda_event_maps(trace)
        self.current_time_us = 0.0
        self._queue_sequence = 0
        self._known_streams_by_rank: dict[int, set[_StreamKey]] = defaultdict(set)
        self._ordered_stream_keys: list[_StreamKey] = []
        self._known_stream_key_set: set[_StreamKey] = set()
        self._pending_schedule_times: set[float] = set()
        self._host_keys_with_pending: set[_HostKey] = set()
        self._stream_keys_with_pending: set[_StreamKey] = set()
        self._ordered_pending_host_keys: list[_HostKey] = []
        self._ordered_pending_stream_keys: list[_StreamKey] = []
        self._ready_host_keys: set[_HostKey] = set()
        self._ready_stream_keys: set[_StreamKey] = set()
        self._ordered_ready_host_keys: list[_HostKey] = []
        self._ordered_ready_stream_keys: list[_StreamKey] = []
        self._busy_stream_count_by_rank: dict[int, int] = defaultdict(int)
        self._pending_stream_keys_by_rank: dict[int, set[_StreamKey]] = defaultdict(set)

    def _emit_diagnostic(self, kind: str, pending: _PendingOp, **extra: object) -> None:
        if self.diagnostic_events is None:
            return
        event = pending.event
        payload: dict[str, object] = {
            "kind": kind,
            "time_us": float(self.current_time_us),
            "event_id": event.id,
            "rank": int(event.rank),
            "api": event.api,
            "op_type": event.op_type,
            "collective_group_id": event.collective_group_id,
            "host_key": str(pending.host_key),
            "stream_key": str(pending.stream_key) if pending.stream_key is not None else None,
            "wait_key": str(pending.wait_key) if pending.wait_key is not None else None,
        }
        payload.update(extra)
        self.diagnostic_events.append(payload)

    def _mark_wait(self, pending: _PendingOp, reason: str, **extra: object) -> None:
        pending.diagnostic_wait_reason = reason
        pending.diagnostic_wait_start_us = float(self.current_time_us)
        pending.diagnostic_release_reason = None
        pending.diagnostic_released_by_event_id = None
        pending.diagnostic_release_us = None
        self._emit_diagnostic("wait_start", pending, wait_reason=reason, **extra)

    def _mark_release(
        self,
        pending: _PendingOp,
        reason: str,
        *,
        released_by_event_id: str | None = None,
        **extra: object,
    ) -> None:
        pending.diagnostic_release_reason = reason
        pending.diagnostic_released_by_event_id = released_by_event_id
        pending.diagnostic_release_us = float(self.current_time_us)
        waited_us = None
        if pending.diagnostic_wait_start_us is not None:
            waited_us = float(self.current_time_us) - float(pending.diagnostic_wait_start_us)
        self._emit_diagnostic(
            "wait_release",
            pending,
            wait_reason=pending.diagnostic_wait_reason,
            release_reason=reason,
            released_by_event_id=released_by_event_id,
            waited_us=waited_us,
            **extra,
        )

    def _start_reason_payload(self, pending: _PendingOp) -> dict[str, object]:
        waited_us = None
        if pending.diagnostic_wait_start_us is not None:
            end_us = pending.diagnostic_release_us if pending.diagnostic_release_us is not None else self.current_time_us
            waited_us = float(end_us) - float(pending.diagnostic_wait_start_us)
        return {
            "wait_reason": pending.diagnostic_wait_reason,
            "release_reason": pending.diagnostic_release_reason,
            "released_by_event_id": pending.diagnostic_released_by_event_id,
            "wait_start_us": pending.diagnostic_wait_start_us,
            "release_us": pending.diagnostic_release_us,
            "waited_us": waited_us,
        }

    def _host_state(self, host_key: _HostKey) -> _HostState:
        state = self.host_states.get(host_key)
        if state is None:
            state = _HostState()
            self.host_states[host_key] = state
        return state

    def _stream_state(self, stream_key: _StreamKey) -> _StreamState:
        rank, _ = stream_key
        state = self.stream_states.get(stream_key)
        if state is None:
            state = _StreamState()
            self.stream_states[stream_key] = state
        if stream_key not in self._known_stream_key_set:
            self._known_stream_key_set.add(stream_key)
            self._known_streams_by_rank[rank].add(stream_key)
            insort(self._ordered_stream_keys, stream_key)
        return state

    def _set_stream_busy_state(self, stream_key: _StreamKey, busy: bool) -> None:
        stream_state = self._stream_state(stream_key)
        if stream_state.busy == busy:
            return
        rank = stream_key[0]
        if busy:
            self._busy_stream_count_by_rank[rank] += 1
        else:
            current = self._busy_stream_count_by_rank.get(rank, 0)
            next_value = max(current - 1, 0)
            if next_value:
                self._busy_stream_count_by_rank[rank] = next_value
            else:
                self._busy_stream_count_by_rank.pop(rank, None)
        stream_state.busy = busy

    def _mark_host_key_pending(self, host_key: _HostKey) -> None:
        if host_key in self._host_keys_with_pending:
            return
        self._host_keys_with_pending.add(host_key)
        insort(self._ordered_pending_host_keys, host_key)

    def _clear_host_key_pending(self, host_key: _HostKey) -> None:
        if host_key not in self._host_keys_with_pending:
            return
        self._host_keys_with_pending.discard(host_key)
        try:
            self._ordered_pending_host_keys.remove(host_key)
        except ValueError:
            pass

    def _mark_stream_key_pending(self, stream_key: _StreamKey) -> None:
        if stream_key in self._stream_keys_with_pending:
            return
        self._stream_keys_with_pending.add(stream_key)
        insort(self._ordered_pending_stream_keys, stream_key)

    def _clear_stream_key_pending(self, stream_key: _StreamKey) -> None:
        if stream_key not in self._stream_keys_with_pending:
            return
        self._stream_keys_with_pending.discard(stream_key)
        try:
            self._ordered_pending_stream_keys.remove(stream_key)
        except ValueError:
            pass

    def _mark_host_key_ready(self, host_key: _HostKey) -> None:
        if host_key in self._ready_host_keys:
            return
        self._ready_host_keys.add(host_key)
        insort(self._ordered_ready_host_keys, host_key)

    def _clear_host_key_ready(self, host_key: _HostKey) -> None:
        if host_key not in self._ready_host_keys:
            return
        self._ready_host_keys.discard(host_key)
        try:
            self._ordered_ready_host_keys.remove(host_key)
        except ValueError:
            pass

    def _mark_stream_key_ready(self, stream_key: _StreamKey) -> None:
        if stream_key in self._ready_stream_keys:
            return
        self._ready_stream_keys.add(stream_key)
        insort(self._ordered_ready_stream_keys, stream_key)

    def _clear_stream_key_ready(self, stream_key: _StreamKey) -> None:
        if stream_key not in self._ready_stream_keys:
            return
        self._ready_stream_keys.discard(stream_key)
        try:
            self._ordered_ready_stream_keys.remove(stream_key)
        except ValueError:
            pass

    def _push_queue_event(
        self,
        queue: list[_QueueEvent],
        *,
        time_us: float,
        kind: str,
        payload: object,
    ) -> None:
        heappush(queue, _QueueEvent(float(time_us), self._queue_sequence, kind, payload))
        self._queue_sequence += 1

    def _push_schedule(self, queue: list[_QueueEvent], time_us: float | None = None) -> None:
        schedule_time_us = self.current_time_us if time_us is None else float(time_us)
        if schedule_time_us in self._pending_schedule_times:
            return
        self._pending_schedule_times.add(schedule_time_us)
        self._push_queue_event(
            queue,
            time_us=schedule_time_us,
            kind="schedule",
            payload=None,
        )

    def handle_arrival(self, pending: _PendingOp, queue: list[_QueueEvent]) -> None:
        host_state = self._host_state(pending.host_key)
        if host_state.busy:
            self.pending_host_arrivals[pending.host_key].append(pending)
            self._mark_wait(
                pending,
                "host_queue_busy",
                queue_depth=len(self.pending_host_arrivals[pending.host_key]),
            )
            self._mark_host_key_pending(pending.host_key)
            self._clear_host_key_ready(pending.host_key)
        else:
            self._submit_from_host(pending, queue)
        self._push_schedule(queue)

    def handle_end(self, running: _RunningOp, queue: list[_QueueEvent]) -> None:
        event = running.pending.event
        if running.completion_kind == "host_dispatch":
            self._complete_host_dispatch(running, queue)
            return
        if running.pending.stream_key is None:
            host_state = self._host_state(running.pending.host_key)
            host_state.busy = False
            host_state.blocked = False
            host_state.last_end_us = max(host_state.last_end_us, running.end_us)
            self.rank_host_end_us[event.rank] = max(self.rank_host_end_us[event.rank], running.end_us)
            self._apply_host_side_effects(event)
            if self.pending_host_arrivals.get(running.pending.host_key):
                self._mark_host_key_ready(running.pending.host_key)
        else:
            stream_state = self._stream_state(running.pending.stream_key)
            if running.stream_busy_claimed:
                self._set_stream_busy_state(running.pending.stream_key, False)
            stream_state.blocked = False
            stream_state.last_end_us = max(stream_state.last_end_us, running.end_us)
            self.rank_stream_end_us[event.rank] = max(self.rank_stream_end_us[event.rank], running.end_us)
            if event.api in _CUDA_EVENT_RECORD_APIS and running.pending.wait_key is not None:
                self.cuda_event_ready_times[running.pending.wait_key] = running.end_us
                self._release_cuda_waiters(running.pending.wait_key, queue)
            self._release_stream_and_device_waiters(running.pending.stream_key, queue)
            if self.pending_stream_ops.get(running.pending.stream_key):
                self._mark_stream_key_ready(running.pending.stream_key)
        self._push_schedule(queue)

    def _complete_host_dispatch(self, running: _RunningOp, queue: list[_QueueEvent]) -> None:
        pending = running.pending
        event = pending.event
        host_state = self._host_state(pending.host_key)
        host_state.busy = False
        host_state.blocked = False
        host_state.last_end_us = max(host_state.last_end_us, running.end_us)
        self.rank_host_end_us[event.rank] = max(self.rank_host_end_us[event.rank], running.end_us)

        stream_key = pending.dispatch_stream_key
        if stream_key is None:
            raise RuntimeError(f"host dispatch completed without stream key for event {event.id}")
        pending.stream_key = stream_key
        pending.dispatch_stream_key = None
        self._enqueue_stream_op_after_dispatch(pending, queue)

        if self.pending_host_arrivals.get(pending.host_key):
            self._mark_host_key_ready(pending.host_key)
        self._push_schedule(queue)

    def handle_schedule(self, queue: list[_QueueEvent]) -> None:
        progress = True
        while progress:
            progress = False
            for host_key in tuple(self._ordered_ready_host_keys):
                if self._drain_host_arrivals(host_key, queue):
                    progress = True
            for stream_key in tuple(self._ordered_ready_stream_keys):
                if self._try_start_next_stream_op(stream_key, queue):
                    progress = True

    def finalize(self) -> None:
        busy_hosts = [thread for thread, state in self.host_states.items() if state.busy]
        busy_streams = [stream for stream, state in self.stream_states.items() if state.busy]
        pending_thread = {host: len(queue) for host, queue in self.pending_host_arrivals.items() if queue}
        pending_stream = {stream: len(queue) for stream, queue in self.pending_stream_ops.items() if queue}
        cuda_waits = {key: len(waiters) for key, waiters in self.cuda_event_waiters.items() if waiters}
        collective_waits = {key: len(waiters) for key, waiters in self.collective_waiters.items() if waiters}
        if busy_hosts or busy_streams or pending_thread or pending_stream or cuda_waits or collective_waits:
            raise RuntimeError(
                "paper-style replay ended with unresolved state: "
                f"busy_hosts={busy_hosts}, busy_streams={busy_streams}, "
                f"pending_thread={pending_thread}, pending_stream={pending_stream}, "
                f"cuda_waits={cuda_waits}, collective_waits={collective_waits}"
            )

    def build_result(self, *, expand_profiled_rank_groups: bool) -> ReplayResult:
        rank_metrics: list[RankReplayMetrics] = []
        for rank in self.rank_ids:
            totals = self.rank_totals[rank]
            host_time_us = self.rank_host_end_us.get(rank, 0.0)
            stream_time_us = self.rank_stream_end_us.get(rank, 0.0)
            total_time_us = max(host_time_us, stream_time_us)
            start_offset_us = min(
                (
                    event.start_us
                    for event in (self.simulated_events or ())
                    if event.rank == rank
                ),
                default=0.0,
            )
            end_time_us = start_offset_us + total_time_us
            compute_time_us = totals["compute"]
            utilization = compute_time_us / total_time_us if total_time_us > 0 else 0.0
            rank_metrics.append(
                RankReplayMetrics(
                    rank=rank,
                    compute_time_us=compute_time_us,
                    communication_time_us=totals["comm"],
                    memory_time_us=totals["mem"],
                    other_time_us=totals["other"],
                    total_time_us=total_time_us,
                    num_events=totals["count"],
                    utilization=utilization,
                    start_offset_us=start_offset_us,
                    end_time_us=end_time_us,
                )
            )

        critical_path_us = max((metric.total_time_us for metric in rank_metrics), default=0.0)
        global_makespan_us = max((metric.end_time_us for metric in rank_metrics), default=0.0) - min(
            (metric.start_offset_us for metric in rank_metrics),
            default=0.0,
        )
        rank0_time_us = next((metric.total_time_us for metric in rank_metrics if metric.rank == 0), None)
        final_rank_metrics: tuple[RankReplayMetrics, ...]
        if (
            expand_profiled_rank_groups
            and self.trace.profiled_rank_groups
            and not self.trace.logical_rank_materialized
        ):
            final_rank_metrics = _expand_rank_metrics(rank_metrics, self.trace.profiled_rank_groups)
        else:
            final_rank_metrics = tuple(rank_metrics)
        return ReplayResult(
            total_time_us=critical_path_us,
            critical_path_us=critical_path_us,
            global_makespan_us=global_makespan_us,
            rank0_time_us=rank0_time_us,
            success=True,
            rank_metrics=final_rank_metrics,
            simulated_events=(
                ()
                if self.simulated_events is None
                else tuple(sorted(self.simulated_events, key=lambda event: (event.start_us, event.rank, event.event_id)))
            ),
        )

    def _apply_host_side_effects(self, event: AnnotatedEvent) -> None:
        if event.api == "cublasCreate_v2":
            handle_id = event.extras.get("handle_id")
            if handle_id not in (None, "", "0", "0x0"):
                self.handle_streams[event.rank][str(handle_id)] = "__default_stream__"
        elif event.api == "cublasSetStream_v2":
            handle_id = event.extras.get("handle_id")
            stream_id = event.extras.get("stream_id")
            if handle_id not in (None, "", "0", "0x0"):
                self.handle_streams[event.rank][str(handle_id)] = _normalize_stream_id(stream_id)

    def _submit_from_host(self, pending: _PendingOp, queue: list[_QueueEvent]) -> None:
        if pending.stream_key is None and (
            pending.targets_stream or pending.event.api == "cudaStreamSynchronize"
        ):
            pending.stream_key = _event_stream_key(pending.event, self.handle_streams)
        if pending.stream_key is None:
            self._try_start_host_op(pending, queue)
            return
        dispatch_duration_us = _host_dispatch_duration_us(pending.event)
        if dispatch_duration_us > 0.0 and not pending.host_dispatch_modeled:
            pending.dispatch_stream_key = pending.stream_key
            pending.stream_key = None
            pending.host_dispatch_modeled = True
            self._start_host_dispatch(pending, queue, self.current_time_us, dispatch_duration_us)
            return
        self._enqueue_stream_op_after_dispatch(pending, queue)

    def _enqueue_stream_op_after_dispatch(self, pending: _PendingOp, queue: list[_QueueEvent]) -> None:
        if pending.stream_key is None:
            raise RuntimeError(f"stream op missing stream key for event {pending.event.id}")
        stream_state = self._stream_state(pending.stream_key)
        pending.diagnostic_stream_enqueue_us = float(self.current_time_us)
        self.pending_stream_ops[pending.stream_key].append(pending)
        self._mark_wait(
            pending,
            "stream_queue_wait",
            queue_depth=len(self.pending_stream_ops[pending.stream_key]),
            stream_busy=bool(stream_state.busy),
        )
        self._mark_stream_key_pending(pending.stream_key)
        self._pending_stream_keys_by_rank[pending.stream_key[0]].add(pending.stream_key)
        if not stream_state.busy:
            self._mark_stream_key_ready(pending.stream_key)

    def _drain_host_arrivals(self, host_key: _HostKey, queue: list[_QueueEvent]) -> bool:
        host_state = self._host_state(host_key)
        if host_state.busy:
            self._clear_host_key_ready(host_key)
            return False
        pending_arrivals = self.pending_host_arrivals.get(host_key)
        if not pending_arrivals:
            self._clear_host_key_ready(host_key)
            return False

        progressed = False
        while pending_arrivals and not host_state.busy:
            pending = pending_arrivals.popleft()
            self._submit_from_host(pending, queue)
            progressed = True
            host_state = self._host_state(host_key)
        if not pending_arrivals:
            self._clear_host_key_pending(host_key)
            self._clear_host_key_ready(host_key)
        elif host_state.busy:
            self._clear_host_key_ready(host_key)
        return progressed

    def _try_start_host_op(self, pending: _PendingOp, queue: list[_QueueEvent]) -> None:
        event = pending.event
        if event.api == "cudaEventQuery":
            # cudaEventQuery is a non-blocking poll.  If an application spins on
            # it, Maya should see that as explicit query calls plus host_delay
            # gaps between them.  Blocking the query itself on CudaEventWaitMap
            # double-counts direct emulation wall-clock hostDelay and makes the
            # replay stricter than CUDA / the paper's wait-map semantics.
            self._start_host_op(pending, queue, self.current_time_us)
            return
        if event.api in _CUDA_EVENT_HOST_WAIT_APIS and pending.wait_key is not None:
            if pending.wait_key not in self.cuda_event_record_keys:
                self._start_host_op(pending, queue, self.current_time_us)
            elif pending.wait_key in self.cuda_event_ready_times:
                self._start_host_op(pending, queue, self.current_time_us)
            else:
                self._block_host_on_cuda_event(pending)
            return
        if event.api == "cudaStreamSynchronize" and pending.stream_key is not None:
            if self._stream_is_quiescent(pending.stream_key):
                self._start_host_op(pending, queue, self.current_time_us)
            else:
                self._block_host_on_stream(pending)
            return
        if event.api == "cudaDeviceSynchronize":
            if self._rank_streams_quiescent(event.rank):
                self._start_host_op(pending, queue, self.current_time_us)
            else:
                self._block_host_on_device(pending)
            return
        self._start_host_op(pending, queue, self.current_time_us)

    def _try_start_next_stream_op(self, stream_key: _StreamKey, queue: list[_QueueEvent]) -> bool:
        stream_state = self._stream_state(stream_key)
        if stream_state.busy:
            self._clear_stream_key_ready(stream_key)
            return False
        pending_ops = self.pending_stream_ops.get(stream_key)
        if not pending_ops:
            self._clear_stream_key_pending(stream_key)
            self._pending_stream_keys_by_rank[stream_key[0]].discard(stream_key)
            self._clear_stream_key_ready(stream_key)
            return False

        pending = pending_ops.popleft()
        self._mark_release(
            pending,
            "stream_queue_head",
            queue_depth_after_pop=len(pending_ops),
        )
        if not pending_ops:
            self._clear_stream_key_pending(stream_key)
            self._pending_stream_keys_by_rank[stream_key[0]].discard(stream_key)
            self._clear_stream_key_ready(stream_key)
        event = pending.event
        if event.collective_group_id is not None:
            self._join_collective(pending, queue)
            return True
        if event.api == "cudaStreamWaitEvent" and pending.wait_key is not None:
            if pending.wait_key not in self.cuda_event_record_keys:
                self._start_stream_op_with_override(pending, queue)
            elif pending.wait_key in self.cuda_event_ready_times:
                self._start_stream_op_with_override(pending, queue)
            elif (
                self.cuda_event_wait_ablation_predicate is not None
                and self.cuda_event_wait_ablation_predicate(pending)
            ):
                self._mark_release(
                    pending,
                    "cuda_event_wait_ablation",
                    released_by_event_id=None,
                    ablated_cuda_event_wait=True,
                )
                self._start_stream_op_with_override(pending, queue)
            else:
                self._block_stream_on_cuda_event(pending)
            return True
        self._start_stream_op_with_override(pending, queue)
        return True

    def _start_stream_op_with_override(self, pending: _PendingOp, queue: list[_QueueEvent]) -> None:
        start_us = float(self.current_time_us)
        if self.stream_start_time_override is not None:
            start_us = max(float(self.stream_start_time_override(pending, start_us, self)), start_us)
        self._start_stream_op(pending, queue, start_us)

    def _start_host_op(self, pending: _PendingOp, queue: list[_QueueEvent], start_us: float) -> None:
        host_state = self._host_state(pending.host_key)
        host_state.busy = True
        host_state.blocked = False
        end_us = start_us + _host_op_duration_us(pending.event)
        self._emit_diagnostic(
            "start_host_op",
            pending,
            start_us=float(start_us),
            end_us=float(end_us),
            **self._start_reason_payload(pending),
        )
        _record_simulated_event(
            self.simulated_events,
            self.rank_totals,
            pending.event,
            start_us=start_us,
            end_us=end_us,
            resource_kind="host",
            host_key=pending.host_key,
        )
        self._push_queue_event(
            queue,
            time_us=end_us,
            kind="end",
            payload=_RunningOp(pending=pending, start_us=start_us, end_us=end_us),
        )

    def _start_host_dispatch(
        self,
        pending: _PendingOp,
        queue: list[_QueueEvent],
        start_us: float,
        dispatch_duration_us: float,
    ) -> None:
        host_state = self._host_state(pending.host_key)
        host_state.busy = True
        host_state.blocked = False
        end_us = start_us + max(dispatch_duration_us, 0.0)
        self._emit_diagnostic(
            "start_host_dispatch",
            pending,
            start_us=float(start_us),
            end_us=float(end_us),
            dispatch_duration_us=float(dispatch_duration_us),
            **self._start_reason_payload(pending),
        )
        _record_host_dispatch_event(
            self.simulated_events,
            self.rank_totals,
            pending.event,
            start_us=start_us,
            end_us=end_us,
            host_key=pending.host_key,
        )
        self._push_queue_event(
            queue,
            time_us=end_us,
            kind="end",
            payload=_RunningOp(
                pending=pending,
                start_us=start_us,
                end_us=end_us,
                completion_kind="host_dispatch",
            ),
        )

    def _start_stream_op(self, pending: _PendingOp, queue: list[_QueueEvent], start_us: float) -> None:
        if pending.stream_key is None:
            raise RuntimeError(f"stream op missing stream key for event {pending.event.id}")
        stream_state = self._stream_state(pending.stream_key)
        ablate_stream_serialization = (
            self.stream_serialization_ablation_predicate is not None
            and self.stream_serialization_ablation_predicate(pending)
        )
        if not ablate_stream_serialization:
            self._set_stream_busy_state(pending.stream_key, True)
        stream_state.blocked = False
        end_us = start_us + max(pending.event.duration_us, 0.0)
        self._emit_diagnostic(
            "start_stream_op",
            pending,
            start_us=float(start_us),
            end_us=float(end_us),
            ablated_stream_serialization=ablate_stream_serialization,
            **self._start_reason_payload(pending),
        )
        _record_simulated_event(
            self.simulated_events,
            self.rank_totals,
            pending.event,
            start_us=start_us,
            end_us=end_us,
            collective_group_id=pending.event.collective_group_id,
            resource_kind="stream",
            stream_key=pending.stream_key,
        )
        self._push_queue_event(
            queue,
            time_us=end_us,
            kind="end",
            payload=_RunningOp(
                pending=pending,
                start_us=start_us,
                end_us=end_us,
                stream_busy_claimed=not ablate_stream_serialization,
            ),
        )

    def _block_host_on_cuda_event(self, pending: _PendingOp) -> None:
        host_state = self._host_state(pending.host_key)
        host_state.busy = True
        host_state.blocked = True
        self._mark_wait(pending, "cuda_event_host_wait")
        self.cuda_event_waiters[pending.wait_key].append(pending)

    def _block_stream_on_cuda_event(self, pending: _PendingOp) -> None:
        if pending.stream_key is None:
            raise RuntimeError(f"stream wait missing stream key for event {pending.event.id}")
        stream_state = self._stream_state(pending.stream_key)
        self._set_stream_busy_state(pending.stream_key, True)
        stream_state.blocked = True
        self._mark_wait(pending, "cuda_event_stream_wait")
        self.cuda_event_waiters[pending.wait_key].append(pending)

    def _block_host_on_stream(self, pending: _PendingOp) -> None:
        host_state = self._host_state(pending.host_key)
        host_state.busy = True
        host_state.blocked = True
        self._mark_wait(pending, "host_stream_sync_wait")
        self.stream_waiters[pending.stream_key].append(pending)

    def _block_host_on_device(self, pending: _PendingOp) -> None:
        host_state = self._host_state(pending.host_key)
        host_state.busy = True
        host_state.blocked = True
        self._mark_wait(pending, "host_device_sync_wait")
        self.device_waiters[pending.event.rank].append(pending)

    def _join_collective(self, pending: _PendingOp, queue: list[_QueueEvent]) -> None:
        if pending.stream_key is None:
            raise RuntimeError(f"collective op missing stream key for event {pending.event.id}")
        stream_state = self._stream_state(pending.stream_key)
        self._set_stream_busy_state(pending.stream_key, True)
        stream_state.blocked = True
        group_id = pending.event.collective_group_id
        if group_id is None:
            raise RuntimeError(f"collective op missing group id for event {pending.event.id}")
        group = self.trace.collective_groups[group_id]
        expected_participants = len(group.event_ids)
        if group.participant_count is not None:
            expected_participants = min(expected_participants, int(group.participant_count))
        if (
            self.collective_ablation_predicate is not None
            and self.collective_ablation_predicate(group_id, (pending,))
        ):
            # Causal diagnostic ablation: model selected collectives as
            # participant-local stream operations as soon as each participant
            # reaches the collective.  This removes the selected cross-rank
            # participant barrier without inserting events in the past.
            self._mark_wait(
                pending,
                "collective_participant_wait",
                group_id=group_id,
                ready_count=1,
                expected_participants=expected_participants,
                ablated_collective_participant_wait=True,
            )
            self._mark_release(
                pending,
                "collective_ablation_participant_local",
                released_by_event_id=pending.event.id,
                group_id=group_id,
                ready_count=1,
                expected_participants=expected_participants,
                ablated_collective_participant_wait=True,
            )
            self._start_stream_op(pending, queue, self.current_time_us)
            return
        waiters = self.collective_waiters[group_id]
        waiters.append(pending)
        self._mark_wait(
            pending,
            "collective_participant_wait",
            group_id=group_id,
            ready_count=len(waiters),
            expected_participants=expected_participants,
        )
        if len(waiters) < expected_participants:
            return
        ready_waiters = list(waiters)
        self.collective_waiters.pop(group_id, None)
        duration_us = self.collective_group_durations_us.get(group_id)
        if duration_us is None:
            duration_us = _collective_group_duration_us(ready_waiters)
        start_us = self.current_time_us
        end_us = start_us + duration_us
        for waiter in ready_waiters:
            if waiter.stream_key is None:
                raise RuntimeError(f"collective waiter missing stream key for event {waiter.event.id}")
            self._mark_release(
                waiter,
                "collective_all_participants_ready",
                released_by_event_id=pending.event.id,
                group_id=group_id,
                ready_count=len(ready_waiters),
                expected_participants=expected_participants,
            )
            waiter_stream = self._stream_state(waiter.stream_key)
            self._set_stream_busy_state(waiter.stream_key, True)
            waiter_stream.blocked = False
            self._emit_diagnostic(
                "start_collective_op",
                waiter,
                start_us=float(start_us),
                end_us=float(end_us),
                group_id=group_id,
                ready_count=len(ready_waiters),
                expected_participants=expected_participants,
                **self._start_reason_payload(waiter),
            )
            _record_simulated_event(
                self.simulated_events,
                self.rank_totals,
                waiter.event,
                start_us=start_us,
                end_us=end_us,
                collective_group_id=group_id,
                resource_kind="stream",
                stream_key=waiter.stream_key,
            )
            self._push_queue_event(
                queue,
                time_us=end_us,
                kind="end",
                payload=_RunningOp(pending=waiter, start_us=start_us, end_us=end_us),
            )

    def _release_cuda_waiters(self, wait_key: _CudaWaitKey, queue: list[_QueueEvent]) -> None:
        released = self.cuda_event_waiters.pop(wait_key, [])
        for pending in released:
            if pending.stream_key is None:
                host_state = self._host_state(pending.host_key)
                host_state.busy = True
                host_state.blocked = False
                self._mark_release(pending, "cuda_event_ready", released_by_event_id=None)
                self._start_host_op(pending, queue, self.current_time_us)
            else:
                stream_state = self._stream_state(pending.stream_key)
                self._set_stream_busy_state(pending.stream_key, True)
                stream_state.blocked = False
                self._clear_stream_key_ready(pending.stream_key)
                self._mark_release(pending, "cuda_event_ready", released_by_event_id=None)
                self._start_stream_op(pending, queue, self.current_time_us)

    def _release_stream_and_device_waiters(self, stream_key: _StreamKey, queue: list[_QueueEvent]) -> None:
        if self._stream_is_quiescent(stream_key):
            for pending in self.stream_waiters.pop(stream_key, []):
                self._mark_release(pending, "stream_quiescent", released_by_event_id=None)
                self._start_host_op(pending, queue, self.current_time_us)
        rank = stream_key[0]
        if self._rank_streams_quiescent(rank):
            for pending in self.device_waiters.pop(rank, []):
                self._mark_release(pending, "device_quiescent", released_by_event_id=None)
                self._start_host_op(pending, queue, self.current_time_us)

    def _stream_is_quiescent(self, stream_key: _StreamKey) -> bool:
        stream_state = self._stream_state(stream_key)
        return not stream_state.busy and not self.pending_stream_ops.get(stream_key)

    def _rank_streams_quiescent(self, rank: int) -> bool:
        return (
            self._busy_stream_count_by_rank.get(rank, 0) == 0
            and not self._pending_stream_keys_by_rank.get(rank)
        )


def replay_annotated_trace(
    trace: AnnotatedTrace,
    *,
    expand_profiled_rank_groups: bool = False,
    record_simulated_events: bool = True,
    diagnostic_events: list[dict[str, object]] | None = None,
    collective_ablation_predicate: CollectiveAblationPredicate | None = None,
    stream_serialization_ablation_predicate: StreamSerializationAblationPredicate | None = None,
    stream_start_time_override: StreamStartTimeOverride | None = None,
    cuda_event_wait_ablation_predicate: CudaEventWaitAblationPredicate | None = None,
) -> ReplayResult:
    """Replay an annotated trace using a paper-style discrete-event scheduler."""
    scheduler = _ReplayScheduler(
        trace,
        record_simulated_events=record_simulated_events,
        diagnostic_events=diagnostic_events,
        collective_ablation_predicate=collective_ablation_predicate,
        stream_serialization_ablation_predicate=stream_serialization_ablation_predicate,
        stream_start_time_override=stream_start_time_override,
        cuda_event_wait_ablation_predicate=cuda_event_wait_ablation_predicate,
    )
    event_queue: list[_QueueEvent] = []

    ordered_events = trace.global_events or tuple(
        event
        for rank in sorted(trace.rank_events)
        for event in trace.rank_events[rank]
    )
    rank_window_starts = {
        int(rank): float(window.start_ts)
        for rank, window in trace.fidelity_windows.items()
        if window.is_paper_valid_step_window
    }
    for event in ordered_events:
        pending = _PendingOp(
            event=event,
            host_key=_host_key(trace, event),
            arrival_us=max(float(event.ts) - rank_window_starts.get(event.rank, float(event.ts)), 0.0),
            stream_key=None,
            wait_key=_event_wait_key(event, scheduler.event_wait_keys),
            targets_stream=_event_targets_stream(event),
        )
        scheduler._push_queue_event(
            event_queue,
            time_us=pending.arrival_us,
            kind="arrival",
            payload=pending,
        )

    while event_queue:
        queued = heappop(event_queue)
        scheduler.current_time_us = queued.time_us
        if queued.kind == "end":
            scheduler.handle_end(queued.payload, event_queue)
        elif queued.kind == "arrival":
            scheduler.handle_arrival(queued.payload, event_queue)
        elif queued.kind == "schedule":
            scheduler._pending_schedule_times.discard(queued.time_us)
            scheduler.handle_schedule(event_queue)
        else:
            raise RuntimeError(f"unknown replay queue event kind: {queued.kind}")

    scheduler.finalize()
    return scheduler.build_result(expand_profiled_rank_groups=expand_profiled_rank_groups)
