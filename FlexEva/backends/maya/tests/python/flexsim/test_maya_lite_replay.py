from pathlib import Path

import pytest

from flexsim.estimator import Estimator
from flexsim.maya_lite import (
    AnnotatedEvent,
    AnnotatedTrace,
    CollectiveGroup,
    CollatedEvent,
    CollatedTrace,
    collate_trace_bundle,
    annotate_collated_trace,
    export_replay_edge_diagnostics,
    load_trace_directory,
    replay_annotated_trace,
    pair_seq_collective_ablation_predicate,
)
from flexsim.maya_lite.schema import FidelityWindow, TraceSource


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
}

_HOSTDELAY_SEMANTIC_CLASSIFICATION_FIELDS = {
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
}
_HOSTDELAY_OCCURRENCE_METADATA_FIELDS = {
    "hostdelay_occurrence_metadata_schema_version",
    "hostdelay_occurrence_metadata_opt_in_flag",
    "stable_hostdelay_occurrence_id",
    "hostdelay_occurrence_interval_start_ts_us",
    "hostdelay_occurrence_interval_end_ts_us",
    "hostdelay_occurrence_duration_us",
    "hostdelay_occurrence_raw_boundary_family",
    "hostdelay_occurrence_semantic_boundary_family",
    "hostdelay_occurrence_count_once_status",
    "hostdelay_occurrence_cuda_event_wait_map_safety_status",
    "hostdelay_occurrence_repair_ready",
    "hostdelay_occurrence_safe_to_use_as_subtraction_delta",
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
_COLLECTIVE_EVENT_POLLING_REPLAY_WAITMAP_METADATA_PREFIX = (
    "collective_event_polling_replay_waitmap_"
)
_EVENT_POLLING_BOUNDARY_METADATA_PREFIX = "event_polling_boundary_"
_EVENT_POLLING_BOUNDARY_METADATA_FIELDS = {
    "event_polling_boundary_metadata_schema_version",
    "event_polling_boundary_metadata_opt_in_flag",
    "event_polling_boundary_occurrence_id",
    "event_polling_boundary_raw_family",
    "event_polling_boundary_semantic_family",
    "event_polling_boundary_target_class",
    "event_polling_boundary_polling_class",
    "event_polling_boundary_origin_status",
    "event_polling_boundary_visibility_status",
    "event_polling_boundary_paper_visibility_class",
    "event_polling_boundary_count_once_status",
    "event_polling_boundary_nonoverlap_status",
    "event_polling_boundary_wait_map_safety_status",
    "event_polling_boundary_repair_ready",
    "event_polling_boundary_safe_to_use_as_repair_evidence",
    "event_polling_boundary_safe_to_use_as_subtraction_delta",
    "event_polling_boundary_safe_delta_us",
}
_BOUNDARY_ORIGIN_SUBREGION_METADATA_PREFIX = "boundary_origin_subregion_"
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
    "boundary_origin_subregion_repair_ready",
    "boundary_origin_subregion_safe_to_use_as_repair_evidence",
    "boundary_origin_subregion_safe_to_use_as_subtraction_delta",
    "boundary_origin_subregion_safe_delta_us",
}
_STRICT_SUBREGION_EXTENT_METADATA_PREFIX = "strict_subregion_extent_"
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
    "strict_subregion_extent_timestamp_source_kind",
    "strict_subregion_extent_source_proof_status",
    "strict_subregion_extent_origin_status",
    "strict_subregion_extent_visibility_status",
    "strict_subregion_extent_paper_visibility_class",
    "strict_subregion_extent_count_once_status",
    "strict_subregion_extent_nonoverlap_status",
    "strict_subregion_extent_cuda_event_wait_map_safety_status",
    "strict_subregion_extent_collective_wait_map_safety_status",
    "strict_subregion_extent_repair_ready",
    "strict_subregion_extent_safe_to_use_as_repair_evidence",
    "strict_subregion_extent_safe_to_use_as_subtraction_delta",
    "strict_subregion_extent_safe_delta_us",
    "strict_subregion_extent_runtime_or_endpoint_substitution_used",
    "strict_subregion_extent_hostdelay_shortening_used",
    "strict_subregion_extent_rank_workload_special_case_used",
}


def _manual_p2p_kernel1_trace(
    *,
    send_ts: int = 0,
    send_end_ts: int | None = None,
    recv_ts: int = 5,
    rank0_window: tuple[int, int] = (0, 100),
    rank1_window: tuple[int, int] = (0, 100),
) -> AnnotatedTrace:
    motif = "boundary={api}|kernel=1|gemm=0|strided=0|send={send}|recv={recv}|allreduce=0"
    send_extras = {
        "motif_key": motif.format(api="ncclSend", send=1, recv=0),
        "collective_group_duration_us": 7.0,
        "collective_group_duration_basis": "group_provider:test_provider",
        "provider_name": "test_provider",
        "provider_tier": "trace_signature_stats",
        "material_signature": "group_api=ncclP2P;collective=p2p;count=8;datatype=9",
        "material_signature_inputs": {"count": 8, "datatype": 9},
        "stream_id": "s0",
        "peer": "1",
        "comm_id": "comm-0",
        "host_dispatch_queue_id": "host0:rank:0",
    }
    if send_end_ts is not None:
        send_extras["end_ts"] = send_end_ts
    send = AnnotatedEvent(
        id="r0:p2p-send",
        rank=0,
        ordinal=0,
        source=TraceSource.UNKNOWN,
        ts=send_ts,
        pid=10,
        tid=20,
        module="libnccl.so",
        api="ncclSend",
        op_type="nccl_collective",
        duration_us=3.0,
        duration_source="estimator_provider:test_provider",
        extras=send_extras,
        collective_group_id="ncclP2P|comm:comm-0|members:0-1|pair_seq:0",
    )
    recv = AnnotatedEvent(
        id="r1:p2p-recv",
        rank=1,
        ordinal=0,
        source=TraceSource.UNKNOWN,
        ts=recv_ts,
        pid=11,
        tid=21,
        module="libnccl.so",
        api="ncclRecv",
        op_type="nccl_collective",
        duration_us=4.0,
        duration_source="estimator_provider:test_provider",
        extras={
            "motif_key": motif.format(api="ncclRecv", send=0, recv=1),
            "collective_group_duration_us": 7.0,
            "collective_group_duration_basis": "group_provider:test_provider",
            "provider_name": "test_provider",
            "provider_tier": "trace_signature_stats",
            "material_signature": "group_api=ncclP2P;collective=p2p;count=8;datatype=9",
            "material_signature_inputs": {"count": 8, "datatype": 9},
            "stream_id": "s1",
            "peer": "0",
            "comm_id": "comm-0",
            "host_dispatch_queue_id": "host0:rank:1",
        },
        collective_group_id="ncclP2P|comm:comm-0|members:0-1|pair_seq:0",
    )
    group = CollectiveGroup(
        id="ncclP2P|comm:comm-0|members:0-1|pair_seq:0",
        api="ncclP2P",
        op_type="nccl_collective",
        ranks=(0, 1),
        event_ids=(send.id, recv.id),
        communicator_id="comm-0",
        sequence_number=0,
        communicator_size=2,
        participant_count=2,
        match_basis="communicator_pair_sequence",
    )
    return AnnotatedTrace(
        trace_dir=Path("/tmp/manual-p2p-kernel1"),
        source=TraceSource.UNKNOWN,
        rank_events={0: (send,), 1: (recv,)},
        global_events=(send, recv),
        collective_groups={group.id: group},
        fidelity_windows={
            0: FidelityWindow(
                start_ts=rank0_window[0],
                end_ts=rank0_window[1],
                source="manual",
                is_paper_valid_step_window=True,
            ),
            1: FidelityWindow(
                start_ts=rank1_window[0],
                end_ts=rank1_window[1],
                source="manual",
                is_paper_valid_step_window=True,
            ),
        },
    )


def _manual_allreduce_trace(
    *,
    rank0_ts: int = 0,
    rank0_end_ts: int | None = None,
    rank1_ts: int = 0,
    rank1_end_ts: int | None = None,
    rank0_window: tuple[int, int] = (0, 100),
    rank1_window: tuple[int, int] = (0, 100),
    host_dispatch_duration_us: float | None = None,
) -> AnnotatedTrace:
    group_id = "ncclAllReduce|comm:flexsim-members:0,1|call:42"
    motif_key = (
        "boundary=ncclAllReduce|kernel=4-8|gemm=2-3|strided=2-3|send=0|recv=0|allreduce=1"
    )
    rank_events: dict[int, tuple[AnnotatedEvent, ...]] = {}
    event_ids: list[str] = []
    for rank, ts, end_ts in ((0, rank0_ts, rank0_end_ts), (1, rank1_ts, rank1_end_ts)):
        event_id = f"r{rank}:allreduce"
        event_ids.append(event_id)
        extras = {
            "motif_key": motif_key,
            "collective_group_duration_us": 6.0,
            "collective_group_duration_basis": "group_provider:test_provider",
            "provider_name": "test_provider",
            "provider_tier": "trace_signature_stats",
            "provider_duration_source_expected": "trace_signature_stats",
            "material_signature": "group_api=ncclAllReduce;collective=allreduce;count=16;datatype=9",
            "material_signature_inputs": {"count": 16, "datatype": 9},
            "collective": "allreduce",
            "count": 16,
            "datatype": 9,
            "stream_id": f"s{rank}",
            "comm_id": "comm-0",
            "host_dispatch_queue_id": f"host0:rank:{rank}",
        }
        if end_ts is not None:
            extras["end_ts"] = end_ts
        if host_dispatch_duration_us is not None:
            extras["wrapper_runtime_contract"] = "dispatch_only"
            extras["host_duration_us"] = host_dispatch_duration_us
        rank_events[rank] = (
            AnnotatedEvent(
                id=event_id,
                rank=rank,
                ordinal=0,
                source=TraceSource.UNKNOWN,
                ts=ts,
                pid=10 + rank,
                tid=20 + rank,
                module="libnccl.so",
                api="ncclAllReduce",
                op_type="nccl_collective",
                duration_us=6.0,
                duration_source="manual",
                extras=extras,
                collective_group_id=group_id,
            ),
        )
    return AnnotatedTrace(
        trace_dir=Path("/tmp/manual-allreduce"),
        source=TraceSource.UNKNOWN,
        rank_events=rank_events,
        global_events=rank_events[0] + rank_events[1],
        collective_groups={
            group_id: CollectiveGroup(
                id=group_id,
                api="ncclAllReduce",
                op_type="nccl_collective",
                ranks=(0, 1),
                event_ids=tuple(event_ids),
                communicator_id="comm-0",
                sequence_number=42,
                communicator_size=2,
                participant_count=2,
            )
        },
        fidelity_windows={
            0: FidelityWindow(
                start_ts=rank0_window[0],
                end_ts=rank0_window[1],
                source="manual",
                is_paper_valid_step_window=True,
            ),
            1: FidelityWindow(
                start_ts=rank1_window[0],
                end_ts=rank1_window[1],
                source="manual",
                is_paper_valid_step_window=True,
            ),
        },
    )


def _collated_p2p_kernel1_without_motif_trace() -> CollatedTrace:
    group_id = "ncclP2P|comm:comm-0|members:0-1|pair_seq:0"
    kernel0 = CollatedEvent(
        id="r0:k0",
        rank=0,
        ordinal=0,
        source=TraceSource.UNKNOWN,
        ts=0,
        pid=10,
        tid=20,
        module="libcudart.so",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={"stream_id": "s0", "host_dispatch_queue_id": "host0:rank:0"},
    )
    send = CollatedEvent(
        id="r0:p2p-send",
        rank=0,
        ordinal=1,
        source=TraceSource.UNKNOWN,
        ts=1,
        pid=10,
        tid=20,
        module="libnccl.so",
        api="ncclSend",
        op_type="nccl_collective",
        extras={
            "stream_id": "s0",
            "peer": "1",
            "comm_id": "comm-0",
            "count": 8,
            "datatype": 9,
            "host_dispatch_queue_id": "host0:rank:0",
        },
        prev_event_id=kernel0.id,
        collective_group_id=group_id,
    )
    kernel1 = CollatedEvent(
        id="r1:k0",
        rank=1,
        ordinal=0,
        source=TraceSource.UNKNOWN,
        ts=0,
        pid=11,
        tid=21,
        module="libcudart.so",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={"stream_id": "s1", "host_dispatch_queue_id": "host0:rank:1"},
    )
    recv = CollatedEvent(
        id="r1:p2p-recv",
        rank=1,
        ordinal=1,
        source=TraceSource.UNKNOWN,
        ts=1,
        pid=11,
        tid=21,
        module="libnccl.so",
        api="ncclRecv",
        op_type="nccl_collective",
        extras={
            "stream_id": "s1",
            "peer": "0",
            "comm_id": "comm-0",
            "count": 8,
            "datatype": 9,
            "host_dispatch_queue_id": "host0:rank:1",
        },
        prev_event_id=kernel1.id,
        collective_group_id=group_id,
    )
    group = CollectiveGroup(
        id=group_id,
        api="ncclP2P",
        op_type="nccl_collective",
        ranks=(0, 1),
        event_ids=(send.id, recv.id),
        communicator_id="comm-0",
        sequence_number=0,
        communicator_size=2,
        participant_count=2,
        match_basis="communicator_pair_sequence",
    )
    return CollatedTrace(
        trace_dir=Path("/tmp/collated-p2p-kernel1-no-motif"),
        source=TraceSource.UNKNOWN,
        rank_events={0: (kernel0, send), 1: (kernel1, recv)},
        global_events=(kernel0, kernel1, send, recv),
        collective_groups={group_id: group},
        fidelity_windows={
            0: FidelityWindow(
                start_ts=0,
                end_ts=100,
                source="manual",
                is_paper_valid_step_window=True,
            ),
            1: FidelityWindow(
                start_ts=0,
                end_ts=100,
                source="manual",
                is_paper_valid_step_window=True,
            ),
        },
    )


def _collated_allreduce_selected_without_motif_trace() -> CollatedTrace:
    group_id = "ncclAllReduce|comm:flexsim-members:0,1|call:42"
    rank_events: dict[int, tuple[CollatedEvent, ...]] = {}
    group_event_ids: list[str] = []
    for rank in (0, 1):
        events: list[CollatedEvent] = []
        for index in range(4):
            events.append(
                CollatedEvent(
                    id=f"r{rank}:k{index}",
                    rank=rank,
                    ordinal=len(events),
                    source=TraceSource.UNKNOWN,
                    ts=len(events),
                    pid=10 + rank,
                    tid=20 + rank,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    extras={"stream_id": f"s{rank}", "host_dispatch_queue_id": f"host0:rank:{rank}"},
                )
            )
        for index in range(2):
            events.append(
                CollatedEvent(
                    id=f"r{rank}:gemm{index}",
                    rank=rank,
                    ordinal=len(events),
                    source=TraceSource.UNKNOWN,
                    ts=len(events),
                    pid=10 + rank,
                    tid=20 + rank,
                    module="libcublas.so",
                    api="cublasGemmEx",
                    op_type="cublas_gemm",
                    extras={"stream_id": f"s{rank}", "host_dispatch_queue_id": f"host0:rank:{rank}"},
                )
            )
        for index in range(2):
            events.append(
                CollatedEvent(
                    id=f"r{rank}:strided{index}",
                    rank=rank,
                    ordinal=len(events),
                    source=TraceSource.UNKNOWN,
                    ts=len(events),
                    pid=10 + rank,
                    tid=20 + rank,
                    module="libcublas.so",
                    api="cublasGemmStridedBatchedEx",
                    op_type="cublas_gemm",
                    extras={"stream_id": f"s{rank}", "host_dispatch_queue_id": f"host0:rank:{rank}"},
                )
            )
        event_id = f"r{rank}:allreduce"
        group_event_ids.append(event_id)
        events.append(
            CollatedEvent(
                id=event_id,
                rank=rank,
                ordinal=len(events),
                source=TraceSource.UNKNOWN,
                ts=len(events),
                pid=10 + rank,
                tid=20 + rank,
                module="libnccl.so",
                api="ncclAllReduce",
                op_type="nccl_collective",
                extras={
                    "stream_id": f"s{rank}",
                    "comm_id": "comm-0",
                    "count": 16,
                    "datatype": 9,
                    "host_dispatch_queue_id": f"host0:rank:{rank}",
                },
                prev_event_id=events[-1].id,
                collective_group_id=group_id,
            )
        )
        rank_events[rank] = tuple(events)

    group = CollectiveGroup(
        id=group_id,
        api="ncclAllReduce",
        op_type="nccl_collective",
        ranks=(0, 1),
        event_ids=tuple(group_event_ids),
        communicator_id="comm-0",
        sequence_number=42,
        communicator_size=2,
        participant_count=2,
        match_basis="communicator_call_sequence",
    )
    return CollatedTrace(
        trace_dir=Path("/tmp/collated-allreduce-selected-no-motif"),
        source=TraceSource.UNKNOWN,
        rank_events=rank_events,
        global_events=rank_events[0] + rank_events[1],
        collective_groups={group_id: group},
        fidelity_windows={
            0: FidelityWindow(
                start_ts=0,
                end_ts=100,
                source="manual",
                is_paper_valid_step_window=True,
            ),
            1: FidelityWindow(
                start_ts=0,
                end_ts=100,
                source="manual",
                is_paper_valid_step_window=True,
            ),
        },
    )


def test_replay_collective_synchronizes_ranks():
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:e0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    duration_us=5.0,
                    duration_source="manual",
                ),
                AnnotatedEvent(
                    id="r0:e1",
                    rank=0,
                    ordinal=1,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=0,
                    tid=0,
                    module="libnccl.so",
                    api="ncclAllReduce",
                    op_type="nccl_collective",
                    duration_us=7.0,
                    duration_source="manual",
                    prev_event_id="r0:e0",
                    collective_group_id="ncclAllReduce#0",
                ),
            ),
            1: (
                AnnotatedEvent(
                    id="r1:e0",
                    rank=1,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=1,
                    tid=1,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    duration_us=3.0,
                    duration_source="manual",
                ),
                AnnotatedEvent(
                    id="r1:e1",
                    rank=1,
                    ordinal=1,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=1,
                    tid=1,
                    module="libnccl.so",
                    api="ncclAllReduce",
                    op_type="nccl_collective",
                    duration_us=7.0,
                    duration_source="manual",
                    prev_event_id="r1:e0",
                    collective_group_id="ncclAllReduce#0",
                ),
            ),
        },
        global_events=(),
        collective_groups={
            "ncclAllReduce#0": CollectiveGroup(
                id="ncclAllReduce#0",
                api="ncclAllReduce",
                op_type="nccl_collective",
                ranks=(0, 1),
                event_ids=("r0:e1", "r1:e1"),
            )
        },
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0] + trace.rank_events[1],
        collective_groups=trace.collective_groups,
    )

    result = replay_annotated_trace(trace)

    assert result.success
    assert result.total_time_us == 12.0
    collective_events = [event for event in result.simulated_events if event.collective_group_id]
    assert len(collective_events) == 2
    assert {event.start_us for event in collective_events} == {5.0}
    assert {event.end_us for event in collective_events} == {12.0}


def test_replay_edge_export_reports_intervals_and_wait_edges():
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:e0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaEventRecord",
                    op_type="event_record",
                    duration_us=10.0,
                    duration_source="manual",
                    extras={"event_id": "evt0", "stream_id": "stream0"},
                ),
                AnnotatedEvent(
                    id="r0:e1",
                    rank=0,
                    ordinal=1,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaStreamWaitEvent",
                    op_type="event_wait",
                    duration_us=1.0,
                    duration_source="manual",
                    extras={"event_id": "evt0", "stream_id": "stream1"},
                ),
            )
        },
        global_events=(),
        collective_groups={},
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0],
        collective_groups={},
    )
    diagnostics = []
    result = replay_annotated_trace(trace, diagnostic_events=diagnostics)

    export = export_replay_edge_diagnostics(trace, result, diagnostic_events=diagnostics)

    assert export["scope"] == "diagnostic_only_no_replay_behavior_change"
    assert export["field_coverage"]["simulated_intervals"] == 2
    assert export["field_coverage"]["cuda_event_edges"] == 1
    assert all("wait_start_us" in edge for edge in export["predecessor_edges"])
    cuda_edges = [
        edge for edge in export["predecessor_edges"]
        if edge["edge_kind"] == "cuda_event_wait"
    ]
    assert cuda_edges[0]["predecessor_event_id"] == "r0:e0"
    assert cuda_edges[0]["successor_event_id"] == "r0:e1"
    assert cuda_edges[0]["predecessor_api"] == "cudaEventRecord"
    assert cuda_edges[0]["successor_api"] == "cudaStreamWaitEvent"
    assert cuda_edges[0]["cuda_event_id"] == "evt0"
    assert cuda_edges[0]["cuda_event_version"] == 1
    assert cuda_edges[0]["wait_start_us"] == 0.0
    assert cuda_edges[0]["release_us"] == 10.0
    assert cuda_edges[0]["waited_us"] == 10.0
    assert cuda_edges[0]["affected_interval_duration_us"] == 10.0
    assert cuda_edges[0]["stream_queue_position"] == 0
    assert export["simulated_events"][0]["link_fields"]["event_id"] == "evt0"


def test_replay_edge_export_appendix_ab_p2p_diagnostics_default_off(monkeypatch):
    monkeypatch.delenv("MAYA_ENABLE_P2P_PER_BLOCK_COMPONENT_DIAGNOSTICS", raising=False)
    monkeypatch.delenv("FLEXSIM_MAYA_ENABLE_P2P_PER_BLOCK_COMPONENT_DIAGNOSTICS", raising=False)
    monkeypatch.delenv(
        "MAYA_ENABLE_APPENDIX_AB_SELECTED_NCCL_ALLREDUCE_PER_BLOCK_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_APPENDIX_AB_SELECTED_NCCL_ALLREDUCE_PER_BLOCK_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_PHASE1_METADATA_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_PHASE1_METADATA_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
        raising=False,
    )
    trace = _manual_p2p_kernel1_trace()
    diagnostics: list[dict[str, object]] = []
    result = replay_annotated_trace(trace, diagnostic_events=diagnostics)

    export = export_replay_edge_diagnostics(trace, result, diagnostic_events=diagnostics)

    assert "appendix_ab_selected_p2p_per_block_diagnostics" not in export
    assert "appendix_ab_selected_nccl_allreduce_per_block_diagnostics" not in export
    assert "shared_all_rank_phase_anchor_causal_edge_metadata" not in export
    assert "generic_replay_placement_envelope_phase1_metadata" not in export
    assert "component_strict_counterpart_metadata_evidence" not in export
    assert "appendix_ab_selected_p2p_per_block_rows" not in export["field_coverage"]
    assert "appendix_ab_selected_nccl_allreduce_per_block_rows" not in export["field_coverage"]
    assert "shared_phase_anchor_causal_edge_rows" not in export["field_coverage"]
    assert "generic_replay_placement_envelope_phase1_rows" not in export["field_coverage"]
    assert "component_strict_counterpart_metadata_rows" not in export["field_coverage"]
    assert result.total_time_us == 12.0


@pytest.mark.parametrize(
    "env_key",
    [
        "MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_COMPONENT_STRICT_COUNTERPART_METADATA_DIAGNOSTICS",
    ],
)
def test_replay_edge_export_component_strict_counterpart_metadata_opt_in(
    monkeypatch,
    env_key,
):
    monkeypatch.setenv(env_key, "yes")
    monkeypatch.delenv(
        "MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_PHASE1_METADATA_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_PHASE1_METADATA_DIAGNOSTICS",
        raising=False,
    )
    trace = _manual_p2p_kernel1_trace()
    diagnostics: list[dict[str, object]] = []
    result = replay_annotated_trace(trace, diagnostic_events=diagnostics)

    export = export_replay_edge_diagnostics(trace, result, diagnostic_events=diagnostics)

    assert "generic_replay_placement_envelope_phase1_metadata" not in export
    component = export["component_strict_counterpart_metadata_evidence"]
    assert component["schema_version"] == "component_strict_counterpart_metadata_evidence_v1"
    assert component["diagnostic_only"] is True
    assert component["safety"]["additive_metadata_only"] is True
    assert component["safety"]["behavior_changing_replay_or_provider_edit"] is False
    assert component["safety"]["actual_runtime_direct_substitution"] is False
    assert component["safety"]["repair_ready"] is False
    assert component["row_count"] == (
        len(export["simulated_events"])
        + len(export["predecessor_edges"])
        + len(export["metrics"]["rank_metrics"])
        + 1
    )
    assert export["field_coverage"]["component_strict_counterpart_metadata_rows"] == (
        component["row_count"]
    )

    rows = {
        row["materialized_event_id"]: row
        for row in component["rows"]
        if row["predicted_replay_component"] == "interval"
    }
    send = rows["r0:p2p-send"]
    first_id = send["stable_predicted_component_row_id"]
    assert first_id.startswith(
        "generic_replay_placement_envelope:stream_operation_interval:"
    )
    assert send["component_strict_counterpart_opt_in_flag"] is True
    assert send["source_side"] == "predicted_component_metadata"
    assert send["stable_predicted_interval_row_id"] == send["stable_predicted_count_once_group_id"].removeprefix("replay_interval:")
    assert send["component_kind"] == "stream_operation_interval"
    assert send["api_or_kernel_family"] == "ncclSend"
    assert send["rank"] == 0
    assert send["world_size"] == 2
    assert send["predicted_interval_duration_us"] == 7.0
    assert send["predicted_interval_resource_kind"] == "stream"
    assert send["predicted_stream_resource_id"] == "rank:0:stream:s0"
    assert send["stream_namespace_alignment_status"] == (
        "predicted_replay_stream_namespace_only_actual_alignment_unavailable"
    )
    assert send["strict_actual_timing_status"] == "unavailable"
    assert send["strict_actual_timing_available"] is False
    assert send["actual_start_us"] is None
    assert send["actual_end_us"] is None
    assert send["actual_duration_us"] is None
    assert send["actual_endpoint_timestamps_used_as_strict_timing"] is False
    assert send["actual_host_duration_used_as_strict_timing"] is False
    assert send["actual_runtime_direct_substitution"] is False
    assert send["actual_observed_runtime_used_as_prediction"] is False
    assert send["count_once_status"] == (
        "metadata_only_count_once_group_not_strict_non_overlap_proof"
    )
    assert send["nonoverlap_status"] == "unavailable"
    assert send["wait_map_safety_status"] == "unavailable"
    assert send["producer_visibility_status"] == "unavailable"
    assert send["repair_ready"] is False
    assert send["safe_to_use_as_repair_evidence"] is False
    assert send["safe_to_use_as_subtraction_delta"] is False

    repeat = export_replay_edge_diagnostics(trace, result, diagnostic_events=diagnostics)
    repeat_rows = {
        row["materialized_event_id"]: row
        for row in repeat["component_strict_counterpart_metadata_evidence"]["rows"]
        if row["predicted_replay_component"] == "interval"
    }
    assert repeat_rows["r0:p2p-send"]["stable_predicted_component_row_id"] == first_id


@pytest.mark.parametrize(
    "env_key",
    [
        "MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_PHASE1_METADATA_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_PHASE1_METADATA_DIAGNOSTICS",
    ],
)
def test_replay_edge_export_generic_placement_envelope_metadata_opt_in(monkeypatch, env_key):
    monkeypatch.setenv(env_key, "1")
    trace = _manual_p2p_kernel1_trace()
    diagnostics: list[dict[str, object]] = []
    result = replay_annotated_trace(trace, diagnostic_events=diagnostics)

    export = export_replay_edge_diagnostics(trace, result, diagnostic_events=diagnostics)

    generic = export["generic_replay_placement_envelope_phase1_metadata"]
    assert generic["schema_version"] == "generic_replay_placement_envelope_phase1_metadata_v1"
    assert generic["diagnostic_only"] is True
    assert generic["safety"]["behavior_changing_replay_or_provider_edit"] is False
    assert generic["safety"]["actual_runtime_direct_substitution"] is False
    assert generic["safety"]["actual_endpoint_timestamp_substitution"] is False
    assert generic["row_count"] == (
        len(export["simulated_events"])
        + len(export["predecessor_edges"])
        + len(export["metrics"]["rank_metrics"])
        + 1
    )
    assert export["field_coverage"]["generic_replay_placement_envelope_phase1_rows"] == (
        generic["row_count"]
    )

    component_kinds = {row["component_kind"] for row in generic["rows"]}
    assert "stream_operation_interval" in component_kinds
    assert "collective_wait" in component_kinds
    assert "rank_completion_envelope" in component_kinds
    assert "global_completion_envelope" in component_kinds
    rows = {
        row["materialized_event_id"]: row
        for row in generic["rows"]
        if row["component_row_type"] == "interval"
    }
    send = rows["r0:p2p-send"]
    assert send["stable_component_row_id"].startswith(
        "generic_replay_placement_envelope:stream_operation_interval:"
    )
    assert send["component_kind"] == "stream_operation_interval"
    assert send["count_once_interval_id"] == send["simulated_event_id"]
    assert send["count_once_group_id"] == f"replay_interval:{send['simulated_event_id']}"
    assert send["count_once_status"] == (
        "metadata_only_count_once_group_not_strict_non_overlap_proof"
    )
    assert send["count_once_non_overlap_status"] == "unavailable"
    assert send["resource_kind"] == "stream"
    assert send["resource_id"] == "rank:0:stream:s0"
    assert send["predicted_stream_resource_id"] == "rank:0:stream:s0"
    assert send["stream_namespace_alignment_status"] == (
        "predicted_replay_stream_namespace_only_actual_alignment_unavailable"
    )
    assert send["predecessor_successor_status"] == "available_in_replay_edge_export"
    assert send["predecessor_tags"]
    assert "critical_placement_status" in send
    assert isinstance(send["critical_placement_tags"], list)
    assert send["actual_timing_status"] == "unavailable"
    assert "endpoint_timestamps_not_used" in send["actual_timing_unavailable_reason"]
    assert send["actual_start_us"] is None
    assert send["actual_end_us"] is None
    assert send["actual_duration_us"] is None
    assert send["actual_wait_start_us"] is None
    assert send["actual_release_us"] is None
    assert send["actual_runtime_us"] is None
    assert send["actual_endpoint_timestamps_used"] is False
    assert send["actual_runtime_direct_substitution"] is False
    assert send["safe_to_use_as_repair_evidence"] is False
    assert send["safe_to_use_as_subtraction_delta"] is False
    edge_rows = [row for row in generic["rows"] if row["component_row_type"] == "edge"]
    collective = [row for row in edge_rows if row["component_kind"] == "collective_wait"][0]
    assert collective["stable_replay_edge_id"]
    assert collective["actual_timing_status"] == "unavailable"
    assert collective["predecessor_successor_status"] == "available_in_replay_edge_export"
    global_rows = [
        row for row in generic["rows"]
        if row["component_kind"] == "global_completion_envelope"
    ]
    assert global_rows[0]["resource_kind"] == "global"
    assert global_rows[0]["actual_endpoint_timestamps_used"] is False
    assert result.total_time_us == 12.0


@pytest.mark.parametrize(
    "env_key",
    [
        "MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
    ],
)
def test_replay_edge_export_shared_phase_anchor_metadata_opt_in(monkeypatch, env_key):
    monkeypatch.setenv(env_key, "true")
    trace = _manual_p2p_kernel1_trace()
    diagnostics: list[dict[str, object]] = []
    result = replay_annotated_trace(trace, diagnostic_events=diagnostics)

    export = export_replay_edge_diagnostics(trace, result, diagnostic_events=diagnostics)

    shared = export["shared_all_rank_phase_anchor_causal_edge_metadata"]
    assert shared["schema_version"] == (
        "shared_all_rank_phase_anchor_counterpart_replay_causal_edge_metadata_v1"
    )
    assert shared["diagnostic_only"] is True
    assert shared["safety"]["behavior_changing_replay_or_provider_edit"] is False
    assert shared["safety"]["actual_endpoint_end_ts_used_as_release"] is False
    assert shared["safety"]["safe_to_use_as_repair_evidence"] is False
    assert shared["phase_anchor_block_row_count"] == 2
    assert shared["causal_edge_row_count"] == len(export["predecessor_edges"])
    assert export["field_coverage"]["shared_phase_anchor_block_rows"] == 2
    assert export["field_coverage"]["shared_phase_anchor_causal_edge_rows"] == len(
        export["predecessor_edges"]
    )

    block_rows = {row["api"]: row for row in shared["phase_anchor_block_rows"]}
    assert set(block_rows) == {"ncclSend", "ncclRecv"}
    send_block = block_rows["ncclSend"]
    assert send_block["stable_block_id"].startswith("shared_phase_anchor_block:rank:0:api:ncclSend:")
    assert send_block["phase_anchor_id"] == "shared_all_rank_phase_anchor:ncclP2P:ncclP2P|comm:comm-0|members:0-1|pair_seq:0"
    assert send_block["participant_rank_ids"] == [0, 1]
    assert send_block["provider_runtime_us"] == 7.0
    assert send_block["collective_wait_us"] == 5.0
    assert send_block["repair_authorization_status"] == "diagnostic_only_no_repair_authorized"

    allowed_edge_kinds = {
        "host_order",
        "stream_order",
        "stream_queue_wait",
        "cuda_event_wait",
        "collective_wait",
        "host_dispatch",
        "host_sync_wait",
        "device_sync_wait",
        "rank_completion_aggregation_context",
    }
    assert {row["edge_kind"] for row in shared["causal_edge_rows"]} <= allowed_edge_kinds
    collective_edges = [
        row for row in shared["causal_edge_rows"] if row["edge_kind"] == "collective_wait"
    ]
    assert collective_edges
    assert collective_edges[0]["shared_anchor_causal_edge_schema_version"] == shared[
        "schema_version"
    ]
    assert collective_edges[0]["diagnostic_opt_in_flag"] is True
    assert collective_edges[0]["source_side"] == "predicted_replay"
    assert collective_edges[0]["stable_block_id"] in {
        row["stable_block_id"] for row in shared["phase_anchor_block_rows"]
    }
    assert collective_edges[0]["release_us"] == 5.0
    assert collective_edges[0]["waited_us"] == 5.0
    assert result.total_time_us == 12.0


@pytest.mark.parametrize(
    "env_key",
    [
        "MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
    ],
)
def test_replay_edge_export_shared_phase_anchor_common_basis_fields(monkeypatch, env_key):
    monkeypatch.setenv(env_key, "yes")
    monkeypatch.delenv(
        "MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_SHARED_ALL_RANK_PHASE_ANCHOR_CAUSAL_EDGE_METADATA_DIAGNOSTICS",
        raising=False,
    )
    trace = _manual_p2p_kernel1_trace()
    diagnostics: list[dict[str, object]] = []
    result = replay_annotated_trace(trace, diagnostic_events=diagnostics)

    export = export_replay_edge_diagnostics(trace, result, diagnostic_events=diagnostics)

    shared = export["shared_all_rank_phase_anchor_causal_edge_metadata"]
    assert shared["common_basis_key_fields_enabled"] is True
    assert shared["common_basis_schema_version"] == (
        "shared_phase_anchor_common_basis_key_fields_v1"
    )
    assert export["field_coverage"]["shared_phase_anchor_common_basis_block_rows"] == 2
    block_rows = {row["api"]: row for row in shared["phase_anchor_block_rows"]}
    send = block_rows["ncclSend"]
    recv = block_rows["ncclRecv"]

    assert send["common_basis_schema_version"] == (
        "shared_phase_anchor_common_basis_key_fields_v1"
    )
    assert send["common_call_order_basis"] == "group_id_pair_seq"
    assert send["common_call_order_index"] == 0
    assert send["common_pair_seq"] == 0
    assert send["common_group_id_call_index"] is None
    assert send["common_rank_window_index"] == 0
    assert send["common_payload_signature"] == (
        "group_api=ncclP2P;kind=p2p;api=ncclSend;"
        "members=members:0-1;count=8;datatype=9;op=null"
    )
    assert send["common_payload_signature_inputs"] == {
        "api": "ncclSend",
        "group_api": "ncclP2P",
        "collective_kind": "p2p",
        "count": 8,
        "datatype": 9,
        "op": None,
        "membership_signature": "members:0-1",
        "pair_members": [0, 1],
    }
    assert send["payload_basis"] == "raw_operation_semantics_not_stream_only_key"
    assert send["common_membership_signature"] == "members:0-1"
    assert send["common_pair_members"] == [0, 1]
    assert send["shape_signature"] == "group_api=ncclP2P;collective=p2p;count=8;datatype=9"
    assert recv["common_api_direction"] == "recv"
    assert recv["common_pair_seq"] == 0
    assert result.total_time_us == 12.0


def test_replay_edge_export_shared_phase_anchor_allreduce_common_call_basis(monkeypatch):
    monkeypatch.setenv(
        "MAYA_ENABLE_SHARED_PHASE_ANCHOR_COMMON_BASIS_KEY_FIELDS_DIAGNOSTICS",
        "1",
    )
    trace = _manual_allreduce_trace()
    diagnostics: list[dict[str, object]] = []
    result = replay_annotated_trace(trace, diagnostic_events=diagnostics)

    export = export_replay_edge_diagnostics(trace, result, diagnostic_events=diagnostics)

    rows = export["shared_all_rank_phase_anchor_causal_edge_metadata"][
        "phase_anchor_block_rows"
    ]
    assert len(rows) == 2
    for row in rows:
        assert row["api"] == "ncclAllReduce"
        assert row["common_call_order_basis"] == "group_id_call_ordinal"
        assert row["common_call_order_index"] == 42
        assert row["common_group_id_call_index"] == 42
        assert row["common_pair_seq"] is None
        assert row["common_collective_kind"] == "allreduce"
        assert row["common_payload_signature"] == (
            "group_api=ncclAllReduce;kind=allreduce;api=ncclAllReduce;"
            "members=members:0-1;count=16;datatype=9;op=null"
        )
        assert row["payload_basis"] == "raw_operation_semantics_not_stream_only_key"
    assert result.total_time_us == 6.0


@pytest.mark.parametrize(
    "env_key",
    [
        "MAYA_ENABLE_APPENDIX_AB_SELECTED_NCCL_ALLREDUCE_PER_BLOCK_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_APPENDIX_AB_SELECTED_NCCL_ALLREDUCE_PER_BLOCK_DIAGNOSTICS",
    ],
)
def test_replay_edge_export_appendix_b_allreduce_diagnostics_opt_in_row_shape(
    monkeypatch,
    env_key,
):
    monkeypatch.setenv(env_key, "1")
    trace = _manual_allreduce_trace()
    diagnostics: list[dict[str, object]] = []
    result = replay_annotated_trace(trace, diagnostic_events=diagnostics)

    export = export_replay_edge_diagnostics(trace, result, diagnostic_events=diagnostics)

    appendix = export["appendix_ab_selected_nccl_allreduce_per_block_diagnostics"]
    assert appendix["schema_version"] == (
        "appendix_ab_selected_nccl_allreduce_per_block_component_diagnostics_v1"
    )
    assert appendix["diagnostic_only"] is True
    assert appendix["row_count"] == 2
    assert appendix["selected_family_label"] == "nccl_allreduce_kernel4_8_gemm2_3_strided2_3"
    assert appendix["safety"]["actual_runtime_direct_substitution"] is False
    assert appendix["safety"]["actual_api_end_ts_used_as_wait_map_release"] is False
    assert appendix["safety"]["actual_api_end_ts_used_as_block_timing"] is False
    assert appendix["safety"]["safe_to_use_as_repair_evidence"] is False
    assert export["field_coverage"]["appendix_ab_selected_nccl_allreduce_per_block_rows"] == 2

    rows = {row["rank"]: row for row in appendix["rows"]}
    rank0 = rows[0]
    assert rank0["stable_block_id"].startswith(
        "appendix_ab_nccl_allreduce_k4_8_g2_3_s2_3:0:ncclAllReduce:"
    )
    assert rank0["family_label"] == "nccl_allreduce_kernel4_8_gemm2_3_strided2_3"
    assert rank0["motif_key"] == appendix["selected_motif_key"]
    assert rank0["api"] == "ncclAllReduce"
    assert rank0["group_api"] == "ncclAllReduce"
    assert rank0["members"] == [0, 1]
    assert rank0["group_call_ordinal"] == 42
    assert rank0["participant_count"] == 2
    assert rank0["stream_resource_id"] == "rank:0:stream:s0"
    assert rank0["paper_valid_window_id"] == "rank0:step_window"
    assert rank0["predicted_block_duration_us"] == 6.0
    assert rank0["component_duration_split_us"]["provider_runtime"] == 6.0
    assert "hostDelay" in rank0["component_duration_split_us"]
    assert rank0["collective_group_duration_us"] == 6.0
    assert rank0["collective_group_duration_basis"] == "group_provider:test_provider"
    assert rank0["duration_source"] == "manual"
    assert rank0["provider_name"] == "test_provider"
    assert rank0["provider_tier"] == "trace_signature_stats"
    assert rank0["provider_duration_source_expected"] == "trace_signature_stats"
    assert rank0["material_signature"] == (
        "group_api=ncclAllReduce;collective=allreduce;count=16;datatype=9"
    )
    assert rank0["predecessor_edge_ids"]
    assert rank0["wait_start_us"] is not None
    assert rank0["release_us"] is not None
    assert rank0["release_reason"] == "collective_all_participants_ready"
    assert rank0["actual_block_start_us"] is None
    assert rank0["actual_block_end_us"] is None
    assert rank0["actual_wait_start_us"] is None
    assert rank0["actual_release_us"] is None
    assert rank0["actual_api_end_ts_used_as_wait_map_release"] is False
    assert rank0["actual_api_end_ts_used_as_block_timing"] is False
    assert "actual API end_ts is intentionally not used" in (
        rank0["actual_release_unavailable_reason"]
    )
    assert result.total_time_us == 6.0


def test_replay_edge_export_appendix_b_allreduce_dispatch_only_keeps_one_stream_row(
    monkeypatch,
):
    monkeypatch.setenv(
        "MAYA_ENABLE_APPENDIX_AB_SELECTED_NCCL_ALLREDUCE_PER_BLOCK_DIAGNOSTICS",
        "1",
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_APPENDIX_AB_SELECTED_NCCL_ALLREDUCE_PER_BLOCK_DIAGNOSTICS",
        raising=False,
    )
    trace = _manual_allreduce_trace(host_dispatch_duration_us=2.5)
    diagnostics: list[dict[str, object]] = []
    result = replay_annotated_trace(trace, diagnostic_events=diagnostics)

    export = export_replay_edge_diagnostics(trace, result, diagnostic_events=diagnostics)

    appendix = export["appendix_ab_selected_nccl_allreduce_per_block_diagnostics"]
    assert appendix["row_count"] == 2
    assert export["field_coverage"]["appendix_ab_selected_nccl_allreduce_per_block_rows"] == 2
    assert {row["event_ids"][0] for row in appendix["rows"]} == {
        "r0:allreduce",
        "r1:allreduce",
    }
    for row in appendix["rows"]:
        assert row["api"] == "ncclAllReduce"
        assert row["schedule_resource_kind"] == "stream"
        assert not row["stable_block_id"].endswith(":host_dispatch")
        assert row["predicted_block_duration_us"] == 6.0
        assert row["component_duration_split_us"]["provider_runtime"] == 6.0
        assert row["component_duration_split_us"]["host_dispatch"] == 2.5
        assert "host_dispatch" not in row["component_duration_split_unavailable_reasons"]
    assert result.total_time_us == 8.5


def test_replay_edge_export_appendix_b_allreduce_diagnostics_trace_window_compatibility(
    monkeypatch,
):
    monkeypatch.delenv(
        "MAYA_ENABLE_APPENDIX_AB_SELECTED_NCCL_ALLREDUCE_PER_BLOCK_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_APPENDIX_AB_SELECTED_NCCL_ALLREDUCE_PER_BLOCK_DIAGNOSTICS",
        raising=False,
    )
    trace = _manual_allreduce_trace(
        rank0_ts=90,
        rank0_end_ts=120,
        rank1_ts=100,
        rank1_end_ts=130,
        rank0_window=(100, 200),
        rank1_window=(100, 200),
    )
    diagnostics: list[dict[str, object]] = []
    result = replay_annotated_trace(trace, diagnostic_events=diagnostics)

    default_off_export = export_replay_edge_diagnostics(
        trace,
        result,
        diagnostic_events=diagnostics,
    )

    assert "appendix_ab_selected_nccl_allreduce_per_block_diagnostics" not in default_off_export

    monkeypatch.setenv(
        "MAYA_ENABLE_APPENDIX_AB_SELECTED_NCCL_ALLREDUCE_PER_BLOCK_DIAGNOSTICS",
        "1",
    )
    opt_in_export = export_replay_edge_diagnostics(
        trace,
        result,
        diagnostic_events=diagnostics,
    )

    appendix = opt_in_export["appendix_ab_selected_nccl_allreduce_per_block_diagnostics"]
    assert appendix["trace_window_compatibility"] == {
        "paper_valid_step_windows_authoritative": True,
        "step_window_assignment_bypassed": False,
        "paper_error_basis_changed": False,
        "paper_valid_window_count": 2,
        "paper_valid_window_sources": ["manual"],
        "actual_terms_require_in_window_aligned_counterpart": True,
        "outside_window_actual_timings_included": False,
        "actual_api_end_ts_used_as_wait_map_release": False,
        "actual_side_unavailable_policy": (
            "Actual Appendix B terms remain null unless a strict actual counterpart is aligned "
            "to the existing paper-valid step window; outside-window or unaligned actual timings "
            "are not included."
        ),
    }
    rows = {row["rank"]: row for row in appendix["rows"]}
    rank0 = rows[0]
    membership = rank0["paper_valid_window_membership"]
    assert membership["in_paper_valid_window"] is True
    assert membership["start_ts"] == 100
    assert membership["end_ts"] == 200
    assert membership["membership_basis"] == "existing_trace_fidelity_window_ts_end_ts_overlap"
    assert rank0["paper_valid_window_id"] == "rank0:step_window"
    assert rank0["actual_paper_valid_window_id"] is None
    assert rank0["actual_counterpart_window_status"] == "unavailable"
    assert rank0["actual_block_start_us"] is None
    assert rank0["actual_block_end_us"] is None
    assert rank0["actual_api_end_ts_used_as_wait_map_release"] is False
    assert rank0["actual_api_end_ts_used_as_block_timing"] is False


def test_appendix_ab_p2p_selection_metadata_forwarding_default_off(monkeypatch):
    monkeypatch.delenv("MAYA_ENABLE_P2P_PER_BLOCK_COMPONENT_DIAGNOSTICS", raising=False)
    monkeypatch.delenv("FLEXSIM_MAYA_ENABLE_P2P_PER_BLOCK_COMPONENT_DIAGNOSTICS", raising=False)
    monkeypatch.delenv(
        "MAYA_ENABLE_APPENDIX_AB_SELECTED_NCCL_ALLREDUCE_PER_BLOCK_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_APPENDIX_AB_SELECTED_NCCL_ALLREDUCE_PER_BLOCK_DIAGNOSTICS",
        raising=False,
    )

    trace = annotate_collated_trace(
        _collated_p2p_kernel1_without_motif_trace(),
        Estimator(),
        allow_weak_runtime_fallback=True,
        allow_collective_group_fallback=True,
    )

    p2p_events = [event for event in trace.global_events if event.api in {"ncclSend", "ncclRecv"}]
    assert p2p_events
    assert all("motif_key" not in event.extras for event in p2p_events)
    assert all("appendix_ab_p2p_motif_key" not in event.extras for event in p2p_events)


def test_appendix_b_allreduce_selection_metadata_forwarding_default_off(monkeypatch):
    monkeypatch.delenv(
        "MAYA_ENABLE_APPENDIX_AB_SELECTED_NCCL_ALLREDUCE_PER_BLOCK_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_APPENDIX_AB_SELECTED_NCCL_ALLREDUCE_PER_BLOCK_DIAGNOSTICS",
        raising=False,
    )

    trace = annotate_collated_trace(
        _collated_allreduce_selected_without_motif_trace(),
        Estimator(),
        allow_weak_runtime_fallback=True,
        allow_collective_group_fallback=True,
    )

    allreduce_events = [event for event in trace.global_events if event.api == "ncclAllReduce"]
    assert len(allreduce_events) == 2
    assert all("motif_key" not in event.extras for event in allreduce_events)
    assert all("appendix_ab_allreduce_motif_key" not in event.extras for event in allreduce_events)
    assert all(
        "appendix_ab_allreduce_selection_basis" not in event.extras
        for event in allreduce_events
    )


def test_appendix_b_allreduce_selection_metadata_forwarding_opt_in(monkeypatch):
    monkeypatch.setenv(
        "MAYA_ENABLE_APPENDIX_AB_SELECTED_NCCL_ALLREDUCE_PER_BLOCK_DIAGNOSTICS",
        "1",
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_APPENDIX_AB_SELECTED_NCCL_ALLREDUCE_PER_BLOCK_DIAGNOSTICS",
        raising=False,
    )

    trace = annotate_collated_trace(
        _collated_allreduce_selected_without_motif_trace(),
        Estimator(),
        allow_weak_runtime_fallback=True,
        allow_collective_group_fallback=True,
    )

    allreduce_events = [event for event in trace.global_events if event.api == "ncclAllReduce"]
    assert len(allreduce_events) == 2
    for event in allreduce_events:
        assert event.extras["appendix_ab_allreduce_motif_key"] == (
            "boundary=ncclAllReduce|kernel=4-8|gemm=2-3|strided=2-3|send=0|recv=0|allreduce=1"
        )
        assert event.extras["appendix_ab_allreduce_family_label"] == (
            "nccl_allreduce_kernel4_8_gemm2_3_strided2_3"
        )
        assert event.extras["appendix_ab_allreduce_selection_basis"] == (
            "opt_in_rank_order_semantic_block_api_counts_v1"
        )

    diagnostics: list[dict[str, object]] = []
    result = replay_annotated_trace(trace, diagnostic_events=diagnostics)
    export = export_replay_edge_diagnostics(
        trace,
        result,
        diagnostic_events=diagnostics,
        target_event_ids={event.id for event in allreduce_events},
    )

    appendix = export["appendix_ab_selected_nccl_allreduce_per_block_diagnostics"]
    assert appendix["row_count"] == 2
    assert {row["event_ids"][0] for row in appendix["rows"]} == {
        event.id for event in allreduce_events
    }


def test_replay_edge_export_appendix_ab_p2p_diagnostics_forwards_kernel1_metadata(monkeypatch):
    monkeypatch.setenv("MAYA_ENABLE_P2P_PER_BLOCK_COMPONENT_DIAGNOSTICS", "1")
    monkeypatch.delenv("FLEXSIM_MAYA_ENABLE_P2P_PER_BLOCK_COMPONENT_DIAGNOSTICS", raising=False)
    trace = annotate_collated_trace(
        _collated_p2p_kernel1_without_motif_trace(),
        Estimator(),
        allow_weak_runtime_fallback=True,
        allow_collective_group_fallback=True,
    )

    p2p_events = {event.api: event for event in trace.global_events if event.api in {"ncclSend", "ncclRecv"}}
    assert "motif_key" not in p2p_events["ncclSend"].extras
    assert p2p_events["ncclSend"].extras["appendix_ab_p2p_motif_key"] == (
        "boundary=ncclSend|kernel=1|gemm=0|strided=0|send=1|recv=0|allreduce=0"
    )
    assert p2p_events["ncclRecv"].extras["appendix_ab_p2p_motif_key"] == (
        "boundary=ncclRecv|kernel=1|gemm=0|strided=0|send=0|recv=1|allreduce=0"
    )

    diagnostics: list[dict[str, object]] = []
    result = replay_annotated_trace(trace, diagnostic_events=diagnostics)
    selected_event_ids = {event.id for event in p2p_events.values()}
    export = export_replay_edge_diagnostics(
        trace,
        result,
        diagnostic_events=diagnostics,
        target_event_ids=selected_event_ids,
    )

    appendix = export["appendix_ab_selected_p2p_per_block_diagnostics"]
    rows = {row["api"]: row for row in appendix["rows"]}
    assert appendix["row_count"] == 2
    assert set(rows) == {"ncclSend", "ncclRecv"}
    assert rows["ncclSend"]["motif_key"] == p2p_events["ncclSend"].extras["appendix_ab_p2p_motif_key"]
    assert rows["ncclRecv"]["motif_key"] == p2p_events["ncclRecv"].extras["appendix_ab_p2p_motif_key"]
    assert rows["ncclSend"]["actual_api_end_ts_used_as_wait_map_release"] is False
    assert rows["ncclRecv"]["actual_block_start_us"] is None


def test_replay_edge_export_appendix_ab_p2p_diagnostics_uses_end_ts_window_overlap(monkeypatch):
    monkeypatch.delenv("MAYA_ENABLE_P2P_PER_BLOCK_COMPONENT_DIAGNOSTICS", raising=False)
    monkeypatch.delenv("FLEXSIM_MAYA_ENABLE_P2P_PER_BLOCK_COMPONENT_DIAGNOSTICS", raising=False)
    trace = _manual_p2p_kernel1_trace(
        send_ts=90,
        send_end_ts=120,
        recv_ts=100,
        rank0_window=(100, 200),
        rank1_window=(100, 200),
    )
    diagnostics: list[dict[str, object]] = []
    result = replay_annotated_trace(trace, diagnostic_events=diagnostics)

    default_off_export = export_replay_edge_diagnostics(trace, result, diagnostic_events=diagnostics)

    assert "appendix_ab_selected_p2p_per_block_diagnostics" not in default_off_export

    monkeypatch.setenv("MAYA_ENABLE_P2P_PER_BLOCK_COMPONENT_DIAGNOSTICS", "1")
    opt_in_export = export_replay_edge_diagnostics(trace, result, diagnostic_events=diagnostics)

    rows = {
        (row["rank"], row["api"]): row
        for row in opt_in_export["appendix_ab_selected_p2p_per_block_diagnostics"]["rows"]
    }
    send = rows[(0, "ncclSend")]
    membership = send["paper_valid_window_membership"]
    assert membership["in_paper_valid_window"] is True
    assert membership["start_ts"] == 100
    assert membership["end_ts"] == 200
    assert membership["membership_basis"] == "existing_trace_fidelity_window_ts_end_ts_overlap"
    assert send["paper_valid_window_id"] == "rank0:step_window"
    assert send["actual_block_start_us"] is None
    assert send["actual_block_end_us"] is None
    assert send["actual_api_end_ts_used_as_wait_map_release"] is False


def test_replay_edge_export_appendix_ab_p2p_diagnostics_clamps_malformed_end_ts(monkeypatch):
    monkeypatch.setenv("MAYA_ENABLE_P2P_PER_BLOCK_COMPONENT_DIAGNOSTICS", "1")
    trace = _manual_p2p_kernel1_trace(
        send_ts=150,
        send_end_ts=90,
        recv_ts=150,
        rank0_window=(100, 200),
        rank1_window=(100, 200),
    )
    diagnostics: list[dict[str, object]] = []
    result = replay_annotated_trace(trace, diagnostic_events=diagnostics)

    export = export_replay_edge_diagnostics(trace, result, diagnostic_events=diagnostics)

    rows = {
        (row["rank"], row["api"]): row
        for row in export["appendix_ab_selected_p2p_per_block_diagnostics"]["rows"]
    }
    send = rows[(0, "ncclSend")]
    membership = send["paper_valid_window_membership"]
    assert membership["in_paper_valid_window"] is True
    assert membership["membership_basis"] == "existing_trace_fidelity_window_ts_end_ts_overlap"
    assert send["actual_block_start_us"] is None
    assert send["actual_block_end_us"] is None
    assert send["actual_api_end_ts_used_as_wait_map_release"] is False


def test_replay_edge_export_appendix_ab_p2p_diagnostics_opt_in(monkeypatch):
    monkeypatch.setenv("MAYA_ENABLE_P2P_PER_BLOCK_COMPONENT_DIAGNOSTICS", "1")
    trace = _manual_p2p_kernel1_trace()
    diagnostics: list[dict[str, object]] = []
    result = replay_annotated_trace(trace, diagnostic_events=diagnostics)

    export = export_replay_edge_diagnostics(trace, result, diagnostic_events=diagnostics)

    appendix = export["appendix_ab_selected_p2p_per_block_diagnostics"]
    assert appendix["diagnostic_only"] is True
    assert appendix["row_count"] == 2
    assert appendix["safety"]["actual_api_end_ts_used_as_wait_map_release"] is False
    assert appendix["safety"]["outside_window_actual_timings_included"] is False
    assert appendix["safety"]["step_window_assignment_bypassed"] is False
    assert appendix["trace_window_compatibility"] == {
        "paper_valid_step_windows_authoritative": True,
        "step_window_assignment_bypassed": False,
        "paper_error_basis_changed": False,
        "paper_valid_window_count": 2,
        "paper_valid_window_sources": ["manual"],
        "actual_terms_require_in_window_aligned_counterpart": True,
        "outside_window_actual_timings_included": False,
        "actual_api_end_ts_used_as_wait_map_release": False,
        "actual_side_unavailable_policy": (
            "Actual Appendix B terms remain null unless a strict actual counterpart is aligned "
            "to the existing paper-valid step window; outside-window or unaligned actual timings "
            "are not included."
        ),
    }
    rows = {row["api"]: row for row in appendix["rows"]}
    send = rows["ncclSend"]
    recv = rows["ncclRecv"]
    assert send["stable_block_id"].startswith("appendix_ab_p2p_kernel1:0:ncclSend:")
    assert recv["stable_block_id"].startswith("appendix_ab_p2p_kernel1:1:ncclRecv:")
    assert send["predicted_block_start_us"] == 5.0
    assert send["predicted_block_end_us"] == 12.0
    assert send["predicted_block_duration_us"] == 7.0
    assert send["component_duration_split_us"]["provider_runtime"] == 7.0
    assert send["component_duration_split_us"]["collective_wait"] == 5.0
    assert send["collective_group_duration_us"] == 7.0
    assert send["collective_group_duration_basis"] == "group_provider:test_provider"
    assert send["participant_count"] == 2
    assert send["members"] == [0, 1]
    assert send["pair_seq"] == 0
    assert send["paper_valid_window_id"] == "rank0:step_window"
    assert send["paper_valid_window_membership"]["in_paper_valid_window"] is True
    assert send["paper_valid_window_membership"]["window_source"] == "manual"
    assert send["actual_paper_valid_window_id"] is None
    assert send["actual_counterpart_window_status"] == "unavailable"
    assert "outside-window or unaligned actual timings are intentionally excluded" in (
        send["actual_counterpart_window_unavailable_reason"]
    )
    assert "outside-window actual timings are not included" in send["actual_block_unavailable_reason"]
    assert send["actual_release_us"] is None
    assert send["actual_api_end_ts_used_as_wait_map_release"] is False
    assert "outside-window actual timings are not included" in send["actual_release_unavailable_reason"]
    assert "actual_end_ts" not in send
    assert recv["component_duration_split_us"]["collective_wait"] == 0.0


def test_replay_edge_export_passes_hostdelay_occurrence_metadata_without_timing_change():
    host_extras = {
        "host_machine_id": "host0",
        "host_dispatch_queue_id": "host0:rank:0",
        "host_timing_dispatch_scope": "host_machine",
        "host_dispatch_model": "single_dispatch_queue_per_host_execution_context",
    }
    occurrence_extras = {
        **host_extras,
        "observed_gap_us": 40,
        "hostdelay_occurrence_metadata_schema_version": (
            "hostdelay_occurrence_metadata_export_v1"
        ),
        "hostdelay_occurrence_metadata_opt_in_flag": True,
        "stable_hostdelay_occurrence_id": (
            "rank:0:hostdelay_occurrence:raw_ordinal:0->raw_ordinal:1"
        ),
        "hostdelay_occurrence_interval_start_ts_us": 100,
        "hostdelay_occurrence_interval_end_ts_us": 140,
        "hostdelay_occurrence_duration_us": 40,
        "hostdelay_occurrence_raw_predecessor_event_id": "r0:e0",
        "hostdelay_occurrence_raw_predecessor_api": "cudaLaunchKernel",
        "hostdelay_occurrence_raw_successor_event_id": "r0:e1",
        "hostdelay_occurrence_raw_successor_api": "cudaEventRecord",
        "hostdelay_occurrence_semantic_predecessor_event_id": "r0:e0",
        "hostdelay_occurrence_semantic_predecessor_api": "cudaLaunchKernel",
        "hostdelay_occurrence_semantic_successor_event_id": "r0:e1",
        "hostdelay_occurrence_semantic_successor_api": "cudaEventRecord",
        "hostdelay_occurrence_raw_boundary_family": (
            "cudaLaunchKernel -> cudaEventRecord"
        ),
        "hostdelay_occurrence_semantic_boundary_family": (
            "cudaLaunchKernel -> cudaEventRecord"
        ),
        "hostdelay_occurrence_count_once_group_id": (
            "hostdelay_occurrence_count_once:"
            "rank:0:hostdelay_occurrence:raw_ordinal:0->raw_ordinal:1"
        ),
        "hostdelay_occurrence_count_once_status": "unavailable",
        "hostdelay_occurrence_nonoverlap_status": "unavailable",
        "hostdelay_occurrence_cuda_event_wait_map_safety_status": "unavailable",
        "hostdelay_occurrence_collective_wait_map_safety_status": "unavailable",
        "hostdelay_occurrence_join_metadata_schema_version": (
            "hostdelay_occurrence_join_metadata_export_v1"
        ),
        "hostdelay_occurrence_join_metadata_opt_in_flag": True,
        "hostdelay_occurrence_boundary_origin_join_key": (
            "rank:0:hostdelay_occurrence:raw_ordinal:0->raw_ordinal:1"
        ),
        "hostdelay_occurrence_boundary_origin_status": "unresolved",
        "hostdelay_occurrence_boundary_visibility_join_key": (
            "rank:0:hostdelay_occurrence:raw_ordinal:0->raw_ordinal:1"
        ),
        "hostdelay_occurrence_boundary_visibility_status": "unresolved",
        "hostdelay_occurrence_count_once_join_key": (
            "hostdelay_occurrence_count_once:"
            "rank:0:hostdelay_occurrence:raw_ordinal:0->raw_ordinal:1"
        ),
        "hostdelay_occurrence_nonoverlap_join_key": (
            "rank:0:hostdelay_occurrence:raw_ordinal:0->raw_ordinal:1"
        ),
        "hostdelay_occurrence_cuda_event_waitmap_join_key": (
            "rank:0:hostdelay_occurrence:raw_ordinal:0->raw_ordinal:1"
        ),
        "hostdelay_occurrence_collective_waitmap_join_key": (
            "rank:0:hostdelay_occurrence:raw_ordinal:0->raw_ordinal:1"
        ),
        "hostdelay_occurrence_join_repair_ready": False,
        "hostdelay_occurrence_join_safe_to_use_as_repair_evidence": False,
        "hostdelay_occurrence_join_safe_to_use_as_subtraction_delta": False,
        "hostdelay_occurrence_join_safe_delta_us": None,
        "hostdelay_occurrence_repair_ready": False,
        "hostdelay_occurrence_safe_to_use_as_repair_evidence": False,
        "hostdelay_occurrence_safe_to_use_as_subtraction_delta": False,
        "hostdelay_occurrence_safe_delta_us": None,
        "repair_ready": False,
        "safe_to_use_as_repair_evidence": False,
        "safe_to_use_as_subtraction_delta": False,
        "safe_delta_us": None,
    }
    events = (
        AnnotatedEvent(
            id="r0:e0",
            rank=0,
            ordinal=0,
            source=TraceSource.FAKE,
            ts=100,
            pid=1,
            tid=2,
            module="libcudart.so.12",
            api="cudaLaunchKernel",
            op_type="kernel_launch",
            extras={**host_extras, "stream_id": "7"},
            duration_us=1.0,
            duration_source="test",
        ),
        AnnotatedEvent(
            id="r0:h1",
            rank=0,
            ordinal=1,
            source=TraceSource.FAKE,
            ts=100,
            pid=1,
            tid=2,
            module="host.dispatch",
            api="__hostDelay__",
            op_type="host_delay",
            extras=occurrence_extras,
            prev_event_id="r0:e0",
            duration_us=40.0,
            duration_source="host_delay",
        ),
        AnnotatedEvent(
            id="r0:e1",
            rank=0,
            ordinal=1,
            source=TraceSource.FAKE,
            ts=140,
            pid=1,
            tid=2,
            module="libcudart.so.12",
            api="cudaEventRecord",
            op_type="event_record",
            extras={**host_extras, "event_id": "ev0", "stream_id": "7"},
            prev_event_id="r0:h1",
            duration_us=1.0,
            duration_source="test",
        ),
    )
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/replay"),
        source=TraceSource.FAKE,
        rank_events={0: events},
        global_events=events,
        collective_groups={},
    )

    result = replay_annotated_trace(trace)
    export = export_replay_edge_diagnostics(trace, result)
    hostdelay = [
        row
        for row in export["simulated_events"]
        if row["materialized_event_id"] == "r0:h1"
    ][0]
    metadata = hostdelay["link_fields"]

    assert result.total_time_us == 42.0
    assert metadata["hostdelay_occurrence_metadata_schema_version"] == (
        "hostdelay_occurrence_metadata_export_v1"
    )
    assert metadata["stable_hostdelay_occurrence_id"] == (
        "rank:0:hostdelay_occurrence:raw_ordinal:0->raw_ordinal:1"
    )
    assert metadata["hostdelay_occurrence_interval_start_ts_us"] == 100
    assert metadata["hostdelay_occurrence_interval_end_ts_us"] == 140
    assert metadata["hostdelay_occurrence_duration_us"] == 40
    assert metadata["hostdelay_occurrence_raw_boundary_family"] == (
        "cudaLaunchKernel -> cudaEventRecord"
    )
    assert metadata["hostdelay_occurrence_semantic_boundary_family"] == (
        "cudaLaunchKernel -> cudaEventRecord"
    )
    assert metadata["hostdelay_occurrence_count_once_status"] == "unavailable"
    assert metadata["hostdelay_occurrence_nonoverlap_status"] == "unavailable"
    assert metadata["hostdelay_occurrence_cuda_event_wait_map_safety_status"] == (
        "unavailable"
    )
    assert metadata["hostdelay_occurrence_collective_wait_map_safety_status"] == (
        "unavailable"
    )
    assert metadata["hostdelay_occurrence_join_metadata_schema_version"] == (
        "hostdelay_occurrence_join_metadata_export_v1"
    )
    assert metadata["hostdelay_occurrence_boundary_origin_status"] == "unresolved"
    assert metadata["hostdelay_occurrence_boundary_visibility_status"] == "unresolved"
    assert metadata["hostdelay_occurrence_join_repair_ready"] is False
    assert metadata["hostdelay_occurrence_join_safe_to_use_as_repair_evidence"] is False
    assert metadata["hostdelay_occurrence_join_safe_to_use_as_subtraction_delta"] is False
    assert metadata["hostdelay_occurrence_repair_ready"] is False
    assert metadata["hostdelay_occurrence_safe_to_use_as_subtraction_delta"] is False


def test_replay_collective_event_polling_metadata_absent_when_only_base_occurrence_metadata():
    host_extras = {
        "host_machine_id": "host0",
        "host_dispatch_queue_id": "host0:rank:0",
        "host_timing_dispatch_scope": "host_machine",
        "host_dispatch_model": "single_dispatch_queue_per_host_execution_context",
    }
    occurrence_extras = {
        **host_extras,
        "observed_gap_us": 40,
        "hostdelay_occurrence_metadata_schema_version": (
            "hostdelay_occurrence_metadata_export_v1"
        ),
        "hostdelay_occurrence_metadata_opt_in_flag": True,
        "stable_hostdelay_occurrence_id": (
            "rank:0:hostdelay_occurrence:raw_ordinal:0->raw_ordinal:1"
        ),
        "hostdelay_occurrence_interval_start_ts_us": 100,
        "hostdelay_occurrence_interval_end_ts_us": 140,
        "hostdelay_occurrence_duration_us": 40,
        "hostdelay_occurrence_raw_boundary_family": (
            "ncclAllReduce -> cudaEventRecord"
        ),
        "hostdelay_occurrence_semantic_boundary_family": (
            "ncclAllReduce -> cudaEventRecord"
        ),
        "hostdelay_occurrence_count_once_status": "unavailable",
        "hostdelay_occurrence_cuda_event_wait_map_safety_status": "unavailable",
        "hostdelay_occurrence_collective_wait_map_safety_status": "unavailable",
        "hostdelay_occurrence_repair_ready": False,
        "hostdelay_occurrence_safe_to_use_as_repair_evidence": False,
        "hostdelay_occurrence_safe_to_use_as_subtraction_delta": False,
        "hostdelay_occurrence_safe_delta_us": None,
    }
    events = (
        AnnotatedEvent(
            id="r0:e0",
            rank=0,
            ordinal=0,
            source=TraceSource.FAKE,
            ts=100,
            pid=1,
            tid=2,
            module="libnccl.so.2",
            api="ncclAllReduce",
            op_type="nccl_collective",
            extras={**host_extras, "stream_id": "7"},
            duration_us=1.0,
            duration_source="test",
        ),
        AnnotatedEvent(
            id="r0:h1",
            rank=0,
            ordinal=1,
            source=TraceSource.FAKE,
            ts=100,
            pid=1,
            tid=2,
            module="host.dispatch",
            api="__hostDelay__",
            op_type="host_delay",
            extras=occurrence_extras,
            prev_event_id="r0:e0",
            duration_us=40.0,
            duration_source="host_delay",
        ),
        AnnotatedEvent(
            id="r0:e1",
            rank=0,
            ordinal=2,
            source=TraceSource.FAKE,
            ts=140,
            pid=1,
            tid=2,
            module="libcudart.so.12",
            api="cudaEventRecord",
            op_type="event_record",
            extras={**host_extras, "event_id": "ev0", "stream_id": "7"},
            prev_event_id="r0:h1",
            duration_us=1.0,
            duration_source="test",
        ),
    )
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/replay"),
        source=TraceSource.FAKE,
        rank_events={0: events},
        global_events=events,
        collective_groups={},
    )

    result = replay_annotated_trace(trace)
    export = export_replay_edge_diagnostics(trace, result)
    hostdelay = [
        row
        for row in export["simulated_events"]
        if row["materialized_event_id"] == "r0:h1"
    ][0]
    metadata = hostdelay["link_fields"]

    assert result.total_time_us == 42.0
    assert metadata["hostdelay_occurrence_metadata_opt_in_flag"] is True
    assert not {
        key for key in metadata if key.startswith("collective_event_polling_")
    }


def test_replay_edge_export_passes_collective_event_polling_metadata_without_timing_change():
    host_extras = {
        "host_machine_id": "host0",
        "host_dispatch_queue_id": "host0:rank:0",
        "host_timing_dispatch_scope": "host_machine",
        "host_dispatch_model": "single_dispatch_queue_per_host_execution_context",
    }
    occurrence_key = "rank:0:hostdelay_occurrence:raw_ordinal:0->raw_ordinal:1"
    occurrence_extras = {
        **host_extras,
        "observed_gap_us": 40,
        "hostdelay_occurrence_metadata_schema_version": (
            "hostdelay_occurrence_metadata_export_v1"
        ),
        "hostdelay_occurrence_metadata_opt_in_flag": True,
        "stable_hostdelay_occurrence_id": occurrence_key,
        "hostdelay_occurrence_interval_start_ts_us": 100,
        "hostdelay_occurrence_interval_end_ts_us": 140,
        "hostdelay_occurrence_duration_us": 40,
        "hostdelay_occurrence_raw_boundary_family": (
            "ncclAllReduce -> cudaEventRecord"
        ),
        "hostdelay_occurrence_semantic_boundary_family": (
            "ncclAllReduce -> cudaEventRecord"
        ),
        "hostdelay_occurrence_count_once_status": "unavailable",
        "hostdelay_occurrence_cuda_event_wait_map_safety_status": "unavailable",
        "hostdelay_occurrence_collective_wait_map_safety_status": "unavailable",
        "hostdelay_occurrence_repair_ready": False,
        "hostdelay_occurrence_safe_to_use_as_repair_evidence": False,
        "hostdelay_occurrence_safe_to_use_as_subtraction_delta": False,
        "hostdelay_occurrence_safe_delta_us": None,
        "collective_event_polling_metadata_schema_version": (
            "collective_event_polling_boundary_metadata_v1"
        ),
        "collective_event_polling_metadata_opt_in_flag": True,
        "collective_event_polling_metadata_behavior_effect": (
            "diagnostic_export_only_no_replay_or_timing_behavior_change"
        ),
        "collective_event_polling_occurrence_join_key": occurrence_key,
        "collective_event_polling_raw_boundary_family": (
            "ncclAllReduce -> cudaEventRecord"
        ),
        "collective_event_polling_semantic_boundary_family": (
            "ncclAllReduce -> cudaEventRecord"
        ),
        "collective_event_polling_target_family": "ncclAllReduce",
        "collective_event_polling_collective_group_id": "cg0",
        "collective_event_polling_collective_member_id": "rank0",
        "collective_event_polling_collective_api": "ncclAllReduce",
        "collective_event_polling_collective_call_order": 5,
        "collective_event_polling_communicator_id": "comm0",
        "collective_event_polling_participant_count": 16,
        "collective_event_polling_cuda_event_handle": "ev0",
        "collective_event_polling_cuda_event_version": 3,
        "collective_event_polling_cuda_event_record_id": "r0:e1",
        "collective_event_polling_boundary_origin_status": "unresolved",
        "collective_event_polling_boundary_visibility_status": "unresolved",
        "collective_event_polling_wait_map_release_status": "unavailable",
        "collective_event_polling_count_once_status": "unavailable",
        "collective_event_polling_nonoverlap_status": "unavailable",
        "collective_event_polling_repair_ready": False,
        "collective_event_polling_safe_to_use_as_repair_evidence": False,
        "collective_event_polling_safe_to_use_as_subtraction_delta": False,
        "collective_event_polling_safe_delta_us": None,
        "repair_ready": False,
        "safe_to_use_as_repair_evidence": False,
        "safe_to_use_as_subtraction_delta": False,
        "safe_delta_us": None,
    }
    events = (
        AnnotatedEvent(
            id="r0:e0",
            rank=0,
            ordinal=0,
            source=TraceSource.FAKE,
            ts=100,
            pid=1,
            tid=2,
            module="libnccl.so.2",
            api="ncclAllReduce",
            op_type="nccl_collective",
            extras={**host_extras, "stream_id": "7"},
            duration_us=1.0,
            duration_source="test",
        ),
        AnnotatedEvent(
            id="r0:h1",
            rank=0,
            ordinal=1,
            source=TraceSource.FAKE,
            ts=100,
            pid=1,
            tid=2,
            module="host.dispatch",
            api="__hostDelay__",
            op_type="host_delay",
            extras=occurrence_extras,
            prev_event_id="r0:e0",
            duration_us=40.0,
            duration_source="host_delay",
        ),
        AnnotatedEvent(
            id="r0:e1",
            rank=0,
            ordinal=2,
            source=TraceSource.FAKE,
            ts=140,
            pid=1,
            tid=2,
            module="libcudart.so.12",
            api="cudaEventRecord",
            op_type="event_record",
            extras={**host_extras, "event_id": "ev0", "stream_id": "7"},
            prev_event_id="r0:h1",
            duration_us=1.0,
            duration_source="test",
        ),
    )
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/replay"),
        source=TraceSource.FAKE,
        rank_events={0: events},
        global_events=events,
        collective_groups={},
    )

    result = replay_annotated_trace(trace)
    export = export_replay_edge_diagnostics(trace, result)
    hostdelay = [
        row
        for row in export["simulated_events"]
        if row["materialized_event_id"] == "r0:h1"
    ][0]
    metadata = hostdelay["link_fields"]

    assert result.total_time_us == 42.0
    assert _COLLECTIVE_EVENT_POLLING_METADATA_FIELDS <= metadata.keys()
    assert metadata["collective_event_polling_metadata_schema_version"] == (
        "collective_event_polling_boundary_metadata_v1"
    )
    assert metadata["collective_event_polling_raw_boundary_family"] == (
        "ncclAllReduce -> cudaEventRecord"
    )
    assert metadata["collective_event_polling_semantic_boundary_family"] == (
        "ncclAllReduce -> cudaEventRecord"
    )
    assert metadata["collective_event_polling_target_family"] == "ncclAllReduce"
    assert metadata["collective_event_polling_collective_group_id"] == "cg0"
    assert metadata["collective_event_polling_collective_api"] == "ncclAllReduce"
    assert metadata["collective_event_polling_cuda_event_handle"] == "ev0"
    assert metadata["collective_event_polling_boundary_origin_status"] == (
        "unresolved"
    )
    assert metadata["collective_event_polling_boundary_visibility_status"] == (
        "unresolved"
    )
    assert metadata["collective_event_polling_wait_map_release_status"] == (
        "unavailable"
    )
    assert metadata["collective_event_polling_count_once_status"] == "unavailable"
    assert metadata["collective_event_polling_nonoverlap_status"] == "unavailable"
    assert metadata["collective_event_polling_repair_ready"] is False
    assert metadata[
        "collective_event_polling_safe_to_use_as_repair_evidence"
    ] is False
    assert metadata[
        "collective_event_polling_safe_to_use_as_subtraction_delta"
    ] is False
    assert metadata["collective_event_polling_safe_delta_us"] is None


def _event_polling_boundary_trace() -> AnnotatedTrace:
    host_extras = {
        "host_machine_id": "host0",
        "host_dispatch_queue_id": "host0:rank:0",
        "host_timing_dispatch_scope": "host_machine",
        "host_dispatch_model": "single_dispatch_queue_per_host_execution_context",
    }
    occurrence_key = "rank:0:hostdelay_occurrence:raw_ordinal:0->raw_ordinal:1"
    occurrence_extras = {
        **host_extras,
        "observed_gap_us": 40,
        "hostdelay_occurrence_metadata_schema_version": (
            "hostdelay_occurrence_metadata_export_v1"
        ),
        "hostdelay_occurrence_metadata_opt_in_flag": True,
        "stable_hostdelay_occurrence_id": occurrence_key,
        "hostdelay_occurrence_interval_start_ts_us": 100,
        "hostdelay_occurrence_interval_end_ts_us": 140,
        "hostdelay_occurrence_duration_us": 40,
        "hostdelay_occurrence_raw_boundary_family": (
            "cudaEventQuery -> cudaEventQuery"
        ),
        "hostdelay_occurrence_semantic_boundary_family": (
            "cudaEventQuery -> cudaEventQuery"
        ),
        "hostdelay_occurrence_repair_ready": False,
        "hostdelay_occurrence_safe_to_use_as_repair_evidence": False,
        "hostdelay_occurrence_safe_to_use_as_subtraction_delta": False,
        "hostdelay_occurrence_safe_delta_us": None,
        "event_polling_boundary_metadata_schema_version": (
            "event_polling_boundary_origin_visibility_trace_processing_metadata_v1"
        ),
        "event_polling_boundary_metadata_opt_in_flag": True,
        "event_polling_boundary_behavior_effect": (
            "diagnostic_only_no_duration_materialization_or_replay_change"
        ),
        "event_polling_boundary_occurrence_id": occurrence_key,
        "event_polling_boundary_raw_family": "cudaEventQuery -> cudaEventQuery",
        "event_polling_boundary_semantic_family": (
            "cudaEventQuery -> cudaEventQuery"
        ),
        "event_polling_boundary_target_class": (
            "nonblocking_cudaEventQuery_polling_pressure"
        ),
        "event_polling_boundary_polling_class": (
            "nonblocking_cudaEventQuery_polling"
        ),
        "event_polling_boundary_origin_status": (
            "classified_paper_visible_by_default"
        ),
        "event_polling_boundary_visibility_status": "paper_visible_by_default",
        "event_polling_boundary_paper_visibility_class": "paper_visible_by_default",
        "event_polling_boundary_count_once_status": "unavailable",
        "event_polling_boundary_nonoverlap_status": "unavailable",
        "event_polling_boundary_wait_map_safety_status": "unavailable",
        "event_polling_boundary_repair_ready": False,
        "event_polling_boundary_safe_to_use_as_repair_evidence": False,
        "event_polling_boundary_safe_to_use_as_subtraction_delta": False,
        "event_polling_boundary_safe_delta_us": None,
    }
    events = (
        AnnotatedEvent(
            id="r0:e0",
            rank=0,
            ordinal=0,
            source=TraceSource.FAKE,
            ts=100,
            pid=1,
            tid=2,
            module="libcudart.so.12",
            api="cudaEventQuery",
            op_type="event_query",
            extras={**host_extras, "event_id": "ev0", "event_version": 1},
            duration_us=1.0,
            duration_source="test",
        ),
        AnnotatedEvent(
            id="r0:h1",
            rank=0,
            ordinal=1,
            source=TraceSource.FAKE,
            ts=100,
            pid=1,
            tid=2,
            module="host.dispatch",
            api="__hostDelay__",
            op_type="host_delay",
            extras=occurrence_extras,
            prev_event_id="r0:e0",
            duration_us=40.0,
            duration_source="host_delay",
        ),
        AnnotatedEvent(
            id="r0:e1",
            rank=0,
            ordinal=2,
            source=TraceSource.FAKE,
            ts=140,
            pid=1,
            tid=2,
            module="libcudart.so.12",
            api="cudaEventQuery",
            op_type="event_query",
            extras={**host_extras, "event_id": "ev0", "event_version": 1},
            prev_event_id="r0:h1",
            duration_us=1.0,
            duration_source="test",
        ),
    )
    return AnnotatedTrace(
        trace_dir=Path("/tmp/replay"),
        source=TraceSource.FAKE,
        rank_events={0: events},
        global_events=events,
        collective_groups={},
    )


def test_replay_event_polling_boundary_metadata_default_off_absent(monkeypatch):
    monkeypatch.delenv(
        "MAYA_ENABLE_EVENT_POLLING_BOUNDARY_ORIGIN_VISIBILITY_METADATA_EXPORT",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_EVENT_POLLING_BOUNDARY_ORIGIN_VISIBILITY_METADATA_EXPORT",
        raising=False,
    )
    trace = _event_polling_boundary_trace()

    result = replay_annotated_trace(trace)
    export = export_replay_edge_diagnostics(trace, result)
    hostdelay = [
        row
        for row in export["simulated_events"]
        if row["materialized_event_id"] == "r0:h1"
    ][0]
    metadata = hostdelay["link_fields"]

    assert result.total_time_us == 42.0
    assert not {
        key
        for key in metadata
        if key.startswith(_EVENT_POLLING_BOUNDARY_METADATA_PREFIX)
    }


def test_replay_event_polling_boundary_metadata_opt_in_passes_diagnostics_only(
    monkeypatch,
):
    monkeypatch.setenv(
        "MAYA_ENABLE_EVENT_POLLING_BOUNDARY_ORIGIN_VISIBILITY_METADATA_EXPORT",
        "1",
    )
    trace = _event_polling_boundary_trace()

    result = replay_annotated_trace(trace)
    export = export_replay_edge_diagnostics(trace, result)
    hostdelay = [
        row
        for row in export["simulated_events"]
        if row["materialized_event_id"] == "r0:h1"
    ][0]
    metadata = hostdelay["link_fields"]

    assert result.total_time_us == 42.0
    assert _EVENT_POLLING_BOUNDARY_METADATA_FIELDS <= metadata.keys()
    assert metadata["event_polling_boundary_metadata_schema_version"] == (
        "event_polling_boundary_origin_visibility_trace_processing_metadata_v1"
    )
    assert metadata["event_polling_boundary_target_class"] == (
        "nonblocking_cudaEventQuery_polling_pressure"
    )
    assert metadata["event_polling_boundary_polling_class"] == (
        "nonblocking_cudaEventQuery_polling"
    )
    assert metadata["event_polling_boundary_origin_status"] == (
        "classified_paper_visible_by_default"
    )
    assert metadata["event_polling_boundary_visibility_status"] == (
        "paper_visible_by_default"
    )
    assert metadata["event_polling_boundary_paper_visibility_class"] == (
        "paper_visible_by_default"
    )
    assert metadata["event_polling_boundary_count_once_status"] == "unavailable"
    assert metadata["event_polling_boundary_nonoverlap_status"] == "unavailable"
    assert metadata["event_polling_boundary_wait_map_safety_status"] == "unavailable"
    assert metadata["event_polling_boundary_repair_ready"] is False
    assert metadata["event_polling_boundary_safe_to_use_as_repair_evidence"] is False
    assert metadata["event_polling_boundary_safe_to_use_as_subtraction_delta"] is False
    assert metadata["event_polling_boundary_safe_delta_us"] is None


def _boundary_origin_subregion_trace() -> AnnotatedTrace:
    host_extras = {
        "host_machine_id": "host0",
        "host_dispatch_queue_id": "host0:rank:0",
        "host_timing_dispatch_scope": "host_machine",
        "host_dispatch_model": "single_dispatch_queue_per_host_execution_context",
    }
    occurrence_key = "rank:0:hostdelay_occurrence:raw_ordinal:0->raw_ordinal:1"
    occurrence_extras = {
        **host_extras,
        "observed_gap_us": 40,
        "hostdelay_occurrence_metadata_schema_version": (
            "hostdelay_occurrence_metadata_export_v1"
        ),
        "hostdelay_occurrence_metadata_opt_in_flag": True,
        "stable_hostdelay_occurrence_id": occurrence_key,
        "hostdelay_occurrence_interval_start_ts_us": 100,
        "hostdelay_occurrence_interval_end_ts_us": 140,
        "hostdelay_occurrence_duration_us": 40,
        "hostdelay_occurrence_raw_boundary_family": (
            "cudaLaunchKernel -> cudaEventQuery"
        ),
        "hostdelay_occurrence_semantic_boundary_family": (
            "cudaLaunchKernel -> cudaEventQuery"
        ),
        "hostdelay_occurrence_repair_ready": False,
        "hostdelay_occurrence_safe_to_use_as_repair_evidence": False,
        "hostdelay_occurrence_safe_to_use_as_subtraction_delta": False,
        "hostdelay_occurrence_safe_delta_us": None,
        "boundary_origin_subregion_metadata_schema_version": (
            "boundary_origin_subregion_proof_metadata_v1"
        ),
        "boundary_origin_subregion_metadata_opt_in_flag": True,
        "boundary_origin_subregion_behavior_effect": (
            "diagnostic_only_no_duration_materialization_or_replay_change"
        ),
        "boundary_origin_subregion_occurrence_id": occurrence_key,
        "boundary_origin_subregion_id": (
            f"boundary_origin_subregion:{occurrence_key}"
        ),
        "boundary_origin_subregion_kind": (
            "candidate_internal_event_launch_library_control_gap"
        ),
        "boundary_origin_subregion_candidate_role": (
            "candidate_internal_subregion_needs_strict_proof"
        ),
        "boundary_origin_subregion_raw_family": (
            "cudaLaunchKernel -> cudaEventQuery"
        ),
        "boundary_origin_subregion_semantic_family": (
            "cudaLaunchKernel -> cudaEventQuery"
        ),
        "boundary_origin_subregion_raw_semantic_pair": (
            "cudaLaunchKernel -> cudaEventQuery || "
            "cudaLaunchKernel -> cudaEventQuery"
        ),
        "boundary_origin_subregion_origin_status": (
            "candidate_needs_strict_boundary_origin_proof"
        ),
        "boundary_origin_subregion_visibility_status": (
            "candidate_needs_strict_visibility_proof"
        ),
        "boundary_origin_subregion_paper_visibility_class": "unresolved_mixed",
        "boundary_origin_subregion_strict_extent_status": "unavailable_or_unproven",
        "boundary_origin_subregion_strict_proof_status": "unavailable_or_unproven",
        "boundary_origin_subregion_count_once_status": "unavailable",
        "boundary_origin_subregion_nonoverlap_status": "unavailable",
        "boundary_origin_subregion_wait_map_safety_status": "unavailable",
        "boundary_origin_subregion_fresh16_fresh8_join_key_status": (
            "diagnostic_context_only_not_strict_counterpart"
        ),
        "boundary_origin_subregion_repair_ready": False,
        "boundary_origin_subregion_safe_to_use_as_repair_evidence": False,
        "boundary_origin_subregion_safe_to_use_as_subtraction_delta": False,
        "boundary_origin_subregion_safe_delta_us": None,
        "strict_subregion_extent_metadata_schema_version": (
            "strict_subregion_extent_source_metadata_v1"
        ),
        "strict_subregion_extent_metadata_opt_in_flag": True,
        "strict_subregion_extent_behavior_effect": (
            "diagnostic_only_no_duration_materialization_or_replay_change"
        ),
        "strict_subregion_extent_occurrence_id": occurrence_key,
        "strict_subregion_extent_boundary_id": (
            f"strict_subregion_extent:{occurrence_key}"
        ),
        "strict_subregion_extent_raw_family": (
            "cudaLaunchKernel -> cudaEventQuery"
        ),
        "strict_subregion_extent_semantic_family": (
            "cudaLaunchKernel -> cudaEventQuery"
        ),
        "strict_subregion_extent_raw_semantic_pair": (
            "cudaLaunchKernel -> cudaEventQuery || "
            "cudaLaunchKernel -> cudaEventQuery"
        ),
        "strict_subregion_extent_target_family_class": (
            "unresolved_launch_event_library_control_boundary"
        ),
        "strict_subregion_extent_candidate_subregion_kind": (
            "candidate_internal_event_launch_library_control_gap"
        ),
        "strict_subregion_extent_candidate_subregion_role": (
            "requires_strict_subregion_extent_source_proof"
        ),
        "strict_subregion_extent_start_ts_us": None,
        "strict_subregion_extent_end_ts_us": None,
        "strict_subregion_extent_duration_us_context_only": 40,
        "strict_subregion_extent_timestamp_source_kind": (
            "none_strict_source_unavailable"
        ),
        "strict_subregion_extent_source_is_non_perturbing": False,
        "strict_subregion_extent_source_uses_runtime_endpoint_substitution": False,
        "strict_subregion_extent_source_uses_measured_actual_runtime": False,
        "strict_subregion_extent_source_uses_hostdelay_shortening": False,
        "strict_subregion_extent_source_proof_status": "unavailable_or_unproven",
        "strict_subregion_extent_origin_status": (
            "candidate_needs_strict_boundary_origin_proof"
        ),
        "strict_subregion_extent_visibility_status": (
            "candidate_needs_strict_visibility_proof"
        ),
        "strict_subregion_extent_paper_visibility_class": "unresolved_mixed",
        "strict_subregion_extent_count_once_status": "unavailable",
        "strict_subregion_extent_nonoverlap_status": "unavailable",
        "strict_subregion_extent_cuda_event_wait_map_safety_status": "unavailable",
        "strict_subregion_extent_collective_wait_map_safety_status": "unavailable",
        "strict_subregion_extent_repair_ready": False,
        "strict_subregion_extent_safe_to_use_as_repair_evidence": False,
        "strict_subregion_extent_safe_to_use_as_subtraction_delta": False,
        "strict_subregion_extent_safe_delta_us": None,
        "strict_subregion_extent_runtime_or_endpoint_substitution_used": False,
        "strict_subregion_extent_hostdelay_shortening_used": False,
        "strict_subregion_extent_rank_workload_special_case_used": False,
    }
    events = (
        AnnotatedEvent(
            id="r0:e0",
            rank=0,
            ordinal=0,
            source=TraceSource.FAKE,
            ts=100,
            pid=1,
            tid=2,
            module="libcudart.so.12",
            api="cudaLaunchKernel",
            op_type="kernel_launch",
            extras={**host_extras, "stream_id": "7"},
            duration_us=1.0,
            duration_source="test",
        ),
        AnnotatedEvent(
            id="r0:h1",
            rank=0,
            ordinal=1,
            source=TraceSource.FAKE,
            ts=100,
            pid=1,
            tid=2,
            module="host.dispatch",
            api="__hostDelay__",
            op_type="host_delay",
            extras=occurrence_extras,
            prev_event_id="r0:e0",
            duration_us=40.0,
            duration_source="host_delay",
        ),
        AnnotatedEvent(
            id="r0:e1",
            rank=0,
            ordinal=2,
            source=TraceSource.FAKE,
            ts=140,
            pid=1,
            tid=2,
            module="libcudart.so.12",
            api="cudaEventQuery",
            op_type="event_query",
            extras={**host_extras, "event_id": "ev0", "event_version": 1},
            prev_event_id="r0:h1",
            duration_us=1.0,
            duration_source="test",
        ),
    )
    return AnnotatedTrace(
        trace_dir=Path("/tmp/replay"),
        source=TraceSource.FAKE,
        rank_events={0: events},
        global_events=events,
        collective_groups={},
    )


def test_replay_boundary_origin_subregion_metadata_default_off_absent(monkeypatch):
    monkeypatch.delenv(
        "MAYA_ENABLE_BOUNDARY_ORIGIN_SUBREGION_PROOF_METADATA_EXPORT",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_BOUNDARY_ORIGIN_SUBREGION_PROOF_METADATA_EXPORT",
        raising=False,
    )
    trace = _boundary_origin_subregion_trace()

    result = replay_annotated_trace(trace)
    export = export_replay_edge_diagnostics(trace, result)
    hostdelay = [
        row
        for row in export["simulated_events"]
        if row["materialized_event_id"] == "r0:h1"
    ][0]
    metadata = hostdelay["link_fields"]

    assert result.total_time_us == 41.0
    assert not {
        key
        for key in metadata
        if key.startswith(_BOUNDARY_ORIGIN_SUBREGION_METADATA_PREFIX)
    }


def test_replay_boundary_origin_subregion_metadata_opt_in_passes_diagnostics_only(
    monkeypatch,
):
    monkeypatch.setenv(
        "MAYA_ENABLE_BOUNDARY_ORIGIN_SUBREGION_PROOF_METADATA_EXPORT",
        "1",
    )
    trace = _boundary_origin_subregion_trace()

    result = replay_annotated_trace(trace)
    export = export_replay_edge_diagnostics(trace, result)
    hostdelay = [
        row
        for row in export["simulated_events"]
        if row["materialized_event_id"] == "r0:h1"
    ][0]
    metadata = hostdelay["link_fields"]

    assert result.total_time_us == 41.0
    assert _BOUNDARY_ORIGIN_SUBREGION_METADATA_FIELDS <= metadata.keys()
    assert metadata["boundary_origin_subregion_metadata_schema_version"] == (
        "boundary_origin_subregion_proof_metadata_v1"
    )
    assert metadata["boundary_origin_subregion_kind"] == (
        "candidate_internal_event_launch_library_control_gap"
    )
    assert metadata["boundary_origin_subregion_candidate_role"] == (
        "candidate_internal_subregion_needs_strict_proof"
    )
    assert metadata["boundary_origin_subregion_raw_family"] == (
        "cudaLaunchKernel -> cudaEventQuery"
    )
    assert metadata["boundary_origin_subregion_semantic_family"] == (
        "cudaLaunchKernel -> cudaEventQuery"
    )
    assert metadata["boundary_origin_subregion_origin_status"] == (
        "candidate_needs_strict_boundary_origin_proof"
    )
    assert metadata["boundary_origin_subregion_visibility_status"] == (
        "candidate_needs_strict_visibility_proof"
    )
    assert metadata["boundary_origin_subregion_paper_visibility_class"] == (
        "unresolved_mixed"
    )
    assert metadata["boundary_origin_subregion_strict_extent_status"] == (
        "unavailable_or_unproven"
    )
    assert metadata["boundary_origin_subregion_strict_proof_status"] == (
        "unavailable_or_unproven"
    )
    assert metadata["boundary_origin_subregion_count_once_status"] == "unavailable"
    assert metadata["boundary_origin_subregion_nonoverlap_status"] == "unavailable"
    assert metadata["boundary_origin_subregion_wait_map_safety_status"] == (
        "unavailable"
    )
    assert metadata["boundary_origin_subregion_repair_ready"] is False
    assert metadata[
        "boundary_origin_subregion_safe_to_use_as_repair_evidence"
    ] is False
    assert metadata[
        "boundary_origin_subregion_safe_to_use_as_subtraction_delta"
    ] is False
    assert metadata["boundary_origin_subregion_safe_delta_us"] is None


def test_replay_strict_subregion_extent_metadata_default_off_absent_when_related_opted_in(
    monkeypatch,
):
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
    trace = _boundary_origin_subregion_trace()

    result = replay_annotated_trace(trace)
    export = export_replay_edge_diagnostics(trace, result)
    hostdelay = [
        row
        for row in export["simulated_events"]
        if row["materialized_event_id"] == "r0:h1"
    ][0]
    metadata = hostdelay["link_fields"]

    assert result.total_time_us == 41.0
    assert _BOUNDARY_ORIGIN_SUBREGION_METADATA_FIELDS <= metadata.keys()
    assert not {
        key
        for key in metadata
        if key.startswith(_STRICT_SUBREGION_EXTENT_METADATA_PREFIX)
    }


def test_replay_strict_subregion_extent_metadata_opt_in_passes_diagnostics_only(
    monkeypatch,
):
    monkeypatch.setenv(
        "MAYA_ENABLE_STRICT_SUBREGION_EXTENT_SOURCE_METADATA_EXPORT",
        "1",
    )
    trace = _boundary_origin_subregion_trace()

    result = replay_annotated_trace(trace)
    export = export_replay_edge_diagnostics(trace, result)
    hostdelay = [
        row
        for row in export["simulated_events"]
        if row["materialized_event_id"] == "r0:h1"
    ][0]
    metadata = hostdelay["link_fields"]

    assert result.total_time_us == 41.0
    assert _STRICT_SUBREGION_EXTENT_METADATA_FIELDS <= metadata.keys()
    assert metadata["strict_subregion_extent_metadata_schema_version"] == (
        "strict_subregion_extent_source_metadata_v1"
    )
    assert metadata["strict_subregion_extent_target_family_class"] == (
        "unresolved_launch_event_library_control_boundary"
    )
    assert metadata["strict_subregion_extent_candidate_subregion_kind"] == (
        "candidate_internal_event_launch_library_control_gap"
    )
    assert metadata["strict_subregion_extent_candidate_subregion_role"] == (
        "requires_strict_subregion_extent_source_proof"
    )
    assert metadata["strict_subregion_extent_raw_family"] == (
        "cudaLaunchKernel -> cudaEventQuery"
    )
    assert metadata["strict_subregion_extent_semantic_family"] == (
        "cudaLaunchKernel -> cudaEventQuery"
    )
    assert metadata["strict_subregion_extent_start_ts_us"] is None
    assert metadata["strict_subregion_extent_end_ts_us"] is None
    assert metadata["strict_subregion_extent_duration_us_context_only"] == 40
    assert metadata["strict_subregion_extent_timestamp_source_kind"] == (
        "none_strict_source_unavailable"
    )
    assert metadata["strict_subregion_extent_source_proof_status"] == (
        "unavailable_or_unproven"
    )
    assert metadata["strict_subregion_extent_origin_status"] == (
        "candidate_needs_strict_boundary_origin_proof"
    )
    assert metadata["strict_subregion_extent_visibility_status"] == (
        "candidate_needs_strict_visibility_proof"
    )
    assert metadata["strict_subregion_extent_paper_visibility_class"] == (
        "unresolved_mixed"
    )
    assert metadata["strict_subregion_extent_count_once_status"] == "unavailable"
    assert metadata["strict_subregion_extent_nonoverlap_status"] == "unavailable"
    assert metadata["strict_subregion_extent_cuda_event_wait_map_safety_status"] == (
        "unavailable"
    )
    assert metadata[
        "strict_subregion_extent_collective_wait_map_safety_status"
    ] == "unavailable"
    assert metadata["strict_subregion_extent_repair_ready"] is False
    assert metadata[
        "strict_subregion_extent_safe_to_use_as_repair_evidence"
    ] is False
    assert metadata[
        "strict_subregion_extent_safe_to_use_as_subtraction_delta"
    ] is False
    assert metadata["strict_subregion_extent_safe_delta_us"] is None
    assert metadata[
        "strict_subregion_extent_runtime_or_endpoint_substitution_used"
    ] is False
    assert metadata["strict_subregion_extent_hostdelay_shortening_used"] is False
    assert metadata[
        "strict_subregion_extent_rank_workload_special_case_used"
    ] is False


def _collective_waitmap_keys(metadata: dict[str, object]) -> set[str]:
    return {
        key
        for key in metadata
        if key.startswith(_COLLECTIVE_EVENT_POLLING_REPLAY_WAITMAP_METADATA_PREFIX)
    }


def test_replay_collective_event_polling_replay_waitmap_metadata_default_off_absent(
    monkeypatch,
):
    monkeypatch.setenv(
        "MAYA_ENABLE_COLLECTIVE_EVENT_POLLING_BOUNDARY_METADATA_EXPORT",
        "1",
    )
    monkeypatch.delenv(
        "MAYA_ENABLE_COLLECTIVE_EVENT_POLLING_REPLAY_WAITMAP_RELEASE_METADATA_EXPORT",
        raising=False,
    )
    monkeypatch.delenv(
        "FLEXSIM_MAYA_ENABLE_COLLECTIVE_EVENT_POLLING_REPLAY_WAITMAP_RELEASE_METADATA_EXPORT",
        raising=False,
    )
    host_extras = {
        "host_machine_id": "host0",
        "host_dispatch_queue_id": "host0:rank:0",
        "host_timing_dispatch_scope": "host_machine",
        "host_dispatch_model": "single_dispatch_queue_per_host_execution_context",
    }
    hostdelay_extras = {
        **host_extras,
        "observed_gap_us": 3,
        "hostdelay_occurrence_metadata_opt_in_flag": True,
        "stable_hostdelay_occurrence_id": "hd:query",
        "hostdelay_occurrence_raw_successor_event_id": "r0:e1",
        "hostdelay_occurrence_raw_successor_api": "cudaEventQuery",
        "hostdelay_occurrence_raw_boundary_family": (
            "cudaEventRecord -> cudaEventQuery"
        ),
        "hostdelay_occurrence_semantic_boundary_family": (
            "cudaEventRecord -> cudaEventQuery"
        ),
        "collective_event_polling_metadata_opt_in_flag": True,
        "collective_event_polling_raw_boundary_family": (
            "cudaEventRecord -> cudaEventQuery"
        ),
        "collective_event_polling_semantic_boundary_family": (
            "cudaEventRecord -> cudaEventQuery"
        ),
    }
    events = (
        AnnotatedEvent(
            id="r0:e0",
            rank=0,
            ordinal=0,
            source=TraceSource.FAKE,
            ts=0,
            pid=1,
            tid=2,
            module="libcudart.so.12",
            api="cudaEventRecord",
            op_type="event_record",
            extras={**host_extras, "event_id": "ev0", "stream_id": "1"},
            duration_us=1.0,
            duration_source="test",
        ),
        AnnotatedEvent(
            id="r0:h0",
            rank=0,
            ordinal=1,
            source=TraceSource.FAKE,
            ts=1,
            pid=1,
            tid=2,
            module="host.dispatch",
            api="__hostDelay__",
            op_type="host_delay",
            extras=hostdelay_extras,
            duration_us=3.0,
            duration_source="host_delay",
        ),
        AnnotatedEvent(
            id="r0:e1",
            rank=0,
            ordinal=2,
            source=TraceSource.FAKE,
            ts=4,
            pid=1,
            tid=2,
            module="libcudart.so.12",
            api="cudaEventQuery",
            op_type="context_op",
            extras={**host_extras, "event_id": "ev0"},
            duration_us=1.0,
            duration_source="test",
        ),
    )
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/replay"),
        source=TraceSource.FAKE,
        rank_events={0: events},
        global_events=events,
        collective_groups={},
    )
    diagnostics: list[dict[str, object]] = []
    result = replay_annotated_trace(trace, diagnostic_events=diagnostics)
    export = export_replay_edge_diagnostics(trace, result, diagnostic_events=diagnostics)
    hostdelay = [
        row
        for row in export["simulated_events"]
        if row["materialized_event_id"] == "r0:h0"
    ][0]

    assert result.total_time_us == 4.0
    assert hostdelay["link_fields"]["collective_event_polling_metadata_opt_in_flag"] is True
    assert not _collective_waitmap_keys(hostdelay["link_fields"])


def test_replay_collective_event_polling_replay_waitmap_metadata_cuda_event_query_polling_only(
    monkeypatch,
):
    monkeypatch.setenv(
        "MAYA_ENABLE_COLLECTIVE_EVENT_POLLING_REPLAY_WAITMAP_RELEASE_METADATA_EXPORT",
        "1",
    )
    host_extras = {
        "host_machine_id": "host0",
        "host_dispatch_queue_id": "host0:rank:0",
        "host_timing_dispatch_scope": "host_machine",
        "host_dispatch_model": "single_dispatch_queue_per_host_execution_context",
    }
    hostdelay_extras = {
        **host_extras,
        "observed_gap_us": 3,
        "hostdelay_occurrence_metadata_opt_in_flag": True,
        "stable_hostdelay_occurrence_id": "hd:query",
        "hostdelay_occurrence_raw_successor_event_id": "r0:e1",
        "hostdelay_occurrence_raw_successor_api": "cudaEventQuery",
        "hostdelay_occurrence_raw_boundary_family": (
            "cudaEventRecord -> cudaEventQuery"
        ),
        "hostdelay_occurrence_semantic_boundary_family": (
            "cudaEventRecord -> cudaEventQuery"
        ),
    }
    events = (
        AnnotatedEvent(
            id="r0:e0",
            rank=0,
            ordinal=0,
            source=TraceSource.FAKE,
            ts=0,
            pid=1,
            tid=2,
            module="libcudart.so.12",
            api="cudaEventRecord",
            op_type="event_record",
            extras={**host_extras, "event_id": "ev0", "stream_id": "1"},
            duration_us=1.0,
            duration_source="test",
        ),
        AnnotatedEvent(
            id="r0:h0",
            rank=0,
            ordinal=1,
            source=TraceSource.FAKE,
            ts=1,
            pid=1,
            tid=2,
            module="host.dispatch",
            api="__hostDelay__",
            op_type="host_delay",
            extras=hostdelay_extras,
            duration_us=3.0,
            duration_source="host_delay",
        ),
        AnnotatedEvent(
            id="r0:e1",
            rank=0,
            ordinal=2,
            source=TraceSource.FAKE,
            ts=4,
            pid=1,
            tid=2,
            module="libcudart.so.12",
            api="cudaEventQuery",
            op_type="context_op",
            extras={**host_extras, "event_id": "ev0"},
            duration_us=1.0,
            duration_source="test",
        ),
    )
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/replay"),
        source=TraceSource.FAKE,
        rank_events={0: events},
        global_events=events,
        collective_groups={},
    )
    diagnostics: list[dict[str, object]] = []
    result = replay_annotated_trace(trace, diagnostic_events=diagnostics)
    export = export_replay_edge_diagnostics(trace, result, diagnostic_events=diagnostics)
    hostdelay = [
        row
        for row in export["simulated_events"]
        if row["materialized_event_id"] == "r0:h0"
    ][0]
    metadata = hostdelay["link_fields"]

    assert result.total_time_us == 4.0
    assert metadata[
        "collective_event_polling_replay_waitmap_nonblocking_poll_status"
    ] == "nonblocking_poll_no_replay_wait_edge_expected"
    assert metadata["collective_event_polling_replay_waitmap_no_wait_edge_expected"] is True
    assert metadata["collective_event_polling_replay_waitmap_edge_id"] is None
    assert metadata["collective_event_polling_replay_waitmap_release_us"] is None
    assert metadata["collective_event_polling_replay_waitmap_safe_delta_us"] is None
    assert metadata["collective_event_polling_replay_waitmap_repair_ready"] is False
    assert metadata[
        "collective_event_polling_replay_waitmap_safe_to_use_as_subtraction_delta"
    ] is False


def test_replay_collective_event_polling_replay_waitmap_metadata_cuda_stream_wait_event_edge(
    monkeypatch,
):
    monkeypatch.setenv(
        "MAYA_ENABLE_COLLECTIVE_EVENT_POLLING_REPLAY_WAITMAP_RELEASE_METADATA_EXPORT",
        "1",
    )
    host_extras = {
        "host_machine_id": "host0",
        "host_dispatch_queue_id": "host0:rank:0",
        "host_timing_dispatch_scope": "host_machine",
        "host_dispatch_model": "single_dispatch_queue_per_host_execution_context",
    }
    hostdelay_extras = {
        **host_extras,
        "observed_gap_us": 1,
        "hostdelay_occurrence_metadata_opt_in_flag": True,
        "stable_hostdelay_occurrence_id": "hd:wait",
        "hostdelay_occurrence_raw_successor_event_id": "r0:e1",
        "hostdelay_occurrence_raw_successor_api": "cudaStreamWaitEvent",
        "hostdelay_occurrence_raw_boundary_family": (
            "cudaEventRecord -> cudaStreamWaitEvent"
        ),
        "hostdelay_occurrence_semantic_boundary_family": (
            "cudaEventRecord -> cudaStreamWaitEvent"
        ),
    }
    events = (
        AnnotatedEvent(
            id="r0:e0",
            rank=0,
            ordinal=0,
            source=TraceSource.FAKE,
            ts=0,
            pid=1,
            tid=2,
            module="libcudart.so.12",
            api="cudaEventRecord",
            op_type="event_record",
            extras={**host_extras, "event_id": "ev0", "stream_id": "1"},
            duration_us=10.0,
            duration_source="test",
        ),
        AnnotatedEvent(
            id="r0:h0",
            rank=0,
            ordinal=1,
            source=TraceSource.FAKE,
            ts=1,
            pid=1,
            tid=2,
            module="host.dispatch",
            api="__hostDelay__",
            op_type="host_delay",
            extras=hostdelay_extras,
            duration_us=1.0,
            duration_source="host_delay",
        ),
        AnnotatedEvent(
            id="r0:e1",
            rank=0,
            ordinal=2,
            source=TraceSource.FAKE,
            ts=2,
            pid=1,
            tid=2,
            module="libcudart.so.12",
            api="cudaStreamWaitEvent",
            op_type="stream_op",
            extras={**host_extras, "event_id": "ev0", "stream_id": "2"},
            duration_us=0.0,
            duration_source="test",
        ),
    )
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/replay"),
        source=TraceSource.FAKE,
        rank_events={0: events},
        global_events=events,
        collective_groups={},
    )
    diagnostics: list[dict[str, object]] = []
    result = replay_annotated_trace(trace, diagnostic_events=diagnostics)
    export = export_replay_edge_diagnostics(trace, result, diagnostic_events=diagnostics)
    hostdelay = [
        row
        for row in export["simulated_events"]
        if row["materialized_event_id"] == "r0:h0"
    ][0]
    metadata = hostdelay["link_fields"]

    assert result.total_time_us == 10.0
    assert metadata["collective_event_polling_replay_waitmap_join_status"] == (
        "joined_predicted_cuda_event_wait_edge"
    )
    assert metadata["collective_event_polling_replay_waitmap_edge_kind"] == (
        "cuda_event_wait"
    )
    assert metadata["collective_event_polling_replay_waitmap_wait_key"] == (
        "(0, 'ev0', 1)"
    )
    assert metadata["collective_event_polling_replay_waitmap_cuda_event_id"] == "ev0"
    assert metadata["collective_event_polling_replay_waitmap_cuda_event_version"] == 1
    assert metadata["collective_event_polling_replay_waitmap_record_event_id"] == "r0:e0"
    assert metadata["collective_event_polling_replay_waitmap_wait_event_id"] == "r0:e1"
    assert metadata["collective_event_polling_replay_waitmap_wait_start_us"] is not None
    assert metadata["collective_event_polling_replay_waitmap_release_us"] is not None
    assert metadata[
        "collective_event_polling_replay_waitmap_waited_us_context_only"
    ] is not None
    assert metadata["collective_event_polling_replay_waitmap_repair_ready"] is False
    assert metadata[
        "collective_event_polling_replay_waitmap_safe_to_use_as_subtraction_delta"
    ] is False
    assert metadata["collective_event_polling_replay_waitmap_safe_delta_us"] is None


def test_replay_collective_event_polling_replay_waitmap_metadata_collective_edge(
    monkeypatch,
):
    monkeypatch.setenv(
        "MAYA_ENABLE_COLLECTIVE_EVENT_POLLING_REPLAY_WAITMAP_RELEASE_METADATA_EXPORT",
        "1",
    )
    group_id = "ncclAllReduce#0"
    host_extras = {
        "host_machine_id": "host0",
        "host_dispatch_queue_id": "host0:rank:0",
        "host_timing_dispatch_scope": "host_machine",
        "host_dispatch_model": "single_dispatch_queue_per_host_execution_context",
    }
    rank0_event = AnnotatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.FAKE,
        ts=0,
        pid=1,
        tid=2,
        module="libnccl.so.2",
        api="ncclAllReduce",
        op_type="nccl_collective",
        extras={
            **host_extras,
            "stream_id": "7",
            "collective_event_polling_metadata_opt_in_flag": True,
            "collective_event_polling_semantic_boundary_family": "ncclAllReduce",
        },
        duration_us=4.0,
        duration_source="test",
        collective_group_id=group_id,
    )
    rank1_event = AnnotatedEvent(
        id="r1:e0",
        rank=1,
        ordinal=0,
        source=TraceSource.FAKE,
        ts=5,
        pid=1,
        tid=2,
        module="libnccl.so.2",
        api="ncclAllReduce",
        op_type="nccl_collective",
        extras={
            **host_extras,
            "host_dispatch_queue_id": "host0:rank:1",
            "stream_id": "7",
        },
        duration_us=4.0,
        duration_source="test",
        collective_group_id=group_id,
    )
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/replay"),
        source=TraceSource.FAKE,
        rank_events={0: (rank0_event,), 1: (rank1_event,)},
        global_events=(rank0_event, rank1_event),
        collective_groups={
            group_id: CollectiveGroup(
                id=group_id,
                api="ncclAllReduce",
                op_type="nccl_collective",
                ranks=(0, 1),
                event_ids=("r0:e0", "r1:e0"),
                participant_count=2,
            )
        },
    )
    diagnostics: list[dict[str, object]] = []
    result = replay_annotated_trace(trace, diagnostic_events=diagnostics)
    export = export_replay_edge_diagnostics(trace, result, diagnostic_events=diagnostics)
    row = [
        row
        for row in export["simulated_events"]
        if row["materialized_event_id"] == "r0:e0"
    ][0]
    metadata = row["link_fields"]

    assert result.total_time_us == 4.0
    assert metadata["collective_event_polling_replay_waitmap_join_status"] == (
        "joined_predicted_collective_wait_edge"
    )
    assert metadata["collective_event_polling_replay_waitmap_edge_kind"] == (
        "collective_wait"
    )
    assert metadata["collective_event_polling_replay_waitmap_collective_group_id"] == (
        group_id
    )
    assert metadata[
        "collective_event_polling_replay_waitmap_collective_member_event_id"
    ] == "r0:e0"
    assert metadata[
        "collective_event_polling_replay_waitmap_collective_released_by_event_id"
    ] == "r1:e0"
    assert metadata["collective_event_polling_replay_waitmap_repair_ready"] is False
    assert metadata[
        "collective_event_polling_replay_waitmap_safe_to_use_as_subtraction_delta"
    ] is False
    assert metadata["collective_event_polling_replay_waitmap_safe_delta_us"] is None


def test_replay_edge_export_normalizes_host_control_boundary_metadata_without_timing_change(
    monkeypatch,
):
    def make_trace(with_metadata: bool) -> AnnotatedTrace:
        host_extras = {
            "host_machine_id": "host0",
            "host_dispatch_queue_id": "host0:rank:0",
            "host_timing_dispatch_scope": "host_machine",
            "host_dispatch_model": "single_dispatch_queue_per_host_execution_context",
        }
        host_delay_extras = dict(host_extras)
        if with_metadata:
            host_delay_extras.update({
                "observed_gap_us": 5.0,
                "raw_prev_event_id": "r0:e0",
                "raw_prev_api": "__cudaPopCallConfiguration",
                "raw_prev_ts_us": 100,
                "raw_prev_end_ts_us": 104,
                "raw_prev_end_ts_source": "raw_end_ts",
                "raw_prev_host_duration_us": 4.0,
                "raw_current_event_id": "r0:e2",
                "raw_current_api": "cudaLaunchKernel",
                "raw_current_ts_us": 150,
                "raw_current_end_ts_us": 160,
                "raw_current_end_ts_source": "computed_from_host_duration_us",
                "raw_current_host_duration_us": 10.0,
                "raw_boundary_family": "__cudaPopCallConfiguration -> cudaLaunchKernel",
                "previous_materialized_event_id": "r0:e0",
                "previous_materialized_api": "cudaLaunchKernel",
                "current_materialized_event_id": "r0:e2",
                "current_materialized_api": "cudaLaunchKernel",
                "materialized_boundary_family": "cudaLaunchKernel -> cudaLaunchKernel",
                "host_control_boundary_counterpart_schema_version": (
                    "host_control_boundary_visibility_unblocker_v2_row_evidence_v1"
                ),
                "host_control_visibility_schema_version": (
                    "host_control_launch_neighborhood_visibility_counterpart_isolation_v1"
                ),
                "host_control_visibility_opt_in_flag": True,
                "host_control_envelope_counterpart_schema_version": (
                    "host_control_replay_envelope_counterpart_metadata_v1"
                ),
                "host_control_envelope_counterpart_opt_in_flag": True,
                "host_control_envelope_counterpart_key": (
                    "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:2"
                ),
                "hostdelay_counterpart_key": (
                    "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:2"
                ),
                "host_control_envelope_materialized_interval_id": (
                    "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:2:"
                    "materialized_hostDelay"
                ),
                "host_control_envelope_counterpart_interval_id": (
                    "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:2:"
                    "endpoint_gap"
                ),
                "host_control_envelope_hostdelay_interval_id": "r0:h1",
                "hostdelay_interval_id": "r0:h1",
                "host_control_envelope_prev_raw_event_id": "rank:0:raw_ordinal:0",
                "host_control_envelope_current_raw_event_id": "rank:0:raw_ordinal:2",
                "host_control_envelope_prev_api": "__cudaPopCallConfiguration",
                "host_control_envelope_current_api": "cudaLaunchKernel",
                "host_control_envelope_rank": 0,
                "host_control_envelope_stream_id": "stream0",
                "host_control_envelope_host_dispatch_queue_id": "host0:rank:0",
                "host_control_envelope_paper_valid_window_id": "rank0:step_window",
                "host_control_envelope_prev_raw_ordinal": 0,
                "host_control_envelope_current_raw_ordinal": 2,
                "host_control_envelope_timestamp_basis": (
                    "raw_prev_end_ts_to_raw_current_ts_materialized_by_collate"
                ),
                "host_control_envelope_interval_start_ts_us": 104,
                "host_control_envelope_interval_end_ts_us": 150,
                "host_control_envelope_interval_duration_us": 5.0,
                "host_control_envelope_interval_time_basis": (
                    "materialized_hostDelay_ts_plus_observed_gap_us"
                ),
                "host_control_envelope_visibility_basis_status": (
                    "structural_metadata_only_no_mechanical_visibility_split"
                ),
                "host_control_envelope_visibility_kind": "mixed_or_unresolved",
                "host_control_envelope_replay_overlap_status": "unavailable",
                "host_control_envelope_replay_overlap_unavailable_reason": (
                    "requires_replay_interval_export_with_wait_map_and_count_once_context"
                ),
                "selected_occurrence_id": (
                    "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:2"
                ),
                "host_control_boundary_row_id": "rank:0:raw_ordinal:2",
                "host_control_boundary_occurrence_id": (
                    "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:2"
                ),
                "host_control_boundary_selection_status": "selected_family",
                "host_control_boundary_prev_raw_event_id": "rank:0:raw_ordinal:0",
                "host_control_boundary_current_raw_event_id": "rank:0:raw_ordinal:2",
                "host_control_boundary_family": "__cudaPopCallConfiguration -> cudaLaunchKernel",
                "host_control_visibility_split_status": "unavailable",
                "host_control_visibility_split_unavailable_reason": (
                    "mechanical_visibility_split_not_measured_to_preserve_wrapper_host_timing"
                ),
                "host_control_visibility_split_basis": (
                    "capture_real_adjacent_raw_row_export_without_internal_wrapper_clocks"
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
                "host_control_producer_numeric_split_unavailable_reason": (
                    "real_api_body_or_instrumentation_split_not_emitted_without_nonperturbing_brackets"
                ),
                "host_control_producer_nonoverlap_status": "unavailable",
                "host_control_producer_nonoverlap_unavailable_reason": (
                    "producer_cannot_observe_replay_wait_map_stream_collective_host_sync_or_rank_global_overlap"
                ),
                "host_control_producer_wait_map_nonoverlap_status": "unavailable",
                "host_control_producer_double_counting_nonoverlap_status": "unavailable",
                "host_control_compat_launch_pop_coverage_status": (
                    "unavailable_not_exported_by_current_real_wrapper_producer"
                ),
                "host_control_compat_launch_pop_coverage_unavailable_reason": (
                    "__cudaPopCallConfiguration_interposition_not_proven_for_real_libcudart;"
                    "do_not_synthesize_compat_launch_family_from_cudaLaunchKernel"
                ),
                "actual_counterpart_id": (
                    "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:2"
                ),
                "actual_counterpart_status": (
                    "actual_boundary_row_id_exported_selected_occurrence_join_not_attempted"
                ),
                "actual_counterpart_unavailable_reason": (
                    "requires_offline_emulated_selected_occurrence_join_not_available_during_capture"
                ),
                "actual_rank": 0,
                "actual_raw_prev_event_id": "rank:0:raw_ordinal:0",
                "actual_raw_current_event_id": "rank:0:raw_ordinal:2",
                "actual_boundary_family": "__cudaPopCallConfiguration -> cudaLaunchKernel",
                "counterpart_join_key": (
                    "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:2"
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
                "affected_interval_id": (
                    "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:2:endpoint_gap"
                ),
                "interval_kind": "actual_endpoint_gap_context_only",
                "start_ts_us": 104,
                "end_ts_us": 150,
                "duration_us": 5.0,
                "count_once_status": "unavailable",
                "paper_valid_window_membership": {
                    "window_id": "rank0:step_window",
                    "in_paper_valid_window": True,
                    "window_source": "manifest",
                    "start_ts": 90,
                    "end_ts": 200,
                    "is_paper_valid_step_window": True,
                    "membership_basis": "collate_step_window",
                    "unavailable_reason": None,
                },
            })
        events = (
            AnnotatedEvent(
                id="r0:e0",
                rank=0,
                ordinal=0,
                source=TraceSource.UNKNOWN,
                ts=0,
                pid=0,
                tid=0,
                module="libcudart.so",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={**host_extras, "stream_id": "stream0"},
                duration_us=2.0,
                duration_source="manual",
            ),
            AnnotatedEvent(
                id="r0:h1",
                rank=0,
                ordinal=1,
                source=TraceSource.UNKNOWN,
                ts=1,
                pid=0,
                tid=0,
                module="host.dispatch",
                api="__hostDelay__",
                op_type="host_delay",
                extras=host_delay_extras,
                duration_us=5.0,
                duration_source="manual",
                prev_event_id="r0:e0",
            ),
            AnnotatedEvent(
                id="r0:e2",
                rank=0,
                ordinal=2,
                source=TraceSource.UNKNOWN,
                ts=2,
                pid=0,
                tid=0,
                module="libcudart.so",
                api="cudaLaunchKernel",
                op_type="kernel_launch",
                extras={**host_extras, "stream_id": "stream0"},
                duration_us=3.0,
                duration_source="manual",
                prev_event_id="r0:h1",
            ),
        )
        return AnnotatedTrace(
            trace_dir=Path("/tmp/manual"),
            source=TraceSource.UNKNOWN,
            rank_events={0: events},
            global_events=events,
            collective_groups={},
        )

    baseline = replay_annotated_trace(make_trace(False))
    diagnostic = replay_annotated_trace(make_trace(True))

    assert diagnostic.success
    assert diagnostic.total_time_us == baseline.total_time_us
    assert diagnostic.critical_path_us == baseline.critical_path_us
    assert [event.start_us for event in diagnostic.simulated_events] == [
        event.start_us for event in baseline.simulated_events
    ]

    baseline_export = export_replay_edge_diagnostics(make_trace(False), baseline)
    baseline_hostdelay_row = [
        row for row in baseline_export["simulated_events"]
        if row["materialized_event_id"] == "r0:h1"
    ][0]
    assert not (
        _HOST_CONTROL_DEFAULT_OFF_DIAGNOSTIC_FIELDS
        & baseline_hostdelay_row["link_fields"].keys()
    )

    monkeypatch.setenv(
        "MAYA_ENABLE_GENERIC_REPLAY_PLACEMENT_ENVELOPE_PHASE1_METADATA_DIAGNOSTICS",
        "1",
    )
    export = export_replay_edge_diagnostics(make_trace(True), diagnostic)
    hostdelay_row = [
        row for row in export["simulated_events"]
        if row["materialized_event_id"] == "r0:h1"
    ][0]
    metadata = hostdelay_row["link_fields"]
    generic_rows = {
        row["materialized_event_id"]: row
        for row in export["generic_replay_placement_envelope_phase1_metadata"]["rows"]
        if row["component_row_type"] == "interval"
    }
    generic_hostdelay = generic_rows["r0:h1"]

    assert metadata["hostdelay_event_id"] == "r0:h1"
    assert metadata["rank"] == 0
    assert metadata["ordinal"] == 1
    assert metadata["source"] == "collate_host_gap"
    assert metadata["hostdelay_source"] == "collate_host_gap"
    assert metadata["trace_source"] == "unknown"
    assert metadata["observed_gap_us"] == 5.0
    assert metadata["raw_prev_event_id"] == "r0:e0"
    assert metadata["raw_prev_api"] == "__cudaPopCallConfiguration"
    assert metadata["raw_prev_ts_us"] == 100
    assert metadata["raw_prev_end_ts_us"] == 104
    assert metadata["raw_prev_end_ts_source"] == "raw_end_ts"
    assert metadata["raw_prev_host_duration_us"] == 4.0
    assert metadata["raw_current_event_id"] == "r0:e2"
    assert metadata["raw_current_api"] == "cudaLaunchKernel"
    assert metadata["raw_current_ts_us"] == 150
    assert metadata["raw_current_end_ts_us"] == 160
    assert metadata["raw_current_end_ts_source"] == "computed_from_host_duration_us"
    assert metadata["raw_current_host_duration_us"] == 10.0
    assert metadata["previous_materialized_event_id"] == "r0:e0"
    assert metadata["current_materialized_event_id"] == "r0:e2"
    assert metadata["host_dispatch_queue_id"] == "host0:rank:0"
    assert metadata["host_timing_dispatch_scope"] == "host_machine"
    assert metadata["paper_valid_window_membership"]["window_id"] == "rank0:step_window"
    assert metadata["boundary_origin_kind"] == "unavailable"
    assert metadata["boundary_visibility_kind"] == "unavailable"
    assert metadata["boundary_origin_field_sources"] == {}
    assert metadata["boundary_origin_conflicting_fields"] == {}
    assert metadata["wrapper_segment_coverage"] == "unavailable"
    assert metadata["boundary_visibility_segments"] == []
    assert metadata["actual_host_dispatch_duration_us"] is None
    assert metadata["host_control_boundary_counterpart_schema_version"] == (
        "host_control_boundary_visibility_unblocker_v2_row_evidence_v1"
    )
    assert metadata["host_control_visibility_schema_version"] == (
        "host_control_launch_neighborhood_visibility_counterpart_isolation_v1"
    )
    assert metadata["host_control_visibility_opt_in_flag"] is True
    assert metadata["host_control_envelope_counterpart_schema_version"] == (
        "host_control_replay_envelope_counterpart_metadata_v1"
    )
    assert metadata["host_control_envelope_counterpart_opt_in_flag"] is True
    assert metadata["host_control_envelope_counterpart_key"] == (
        "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:2"
    )
    assert metadata["hostdelay_counterpart_key"] == (
        "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:2"
    )
    assert metadata["host_control_envelope_replay_interval_id"] == "r0:h1"
    assert metadata["host_control_envelope_count_once_interval_id"] == "r0:h1"
    assert metadata["host_control_envelope_replay_resource_kind"] == "host"
    assert metadata["host_control_envelope_replay_resource_id"] == "host0:rank:0"
    assert metadata["host_control_envelope_replay_predecessor_successor_status"] == (
        "available_in_replay_edge_export"
    )
    assert metadata["selected_occurrence_id"] == (
        "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:2"
    )
    assert metadata["host_control_boundary_occurrence_id"] == (
        "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:2"
    )
    assert metadata["host_control_boundary_selection_status"] == "selected_family"
    assert metadata["host_control_visibility_split_status"] == "unavailable"
    assert metadata["host_control_producer_visibility_schema_version"] == (
        "host_control_producer_visibility_nonoverlap_v1"
    )
    assert metadata["host_control_producer_visibility_status"] == "structural_unavailable"
    assert metadata["host_control_producer_visibility_segments"][0]["duration_us"] is None
    assert metadata["host_control_producer_numeric_split_status"] == "unavailable"
    assert metadata["host_control_producer_nonoverlap_status"] == "unavailable"
    assert metadata["host_control_producer_wait_map_nonoverlap_status"] == "unavailable"
    assert metadata["host_control_producer_double_counting_nonoverlap_status"] == (
        "unavailable"
    )
    assert metadata["host_control_compat_launch_pop_coverage_status"] == (
        "unavailable_not_exported_by_current_real_wrapper_producer"
    )
    assert metadata["actual_counterpart_id"] == (
        "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:2"
    )
    assert metadata["actual_counterpart_status"] == (
        "actual_boundary_row_id_exported_selected_occurrence_join_not_attempted"
    )
    assert metadata["actual_rank"] == 0
    assert metadata["actual_raw_prev_event_id"] == "rank:0:raw_ordinal:0"
    assert metadata["actual_raw_current_event_id"] == "rank:0:raw_ordinal:2"
    assert metadata["actual_boundary_family"] == (
        "__cudaPopCallConfiguration -> cudaLaunchKernel"
    )
    assert metadata["counterpart_join_key"] == (
        "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:2"
    )
    assert metadata["counterpart_join_method"] == (
        "actual_row_id_only_emulated_selected_occurrence_join_not_attempted"
    )
    assert metadata["counterpart_join_confidence"] == "unavailable"
    assert metadata["comparable_actual_context_only"] is True
    assert metadata["split_sum_check_status"] == "unavailable"
    assert metadata["affected_interval_id"] == (
        "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:2:endpoint_gap"
    )
    assert metadata["interval_kind"] == "actual_endpoint_gap_context_only"
    assert metadata["start_ts_us"] == 104
    assert metadata["end_ts_us"] == 150
    assert metadata["duration_us"] == 5.0
    assert metadata["count_once_status"] == "unavailable"
    assert metadata["exact_counterpart_status"] == "unavailable"
    assert metadata["exact_counterpart_unavailable_reason"] == (
        "offline_exact_counterpart_join_not_run_by_replay_export"
    )
    assert metadata["wait_map_safety_status"] == "unavailable"
    assert metadata["double_counting_overlap_status"] == "unavailable"
    assert metadata["safe_to_use_as_repair_evidence"] is False
    assert metadata["safe_to_use_as_subtraction_delta"] is False
    assert hostdelay_row["host_queue_position"] == 0
    assert hostdelay_row["host_resource_id"] == "host0:rank:0"
    assert generic_hostdelay["component_kind"] == "host_control_interval"
    assert generic_hostdelay["count_once_interval_id"] == "r0:h1"
    assert generic_hostdelay["actual_timing_status"] == "unavailable"
    assert generic_hostdelay["stream_namespace_alignment_status"] == (
        "not_applicable_no_predicted_stream_resource"
    )

    host_edges = [edge for edge in export["predecessor_edges"] if edge["edge_kind"] == "host_order"]
    assert host_edges[0]["successor_materialized_event_id"] == "r0:h1"
    assert host_edges[0]["successor_api"] == "__hostDelay__"
    assert host_edges[0]["host_queue_position"] == 0
    assert host_edges[0]["affected_interval_duration_us"] == 5.0
    assert "wait_start_us" in host_edges[0]
    assert host_edges[0]["wait_start_us"] is None


def test_replay_edge_export_preserves_gemm_adjacent_metadata_without_timing_change():
    def make_trace(with_metadata: bool) -> AnnotatedTrace:
        host_extras = {
            "host_machine_id": "host0",
            "host_dispatch_queue_id": "host0:rank:0",
            "host_timing_dispatch_scope": "host_machine",
            "host_dispatch_model": "single_dispatch_queue_per_host_execution_context",
        }
        host_delay_extras = {
            **host_extras,
            "observed_gap_us": 7.0,
            "raw_prev_event_id": "r0:e0",
            "raw_prev_api": "cublasSetStream_v2",
            "raw_current_event_id": "r0:e2",
            "raw_current_api": "cublasGemmEx",
        }
        if with_metadata:
            host_delay_extras.update(
                {
                    "gemm_adjacent_hostdelay_schema_version": (
                        "gemm_adjacent_hostdelay_boundary_counterpart_visibility_count_once_metadata_v1"
                    ),
                    "gemm_adjacent_hostdelay_opt_in_flag": True,
                    "gemm_adjacent_source_side": "predicted_materialized_hostdelay_boundary",
                    "gemm_adjacent_stable_boundary_row_id": (
                        "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:1"
                    ),
                    "gemm_adjacent_rank": 0,
                    "gemm_adjacent_materialized_hostdelay_event_id": "r0:h1",
                    "gemm_adjacent_hostdelay_event_id": "r0:h1",
                    "gemm_adjacent_hostdelay_source": "collate_host_gap",
                    "gemm_adjacent_observed_gap_us": 7.0,
                    "gemm_adjacent_hostdelay_duration_us": 7.0,
                    "gemm_adjacent_host_dispatch_queue_id": "host0:rank:0",
                    "gemm_adjacent_host_machine_id": "host0",
                    "gemm_adjacent_boundary_direction": "raw_or_materialized_adjacent_boundary",
                    "gemm_adjacent_target_gemm_api": "cublasGemmEx",
                    "gemm_adjacent_adjacent_api": "cublasSetStream_v2",
                    "gemm_adjacent_boundary_family_in_design_scope": True,
                    "gemm_adjacent_current_stream_id": "stream0",
                    "gemm_adjacent_current_stream_resource_id": "rank:0:stream:stream0",
                    "gemm_adjacent_stream_namespace_basis": "predicted_collate_raw_stream_id_rank_local",
                    "gemm_adjacent_stream_namespace_alignment_status": (
                        "predicted_only_actual_alignment_unavailable"
                    ),
                    "gemm_adjacent_algorithm": "23",
                    "gemm_adjacent_gemm_shape_signature": "m=256|n=128|k=64|algorithm=23",
                    "gemm_adjacent_gemm_material_metadata_status": "available",
                    "gemm_adjacent_actual_counterpart_join_status": (
                        "predicted_metadata_only_actual_join_deferred"
                    ),
                    "gemm_adjacent_actual_counterpart_join_confidence": "unavailable",
                    "gemm_adjacent_actual_timing_status": "unavailable",
                    "gemm_adjacent_actual_endpoint_timestamps_used_as_strict_timing": False,
                    "gemm_adjacent_actual_runtime_direct_substitution": False,
                    "gemm_adjacent_producer_visibility_schema_version": (
                        "gemm_adjacent_hostdelay_producer_visibility_v1"
                    ),
                    "gemm_adjacent_producer_visibility_status": "structural_unavailable",
                    "gemm_adjacent_predicted_count_once_group_id": (
                        "predicted_hostdelay_boundary:rank:0:host_control_boundary:"
                        "raw_ordinal:0->raw_ordinal:1"
                    ),
                    "gemm_adjacent_count_once_status": "unavailable",
                    "gemm_adjacent_count_once_non_overlap_status": "unavailable",
                    "gemm_adjacent_double_counting_overlap_status": "unavailable",
                    "gemm_adjacent_wait_map_safety_status": "unavailable",
                    "gemm_adjacent_repair_ready": False,
                    "gemm_adjacent_safe_to_use_as_repair_evidence": False,
                    "gemm_adjacent_safe_to_use_as_subtraction_delta": False,
                }
            )
        events = (
            AnnotatedEvent(
                id="r0:e0",
                rank=0,
                ordinal=0,
                source=TraceSource.UNKNOWN,
                ts=0,
                pid=0,
                tid=0,
                module="libcublas.so",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={**host_extras, "stream_id": "stream0"},
                duration_us=2.0,
                duration_source="manual",
            ),
            AnnotatedEvent(
                id="r0:h1",
                rank=0,
                ordinal=1,
                source=TraceSource.UNKNOWN,
                ts=1,
                pid=0,
                tid=0,
                module="host.dispatch",
                api="__hostDelay__",
                op_type="host_delay",
                extras=host_delay_extras,
                duration_us=7.0,
                duration_source="manual",
                prev_event_id="r0:e0",
            ),
            AnnotatedEvent(
                id="r0:e2",
                rank=0,
                ordinal=2,
                source=TraceSource.UNKNOWN,
                ts=2,
                pid=0,
                tid=0,
                module="libcublas.so",
                api="cublasGemmEx",
                op_type="blas_compute",
                extras={**host_extras, "stream_id": "stream0"},
                duration_us=11.0,
                duration_source="manual",
                prev_event_id="r0:h1",
            ),
        )
        return AnnotatedTrace(
            trace_dir=Path("/tmp/manual"),
            source=TraceSource.UNKNOWN,
            rank_events={0: events},
            global_events=events,
            collective_groups={},
        )

    baseline = replay_annotated_trace(make_trace(False))
    diagnostic = replay_annotated_trace(make_trace(True))

    assert diagnostic.success
    assert diagnostic.total_time_us == baseline.total_time_us
    assert diagnostic.critical_path_us == baseline.critical_path_us
    assert [event.start_us for event in diagnostic.simulated_events] == [
        event.start_us for event in baseline.simulated_events
    ]

    baseline_export = export_replay_edge_diagnostics(make_trace(False), baseline)
    baseline_hostdelay = [
        row for row in baseline_export["simulated_events"]
        if row["materialized_event_id"] == "r0:h1"
    ][0]
    assert not any(
        key.startswith("gemm_adjacent_")
        for key in baseline_hostdelay["link_fields"]
    )

    export = export_replay_edge_diagnostics(make_trace(True), diagnostic)
    hostdelay = [
        row for row in export["simulated_events"]
        if row["materialized_event_id"] == "r0:h1"
    ][0]
    metadata = hostdelay["link_fields"]

    assert metadata["gemm_adjacent_hostdelay_opt_in_flag"] is True
    assert metadata["gemm_adjacent_target_gemm_api"] == "cublasGemmEx"
    assert metadata["gemm_adjacent_adjacent_api"] == "cublasSetStream_v2"
    assert metadata["gemm_adjacent_predicted_replay_interval_id"] == "r0:h1"
    assert metadata["gemm_adjacent_predicted_replay_component_kind"] == (
        "host_control_interval"
    )
    assert metadata["gemm_adjacent_predicted_replay_resource_kind"] == "host"
    assert metadata["gemm_adjacent_predicted_duration_us"] == 7.0
    assert metadata["gemm_adjacent_predicted_count_once_interval_id"] == "r0:h1"
    assert metadata["gemm_adjacent_actual_timing_status"] == "unavailable"
    assert metadata["gemm_adjacent_actual_endpoint_timestamps_used_as_strict_timing"] is False
    assert metadata["gemm_adjacent_actual_runtime_direct_substitution"] is False
    assert metadata["gemm_adjacent_count_once_status"] == "unavailable"
    assert metadata["gemm_adjacent_count_once_non_overlap_status"] == "unavailable"
    assert metadata["gemm_adjacent_double_counting_overlap_status"] == "unavailable"
    assert metadata["gemm_adjacent_wait_map_safety_status"] == "unavailable"
    assert metadata["gemm_adjacent_repair_ready"] is False
    assert metadata["gemm_adjacent_safe_to_use_as_repair_evidence"] is False
    assert metadata["gemm_adjacent_safe_to_use_as_subtraction_delta"] is False


def test_replay_edge_export_opt_in_hostdelay_semantic_classification_metadata_without_timing_change(
    monkeypatch,
):
    env_keys = (
        "MAYA_ENABLE_CRITICAL_PATH_HOSTDELAY_BOUNDARY_SEMANTIC_CLASSIFICATION_METADATA_DIAGNOSTICS",
        "FLEXSIM_MAYA_ENABLE_CRITICAL_PATH_HOSTDELAY_BOUNDARY_SEMANTIC_CLASSIFICATION_METADATA_DIAGNOSTICS",
    )

    def make_trace(
        *,
        query_boundary: bool = False,
        stale_extras: bool = False,
        stale_non_hostdelay_extras: bool = False,
        extra_hostdelay_extras: dict[str, object] | None = None,
    ) -> AnnotatedTrace:
        host_extras = {
            "host_machine_id": "host0",
            "host_dispatch_queue_id": "host0:rank:0",
            "host_timing_dispatch_scope": "host_machine",
            "host_dispatch_model": "single_dispatch_queue_per_host_execution_context",
        }
        prev_api = "cudaEventQuery" if query_boundary else "cublasSetStream_v2"
        target_api = "cudaLaunchKernel" if query_boundary else "cublasGemmEx"
        target_op_type = "kernel_launch" if query_boundary else "blas_compute"
        host_delay_extras = {
            **host_extras,
            "observed_gap_us": 7.0,
            "raw_prev_event_id": "r0:e0",
            "raw_prev_api": prev_api,
            "raw_current_event_id": "r0:e2",
            "raw_current_api": target_api,
            "previous_materialized_event_id": "r0:e0",
            "previous_materialized_api": prev_api,
            "current_materialized_event_id": "r0:e2",
            "current_materialized_api": target_api,
            "materialized_boundary_family": f"{prev_api} -> {target_api}",
            "selected_occurrence_id": "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:2",
            "predicted_count_once_group_id": "predicted_hostDelay_boundary:r0:h1",
        }
        if stale_extras:
            host_delay_extras.update({
                "hostdelay_semantic_classification_schema_version": "stale",
                "hostdelay_semantic_classification_opt_in_flag": True,
                "hostdelay_origin_semantic_class": "instrumentation_only",
                "hostdelay_safe_to_use_as_repair_evidence": True,
            })
        if extra_hostdelay_extras:
            host_delay_extras.update(extra_hostdelay_extras)
        prev_extras = {**host_extras, "stream_id": "stream0"}
        if stale_non_hostdelay_extras:
            prev_extras.update({
                "hostdelay_semantic_classification_schema_version": "stale_non_target",
                "hostdelay_semantic_classification_opt_in_flag": True,
                "hostdelay_origin_semantic_class": "instrumentation_only",
                "hostdelay_safe_to_use_as_repair_evidence": True,
            })
        events = (
            AnnotatedEvent(
                id="r0:e0",
                rank=0,
                ordinal=0,
                source=TraceSource.UNKNOWN,
                ts=0,
                pid=0,
                tid=0,
                module="libcudart.so",
                api=prev_api,
                op_type="stream_op",
                extras=prev_extras,
                duration_us=2.0,
                duration_source="manual",
            ),
            AnnotatedEvent(
                id="r0:h1",
                rank=0,
                ordinal=1,
                source=TraceSource.UNKNOWN,
                ts=1,
                pid=0,
                tid=0,
                module="host.dispatch",
                api="__hostDelay__",
                op_type="host_delay",
                extras=host_delay_extras,
                duration_us=7.0,
                duration_source="manual",
                prev_event_id="r0:e0",
            ),
            AnnotatedEvent(
                id="r0:e2",
                rank=0,
                ordinal=2,
                source=TraceSource.UNKNOWN,
                ts=2,
                pid=0,
                tid=0,
                module="libcudart.so",
                api=target_api,
                op_type=target_op_type,
                extras={**host_extras, "stream_id": "stream0"},
                duration_us=11.0,
                duration_source="manual",
                prev_event_id="r0:h1",
            ),
        )
        return AnnotatedTrace(
            trace_dir=Path("/tmp/manual"),
            source=TraceSource.UNKNOWN,
            rank_events={0: events},
            global_events=events,
            collective_groups={},
        )

    for key in env_keys:
        monkeypatch.delenv(key, raising=False)
    baseline = replay_annotated_trace(make_trace(stale_extras=True))
    default_off_export = export_replay_edge_diagnostics(
        make_trace(stale_extras=True),
        baseline,
    )
    default_off_hostdelay = [
        row
        for row in default_off_export["simulated_events"]
        if row["materialized_event_id"] == "r0:h1"
    ][0]
    assert not (
        _HOSTDELAY_SEMANTIC_CLASSIFICATION_FIELDS
        & default_off_hostdelay["link_fields"].keys()
    )

    monkeypatch.setenv(env_keys[0], "1")
    diagnostic = replay_annotated_trace(make_trace())
    assert diagnostic.total_time_us == baseline.total_time_us
    assert diagnostic.critical_path_us == baseline.critical_path_us
    assert [event.start_us for event in diagnostic.simulated_events] == [
        event.start_us for event in baseline.simulated_events
    ]

    export = export_replay_edge_diagnostics(make_trace(), diagnostic)
    hostdelay = [
        row
        for row in export["simulated_events"]
        if row["materialized_event_id"] == "r0:h1"
    ][0]
    metadata = hostdelay["link_fields"]

    assert metadata["hostdelay_semantic_classification_schema_version"] == (
        "critical_path_hostDelay_boundary_semantic_classification_metadata_v1"
    )
    assert metadata["hostdelay_semantic_classification_opt_in_flag"] is True
    assert metadata["hostdelay_boundary_semantic_source_side"] == (
        "predicted_hostDelay_boundary_metadata"
    )
    assert metadata["hostdelay_boundary_family"] == "cublasSetStream_v2 -> cublasGemmEx"
    assert metadata["hostdelay_boundary_target_api"] == "cublasGemmEx"
    assert metadata["hostdelay_boundary_rank_stream_queue_key"] == (
        "rank:0|queue:host0:rank:0|stream:unavailable"
    )
    assert metadata["hostdelay_origin_semantic_class"] == "paper_visible_host_overhead"
    assert metadata["hostdelay_origin_semantic_basis"] == "figure6_inter_api_host_overhead"
    assert metadata["hostdelay_paper_visible_status"] == "paper_visible_by_default"
    assert metadata["hostdelay_classification_confidence"] == "metadata_only_context"
    assert metadata["hostdelay_paper_visible_host_overhead_us"] is None
    assert metadata["hostdelay_instrumentation_only_us"] is None
    assert metadata["hostdelay_already_counted_host_dispatch_us"] is None
    assert metadata["hostdelay_already_counted_provider_runtime_us"] is None
    assert metadata["hostdelay_unresolved_mixed_us"] == 7.0
    assert metadata["hostdelay_same_occurrence_key"] == (
        "rank:0:host_control_boundary:raw_ordinal:0->raw_ordinal:2"
    )
    assert metadata["hostdelay_same_occurrence_key_status"] == (
        "available_metadata_only_not_strict_counterpart"
    )
    assert metadata["hostdelay_predicted_count_once_group_id"] == (
        "predicted_hostDelay_boundary:r0:h1"
    )
    assert metadata["hostdelay_actual_count_once_group_id"] is None
    assert metadata["hostdelay_count_once_status"] == "unavailable"
    assert metadata["hostdelay_nonoverlap_status"] == "unavailable"
    assert metadata["hostdelay_wait_map_safety_status"] == "unavailable"
    assert metadata["hostdelay_host_dispatch_overlap_status"] == "unavailable"
    assert metadata["hostdelay_provider_runtime_overlap_status"] == "unavailable"
    assert metadata["hostdelay_stream_queue_wait_overlap_status"] == "unavailable"
    assert metadata["hostdelay_endpoint_duration_used_as_subtraction_delta"] is False
    assert metadata["hostdelay_runtime_used_as_substitution"] is False
    assert metadata["hostdelay_projection_overlap_used_as_subtraction_delta"] is False
    assert metadata["hostdelay_safe_to_use_as_repair_evidence"] is False
    assert metadata["hostdelay_safe_to_use_as_subtraction_delta"] is False
    assert metadata["hostdelay_repair_ready"] is False

    query_export = export_replay_edge_diagnostics(
        make_trace(query_boundary=True),
        replay_annotated_trace(make_trace(query_boundary=True)),
    )
    query_hostdelay = [
        row
        for row in query_export["simulated_events"]
        if row["materialized_event_id"] == "r0:h1"
    ][0]
    query_metadata = query_hostdelay["link_fields"]
    assert query_metadata["hostdelay_origin_semantic_class"] == (
        "control_plane_or_polling_unresolved"
    )
    assert query_metadata["hostdelay_paper_visible_status"] == "unresolved"

    instrumentation_export = export_replay_edge_diagnostics(
        make_trace(
            extra_hostdelay_extras={
                "boundary_visibility_kind": "instrumentation_only",
                "instrumentation_only_duration_us": 3.0,
            }
        ),
        replay_annotated_trace(
            make_trace(
                extra_hostdelay_extras={
                    "boundary_visibility_kind": "instrumentation_only",
                    "instrumentation_only_duration_us": 3.0,
                }
            )
        ),
    )
    instrumentation_metadata = [
        row
        for row in instrumentation_export["simulated_events"]
        if row["materialized_event_id"] == "r0:h1"
    ][0]["link_fields"]
    assert instrumentation_metadata["hostdelay_origin_semantic_class"] == (
        "instrumentation_only"
    )
    assert instrumentation_metadata["hostdelay_instrumentation_only_us"] == 3.0
    assert instrumentation_metadata["hostdelay_unresolved_mixed_us"] is None

    already_counted_export = export_replay_edge_diagnostics(
        make_trace(
            extra_hostdelay_extras={
                "host_dispatch_overlap_status": "proven_counted_once",
                "count_once_status": "proven",
            }
        ),
        replay_annotated_trace(
            make_trace(
                extra_hostdelay_extras={
                    "host_dispatch_overlap_status": "proven_counted_once",
                    "count_once_status": "proven",
                }
            )
        ),
    )
    already_counted_metadata = [
        row
        for row in already_counted_export["simulated_events"]
        if row["materialized_event_id"] == "r0:h1"
    ][0]["link_fields"]
    assert already_counted_metadata["hostdelay_origin_semantic_class"] == (
        "already_counted_host_dispatch"
    )
    assert already_counted_metadata["hostdelay_host_dispatch_overlap_status"] == (
        "proven_counted_once"
    )
    assert already_counted_metadata["hostdelay_count_once_status"] == "proven"
    assert already_counted_metadata["hostdelay_unresolved_mixed_us"] is None

    stale_non_target_export = export_replay_edge_diagnostics(
        make_trace(stale_non_hostdelay_extras=True),
        replay_annotated_trace(make_trace(stale_non_hostdelay_extras=True)),
    )
    non_hostdelay_metadata = [
        row
        for row in stale_non_target_export["simulated_events"]
        if row["materialized_event_id"] == "r0:e0"
    ][0]["link_fields"]
    assert not (
        _HOSTDELAY_SEMANTIC_CLASSIFICATION_FIELDS
        & non_hostdelay_metadata.keys()
    )


def test_replay_edge_export_preserves_cuda_gemm_hostdispatch_strict_occurrence_gap_metadata_without_timing_change():
    def make_trace(with_metadata: bool) -> AnnotatedTrace:
        host_extras = {
            "host_machine_id": "host0",
            "host_dispatch_queue_id": "host0:rank:0",
            "host_timing_dispatch_scope": "host_machine",
            "host_dispatch_model": "single_dispatch_queue_per_host_execution_context",
        }
        host_delay_extras = {
            **host_extras,
            "observed_gap_us": 7.0,
            "raw_prev_event_id": "r0:e0",
            "raw_prev_api": "cublasSetStream_v2",
            "raw_current_event_id": "r0:e2",
            "raw_current_api": "cublasGemmEx",
        }
        if with_metadata:
            host_delay_extras.update(
                {
                    "cuda_gemm_hostdispatch_strict_occurrence_gap_schema_version": (
                        "cudaLaunch_GEMM_hostdispatch_strict_occurrence_gap_metadata_v1"
                    ),
                    "cuda_gemm_hostdispatch_strict_occurrence_gap_opt_in_flag": True,
                    "cuda_gemm_hostdispatch_strict_occurrence_gap_source_side": (
                        "predicted_hostDelay_boundary_metadata"
                    ),
                    "cuda_gemm_hostdispatch_strict_occurrence_gap_predicted_row_id": (
                        "rank:0:strict_occurrence_gap_hostDelay:r0:e0->r0:e2"
                    ),
                    "strict_occurrence_common_basis_key": (
                        "rank:0|queue:host0:rank:0|api:cublasGemmEx|api_seq:0|"
                        "host_queue_seq:1|stream_seq:1|"
                        "material:m=256|n=128|k=64|algorithm=23|algorithm:23|"
                        "boundary:cublasSetStream_v2 -> cublasGemmEx"
                    ),
                    "strict_occurrence_material_without_embedded_algo": (
                        "m=256|n=128|k=64|algorithm=23"
                    ),
                    "strict_occurrence_boundary_target_side": "incoming_to_target",
                    "strict_occurrence_endpoint_identity_projection_key": (
                        "rank:0|queue:host0:rank:0|api:cublasGemmEx|api_seq:0|"
                        "material:m=256|n=128|k=64|algorithm=23|algorithm:23"
                    ),
                    "strict_occurrence_boundary_target_side_projection_key": (
                        "rank:0|queue:host0:rank:0|api:cublasGemmEx|api_seq:0|"
                        "material:m=256|n=128|k=64|algorithm=23|algorithm:23|"
                        "boundary_target_side:incoming_to_target"
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
                        "rank": 0,
                        "paper_valid_window_id": "rank0:step_window",
                        "host_dispatch_queue_id": "host0:rank:0",
                        "api_family": "cublasGemmEx",
                        "component_role": "hostDelay",
                        "api_sequence_ordinal_in_window": 0,
                        "host_queue_sequence_ordinal_in_window": 1,
                        "stream_sequence_ordinal_in_window": 1,
                        "material_signature": "m=256|n=128|k=64|algorithm=23",
                        "algorithm": "23",
                        "gemm_shape_signature": "m=256|n=128|k=64|algorithm=23",
                        "boundary_family": "cublasSetStream_v2 -> cublasGemmEx",
                    },
                    "strict_occurrence_count_basis_side": "predicted_hostDelay_boundary",
                    "paper_valid_window_id": "rank0:step_window",
                    "rank": 0,
                    "api_family": "cublasGemmEx",
                    "component_role": "hostDelay",
                    "api_sequence_ordinal_in_window": 0,
                    "host_queue_sequence_ordinal_in_window": 1,
                    "stream_sequence_ordinal_in_window": 1,
                    "material_signature": "m=256|n=128|k=64|algorithm=23",
                    "algorithm": "23",
                    "gemm_shape_signature": "m=256|n=128|k=64|algorithm=23",
                    "boundary_family": "cublasSetStream_v2 -> cublasGemmEx",
                    "key_completeness_status": (
                        "predicted_hostDelay_key_parts_available_actual_join_deferred"
                    ),
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
                    "predicted_stream_resource_id": "rank:0:stream:stream0",
                    "actual_stream_resource_id": None,
                    "stream_namespace_basis": "rank_scoped_predicted_raw_stream_id",
                    "stream_alignment_status": (
                        "predicted_only_actual_alignment_unavailable"
                    ),
                    "exact_stream_identity_proven": False,
                    "default_stream_equivalence_reviewed": False,
                    "predicted_count_once_group_id": (
                        "predicted_hostDelay_boundary:rank:0:"
                        "strict_occurrence_gap_hostDelay:r0:e0->r0:e2"
                    ),
                    "actual_count_once_group_id": None,
                    "count_once_status": (
                        "metadata_only_count_once_group_not_strict_nonoverlap_proof"
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
            )
        events = (
            AnnotatedEvent(
                id="r0:e0",
                rank=0,
                ordinal=0,
                source=TraceSource.UNKNOWN,
                ts=0,
                pid=0,
                tid=0,
                module="libcublas.so",
                api="cublasSetStream_v2",
                op_type="stream_op",
                extras={**host_extras, "stream_id": "stream0"},
                duration_us=2.0,
                duration_source="manual",
            ),
            AnnotatedEvent(
                id="r0:h1",
                rank=0,
                ordinal=1,
                source=TraceSource.UNKNOWN,
                ts=1,
                pid=0,
                tid=0,
                module="host.dispatch",
                api="__hostDelay__",
                op_type="host_delay",
                extras=host_delay_extras,
                duration_us=7.0,
                duration_source="manual",
                prev_event_id="r0:e0",
            ),
            AnnotatedEvent(
                id="r0:e2",
                rank=0,
                ordinal=2,
                source=TraceSource.UNKNOWN,
                ts=2,
                pid=0,
                tid=0,
                module="libcublas.so",
                api="cublasGemmEx",
                op_type="blas_compute",
                extras={
                    **host_extras,
                    "stream_id": "stream0",
                    "m": "256",
                    "n": "128",
                    "k": "64",
                    "algorithm": "23",
                },
                duration_us=11.0,
                duration_source="manual",
                prev_event_id="r0:h1",
            ),
        )
        return AnnotatedTrace(
            trace_dir=Path("/tmp/manual"),
            source=TraceSource.UNKNOWN,
            rank_events={0: events},
            global_events=events,
            collective_groups={},
        )

    baseline = replay_annotated_trace(make_trace(False))
    diagnostic = replay_annotated_trace(make_trace(True))

    assert diagnostic.success
    assert diagnostic.total_time_us == baseline.total_time_us
    assert diagnostic.critical_path_us == baseline.critical_path_us
    assert [event.start_us for event in diagnostic.simulated_events] == [
        event.start_us for event in baseline.simulated_events
    ]

    baseline_export = export_replay_edge_diagnostics(make_trace(False), baseline)
    baseline_hostdelay = [
        row
        for row in baseline_export["simulated_events"]
        if row["materialized_event_id"] == "r0:h1"
    ][0]
    assert "cuda_gemm_hostdispatch_strict_occurrence_gap_schema_version" not in (
        baseline_hostdelay["link_fields"]
    )
    assert "strict_occurrence_common_basis_key" not in baseline_hostdelay["link_fields"]

    export = export_replay_edge_diagnostics(make_trace(True), diagnostic)
    hostdelay = [
        row
        for row in export["simulated_events"]
        if row["materialized_event_id"] == "r0:h1"
    ][0]
    metadata = hostdelay["link_fields"]

    assert metadata["cuda_gemm_hostdispatch_strict_occurrence_gap_schema_version"] == (
        "cudaLaunch_GEMM_hostdispatch_strict_occurrence_gap_metadata_v1"
    )
    assert metadata["cuda_gemm_hostdispatch_strict_occurrence_gap_opt_in_flag"] is True
    assert metadata["cuda_gemm_hostdispatch_strict_occurrence_gap_source_side"] == (
        "predicted_hostDelay_boundary_metadata"
    )
    assert metadata["cuda_gemm_hostdispatch_strict_occurrence_gap_predicted_row_id"] == (
        "rank:0:strict_occurrence_gap_hostDelay:r0:e0->r0:e2"
    )
    assert metadata["strict_occurrence_count_basis_side"] == (
        "predicted_hostDelay_boundary"
    )
    common_key = metadata["strict_occurrence_common_basis_key"]
    assert "window:" not in common_key
    assert "role:" not in common_key
    assert "api:cublasGemmEx" in common_key
    assert "host_queue_seq:1" in common_key
    assert metadata["strict_occurrence_boundary_target_side"] == "incoming_to_target"
    assert metadata["strict_occurrence_projection_keys_status"] == (
        "diagnostic_only_projection_not_strict_join_key"
    )
    assert metadata["strict_occurrence_projection_keys_repair_ready"] is False
    assert metadata["strict_occurrence_endpoint_identity_projection_key"] == (
        "rank:0|queue:host0:rank:0|api:cublasGemmEx|api_seq:0|"
        "material:m=256|n=128|k=64|algorithm=23|algorithm:23"
    )
    assert metadata["strict_occurrence_boundary_target_side_projection_key"] == (
        "rank:0|queue:host0:rank:0|api:cublasGemmEx|api_seq:0|"
        "material:m=256|n=128|k=64|algorithm=23|algorithm:23|"
        "boundary_target_side:incoming_to_target"
    )
    assert metadata["strict_occurrence_key_parts"]["paper_valid_window_id"] == (
        "rank0:step_window"
    )
    assert metadata["strict_occurrence_key_parts"]["component_role"] == "hostDelay"
    assert metadata["actual_mechanical_dispatch_split_status"] == "unavailable"
    assert metadata["actual_control_dispatch_us"] is None
    assert metadata["actual_api_body_us"] is None
    assert metadata["actual_instrumentation_only_us"] is None
    assert metadata["actual_endpoint_timestamps_used_as_dispatch_split"] is False
    assert metadata["actual_host_duration_used_as_dispatch_split"] is False
    assert metadata["actual_runtime_used_as_dispatch_split"] is False
    assert metadata["stream_alignment_status"] == (
        "predicted_only_actual_alignment_unavailable"
    )
    assert metadata["count_once_status"] == (
        "metadata_only_count_once_group_not_strict_nonoverlap_proof"
    )
    assert metadata["nonoverlap_status"] == "unavailable"
    assert metadata["wait_map_safety_status"] == "unavailable"
    assert metadata["hostdispatch_producer_visibility_status"] == "unavailable"
    assert metadata["strict_occurrence_join_ready"] is False
    assert metadata["strict_actual_timing_or_mechanical_split_ready"] is False
    assert metadata["strict_apples_to_apples_delta_ready"] is False
    assert metadata["repair_ready"] is False
    assert metadata["safe_to_use_as_repair_evidence"] is False
    assert metadata["safe_to_use_as_subtraction_delta"] is False
    assert metadata["safe_to_use_for_runtime_substitution"] is False
    assert metadata["safe_to_use_for_endpoint_timestamp_substitution"] is False


def test_replay_collective_prefers_group_level_duration_metadata():
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:e0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=0,
                    module="libnccl.so",
                    api="ncclAllReduce",
                    op_type="nccl_collective",
                    extras={"collective_group_duration_us": 9.0},
                    duration_us=3.0,
                    duration_source="manual",
                    collective_group_id="ncclAllReduce#0",
                ),
            ),
            1: (
                AnnotatedEvent(
                    id="r1:e0",
                    rank=1,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=1,
                    tid=1,
                    module="libnccl.so",
                    api="ncclAllReduce",
                    op_type="nccl_collective",
                    extras={"collective_group_duration_us": 9.0},
                    duration_us=7.0,
                    duration_source="manual",
                    collective_group_id="ncclAllReduce#0",
                ),
            ),
        },
        global_events=(),
        collective_groups={
            "ncclAllReduce#0": CollectiveGroup(
                id="ncclAllReduce#0",
                api="ncclAllReduce",
                op_type="nccl_collective",
                ranks=(0, 1),
                event_ids=("r0:e0", "r1:e0"),
            )
        },
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0] + trace.rank_events[1],
        collective_groups=trace.collective_groups,
    )

    result = replay_annotated_trace(trace)

    assert result.success
    assert result.total_time_us == 9.0
    collective_events = [event for event in result.simulated_events if event.collective_group_id]
    assert len(collective_events) == 2
    assert {event.start_us for event in collective_events} == {0.0}
    assert {event.end_us for event in collective_events} == {9.0}


def test_replay_host_delay_uses_separate_host_clock():
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:e0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=0,
                    module="libcublas.so",
                    api="cublasSgemm_v2",
                    op_type="blas_compute",
                    duration_us=10.0,
                    duration_source="manual",
                ),
                AnnotatedEvent(
                    id="r0:h1",
                    rank=0,
                    ordinal=1,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=0,
                    tid=0,
                    module="host.dispatch",
                    api="__hostDelay__",
                    op_type="host_delay",
                    duration_us=5.0,
                    duration_source="manual",
                    prev_event_id="r0:e0",
                ),
                AnnotatedEvent(
                    id="r0:e2",
                    rank=0,
                    ordinal=2,
                    source=TraceSource.UNKNOWN,
                    ts=2,
                    pid=0,
                    tid=0,
                    module="libcublas.so",
                    api="cublasSgemm_v2",
                    op_type="blas_compute",
                    duration_us=1.0,
                    duration_source="manual",
                    prev_event_id="r0:h1",
                ),
            ),
        },
        global_events=(),
        collective_groups={},
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0],
        collective_groups=trace.collective_groups,
    )

    result = replay_annotated_trace(trace)

    assert result.success
    assert result.total_time_us == 11.0
    assert [event.start_us for event in result.simulated_events] == [0.0, 0.0, 10.0]
    assert [event.end_us for event in result.simulated_events] == [10.0, 5.0, 11.0]
    assert [event.resource_kind for event in result.simulated_events] == ["stream", "host", "stream"]
    assert result.simulated_events[1].host_machine_id == "legacy_pid:0"
    assert result.simulated_events[1].host_tid is None
    assert result.simulated_events[0].stream_id == "__default_stream__"


def test_replay_host_machine_dispatch_queue_serializes_cross_process_host_ops():
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:h0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=10,
                    module="host.dispatch",
                    api="__hostDelay__",
                    op_type="host_delay",
                    duration_us=10.0,
                    duration_source="manual",
                    extras={
                        "host_machine_id": "host0",
                        "host_dispatch_queue_id": "host0:queue0",
                    },
                ),
            ),
            1: (
                AnnotatedEvent(
                    id="r1:h0",
                    rank=1,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=1,
                    tid=11,
                    module="host.dispatch",
                    api="__hostDelay__",
                    op_type="host_delay",
                    duration_us=7.0,
                    duration_source="manual",
                    extras={
                        "host_machine_id": "host0",
                        "host_dispatch_queue_id": "host0:queue0",
                    },
                ),
            ),
        },
        global_events=(),
        collective_groups={},
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0] + trace.rank_events[1],
        collective_groups=trace.collective_groups,
    )

    result = replay_annotated_trace(trace)

    assert result.success
    assert result.total_time_us == 17.0
    assert [event.start_us for event in result.simulated_events] == [0.0, 10.0]
    assert [event.end_us for event in result.simulated_events] == [10.0, 17.0]
    assert [event.host_machine_id for event in result.simulated_events] == ["host0", "host0"]
    assert [event.host_dispatch_queue_id for event in result.simulated_events] == [
        "host0:queue0",
        "host0:queue0",
    ]
    assert [event.host_tid for event in result.simulated_events] == [None, None]


def test_replay_separates_physical_host_topology_from_dispatch_queues():
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:h0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=10,
                    module="host.dispatch",
                    api="__hostDelay__",
                    op_type="host_delay",
                    duration_us=10.0,
                    duration_source="manual",
                    extras={
                        "host_machine_id": "host0",
                        "host_dispatch_queue_id": "host0:rank:0",
                    },
                ),
            ),
            1: (
                AnnotatedEvent(
                    id="r1:h0",
                    rank=1,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=1,
                    tid=11,
                    module="host.dispatch",
                    api="__hostDelay__",
                    op_type="host_delay",
                    duration_us=7.0,
                    duration_source="manual",
                    extras={
                        "host_machine_id": "host0",
                        "host_dispatch_queue_id": "host0:rank:1",
                    },
                ),
            ),
        },
        global_events=(),
        collective_groups={},
        rank_host_machines={0: "host0", 1: "host0"},
        rank_host_dispatch_queues={0: "host0:rank:0", 1: "host0:rank:1"},
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0] + trace.rank_events[1],
        collective_groups=trace.collective_groups,
        rank_host_machines=trace.rank_host_machines,
        rank_host_dispatch_queues=trace.rank_host_dispatch_queues,
    )

    result = replay_annotated_trace(trace)

    assert result.success
    assert result.total_time_us == 10.0
    assert [event.start_us for event in result.simulated_events] == [0.0, 0.0]
    assert [event.end_us for event in result.simulated_events] == [10.0, 7.0]
    assert [event.host_machine_id for event in result.simulated_events] == ["host0", "host0"]
    assert [event.host_dispatch_queue_id for event in result.simulated_events] == [
        "host0:rank:0",
        "host0:rank:1",
    ]


def test_replay_thread_dispatch_scope_keeps_host_lanes_independent():
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:h0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=10,
                    module="host.dispatch",
                    api="__hostDelay__",
                    op_type="host_delay",
                    duration_us=10.0,
                    duration_source="manual",
                    extras={"host_machine_id": "host0", "host_timing_dispatch_scope": "thread"},
                ),
            ),
            1: (
                AnnotatedEvent(
                    id="r1:h0",
                    rank=1,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=1,
                    tid=11,
                    module="host.dispatch",
                    api="__hostDelay__",
                    op_type="host_delay",
                    duration_us=7.0,
                    duration_source="manual",
                    extras={"host_machine_id": "host0", "host_timing_dispatch_scope": "thread"},
                ),
            ),
        },
        global_events=(),
        collective_groups={},
        host_timing_dispatch_scope_resolved="thread",
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0] + trace.rank_events[1],
        collective_groups=trace.collective_groups,
        host_timing_dispatch_scope_resolved=trace.host_timing_dispatch_scope_resolved,
    )

    result = replay_annotated_trace(trace)

    assert result.success
    assert result.total_time_us == 10.0
    assert [event.start_us for event in result.simulated_events] == [0.0, 0.0]
    assert [event.end_us for event in result.simulated_events] == [10.0, 7.0]
    assert [event.host_machine_id for event in result.simulated_events] == ["host0", "host0"]
    assert [event.host_tid for event in result.simulated_events] == [10, 11]


def test_replay_can_skip_simulated_event_recording_without_changing_metrics():
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:e0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    duration_us=5.0,
                    duration_source="manual",
                ),
                AnnotatedEvent(
                    id="r0:h1",
                    rank=0,
                    ordinal=1,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=0,
                    tid=0,
                    module="host.dispatch",
                    api="__hostDelay__",
                    op_type="host_delay",
                    duration_us=3.0,
                    duration_source="manual",
                    prev_event_id="r0:e0",
                ),
            ),
        },
        global_events=(),
        collective_groups={},
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0],
        collective_groups=trace.collective_groups,
    )

    default_result = replay_annotated_trace(trace)
    compact_result = replay_annotated_trace(trace, record_simulated_events=False)

    assert default_result.success
    assert compact_result.success
    assert default_result.total_time_us == compact_result.total_time_us
    assert default_result.critical_path_us == compact_result.critical_path_us
    assert default_result.rank_metrics == compact_result.rank_metrics
    assert len(default_result.simulated_events) == 2
    assert compact_result.simulated_events == ()


def test_replay_collective_waits_for_same_host_machine_dispatch_queue():
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:h0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=10,
                    module="host.dispatch",
                    api="__hostDelay__",
                    op_type="host_delay",
                    duration_us=50.0,
                    duration_source="manual",
                    extras={"host_machine_id": "host0"},
                ),
            ),
            1: (
                AnnotatedEvent(
                    id="r1:e0",
                    rank=1,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=1,
                    tid=11,
                    module="libnccl.so",
                    api="ncclAllReduce",
                    op_type="nccl_collective",
                    duration_us=5.0,
                    duration_source="manual",
                    extras={"host_machine_id": "host0"},
                    collective_group_id="ncclAllReduce#0",
                ),
            ),
            2: (
                AnnotatedEvent(
                    id="r2:e0",
                    rank=2,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=2,
                    tid=12,
                    module="libnccl.so",
                    api="ncclAllReduce",
                    op_type="nccl_collective",
                    duration_us=5.0,
                    duration_source="manual",
                    extras={"host_machine_id": "host1"},
                    collective_group_id="ncclAllReduce#0",
                ),
            ),
        },
        global_events=(),
        collective_groups={
            "ncclAllReduce#0": CollectiveGroup(
                id="ncclAllReduce#0",
                api="ncclAllReduce",
                op_type="nccl_collective",
                ranks=(1, 2),
                event_ids=("r1:e0", "r2:e0"),
            )
        },
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0] + trace.rank_events[1] + trace.rank_events[2],
        collective_groups=trace.collective_groups,
    )

    result = replay_annotated_trace(trace)

    assert result.success
    assert result.total_time_us == 55.0
    collective_events = [event for event in result.simulated_events if event.collective_group_id == "ncclAllReduce#0"]
    assert len(collective_events) == 2
    assert {event.start_us for event in collective_events} == {50.0}
    assert {event.end_us for event in collective_events} == {55.0}


def test_replay_pipeline_smoke_on_corpus():
    trace_dir = Path("paper/traces/fake/e3")
    real_dir = Path("paper/traces/real/e1")
    if not trace_dir.exists() or not real_dir.exists():
        pytest.skip("trace dirs not available")

    estimator = Estimator.fit_from_traces(str(real_dir), max_files=2)
    bundle = load_trace_directory(trace_dir, max_events_per_rank=256)
    collated = collate_trace_bundle(bundle)
    annotated = annotate_collated_trace(collated, estimator)
    result = replay_annotated_trace(annotated)

    assert result.success
    assert result.total_time_us > 0
    assert len(result.rank_metrics) == collated.world_size


def test_replay_nccl_send_recv_without_collective_group_does_not_stall():
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:e0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=0,
                    module="libnccl.so",
                    api="ncclSend",
                    op_type="nccl_collective",
                    duration_us=4.0,
                    duration_source="manual",
                ),
            ),
            1: (
                AnnotatedEvent(
                    id="r1:e0",
                    rank=1,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=1,
                    tid=1,
                    module="libnccl.so",
                    api="ncclRecv",
                    op_type="nccl_collective",
                    duration_us=4.0,
                    duration_source="manual",
                ),
            ),
        },
        global_events=(),
        collective_groups={},
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0] + trace.rank_events[1],
        collective_groups=trace.collective_groups,
    )

    result = replay_annotated_trace(trace)

    assert result.success
    assert result.total_time_us == 4.0
    assert len(result.simulated_events) == 2


@pytest.mark.parametrize("record_api", ["cudaEventRecord", "cudaEventRecordWithFlags"])
def test_replay_respects_cuda_event_wait_across_streams(record_api: str):
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:e0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    extras={"stream_id": "stream-a"},
                    duration_us=10.0,
                    duration_source="manual",
                ),
                AnnotatedEvent(
                    id="r0:e1",
                    rank=0,
                    ordinal=1,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api=record_api,
                    op_type="stream_op",
                    extras={"stream_id": "stream-a", "event_id": "evt-1"},
                    duration_us=0.0,
                    duration_source="manual",
                    prev_event_id="r0:e0",
                ),
                AnnotatedEvent(
                    id="r0:e2",
                    rank=0,
                    ordinal=2,
                    source=TraceSource.UNKNOWN,
                    ts=2,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaStreamWaitEvent",
                    op_type="stream_op",
                    extras={"stream_id": "stream-b", "event_id": "evt-1"},
                    duration_us=0.0,
                    duration_source="manual",
                    prev_event_id="r0:e1",
                ),
                AnnotatedEvent(
                    id="r0:e3",
                    rank=0,
                    ordinal=3,
                    source=TraceSource.UNKNOWN,
                    ts=3,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    extras={"stream_id": "stream-b"},
                    duration_us=2.0,
                    duration_source="manual",
                    prev_event_id="r0:e2",
                ),
            ),
        },
        global_events=(),
        collective_groups={},
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0],
        collective_groups=trace.collective_groups,
    )

    result = replay_annotated_trace(trace)

    launches = [event for event in result.simulated_events if event.api == "cudaLaunchKernel"]
    assert len(launches) == 2
    assert launches[0].start_us == 0.0
    assert launches[0].end_us == 10.0
    assert launches[1].start_us == 10.0
    assert launches[1].end_us == 12.0


def test_replay_treats_cuda_event_query_as_nonblocking_poll():
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:e0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    extras={"stream_id": "comm"},
                    duration_us=10.0,
                    duration_source="manual",
                ),
                AnnotatedEvent(
                    id="r0:e1",
                    rank=0,
                    ordinal=1,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaEventRecord",
                    op_type="stream_op",
                    extras={"stream_id": "comm", "event_id": "done"},
                    duration_us=0.0,
                    duration_source="manual",
                    prev_event_id="r0:e0",
                ),
                AnnotatedEvent(
                    id="r0:e2",
                    rank=0,
                    ordinal=2,
                    source=TraceSource.UNKNOWN,
                    ts=2,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaEventQuery",
                    op_type="stream_op",
                    extras={
                        "event_id": "done",
                        "host_timing_mode": "measure",
                        "host_timing_source": "direct_wallclock",
                    },
                    duration_us=1.0,
                    duration_source="manual",
                    prev_event_id="r0:e1",
                ),
                AnnotatedEvent(
                    id="r0:e3",
                    rank=0,
                    ordinal=3,
                    source=TraceSource.UNKNOWN,
                    ts=3,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    extras={"stream_id": "compute"},
                    duration_us=2.0,
                    duration_source="manual",
                    prev_event_id="r0:e2",
                ),
            ),
        },
        global_events=(),
        collective_groups={},
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0],
        collective_groups=trace.collective_groups,
    )

    result = replay_annotated_trace(trace)

    events = {event.event_id: event for event in result.simulated_events}
    assert events["r0:e2"].start_us == 0.0
    assert events["r0:e2"].end_us == 1.0
    assert events["r0:e3"].start_us == 1.0


def test_replay_treats_repeated_cuda_event_waits_between_records_as_ready():
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:e0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaEventRecord",
                    op_type="stream_op",
                    extras={"stream_id": "producer", "event_id": "reuse"},
                    duration_us=1.0,
                    duration_source="manual",
                ),
                AnnotatedEvent(
                    id="r0:e1",
                    rank=0,
                    ordinal=1,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaStreamWaitEvent",
                    op_type="stream_op",
                    extras={"stream_id": "consumer", "event_id": "reuse"},
                    duration_us=1.0,
                    duration_source="manual",
                    prev_event_id="r0:e0",
                ),
                AnnotatedEvent(
                    id="r0:e2",
                    rank=0,
                    ordinal=2,
                    source=TraceSource.UNKNOWN,
                    ts=2,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaStreamWaitEvent",
                    op_type="stream_op",
                    extras={"stream_id": "consumer", "event_id": "reuse"},
                    duration_us=1.0,
                    duration_source="manual",
                    prev_event_id="r0:e1",
                ),
                AnnotatedEvent(
                    id="r0:e3",
                    rank=0,
                    ordinal=3,
                    source=TraceSource.UNKNOWN,
                    ts=3,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    extras={"stream_id": "consumer"},
                    duration_us=2.0,
                    duration_source="manual",
                    prev_event_id="r0:e2",
                ),
            ),
        },
        global_events=(),
        collective_groups={},
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0],
        collective_groups=trace.collective_groups,
    )

    result = replay_annotated_trace(trace)

    events = {event.event_id: event for event in result.simulated_events}
    assert events["r0:e1"].start_us == 1.0
    assert events["r0:e2"].start_us == 2.0
    assert events["r0:e3"].start_us == 3.0


def test_replay_real_cuda_event_query_does_not_double_count_observed_poll_gap():
    events = (
        AnnotatedEvent(
            id="r0:e0",
            rank=0,
            ordinal=0,
            source=TraceSource.REAL,
            ts=0,
            pid=0,
            tid=0,
            module="libcudart.so",
            api="cudaLaunchKernel",
            op_type="kernel_launch",
            extras={"stream_id": "comm"},
            duration_us=10.0,
            duration_source="manual",
        ),
        AnnotatedEvent(
            id="r0:e1",
            rank=0,
            ordinal=1,
            source=TraceSource.REAL,
            ts=1,
            pid=0,
            tid=0,
            module="libcudart.so",
            api="cudaEventRecord",
            op_type="stream_op",
            extras={"stream_id": "comm", "event_id": "done"},
            duration_us=0.0,
            duration_source="manual",
            prev_event_id="r0:e0",
        ),
        AnnotatedEvent(
            id="r0:e2",
            rank=0,
            ordinal=2,
            source=TraceSource.REAL,
            ts=2,
            pid=0,
            tid=0,
            module="libcudart.so",
            api="__hostDelay__",
            op_type="host_delay",
            duration_us=7.0,
            duration_source="manual",
            prev_event_id="r0:e1",
        ),
        AnnotatedEvent(
            id="r0:e3",
            rank=0,
            ordinal=3,
            source=TraceSource.REAL,
            ts=3,
            pid=0,
            tid=0,
            module="libcudart.so",
            api="cudaEventQuery",
            op_type="stream_op",
            extras={"event_id": "done"},
            duration_us=1.0,
            duration_source="manual",
            prev_event_id="r0:e2",
        ),
    )
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.REAL,
        rank_events={0: events},
        global_events=events,
        collective_groups={},
    )

    result = replay_annotated_trace(trace)

    by_id = {event.event_id: event for event in result.simulated_events}
    assert by_id["r0:e2"].start_us == 0.0
    assert by_id["r0:e2"].end_us == 7.0
    assert by_id["r0:e3"].start_us == 7.0
    assert by_id["r0:e3"].end_us == 8.0
    assert result.total_time_us == 10.0


def test_replay_tracks_cuda_event_handle_versions():
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:e0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    extras={"stream_id": "stream-a"},
                    duration_us=5.0,
                    duration_source="manual",
                ),
                AnnotatedEvent(
                    id="r0:e1",
                    rank=0,
                    ordinal=1,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaEventRecord",
                    op_type="stream_op",
                    extras={"stream_id": "stream-a", "event_id": "evt-1"},
                    duration_us=0.0,
                    duration_source="manual",
                    prev_event_id="r0:e0",
                ),
                AnnotatedEvent(
                    id="r0:e2",
                    rank=0,
                    ordinal=2,
                    source=TraceSource.UNKNOWN,
                    ts=2,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaStreamWaitEvent",
                    op_type="stream_op",
                    extras={"stream_id": "stream-b", "event_id": "evt-1"},
                    duration_us=0.0,
                    duration_source="manual",
                    prev_event_id="r0:e1",
                ),
                AnnotatedEvent(
                    id="r0:e3",
                    rank=0,
                    ordinal=3,
                    source=TraceSource.UNKNOWN,
                    ts=3,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    extras={"stream_id": "stream-b"},
                    duration_us=1.0,
                    duration_source="manual",
                    prev_event_id="r0:e2",
                ),
                AnnotatedEvent(
                    id="r0:e4",
                    rank=0,
                    ordinal=4,
                    source=TraceSource.UNKNOWN,
                    ts=4,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    extras={"stream_id": "stream-a"},
                    duration_us=7.0,
                    duration_source="manual",
                    prev_event_id="r0:e3",
                ),
                AnnotatedEvent(
                    id="r0:e5",
                    rank=0,
                    ordinal=5,
                    source=TraceSource.UNKNOWN,
                    ts=5,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaEventRecord",
                    op_type="stream_op",
                    extras={"stream_id": "stream-a", "event_id": "evt-1"},
                    duration_us=0.0,
                    duration_source="manual",
                    prev_event_id="r0:e4",
                ),
                AnnotatedEvent(
                    id="r0:e6",
                    rank=0,
                    ordinal=6,
                    source=TraceSource.UNKNOWN,
                    ts=6,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaStreamWaitEvent",
                    op_type="stream_op",
                    extras={"stream_id": "stream-c", "event_id": "evt-1"},
                    duration_us=0.0,
                    duration_source="manual",
                    prev_event_id="r0:e5",
                ),
                AnnotatedEvent(
                    id="r0:e7",
                    rank=0,
                    ordinal=7,
                    source=TraceSource.UNKNOWN,
                    ts=7,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    extras={"stream_id": "stream-c"},
                    duration_us=1.0,
                    duration_source="manual",
                    prev_event_id="r0:e6",
                ),
            ),
        },
        global_events=(),
        collective_groups={},
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0],
        collective_groups=trace.collective_groups,
    )

    result = replay_annotated_trace(trace)

    launches = {
        event.event_id: event
        for event in result.simulated_events
        if event.api == "cudaLaunchKernel"
    }
    assert launches["r0:e0"].start_us == 0.0
    assert launches["r0:e0"].end_us == 5.0
    assert launches["r0:e3"].start_us == 5.0
    assert launches["r0:e3"].end_us == 6.0
    assert launches["r0:e4"].start_us == 5.0
    assert launches["r0:e4"].end_us == 12.0
    assert launches["r0:e7"].start_us == 12.0
    assert launches["r0:e7"].end_us == 13.0


def test_replay_treats_unrecorded_cuda_event_wait_as_noop():
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:e0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaStreamWaitEvent",
                    op_type="stream_op",
                    extras={"stream_id": "stream-a", "event_id": "evt-never-recorded"},
                    duration_us=0.0,
                    duration_source="manual",
                ),
                AnnotatedEvent(
                    id="r0:e1",
                    rank=0,
                    ordinal=1,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    extras={"stream_id": "stream-a"},
                    duration_us=3.0,
                    duration_source="manual",
                    prev_event_id="r0:e0",
                ),
            ),
        },
        global_events=(),
        collective_groups={},
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0],
        collective_groups=trace.collective_groups,
    )

    result = replay_annotated_trace(trace)

    events = {event.event_id: event for event in result.simulated_events}
    assert events["r0:e0"].start_us == 0.0
    assert events["r0:e0"].end_us == 0.0
    assert events["r0:e1"].start_us == 0.0
    assert events["r0:e1"].end_us == 3.0
    assert result.total_time_us == 3.0


def test_replay_does_not_bind_step_boundary_event_wait_to_future_record():
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:e0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaStreamWaitEvent",
                    op_type="stream_op",
                    extras={"stream_id": "stream-b", "event_id": "evt-reused"},
                    duration_us=0.0,
                    duration_source="manual",
                ),
                AnnotatedEvent(
                    id="r0:e1",
                    rank=0,
                    ordinal=1,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    extras={"stream_id": "stream-b"},
                    duration_us=2.0,
                    duration_source="manual",
                    prev_event_id="r0:e0",
                ),
                AnnotatedEvent(
                    id="r0:e2",
                    rank=0,
                    ordinal=2,
                    source=TraceSource.UNKNOWN,
                    ts=2,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    extras={"stream_id": "stream-a"},
                    duration_us=5.0,
                    duration_source="manual",
                    prev_event_id="r0:e1",
                ),
                AnnotatedEvent(
                    id="r0:e3",
                    rank=0,
                    ordinal=3,
                    source=TraceSource.UNKNOWN,
                    ts=3,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaEventRecord",
                    op_type="stream_op",
                    extras={"stream_id": "stream-a", "event_id": "evt-reused"},
                    duration_us=0.0,
                    duration_source="manual",
                    prev_event_id="r0:e2",
                ),
                AnnotatedEvent(
                    id="r0:e4",
                    rank=0,
                    ordinal=4,
                    source=TraceSource.UNKNOWN,
                    ts=4,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaStreamWaitEvent",
                    op_type="stream_op",
                    extras={"stream_id": "stream-c", "event_id": "evt-reused"},
                    duration_us=0.0,
                    duration_source="manual",
                    prev_event_id="r0:e3",
                ),
                AnnotatedEvent(
                    id="r0:e5",
                    rank=0,
                    ordinal=5,
                    source=TraceSource.UNKNOWN,
                    ts=5,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    extras={"stream_id": "stream-c"},
                    duration_us=1.0,
                    duration_source="manual",
                    prev_event_id="r0:e4",
                ),
            ),
        },
        global_events=(),
        collective_groups={},
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0],
        collective_groups=trace.collective_groups,
    )

    result = replay_annotated_trace(trace)

    events = {event.event_id: event for event in result.simulated_events}
    assert events["r0:e1"].start_us == 0.0
    assert events["r0:e1"].end_us == 2.0
    assert events["r0:e5"].start_us == 5.0
    assert events["r0:e5"].end_us == 6.0


def test_replay_uses_cublas_handle_stream_binding():
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:e0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=0,
                    module="libcublas.so",
                    api="cublasCreate_v2",
                    op_type="context_op",
                    extras={"handle_id": "h1"},
                    duration_us=0.0,
                    duration_source="manual",
                ),
                AnnotatedEvent(
                    id="r0:e1",
                    rank=0,
                    ordinal=1,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=0,
                    tid=0,
                    module="libcublas.so",
                    api="cublasSetStream_v2",
                    op_type="stream_op",
                    extras={"handle_id": "h1", "stream_id": "s1"},
                    duration_us=0.0,
                    duration_source="manual",
                    prev_event_id="r0:e0",
                ),
                AnnotatedEvent(
                    id="r0:e2",
                    rank=0,
                    ordinal=2,
                    source=TraceSource.UNKNOWN,
                    ts=2,
                    pid=0,
                    tid=0,
                    module="libcublas.so",
                    api="cublasSgemm_v2",
                    op_type="blas_compute",
                    extras={"handle_id": "h1"},
                    duration_us=7.0,
                    duration_source="manual",
                    prev_event_id="r0:e1",
                ),
                AnnotatedEvent(
                    id="r0:e3",
                    rank=0,
                    ordinal=3,
                    source=TraceSource.UNKNOWN,
                    ts=3,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaStreamSynchronize",
                    op_type="stream_op",
                    extras={"stream_id": "s1"},
                    duration_us=1.0,
                    duration_source="manual",
                    prev_event_id="r0:e2",
                ),
            ),
        },
        global_events=(),
        collective_groups={},
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0],
        collective_groups=trace.collective_groups,
    )

    result = replay_annotated_trace(trace)

    events = {event.event_id: event for event in result.simulated_events}
    assert events["r0:e2"].start_us == 0.0
    assert events["r0:e2"].end_us == 7.0
    assert events["r0:e3"].start_us == 7.0
    assert events["r0:e3"].end_us == 8.0
    assert result.total_time_us == 8.0


def test_replay_uses_rank_quiescent_state_for_cuda_device_synchronize():
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:e0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    extras={"stream_id": "s0"},
                    duration_us=5.0,
                    duration_source="manual",
                ),
                AnnotatedEvent(
                    id="r0:e1",
                    rank=0,
                    ordinal=1,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    extras={"stream_id": "s1"},
                    duration_us=7.0,
                    duration_source="manual",
                    prev_event_id="r0:e0",
                ),
                AnnotatedEvent(
                    id="r0:e2",
                    rank=0,
                    ordinal=2,
                    source=TraceSource.UNKNOWN,
                    ts=2,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaDeviceSynchronize",
                    op_type="stream_op",
                    duration_us=1.0,
                    duration_source="manual",
                    prev_event_id="r0:e1",
                ),
            ),
        },
        global_events=(),
        collective_groups={},
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0],
        collective_groups=trace.collective_groups,
    )

    result = replay_annotated_trace(trace)

    events = {event.event_id: event for event in result.simulated_events}
    assert events["r0:e0"].start_us == 0.0
    assert events["r0:e0"].end_us == 5.0
    assert events["r0:e1"].start_us == 0.0
    assert events["r0:e1"].end_us == 7.0
    assert events["r0:e2"].start_us == 7.0
    assert events["r0:e2"].end_us == 8.0
    assert result.total_time_us == 8.0


def test_replay_real_two_rank_ddp_trace_skips_nccl_control_polls():
    trace_dir = Path("paper/maya_lite/remote_traces/tc30033_gpt2_ddp2_20260324")
    fit_dir = Path("paper/maya_lite/remote_traces/tc30033_simple_full_20260324")
    if not trace_dir.exists() or not fit_dir.exists():
        pytest.skip("real DDP trace dirs not available")

    estimator = Estimator.fit_from_traces(str(fit_dir), max_files=2)
    bundle = load_trace_directory(trace_dir)
    collated = collate_trace_bundle(bundle)
    annotated = annotate_collated_trace(collated, estimator)
    result = replay_annotated_trace(annotated)

    assert result.success
    assert result.total_time_us > 0
    assert len(result.rank_metrics) == 2


def test_replay_can_expand_profiled_rank_groups():
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:e0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    duration_us=5.0,
                    duration_source="manual",
                ),
            ),
            2: (
                AnnotatedEvent(
                    id="r2:e0",
                    rank=2,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=2,
                    tid=2,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    duration_us=7.0,
                    duration_source="manual",
                ),
            ),
        },
        global_events=(),
        collective_groups={},
        original_world_size=4,
        profiled_rank_groups={0: (0, 1), 2: (2, 3)},
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0] + trace.rank_events[2],
        collective_groups=trace.collective_groups,
        original_world_size=trace.original_world_size,
        profiled_rank_groups=trace.profiled_rank_groups,
    )

    result = replay_annotated_trace(trace, expand_profiled_rank_groups=True)

    assert result.success
    assert [metric.rank for metric in result.rank_metrics] == [0, 1, 2, 3]
    assert result.rank_metrics[0].total_time_us == result.rank_metrics[1].total_time_us
    assert result.rank_metrics[2].total_time_us == result.rank_metrics[3].total_time_us


def test_replay_collective_ablation_predicate_is_causal_and_default_unchanged():
    group_id = "ncclP2P|comm:flexsim-members:0,1|members:0-1|pair_seq:124"
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:e0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=0,
                    module="libnccl.so",
                    api="ncclSend",
                    op_type="nccl_collective",
                    duration_us=2.0,
                    duration_source="manual",
                    collective_group_id=group_id,
                ),
            ),
            1: (
                AnnotatedEvent(
                    id="r1:k0",
                    rank=1,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=1,
                    tid=1,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    duration_us=5.0,
                    duration_source="manual",
                ),
                AnnotatedEvent(
                    id="r1:e1",
                    rank=1,
                    ordinal=1,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=1,
                    tid=1,
                    module="libnccl.so",
                    api="ncclRecv",
                    op_type="nccl_collective",
                    duration_us=2.0,
                    duration_source="manual",
                    prev_event_id="r1:k0",
                    collective_group_id=group_id,
                ),
            ),
        },
        global_events=(),
        collective_groups={
            group_id: CollectiveGroup(
                id=group_id,
                api="ncclP2P",
                op_type="nccl_collective",
                ranks=(0, 1),
                event_ids=("r0:e0", "r1:e1"),
                participant_count=2,
            )
        },
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0] + trace.rank_events[1],
        collective_groups=trace.collective_groups,
    )

    default = replay_annotated_trace(trace)
    ablated = replay_annotated_trace(
        trace,
        collective_ablation_predicate=pair_seq_collective_ablation_predicate({124}),
    )

    default_collectives = [event for event in default.simulated_events if event.collective_group_id]
    assert [(event.event_id, event.start_us, event.end_us) for event in default_collectives] == [
        ("r0:e0", 5.0, 7.0),
        ("r1:e1", 5.0, 7.0),
    ]
    ablated_collectives = [event for event in ablated.simulated_events if event.collective_group_id]
    assert [(event.event_id, event.start_us, event.end_us) for event in ablated_collectives] == [
        ("r0:e0", 0.0, 2.0),
        ("r1:e1", 5.0, 7.0),
    ]
    assert all(event.end_us >= event.start_us for event in ablated.simulated_events)
    assert ablated.total_time_us == default.total_time_us


def test_pair_seq_collective_ablation_predicate_matches_only_selected_p2p_sequences():
    predicate = pair_seq_collective_ablation_predicate({124, 127})

    assert predicate(
        "ncclP2P|comm:flexsim-members:0,1|members:0-1|pair_seq:124",
        (),
    )
    assert not predicate(
        "ncclP2P|comm:flexsim-members:0,1|members:0-1|pair_seq:125",
        (),
    )
    assert not predicate("ncclAllReduce#pair_seq:124", ())


def test_stream_serialization_ablation_partial_predicate_does_not_release_busy_stream():
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:e0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    duration_us=10.0,
                    duration_source="manual",
                    extras={"stream_id": "0"},
                ),
                AnnotatedEvent(
                    id="r0:e1",
                    rank=0,
                    ordinal=1,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    duration_us=1.0,
                    duration_source="manual",
                    extras={"stream_id": "0"},
                ),
                AnnotatedEvent(
                    id="r0:e2",
                    rank=0,
                    ordinal=2,
                    source=TraceSource.UNKNOWN,
                    ts=2,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    duration_us=1.0,
                    duration_source="manual",
                    extras={"stream_id": "0"},
                ),
            ),
        },
        global_events=(),
        collective_groups={},
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0],
        collective_groups=trace.collective_groups,
    )

    result = replay_annotated_trace(
        trace,
        stream_serialization_ablation_predicate=lambda pending: pending.event.id == "r0:e1",
    )

    assert result.success
    stream_events = [event for event in result.simulated_events if event.resource_kind == "stream"]
    assert [(event.event_id, event.start_us, event.end_us) for event in stream_events] == [
        ("r0:e0", 0.0, 10.0),
        ("r0:e1", 10.0, 11.0),
        ("r0:e2", 10.0, 11.0),
    ]


def test_stream_serialization_ablation_allows_matched_following_stream_op_overlap():
    trace = AnnotatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.UNKNOWN,
        rank_events={
            0: (
                AnnotatedEvent(
                    id="r0:e0",
                    rank=0,
                    ordinal=0,
                    source=TraceSource.UNKNOWN,
                    ts=0,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    duration_us=10.0,
                    duration_source="manual",
                    extras={"stream_id": "0"},
                ),
                AnnotatedEvent(
                    id="r0:e1",
                    rank=0,
                    ordinal=1,
                    source=TraceSource.UNKNOWN,
                    ts=1,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    duration_us=10.0,
                    duration_source="manual",
                    extras={"stream_id": "0"},
                ),
                AnnotatedEvent(
                    id="r0:e2",
                    rank=0,
                    ordinal=2,
                    source=TraceSource.UNKNOWN,
                    ts=2,
                    pid=0,
                    tid=0,
                    module="libcudart.so",
                    api="cudaLaunchKernel",
                    op_type="kernel_launch",
                    duration_us=1.0,
                    duration_source="manual",
                    extras={"stream_id": "0"},
                ),
            ),
        },
        global_events=(),
        collective_groups={},
    )
    trace = AnnotatedTrace(
        trace_dir=trace.trace_dir,
        source=trace.source,
        rank_events=trace.rank_events,
        global_events=trace.rank_events[0],
        collective_groups=trace.collective_groups,
    )

    result = replay_annotated_trace(
        trace,
        stream_serialization_ablation_predicate=lambda pending: pending.event.id == "r0:e1",
    )

    assert result.success
    stream_events = [event for event in result.simulated_events if event.resource_kind == "stream"]
    assert [(event.event_id, event.start_us, event.end_us) for event in stream_events] == [
        ("r0:e0", 0.0, 10.0),
        ("r0:e1", 10.0, 20.0),
        ("r0:e2", 10.0, 11.0),
    ]
