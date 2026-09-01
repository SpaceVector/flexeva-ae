"""Default-off launch-neighborhood equivalence diagnostic metadata."""

from __future__ import annotations

from typing import Mapping


LAUNCH_NEIGHBORHOOD_EQUIVALENCE_ENV_KEYS = (
    "MAYA_ENABLE_LAUNCH_NEIGHBORHOOD_EQUIVALENCE_DIAGNOSTICS",
    "FLEXSIM_MAYA_ENABLE_LAUNCH_NEIGHBORHOOD_EQUIVALENCE_DIAGNOSTICS",
)

LAUNCH_NEIGHBORHOOD_EQUIVALENCE_SCHEMA_VERSION = (
    "launch_neighborhood_occurrence_equivalence_diagnostics_v1"
)

LAUNCH_NEIGHBORHOOD_EQUIVALENCE_EXPORT_FIELDS = (
    "launch_neighborhood_equivalence_schema_version",
    "launch_neighborhood_equivalence_opt_in_flag",
    "launch_neighborhood_occurrence_id",
    "launch_neighborhood_id",
    "launch_neighborhood_role",
    "launch_neighborhood_occurrence_basis",
    "launch_neighborhood_normalized_signature",
    "launch_neighborhood_prev_raw_event_id",
    "launch_neighborhood_current_raw_event_id",
    "launch_neighborhood_prev_api",
    "launch_neighborhood_current_api",
    "launch_neighborhood_prev_api_visibility_label",
    "launch_neighborhood_current_api_visibility_label",
    "launch_neighborhood_visibility_status",
    "launch_neighborhood_boundary_exclusion_reasons",
    "launch_neighborhood_rank",
    "launch_neighborhood_paper_valid_window_id",
    "launch_neighborhood_host_dispatch_queue_id",
    "launch_neighborhood_stream_id",
    "launch_neighborhood_prev_raw_ordinal",
    "launch_neighborhood_current_raw_ordinal",
    "launch_neighborhood_order_key",
    "launch_neighborhood_wait_map_nonoverlap_status",
    "launch_neighborhood_wait_map_nonoverlap_unavailable_reason",
    "launch_neighborhood_replay_interval_id",
    "launch_neighborhood_replay_interval_unavailable_reason",
    "launch_neighborhood_counterpart_join_status",
    "launch_neighborhood_counterpart_join_unavailable_reason",
    "launch_neighborhood_safe_to_use_as_repair_evidence",
    "launch_neighborhood_safe_to_use_as_subtraction_delta",
)

EVENT_WAIT_SYNC_APIS = {
    "cudaEventRecord",
    "cudaEventRecordWithFlags",
    "cudaEventQuery",
    "cudaEventSynchronize",
    "cudaStreamWaitEvent",
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
}
COLLECTIVE_OR_COMM_APIS = {
    "ncclAllGather",
    "ncclAllReduce",
    "ncclAllToAll",
    "ncclAllToAllv",
    "ncclBcast",
    "ncclBroadcast",
    "ncclGather",
    "ncclRecv",
    "ncclReduce",
    "ncclReduceScatter",
    "ncclScatter",
    "ncclSend",
}
COLLECTIVE_CONTROL_APIS = {
    "ncclCommGetAsyncError",
    "ncclGroupStart",
    "ncclGroupEnd",
}
PAPER_VISIBLE_OPERATION_APIS = {
    "cudaLaunchKernel",
    "cublasGemmEx",
    "cublasGemmStridedBatchedEx",
    "cublasGemmBatchedEx",
    "cublasLtMatmul",
}
LAUNCH_CONTROL_APIS = {
    "__cudaPopCallConfiguration",
    "__cudaPushCallConfiguration",
    "cudaGetDevice",
    "cudaGetLastError",
    "cudaPeekAtLastError",
}
STREAM_BINDING_APIS = {"cublasSetStream_v2"}

LAUNCH_NEIGHBORHOOD_EQUIVALENCE_APIS = (
    EVENT_WAIT_SYNC_APIS
    | COLLECTIVE_OR_COMM_APIS
    | COLLECTIVE_CONTROL_APIS
    | PAPER_VISIBLE_OPERATION_APIS
    | LAUNCH_CONTROL_APIS
    | STREAM_BINDING_APIS
)


def launch_neighborhood_visibility_label(api: str | None) -> str:
    if api in {"__cudaPopCallConfiguration", "__cudaPushCallConfiguration"}:
        return "compat_launch_config_control_candidate"
    if api in {"cudaGetDevice", "cudaGetLastError", "cudaPeekAtLastError"}:
        return "unresolved_wrapper_control_cpu_work"
    if api in STREAM_BINDING_APIS:
        return "stream_binding_control_safety_sensitive"
    if api in EVENT_WAIT_SYNC_APIS:
        return "event_wait_sync_safety_sensitive"
    if api in COLLECTIVE_OR_COMM_APIS:
        return "collective_or_comm_safety_sensitive"
    if api in COLLECTIVE_CONTROL_APIS:
        return "collective_control_polling_safety_sensitive"
    if api in PAPER_VISIBLE_OPERATION_APIS:
        return "paper_visible_operation_boundary"
    if api in (None, ""):
        return "leading_boundary_no_previous_api"
    return "unresolved_other_api"


def launch_neighborhood_normalized_signature(
    previous_api: str | None,
    current_api: str | None,
) -> str:
    return (
        f"{launch_neighborhood_visibility_label(previous_api)}"
        f" -> {launch_neighborhood_visibility_label(current_api)}"
    )


def launch_neighborhood_boundary_exclusion_reasons(
    previous_api: str | None,
    current_api: str | None,
) -> list[str]:
    reasons: list[str] = []
    for side, api in (("prev", previous_api), ("current", current_api)):
        label = launch_neighborhood_visibility_label(api)
        if label == "event_wait_sync_safety_sensitive":
            reasons.append(f"{side}:{api}:event_wait_sync_release_surface")
        elif label == "stream_binding_control_safety_sensitive":
            reasons.append(f"{side}:{api}:stateful_stream_binding_surface")
        elif label in {
            "collective_or_comm_safety_sensitive",
            "collective_control_polling_safety_sensitive",
        }:
            reasons.append(f"{side}:{api}:collective_wait_map_or_polling_surface")
        elif label == "paper_visible_operation_boundary":
            reasons.append(f"{side}:{api}:paper_visible_operation_boundary")
        elif label == "unresolved_wrapper_control_cpu_work":
            reasons.append(f"{side}:{api}:wrapper_control_visibility_unresolved")
    return reasons


def build_launch_neighborhood_equivalence_metadata(
    *,
    rank: int,
    previous_api: str | None,
    current_api: str | None,
    previous_raw_event_id: str | None,
    current_raw_event_id: str,
    previous_raw_ordinal: int | None,
    current_raw_ordinal: int,
    host_dispatch_queue_id: str | None,
    stream_id: object | None,
    paper_valid_window_id: str | None,
    role: str,
) -> dict[str, object]:
    previous_key = (
        "leading" if previous_raw_ordinal is None else f"raw_ordinal:{int(previous_raw_ordinal)}"
    )
    current_key = f"raw_ordinal:{int(current_raw_ordinal)}"
    occurrence_id = (
        f"rank:{int(rank)}:launch_neighborhood:{previous_key}->{current_key}"
    )
    exclusion_reasons = launch_neighborhood_boundary_exclusion_reasons(
        previous_api,
        current_api,
    )
    return {
        "launch_neighborhood_equivalence_schema_version": (
            LAUNCH_NEIGHBORHOOD_EQUIVALENCE_SCHEMA_VERSION
        ),
        "launch_neighborhood_equivalence_opt_in_flag": True,
        "launch_neighborhood_occurrence_id": occurrence_id,
        "launch_neighborhood_id": occurrence_id,
        "launch_neighborhood_role": role,
        "launch_neighborhood_occurrence_basis": (
            "rank_plus_adjacent_raw_ordinals_no_duration_keys"
        ),
        "launch_neighborhood_normalized_signature": (
            launch_neighborhood_normalized_signature(previous_api, current_api)
        ),
        "launch_neighborhood_prev_raw_event_id": previous_raw_event_id,
        "launch_neighborhood_current_raw_event_id": current_raw_event_id,
        "launch_neighborhood_prev_api": previous_api,
        "launch_neighborhood_current_api": current_api,
        "launch_neighborhood_prev_api_visibility_label": (
            launch_neighborhood_visibility_label(previous_api)
        ),
        "launch_neighborhood_current_api_visibility_label": (
            launch_neighborhood_visibility_label(current_api)
        ),
        "launch_neighborhood_visibility_status": (
            "structural_labels_only_visibility_equivalence_unproven"
        ),
        "launch_neighborhood_boundary_exclusion_reasons": exclusion_reasons,
        "launch_neighborhood_rank": int(rank),
        "launch_neighborhood_paper_valid_window_id": paper_valid_window_id,
        "launch_neighborhood_host_dispatch_queue_id": host_dispatch_queue_id,
        "launch_neighborhood_stream_id": stream_id,
        "launch_neighborhood_prev_raw_ordinal": (
            int(previous_raw_ordinal) if previous_raw_ordinal is not None else None
        ),
        "launch_neighborhood_current_raw_ordinal": int(current_raw_ordinal),
        "launch_neighborhood_order_key": f"rank:{int(rank)}:{previous_key}->{current_key}",
        "launch_neighborhood_wait_map_nonoverlap_status": "unavailable",
        "launch_neighborhood_wait_map_nonoverlap_unavailable_reason": (
            "requires_replay_wait_map_stream_fifo_host_queue_collective_nonoverlap_ledger"
        ),
        "launch_neighborhood_replay_interval_id": None,
        "launch_neighborhood_replay_interval_unavailable_reason": (
            "replay_interval_id_not_assigned_until_replay_export"
        ),
        "launch_neighborhood_counterpart_join_status": "unavailable",
        "launch_neighborhood_counterpart_join_unavailable_reason": (
            "predicted_actual_normalized_neighborhood_join_not_run"
        ),
        "launch_neighborhood_safe_to_use_as_repair_evidence": False,
        "launch_neighborhood_safe_to_use_as_subtraction_delta": False,
    }


def metadata_has_launch_neighborhood_equivalence(extras: Mapping[str, object]) -> bool:
    if extras.get("launch_neighborhood_equivalence_opt_in_flag") is True:
        return True
    return any(
        extras.get(field) not in (None, "")
        for field in LAUNCH_NEIGHBORHOOD_EQUIVALENCE_EXPORT_FIELDS
    )
