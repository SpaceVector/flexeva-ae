#!/usr/bin/env python3
"""Validate split boundary-origin producer smoke traces.

This helper checks only additive producer-side raw trace fields. It does not run
or modify collate/replay and it intentionally does not infer paper-visible host
attribution. Launch-boundary visibility diagnostics are opt-in/default-off; use
the CLI mode flags to assert either default-off or opted-in output.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SELECTED_APIS = [
    "cudaGetDevice",
    "__cudaPushCallConfiguration",
    "cudaGetLastError",
    "cublasGemmEx",
    "cublasGemmStridedBatchedEx",
    "__cudaPopCallConfiguration",
    "cudaLaunchKernel",
    "cublasSetStream_v2",
    "cudaEventRecord",
    "cudaStreamWaitEvent",
]

PREPOP_LAUNCH_NEIGHBORHOOD_APIS = {
    "__cudaPushCallConfiguration",
    "cudaGetLastError",
    "cublasGemmEx",
    "cublasGemmStridedBatchedEx",
    "__cudaPopCallConfiguration",
}

PREPOP_LAUNCH_NEIGHBORHOOD_FIELDS = [
    "prepop_launch_neighborhood_schema_version",
    "prepop_launch_neighborhood_opt_in_flag",
    "prepop_launch_neighborhood_api",
    "prepop_launch_neighborhood_row_role",
    "prepop_launch_neighborhood_boundary_family",
    "prepop_launch_neighborhood_visibility_kind",
    "prepop_launch_neighborhood_visibility_status",
    "prepop_launch_neighborhood_classification_basis",
    "prepop_launch_neighborhood_mechanical_split_status",
    "prepop_launch_neighborhood_mechanical_split_unavailable_reason",
    "prepop_launch_neighborhood_count_once_status",
    "prepop_launch_neighborhood_count_once_unavailable_reason",
    "prepop_launch_neighborhood_wait_map_safety_status",
    "prepop_launch_neighborhood_wait_map_safety_unavailable_reason",
    "prepop_launch_neighborhood_double_counting_overlap_status",
    "prepop_launch_neighborhood_runtime_substitution_status",
    "prepop_launch_neighborhood_endpoint_timestamp_substitution_status",
    "prepop_launch_neighborhood_repair_ready",
    "prepop_launch_neighborhood_safe_to_use_as_repair_evidence",
    "prepop_launch_neighborhood_safe_to_use_as_subtraction_delta",
    "prepop_launch_neighborhood_candidate_predecessor_api",
    "prepop_launch_neighborhood_expected_successor_api",
]

UNKNOWN_COMPONENT_FIELDS = [
    "fake_api_body_duration_us",
    "actual_host_dispatch_duration_us",
    "wrapper_internal_duration_us",
    "instrumentation_only_duration_us",
]

GEMM_ALGORITHM_PAYLOAD_APIS = {
    "cublasGemmEx",
    "cublasGemmStridedBatchedEx",
}

LAUNCH_BOUNDARY_DIAGNOSTIC_FIELDS = [
    "boundary_origin_kind",
    "boundary_segment_schema_version",
    "launch_boundary_id",
    "launch_boundary_id_unavailable_reason",
    "wrapper_segment_coverage",
    "wrapper_segment_sum_us",
    "wrapper_segment_unattributed_us",
    "paper_visible_host_duration_us",
    "boundary_origin_classification_basis",
    "boundary_visibility_segments",
    "caller_visible_elapsed_us",
    "fake_api_body_duration_us",
    "actual_host_dispatch_duration_us",
    "actual_counterpart_component_id",
    "actual_counterpart_visibility_kind",
    "wrapper_internal_duration_us",
    "instrumentation_only_duration_us",
    "unresolved_mixed_duration_us",
    "actual_launch_control_dispatch_us",
    "actual_launch_api_body_us",
    "actual_launch_instrumentation_only_us",
    "actual_launch_visibility_kind",
    "actual_launch_unavailable_reason",
    *PREPOP_LAUNCH_NEIGHBORHOOD_FIELDS,
]

TOLERANCE_US = 2.0


def _as_float(event: dict, key: str) -> float:
    if key not in event:
        raise AssertionError(f"{event.get('api')}: missing {key}")
    value = event[key]
    if isinstance(value, bool):
        raise AssertionError(f"{event.get('api')}: {key} is bool, expected number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{event.get('api')}: {key} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise AssertionError(f"{event.get('api')}: {key} is not finite: {value!r}")
    return result


def _as_optional_float(event: dict, key: str) -> float | None:
    if key not in event or event[key] in (None, ""):
        return None
    return _as_float(event, key)


def load_events(path: Path) -> list[dict]:
    events: list[dict] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AssertionError(f"line {line_no}: invalid JSON: {exc}") from exc
    return events


def validate(
    path: Path,
    require_positive_body: bool = True,
    *,
    expect_launch_boundary_diagnostics: bool = True,
    expect_gemm_algorithm_aliases: bool = False,
) -> dict:
    events = load_events(path)
    by_api = {event.get("api"): event for event in events if event.get("api") in SELECTED_APIS}
    missing = [api for api in SELECTED_APIS if api not in by_api]
    if missing:
        raise AssertionError(f"missing selected APIs: {missing}")

    summary: dict[str, dict[str, float | str]] = {}
    for api in SELECTED_APIS:
        event = by_api[api]
        if expect_gemm_algorithm_aliases and api in GEMM_ALGORITHM_PAYLOAD_APIS:
            if "algorithm" not in event:
                raise AssertionError(f"{api}: missing algorithm payload field")
            if "algo" not in event:
                raise AssertionError(f"{api}: missing algo payload alias")
            if str(event["algorithm"]) != str(event["algo"]):
                raise AssertionError(
                    f"{api}: algorithm/algo mismatch: "
                    f"{event['algorithm']!r} != {event['algo']!r}"
                )
        if not expect_launch_boundary_diagnostics:
            present = [field for field in LAUNCH_BOUNDARY_DIAGNOSTIC_FIELDS if field in event]
            if present:
                raise AssertionError(
                    f"{api}: launch-boundary diagnostics must be default-off, found {present}"
                )
            host_duration = _as_optional_float(event, "host_duration_us")
            summary[api] = {
                "launch_boundary_diagnostics": "disabled",
                "host_duration_us": host_duration if host_duration is not None else "unavailable",
            }
            continue

        if api in PREPOP_LAUNCH_NEIGHBORHOOD_APIS:
            if event.get("prepop_launch_neighborhood_schema_version") != (
                "prepop_launch_gemm_visibility_row_evidence_v1"
            ):
                raise AssertionError(f"{api}: missing pre-pop launch/GEMM row schema")
            if event.get("prepop_launch_neighborhood_opt_in_flag") is not True:
                raise AssertionError(f"{api}: missing pre-pop opt-in marker")
            if event.get("prepop_launch_neighborhood_api") != api:
                raise AssertionError(f"{api}: row api mismatch in pre-pop metadata")
            if event.get("prepop_launch_neighborhood_visibility_kind") != "mixed_or_unresolved":
                raise AssertionError(f"{api}: pre-pop visibility must remain unresolved")
            if event.get("prepop_launch_neighborhood_visibility_status") != (
                "structural_row_label_only_unresolved"
            ):
                raise AssertionError(f"{api}: pre-pop status must be structural/unresolved")
            for status_field in (
                "prepop_launch_neighborhood_mechanical_split_status",
                "prepop_launch_neighborhood_count_once_status",
                "prepop_launch_neighborhood_wait_map_safety_status",
                "prepop_launch_neighborhood_double_counting_overlap_status",
            ):
                if event.get(status_field) != "unavailable":
                    raise AssertionError(f"{api}: {status_field} must remain unavailable")
            for forbidden_status in (
                "prepop_launch_neighborhood_runtime_substitution_status",
                "prepop_launch_neighborhood_endpoint_timestamp_substitution_status",
            ):
                if event.get(forbidden_status) != "forbidden":
                    raise AssertionError(f"{api}: {forbidden_status} must remain forbidden")
            for safety_field in (
                "prepop_launch_neighborhood_repair_ready",
                "prepop_launch_neighborhood_safe_to_use_as_repair_evidence",
                "prepop_launch_neighborhood_safe_to_use_as_subtraction_delta",
            ):
                if event.get(safety_field) is not False:
                    raise AssertionError(f"{api}: {safety_field} must remain false")
            expected_role = (
                "prepop_predecessor_candidate"
                if api in {"cudaGetLastError", "cublasGemmEx", "cublasGemmStridedBatchedEx"}
                else "launch_config_push_context"
                if api == "__cudaPushCallConfiguration"
                else "launch_config_pop_prepop_endpoint"
            )
            if event.get("prepop_launch_neighborhood_row_role") != expected_role:
                raise AssertionError(f"{api}: unexpected pre-pop row role")
            if api in {"cudaGetLastError", "cublasGemmEx", "cublasGemmStridedBatchedEx"}:
                if event.get("prepop_launch_neighborhood_candidate_predecessor_api") != api:
                    raise AssertionError(f"{api}: missing predecessor API label")
                if event.get("prepop_launch_neighborhood_expected_successor_api") != (
                    "__cudaPopCallConfiguration"
                ):
                    raise AssertionError(f"{api}: expected successor must be pop")

        kind = event.get("boundary_origin_kind")
        if "boundary_segment_schema_version" not in event:
            summary[api] = {
                "prepop_launch_neighborhood": "structural_row_label_only_unresolved",
            }
            continue
        if kind != "mixed_or_unresolved":
            raise AssertionError(f"{api}: boundary_origin_kind={kind!r}, expected mixed_or_unresolved")
        if event.get("paper_visible_host_duration_us") is not None:
            raise AssertionError(f"{api}: paper_visible_host_duration_us must remain unknown")
        if "actual_launch_control_dispatch_us" in event:
            raise AssertionError(f"{api}: actual_launch_control_dispatch_us is out of scope for this smoke")
        if "actual_device_runtime_us" in event:
            raise AssertionError(f"{api}: actual_device_runtime_us is out of scope for this smoke")
        if event.get("actual_counterpart_visibility_kind") != "mixed_or_unresolved":
            raise AssertionError(
                f"{api}: actual_counterpart_visibility_kind must remain mixed_or_unresolved, "
                f"got {event.get('actual_counterpart_visibility_kind')!r}"
            )
        caller = _as_float(event, "caller_visible_elapsed_us")
        if caller <= 0.0:
            raise AssertionError(f"{api}: caller_visible_elapsed_us must be positive, got {caller}")
        host_duration = _as_float(event, "host_duration_us")
        if abs(host_duration - caller) > TOLERANCE_US:
            raise AssertionError(
                f"{api}: caller_visible_elapsed_us {caller:.3f} must preserve "
                f"host_duration_us {host_duration:.3f}"
            )
        if event.get("boundary_segment_schema_version") != "launch_boundary_visibility_v1":
            raise AssertionError(f"{api}: missing launch boundary segment schema")
        if event.get("wrapper_segment_coverage") not in {
            "partial",
            "complete",
            "unavailable",
            "structural_labels_only_unmeasured",
        }:
            raise AssertionError(f"{api}: invalid wrapper_segment_coverage={event.get('wrapper_segment_coverage')!r}")
        wrapper_segment_sum = _as_float(event, "wrapper_segment_sum_us")
        wrapper_segment_unattributed = _as_float(event, "wrapper_segment_unattributed_us")
        if abs((wrapper_segment_sum + wrapper_segment_unattributed) - caller) > TOLERANCE_US:
            raise AssertionError(
                f"{api}: wrapper segment accounting {wrapper_segment_sum:.3f}+"
                f"{wrapper_segment_unattributed:.3f} does not match caller {caller:.3f}"
            )
        segments = event.get("boundary_visibility_segments")
        if not isinstance(segments, list) or not segments:
            raise AssertionError(f"{api}: boundary_visibility_segments must be a non-empty list")
        for segment in segments:
            if not isinstance(segment, dict):
                raise AssertionError(f"{api}: segment is not an object: {segment!r}")
            for field in (
                "name",
                "visibility_kind",
                "duration_us",
                "classification_basis",
                "included_in_paper_visible_host_duration",
                "included_in_instrumentation_only_duration",
            ):
                if field not in segment:
                    raise AssertionError(f"{api}: segment missing {field}: {segment!r}")
            if segment.get("duration_us") is not None:
                raise AssertionError(f"{api}: segment durations must be unknown: {segment!r}")
            if segment.get("start_offset_us") is not None or segment.get("end_offset_us") is not None:
                raise AssertionError(f"{api}: segment offsets must be unknown: {segment!r}")
            if segment.get("clock") != "unmeasured":
                raise AssertionError(f"{api}: segment clock must be unmeasured: {segment!r}")
        component_values: dict[str, float | None] = {}
        for field in UNKNOWN_COMPONENT_FIELDS:
            value = _as_optional_float(event, field)
            if value is not None:
                raise AssertionError(f"{api}: {field} must remain unknown, got {value}")
            component_values[field] = value
        unresolved_mixed = _as_float(event, "unresolved_mixed_duration_us")
        if unresolved_mixed < -1e-9:
            raise AssertionError(f"{api}: unresolved fallback invalid")
        if abs(unresolved_mixed - caller) > TOLERANCE_US:
            raise AssertionError(
                f"{api}: unresolved_mixed_duration_us {unresolved_mixed:.3f} "
                f"must match caller_visible_elapsed_us {caller:.3f}"
            )
        summary[api] = {
            "boundary_origin_kind": kind,
            "caller_visible_elapsed_us": caller,
            "host_duration_us": host_duration,
            "wrapper_segment_sum_us": wrapper_segment_sum,
            "wrapper_segment_unattributed_us": wrapper_segment_unattributed,
            "segment_count": len(segments),
            **component_values,
            "unresolved_mixed_duration_us": unresolved_mixed,
        }
    if expect_launch_boundary_diagnostics:
        for api in ("__cudaPopCallConfiguration", "cudaLaunchKernel"):
            if by_api[api].get("launch_boundary_id"):
                raise AssertionError(f"{api}: launch_boundary_id must be disabled for diagnostic-only timing safety")
            if by_api[api].get("launch_boundary_id_unavailable_reason") != (
                "fakecuda_launch_pair_id_disabled_to_preserve_host_duration"
            ):
                raise AssertionError(f"{api}: missing launch_boundary_id_unavailable_reason")
    return summary


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_jsonl", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--expect-launch-boundary-diagnostics-enabled",
        action="store_true",
        help="Require opt-in launch-boundary visibility diagnostic fields.",
    )
    mode.add_argument(
        "--expect-launch-boundary-diagnostics-disabled",
        action="store_true",
        help="Require default-off output with no launch-boundary diagnostic fields.",
    )
    parser.add_argument(
        "--allow-zero-body",
        action="store_true",
        help="Allow fake_api_body_duration_us == 0 for logger/schema-only fallback checks.",
    )
    parser.add_argument(
        "--expect-gemm-algorithm-aliases",
        action="store_true",
        help="Require cublasGemmEx/cublasGemmStridedBatchedEx trace rows to carry both algorithm and algo.",
    )
    args = parser.parse_args(argv[1:])
    expect_enabled = not args.expect_launch_boundary_diagnostics_disabled
    summary = validate(
        args.trace_jsonl,
        require_positive_body=not args.allow_zero_body,
        expect_launch_boundary_diagnostics=expect_enabled,
        expect_gemm_algorithm_aliases=args.expect_gemm_algorithm_aliases,
    )
    print(json.dumps({"validated_apis": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
