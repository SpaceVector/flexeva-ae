#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from flexsim.maya_lite.markers import load_step_markers, resolve_step_window_from_markers
from flexsim.maya_lite.filters import (
    canonicalize_trace_api,
    is_host_timing_traced_api,
    is_semantic_traced_api,
)
from flexsim.maya_lite.io import fidelity_window_from_payload
from flexsim.maya_lite.schema import TraceSource


DEFAULT_PROFILE_STATISTIC = "percentile"
# Accuracy-facing host timing profiles are often built from only one or two
# captured steps. In that setting, aggressively truncating pairocc exports to
# the first few occurrences erases recurring control-plane stalls that happen
# throughout the step, and requiring 4+ samples suppresses occurrence-aware
# overrides entirely. Keep the defaults fidelity-first; callers that want a
# more compact/regularized profile can still opt in via CLI flags.
DEFAULT_MAX_PAIR_OCCURRENCE_INDEX = 0
DEFAULT_MIN_PAIR_OCCURRENCE_SAMPLES = 1
DEFAULT_PAIR_OCCURRENCE_MIN_DELTA_US = 0.0
DEFAULT_PAIR_OCCURRENCE_MIN_RATIO = 0.0
DEFAULT_MAX_PAIR_OCCURRENCE_DELAY_US = 0.0

_HOST_TIMING_API_ALIASES = {
    "cudaEventRecordWithFlags": "cudaEventRecord",
    "cudaEventCreate": "cudaEventCreateWithFlags",
    "cudaStreamCreateWithFlags": "cudaStreamCreate",
    "cudaStreamCreateWithPriority": "cudaStreamCreate",
}

_HELPER_THREAD_PRIMARY_APIS = {
    "ncclCommGetAsyncError",
    "ncclGetVersion",
    "cudaSetDevice",
    "cudaGetDevice",
    "cudaGetLastError",
}

_HELPER_THREAD_FORBIDDEN_APIS = {
    "__cudaRegisterFunction",
    "cudaLaunchKernel",
    "cublasCreate_v2",
    "cublasDestroy_v2",
    "cublasSetStream_v2",
    "cublasGemmEx",
    "cublasGemmStridedBatchedEx",
    "cublasGemmBatchedEx",
    "cublasLtCreate",
    "cublasLtDestroy",
    "cublasLtMatmulDescCreate",
    "cublasLtMatmulDescSetAttribute",
    "cublasLtMatmulDescDestroy",
    "cublasLtMatmulPreferenceCreate",
    "cublasLtMatmulPreferenceSetAttribute",
    "cublasLtMatmulPreferenceDestroy",
    "cublasLtMatrixLayoutCreate",
    "cublasLtMatrixLayoutDestroy",
    "cublasLtMatmul",
    "ncclAllReduce",
    "ncclAllGather",
    "ncclAllToAll",
    "ncclAllToAllv",
    "ncclBroadcast",
    "ncclReduce",
    "ncclReduceScatter",
    "ncclSend",
    "ncclRecv",
    "cudaMemcpy",
    "cudaMemcpyAsync",
    "cudaMalloc",
    "cudaMallocAsync",
    "cudaFree",
    "cudaFreeAsync",
    "cudaEventRecord",
    "cudaEventRecordWithFlags",
    "cudaStreamWaitEvent",
}

_SYNTHETIC_HELPER_START_APIS = {
    "ncclCommGetAsyncError",
    "ncclGetVersion",
    "cudaSetDevice",
    "cudaGetDevice",
    "cudaGetLastError",
}


@dataclass(frozen=True)
class _ThreadState:
    previous_api: str
    previous_type: str
    previous_mod: str
    dispatch_high_watermark_ts: int


@dataclass(frozen=True)
class _HelperThreadTemplate:
    source_tid: int
    start_offset_us: int
    end_offset_us: int
    first_api: str
    dominant_api: str
    event_count: int
    api_sequence: tuple[str, ...]


def _canonicalize_host_timing_api(api: str) -> str:
    return canonicalize_trace_api(_HOST_TIMING_API_ALIASES.get(api, api))


def _is_host_timing_profile_event(
    api: str,
    event_type: str,
    *,
    profile_surface: str = "semantic",
) -> bool:
    if profile_surface == "semantic":
        return is_semantic_traced_api(api, event_type)
    return is_host_timing_traced_api(api, event_type)


def _is_helper_thread(api_counts: dict[str, int]) -> bool:
    total = sum(api_counts.values())
    if total <= 0:
        return False
    if any(api in _HELPER_THREAD_FORBIDDEN_APIS for api in api_counts):
        return False
    helper_total = sum(count for api, count in api_counts.items() if api in _HELPER_THREAD_PRIMARY_APIS)
    return helper_total / total >= 0.8


def _collect_helper_thread_templates(
    trace_dir: Path,
    *,
    step_windows: dict[int, tuple[int, int]] | None = None,
    ranks: set[int] | None = None,
) -> dict[int, list[_HelperThreadTemplate]]:
    templates_by_rank: dict[int, list[_HelperThreadTemplate]] = {}

    for trace_path in _iter_rank_trace_files(trace_dir):
        rank = _rank_from_trace_path(trace_path)
        if rank is None:
            continue
        if ranks is not None and rank not in ranks:
            continue

        step_window = step_windows.get(rank) if step_windows is not None else None
        rank_base_ts = step_window[0] if step_window is not None else None
        thread_stats: dict[int, dict[str, object]] = {}

        with trace_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = line.strip()
                if not payload:
                    continue
                record = json.loads(payload)
                try:
                    tid = int(record["tid"])
                    ts = int(record["ts"])
                    api = _canonicalize_host_timing_api(str(record["api"]))
                except KeyError as exc:
                    raise ValueError(f"missing field {exc} in {trace_path}:{line_number}") from exc

                if step_window is not None:
                    window_start_ts, window_end_ts = step_window
                    if ts < window_start_ts or ts > window_end_ts:
                        continue
                if rank_base_ts is None or ts < rank_base_ts:
                    rank_base_ts = ts

                stats = thread_stats.get(tid)
                if stats is None:
                    stats = {
                        "start_ts": ts,
                        "end_ts": ts,
                        "first_api": api,
                        "event_count": 0,
                        "api_counts": defaultdict(int),
                        "sequence": [],
                    }
                    thread_stats[tid] = stats

                previous_start_ts = int(stats["start_ts"])
                stats["start_ts"] = min(previous_start_ts, ts)
                stats["end_ts"] = max(int(stats["end_ts"]), ts)
                stats["event_count"] = int(stats["event_count"]) + 1
                if ts < previous_start_ts:
                    stats["first_api"] = api
                if ts == previous_start_ts:
                    stats["first_api"] = api
                stats["api_counts"][api] += 1
                stats["sequence"].append((ts, api))

        if rank_base_ts is None:
            templates_by_rank[rank] = []
            continue

        helper_templates: list[_HelperThreadTemplate] = []
        for tid, stats in thread_stats.items():
            api_counts = {str(key): int(value) for key, value in stats["api_counts"].items()}
            if not _is_helper_thread(api_counts):
                continue
            first_api = str(stats["first_api"])
            dominant_api = max(api_counts.items(), key=lambda item: (item[1], item[0]))[0]
            api_sequence = tuple(
                api
                for _, api in sorted(
                    ((int(ts), str(api)) for ts, api in stats["sequence"]),
                    key=lambda item: (item[0], item[1]),
                )
            )
            helper_templates.append(
                _HelperThreadTemplate(
                    source_tid=tid,
                    start_offset_us=max(int(stats["start_ts"]) - int(rank_base_ts), 0),
                    end_offset_us=max(int(stats["end_ts"]) - int(rank_base_ts), 0),
                    first_api=first_api,
                    dominant_api=dominant_api,
                    event_count=int(stats["event_count"]),
                    api_sequence=api_sequence,
                )
            )

        helper_templates.sort(key=lambda item: (item.start_offset_us, item.source_tid))
        templates_by_rank[rank] = helper_templates

    return templates_by_rank


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a fake-cuda host timing profile from real rank traces"
    )
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-type",
        action="append",
        dest="include_types",
        default=None,
        help=(
            "Previous-event type to include when attributing dispatch gaps. "
            "May be passed multiple times. When omitted, all observed event types are used."
        ),
    )
    parser.add_argument(
        "--statistic",
        choices=["mean", "percentile"],
        default=DEFAULT_PROFILE_STATISTIC,
        help=(
            "Aggregation used when exporting synthetic host-delay entries. "
            "mean better preserves cumulative host time for long-tailed gaps; "
            "percentile preserves the legacy behavior."
        ),
    )
    parser.add_argument(
        "--profile-surface",
        choices=["supported", "semantic"],
        default="semantic",
        help=(
            "Which API surface advances host-dispatch timing state. "
            "'semantic' is the paper-facing default: it advances state only "
            "on semantic-traced APIs while accumulating gaps across "
            "intervening compat/control-plane APIs. 'supported' is retained "
            "for diagnostics."
        ),
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=50.0,
        help="Percentile used when --statistic=percentile (default: 50).",
    )
    parser.add_argument(
        "--min-api-samples",
        type=int,
        default=32,
        help="Minimum per-API sample count before exporting an API-specific entry.",
    )
    parser.add_argument(
        "--min-pair-samples",
        type=int,
        default=8,
        help="Minimum per-transition sample count before exporting a pair:<prev_api>-><api> entry.",
    )
    parser.add_argument(
        "--min-pair-occurrence-samples",
        type=int,
        default=DEFAULT_MIN_PAIR_OCCURRENCE_SAMPLES,
        help=(
            "Minimum sample count before exporting a pairocc:<prev_api>-><api>#<occurrence> "
            "entry for early per-transition occurrences. Occurrence-aware entries are intended "
            "for warmup-like transitions and should only be emitted when multiple independent "
            "observations agree."
        ),
    )
    parser.add_argument(
        "--max-pair-occurrence-index",
        type=int,
        default=DEFAULT_MAX_PAIR_OCCURRENCE_INDEX,
        help=(
            "Export pairocc entries for occurrence indices in [0, max_pair_occurrence_index). "
            "Use 0 or a negative value to export all observed occurrence indices. Large values "
            "can overfit an exact trace schedule, so the default intentionally keeps this small."
        ),
    )
    parser.add_argument(
        "--pair-occurrence-min-delta-us",
        type=float,
        default=DEFAULT_PAIR_OCCURRENCE_MIN_DELTA_US,
        help=(
            "Only export a pairocc override when its aggregated delay differs from the steady-state "
            "pair delay by at least this many microseconds. Use 0 together with "
            "--pair-occurrence-min-ratio 0 to export every observed occurrence."
        ),
    )
    parser.add_argument(
        "--pair-occurrence-min-ratio",
        type=float,
        default=DEFAULT_PAIR_OCCURRENCE_MIN_RATIO,
        help=(
            "Only export a pairocc override when its aggregated delay is at least this multiple "
            "of the steady-state pair delay. Use 0 together with "
            "--pair-occurrence-min-delta-us 0 to export every observed occurrence."
        ),
    )
    parser.add_argument(
        "--max-pair-occurrence-delay-us",
        type=float,
        default=DEFAULT_MAX_PAIR_OCCURRENCE_DELAY_US,
        help=(
            "Clamp exported pairocc override delays to this ceiling. Use 0 or a negative value "
            "to disable clamping."
        ),
    )
    parser.add_argument(
        "--rank",
        action="append",
        dest="ranks",
        type=int,
        default=None,
        help=(
            "Optional rank id to include when sampling host gaps. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--window-mode",
        choices=["auto", "full"],
        default="auto",
        help=(
            "auto limits sampling to per-rank step windows when capture markers/manifest are "
            "available; full uses the entire trace."
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON summary path for diagnostics.",
    )
    parser.add_argument(
        "--dispatch-scope",
        choices=["thread", "process", "host_machine"],
        default="host_machine",
        help=(
            "How host-dispatch gaps are attributed when building the timing profile. "
            "host_machine is the Maya paper default single dispatch queue per host; "
            "process and thread are diagnostic scopes."
        ),
    )
    return parser


def _iter_rank_trace_files(trace_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in trace_dir.glob("rank_*.jsonl")
        if not path.name.endswith(".markers.jsonl") and not path.name.endswith(".communicators.jsonl")
    )


def _rank_from_trace_path(trace_path: Path) -> int | None:
    stem = trace_path.stem
    if not stem.startswith("rank_"):
        return None
    try:
        return int(stem.split("_", 1)[1])
    except ValueError:
        return None


def _load_step_windows_from_manifest(trace_dir: Path) -> dict[int, tuple[int, int]]:
    manifest_path = trace_dir / "capture_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    raw_windows = payload.get("step_windows")
    if not isinstance(raw_windows, dict):
        raw_windows = payload.get("fidelity_windows")
    if not isinstance(raw_windows, dict):
        return {}
    resolved: dict[int, tuple[int, int]] = {}
    for raw_rank, raw_window in raw_windows.items():
        fidelity_window = fidelity_window_from_payload(raw_window, default_source="manifest")
        if fidelity_window is None or not fidelity_window.is_paper_valid_step_window:
            continue
        try:
            rank = int(raw_rank)
        except (TypeError, ValueError):
            continue
        resolved[rank] = (fidelity_window.start_ts, fidelity_window.end_ts)
    return resolved


def _load_step_windows_from_markers(trace_dir: Path) -> dict[int, tuple[int, int]]:
    resolved: dict[int, tuple[int, int]] = {}
    for trace_path in _iter_rank_trace_files(trace_dir):
        rank = _rank_from_trace_path(trace_path)
        if rank is None:
            continue
        markers_path = trace_path.with_suffix(".markers.jsonl")
        marker_records = load_step_markers(markers_path)
        if not marker_records:
            continue
        window = resolve_step_window_from_markers(marker_records, source=TraceSource.REAL)
        if window is None:
            continue
        resolved[rank] = (int(window["start_ts"]), int(window["end_ts"]))
    return resolved


def _load_step_windows(trace_dir: Path, *, window_mode: str) -> dict[int, tuple[int, int]]:
    if window_mode == "full":
        return {}
    windows = dict(_load_step_windows_from_manifest(trace_dir))
    marker_windows = _load_step_windows_from_markers(trace_dir)
    for rank, window in marker_windows.items():
        windows.setdefault(rank, window)
    return windows


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if percentile <= 0.0:
        return min(values)
    if percentile >= 100.0:
        return max(values)
    ordered = sorted(values)
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def collect_host_gap_samples(
    trace_dir: Path,
    *,
    include_types: Iterable[str] | None = None,
    step_windows: dict[int, tuple[int, int]] | None = None,
    ranks: set[int] | None = None,
    dispatch_scope: str = "host_machine",
    profile_surface: str = "semantic",
) -> tuple[
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    int,
    dict[str, list[float]],
    dict[str, list[float]],
]:
    pair_occurrence_samples: dict[str, list[float]] = defaultdict(list)
    pair_samples: dict[str, list[float]] = defaultdict(list)
    api_samples: dict[str, list[float]] = defaultdict(list)
    type_samples: dict[str, list[float]] = defaultdict(list)
    thread_start_occurrence_samples: dict[str, list[float]] = defaultdict(list)
    thread_start_samples: dict[str, list[float]] = defaultdict(list)
    included = set(include_types) if include_types is not None else None
    processed_events = 0

    for trace_path in _iter_rank_trace_files(trace_dir):
        rank = _rank_from_trace_path(trace_path)
        if ranks is not None and rank is not None and rank not in ranks:
            continue
        step_window = (
            step_windows.get(rank)
            if step_windows is not None and rank is not None
            else None
        )
        states: dict[int, _ThreadState] = {}
        pair_occurrence_counts_by_dispatch: dict[int, dict[str, int]] = defaultdict(dict)
        first_semantic_event_by_dispatch: dict[int, tuple[int, str]] = {}
        dispatch_high_watermark_ts: dict[int, int] = {}
        pending_host_gap_by_dispatch: dict[int, float] = defaultdict(float)
        rank_base_ts: int | None = step_window[0] if step_window is not None else None
        with trace_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = line.strip()
                if not payload:
                    continue
                record = json.loads(payload)
                try:
                    tid = int(record["tid"])
                    pid = int(record["pid"])
                    ts = int(record["ts"])
                    api = _canonicalize_host_timing_api(str(record["api"]))
                    event_type = str(record["type"])
                    mod = str(record["mod"])
                except KeyError as exc:
                    raise ValueError(f"missing field {exc} in {trace_path}:{line_number}") from exc
                if step_window is not None:
                    window_start_ts, window_end_ts = step_window
                    if ts < window_start_ts or ts > window_end_ts:
                        continue
                processed_events += 1
                if rank_base_ts is None or ts < rank_base_ts:
                    rank_base_ts = ts
                if dispatch_scope == "thread":
                    dispatch_key = tid
                elif dispatch_scope == "process":
                    dispatch_key = pid
                else:
                    dispatch_key = 0
                previous_high_watermark_ts = dispatch_high_watermark_ts.get(dispatch_key)
                if previous_high_watermark_ts is not None:
                    pending_host_gap_by_dispatch[dispatch_key] += max(
                        float(ts - previous_high_watermark_ts),
                        0.0,
                    )
                dispatch_high_watermark_ts[dispatch_key] = max(
                    dispatch_high_watermark_ts.get(dispatch_key, ts),
                    ts,
                )
                if not _is_host_timing_profile_event(
                    api,
                    event_type,
                    profile_surface=profile_surface,
                ):
                    continue

                previous = states.get(dispatch_key)
                existing_first_semantic = first_semantic_event_by_dispatch.get(dispatch_key)
                if existing_first_semantic is None or ts < existing_first_semantic[0]:
                    first_semantic_event_by_dispatch[dispatch_key] = (ts, api)
                if previous is not None:
                    gap_us = pending_host_gap_by_dispatch[dispatch_key]
                    if included is None or previous.previous_type in included:
                        pair_key = f"pair:{previous.previous_api}->{api}"
                        pair_samples[pair_key].append(gap_us)
                        pair_occurrence_counts = pair_occurrence_counts_by_dispatch[dispatch_key]
                        pair_occurrence_index = pair_occurrence_counts.get(pair_key, 0)
                        pair_occurrence_samples[
                            f"pairocc:{previous.previous_api}->{api}#{pair_occurrence_index}"
                        ].append(gap_us)
                        pair_occurrence_counts[pair_key] = pair_occurrence_index + 1
                        api_samples[previous.previous_api].append(gap_us)
                        type_samples[previous.previous_type].append(gap_us)
                pending_host_gap_by_dispatch[dispatch_key] = 0.0
                states[dispatch_key] = _ThreadState(
                    previous_api=api,
                    previous_type=event_type,
                    previous_mod=mod,
                    dispatch_high_watermark_ts=dispatch_high_watermark_ts[dispatch_key],
                )

        if rank_base_ts is not None:
            thread_start_counts_by_api: dict[str, int] = {}
            for dispatch_key, (first_ts, first_api) in sorted(
                first_semantic_event_by_dispatch.items(),
                key=lambda item: (item[1][0], item[0]),
            ):
                del dispatch_key
                start_gap_us = max(float(first_ts - rank_base_ts), 0.0)
                occurrence_index = thread_start_counts_by_api.get(first_api, 0)
                thread_start_occurrence_samples[
                    f"threadstartocc:{first_api}#{occurrence_index}"
                ].append(start_gap_us)
                thread_start_samples[f"threadstart:{first_api}"].append(start_gap_us)
                thread_start_counts_by_api[first_api] = occurrence_index + 1

    return (
        pair_occurrence_samples,
        pair_samples,
        api_samples,
        type_samples,
        processed_events,
        thread_start_occurrence_samples,
        thread_start_samples,
    )


def _aggregate_samples(
    values: list[float],
    *,
    statistic: str,
    percentile: float,
) -> float:
    if statistic == "mean":
        return (sum(values) / len(values)) if values else 0.0
    return _percentile(values, percentile)


def build_profile_lines(
    pair_occurrence_samples: dict[str, list[float]],
    pair_samples: dict[str, list[float]],
    api_samples: dict[str, list[float]],
    type_samples: dict[str, list[float]],
    thread_start_occurrence_samples: dict[str, list[float]],
    thread_start_samples: dict[str, list[float]],
    *,
    statistic: str,
    percentile: float,
    max_pair_occurrence_index: int,
    min_pair_occurrence_samples: int,
    pair_occurrence_min_delta_us: float,
    pair_occurrence_min_ratio: float,
    max_pair_occurrence_delay_us: float,
    min_pair_samples: int,
    min_api_samples: int,
) -> list[str]:
    lines = [
        "# fake-cuda host timing profile",
        "# values are synthetic host dispatch delays in microseconds",
        "# threadstartocc:<api>#<occurrence> entries model per-thread first semantic-dispatch offsets",
        "# threadstart:<api> entries provide coarse per-thread first semantic-dispatch fallbacks",
        "# pairocc:<prev_api>-><api>#<occurrence> entries model early semantic-transition stalls",
        "# pair:<prev_api>-><api> entries model semantic host-delay between semantic events",
        "default=0",
    ]
    for thread_start_occurrence_key in sorted(thread_start_occurrence_samples):
        samples = thread_start_occurrence_samples[thread_start_occurrence_key]
        if not samples:
            continue
        value = _aggregate_samples(samples, statistic=statistic, percentile=percentile)
        lines.append(f"{thread_start_occurrence_key}={value:.6f}")
    for thread_start_key in sorted(thread_start_samples):
        samples = thread_start_samples[thread_start_key]
        if not samples:
            continue
        value = _aggregate_samples(samples, statistic=statistic, percentile=percentile)
        lines.append(f"{thread_start_key}={value:.6f}")
    pair_aggregates = {
        pair_key: _aggregate_samples(samples, statistic=statistic, percentile=percentile)
        for pair_key, samples in pair_samples.items()
    }
    for pair_occurrence_key in sorted(pair_occurrence_samples):
        if max_pair_occurrence_index > 0:
            _, _, occurrence_index_text = pair_occurrence_key.rpartition("#")
            try:
                occurrence_index = int(occurrence_index_text)
            except ValueError:
                occurrence_index = max_pair_occurrence_index + 1
            if occurrence_index >= max_pair_occurrence_index:
                continue
        samples = pair_occurrence_samples[pair_occurrence_key]
        if len(samples) < min_pair_occurrence_samples:
            continue
        occurrence_value = _aggregate_samples(samples, statistic=statistic, percentile=percentile)
        pair_identity, _, _ = pair_occurrence_key.partition("#")
        pair_key = f"pair:{pair_identity[len('pairocc:'):]}"
        steady_state_value = pair_aggregates.get(pair_key)
        if pair_occurrence_min_delta_us <= 0.0 and pair_occurrence_min_ratio <= 0.0:
            materially_differs = True
        elif steady_state_value is not None:
            if steady_state_value <= 0.0:
                materially_differs = occurrence_value >= pair_occurrence_min_delta_us
            else:
                materially_differs = (
                    (occurrence_value - steady_state_value) >= pair_occurrence_min_delta_us
                    and occurrence_value >= (steady_state_value * pair_occurrence_min_ratio)
                )
        else:
            materially_differs = True
        if not materially_differs:
            continue
        if max_pair_occurrence_delay_us > 0.0:
            occurrence_value = min(occurrence_value, max_pair_occurrence_delay_us)
        lines.append(f"{pair_occurrence_key}={occurrence_value:.6f}")
    for pair_key in sorted(pair_samples):
        samples = pair_samples[pair_key]
        if len(samples) < min_pair_samples:
            continue
        lines.append(f"{pair_key}={pair_aggregates[pair_key]:.6f}")
    for event_type in sorted(type_samples):
        lines.append(
            f"type:{event_type}={_aggregate_samples(type_samples[event_type], statistic=statistic, percentile=percentile):.6f}"
        )
    for api in sorted(api_samples):
        samples = api_samples[api]
        if len(samples) < min_api_samples:
            continue
        lines.append(
            f"{api}={_aggregate_samples(samples, statistic=statistic, percentile=percentile):.6f}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    include_types = tuple(args.include_types) if args.include_types else None
    ranks = set(args.ranks) if args.ranks else None
    trace_dir = args.trace_dir.resolve()
    step_windows = _load_step_windows(trace_dir, window_mode=args.window_mode)
    (
        pair_occurrence_samples,
        pair_samples,
        api_samples,
        type_samples,
        processed_events,
        thread_start_occurrence_samples,
        thread_start_samples,
    ) = collect_host_gap_samples(
        trace_dir,
        include_types=include_types,
        step_windows=step_windows,
        ranks=ranks,
        dispatch_scope=args.dispatch_scope,
        profile_surface=args.profile_surface,
    )
    helper_thread_templates = _collect_helper_thread_templates(
        trace_dir,
        step_windows=step_windows,
        ranks=ranks,
    )
    lines = build_profile_lines(
        pair_occurrence_samples,
        pair_samples,
        api_samples,
        type_samples,
        thread_start_occurrence_samples,
        thread_start_samples,
        statistic=args.statistic,
        percentile=args.percentile,
        max_pair_occurrence_index=args.max_pair_occurrence_index,
        min_pair_occurrence_samples=args.min_pair_occurrence_samples,
        pair_occurrence_min_delta_us=args.pair_occurrence_min_delta_us,
        pair_occurrence_min_ratio=args.pair_occurrence_min_ratio,
        max_pair_occurrence_delay_us=args.max_pair_occurrence_delay_us,
        min_pair_samples=args.min_pair_samples,
        min_api_samples=args.min_api_samples,
    )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.summary_json is not None:
        summary = {
            "trace_dir": str(trace_dir),
            "include_types": list(include_types) if include_types is not None else None,
            "ranks": sorted(ranks) if ranks is not None else None,
            "statistic": args.statistic,
            "processed_events": processed_events,
            "window_mode": args.window_mode,
            "dispatch_scope": args.dispatch_scope,
            "profile_surface": args.profile_surface,
            "step_window_rank_count": len(step_windows),
            "pair_occurrence_sample_counts": {
                pair_occurrence_key: len(samples)
                for pair_occurrence_key, samples in sorted(pair_occurrence_samples.items())
            },
            "pair_sample_counts": {
                pair_key: len(samples) for pair_key, samples in sorted(pair_samples.items())
            },
            "api_sample_counts": {api: len(samples) for api, samples in sorted(api_samples.items())},
            "type_sample_counts": {
                event_type: len(samples) for event_type, samples in sorted(type_samples.items())
            },
            "thread_start_occurrence_sample_counts": {
                key: len(samples)
                for key, samples in sorted(thread_start_occurrence_samples.items())
            },
            "thread_start_sample_counts": {
                key: len(samples)
                for key, samples in sorted(thread_start_samples.items())
            },
            "percentile": args.percentile,
            "min_pair_occurrence_samples": args.min_pair_occurrence_samples,
            "max_pair_occurrence_index": args.max_pair_occurrence_index,
            "pair_occurrence_min_delta_us": args.pair_occurrence_min_delta_us,
            "pair_occurrence_min_ratio": args.pair_occurrence_min_ratio,
            "max_pair_occurrence_delay_us": args.max_pair_occurrence_delay_us,
            "min_pair_samples": args.min_pair_samples,
            "min_api_samples": args.min_api_samples,
            "output": str(output_path),
            "helper_thread_templates_by_rank": {
                str(rank): [
                    {
                        "source_tid": template.source_tid,
                        "start_offset_us": template.start_offset_us,
                        "end_offset_us": template.end_offset_us,
                        "first_api": template.first_api,
                        "dominant_api": template.dominant_api,
                        "event_count": template.event_count,
                        "api_sequence": list(template.api_sequence),
                    }
                    for template in templates
                    if template.first_api in _SYNTHETIC_HELPER_START_APIS
                ]
                for rank, templates in sorted(helper_thread_templates.items())
            },
        }
        args.summary_json.resolve().write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
