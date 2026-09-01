"""
Step-sliced real-vs-emulated trace comparison for the current Figure 13 workload.

The metrics here are intentionally low-level. They validate whether the dry-run
emulation phase emits a similar step trace shape to the real CUDA/NCCL trace on
the same small realized workload.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from .filters import is_compat_only_api, is_semantic_traced_api
from .io import load_trace_directory
from .schema import RankTrace, TraceBundle, TraceEvent

_MAX_EXACT_LCS_SEQUENCE_CELLS = 4_000_000


def _api_frequency(events: list[TraceEvent]) -> Counter[str]:
    return Counter(event.api for event in events)


def _cosine_similarity(vec_a: Counter[str], vec_b: Counter[str]) -> float:
    all_keys = set(vec_a) | set(vec_b)
    if not all_keys:
        return 1.0
    dot = sum(vec_a.get(key, 0) * vec_b.get(key, 0) for key in all_keys)
    mag_a = math.sqrt(sum(value * value for value in vec_a.values()))
    mag_b = math.sqrt(sum(value * value for value in vec_b.values()))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _api_coverage(real_freq: Counter[str], emulated_freq: Counter[str]) -> float:
    if not real_freq:
        return 1.0
    covered = sum(1 for api in real_freq if api in emulated_freq)
    return covered / len(real_freq)


def _nccl_sequence(events: list[TraceEvent]) -> list[str]:
    return [event.api for event in events if event.op_type == "nccl_collective"]


def _sequence_match_ratio(seq_a: list[str], seq_b: list[str]) -> float:
    if not seq_a and not seq_b:
        return 1.0
    if not seq_a or not seq_b:
        return 0.0
    m, n = len(seq_a), len(seq_b)
    if m * n > _MAX_EXACT_LCS_SEQUENCE_CELLS:
        matches = sum(1 for left, right in zip(seq_a, seq_b) if left == right)
        return 2.0 * matches / (m + n)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs_len = prev[n]
    return 2.0 * lcs_len / (m + n)


def _bundle_events(bundle: TraceBundle) -> list[TraceEvent]:
    return [event for rank_trace in bundle.rank_traces for event in rank_trace.events]


def _rank_trace_map(bundle: TraceBundle) -> dict[int, RankTrace]:
    return {rank_trace.rank: rank_trace for rank_trace in bundle.rank_traces}


def _project_bundle_to_ranks(bundle: TraceBundle, rank_ids: list[int] | tuple[int, ...]) -> TraceBundle:
    wanted = {int(rank) for rank in rank_ids}
    projected_rank_traces = tuple(
        rank_trace
        for rank_trace in bundle.rank_traces
        if rank_trace.rank in wanted
    )
    projected_groups = {
        int(rank): tuple(int(member) for member in members)
        for rank, members in bundle.profiled_rank_groups.items()
        if int(rank) in wanted
    }
    projected_world_size = len(projected_rank_traces)
    return TraceBundle(
        trace_dir=bundle.trace_dir,
        source=bundle.source,
        rank_traces=projected_rank_traces,
        original_world_size=projected_world_size,
        captured_world_size=projected_world_size,
        profiled_rank_groups=projected_groups,
        rank_host_machines={
            int(rank): host_machine_id
            for rank, host_machine_id in bundle.rank_host_machines.items()
            if int(rank) in wanted
        },
        rank_host_dispatch_queues={
            int(rank): host_dispatch_queue_id
            for rank, host_dispatch_queue_id in bundle.rank_host_dispatch_queues.items()
            if int(rank) in wanted
        },
        communicator_memberships=dict(bundle.communicator_memberships),
        host_timing_dispatch_scope_resolved=bundle.host_timing_dispatch_scope_resolved,
        step_windows={
            int(rank): tuple(window)
            for rank, window in bundle.step_windows.items()
            if int(rank) in wanted
        },
        fidelity_windows={
            int(rank): bundle.fidelity_windows[rank]
            for rank in bundle.fidelity_windows
            if int(rank) in wanted
        },
        logical_rank_materialized=bundle.logical_rank_materialized,
        trace_window=bundle.trace_window,
    )


def _representative_alignment_view(
    real_bundle: TraceBundle,
    emulated_bundle: TraceBundle,
) -> tuple[TraceBundle, TraceBundle, dict[str, object]] | None:
    candidates = (
        ("emulated", emulated_bundle, "real", real_bundle),
        ("real", real_bundle, "emulated", emulated_bundle),
    )
    for compact_label, compact_bundle, full_label, full_bundle in candidates:
        if compact_bundle.profiled_world_size >= compact_bundle.world_size:
            continue
        if not compact_bundle.profiled_rank_groups:
            continue
        representative_ranks = tuple(sorted(int(rank) for rank in compact_bundle.profiled_rank_groups))
        if not representative_ranks:
            continue
        full_rank_ids = set(full_bundle.rank_ids())
        if not set(representative_ranks).issubset(full_rank_ids):
            continue
        projected_full_bundle = _project_bundle_to_ranks(full_bundle, representative_ranks)
        return (
            projected_full_bundle if full_label == "real" else compact_bundle,
            compact_bundle if compact_label == "emulated" else projected_full_bundle,
            {
                "mode": "representative_projection",
                "compact_side": compact_label,
                "full_side": full_label,
                "representative_ranks": list(representative_ranks),
                "compact_profiled_world_size": compact_bundle.profiled_world_size,
                "compact_world_size": compact_bundle.world_size,
            },
        )
    return None


def _top_api_comparison(
    real_freq: Counter[str],
    emulated_freq: Counter[str],
    *,
    limit: int = 20,
) -> dict[str, dict[str, int]]:
    ranked = sorted(
        set(real_freq) | set(emulated_freq),
        key=lambda api: max(real_freq.get(api, 0), emulated_freq.get(api, 0)),
        reverse=True,
    )
    return {
        api: {
            "real": real_freq.get(api, 0),
            "emulated": emulated_freq.get(api, 0),
        }
        for api in ranked[:limit]
    }


def _semantic_events(events: list[TraceEvent]) -> list[TraceEvent]:
    return [
        event
        for event in events
        if is_semantic_traced_api(event.api, event.op_type)
    ]


def _stream_event_events(events: list[TraceEvent]) -> list[TraceEvent]:
    return [
        event
        for event in events
        if event.api.startswith("cudaStream") or event.api.startswith("cudaEvent")
    ]


def _compat_only_events(events: list[TraceEvent]) -> list[TraceEvent]:
    return [
        event
        for event in events
        if is_compat_only_api(event.api, event.op_type)
    ]


def _metric_block(real_events: list[TraceEvent], emulated_events: list[TraceEvent]) -> dict[str, object]:
    real_freq = _api_frequency(real_events)
    emulated_freq = _api_frequency(emulated_events)
    real_nccl = _nccl_sequence(real_events)
    emulated_nccl = _nccl_sequence(emulated_events)
    return {
        "api_cosine_similarity": round(_cosine_similarity(real_freq, emulated_freq), 6),
        "api_coverage": round(_api_coverage(real_freq, emulated_freq), 6),
        "nccl_sequence_match": round(_sequence_match_ratio(real_nccl, emulated_nccl), 6),
        "real_total_calls": len(real_events),
        "emulated_total_calls": len(emulated_events),
        "call_count_ratio": round(len(emulated_events) / len(real_events), 6)
        if real_events
        else 0.0,
        "real_only_apis": sorted(set(real_freq) - set(emulated_freq)),
        "emulated_only_apis": sorted(set(emulated_freq) - set(real_freq)),
        "top_api_comparison": _top_api_comparison(real_freq, emulated_freq),
    }


def _events_excluding_apis(events: list[TraceEvent], excluded_apis: set[str]) -> list[TraceEvent]:
    return [event for event in events if event.api not in excluded_apis]


def _fig13_normalized_semantic_metrics(
    real_events: list[TraceEvent],
    emulated_events: list[TraceEvent],
) -> dict[str, object]:
    launch_config_apis = {"__cudaPopCallConfiguration", "__cudaPushCallConfiguration"}
    allocator_jitter_apis = {"cudaMalloc"}
    host_progress_query_apis = {"cudaEventQuery"}

    without_launch_config = _metric_block(
        _events_excluding_apis(real_events, launch_config_apis),
        _events_excluding_apis(emulated_events, launch_config_apis),
    )
    without_launch_config["excluded_apis"] = sorted(launch_config_apis)
    without_launch_config["policy"] = "diagnostic_only"

    without_launch_config_and_allocator_jitter = _metric_block(
        _events_excluding_apis(real_events, launch_config_apis | allocator_jitter_apis),
        _events_excluding_apis(emulated_events, launch_config_apis | allocator_jitter_apis),
    )
    without_launch_config_and_allocator_jitter["excluded_apis"] = sorted(
        launch_config_apis | allocator_jitter_apis
    )
    without_launch_config_and_allocator_jitter["policy"] = "diagnostic_only"

    core_workload_nccl = _metric_block(
        _events_excluding_apis(
            real_events,
            launch_config_apis | allocator_jitter_apis | host_progress_query_apis,
        ),
        _events_excluding_apis(
            emulated_events,
            launch_config_apis | allocator_jitter_apis | host_progress_query_apis,
        ),
    )
    core_workload_nccl["excluded_apis"] = sorted(
        launch_config_apis | allocator_jitter_apis | host_progress_query_apis
    )
    core_workload_nccl["policy"] = "diagnostic_only"

    return {
        "without_launch_config": without_launch_config,
        "without_launch_config_and_allocator_jitter": without_launch_config_and_allocator_jitter,
        "core_workload_nccl": core_workload_nccl,
        "interpretation": (
            "Diagnostic Figure 13 views that remove known emulator/driver boundary noise. "
            "core_workload_nccl excludes launch-configuration shims, allocator jitter, "
            "and host-progress cudaEventQuery polling to expose model-shape and "
            "NCCL-sequence conformance."
        ),
    }


def _side_summary(bundle: TraceBundle) -> dict[str, object]:
    rank_ids = list(bundle.rank_ids())
    observed_apis = sorted({event.api for event in _bundle_events(bundle)})
    observed_nccl = sorted({event.api for event in _bundle_events(bundle) if event.api.startswith("nccl")})
    return {
        "trace_dir": str(bundle.trace_dir),
        "trace_window": bundle.trace_window,
        "paper_valid_step_window_rank_count": sum(
            1
            for fidelity_window in bundle.fidelity_windows.values()
            if fidelity_window.is_paper_valid_step_window
        ),
        "step_window_sources": sorted(
            {fidelity_window.source for fidelity_window in bundle.fidelity_windows.values()}
        ),
        "world_size": bundle.world_size,
        "profiled_world_size": bundle.profiled_world_size,
        "rank_ids": rank_ids,
        "total_events": bundle.total_events,
        "observed_apis": observed_apis,
        "observed_nccl_apis": observed_nccl,
    }


def compare_fig13_step_trace_dirs(
    real_trace_dir: Path,
    emulated_trace_dir: Path,
    *,
    max_events_per_rank: int | None = None,
    pre_step_context_us: int = 0,
) -> dict[str, object]:
    real_bundle = load_trace_directory(
        real_trace_dir,
        max_events_per_rank=max_events_per_rank,
        trace_window="step",
    )
    emulated_bundle = load_trace_directory(
        emulated_trace_dir,
        max_events_per_rank=max_events_per_rank,
        trace_window="step",
    )

    real_events = _bundle_events(real_bundle)
    emulated_events = _bundle_events(emulated_bundle)
    real_freq = _api_frequency(real_events)
    emulated_freq = _api_frequency(emulated_events)
    real_nccl = _nccl_sequence(real_events)
    emulated_nccl = _nccl_sequence(emulated_events)
    semantic_real_events = _semantic_events(real_events)
    semantic_emulated_events = _semantic_events(emulated_events)
    compat_real_events = _compat_only_events(real_events)
    compat_emulated_events = _compat_only_events(emulated_events)
    representative_alignment = _representative_alignment_view(real_bundle, emulated_bundle)
    representative_aligned_metrics: dict[str, object] | None = None
    representative_aligned_semantic_metrics: dict[str, object] | None = None
    representative_alignment_info: dict[str, object] | None = None
    if representative_alignment is not None:
        aligned_real_bundle, aligned_emulated_bundle, representative_alignment_info = representative_alignment
        aligned_real_events = _bundle_events(aligned_real_bundle)
        aligned_emulated_events = _bundle_events(aligned_emulated_bundle)
        representative_aligned_metrics = _metric_block(aligned_real_events, aligned_emulated_events)
        representative_aligned_semantic_metrics = _metric_block(
            _semantic_events(aligned_real_events),
            _semantic_events(aligned_emulated_events),
        )
    context_metrics: dict[str, object] | None = None
    stream_event_metrics: dict[str, object] | None = None
    if pre_step_context_us > 0:
        real_context_bundle = load_trace_directory(
            real_trace_dir,
            max_events_per_rank=max_events_per_rank,
            trace_window="step",
            step_pre_padding_us=pre_step_context_us,
        )
        emulated_context_bundle = load_trace_directory(
            emulated_trace_dir,
            max_events_per_rank=max_events_per_rank,
            trace_window="step",
            step_pre_padding_us=pre_step_context_us,
        )
        context_metrics = _metric_block(
            _bundle_events(real_context_bundle),
            _bundle_events(emulated_context_bundle),
        )
        context_metrics["pre_step_context_us"] = int(pre_step_context_us)
        stream_event_metrics = _metric_block(
            _stream_event_events(_bundle_events(real_context_bundle)),
            _stream_event_events(_bundle_events(emulated_context_bundle)),
        )
        stream_event_metrics["pre_step_context_us"] = int(pre_step_context_us)

    per_rank: list[dict[str, object]] = []
    real_rank_map = _rank_trace_map(real_bundle)
    emulated_rank_map = _rank_trace_map(emulated_bundle)
    all_ranks = sorted(set(real_rank_map) | set(emulated_rank_map))
    for rank in all_ranks:
        real_rank_events = list(real_rank_map.get(rank, RankTrace(rank, Path(), real_bundle.source, ())).events)
        emulated_rank_events = list(
            emulated_rank_map.get(rank, RankTrace(rank, Path(), emulated_bundle.source, ())).events
        )
        real_rank_freq = _api_frequency(real_rank_events)
        emulated_rank_freq = _api_frequency(emulated_rank_events)
        per_rank.append(
            {
                "rank": rank,
                "real_events": len(real_rank_events),
                "emulated_events": len(emulated_rank_events),
                "api_cosine_similarity": round(
                    _cosine_similarity(real_rank_freq, emulated_rank_freq), 6
                ),
                "api_coverage": round(_api_coverage(real_rank_freq, emulated_rank_freq), 6),
                "nccl_sequence_match": round(
                    _sequence_match_ratio(
                        _nccl_sequence(real_rank_events),
                        _nccl_sequence(emulated_rank_events),
                    ),
                    6,
                ),
            }
        )

    real_only_apis = sorted(set(real_freq) - set(emulated_freq))
    emulated_only_apis = sorted(set(emulated_freq) - set(real_freq))
    semantic_metrics = _metric_block(semantic_real_events, semantic_emulated_events)
    compat_only_metrics = _metric_block(compat_real_events, compat_emulated_events)
    compat_only_metrics["policy"] = "diagnostic_only"
    return {
        "real": _side_summary(real_bundle),
        "emulated": _side_summary(emulated_bundle),
        "global_metrics": {
            "api_cosine_similarity": round(_cosine_similarity(real_freq, emulated_freq), 6),
            "api_coverage": round(_api_coverage(real_freq, emulated_freq), 6),
            "nccl_sequence_match": round(_sequence_match_ratio(real_nccl, emulated_nccl), 6),
            "real_total_calls": len(real_events),
            "emulated_total_calls": len(emulated_events),
            "call_count_ratio": round(len(emulated_events) / len(real_events), 6)
            if real_events
            else 0.0,
        },
        "real_only_apis": real_only_apis,
        "emulated_only_apis": emulated_only_apis,
        "top_api_comparison": _top_api_comparison(real_freq, emulated_freq),
        "semantic_metrics": semantic_metrics,
        "conformance_metrics": semantic_metrics,
        "normalized_semantic_metrics": _fig13_normalized_semantic_metrics(
            semantic_real_events,
            semantic_emulated_events,
        ),
        "compat_only_metrics": compat_only_metrics,
        "representative_alignment": representative_alignment_info,
        "representative_aligned_metrics": representative_aligned_metrics,
        "representative_aligned_semantic_metrics": representative_aligned_semantic_metrics,
        "pre_step_context_metrics": context_metrics,
        "stream_event_metrics": stream_event_metrics,
        "per_rank": per_rank,
    }
