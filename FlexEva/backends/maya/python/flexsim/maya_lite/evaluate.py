"""
Batch black-box evaluation for Maya-lite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from flexsim.estimator import Estimator

from .annotate import annotate_collated_trace, collective_group_duration_summary
from .collate import collate_trace_bundle
from .host_delay_profile import HostDelayProfile
from .io import (
    dedup_identical_rank_traces,
    dedup_pattern_rank_traces,
    load_trace_directory,
    materialize_profiled_rank_traces,
)
from .replay import replay_annotated_trace
from .schema import CandidateEvaluation, CollatedTrace, ReplayResult


def build_candidate_evaluation(
    collated: CollatedTrace,
    replay: ReplayResult,
    *,
    candidate_name: str | None = None,
    expand_profiled_rank_groups: bool = False,
    annotation_diagnostics: dict[str, object] | None = None,
) -> CandidateEvaluation:
    """Convert a replay result into the standard Maya-lite candidate report."""
    if replay.rank_metrics:
        weighted_utilization = 0.0
        weighted_count = 0
        for metric in replay.rank_metrics:
            multiplier = 1
            if not expand_profiled_rank_groups and not collated.logical_rank_materialized:
                multiplier = len(collated.profiled_rank_groups.get(metric.rank, (metric.rank,)))
            weighted_utilization += metric.utilization * multiplier
            weighted_count += multiplier
        average_utilization = weighted_utilization / weighted_count if weighted_count else 0.0
    else:
        average_utilization = 0.0
    return CandidateEvaluation(
        candidate_name=candidate_name or Path(collated.trace_dir).name,
        trace_dir=Path(collated.trace_dir),
        source=collated.source,
        world_size=collated.world_size,
        profiled_world_size=collated.profiled_world_size,
        trace_window=collated.trace_window,
        paper_valid_step_window_rank_count=sum(
            1
            for fidelity_window in collated.fidelity_windows.values()
            if fidelity_window.is_paper_valid_step_window
        ),
        step_window_sources=tuple(
            sorted({fidelity_window.source for fidelity_window in collated.fidelity_windows.values()})
        ),
        total_events=collated.total_events,
        total_time_us=replay.total_time_us,
        critical_path_us=replay.critical_path_us,
        global_makespan_us=replay.global_makespan_us,
        rank0_time_us=replay.rank0_time_us,
        average_utilization=average_utilization,
        rank_metrics=replay.rank_metrics,
        profiled_rank_groups=dict(collated.profiled_rank_groups),
        annotation_diagnostics=dict(annotation_diagnostics or {}),
    )


def evaluate_collated_trace(
    collated: CollatedTrace,
    estimator: Estimator,
    *,
    candidate_name: str | None = None,
    expand_profiled_rank_groups: bool = False,
    percentile: str = "p50",
    allow_kernel_launch_heuristic_fallback: bool = False,
    allow_weak_runtime_fallback: bool = False,
    use_observed_control_plane_wrapper_durations: bool = True,
    use_observed_semantic_wrapper_durations: bool = True,
    host_delay_profile: HostDelayProfile | None = None,
    predictor_workers: int = 1,
) -> CandidateEvaluation:
    """Run annotation + replay for an already-collated low-level trace."""
    annotated = annotate_collated_trace(
        collated,
        estimator,
        percentile=percentile,
        allow_kernel_launch_heuristic_fallback=allow_kernel_launch_heuristic_fallback,
        allow_weak_runtime_fallback=allow_weak_runtime_fallback,
        use_observed_control_plane_wrapper_durations=use_observed_control_plane_wrapper_durations,
        use_observed_semantic_wrapper_durations=use_observed_semantic_wrapper_durations,
        host_delay_profile=host_delay_profile,
        parallel_workers=predictor_workers,
    )
    replay = replay_annotated_trace(
        annotated,
        expand_profiled_rank_groups=expand_profiled_rank_groups,
        record_simulated_events=False,
    )
    return build_candidate_evaluation(
        collated,
        replay,
        candidate_name=candidate_name,
        expand_profiled_rank_groups=expand_profiled_rank_groups,
        annotation_diagnostics=collective_group_duration_summary(annotated),
    )


def evaluate_trace_directory(
    trace_dir: str | Path,
    estimator: Estimator,
    *,
    candidate_name: str | None = None,
    max_events_per_rank: int | None = None,
    trace_window: str = "auto",
    dedup_identical_ranks: bool = False,
    dedup_pattern_ranks: bool = False,
    materialize_logical_ranks: bool = False,
    expand_profiled_rank_groups: bool = False,
    percentile: str = "p50",
    allow_kernel_launch_heuristic_fallback: bool = False,
    allow_weak_runtime_fallback: bool = False,
    use_observed_control_plane_wrapper_durations: bool = True,
    use_observed_semantic_wrapper_durations: bool = True,
    host_delay_profile: HostDelayProfile | None = None,
    predictor_workers: int = 1,
) -> CandidateEvaluation:
    """Run the full Maya-lite pipeline for one low-level trace directory."""
    bundle = load_trace_directory(
        trace_dir,
        max_events_per_rank=max_events_per_rank,
        trace_window=trace_window,
    )
    if dedup_identical_ranks:
        bundle = dedup_identical_rank_traces(bundle)
    if dedup_pattern_ranks:
        bundle = dedup_pattern_rank_traces(bundle)
    if materialize_logical_ranks:
        bundle = materialize_profiled_rank_traces(bundle)
    collated = collate_trace_bundle(bundle)
    return evaluate_collated_trace(
        collated,
        estimator,
        candidate_name=candidate_name or Path(trace_dir).name,
        expand_profiled_rank_groups=expand_profiled_rank_groups,
        percentile=percentile,
        allow_kernel_launch_heuristic_fallback=allow_kernel_launch_heuristic_fallback,
        allow_weak_runtime_fallback=allow_weak_runtime_fallback,
        use_observed_control_plane_wrapper_durations=use_observed_control_plane_wrapper_durations,
        use_observed_semantic_wrapper_durations=use_observed_semantic_wrapper_durations,
        host_delay_profile=host_delay_profile,
        predictor_workers=predictor_workers,
    )


def evaluate_candidate_set(
    candidates: dict[str, str | Path] | Iterable[tuple[str, str | Path]],
    estimator: Estimator,
    *,
    max_events_per_rank: int | None = None,
    trace_window: str = "auto",
    dedup_identical_ranks: bool = False,
    dedup_pattern_ranks: bool = False,
    materialize_logical_ranks: bool = False,
    expand_profiled_rank_groups: bool = False,
    percentile: str = "p50",
    allow_kernel_launch_heuristic_fallback: bool = False,
    allow_weak_runtime_fallback: bool = False,
    use_observed_control_plane_wrapper_durations: bool = True,
    use_observed_semantic_wrapper_durations: bool = True,
    host_delay_profile: HostDelayProfile | None = None,
    predictor_workers: int = 1,
) -> list[CandidateEvaluation]:
    """Evaluate multiple candidates independently, Maya-style."""
    if isinstance(candidates, dict):
        items = candidates.items()
    else:
        items = list(candidates)

    results = [
        evaluate_trace_directory(
            trace_dir,
            estimator,
            candidate_name=name,
            max_events_per_rank=max_events_per_rank,
            trace_window=trace_window,
            dedup_identical_ranks=dedup_identical_ranks,
            dedup_pattern_ranks=dedup_pattern_ranks,
            materialize_logical_ranks=materialize_logical_ranks,
            expand_profiled_rank_groups=expand_profiled_rank_groups,
            percentile=percentile,
            allow_kernel_launch_heuristic_fallback=allow_kernel_launch_heuristic_fallback,
            allow_weak_runtime_fallback=allow_weak_runtime_fallback,
            use_observed_control_plane_wrapper_durations=use_observed_control_plane_wrapper_durations,
            use_observed_semantic_wrapper_durations=use_observed_semantic_wrapper_durations,
            host_delay_profile=host_delay_profile,
            predictor_workers=predictor_workers,
        )
        for name, trace_dir in items
    ]
    return sorted(results, key=lambda result: result.total_time_us)
