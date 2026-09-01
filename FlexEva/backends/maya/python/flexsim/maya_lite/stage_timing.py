"""
Stage-timing helpers for Maya-lite Figure 13 style measurements.

This separates Maya-lite's post-emulation stack into the same paper-facing
stages used in Figure 13:

1. emulator
2. collator
3. predictor
4. simulator

For paper-facing reporting we treat the Emulator stage as the capture
wall-clock runtime. We still carry active-emulation seconds separately as
diagnostic metadata, especially because representative-trace dedup can make
active time much smaller than the real full capture wall-clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from flexsim.estimator import Estimator

from .annotate import (
    AnnotationTimingRecorder,
    annotate_collated_trace,
    collective_group_duration_summary,
)
from .collate import collate_trace_bundle
from .evaluate import build_candidate_evaluation
from .host_delay_profile import HostDelayProfile, HostGapProfile
from .io import (
    dedup_identical_rank_traces,
    dedup_pattern_rank_traces,
    load_capture_manifest,
    load_trace_directory,
    materialize_profiled_rank_traces,
)
from .replay import replay_annotated_trace
from .schema import CandidateEvaluation, TraceBundle


@dataclass(frozen=True)
class StageTimingBreakdown:
    """Candidate evaluation with per-stage compute timing."""

    evaluation: CandidateEvaluation
    collator_seconds: float
    predictor_seconds: float
    simulator_seconds: float
    emulator_seconds: float | None = None
    emulator_wall_seconds: float | None = None
    trace_load_seconds: float = 0.0
    predictor_total_annotation_seconds: float = 0.0
    predictor_runtime_estimation_seconds: float = 0.0
    predictor_pass_through_annotation_seconds: float = 0.0
    predictor_rank_runtime_estimation_seconds: float = 0.0
    predictor_collective_group_estimation_seconds: float = 0.0
    predictor_annotation_timing: dict[str, object] = field(default_factory=dict)

    @property
    def processing_seconds(self) -> float:
        return (
            self.collator_seconds + self.predictor_seconds + self.simulator_seconds
        )

    @property
    def implementation_processing_seconds(self) -> float:
        return (
            self.collator_seconds + self.predictor_seconds + self.simulator_seconds
        )

    @property
    def emulator_stage_seconds(self) -> float:
        if self.emulator_wall_seconds is not None:
            return self.emulator_wall_seconds
        return self.emulator_seconds or 0.0

    @property
    def emulator_stage_basis(self) -> str:
        if self.emulator_wall_seconds is not None:
            return "capture_elapsed_seconds"
        if self.emulator_seconds is not None:
            return "active_emulator_seconds"
        return "unavailable"

    @property
    def stack_seconds(self) -> float:
        return self.processing_seconds + self.emulator_stage_seconds

    @property
    def implementation_stack_seconds(self) -> float:
        return self.implementation_processing_seconds + self.emulator_stage_seconds

    def to_dict(self) -> dict[str, object]:
        payload = self.evaluation.to_dict()
        payload["stage_timing"] = {
            "emulator_seconds": self.emulator_seconds,
            "emulator_wall_seconds": self.emulator_wall_seconds,
            "emulator_stage_seconds": self.emulator_stage_seconds,
            "emulator_stage_basis": self.emulator_stage_basis,
            "collator_seconds": self.collator_seconds,
            "collator_stage_basis": "collate_only",
            "predictor_seconds": self.predictor_seconds,
            "predictor_stage_basis": "runtime_annotation_wall_time",
            "predictor_total_annotation_seconds": self.predictor_total_annotation_seconds,
            "predictor_runtime_estimation_seconds": (
                self.predictor_runtime_estimation_seconds
            ),
            "predictor_pass_through_annotation_seconds": (
                self.predictor_pass_through_annotation_seconds
            ),
            "predictor_rank_runtime_estimation_seconds": (
                self.predictor_rank_runtime_estimation_seconds
            ),
            "predictor_collective_group_estimation_seconds": (
                self.predictor_collective_group_estimation_seconds
            ),
            "predictor_annotation_timing": dict(self.predictor_annotation_timing),
            "simulator_seconds": self.simulator_seconds,
            "simulator_stage_basis": "replay_only",
            "processing_seconds": self.processing_seconds,
            "stack_seconds": self.stack_seconds,
            "paper_processing_seconds": self.processing_seconds,
            "paper_stack_seconds": self.stack_seconds,
            "implementation_processing_seconds": self.implementation_processing_seconds,
            "implementation_stack_seconds": self.implementation_stack_seconds,
            "trace_load_seconds": self.trace_load_seconds,
            "trace_load_included_in_collator": False,
        }
        return payload


def benchmark_trace_directory(
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
    emulator_seconds: float | None = None,
    emulator_wall_seconds: float | None = None,
    allow_kernel_launch_heuristic_fallback: bool = False,
    allow_weak_runtime_fallback: bool = False,
    use_observed_control_plane_wrapper_durations: bool = True,
    use_observed_semantic_wrapper_durations: bool = True,
    host_delay_profile: HostDelayProfile | None = None,
    host_gap_profile: HostGapProfile | None = None,
    predictor_workers: int = 1,
) -> StageTimingBreakdown:
    """Measure collate / predict / simulate compute time on one trace directory."""
    emulator_seconds, emulator_wall_seconds = _resolve_emulator_timings(
        trace_dir,
        emulator_seconds=emulator_seconds,
        emulator_wall_seconds=emulator_wall_seconds,
    )
    load_start = perf_counter()
    bundle = load_trace_directory(
        trace_dir,
        max_events_per_rank=max_events_per_rank,
        trace_window=trace_window,
    )
    trace_load_seconds = perf_counter() - load_start
    return benchmark_trace_bundle(
        bundle,
        estimator,
        candidate_name=candidate_name,
        dedup_identical_ranks=dedup_identical_ranks,
        dedup_pattern_ranks=dedup_pattern_ranks,
        materialize_logical_ranks=materialize_logical_ranks,
        expand_profiled_rank_groups=expand_profiled_rank_groups,
        percentile=percentile,
        emulator_seconds=emulator_seconds,
        emulator_wall_seconds=emulator_wall_seconds,
        trace_load_seconds=trace_load_seconds,
        allow_kernel_launch_heuristic_fallback=allow_kernel_launch_heuristic_fallback,
        allow_weak_runtime_fallback=allow_weak_runtime_fallback,
        use_observed_control_plane_wrapper_durations=use_observed_control_plane_wrapper_durations,
        use_observed_semantic_wrapper_durations=use_observed_semantic_wrapper_durations,
        host_delay_profile=host_delay_profile,
        host_gap_profile=host_gap_profile,
        predictor_workers=predictor_workers,
    )


def _resolve_emulator_timings(
    trace_dir: str | Path,
    *,
    emulator_seconds: float | None,
    emulator_wall_seconds: float | None,
) -> tuple[float | None, float | None]:
    manifest = load_capture_manifest(trace_dir)
    raw_capture_seconds = None if manifest is None else manifest.get("capture_elapsed_seconds")
    raw_active_seconds = None if manifest is None else manifest.get("active_emulator_seconds")

    resolved_emulator_wall_seconds = emulator_wall_seconds
    resolved_emulator_seconds = emulator_seconds

    if resolved_emulator_wall_seconds is None and raw_capture_seconds not in (None, ""):
        resolved_emulator_wall_seconds = float(raw_capture_seconds)
    if resolved_emulator_seconds is None and raw_active_seconds not in (None, ""):
        resolved_emulator_seconds = float(raw_active_seconds)
    if resolved_emulator_seconds is None and resolved_emulator_wall_seconds is not None:
        resolved_emulator_seconds = resolved_emulator_wall_seconds
    return resolved_emulator_seconds, resolved_emulator_wall_seconds


def benchmark_trace_bundle(
    bundle: TraceBundle,
    estimator: Estimator,
    *,
    candidate_name: str | None = None,
    dedup_identical_ranks: bool = False,
    dedup_pattern_ranks: bool = False,
    materialize_logical_ranks: bool = False,
    expand_profiled_rank_groups: bool = False,
    percentile: str = "p50",
    emulator_seconds: float | None = None,
    emulator_wall_seconds: float | None = None,
    trace_load_seconds: float = 0.0,
    allow_kernel_launch_heuristic_fallback: bool = False,
    allow_weak_runtime_fallback: bool = False,
    allow_collective_group_fallback: bool = False,
    use_observed_control_plane_wrapper_durations: bool = True,
    use_observed_semantic_wrapper_durations: bool = True,
    host_delay_profile: HostDelayProfile | None = None,
    host_gap_profile: HostGapProfile | None = None,
    predictor_workers: int = 1,
) -> StageTimingBreakdown:
    """Measure collate / predict / simulate compute time on a preloaded trace bundle.

    Performance optimizations are allowed to reduce the real wall time of these
    stages, but they must not change the paper-facing accounting boundary:

    - emulator: capture / emulator-owned artifact production
    - collator: low-level trace collation only
    - predictor: full runtime annotation wall time
    - simulator: replay only

    In other words, we can make a stage faster, but we must not make Figure 13
    look better by silently moving work across stage boundaries.
    """
    emulator_seconds, emulator_wall_seconds = _resolve_emulator_timings(
        bundle.trace_dir,
        emulator_seconds=emulator_seconds,
        emulator_wall_seconds=emulator_wall_seconds,
    )
    collate_start = perf_counter()
    if dedup_identical_ranks:
        bundle = dedup_identical_rank_traces(bundle)
    if dedup_pattern_ranks:
        bundle = dedup_pattern_rank_traces(bundle)
    if materialize_logical_ranks:
        bundle = materialize_profiled_rank_traces(bundle)
    collated = collate_trace_bundle(bundle)
    collator_seconds = perf_counter() - collate_start

    predictor_timing_recorder = AnnotationTimingRecorder()
    predictor_start = perf_counter()
    annotated = annotate_collated_trace(
        collated,
        estimator,
        percentile=percentile,
        allow_kernel_launch_heuristic_fallback=allow_kernel_launch_heuristic_fallback,
        allow_weak_runtime_fallback=allow_weak_runtime_fallback,
        allow_collective_group_fallback=allow_collective_group_fallback,
        use_observed_control_plane_wrapper_durations=use_observed_control_plane_wrapper_durations,
        use_observed_semantic_wrapper_durations=use_observed_semantic_wrapper_durations,
        host_delay_profile=host_delay_profile,
        host_gap_profile=host_gap_profile,
        parallel_workers=predictor_workers,
        timing_recorder=predictor_timing_recorder,
    )
    predictor_total_annotation_seconds = perf_counter() - predictor_start
    predictor_annotation_timing = predictor_timing_recorder.summary(
        total_annotation_seconds=predictor_total_annotation_seconds,
    )
    predictor_runtime_estimation_seconds = float(
        predictor_annotation_timing["runtime_estimation_wall_seconds"]
    )

    simulator_start = perf_counter()
    replay = replay_annotated_trace(
        annotated,
        expand_profiled_rank_groups=expand_profiled_rank_groups,
        # Figure 13 stage-share accounting should measure the full simulator
        # work, including construction of the paper-facing simulation output.
        # The lightweight replay path remains available in evaluation-only
        # flows, but paper-facing benchmark timing should not undercount the
        # simulator stage by skipping simulated-event materialization.
        record_simulated_events=True,
    )
    simulator_seconds = perf_counter() - simulator_start

    return StageTimingBreakdown(
        evaluation=build_candidate_evaluation(
            collated,
            replay,
            candidate_name=candidate_name or bundle.trace_dir.name,
            expand_profiled_rank_groups=expand_profiled_rank_groups,
            annotation_diagnostics=collective_group_duration_summary(annotated),
        ),
        collator_seconds=collator_seconds,
        predictor_seconds=predictor_total_annotation_seconds,
        simulator_seconds=simulator_seconds,
        emulator_seconds=emulator_seconds,
        emulator_wall_seconds=emulator_wall_seconds,
        trace_load_seconds=trace_load_seconds,
        predictor_total_annotation_seconds=predictor_total_annotation_seconds,
        predictor_runtime_estimation_seconds=predictor_runtime_estimation_seconds,
        predictor_pass_through_annotation_seconds=float(
            predictor_annotation_timing["pass_through_annotation_seconds"]
        ),
        predictor_rank_runtime_estimation_seconds=float(
            predictor_annotation_timing["rank_runtime_estimation_wall_seconds"]
        ),
        predictor_collective_group_estimation_seconds=float(
            predictor_annotation_timing["collective_group_estimation_seconds"]
        ),
        predictor_annotation_timing=predictor_annotation_timing,
    )
