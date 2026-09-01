"""Phase 2 – simulate.

Loads artifacts from prepare and runs the four Figure 13 stages:

  Emulator  = emulator_wall_seconds (external, from capture_manifest or arg)
  Collator  = collate_trace_bundle(emu_bundle)
  Predictor = runtime_annotation_wall_time
  Simulator = replay_annotated_trace

Uses benchmark_trace_bundle so stage accounting is identical to
reproduce_fig13.py and the paper-facing stage_timing contract is satisfied.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

from flexsim.maya_lite.paper_error import paper_absolute_error_pct
from flexsim.maya_lite.stage_timing import benchmark_trace_bundle


_ARTIFACT_NAMES = (
    "emu_bundle.pkl",
    "estimator.pkl",
    "host_delay_profile.pkl",
    "host_gap_profile.pkl",
    "paper_actual_runtime.pkl",
)


def _load_artifacts(cache_dir: Path):
    missing = [n for n in _ARTIFACT_NAMES if not (cache_dir / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"Cache artifacts missing in {cache_dir}: {missing}. "
            "Run the prepare phase first."
        )
    results = []
    for name in _ARTIFACT_NAMES:
        with open(cache_dir / name, "rb") as f:
            results.append(pickle.load(f))
    return tuple(results)


def _assert_paper_host_timing_inputs(
    host_delay_profile: object | None,
    host_gap_profile: object | None,
) -> None:
    """Fail if paper-facing replay is wired to answer-trace host timing."""
    if host_delay_profile is not None or host_gap_profile is not None:
        raise RuntimeError("paper-facing simulate cannot use real-derived host profiles")


def simulate(
    cache_dir: str | Path,
    *,
    real_total_us: float | None = None,
    emulator_wall_seconds: float | None = None,
    percentile: str = "mean",
    parallel_workers: int = 16,
    verbose: bool = True,
) -> dict:
    """Run the simulate phase using Figure 13 four-stage accounting."""
    cache_path = Path(cache_dir)
    marks: list[tuple[str, float]] = []

    def mark(name: str) -> None:
        marks.append((name, time.perf_counter()))
        if verbose:
            print(f"SIMULATE {name}", flush=True)

    mark("start")
    emu_bundle, estimator, _host_delay_profile, _host_gap_profile, paper_actual_runtime = _load_artifacts(cache_path)
    mark("load_artifacts")

    expand_profiled_rank_groups = len(emu_bundle.rank_traces) < int(emu_bundle.world_size)
    # Native dynamic dedup keeps all first-iteration rank traces when available.
    # In that case replay must use those rank-specific timings directly instead
    # of copying representative metrics to every group member.
    replay_host_delay_profile = None
    replay_host_gap_profile = None
    _assert_paper_host_timing_inputs(replay_host_delay_profile, replay_host_gap_profile)
    breakdown = benchmark_trace_bundle(
        emu_bundle,
        estimator,
        percentile=percentile,
        emulator_wall_seconds=emulator_wall_seconds,
        # Paper-facing Maya uses host-side computation/dispatch overheads
        # measured as wall-clock deltas during emulation.  Real traces are the
        # validation target and the source for device runtime estimators; they
        # must not inject an additional fitted host delay/gap profile into
        # replay.
        host_delay_profile=replay_host_delay_profile,
        host_gap_profile=replay_host_gap_profile,
        predictor_workers=parallel_workers,
        expand_profiled_rank_groups=expand_profiled_rank_groups,
        # Maya models host-side control overheads from wall-clock measurements
        # collected in the emulated trace.  Keep observed control-plane wrapper
        # durations in the native path; otherwise 64GPU underestimates host
        # dispatch/synchronization time by dropping measured CPU-side gaps.
        use_observed_control_plane_wrapper_durations=True,
        use_observed_semantic_wrapper_durations=True,
    )
    mark("done")

    stage_timing = breakdown.to_dict().get("stage_timing", {})
    evaluation = breakdown.evaluation
    total_time_us = evaluation.total_time_us
    paper_prediction_runtime_us = evaluation.critical_path_us
    if real_total_us is not None:
        provided = float(real_total_us)
        derived = float(paper_actual_runtime.actual_per_iteration_runtime_us)
        tolerance = max(1.0, 0.001 * derived)
        if abs(provided - derived) > tolerance:
            raise ValueError(
                "--real-total-us is only a compatibility check and must match "
                f"real-trace step-window actual runtime; provided={provided}, derived={derived}"
            )
    actual_per_iteration_runtime_us = float(paper_actual_runtime.actual_per_iteration_runtime_us)

    summary: dict = {
        "phase": "simulate",
        "native_dynamic_dedup_first_iteration_trace_retention": not expand_profiled_rank_groups,
        "expand_profiled_rank_groups": expand_profiled_rank_groups,
        "emu_rank_trace_count": len(emu_bundle.rank_traces),
        "emu_world_size": int(emu_bundle.world_size),
        "paper_host_timing_source": "emulation_wall_clock_deltas",
        "real_derived_host_profiles_used": False,
        "load_artifacts_seconds": marks[1][1] - marks[0][1],
        "collator_seconds": breakdown.collator_seconds,
        "predictor_seconds": breakdown.predictor_seconds,
        "predictor_runtime_estimation_seconds": (
            breakdown.predictor_runtime_estimation_seconds
        ),
        "predictor_total_annotation_seconds": breakdown.predictor_total_annotation_seconds,
        "simulator_seconds": breakdown.simulator_seconds,
        "emulator_wall_seconds": breakdown.emulator_wall_seconds,
        "processing_seconds": breakdown.processing_seconds,
        "stack_seconds": breakdown.stack_seconds,
        "paper_prediction_metric": "absolute_percentage_error_of_predicted_vs_actual_per_iteration_runtime",
        "paper_prediction_runtime_basis": "critical_rank_completed_iteration_runtime",
        "predicted_per_iteration_runtime_us": paper_prediction_runtime_us,
        "predicted_critical_rank_runtime_us": evaluation.critical_path_us,
        "predicted_global_makespan_us": evaluation.global_makespan_us,
        "predicted_rank0_runtime_us": evaluation.rank0_time_us,
        "stage_timing": stage_timing,
    }
    paper_error_pct = paper_absolute_error_pct(
        paper_prediction_runtime_us,
        actual_per_iteration_runtime_us,
    )
    summary["actual_per_iteration_runtime_us"] = actual_per_iteration_runtime_us
    summary["paper_actual_runtime_basis"] = paper_actual_runtime.basis
    summary["paper_actual_runtime"] = paper_actual_runtime.to_dict()
    summary["paper_absolute_error_pct"] = paper_error_pct
    summary["absolute_error_pct_by_basis"] = {
        "critical_rank": paper_absolute_error_pct(evaluation.critical_path_us, actual_per_iteration_runtime_us),
        "global_makespan": paper_absolute_error_pct(evaluation.global_makespan_us, actual_per_iteration_runtime_us),
        "rank0": (
            None
            if evaluation.rank0_time_us is None
            else paper_absolute_error_pct(evaluation.rank0_time_us, actual_per_iteration_runtime_us)
        ),
    }
    (cache_path / "simulate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if verbose:
        print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Maya-lite simulate phase (Fig13 aligned)")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--real-total-us", type=float, default=None)
    parser.add_argument("--emulator-wall-seconds", type=float, default=None)
    parser.add_argument("--percentile", default="mean")
    parser.add_argument("--parallel-workers", type=int, default=16)
    args = parser.parse_args()
    simulate(
        args.cache,
        real_total_us=args.real_total_us,
        emulator_wall_seconds=args.emulator_wall_seconds,
        percentile=args.percentile,
        parallel_workers=args.parallel_workers,
    )


if __name__ == "__main__":
    main()
