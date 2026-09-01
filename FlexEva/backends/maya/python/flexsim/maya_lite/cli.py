"""
CLI for Maya-lite batch evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flexsim.estimator import (
    DEFAULT_GPU_ESTIMATOR_BUNDLE,
    Estimator,
    probe_gpu_estimator_provider,
)

from .evaluate import evaluate_trace_directory
def resolve_estimator_mode(
    trace_dir: Path,
    requested_mode: str,
    *,
    fit_trace_dir: Path | None = None,
    fit_max_files: int = 0,
    fit_workers: int = 1,
    gpu_estimator_bundle: Path = DEFAULT_GPU_ESTIMATOR_BUNDLE,
) -> str:
    """
    Choose a workload-appropriate estimator mode for Maya-lite.

    Auto mode keeps the paper-facing transparent-profiling path first.  When a
    GPU estimator bundle is useful for a large share of the fitted trace, choose
    `hybrid`, whose provider order is trace-signature profiling, trace-learned
    profiling, then GPU-estimator fallback.  That preserves Maya's
    profile-then-annotate contract while still allowing a pluggable fallback.
    """
    mode = (requested_mode or "hybrid").strip().lower()
    if mode != "auto":
        return mode

    status = probe_gpu_estimator_provider(gpu_estimator_bundle)
    if status.provider is None:
        return "learned_trace"

    coverage_trace_dir = fit_trace_dir or trace_dir
    try:
        probe_estimator = Estimator.fit_from_traces(
            str(coverage_trace_dir),
            max_files=max(int(fit_max_files), 0),
            learned_method="trace_stats",
            fit_workers=max(int(fit_workers), 1),
        )
    except Exception:
        return "learned_trace"

    probe_estimator.add_provider(status.provider, prepend=True)
    coverage = probe_estimator.provider_coverage_summary(
        "gpu_estimator_xgboost",
        limit=5,
    )
    if coverage["covered_time_share"] >= 0.35:
        return "hybrid"
    return "learned_trace"


def fit_estimator_with_resolved_mode(
    trace_dir: Path,
    requested_mode: str,
    *,
    fit_trace_dir: Path | None = None,
    fit_max_files: int = 0,
    fit_workers: int = 1,
    fit_trace_window: str = "auto",
    gpu_estimator_bundle: Path = DEFAULT_GPU_ESTIMATOR_BUNDLE,
) -> tuple[Estimator, str]:
    """
    Fit the paper-facing estimator while resolving auto mode without a second
    full trace ingest.

    Maya's profiling path should dispatch/profile once and then annotate from
    that profile.  The previous auto-mode helper fitted once to choose a mode
    and then fitted again to build the final estimator.  This helper keeps the
    same mode-selection policy but reuses the trace statistics gathered during
    the probe.
    """
    mode = (requested_mode or "hybrid").strip().lower()
    fit_dir = fit_trace_dir or trace_dir
    max_files = max(int(fit_max_files), 0)
    workers = max(int(fit_workers), 1)

    if mode != "auto":
        estimator = Estimator.fit_from_traces(
            str(fit_dir),
            max_files=max_files,
            learned_method=mode,
            gpu_estimator_bundle=gpu_estimator_bundle,
            fit_workers=workers,
            trace_window=fit_trace_window,
        )
        return estimator, mode

    status = probe_gpu_estimator_provider(gpu_estimator_bundle)
    if status.provider is None:
        resolved_mode = "learned_trace"
        estimator = Estimator.fit_from_traces(
            str(fit_dir),
            max_files=max_files,
            learned_method=resolved_mode,
            gpu_estimator_bundle=gpu_estimator_bundle,
            fit_workers=workers,
            trace_window=fit_trace_window,
        )
        return estimator, resolved_mode

    estimator = Estimator.fit_from_traces(
        str(fit_dir),
        max_files=max_files,
        learned_method="trace_stats",
        gpu_estimator_bundle=gpu_estimator_bundle,
        fit_workers=workers,
        trace_window=fit_trace_window,
    )
    coverage = estimator.provider_coverage_summary(
        "gpu_estimator_xgboost",
        limit=5,
        providers=[status.provider],
    )
    resolved_mode = "hybrid" if coverage["covered_time_share"] >= 0.35 else "learned_trace"
    estimator.attach_learned_methods(
        resolved_mode,
        gpu_estimator_bundle=gpu_estimator_bundle,
    )
    return estimator, resolved_mode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Maya-lite low-level evaluator")
    parser.add_argument("trace_dir", type=Path, help="Directory with rank_*.jsonl traces")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Existing estimator JSON file",
    )
    parser.add_argument(
        "--fit-traces",
        type=Path,
        default=None,
        help="Real-trace directory used to fit an estimator on the fly",
    )
    parser.add_argument(
        "--fit-max-files",
        type=int,
        default=0,
        help="Maximum rank trace files to use when fitting the estimator (0 = all)",
    )
    parser.add_argument(
        "--max-events-per-rank",
        type=int,
        default=None,
        help="Limit events per rank for faster experiments",
    )
    parser.add_argument(
        "--dedup-identical-ranks",
        action="store_true",
        help="Replay only one representative for byte-identical low-level rank traces while preserving original world_size metadata.",
    )
    parser.add_argument(
        "--dedup-rank-patterns",
        action="store_true",
        help="Run a post-capture pattern-dedup stage that buckets semantically equivalent rank traces with rolling hashes before replay.",
    )
    parser.add_argument(
        "--expand-profiled-ranks",
        action="store_true",
        help="Expand representative-rank replay metrics back to one metric entry per original rank using profiled_rank_groups metadata.",
    )
    parser.add_argument(
        "--percentile",
        default="p50",
        choices=["p50", "mean", "p95"],
        help="Estimator percentile to use",
    )
    parser.add_argument(
        "--estimator-mode",
        default="auto",
        choices=["auto", "trace_stats", "learned_trace", "gpu_xgboost", "hybrid"],
        help="Estimator strategy for --fit-traces. auto selects a workload-appropriate mode from the target trace; hybrid = trace-signature profiling first, trace-learned profiling second, GPU-estimator fallback.",
    )
    parser.add_argument(
        "--gpu-estimator-bundle",
        type=Path,
        default=None,
        help="Optional gpu_estimator runtime bundle path used by gpu_xgboost/hybrid modes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    # Paper-aligned CLI entrypoints do not expose old weak timing fallback
    # toggles. Internal APIs still default these booleans to False so tests can
    # assert the guardrails without offering a user-facing escape hatch.
    parser.set_defaults(
        allow_heuristic_kernel_launch_fallback=False,
        allow_weak_runtime_fallback=False,
    )
    # The current Maya-aligned route always consumes direct semantic wrapper
    # observations when present; the old compatibility toggle was removed to
    # avoid paper-facing runs silently falling back to non-canonical timing.
    parser.set_defaults(use_observed_semantic_wrapper_durations=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.model is not None:
        estimator = Estimator.load(str(args.model))
        resolved_mode = "loaded_model"
    elif args.fit_traces is not None:
        resolved_mode = resolve_estimator_mode(
            args.trace_dir,
            args.estimator_mode,
            fit_trace_dir=args.fit_traces,
            fit_max_files=args.fit_max_files,
            gpu_estimator_bundle=(
                args.gpu_estimator_bundle
                if args.gpu_estimator_bundle is not None
                else DEFAULT_GPU_ESTIMATOR_BUNDLE
            ),
        )
        estimator = Estimator.fit_from_traces(
            str(args.fit_traces),
            max_files=max(int(args.fit_max_files), 0),
            learned_method=resolved_mode,
            gpu_estimator_bundle=(
                args.gpu_estimator_bundle
                if args.gpu_estimator_bundle is not None
                else DEFAULT_GPU_ESTIMATOR_BUNDLE
            ),
        )
    else:
        parser.error("pass either --model or --fit-traces")

    result = evaluate_trace_directory(
        args.trace_dir,
        estimator,
        max_events_per_rank=args.max_events_per_rank,
        dedup_identical_ranks=args.dedup_identical_ranks,
        dedup_pattern_ranks=args.dedup_rank_patterns,
        expand_profiled_rank_groups=args.expand_profiled_ranks,
        percentile=args.percentile,
        allow_kernel_launch_heuristic_fallback=args.allow_heuristic_kernel_launch_fallback,
        allow_weak_runtime_fallback=args.allow_weak_runtime_fallback,
        use_observed_semantic_wrapper_durations=args.use_observed_semantic_wrapper_durations,
    )
    payload = result.to_dict()
    payload["estimator_mode_requested"] = args.estimator_mode
    payload["estimator_mode_resolved"] = resolved_mode
    payload["estimator_provider_names"] = list(estimator.provider_names())
    payload["estimator_provider_diagnostics"] = estimator.provider_diagnostics()
    payload["estimator_operator_families"] = estimator.operator_family_summary(limit=10)
    payload["estimator_kernel_launch_metadata"] = estimator.kernel_launch_metadata_summary()
    payload["estimator_transparent_profiling"] = estimator.transparent_profiling_summary()
    payload["estimator_xgboost_coverage"] = estimator.provider_coverage_summary(
        "gpu_estimator_xgboost",
        limit=10,
    )
    payload["heuristic_kernel_launch_fallback_enabled"] = (
        args.allow_heuristic_kernel_launch_fallback
    )
    payload["weak_runtime_fallback_enabled"] = args.allow_weak_runtime_fallback
    payload["observed_semantic_wrapper_durations_enabled"] = (
        args.use_observed_semantic_wrapper_durations
    )

    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
