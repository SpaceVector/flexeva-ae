"""Phase 1 – prepare.

Loads the evaluation real trace only to extract the measured actual runtime,
loads the emulated trace to cache replay input, and loads an independent
estimator model.  Paper-facing prepare must not fit timing models from the same
real trace that is later used as the accuracy target.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

from flexsim.estimator import Estimator
from flexsim.maya_lite.io import load_trace_directory
from flexsim.maya_lite.paper_error import actual_runtime_from_real_trace


def prepare(
    real_dir: str | Path,
    emu_dir: str | Path,
    cache_dir: str | Path,
    *,
    estimator_model: str | Path,
    gpu_estimator_bundle: str | Path | None = None,
    capture_estimator_load_warnings: bool = False,
    trace_window: str = "step",
    verbose: bool = True,
) -> dict:
    """Run paper-facing prepare without target-trace estimator leakage."""
    marks: list[tuple[str, float]] = []

    def mark(name: str) -> None:
        marks.append((name, time.perf_counter()))
        if verbose:
            print(f"PREPARE  {name}", flush=True)

    mark("start")
    real_bundle = load_trace_directory(real_dir, trace_window=trace_window)
    paper_actual_runtime = actual_runtime_from_real_trace(real_bundle)
    mark("load_real_actual_only")
    emu_bundle = load_trace_directory(emu_dir, trace_window=trace_window)
    mark("load_emu")

    estimator_model_path = Path(estimator_model)
    if not estimator_model_path.exists():
        raise FileNotFoundError(
            f"paper-facing native prepare requires an independent estimator model: {estimator_model_path}. "
            "Do not fit the estimator from the evaluation real trace."
        )
    estimator_load_warnings: list[dict[str, str]] = []
    if capture_estimator_load_warnings:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator = Estimator.load(
                str(estimator_model_path),
                gpu_estimator_bundle=gpu_estimator_bundle,
            )
        estimator_load_warnings = list(
            {
                (warning.category.__name__, str(warning.message)): {
                    "category": warning.category.__name__,
                    "message": str(warning.message),
                }
                for warning in caught
            }.values()
        )
    else:
        estimator = Estimator.load(
            str(estimator_model_path),
            gpu_estimator_bundle=gpu_estimator_bundle,
        )
    mark("load_independent_estimator")

    host_delay_profile = None
    host_gap_profile = None
    mark("disable_real_derived_host_profiles")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    for name, obj in (
        ("emu_bundle.pkl", emu_bundle),
        ("estimator.pkl", estimator),
        ("host_delay_profile.pkl", host_delay_profile),
        ("host_gap_profile.pkl", host_gap_profile),
        ("paper_actual_runtime.pkl", paper_actual_runtime),
    ):
        with open(cache_path / name, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    mark("save_artifacts")

    stages = [
        {"stage": marks[i][0], "seconds": marks[i][1] - marks[i - 1][1]}
        for i in range(1, len(marks))
    ]
    total = marks[-1][1] - marks[0][1]
    summary = {
        "phase": "prepare",
        "total_seconds": total,
        "stages": stages,
        "event_counts": {
            "real_raw": sum(len(r.events) for r in real_bundle.rank_traces),
            "emu_raw": sum(len(r.events) for r in emu_bundle.rank_traces),
            "real_collated": None,
        },
        "paper_actual_runtime": paper_actual_runtime.to_dict(),
        "estimator_source": "independent_model",
        "estimator_model": str(estimator_model_path),
        "gpu_estimator_bundle_override": (
            None if gpu_estimator_bundle is None else str(gpu_estimator_bundle)
        ),
        "estimator_load_warnings": estimator_load_warnings,
        "target_trace_used_for_estimator_fit": False,
        "host_gap_profile": {
            "rank_pair_count": 0,
            "rank_api_count": 0,
            "global_pair_count": 0,
            "global_api_count": 0,
            "used_for_replay": False,
        },
    }
    (cache_path / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if verbose:
        print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Maya-lite paper-facing prepare phase")
    parser.add_argument("--real", required=True)
    parser.add_argument("--emu", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--estimator-model", required=True)
    parser.add_argument("--gpu-estimator-bundle")
    parser.add_argument("--capture-estimator-load-warnings", action="store_true")
    parser.add_argument("--trace-window", default="step")
    args = parser.parse_args()
    prepare(
        args.real,
        args.emu,
        args.cache,
        estimator_model=args.estimator_model,
        gpu_estimator_bundle=args.gpu_estimator_bundle,
        capture_estimator_load_warnings=args.capture_estimator_load_warnings,
        trace_window=args.trace_window,
    )


if __name__ == "__main__":
    main()
