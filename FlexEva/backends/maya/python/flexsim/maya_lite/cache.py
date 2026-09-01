"""Maya-lite artifact cache.

Separates the pipeline into two phases that match Maya paper's oral:

  Phase 1 – prepare (one-time, like Maya's profiling/emulation phase):
    load real trace + emulated trace
    collate both
    fit Estimator from memory
    fit HostDelayProfile + HostGapProfile
    save compact artifacts to disk

  Phase 2 – simulate (repeated, like Maya's simulation phase):
    load compact artifacts
    annotate emulated collated trace
    replay
    report total_time_us / error

The simulate phase is the one that should be compared to Maya's reported
simulation speed.  The prepare phase is analogous to Maya's profiling/
emulation overhead and is expected to be slow.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


_ARTIFACT_NAMES = (
    "emu_collated.pkl",
    "estimator.pkl",
    "host_delay_profile.pkl",
    "host_gap_profile.pkl",
)


def save_artifacts(
    cache_dir: str | Path,
    *,
    emu_coll: Any,
    estimator: Any,
    host_delay_profile: Any,
    host_gap_profile: Any,
) -> None:
    """Persist all simulation-phase artifacts to *cache_dir*."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    for name, obj in (
        ("emu_collated.pkl", emu_coll),
        ("estimator.pkl", estimator),
        ("host_delay_profile.pkl", host_delay_profile),
        ("host_gap_profile.pkl", host_gap_profile),
    ):
        with open(cache_path / name, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_artifacts(
    cache_dir: str | Path,
) -> tuple[Any, Any, Any, Any]:
    """Load artifacts saved by :func:`save_artifacts`.

    Returns ``(emu_coll, estimator, host_delay_profile, host_gap_profile)``.
    """
    cache_path = Path(cache_dir)
    missing = [n for n in _ARTIFACT_NAMES if not (cache_path / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"Cache artifacts missing in {cache_dir}: {missing}. "
            "Run the prepare phase first."
        )
    results = []
    for name in _ARTIFACT_NAMES:
        with open(cache_path / name, "rb") as f:
            results.append(pickle.load(f))
    return tuple(results)  # type: ignore[return-value]


def artifacts_exist(cache_dir: str | Path) -> bool:
    """Return True if all artifacts are present in *cache_dir*."""
    cache_path = Path(cache_dir)
    return all((cache_path / n).exists() for n in _ARTIFACT_NAMES)
