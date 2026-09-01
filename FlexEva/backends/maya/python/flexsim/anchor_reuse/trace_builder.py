"""Build coarse anchor witnesses directly from audited micro-trace artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import (
    AnchorRegion,
    AnchorWitness,
    DependencyType,
    RegionKind,
    Witness,
    WitnessGranularity,
    WitnessSourceKind,
)

WINDOW_COUNT = 24


def _category(event: dict[str, Any]) -> str:
    event_type = event["type"]
    mod = event["mod"]
    if event_type == "nccl_collective" or mod.startswith("libnccl"):
        return "nccl"
    if event_type == "blas_compute" or mod.startswith("libcublas"):
        return "blas"
    if event_type == "kernel_launch":
        return "kernel"
    if event_type == "mem_copy":
        return "memcpy"
    if event_type == "stream_op":
        return "stream"
    if event_type == "mem_alloc":
        return "alloc"
    if event_type == "context_op":
        return "context"
    return "other"


def _dominant_meaningful_category(counter: Counter[str]) -> str:
    meaningful = {key: value for key, value in counter.items() if key not in {"context", "other"}}
    if meaningful:
        return max(sorted(meaningful), key=lambda key: meaningful[key])
    if counter:
        return max(sorted(counter), key=lambda key: counter[key])
    return "empty"


def _region_kind_for_category(category: str) -> RegionKind:
    if category == "nccl":
        return RegionKind.COLLECTIVE
    if category in {"blas", "kernel"}:
        return RegionKind.EXPERT_COMPUTE
    if category in {"memcpy", "alloc"}:
        return RegionKind.MEMORY
    if category == "stream":
        return RegionKind.OVERLAP
    return RegionKind.OTHER


def _world_size_from_manifest(trace_dir: Path) -> int:
    manifest_path = trace_dir / "capture_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = payload.get("original_world_size")
        if value is not None:
            return int(value)
    rank_files = sorted(trace_dir.glob("rank_*.jsonl"))
    if not rank_files:
        raise FileNotFoundError(f"no rank traces found in {trace_dir}")
    first_line = next(
        line for line in rank_files[0].read_text(encoding="utf-8").splitlines() if line.strip()
    )
    return int(json.loads(first_line).get("world_size", 1))


def build_witness_from_micro_trace_dir(
    trace_dir: Path | str,
    *,
    anchor_candidate_id: str,
    workload_family: str,
    window_count: int = WINDOW_COUNT,
) -> AnchorWitness:
    """Construct a coarse region-level witness from multi-rank audited traces.

    The builder intentionally mirrors the normalized-window aggregation already
    used by the preliminary locality diagnostic, then merges adjacent windows of
    the same coarse region kind.
    """

    trace_dir = Path(trace_dir)
    rank_files = sorted(trace_dir.glob("rank_*.jsonl"))
    if not rank_files:
        raise FileNotFoundError(f"no rank traces found in {trace_dir}")

    window_counts = [Counter() for _ in range(window_count)]
    window_share = [0.0 for _ in range(window_count)]
    rank_count = 0

    for rank_file in rank_files:
        events = [
            json.loads(line)
            for line in rank_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not events:
            continue
        ts0 = events[0]["ts"]
        ts1 = events[-1]["ts"]
        span = max(ts1 - ts0, 1)
        per_counts = [Counter() for _ in range(window_count)]
        min_ts: list[int | None] = [None for _ in range(window_count)]
        max_ts: list[int | None] = [None for _ in range(window_count)]

        for event in events:
            position = (event["ts"] - ts0) / span
            idx = min(window_count - 1, int(position * window_count))
            per_counts[idx][_category(event)] += 1
            event_ts = event["ts"]
            min_ts[idx] = event_ts if min_ts[idx] is None else min(min_ts[idx], event_ts)
            max_ts[idx] = event_ts if max_ts[idx] is None else max(max_ts[idx], event_ts)

        for idx in range(window_count):
            window_counts[idx].update(per_counts[idx])
            if min_ts[idx] is not None and max_ts[idx] is not None:
                window_share[idx] += (max_ts[idx] - min_ts[idx]) / span
        rank_count += 1

    if rank_count == 0:
        raise ValueError(f"trace directory contains no non-empty rank traces: {trace_dir}")

    normalized_window_share = [value / rank_count for value in window_share]
    total_share = sum(normalized_window_share)
    normalized_window_share = [
        value / total_share if total_share else 0.0 for value in normalized_window_share
    ]

    window_regions: list[tuple[int, int, RegionKind, float, Counter[str]]] = []
    for idx, counter in enumerate(window_counts):
        share = normalized_window_share[idx]
        if share <= 0.0:
            continue
        category = _dominant_meaningful_category(counter)
        region_kind = _region_kind_for_category(category)
        if window_regions and window_regions[-1][2] == region_kind:
            start, _, kind, prev_share, prev_counter = window_regions[-1]
            prev_counter.update(counter)
            window_regions[-1] = (start, idx, kind, prev_share + share, prev_counter)
        else:
            window_regions.append((idx, idx, region_kind, share, Counter(counter)))

    max_share = max((share for _, _, _, share, _ in window_regions), default=0.0)
    witness = Witness(
        source_kind=WitnessSourceKind.TRACE_WINDOW_AGGREGATE,
        confidence=0.78,
        evidence=tuple(str(path) for path in ([trace_dir / "capture_manifest.json"] + rank_files)),
        rationale=("built from normalized multi-rank trace windows",),
    )

    regions = []
    for order_index, (start_idx, end_idx, region_kind, share, counter) in enumerate(window_regions):
        region_id = f"{region_kind.value}_{order_index:02d}"
        top_categories = ",".join(
            f"{name}:{count}" for name, count in counter.most_common(3)
        )
        value_sensitive = region_kind in {
            RegionKind.DISPATCH,
            RegionKind.EXPERT_COMPUTE,
            RegionKind.OVERLAP,
        }
        hardware_sensitive = region_kind in {
            RegionKind.COLLECTIVE,
            RegionKind.OVERLAP,
            RegionKind.MEMORY,
        }
        dependency_type = (
            DependencyType.WIDE
            if region_kind in {RegionKind.DISPATCH, RegionKind.COLLECTIVE, RegionKind.OVERLAP}
            else DependencyType.NARROW
        )
        regions.append(
            AnchorRegion(
                region_id=region_id,
                region_kind=region_kind,
                order_index=order_index,
                timing_share=share,
                start_window=start_idx,
                end_window=end_idx,
                notes=(
                    f"windows={start_idx}-{end_idx}",
                    f"dominant={_dominant_meaningful_category(counter)}",
                    f"counts={top_categories}",
                ),
                witness=witness,
                provenance=(f"trace_dir={trace_dir}", f"window_range={start_idx}:{end_idx}"),
                value_sensitive=value_sensitive,
                hardware_sensitive=hardware_sensitive,
                criticality_slack=max(max_share - share, 0.0),
                dependency_type=dependency_type,
            )
        )

    artifacts = tuple(str(path) for path in ([trace_dir / "capture_manifest.json"] + rank_files))
    return AnchorWitness(
        anchor_candidate_id=anchor_candidate_id,
        workload_family=workload_family,
        world_size=_world_size_from_manifest(trace_dir),
        granularity=WitnessGranularity.PHASE_REGION,
        regions=tuple(regions),
        artifacts=artifacts,
        notes=("built from normalized multi-rank trace windows",),
    )
