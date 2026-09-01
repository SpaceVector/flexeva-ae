"""Logical replay-segment extraction for anchor-based partial refresh."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from flexsim.maya_lite.collate import collate_trace_bundle
from flexsim.maya_lite.io import load_trace_directory
from flexsim.maya_lite.schema import CollatedEvent, CollatedTrace, CollectiveGroup


@dataclass(frozen=True)
class ReplaySegment:
    segment_id: str
    segment_kind: str
    start_ts: int
    end_ts: int
    start_window: int
    end_window: int
    event_count: int
    ranks: tuple[int, ...]
    collective_group_id: str | None = None
    event_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplaySegmentBundle:
    trace_dir: str
    window_count: int
    segments: tuple[ReplaySegment, ...]
    collated: CollatedTrace | None = None
    events_by_id: dict[str, CollatedEvent] = field(default_factory=dict)
    segment_indices_by_window: dict[int, tuple[int, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class SegmentCompatibility:
    refresh_windows: tuple[int, ...]
    anchor_segment_count: int
    target_segment_count: int
    anchor_touched_fraction: float
    target_touched_fraction: float
    kind_match_count: int
    length_ratio: float
    kind_match_fraction: float
    compatibility_score: float
    should_refresh: bool


def _window_index(ts: int, ts0: int, ts1: int, *, window_count: int) -> int:
    span = max(ts1 - ts0, 1)
    return min(window_count - 1, int(((ts - ts0) / span) * window_count))


def _segment_kind_for_event(event) -> str:
    if event.collective_group_id is not None:
        return "collective"
    if event.api == "__hostDelay__":
        return "host_delay"
    if event.op_type in {"blas_compute", "kernel_launch"}:
        return "compute"
    if event.op_type in {"mem_copy", "mem_alloc"}:
        return "memory"
    if event.op_type == "stream_op":
        return "stream"
    return "other"


def build_replay_segments_from_collated(
    collated: CollatedTrace,
    *,
    window_count: int = 24,
) -> ReplaySegmentBundle:
    events = collated.global_events
    if not events:
        return ReplaySegmentBundle(
            trace_dir=str(collated.trace_dir),
            window_count=window_count,
            segments=(),
            collated=collated,
        )

    ts0 = int(events[0].ts)
    ts1 = int(events[-1].ts)
    segments: list[ReplaySegment] = []
    current_events = []
    current_kind = None
    current_collective_group = None

    def flush() -> None:
        nonlocal current_events, current_kind, current_collective_group
        if not current_events:
            return
        start_ts = int(current_events[0].ts)
        end_ts = int(current_events[-1].ts)
        start_window = _window_index(start_ts, ts0, ts1, window_count=window_count)
        end_window = _window_index(end_ts, ts0, ts1, window_count=window_count)
        segment_id = f"seg_{len(segments):03d}"
        segments.append(
            ReplaySegment(
                segment_id=segment_id,
                segment_kind=str(current_kind),
                start_ts=start_ts,
                end_ts=end_ts,
                start_window=start_window,
                end_window=end_window,
                event_count=len(current_events),
                ranks=tuple(sorted({int(event.rank) for event in current_events})),
                collective_group_id=current_collective_group,
                event_ids=tuple(str(event.id) for event in current_events),
            )
        )
        current_events = []
        current_kind = None
        current_collective_group = None

    for event in events:
        event_kind = _segment_kind_for_event(event)
        event_group = event.collective_group_id
        if current_events:
            same_collective = (
                event_kind == "collective"
                and current_kind == "collective"
                and event_group == current_collective_group
            )
            same_non_collective = (
                event_kind != "collective"
                and current_kind == event_kind
                and current_collective_group is None
            )
            if not (same_collective or same_non_collective):
                flush()
        if not current_events:
            current_kind = event_kind
            current_collective_group = event_group if event_kind == "collective" else None
        current_events.append(event)
    flush()

    events_by_id = {event.id: event for event in collated.global_events}
    segment_indices_by_window: dict[int, list[int]] = {}
    for idx, segment in enumerate(segments):
        for window in range(segment.start_window, segment.end_window + 1):
            segment_indices_by_window.setdefault(window, []).append(idx)

    return ReplaySegmentBundle(
        trace_dir=str(collated.trace_dir),
        window_count=window_count,
        segments=tuple(segments),
        collated=collated,
        events_by_id=events_by_id,
        segment_indices_by_window={
            window: tuple(indices) for window, indices in sorted(segment_indices_by_window.items())
        },
    )


def _segment_indices_for_windows(
    bundle: ReplaySegmentBundle,
    refresh_windows: set[int],
) -> tuple[int, ...]:
    touched: set[int] = set()
    for window in refresh_windows:
        touched.update(bundle.segment_indices_by_window.get(window, ()))
    return tuple(sorted(touched))


def build_replay_segments_from_trace_dir(
    trace_dir: str | Path,
    *,
    max_events_per_rank: int | None = None,
    window_count: int = 24,
) -> ReplaySegmentBundle:
    bundle = load_trace_directory(trace_dir, max_events_per_rank=max_events_per_rank)
    collated = collate_trace_bundle(bundle)
    return build_replay_segments_from_collated(collated, window_count=window_count)


def assess_segment_compatibility(
    anchor_segments: ReplaySegmentBundle,
    target_segments: ReplaySegmentBundle,
    *,
    refresh_windows: tuple[int, ...] | list[int] | set[int],
) -> SegmentCompatibility:
    refresh_set = {int(value) for value in refresh_windows}
    anchor_touched = [anchor_segments.segments[idx] for idx in _segment_indices_for_windows(anchor_segments, refresh_set)]
    target_touched = [target_segments.segments[idx] for idx in _segment_indices_for_windows(target_segments, refresh_set)]
    pair_count = max(len(anchor_touched), len(target_touched), 1)
    kind_match_count = 0
    for idx in range(pair_count):
        anchor_segment = anchor_touched[idx] if idx < len(anchor_touched) else None
        target_segment = target_touched[idx] if idx < len(target_touched) else None
        if anchor_segment is None or target_segment is None:
            continue
        if anchor_segment.segment_kind == target_segment.segment_kind:
            kind_match_count += 1
    length_ratio = min(len(anchor_touched), len(target_touched)) / max(len(anchor_touched), len(target_touched), 1)
    kind_match_fraction = kind_match_count / pair_count
    anchor_touched_fraction = len(anchor_touched) / max(len(anchor_segments.segments), 1)
    target_touched_fraction = len(target_touched) / max(len(target_segments.segments), 1)
    max_touched_fraction = max(anchor_touched_fraction, target_touched_fraction)
    compatibility_score = 0.4 * kind_match_fraction + 0.2 * length_ratio + 0.4 * (1.0 - max_touched_fraction)
    return SegmentCompatibility(
        refresh_windows=tuple(sorted(refresh_set)),
        anchor_segment_count=len(anchor_touched),
        target_segment_count=len(target_touched),
        anchor_touched_fraction=anchor_touched_fraction,
        target_touched_fraction=target_touched_fraction,
        kind_match_count=kind_match_count,
        length_ratio=length_ratio,
        kind_match_fraction=kind_match_fraction,
        compatibility_score=compatibility_score,
        should_refresh=max_touched_fraction <= 0.3,
    )


def compose_hybrid_segment_collated_trace(
    anchor_segments: ReplaySegmentBundle,
    target_segments: ReplaySegmentBundle,
    *,
    refresh_windows: tuple[int, ...] | list[int] | set[int],
    trace_name: str = "hybrid_segment",
) -> CollatedTrace:
    if anchor_segments.collated is None or target_segments.collated is None:
        raise ValueError("segment bundles must carry collated traces")

    refresh_set = {int(value) for value in refresh_windows}
    anchor_collated = anchor_segments.collated
    target_collated = target_segments.collated
    target_by_id = target_segments.events_by_id
    anchor_by_id = anchor_segments.events_by_id

    mixed_events: list[CollatedEvent] = []
    selected_collective_groups: dict[str, CollectiveGroup] = {}

    for segment_index in _segment_indices_for_windows(anchor_segments, refresh_set):
        anchor_segment = anchor_segments.segments[segment_index]
        target_segment = (
            target_segments.segments[segment_index]
            if segment_index < len(target_segments.segments)
            else None
        )
        can_swap = (
            target_segment is not None
            and target_segment.segment_kind == anchor_segment.segment_kind
            and len(target_segment.event_ids) > 0
        )
        if not can_swap:
            chosen_events = [
                anchor_by_id[event_id]
                for event_id in anchor_segment.event_ids
                if event_id in anchor_by_id
            ]
            if not chosen_events:
                continue
            mixed_events.extend(chosen_events)
            if anchor_segment.collective_group_id is not None:
                original_group = anchor_collated.collective_groups.get(anchor_segment.collective_group_id)
                if original_group is not None:
                    selected_collective_groups[original_group.id] = original_group
            continue

        chosen_events = [
            target_by_id[event_id]
            for event_id in target_segment.event_ids
            if event_id in target_by_id
        ]
        if not chosen_events:
            continue

        source_start = int(chosen_events[0].ts)
        source_end = int(chosen_events[-1].ts)
        source_duration = max(source_end - source_start, 1)
        slot_start = int(anchor_segment.start_ts)
        slot_duration = max(int(anchor_segment.end_ts) - int(anchor_segment.start_ts), 1)
        local_collective_map: dict[str, str] = {}
        new_group_events: dict[str, list[CollatedEvent]] = {}

        for local_index, event in enumerate(chosen_events):
            relative = min(max(int(event.ts) - source_start, 0), source_duration)
            rebased_ts = slot_start + int((relative / source_duration) * slot_duration)
            new_group_id = None
            if event.collective_group_id is not None:
                new_group_id = local_collective_map.setdefault(
                    event.collective_group_id,
                    f"hyb_cg_{segment_index:04d}_{len(local_collective_map):02d}",
                )
            new_id = f"hyb:s{segment_index:04d}:e{local_index:04d}:r{event.rank}"
            new_event = CollatedEvent(
                id=new_id,
                rank=event.rank,
                ordinal=event.ordinal,
                source=event.source,
                ts=rebased_ts,
                pid=event.pid,
                tid=event.tid,
                module=event.module,
                api=event.api,
                op_type=event.op_type,
                extras=dict(event.extras),
                prev_event_id=event.prev_event_id,
                collective_group_id=new_group_id,
            )
            mixed_events.append(new_event)
            if new_group_id is not None:
                new_group_events.setdefault(new_group_id, []).append(new_event)

        if target_segment.collective_group_id is not None and target_segment.collective_group_id in local_collective_map:
            original_group_id = target_segment.collective_group_id
            original_group = (
                target_collated.collective_groups.get(original_group_id)
            )
            if original_group is not None:
                new_group_id = local_collective_map[original_group_id]
                group_events = tuple(new_group_events.get(new_group_id, ()))
                group_event_ids = tuple(event.id for event in group_events)
                group_ranks = tuple(sorted({event.rank for event in group_events}))
                selected_collective_groups[new_group_id] = CollectiveGroup(
                    id=new_group_id,
                    api=original_group.api,
                    op_type=original_group.op_type,
                    ranks=group_ranks,
                    event_ids=group_event_ids,
                )
    rank_events: dict[int, tuple[CollatedEvent, ...]] = {}
    for event in mixed_events:
        rank_events.setdefault(event.rank, []).append(event)

    return CollatedTrace(
        trace_dir=Path(trace_name),
        source=anchor_collated.source,
        rank_events={rank: tuple(events) for rank, events in sorted(rank_events.items())},
        global_events=tuple(mixed_events),
        collective_groups=selected_collective_groups,
        original_world_size=anchor_collated.world_size,
        profiled_rank_groups=dict(anchor_collated.profiled_rank_groups),
    )


def compose_segment_window_collated_trace(
    bundle: ReplaySegmentBundle,
    *,
    refresh_windows: tuple[int, ...] | list[int] | set[int],
    trace_name: str = "segment_window",
) -> CollatedTrace:
    if bundle.collated is None:
        raise ValueError("segment bundle must carry collated trace")

    refresh_set = {int(value) for value in refresh_windows}
    collated = bundle.collated
    by_id = bundle.events_by_id
    selected_events: list[CollatedEvent] = []
    selected_collective_groups: dict[str, CollectiveGroup] = {}

    for segment_index in _segment_indices_for_windows(bundle, refresh_set):
        segment = bundle.segments[segment_index]
        selected_events.extend(
            by_id[event_id]
            for event_id in segment.event_ids
            if event_id in by_id
        )
        if segment.collective_group_id is not None:
            original_group = collated.collective_groups.get(segment.collective_group_id)
            if original_group is not None:
                selected_collective_groups[original_group.id] = original_group

    rank_events: dict[int, tuple[CollatedEvent, ...]] = {}
    for event in selected_events:
        rank_events.setdefault(event.rank, []).append(event)

    return CollatedTrace(
        trace_dir=Path(trace_name),
        source=collated.source,
        rank_events={rank: tuple(events) for rank, events in sorted(rank_events.items())},
        global_events=tuple(selected_events),
        collective_groups=selected_collective_groups,
        original_world_size=collated.world_size,
        profiled_rank_groups=dict(collated.profiled_rank_groups),
    )


def filter_collated_trace_by_op_types(
    collated: CollatedTrace,
    *,
    op_types: tuple[str, ...] | list[str] | set[str],
    trace_name: str = "filtered_collated",
) -> CollatedTrace:
    selected_types = {str(value) for value in op_types}
    selected_events = [
        event for event in collated.global_events if event.op_type in selected_types
    ]
    selected_event_ids = {event.id for event in selected_events}
    selected_collective_groups: dict[str, CollectiveGroup] = {}

    if selected_event_ids:
        events_by_id = {event.id: event for event in selected_events}
        for group in collated.collective_groups.values():
            group_event_ids = tuple(
                event_id for event_id in group.event_ids if event_id in selected_event_ids
            )
            if not group_event_ids:
                continue
            group_ranks = tuple(
                sorted({events_by_id[event_id].rank for event_id in group_event_ids})
            )
            selected_collective_groups[group.id] = CollectiveGroup(
                id=group.id,
                api=group.api,
                op_type=group.op_type,
                ranks=group_ranks,
                event_ids=group_event_ids,
            )

    rank_events: dict[int, list[CollatedEvent]] = {}
    for event in selected_events:
        rank_events.setdefault(event.rank, []).append(event)

    return CollatedTrace(
        trace_dir=Path(trace_name),
        source=collated.source,
        rank_events={rank: tuple(events) for rank, events in sorted(rank_events.items())},
        global_events=tuple(selected_events),
        collective_groups=selected_collective_groups,
        original_world_size=collated.world_size,
        profiled_rank_groups=dict(collated.profiled_rank_groups),
    )


def compose_segment_window_filtered_collated_trace(
    bundle: ReplaySegmentBundle,
    *,
    refresh_windows: tuple[int, ...] | list[int] | set[int],
    op_types: tuple[str, ...] | list[str] | set[str],
    trace_name: str = "segment_window_filtered",
) -> CollatedTrace:
    selected = compose_segment_window_collated_trace(
        bundle,
        refresh_windows=refresh_windows,
        trace_name=trace_name,
    )
    return filter_collated_trace_by_op_types(
        selected,
        op_types=op_types,
        trace_name=trace_name,
    )


def compose_hybrid_segment_window_collated_trace(
    anchor_segments: ReplaySegmentBundle,
    target_segments: ReplaySegmentBundle,
    *,
    refresh_windows: tuple[int, ...] | list[int] | set[int],
    trace_name: str = "hybrid_segment_window",
) -> CollatedTrace:
    if anchor_segments.collated is None or target_segments.collated is None:
        raise ValueError("segment bundles must carry collated traces")

    refresh_set = {int(value) for value in refresh_windows}
    anchor_collated = anchor_segments.collated
    target_collated = target_segments.collated
    target_by_id = target_segments.events_by_id
    anchor_by_id = anchor_segments.events_by_id

    selected_events: list[CollatedEvent] = []
    selected_collective_groups: dict[str, CollectiveGroup] = {}

    for segment_index in _segment_indices_for_windows(anchor_segments, refresh_set):
        anchor_segment = anchor_segments.segments[segment_index]
        target_segment = (
            target_segments.segments[segment_index]
            if segment_index < len(target_segments.segments)
            else None
        )
        can_swap = (
            target_segment is not None
            and target_segment.segment_kind == anchor_segment.segment_kind
            and len(target_segment.event_ids) > 0
        )
        if not can_swap:
            chosen_events = [
                anchor_by_id[event_id]
                for event_id in anchor_segment.event_ids
                if event_id in anchor_by_id
            ]
            if not chosen_events:
                continue
            selected_events.extend(chosen_events)
            if anchor_segment.collective_group_id is not None:
                original_group = anchor_collated.collective_groups.get(anchor_segment.collective_group_id)
                if original_group is not None:
                    selected_collective_groups[original_group.id] = original_group
            continue

        chosen_events = [
            target_by_id[event_id]
            for event_id in target_segment.event_ids
            if event_id in target_by_id
        ]
        if not chosen_events:
            continue

        source_start = int(chosen_events[0].ts)
        source_end = int(chosen_events[-1].ts)
        source_duration = max(source_end - source_start, 1)
        slot_start = int(anchor_segment.start_ts)
        slot_duration = max(int(anchor_segment.end_ts) - int(anchor_segment.start_ts), 1)
        local_collective_map: dict[str, str] = {}
        new_group_events: dict[str, list[CollatedEvent]] = {}

        for local_index, event in enumerate(chosen_events):
            relative = min(max(int(event.ts) - source_start, 0), source_duration)
            rebased_ts = slot_start + int((relative / source_duration) * slot_duration)
            new_group_id = None
            if event.collective_group_id is not None:
                new_group_id = local_collective_map.setdefault(
                    event.collective_group_id,
                    f"hyb_delta_cg_{segment_index:04d}_{len(local_collective_map):02d}",
                )
            new_event = CollatedEvent(
                id=f"hyb_delta:s{segment_index:04d}:e{local_index:04d}:r{event.rank}",
                rank=event.rank,
                ordinal=event.ordinal,
                source=event.source,
                ts=rebased_ts,
                pid=event.pid,
                tid=event.tid,
                module=event.module,
                api=event.api,
                op_type=event.op_type,
                extras=dict(event.extras),
                prev_event_id=event.prev_event_id,
                collective_group_id=new_group_id,
            )
            selected_events.append(new_event)
            if new_group_id is not None:
                new_group_events.setdefault(new_group_id, []).append(new_event)

        if target_segment.collective_group_id is not None and target_segment.collective_group_id in local_collective_map:
            original_group = target_collated.collective_groups.get(target_segment.collective_group_id)
            if original_group is not None:
                new_group_id = local_collective_map[target_segment.collective_group_id]
                group_events = tuple(new_group_events.get(new_group_id, ()))
                selected_collective_groups[new_group_id] = CollectiveGroup(
                    id=new_group_id,
                    api=original_group.api,
                    op_type=original_group.op_type,
                    ranks=tuple(sorted({event.rank for event in group_events})),
                    event_ids=tuple(event.id for event in group_events),
                )

    rank_events: dict[int, tuple[CollatedEvent, ...]] = {}
    for event in selected_events:
        rank_events.setdefault(event.rank, []).append(event)

    return CollatedTrace(
        trace_dir=Path(trace_name),
        source=anchor_collated.source,
        rank_events={rank: tuple(events) for rank, events in sorted(rank_events.items())},
        global_events=tuple(selected_events),
        collective_groups=selected_collective_groups,
        original_world_size=anchor_collated.world_size,
        profiled_rank_groups=dict(anchor_collated.profiled_rank_groups),
    )
