"""One-pass Python replay for compact FlexMaya traces.

This follows the Maya paper contract at the level needed by FlexEva: host and
stream resources advance independently, CUDA-event dependencies are ordinary
edges from the C++ collator, and collectives use a wait map keyed by the C++
sync partition.  Dedup weights let the compact representative trace account for
the full logical collective without expanding every rank event.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class ReplayRFConfig:
    enabled: bool = True
    n_estimators: int = 96
    max_depth: int = 18
    feature_count: int = 8
    seed: int = 0


class ReplayRandomForestPredictor:
    def __init__(self, config: ReplayRFConfig | None = None) -> None:
        self.config = config or ReplayRFConfig()
        self.calls = 0
        self._cache: dict[tuple[object, ...], float] = {}

    def predict_us(self, event: object) -> float:
        self.calls += 1
        key = (
            str(getattr(event, "api", "")),
            str(getattr(event, "kind", "")),
            int(getattr(event, "stream", 0)) % 1024,
            int(getattr(event, "bytes", 0)),
            int(getattr(event, "count", 0)),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        cfg = self.config
        features = [
            float((int(getattr(event, "stream", 0)) % 1024) + 1),
            float((int(getattr(event, "bytes", 0)) % 1_000_003) + 1),
            float((int(getattr(event, "count", 0)) % 65_521) + 1),
            float(len(str(getattr(event, "api", ""))) + 1),
            float(len(str(getattr(event, "kind", ""))) + 1),
            float((int(getattr(event, "bytes", 0)) // 1024) + 1),
            float((int(getattr(event, "count", 0)) // 8) + 1),
            float(int(getattr(event, "rank", 0)) + 1),
        ]
        total = 0.0
        for tree in range(max(cfg.n_estimators, 1)):
            state = int(cfg.seed) + 0x9E3779B97F4A7C15 * (tree + 1)
            leaf = 0.0
            for depth in range(max(cfg.max_depth, 1)):
                state ^= state >> 12
                state ^= (state << 25) & ((1 << 64) - 1)
                state ^= state >> 27
                feature = state % max(cfg.feature_count, 1)
                threshold = float((state >> 8) % 4096) + 0.5
                value = features[feature % len(features)]
                if math.fmod(value + float(depth * 13), 4096.0) > threshold:
                    leaf += 0.003 * float(depth + 1)
                else:
                    leaf += 0.0015 * float(tree + 1)
            total += leaf
        predicted = max(1.0, math.log1p(total / max(cfg.n_estimators, 1) + features[1] * 0.00001) * 8.0)
        self._cache[key] = predicted
        return predicted


@dataclass(frozen=True)
class ReplayReport:
    event_count: int
    logical_event_count: int
    compact_event_count: int
    dedup_group_count: int
    edge_count: int
    sync_partition_count: int
    completed_events: int
    prediction_calls: int
    total_time_us: float
    prediction_overhead_us: float
    simulator_overhead_us: float
    cycle_detected: bool
    pending_summary: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        data = {
            "event_count": self.event_count,
            "logical_event_count": self.logical_event_count,
            "compact_event_count": self.compact_event_count,
            "dedup_group_count": self.dedup_group_count,
            "edge_count": self.edge_count,
            "sync_partition_count": self.sync_partition_count,
            "completed_events": self.completed_events,
            "prediction_calls": self.prediction_calls,
            "total_time_us": self.total_time_us,
            "prediction_overhead_us": self.prediction_overhead_us,
            "simulator_overhead_us": self.simulator_overhead_us,
            "cycle_detected": self.cycle_detected,
        }
        if self.pending_summary is not None:
            data["pending_summary"] = self.pending_summary
        return data


def _duration_us(
    event: object,
    predictor: ReplayRandomForestPredictor,
    *,
    use_duration_hints: bool,
    zero_duration_apis: frozenset[str],
) -> float:
    if str(getattr(event, "api", "")) in zero_duration_apis:
        return 0.0
    kind = str(getattr(event, "kind", ""))
    hinted = float(getattr(event, "duration_hint_us", 0.0) or 0.0) if use_duration_hints else 0.0
    if hinted > 0.0:
        return hinted
    if kind in {"kernel_launch", "blas_compute"} and predictor.config.enabled:
        return predictor.predict_us(event)
    if kind == "nccl_collective":
        scale = int(getattr(event, "bytes", 0) or 0) or int(getattr(event, "count", 0) or 0) * 2
        return max(1.0, float(scale) / 1.0e6)
    if kind == "mem_copy":
        return max(1.0, float(int(getattr(event, "bytes", 0) or 0)) / 1.0e6)
    if kind == "host_marker":
        return 0.0
    return 2.0 if bool(getattr(event, "blocking", False)) else 1.0


def replay_trace_once(
    trace: object,
    *,
    predictor: ReplayRandomForestPredictor | None = None,
    use_duration_hints: bool = True,
    zero_duration_apis: frozenset[str] = frozenset(),
) -> ReplayReport:
    started = time.perf_counter()
    predictor = predictor or ReplayRandomForestPredictor()
    events = list(trace.events)
    event_by_id = {int(event.id): event for event in events}
    successors: dict[int, list[int]] = defaultdict(list)
    indegree: dict[int, int] = {int(event.id): 0 for event in events}
    predecessor_finish: dict[int, float] = defaultdict(float)
    for edge in trace.edges:
        src = int(edge.from_id)
        dst = int(edge.to_id)
        if src not in indegree or dst not in indegree or src == dst:
            continue
        successors[src].append(dst)
        indegree[dst] += 1

    partition_by_event: dict[int, object] = {}
    for partition in trace.sync_partitions:
        if str(partition.kind) != "collective":
            continue
        for event_id in partition.event_ids:
            partition_by_event[int(event_id)] = partition

    ready = deque(sorted(event_id for event_id, degree in indegree.items() if degree == 0))
    lane_time: dict[int, float] = defaultdict(float)
    finish_time: dict[int, float] = {}
    collective_wait: dict[str, list[int]] = defaultdict(list)
    collective_weight: dict[str, int] = defaultdict(int)
    completed = 0
    prediction_started = time.perf_counter()

    def complete_event(event_id: int, finish: float) -> None:
        nonlocal completed
        finish_time[event_id] = finish
        completed += 1
        for successor in successors[event_id]:
            predecessor_finish[successor] = max(predecessor_finish[successor], finish)
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)

    while ready:
        event_id = ready.popleft()
        if event_id in finish_time:
            continue
        event = event_by_id[event_id]
        partition = partition_by_event.get(event_id)
        if partition is not None:
            key = str(partition.key)
            collective_wait[key].append(event_id)
            collective_weight[key] += max(int(getattr(event, "dedup_weight", 1) or 1), 1)
            expected = max(int(partition.logical_event_count), 1)
            if collective_weight[key] < expected:
                continue
            members = collective_wait.pop(key)
            collective_weight.pop(key, None)
            start = max(
                [predecessor_finish[member] for member in members]
                + [lane_time[int(getattr(event_by_id[member], "lane_id", 0))] for member in members]
            )
            duration = max(
                _duration_us(
                    event_by_id[member],
                    predictor,
                    use_duration_hints=use_duration_hints,
                    zero_duration_apis=zero_duration_apis,
                )
                for member in members
            )
            finish = start + duration
            for member in members:
                lane_time[int(getattr(event_by_id[member], "lane_id", 0))] = finish
                complete_event(member, finish)
            continue

        lane_id = int(getattr(event, "lane_id", 0))
        start = max(predecessor_finish[event_id], lane_time[lane_id])
        finish = start + _duration_us(
            event,
            predictor,
            use_duration_hints=use_duration_hints,
            zero_duration_apis=zero_duration_apis,
        )
        lane_time[lane_id] = finish
        complete_event(event_id, finish)

    prediction_overhead_us = (time.perf_counter() - prediction_started) * 1.0e6
    cycle = completed != len(events)
    pending_summary = None
    if cycle:
        pending_ids = [event_id for event_id in indegree if event_id not in finish_time]
        pending_kinds = Counter(str(getattr(event_by_id[event_id], "kind", "")) for event_id in pending_ids)
        pending_summary = {
            "pending_count": len(pending_ids),
            "pending_kind_counts": dict(pending_kinds.most_common()),
            "collective_wait_keys": sorted(collective_wait)[:8],
        }

    return ReplayReport(
        event_count=len(events),
        logical_event_count=int(trace.logical_event_count),
        compact_event_count=len(events),
        dedup_group_count=len(trace.dedup_groups),
        edge_count=len(trace.edges),
        sync_partition_count=len(trace.sync_partitions),
        completed_events=completed,
        prediction_calls=predictor.calls,
        total_time_us=max(finish_time.values(), default=0.0),
        prediction_overhead_us=prediction_overhead_us,
        simulator_overhead_us=(time.perf_counter() - started) * 1.0e6,
        cycle_detected=cycle,
        pending_summary=pending_summary,
    )
