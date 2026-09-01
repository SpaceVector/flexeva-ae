#!/usr/bin/env python3
"""Shared Table 4 case definitions and trace-comparison helpers."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class MegatronCase:
    name: str
    parameter_scale: str
    steps: int
    global_batch_size: int
    seq_len: int
    hidden_size: int
    num_layers: int
    num_heads: int
    vocab_size: int
    tp: int
    pp: int
    dp: int
    world_size: int
    micro_batches: int
    schedule: str
    dtype: str


def default_cases() -> tuple[MegatronCase, ...]:
    return (
        MegatronCase("megatron_2p7b_8gpu", "2.7B", 1, 256, 2048, 2560, 32, 32, 32000, 1, 8, 1, 8, 256, "1f1b", "bf16"),
        MegatronCase("megatron_18p4b_16gpu", "18.4B", 1, 512, 2048, 6144, 40, 48, 32000, 2, 8, 1, 16, 512, "1f1b", "bf16"),
        MegatronCase("megatron_2p7b_16gpu_dp2", "2.7B", 1, 512, 2048, 2560, 32, 32, 32000, 1, 8, 2, 16, 256, "1f1b", "bf16"),
        MegatronCase("megatron_18p4b_16gpu_dp2", "18.4B", 1, 512, 2048, 6144, 40, 48, 32000, 1, 8, 2, 16, 512, "1f1b", "bf16"),
    )


def event_signature(event: object) -> tuple[object, ...]:
    collective_group = str(getattr(event, "collective_group", ""))
    collective_base = collective_group.split("|occ:", 1)[0]
    kind = str(getattr(event, "kind", ""))
    return (
        str(getattr(event, "api", "")),
        kind,
        str(getattr(event, "code_partition", "")),
        int(getattr(event, "stream", 0)),
        int(getattr(event, "bytes", 0)),
        int(getattr(event, "count", 0)),
        collective_base if kind == "nccl_collective" else "",
    )


def weighted_event_counter(trace: object, *, use_dedup_weight: bool) -> Counter[tuple[object, ...]]:
    counter: Counter[tuple[object, ...]] = Counter()
    for event in trace.events:
        weight = int(getattr(event, "dedup_weight", 1) or 1) if use_dedup_weight else 1
        counter[event_signature(event)] += weight
    return counter


def filtered_weighted_event_counter(
    trace: object,
    *,
    use_dedup_weight: bool,
    kinds: frozenset[str] | None = None,
    excluded_apis: frozenset[str] = frozenset(),
) -> Counter[tuple[object, ...]]:
    counter: Counter[tuple[object, ...]] = Counter()
    for event in trace.events:
        if kinds is not None and str(getattr(event, "kind", "")) not in kinds:
            continue
        if str(getattr(event, "api", "")) in excluded_apis:
            continue
        weight = int(getattr(event, "dedup_weight", 1) or 1) if use_dedup_weight else 1
        counter[event_signature(event)] += weight
    return counter


def weighted_edge_counter(trace: object, *, reason: str | None = None) -> Counter[tuple[object, ...]]:
    events = {int(event.id): event for event in trace.events}
    counter: Counter[tuple[object, ...]] = Counter()
    for edge in trace.edges:
        edge_reason = str(edge.reason)
        if reason is not None and edge_reason != reason:
            continue
        source = events.get(int(edge.from_id))
        target = events.get(int(edge.to_id))
        if source is None or target is None:
            continue
        source_weight = int(getattr(source, "dedup_weight", 1) or 1)
        target_weight = int(getattr(target, "dedup_weight", 1) or 1)
        counter[(edge_reason, event_signature(source), event_signature(target))] += max(source_weight, target_weight)
    return counter


def trace_window_counter(trace: object) -> Counter[tuple[object, ...]]:
    events_by_id = {int(event.id): event for event in trace.events}
    counter: Counter[tuple[object, ...]] = Counter()
    for partition in trace.sync_partitions:
        if str(partition.kind) != "trace_window":
            continue
        signatures = tuple(
            event_signature(events_by_id[event_id])
            for event_id in partition.event_ids
            if event_id in events_by_id
        )
        counter[(str(partition.code_partition), signatures)] += max(int(partition.logical_event_count), 1)
    return counter


def weighted_jaccard(left: Counter[tuple[object, ...]], right: Counter[tuple[object, ...]]) -> float:
    keys = set(left) | set(right)
    numerator = sum(min(left.get(key, 0), right.get(key, 0)) for key in keys)
    denominator = sum(max(left.get(key, 0), right.get(key, 0)) for key in keys)
    return float(numerator) / float(denominator or 1)


def cosine(left: Counter[tuple[object, ...]], right: Counter[tuple[object, ...]]) -> float:
    keys = set(left) | set(right)
    dot = sum(left.get(key, 0) * right.get(key, 0) for key in keys)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return float(dot) / float(left_norm * right_norm or 1.0)
