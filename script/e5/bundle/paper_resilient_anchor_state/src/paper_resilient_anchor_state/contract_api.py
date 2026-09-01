"""Paper-facing RAS contract prototype.

This module implements the small API surface used by the Section 4 RAS and AL
interfaces.  It is intentionally deterministic and self-contained; it adapts
the public anchor-state objects in this package without importing the larger
internal analyzer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .state import (
    AnchorCodeState,
    BoundaryContextCapsule,
    DryRunProgramLogicCapture,
    ProgramLogicPoint,
)


@dataclass(frozen=True)
class LayerEvent:
    layer_id: str
    event_id: str
    lane_id: str
    order: int
    control_region_id: str | None
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class DataDependency:
    producer_event: str
    consumer_event: str
    dependency_fields: tuple[str, ...]


@dataclass(frozen=True)
class RASPartition:
    layer_id: str
    partition_id: str
    event_ids: tuple[str, ...]
    lane_ids: tuple[str, ...]
    control_region_id: str | None


@dataclass(frozen=True)
class BoundaryRecord:
    upper_partition: str
    boundary_event: str
    transform_name: str
    lower_region_key: str


@dataclass(frozen=True)
class AnchorLineageEdge:
    lower_partition: str
    upper_partition: str
    upper_event: str
    edge_kind: str
    dependency_fields: tuple[str, ...] = ()
    lower_event: str | None = None


TransformCallback = Callable[[BoundaryRecord, LayerEvent | None], Iterable[LayerEvent | Mapping[str, Any]]]


def lane(event: LayerEvent) -> tuple[str, int, str | None]:
    return event.lane_id, event.order, event.control_region_id


def payload(event: LayerEvent) -> Mapping[str, Any]:
    return event.payload


def datadep(
    events: Iterable[LayerEvent],
    partitions: Iterable[RASPartition] = (),
) -> tuple[DataDependency, ...]:
    del partitions
    ordered = _ordered_events(events)
    all_event_ids = {event.event_id for event in ordered}
    seen_events: set[str] = set()
    latest_writer: dict[str, str] = {}
    edge_fields: dict[tuple[str, str], set[str]] = defaultdict(set)

    for event in ordered:
        event_payload = event.payload
        depends_on = set(_as_tuple(event_payload.get("depends_on")))

        for explicit in sorted(depends_on & seen_events):
            edge_fields[(explicit, event.event_id)].add(f"event:{explicit}")

        read_fields = set(_as_tuple(event_payload.get("reads")))
        read_fields.update(_as_tuple(event_payload.get("source_fields")))
        read_fields.update(item for item in depends_on if item not in all_event_ids)

        for field in sorted(read_fields):
            producer = latest_writer.get(field)
            if producer is not None and producer != event.event_id:
                edge_fields[(producer, event.event_id)].add(field)

        write_fields = set(_as_tuple(event_payload.get("writes")))
        write_fields.update(_as_tuple(event_payload.get("produces")))
        for field in sorted(write_fields):
            latest_writer[field] = event.event_id
        seen_events.add(event.event_id)

    return tuple(
        DataDependency(
            producer_event=producer,
            consumer_event=consumer,
            dependency_fields=tuple(sorted(fields)),
        )
        for (producer, consumer), fields in sorted(edge_fields.items())
    )


def partition(
    events: Iterable[LayerEvent],
    old_partitions: Iterable[RASPartition] = (),
) -> tuple[RASPartition, ...]:
    ordered = _ordered_events(events)
    if not ordered:
        return ()

    old_by_event: dict[str, RASPartition] = {}
    for old_partition in old_partitions:
        for event_id in old_partition.event_ids:
            old_by_event[event_id] = old_partition

    groups: list[list[LayerEvent]] = []
    current: list[LayerEvent] = []
    for event in ordered:
        if current and (
            _context_changed(current[-1], event)
            or _is_split_event(current[-1])
            or _is_split_event(event)
        ):
            groups.append(current)
            current = []
        current.append(event)
        if _is_split_event(event):
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    used_partition_ids: set[str] = set()
    rows: list[RASPartition] = []
    for index, group in enumerate(groups):
        layer_id = group[0].layer_id
        event_ids = tuple(event.event_id for event in group)
        lane_ids = tuple(dict.fromkeys(event.lane_id for event in group))
        control_regions = tuple(dict.fromkeys(event.control_region_id for event in group))
        control_region_id = control_regions[0] if len(control_regions) == 1 else None
        partition_id = _matching_old_partition_id(event_ids, old_by_event)
        if partition_id is None or partition_id in used_partition_ids:
            partition_id = f"{layer_id}:p{index:04d}"
        used_partition_ids.add(partition_id)
        rows.append(
            RASPartition(
                layer_id=layer_id,
                partition_id=partition_id,
                event_ids=event_ids,
                lane_ids=lane_ids,
                control_region_id=control_region_id,
            )
        )
    return tuple(rows)


def boundary(
    partitions: Iterable[RASPartition],
    events: Iterable[LayerEvent],
    *,
    transform_name: str = "backend",
) -> tuple[BoundaryRecord, ...]:
    event_by_id = {event.event_id: event for event in events}
    rows: list[BoundaryRecord] = []
    for ras_partition in partitions:
        for event_id in ras_partition.event_ids:
            event = event_by_id.get(event_id)
            if event is None or not _is_boundary_event(event):
                continue
            lower_region_key = event.payload.get("lower_region_key")
            if lower_region_key is None:
                lower_region_key = event.payload.get("coarse_lower_region_key")
            if lower_region_key is None:
                lower_region_key = event.payload.get("region_key")
            if lower_region_key is None:
                lower_region_key = f"{transform_name}:{event.event_id}"
            rows.append(
                BoundaryRecord(
                    upper_partition=ras_partition.partition_id,
                    boundary_event=event.event_id,
                    transform_name=transform_name,
                    lower_region_key=str(lower_region_key),
                )
            )
    return tuple(rows)


def apply(
    boundary_records: Iterable[BoundaryRecord],
    events: Iterable[LayerEvent] = (),
    *,
    transform: TransformCallback | None = None,
    lower_layer_id: str | None = None,
) -> tuple[LayerEvent, ...]:
    event_by_id = {event.event_id: event for event in events}
    rows: list[LayerEvent] = []
    for record in boundary_records:
        boundary_event = event_by_id.get(record.boundary_event)
        if transform is not None:
            produced = transform(record, boundary_event)
            rows.extend(
                _coerce_lower_event(
                    item,
                    record=record,
                    boundary_event=boundary_event,
                    lower_layer_id=lower_layer_id,
                    index=index,
                )
                for index, item in enumerate(produced)
            )
            continue

        declared_events = _as_declared_lower_events(boundary_event)
        rows.extend(
            _coerce_lower_event(
                item,
                record=record,
                boundary_event=boundary_event,
                lower_layer_id=lower_layer_id,
                index=index,
            )
            for index, item in enumerate(declared_events)
        )
    return tuple(sorted(rows, key=lambda event: (event.layer_id, event.order, event.event_id)))


def link(
    lower_partitions: Iterable[RASPartition],
    upper_partitions: Iterable[RASPartition],
    boundary_records: Iterable[BoundaryRecord],
    upper_dependencies: Iterable[DataDependency] = (),
    lower_dependencies: Iterable[DataDependency] = (),
    *,
    lower_events: Iterable[LayerEvent] = (),
    upper_events: Iterable[LayerEvent] = (),
) -> tuple[AnchorLineageEdge, ...]:
    del upper_events
    lower_partition_rows = tuple(lower_partitions)
    upper_partition_rows = tuple(upper_partitions)
    boundary_rows = tuple(boundary_records)
    upper_dep_rows = tuple(upper_dependencies)
    lower_dep_rows = tuple(lower_dependencies)
    lower_event_by_id = {event.event_id: event for event in lower_events}

    upper_event_to_partition = _event_to_partition(upper_partition_rows)
    lower_event_to_partition = _event_to_partition(lower_partition_rows)
    upper_dep_fields = {
        (dep.producer_event, dep.consumer_event): dep.dependency_fields
        for dep in upper_dep_rows
    }
    field_to_upper_producers: dict[str, set[str]] = defaultdict(set)
    for dep in upper_dep_rows:
        for field in dep.dependency_fields:
            if not field.startswith("event:"):
                field_to_upper_producers[field].add(dep.producer_event)

    rows: list[AnchorLineageEdge] = []
    seen: set[tuple[Any, ...]] = set()

    def add(edge: AnchorLineageEdge) -> None:
        key = (
            edge.lower_partition,
            edge.upper_partition,
            edge.upper_event,
            edge.edge_kind,
            edge.dependency_fields,
            edge.lower_event,
        )
        if key not in seen:
            seen.add(key)
            rows.append(edge)

    record_to_lower_partitions: dict[BoundaryRecord, tuple[RASPartition, ...]] = {}
    for record in boundary_rows:
        matching_lower_partitions = tuple(
            ras_partition
            for ras_partition in lower_partition_rows
            if _lower_partition_matches_record(ras_partition, record, lower_event_by_id)
        )
        record_to_lower_partitions[record] = matching_lower_partitions
        for lower_partition in matching_lower_partitions:
            add(
                AnchorLineageEdge(
                    lower_partition=lower_partition.partition_id,
                    upper_partition=record.upper_partition,
                    upper_event=record.boundary_event,
                    edge_kind="boundary",
                    dependency_fields=(record.lower_region_key,),
                )
            )

    for record in boundary_rows:
        for producer_event in _upstream_events(record.boundary_event, upper_dep_rows):
            upper_partition = upper_event_to_partition.get(producer_event)
            if upper_partition is None:
                continue
            fields = _fields_between(producer_event, record.boundary_event, upper_dep_rows, upper_dep_fields)
            for lower_partition in record_to_lower_partitions.get(record, ()):
                add(
                    AnchorLineageEdge(
                        lower_partition=lower_partition.partition_id,
                        upper_partition=upper_partition,
                        upper_event=producer_event,
                        edge_kind="data",
                        dependency_fields=fields,
                    )
                )

    for dep in lower_dep_rows:
        lower_partition_id = lower_event_to_partition.get(dep.consumer_event)
        if lower_partition_id is None:
            continue
        upper_refs = set(_upper_refs_from_lower_event(lower_event_by_id.get(dep.producer_event)))
        upper_refs.update(_upper_refs_from_lower_event(lower_event_by_id.get(dep.consumer_event)))
        for field in dep.dependency_fields:
            upper_refs.update(field_to_upper_producers.get(field, ()))
        for upper_ref in sorted(upper_refs):
            upper_partition = upper_event_to_partition.get(upper_ref)
            if upper_partition is None:
                continue
            add(
                AnchorLineageEdge(
                    lower_partition=lower_partition_id,
                    upper_partition=upper_partition,
                    upper_event=upper_ref,
                    edge_kind="data",
                    dependency_fields=dep.dependency_fields,
                    lower_event=dep.consumer_event,
                )
            )

    return tuple(rows)


def event_from_program_logic_point(
    point: ProgramLogicPoint,
    *,
    layer_id: str = "source",
    order: int = 0,
    lane_id: str = "program:logic",
    control_region_id: str | None = None,
) -> LayerEvent:
    source_fields = tuple(str(field) for field in point.source_fields)
    return LayerEvent(
        layer_id=layer_id,
        event_id=point.name,
        lane_id=lane_id,
        order=order,
        control_region_id=control_region_id or point.source,
        payload={
            "kind": "program_logic",
            "logic_kind": point.kind.value,
            "value": point.value,
            "source": point.source,
            "source_path": point.source_path,
            "lineno": point.lineno,
            "branch_ids": point.branch_ids,
            "source_fields": source_fields,
            "reads": source_fields,
            "writes": (point.name,),
        },
    )


def event_from_boundary_capsule(
    capsule: BoundaryContextCapsule,
    *,
    layer_id: str = "source",
    order: int = 0,
    lane_id: str = "program:boundary",
    control_region_id: str | None = None,
) -> LayerEvent:
    return LayerEvent(
        layer_id=layer_id,
        event_id=capsule.capsule_id,
        lane_id=lane_id,
        order=order,
        control_region_id=control_region_id or capsule.site_signature,
        payload={
            "kind": "boundary",
            "backend_visible": True,
            "boundary_kind": capsule.boundary_kind,
            "callee_name": capsule.callee_name,
            "source_path": capsule.source_path,
            "lineno": capsule.lineno,
            "branch_ids": capsule.branch_ids,
            "positional_arg_kinds": capsule.positional_arg_kinds,
            "keyword_arg_names": capsule.keyword_arg_names,
            "reads": capsule.keyword_arg_names,
            "lower_region_key": f"{capsule.callee_name}:{capsule.capsule_id}",
        },
    )


def events_from_dry_run_capture(
    capture: DryRunProgramLogicCapture,
    *,
    layer_id: str = "source",
) -> tuple[LayerEvent, ...]:
    rows: list[LayerEvent] = []
    control_region = capture.logic_scope.scope_id if capture.logic_scope is not None else capture.code_path
    for index, point in enumerate(capture.program_logic.points):
        rows.append(
            event_from_program_logic_point(
                point,
                layer_id=layer_id,
                order=index,
                control_region_id=control_region,
            )
        )
    boundary_start = len(rows)
    for offset, capsule in enumerate(capture.boundary_capsules):
        rows.append(
            event_from_boundary_capsule(
                capsule,
                layer_id=layer_id,
                order=boundary_start + offset,
                control_region_id=control_region,
            )
        )
    return tuple(rows)


def events_from_anchor_code_state(
    state: AnchorCodeState,
    *,
    layer_id: str = "code",
) -> tuple[LayerEvent, ...]:
    rows: list[LayerEvent] = []
    for index, hunk in enumerate(state.mutation_hunks):
        after_path = hunk.after_path or hunk.before_path
        path_name = Path(after_path).name
        rows.append(
            LayerEvent(
                layer_id=layer_id,
                event_id=f"code_hunk:{index:04d}",
                lane_id=f"file:{path_name}",
                order=index,
                control_region_id=after_path,
                payload={
                    "kind": "code_mutation",
                    "before_path": hunk.before_path,
                    "after_path": hunk.after_path,
                    "before_lines": (hunk.before_start_line, hunk.before_end_line),
                    "after_lines": (hunk.after_start_line, hunk.after_end_line),
                    "summary": hunk.summary,
                    "writes": _fields_from_hunk_text(
                        "\n".join(hunk.before_lines + hunk.after_lines + (hunk.summary,))
                    ),
                },
            )
        )
    return tuple(rows)


def _ordered_events(events: Iterable[LayerEvent]) -> tuple[LayerEvent, ...]:
    return tuple(sorted(events, key=lambda event: (event.layer_id, event.order, event.event_id)))


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(str(key) for key in value.keys())
    try:
        return tuple(str(item) for item in value)
    except TypeError:
        return (str(value),)


def _is_boundary_event(event: LayerEvent) -> bool:
    event_payload = event.payload
    kind = str(event_payload.get("kind", "")).lower()
    return bool(
        event_payload.get("backend_visible")
        or event_payload.get("boundary")
        or event_payload.get("is_boundary")
        or kind in {"boundary", "backend_call", "opaque_call"}
    )


def _is_split_event(event: LayerEvent) -> bool:
    event_payload = event.payload
    kind = str(event_payload.get("kind", "")).lower()
    return bool(
        _is_boundary_event(event)
        or event_payload.get("synchronization")
        or event_payload.get("sync")
        or kind in {"sync", "synchronization", "collective_wait", "barrier"}
    )


def _context_changed(left: LayerEvent, right: LayerEvent) -> bool:
    return (
        left.lane_id != right.lane_id
        or left.control_region_id != right.control_region_id
        or left.layer_id != right.layer_id
    )


def _matching_old_partition_id(
    event_ids: tuple[str, ...],
    old_by_event: Mapping[str, RASPartition],
) -> str | None:
    candidates = [old_by_event[event_id] for event_id in event_ids if event_id in old_by_event]
    if not candidates:
        return None
    first = candidates[0]
    if all(candidate.partition_id == first.partition_id for candidate in candidates):
        return first.partition_id
    return None


def _as_declared_lower_events(boundary_event: LayerEvent | None) -> tuple[Any, ...]:
    if boundary_event is None:
        return ()
    for key in ("lower_events", "generated_events", "emits"):
        value = boundary_event.payload.get(key)
        if value is not None:
            if isinstance(value, tuple):
                return value
            if isinstance(value, list):
                return tuple(value)
            return (value,)
    return ()


def _coerce_lower_event(
    item: LayerEvent | Mapping[str, Any],
    *,
    record: BoundaryRecord,
    boundary_event: LayerEvent | None,
    lower_layer_id: str | None,
    index: int,
) -> LayerEvent:
    if isinstance(item, LayerEvent):
        return item
    item_payload = item.get("payload")
    event_payload = dict(item_payload) if isinstance(item_payload, Mapping) else dict(item)
    for key in ("layer_id", "event_id", "lane_id", "order", "control_region_id", "payload"):
        event_payload.pop(key, None)
    if "upper_boundary_event" not in event_payload:
        event_payload["upper_boundary_event"] = record.boundary_event
    if "upper_partition" not in event_payload:
        event_payload["upper_partition"] = record.upper_partition
    if "lower_region_key" not in event_payload:
        event_payload["lower_region_key"] = record.lower_region_key
    source_lane = boundary_event.lane_id if boundary_event is not None else "lower:any"
    source_order = boundary_event.order if boundary_event is not None else 0
    return LayerEvent(
        layer_id=str(item.get("layer_id", lower_layer_id or f"{record.transform_name}:lower")),
        event_id=str(item.get("event_id", f"{record.boundary_event}:lower:{index}")),
        lane_id=str(item.get("lane_id", source_lane)),
        order=int(item.get("order", source_order * 1000 + index)),
        control_region_id=str(item.get("control_region_id", record.lower_region_key)),
        payload=event_payload,
    )


def _event_to_partition(partitions: Iterable[RASPartition]) -> dict[str, str]:
    rows: dict[str, str] = {}
    for ras_partition in partitions:
        for event_id in ras_partition.event_ids:
            rows[event_id] = ras_partition.partition_id
    return rows


def _lower_partition_matches_record(
    ras_partition: RASPartition,
    record: BoundaryRecord,
    lower_event_by_id: Mapping[str, LayerEvent],
) -> bool:
    if ras_partition.control_region_id == record.lower_region_key:
        return True
    if ras_partition.partition_id == record.lower_region_key:
        return True
    for event_id in ras_partition.event_ids:
        event = lower_event_by_id.get(event_id)
        if event is None:
            continue
        if event.payload.get("lower_region_key") == record.lower_region_key:
            return True
        if event.payload.get("upper_boundary_event") == record.boundary_event:
            return True
    return False


def _upstream_events(boundary_event: str, dependencies: tuple[DataDependency, ...]) -> tuple[str, ...]:
    producers_by_consumer: dict[str, set[str]] = defaultdict(set)
    for dep in dependencies:
        producers_by_consumer[dep.consumer_event].add(dep.producer_event)

    rows: list[str] = []
    seen: set[str] = set()

    def visit(event_id: str) -> None:
        for producer in sorted(producers_by_consumer.get(event_id, ())):
            if producer in seen:
                continue
            seen.add(producer)
            rows.append(producer)
            visit(producer)

    visit(boundary_event)
    return tuple(rows)


def _fields_between(
    producer_event: str,
    boundary_event: str,
    dependencies: tuple[DataDependency, ...],
    upper_dep_fields: Mapping[tuple[str, str], tuple[str, ...]],
) -> tuple[str, ...]:
    direct = upper_dep_fields.get((producer_event, boundary_event))
    if direct is not None:
        return direct
    fields: set[str] = set()
    consumers_by_producer: dict[str, set[str]] = defaultdict(set)
    for dep in dependencies:
        consumers_by_producer[dep.producer_event].add(dep.consumer_event)

    queue = [producer_event]
    seen = {producer_event}
    while queue:
        current = queue.pop(0)
        for consumer in sorted(consumers_by_producer.get(current, ())):
            fields.update(upper_dep_fields.get((current, consumer), ()))
            if consumer == boundary_event:
                continue
            if consumer not in seen:
                seen.add(consumer)
                queue.append(consumer)
    return tuple(sorted(fields))


def _upper_refs_from_lower_event(event: LayerEvent | None) -> tuple[str, ...]:
    if event is None:
        return ()
    refs: set[str] = set()
    for key in (
        "upper_event",
        "upper_event_id",
        "upper_producer_event",
        "source_event_id",
        "source_events",
    ):
        refs.update(_as_tuple(event.payload.get(key)))
    refs.discard("")
    return tuple(sorted(refs))


def _fields_from_hunk_text(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    fields: list[str] = []
    if "top_k" in lowered or "torch.max" in text or "switchtop1" in lowered:
        fields.extend(("routing.top_k", "routing.policy", "routing.dispatch_footprint"))
    if "tokens_rerouted" in lowered:
        fields.append("routing.tokens_rerouted")
    if "all_to_all" in lowered or "a2a" in lowered:
        fields.append("collective.all_to_all")
    if "scaled_dot_product_attention" in lowered:
        fields.append("attention.kernel")
    if "expert_parallel" in lowered or "num_local_experts" in lowered:
        fields.append("parallelism.ep_size")
    if "historical_reference" in lowered:
        fields.append("workload.historical_reference")
    if not fields:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        fields.append(f"source.change.{digest}")
    return tuple(dict.fromkeys(fields))
