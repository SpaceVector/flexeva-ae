"""Program-logic dependency extraction for anchor-region evaluation.

This layer captures candidate changes that alter *what work exists* before the
operator-DAG layer explains *how that work executes*.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .schema import DeltaKind, RegionKind, RegionTrust
from .semantic import source_fields_for_semantic_names


class ProgramLogicKind(str, Enum):
    RANK_PARTITION = "rank_partition"
    REGION_ACTIVATION = "region_activation"
    STAGE_ORDER = "stage_order"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProgramLogicPoint:
    name: str
    kind: ProgramLogicKind
    value: Any
    source: str
    source_path: str | None = None
    lineno: int | None = None
    end_lineno: int | None = None
    branch_ids: tuple[int, ...] = ()
    rank_groups: tuple[tuple[int, ...], ...] = ()
    source_fields: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProgramLogicDelta:
    point: ProgramLogicPoint
    before: Any
    after: Any
    delta_kind: DeltaKind


@dataclass(frozen=True)
class ProgramLogicInfluence:
    logic_kind: ProgramLogicKind
    region_kind: RegionKind
    trust: RegionTrust
    rationale: str


@dataclass(frozen=True)
class ProgramLogicCarrier:
    source: str
    points: tuple[ProgramLogicPoint, ...]
    notes: tuple[str, ...] = ()

    def point_map(self) -> dict[str, ProgramLogicPoint]:
        return {point.name: point for point in self.points}

    def signature(self) -> tuple[str, ...]:
        rows: list[str] = [f"source={self.source}"]
        for point in self.points:
            rows.append(
                "|".join(
                    (
                        point.name,
                        point.kind.value,
                        repr(point.value),
                        ",".join(str(item) for item in point.branch_ids),
                        ";".join(",".join(str(rank) for rank in group) for group in point.rank_groups),
                    )
                )
            )
        return tuple(rows)


PROGRAM_LOGIC_INFLUENCE_RULES: dict[ProgramLogicKind, tuple[ProgramLogicInfluence, ...]] = {
    ProgramLogicKind.RANK_PARTITION: (
        ProgramLogicInfluence(
            ProgramLogicKind.RANK_PARTITION,
            RegionKind.DISPATCH,
            RegionTrust.INVALID,
            "rank-partition changes directly alter dispatch structure",
        ),
        ProgramLogicInfluence(
            ProgramLogicKind.RANK_PARTITION,
            RegionKind.COLLECTIVE,
            RegionTrust.INVALID,
            "rank-partition changes alter communication participants",
        ),
        ProgramLogicInfluence(
            ProgramLogicKind.RANK_PARTITION,
            RegionKind.EXPERT_COMPUTE,
            RegionTrust.UNCERTAIN,
            "rank-partition changes can perturb compute placement",
        ),
    ),
    ProgramLogicKind.REGION_ACTIVATION: (
        ProgramLogicInfluence(
            ProgramLogicKind.REGION_ACTIVATION,
            RegionKind.EXPERT_COMPUTE,
            RegionTrust.INVALID,
            "activation logic changes the compute regions that execute",
        ),
        ProgramLogicInfluence(
            ProgramLogicKind.REGION_ACTIVATION,
            RegionKind.MEMORY,
            RegionTrust.INVALID,
            "activation logic changes activation and buffer behavior",
        ),
        ProgramLogicInfluence(
            ProgramLogicKind.REGION_ACTIVATION,
            RegionKind.COLLECTIVE,
            RegionTrust.UNCERTAIN,
            "activation logic can perturb collective cadence",
        ),
    ),
    ProgramLogicKind.STAGE_ORDER: (
        ProgramLogicInfluence(
            ProgramLogicKind.STAGE_ORDER,
            RegionKind.OVERLAP,
            RegionTrust.INVALID,
            "stage-order logic directly changes overlap schedule",
        ),
        ProgramLogicInfluence(
            ProgramLogicKind.STAGE_ORDER,
            RegionKind.COLLECTIVE,
            RegionTrust.INVALID,
            "stage-order logic changes synchronization ordering",
        ),
        ProgramLogicInfluence(
            ProgramLogicKind.STAGE_ORDER,
            RegionKind.MEMORY,
            RegionTrust.UNCERTAIN,
            "stage-order logic can shift memory exposure timing",
        ),
    ),
}


def _logic_point(
    *,
    name: str,
    kind: ProgramLogicKind,
    value: Any,
    source: str,
    source_path: str | None = None,
    lineno: int | None = None,
    end_lineno: int | None = None,
    branch_ids: tuple[int, ...] = (),
    rank_groups: tuple[tuple[int, ...], ...] = (),
    source_fields: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
) -> ProgramLogicPoint:
    return ProgramLogicPoint(
        name=name,
        kind=kind,
        value=value,
        source=source,
        source_path=source_path,
        lineno=lineno,
        end_lineno=end_lineno,
        branch_ids=branch_ids,
        rank_groups=rank_groups,
        source_fields=source_fields,
        notes=notes,
    )


def _sidecar_points(sidecar: Mapping[str, Any]) -> tuple[ProgramLogicPoint, ...]:
    parallel = sidecar.get("parallelism", {})
    routing = sidecar.get("routing", {})
    memory = sidecar.get("memory_compute", {})
    loop = sidecar.get("loop_semantics", {})

    points = (
        _logic_point(
            name="routing_partition",
            kind=ProgramLogicKind.RANK_PARTITION,
            value=(
                parallel.get("expert_layout"),
                parallel.get("ep_size"),
                routing.get("top_k"),
                routing.get("dispatch_footprint"),
            ),
            source="sidecar.routing_partition",
            source_fields=(
                "parallelism.expert_layout",
                "parallelism.ep_size",
                "routing.top_k",
                "routing.dispatch_footprint",
            ),
            notes=("derived from parallelism/routing sidecar fields",),
        ),
        _logic_point(
            name="activation_policy",
            kind=ProgramLogicKind.REGION_ACTIVATION,
            value=(
                memory.get("recompute"),
                routing.get("capacity_factor"),
            ),
            source="sidecar.activation_policy",
            source_fields=(
                "memory_compute.recompute",
                "routing.capacity_factor",
            ),
            notes=("derived from recompute/capacity sidecar fields",),
        ),
        _logic_point(
            name="execution_order",
            kind=ProgramLogicKind.STAGE_ORDER,
            value=(
                memory.get("micro_batches"),
                loop.get("overlap_policy"),
                loop.get("sync_policy"),
            ),
            source="sidecar.execution_order",
            source_fields=(
                "memory_compute.micro_batches",
                "loop_semantics.overlap_policy",
                "loop_semantics.sync_policy",
            ),
            notes=("derived from microbatch/overlap/sync sidecar fields",),
        ),
    )
    return tuple(point for point in points if any(item is not None for item in point.value))


def _delta_kind_for_logic_kind(kind: ProgramLogicKind) -> DeltaKind:
    mapping = {
        ProgramLogicKind.RANK_PARTITION: DeltaKind.ROUTING_PRESSURE,
        ProgramLogicKind.REGION_ACTIVATION: DeltaKind.RECOMPUTE_POLICY,
        ProgramLogicKind.STAGE_ORDER: DeltaKind.OVERLAP_POLICY,
    }
    return mapping.get(kind, DeltaKind.UNKNOWN)


def program_logic_points_from_sidecar(sidecar: Mapping[str, Any]) -> tuple[ProgramLogicPoint, ...]:
    return _sidecar_points(sidecar)


def build_program_logic_carrier_from_sidecar(
    sidecar: Mapping[str, Any],
) -> ProgramLogicCarrier:
    return ProgramLogicCarrier(
        source="sidecar_helper",
        points=program_logic_points_from_sidecar(sidecar),
        notes=("helper carrier built from sidecar",),
    )


def program_logic_deltas_from_sidecars(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[ProgramLogicDelta, ...]:
    before_map = {point.name: point for point in program_logic_points_from_sidecar(before)}
    after_map = {point.name: point for point in program_logic_points_from_sidecar(after)}
    deltas: list[ProgramLogicDelta] = []
    for name in sorted(set(before_map) | set(after_map)):
        before_point = before_map.get(name)
        after_point = after_map.get(name)
        before_value = before_point.value if before_point is not None else None
        after_value = after_point.value if after_point is not None else None
        if before_value == after_value:
            continue
        point = after_point if after_point is not None else before_point
        assert point is not None
        deltas.append(
            ProgramLogicDelta(
                point=point,
                before=before_value,
                after=after_value,
                delta_kind=_delta_kind_for_logic_kind(point.kind),
            )
        )
    return tuple(deltas)


def program_logic_deltas_from_carriers(
    before: ProgramLogicCarrier | None,
    after: ProgramLogicCarrier | None,
) -> tuple[ProgramLogicDelta, ...]:
    before_map = before.point_map() if before is not None else {}
    after_map = after.point_map() if after is not None else {}
    deltas: list[ProgramLogicDelta] = []
    for name in sorted(set(before_map) | set(after_map)):
        before_point = before_map.get(name)
        after_point = after_map.get(name)
        before_value = before_point.value if before_point is not None else None
        after_value = after_point.value if after_point is not None else None
        if before_value == after_value:
            continue
        point = after_point if after_point is not None else before_point
        assert point is not None
        deltas.append(
            ProgramLogicDelta(
                point=point,
                before=before_value,
                after=after_value,
                delta_kind=_delta_kind_for_logic_kind(point.kind),
            )
        )
    return tuple(deltas)


def program_logic_points_from_dryrun(
    branch_signatures: Mapping[int, list[tuple[int, bool, bool]]],
    semantic_summaries: Mapping[int, Mapping[str, Mapping[str, Any]]] | None = None,
    branch_metadata: Mapping[int, tuple[str, int, int | None]] | None = None,
    branch_var_map: Mapping[int, list[str] | tuple[str, ...]] | None = None,
) -> tuple[ProgramLogicPoint, ...]:
    """Extract first-pass logical dependency points from dry-run output.

    Current focus:
    - rank partitions induced by value-tainted branches
    - optional variable provenance summaries attached as notes
    """

    semantic_summaries = semantic_summaries or {}
    branch_metadata = branch_metadata or {}
    branch_var_map = branch_var_map or {}
    semantic_dependencies: dict[str, set[str]] = {}
    branch_rows: dict[int, dict[bool, list[int]]] = {}
    tainted_branches: set[int] = set()
    for summary in semantic_summaries.values():
        for name, payload in summary.items():
            dependencies = semantic_dependencies.setdefault(str(name), set())
            dependencies.update(str(item) for item in payload.get("dependencies", []))
    for rank, signature in branch_signatures.items():
        for branch_id, outcome, is_rtainted in signature:
            branch_rows.setdefault(int(branch_id), {True: [], False: []})[bool(outcome)].append(int(rank))
            if is_rtainted:
                tainted_branches.add(int(branch_id))

    points: list[ProgramLogicPoint] = []
    for branch_id in sorted(tainted_branches):
        rows = branch_rows.get(branch_id, {True: [], False: []})
        true_ranks = tuple(sorted(rows.get(True, ())))
        false_ranks = tuple(sorted(rows.get(False, ())))
        related_vars = sorted(
            name
            for summary in semantic_summaries.values()
            for name, payload in summary.items()
            if branch_id in {int(item) for item in payload.get("branch_ids", [])}
        )
        branch_vars = tuple(
            dict.fromkeys(
                str(name)
                for name in branch_var_map.get(branch_id, ())
                if str(name)
            )
        )
        related_semantic_names = set(related_vars) | set(branch_vars)
        pending_names = list(related_semantic_names)
        while pending_names:
            dependency_name = pending_names.pop()
            for next_name in sorted(semantic_dependencies.get(dependency_name, ())):
                if next_name in related_semantic_names:
                    continue
                related_semantic_names.add(next_name)
                pending_names.append(next_name)
        source_fields = source_fields_for_semantic_names(related_semantic_names)
        notes = ("derived from dry-run branch signature",)
        if not true_ranks or not false_ranks:
            uniform_outcome = "true" if true_ranks else "false"
            notes = notes + (f"uniform_outcome={uniform_outcome}",)
        if related_vars:
            notes = notes + (f"related_vars={','.join(related_vars)}",)
        if branch_vars:
            notes = notes + (f"branch_vars={','.join(branch_vars)}",)
        if source_fields:
            notes = notes + (f"source_fields={','.join(source_fields)}",)
        points.append(
            ProgramLogicPoint(
                name=f"branch_{branch_id}",
                kind=ProgramLogicKind.RANK_PARTITION,
                value=(true_ranks, false_ranks),
                source="dryrun.branch",
                source_path=branch_metadata.get(branch_id, (None, None, None))[0],
                lineno=branch_metadata.get(branch_id, (None, None, None))[1],
                end_lineno=branch_metadata.get(branch_id, (None, None, None))[2],
                branch_ids=(branch_id,),
                rank_groups=(true_ranks, false_ranks),
                source_fields=source_fields,
                notes=notes,
            )
        )
    return tuple(points)


def build_program_logic_carrier_from_dryrun(
    branch_signatures: Mapping[int, list[tuple[int, bool, bool]]],
    semantic_summaries: Mapping[int, Mapping[str, Mapping[str, Any]]] | None = None,
    branch_metadata: Mapping[int, tuple[str, int, int | None]] | None = None,
    branch_var_map: Mapping[int, list[str] | tuple[str, ...]] | None = None,
) -> ProgramLogicCarrier:
    return ProgramLogicCarrier(
        source="dryrun",
        points=program_logic_points_from_dryrun(
            branch_signatures,
            semantic_summaries,
            branch_metadata,
            branch_var_map,
        ),
        notes=("carrier built from dry-run branch signatures",),
    )


def build_program_logic_carrier_from_executor(
    executor: Any,
    branch_metadata: Mapping[int, tuple[str, int, int | None]] | None = None,
) -> ProgramLogicCarrier:
    branch_signatures = executor.get_all_branch_signatures()
    semantic_summaries_getter = getattr(executor, "get_all_semantic_summaries", None)
    semantic_summaries = semantic_summaries_getter() if callable(semantic_summaries_getter) else None
    branch_var_map = getattr(executor, "branch_var_map", None)
    return build_program_logic_carrier_from_dryrun(
        branch_signatures,
        semantic_summaries,
        branch_metadata,
        branch_var_map,
    )


def influences_for_program_logic_delta(
    delta: ProgramLogicDelta,
) -> tuple[ProgramLogicInfluence, ...]:
    return PROGRAM_LOGIC_INFLUENCE_RULES.get(delta.point.kind, ())
