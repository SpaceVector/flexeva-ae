"""Candidate-metadata helpers and runtime-value-point assessment.

This layer makes the anchor-first picture executable:

- one observed anchor defines the current semantic basis,
- generator-side sidecars request runtime-value points,
- points are classified as derivable, inferable, or unmodeled,
- and the runtime can decide whether local generation is plausible or escalation is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .semantic import SemanticVariableKind, semantic_changes_from_sidecars


class RuntimeValuePointKind(str, Enum):
    BRANCH = "branch"
    COUNT = "count"
    BUCKET = "bucket"
    STAGE = "stage"


class RuntimeValueStatus(str, Enum):
    DERIVABLE = "derivable"
    INFERABLE = "inferable"
    UNMODELED = "unmodeled"


class BasisAction(str, Enum):
    LOCAL_GENERATE = "local_generate"
    LOCAL_GENERATE_WITH_INFERENCE = "local_generate_with_inference"
    ESCALATE_ANCHOR_REFRESH = "escalate_anchor_refresh"


@dataclass(frozen=True)
class RuntimeValuePoint:
    name: str
    kind: RuntimeValuePointKind
    description: str
    required_fields: tuple[str, ...]
    inferable_fields: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkloadSemanticContract:
    workload_family: str
    stable_entities: tuple[str, ...]
    execution_summary_fields: tuple[str, ...]
    runtime_value_points: tuple[RuntimeValuePoint, ...]
    notes: tuple[str, ...] = ()

    def point_by_name(self) -> dict[str, RuntimeValuePoint]:
        return {point.name: point for point in self.runtime_value_points}


@dataclass(frozen=True)
class RuntimeValueResolution:
    point_name: str
    status: RuntimeValueStatus
    available_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]
    rationale: tuple[str, ...] = ()


@dataclass(frozen=True)
class MutationBasisAssessment:
    workload_family: str
    action: BasisAction
    changed_fields: tuple[str, ...]
    requested_points: tuple[str, ...]
    resolutions: tuple[RuntimeValueResolution, ...]
    notes: tuple[str, ...] = ()

    @property
    def derivable_points(self) -> tuple[str, ...]:
        return tuple(
            resolution.point_name
            for resolution in self.resolutions
            if resolution.status == RuntimeValueStatus.DERIVABLE
        )

    @property
    def inferable_points(self) -> tuple[str, ...]:
        return tuple(
            resolution.point_name
            for resolution in self.resolutions
            if resolution.status == RuntimeValueStatus.INFERABLE
        )

    @property
    def unmodeled_points(self) -> tuple[str, ...]:
        return tuple(
            resolution.point_name
            for resolution in self.resolutions
            if resolution.status == RuntimeValueStatus.UNMODELED
        )

    @property
    def in_basis(self) -> bool:
        return not self.unmodeled_points


_REQUIRED_POINTS_BY_VARIABLE_KIND: dict[SemanticVariableKind, tuple[str, ...]] = {
    SemanticVariableKind.EXPERT_LAYOUT: ("remote_dispatch", "expert_load"),
    SemanticVariableKind.EXPERT_PARALLELISM: ("remote_dispatch", "expert_load", "collective_stage"),
    SemanticVariableKind.ROUTING_TOPK: ("route_decision", "expert_load", "remote_dispatch"),
    SemanticVariableKind.DISPATCH_PRESSURE: ("route_decision", "expert_load", "remote_dispatch"),
    SemanticVariableKind.CAPACITY_POLICY: ("overflow_state", "expert_load"),
    SemanticVariableKind.RECOMPUTE: ("collective_stage",),
    SemanticVariableKind.MICROBATCHES: ("collective_stage",),
    SemanticVariableKind.OVERLAP_MODE: ("collective_stage",),
    SemanticVariableKind.SYNC_MODE: ("collective_stage",),
    SemanticVariableKind.BOTTLENECK_HINT: (),
    SemanticVariableKind.UNKNOWN: (),
}


def _flatten_mapping(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, item in value.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flat.update(_flatten_mapping(item, prefix=full_key))
        else:
            flat[full_key] = item
    return flat


def _merged_sidecar_fields(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    merged = _flatten_mapping(before)
    merged.update(_flatten_mapping(after))
    return merged


def routed_moe_workload_contract() -> WorkloadSemanticContract:
    return WorkloadSemanticContract(
        workload_family="routed_moe",
        stable_entities=("tokens", "experts", "ranks", "stages"),
        execution_summary_fields=(
            "parallelism.expert_layout",
            "parallelism.ep_size",
            "routing.top_k",
            "routing.capacity_factor",
            "routing.dispatch_footprint",
            "memory_compute.micro_batches",
            "memory_compute.recompute",
            "loop_semantics.overlap_policy",
            "loop_semantics.sync_policy",
            "intended_bottleneck",
        ),
        runtime_value_points=(
            RuntimeValuePoint(
                name="route_decision",
                kind=RuntimeValuePointKind.BRANCH,
                description="Token routing choice and fanout bucket.",
                required_fields=("routing.top_k", "routing.dispatch_footprint"),
            ),
            RuntimeValuePoint(
                name="expert_load",
                kind=RuntimeValuePointKind.COUNT,
                description="Per-expert or per-rank load summary needed for hot-expert and spill reasoning.",
                required_fields=("routing.dispatch_footprint", "routing.capacity_factor", "parallelism.ep_size"),
            ),
            RuntimeValuePoint(
                name="overflow_state",
                kind=RuntimeValuePointKind.COUNT,
                description="Overflow / drop / reroute summary bucket.",
                required_fields=("routing.dispatch_footprint", "routing.capacity_factor"),
            ),
            RuntimeValuePoint(
                name="remote_dispatch",
                kind=RuntimeValuePointKind.BUCKET,
                description="Local-vs-remote dispatch pressure summary.",
                required_fields=("parallelism.expert_layout", "parallelism.ep_size", "routing.dispatch_footprint"),
                inferable_fields=("routing.dispatch_footprint",),
                notes=(
                    "When dispatch_footprint is absent, a coarse locality bucket may still be inferred from placement and EP degree.",
                ),
            ),
            RuntimeValuePoint(
                name="collective_stage",
                kind=RuntimeValuePointKind.STAGE,
                description="Collective and overlap stage ordering / cadence summary.",
                required_fields=(
                    "loop_semantics.overlap_policy",
                    "loop_semantics.sync_policy",
                    "memory_compute.micro_batches",
                ),
                inferable_fields=("loop_semantics.sync_policy",),
                notes=("If sync policy is missing, overlap and micro-batch cadence still permit a coarse stage estimate.",),
            ),
        ),
        notes=("Family-level contract for routed-MoE code mutations under a fixed algorithm/model contract.",),
    )


def default_contract_for_workload_family(workload_family: str) -> WorkloadSemanticContract | None:
    lowered = workload_family.lower()
    if "moe" in lowered:
        return routed_moe_workload_contract()
    return None


def required_runtime_points_for_sidecar_mutation(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[str, ...]:
    requested: list[str] = []
    for change in semantic_changes_from_sidecars(before, after):
        for point_name in _REQUIRED_POINTS_BY_VARIABLE_KIND.get(change.variable.kind, ()):
            if point_name not in requested:
                requested.append(point_name)
    return tuple(requested)


def assess_runtime_value_points(
    contract: WorkloadSemanticContract,
    *,
    available_fields: Mapping[str, Any],
    requested_point_names: tuple[str, ...],
) -> tuple[RuntimeValueResolution, ...]:
    points = contract.point_by_name()
    available_field_names = set(available_fields)
    resolutions: list[RuntimeValueResolution] = []
    for point_name in requested_point_names:
        point = points.get(point_name)
        if point is None:
            resolutions.append(
                RuntimeValueResolution(
                    point_name=point_name,
                    status=RuntimeValueStatus.UNMODELED,
                    available_fields=(),
                    missing_fields=(),
                    rationale=("point not present in workload semantic contract",),
                )
            )
            continue
        missing = tuple(field for field in point.required_fields if field not in available_field_names)
        available = tuple(field for field in point.required_fields if field in available_field_names)
        if not missing:
            status = RuntimeValueStatus.DERIVABLE
            rationale = ("all required summary fields are available",)
        elif set(missing).issubset(set(point.inferable_fields)):
            status = RuntimeValueStatus.INFERABLE
            rationale = (
                "required fields are missing, but the gap stays within the point's inferable field set",
            )
        else:
            status = RuntimeValueStatus.UNMODELED
            rationale = ("required summary fields are missing outside the inferable field set",)
        resolutions.append(
            RuntimeValueResolution(
                point_name=point_name,
                status=status,
                available_fields=available,
                missing_fields=missing,
                rationale=rationale + point.notes,
            )
        )
    return tuple(resolutions)


def assess_sidecar_mutation_against_contract(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    contract: WorkloadSemanticContract,
    requested_point_names: tuple[str, ...] | None = None,
) -> MutationBasisAssessment:
    merged_fields = _merged_sidecar_fields(before, after)
    changed_fields = []
    before_fields = _flatten_mapping(before)
    after_fields = _flatten_mapping(after)
    for field_name in sorted(set(before_fields) | set(after_fields)):
        if before_fields.get(field_name) != after_fields.get(field_name):
            changed_fields.append(field_name)
    requested_points = (
        requested_point_names
        if requested_point_names is not None
        else required_runtime_points_for_sidecar_mutation(before, after)
    )
    resolutions = assess_runtime_value_points(
        contract,
        available_fields=merged_fields,
        requested_point_names=tuple(requested_points),
    )
    if any(resolution.status == RuntimeValueStatus.UNMODELED for resolution in resolutions):
        action = BasisAction.ESCALATE_ANCHOR_REFRESH
        notes = ("mutation requests runtime-value points outside the current semantic basis",)
    elif any(resolution.status == RuntimeValueStatus.INFERABLE for resolution in resolutions):
        action = BasisAction.LOCAL_GENERATE_WITH_INFERENCE
        notes = ("mutation stays in-basis but requires lightweight runtime-value inference",)
    else:
        action = BasisAction.LOCAL_GENERATE
        notes = ("all requested runtime-value points are derivable from tracked summary fields",)
    return MutationBasisAssessment(
        workload_family=contract.workload_family,
        action=action,
        changed_fields=tuple(changed_fields),
        requested_points=tuple(requested_points),
        resolutions=resolutions,
        notes=notes,
    )
