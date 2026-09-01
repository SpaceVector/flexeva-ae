"""Semantic variable modeling for anchor-reuse invalidation.

This layer sits between raw sidecar/dry-run signals and witness-region
invalidation. The immediate goal is to make decision-critical semantic state
first-class instead of flattening everything into anonymous delta kinds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from .schema import DeltaKind, RegionKind, RegionTrust


class SemanticVariableKind(str, Enum):
    EXPERT_LAYOUT = "expert_layout"
    EXPERT_PARALLELISM = "expert_parallelism"
    ROUTING_TOPK = "routing_topk"
    DISPATCH_PRESSURE = "dispatch_pressure"
    CAPACITY_POLICY = "capacity_policy"
    RECOMPUTE = "recompute"
    MICROBATCHES = "microbatches"
    OVERLAP_MODE = "overlap_mode"
    SYNC_MODE = "sync_mode"
    BOTTLENECK_HINT = "bottleneck_hint"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SemanticVariable:
    name: str
    kind: SemanticVariableKind
    value: Any
    source: str
    branch_ids: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticVariableChange:
    variable: SemanticVariable
    before: Any
    after: Any
    delta_kind: DeltaKind


@dataclass(frozen=True)
class RegionInfluence:
    variable_kind: SemanticVariableKind
    region_kind: RegionKind
    trust: RegionTrust
    rationale: str


SEMANTIC_INFLUENCE_RULES: dict[SemanticVariableKind, tuple[RegionInfluence, ...]] = {
    SemanticVariableKind.EXPERT_LAYOUT: (
        RegionInfluence(SemanticVariableKind.EXPERT_LAYOUT, RegionKind.EXPERT_COMPUTE, RegionTrust.INVALID, "expert placement changes expert compute ownership"),
        RegionInfluence(SemanticVariableKind.EXPERT_LAYOUT, RegionKind.COLLECTIVE, RegionTrust.INVALID, "expert placement changes collective dispatch pattern"),
        RegionInfluence(SemanticVariableKind.EXPERT_LAYOUT, RegionKind.DISPATCH, RegionTrust.INVALID, "expert placement changes token dispatch targets"),
        RegionInfluence(SemanticVariableKind.EXPERT_LAYOUT, RegionKind.OVERLAP, RegionTrust.UNCERTAIN, "layout changes can perturb overlap behavior"),
    ),
    SemanticVariableKind.EXPERT_PARALLELISM: (
        RegionInfluence(SemanticVariableKind.EXPERT_PARALLELISM, RegionKind.EXPERT_COMPUTE, RegionTrust.INVALID, "ep degree changes compute ownership"),
        RegionInfluence(SemanticVariableKind.EXPERT_PARALLELISM, RegionKind.COLLECTIVE, RegionTrust.INVALID, "ep degree changes dispatch collectives"),
        RegionInfluence(SemanticVariableKind.EXPERT_PARALLELISM, RegionKind.OVERLAP, RegionTrust.UNCERTAIN, "ep degree can perturb overlap"),
    ),
    SemanticVariableKind.ROUTING_TOPK: (
        RegionInfluence(SemanticVariableKind.ROUTING_TOPK, RegionKind.DISPATCH, RegionTrust.INVALID, "top-k changes dispatch fanout"),
        RegionInfluence(SemanticVariableKind.ROUTING_TOPK, RegionKind.COLLECTIVE, RegionTrust.INVALID, "top-k changes communication volume"),
        RegionInfluence(SemanticVariableKind.ROUTING_TOPK, RegionKind.EXPERT_COMPUTE, RegionTrust.UNCERTAIN, "top-k changes expert load distribution"),
    ),
    SemanticVariableKind.DISPATCH_PRESSURE: (
        RegionInfluence(SemanticVariableKind.DISPATCH_PRESSURE, RegionKind.DISPATCH, RegionTrust.INVALID, "dispatch pressure directly affects dispatch regions"),
        RegionInfluence(SemanticVariableKind.DISPATCH_PRESSURE, RegionKind.COLLECTIVE, RegionTrust.INVALID, "dispatch pressure affects all-to-all / collective traffic"),
        RegionInfluence(SemanticVariableKind.DISPATCH_PRESSURE, RegionKind.EXPERT_COMPUTE, RegionTrust.UNCERTAIN, "dispatch pressure shifts expert load"),
        RegionInfluence(SemanticVariableKind.DISPATCH_PRESSURE, RegionKind.OVERLAP, RegionTrust.UNCERTAIN, "dispatch pressure perturbs overlap"),
    ),
    SemanticVariableKind.CAPACITY_POLICY: (
        RegionInfluence(SemanticVariableKind.CAPACITY_POLICY, RegionKind.MEMORY, RegionTrust.INVALID, "capacity policy changes buffering and overflow behavior"),
        RegionInfluence(SemanticVariableKind.CAPACITY_POLICY, RegionKind.DISPATCH, RegionTrust.UNCERTAIN, "capacity changes can alter dispatch shape"),
    ),
    SemanticVariableKind.RECOMPUTE: (
        RegionInfluence(SemanticVariableKind.RECOMPUTE, RegionKind.EXPERT_COMPUTE, RegionTrust.INVALID, "recompute changes compute work"),
        RegionInfluence(SemanticVariableKind.RECOMPUTE, RegionKind.MEMORY, RegionTrust.INVALID, "recompute changes activation/memory traffic"),
        RegionInfluence(SemanticVariableKind.RECOMPUTE, RegionKind.COLLECTIVE, RegionTrust.UNCERTAIN, "recompute can shift synchronization timing"),
    ),
    SemanticVariableKind.MICROBATCHES: (
        RegionInfluence(SemanticVariableKind.MICROBATCHES, RegionKind.OVERLAP, RegionTrust.INVALID, "microbatch count changes overlap schedule"),
        RegionInfluence(SemanticVariableKind.MICROBATCHES, RegionKind.COLLECTIVE, RegionTrust.INVALID, "microbatch count changes sync cadence"),
        RegionInfluence(SemanticVariableKind.MICROBATCHES, RegionKind.EXPERT_COMPUTE, RegionTrust.UNCERTAIN, "microbatch count can shift compute grouping"),
    ),
    SemanticVariableKind.OVERLAP_MODE: (
        RegionInfluence(SemanticVariableKind.OVERLAP_MODE, RegionKind.OVERLAP, RegionTrust.INVALID, "overlap mode directly changes overlap regions"),
        RegionInfluence(SemanticVariableKind.OVERLAP_MODE, RegionKind.COLLECTIVE, RegionTrust.UNCERTAIN, "overlap mode perturbs collective timing"),
    ),
    SemanticVariableKind.SYNC_MODE: (
        RegionInfluence(SemanticVariableKind.SYNC_MODE, RegionKind.COLLECTIVE, RegionTrust.INVALID, "sync mode directly changes collective sequencing"),
        RegionInfluence(SemanticVariableKind.SYNC_MODE, RegionKind.OVERLAP, RegionTrust.INVALID, "sync mode directly changes overlap coordination"),
        RegionInfluence(SemanticVariableKind.SYNC_MODE, RegionKind.EXPERT_COMPUTE, RegionTrust.UNCERTAIN, "sync mode can perturb compute timing"),
    ),
    SemanticVariableKind.BOTTLENECK_HINT: (
        RegionInfluence(SemanticVariableKind.BOTTLENECK_HINT, RegionKind.COLLECTIVE, RegionTrust.UNCERTAIN, "bottleneck hint shifts trust toward communication-heavy regions"),
        RegionInfluence(SemanticVariableKind.BOTTLENECK_HINT, RegionKind.EXPERT_COMPUTE, RegionTrust.UNCERTAIN, "bottleneck hint shifts trust toward compute-heavy regions"),
    ),
}


_SEMANTIC_NAME_TO_SOURCE_FIELDS: dict[str, tuple[str, ...]] = {
    "expert_layout": ("parallelism.expert_layout",),
    "ep_size": ("parallelism.ep_size",),
    "top_k": ("routing.top_k",),
    "dispatch_footprint": ("routing.dispatch_footprint",),
    "capacity_factor": ("routing.capacity_factor",),
    "recompute": ("memory_compute.recompute",),
    "micro_batches": ("memory_compute.micro_batches",),
    "overlap_policy": ("loop_semantics.overlap_policy",),
    "sync_policy": ("loop_semantics.sync_policy",),
    "intended_bottleneck": ("intended_bottleneck",),
}


def _variable(
    *,
    name: str,
    kind: SemanticVariableKind,
    value: Any,
    source: str,
    branch_ids: tuple[int, ...] = (),
) -> SemanticVariable:
    return SemanticVariable(name=name, kind=kind, value=value, source=source, branch_ids=branch_ids)


def source_fields_for_semantic_names(names: Iterable[str]) -> tuple[str, ...]:
    fields = {
        field
        for name in names
        for field in _SEMANTIC_NAME_TO_SOURCE_FIELDS.get(str(name), ())
    }
    return tuple(sorted(fields))


def semantic_variables_from_sidecar(sidecar: Mapping[str, Any]) -> tuple[SemanticVariable, ...]:
    parallel = sidecar.get("parallelism", {})
    routing = sidecar.get("routing", {})
    memory = sidecar.get("memory_compute", {})
    loop = sidecar.get("loop_semantics", {})

    variables = (
        _variable(name="expert_layout", kind=SemanticVariableKind.EXPERT_LAYOUT, value=parallel.get("expert_layout"), source="parallelism.expert_layout"),
        _variable(name="ep_size", kind=SemanticVariableKind.EXPERT_PARALLELISM, value=parallel.get("ep_size"), source="parallelism.ep_size"),
        _variable(name="top_k", kind=SemanticVariableKind.ROUTING_TOPK, value=routing.get("top_k"), source="routing.top_k"),
        _variable(name="dispatch_footprint", kind=SemanticVariableKind.DISPATCH_PRESSURE, value=routing.get("dispatch_footprint"), source="routing.dispatch_footprint"),
        _variable(name="capacity_factor", kind=SemanticVariableKind.CAPACITY_POLICY, value=routing.get("capacity_factor"), source="routing.capacity_factor"),
        _variable(name="recompute", kind=SemanticVariableKind.RECOMPUTE, value=memory.get("recompute"), source="memory_compute.recompute"),
        _variable(name="micro_batches", kind=SemanticVariableKind.MICROBATCHES, value=memory.get("micro_batches"), source="memory_compute.micro_batches"),
        _variable(name="overlap_policy", kind=SemanticVariableKind.OVERLAP_MODE, value=loop.get("overlap_policy"), source="loop_semantics.overlap_policy"),
        _variable(name="sync_policy", kind=SemanticVariableKind.SYNC_MODE, value=loop.get("sync_policy"), source="loop_semantics.sync_policy"),
        _variable(name="intended_bottleneck", kind=SemanticVariableKind.BOTTLENECK_HINT, value=sidecar.get("intended_bottleneck"), source="intended_bottleneck"),
    )
    return tuple(item for item in variables if item.value is not None)


def semantic_variables_from_dryrun_summary(
    summary: Mapping[str, Mapping[str, Any]],
    *,
    kind_hints: Mapping[str, SemanticVariableKind] | None = None,
) -> tuple[SemanticVariable, ...]:
    hints = dict(kind_hints or {})
    variables = []
    for name, payload in summary.items():
        variables.append(
            SemanticVariable(
                name=name,
                kind=hints.get(name, SemanticVariableKind.UNKNOWN),
                value=payload.get("value"),
                source="dryrun",
                branch_ids=tuple(int(value) for value in payload.get("branch_ids", [])),
                notes=(f"repr={payload.get('repr')}",),
            )
        )
    return tuple(variables)


def _delta_kind_for_variable(kind: SemanticVariableKind) -> DeltaKind:
    mapping = {
        SemanticVariableKind.EXPERT_LAYOUT: DeltaKind.EXPERT_LAYOUT,
        SemanticVariableKind.EXPERT_PARALLELISM: DeltaKind.EXPERT_LAYOUT,
        SemanticVariableKind.ROUTING_TOPK: DeltaKind.ROUTING_PRESSURE,
        SemanticVariableKind.DISPATCH_PRESSURE: DeltaKind.ROUTING_PRESSURE,
        SemanticVariableKind.CAPACITY_POLICY: DeltaKind.MEMORY_POLICY,
        SemanticVariableKind.RECOMPUTE: DeltaKind.RECOMPUTE_POLICY,
        SemanticVariableKind.MICROBATCHES: DeltaKind.MICROBATCH_POLICY,
        SemanticVariableKind.OVERLAP_MODE: DeltaKind.OVERLAP_POLICY,
        SemanticVariableKind.SYNC_MODE: DeltaKind.SYNC_POLICY,
        SemanticVariableKind.BOTTLENECK_HINT: DeltaKind.UNKNOWN,
    }
    return mapping.get(kind, DeltaKind.UNKNOWN)


def semantic_changes_from_sidecars(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[SemanticVariableChange, ...]:
    before_map = {item.name: item for item in semantic_variables_from_sidecar(before)}
    after_map = {item.name: item for item in semantic_variables_from_sidecar(after)}
    changes = []
    for name in sorted(set(before_map) | set(after_map)):
        before_var = before_map.get(name)
        after_var = after_map.get(name)
        before_value = before_var.value if before_var is not None else None
        after_value = after_var.value if after_var is not None else None
        if before_value == after_value:
            continue
        variable = after_var if after_var is not None else before_var
        assert variable is not None
        changes.append(
            SemanticVariableChange(
                variable=variable,
                before=before_value,
                after=after_value,
                delta_kind=_delta_kind_for_variable(variable.kind),
            )
        )
    return tuple(changes)


def influences_for_semantic_change(
    change: SemanticVariableChange,
) -> tuple[RegionInfluence, ...]:
    return SEMANTIC_INFLUENCE_RULES.get(change.variable.kind, ())
