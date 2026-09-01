"""First-pass object model for anchor-based sibling evaluation reuse.

The goal is to make the paper's current center concrete:

- anchor regions as the reusable unit
- witness as per-region reuse justification
- narrow vs wide invalidation
- obligations as unresolved uncertainty
- bounded regeneration / rerun decisions

Naming decision:

- ``AnchorRegion`` is the reusable unit.
- ``Witness`` is attached to one region.
- ``AnchorWitness`` is only the current container for anchor regions; it is a
  transitional implementation name, not the final paper object name.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class WitnessGranularity(str, Enum):
    PHASE_REGION = "phase_region"
    COLLECTIVE_REGION = "collective_region"
    STREAM_REGION = "stream_region"


class RegionKind(str, Enum):
    DISPATCH = "dispatch"
    EXPERT_COMPUTE = "expert_compute"
    COLLECTIVE = "collective"
    OVERLAP = "overlap"
    MEMORY = "memory"
    OTHER = "other"


class DeltaKind(str, Enum):
    EXPERT_LAYOUT = "expert_layout"
    ROUTING_PRESSURE = "routing_pressure"
    OVERLAP_POLICY = "overlap_policy"
    RECOMPUTE_POLICY = "recompute_policy"
    MICROBATCH_POLICY = "microbatch_policy"
    MEMORY_POLICY = "memory_policy"
    SYNC_POLICY = "sync_policy"
    UNKNOWN = "unknown"


class RegionTrust(str, Enum):
    TRUSTED = "trusted"
    UNCERTAIN = "uncertain"
    INVALID = "invalid"


class RepairActionKind(str, Enum):
    REUSE = "reuse"
    PATCH_TIMING = "patch_timing"
    REFRESH_REGION = "refresh_region"
    PARTIAL_RERUN = "partial_rerun"
    FULL_RERUN = "full_rerun"
    ABSTAIN = "abstain"


class DependencyType(str, Enum):
    NARROW = "narrow"
    WIDE = "wide"


class WitnessSourceKind(str, Enum):
    TRACE_WINDOW_AGGREGATE = "trace_window_aggregate"
    SEMANTIC_DRYRUN = "semantic_dryrun"
    GROUNDED_SUBSET = "grounded_subset"
    INFERRED = "inferred"


class ObligationKind(str, Enum):
    RUNTIME_VALUE = "runtime_value"
    HARDWARE_BEHAVIOR = "hardware_behavior"
    CLUSTER_REALIZATION = "cluster_realization"
    STRUCTURAL_REPLAY = "structural_replay"


class ObligationStatus(str, Enum):
    OPEN = "open"
    DISCHARGED = "discharged"
    DEFERRED = "deferred"


class ObligationSourceKind(str, Enum):
    EMITTED_STUB = "emitted_stub"
    BOUNDARY_CAPSULE = "boundary_capsule"
    RUNTIME_VALUE_POINT = "runtime_value_point"
    INVALIDATION_REGION = "invalidation_region"


class ObligationActionHint(str, Enum):
    REFRESH_OPERATOR_MAPPING = "refresh_operator_mapping"
    REFRESH_OPERATOR_PROJECTION = "refresh_operator_projection"
    REFRESH_BLACK_BOX_BOUNDARY = "refresh_black_box_boundary"
    PARTIAL_REPLAY_OR_REFRESH = "partial_replay_or_refresh"
    SEMANTIC_OR_VALUE_REGENERATION = "semantic_or_value_regeneration"
    GROUND_OR_RECALIBRATE = "ground_or_recalibrate"
    LOCAL_INFERENCE = "local_inference"
    ANCHOR_REFRESH = "anchor_refresh"


class ObligationEvidenceKind(str, Enum):
    SUMMARY = "summary"
    NOTE = "note"
    CALLEE = "callee"
    CANDIDATE_REGIONS = "candidate_regions"
    INVALIDATED_FIELDS = "invalidated_fields"
    CONTRACT_FIELDS = "contract_fields"
    PRESERVATION_REQUIREMENTS = "preservation_requirements"


@dataclass(frozen=True)
class ObligationEvidenceItem:
    kind: ObligationEvidenceKind
    value: str

    def as_text(self) -> str:
        if self.kind in {ObligationEvidenceKind.SUMMARY, ObligationEvidenceKind.NOTE}:
            return self.value
        return f"{self.kind.value}={self.value}"


@dataclass(frozen=True)
class Witness:
    source_kind: WitnessSourceKind
    confidence: float = 1.0
    evidence: tuple[str, ...] = ()
    rationale: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnchorRegion:
    region_id: str
    region_kind: RegionKind
    order_index: int
    timing_share: float
    start_window: int | None = None
    end_window: int | None = None
    collective_group: str | None = None
    stream_label: str | None = None
    notes: tuple[str, ...] = ()
    witness: Witness | None = None
    provenance: tuple[str, ...] = ()
    value_sensitive: bool = False
    hardware_sensitive: bool = False
    criticality_slack: float | None = None
    dependency_type: DependencyType = DependencyType.NARROW

    @property
    def is_critical(self) -> bool:
        return self.criticality_slack is not None and self.criticality_slack <= 0.05


# Backward-compatible name used by the existing code/tests.
WitnessRegion = AnchorRegion


@dataclass(frozen=True)
class AnchorDependencyEdge:
    src_region_id: str
    dst_region_id: str
    dependency_type: DependencyType = DependencyType.NARROW
    rationale: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnchorRegionGraph:
    anchor_candidate_id: str
    workload_family: str
    world_size: int
    regions: tuple[AnchorRegion, ...]
    edges: tuple[AnchorDependencyEdge, ...]
    notes: tuple[str, ...] = ()

    def region_ids(self) -> tuple[str, ...]:
        return tuple(region.region_id for region in self.regions)

    def region_by_id(self) -> dict[str, AnchorRegion]:
        return {region.region_id: region for region in self.regions}

    def outgoing_edges(self, region_id: str) -> tuple[AnchorDependencyEdge, ...]:
        return tuple(edge for edge in self.edges if edge.src_region_id == region_id)

    def incoming_edges(self, region_id: str) -> tuple[AnchorDependencyEdge, ...]:
        return tuple(edge for edge in self.edges if edge.dst_region_id == region_id)

    def adjacent_region_ids(self, region_id: str) -> tuple[str, ...]:
        adjacent: list[str] = []
        seen: set[str] = set()
        for edge in self.outgoing_edges(region_id):
            if edge.dst_region_id not in seen:
                seen.add(edge.dst_region_id)
                adjacent.append(edge.dst_region_id)
        for edge in self.incoming_edges(region_id):
            if edge.src_region_id not in seen:
                seen.add(edge.src_region_id)
                adjacent.append(edge.src_region_id)
        return tuple(adjacent)


@dataclass(frozen=True)
class AnchorWitness:
    anchor_candidate_id: str
    workload_family: str
    world_size: int
    granularity: WitnessGranularity
    regions: tuple[AnchorRegion, ...]
    artifacts: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def anchor_regions(self) -> tuple[AnchorRegion, ...]:
        return self.regions

    def region_ids(self) -> tuple[str, ...]:
        return tuple(region.region_id for region in self.regions)

    def total_timing_share(self) -> float:
        return sum(region.timing_share for region in self.regions)

    def region_by_id(self) -> dict[str, AnchorRegion]:
        return {region.region_id: region for region in self.regions}

    def critical_region_ids(self, *, slack_threshold: float = 0.10) -> tuple[str, ...]:
        return tuple(
            region.region_id
            for region in self.regions
            if region.criticality_slack is not None and region.criticality_slack <= slack_threshold
        )


@dataclass(frozen=True)
class CandidateDelta:
    kind: DeltaKind
    before: Any
    after: Any
    source: str
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Obligation:
    obligation_id: str
    kind: ObligationKind
    status: ObligationStatus = ObligationStatus.OPEN
    region_id: str | None = None
    related_region_ids: tuple[str, ...] = ()
    source_kind: ObligationSourceKind | None = None
    source_id: str | None = None
    suggested_action: ObligationActionHint | None = None
    evidence: tuple[ObligationEvidenceItem, ...] = ()
    rationale: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        related: list[str] = []
        seen: set[str] = set()
        if self.region_id is not None:
            normalized = str(self.region_id)
            if normalized:
                seen.add(normalized)
                related.append(normalized)
        for item in self.related_region_ids:
            normalized = str(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            related.append(normalized)
        object.__setattr__(self, "related_region_ids", tuple(related))
        if self.source_id is not None:
            object.__setattr__(self, "source_id", str(self.source_id))
        if self.suggested_action is not None and not isinstance(
            self.suggested_action,
            ObligationActionHint,
        ):
            object.__setattr__(
                self,
                "suggested_action",
                ObligationActionHint(str(self.suggested_action)),
            )
        normalized_evidence: list[ObligationEvidenceItem] = []
        for item in self.evidence:
            if isinstance(item, ObligationEvidenceItem):
                normalized_evidence.append(item)
                continue
            kind, value = item  # type: ignore[misc]
            normalized_evidence.append(
                ObligationEvidenceItem(
                    kind=kind if isinstance(kind, ObligationEvidenceKind) else ObligationEvidenceKind(str(kind)),
                    value=str(value),
                )
            )
        if not normalized_evidence and self.rationale:
            normalized_evidence = [
                ObligationEvidenceItem(ObligationEvidenceKind.NOTE, str(item))
                for item in self.rationale
            ]
        object.__setattr__(self, "evidence", tuple(normalized_evidence))
        object.__setattr__(
            self,
            "rationale",
            tuple(item.as_text() for item in normalized_evidence),
        )

    def has_source(self) -> bool:
        return self.source_kind is not None and self.source_id is not None

    def source_descriptor(self) -> str | None:
        if self.source_kind is None or self.source_id is None:
            return None
        return f"{self.source_kind.value}:{self.source_id}"

    def matches_source(
        self,
        source_kind: ObligationSourceKind,
        source_id: str | None = None,
    ) -> bool:
        if self.source_kind != source_kind:
            return False
        if source_id is None:
            return self.source_id is not None
        return self.source_id == source_id

    def suggested_action_name(self) -> str | None:
        if self.suggested_action is None:
            return None
        return self.suggested_action.value

    def evidence_text(self) -> tuple[str, ...]:
        return tuple(item.as_text() for item in self.evidence)


@dataclass(frozen=True)
class AffectedRegion:
    region_id: str
    trust: RegionTrust
    triggering_deltas: tuple[DeltaKind, ...]
    estimated_share_impact: float | None = None
    rationale: tuple[str, ...] = ()
    dependency_type: DependencyType = DependencyType.NARROW
    invalidated_witness_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class InvalidationReport:
    anchor_candidate_id: str
    preserved_region_ids: tuple[str, ...]
    affected_regions: tuple[AffectedRegion, ...]
    unknown_region_ids: tuple[str, ...] = ()
    witness_backed_region_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    dependency_type: DependencyType = DependencyType.NARROW
    obligations: tuple[Obligation, ...] = ()

    def affected_fraction(self, witness: AnchorWitness) -> float:
        impacted = {region.region_id for region in witness.regions if region.region_id in self.unknown_region_ids}
        impacted.update(region.region_id for region in self.affected_regions)
        total = witness.total_timing_share()
        if total <= 0:
            return 0.0
        impacted_share = sum(
            region.timing_share for region in witness.regions if region.region_id in impacted
        )
        return impacted_share / total

    def preserved_fraction(self, witness: AnchorWitness) -> float:
        total = witness.total_timing_share()
        if total <= 0:
            return 0.0
        preserved_share = sum(
            region.timing_share
            for region in witness.regions
            if region.region_id in self.preserved_region_ids
        )
        return preserved_share / total

    def has_unknowns(self) -> bool:
        return bool(self.unknown_region_ids)

    def open_obligations(self) -> tuple[Obligation, ...]:
        return tuple(item for item in self.obligations if item.status == ObligationStatus.OPEN)


@dataclass(frozen=True)
class RepairDecision:
    action: RepairActionKind
    rationale: tuple[str, ...]
    target_region_ids: tuple[str, ...] = ()
    estimated_cost_tier: str = "unknown"
    dependency_type: DependencyType | None = None


def _delta_if_changed(
    *,
    kind: DeltaKind,
    source: str,
    before: Any,
    after: Any,
) -> CandidateDelta | None:
    if before == after:
        return None
    return CandidateDelta(kind=kind, before=before, after=after, source=source)


def candidate_deltas_from_sidecars(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[CandidateDelta, ...]:
    """Extract first-pass typed deltas from semantic sidecars."""

    before_parallel = before.get("parallelism", {})
    after_parallel = after.get("parallelism", {})
    before_routing = before.get("routing", {})
    after_routing = after.get("routing", {})
    before_memory = before.get("memory_compute", {})
    after_memory = after.get("memory_compute", {})
    before_loop = before.get("loop_semantics", {})
    after_loop = after.get("loop_semantics", {})

    deltas = [
        _delta_if_changed(
            kind=DeltaKind.EXPERT_LAYOUT,
            source="parallelism.expert_layout",
            before=before_parallel.get("expert_layout"),
            after=after_parallel.get("expert_layout"),
        ),
        _delta_if_changed(
            kind=DeltaKind.ROUTING_PRESSURE,
            source="routing.dispatch_footprint",
            before=before_routing.get("dispatch_footprint"),
            after=after_routing.get("dispatch_footprint"),
        ),
        _delta_if_changed(
            kind=DeltaKind.ROUTING_PRESSURE,
            source="routing.top_k",
            before=before_routing.get("top_k"),
            after=after_routing.get("top_k"),
        ),
        _delta_if_changed(
            kind=DeltaKind.OVERLAP_POLICY,
            source="loop_semantics.overlap_policy",
            before=before_loop.get("overlap_policy"),
            after=after_loop.get("overlap_policy"),
        ),
        _delta_if_changed(
            kind=DeltaKind.SYNC_POLICY,
            source="loop_semantics.sync_policy",
            before=before_loop.get("sync_policy"),
            after=after_loop.get("sync_policy"),
        ),
        _delta_if_changed(
            kind=DeltaKind.RECOMPUTE_POLICY,
            source="memory_compute.recompute",
            before=before_memory.get("recompute"),
            after=after_memory.get("recompute"),
        ),
        _delta_if_changed(
            kind=DeltaKind.MICROBATCH_POLICY,
            source="memory_compute.micro_batches",
            before=before_memory.get("micro_batches"),
            after=after_memory.get("micro_batches"),
        ),
        _delta_if_changed(
            kind=DeltaKind.MEMORY_POLICY,
            source="routing.capacity_factor",
            before=before_routing.get("capacity_factor"),
            after=after_routing.get("capacity_factor"),
        ),
    ]
    return tuple(delta for delta in deltas if delta is not None)
