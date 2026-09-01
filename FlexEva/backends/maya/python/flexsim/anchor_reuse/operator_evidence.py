"""Operator-stub evidence matching against the anchor operator DAG."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .dryrun_bridge import BoundaryContextCapsule, EmissionDAG, OperatorStub
from .source_diff import SourceDiffHint
from .schema import (
    AnchorRegionGraph,
    AnchorWitness,
    DependencyType,
    DeltaKind,
    Obligation,
    ObligationActionHint,
    ObligationEvidenceItem,
    ObligationEvidenceKind,
    ObligationKind,
    ObligationSourceKind,
    ObligationStatus,
    RegionKind,
)


class OperatorEvidenceKind(str, Enum):
    COLLECTIVE_SYNC = "collective_sync"
    MEMORY_EFFECT = "memory_effect"
    DISPATCH_FLOW = "dispatch_flow"
    OVERLAP_EFFECT = "overlap_effect"


class EmissionOperatorProvenanceKind(str, Enum):
    DIRECT_EVIDENCE_MATCH = "direct_evidence_match"
    SYNTHESIZED_REGION_MAPPING = "synthesized_region_mapping"
    CONTROL_HIDDEN_NOOP = "control_hidden_noop"


@dataclass(frozen=True)
class OperatorStubDelta:
    stub: OperatorStub
    change: str


@dataclass(frozen=True)
class OperatorEdgeEvidence:
    stub_id: str
    callee_name: str
    src_region_id: str
    dst_region_id: str
    evidence_kind: OperatorEvidenceKind
    delta_kind: DeltaKind
    confidence: float = 1.0
    branch_ids: tuple[int, ...] = ()
    rationale: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperatorEvidenceMatchResult:
    matched_evidence: tuple[OperatorEdgeEvidence, ...]
    unmatched_deltas: tuple[OperatorStubDelta, ...] = ()


@dataclass(frozen=True)
class EmissionOperatorLink:
    emission_id: str
    stub_id: str
    emission_signature: str
    site_signature: str
    structure_signature: str
    block_signature: str
    callee_name: str
    boundary_kind: str
    branch_ids: tuple[int, ...]
    touched_region_ids: tuple[str, ...]
    dependency_type: DependencyType
    provenance_kind: EmissionOperatorProvenanceKind
    neighborhood_signature: str = ""
    evidence_rows: tuple[OperatorEdgeEvidence, ...] = ()
    rationale: tuple[str, ...] = ()


@dataclass(frozen=True)
class EmissionOperatorMap:
    scope_id: str
    links: tuple[EmissionOperatorLink, ...]
    notes: tuple[str, ...] = ()

    def links_for_stub(
        self,
        stub: OperatorStub,
        *,
        source_diff_hint: SourceDiffHint | None = None,
    ) -> tuple[EmissionOperatorLink, ...]:
        if source_diff_hint is not None:
            preferred_site_signature = source_diff_hint.preferred_site_signature(stub.stub_id)
            if preferred_site_signature:
                hinted = tuple(
                    link for link in self.links if link.site_signature == preferred_site_signature
                )
                if len(hinted) == 1:
                    return hinted
            if source_diff_hint.is_ambiguous(stub.stub_id):
                exact = ()
            else:
                exact = tuple(link for link in self.links if link.stub_id == stub.stub_id)
                if exact:
                    return exact
        else:
            exact = tuple(link for link in self.links if link.stub_id == stub.stub_id)
            if exact:
                return exact
        same_site = tuple(
            link
            for link in self.links
            if stub.site_signature and link.site_signature == stub.site_signature
        )
        if same_site and not (source_diff_hint is not None and source_diff_hint.is_ambiguous(stub.stub_id)):
            return same_site
        same_block = tuple(
            link
            for link in self.links
            if stub.block_signature and link.block_signature == stub.block_signature
        )
        if same_block:
            return same_block
        same_structure = tuple(
            link
            for link in self.links
            if stub.structure_signature and link.structure_signature == stub.structure_signature
        )
        if same_structure:
            return same_structure
        same_neighborhood = tuple(
            link
            for link in self.links
            if stub.neighborhood_signature and link.neighborhood_signature == stub.neighborhood_signature
        )
        if same_neighborhood:
            return same_neighborhood
        same_signature = tuple(
            link
            for link in self.links
            if stub.emission_signature and link.emission_signature == stub.emission_signature
        )
        if same_signature:
            return same_signature
        same_context = tuple(
            link
            for link in self.links
            if link.callee_name == stub.callee_name
            and link.boundary_kind == stub.boundary_kind
            and link.branch_ids == stub.branch_ids
        )
        if same_context:
            return same_context
        same_callee = tuple(
            link
            for link in self.links
            if link.callee_name == stub.callee_name and link.boundary_kind == stub.boundary_kind
        )
        return same_callee


@dataclass(frozen=True)
class ProjectionSufficiencyResult:
    sufficient: bool
    total_emitted_count: int
    resolved_emitted_count: int
    unresolved_stub_ids: tuple[str, ...] = ()
    ambiguous_stub_ids: tuple[str, ...] = ()
    incompatible_capsule_ids: tuple[str, ...] = ()
    residual_ratio: float = 0.0
    ambiguity_ratio: float = 0.0
    obligations: tuple[Obligation, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperatorEffectUpdate:
    emitted_stub_ids: tuple[str, ...]
    touched_region_ids: tuple[str, ...]
    dependency_type: DependencyType
    crosses_coordination_cut: bool = False
    connected_subgraph: bool = True
    affected_timing_share: float = 0.0
    bounded_affected_subgraph: bool = True
    evidence_rows: tuple[OperatorEdgeEvidence, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectionWitness:
    subject_kind: str
    requested_stub_ids: tuple[str, ...]
    subject_stub_ids: tuple[str, ...]
    matched_stub_ids: tuple[str, ...]
    unresolved_stub_ids: tuple[str, ...]
    ambiguous_stub_ids: tuple[str, ...]
    incompatible_capsule_ids: tuple[str, ...]
    total_subject_count: int
    resolved_subject_count: int
    match_result: OperatorEvidenceMatchResult
    operator_effect_update: OperatorEffectUpdate | None
    projection_sufficiency: ProjectionSufficiencyResult
    obligations: tuple[Obligation, ...] = ()
    notes: tuple[str, ...] = ()


_DEFAULT_AFFECTED_TIMING_SHARE_BOUND = 0.60


def _stub_compare_key(stub: OperatorStub) -> tuple:
    if stub.site_signature:
        return ("site", stub.site_signature)
    if stub.neighborhood_signature:
        return ("neighborhood", stub.neighborhood_signature)
    if stub.emission_signature:
        return ("signature", stub.emission_signature)
    return (
        "legacy",
        stub.callee_name,
        stub.lineno,
        stub.branch_ids,
        stub.boundary_kind,
    )


def _structure_tiebreak_key(stub: OperatorStub) -> tuple | None:
    if not stub.structure_signature:
        return None
    return (
        stub.emission_signature,
        stub.structure_signature,
    )


def _block_tiebreak_key(stub: OperatorStub) -> tuple | None:
    if not stub.block_signature:
        return None
    return (
        stub.emission_signature,
        stub.block_signature,
    )


def compare_operator_stubs(
    before: tuple[OperatorStub, ...] | list[OperatorStub] = (),
    after: tuple[OperatorStub, ...] | list[OperatorStub] = (),
    *,
    source_diff_hint: SourceDiffHint | None = None,
) -> tuple[OperatorStubDelta, ...]:
    before_remaining = list(before)
    after_remaining = list(after)
    forced_added_after: list[OperatorStub] = []

    if source_diff_hint is not None:
        before_by_site = {stub.site_signature: stub for stub in before_remaining if stub.site_signature}
        paired_before_ids: set[str] = set()
        paired_after_ids: set[str] = set()
        for after_stub in after_remaining:
            preferred_site_signature = source_diff_hint.preferred_site_signature(after_stub.stub_id)
            if not preferred_site_signature:
                continue
            before_stub = before_by_site.get(preferred_site_signature)
            if before_stub is None or before_stub.stub_id in paired_before_ids:
                continue
            paired_before_ids.add(before_stub.stub_id)
            paired_after_ids.add(after_stub.stub_id)
        if paired_before_ids or paired_after_ids:
            before_remaining = [
                stub for stub in before_remaining if stub.stub_id not in paired_before_ids
            ]
            after_remaining = [
                stub for stub in after_remaining if stub.stub_id not in paired_after_ids
            ]
        ambiguous_after_ids = {
            stub.stub_id
            for stub in after_remaining
            if source_diff_hint.is_ambiguous(stub.stub_id)
        }
        if ambiguous_after_ids:
            forced_added_after = [
                stub for stub in after_remaining if stub.stub_id in ambiguous_after_ids
            ]
            after_remaining = [
                stub for stub in after_remaining if stub.stub_id not in ambiguous_after_ids
            ]

    def _group_by_compare_key(
        stubs: tuple[OperatorStub, ...] | list[OperatorStub],
    ) -> dict[tuple, list[OperatorStub]]:
        grouped: dict[tuple, list[OperatorStub]] = {}
        for stub in stubs:
            grouped.setdefault(_stub_compare_key(stub), []).append(stub)
        return grouped

    before_groups = _group_by_compare_key(before_remaining)
    after_groups = _group_by_compare_key(after_remaining)
    unmatched_before: list[OperatorStub] = []
    unmatched_after: list[OperatorStub] = []
    for key in sorted(set(before_groups) | set(after_groups), key=repr):
        before_items = list(before_groups.get(key, ()))
        after_items = list(after_groups.get(key, ()))
        paired = min(len(before_items), len(after_items))
        unmatched_before.extend(before_items[paired:])
        unmatched_after.extend(after_items[paired:])

    before_block_groups: dict[tuple, list[OperatorStub]] = {}
    after_block_groups: dict[tuple, list[OperatorStub]] = {}
    remaining_before: list[OperatorStub] = []
    remaining_after: list[OperatorStub] = []
    for stub in unmatched_before:
        key = _block_tiebreak_key(stub)
        if key is None:
            remaining_before.append(stub)
            continue
        before_block_groups.setdefault(key, []).append(stub)
    for stub in unmatched_after:
        key = _block_tiebreak_key(stub)
        if key is None:
            remaining_after.append(stub)
            continue
        after_block_groups.setdefault(key, []).append(stub)
    unmatched_before = []
    unmatched_after = []
    for key in sorted(set(before_block_groups) | set(after_block_groups), key=repr):
        before_items = list(before_block_groups.get(key, ()))
        after_items = list(after_block_groups.get(key, ()))
        paired = min(len(before_items), len(after_items))
        unmatched_before.extend(before_items[paired:])
        unmatched_after.extend(after_items[paired:])

    before_structure_groups: dict[tuple, list[OperatorStub]] = {}
    after_structure_groups: dict[tuple, list[OperatorStub]] = {}
    remaining_before = []
    remaining_after = []
    for stub in unmatched_before:
        key = _structure_tiebreak_key(stub)
        if key is None:
            remaining_before.append(stub)
            continue
        before_structure_groups.setdefault(key, []).append(stub)
    for stub in unmatched_after:
        key = _structure_tiebreak_key(stub)
        if key is None:
            remaining_after.append(stub)
            continue
        after_structure_groups.setdefault(key, []).append(stub)
    for key in sorted(set(before_structure_groups) | set(after_structure_groups), key=repr):
        before_items = list(before_structure_groups.get(key, ()))
        after_items = list(after_structure_groups.get(key, ()))
        paired = min(len(before_items), len(after_items))
        remaining_before.extend(before_items[paired:])
        remaining_after.extend(after_items[paired:])

    def _stub_sort_key(stub: OperatorStub) -> tuple:
        return (stub.callee_name, stub.lineno, stub.stub_id)

    deltas: list[OperatorStubDelta] = []
    for after_stub in sorted(remaining_after + forced_added_after, key=_stub_sort_key):
        deltas.append(OperatorStubDelta(stub=after_stub, change="added"))
    for before_stub in sorted(remaining_before, key=_stub_sort_key):
        deltas.append(OperatorStubDelta(stub=before_stub, change="removed"))
    return tuple(deltas)


def _stub_evidence_spec(stub: OperatorStub) -> tuple[OperatorEvidenceKind, DeltaKind, set[RegionKind]] | None:
    name = stub.callee_name.lower()
    if any(token in name for token in ("allreduce", "all_reduce", "alltoall", "all_to_all", "barrier", "group", "collective", "nccl")):
        return (
            OperatorEvidenceKind.COLLECTIVE_SYNC,
            DeltaKind.SYNC_POLICY,
            {RegionKind.COLLECTIVE, RegionKind.OVERLAP, RegionKind.DISPATCH},
        )
    if any(token in name for token in ("checkpoint", "save", "recompute", "memory")):
        return (
            OperatorEvidenceKind.MEMORY_EFFECT,
            DeltaKind.MEMORY_POLICY,
            {RegionKind.MEMORY, RegionKind.EXPERT_COMPUTE},
        )
    if any(token in name for token in ("dispatch", "route", "expert")):
        return (
            OperatorEvidenceKind.DISPATCH_FLOW,
            DeltaKind.ROUTING_PRESSURE,
            {RegionKind.DISPATCH, RegionKind.EXPERT_COMPUTE, RegionKind.COLLECTIVE},
        )
    if any(token in name for token in ("overlap", "sync")):
        return (
            OperatorEvidenceKind.OVERLAP_EFFECT,
            DeltaKind.OVERLAP_POLICY,
            {RegionKind.OVERLAP, RegionKind.COLLECTIVE},
        )
    return None


def _capsule_allows_noop_operator_mapping(
    capsule: BoundaryContextCapsule | None,
) -> bool:
    if capsule is None:
        return False
    return_summary = capsule.return_summary
    side_effect_summary = capsule.side_effect_summary
    if return_summary is not None and (
        return_summary.logic_observable
        or return_summary.return_kind not in {"void_like"}
    ):
        return False
    if side_effect_summary is not None and (
        side_effect_summary.advances_group_state or side_effect_summary.mutates_state
    ):
        return False
    return True


def _anchor_map_target_kinds(
    stub: OperatorStub,
    target_kinds: set[RegionKind],
) -> set[RegionKind]:
    lowered = stub.callee_name.lower()
    if any(token in lowered for token in ("save", "checkpoint")):
        return {RegionKind.MEMORY}
    return set(target_kinds)


def _shortest_region_path(
    graph: AnchorRegionGraph,
    src_region_id: str,
    dst_region_id: str,
) -> tuple[str, ...]:
    if src_region_id == dst_region_id:
        return (src_region_id,)
    adjacency: dict[str, list[str]] = {}
    for edge in graph.edges:
        adjacency.setdefault(edge.src_region_id, []).append(edge.dst_region_id)
        adjacency.setdefault(edge.dst_region_id, []).append(edge.src_region_id)
    queue: list[tuple[str, tuple[str, ...]]] = [(src_region_id, (src_region_id,))]
    seen = {src_region_id}
    while queue:
        current, path = queue.pop(0)
        for neighbor in adjacency.get(current, ()):
            if neighbor in seen:
                continue
            next_path = path + (neighbor,)
            if neighbor == dst_region_id:
                return next_path
            seen.add(neighbor)
            queue.append((neighbor, next_path))
    return (src_region_id, dst_region_id)


def _minimal_connected_region_closure(
    graph: AnchorRegionGraph,
    seed_region_ids: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    seeds = tuple(dict.fromkeys(str(region_id) for region_id in seed_region_ids))
    if len(seeds) <= 1:
        return seeds
    ordered = list(seeds[:1])
    seen = set(ordered)
    for seed_region_id in seeds[1:]:
        path = _shortest_region_path(graph, ordered[-1], seed_region_id)
        for region_id in path[1:]:
            if region_id in seen:
                continue
            seen.add(region_id)
            ordered.append(region_id)
    return tuple(ordered)


def _synthesized_evidence_rows_for_regions(
    graph: AnchorRegionGraph,
    *,
    stub: OperatorStub,
    touched_region_ids: tuple[str, ...],
    evidence_kind: OperatorEvidenceKind,
    delta_kind: DeltaKind,
    rationale: tuple[str, ...],
) -> tuple[OperatorEdgeEvidence, ...]:
    if len(touched_region_ids) <= 1:
        return ()
    edge_set = {
        (edge.src_region_id, edge.dst_region_id)
        for edge in graph.edges
    }
    rows: list[OperatorEdgeEvidence] = []
    for left, right in zip(touched_region_ids, touched_region_ids[1:], strict=False):
        if (left, right) not in edge_set and (right, left) not in edge_set:
            continue
        rows.append(
            OperatorEdgeEvidence(
                stub_id=stub.stub_id,
                callee_name=stub.callee_name,
                src_region_id=left,
                dst_region_id=right,
                evidence_kind=evidence_kind,
                delta_kind=delta_kind,
                confidence=0.95,
                branch_ids=stub.branch_ids,
                rationale=rationale,
            )
        )
    return tuple(rows)


def build_emission_operator_map(
    graph: AnchorRegionGraph,
    emission_dag: EmissionDAG | None,
    *,
    boundary_capsules: tuple[BoundaryContextCapsule, ...] | list[BoundaryContextCapsule] = (),
) -> EmissionOperatorMap | None:
    if emission_dag is None or not emission_dag.nodes:
        return None
    links: list[EmissionOperatorLink] = []
    notes: list[str] = []
    capsule_by_stub_id = {
        capsule.stub_id: capsule
        for capsule in boundary_capsules
    }
    for node in emission_dag.nodes:
        stub = OperatorStub(
            stub_id=node.stub_id,
            emission_signature=node.emission_signature,
            site_signature=node.site_signature,
            structure_signature=node.structure_signature,
            block_signature=node.block_signature,
            neighborhood_signature=node.neighborhood_signature,
            callee_name=node.callee_name,
            source_path=node.source_path,
            lineno=node.lineno,
            boundary_kind="opaque_call",
            branch_ids=node.branch_ids,
            notes=node.notes,
        )
        match_result = resolve_operator_stub_evidence(
            graph,
            (OperatorStubDelta(stub=stub, change="anchor"),),
        )
        if not match_result.matched_evidence:
            capsule = capsule_by_stub_id.get(stub.stub_id)
            if _capsule_allows_noop_operator_mapping(capsule):
                links.append(
                    EmissionOperatorLink(
                        emission_id=node.emission_id,
                        stub_id=node.stub_id,
                        emission_signature=node.emission_signature,
                        site_signature=node.site_signature,
                        structure_signature=node.structure_signature,
                        block_signature=node.block_signature,
                        neighborhood_signature=node.neighborhood_signature,
                        callee_name=node.callee_name,
                        boundary_kind="opaque_call",
                        branch_ids=node.branch_ids,
                        touched_region_ids=(),
                        dependency_type=DependencyType.NARROW,
                        provenance_kind=EmissionOperatorProvenanceKind.CONTROL_HIDDEN_NOOP,
                        evidence_rows=(),
                        rationale=(
                            "anchor-time emission-to-operator mapping",
                            "control-hidden boundary emission with no operator-DAG effect",
                        ),
                    )
                )
                continue
            notes.append(f"no anchor operator mapping for emission {node.emission_id}")
            continue
        spec = _stub_evidence_spec(stub)
        if spec is not None:
            evidence_kind, delta_kind, target_kinds = spec
            map_target_kinds = _anchor_map_target_kinds(stub, target_kinds)
            seed_region_ids = tuple(
                dict.fromkeys(
                    region_id
                    for evidence in match_result.matched_evidence
                    for region_id in (evidence.src_region_id, evidence.dst_region_id)
                    if graph.region_by_id()[region_id].region_kind in map_target_kinds
                )
            )
            if not seed_region_ids:
                seed_region_ids = tuple(
                    region.region_id
                    for region in graph.regions
                    if region.region_kind in map_target_kinds
                )
            if len(map_target_kinds) == 1 and len(seed_region_ids) > 1:
                region_by_id = graph.region_by_id()
                seed_region_ids = (
                    min(
                        seed_region_ids,
                        key=lambda region_id: region_by_id[region_id].timing_share,
                    ),
                )
            touched_region_ids = _minimal_connected_region_closure(graph, seed_region_ids)
            dependency_type = (
                DependencyType.WIDE
                if evidence_kind in {
                    OperatorEvidenceKind.COLLECTIVE_SYNC,
                    OperatorEvidenceKind.OVERLAP_EFFECT,
                }
                else DependencyType.NARROW
            )
            evidence_rows = _synthesized_evidence_rows_for_regions(
                graph,
                stub=stub,
                touched_region_ids=touched_region_ids,
                evidence_kind=evidence_kind,
                delta_kind=delta_kind,
                rationale=("anchor-time emission-to-operator mapping",),
            )
        else:
            touched_region_ids_list: list[str] = []
            seen_region_ids: set[str] = set()
            dependency_type = DependencyType.NARROW
            for evidence in match_result.matched_evidence:
                for region_id in (evidence.src_region_id, evidence.dst_region_id):
                    if region_id in seen_region_ids:
                        continue
                    seen_region_ids.add(region_id)
                    touched_region_ids_list.append(region_id)
                if evidence.evidence_kind in {
                    OperatorEvidenceKind.COLLECTIVE_SYNC,
                    OperatorEvidenceKind.OVERLAP_EFFECT,
                }:
                    dependency_type = DependencyType.WIDE
            touched_region_ids = tuple(touched_region_ids_list)
            evidence_rows = match_result.matched_evidence
        links.append(
            EmissionOperatorLink(
                emission_id=node.emission_id,
                stub_id=node.stub_id,
                emission_signature=node.emission_signature,
                site_signature=node.site_signature,
                structure_signature=node.structure_signature,
                block_signature=node.block_signature,
                neighborhood_signature=node.neighborhood_signature,
                callee_name=node.callee_name,
                boundary_kind="opaque_call",
                branch_ids=node.branch_ids,
                touched_region_ids=touched_region_ids,
                dependency_type=dependency_type,
                provenance_kind=(
                    EmissionOperatorProvenanceKind.SYNTHESIZED_REGION_MAPPING
                    if spec is not None
                    else EmissionOperatorProvenanceKind.DIRECT_EVIDENCE_MATCH
                ),
                evidence_rows=evidence_rows,
                rationale=("anchor-time emission-to-operator mapping",),
            )
        )
    return EmissionOperatorMap(
        scope_id=emission_dag.scope_id,
        links=tuple(links),
        notes=tuple(notes) or ("anchor-time emission-to-operator map",),
    )


def _resolve_operator_stub_evidence_via_map(
    emission_operator_map: EmissionOperatorMap,
    stub_delta: OperatorStubDelta,
    *,
    source_diff_hint: SourceDiffHint | None = None,
) -> tuple[tuple[OperatorEdgeEvidence, ...], bool, bool]:
    links = emission_operator_map.links_for_stub(
        stub_delta.stub,
        source_diff_hint=source_diff_hint,
    )
    if not links:
        return (), False, False
    if len(links) != 1:
        return (), True, False
    link = links[0]
    if not link.touched_region_ids and not link.evidence_rows:
        return (), False, True
    evidence_rows = tuple(
        OperatorEdgeEvidence(
            stub_id=stub_delta.stub.stub_id,
            callee_name=stub_delta.stub.callee_name,
            src_region_id=row.src_region_id,
            dst_region_id=row.dst_region_id,
            evidence_kind=row.evidence_kind,
            delta_kind=row.delta_kind,
            confidence=max(row.confidence, 0.95),
            branch_ids=stub_delta.stub.branch_ids,
            rationale=row.rationale + ("matched via anchor emission-to-operator map",),
        )
        for row in link.evidence_rows
    )
    if evidence_rows:
        return evidence_rows, False, True
    region_ids = tuple(dict.fromkeys(link.touched_region_ids))
    if len(region_ids) >= 2:
        fallback_rows = tuple(
            OperatorEdgeEvidence(
                stub_id=stub_delta.stub.stub_id,
                callee_name=stub_delta.stub.callee_name,
                src_region_id=left,
                dst_region_id=right,
                evidence_kind=(
                    OperatorEvidenceKind.COLLECTIVE_SYNC
                    if link.dependency_type == DependencyType.WIDE
                    else OperatorEvidenceKind.MEMORY_EFFECT
                ),
                delta_kind=(
                    DeltaKind.SYNC_POLICY
                    if link.dependency_type == DependencyType.WIDE
                    else DeltaKind.MEMORY_POLICY
                ),
                confidence=0.95,
                branch_ids=stub_delta.stub.branch_ids,
                rationale=link.rationale + ("fallback edge synthesis from mapped touched regions",),
            )
            for left, right in zip(region_ids, region_ids[1:], strict=False)
        )
        return fallback_rows, False, True
    return (), False, True


def resolve_operator_stub_evidence(
    graph: AnchorRegionGraph,
    stub_deltas: tuple[OperatorStubDelta, ...] | list[OperatorStubDelta],
    *,
    emission_operator_map: EmissionOperatorMap | None = None,
    allow_heuristic_fallback: bool = True,
    source_diff_hint: SourceDiffHint | None = None,
) -> OperatorEvidenceMatchResult:
    evidence_rows: list[OperatorEdgeEvidence] = []
    unmatched: list[OperatorStubDelta] = []
    edges = tuple(graph.edges)
    for stub_delta in stub_deltas:
        if emission_operator_map is not None:
            mapped_rows, ambiguous, resolved_via_map = _resolve_operator_stub_evidence_via_map(
                emission_operator_map,
                stub_delta,
                source_diff_hint=source_diff_hint,
            )
            if mapped_rows:
                evidence_rows.extend(mapped_rows)
                continue
            if ambiguous:
                unmatched.append(stub_delta)
                continue
            if resolved_via_map:
                continue
            if not allow_heuristic_fallback:
                unmatched.append(stub_delta)
                continue
        spec = _stub_evidence_spec(stub_delta.stub)
        if spec is None:
            unmatched.append(stub_delta)
            continue
        evidence_kind, delta_kind, target_kinds = spec
        matched_any = False
        for edge in edges:
            src_kind = graph.region_by_id()[edge.src_region_id].region_kind
            dst_kind = graph.region_by_id()[edge.dst_region_id].region_kind
            if src_kind not in target_kinds and dst_kind not in target_kinds:
                continue
            matched_any = True
            evidence_rows.append(
                OperatorEdgeEvidence(
                    stub_id=stub_delta.stub.stub_id,
                    callee_name=stub_delta.stub.callee_name,
                    src_region_id=edge.src_region_id,
                    dst_region_id=edge.dst_region_id,
                    evidence_kind=evidence_kind,
                    delta_kind=delta_kind,
                    confidence=1.0 if stub_delta.change == "added" else 0.85,
                    branch_ids=stub_delta.stub.branch_ids,
                    rationale=(
                        f"operator-stub {stub_delta.change}",
                        f"callee={stub_delta.stub.callee_name}",
                    ),
                )
            )
        if not matched_any:
            unmatched.append(stub_delta)
    return OperatorEvidenceMatchResult(
        matched_evidence=tuple(evidence_rows),
        unmatched_deltas=tuple(unmatched),
    )


def match_operator_stub_evidence(
    graph: AnchorRegionGraph,
    stub_deltas: tuple[OperatorStubDelta, ...] | list[OperatorStubDelta],
    *,
    emission_operator_map: EmissionOperatorMap | None = None,
    allow_heuristic_fallback: bool = True,
    source_diff_hint: SourceDiffHint | None = None,
) -> tuple[OperatorEdgeEvidence, ...]:
    return resolve_operator_stub_evidence(
        graph,
        stub_deltas,
        emission_operator_map=emission_operator_map,
        allow_heuristic_fallback=allow_heuristic_fallback,
        source_diff_hint=source_diff_hint,
    ).matched_evidence


def obligations_for_unmatched_stub_deltas(
    stub_deltas: tuple[OperatorStubDelta, ...] | list[OperatorStubDelta],
    *,
    emission_operator_map: EmissionOperatorMap | None = None,
    source_diff_hint: SourceDiffHint | None = None,
) -> tuple[Obligation, ...]:
    obligations: list[Obligation] = []
    seen_ids: set[str] = set()
    for stub_delta in stub_deltas:
        obligation_id = f"operator_stub:{stub_delta.change}:{stub_delta.stub.stub_id}"
        if obligation_id in seen_ids:
            continue
        seen_ids.add(obligation_id)
        related_region_ids = _projection_region_hints_for_stub(
            stub_delta.stub,
            emission_operator_map=emission_operator_map,
            source_diff_hint=source_diff_hint,
        )
        evidence = [
            ObligationEvidenceItem(
                ObligationEvidenceKind.SUMMARY,
                f"unmatched operator stub {stub_delta.change}",
            ),
            ObligationEvidenceItem(
                ObligationEvidenceKind.CALLEE,
                stub_delta.stub.callee_name,
            ),
        ]
        if related_region_ids:
            evidence.append(
                ObligationEvidenceItem(
                    ObligationEvidenceKind.CANDIDATE_REGIONS,
                    ",".join(related_region_ids),
                )
            )
        obligations.append(
            Obligation(
                obligation_id=obligation_id,
                kind=ObligationKind.STRUCTURAL_REPLAY,
                status=ObligationStatus.OPEN,
                region_id=related_region_ids[0] if len(related_region_ids) == 1 else None,
                related_region_ids=related_region_ids,
                source_kind=ObligationSourceKind.EMITTED_STUB,
                source_id=stub_delta.stub.stub_id,
                suggested_action=ObligationActionHint.REFRESH_OPERATOR_MAPPING,
                evidence=tuple(evidence),
            )
        )
    return tuple(obligations)


def _projection_obligations_for_stub_ids(
    stub_ids: tuple[str, ...],
    *,
    reason: str,
    region_ids_by_stub_id: dict[str, tuple[str, ...]] | None = None,
) -> tuple[Obligation, ...]:
    obligations: list[Obligation] = []
    for stub_id in stub_ids:
        related_region_ids = tuple(region_ids_by_stub_id.get(stub_id, ())) if region_ids_by_stub_id else ()
        evidence = [
            ObligationEvidenceItem(
                ObligationEvidenceKind.SUMMARY,
                f"operator projection {reason}",
            ),
            ObligationEvidenceItem(
                ObligationEvidenceKind.NOTE,
                f"stub_id={stub_id}",
            ),
        ]
        if related_region_ids:
            evidence.append(
                ObligationEvidenceItem(
                    ObligationEvidenceKind.CANDIDATE_REGIONS,
                    ",".join(related_region_ids),
                )
            )
        obligations.append(
            Obligation(
                obligation_id=f"projection::{reason}:{stub_id}",
                kind=ObligationKind.STRUCTURAL_REPLAY,
                status=ObligationStatus.OPEN,
                region_id=related_region_ids[0] if len(related_region_ids) == 1 else None,
                related_region_ids=related_region_ids,
                source_kind=ObligationSourceKind.EMITTED_STUB,
                source_id=stub_id,
                suggested_action=ObligationActionHint.REFRESH_OPERATOR_PROJECTION,
                evidence=tuple(evidence),
            )
        )
    return tuple(obligations)


def _projection_region_hints_for_stub(
    stub: OperatorStub,
    *,
    match_region_ids: tuple[str, ...] | list[str] = (),
    emission_operator_map: EmissionOperatorMap | None = None,
    source_diff_hint: SourceDiffHint | None = None,
) -> tuple[str, ...]:
    region_ids: list[str] = []
    seen: set[str] = set()
    for region_id in match_region_ids:
        normalized = str(region_id)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        region_ids.append(normalized)
    if region_ids or emission_operator_map is None:
        return tuple(region_ids)
    for link in emission_operator_map.links_for_stub(stub, source_diff_hint=source_diff_hint):
        for region_id in link.touched_region_ids:
            normalized = str(region_id)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            region_ids.append(normalized)
    return tuple(region_ids)


def _connected_region_set(
    graph: AnchorRegionGraph,
    region_ids: set[str],
) -> bool:
    if len(region_ids) <= 1:
        return True
    adjacency: dict[str, set[str]] = {region_id: set() for region_id in region_ids}
    for edge in graph.edges:
        if edge.src_region_id in region_ids and edge.dst_region_id in region_ids:
            adjacency[edge.src_region_id].add(edge.dst_region_id)
            adjacency[edge.dst_region_id].add(edge.src_region_id)
    start = next(iter(region_ids))
    seen = {start}
    queue = [start]
    while queue:
        current = queue.pop(0)
        for neighbor in adjacency.get(current, ()):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
    return seen == region_ids


def _affected_timing_share(
    witness: AnchorWitness,
    region_ids: tuple[str, ...] | list[str] | set[str],
) -> float:
    selected = {str(region_id) for region_id in region_ids}
    if not selected:
        return 0.0
    return sum(
        region.timing_share
        for region in witness.regions
        if region.region_id in selected
    )


def _normalized_incompatible_stub_ids(
    incompatible_capsule_ids: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            capsule_id.removeprefix("capsule::")
            for capsule_id in incompatible_capsule_ids
            if capsule_id.startswith("capsule::")
        )
    )


def _projection_sufficiency_from_match_result(
    graph: AnchorRegionGraph,
    *,
    match_result: OperatorEvidenceMatchResult,
    subject_stubs: tuple[OperatorStub, ...],
    incompatible_capsule_ids: tuple[str, ...] | list[str] = (),
    emission_operator_map: EmissionOperatorMap | None = None,
    source_diff_hint: SourceDiffHint | None = None,
    notes_prefix: str,
) -> ProjectionSufficiencyResult:
    evidence_by_stub: dict[str, list[OperatorEdgeEvidence]] = {}
    for item in match_result.matched_evidence:
        evidence_by_stub.setdefault(item.stub_id, []).append(item)
    unresolved_stub_ids = tuple(sorted(item.stub.stub_id for item in match_result.unmatched_deltas))
    ambiguous_stub_ids: list[str] = []
    region_ids_by_stub_id: dict[str, tuple[str, ...]] = {}
    for stub in subject_stubs:
        evidence_rows = evidence_by_stub.get(stub.stub_id, [])
        matched_region_ids = tuple(
            dict.fromkeys(
                region_id
                for row in evidence_rows
                for region_id in (row.src_region_id, row.dst_region_id)
            )
        )
        region_ids = _projection_region_hints_for_stub(
            stub,
            match_region_ids=matched_region_ids,
            emission_operator_map=emission_operator_map,
            source_diff_hint=source_diff_hint,
        )
        if region_ids:
            region_ids_by_stub_id[stub.stub_id] = region_ids
        connected_region_ids = {
            region_id
            for region_id in region_ids
        }
        if evidence_rows and not _connected_region_set(graph, connected_region_ids):
            ambiguous_stub_ids.append(stub.stub_id)
    incompatible_stub_ids = _normalized_incompatible_stub_ids(incompatible_capsule_ids)
    total = len(subject_stubs)
    unresolved_count = len(unresolved_stub_ids) + len(ambiguous_stub_ids) + len(incompatible_stub_ids)
    resolved_count = max(total - unresolved_count, 0)
    obligations = (
        obligations_for_unmatched_stub_deltas(
            match_result.unmatched_deltas,
            emission_operator_map=emission_operator_map,
            source_diff_hint=source_diff_hint,
        )
        + _projection_obligations_for_stub_ids(
            tuple(sorted(set(ambiguous_stub_ids))),
            reason="ambiguous",
            region_ids_by_stub_id=region_ids_by_stub_id,
        )
        + _projection_obligations_for_stub_ids(
            incompatible_stub_ids,
            reason="incompatible_capsule",
            region_ids_by_stub_id=region_ids_by_stub_id,
        )
    )
    residual_ratio = (unresolved_count / total) if total else 0.0
    ambiguity_ratio = (len(ambiguous_stub_ids) / total) if total else 0.0
    sufficient = not unresolved_stub_ids and not ambiguous_stub_ids and not incompatible_stub_ids
    return ProjectionSufficiencyResult(
        sufficient=sufficient,
        total_emitted_count=total,
        resolved_emitted_count=resolved_count,
        unresolved_stub_ids=unresolved_stub_ids,
        ambiguous_stub_ids=tuple(sorted(set(ambiguous_stub_ids))),
        incompatible_capsule_ids=tuple(sorted(set(incompatible_capsule_ids))),
        residual_ratio=residual_ratio,
        ambiguity_ratio=ambiguity_ratio,
        obligations=obligations,
        notes=(
            f"{notes_prefix}_residual_ratio={residual_ratio:.3f}",
            f"{notes_prefix}_ambiguity_ratio={ambiguity_ratio:.3f}",
        ),
    )


def _operator_effect_update_from_match_result(
    graph: AnchorRegionGraph,
    *,
    witness: AnchorWitness | None,
    stub_deltas: tuple[OperatorStubDelta, ...],
    match_result: OperatorEvidenceMatchResult,
    effect_stub_ids: tuple[str, ...],
    notes_prefix: str,
    affected_timing_share_bound: float = _DEFAULT_AFFECTED_TIMING_SHARE_BOUND,
    emission_operator_map: EmissionOperatorMap | None = None,
    source_diff_hint: SourceDiffHint | None = None,
) -> OperatorEffectUpdate:
    touched_region_ids: list[str] = []
    seen_regions: set[str] = set()
    dependency_type = DependencyType.NARROW
    crosses_coordination_cut = False
    map_links: list[EmissionOperatorLink] = []
    if emission_operator_map is not None:
        for stub_delta in stub_deltas:
            links = emission_operator_map.links_for_stub(
                stub_delta.stub,
                source_diff_hint=source_diff_hint,
            )
            if len(links) == 1:
                map_links.append(links[0])
        if map_links:
            for link in map_links:
                for region_id in link.touched_region_ids:
                    if region_id in seen_regions:
                        continue
                    seen_regions.add(region_id)
                    touched_region_ids.append(region_id)
                if link.dependency_type == DependencyType.WIDE:
                    dependency_type = DependencyType.WIDE
                    crosses_coordination_cut = True
    if not touched_region_ids:
        edge_type_by_key = {
            (edge.src_region_id, edge.dst_region_id): edge.dependency_type
            for edge in graph.edges
        }
        for item in match_result.matched_evidence:
            for region_id in (item.src_region_id, item.dst_region_id):
                if region_id in seen_regions:
                    continue
                seen_regions.add(region_id)
                touched_region_ids.append(region_id)
            edge_type = edge_type_by_key.get((item.src_region_id, item.dst_region_id))
            if edge_type == DependencyType.WIDE or item.evidence_kind in {
                OperatorEvidenceKind.COLLECTIVE_SYNC,
                OperatorEvidenceKind.OVERLAP_EFFECT,
            }:
                dependency_type = DependencyType.WIDE
                crosses_coordination_cut = True
    connected_subgraph = _connected_region_set(graph, set(touched_region_ids))
    affected_share = _affected_timing_share(witness, touched_region_ids) if witness is not None else 0.0
    bounded_affected_subgraph = connected_subgraph and affected_share <= affected_timing_share_bound
    notes = (
        f"{notes_prefix}={','.join(effect_stub_ids)}",
        f"matched_operator_regions={','.join(touched_region_ids) if touched_region_ids else '<none>'}",
        f"crosses_coordination_cut={crosses_coordination_cut}",
        f"connected_subgraph={connected_subgraph}",
        f"affected_timing_share={affected_share:.3f}",
        f"bounded_affected_subgraph={bounded_affected_subgraph}",
    )
    if match_result.unmatched_deltas:
        notes = notes + (
            "some projected stubs had no operator-effect match",
        )
    return OperatorEffectUpdate(
        emitted_stub_ids=effect_stub_ids,
        touched_region_ids=tuple(touched_region_ids),
        dependency_type=dependency_type,
        crosses_coordination_cut=crosses_coordination_cut,
        connected_subgraph=connected_subgraph,
        affected_timing_share=affected_share,
        bounded_affected_subgraph=bounded_affected_subgraph,
        evidence_rows=match_result.matched_evidence,
        notes=notes,
    )


def build_projection_witness_for_emitted_stub_ids(
    graph: AnchorRegionGraph,
    *,
    witness: AnchorWitness | None = None,
    operator_stubs: tuple[OperatorStub, ...] | list[OperatorStub],
    emitted_stub_ids: tuple[str, ...] | list[str],
    incompatible_capsule_ids: tuple[str, ...] | list[str] = (),
    affected_timing_share_bound: float = _DEFAULT_AFFECTED_TIMING_SHARE_BOUND,
    emission_operator_map: EmissionOperatorMap | None = None,
    allow_heuristic_fallback: bool = True,
    source_diff_hint: SourceDiffHint | None = None,
) -> ProjectionWitness:
    requested_stub_ids = tuple(dict.fromkeys(str(item) for item in emitted_stub_ids))
    emitted_set = set(requested_stub_ids)
    if not emitted_set:
        projection_sufficiency = ProjectionSufficiencyResult(
            sufficient=True,
            total_emitted_count=0,
            resolved_emitted_count=0,
            notes=("no emitted stubs to project",),
        )
        return ProjectionWitness(
            subject_kind="emitted_stubs",
            requested_stub_ids=requested_stub_ids,
            subject_stub_ids=(),
            matched_stub_ids=(),
            unresolved_stub_ids=(),
            ambiguous_stub_ids=(),
            incompatible_capsule_ids=(),
            total_subject_count=0,
            resolved_subject_count=0,
            match_result=OperatorEvidenceMatchResult((), ()),
            operator_effect_update=None,
            projection_sufficiency=projection_sufficiency,
            obligations=projection_sufficiency.obligations,
            notes=projection_sufficiency.notes,
        )
    emitted_stubs = tuple(stub for stub in operator_stubs if stub.stub_id in emitted_set)
    if not emitted_stubs:
        projection_sufficiency = ProjectionSufficiencyResult(
            sufficient=True,
            total_emitted_count=0,
            resolved_emitted_count=0,
            notes=("no emitted stubs matched captured boundary stubs",),
        )
        operator_effect_update = OperatorEffectUpdate(
            emitted_stub_ids=tuple(sorted(emitted_set)),
            touched_region_ids=(),
            dependency_type=DependencyType.NARROW,
            crosses_coordination_cut=False,
            connected_subgraph=True,
            affected_timing_share=0.0,
            bounded_affected_subgraph=True,
            notes=("no emitted operator stubs matched captured boundary stubs",),
        )
        return ProjectionWitness(
            subject_kind="emitted_stubs",
            requested_stub_ids=requested_stub_ids,
            subject_stub_ids=(),
            matched_stub_ids=(),
            unresolved_stub_ids=(),
            ambiguous_stub_ids=(),
            incompatible_capsule_ids=(),
            total_subject_count=0,
            resolved_subject_count=0,
            match_result=OperatorEvidenceMatchResult((), ()),
            operator_effect_update=operator_effect_update,
            projection_sufficiency=projection_sufficiency,
            obligations=projection_sufficiency.obligations,
            notes=projection_sufficiency.notes + operator_effect_update.notes,
        )
    stub_deltas = tuple(OperatorStubDelta(stub=stub, change="executed") for stub in emitted_stubs)
    match_result = resolve_operator_stub_evidence(
        graph,
        stub_deltas,
        emission_operator_map=emission_operator_map,
        allow_heuristic_fallback=allow_heuristic_fallback,
        source_diff_hint=source_diff_hint,
    )
    projection_sufficiency = _projection_sufficiency_from_match_result(
        graph,
        match_result=match_result,
        subject_stubs=emitted_stubs,
        incompatible_capsule_ids=incompatible_capsule_ids,
        emission_operator_map=emission_operator_map,
        source_diff_hint=source_diff_hint,
        notes_prefix="projection",
    )
    operator_effect_update = _operator_effect_update_from_match_result(
        graph,
        witness=witness,
        stub_deltas=stub_deltas,
        match_result=match_result,
        effect_stub_ids=tuple(stub.stub_id for stub in emitted_stubs),
        notes_prefix="projected_emitted_stubs",
        affected_timing_share_bound=affected_timing_share_bound,
        emission_operator_map=emission_operator_map,
        source_diff_hint=source_diff_hint,
    )
    incompatible_stub_ids = set(_normalized_incompatible_stub_ids(incompatible_capsule_ids))
    matched_stub_ids = tuple(
        stub.stub_id
        for stub in emitted_stubs
        if stub.stub_id not in set(projection_sufficiency.unresolved_stub_ids)
        and stub.stub_id not in set(projection_sufficiency.ambiguous_stub_ids)
        and stub.stub_id not in incompatible_stub_ids
    )
    return ProjectionWitness(
        subject_kind="emitted_stubs",
        requested_stub_ids=requested_stub_ids,
        subject_stub_ids=tuple(stub.stub_id for stub in emitted_stubs),
        matched_stub_ids=matched_stub_ids,
        unresolved_stub_ids=projection_sufficiency.unresolved_stub_ids,
        ambiguous_stub_ids=projection_sufficiency.ambiguous_stub_ids,
        incompatible_capsule_ids=projection_sufficiency.incompatible_capsule_ids,
        total_subject_count=projection_sufficiency.total_emitted_count,
        resolved_subject_count=projection_sufficiency.resolved_emitted_count,
        match_result=match_result,
        operator_effect_update=operator_effect_update,
        projection_sufficiency=projection_sufficiency,
        obligations=projection_sufficiency.obligations,
        notes=projection_sufficiency.notes + operator_effect_update.notes,
    )


def build_projection_witness_for_stub_deltas(
    graph: AnchorRegionGraph,
    *,
    witness: AnchorWitness | None = None,
    stub_deltas: tuple[OperatorStubDelta, ...] | list[OperatorStubDelta],
    incompatible_capsule_ids: tuple[str, ...] | list[str] = (),
    affected_timing_share_bound: float = _DEFAULT_AFFECTED_TIMING_SHARE_BOUND,
    emission_operator_map: EmissionOperatorMap | None = None,
    allow_heuristic_fallback: bool = True,
    source_diff_hint: SourceDiffHint | None = None,
) -> ProjectionWitness:
    delta_items = tuple(stub_deltas)
    requested_stub_ids = tuple(f"{item.change}:{item.stub.stub_id}" for item in delta_items)
    if not delta_items:
        projection_sufficiency = ProjectionSufficiencyResult(
            sufficient=True,
            total_emitted_count=0,
            resolved_emitted_count=0,
            notes=("no operator stub deltas to project",),
        )
        return ProjectionWitness(
            subject_kind="stub_deltas",
            requested_stub_ids=requested_stub_ids,
            subject_stub_ids=(),
            matched_stub_ids=(),
            unresolved_stub_ids=(),
            ambiguous_stub_ids=(),
            incompatible_capsule_ids=(),
            total_subject_count=0,
            resolved_subject_count=0,
            match_result=OperatorEvidenceMatchResult((), ()),
            operator_effect_update=None,
            projection_sufficiency=projection_sufficiency,
            obligations=projection_sufficiency.obligations,
            notes=projection_sufficiency.notes,
        )
    match_result = resolve_operator_stub_evidence(
        graph,
        delta_items,
        emission_operator_map=emission_operator_map,
        allow_heuristic_fallback=allow_heuristic_fallback,
        source_diff_hint=source_diff_hint,
    )
    subject_stubs = tuple(item.stub for item in delta_items)
    projection_sufficiency = _projection_sufficiency_from_match_result(
        graph,
        match_result=match_result,
        subject_stubs=subject_stubs,
        incompatible_capsule_ids=incompatible_capsule_ids,
        emission_operator_map=emission_operator_map,
        source_diff_hint=source_diff_hint,
        notes_prefix="stub_delta_projection",
    )
    operator_effect_update = _operator_effect_update_from_match_result(
        graph,
        witness=witness,
        stub_deltas=delta_items,
        match_result=match_result,
        effect_stub_ids=requested_stub_ids,
        notes_prefix="projected_stub_deltas",
        affected_timing_share_bound=affected_timing_share_bound,
        emission_operator_map=emission_operator_map,
        source_diff_hint=source_diff_hint,
    )
    incompatible_stub_ids = set(_normalized_incompatible_stub_ids(incompatible_capsule_ids))
    matched_stub_ids = tuple(
        f"{item.change}:{item.stub.stub_id}"
        for item in delta_items
        if item.stub.stub_id not in set(projection_sufficiency.unresolved_stub_ids)
        and item.stub.stub_id not in set(projection_sufficiency.ambiguous_stub_ids)
        and item.stub.stub_id not in incompatible_stub_ids
    )
    return ProjectionWitness(
        subject_kind="stub_deltas",
        requested_stub_ids=requested_stub_ids,
        subject_stub_ids=tuple(item.stub.stub_id for item in delta_items),
        matched_stub_ids=matched_stub_ids,
        unresolved_stub_ids=projection_sufficiency.unresolved_stub_ids,
        ambiguous_stub_ids=projection_sufficiency.ambiguous_stub_ids,
        incompatible_capsule_ids=projection_sufficiency.incompatible_capsule_ids,
        total_subject_count=projection_sufficiency.total_emitted_count,
        resolved_subject_count=projection_sufficiency.resolved_emitted_count,
        match_result=match_result,
        operator_effect_update=operator_effect_update,
        projection_sufficiency=projection_sufficiency,
        obligations=projection_sufficiency.obligations,
        notes=projection_sufficiency.notes + operator_effect_update.notes,
    )


def assess_operator_projection_sufficiency(
    graph: AnchorRegionGraph,
    *,
    operator_stubs: tuple[OperatorStub, ...] | list[OperatorStub],
    emitted_stub_ids: tuple[str, ...] | list[str],
    incompatible_capsule_ids: tuple[str, ...] | list[str] = (),
    emission_operator_map: EmissionOperatorMap | None = None,
    allow_heuristic_fallback: bool = True,
    source_diff_hint: SourceDiffHint | None = None,
) -> ProjectionSufficiencyResult:
    return build_projection_witness_for_emitted_stub_ids(
        graph,
        operator_stubs=operator_stubs,
        emitted_stub_ids=emitted_stub_ids,
        incompatible_capsule_ids=incompatible_capsule_ids,
        emission_operator_map=emission_operator_map,
        allow_heuristic_fallback=allow_heuristic_fallback,
        source_diff_hint=source_diff_hint,
    ).projection_sufficiency


def assess_operator_stub_delta_projection_sufficiency(
    graph: AnchorRegionGraph,
    *,
    stub_deltas: tuple[OperatorStubDelta, ...] | list[OperatorStubDelta],
    incompatible_capsule_ids: tuple[str, ...] | list[str] = (),
    emission_operator_map: EmissionOperatorMap | None = None,
    allow_heuristic_fallback: bool = True,
    source_diff_hint: SourceDiffHint | None = None,
) -> ProjectionSufficiencyResult:
    return build_projection_witness_for_stub_deltas(
        graph,
        stub_deltas=stub_deltas,
        incompatible_capsule_ids=incompatible_capsule_ids,
        emission_operator_map=emission_operator_map,
        allow_heuristic_fallback=allow_heuristic_fallback,
        source_diff_hint=source_diff_hint,
    ).projection_sufficiency


def project_emitted_stub_ids_to_operator_effects(
    graph: AnchorRegionGraph,
    *,
    witness: AnchorWitness | None = None,
    operator_stubs: tuple[OperatorStub, ...] | list[OperatorStub],
    emitted_stub_ids: tuple[str, ...] | list[str],
    affected_timing_share_bound: float = _DEFAULT_AFFECTED_TIMING_SHARE_BOUND,
    emission_operator_map: EmissionOperatorMap | None = None,
    allow_heuristic_fallback: bool = True,
    source_diff_hint: SourceDiffHint | None = None,
) -> OperatorEffectUpdate | None:
    return build_projection_witness_for_emitted_stub_ids(
        graph,
        witness=witness,
        operator_stubs=operator_stubs,
        emitted_stub_ids=emitted_stub_ids,
        affected_timing_share_bound=affected_timing_share_bound,
        emission_operator_map=emission_operator_map,
        allow_heuristic_fallback=allow_heuristic_fallback,
        source_diff_hint=source_diff_hint,
    ).operator_effect_update


def project_operator_stub_deltas_to_operator_effects(
    graph: AnchorRegionGraph,
    *,
    witness: AnchorWitness | None = None,
    stub_deltas: tuple[OperatorStubDelta, ...] | list[OperatorStubDelta],
    affected_timing_share_bound: float = _DEFAULT_AFFECTED_TIMING_SHARE_BOUND,
    emission_operator_map: EmissionOperatorMap | None = None,
    allow_heuristic_fallback: bool = True,
    source_diff_hint: SourceDiffHint | None = None,
) -> OperatorEffectUpdate | None:
    return build_projection_witness_for_stub_deltas(
        graph,
        witness=witness,
        stub_deltas=stub_deltas,
        affected_timing_share_bound=affected_timing_share_bound,
        emission_operator_map=emission_operator_map,
        allow_heuristic_fallback=allow_heuristic_fallback,
        source_diff_hint=source_diff_hint,
    ).operator_effect_update
