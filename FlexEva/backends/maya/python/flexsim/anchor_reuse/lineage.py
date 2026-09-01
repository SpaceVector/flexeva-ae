"""Anchor-lineage object for cached segment-level reuse planning."""

from __future__ import annotations

import json
import pickle
import gzip
from dataclasses import dataclass
from pathlib import Path

from .control_regions import ControlRegion, ControlRegionKind, ControlRegionTree
from .dryrun_bridge import (
    AbstractValueSummary,
    BlackBoxReturnSummary,
    BoundaryContextCapsule,
    EmissionDAG,
    EmissionEdge,
    EmissionNode,
    EmissionOwnership,
    EmissionOwnershipMap,
    LogicEmissionLink,
    LogicEmissionMap,
    LogicPointOwnership,
    LogicPointOwnershipMap,
    LogicSliceEdge,
    LogicSliceGraph,
    LogicSliceState,
    LogicStateStore,
    SideEffectSummary,
)
from .operator_evidence import EmissionOperatorLink, EmissionOperatorMap, build_emission_operator_map
from .logical_segments import ReplaySegment, ReplaySegmentBundle, build_replay_segments_from_trace_dir
from .program_logic import ProgramLogicCarrier, ProgramLogicKind, ProgramLogicPoint
from .schema import (
    AnchorWitness,
    AnchorRegion,
    DependencyType,
    Obligation,
    ObligationKind,
    ObligationStatus,
    RegionKind,
    Witness,
    WitnessGranularity,
    WitnessSourceKind,
    WitnessRegion,
)


@dataclass(frozen=True)
class RegionSegmentSlice:
    region_id: str
    start_window: int
    end_window: int
    segment_ids: tuple[str, ...]
    segment_fraction: float


@dataclass(frozen=True)
class AnchorLineage:
    witness: AnchorWitness
    anchor_trace_dir: str
    segment_max_events_per_rank: int | None
    replay_max_events_per_rank: int | None
    segment_bundle: ReplaySegmentBundle
    region_slices: tuple[RegionSegmentSlice, ...]
    program_logic_carrier: ProgramLogicCarrier | None = None
    boundary_capsules: tuple[BoundaryContextCapsule, ...] = ()
    control_region_tree: ControlRegionTree | None = None
    emission_dag: EmissionDAG | None = None
    logic_emission_map: LogicEmissionMap | None = None
    emission_ownership_map: EmissionOwnershipMap | None = None
    logic_point_ownership_map: LogicPointOwnershipMap | None = None
    logic_state_store: LogicStateStore | None = None
    logic_slice_graph: LogicSliceGraph | None = None
    emission_operator_map: EmissionOperatorMap | None = None

    def segment_ids_for_regions(self, region_ids: tuple[str, ...] | list[str] | set[str]) -> tuple[str, ...]:
        selected = {str(value) for value in region_ids}
        segment_ids: list[str] = []
        seen: set[str] = set()
        for region_slice in self.region_slices:
            if region_slice.region_id not in selected:
                continue
            for segment_id in region_slice.segment_ids:
                if segment_id in seen:
                    continue
                seen.add(segment_id)
                segment_ids.append(segment_id)
        return tuple(segment_ids)

    def segment_fraction_for_regions(self, region_ids: tuple[str, ...] | list[str] | set[str]) -> float:
        selected = {str(value) for value in region_ids}
        if not self.segment_bundle.segments:
            return 0.0
        touched = {
            segment_id
            for region_slice in self.region_slices
            if region_slice.region_id in selected
            for segment_id in region_slice.segment_ids
        }
        return len(touched) / len(self.segment_bundle.segments)


def build_anchor_lineage(
    witness: AnchorWitness,
    *,
    anchor_trace_dir: str | Path,
    max_events_per_rank: int | None = None,
    replay_max_events_per_rank: int | None = 500,
    program_logic_carrier: ProgramLogicCarrier | None = None,
    boundary_capsules: tuple[BoundaryContextCapsule, ...] = (),
    control_region_tree: ControlRegionTree | None = None,
    emission_dag: EmissionDAG | None = None,
    logic_emission_map: LogicEmissionMap | None = None,
    emission_ownership_map: EmissionOwnershipMap | None = None,
    logic_point_ownership_map: LogicPointOwnershipMap | None = None,
    logic_state_store: LogicStateStore | None = None,
    logic_slice_graph: LogicSliceGraph | None = None,
    emission_operator_map: EmissionOperatorMap | None = None,
) -> AnchorLineage:
    segment_bundle = build_replay_segments_from_trace_dir(
        anchor_trace_dir,
        max_events_per_rank=max_events_per_rank,
    )
    region_slices: list[RegionSegmentSlice] = []
    for region in witness.regions:
        if region.start_window is None or region.end_window is None:
            continue
        touched = tuple(
            segment.segment_id
            for segment in segment_bundle.segments
            if segment.start_window <= region.end_window and region.start_window <= segment.end_window
        )
        segment_fraction = len(touched) / max(len(segment_bundle.segments), 1)
        region_slices.append(
            RegionSegmentSlice(
                region_id=region.region_id,
                start_window=region.start_window,
                end_window=region.end_window,
                segment_ids=touched,
                segment_fraction=segment_fraction,
            )
        )
    resolved_emission_operator_map = emission_operator_map
    if resolved_emission_operator_map is None and emission_dag is not None:
        from .graph import build_anchor_region_graph

        resolved_emission_operator_map = build_emission_operator_map(
            build_anchor_region_graph(witness),
            emission_dag,
            boundary_capsules=boundary_capsules,
        )
    return AnchorLineage(
        witness=witness,
        anchor_trace_dir=str(anchor_trace_dir),
        segment_max_events_per_rank=max_events_per_rank,
        replay_max_events_per_rank=replay_max_events_per_rank,
        segment_bundle=segment_bundle,
        region_slices=tuple(region_slices),
        program_logic_carrier=program_logic_carrier,
        boundary_capsules=tuple(boundary_capsules),
        control_region_tree=control_region_tree,
        emission_dag=emission_dag,
        logic_emission_map=logic_emission_map,
        emission_ownership_map=emission_ownership_map,
        logic_point_ownership_map=logic_point_ownership_map,
        logic_state_store=logic_state_store,
        logic_slice_graph=logic_slice_graph,
        emission_operator_map=resolved_emission_operator_map,
    )


def _program_logic_carrier_to_jsonable(carrier: ProgramLogicCarrier | None) -> dict | None:
    if carrier is None:
        return None
    return {
        "source": carrier.source,
        "notes": list(carrier.notes),
        "points": [
            {
                "name": point.name,
                "kind": point.kind.value,
                "value": _freeze_program_logic_value(point.value),
                "source": point.source,
                "source_path": point.source_path,
                "lineno": point.lineno,
                "end_lineno": point.end_lineno,
                "branch_ids": list(point.branch_ids),
                "rank_groups": [list(group) for group in point.rank_groups],
                "source_fields": list(point.source_fields),
                "notes": list(point.notes),
            }
            for point in carrier.points
        ],
    }


def _freeze_program_logic_value(value):
    if isinstance(value, tuple):
        return {"__tuple__": [_freeze_program_logic_value(item) for item in value]}
    if isinstance(value, list):
        return [_freeze_program_logic_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _freeze_program_logic_value(item) for key, item in value.items()}
    return value


def _thaw_program_logic_value(value):
    if isinstance(value, dict) and "__tuple__" in value:
        return tuple(_thaw_program_logic_value(item) for item in value["__tuple__"])
    if isinstance(value, list):
        return tuple(_thaw_program_logic_value(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _thaw_program_logic_value(item) for key, item in value.items()}
    return value


def _program_logic_carrier_from_jsonable(payload: dict | None) -> ProgramLogicCarrier | None:
    if not isinstance(payload, dict):
        return None
    return ProgramLogicCarrier(
        source=str(payload["source"]),
        points=tuple(
            ProgramLogicPoint(
                name=str(point["name"]),
                kind=ProgramLogicKind(str(point["kind"])),
                value=_thaw_program_logic_value(point.get("value")),
                source=str(point["source"]),
                source_path=(
                    str(point["source_path"])
                    if point.get("source_path") is not None
                    else None
                ),
                lineno=int(point["lineno"]) if point.get("lineno") is not None else None,
                end_lineno=(
                    int(point["end_lineno"]) if point.get("end_lineno") is not None else None
                ),
                branch_ids=tuple(int(item) for item in point.get("branch_ids", [])),
                rank_groups=tuple(
                    tuple(int(rank) for rank in group)
                    for group in point.get("rank_groups", [])
                ),
                source_fields=tuple(str(item) for item in point.get("source_fields", [])),
                notes=tuple(str(item) for item in point.get("notes", [])),
            )
            for point in payload.get("points", [])
        ),
        notes=tuple(str(item) for item in payload.get("notes", [])),
    )


def _boundary_capsules_to_jsonable(
    boundary_capsules: tuple[BoundaryContextCapsule, ...] | list[BoundaryContextCapsule],
) -> list[dict]:
    return [
        {
            "capsule_id": capsule.capsule_id,
            "stub_id": capsule.stub_id,
            "emission_signature": capsule.emission_signature,
            "site_signature": capsule.site_signature,
            "structure_signature": capsule.structure_signature,
            "block_signature": capsule.block_signature,
            "callee_name": capsule.callee_name,
            "source_path": capsule.source_path,
            "lineno": capsule.lineno,
            "boundary_kind": capsule.boundary_kind,
            "branch_ids": list(capsule.branch_ids),
            "positional_arg_kinds": list(capsule.positional_arg_kinds),
            "keyword_arg_names": list(capsule.keyword_arg_names),
            "return_summary": (
                {
                    "return_kind": capsule.return_summary.return_kind,
                    "logic_observable": capsule.return_summary.logic_observable,
                    "abstract_values": [
                        {
                            "value_kind": value.value_kind,
                            "shape_hint": list(value.shape_hint),
                            "dtype_hint": value.dtype_hint,
                            "device_hint": value.device_hint,
                            "scalar_hint": value.scalar_hint,
                            "notes": list(value.notes),
                        }
                        for value in capsule.return_summary.abstract_values
                    ],
                    "notes": list(capsule.return_summary.notes),
                }
                if capsule.return_summary is not None
                else None
            ),
            "side_effect_summary": (
                {
                    "emits_operator_stub_ids": list(capsule.side_effect_summary.emits_operator_stub_ids),
                    "mutates_state": capsule.side_effect_summary.mutates_state,
                    "advances_group_state": capsule.side_effect_summary.advances_group_state,
                    "notes": list(capsule.side_effect_summary.notes),
                }
                if capsule.side_effect_summary is not None
                else None
            ),
            "notes": list(capsule.notes),
        }
        for capsule in boundary_capsules
    ]


def _boundary_capsules_from_jsonable(payload: list[dict] | None) -> tuple[BoundaryContextCapsule, ...]:
    if not isinstance(payload, list):
        return ()
    capsules: list[BoundaryContextCapsule] = []
    for capsule in payload:
        return_payload = capsule.get("return_summary")
        side_effect_payload = capsule.get("side_effect_summary")
        capsules.append(
            BoundaryContextCapsule(
                capsule_id=str(capsule["capsule_id"]),
                stub_id=str(capsule["stub_id"]),
                emission_signature=str(capsule.get("emission_signature", "")),
                site_signature=str(capsule.get("site_signature", "")),
                structure_signature=str(capsule.get("structure_signature", "")),
                block_signature=str(capsule.get("block_signature", "")),
                callee_name=str(capsule["callee_name"]),
                source_path=str(capsule["source_path"]),
                lineno=int(capsule["lineno"]),
                boundary_kind=str(capsule["boundary_kind"]),
                branch_ids=tuple(int(item) for item in capsule.get("branch_ids", [])),
                positional_arg_kinds=tuple(str(item) for item in capsule.get("positional_arg_kinds", [])),
                keyword_arg_names=tuple(str(item) for item in capsule.get("keyword_arg_names", [])),
                return_summary=(
                    BlackBoxReturnSummary(
                        return_kind=str(return_payload["return_kind"]),
                        abstract_values=tuple(
                            AbstractValueSummary(
                                value_kind=str(value["value_kind"]),
                                shape_hint=tuple(str(item) for item in value.get("shape_hint", [])),
                                dtype_hint=value.get("dtype_hint"),
                                device_hint=value.get("device_hint"),
                                scalar_hint=value.get("scalar_hint"),
                                notes=tuple(str(item) for item in value.get("notes", [])),
                            )
                            for value in return_payload.get("abstract_values", [])
                        ),
                        logic_observable=bool(return_payload.get("logic_observable", False)),
                        notes=tuple(str(item) for item in return_payload.get("notes", [])),
                    )
                    if isinstance(return_payload, dict)
                    else None
                ),
                side_effect_summary=(
                    SideEffectSummary(
                        emits_operator_stub_ids=tuple(
                            str(item) for item in side_effect_payload.get("emits_operator_stub_ids", [])
                        ),
                        mutates_state=bool(side_effect_payload.get("mutates_state", False)),
                        advances_group_state=bool(side_effect_payload.get("advances_group_state", False)),
                        notes=tuple(str(item) for item in side_effect_payload.get("notes", [])),
                    )
                    if isinstance(side_effect_payload, dict)
                    else None
                ),
                notes=tuple(str(item) for item in capsule.get("notes", [])),
            )
        )
    return tuple(capsules)


def _control_region_tree_to_jsonable(control_region_tree: ControlRegionTree | None) -> dict | None:
    if control_region_tree is None:
        return None
    return {
        "scope_id": control_region_tree.scope_id,
        "root_region_id": control_region_tree.root_region_id,
        "regions": [
            {
                "region_id": region.region_id,
                "region_name": region.region_name,
                "kind": region.kind.value,
                "source_path": region.source_path,
                "lineno": region.lineno,
                "end_lineno": region.end_lineno,
                "parent_region_id": region.parent_region_id,
                "child_region_ids": list(region.child_region_ids),
                "branch_ids": list(region.branch_ids),
                "logic_point_names": list(region.logic_point_names),
                "notes": list(region.notes),
            }
            for region in control_region_tree.regions
        ],
        "notes": list(control_region_tree.notes),
    }


def _control_region_tree_from_jsonable(payload: dict | None) -> ControlRegionTree | None:
    if not isinstance(payload, dict):
        return None
    return ControlRegionTree(
        scope_id=str(payload["scope_id"]),
        root_region_id=str(payload["root_region_id"]),
        regions=tuple(
            ControlRegion(
                region_id=str(region["region_id"]),
                region_name=str(region["region_name"]),
                kind=ControlRegionKind(str(region["kind"])),
                source_path=str(region["source_path"]),
                lineno=int(region["lineno"]),
                end_lineno=(
                    int(region["end_lineno"]) if region.get("end_lineno") is not None else None
                ),
                parent_region_id=(
                    str(region["parent_region_id"])
                    if region.get("parent_region_id") is not None
                    else None
                ),
                child_region_ids=tuple(str(item) for item in region.get("child_region_ids", [])),
                branch_ids=tuple(int(item) for item in region.get("branch_ids", [])),
                logic_point_names=tuple(str(item) for item in region.get("logic_point_names", [])),
                notes=tuple(str(item) for item in region.get("notes", [])),
            )
            for region in payload.get("regions", [])
        ),
        notes=tuple(str(item) for item in payload.get("notes", [])),
    )


def _emission_dag_to_jsonable(emission_dag: EmissionDAG | None) -> dict | None:
    if emission_dag is None:
        return None
    return {
        "scope_id": emission_dag.scope_id,
        "nodes": [
            {
                "emission_id": node.emission_id,
                "stub_id": node.stub_id,
                "emission_signature": node.emission_signature,
                "site_signature": node.site_signature,
                "structure_signature": node.structure_signature,
                "block_signature": node.block_signature,
                "neighborhood_signature": node.neighborhood_signature,
                "callee_name": node.callee_name,
                "source_path": node.source_path,
                "lineno": node.lineno,
                "branch_ids": list(node.branch_ids),
                "notes": list(node.notes),
            }
            for node in emission_dag.nodes
        ],
        "edges": [
            {
                "src_emission_id": edge.src_emission_id,
                "dst_emission_id": edge.dst_emission_id,
                "rationale": list(edge.rationale),
            }
            for edge in emission_dag.edges
        ],
        "notes": list(emission_dag.notes),
    }


def _emission_dag_from_jsonable(payload: dict | None) -> EmissionDAG | None:
    if not isinstance(payload, dict):
        return None
    return EmissionDAG(
        scope_id=str(payload["scope_id"]),
        nodes=tuple(
            EmissionNode(
                emission_id=str(node["emission_id"]),
                stub_id=str(node["stub_id"]),
                emission_signature=str(node.get("emission_signature", "")),
                site_signature=str(node.get("site_signature", "")),
                structure_signature=str(node.get("structure_signature", "")),
                block_signature=str(node.get("block_signature", "")),
                neighborhood_signature=str(node.get("neighborhood_signature", "")),
                callee_name=str(node["callee_name"]),
                source_path=str(node["source_path"]),
                lineno=int(node["lineno"]),
                branch_ids=tuple(int(item) for item in node.get("branch_ids", [])),
                notes=tuple(str(item) for item in node.get("notes", [])),
            )
            for node in payload.get("nodes", [])
        ),
        edges=tuple(
            EmissionEdge(
                src_emission_id=str(edge["src_emission_id"]),
                dst_emission_id=str(edge["dst_emission_id"]),
                rationale=tuple(str(item) for item in edge.get("rationale", [])),
            )
            for edge in payload.get("edges", [])
        ),
        notes=tuple(str(item) for item in payload.get("notes", [])),
    )


def _logic_emission_map_to_jsonable(logic_emission_map: LogicEmissionMap | None) -> dict | None:
    if logic_emission_map is None:
        return None
    return {
        "scope_id": logic_emission_map.scope_id,
        "links": [
            {
                "logic_point_name": link.logic_point_name,
                "emission_id": link.emission_id,
                "rationale": list(link.rationale),
            }
            for link in logic_emission_map.links
        ],
        "notes": list(logic_emission_map.notes),
    }


def _logic_emission_map_from_jsonable(payload: dict | None) -> LogicEmissionMap | None:
    if not isinstance(payload, dict):
        return None
    return LogicEmissionMap(
        scope_id=str(payload["scope_id"]),
        links=tuple(
            LogicEmissionLink(
                logic_point_name=str(link["logic_point_name"]),
                emission_id=str(link["emission_id"]),
                rationale=tuple(str(item) for item in link.get("rationale", [])),
            )
            for link in payload.get("links", [])
        ),
        notes=tuple(str(item) for item in payload.get("notes", [])),
    )


def _logic_point_ownership_map_to_jsonable(
    logic_point_ownership_map: LogicPointOwnershipMap | None,
) -> dict | None:
    if logic_point_ownership_map is None:
        return None
    return {
        "scope_id": logic_point_ownership_map.scope_id,
        "ownerships": [
            {
                "point_name": ownership.point_name,
                "owning_region_ids": list(ownership.owning_region_ids),
                "owning_cfg_block_ids": list(ownership.owning_cfg_block_ids),
                "rationale": list(ownership.rationale),
            }
            for ownership in logic_point_ownership_map.ownerships
        ],
        "notes": list(logic_point_ownership_map.notes),
    }


def _logic_point_ownership_map_from_jsonable(payload: dict | None) -> LogicPointOwnershipMap | None:
    if not isinstance(payload, dict):
        return None
    return LogicPointOwnershipMap(
        scope_id=str(payload["scope_id"]),
        ownerships=tuple(
            LogicPointOwnership(
                point_name=str(ownership["point_name"]),
                owning_region_ids=tuple(str(item) for item in ownership.get("owning_region_ids", [])),
                owning_cfg_block_ids=tuple(
                    str(item) for item in ownership.get("owning_cfg_block_ids", [])
                ),
                rationale=tuple(str(item) for item in ownership.get("rationale", [])),
            )
            for ownership in payload.get("ownerships", [])
        ),
        notes=tuple(str(item) for item in payload.get("notes", [])),
    )


def _emission_ownership_map_to_jsonable(
    emission_ownership_map: EmissionOwnershipMap | None,
) -> dict | None:
    if emission_ownership_map is None:
        return None
    return {
        "scope_id": emission_ownership_map.scope_id,
        "ownerships": [
            {
                "emission_id": ownership.emission_id,
                "owning_region_ids": list(ownership.owning_region_ids),
                "owning_cfg_block_ids": list(ownership.owning_cfg_block_ids),
                "rationale": list(ownership.rationale),
            }
            for ownership in emission_ownership_map.ownerships
        ],
        "notes": list(emission_ownership_map.notes),
    }


def _emission_ownership_map_from_jsonable(payload: dict | None) -> EmissionOwnershipMap | None:
    if not isinstance(payload, dict):
        return None
    return EmissionOwnershipMap(
        scope_id=str(payload["scope_id"]),
        ownerships=tuple(
            EmissionOwnership(
                emission_id=str(ownership["emission_id"]),
                owning_region_ids=tuple(str(item) for item in ownership.get("owning_region_ids", [])),
                owning_cfg_block_ids=tuple(
                    str(item) for item in ownership.get("owning_cfg_block_ids", [])
                ),
                rationale=tuple(str(item) for item in ownership.get("rationale", [])),
            )
            for ownership in payload.get("ownerships", [])
        ),
        notes=tuple(str(item) for item in payload.get("notes", [])),
    )


def _logic_state_store_to_jsonable(logic_state_store: LogicStateStore | None) -> dict | None:
    if logic_state_store is None:
        return None
    return {
        "scope_id": logic_state_store.scope_id,
        "slices": [
            {
                "slice_id": slice_state.slice_id,
                "logic_point_names": list(slice_state.logic_point_names),
                "entry_branch_ids": list(slice_state.entry_branch_ids),
                "entry_rank_groups": [list(group) for group in slice_state.entry_rank_groups],
                "emission_ids": list(slice_state.emission_ids),
                "owned_region_ids": list(slice_state.owned_region_ids),
                "owned_cfg_block_ids": list(slice_state.owned_cfg_block_ids),
                "next_slice_ids": list(slice_state.next_slice_ids),
                "checkpointable": slice_state.checkpointable,
                "notes": list(slice_state.notes),
            }
            for slice_state in logic_state_store.slices
        ],
        "notes": list(logic_state_store.notes),
    }


def _logic_state_store_from_jsonable(payload: dict | None) -> LogicStateStore | None:
    if not isinstance(payload, dict):
        return None
    return LogicStateStore(
        scope_id=str(payload["scope_id"]),
        slices=tuple(
            LogicSliceState(
                slice_id=str(slice_state["slice_id"]),
                logic_point_names=tuple(str(item) for item in slice_state.get("logic_point_names", [])),
                entry_branch_ids=tuple(int(item) for item in slice_state.get("entry_branch_ids", [])),
                entry_rank_groups=tuple(
                    tuple(int(rank) for rank in group)
                    for group in slice_state.get("entry_rank_groups", [])
                ),
                emission_ids=tuple(str(item) for item in slice_state.get("emission_ids", [])),
                owned_region_ids=tuple(str(item) for item in slice_state.get("owned_region_ids", [])),
                owned_cfg_block_ids=tuple(
                    str(item) for item in slice_state.get("owned_cfg_block_ids", [])
                ),
                next_slice_ids=tuple(str(item) for item in slice_state.get("next_slice_ids", [])),
                checkpointable=bool(slice_state.get("checkpointable", True)),
                notes=tuple(str(item) for item in slice_state.get("notes", [])),
            )
            for slice_state in payload.get("slices", [])
        ),
        notes=tuple(str(item) for item in payload.get("notes", [])),
    )


def _logic_slice_graph_to_jsonable(logic_slice_graph: LogicSliceGraph | None) -> dict | None:
    if logic_slice_graph is None:
        return None
    return {
        "scope_id": logic_slice_graph.scope_id,
        "slice_ids": list(logic_slice_graph.slice_ids),
        "edges": [
            {
                "src_slice_id": edge.src_slice_id,
                "dst_slice_id": edge.dst_slice_id,
                "kind": edge.kind,
                "rationale": list(edge.rationale),
            }
            for edge in logic_slice_graph.edges
        ],
        "notes": list(logic_slice_graph.notes),
    }


def _logic_slice_graph_from_jsonable(payload: dict | None) -> LogicSliceGraph | None:
    if not isinstance(payload, dict):
        return None
    return LogicSliceGraph(
        scope_id=str(payload["scope_id"]),
        slice_ids=tuple(str(item) for item in payload.get("slice_ids", [])),
        edges=tuple(
            LogicSliceEdge(
                src_slice_id=str(edge["src_slice_id"]),
                dst_slice_id=str(edge["dst_slice_id"]),
                kind=str(edge.get("kind", "normal")),
                rationale=tuple(str(item) for item in edge.get("rationale", [])),
            )
            for edge in payload.get("edges", [])
        ),
        notes=tuple(str(item) for item in payload.get("notes", [])),
    )


def _emission_operator_map_to_jsonable(emission_operator_map: EmissionOperatorMap | None) -> dict | None:
    if emission_operator_map is None:
        return None
    return {
        "scope_id": emission_operator_map.scope_id,
        "links": [
            {
                "emission_id": link.emission_id,
                "stub_id": link.stub_id,
                "emission_signature": link.emission_signature,
                "site_signature": link.site_signature,
                "structure_signature": link.structure_signature,
                "block_signature": link.block_signature,
                "neighborhood_signature": link.neighborhood_signature,
                "callee_name": link.callee_name,
                "boundary_kind": link.boundary_kind,
                "branch_ids": list(link.branch_ids),
                "touched_region_ids": list(link.touched_region_ids),
                "dependency_type": link.dependency_type.value,
                "provenance_kind": link.provenance_kind.value,
                "evidence_rows": [
                    {
                        "stub_id": row.stub_id,
                        "callee_name": row.callee_name,
                        "src_region_id": row.src_region_id,
                        "dst_region_id": row.dst_region_id,
                        "evidence_kind": row.evidence_kind.value,
                        "delta_kind": row.delta_kind.value,
                        "confidence": row.confidence,
                        "branch_ids": list(row.branch_ids),
                        "rationale": list(row.rationale),
                    }
                    for row in link.evidence_rows
                ],
                "rationale": list(link.rationale),
            }
            for link in emission_operator_map.links
        ],
        "notes": list(emission_operator_map.notes),
    }


def _emission_operator_map_from_jsonable(payload: dict | None) -> EmissionOperatorMap | None:
    if not isinstance(payload, dict):
        return None
    from .operator_evidence import (
        EmissionOperatorProvenanceKind,
        OperatorEdgeEvidence,
        OperatorEvidenceKind,
    )
    from .schema import DeltaKind

    return EmissionOperatorMap(
        scope_id=str(payload["scope_id"]),
        links=tuple(
            EmissionOperatorLink(
                emission_id=str(link["emission_id"]),
                stub_id=str(link["stub_id"]),
                emission_signature=str(link.get("emission_signature", "")),
                site_signature=str(link.get("site_signature", "")),
                structure_signature=str(link.get("structure_signature", "")),
                block_signature=str(link.get("block_signature", "")),
                neighborhood_signature=str(link.get("neighborhood_signature", "")),
                callee_name=str(link["callee_name"]),
                boundary_kind=str(link.get("boundary_kind", "opaque_call")),
                branch_ids=tuple(int(item) for item in link.get("branch_ids", [])),
                touched_region_ids=tuple(str(item) for item in link.get("touched_region_ids", [])),
                dependency_type=DependencyType(str(link.get("dependency_type", DependencyType.NARROW.value))),
                provenance_kind=EmissionOperatorProvenanceKind(
                    str(
                        link.get(
                            "provenance_kind",
                            EmissionOperatorProvenanceKind.DIRECT_EVIDENCE_MATCH.value,
                        )
                    )
                ),
                evidence_rows=tuple(
                    OperatorEdgeEvidence(
                        stub_id=str(row["stub_id"]),
                        callee_name=str(row["callee_name"]),
                        src_region_id=str(row["src_region_id"]),
                        dst_region_id=str(row["dst_region_id"]),
                        evidence_kind=OperatorEvidenceKind(str(row["evidence_kind"])),
                        delta_kind=DeltaKind(str(row["delta_kind"])),
                        confidence=float(row.get("confidence", 1.0)),
                        branch_ids=tuple(int(item) for item in row.get("branch_ids", [])),
                        rationale=tuple(str(item) for item in row.get("rationale", [])),
                    )
                    for row in link.get("evidence_rows", [])
                ),
                rationale=tuple(str(item) for item in link.get("rationale", [])),
            )
            for link in payload.get("links", [])
        ),
        notes=tuple(str(item) for item in payload.get("notes", [])),
    )


def _witness_region_to_jsonable(region: WitnessRegion) -> dict:
    return {
        "region_id": region.region_id,
        "region_kind": region.region_kind.value,
        "order_index": region.order_index,
        "timing_share": region.timing_share,
        "start_window": region.start_window,
        "end_window": region.end_window,
        "collective_group": region.collective_group,
        "stream_label": region.stream_label,
        "notes": list(region.notes),
        "witness": (
            {
                "source_kind": region.witness.source_kind.value,
                "confidence": region.witness.confidence,
                "evidence": list(region.witness.evidence),
                "rationale": list(region.witness.rationale),
            }
            if region.witness is not None
            else None
        ),
        "provenance": list(region.provenance),
        "value_sensitive": region.value_sensitive,
        "hardware_sensitive": region.hardware_sensitive,
        "criticality_slack": region.criticality_slack,
        "dependency_type": region.dependency_type.value,
    }


def _witness_region_from_jsonable(payload: dict) -> WitnessRegion:
    witness_payload = payload.get("witness")
    witness = None
    if isinstance(witness_payload, dict):
        witness = Witness(
            source_kind=WitnessSourceKind(str(witness_payload["source_kind"])),
            confidence=float(witness_payload.get("confidence", 1.0)),
            evidence=tuple(str(item) for item in witness_payload.get("evidence", [])),
            rationale=tuple(str(item) for item in witness_payload.get("rationale", [])),
        )
    return AnchorRegion(
        region_id=str(payload["region_id"]),
        region_kind=RegionKind(str(payload["region_kind"])),
        order_index=int(payload["order_index"]),
        timing_share=float(payload["timing_share"]),
        start_window=payload.get("start_window"),
        end_window=payload.get("end_window"),
        collective_group=payload.get("collective_group"),
        stream_label=payload.get("stream_label"),
        notes=tuple(str(item) for item in payload.get("notes", [])),
        witness=witness,
        provenance=tuple(str(item) for item in payload.get("provenance", [])),
        value_sensitive=bool(payload.get("value_sensitive", False)),
        hardware_sensitive=bool(payload.get("hardware_sensitive", False)),
        criticality_slack=payload.get("criticality_slack"),
        dependency_type=DependencyType(str(payload.get("dependency_type", DependencyType.NARROW.value))),
    )


def _segment_to_jsonable(segment: ReplaySegment) -> dict:
    return {
        "segment_id": segment.segment_id,
        "segment_kind": segment.segment_kind,
        "start_ts": segment.start_ts,
        "end_ts": segment.end_ts,
        "start_window": segment.start_window,
        "end_window": segment.end_window,
        "event_count": segment.event_count,
        "ranks": list(segment.ranks),
        "collective_group_id": segment.collective_group_id,
        "event_ids": list(segment.event_ids),
    }


def _segment_from_jsonable(payload: dict) -> ReplaySegment:
    return ReplaySegment(
        segment_id=str(payload["segment_id"]),
        segment_kind=str(payload["segment_kind"]),
        start_ts=int(payload["start_ts"]),
        end_ts=int(payload["end_ts"]),
        start_window=int(payload["start_window"]),
        end_window=int(payload["end_window"]),
        event_count=int(payload["event_count"]),
        ranks=tuple(int(item) for item in payload.get("ranks", [])),
        collective_group_id=payload.get("collective_group_id"),
        event_ids=tuple(str(item) for item in payload.get("event_ids", [])),
    )


def anchor_lineage_to_jsonable(lineage: AnchorLineage) -> dict:
    return {
        "anchor_trace_dir": lineage.anchor_trace_dir,
        "segment_max_events_per_rank": lineage.segment_max_events_per_rank,
        "replay_max_events_per_rank": lineage.replay_max_events_per_rank,
        "program_logic_carrier": _program_logic_carrier_to_jsonable(lineage.program_logic_carrier),
        "boundary_capsules": _boundary_capsules_to_jsonable(lineage.boundary_capsules),
        "control_region_tree": _control_region_tree_to_jsonable(lineage.control_region_tree),
        "emission_dag": _emission_dag_to_jsonable(lineage.emission_dag),
        "logic_emission_map": _logic_emission_map_to_jsonable(lineage.logic_emission_map),
        "emission_ownership_map": _emission_ownership_map_to_jsonable(
            lineage.emission_ownership_map
        ),
        "logic_point_ownership_map": _logic_point_ownership_map_to_jsonable(
            lineage.logic_point_ownership_map
        ),
        "logic_state_store": _logic_state_store_to_jsonable(lineage.logic_state_store),
        "logic_slice_graph": _logic_slice_graph_to_jsonable(lineage.logic_slice_graph),
        "emission_operator_map": _emission_operator_map_to_jsonable(lineage.emission_operator_map),
        "witness": {
            "anchor_candidate_id": lineage.witness.anchor_candidate_id,
            "workload_family": lineage.witness.workload_family,
            "world_size": lineage.witness.world_size,
            "granularity": lineage.witness.granularity.value,
            "artifacts": list(lineage.witness.artifacts),
            "notes": list(lineage.witness.notes),
            "regions": [_witness_region_to_jsonable(region) for region in lineage.witness.regions],
        },
        "segment_bundle": {
            "trace_dir": lineage.segment_bundle.trace_dir,
            "window_count": lineage.segment_bundle.window_count,
            "segments": [_segment_to_jsonable(segment) for segment in lineage.segment_bundle.segments],
        },
        "region_slices": [
            {
                "region_id": item.region_id,
                "start_window": item.start_window,
                "end_window": item.end_window,
                "segment_ids": list(item.segment_ids),
                "segment_fraction": item.segment_fraction,
            }
            for item in lineage.region_slices
        ],
    }


def anchor_lineage_from_jsonable(payload: dict) -> AnchorLineage:
    witness_payload = payload["witness"]
    witness = AnchorWitness(
        anchor_candidate_id=str(witness_payload["anchor_candidate_id"]),
        workload_family=str(witness_payload["workload_family"]),
        world_size=int(witness_payload["world_size"]),
        granularity=WitnessGranularity(str(witness_payload["granularity"])),
        regions=tuple(
            _witness_region_from_jsonable(region_payload)
            for region_payload in witness_payload.get("regions", [])
        ),
        artifacts=tuple(str(item) for item in witness_payload.get("artifacts", [])),
        notes=tuple(str(item) for item in witness_payload.get("notes", [])),
    )
    segment_bundle_payload = payload["segment_bundle"]
    segment_bundle = ReplaySegmentBundle(
        trace_dir=str(segment_bundle_payload["trace_dir"]),
        window_count=int(segment_bundle_payload["window_count"]),
        segments=tuple(
            _segment_from_jsonable(segment_payload)
            for segment_payload in segment_bundle_payload.get("segments", [])
        ),
        collated=None,
    )
    region_slices = tuple(
        RegionSegmentSlice(
            region_id=str(item["region_id"]),
            start_window=int(item["start_window"]),
            end_window=int(item["end_window"]),
            segment_ids=tuple(str(seg_id) for seg_id in item.get("segment_ids", [])),
            segment_fraction=float(item["segment_fraction"]),
        )
        for item in payload.get("region_slices", [])
    )
    return AnchorLineage(
        witness=witness,
        anchor_trace_dir=str(payload["anchor_trace_dir"]),
        segment_max_events_per_rank=payload.get("segment_max_events_per_rank"),
        replay_max_events_per_rank=payload.get("replay_max_events_per_rank"),
        segment_bundle=segment_bundle,
        region_slices=region_slices,
        program_logic_carrier=_program_logic_carrier_from_jsonable(payload.get("program_logic_carrier")),
        boundary_capsules=_boundary_capsules_from_jsonable(payload.get("boundary_capsules")),
        control_region_tree=_control_region_tree_from_jsonable(payload.get("control_region_tree")),
        emission_dag=_emission_dag_from_jsonable(payload.get("emission_dag")),
        logic_emission_map=_logic_emission_map_from_jsonable(payload.get("logic_emission_map")),
        emission_ownership_map=_emission_ownership_map_from_jsonable(
            payload.get("emission_ownership_map")
        ),
        logic_point_ownership_map=_logic_point_ownership_map_from_jsonable(
            payload.get("logic_point_ownership_map")
        ),
        logic_state_store=_logic_state_store_from_jsonable(payload.get("logic_state_store")),
        logic_slice_graph=_logic_slice_graph_from_jsonable(payload.get("logic_slice_graph")),
        emission_operator_map=_emission_operator_map_from_jsonable(payload.get("emission_operator_map")),
    )


def save_anchor_lineage(lineage: AnchorLineage, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(anchor_lineage_to_jsonable(lineage), indent=2), encoding="utf-8")
    return path


def load_anchor_lineage(path: str | Path) -> AnchorLineage:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return anchor_lineage_from_jsonable(payload)


def save_anchor_lineage_binary(lineage: AnchorLineage, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = anchor_lineage_to_jsonable(lineage)
    with gzip.open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_anchor_lineage_binary(path: str | Path) -> AnchorLineage:
    with gzip.open(Path(path), "rb") as handle:
        payload = pickle.load(handle)
    return anchor_lineage_from_jsonable(payload)


def save_anchor_lineage_pickle(lineage: AnchorLineage, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = anchor_lineage_to_jsonable(lineage)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_anchor_lineage_pickle(path: str | Path) -> AnchorLineage:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    return anchor_lineage_from_jsonable(payload)
