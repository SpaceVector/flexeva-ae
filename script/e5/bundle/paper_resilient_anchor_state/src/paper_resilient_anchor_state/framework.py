from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from .state import (
    AnchorCodeState,
    AnchorRuntimeValueState,
    AnchorSemanticState,
    AnchorTraceState,
    BoundaryContextCapsule,
    DryRunProgramLogicCapture,
    LogicScopeSpec,
    ProgramLogicCarrier,
    ProgramLogicKind,
    RuntimeValueDistribution,
    RuntimeValuePoint,
    RuntimeValuePointKind,
    WorkloadSemanticContract,
    build_anchor_code_state,
    build_anchor_runtime_value_state,
    build_anchor_semantic_state,
    build_anchor_trace_state,
    default_contract_for_workload_family,
)


@dataclass(frozen=True)
class AnchorWindowAnalysis:
    anchor_candidate_id: str
    workload_family: str
    code: AnchorCodeState
    semantic: AnchorSemanticState
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnchorGroundingTarget:
    point_name: str
    point_kind: RuntimeValuePointKind
    required_fields: tuple[str, ...]
    producer_logic_points: tuple[str, ...]
    source_capsule_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticDedupGroup:
    group_id: str
    representative_rank: int
    member_ranks: tuple[int, ...]
    source_logic_points: tuple[str, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnchorGroundingState:
    workload_family: str
    semantic_scope_id: str | None
    logical_world_size: int
    contract: WorkloadSemanticContract | None
    targets: tuple[AnchorGroundingTarget, ...]
    semantic_dedup_groups: tuple[SemanticDedupGroup, ...]
    representative_ranks: tuple[int, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class OptimizedMayaRequest:
    anchor_candidate_id: str
    workload_family: str
    logical_world_size: int
    representative_ranks: tuple[int, ...]
    semantic_dedup_groups: tuple[SemanticDedupGroup, ...]
    trace_window_source: str
    lane_projection: str
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class OptimizedMayaArtifact:
    request: OptimizedMayaRequest
    trace: AnchorTraceState
    notes: tuple[str, ...] = ()


def analyze_user_window(
    *,
    anchor_candidate_id: str,
    workload_family: str,
    anchor_code_paths: Iterable[str] = (),
    code_mutation_pairs: Iterable[tuple[str, str]] = (),
    anchor_capture: DryRunProgramLogicCapture | None = None,
    anchor_program_logic: ProgramLogicCarrier | None = None,
    candidate_capture: DryRunProgramLogicCapture | None = None,
) -> AnchorWindowAnalysis:
    code_state = build_anchor_code_state(
        anchor_code_paths=anchor_code_paths,
        code_mutation_pairs=code_mutation_pairs,
    )
    semantic_state = build_anchor_semantic_state(
        anchor_capture=anchor_capture,
        anchor_program_logic=anchor_program_logic,
        candidate_capture=candidate_capture,
    )
    notes = (
        "window analysis is the user-acquired source stage",
        "outputs AnchorCodeState and AnchorSemanticState only",
    )
    return AnchorWindowAnalysis(
        anchor_candidate_id=anchor_candidate_id,
        workload_family=workload_family,
        code=code_state,
        semantic=semantic_state,
        notes=notes,
    )


def _logic_field_map(semantic_state: AnchorSemanticState) -> dict[str, set[str]]:
    rows: dict[str, set[str]] = {}
    for point in semantic_state.program_logic_carrier.points:
        rows[point.name] = set(point.source_fields)
    return rows


def _capsule_map(semantic_state: AnchorSemanticState) -> dict[str, BoundaryContextCapsule]:
    return {capsule.capsule_id: capsule for capsule in semantic_state.boundary_capsules}


def _resolve_target_from_contract_point(
    contract_point: RuntimeValuePoint,
    semantic_state: AnchorSemanticState,
) -> AnchorGroundingTarget:
    logic_field_map = _logic_field_map(semantic_state)
    producer_logic_points = tuple(
        name
        for name, source_fields in sorted(logic_field_map.items())
        if set(contract_point.required_fields) & source_fields
    )
    source_capsule_ids = tuple(
        capsule.capsule_id
        for capsule in semantic_state.boundary_capsules
        if capsule.callee_name == contract_point.name
        or capsule.site_signature.endswith(contract_point.name)
        or any(field in capsule.keyword_arg_names for field in contract_point.required_fields)
    )
    notes: list[str] = []
    if producer_logic_points:
        notes.append("producer logic points resolved from overlapping semantic source fields")
    if source_capsule_ids:
        notes.append("grounding target is attached to black-box boundary capsules")
    if not producer_logic_points and contract_point.required_fields:
        notes.append("required fields remain contract-only and need actual grounding execution")
    return AnchorGroundingTarget(
        point_name=contract_point.name,
        point_kind=contract_point.kind,
        required_fields=contract_point.required_fields,
        producer_logic_points=producer_logic_points,
        source_capsule_ids=source_capsule_ids,
        notes=tuple(notes),
    )


def _resolve_semantic_dedup_groups(
    semantic_state: AnchorSemanticState,
    logical_world_size: int,
) -> tuple[SemanticDedupGroup, ...]:
    groups: list[SemanticDedupGroup] = []
    for point in semantic_state.program_logic_carrier.points:
        if point.kind != ProgramLogicKind.RANK_PARTITION:
            continue
        if not isinstance(point.value, (list, tuple)):
            continue
        partitions = []
        for item in point.value:
            if isinstance(item, (list, tuple)) and item:
                try:
                    partitions.append(tuple(int(rank) for rank in item))
                except (TypeError, ValueError):
                    continue
        for index, members in enumerate(partitions):
            if not members:
                continue
            groups.append(
                SemanticDedupGroup(
                    group_id=f"{point.name}::group_{index}",
                    representative_rank=min(members),
                    member_ranks=members,
                    source_logic_points=(point.name,),
                    notes=("semantic rank partition resolved before optimized Maya dryrun",),
                )
            )
    if groups:
        return tuple(groups)
    return (
        SemanticDedupGroup(
            group_id="default::all_ranks",
            representative_rank=0,
            member_ranks=tuple(range(logical_world_size)),
            source_logic_points=(),
            notes=("no semantic rank partition was attached; fall back to full logical world",),
        ),
    )


def resolve_anchor_grounding_state(
    *,
    workload_family: str,
    semantic_state: AnchorSemanticState,
    logical_world_size: int,
    contract: WorkloadSemanticContract | None = None,
) -> AnchorGroundingState:
    resolved_contract = contract or default_contract_for_workload_family(workload_family)
    targets = ()
    notes: list[str] = []
    if resolved_contract is not None:
        targets = tuple(
            _resolve_target_from_contract_point(point, semantic_state)
            for point in resolved_contract.runtime_value_points
        )
        notes.append("grounding targets are resolved from semantic state plus workload contract")
    else:
        notes.append("no workload contract was found; grounding has no explicit runtime targets")
    dedup_groups = _resolve_semantic_dedup_groups(semantic_state, logical_world_size)
    representative_ranks = tuple(group.representative_rank for group in dedup_groups)
    notes.append("semantic dedup groups are resolved before optimized Maya dryrun")
    return AnchorGroundingState(
        workload_family=workload_family,
        semantic_scope_id=None if semantic_state.logic_scope is None else semantic_state.logic_scope.scope_id,
        logical_world_size=logical_world_size,
        contract=resolved_contract,
        targets=targets,
        semantic_dedup_groups=dedup_groups,
        representative_ranks=representative_ranks,
        notes=tuple(notes),
    )


def _filter_runtime_distributions(
    distributions: tuple[RuntimeValueDistribution, ...],
    grounding_state: AnchorGroundingState,
) -> tuple[RuntimeValueDistribution, ...]:
    allowed = {target.point_name for target in grounding_state.targets}
    return tuple(item for item in distributions if item.point_name in allowed)


def materialize_anchor_runtime_values(
    *,
    workload_family: str,
    grounding_state: AnchorGroundingState,
    anchor_capture: DryRunProgramLogicCapture | None = None,
    candidate_capture: DryRunProgramLogicCapture | None = None,
) -> AnchorRuntimeValueState:
    runtime_state = build_anchor_runtime_value_state(
        workload_family=workload_family,
        anchor_capture=anchor_capture,
        candidate_capture=candidate_capture,
    )
    if not grounding_state.targets:
        return replace(
            runtime_state,
            notes=runtime_state.notes + ("no grounding targets were resolved; runtime values remain unfiltered",),
        )
    filtered = _filter_runtime_distributions(runtime_state.distributions, grounding_state)
    return replace(
        runtime_state,
        distributions=filtered,
        notes=runtime_state.notes
        + (
            "runtime values are filtered by AnchorGroundingState targets",
            f"representative_ranks={','.join(str(rank) for rank in grounding_state.representative_ranks)}",
        ),
    )


def build_optimized_maya_request(
    *,
    anchor_candidate_id: str,
    workload_family: str,
    grounding_state: AnchorGroundingState,
    trace_window_source: str = "workload_heuristic",
    lane_projection: str = "rank/cpu-helper/gpu-stream",
) -> OptimizedMayaRequest:
    return OptimizedMayaRequest(
        anchor_candidate_id=anchor_candidate_id,
        workload_family=workload_family,
        logical_world_size=grounding_state.logical_world_size,
        representative_ranks=grounding_state.representative_ranks,
        semantic_dedup_groups=grounding_state.semantic_dedup_groups,
        trace_window_source=trace_window_source,
        lane_projection=lane_projection,
        notes=(
            "optimized Maya dryrun uses semantic dedup before trace generation",
            "request is paper-facing and assumes projected host/main/helper/stream lanes",
        ),
    )


def materialize_anchor_trace_from_optimized_maya(
    *,
    request: OptimizedMayaRequest,
    trace_dir: str,
    witness=None,
    candidate_trace_dir: str | None = None,
) -> OptimizedMayaArtifact:
    trace_state = build_anchor_trace_state(
        trace_dir=trace_dir,
        anchor_candidate_id=request.anchor_candidate_id,
        workload_family=request.workload_family,
        witness=witness,
        candidate_trace_dir=candidate_trace_dir,
    )
    return OptimizedMayaArtifact(
        request=request,
        trace=trace_state,
        notes=(
            "AnchorTraceState is produced after semantic dedup and optimized Maya dryrun",
            f"trace_window_source={request.trace_window_source}",
            f"lane_projection={request.lane_projection}",
        ),
    )
