"""Thin bridge from pyextend dry-run into program-logic carriers."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from pyextend.runtime import DryRunExecutor, SimEnv

from .control_flow import build_control_flow_graphs
from .control_regions import ControlRegion, ControlRegionKind, ControlRegionTree, build_control_region_tree
from .program_logic import ProgramLogicCarrier, build_program_logic_carrier_from_executor
from .rank_reduction import RankReductionRequirementKind
from .source_frontend import defined_function_names_for_path, is_cpp_like_path, iter_cpp_opaque_calls


@dataclass(frozen=True)
class LogicScopeSpec:
    scope_id: str
    selected_paths: tuple[str, ...]
    selected_functions: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BlackBoxBoundaryRule:
    opaque_call_names: tuple[str, ...] = ()
    opaque_module_prefixes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperatorStub:
    stub_id: str
    emission_signature: str
    site_signature: str
    callee_name: str
    source_path: str
    lineno: int
    structure_signature: str = ""
    block_signature: str = ""
    neighborhood_signature: str = ""
    boundary_kind: str = "opaque_call"
    branch_ids: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AbstractValueSummary:
    value_kind: str
    shape_hint: tuple[str, ...] = ()
    dtype_hint: str | None = None
    device_hint: str | None = None
    scalar_hint: str | int | float | bool | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BlackBoxReturnSummary:
    return_kind: str
    abstract_values: tuple[AbstractValueSummary, ...] = ()
    logic_observable: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SideEffectSummary:
    emits_operator_stub_ids: tuple[str, ...] = ()
    mutates_state: bool = False
    advances_group_state: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BoundaryContract:
    callee_name: str
    boundary_kind: str
    branch_ids: tuple[int, ...] = ()
    positional_arg_kinds: tuple[str, ...] = ()
    keyword_arg_names: tuple[str, ...] = ()
    return_kind: str = "none"
    logic_observable: bool = False
    mutates_state: bool = False
    advances_group_state: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BoundaryContractComparison:
    compatible: bool
    changed_fields: tuple[str, ...] = ()
    required_preservation_requirements: tuple[RankReductionRequirementKind, ...] = ()
    rationale: tuple[str, ...] = ()


@dataclass(frozen=True)
class BoundaryContextCapsule:
    capsule_id: str
    stub_id: str
    emission_signature: str
    site_signature: str
    callee_name: str
    source_path: str
    lineno: int
    boundary_kind: str
    branch_ids: tuple[int, ...] = ()
    positional_arg_kinds: tuple[str, ...] = ()
    keyword_arg_names: tuple[str, ...] = ()
    structure_signature: str = ""
    block_signature: str = ""
    return_summary: BlackBoxReturnSummary | None = None
    side_effect_summary: SideEffectSummary | None = None
    notes: tuple[str, ...] = ()

    @property
    def boundary_contract(self) -> BoundaryContract:
        return BoundaryContract(
            callee_name=self.callee_name,
            boundary_kind=self.boundary_kind,
            branch_ids=self.branch_ids,
            positional_arg_kinds=self.positional_arg_kinds,
            keyword_arg_names=self.keyword_arg_names,
            return_kind=self.return_summary.return_kind if self.return_summary is not None else "none",
            logic_observable=(
                self.return_summary.logic_observable if self.return_summary is not None else False
            ),
            mutates_state=(
                self.side_effect_summary.mutates_state if self.side_effect_summary is not None else False
            ),
            advances_group_state=(
                self.side_effect_summary.advances_group_state
                if self.side_effect_summary is not None
                else False
            ),
            notes=self.notes,
        )


@dataclass(frozen=True)
class EmissionNode:
    emission_id: str
    stub_id: str
    emission_signature: str
    site_signature: str
    callee_name: str
    source_path: str
    lineno: int
    structure_signature: str = ""
    block_signature: str = ""
    neighborhood_signature: str = ""
    branch_ids: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class EmissionEdge:
    src_emission_id: str
    dst_emission_id: str
    rationale: tuple[str, ...] = ()


@dataclass(frozen=True)
class EmissionDAG:
    scope_id: str
    nodes: tuple[EmissionNode, ...]
    edges: tuple[EmissionEdge, ...] = ()
    notes: tuple[str, ...] = ()

    def node_by_id(self) -> dict[str, EmissionNode]:
        return {node.emission_id: node for node in self.nodes}


@dataclass(frozen=True)
class LogicEmissionLink:
    logic_point_name: str
    emission_id: str
    rationale: tuple[str, ...] = ()


@dataclass(frozen=True)
class LogicEmissionMap:
    scope_id: str
    links: tuple[LogicEmissionLink, ...]
    notes: tuple[str, ...] = ()

    def links_for_point(self, logic_point_name: str) -> tuple[LogicEmissionLink, ...]:
        return tuple(link for link in self.links if link.logic_point_name == logic_point_name)


@dataclass(frozen=True)
class EmissionOwnership:
    emission_id: str
    owning_region_ids: tuple[str, ...] = ()
    owning_cfg_block_ids: tuple[str, ...] = ()
    rationale: tuple[str, ...] = ()


@dataclass(frozen=True)
class EmissionOwnershipMap:
    scope_id: str
    ownerships: tuple[EmissionOwnership, ...]
    notes: tuple[str, ...] = ()

    def ownership_for_emission(self, emission_id: str) -> EmissionOwnership | None:
        for ownership in self.ownerships:
            if ownership.emission_id == emission_id:
                return ownership
        return None


@dataclass(frozen=True)
class LogicPointOwnership:
    point_name: str
    owning_region_ids: tuple[str, ...] = ()
    owning_cfg_block_ids: tuple[str, ...] = ()
    rationale: tuple[str, ...] = ()


@dataclass(frozen=True)
class LogicPointOwnershipMap:
    scope_id: str
    ownerships: tuple[LogicPointOwnership, ...]
    notes: tuple[str, ...] = ()

    def ownership_for_point(self, logic_point_name: str) -> LogicPointOwnership | None:
        for ownership in self.ownerships:
            if ownership.point_name == logic_point_name:
                return ownership
        return None


class LogicSliceGranularityMode(str, Enum):
    LEGACY_REGION_LEAVES = "legacy_region_leaves"
    CFG_BOUNDARY_AWARE = "cfg_boundary_aware"


@dataclass(frozen=True)
class LogicSliceGranularityPolicy:
    mode: LogicSliceGranularityMode = LogicSliceGranularityMode.CFG_BOUNDARY_AWARE
    max_cfg_blocks_per_slice: int = 3
    split_on_merge: bool = True
    split_on_emission: bool = True
    coalesce_trivial_cfg_slices: bool = True
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LogicSliceState:
    slice_id: str
    logic_point_names: tuple[str, ...]
    entry_branch_ids: tuple[int, ...] = ()
    entry_rank_groups: tuple[tuple[int, ...], ...] = ()
    emission_ids: tuple[str, ...] = ()
    owned_region_ids: tuple[str, ...] = ()
    owned_cfg_block_ids: tuple[str, ...] = ()
    next_slice_ids: tuple[str, ...] = ()
    checkpointable: bool = True
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LogicStateStore:
    scope_id: str
    slices: tuple[LogicSliceState, ...]
    notes: tuple[str, ...] = ()

    def slice_by_id(self) -> dict[str, LogicSliceState]:
        return {item.slice_id: item for item in self.slices}


@dataclass(frozen=True)
class LogicSliceEdge:
    src_slice_id: str
    dst_slice_id: str
    kind: str = "normal"
    rationale: tuple[str, ...] = ()


@dataclass(frozen=True)
class LogicSliceGraph:
    scope_id: str
    slice_ids: tuple[str, ...]
    edges: tuple[LogicSliceEdge, ...]
    notes: tuple[str, ...] = ()

    def successor_map(self) -> dict[str, tuple[str, ...]]:
        buckets: dict[str, list[str]] = {slice_id: [] for slice_id in self.slice_ids}
        for edge in self.edges:
            buckets.setdefault(edge.src_slice_id, []).append(edge.dst_slice_id)
        return {slice_id: tuple(values) for slice_id, values in buckets.items()}

    def predecessor_map(self) -> dict[str, tuple[str, ...]]:
        buckets: dict[str, list[str]] = {slice_id: [] for slice_id in self.slice_ids}
        for edge in self.edges:
            buckets.setdefault(edge.dst_slice_id, []).append(edge.src_slice_id)
        return {slice_id: tuple(values) for slice_id, values in buckets.items()}


@dataclass(frozen=True)
class DryRunProgramLogicCapture:
    code_path: str
    world_size: int
    branch_signatures: dict[int, tuple[tuple[int, bool, bool], ...]]
    semantic_summaries: dict[int, dict[str, dict[str, Any]]]
    program_logic: ProgramLogicCarrier
    logic_scope: LogicScopeSpec
    boundary_rule: BlackBoxBoundaryRule | None = None
    operator_stubs: tuple[OperatorStub, ...] = ()
    boundary_capsules: tuple[BoundaryContextCapsule, ...] = ()
    emission_dag: EmissionDAG | None = None
    logic_emission_map: LogicEmissionMap | None = None
    emission_ownership_map: EmissionOwnershipMap | None = None
    logic_point_ownership_map: LogicPointOwnershipMap | None = None
    logic_state_store: LogicStateStore | None = None
    logic_slice_graph: LogicSliceGraph | None = None
    control_region_tree: ControlRegionTree | None = None

    def signature(self) -> tuple[str, ...]:
        rows = [
            f"code_path={self.code_path}",
            f"world_size={self.world_size}",
            f"logic_scope={self.logic_scope.scope_id}",
        ]
        rows.extend(self.program_logic.signature())
        for stub in self.operator_stubs:
            rows.append(
                "|".join(
                    (
                        stub.stub_id,
                        stub.emission_signature,
                        stub.site_signature,
                        stub.structure_signature,
                        stub.block_signature,
                        stub.neighborhood_signature,
                        stub.callee_name,
                        str(stub.lineno),
                        ",".join(str(item) for item in stub.branch_ids),
                    )
                )
            )
        for capsule in self.boundary_capsules:
            rows.append(
                "|".join(
                    (
                        capsule.capsule_id,
                        capsule.emission_signature,
                        capsule.site_signature,
                        capsule.structure_signature,
                        capsule.block_signature,
                        capsule.callee_name,
                        capsule.return_summary.return_kind if capsule.return_summary is not None else "none",
                        ",".join(capsule.positional_arg_kinds),
                        ",".join(capsule.keyword_arg_names),
                    )
                )
            )
        if self.control_region_tree is not None:
            for region in self.control_region_tree.regions:
                rows.append(
                    "|".join(
                        (
                            region.region_id,
                            region.kind.value,
                            ",".join(region.logic_point_names),
                            ",".join(region.child_region_ids),
                        )
                    )
                )
        return tuple(rows)


def _selected_boundary_roots(
    module: ast.Module,
    selected_functions: tuple[str, ...],
) -> tuple[ast.AST, ...]:
    if not selected_functions:
        return (module,)
    wanted = set(selected_functions)
    selected: list[ast.AST] = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            selected.append(node)
            continue
        if isinstance(node, ast.ClassDef):
            for statement in node.body:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name in wanted:
                    selected.append(statement)
    return tuple(selected)


def _default_logic_scope_for_path(code_path: Path) -> LogicScopeSpec:
    return LogicScopeSpec(
        scope_id=f"file::{code_path.stem}",
        selected_paths=(str(code_path),),
        notes=("default single-file logic scope",),
    )


def _collect_branch_source_metadata(
    selected_paths: tuple[str, ...],
    selected_functions: tuple[str, ...],
) -> dict[int, tuple[str, int, int | None]]:
    metadata: dict[int, tuple[str, int, int | None]] = {}
    for raw_path in selected_paths:
        selected_path = Path(raw_path)
        if is_cpp_like_path(selected_path):
            continue
        module = ast.parse(selected_path.read_text(encoding="utf-8"), filename=str(selected_path))
        for root in _selected_boundary_roots(module, selected_functions):
            for node in ast.walk(root):
                if not isinstance(node, ast.If):
                    continue
                branch_id = _extract_mark_cond_branch_id(node.test)
                if branch_id is None or branch_id in metadata:
                    continue
                metadata[int(branch_id)] = (
                    str(selected_path),
                    int(getattr(node, "lineno", 0)),
                    getattr(node, "end_lineno", None),
                )
    return metadata


def _resolve_call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _extract_mark_cond_branch_id(node: ast.AST) -> int | None:
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Name) or node.func.id != "mark_cond":
        return None
    if len(node.args) < 2:
        return None
    branch_arg = node.args[1]
    if isinstance(branch_arg, ast.Constant) and isinstance(branch_arg.value, int):
        return int(branch_arg.value)
    return None


def _classify_arg_kind(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return "name"
    if isinstance(node, ast.Attribute):
        return "attribute"
    if isinstance(node, ast.Call):
        return "call"
    if isinstance(node, ast.Constant):
        return f"constant:{type(node.value).__name__}"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if isinstance(node, ast.List):
        return "list"
    if isinstance(node, ast.Dict):
        return "dict"
    if isinstance(node, ast.Subscript):
        return "subscript"
    return node.__class__.__name__.lower()


def _make_emission_signature(
    *,
    callee_name: str,
    boundary_kind: str,
    branch_ids: tuple[int, ...],
    positional_arg_kinds: tuple[str, ...],
    keyword_arg_names: tuple[str, ...],
) -> str:
    return "|".join(
        (
            boundary_kind,
            callee_name,
            ",".join(str(item) for item in branch_ids),
            ",".join(positional_arg_kinds),
            ",".join(keyword_arg_names),
        )
    )


def _default_return_summary(call_name: str) -> BlackBoxReturnSummary:
    lowered = call_name.lower()
    if any(token in lowered for token in ("save", "checkpoint", "log", "print")):
        return BlackBoxReturnSummary(
            return_kind="void_like",
            abstract_values=(),
            logic_observable=False,
            notes=("assumed control-hidden side-effect-only helper",),
        )
    if any(token in lowered for token in ("group", "collective", "allreduce", "all_to_all", "alltoall")):
        return BlackBoxReturnSummary(
            return_kind="opaque_group_result",
            abstract_values=(
                AbstractValueSummary(
                    value_kind="opaque_group_token",
                    notes=("collective/group boundary result placeholder",),
                ),
            ),
            logic_observable=False,
            notes=("assumed control-hidden collective helper",),
        )
    return BlackBoxReturnSummary(
        return_kind="opaque_result",
        abstract_values=(
            AbstractValueSummary(
                value_kind="opaque_token",
                notes=("generic black-box return placeholder",),
            ),
        ),
        logic_observable=False,
        notes=("assumed control-hidden black-box return",),
    )


def _default_side_effect_summary(stub_id: str, call_name: str) -> SideEffectSummary:
    lowered = call_name.lower()
    return SideEffectSummary(
        emits_operator_stub_ids=(stub_id,),
        mutates_state=any(token in lowered for token in ("save", "checkpoint", "update")),
        advances_group_state=any(
            token in lowered for token in ("group", "collective", "allreduce", "all_to_all", "alltoall", "barrier")
        ),
        notes=("first-pass boundary side-effect summary",),
    )


def compare_boundary_contracts(
    before: BoundaryContextCapsule | BoundaryContract,
    after: BoundaryContextCapsule | BoundaryContract,
) -> BoundaryContractComparison:
    before_contract = before.boundary_contract if isinstance(before, BoundaryContextCapsule) else before
    after_contract = after.boundary_contract if isinstance(after, BoundaryContextCapsule) else after
    changed_fields: list[str] = []
    field_pairs = (
        ("callee_name", before_contract.callee_name, after_contract.callee_name),
        ("boundary_kind", before_contract.boundary_kind, after_contract.boundary_kind),
        ("branch_ids", before_contract.branch_ids, after_contract.branch_ids),
        (
            "positional_arg_kinds",
            before_contract.positional_arg_kinds,
            after_contract.positional_arg_kinds,
        ),
        (
            "keyword_arg_names",
            before_contract.keyword_arg_names,
            after_contract.keyword_arg_names,
        ),
        ("return_kind", before_contract.return_kind, after_contract.return_kind),
        ("logic_observable", before_contract.logic_observable, after_contract.logic_observable),
        ("mutates_state", before_contract.mutates_state, after_contract.mutates_state),
        (
            "advances_group_state",
            before_contract.advances_group_state,
            after_contract.advances_group_state,
        ),
    )
    for field_name, before_value, after_value in field_pairs:
        if before_value != after_value:
            changed_fields.append(field_name)
    required_requirements: list[RankReductionRequirementKind] = []
    if any(
        field_name in {
            "callee_name",
            "boundary_kind",
            "branch_ids",
            "positional_arg_kinds",
            "keyword_arg_names",
            "return_kind",
            "logic_observable",
            "mutates_state",
        }
        for field_name in changed_fields
    ):
        required_requirements.append(RankReductionRequirementKind.PORTABLE_PROGRAM_LOGIC)
    if "advances_group_state" in changed_fields:
        required_requirements.append(RankReductionRequirementKind.RANK_PARTITION_BEHAVIOR)
    normalized_requirements = tuple(dict.fromkeys(required_requirements))
    if not changed_fields:
        return BoundaryContractComparison(
            compatible=True,
            changed_fields=(),
            required_preservation_requirements=(),
            rationale=("boundary contract compatible",),
        )
    rationale = [f"boundary contract fields changed: {','.join(changed_fields)}"]
    if normalized_requirements:
        rationale.append(
            "boundary contract change requires preservation: "
            + ",".join(item.value for item in normalized_requirements)
        )
    return BoundaryContractComparison(
        compatible=False,
        changed_fields=tuple(changed_fields),
        required_preservation_requirements=normalized_requirements,
        rationale=tuple(rationale),
    )


class _OperatorStubVisitor(ast.NodeVisitor):
    def __init__(self, code_path: Path, boundary_rule: BlackBoxBoundaryRule) -> None:
        self.code_path = code_path
        self.boundary_rule = boundary_rule
        self.branch_stack: list[int] = []
        self.operator_stubs: list[OperatorStub] = []
        self.boundary_capsules: list[BoundaryContextCapsule] = []
        self._opaque_call_names = set(boundary_rule.opaque_call_names)
        self._site_signature_occurrences: dict[str, int] = {}
        self._structure_stack: list[str] = []
        self._block_stack: list[str] = []

    @staticmethod
    def _hash_block_signature(node: ast.AST) -> str:
        dumped = ast.dump(node, include_attributes=False)
        return hashlib.sha1(dumped.encode("utf-8")).hexdigest()[:16]

    def _push_structure(self, label: str, node: ast.AST) -> None:
        self._structure_stack.append(label)
        self._block_stack.append(self._hash_block_signature(node))

    def _pop_structure(self) -> None:
        if self._structure_stack:
            self._structure_stack.pop()
        if self._block_stack:
            self._block_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # noqa: N802
        self._push_structure(f"fn:{node.name}", node)
        for statement in node.body:
            self.visit(statement)
        self._pop_structure()
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:  # noqa: N802
        self._push_structure(f"fn:{node.name}", node)
        for statement in node.body:
            self.visit(statement)
        self._pop_structure()
        return None

    def visit_If(self, node: ast.If) -> Any:  # noqa: N802
        branch_id = _extract_mark_cond_branch_id(node.test)
        if branch_id is None:
            self._push_structure(f"if:{int(getattr(node, 'lineno', 0))}", node)
            for statement in node.body:
                self.visit(statement)
            for statement in node.orelse:
                self.visit(statement)
            self._pop_structure()
            return None
        self._push_structure(f"branch:{branch_id}", node)
        self.branch_stack.append(branch_id)
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)
        self.branch_stack.pop()
        self._pop_structure()
        return None

    def visit_For(self, node: ast.For) -> Any:  # noqa: N802
        self._push_structure("loop:for", node)
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)
        self._pop_structure()
        return None

    def visit_AsyncFor(self, node: ast.AsyncFor) -> Any:  # noqa: N802
        self._push_structure("loop:async_for", node)
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)
        self._pop_structure()
        return None

    def visit_While(self, node: ast.While) -> Any:  # noqa: N802
        self._push_structure("loop:while", node)
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)
        self._pop_structure()
        return None

    def visit_With(self, node: ast.With) -> Any:  # noqa: N802
        self._push_structure("with", node)
        for statement in node.body:
            self.visit(statement)
        self._pop_structure()
        return None

    def visit_AsyncWith(self, node: ast.AsyncWith) -> Any:  # noqa: N802
        self._push_structure("with:async", node)
        for statement in node.body:
            self.visit(statement)
        self._pop_structure()
        return None

    def visit_Try(self, node: ast.Try) -> Any:  # noqa: N802
        self._push_structure("try", node)
        for statement in node.body:
            self.visit(statement)
        for handler in node.handlers:
            for statement in handler.body:
                self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)
        for statement in node.finalbody:
            self.visit(statement)
        self._pop_structure()
        return None

    def visit_Call(self, node: ast.Call) -> Any:  # noqa: N802
        call_name = _resolve_call_name(node)
        if call_name and call_name in self._opaque_call_names:
            positional_arg_kinds = tuple(_classify_arg_kind(arg) for arg in node.args)
            keyword_arg_names = tuple(
                str(keyword.arg)
                for keyword in node.keywords
                if keyword.arg is not None
            )
            emission_signature = _make_emission_signature(
                callee_name=call_name,
                boundary_kind="opaque_call",
                branch_ids=tuple(self.branch_stack),
                positional_arg_kinds=positional_arg_kinds,
                keyword_arg_names=keyword_arg_names,
            )
            site_occurrence = self._site_signature_occurrences.get(emission_signature, 0)
            self._site_signature_occurrences[emission_signature] = site_occurrence + 1
            site_signature = f"{emission_signature}|site_occ={site_occurrence}"
            structure_signature = "|".join(self._structure_stack) if self._structure_stack else "<module>"
            block_signature = self._block_stack[-1] if self._block_stack else "<module>"
            stub = OperatorStub(
                stub_id=f"{call_name}@{getattr(node, 'lineno', 0)}",
                emission_signature=emission_signature,
                site_signature=site_signature,
                callee_name=call_name,
                source_path=str(self.code_path),
                lineno=int(getattr(node, "lineno", 0)),
                structure_signature=structure_signature,
                block_signature=block_signature,
                boundary_kind="opaque_call",
                branch_ids=tuple(self.branch_stack),
                notes=("captured at black-box boundary from instrumented code",),
            )
            self.operator_stubs.append(stub)
            self.boundary_capsules.append(
                BoundaryContextCapsule(
                    capsule_id=f"capsule::{stub.stub_id}",
                    stub_id=stub.stub_id,
                    emission_signature=stub.emission_signature,
                    site_signature=stub.site_signature,
                    callee_name=stub.callee_name,
                    source_path=stub.source_path,
                    lineno=stub.lineno,
                    structure_signature=stub.structure_signature,
                    block_signature=stub.block_signature,
                    boundary_kind=stub.boundary_kind,
                    branch_ids=stub.branch_ids,
                    positional_arg_kinds=positional_arg_kinds,
                    keyword_arg_names=keyword_arg_names,
                    return_summary=_default_return_summary(call_name),
                    side_effect_summary=_default_side_effect_summary(stub.stub_id, call_name),
                    notes=("boundary capsule derived from opaque call site",),
                )
            )
        return self.generic_visit(node)


def _extract_boundary_artifacts_from_instrumented_code(
    code_path: Path,
    logic_scope: LogicScopeSpec,
    boundary_rule: BlackBoxBoundaryRule | None,
) -> tuple[tuple[OperatorStub, ...], tuple[BoundaryContextCapsule, ...]]:
    if boundary_rule is None or not boundary_rule.opaque_call_names:
        return (), ()
    selected_paths = tuple(
        Path(path)
        for path in (logic_scope.selected_paths or (str(code_path),))
    )
    operator_stubs: list[OperatorStub] = []
    boundary_capsules: list[BoundaryContextCapsule] = []
    for selected_path in selected_paths:
        if is_cpp_like_path(selected_path):
            site_signature_occurrences: dict[str, int] = {}
            for block, lineno, call_name in iter_cpp_opaque_calls(
                selected_path,
                selected_functions=logic_scope.selected_functions,
                opaque_call_names=boundary_rule.opaque_call_names,
            ):
                positional_arg_kinds: tuple[str, ...] = ()
                keyword_arg_names: tuple[str, ...] = ()
                emission_signature = _make_emission_signature(
                    callee_name=call_name,
                    boundary_kind="opaque_call",
                    branch_ids=(),
                    positional_arg_kinds=positional_arg_kinds,
                    keyword_arg_names=keyword_arg_names,
                )
                site_occurrence = site_signature_occurrences.get(emission_signature, 0)
                site_signature_occurrences[emission_signature] = site_occurrence + 1
                site_signature = f"{emission_signature}|site_occ={site_occurrence}"
                block_signature = hashlib.sha1(block.block_text.encode("utf-8")).hexdigest()[:16]
                stub = OperatorStub(
                    stub_id=f"{call_name}@{lineno}",
                    emission_signature=emission_signature,
                    site_signature=site_signature,
                    callee_name=call_name,
                    source_path=str(selected_path),
                    lineno=lineno,
                    structure_signature=block.structure_signature,
                    block_signature=block_signature,
                    boundary_kind="opaque_call",
                    branch_ids=(),
                    notes=("captured at black-box boundary from selected C++ source",),
                )
                operator_stubs.append(stub)
                boundary_capsules.append(
                    BoundaryContextCapsule(
                        capsule_id=f"capsule::{stub.stub_id}",
                        stub_id=stub.stub_id,
                        emission_signature=stub.emission_signature,
                        site_signature=stub.site_signature,
                        callee_name=stub.callee_name,
                        source_path=stub.source_path,
                        lineno=stub.lineno,
                        structure_signature=stub.structure_signature,
                        block_signature=stub.block_signature,
                        boundary_kind=stub.boundary_kind,
                        branch_ids=(),
                        positional_arg_kinds=positional_arg_kinds,
                        keyword_arg_names=keyword_arg_names,
                        return_summary=_default_return_summary(call_name),
                        side_effect_summary=_default_side_effect_summary(stub.stub_id, call_name),
                        notes=("boundary capsule derived from selected C++ call site",),
                    )
                )
            continue
        tree = ast.parse(selected_path.read_text(encoding="utf-8"), filename=str(selected_path))
        visitor = _OperatorStubVisitor(selected_path, boundary_rule)
        selected_roots = _selected_boundary_roots(tree, logic_scope.selected_functions)
        for root in selected_roots:
            visitor.visit(root)
        operator_stubs.extend(visitor.operator_stubs)
        boundary_capsules.extend(visitor.boundary_capsules)
    return tuple(operator_stubs), tuple(boundary_capsules)


def _annotate_boundary_neighborhoods(
    operator_stubs: tuple[OperatorStub, ...] | list[OperatorStub],
    boundary_capsules: tuple[BoundaryContextCapsule, ...] | list[BoundaryContextCapsule],
) -> tuple[tuple[OperatorStub, ...], tuple[BoundaryContextCapsule, ...]]:
    ordered_stubs = tuple(
        sorted(operator_stubs, key=lambda stub: (stub.source_path, stub.lineno, stub.stub_id))
    )
    if not ordered_stubs:
        return (), tuple(boundary_capsules)
    neighborhood_by_stub_id: dict[str, str] = {}
    for index, stub in enumerate(ordered_stubs):
        prev_signature = ordered_stubs[index - 1].emission_signature if index > 0 else "<start>"
        next_signature = (
            ordered_stubs[index + 1].emission_signature if index + 1 < len(ordered_stubs) else "<end>"
        )
        neighborhood_by_stub_id[stub.stub_id] = "|".join(
            (
                stub.emission_signature,
                f"prev={prev_signature}",
                f"next={next_signature}",
            )
        )
    enriched_stubs = tuple(
        replace(stub, neighborhood_signature=neighborhood_by_stub_id.get(stub.stub_id, ""))
        for stub in operator_stubs
    )
    enriched_capsules = tuple(
        replace(
            capsule,
            notes=capsule.notes
            + (
                f"neighborhood_signature={neighborhood_by_stub_id.get(capsule.stub_id, '')}",
            ),
        )
        for capsule in boundary_capsules
    )
    return enriched_stubs, enriched_capsules


def build_emission_dag(
    operator_stubs: tuple[OperatorStub, ...] | list[OperatorStub],
    *,
    logic_scope: LogicScopeSpec,
) -> EmissionDAG:
    ordered_stubs = tuple(
        sorted(
            operator_stubs,
            key=lambda stub: (stub.source_path, stub.lineno, stub.stub_id),
        )
    )
    nodes = tuple(
        EmissionNode(
            emission_id=f"emit::{logic_scope.scope_id}::{index}",
            stub_id=stub.stub_id,
            emission_signature=stub.emission_signature,
            site_signature=stub.site_signature,
            structure_signature=stub.structure_signature,
            block_signature=stub.block_signature,
            neighborhood_signature=stub.neighborhood_signature,
            callee_name=stub.callee_name,
            source_path=stub.source_path,
            lineno=stub.lineno,
            branch_ids=stub.branch_ids,
            notes=stub.notes + ("emission node derived from black-box boundary stub",),
        )
        for index, stub in enumerate(ordered_stubs)
    )
    edges: list[EmissionEdge] = []
    for left, right in zip(nodes, nodes[1:], strict=False):
        rationale = ("emission order adjacency",)
        if left.branch_ids and left.branch_ids == right.branch_ids:
            rationale = rationale + ("shared branch context",)
        edges.append(
            EmissionEdge(
                src_emission_id=left.emission_id,
                dst_emission_id=right.emission_id,
                rationale=rationale,
            )
        )
    return EmissionDAG(
        scope_id=logic_scope.scope_id,
        nodes=nodes,
        edges=tuple(edges),
        notes=("first-pass emission DAG from black-box boundary stubs",),
    )


def build_logic_emission_map(
    program_logic: ProgramLogicCarrier,
    emission_dag: EmissionDAG,
    *,
    control_region_tree: ControlRegionTree | None = None,
) -> LogicEmissionMap:
    links: list[LogicEmissionLink] = []
    linked_emission_ids: set[str] = set()
    for point in program_logic.points:
        point_branch_ids = set(point.branch_ids)
        if not point_branch_ids:
            continue
        for node in emission_dag.nodes:
            shared_branch_ids = tuple(
                branch_id for branch_id in node.branch_ids if branch_id in point_branch_ids
            )
            if not shared_branch_ids:
                continue
            links.append(
                LogicEmissionLink(
                    logic_point_name=point.name,
                    emission_id=node.emission_id,
                    rationale=(
                        "shared branch-context linkage",
                        f"branch_ids={','.join(str(item) for item in shared_branch_ids)}",
                    ),
                )
            )
            linked_emission_ids.add(node.emission_id)
    if control_region_tree is not None:
        ordered_regions = tuple(control_region_tree.regions)
        region_index_by_name = {
            region.region_name: index for index, region in enumerate(ordered_regions)
        }
        for node in emission_dag.nodes:
            if node.emission_id in linked_emission_ids:
                continue
            structure_tokens = tuple(token for token in node.structure_signature.split("|") if token)
            function_names = tuple(
                token.split("fn:", 1)[1] for token in structure_tokens if token.startswith("fn:")
            )
            anchor_index = len(ordered_regions) - 1
            for function_name in reversed(function_names):
                if function_name in region_index_by_name:
                    anchor_index = region_index_by_name[function_name]
                    break
            fallback_region = next(
                (
                    region
                    for region in reversed(ordered_regions[: anchor_index + 1])
                    if region.logic_point_names
                ),
                None,
            )
            if fallback_region is None:
                continue
            for point_name in fallback_region.logic_point_names:
                links.append(
                    LogicEmissionLink(
                        logic_point_name=point_name,
                        emission_id=node.emission_id,
                        rationale=(
                            "nearest preceding control-region fallback",
                            f"region={fallback_region.region_name}",
                        ),
                    )
                )
            linked_emission_ids.add(node.emission_id)
    return LogicEmissionMap(
        scope_id=emission_dag.scope_id,
        links=tuple(links),
        notes=("first-pass program-logic to emission linkage",),
    )


def _control_region_note_value(region: ControlRegion, prefix: str) -> str | None:
    for note in region.notes:
        if note.startswith(prefix):
            return note[len(prefix) :]
    return None


def build_emission_ownership_map(
    emission_dag: EmissionDAG | None,
    control_region_tree: ControlRegionTree | None,
) -> EmissionOwnershipMap | None:
    if emission_dag is None or control_region_tree is None:
        return None
    block_regions = tuple(
        region for region in control_region_tree.regions if region.kind == ControlRegionKind.BLOCK
    )
    if not block_regions:
        return EmissionOwnershipMap(
            scope_id=control_region_tree.scope_id,
            ownerships=(),
            notes=("no cfg block regions available for emission ownership mapping",),
        )

    known_cfg_branch_ids = {
        int(branch_id_text)
        for region in block_regions
        for branch_id_text in (_control_region_note_value(region, "branch_id="),)
        if branch_id_text is not None
    }

    def region_span_width(region: ControlRegion) -> int:
        region_end = region.end_lineno if region.end_lineno is not None else region.lineno
        return max(int(region_end) - int(region.lineno), 0)

    def narrowest_regions(
        regions: tuple[ControlRegion, ...] | list[ControlRegion],
    ) -> tuple[ControlRegion, ...]:
        if not regions:
            return ()
        width = min(region_span_width(region) for region in regions)
        return tuple(region for region in regions if region_span_width(region) == width)

    ownerships: list[EmissionOwnership] = []
    for node in emission_dag.nodes:
        owning_regions: tuple[ControlRegion, ...] = ()
        rationale: tuple[str, ...] = ()

        if node.source_path and node.lineno is not None:
            span_matches = tuple(
                region
                for region in block_regions
                if region.source_path == node.source_path
                and int(region.lineno)
                <= int(node.lineno)
                <= int(region.end_lineno if region.end_lineno is not None else region.lineno)
            )
            if span_matches:
                owning_regions = narrowest_regions(span_matches)
                rationale = ("cfg source-span emission ownership",)

        if not owning_regions and len(node.branch_ids) == 1 and node.branch_ids[0] in known_cfg_branch_ids:
            branch_matches = tuple(
                region
                for region in block_regions
                if _control_region_note_value(region, "branch_id=") == str(node.branch_ids[0])
            )
            if branch_matches:
                owning_regions = narrowest_regions(branch_matches)
                rationale = ("cfg branch-header emission fallback",)

        ownerships.append(
            EmissionOwnership(
                emission_id=node.emission_id,
                owning_region_ids=tuple(region.region_id for region in owning_regions),
                owning_cfg_block_ids=tuple(
                    cfg_block_id
                    for region in owning_regions
                    for cfg_block_id in (_control_region_note_value(region, "cfg_block_id="),)
                    if cfg_block_id
                ),
                rationale=rationale or ("no cfg emission ownership match",),
            )
        )

    return EmissionOwnershipMap(
        scope_id=control_region_tree.scope_id,
        ownerships=tuple(ownerships),
        notes=("emission-to-cfg ownership map derived from selected control-region blocks",),
    )


def build_logic_point_ownership_map(
    program_logic: ProgramLogicCarrier,
    control_region_tree: ControlRegionTree | None,
) -> LogicPointOwnershipMap | None:
    if control_region_tree is None:
        return None
    block_regions = tuple(
        region for region in control_region_tree.regions if region.kind == ControlRegionKind.BLOCK
    )
    if not block_regions:
        return LogicPointOwnershipMap(
            scope_id=control_region_tree.scope_id,
            ownerships=(),
            notes=("no cfg block regions available for ownership mapping",),
        )

    known_cfg_branch_ids = {
        int(branch_id_text)
        for region in block_regions
        for branch_id_text in (_control_region_note_value(region, "branch_id="),)
        if branch_id_text is not None
    }

    def region_span_width(region: ControlRegion) -> int:
        region_end = region.end_lineno if region.end_lineno is not None else region.lineno
        return max(int(region_end) - int(region.lineno), 0)

    def narrowest_regions(regions: tuple[ControlRegion, ...] | list[ControlRegion]) -> tuple[ControlRegion, ...]:
        if not regions:
            return ()
        width = min(region_span_width(region) for region in regions)
        return tuple(region for region in regions if region_span_width(region) == width)

    ownerships: list[LogicPointOwnership] = []
    for point in program_logic.points:
        owning_regions: tuple[ControlRegion, ...] = ()
        rationale: tuple[str, ...] = ()

        if len(point.branch_ids) == 1 and point.branch_ids[0] in known_cfg_branch_ids:
            matches = tuple(
                region
                for region in block_regions
                if _control_region_note_value(region, "branch_id=") == str(point.branch_ids[0])
            )
            if matches:
                owning_regions = narrowest_regions(matches)
                rationale = ("exact cfg branch-header ownership",)

        if not owning_regions and point.source_path and point.lineno is not None:
            span_matches = tuple(
                region
                for region in block_regions
                if region.source_path == point.source_path
                and int(region.lineno)
                <= int(point.lineno)
                <= int(region.end_lineno if region.end_lineno is not None else region.lineno)
            )
            if span_matches:
                owning_regions = narrowest_regions(span_matches)
                rationale = ("narrowest source-span ownership",)

        if not owning_regions and point.branch_ids:
            branch_matches = tuple(
                region for region in block_regions if region.branch_ids == point.branch_ids
            )
            if branch_matches:
                owning_regions = narrowest_regions(branch_matches)
                rationale = ("branch-id fallback ownership",)

        ownerships.append(
            LogicPointOwnership(
                point_name=point.name,
                owning_region_ids=tuple(region.region_id for region in owning_regions),
                owning_cfg_block_ids=tuple(
                    cfg_block_id
                    for region in owning_regions
                    for cfg_block_id in (_control_region_note_value(region, "cfg_block_id="),)
                    if cfg_block_id
                ),
                rationale=rationale or ("no cfg ownership match",),
            )
        )

    return LogicPointOwnershipMap(
        scope_id=control_region_tree.scope_id,
        ownerships=tuple(ownerships),
        notes=("point-to-cfg ownership map derived from selected control-region blocks",),
    )


def build_logic_state_store(
    program_logic: ProgramLogicCarrier,
    logic_emission_map: LogicEmissionMap,
    control_region_tree: ControlRegionTree | None = None,
    emission_ownership_map: EmissionOwnershipMap | None = None,
    logic_point_ownership_map: LogicPointOwnershipMap | None = None,
    slice_granularity_policy: LogicSliceGranularityPolicy | None = None,
) -> LogicStateStore:
    policy = slice_granularity_policy or LogicSliceGranularityPolicy()
    slices: list[LogicSliceState] = []
    point_by_name = {point.name: point for point in program_logic.points}

    def finalize_next_slice_ids(raw_slices: list[LogicSliceState]) -> LogicStateStore:
        finalized: list[LogicSliceState] = []
        for index, slice_state in enumerate(raw_slices):
            next_slice_ids = ()
            if index + 1 < len(raw_slices):
                next_slice_ids = (raw_slices[index + 1].slice_id,)
            finalized.append(replace(slice_state, next_slice_ids=next_slice_ids))
        notes: tuple[str, ...]
        if policy.mode == LogicSliceGranularityMode.LEGACY_REGION_LEAVES:
            notes = ("logic-state store derived from control-region leaves",)
        else:
            notes = (
                "logic-state store derived from cfg-aware slices",
                f"slice_policy={policy.mode.value}",
                f"max_cfg_blocks_per_slice={policy.max_cfg_blocks_per_slice}",
                f"coalesce_trivial_cfg_slices={policy.coalesce_trivial_cfg_slices}",
            ) + tuple(policy.notes)
        return LogicStateStore(
            scope_id=logic_emission_map.scope_id,
            slices=tuple(finalized),
            notes=notes,
        )

    def coalesce_cfg_slices(raw_slices: list[LogicSliceState]) -> list[LogicSliceState]:
        if not policy.coalesce_trivial_cfg_slices:
            return raw_slices

        def is_trivial_cfg_slice(slice_state: LogicSliceState) -> bool:
            return (
                not slice_state.logic_point_names
                and not slice_state.emission_ids
                and not slice_state.checkpointable
                and "cfg-derived logic slice" in slice_state.notes
            )

        meaningful_slices = [item for item in raw_slices if not is_trivial_cfg_slice(item)]
        if not meaningful_slices:
            return raw_slices
        return meaningful_slices

    def build_cfg_aware_slices() -> list[LogicSliceState]:
        if control_region_tree is None:
            return []
        block_regions = tuple(
            region
            for region in control_region_tree.regions
            if region.kind == ControlRegionKind.BLOCK
        )
        if not block_regions:
            return []
        ownership_map = logic_point_ownership_map or build_logic_point_ownership_map(
            program_logic,
            control_region_tree,
        )
        ownership_for_point_name = {
            ownership.point_name: ownership
            for ownership in (ownership_map.ownerships if ownership_map is not None else ())
        }
        emission_map = emission_ownership_map
        ownership_for_emission_id = {
            ownership.emission_id: ownership
            for ownership in (emission_map.ownerships if emission_map is not None else ())
        }
        explicitly_owned_emission_ids = {
            ownership.emission_id
            for ownership in ownership_for_emission_id.values()
            if ownership.owning_region_ids
        }
        fallback_emission_ids_for_point_name = {
            point_name: tuple(
                dict.fromkeys(
                    link.emission_id
                    for link in logic_emission_map.links_for_point(point_name)
                    if link.emission_id not in explicitly_owned_emission_ids
                )
            )
            for point_name in point_by_name
        }
        owned_emission_ids_for_region_id: dict[str, tuple[str, ...]] = {}
        for ownership in ownership_for_emission_id.values():
            if not ownership.owning_region_ids:
                continue
            for region_id in ownership.owning_region_ids:
                owned_emission_ids_for_region_id[region_id] = tuple(
                    dict.fromkeys(
                        owned_emission_ids_for_region_id.get(region_id, ())
                        + (ownership.emission_id,)
                    )
                )

        grouped_regions: list[list[ControlRegion]] = []
        current_group: list[ControlRegion] = []
        current_key: tuple[str, str | None] | None = None
        for region in block_regions:
            group_key = (region.source_path, region.parent_region_id)
            if current_key != group_key:
                if current_group:
                    grouped_regions.append(current_group)
                current_group = [region]
                current_key = group_key
            else:
                current_group.append(region)
        if current_group:
            grouped_regions.append(current_group)

        assigned_points: set[str] = set()
        built_slices: list[LogicSliceState] = []

        def finalize_cfg_slice(regions_for_slice: list[ControlRegion]) -> None:
            if not regions_for_slice:
                return
            covered_point_names: list[str] = []
            owned_region_ids = {region.region_id for region in regions_for_slice}
            for point in program_logic.points:
                if point.name in assigned_points:
                    continue
                ownership = ownership_for_point_name.get(point.name)
                if ownership is None or not ownership.owning_region_ids:
                    continue
                if owned_region_ids.intersection(ownership.owning_region_ids):
                    covered_point_names.append(point.name)
            for point_name in covered_point_names:
                assigned_points.add(point_name)
            entry_branch_ids = next(
                (region.branch_ids for region in regions_for_slice if region.branch_ids),
                (),
            )
            rank_groups: tuple[tuple[int, ...], ...] = ()
            for point_name in covered_point_names:
                point = point_by_name.get(point_name)
                if point is None:
                    continue
                if not rank_groups and point.rank_groups:
                    rank_groups = point.rank_groups
                if not entry_branch_ids and point.branch_ids:
                    entry_branch_ids = point.branch_ids
            explicitly_owned_emission_ids_for_slice = tuple(
                dict.fromkeys(
                    emission_id
                    for region_id in owned_region_ids
                    for emission_id in owned_emission_ids_for_region_id.get(region_id, ())
                )
            )
            fallback_emission_ids = tuple(
                dict.fromkeys(
                    emission_id
                    for point_name in covered_point_names
                    for emission_id in fallback_emission_ids_for_point_name.get(point_name, ())
                )
            )
            emission_ids = tuple(
                dict.fromkeys(
                    explicitly_owned_emission_ids_for_slice + fallback_emission_ids
                )
            )
            block_ids = tuple(
                value
                for region in regions_for_slice
                for value in (_control_region_note_value(region, "cfg_block_id="),)
                if value
            )
            slice_name = (
                covered_point_names[0]
                if len(covered_point_names) == 1
                else regions_for_slice[0].region_name
            )
            built_slices.append(
                LogicSliceState(
                    slice_id=f"logic::{slice_name}",
                    logic_point_names=tuple(covered_point_names),
                    entry_branch_ids=entry_branch_ids,
                    entry_rank_groups=rank_groups,
                    emission_ids=emission_ids,
                    owned_region_ids=tuple(region.region_id for region in regions_for_slice),
                    owned_cfg_block_ids=block_ids,
                    checkpointable=bool(entry_branch_ids or rank_groups),
                    notes=(
                        "cfg-derived logic slice",
                        f"cfg_blocks={','.join(block_ids) if block_ids else '<none>'}",
                        f"region_count={len(regions_for_slice)}",
                    ),
                )
            )

        for group in grouped_regions:
            current_regions: list[ControlRegion] = []
            for region in group:
                cfg_kind = _control_region_note_value(region, "cfg_kind=") or "basic"
                candidate_point_names = tuple(
                    point.name
                    for point in program_logic.points
                    if point.name not in assigned_points
                    and (
                        ownership_for_point_name.get(point.name) is not None
                        and region.region_id in ownership_for_point_name[point.name].owning_region_ids
                    )
                )
                starts_boundary = (
                    not current_regions
                    or cfg_kind in {"branch_header", "loop_header", "return", "break", "continue"}
                    or (
                        policy.split_on_merge
                        and cfg_kind == "merge"
                    )
                    or len(current_regions) >= max(1, int(policy.max_cfg_blocks_per_slice))
                )
                if starts_boundary and current_regions:
                    finalize_cfg_slice(current_regions)
                    current_regions = []
                current_regions.append(region)
                region_owned_emission_ids = owned_emission_ids_for_region_id.get(region.region_id, ())
                fallback_candidate_emission_ids = tuple(
                    dict.fromkeys(
                        emission_id
                        for point_name in candidate_point_names
                        for emission_id in fallback_emission_ids_for_point_name.get(point_name, ())
                    )
                )
                closes_after = (
                    cfg_kind in {"return", "break", "continue"}
                    or (policy.split_on_merge and cfg_kind == "merge")
                    or (
                        policy.split_on_emission
                        and bool(region_owned_emission_ids or fallback_candidate_emission_ids)
                    )
                )
                if closes_after:
                    finalize_cfg_slice(current_regions)
                    current_regions = []
            if current_regions:
                finalize_cfg_slice(current_regions)

        for point in program_logic.points:
            if point.name in assigned_points:
                continue
            built_slices.append(
                LogicSliceState(
                    slice_id=f"logic::{point.name}",
                    logic_point_names=(point.name,),
                    entry_branch_ids=point.branch_ids,
                    entry_rank_groups=point.rank_groups,
                    emission_ids=tuple(
                        link.emission_id for link in logic_emission_map.links_for_point(point.name)
                    ),
                    owned_region_ids=(),
                    owned_cfg_block_ids=(),
                    checkpointable=bool(point.branch_ids or point.rank_groups),
                    notes=(
                        "cfg-slice fallback for unmatched logic point",
                        f"source={point.source}",
                    ),
                )
            )
        return coalesce_cfg_slices(built_slices)

    if policy.mode == LogicSliceGranularityMode.CFG_BOUNDARY_AWARE:
        cfg_slices = build_cfg_aware_slices()
        if cfg_slices:
            return finalize_next_slice_ids(cfg_slices)

    leaf_regions = ()
    if control_region_tree is not None:
        leaf_regions = control_region_tree.leaf_regions(require_logic_points=True)
    if leaf_regions:
        for region in leaf_regions:
            rank_groups: tuple[tuple[int, ...], ...] = ()
            branch_ids: tuple[int, ...] = region.branch_ids
            for point_name in region.logic_point_names:
                point = point_by_name.get(point_name)
                if point is None:
                    continue
                if not branch_ids and point.branch_ids:
                    branch_ids = point.branch_ids
                if not rank_groups and point.rank_groups:
                    rank_groups = point.rank_groups
            emission_ids = tuple(
                dict.fromkeys(
                    emission_id
                    for point_name in region.logic_point_names
                    for emission_id in (
                        link.emission_id for link in logic_emission_map.links_for_point(point_name)
                    )
                )
            )
            slices.append(
                LogicSliceState(
                    slice_id=f"logic::{region.region_name}",
                    logic_point_names=region.logic_point_names,
                    entry_branch_ids=branch_ids,
                    entry_rank_groups=rank_groups,
                    emission_ids=emission_ids,
                    owned_region_ids=(region.region_id,),
                    owned_cfg_block_ids=tuple(
                        value
                        for value in (_control_region_note_value(region, "cfg_block_id="),)
                        if value
                    ),
                    checkpointable=bool(branch_ids or rank_groups),
                    notes=(
                        "control-region leaf checkpoint slice",
                        f"region={region.region_id}",
                    ),
                )
            )
    else:
        ordered_points = tuple(program_logic.points)
        for point in ordered_points:
            slices.append(
                LogicSliceState(
                    slice_id=f"logic::{point.name}",
                    logic_point_names=(point.name,),
                    entry_branch_ids=point.branch_ids,
                    entry_rank_groups=point.rank_groups,
                    emission_ids=tuple(
                        link.emission_id for link in logic_emission_map.links_for_point(point.name)
                    ),
                    owned_region_ids=(),
                    owned_cfg_block_ids=(),
                    checkpointable=bool(point.branch_ids or point.rank_groups),
                    notes=(
                        "fallback point-ordered logic checkpoint slice",
                        f"source={point.source}",
                    ),
                )
            )
    return finalize_next_slice_ids(slices)


def build_logic_slice_graph(
    logic_scope: LogicScopeSpec,
    logic_state_store: LogicStateStore | None,
) -> LogicSliceGraph | None:
    if logic_state_store is None:
        return None
    slice_ids = tuple(slice_state.slice_id for slice_state in logic_state_store.slices)
    if not slice_ids:
        return LogicSliceGraph(
            scope_id=logic_scope.scope_id,
            slice_ids=(),
            edges=(),
            notes=("empty logic-state store; no slice graph edges",),
        )

    slice_order = {slice_id: index for index, slice_id in enumerate(slice_ids)}
    block_to_slice_id: dict[str, str] = {}
    duplicate_block_ids: list[str] = []
    unowned_slice_ids = tuple(
        slice_state.slice_id
        for slice_state in logic_state_store.slices
        if not slice_state.owned_cfg_block_ids
    )
    for slice_state in logic_state_store.slices:
        for block_id in slice_state.owned_cfg_block_ids:
            prior = block_to_slice_id.get(block_id)
            if prior is not None and prior != slice_state.slice_id:
                duplicate_block_ids.append(block_id)
                continue
            block_to_slice_id[block_id] = slice_state.slice_id

    edge_map: dict[tuple[str, str, str], LogicSliceEdge] = {}
    cfg_paths = tuple(Path(raw_path) for raw_path in logic_scope.selected_paths)
    for selected_path in cfg_paths:
        for cfg in build_control_flow_graphs(
            selected_path,
            selected_functions=logic_scope.selected_functions,
        ):
            for edge in cfg.edges:
                src_slice_id = block_to_slice_id.get(edge.src_block_id)
                dst_slice_id = block_to_slice_id.get(edge.dst_block_id)
                if src_slice_id is None or dst_slice_id is None or src_slice_id == dst_slice_id:
                    continue
                edge_key = (src_slice_id, dst_slice_id, edge.kind)
                if edge_key in edge_map:
                    continue
                edge_map[edge_key] = LogicSliceEdge(
                    src_slice_id=src_slice_id,
                    dst_slice_id=dst_slice_id,
                    kind=edge.kind,
                    rationale=(
                        "collapsed cfg block edge",
                        f"src_block={edge.src_block_id}",
                        f"dst_block={edge.dst_block_id}",
                    ),
                )

    ordered_edges = tuple(
        edge
        for _, edge in sorted(
            edge_map.items(),
            key=lambda item: (
                slice_order.get(item[1].src_slice_id, len(slice_order)),
                slice_order.get(item[1].dst_slice_id, len(slice_order)),
                item[1].kind,
            ),
        )
    )
    ordered_edge_list = list(ordered_edges)
    existing_pairs = {(edge.src_slice_id, edge.dst_slice_id) for edge in ordered_edge_list}
    fallback_edge_count = 0
    for src_slice_id, dst_slice_id in zip(slice_ids, slice_ids[1:], strict=False):
        if (src_slice_id, dst_slice_id) in existing_pairs:
            continue
        ordered_edge_list.append(
            LogicSliceEdge(
                src_slice_id=src_slice_id,
                dst_slice_id=dst_slice_id,
                kind="legacy_linear_fallback",
                rationale=(
                    "fallback linear successor edge",
                    "preserves legacy downstream rerun closure",
                ),
            )
        )
        existing_pairs.add((src_slice_id, dst_slice_id))
        fallback_edge_count += 1

    ordered_edges = tuple(
        sorted(
            ordered_edge_list,
            key=lambda edge: (
                slice_order.get(edge.src_slice_id, len(slice_order)),
                slice_order.get(edge.dst_slice_id, len(slice_order)),
                edge.kind,
            ),
        )
    )

    notes = ["logic slice graph derived from cfg adjacency plus fallback linear order edges"]
    if not cfg_paths:
        notes.append("no selected paths available for cfg-derived slice edges")
    if unowned_slice_ids:
        notes.append(f"unowned_slices={','.join(unowned_slice_ids)}")
    if duplicate_block_ids:
        notes.append(f"duplicate_cfg_block_ids={','.join(dict.fromkeys(duplicate_block_ids))}")
    if fallback_edge_count:
        notes.append(f"fallback_linear_edges={fallback_edge_count}")
    return LogicSliceGraph(
        scope_id=logic_scope.scope_id,
        slice_ids=slice_ids,
        edges=ordered_edges,
        notes=tuple(notes),
    )


def propose_deeper_capture_retry(
    code_path: str | Path,
    *,
    logic_scope: LogicScopeSpec,
    boundary_rule: BlackBoxBoundaryRule | None,
    deopaque_call_names: tuple[str, ...] | list[str],
) -> tuple[LogicScopeSpec, BlackBoxBoundaryRule] | None:
    if boundary_rule is None:
        return None
    deopaque = tuple(
        dict.fromkeys(
            str(name)
            for name in deopaque_call_names
            if str(name) in set(boundary_rule.opaque_call_names)
        )
    )
    if not deopaque:
        return None
    selected_paths = tuple(
        Path(path)
        for path in (logic_scope.selected_paths or (str(code_path),))
    )
    defined_functions: set[str] = set()
    for selected_path in selected_paths:
        defined_functions.update(defined_function_names_for_path(selected_path))
    expanded_functions = tuple(
        dict.fromkeys(
            logic_scope.selected_functions
            + tuple(name for name in deopaque if name in defined_functions)
        )
    )
    narrowed_opaque_names = tuple(
        name for name in boundary_rule.opaque_call_names if name not in set(deopaque)
    )
    next_scope = LogicScopeSpec(
        scope_id=logic_scope.scope_id,
        selected_paths=logic_scope.selected_paths,
        selected_functions=expanded_functions,
        notes=logic_scope.notes + (f"deeper_retry_scope={','.join(deopaque)}",),
    )
    next_boundary_rule = BlackBoxBoundaryRule(
        opaque_call_names=narrowed_opaque_names,
        opaque_module_prefixes=boundary_rule.opaque_module_prefixes,
        notes=boundary_rule.notes + (f"deeper_retry_deopaque={','.join(deopaque)}",),
    )
    if next_scope == logic_scope and next_boundary_rule == boundary_rule:
        return None
    return next_scope, next_boundary_rule


def capture_program_logic_from_instrumented_code(
    code_path: str | Path,
    *,
    world_size: int,
    branch_var_map: Mapping[int, list[str] | tuple[str, ...]] | None = None,
    logic_scope: LogicScopeSpec | None = None,
    boundary_rule: BlackBoxBoundaryRule | None = None,
    slice_granularity_policy: LogicSliceGranularityPolicy | None = None,
) -> DryRunProgramLogicCapture:
    executor = DryRunExecutor(
        SimEnv(world_size=int(world_size)),
        branch_var_map={
            int(branch_id): [str(name) for name in names]
            for branch_id, names in (branch_var_map or {}).items()
        },
    )
    resolved_code_path = Path(code_path)
    executor.execute_all_ranks(resolved_code_path)
    branch_signatures = {
        int(rank): tuple(tuple(item) for item in signature)
        for rank, signature in executor.get_all_branch_signatures().items()
    }
    semantic_summaries = {
        int(rank): {
            str(name): {
                str(key): value
                for key, value in payload.items()
            }
            for name, payload in summary.items()
        }
        for rank, summary in executor.get_all_semantic_summaries().items()
    }
    resolved_scope = logic_scope or _default_logic_scope_for_path(resolved_code_path)
    operator_stubs, boundary_capsules = _extract_boundary_artifacts_from_instrumented_code(
        resolved_code_path,
        resolved_scope,
        boundary_rule,
    )
    operator_stubs, boundary_capsules = _annotate_boundary_neighborhoods(
        operator_stubs,
        boundary_capsules,
    )
    branch_metadata = _collect_branch_source_metadata(
        resolved_scope.selected_paths,
        resolved_scope.selected_functions,
    )
    program_logic = build_program_logic_carrier_from_executor(
        executor,
        branch_metadata=branch_metadata,
    )
    control_region_tree = build_control_region_tree(
        resolved_code_path,
        scope_id=resolved_scope.scope_id,
        selected_functions=resolved_scope.selected_functions,
        program_logic=program_logic,
        selected_paths=resolved_scope.selected_paths,
    )
    emission_dag = build_emission_dag(
        operator_stubs,
        logic_scope=resolved_scope,
    )
    logic_emission_map = build_logic_emission_map(
        program_logic,
        emission_dag,
        control_region_tree=control_region_tree,
    )
    emission_ownership_map = build_emission_ownership_map(
        emission_dag,
        control_region_tree,
    )
    logic_point_ownership_map = build_logic_point_ownership_map(
        program_logic,
        control_region_tree,
    )
    logic_state_store = build_logic_state_store(
        program_logic,
        logic_emission_map,
        control_region_tree=control_region_tree,
        emission_ownership_map=emission_ownership_map,
        logic_point_ownership_map=logic_point_ownership_map,
        slice_granularity_policy=slice_granularity_policy,
    )
    logic_slice_graph = build_logic_slice_graph(
        resolved_scope,
        logic_state_store,
    )
    return DryRunProgramLogicCapture(
        code_path=str(resolved_code_path),
        world_size=int(world_size),
        branch_signatures=branch_signatures,
        semantic_summaries=semantic_summaries,
        program_logic=program_logic,
        logic_scope=resolved_scope,
        boundary_rule=boundary_rule,
        operator_stubs=operator_stubs,
        boundary_capsules=boundary_capsules,
        emission_dag=emission_dag,
        logic_emission_map=logic_emission_map,
        emission_ownership_map=emission_ownership_map,
        logic_point_ownership_map=logic_point_ownership_map,
        logic_state_store=logic_state_store,
        logic_slice_graph=logic_slice_graph,
        control_region_tree=control_region_tree,
    )
