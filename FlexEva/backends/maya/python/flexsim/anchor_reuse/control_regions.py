"""Standard control-region decomposition for selected logic scopes.

This module is intentionally not a claimed contribution. It provides a formal
substrate for the program-logic layer using a control-region tree with
SESE-style leaf regions over the selected logic scope.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .control_flow import build_control_flow_graphs
from .program_logic import ProgramLogicCarrier
from .source_frontend import cpp_function_blocks, is_cpp_like_path


class ControlRegionKind(str, Enum):
    ROOT = "root"
    FUNCTION = "function"
    BRANCH = "branch"
    LOOP = "loop"
    BLOCK = "block"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class ControlRegion:
    region_id: str
    region_name: str
    kind: ControlRegionKind
    source_path: str
    lineno: int
    end_lineno: int | None = None
    parent_region_id: str | None = None
    child_region_ids: tuple[str, ...] = ()
    branch_ids: tuple[int, ...] = ()
    logic_point_names: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def is_leaf(self) -> bool:
        return not self.child_region_ids


@dataclass(frozen=True)
class ControlRegionTree:
    scope_id: str
    root_region_id: str
    regions: tuple[ControlRegion, ...]
    notes: tuple[str, ...] = ()

    def region_by_id(self) -> dict[str, ControlRegion]:
        return {region.region_id: region for region in self.regions}

    def child_regions(self, parent_region_id: str) -> tuple[ControlRegion, ...]:
        return tuple(region for region in self.regions if region.parent_region_id == parent_region_id)

    def leaf_regions(self, *, require_logic_points: bool = False) -> tuple[ControlRegion, ...]:
        return tuple(
            region
            for region in self.regions
            if region.is_leaf and (not require_logic_points or bool(region.logic_point_names))
        )


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


def _selected_function_nodes(module: ast.Module, selected_functions: tuple[str, ...]) -> tuple[ast.AST, ...]:
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


def _matching_point_names(
    program_logic: ProgramLogicCarrier | None,
    *,
    branch_id: int | None = None,
) -> tuple[str, ...]:
    if program_logic is None:
        return ()
    matches: list[str] = []
    for point in program_logic.points:
        if branch_id is not None and (point.name == f"branch_{branch_id}" or branch_id in point.branch_ids):
            matches.append(point.name)
    return tuple(dict.fromkeys(matches))


def _path_token(path: Path) -> str:
    return path.stem.replace("-", "_").replace(".", "_")


def _region_token(value: str) -> str:
    return (
        value.replace("-", "_")
        .replace(".", "_")
        .replace("<", "")
        .replace(">", "")
        .replace(" ", "_")
    )


def build_control_region_tree(
    code_path: str | Path,
    *,
    scope_id: str,
    selected_functions: tuple[str, ...] = (),
    program_logic: ProgramLogicCarrier | None = None,
    selected_paths: tuple[str, ...] | None = None,
) -> ControlRegionTree:
    resolved_code_path = Path(code_path)
    selected_code_paths = tuple(
        dict.fromkeys(
            Path(path)
            for path in ((str(resolved_code_path),) if not selected_paths else selected_paths)
        )
    )
    regions: list[ControlRegion] = []
    child_map: dict[str, list[str]] = {}
    assigned_points: set[str] = set()
    function_region_ids: dict[tuple[str, str], str] = {}

    def add_region(
        *,
        region_id: str,
        region_name: str,
        kind: ControlRegionKind,
        source_path: str,
        lineno: int,
        end_lineno: int | None,
        parent_region_id: str | None,
        branch_ids: tuple[int, ...] = (),
        logic_point_names: tuple[str, ...] = (),
        notes: tuple[str, ...] = (),
    ) -> str:
        regions.append(
            ControlRegion(
                region_id=region_id,
                region_name=region_name,
                kind=kind,
                source_path=source_path,
                lineno=int(lineno),
                end_lineno=end_lineno,
                parent_region_id=parent_region_id,
                branch_ids=branch_ids,
                logic_point_names=logic_point_names,
                notes=notes,
            )
        )
        if parent_region_id is not None:
            child_map.setdefault(parent_region_id, []).append(region_id)
        assigned_points.update(logic_point_names)
        return region_id

    root_region_id = f"control::{scope_id}::root"
    add_region(
        region_id=root_region_id,
        region_name="root",
        kind=ControlRegionKind.ROOT,
        source_path=str(selected_code_paths[0]),
        lineno=1,
        end_lineno=None,
        parent_region_id=None,
        notes=("selected logic scope root",),
    )

    def visit_body(
        body: list[ast.stmt],
        *,
        parent_region_id: str,
        enclosing_branch_ids: tuple[int, ...],
        region_source_path: Path,
        source_token: str,
    ) -> None:
        for statement in body:
            if isinstance(statement, ast.If):
                branch_id = _extract_mark_cond_branch_id(statement.test)
                if branch_id is not None:
                    region_name = f"branch_{branch_id}"
                    region_id = f"control::{scope_id}::src::{source_token}::{region_name}"
                    point_names = _matching_point_names(program_logic, branch_id=branch_id)
                    add_region(
                        region_id=region_id,
                        region_name=region_name,
                        kind=ControlRegionKind.BRANCH,
                        source_path=str(region_source_path),
                        lineno=int(getattr(statement, "lineno", 0)),
                        end_lineno=getattr(statement, "end_lineno", None),
                        parent_region_id=parent_region_id,
                        branch_ids=enclosing_branch_ids + (branch_id,),
                        logic_point_names=point_names,
                        notes=("mark_cond branch region",),
                    )
                    visit_body(
                        statement.body,
                        parent_region_id=region_id,
                        enclosing_branch_ids=enclosing_branch_ids + (branch_id,),
                        region_source_path=region_source_path,
                        source_token=source_token,
                    )
                    visit_body(
                        statement.orelse,
                        parent_region_id=region_id,
                        enclosing_branch_ids=enclosing_branch_ids + (branch_id,),
                        region_source_path=region_source_path,
                        source_token=source_token,
                    )
                    continue
                visit_body(
                    statement.body,
                    parent_region_id=parent_region_id,
                    enclosing_branch_ids=enclosing_branch_ids,
                    region_source_path=region_source_path,
                    source_token=source_token,
                )
                visit_body(
                    statement.orelse,
                    parent_region_id=parent_region_id,
                    enclosing_branch_ids=enclosing_branch_ids,
                    region_source_path=region_source_path,
                    source_token=source_token,
                )
                continue
            if isinstance(statement, (ast.For, ast.While, ast.AsyncFor)):
                region_name = f"loop_{int(getattr(statement, 'lineno', 0))}"
                region_id = f"control::{scope_id}::src::{source_token}::{region_name}"
                add_region(
                    region_id=region_id,
                    region_name=region_name,
                    kind=ControlRegionKind.LOOP,
                    source_path=str(region_source_path),
                    lineno=int(getattr(statement, "lineno", 0)),
                    end_lineno=getattr(statement, "end_lineno", None),
                    parent_region_id=parent_region_id,
                    branch_ids=enclosing_branch_ids,
                    notes=("loop control region",),
                )
                visit_body(
                    statement.body,
                    parent_region_id=region_id,
                    enclosing_branch_ids=enclosing_branch_ids,
                    region_source_path=region_source_path,
                    source_token=source_token,
                )
                visit_body(
                    statement.orelse,
                    parent_region_id=region_id,
                    enclosing_branch_ids=enclosing_branch_ids,
                    region_source_path=region_source_path,
                    source_token=source_token,
                )
                continue
            if isinstance(statement, ast.Try):
                visit_body(
                    statement.body,
                    parent_region_id=parent_region_id,
                    enclosing_branch_ids=enclosing_branch_ids,
                    region_source_path=region_source_path,
                    source_token=source_token,
                )
                visit_body(
                    statement.orelse,
                    parent_region_id=parent_region_id,
                    enclosing_branch_ids=enclosing_branch_ids,
                    region_source_path=region_source_path,
                    source_token=source_token,
                )
                visit_body(
                    statement.finalbody,
                    parent_region_id=parent_region_id,
                    enclosing_branch_ids=enclosing_branch_ids,
                    region_source_path=region_source_path,
                    source_token=source_token,
                )
                for handler in statement.handlers:
                    visit_body(
                        handler.body,
                        parent_region_id=parent_region_id,
                        enclosing_branch_ids=enclosing_branch_ids,
                        region_source_path=region_source_path,
                        source_token=source_token,
                    )
                continue
            if isinstance(statement, (ast.With, ast.AsyncWith)):
                visit_body(
                    statement.body,
                    parent_region_id=parent_region_id,
                    enclosing_branch_ids=enclosing_branch_ids,
                    region_source_path=region_source_path,
                    source_token=source_token,
                )

    for selected_path in selected_code_paths:
        if is_cpp_like_path(selected_path):
            source_token = _path_token(selected_path)
            for block in cpp_function_blocks(selected_path, selected_functions):
                function_region_id = f"control::{scope_id}::src::{source_token}::fn::{block.name}"
                add_region(
                    region_id=function_region_id,
                    region_name=block.name,
                    kind=ControlRegionKind.FUNCTION,
                    source_path=str(selected_path),
                    lineno=block.lineno,
                    end_lineno=block.end_lineno,
                    parent_region_id=root_region_id,
                    notes=("selected C++ function region",),
                )
                function_region_ids[(str(selected_path), block.name)] = function_region_id
            for cfg in build_control_flow_graphs(
                selected_path,
                selected_functions=selected_functions,
            ):
                parent_region_id = function_region_ids.get(
                    (str(selected_path), cfg.function_name),
                    root_region_id,
                )
                immediate_dominators = cfg.immediate_dominators()
                immediate_post_dominators = cfg.immediate_post_dominators()
                function_token = _region_token(cfg.function_name)
                for block in cfg.blocks:
                    if block.kind.value in {"entry", "exit"}:
                        continue
                    region_name = f"cfg_{source_token}_{function_token}_{block.order_index:03d}"
                    region_id = f"control::{scope_id}::src::{source_token}::{region_name}"
                    notes = (
                        "cfg-backed block region",
                        f"cfg_kind={block.kind.value}",
                        f"cfg_label={block.label}",
                        f"cfg_block_id={block.block_id}",
                        f"idom={immediate_dominators.get(block.block_id)}",
                        f"ipdom={immediate_post_dominators.get(block.block_id)}",
                    ) + tuple(block.notes)
                    add_region(
                        region_id=region_id,
                        region_name=region_name,
                        kind=ControlRegionKind.BLOCK,
                        source_path=str(selected_path),
                        lineno=block.lineno,
                        end_lineno=block.end_lineno,
                        parent_region_id=parent_region_id,
                        branch_ids=block.branch_ids,
                        notes=notes,
                    )
            continue
        module = ast.parse(selected_path.read_text(encoding="utf-8"), filename=str(selected_path))
        source_token = _path_token(selected_path)
        for node in _selected_function_nodes(module, selected_functions):
            if isinstance(node, ast.Module):
                visit_body(
                    node.body,
                    parent_region_id=root_region_id,
                    enclosing_branch_ids=(),
                    region_source_path=selected_path,
                    source_token=source_token,
                )
                continue
            function_region_id = f"control::{scope_id}::src::{source_token}::fn::{node.name}"
            add_region(
                region_id=function_region_id,
                region_name=node.name,
                kind=ControlRegionKind.FUNCTION,
                source_path=str(selected_path),
                lineno=int(getattr(node, "lineno", 0)),
                end_lineno=getattr(node, "end_lineno", None),
                parent_region_id=root_region_id,
                notes=("selected function region",),
            )
            function_region_ids[(str(selected_path), node.name)] = function_region_id
            visit_body(
                node.body,
                parent_region_id=function_region_id,
                enclosing_branch_ids=(),
                region_source_path=selected_path,
                source_token=source_token,
            )
        for cfg in build_control_flow_graphs(
            selected_path,
            selected_functions=selected_functions,
        ):
            parent_region_id = function_region_ids.get(
                (str(selected_path), cfg.function_name),
                root_region_id,
            )
            immediate_dominators = cfg.immediate_dominators()
            immediate_post_dominators = cfg.immediate_post_dominators()
            function_token = _region_token(cfg.function_name)
            for block in cfg.blocks:
                if block.kind.value in {"entry", "exit"}:
                    continue
                region_name = f"cfg_{source_token}_{function_token}_{block.order_index:03d}"
                region_id = f"control::{scope_id}::src::{source_token}::{region_name}"
                notes = (
                    "cfg-backed block region",
                    f"cfg_kind={block.kind.value}",
                    f"cfg_label={block.label}",
                    f"cfg_block_id={block.block_id}",
                    f"idom={immediate_dominators.get(block.block_id)}",
                    f"ipdom={immediate_post_dominators.get(block.block_id)}",
                ) + tuple(block.notes)
                add_region(
                    region_id=region_id,
                    region_name=region_name,
                    kind=ControlRegionKind.BLOCK,
                    source_path=str(selected_path),
                    lineno=block.lineno,
                    end_lineno=block.end_lineno,
                    parent_region_id=parent_region_id,
                    branch_ids=block.branch_ids,
                    notes=notes,
                )

    if program_logic is not None:
        for point in program_logic.points:
            if point.name in assigned_points:
                continue
            synthetic_region_id = f"control::{scope_id}::{point.name}"
            add_region(
                region_id=synthetic_region_id,
                region_name=point.name,
                kind=ControlRegionKind.SYNTHETIC,
                source_path=str(selected_code_paths[0]),
                lineno=0,
                end_lineno=None,
                parent_region_id=root_region_id,
                branch_ids=point.branch_ids,
                logic_point_names=(point.name,),
                notes=("synthetic control-region leaf for unmatched logic point",),
            )

    finalized_regions: list[ControlRegion] = []
    for region in regions:
        finalized_regions.append(
            ControlRegion(
                region_id=region.region_id,
                region_name=region.region_name,
                kind=region.kind,
                source_path=region.source_path,
                lineno=region.lineno,
                end_lineno=region.end_lineno,
                parent_region_id=region.parent_region_id,
                child_region_ids=tuple(child_map.get(region.region_id, ())),
                branch_ids=region.branch_ids,
                logic_point_names=region.logic_point_names,
                notes=region.notes,
            )
        )

    return ControlRegionTree(
        scope_id=scope_id,
        root_region_id=root_region_id,
        regions=tuple(finalized_regions),
        notes=("standard control-region substrate for selected logic scope",),
    )
