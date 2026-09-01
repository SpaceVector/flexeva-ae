"""Lightweight control-flow graphs for selected Python logic scopes.

This module provides a small formal substrate for the program-logic layer:
per-function control-flow graphs plus dominator and post-dominator queries.
It intentionally focuses on the selected Python scope and keeps the builder
lightweight so it can back the control-region tree without destabilizing the
existing local-rerun path.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .source_frontend import cpp_function_blocks, is_cpp_like_path


class ControlFlowBlockKind(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"
    BASIC = "basic"
    BRANCH_HEADER = "branch_header"
    LOOP_HEADER = "loop_header"
    MERGE = "merge"
    RETURN = "return"
    BREAK = "break"
    CONTINUE = "continue"


@dataclass(frozen=True)
class ControlFlowBlock:
    block_id: str
    function_name: str
    source_path: str
    kind: ControlFlowBlockKind
    label: str
    order_index: int
    lineno: int
    end_lineno: int | None = None
    branch_ids: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ControlFlowEdge:
    src_block_id: str
    dst_block_id: str
    kind: str = "normal"


@dataclass(frozen=True)
class ControlFlowGraph:
    source_path: str
    function_name: str
    entry_block_id: str
    exit_block_id: str
    blocks: tuple[ControlFlowBlock, ...]
    edges: tuple[ControlFlowEdge, ...]
    notes: tuple[str, ...] = ()

    def block_by_id(self) -> dict[str, ControlFlowBlock]:
        return {block.block_id: block for block in self.blocks}

    def successor_map(self) -> dict[str, tuple[str, ...]]:
        buckets: dict[str, list[str]] = {block.block_id: [] for block in self.blocks}
        for edge in self.edges:
            buckets.setdefault(edge.src_block_id, []).append(edge.dst_block_id)
        return {block_id: tuple(values) for block_id, values in buckets.items()}

    def predecessor_map(self) -> dict[str, tuple[str, ...]]:
        buckets: dict[str, list[str]] = {block.block_id: [] for block in self.blocks}
        for edge in self.edges:
            buckets.setdefault(edge.dst_block_id, []).append(edge.src_block_id)
        return {block_id: tuple(values) for block_id, values in buckets.items()}

    def dominator_sets(self) -> dict[str, frozenset[str]]:
        block_ids = tuple(block.block_id for block in self.blocks)
        predecessors = self.predecessor_map()
        all_blocks = frozenset(block_ids)
        dominators: dict[str, frozenset[str]] = {}
        for block_id in block_ids:
            if block_id == self.entry_block_id:
                dominators[block_id] = frozenset({block_id})
            else:
                dominators[block_id] = all_blocks
        changed = True
        while changed:
            changed = False
            for block_id in block_ids:
                if block_id == self.entry_block_id:
                    continue
                preds = predecessors.get(block_id, ())
                if preds:
                    shared = set(dominators[preds[0]])
                    for predecessor_id in preds[1:]:
                        shared.intersection_update(dominators[predecessor_id])
                else:
                    shared = set()
                shared.add(block_id)
                new_value = frozenset(shared)
                if new_value != dominators[block_id]:
                    dominators[block_id] = new_value
                    changed = True
        return dominators

    def immediate_dominators(self) -> dict[str, str | None]:
        dominators = self.dominator_sets()
        immediate: dict[str, str | None] = {self.entry_block_id: None}
        for block in self.blocks:
            block_id = block.block_id
            if block_id == self.entry_block_id:
                continue
            candidates = tuple(dominators[block_id] - {block_id})
            chosen: str | None = None
            for candidate in candidates:
                if all(other == candidate or other in dominators[candidate] for other in candidates):
                    chosen = candidate
                    break
            immediate[block_id] = chosen
        return immediate

    def post_dominator_sets(self) -> dict[str, frozenset[str]]:
        block_ids = tuple(block.block_id for block in self.blocks)
        successors = self.successor_map()
        all_blocks = frozenset(block_ids)
        post_dominators: dict[str, frozenset[str]] = {}
        for block_id in block_ids:
            if block_id == self.exit_block_id:
                post_dominators[block_id] = frozenset({block_id})
            else:
                post_dominators[block_id] = all_blocks
        changed = True
        while changed:
            changed = False
            for block_id in reversed(block_ids):
                if block_id == self.exit_block_id:
                    continue
                succs = successors.get(block_id, ())
                if succs:
                    shared = set(post_dominators[succs[0]])
                    for successor_id in succs[1:]:
                        shared.intersection_update(post_dominators[successor_id])
                else:
                    shared = set()
                shared.add(block_id)
                new_value = frozenset(shared)
                if new_value != post_dominators[block_id]:
                    post_dominators[block_id] = new_value
                    changed = True
        return post_dominators

    def immediate_post_dominators(self) -> dict[str, str | None]:
        post_dominators = self.post_dominator_sets()
        immediate: dict[str, str | None] = {self.exit_block_id: None}
        for block in self.blocks:
            block_id = block.block_id
            if block_id == self.exit_block_id:
                continue
            candidates = tuple(post_dominators[block_id] - {block_id})
            chosen: str | None = None
            for candidate in candidates:
                if all(other == candidate or other in post_dominators[candidate] for other in candidates):
                    chosen = candidate
                    break
            immediate[block_id] = chosen
        return immediate


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


def _path_token(path: Path) -> str:
    return path.stem.replace("-", "_").replace(".", "_")


def _selected_python_roots(module: ast.Module, selected_functions: tuple[str, ...]) -> tuple[tuple[str, list[ast.stmt]], ...]:
    if not selected_functions:
        return (("<module>", module.body),)
    wanted = set(selected_functions)
    selected: list[tuple[str, list[ast.stmt]]] = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            selected.append((node.name, node.body))
            continue
        if isinstance(node, ast.ClassDef):
            for statement in node.body:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name in wanted:
                    selected.append((statement.name, statement.body))
    return tuple(selected)


@dataclass(frozen=True)
class _LoopContext:
    break_target: str
    continue_target: str


class _FunctionCFGBuilder:
    def __init__(self, *, source_path: Path, function_name: str) -> None:
        self.source_path = source_path
        self.function_name = function_name
        self._source_token = _path_token(source_path)
        self._block_order = 0
        self._blocks: list[ControlFlowBlock] = []
        self._edges: list[ControlFlowEdge] = []
        self._edge_keys: set[tuple[str, str, str]] = set()
        self.entry_block_id = self._new_block(
            kind=ControlFlowBlockKind.ENTRY,
            label="entry",
            node=None,
            notes=("synthetic cfg entry",),
        )
        self.exit_block_id = self._new_block(
            kind=ControlFlowBlockKind.EXIT,
            label="exit",
            node=None,
            notes=("synthetic cfg exit",),
        )

    def build(self, body: list[ast.stmt]) -> ControlFlowGraph:
        exits = self._visit_body(
            body,
            incoming=(self.entry_block_id,),
            loop_ctx=None,
            active_branch_ids=(),
        )
        for block_id in exits:
            self._add_edge(block_id, self.exit_block_id)
        return ControlFlowGraph(
            source_path=str(self.source_path),
            function_name=self.function_name,
            entry_block_id=self.entry_block_id,
            exit_block_id=self.exit_block_id,
            blocks=tuple(self._blocks),
            edges=tuple(self._edges),
            notes=("lightweight selected-scope control-flow graph",),
        )

    def _next_block_id(self) -> tuple[str, int]:
        self._block_order += 1
        return (
            f"cfg::{self._source_token}::{self.function_name.replace('.', '_')}::{self._block_order:03d}",
            self._block_order,
        )

    def _new_block(
        self,
        *,
        kind: ControlFlowBlockKind,
        label: str,
        node: ast.AST | None,
        branch_ids: tuple[int, ...] = (),
        notes: tuple[str, ...] = (),
    ) -> str:
        block_id, order_index = self._next_block_id()
        lineno = int(getattr(node, "lineno", 0) or 0)
        end_lineno = getattr(node, "end_lineno", None)
        self._blocks.append(
            ControlFlowBlock(
                block_id=block_id,
                function_name=self.function_name,
                source_path=str(self.source_path),
                kind=kind,
                label=label,
                order_index=order_index,
                lineno=lineno,
                end_lineno=end_lineno,
                branch_ids=branch_ids,
                notes=notes,
            )
        )
        return block_id

    def _add_edge(self, src_block_id: str, dst_block_id: str, *, kind: str = "normal") -> None:
        edge_key = (src_block_id, dst_block_id, kind)
        if edge_key in self._edge_keys:
            return
        self._edge_keys.add(edge_key)
        self._edges.append(
            ControlFlowEdge(
                src_block_id=src_block_id,
                dst_block_id=dst_block_id,
                kind=kind,
            )
        )

    @staticmethod
    def _dedupe_block_ids(block_ids: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(block_ids))

    def _connect_all(self, incoming: tuple[str, ...], target: str, *, kind: str = "normal") -> None:
        for block_id in incoming:
            self._add_edge(block_id, target, kind=kind)

    def _visit_body(
        self,
        body: list[ast.stmt],
        *,
        incoming: tuple[str, ...],
        loop_ctx: _LoopContext | None,
        active_branch_ids: tuple[int, ...],
    ) -> tuple[str, ...]:
        current = incoming
        for statement in body:
            current = self._visit_statement(
                statement,
                incoming=current,
                loop_ctx=loop_ctx,
                active_branch_ids=active_branch_ids,
            )
        return current

    def _visit_statement(
        self,
        statement: ast.stmt,
        *,
        incoming: tuple[str, ...],
        loop_ctx: _LoopContext | None,
        active_branch_ids: tuple[int, ...],
    ) -> tuple[str, ...]:
        if not incoming:
            return ()
        if isinstance(statement, ast.If):
            return self._visit_if(
                statement,
                incoming=incoming,
                loop_ctx=loop_ctx,
                active_branch_ids=active_branch_ids,
            )
        if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
            return self._visit_loop(
                statement,
                incoming=incoming,
                loop_ctx=loop_ctx,
                active_branch_ids=active_branch_ids,
            )
        if isinstance(statement, ast.Return):
            block_id = self._new_block(
                kind=ControlFlowBlockKind.RETURN,
                label="return",
                node=statement,
                branch_ids=active_branch_ids,
            )
            self._connect_all(incoming, block_id)
            self._add_edge(block_id, self.exit_block_id, kind="return")
            return ()
        if isinstance(statement, ast.Break) and loop_ctx is not None:
            block_id = self._new_block(
                kind=ControlFlowBlockKind.BREAK,
                label="break",
                node=statement,
                branch_ids=active_branch_ids,
            )
            self._connect_all(incoming, block_id)
            self._add_edge(block_id, loop_ctx.break_target, kind="break")
            return ()
        if isinstance(statement, ast.Continue) and loop_ctx is not None:
            block_id = self._new_block(
                kind=ControlFlowBlockKind.CONTINUE,
                label="continue",
                node=statement,
                branch_ids=active_branch_ids,
            )
            self._connect_all(incoming, block_id)
            self._add_edge(block_id, loop_ctx.continue_target, kind="continue")
            return ()
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            block_id = self._new_block(
                kind=ControlFlowBlockKind.BASIC,
                label="with",
                node=statement,
                branch_ids=active_branch_ids,
                notes=("context-manager header",),
            )
            self._connect_all(incoming, block_id)
            return self._visit_body(
                statement.body,
                incoming=(block_id,),
                loop_ctx=loop_ctx,
                active_branch_ids=active_branch_ids,
            )
        if isinstance(statement, ast.Try):
            block_id = self._new_block(
                kind=ControlFlowBlockKind.BASIC,
                label="try",
                node=statement,
                branch_ids=active_branch_ids,
                notes=("collapsed try region",),
            )
            self._connect_all(incoming, block_id)
            exits: tuple[str, ...] = self._visit_body(
                statement.body,
                incoming=(block_id,),
                loop_ctx=loop_ctx,
                active_branch_ids=active_branch_ids,
            )
            handler_exits: list[str] = []
            for handler in statement.handlers:
                handler_exits.extend(
                    self._visit_body(
                        handler.body,
                        incoming=(block_id,),
                        loop_ctx=loop_ctx,
                        active_branch_ids=active_branch_ids,
                    )
                )
            merged = self._dedupe_block_ids(exits + tuple(handler_exits))
            if statement.orelse:
                merged = self._visit_body(
                    statement.orelse,
                    incoming=merged or (block_id,),
                    loop_ctx=loop_ctx,
                    active_branch_ids=active_branch_ids,
                )
            if statement.finalbody:
                merged = self._visit_body(
                    statement.finalbody,
                    incoming=merged or (block_id,),
                    loop_ctx=loop_ctx,
                    active_branch_ids=active_branch_ids,
                )
            return merged or (block_id,)
        block_id = self._new_block(
            kind=ControlFlowBlockKind.BASIC,
            label=statement.__class__.__name__.lower(),
            node=statement,
            branch_ids=active_branch_ids,
        )
        self._connect_all(incoming, block_id)
        return (block_id,)

    def _visit_if(
        self,
        statement: ast.If,
        *,
        incoming: tuple[str, ...],
        loop_ctx: _LoopContext | None,
        active_branch_ids: tuple[int, ...],
    ) -> tuple[str, ...]:
        branch_id = _extract_mark_cond_branch_id(statement.test)
        branch_ids = active_branch_ids + ((branch_id,) if branch_id is not None else ())
        header_block_id = self._new_block(
            kind=ControlFlowBlockKind.BRANCH_HEADER,
            label="if",
            node=statement,
            branch_ids=branch_ids,
            notes=(f"branch_id={branch_id}",) if branch_id is not None else (),
        )
        self._connect_all(incoming, header_block_id)
        true_exits = self._visit_body(
            statement.body,
            incoming=(header_block_id,),
            loop_ctx=loop_ctx,
            active_branch_ids=branch_ids,
        )
        if statement.orelse:
            false_exits = self._visit_body(
                statement.orelse,
                incoming=(header_block_id,),
                loop_ctx=loop_ctx,
                active_branch_ids=branch_ids,
            )
        else:
            false_exits = (header_block_id,)
        outgoing = self._dedupe_block_ids(true_exits + false_exits)
        if not outgoing:
            return ()
        merge_block_id = self._new_block(
            kind=ControlFlowBlockKind.MERGE,
            label="if_merge",
            node=statement,
            branch_ids=active_branch_ids,
        )
        for block_id in outgoing:
            if block_id == header_block_id:
                self._add_edge(block_id, merge_block_id, kind="false")
            else:
                self._add_edge(block_id, merge_block_id)
        return (merge_block_id,)

    def _visit_loop(
        self,
        statement: ast.For | ast.AsyncFor | ast.While,
        *,
        incoming: tuple[str, ...],
        loop_ctx: _LoopContext | None,
        active_branch_ids: tuple[int, ...],
    ) -> tuple[str, ...]:
        header_block_id = self._new_block(
            kind=ControlFlowBlockKind.LOOP_HEADER,
            label=statement.__class__.__name__.lower(),
            node=statement,
            branch_ids=active_branch_ids,
        )
        self._connect_all(incoming, header_block_id)
        after_loop_block_id = self._new_block(
            kind=ControlFlowBlockKind.MERGE,
            label="loop_exit",
            node=statement,
            branch_ids=active_branch_ids,
        )
        nested_loop_ctx = _LoopContext(
            break_target=after_loop_block_id,
            continue_target=header_block_id,
        )
        body_exits = self._visit_body(
            statement.body,
            incoming=(header_block_id,),
            loop_ctx=nested_loop_ctx,
            active_branch_ids=active_branch_ids,
        )
        for block_id in body_exits:
            self._add_edge(block_id, header_block_id, kind="backedge")
        self._add_edge(header_block_id, after_loop_block_id, kind="loop_exit")
        exits = (after_loop_block_id,)
        if statement.orelse:
            exits = self._visit_body(
                statement.orelse,
                incoming=exits,
                loop_ctx=loop_ctx,
                active_branch_ids=active_branch_ids,
            )
        return exits


_CPP_IF_RE = re.compile(r"^\s*if\s*\(")
_CPP_LOOP_RE = re.compile(r"^\s*(?:for|while)\s*\(")


class _CppFunctionCFGBuilder:
    """Small C++ CFG fallback for selected fixture-scale functions.

    This intentionally covers source-window analysis fixtures and extension
    binding shims. It is not a replacement for a Clang CFG when a compile
    database is available.
    """

    def __init__(self, *, source_path: Path, function_name: str) -> None:
        self.source_path = source_path
        self.function_name = function_name
        self._source_token = _path_token(source_path)
        self._block_order = 0
        self._blocks: list[ControlFlowBlock] = []
        self._edges: list[ControlFlowEdge] = []
        self._edge_keys: set[tuple[str, str, str]] = set()
        self.entry_block_id = self._new_block(
            kind=ControlFlowBlockKind.ENTRY,
            label="entry",
            lineno=0,
            end_lineno=None,
            notes=("synthetic cfg entry",),
        )
        self.exit_block_id = self._new_block(
            kind=ControlFlowBlockKind.EXIT,
            label="exit",
            lineno=0,
            end_lineno=None,
            notes=("synthetic cfg exit",),
        )

    def build(self, lines: tuple[tuple[int, str], ...]) -> ControlFlowGraph:
        body_lines = self._function_body_lines(lines)
        exits = self._visit_lines(body_lines, incoming=(self.entry_block_id,))
        for block_id in exits:
            self._add_edge(block_id, self.exit_block_id)
        return ControlFlowGraph(
            source_path=str(self.source_path),
            function_name=self.function_name,
            entry_block_id=self.entry_block_id,
            exit_block_id=self.exit_block_id,
            blocks=tuple(self._blocks),
            edges=tuple(self._edges),
            notes=(
                "lightweight selected-scope C++ control-flow graph",
                "line-based fallback; use clang cfg when compile context is available",
            ),
        )

    def _next_block_id(self) -> tuple[str, int]:
        self._block_order += 1
        return (
            f"cfg::{self._source_token}::{self.function_name.replace('.', '_')}::{self._block_order:03d}",
            self._block_order,
        )

    def _new_block(
        self,
        *,
        kind: ControlFlowBlockKind,
        label: str,
        lineno: int,
        end_lineno: int | None = None,
        notes: tuple[str, ...] = (),
    ) -> str:
        block_id, order_index = self._next_block_id()
        self._blocks.append(
            ControlFlowBlock(
                block_id=block_id,
                function_name=self.function_name,
                source_path=str(self.source_path),
                kind=kind,
                label=label,
                order_index=order_index,
                lineno=int(lineno),
                end_lineno=end_lineno,
                notes=notes,
            )
        )
        return block_id

    def _add_edge(self, src: str, dst: str, *, kind: str = "normal") -> None:
        key = (src, dst, kind)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self._edges.append(ControlFlowEdge(src, dst, kind))

    def _connect_incoming(self, incoming: tuple[str, ...], block_id: str, *, kind: str = "normal") -> None:
        for predecessor_id in incoming:
            self._add_edge(predecessor_id, block_id, kind=kind)

    def _function_body_lines(self, lines: tuple[tuple[int, str], ...]) -> tuple[tuple[int, str], ...]:
        first_open_index: int | None = None
        for index, (_, text) in enumerate(lines):
            if "{" in text:
                first_open_index = index
                break
        if first_open_index is None:
            return ()
        body = list(lines[first_open_index + 1 :])
        while body and not body[-1][1].strip():
            body.pop()
        if body and body[-1][1].strip() == "}":
            body.pop()
        return tuple(body)

    def _meaningful_line(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        return stripped not in {"{", "}", "};"}

    def _visit_lines(
        self,
        lines: tuple[tuple[int, str], ...],
        *,
        incoming: tuple[str, ...],
    ) -> tuple[str, ...]:
        exits = incoming
        index = 0
        while index < len(lines):
            lineno, text = lines[index]
            stripped = text.strip()
            if not self._meaningful_line(text):
                index += 1
                continue
            if stripped.startswith("else"):
                index += 1
                continue
            if _CPP_IF_RE.match(stripped):
                exits, index = self._visit_if(lines, index=index, incoming=exits)
                continue
            if _CPP_LOOP_RE.match(stripped):
                exits, index = self._visit_loop(lines, index=index, incoming=exits)
                continue
            if stripped.startswith("return"):
                return_block_id = self._new_block(
                    kind=ControlFlowBlockKind.RETURN,
                    label=stripped,
                    lineno=lineno,
                    notes=("cpp return statement",),
                )
                self._connect_incoming(exits, return_block_id)
                self._add_edge(return_block_id, self.exit_block_id, kind="return")
                exits = ()
                index += 1
                continue

            start_index = index
            end_index = index
            labels: list[str] = []
            while end_index < len(lines):
                current_lineno, current_text = lines[end_index]
                current = current_text.strip()
                if (
                    not self._meaningful_line(current_text)
                    or current.startswith("else")
                    or _CPP_IF_RE.match(current)
                    or _CPP_LOOP_RE.match(current)
                    or current.startswith("return")
                ):
                    break
                labels.append(current)
                end_index += 1
            if end_index == start_index:
                index += 1
                continue
            block_id = self._new_block(
                kind=ControlFlowBlockKind.BASIC,
                label="; ".join(labels),
                lineno=lineno,
                end_lineno=lines[end_index - 1][0],
                notes=("cpp basic statement block",),
            )
            self._connect_incoming(exits, block_id)
            exits = (block_id,)
            index = end_index
        return exits

    def _find_braced_range(
        self,
        lines: tuple[tuple[int, str], ...],
        header_index: int,
    ) -> tuple[int, int, int]:
        found_open = False
        depth = 0
        body_start = header_index + 1
        for index in range(header_index, len(lines)):
            text = lines[index][1]
            if "{" in text and not found_open:
                found_open = True
                body_start = index + 1
            if found_open:
                depth += text.count("{") - text.count("}")
                if depth <= 0:
                    return body_start, index, index
            elif index > header_index and self._meaningful_line(text):
                return index, index + 1, index
        return body_start, body_start, header_index

    def _find_else_range(
        self,
        lines: tuple[tuple[int, str], ...],
        after_index: int,
    ) -> tuple[int, int, int] | None:
        index = after_index
        while index < len(lines) and not lines[index][1].strip():
            index += 1
        if index >= len(lines):
            return None
        stripped = lines[index][1].strip()
        if not stripped.startswith("else"):
            return None
        body_start, body_end, close_index = self._find_braced_range(lines, index)
        return body_start, body_end, close_index

    def _visit_if(
        self,
        lines: tuple[tuple[int, str], ...],
        *,
        index: int,
        incoming: tuple[str, ...],
    ) -> tuple[tuple[str, ...], int]:
        lineno, text = lines[index]
        header_block_id = self._new_block(
            kind=ControlFlowBlockKind.BRANCH_HEADER,
            label=text.strip(),
            lineno=lineno,
            notes=("cpp if header",),
        )
        self._connect_incoming(incoming, header_block_id)
        body_start, body_end, close_index = self._find_braced_range(lines, index)
        then_exits = self._visit_lines(tuple(lines[body_start:body_end]), incoming=(header_block_id,))
        else_range = self._find_else_range(lines, close_index + 1)
        if else_range is None:
            else_exits = (header_block_id,)
            next_index = close_index + 1
        else:
            else_start, else_end, else_close = else_range
            else_exits = self._visit_lines(tuple(lines[else_start:else_end]), incoming=(header_block_id,))
            next_index = else_close + 1
        merge_block_id = self._new_block(
            kind=ControlFlowBlockKind.MERGE,
            label=f"merge if@{lineno}",
            lineno=lineno,
            notes=("cpp if merge",),
        )
        for block_id in then_exits:
            self._add_edge(block_id, merge_block_id, kind="true")
        for block_id in else_exits:
            self._add_edge(block_id, merge_block_id, kind="false")
        return (merge_block_id,), next_index

    def _visit_loop(
        self,
        lines: tuple[tuple[int, str], ...],
        *,
        index: int,
        incoming: tuple[str, ...],
    ) -> tuple[tuple[str, ...], int]:
        lineno, text = lines[index]
        header_block_id = self._new_block(
            kind=ControlFlowBlockKind.LOOP_HEADER,
            label=text.strip(),
            lineno=lineno,
            notes=("cpp loop header",),
        )
        self._connect_incoming(incoming, header_block_id)
        body_start, body_end, close_index = self._find_braced_range(lines, index)
        body_exits = self._visit_lines(tuple(lines[body_start:body_end]), incoming=(header_block_id,))
        for block_id in body_exits:
            self._add_edge(block_id, header_block_id, kind="backedge")
        after_loop_block_id = self._new_block(
            kind=ControlFlowBlockKind.MERGE,
            label=f"loop exit@{lineno}",
            lineno=lineno,
            notes=("cpp loop exit merge",),
        )
        self._add_edge(header_block_id, after_loop_block_id, kind="loop_exit")
        return (after_loop_block_id,), close_index + 1


def build_python_control_flow_graphs(
    code_path: str | Path,
    *,
    selected_functions: tuple[str, ...] = (),
) -> tuple[ControlFlowGraph, ...]:
    source_path = Path(code_path)
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    graphs: list[ControlFlowGraph] = []
    for function_name, body in _selected_python_roots(module, selected_functions):
        builder = _FunctionCFGBuilder(
            source_path=source_path,
            function_name=function_name,
        )
        graphs.append(builder.build(body))
    return tuple(graphs)


def build_cpp_control_flow_graphs(
    code_path: str | Path,
    *,
    selected_functions: tuple[str, ...] = (),
) -> tuple[ControlFlowGraph, ...]:
    source_path = Path(code_path)
    graphs: list[ControlFlowGraph] = []
    for block in cpp_function_blocks(source_path, selected_functions):
        builder = _CppFunctionCFGBuilder(
            source_path=source_path,
            function_name=block.name,
        )
        graphs.append(builder.build(block.scan_body_lines or block.body_lines))
    return tuple(graphs)


def build_control_flow_graphs(
    code_path: str | Path,
    *,
    selected_functions: tuple[str, ...] = (),
) -> tuple[ControlFlowGraph, ...]:
    source_path = Path(code_path)
    if is_cpp_like_path(source_path):
        return build_cpp_control_flow_graphs(source_path, selected_functions=selected_functions)
    return build_python_control_flow_graphs(source_path, selected_functions=selected_functions)
