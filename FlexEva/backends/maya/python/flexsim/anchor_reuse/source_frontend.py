"""Shared source-front-end helpers for Python and limited C++ selected scopes."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path


_CPP_SUFFIXES = {".cc", ".cpp", ".cxx", ".c", ".hpp", ".hh", ".hxx", ".h"}
_CPP_FUNCTION_RE = re.compile(
    r"""
    ^
    \s*
    (?:
        (?:inline|static|virtual|constexpr|auto|const|unsigned|signed|long|short)\s+
    )*
    (?:
        [\w:\<\>\~\*&]+\s+
    )+
    (?P<name>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)
    \s*
    \(
        [^;]* 
    \)
    \s*
    (?:const\s*)?
    \{
    """,
    re.VERBOSE,
)
_CPP_CALL_RE = re.compile(r"\b(?P<name>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*\(")
_CPP_KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "catch"}
_CPP_HEADER_CONTROL_RE = re.compile(r"^(?:if|for|while|switch|catch)\s*\(")
_CPP_HEADER_LOOKAHEAD_LINES = 8


@dataclass(frozen=True)
class CppFunctionBlock:
    name: str
    source_path: Path
    lineno: int
    end_lineno: int
    body_lines: tuple[tuple[int, str], ...]
    scan_body_lines: tuple[tuple[int, str], ...] = ()

    @property
    def structure_signature(self) -> str:
        return f"fn:{self.name}"

    @property
    def block_text(self) -> str:
        return "\n".join(text for _, text in self.body_lines)


def is_cpp_like_path(path: Path) -> bool:
    return path.suffix.lower() in _CPP_SUFFIXES


def _strip_cpp_comments_preserve_lines(text: str) -> tuple[str, ...]:
    lines: list[str] = []
    in_block_comment = False
    for raw_line in text.splitlines():
        index = 0
        stripped: list[str] = []
        while index < len(raw_line):
            if in_block_comment:
                end_index = raw_line.find("*/", index)
                if end_index == -1:
                    index = len(raw_line)
                    continue
                in_block_comment = False
                index = end_index + 2
                continue
            if raw_line.startswith("//", index):
                break
            if raw_line.startswith("/*", index):
                in_block_comment = True
                index += 2
                continue
            stripped.append(raw_line[index])
            index += 1
        lines.append("".join(stripped))
    return tuple(lines)


def _match_cpp_function_header(
    scan_lines: tuple[str, ...],
    start_index: int,
) -> tuple[str, int] | None:
    header_lines: list[str] = []
    for end_index in range(start_index, min(len(scan_lines), start_index + _CPP_HEADER_LOOKAHEAD_LINES)):
        current = scan_lines[end_index].strip()
        if not current and not header_lines:
            return None
        if current.startswith("#"):
            return None
        header_lines.append(current)
        joined = " ".join(line for line in header_lines if line)
        if not joined:
            continue
        if _CPP_HEADER_CONTROL_RE.match(joined):
            return None
        if ";" in joined and "{" not in joined:
            return None
        if "{" not in joined:
            continue
        match = _CPP_FUNCTION_RE.match(joined)
        if match is None:
            return None
        return match.group("name").split("::")[-1], end_index
    return None


def defined_function_names_for_path(path: Path) -> tuple[str, ...]:
    if is_cpp_like_path(path):
        return tuple(block.name for block in cpp_function_blocks(path))
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
            continue
        if isinstance(node, ast.ClassDef):
            for statement in node.body:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.append(statement.name)
    return tuple(names)


def cpp_function_blocks(
    path: Path,
    selected_functions: tuple[str, ...] = (),
) -> tuple[CppFunctionBlock, ...]:
    source_text = path.read_text(encoding="utf-8")
    lines = source_text.splitlines()
    scan_lines = _strip_cpp_comments_preserve_lines(source_text)
    wanted = set(selected_functions)
    blocks: list[CppFunctionBlock] = []
    line_index = 0
    while line_index < len(lines):
        header_match = _match_cpp_function_header(scan_lines, line_index)
        if header_match is None:
            line_index += 1
            continue
        name, header_end_index = header_match
        brace_depth = 0
        start_index = line_index
        body: list[tuple[int, str]] = []
        scan_body: list[tuple[int, str]] = []
        for header_index in range(start_index, header_end_index + 1):
            body.append((header_index + 1, lines[header_index]))
            scan_body.append((header_index + 1, scan_lines[header_index]))
            brace_depth += scan_lines[header_index].count("{") - scan_lines[header_index].count("}")
        line_index = header_end_index + 1
        while line_index < len(lines) and brace_depth > 0:
            current = lines[line_index]
            scan_current = scan_lines[line_index]
            body.append((line_index + 1, current))
            scan_body.append((line_index + 1, scan_current))
            brace_depth += scan_current.count("{") - scan_current.count("}")
            line_index += 1
        if wanted and name not in wanted:
            continue
        blocks.append(
            CppFunctionBlock(
                name=name,
                source_path=path,
                lineno=start_index + 1,
                end_lineno=body[-1][0],
                body_lines=tuple(body),
                scan_body_lines=tuple(scan_body),
            )
        )
    return tuple(blocks)


def iter_cpp_opaque_calls(
    path: Path,
    *,
    selected_functions: tuple[str, ...],
    opaque_call_names: tuple[str, ...],
) -> tuple[tuple[CppFunctionBlock, int, str], ...]:
    opaque_names = set(opaque_call_names)
    rows: list[tuple[CppFunctionBlock, int, str]] = []
    for block in cpp_function_blocks(path, selected_functions):
        scan_body_lines = block.scan_body_lines or block.body_lines
        for lineno, line in scan_body_lines[1:]:
            for match in _CPP_CALL_RE.finditer(line):
                call_name = match.group("name").split("::")[-1]
                if call_name in _CPP_KEYWORDS or call_name not in opaque_names:
                    continue
                rows.append((block, lineno, call_name))
    return tuple(rows)
