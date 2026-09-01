"""
Main parser interface for analyzing Python files and extracting branch points.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Set

from astparser.branch_point import BranchPoint, FlexDecorator
from astparser.branch_visitor import BranchVisitor
from astparser.import_resolver import (
    extract_imports_from_ast,
    is_third_party_module,
    resolve_import_to_file,
)


def parse_file(
    file_path: Path | str, entry_files: Set[Path] | None = None
) -> tuple[List[BranchPoint], List[FlexDecorator]]:
    """
    Parse a Python file and extract all branch points and decorators.

    Parameters
    ----------
    file_path : Path | str
        Path to the Python file to parse
    entry_files : Set[Path] | None, optional
        Set of entry files. If provided, only files in this set will be analyzed.
        If None, only the specified file_path is analyzed.

    Returns
    -------
    tuple[List[BranchPoint], List[FlexDecorator]]
        Tuple of (branch points, decorators) found in the file

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    SyntaxError
        If the file contains invalid Python syntax
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # If entry_files is specified, check if this file is in the set
    if entry_files is not None:
        if file_path.resolve() not in {p.resolve() for p in entry_files}:
            return [], []  # Skip non-entry files

    # Read and parse the file
    try:
        source = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            source = file_path.read_text(encoding="latin-1")
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to read file {file_path}: {e}") from e

    if not source.strip():
        # Empty file - return empty results
        return [], []

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        raise SyntaxError(
            f"Syntax error in {file_path} at line {e.lineno}: {e.msg}\n"
            f"  {e.text or ''}"
        ) from e
    except Exception as e:
        raise ValueError(f"Failed to parse AST for {file_path}: {e}") from e

    # Visit the AST to find branch points and decorators
    visitor = BranchVisitor()
    visitor.visit(tree)

    # Associate branch points and decorators with the file path
    for branch in visitor.branches:
        branch.file_path = file_path.resolve()
    # Note: decorators don't need file_path for now, but could be added if needed

    return visitor.branches, visitor.decorators


def parse_entry_files(entry_files: List[Path | str]) -> tuple[List[BranchPoint], List[FlexDecorator]]:
    """
    Parse multiple entry files and collect all branch points and decorators.

    Follows imports into entry files only, skipping third-party and non-entry modules.

    Parameters
    ----------
    entry_files : List[Path | str]
        List of entry file paths to parse

    Returns
    -------
    tuple[List[BranchPoint], List[FlexDecorator]]
        Tuple of (all branch points, all decorators) from all entry files
    """
    entry_set = {Path(f).resolve() for f in entry_files}
    visited_files: Set[Path] = set()
    all_branches: List[BranchPoint] = []
    all_decorators: List[FlexDecorator] = []

    def parse_file_recursive(file_path: Path) -> None:
        """Recursively parse a file and follow imports into entry files."""
        resolved_path = file_path.resolve()

        # Skip if already visited (avoid cycles)
        if resolved_path in visited_files:
            return

        # Skip if not an entry file
        if resolved_path not in entry_set:
            return

        # Mark as visited
        visited_files.add(resolved_path)

        # Parse the file
        branches, decorators = parse_file(file_path, entry_files=entry_set)
        all_branches.extend(branches)
        all_decorators.extend(decorators)

        # Extract imports and follow them
        try:
            source = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                source = file_path.read_text(encoding="latin-1")
            except Exception:
                # Skip files we can't read
                return
        except Exception:
            # Skip files we can't read
            return

        if not source.strip():
            # Skip empty files
            return

        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError:
            # Skip if we can't parse (already handled in parse_file)
            return
        except Exception:
            # Skip files with other parsing errors
            return

        imports = extract_imports_from_ast(tree)

        # Resolve each import and follow if it's an entry file
        for module_name in imports:
            # Skip third-party modules
            if is_third_party_module(module_name):
                continue

            # Resolve import to file path
            imported_file = resolve_import_to_file(module_name, file_path)

            if imported_file is not None:
                resolved_import = imported_file.resolve()
                # Check if it's an entry file
                if resolved_import in entry_set:
                    # Recursively parse the imported entry file
                    parse_file_recursive(imported_file)

    # Parse all entry files
    for entry_file in entry_files:
        parse_file_recursive(Path(entry_file))

    return all_branches, all_decorators

