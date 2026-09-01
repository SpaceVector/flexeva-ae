"""
Code generation module for transforming user code with branch instrumentation.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Set

from astparser.branch_point import BranchPoint, FlexDecorator

# Fixed output path for generated code (for pyextend to consume)
DEFAULT_OUTPUT_PATH = Path("build/astparser/generated_code.py")


class BranchInstrumenter(ast.NodeTransformer):
    """
    AST transformer that instruments branch points for taint tracking.

    Wraps branch conditions with instrumentation calls while preserving
    the original code structure.
    """

    def __init__(self, branch_map: Dict[int, BranchPoint]) -> None:
        """
        Initialize the instrumenter.

        Parameters
        ----------
        branch_map : Dict[int, BranchPoint]
            Mapping from line numbers to branch points
        """
        self.branch_map = branch_map

    def visit_If(self, node: ast.If) -> None:
        """Instrument if statements."""
        # Check if this if statement should be instrumented
        if node.lineno in self.branch_map:
            branch = self.branch_map[node.lineno]
            # Wrap the condition with instrumentation
            node.test = self._wrap_condition(node.test, branch.lineno)

        # Visit children
        self.generic_visit(node)
        return node

    def visit_While(self, node: ast.While) -> None:
        """Instrument while loops."""
        if node.lineno in self.branch_map:
            branch = self.branch_map[node.lineno]
            node.test = self._wrap_condition(node.test, branch.lineno)

        self.generic_visit(node)
        return node

    def visit_IfExp(self, node: ast.IfExp) -> None:
        """Instrument ternary expressions."""
        if node.lineno in self.branch_map:
            branch = self.branch_map[node.lineno]
            node.test = self._wrap_condition(node.test, branch.lineno)

        self.generic_visit(node)
        return node

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Instrument list comprehension filters."""
        self._instrument_comprehension(node)
        return node

    def visit_SetComp(self, node: ast.SetComp) -> None:
        """Instrument set comprehension filters."""
        self._instrument_comprehension(node)
        return node

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """Instrument dict comprehension filters."""
        self._instrument_comprehension(node)
        return node

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        """Instrument generator expression filters."""
        self._instrument_comprehension(node)
        return node

    def _instrument_comprehension(self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp) -> None:
        """Instrument comprehension filter conditions."""
        for gen in node.generators:
            for i, if_cond in enumerate(gen.ifs):
                if if_cond.lineno in self.branch_map:
                    branch = self.branch_map[if_cond.lineno]
                    gen.ifs[i] = self._wrap_condition(if_cond, branch.lineno)

    def visit_Match(self, node: ast.Match) -> None:
        """Instrument match/case statements."""
        for case in node.cases:
            # Instrument guard conditions
            if case.guard and case.guard.lineno in self.branch_map:
                branch = self.branch_map[case.guard.lineno]
                case.guard = self._wrap_condition(case.guard, branch.lineno)
        self.generic_visit(node)
        return node

    def visit_For(self, node: ast.For) -> None:
        """Instrument for loops (record iterable as potential branch point)."""
        if node.lineno in self.branch_map:
            # For loops are recorded, but we don't wrap the iterable
            # The iterable itself might be rank-dependent, but we can't easily instrument it
            # The branch point is recorded for analysis purposes
            pass
        self.generic_visit(node)
        return node

    def _wrap_condition(self, condition: ast.expr, lineno: int) -> ast.Call:
        """
        Wrap a condition expression with mark_cond instrumentation.

        Transforms: `condition` -> `mark_cond(condition, lineno)`

        Parameters
        ----------
        condition : ast.expr
            The condition expression to wrap
        lineno : int
            Line number for the instrumentation

        Returns
        -------
        ast.Call
            Call node: mark_cond(condition, lineno)
        """
        # Create: mark_cond(condition, lineno)
        return ast.Call(
            func=ast.Name(id="mark_cond", ctx=ast.Load()),
            args=[condition, ast.Constant(value=lineno)],
            keywords=[],
        )


def instrument_file(
    source: str, branch_points: List[BranchPoint], filename: str = "<unknown>"
) -> ast.Module:
    """
    Instrument a Python source file with branch point tracking.

    Parameters
    ----------
    source : str
        Source code to instrument
    branch_points : List[BranchPoint]
        List of branch points to instrument
    filename : str, optional
        Filename for error reporting

    Returns
    -------
    ast.Module
        Instrumented AST module
    """
    # Parse the source
    tree = ast.parse(source, filename=filename)

    # Create mapping from line numbers to branch points
    branch_map: Dict[int, BranchPoint] = {}
    for branch in branch_points:
        branch_map[branch.lineno] = branch

    # Instrument the AST
    instrumenter = BranchInstrumenter(branch_map)
    instrumented_tree = instrumenter.visit(tree)

    # Ensure we have a Module node
    if not isinstance(instrumented_tree, ast.Module):
        instrumented_tree = ast.Module(body=[instrumented_tree], type_ignores=[])

    return instrumented_tree


def combine_files(
    file_contents: Dict[Path, str],
    branch_points: List[BranchPoint],
    decorators: List[FlexDecorator],
) -> str:
    """
    Combine multiple entry files into a single Python file.

    Parameters
    ----------
    file_contents : Dict[Path, str]
        Mapping from file paths to their source code
    branch_points : List[BranchPoint]
        All branch points from all files
    decorators : List[FlexDecorator]
        All decorators from all files

    Returns
    -------
    str
        Combined and instrumented Python source code
    """
    # Group branch points by file
    branches_by_file: Dict[Path, List[BranchPoint]] = {}
    for branch in branch_points:
        if branch.file_path is not None:
            resolved_path = branch.file_path.resolve()
            if resolved_path not in branches_by_file:
                branches_by_file[resolved_path] = []
            branches_by_file[resolved_path].append(branch)

    # Process each file: instrument and collect ASTs
    instrumented_modules: List[ast.Module] = []
    all_imports: List[ast.stmt] = []
    # Track imports by (module, imported_name, asname) to handle duplicates correctly
    imports_seen: Set[tuple[str | None, str, str | None]] = set()  # (module, name, asname) tuples

    for file_path, source in file_contents.items():
        resolved_path = file_path.resolve()
        file_branches = branches_by_file.get(resolved_path, [])

        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as e:
            # Log but continue - syntax errors should have been caught earlier
            import sys
            print(
                f"Warning: Skipping file {file_path} due to syntax error: {e}",
                file=sys.stderr
            )
            continue
        except Exception as e:
            # Log but continue for other parsing errors
            import sys
            print(
                f"Warning: Skipping file {file_path} due to parsing error: {e}",
                file=sys.stderr
            )
            continue

        # Extract and deduplicate imports
        # Visit top-level imports only (not nested in functions/classes)
        for stmt in tree.body:
            if isinstance(stmt, ast.Import):
                # Collect unique aliases from this import statement
                aliases_to_add = []
                for alias in stmt.names:
                    # For "import module as alias", key is (None, module_name, asname)
                    key = (None, alias.name, alias.asname)
                    if key not in imports_seen:
                        imports_seen.add(key)
                        aliases_to_add.append(alias)
                if aliases_to_add:
                    all_imports.append(ast.Import(names=aliases_to_add))
            elif isinstance(stmt, ast.ImportFrom):
                if stmt.module:
                    # Collect unique aliases from this import-from statement
                    aliases_to_add = []
                    for alias in stmt.names:
                        # For "from module import name as alias", key is (module, name, asname)
                        key = (stmt.module, alias.name, alias.asname)
                        if key not in imports_seen:
                            imports_seen.add(key)
                            aliases_to_add.append(alias)
                    if aliases_to_add:
                        all_imports.append(
                            ast.ImportFrom(
                                module=stmt.module,
                                names=aliases_to_add,
                                level=stmt.level,
                            )
                        )

        # Instrument this file's AST
        instrumented = instrument_file(source, file_branches, str(file_path))
        instrumented_modules.append(instrumented)

    # Combine all modules into one
    combined_body: List[ast.stmt] = []

    # Add mark_cond import (if not already present)
    mark_cond_key = ("pyextend.runtime.rvariant", "mark_cond", None)
    if mark_cond_key not in imports_seen:
        combined_body.append(
            ast.ImportFrom(
                module="pyextend.runtime.rvariant",
                names=[ast.alias(name="mark_cond", asname=None)],
                level=0,
            )
        )

    # Add other imports
    combined_body.extend(all_imports)

    # Add all instrumented code (excluding imports which we already handled)
    for module in instrumented_modules:
        for stmt in module.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                # Skip imports (we handle them separately)
                continue
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                # Skip module docstrings (they're Expr nodes with string constants)
                # We could preserve the first one, but for simplicity, skip all
                continue
            else:
                combined_body.append(stmt)

    # Create final module
    combined_module = ast.Module(body=combined_body, type_ignores=[])

    # Convert to source code
    try:
        return ast.unparse(combined_module)
    except AttributeError:
        # Fallback for Python < 3.9 - use a simple string builder
        # This is a basic fallback; for production, consider using astor
        lines: List[str] = []
        for stmt in combined_body:
            if isinstance(stmt, ast.ImportFrom):
                names = ", ".join(alias.name if alias.asname is None else f"{alias.name} as {alias.asname}" for alias in stmt.names)
                lines.append(f"from {stmt.module} import {names}")
            elif isinstance(stmt, ast.Import):
                names = ", ".join(alias.name if alias.asname is None else f"{alias.name} as {alias.asname}" for alias in stmt.names)
                lines.append(f"import {names}")
            else:
                # For non-import statements, we'd need a full AST unparser
                # For now, just add a comment
                lines.append(f"# {type(stmt).__name__}")
        return "\n".join(lines)


def generate_code(
    entry_files: List[Path],
    branch_points: List[BranchPoint],
    decorators: List[FlexDecorator],
    output_path: Path,
) -> None:
    """
    Generate instrumented code and write to output file.

    Parameters
    ----------
    entry_files : List[Path]
        List of entry file paths
    branch_points : List[BranchPoint]
        All branch points from all files
    decorators : List[FlexDecorator]
        All decorators from all files
    output_path : Path
        Path to write the generated code
    """
    # Read all entry files
    file_contents: Dict[Path, str] = {}
    for file_path in entry_files:
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="latin-1")
        file_contents[file_path] = content

    # Combine and instrument
    combined_code = combine_files(file_contents, branch_points, decorators)

    # Write to output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(combined_code, encoding="utf-8")

