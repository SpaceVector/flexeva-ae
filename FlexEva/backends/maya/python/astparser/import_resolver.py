"""
Import resolution utilities for multi-file analysis.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Optional, Set


def resolve_import_to_file(module_name: str, current_file: Path, search_paths: list[Path] | None = None) -> Optional[Path]:
    """
    Resolve a module name to a file path.

    Parameters
    ----------
    module_name : str
        The module name to resolve (e.g., "mymodule" or "mypackage.mymodule")
    current_file : Path
        The file where the import occurs (used to resolve relative imports)
    search_paths : list[Path] | None, optional
        Additional search paths to check. If None, uses current file's directory.

    Returns
    -------
    Optional[Path]
        The resolved file path, or None if not found or is a builtin/third-party module
    """
    # First, try resolving as a local file relative to current file
    # This handles the common case of local modules
    if search_paths is None:
        search_paths = [current_file.parent]

    for search_path in search_paths:
        # Try direct file match (e.g., import mymodule -> mymodule.py)
        candidate = search_path / f"{module_name}.py"
        if candidate.exists() and candidate.is_file():
            return candidate

        # Try package structure (module_name/__init__.py)
        candidate = search_path / module_name / "__init__.py"
        if candidate.exists() and candidate.is_file():
            return candidate

        # Try splitting on dots for nested modules (e.g., mypackage.mymodule)
        parts = module_name.split(".")
        candidate_dir = search_path
        for i, part in enumerate(parts):
            candidate_dir = candidate_dir / part
            # Check if it's a package with __init__.py
            candidate_file = candidate_dir / "__init__.py"
            if candidate_file.exists() and candidate_file.is_file():
                return candidate_file
            # Check if it's the final part and is a .py file
            if i == len(parts) - 1 and candidate_dir.is_file() and candidate_dir.suffix == ".py":
                return candidate_dir

    # Try to find the module using importlib (for installed packages)
    # But we'll skip this for third-party modules
    if is_third_party_module(module_name):
        return None

    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            return None

        origin = Path(spec.origin)
        # Skip if it's a builtin module (no file) or a package directory without __init__.py
        if not origin.exists():
            return None

        if origin.is_dir():
            # Check if it's a package with __init__.py
            init_file = origin / "__init__.py"
            if init_file.exists():
                return init_file
            return None

        # Return the file path
        if origin.is_file() and origin.suffix == ".py":
            return origin

    except (ImportError, ValueError, ModuleNotFoundError):
        # Module not found - might be a local file that we already checked
        pass

    return None


def extract_imports_from_ast(tree: ast.AST) -> Set[str]:
    """
    Extract all import statements from an AST.

    Parameters
    ----------
    tree : ast.AST
        The AST to analyze

    Returns
    -------
    Set[str]
        Set of module names being imported
    """
    imports: Set[str] = set()

    class ImportVisitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                imports.add(alias.name)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.module:
                imports.add(node.module)

    visitor = ImportVisitor()
    visitor.visit(tree)
    return imports


def is_third_party_module(module_name: str) -> bool:
    """
    Check if a module is likely a third-party module (not user code).

    This is a heuristic - we check common third-party package prefixes.

    Parameters
    ----------
    module_name : str
        The module name to check

    Returns
    -------
    bool
        True if likely third-party, False otherwise
    """
    third_party_prefixes = {
        "torch",
        "numpy",
        "pandas",
        "tensorflow",
        "transformers",
        "megatron",
        "deepspeed",
        "pytorch",
        "sklearn",
        "scipy",
        "matplotlib",
        "PIL",
        "cv2",
        "tqdm",
    }

    first_part = module_name.split(".")[0]
    return first_part in third_party_prefixes

