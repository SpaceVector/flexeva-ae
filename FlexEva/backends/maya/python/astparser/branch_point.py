"""
Data structures for representing branch points and decorators detected in AST analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional


class BranchType(Enum):
    """Types of branch points in Python code."""

    IF = "if"
    ELIF = "elif"
    ELSE = "else"
    WHILE = "while"
    TERNARY = "ternary"
    COMPREHENSION_FILTER = "comprehension_filter"
    MATCH_CASE = "match_case"
    FOR_LOOP = "for_loop"


@dataclass
class BranchPoint:
    """
    Represents a branch point detected in the AST.

    Attributes
    ----------
    branch_type : BranchType
        Type of branch (if, elif, while, ternary)
    lineno : int
        Line number where the branch condition appears
    col_offset : int
        Column offset of the branch condition
    condition : str
        String representation of the condition expression
    is_nested : bool
        Whether this branch is nested inside another branch
    parent_lineno : Optional[int]
        Line number of parent branch if nested, None otherwise
    nest_path : List[int]
        List of line numbers representing the nesting path from root to immediate parent.
        Branches with the same nest_path belong to the same nest.
        Empty list means the branch is at the root level.
    file_path : Optional[Path]
        Path to the file where this branch point was found (for multi-file analysis)
    """

    branch_type: BranchType
    lineno: int
    col_offset: int
    condition: str
    is_nested: bool = False
    parent_lineno: Optional[int] = None
    nest_path: Optional[List[int]] = None
    file_path: Optional[Path] = None

    def __post_init__(self) -> None:
        """Initialize nest_path to empty list if None."""
        if self.nest_path is None:
            self.nest_path = []

    def __repr__(self) -> str:
        nested_str = f" (nested under line {self.parent_lineno})" if self.is_nested else ""
        nest_path_str = f" nest_path={self.nest_path}" if self.nest_path else ""
        return (
            f"BranchPoint({self.branch_type.value}, line {self.lineno}, "
            f"condition={self.condition!r}{nested_str}{nest_path_str})"
        )


@dataclass
class FlexDecorator:
    """
    Represents a @flex_rank or @flex_group decorator found in the code.

    Attributes
    ----------
    decorator_type : str
        Either "flex_rank" or "flex_group"
    lineno : int
        Line number where the decorator appears
    col_offset : int
        Column offset of the decorator
    active_ranks : Optional[Any]
        For @flex_rank: the active_ranks argument value
    active_group : Optional[Any]
        For @flex_group: the active_group argument value (user explicitly defined process group)
    target_name : Optional[str]
        Name of the function/class being decorated
    """

    decorator_type: str  # "flex_rank" or "flex_group"
    lineno: int
    col_offset: int
    active_ranks: Optional[Any] = None
    active_group: Optional[Any] = None
    target_name: Optional[str] = None

    def __repr__(self) -> str:
        if self.decorator_type == "flex_rank":
            return f"FlexDecorator(flex_rank, line {self.lineno}, active_ranks={self.active_ranks}, target={self.target_name})"
        else:
            return f"FlexDecorator(flex_group, line {self.lineno}, active_group={self.active_group}, target={self.target_name})"

