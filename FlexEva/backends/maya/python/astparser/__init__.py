"""
astparser
=========

AST-based static analysis toolchain that transforms user training scripts to
clean SPMD simulation schema in FlexSim.

The package provides:
* `BranchVisitor` - AST visitor to enumerate all branch points
* `BranchPoint` - Data structures representing branch points
* `parse_file` - Main entry point for parsing user scripts
"""

from __future__ import annotations

__all__ = [
    "BranchPoint",
    "BranchVisitor",
    "FlexDecorator",
    "parse_file",
]

from astparser.branch_point import BranchPoint, FlexDecorator
from astparser.branch_visitor import BranchVisitor
from astparser.parser import parse_file

