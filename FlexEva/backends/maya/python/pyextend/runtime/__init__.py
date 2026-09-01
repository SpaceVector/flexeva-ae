"""
Pyextend runtime module.

Provides:
- R-variant tracking and taint propagation
- SimEnv for simulation environment
- Branch signature collection
- Rank context management
- Dry-run execution framework
- Phase III branch classification and SPSD transformation
"""

from pyextend.runtime.rvariant import R, mark_cond
from pyextend.runtime.simenv import SimEnv
from pyextend.runtime.dryrun import DryRunExecutor
from pyextend.runtime.decorators import flex_rank, flex_group
from pyextend.runtime.phase3 import (
    BranchClassificationType,
    BranchClassification,
    ExecutionStep,
    classify_branch,
    classify_all_branches,
    build_execution_path,
)

__all__ = [
    "R", "mark_cond", "SimEnv", "DryRunExecutor", "flex_rank", "flex_group",
    "BranchClassificationType", "BranchClassification", "ExecutionStep",
    "classify_branch", "classify_all_branches", "build_execution_path",
]

