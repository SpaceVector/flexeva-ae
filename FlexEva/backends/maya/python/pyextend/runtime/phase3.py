"""
Phase III: SPMD → SPSD Transformation

Transforms SPMD code to a single sequential execution path with dynamic active_ranks context.
"""

from typing import Dict, List, Tuple, Set, Optional
from enum import Enum
from dataclasses import dataclass


class BranchClassificationType(Enum):
    """Branch classification type."""
    WORLD_UNIFORM = "WORLD_UNIFORM"
    RANK_PARTITIONING = "RANK_PARTITIONING"


@dataclass
class BranchClassification:
    """Classification result for a branch."""
    branch_id: int
    branch_type: BranchClassificationType
    true_ranks: Set[int]
    false_ranks: Set[int]
    is_rtainted: bool


@dataclass
class ExecutionStep:
    """A single step in the sequential execution path."""
    active_ranks: Set[int]
    branch_id: Optional[int] = None
    path_type: Optional[str] = None  # "true", "false", or None for sequential code
    nested_steps: List['ExecutionStep'] = None  # Nested branches within this step
    
    def __post_init__(self):
        if self.nested_steps is None:
            self.nested_steps = []


def classify_branch(
    branch_id: int,
    branch_signatures: Dict[int, List[Tuple[int, bool, bool]]],
    active_ranks: Optional[Set[int]] = None
) -> BranchClassification:
    """
    Classify a branch as WORLD_UNIFORM or RANK_PARTITIONING.
    
    Parameters
    ----------
    branch_id : int
        Branch identifier (line number)
    branch_signatures : Dict[int, List[Tuple[int, bool, bool]]]
        Branch signatures from Phase II: rank -> list of (branch_id, outcome, is_rtainted)
    active_ranks : Optional[Set[int]]
        Current active ranks (if None, use all ranks that executed this branch)
    
    Returns
    -------
    BranchClassification
        Classification result
    """
    # Collect outcomes for this branch across all ranks
    true_ranks: Set[int] = set()
    false_ranks: Set[int] = set()
    is_rtainted = False
    
    # Determine active ranks if not provided
    if active_ranks is None:
        active_ranks = set(branch_signatures.keys())
    
    # Check each rank's signature for this branch
    for rank in active_ranks:
        if rank not in branch_signatures:
            continue
        
        signature = branch_signatures[rank]
        for bid, outcome, is_rt in signature:
            if bid == branch_id:
                if outcome:
                    true_ranks.add(rank)
                else:
                    false_ranks.add(rank)
                if is_rt:
                    is_rtainted = True
                break
    
    # Classify based on whether all active ranks see the same outcome
    if len(true_ranks) == 0 or len(false_ranks) == 0:
        # All ranks take the same path
        branch_type = BranchClassificationType.WORLD_UNIFORM
    else:
        # Ranks are split
        branch_type = BranchClassificationType.RANK_PARTITIONING
    
    return BranchClassification(
        branch_id=branch_id,
        branch_type=branch_type,
        true_ranks=true_ranks,
        false_ranks=false_ranks,
        is_rtainted=is_rtainted
    )


def classify_all_branches(
    branch_signatures: Dict[int, List[Tuple[int, bool, bool]]],
    active_ranks: Optional[Set[int]] = None
) -> Dict[int, BranchClassification]:
    """
    Classify all branches found in branch signatures.
    
    Parameters
    ----------
    branch_signatures : Dict[int, List[Tuple[int, bool, bool]]]
        Branch signatures from Phase II
    active_ranks : Optional[Set[int]]
        Current active ranks (if None, use all ranks)
    
    Returns
    -------
    Dict[int, BranchClassification]
        Mapping from branch_id to classification
    """
    # Collect all unique branch IDs
    all_branch_ids: Set[int] = set()
    for signature in branch_signatures.values():
        for branch_id, _, _ in signature:
            all_branch_ids.add(branch_id)
    
    # Classify each branch
    classifications: Dict[int, BranchClassification] = {}
    for branch_id in sorted(all_branch_ids):
        classifications[branch_id] = classify_branch(
            branch_id, branch_signatures, active_ranks
        )
    
    return classifications


def build_execution_path(
    branch_signatures: Dict[int, List[Tuple[int, bool, bool]]],
    classifications: Dict[int, BranchClassification],
    branch_metadata: Dict[int, any],  # BranchPoint objects from astparser
    initial_ranks: Set[int]
) -> List[ExecutionStep]:
    """
    Build a single sequential execution path with dynamic active_ranks.
    
    Parameters
    ----------
    branch_signatures : Dict[int, List[Tuple[int, bool, bool]]]
        Branch signatures from Phase II
    classifications : Dict[int, BranchClassification]
        Branch classifications
    branch_metadata : Dict[int, any]
        Branch metadata from astparser (BranchPoint objects with nest_path and parent_lineno)
    initial_ranks : Set[int]
        Initial set of active ranks
    
    Returns
    -------
    List[ExecutionStep]
        Sequential execution path
    """
    # Get all branch IDs in execution order (by line number)
    all_branch_ids = sorted(classifications.keys())
    
    # Build execution path recursively
    execution_path: List[ExecutionStep] = []
    remaining_branches = set(all_branch_ids)
    
    def process_branch(
        branch_id: int,
        current_ranks: Set[int],
        processed_branches: Set[int]
    ) -> List[ExecutionStep]:
        """Process a branch and return execution steps."""
        if branch_id in processed_branches:
            return []
        
        steps: List[ExecutionStep] = []
        cls = classifications[branch_id]
        metadata = branch_metadata.get(branch_id)
        
        # Check if this is an elif branch
        is_elif = metadata and hasattr(metadata, 'branch_type') and metadata.branch_type.value == 'elif'
        parent_lineno = metadata.parent_lineno if metadata else None
        
        # For elif branches, they execute in the false_path of their parent
        # We'll handle this when processing the parent
        
        # Check if this branch is nested
        nest_path = metadata.nest_path if metadata else []
        is_nested = len(nest_path) > 0
        
        if cls.branch_type == BranchClassificationType.RANK_PARTITIONING:
            # Partition ranks and execute both paths
            true_ranks = cls.true_ranks & current_ranks
            false_ranks = cls.false_ranks & current_ranks
            
            # True path
            if true_ranks:
                true_step = ExecutionStep(
                    active_ranks=true_ranks,
                    branch_id=branch_id,
                    path_type="true"
                )
                
                # Process nested branches within true path
                nested_steps = _process_nested_branches(
                    branch_id, true_ranks, remaining_branches, processed_branches,
                    classifications, branch_metadata
                )
                true_step.nested_steps = nested_steps
                steps.append(true_step)
            
            # False path
            if false_ranks:
                # Check for elif branches in false path
                elif_branches = _find_elif_branches(branch_id, branch_metadata, remaining_branches)
                
                if elif_branches:
                    # Process elif chain in false path
                    current_false_ranks = false_ranks
                    for elif_id in elif_branches:
                        elif_cls = classifications[elif_id]
                        elif_true_ranks = elif_cls.true_ranks & current_false_ranks
                        elif_false_ranks = elif_cls.false_ranks & current_false_ranks
                        
                        # Elif true path
                        if elif_true_ranks:
                            elif_true_step = ExecutionStep(
                                active_ranks=elif_true_ranks,
                                branch_id=elif_id,
                                path_type="true"
                            )
                            
                            # Process nested branches within elif true path
                            nested_steps = _process_nested_branches(
                                elif_id, elif_true_ranks, remaining_branches, processed_branches,
                                classifications, branch_metadata
                            )
                            elif_true_step.nested_steps = nested_steps
                            steps.append(elif_true_step)
                        
                        # Update for next elif or else
                        current_false_ranks = elif_false_ranks
                        processed_branches.add(elif_id)
                    
                    # Handle else block (remaining false_ranks after all elifs)
                    if current_false_ranks:
                        else_step = ExecutionStep(
                            active_ranks=current_false_ranks,
                            branch_id=branch_id,  # Use parent branch_id for else
                            path_type="false"
                        )
                        
                        # Process nested branches within else path
                        nested_steps = _process_nested_branches(
                            branch_id, current_false_ranks, remaining_branches, processed_branches,
                            classifications, branch_metadata
                        )
                        else_step.nested_steps = nested_steps
                        steps.append(else_step)
                else:
                    # Regular false path (no elif chain)
                    false_step = ExecutionStep(
                        active_ranks=false_ranks,
                        branch_id=branch_id,
                        path_type="false"
                    )
                    
                    # Process nested branches within false path
                    nested_steps = _process_nested_branches(
                        branch_id, false_ranks, remaining_branches, processed_branches,
                        classifications, branch_metadata
                    )
                    false_step.nested_steps = nested_steps
                    steps.append(false_step)
        
        elif cls.branch_type == BranchClassificationType.WORLD_UNIFORM:
            # All ranks take the same path, no partitioning
            if cls.true_ranks:
                path_type = "true"
                active = cls.true_ranks & current_ranks
            else:
                path_type = "false"
                active = cls.false_ranks & current_ranks
            
            if active:
                uniform_step = ExecutionStep(
                    active_ranks=active,
                    branch_id=branch_id,
                    path_type=path_type
                )
                
                # Process nested branches
                nested_steps = _process_nested_branches(
                    branch_id, active, remaining_branches, processed_branches,
                    classifications, branch_metadata
                )
                uniform_step.nested_steps = nested_steps
                steps.append(uniform_step)
        
        processed_branches.add(branch_id)
        return steps
    
    def _process_nested_branches(
        parent_id: int,
        active_ranks: Set[int],
        remaining: Set[int],
        processed: Set[int],
        classifications: Dict[int, BranchClassification],
        metadata: Dict[int, any]
    ) -> List[ExecutionStep]:
        """Process branches nested within a parent branch's path."""
        nested_steps: List[ExecutionStep] = []
        
        # Find branches nested within this parent
        parent_metadata = metadata.get(parent_id)
        if not parent_metadata:
            return nested_steps
        
        parent_nest_path = parent_metadata.nest_path if parent_metadata else []
        # Nested branches have this parent in their nest_path
        expected_nest_path = parent_nest_path + [parent_id]
        
        for branch_id in sorted(remaining):
            if branch_id in processed:
                continue
            
            branch_meta = metadata.get(branch_id)
            if not branch_meta:
                continue
            
            branch_nest_path = branch_meta.nest_path if branch_meta else []
            
            # Check if this branch is nested within the parent
            if branch_nest_path == expected_nest_path:
                # This is a nested branch
                nested = process_branch(branch_id, active_ranks, processed)
                nested_steps.extend(nested)
        
        return nested_steps
    
    def _find_elif_branches(
        parent_id: int,
        metadata: Dict[int, any],
        remaining: Set[int]
    ) -> List[int]:
        """Find elif branches that belong to the same chain as parent."""
        elif_branches: List[int] = []
        
        # Find branches with parent_lineno pointing to parent_id or another elif in the chain
        candidates = {parent_id}
        
        while candidates:
            current_parent = candidates.pop()
            for branch_id in sorted(remaining):
                branch_meta = metadata.get(branch_id)
                if not branch_meta:
                    continue
                
                if (hasattr(branch_meta, 'branch_type') and 
                    branch_meta.branch_type.value == 'elif' and
                    branch_meta.parent_lineno == current_parent):
                    elif_branches.append(branch_id)
                    candidates.add(branch_id)  # This elif might have more elifs after it
        
        return sorted(elif_branches)
    
    # Process all root-level branches (not nested, not elif)
    processed_branches: Set[int] = set()
    current_ranks = initial_ranks.copy()
    
    for branch_id in all_branch_ids:
        if branch_id in processed_branches:
            continue
        
        metadata = branch_metadata.get(branch_id)
        if not metadata:
            continue
        
        # Skip elif branches (they'll be processed in their parent's false path)
        if (hasattr(metadata, 'branch_type') and 
            metadata.branch_type.value == 'elif'):
            continue
        
        # Skip nested branches (they'll be processed within their parent)
        nest_path = metadata.nest_path if metadata else []
        if len(nest_path) > 0:
            continue
        
        # Process root-level branch
        steps = process_branch(branch_id, current_ranks, processed_branches)
        execution_path.extend(steps)
    
    return execution_path
