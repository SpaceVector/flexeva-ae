"""
R-variant tracking and taint propagation.

R-variants represent values that vary across ranks, enabling detection
of rank-dependent control flow during Phase II dry-run execution.
"""

from typing import Any, Dict, List, Tuple, Optional
from threading import local
import inspect

# Thread-local storage for branch signatures during dry-run
_thread_local = local()


class R:
    """
    R-variant: A value that may vary across ranks.
    
    Tracks both the value and whether it's rank-tainted (rt).
    Rank-tainted values propagate through operations, enabling detection
    of rank-dependent branch conditions.
    """
    __slots__ = ("v", "rt")
    
    def __init__(self, v: Any, rt: bool = True):
        """
        Initialize an R-variant.
        
        Parameters
        ----------
        v : Any
            The actual value
        rt : bool, optional
            Whether this value is rank-tainted (default: True)
        """
        self.v = v
        self.rt = bool(rt)
    
    def _lift(self, o: Any) -> Any:
        """Extract value from R-variant or return as-is."""
        return o.v if isinstance(o, R) else o
    
    def _wrap(self, val: Any, other: Optional[Any] = None) -> "R":
        """
        Wrap a value as R-variant, propagating taint.
        
        Parameters
        ----------
        val : Any
            The value to wrap
        other : Any, optional
            The other operand (for binary operations)
        
        Returns
        -------
        R
            R-variant with taint propagated
        """
        other_rt = isinstance(other, R) and other.rt if other is not None else False
        return R(val, self.rt or other_rt)
    
    # Arithmetic operations
    def __add__(self, o: Any) -> "R":
        return self._wrap(self.v + self._lift(o), o)
    
    def __sub__(self, o: Any) -> "R":
        return self._wrap(self.v - self._lift(o), o)
    
    def __mul__(self, o: Any) -> "R":
        return self._wrap(self.v * self._lift(o), o)
    
    def __truediv__(self, o: Any) -> "R":
        return self._wrap(self.v / self._lift(o), o)
    
    def __floordiv__(self, o: Any) -> "R":
        return self._wrap(self.v // self._lift(o), o)
    
    def __mod__(self, o: Any) -> "R":
        return self._wrap(self.v % self._lift(o), o)
    
    def __pow__(self, o: Any) -> "R":
        return self._wrap(self.v ** self._lift(o), o)
    
    # Reverse arithmetic operations
    def __radd__(self, o: Any) -> "R":
        return self._wrap(self._lift(o) + self.v, o)
    
    def __rsub__(self, o: Any) -> "R":
        return self._wrap(self._lift(o) - self.v, o)
    
    def __rmul__(self, o: Any) -> "R":
        return self._wrap(self._lift(o) * self.v, o)
    
    def __rtruediv__(self, o: Any) -> "R":
        return self._wrap(self._lift(o) / self.v, o)
    
    def __rfloordiv__(self, o: Any) -> "R":
        return self._wrap(self._lift(o) // self.v, o)
    
    def __rmod__(self, o: Any) -> "R":
        return self._wrap(self._lift(o) % self.v, o)
    
    def __rpow__(self, o: Any) -> "R":
        return self._wrap(self._lift(o) ** self.v, o)
    
    # Comparison operations
    def __eq__(self, o: Any) -> "R":
        return self._wrap(self.v == self._lift(o), o)
    
    def __ne__(self, o: Any) -> "R":
        return self._wrap(self.v != self._lift(o), o)
    
    def __lt__(self, o: Any) -> "R":
        return self._wrap(self.v < self._lift(o), o)
    
    def __le__(self, o: Any) -> "R":
        return self._wrap(self.v <= self._lift(o), o)
    
    def __gt__(self, o: Any) -> "R":
        return self._wrap(self.v > self._lift(o), o)
    
    def __ge__(self, o: Any) -> "R":
        return self._wrap(self.v >= self._lift(o), o)
    
    # Boolean conversion (for if/while conditions)
    def __bool__(self) -> bool:
        """Convert to bool for use in if/while conditions."""
        return bool(self.v)
    
    def __repr__(self) -> str:
        return f"R({self.v!r}, rt={self.rt})"


def mark_cond(cond: Any, branch_id: int) -> bool:
    """
    Mark a branch condition and record its evaluation.
    
    This is the central hook for Phase II branch signature collection.
    It records which branches execute and whether they depend on R-variants.
    
    Parameters
    ----------
    cond : Any
        The condition value (may be R-variant or plain bool)
    branch_id : int
        The branch identifier (typically line number from Phase I)
    
    Returns
    -------
    bool
        The condition value (preserves control flow)
    """
    # Extract actual boolean value
    if isinstance(cond, R):
        cond_value = bool(cond.v)
        is_rtainted = cond.rt
    else:
        cond_value = bool(cond)
        is_rtainted = False
    
    # Record branch evaluation if we're in a dry-run context
    if hasattr(_thread_local, 'branch_signature'):
        _thread_local.branch_signature.append((branch_id, cond_value, is_rtainted))
    
    # If the branch is rank-tainted, record related R-variants
    if is_rtainted:
        rank_ctx = getattr(_thread_local, 'rank_context', None)
        branch_vars = getattr(_thread_local, 'branch_variables', None)
        if rank_ctx is not None and branch_vars:
            var_names = branch_vars.get(branch_id, [])
            if var_names:
                frame = inspect.currentframe()
                try:
                    parent = frame.f_back if frame else None
                    if parent is not None:
                        local_vars = parent.f_locals
                        for name in var_names:
                            value = local_vars.get(name)
                            if isinstance(value, R):
                                rank_ctx.register_r_variant(name, value)
                                if hasattr(rank_ctx, "record_branch_influence"):
                                    rank_ctx.record_branch_influence(name, branch_id)
                finally:
                    # Explicitly delete frame references to avoid reference cycles
                    del frame
    
    return cond_value


def get_branch_signature() -> List[Tuple[int, bool, bool]]:
    """
    Get the current branch signature.
    
    Returns
    -------
    List[Tuple[int, bool, bool]]
        List of (branch_id, outcome, is_rtainted) tuples
    """
    if hasattr(_thread_local, 'branch_signature'):
        return _thread_local.branch_signature.copy()
    return []


def reset_branch_signature() -> None:
    """Reset the branch signature for a new dry-run."""
    _thread_local.branch_signature = []


def set_branch_signature_context(signature: List[Tuple[int, bool, bool]]) -> None:
    """
    Set the branch signature context (for dry-run execution).
    
    Parameters
    ----------
    signature : List[Tuple[int, bool, bool]]
        List to store branch evaluations
    """
    _thread_local.branch_signature = signature


def set_rank_context(rank_ctx: Any) -> None:
    """
    Set the current rank context for R-variant tracking.
    
    Parameters
    ----------
    rank_ctx : Any
        RankContext instance (or None to clear)
    """
    _thread_local.rank_context = rank_ctx


def register_branch_variables(branch_var_map: Dict[int, List[str]]) -> None:
    """
    Register mapping from branch_id to variable names used in the condition.
    
    Parameters
    ----------
    branch_var_map : Dict[int, List[str]]
        Mapping from branch_id to list of variable names referenced in the condition
    """
    _thread_local.branch_variables = branch_var_map
