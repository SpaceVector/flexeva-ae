"""
Decorators for flex execution scoping.

@flex_rank and @flex_group decorators provide explicit execution scoping hints
for Phase III execution.
"""

from typing import Callable, Tuple, Optional, Any
from functools import wraps


def flex_rank(active_ranks: Tuple[int, ...]) -> Callable:
    """
    Decorator to specify which ranks should execute a function.
    
    Semantics:
    - The decorated function should logically execute once per rank in active_ranks
    - In Phase III: Pyextend considers the intersection of current active_ranks
      at the call site and the decorator's active_ranks set
    - If intersection is empty → function is skipped
    - If intersection is non-empty → executes once for each rank in intersection
    
    Parameters
    ----------
    active_ranks : Tuple[int, ...]
        Tuple of rank IDs that should execute this function
    
    Returns
    -------
    Callable
        Decorator function
    
    Note
    ----
    Decorated functions are NOT executed during Phase I (AST parsing) or Phase II (dry-run).
    They execute only in Phase III with appropriate active context.
    """
    def decorator(func: Callable) -> Callable:
        # Store metadata on the function
        func._flex_rank_active_ranks = active_ranks
        func._flex_decorator_type = "flex_rank"
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # In Phase II, decorated functions are not executed
            # This wrapper will be replaced in Phase III
            # During Phase II (dry-run), we skip execution and return None
            # The function will execute in Phase III with proper active context
            return None
        
        # Copy metadata to wrapper
        wrapper._flex_rank_active_ranks = active_ranks
        wrapper._flex_decorator_type = "flex_rank"
        
        return wrapper
    
    return decorator


def flex_group(active_group: str) -> Callable:
    """
    Decorator to specify which group should execute a function.
    
    Semantics:
    - The decorated function should execute once for all ranks in the active group
    - In Phase III: At call site, compute subset of current active context that
      matches the decorator's group label
    - Executes function once for the group
    
    Parameters
    ----------
    active_group : str
        Group label (e.g., "dp:0", "pp:stage1")
    
    Returns
    -------
    Callable
        Decorator function
    
    Note
    ----
    Decorated functions are NOT executed during Phase I (AST parsing) or Phase II (dry-run).
    They execute only in Phase III with appropriate active context.
    """
    def decorator(func: Callable) -> Callable:
        # Store metadata on the function
        func._flex_group_active_group = active_group
        func._flex_decorator_type = "flex_group"
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # In Phase II, decorated functions are not executed
            # This wrapper will be replaced in Phase III
            # During Phase II (dry-run), we skip execution and return None
            # The function will execute in Phase III with proper active context
            return None
        
        # Copy metadata to wrapper
        wrapper._flex_group_active_group = active_group
        wrapper._flex_decorator_type = "flex_group"
        
        return wrapper
    
    return decorator

