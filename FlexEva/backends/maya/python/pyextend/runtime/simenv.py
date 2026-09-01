"""
SimEnv: Simulation environment for managing R-variants and execution context.

SimEnv tracks variables that may vary across ranks (R-variants) and provides
access to them during dry-run execution and SPSD code generation.

Storage Model:
- M RankContexts: one per rank, storing R-variants for that rank
- One GroupContext('global'): storing global variables shared by all ranks
- Shadowing: Rank variables shadow global variables
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from pyextend.runtime.rvariant import R


class GroupContext:
    """
    Context for storing global variables shared by all ranks.
    """
    
    def __init__(self, name: str):
        """
        Initialize group context.
        
        Parameters
        ----------
        name : str
            Group name (typically 'global')
        """
        self.name = name
        # Global variables: name -> value
        self._variables: Dict[str, Any] = {}
    
    def assign(self, name: str, value: Any) -> None:
        """
        Assign a global variable.
        
        Parameters
        ----------
        name : str
            Variable name
        value : Any
            Value
        """
        self._variables[name] = value
    
    def resolve(self, name: str) -> Any:
        """
        Resolve a global variable.
        
        Parameters
        ----------
        name : str
            Variable name
        
        Returns
        -------
        Any
            Variable value
        
        Raises
        ------
        KeyError
            If variable is not found
        """
        if name not in self._variables:
            raise KeyError(f"Global variable '{name}' not found in GroupContext('{self.name}')")
        return self._variables[name]
    
    def has(self, name: str) -> bool:
        """
        Check if a variable exists.
        
        Parameters
        ----------
        name : str
            Variable name
        
        Returns
        -------
        bool
            True if variable exists, False otherwise
        """
        return name in self._variables
    
    def get_all_names(self) -> Set[str]:
        """
        Get all variable names.
        
        Returns
        -------
        Set[str]
            Set of variable names
        """
        return set(self._variables.keys())


class RankContext:
    """
    Context for storing R-variants for a specific rank.
    """
    
    def __init__(self, rank: int):
        """
        Initialize rank context.
        
        Parameters
        ----------
        rank : int
            Rank ID
        """
        self.rank = rank
        # R-variants: name -> value
        self._variables: Dict[str, Any] = {}
    
    def assign(self, name: str, value: Any) -> None:
        """
        Assign an R-variant for this rank.
        
        Parameters
        ----------
        name : str
            Variable name
        value : Any
            Value (typically an R-variant)
        """
        self._variables[name] = value
    
    def resolve(self, name: str) -> Any:
        """
        Resolve an R-variant for this rank.
        
        Parameters
        ----------
        name : str
            Variable name
        
        Returns
        -------
        Any
            Variable value
        
        Raises
        ------
        KeyError
            If variable is not found
        """
        if name not in self._variables:
            raise KeyError(f"R-variant '{name}' not found in RankContext(rank={self.rank})")
        return self._variables[name]
    
    def has(self, name: str) -> bool:
        """
        Check if a variable exists.
        
        Parameters
        ----------
        name : str
            Variable name
        
        Returns
        -------
        bool
            True if variable exists, False otherwise
        """
        return name in self._variables
    
    def get_all_names(self) -> Set[str]:
        """
        Get all variable names.
        
        Returns
        -------
        Set[str]
            Set of variable names
        """
        return set(self._variables.keys())


class SimEnv:
    """
    Simulation environment for Phase II dry-run execution and SPSD code generation.
    
    Storage Model:
    - M RankContexts: one per rank (0 to world_size-1), storing R-variants
    - One GroupContext('global'): storing global variables
    - Shadowing: Rank variables shadow global variables
    
    Manages:
    - R-variants: variables that vary across ranks (stored in RankContexts)
    - Global variables: variables that are the same across all ranks (stored in GroupContext('global'))
    - Active rank context: current rank being simulated (for dry-run)
    """
    
    def __init__(self, world_size: int, r_variant_tracker: Optional[Any] = None):
        """
        Initialize simulation environment.
        
        Parameters
        ----------
        world_size : int
            Total number of ranks in the simulation
        r_variant_tracker : Any, optional
            Callback to track R-variants when accessed (e.g., RankContext from rankcontext.py)
        """
        self.world_size = world_size
        # M RankContexts: one per rank
        self._rank_contexts: Dict[int, RankContext] = {
            rank: RankContext(rank) for rank in range(world_size)
        }
        # One GroupContext('global')
        self._global_context = GroupContext('global')
        # Current active rank context (for dry-run)
        self._current_rank: Optional[int] = None
        # R-variant tracker callback (for dry-run)
        self._r_variant_tracker = r_variant_tracker
        
        # Legacy storage for backward compatibility (used by dry-run)
        # R-variants: name -> rank -> value
        self._r_variants: Dict[str, Dict[int, Any]] = {}
        # Global variables: name -> value
        self._globals: Dict[str, Any] = {}
        
        # Group management: group_name -> (ranks, metadata)
        # ranks: List[int] - ranks that belong to this group
        # metadata: Dict[str, Any] - group metadata (e.g., group_meta1)
        self._groups: Dict[str, Tuple[List[int], Dict[str, Any]]] = {}
    
    def assign(self, name: str, rank: int, value: Any) -> None:
        """
        Assign an R-variant value for a specific rank.
        
        Stores the value in RankContext for that rank.
        Also updates legacy storage for backward compatibility.
        
        Parameters
        ----------
        name : str
            Variable name
        rank : int
            Rank ID
        value : Any
            Value (typically an R-variant)
        """
        # Store in RankContext
        if rank not in self._rank_contexts:
            self._rank_contexts[rank] = RankContext(rank)
        self._rank_contexts[rank].assign(name, value)
        
        # Legacy storage for backward compatibility
        if name not in self._r_variants:
            self._r_variants[name] = {}
        self._r_variants[name][rank] = value
    
    def assign_global(self, name: str, value: Any) -> None:
        """
        Assign a global variable (same for all ranks).
        
        Stores the value in GroupContext('global').
        Also updates legacy storage for backward compatibility.
        
        Parameters
        ----------
        name : str
            Variable name
        value : Any
            Value
        """
        # Store in GroupContext('global')
        self._global_context.assign(name, value)
        
        # Legacy storage for backward compatibility
        self._globals[name] = value
    
    def resolve(self, rank: int, name: str) -> Any:
        """
        Resolve a variable for a specific rank.
        
        Implements shadowing: rank variables shadow global variables.
        First checks RankContext for the rank, then falls back to GroupContext('global').
        
        Parameters
        ----------
        rank : int
            Rank ID
        name : str
            Variable name
        
        Returns
        -------
        Any
            Variable value (from rank context if exists, otherwise from global)
        
        Raises
        ------
        KeyError
            If variable is not found in either rank or global context
        """
        # Check rank context first (shadowing)
        if rank in self._rank_contexts and self._rank_contexts[rank].has(name):
            return self._rank_contexts[rank].resolve(name)
        
        # Fall back to global context
        if self._global_context.has(name):
            return self._global_context.resolve(name)
        
        raise KeyError(f"Variable '{name}' not found in RankContext(rank={rank}) or GroupContext('global')")
    
    def global_resolve(self, name: str) -> Any:
        """
        Resolve a global variable (ignoring rank context).
        
        Parameters
        ----------
        name : str
            Variable name
        
        Returns
        -------
        Any
            Global variable value
        
        Raises
        ------
        KeyError
            If variable is not found in global context
        """
        return self._global_context.resolve(name)
    
    def global_assign(self, name: str, value: Any) -> None:
        """
        Assign a computed variable to the global context.
        
        This is used for variables that are computed in the code (not R-variants or known globals).
        They are stored in GroupContext('global') for access across functions.
        
        Parameters
        ----------
        name : str
            Variable name
        value : Any
            Value to assign
        """
        self._global_context.assign(name, value)
    
    def get_rank_context(self, rank: int) -> RankContext:
        """
        Get RankContext for a specific rank.
        
        Parameters
        ----------
        rank : int
            Rank ID
        
        Returns
        -------
        RankContext
            Rank context for the specified rank
        """
        if rank not in self._rank_contexts:
            self._rank_contexts[rank] = RankContext(rank)
        return self._rank_contexts[rank]
    
    def get_global_context(self) -> GroupContext:
        """
        Get GroupContext('global').
        
        Returns
        -------
        GroupContext
            Global context
        """
        return self._global_context
    
    def default_resolve(self, name: str) -> Any:
        """
        Resolve a variable name to its value for the current rank context.
        
        This is the main method used in user code (main function).
        It returns the R-variant value for the current rank if it's an R-variant,
        or the global value if it's a global variable.
        
        Parameters
        ----------
        name : str
            Variable name
        
        Returns
        -------
        Any
            The value for the current rank context
        
        Raises
        ------
        KeyError
            If variable is not found
        RuntimeError
            If no rank context is set
        """
        if self._current_rank is None:
            raise RuntimeError(
                "No rank context set. Use set_rank_context() before calling default_resolve()"
            )
        
        # Check R-variants first
        if name in self._r_variants:
            rank_values = self._r_variants[name]
            if self._current_rank in rank_values:
                value = rank_values[self._current_rank]
                # Track R-variant access if tracker is set
                if self._r_variant_tracker is not None and isinstance(value, R):
                    self._r_variant_tracker.register_r_variant(name, value)
                return value
            else:
                raise KeyError(
                    f"R-variant '{name}' not assigned for rank {self._current_rank}"
                )
        
        # Check globals
        if name in self._globals:
            return self._globals[name]
        
        raise KeyError(f"Variable '{name}' not found (not an R-variant or global)")
    
    def set_rank_context(self, rank: int) -> None:
        """
        Set the current rank context for dry-run execution.
        
        Parameters
        ----------
        rank : int
            Rank ID to simulate
        """
        if rank < 0 or rank >= self.world_size:
            raise ValueError(f"Rank {rank} out of range [0, {self.world_size})")
        self._current_rank = rank
    
    def get_rank_context(self) -> Optional[int]:
        """
        Get the current rank context.
        
        Returns
        -------
        Optional[int]
            Current rank ID, or None if not set
        """
        return self._current_rank
    
    def get_r_variant_ranks(self, name: str) -> Set[int]:
        """
        Get all ranks for which an R-variant is defined.
        
        Parameters
        ----------
        name : str
            R-variant name
        
        Returns
        -------
        Set[int]
            Set of rank IDs
        """
        if name in self._r_variants:
            return set(self._r_variants[name].keys())
        return set()
    
    def get_all_r_variant_names(self) -> Set[str]:
        """
        Get all R-variant names (from all rank contexts).
        
        Returns
        -------
        Set[str]
            Set of R-variant names
        """
        all_names = set()
        for rank_ctx in self._rank_contexts.values():
            all_names.update(rank_ctx.get_all_names())
        return all_names
    
    def get_global_names(self) -> Set[str]:
        """
        Get all global variable names.
        
        Returns
        -------
        Set[str]
            Set of global variable names
        """
        return self._global_context.get_all_names()
    
    def create_group(self, group_name: str, ranks: List[int]) -> None:
        """
        Create a process group with specified ranks.
        
        Parameters
        ----------
        group_name : str
            Group label (e.g., "dp:0", "pp:stage1")
        ranks : List[int]
            List of rank IDs that belong to this group
        """
        if group_name not in self._groups:
            self._groups[group_name] = (ranks, {})
        else:
            # Update ranks if group already exists
            existing_ranks, metadata = self._groups[group_name]
            self._groups[group_name] = (ranks, metadata)
    
    def assign_group(self, group_name: str, meta_name: str, meta_value: Any) -> None:
        """
        Assign metadata to a process group.
        
        Parameters
        ----------
        group_name : str
            Group label
        meta_name : str
            Metadata key
        meta_value : Any
            Metadata value
        """
        if group_name not in self._groups:
            # Create group with empty ranks if it doesn't exist
            self._groups[group_name] = ([], {})
        
        ranks, metadata = self._groups[group_name]
        metadata[meta_name] = meta_value
        self._groups[group_name] = (ranks, metadata)
    
    def resolve_group(self, group_name: str, meta_name: str) -> Any:
        """
        Resolve metadata from a process group.
        
        Parameters
        ----------
        group_name : str
            Group label
        meta_name : str
            Metadata key
        
        Returns
        -------
        Any
            Metadata value
        
        Raises
        ------
        KeyError
            If group or metadata key is not found
        """
        if group_name not in self._groups:
            raise KeyError(f"Group '{group_name}' not found")
        
        ranks, metadata = self._groups[group_name]
        if meta_name not in metadata:
            raise KeyError(f"Metadata '{meta_name}' not found in group '{group_name}'")
        
        return metadata[meta_name]
    
    def get_group_ranks(self, group_name: str) -> List[int]:
        """
        Get ranks that belong to a process group.
        
        Parameters
        ----------
        group_name : str
            Group label
        
        Returns
        -------
        List[int]
            List of rank IDs
        
        Raises
        ------
        KeyError
            If group is not found
        """
        if group_name not in self._groups:
            raise KeyError(f"Group '{group_name}' not found")
        
        ranks, _ = self._groups[group_name]
        return ranks.copy()

