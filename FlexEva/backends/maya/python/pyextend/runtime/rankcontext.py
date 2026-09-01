"""
RankContext: Tracks R-variants and their branch influence per rank.

Used during Phase II dry-run to register and track semantic variables as they are
created and consumed by rank-dependent control flow.
"""

from typing import Any, Dict, Set, Optional
from pyextend.runtime.rvariant import R


class RankContext:
    """
    Context for tracking R-variants during dry-run execution.
    
    Registers R-variants as they are created and tracks their dependencies
    for taint propagation analysis.
    """
    
    def __init__(self, rank: int):
        """
        Initialize rank context.
        
        Parameters
        ----------
        rank : int
            Rank ID for this context
        """
        self.rank = rank
        # Registered R-variants: name -> R instance
        self._r_variants: Dict[str, R] = {}
        # Dependencies: name -> set of names it depends on
        self._dependencies: Dict[str, Set[str]] = {}
        # Branch influence: variable name -> set of branch ids that consume it
        self._branch_influences: Dict[str, Set[int]] = {}
    
    def register_r_variant(self, name: str, r_value: R) -> None:
        """
        Register an R-variant in this rank context.
        
        Parameters
        ----------
        name : str
            Variable name
        r_value : R
            R-variant value
        """
        self._r_variants[name] = r_value
        # Initialize dependencies if not present
        if name not in self._dependencies:
            self._dependencies[name] = set()
        if name not in self._branch_influences:
            self._branch_influences[name] = set()
    
    def get_r_variant(self, name: str) -> Optional[R]:
        """
        Get a registered R-variant.
        
        Parameters
        ----------
        name : str
            Variable name
        
        Returns
        -------
        Optional[R]
            R-variant if found, None otherwise
        """
        return self._r_variants.get(name)
    
    def has_r_variant(self, name: str) -> bool:
        """
        Check if an R-variant is registered.
        
        Parameters
        ----------
        name : str
            Variable name
        
        Returns
        -------
        bool
            True if registered, False otherwise
        """
        return name in self._r_variants
    
    def get_all_r_variant_names(self) -> Set[str]:
        """
        Get all registered R-variant names.
        
        Returns
        -------
        Set[str]
            Set of R-variant names
        """
        return set(self._r_variants.keys())
    
    def add_dependency(self, name: str, depends_on: str) -> None:
        """
        Record that one R-variant depends on another.
        
        Parameters
        ----------
        name : str
            Dependent R-variant name
        depends_on : str
            Dependency R-variant name
        """
        if name not in self._dependencies:
            self._dependencies[name] = set()
        self._dependencies[name].add(depends_on)
    
    def get_dependencies(self, name: str) -> Set[str]:
        """
        Get dependencies for an R-variant.
        
        Parameters
        ----------
        name : str
            R-variant name
        
        Returns
        -------
        Set[str]
            Set of dependency names
        """
        return self._dependencies.get(name, set()).copy()

    def record_branch_influence(self, name: str, branch_id: int) -> None:
        """
        Record that an R-variant influenced a branch condition.

        Parameters
        ----------
        name : str
            Variable name
        branch_id : int
            Branch identifier
        """
        if name not in self._branch_influences:
            self._branch_influences[name] = set()
        self._branch_influences[name].add(branch_id)

    def get_branch_influences(self, name: str) -> Set[int]:
        """
        Get branch identifiers influenced by an R-variant.
        """
        return self._branch_influences.get(name, set()).copy()

    def summarize_semantic_variables(self) -> Dict[str, Dict[str, Any]]:
        """
        Build a small semantic summary for exported dry-run analysis.
        """
        summary: Dict[str, Dict[str, Any]] = {}
        for name in sorted(self._r_variants.keys()):
            value = self._r_variants[name]
            summary[name] = {
                "value": value.v,
                "r_tainted": value.rt,
                "repr": repr(value),
                "dependencies": sorted(self._dependencies.get(name, set())),
                "branch_ids": sorted(self._branch_influences.get(name, set())),
            }
        return summary
