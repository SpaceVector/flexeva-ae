"""
Dry-run execution framework for Phase II.

Executes instrumented code for each rank and collects branch signatures.
"""

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import sys
import importlib.util
import inspect

from pyextend.runtime.simenv import SimEnv
from pyextend.runtime.rvariant import (
    R,
    mark_cond,
    reset_branch_signature,
    set_branch_signature_context,
    get_branch_signature,
    set_rank_context,
    register_branch_variables,
)
from pyextend.runtime.rankcontext import RankContext


class DryRunExecutor:
    """
    Executes instrumented code in dry-run mode for each rank.
    
    Collects branch signatures and tracks R-variant taint propagation.
    """
    
    def __init__(self, env: SimEnv, branch_var_map: Optional[Dict[int, List[str]]] = None):
        """
        Initialize dry-run executor.
        
        Parameters
        ----------
        env : SimEnv
            Simulation environment
        """
        self.env = env
        self.branch_signatures: Dict[int, List[Tuple[int, bool, bool]]] = {}
        self.rank_contexts: Dict[int, RankContext] = {}
        self.branch_var_map = branch_var_map or {}
    
    def execute_for_rank(self, rank: int, code_path: Path, module_cache: Optional[Any] = None) -> List[Tuple[int, bool, bool]]:
        """
        Execute instrumented code for a specific rank.
        
        Parameters
        ----------
        rank : int
            Rank ID to simulate
        code_path : Path
            Path to instrumented code file from Phase I
        module_cache : Any, optional
            Cached module (to avoid reloading for each rank)
        
        Returns
        -------
        List[Tuple[int, bool, bool]]
            Branch signature: list of (branch_id, outcome, is_rtainted) tuples
        """
        # Create rank context for tracking R-variants
        rank_ctx = RankContext(rank)
        self.rank_contexts[rank] = rank_ctx
        
        # Initialize branch signature collection
        branch_signature: List[Tuple[int, bool, bool]] = []
        reset_branch_signature()
        set_branch_signature_context(branch_signature)
        
        # Load module (reuse cache if provided)
        if module_cache is None:
            try:
                spec = importlib.util.spec_from_file_location("instrumented_code", code_path)
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"Failed to load instrumented code from {code_path}")
                
                module = importlib.util.module_from_spec(spec)
                
                # Inject runtime components into module namespace
                module.SimEnv = SimEnv
                module.R = R
                module.mark_cond = mark_cond
                
                # Prevent __main__ block from executing during import
                # We'll manually execute main() for each rank
                module.__name__ = "instrumented_code"
                
                # Execute the module (defines main() and other functions, but not __main__ block)
                spec.loader.exec_module(module)
            except Exception as e:
                raise RuntimeError(f"Error loading instrumented code: {e}") from e
        else:
            module = module_cache
        
        # Now execute main() for this specific rank
        try:
            # Register rank context and branch metadata for this execution
            set_rank_context(rank_ctx)
            register_branch_variables(self.branch_var_map)
            
            # Create a fresh SimEnv for this rank's execution
            # We need to determine world_size from the module or use self.env.world_size
            world_size = self.env.world_size
            
            # Create SimEnv and set up R-variants as the __main__ block would
            # Pass rank_ctx as tracker to automatically register R-variants when accessed
            rank_env = SimEnv(world_size=world_size, r_variant_tracker=rank_ctx)
            rank_env.set_rank_context(rank)
            
            # Initialize R-variants for all ranks (as __main__ block does)
            # But we only care about the current rank's context
            for r in range(world_size):
                r_val = R(r)
                rank_env.assign('rank', r, r_val)
                # Register initial R-variant for this rank
                if r == rank:
                    rank_ctx.register_r_variant('rank', r_val)
            rank_env.assign_global('world_size', world_size)
            
            # Get the main function from the module
            if not hasattr(module, 'main'):
                raise RuntimeError("Instrumented code does not have a 'main' function")
            
            main_func = module.main
            
            # Use a trace function to track R-variant assignments during execution
            # Track the main function's code object to only capture variables from main()
            main_code = main_func.__code__ if hasattr(main_func, '__code__') else None
            
            def trace_assignments(frame, event, arg):
                """Trace function to track when R-variants are assigned to variables."""
                if event == 'line':
                    # Only track variables in the main() function's frame
                    # Skip internal frames (R class operations, etc.)
                    if frame.f_code is not main_code:
                        return trace_assignments
                    
                    # Check local variables in the current frame
                    if frame.f_locals is not None:
                        # List of internal variable names to skip
                        skip_names = {
                            'self', 'o', 'other', 'value', 'cond', 'r_value',
                            'args', 'kwargs', 'arg', 'frame', 'event'
                        }
                        
                        for var_name, var_value in frame.f_locals.items():
                            # Skip special variables, functions, and internal names
                            if (var_name.startswith('_') or 
                                callable(var_value) or 
                                var_name in skip_names):
                                continue
                            
                            # If the variable is an R-variant, register it
                            if isinstance(var_value, R):
                                # Only register if not already registered
                                if not rank_ctx.has_r_variant(var_name):
                                    rank_ctx.register_r_variant(var_name, var_value)
                return trace_assignments
            
            # Set trace to track assignments
            old_trace = sys.gettrace()
            sys.settrace(trace_assignments)
            
            try:
                # Execute main() for this rank
                main_func(rank_env)
            finally:
                # Restore original trace
                sys.settrace(old_trace)
            
        except Exception as e:
            raise RuntimeError(f"Error executing dry-run for rank {rank}: {e}") from e
        finally:
            # Clear rank context after execution
            set_rank_context(None)
        
        # Store branch signature
        self.branch_signatures[rank] = branch_signature.copy()
        
        return branch_signature
    
    def execute_all_ranks(self, code_path: Path) -> Dict[int, List[Tuple[int, bool, bool]]]:
        """
        Execute instrumented code for all ranks.
        
        Parameters
        ----------
        code_path : Path
            Path to instrumented code file from Phase I
        
        Returns
        -------
        Dict[int, List[Tuple[int, bool, bool]]]
            Mapping from rank ID to branch signature
        """
        # Load module once and reuse for all ranks
        module_dir = str(code_path.parent.resolve())
        inserted_module_dir = False
        try:
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
                inserted_module_dir = True
            spec = importlib.util.spec_from_file_location("instrumented_code", code_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Failed to load instrumented code from {code_path}")
            
            module = importlib.util.module_from_spec(spec)
            module.SimEnv = SimEnv
            module.R = R
            module.mark_cond = mark_cond
            module.__name__ = "instrumented_code"
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Error loading instrumented code: {e}") from e
        finally:
            if inserted_module_dir:
                try:
                    sys.path.remove(module_dir)
                except ValueError:
                    pass

        # Execute for each rank
        for rank in range(self.env.world_size):
            self.execute_for_rank(rank, code_path, module_cache=module)
        
        return self.branch_signatures
    
    def get_branch_signature(self, rank: int) -> List[Tuple[int, bool, bool]]:
        """
        Get branch signature for a rank.
        
        Parameters
        ----------
        rank : int
            Rank ID
        
        Returns
        -------
        List[Tuple[int, bool, bool]]
            Branch signature
        """
        return self.branch_signatures.get(rank, [])
    
    def get_all_branch_signatures(self) -> Dict[int, List[Tuple[int, bool, bool]]]:
        """
        Get all branch signatures.
        
        Returns
        -------
        Dict[int, List[Tuple[int, bool, bool]]]
            Mapping from rank ID to branch signature
        """
        return self.branch_signatures.copy()
    
    def get_rank_context(self, rank: int) -> Optional[RankContext]:
        """
        Get rank context for a rank.
        
        Parameters
        ----------
        rank : int
            Rank ID
        
        Returns
        -------
        Optional[RankContext]
            Rank context if found
        """
        return self.rank_contexts.get(rank)

    def get_semantic_summary(self, rank: int) -> Dict[str, Dict[str, Any]]:
        """
        Export the semantic variable summary for one rank.
        """
        rank_ctx = self.get_rank_context(rank)
        if rank_ctx is None:
            return {}
        return rank_ctx.summarize_semantic_variables()

    def get_all_semantic_summaries(self) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """
        Export semantic variable summaries for all executed ranks.
        """
        return {
            rank: self.get_semantic_summary(rank)
            for rank in sorted(self.rank_contexts.keys())
        }
