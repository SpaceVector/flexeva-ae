"""
AST visitor to systematically enumerate all branch points in Python code.
"""

from __future__ import annotations

import ast
from typing import Any, List

from astparser.branch_point import BranchPoint, BranchType, FlexDecorator


class BranchVisitor(ast.NodeVisitor):
    """
    AST visitor that enumerates all branch points in a Python module.

    Treats nested control flow exactly the same as flat control flow.
    Does NOT reconstruct body structures.
    """

    def __init__(self) -> None:
        """Initialize the branch visitor."""
        self.branches: List[BranchPoint] = []
        self.decorators: List[FlexDecorator] = []
        self._branch_stack: List[int] = []  # Track parent line numbers for nesting

    def visit_If(self, node: ast.If) -> None:
        """
        Visit an if/elif/else statement.

        Treats elif exactly like if - records it as a branch point and recursively
        handles the elif chain. This ensures all elif branches are captured correctly,
        regardless of how many elif branches there are or how deeply nested they are.
        """
        # Skip instrumentation of `if __name__ == '__main__'` blocks
        # This is not a rank-based branch and doesn't need instrumentation
        if self._is_name_main_check(node.test):
            # Still visit children to find nested branches, but don't record this as a branch point
            for child in node.body:
                self.visit(child)
            for child in node.orelse:
                self.visit(child)
            return

        # Check if we're nested (inside another branch)
        is_nested = len(self._branch_stack) > 0
        parent_lineno = self._branch_stack[-1] if is_nested else None
        # nest_path is a copy of the current branch_stack (path from root to parent)
        # This is the nest_path for this if and all its elif branches
        nest_path = list(self._branch_stack)

        # Record this if as a branch point
        condition_str = ast.unparse(node.test) if hasattr(ast, "unparse") else self._expr_to_string(node.test)
        self.branches.append(
            BranchPoint(
                branch_type=BranchType.IF,
                lineno=node.lineno,
                col_offset=node.col_offset,
                condition=condition_str,
                is_nested=is_nested,
                parent_lineno=parent_lineno,
                nest_path=nest_path,
            )
        )

        # Push this branch onto the stack for nested tracking
        self._branch_stack.append(node.lineno)

        # Visit body to find nested branches
        for child in node.body:
            self.visit(child)

        # Handle elif/else: orelse contains either another If (elif) or other statements (else)
        # Elif is only when orelse contains exactly one If statement (and nothing else)
        if node.orelse and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            # This is an elif (else-if) - recursively visit it as an elif
            # Pass the current if's lineno as the parent for the first elif
            # Also pass the nest_path (before pushing this if) so elif branches share the same nest_path
            # The elif will be recorded as ELIF and handle its own chain
            self._visit_elif_chain(node.orelse[0], is_nested, node.lineno, nest_path)
        else:
            # This is an else block (or no else) - visit the orelse statements
            # Note: else branches don't have conditions, so we don't record them as branch points
            for child in node.orelse:
                self.visit(child)

        # Pop from stack
        if self._branch_stack and self._branch_stack[-1] == node.lineno:
            self._branch_stack.pop()

    def _visit_elif_chain(self, elif_node: ast.If, is_nested: bool, parent_lineno: int | None, nest_path: List[int]) -> None:
        """
        Recursively visit an elif chain.

        Treats each elif exactly like an if - records it and handles its chain.
        This ensures all elif branches are captured, regardless of depth.
        
        Note: 
        - elif branches share the same nest_path as their corresponding if,
        so we don't push elif onto the branch_stack (it's not a new nesting level).
        - parent_lineno points to the previous elif/if in the chain (not the original parent).
        - nest_path is passed explicitly to ensure elif branches share the same nest_path as the if.
        
        Parameters
        ----------
        elif_node : ast.If
            The elif node to process
        is_nested : bool
            Whether we're nested (same as the original if)
        parent_lineno : int | None
            Line number of the previous elif/if in the chain
        nest_path : List[int]
            The nest_path shared by the if and all its elif branches
        """
        # Record this elif as a branch point
        # parent_lineno now points to the previous elif/if in the chain
        # nest_path is the same as the original if (passed explicitly)
        condition_str = ast.unparse(elif_node.test) if hasattr(ast, "unparse") else self._expr_to_string(elif_node.test)
        self.branches.append(
            BranchPoint(
                branch_type=BranchType.ELIF,
                lineno=elif_node.lineno,
                col_offset=elif_node.col_offset,
                condition=condition_str,
                is_nested=is_nested,
                parent_lineno=parent_lineno,  # Points to previous elif/if in chain
                nest_path=nest_path,  # Same as the original if
            )
        )

        # Don't push elif onto stack - elif is part of the same if statement, not a new nesting level
        # Visit elif body (nested branches inside elif will have the same nest_path as the elif)
        for child in elif_node.body:
            self.visit(child)

        # Handle further elif/else in this elif
        if elif_node.orelse and isinstance(elif_node.orelse[0], ast.If):
            # Another elif - recursively handle the chain
            # Pass this elif's lineno as the parent for the next elif
            # Pass the same nest_path so all elif branches share it
            self._visit_elif_chain(elif_node.orelse[0], is_nested, elif_node.lineno, nest_path)
        else:
            # Else block of this elif
            for child in elif_node.orelse:
                self.visit(child)

    def visit_While(self, node: ast.While) -> None:
        """Visit a while loop statement."""
        is_nested = len(self._branch_stack) > 0
        parent_lineno = self._branch_stack[-1] if is_nested else None
        nest_path = list(self._branch_stack)

        condition_str = ast.unparse(node.test) if hasattr(ast, "unparse") else self._expr_to_string(node.test)
        self.branches.append(
            BranchPoint(
                branch_type=BranchType.WHILE,
                lineno=node.lineno,
                col_offset=node.col_offset,
                condition=condition_str,
                is_nested=is_nested,
                parent_lineno=parent_lineno,
                nest_path=nest_path,
            )
        )

        # Push onto stack and visit children
        self._branch_stack.append(node.lineno)
        for child in node.body:
            self.visit(child)
        for child in node.orelse:
            self.visit(child)
        if self._branch_stack and self._branch_stack[-1] == node.lineno:
            self._branch_stack.pop()

    def visit_IfExp(self, node: ast.IfExp) -> None:
        """Visit a ternary expression (x if condition else y)."""
        is_nested = len(self._branch_stack) > 0
        parent_lineno = self._branch_stack[-1] if is_nested else None
        nest_path = list(self._branch_stack)

        condition_str = ast.unparse(node.test) if hasattr(ast, "unparse") else self._expr_to_string(node.test)
        self.branches.append(
            BranchPoint(
                branch_type=BranchType.TERNARY,
                lineno=node.lineno,
                col_offset=node.col_offset,
                condition=condition_str,
                is_nested=is_nested,
                parent_lineno=parent_lineno,
                nest_path=nest_path,
            )
        )

        # Visit children (body and orelse may contain nested branches)
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Visit a list comprehension and check for filters."""
        self._visit_comprehension(node, BranchType.COMPREHENSION_FILTER)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        """Visit a set comprehension and check for filters."""
        self._visit_comprehension(node, BranchType.COMPREHENSION_FILTER)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """Visit a dict comprehension and check for filters."""
        self._visit_comprehension(node, BranchType.COMPREHENSION_FILTER)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        """Visit a generator expression and check for filters."""
        self._visit_comprehension(node, BranchType.COMPREHENSION_FILTER)

    def _visit_comprehension(self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp, branch_type: BranchType) -> None:
        """
        Visit a comprehension (list, set, dict, or generator) and record filter conditions.

        Comprehensions can have if filters: [x for x in items if x != rank]
        """
        is_nested = len(self._branch_stack) > 0
        parent_lineno = self._branch_stack[-1] if is_nested else None
        nest_path = list(self._branch_stack)

        # Comprehensions have a 'generators' field which is a list of comprehension nodes
        # Each comprehension can have an 'ifs' field with filter conditions
        generators = node.generators

        for gen in generators:
            if gen.ifs:
                for if_cond in gen.ifs:
                    condition_str = ast.unparse(if_cond) if hasattr(ast, "unparse") else self._expr_to_string(if_cond)
                    self.branches.append(
                        BranchPoint(
                            branch_type=branch_type,
                            lineno=if_cond.lineno,
                            col_offset=if_cond.col_offset,
                            condition=condition_str,
                            is_nested=is_nested,
                            parent_lineno=parent_lineno,
                            nest_path=nest_path,
                        )
                    )

        # Visit nested comprehensions
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        """
        Visit a match statement (Python 3.10+).

        Each case in a match statement is a branch point.
        """
        is_nested = len(self._branch_stack) > 0
        parent_lineno = self._branch_stack[-1] if is_nested else None
        nest_path = list(self._branch_stack)

        # Record the match subject (though cases are the actual branches)
        # We'll record each case as a branch point
        for case in node.cases:
            # Case patterns can be complex, but we'll record them
            # For simple cases, we can stringify the pattern
            if case.pattern:
                try:
                    pattern_str = ast.unparse(case.pattern) if hasattr(ast, "unparse") else self._pattern_to_string(case.pattern)
                except:
                    pattern_str = "<pattern>"
            else:
                pattern_str = "<no pattern>"

            # Also check for guard conditions
            if case.guard:
                guard_str = ast.unparse(case.guard) if hasattr(ast, "unparse") else self._expr_to_string(case.guard)
                condition_str = f"{pattern_str} if {guard_str}"
            else:
                condition_str = pattern_str

            self.branches.append(
                BranchPoint(
                    branch_type=BranchType.MATCH_CASE,
                    lineno=case.pattern.lineno if hasattr(case.pattern, 'lineno') else node.lineno,
                    col_offset=case.pattern.col_offset if hasattr(case.pattern, 'col_offset') else node.col_offset,
                    condition=condition_str,
                    is_nested=is_nested,
                    parent_lineno=parent_lineno,
                    nest_path=nest_path,
                )
            )

        # Visit match body and cases
        self.generic_visit(node)

    def _pattern_to_string(self, pattern: ast.pattern) -> str:
        """Convert a match pattern to string representation."""
        if isinstance(pattern, ast.MatchValue):
            return self._expr_to_string(pattern.value)
        elif isinstance(pattern, ast.MatchSingleton):
            return str(pattern.value)
        elif isinstance(pattern, ast.MatchSequence):
            patterns = [self._pattern_to_string(p) for p in pattern.patterns]
            return f"[{', '.join(patterns)}]"
        elif isinstance(pattern, ast.MatchMapping):
            # Simplified representation
            return "<mapping pattern>"
        elif isinstance(pattern, ast.MatchClass):
            return f"{self._expr_to_string(pattern.cls)}(...)"
        elif isinstance(pattern, ast.MatchStar):
            return "*"
        elif isinstance(pattern, ast.MatchAs):
            if pattern.name:
                return pattern.name
            return "_"
        elif isinstance(pattern, ast.MatchOr):
            patterns = [self._pattern_to_string(p) for p in pattern.patterns]
            return f"({' | '.join(patterns)})"
        else:
            return f"<{type(pattern).__name__}>"

    def visit_For(self, node: ast.For) -> None:
        """
        Visit a for loop.

        We record for loops as branch points if they might be rank-dependent.
        The iterable expression is recorded as the condition.
        """
        is_nested = len(self._branch_stack) > 0
        parent_lineno = self._branch_stack[-1] if is_nested else None
        nest_path = list(self._branch_stack)

        # Record the for loop iterable as a potential branch point
        # The iterable might be rank-dependent (e.g., range(rank, world_size))
        iterable_str = ast.unparse(node.iter) if hasattr(ast, "unparse") else self._expr_to_string(node.iter)
        self.branches.append(
            BranchPoint(
                branch_type=BranchType.FOR_LOOP,
                lineno=node.lineno,
                col_offset=node.col_offset,
                condition=f"for {ast.unparse(node.target) if hasattr(ast, 'unparse') else '...'} in {iterable_str}",
                is_nested=is_nested,
                parent_lineno=parent_lineno,
                nest_path=nest_path,
            )
        )

        # Push onto stack and visit children
        self._branch_stack.append(node.lineno)
        for child in node.body:
            self.visit(child)
        for child in node.orelse:
            self.visit(child)
        if self._branch_stack and self._branch_stack[-1] == node.lineno:
            self._branch_stack.pop()

    def _expr_to_string(self, node: ast.expr) -> str:
        """
        Convert an AST expression node to a string representation.

        This is a fallback for Python versions that don't have ast.unparse.
        """
        # Simple fallback: try to reconstruct a readable string
        # This is a simplified version - for production, consider using
        # a more robust approach or requiring Python 3.9+ for ast.unparse
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Compare):
            left = self._expr_to_string(node.left)
            ops = [self._op_to_string(op) for op in node.ops]
            comparators = [self._expr_to_string(comp) for comp in node.comparators]
            parts = [left]
            for i, op in enumerate(ops):
                parts.append(op)
                parts.append(comparators[i])
            return " ".join(parts)
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.BinOp):
            left = self._expr_to_string(node.left)
            op = self._op_to_string(node.op)
            right = self._expr_to_string(node.right)
            return f"({left} {op} {right})"
        elif isinstance(node, ast.Call):
            func = self._expr_to_string(node.func)
            args = [self._expr_to_string(arg) for arg in node.args]
            return f"{func}({', '.join(args)})"
        elif isinstance(node, ast.Attribute):
            value = self._expr_to_string(node.value)
            return f"{value}.{node.attr}"
        else:
            # Fallback for unknown node types
            return f"<{type(node).__name__}>"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function definition and check for @flex_rank/@flex_group decorators."""
        self._check_decorators(node.decorator_list, node.name, node.lineno)
        # Visit function body for nested branches
        for child in node.body:
            self.visit(child)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit an async function definition and check for decorators."""
        self._check_decorators(node.decorator_list, node.name, node.lineno)
        # Visit function body for nested branches
        for child in node.body:
            self.visit(child)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition and check for decorators."""
        self._check_decorators(node.decorator_list, node.name, node.lineno)
        # Visit class body for nested branches
        for child in node.body:
            self.visit(child)

    def _check_decorators(self, decorator_list: List[ast.expr], target_name: str, target_lineno: int) -> None:
        """
        Check decorator list for @flex_rank or @flex_group decorators.

        Parameters
        ----------
        decorator_list : List[ast.expr]
            List of decorator expressions
        target_name : str
            Name of the function/class being decorated
        target_lineno : int
            Line number of the function/class definition
        """
        for decorator in decorator_list:
            # Check if it's a call (e.g., @flex_rank(...))
            if isinstance(decorator, ast.Call):
                func_name = self._get_decorator_name(decorator.func)
                if func_name in ("flex_rank", "flex_group"):
                    self._extract_flex_decorator(decorator, func_name, target_name, target_lineno)
            # Check if it's a name (e.g., @flex_rank without parentheses)
            elif isinstance(decorator, ast.Name):
                if decorator.id in ("flex_rank", "flex_group"):
                    self.decorators.append(
                        FlexDecorator(
                            decorator_type=decorator.id,
                            lineno=decorator.lineno,
                            col_offset=decorator.col_offset,
                            target_name=target_name,
                        )
                    )

    def _get_decorator_name(self, node: ast.expr) -> str:
        """Extract the name from a decorator expression."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # Handle cases like @module.flex_rank
            return node.attr
        return ""

    def _extract_flex_decorator(
        self, decorator: ast.Call, decorator_type: str, target_name: str, target_lineno: int
    ) -> None:
        """
        Extract arguments from a @flex_rank or @flex_group decorator call.

        Parameters
        ----------
        decorator : ast.Call
            The decorator call node
        decorator_type : str
            Either "flex_rank" or "flex_group"
        target_name : str
            Name of the function/class being decorated
        target_lineno : int
            Line number of the function/class definition
        """
        active_ranks = None
        active_group = None

        # Extract keyword arguments
        for keyword in decorator.keywords:
            if keyword.arg == "active_ranks":
                # Try to evaluate the argument value if it's a constant
                active_ranks = self._extract_value(keyword.value)
            elif keyword.arg == "active_group":
                active_group = self._extract_value(keyword.value)

        self.decorators.append(
            FlexDecorator(
                decorator_type=decorator_type,
                lineno=decorator.lineno,
                col_offset=decorator.col_offset,
                active_ranks=active_ranks,
                active_group=active_group,
                target_name=target_name,
            )
        )

    def _extract_value(self, node: ast.expr) -> Any:
        """
        Extract a value from an AST expression node.

        This attempts to evaluate constant expressions. For complex expressions,
        returns a string representation.
        """
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Tuple) or isinstance(node, ast.List):
            return [self._extract_value(item) for item in node.elts]
        elif isinstance(node, ast.Set):
            return {self._extract_value(item) for item in node.elts}
        else:
            # For complex expressions, return string representation
            return ast.unparse(node) if hasattr(ast, "unparse") else self._expr_to_string(node)

    def _op_to_string(self, op: ast.operator | ast.cmpop) -> str:
        """Convert an AST operator to its string representation."""
        op_map = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Is: "is",
            ast.IsNot: "is not",
            ast.In: "in",
            ast.NotIn: "not in",
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
            ast.And: "and",
            ast.Or: "or",
        }
        return op_map.get(type(op), str(op))

    def _is_name_main_check(self, node: ast.expr) -> bool:
        """
        Check if an expression is `__name__ == '__main__'`.
        
        This is a common pattern that doesn't need instrumentation
        as it's not rank-based control flow.
        """
        if isinstance(node, ast.Compare):
            # Check if it's a comparison with exactly one operator
            if len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
                # Check if left side is __name__
                if isinstance(node.left, ast.Name) and node.left.id == "__name__":
                    # Check if right side is '__main__'
                    if len(node.comparators) == 1:
                        comparator = node.comparators[0]
                        if isinstance(comparator, ast.Constant) and comparator.value == "__main__":
                            return True
                        # Also handle string literals in older Python versions
                        if isinstance(comparator, ast.Str) and comparator.s == "__main__":
                            return True
        return False

