"""
Phase III Code Generation: SPSD-Transformed Code with Branch Arm Expansion

NEW DESIGN (per task-instruction.md):
- Preserve main() function structure
- Define _execute_branch_<id>() and _execute_arm_<id>() functions outside main
- Use explicit active_ranks parameter
- Transform at line level (R-variant lines → loops)
- Use resolve()/global_resolve() instead of default_resolve()
"""

import ast
import re
from typing import Dict, List, Set, Optional, Tuple, Any
from pathlib import Path

from pyextend.runtime.phase3 import ExecutionStep, BranchClassification


def extract_main_function(source_code: str) -> Tuple[Optional[str], Optional[ast.FunctionDef]]:
    """
    Extract the main() function from source code.
    
    Parameters
    ----------
    source_code : str
        Full source code
    
    Returns
    -------
    Tuple[Optional[str], Optional[ast.FunctionDef]]
        (main function code, AST node)
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return None, None
    
    # Find main function
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'main':
            # Get source lines for main function
            lines = source_code.splitlines()
            start_line = node.lineno - 1
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1
            
            main_code = '\n'.join(lines[start_line:end_line])
            return main_code, node
    
    return None, None


def extract_decorated_functions(source_code: str, decorator_map: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Extract decorated function definitions from source code.

    Parameters
    ----------
    source_code : str
        Full source code (instrumented)
    decorator_map : Dict[str, Dict[str, Any]]
        Mapping from function name to decorator metadata

    Returns
    -------
    List[str]
        List of function definition code strings (with decorators)
    """
    if not decorator_map:
        return []

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    decorated_funcs = []
    lines = source_code.splitlines()

    # Find all function definitions that are decorated
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            if func_name in decorator_map:
                # Extract the function definition including decorators
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1

                # Extract function definition WITHOUT decorators
                # In Phase III, we handle decorator semantics in the transformation,
                # so we don't need the decorators on the function definitions
                # Just extract the function body
                func_code = '\n'.join(lines[start_line:end_line])
                decorated_funcs.append(func_code)

    return decorated_funcs


def extract_helper_functions(source_code: str) -> List[str]:
    """
    Extract all top-level helper function definitions (excluding main).

    This extracts functions that are not the main function, not decorated
    with SPMD decorators, and not inside the __main__ block.

    Parameters
    ----------
    source_code : str
        Full source code

    Returns
    -------
    List[str]
        List of helper function code strings
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    helper_funcs = []
    lines = source_code.splitlines()

    # Find top-level function definitions (direct children of Module)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            # Skip 'main' function - it's handled separately
            if func_name == 'main':
                continue

            # Extract the function definition
            start_line = node.lineno - 1
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1

            # Include any decorators
            if node.decorator_list:
                first_decorator = node.decorator_list[0]
                start_line = first_decorator.lineno - 1

            func_code = '\n'.join(lines[start_line:end_line])
            helper_funcs.append(func_code)

    return helper_funcs


def extract_branch_arm_code(
    source_code: str,
    branch_id: int,
    path_type: str,
    branch_metadata: Dict[int, any],
    ast_tree: Optional[ast.Module]
) -> str:
    """
    Extract code for a branch arm (true or false path) from instrumented source.
    
    Parameters
    ----------
    source_code : str
        Full source code (instrumented)
    branch_id : int
        Branch identifier (line number from original code, used in mark_cond)
    path_type : str
        "true" or "false"
    branch_metadata : Dict[int, any]
        Branch metadata from astparser
    ast_tree : Optional[ast.Module]
        Parsed AST of the source code
    
    Returns
    -------
    str
        Code for the branch arm
    """
    lines = source_code.splitlines()
    
    # Find the line with mark_cond that has this branch_id
    branch_line_idx = None
    for i, line in enumerate(lines, 1):
        if f'mark_cond(' in line and f', {branch_id})' in line:
            branch_line_idx = i
            break
    
    if branch_line_idx is None:
        return f"# Code for Branch {branch_id} ({path_type} path) - branch line not found"
    
    branch_line = lines[branch_line_idx - 1]
    indent = len(branch_line) - len(branch_line.lstrip())
    
    if path_type == "true":
        # Extract body code (lines after the if statement with greater indent)
        body_lines = []
        base_indent = None
        
        for j in range(branch_line_idx, len(lines)):
            next_line = lines[j]
            if not next_line.strip():
                continue
            
            next_indent = len(next_line) - len(next_line.lstrip())
            if next_indent > indent:
                # This is part of the body
                if base_indent is None:
                    base_indent = next_indent
                
                # Remove mark_cond wrapper if present in the line
                clean_line = next_line
                if 'mark_cond(' in clean_line:
                    pattern = r'mark_cond\s*\((.*?),\s*\d+\)'
                    match = re.search(pattern, clean_line)
                    if match:
                        condition = match.group(1)
                        clean_line = clean_line.replace(match.group(0), condition)
                
                # Normalize indentation - remove base_indent from all lines
                # But preserve relative indentation within the body
                if base_indent is not None:
                    current_indent = len(next_line) - len(next_line.lstrip())
                    relative_indent = current_indent - base_indent
                    # Keep only the relative indentation (normalize to base)
                    if relative_indent > 0:
                        clean_line = "    " * (relative_indent // 4) + clean_line.lstrip()
                    else:
                        clean_line = clean_line.lstrip()
                
                body_lines.append(clean_line)
            elif next_indent <= indent and next_line.strip():
                # Reached next statement at same or lower indent level
                break
        
        if body_lines:
            return "\n".join(body_lines)
        else:
            return "pass"
    else:
        # Extract else/elif code - need to extract the entire elif chain
        else_lines = []
        found_else = False
        
        # Find the first elif/else at the same indent level
        first_elif_idx = None
        for j in range(branch_line_idx, len(lines)):
            next_line = lines[j]
            if not next_line.strip():
                continue
            
            next_indent = len(next_line) - len(next_line.lstrip())
            
            if next_indent == indent:
                if 'elif' in next_line or ('else' in next_line and ':' in next_line):
                    first_elif_idx = j
                    found_else = True
                    break
        
        if first_elif_idx is not None:
            # Extract the entire elif chain (all elif branches and else)
            # Convert the first elif to an if statement for the false path
            elif_indent = indent
            k = first_elif_idx
            is_first_elif = True
            
            while k < len(lines):
                current_line = lines[k]
                if not current_line.strip():
                    k += 1
                    continue
                
                current_indent = len(current_line) - len(current_line.lstrip())
                
                # Check if this is an elif or else at the same level
                if current_indent == elif_indent:
                    if 'elif' in current_line or ('else' in current_line and ':' in current_line):
                        # Check if this is a bare 'else:' (not elif)
                        is_bare_else = 'elif' not in current_line and 'else' in current_line and ':' in current_line

                        # Convert first elif to if, keep others as elif/else
                        clean_line = current_line
                        if 'mark_cond(' in clean_line:
                            pattern = r'mark_cond\s*\((.*?),\s*\d+\)'
                            match = re.search(pattern, clean_line)
                            if match:
                                condition = match.group(1)
                                clean_line = clean_line.replace(match.group(0), condition)

                        # Convert first elif to if statement
                        if is_first_elif and 'elif' in clean_line:
                            clean_line = clean_line.replace('elif', 'if', 1)
                            is_first_elif = False

                        # Normalize indentation
                        if len(clean_line) >= elif_indent:
                            clean_line = clean_line[elif_indent:]
                        else:
                            clean_line = clean_line.lstrip()

                        # Only add the else: line if it's part of an elif chain (not a bare else)
                        # For bare else:, we skip it and only extract the body
                        if not is_bare_else:
                            else_lines.append(clean_line)
                        
                        # Extract the body of this elif/else
                        k += 1
                        while k < len(lines):
                            body_line = lines[k]
                            if not body_line.strip():
                                k += 1
                                continue
                            body_indent = len(body_line) - len(body_line.lstrip())
                            
                            if body_indent > elif_indent:
                                # This is part of the elif/else body
                                clean_body_line = body_line
                                if 'mark_cond(' in clean_body_line:
                                    pattern = r'mark_cond\s*\((.*?),\s*\d+\)'
                                    match = re.search(pattern, clean_body_line)
                                    if match:
                                        condition = match.group(1)
                                        clean_body_line = clean_body_line.replace(match.group(0), condition)
                                # Normalize indentation
                                if len(clean_body_line) >= elif_indent:
                                    clean_body_line = clean_body_line[elif_indent:]
                                else:
                                    clean_body_line = clean_body_line.lstrip()
                                else_lines.append(clean_body_line)
                                k += 1
                            elif body_indent == elif_indent and ('elif' in body_line or ('else' in body_line and ':' in body_line)):
                                # Another elif/else at same level - continue the chain
                                break
                            else:
                                # Reached end of elif chain
                                break
                    else:
                        # Not an elif/else, we've reached the end
                        break
                else:
                    # Different indent level, we've left the elif chain
                    break
        
        if else_lines:
            return "\n".join(else_lines)
        elif found_else:
            return "pass"
        else:
            return "# No else block"
    

def detect_r_variant_in_line(line: str, r_variant_names: Set[str]) -> bool:
    """
    Check if a line contains R-variant names.
    
    Parameters
    ----------
    line : str
        Code line
    r_variant_names : Set[str]
        Set of R-variant names
    
    Returns
    -------
    bool
        True if line contains R-variants
    """
    for var_name in r_variant_names:
        pattern = r'\b' + re.escape(var_name) + r'\b'
        if re.search(pattern, line):
            return True
    return False


def transform_assignment_to_global_assign(
    line: str,
    r_variant_names: Set[str],
    global_names: Set[str]
) -> str:
    """
    Transform a variable assignment to use env.global_assign() for computed variables.
    
    R-variants use env.assign(), known globals use env.assign_global(),
    other computed variables use env.global_assign().
    
    Parameters
    ----------
    line : str
        Assignment line (e.g., "steps = 10" or "loss = 1.0 / (step + 1)")
    r_variant_names : Set[str]
        Set of R-variant names
    global_names : Set[str]
        Set of global variable names
    
    Returns
    -------
    str
        Transformed assignment line
    """
    if '=' not in line or line.strip().startswith('if') or line.strip().startswith('for'):
        return line
    
    # Extract variable name and value
    parts = line.split('=', 1)
    if len(parts) != 2:
        return line
    
    left_side = parts[0].strip()
    right_side = parts[1].strip()
    
    # Extract variable name (handle multiple assignment, unpacking, etc.)
    # Simple case: single variable assignment
    var_name_match = re.match(r'^(\w+)', left_side)
    if not var_name_match:
        return line
    
    var_name = var_name_match.group(1)
    
    # Determine assignment type first
    if var_name in r_variant_names:
        # R-variant: use env.assign() in a loop
        # Transform right side: use 'rank' directly for R-variants, resolve others
        transformed_right = transform_line_to_use_resolve(
            right_side, r_variant_names, global_names, rank_var="rank"
        )
        # Replace env.resolve(rank, "rank") with just 'rank' since we're in a loop
        transformed_right = re.sub(r'env\.resolve\(rank,\s*["\']rank["\']\)', 'rank', transformed_right)
        # Also replace env.resolve(rank, "var") with env.resolve(rank, "var") for other R-variants
        # But for 'rank' itself, use the loop variable directly
        # env.assign() signature: assign(name, rank, value)
        return f"env.assign('{var_name}', rank, {transformed_right})"  # Will be wrapped in loop
    
    # Transform right side to use global_resolve for computed variables
    transformed_right = transform_line_to_use_resolve(
        right_side, r_variant_names, global_names, rank_var="rank"
    )
    
    if var_name in global_names:
        # Known global: use env.assign_global()
        # Skip if it's assigning to itself (redundant)
        if f"env.global_resolve('{var_name}')" in transformed_right or f'env.global_resolve("{var_name}")' in transformed_right:
            return None  # Signal to skip this line
        return f"env.assign_global('{var_name}', {transformed_right})"
    else:
        # Computed variable: use env.global_assign()
        return f"env.global_assign('{var_name}', {transformed_right})"


def transform_line_to_use_resolve(
    line: str,
    r_variant_names: Set[str],
    global_names: Set[str],
    rank_var: str = "rank"
) -> str:
    """
    Transform a line to use env.resolve() or env.global_resolve() instead of direct variable access.
    
    Replaces:
    - R-variant access: `var_name` → `env.resolve(rank, "var_name")`
    - Global access: `var_name` → `env.global_resolve("var_name")`
    - Already transformed: `env.default_resolve("var_name")` → `env.resolve(rank, "var_name")` or `env.global_resolve("var_name")`
    
    Parameters
    ----------
    line : str
        Original code line
    r_variant_names : Set[str]
        Set of R-variant names
    global_names : Set[str]
        Set of global variable names
    rank_var : str
        Name of the rank variable in the loop (default: "rank")
    
    Returns
    -------
    str
        Transformed line
    """
    if not line.strip() or line.strip().startswith('#'):
        return line
    
    transformed = line
    
    # Replace computed variable access with env.global_resolve()
    # Extract all variable names from the line (simple heuristic)
    var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
    all_vars = set(re.findall(var_pattern, transformed))
    
    # Filter out: Python keywords, already resolved vars, built-ins
    # NOTE: R-variants and globals should NOT be skipped - they need to be transformed
    # But the loop variable (rank_var) should be skipped since it's the loop variable
    python_keywords = {'if', 'else', 'elif', 'for', 'while', 'def', 'return', 'print', 
                      'range', 'len', 'int', 'str', 'float', 'list', 'dict', 'set',
                      'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is', 'env', 'active_ranks'}
    skip_vars = python_keywords | {rank_var}  # Skip the loop variable itself
    
    # Check if this is a for loop statement - if so, don't transform the loop variable in the "for var in" part
    is_for_loop = transformed.strip().startswith('for ')
    loop_var = None
    if is_for_loop:
        # Extract loop variable: "for var in ..."
        for_match = re.match(r'for\s+(\w+)\s+in', transformed)
        if for_match:
            loop_var = for_match.group(1)
            skip_vars.add(loop_var)
    
    # Handle f-strings specially - they need to preserve the f-string structure
    # Pattern: f'...' or f"..."
    # More robust pattern that handles nested braces and escaped quotes
    f_string_placeholders = {}
    placeholder_counter = 0
    
    def replace_f_string(match):
        nonlocal placeholder_counter
        f_string = match.group(0)
        placeholder = f"__F_STRING_{placeholder_counter}__"
        placeholder_counter += 1
        
        # Extract quote character and content
        quote_char = f_string[1]  # ' or "
        content_start = 2
        content_end = len(f_string) - 1
        f_content = f_string[content_start:content_end]
        
        # Find all {var_name} or {var_name:format} patterns in the f-string
        brace_pattern = r'\{([^}]+)\}'
        
        def transform_brace(m):
            brace_content = m.group(1)
            # Check if it has formatting like {var:.4f}
            if ':' in brace_content:
                var_part, fmt_part = brace_content.split(':', 1)
                var_name = var_part.strip()
                fmt_spec = ':' + fmt_part
            else:
                var_name = brace_content.strip()
                fmt_spec = ''
            
            # Transform the variable
            if var_name in r_variant_names:
                transformed_var = f'env.resolve({rank_var}, "{var_name}")'
            elif var_name in global_names:
                transformed_var = f'env.global_resolve("{var_name}")'
            else:
                # Computed variable
                transformed_var = f'env.global_resolve("{var_name}")'
            
            return f'{{{transformed_var}{fmt_spec}}}'
        
        transformed_content = re.sub(brace_pattern, transform_brace, f_content)
        transformed_f_string = f"f{quote_char}{transformed_content}{quote_char}"
        f_string_placeholders[placeholder] = transformed_f_string
        return placeholder
    
    # Replace f-strings with placeholders first
    if "f'" in transformed or 'f"' in transformed:
        # Pattern: f'...' or f"..."
        f_string_pattern = r"f(['\"])([^'\"]*?)\1"
        transformed = re.sub(f_string_pattern, replace_f_string, transformed)
    
    for var_name in all_vars - skip_vars:
        # Skip if already in resolve format
        # Check for exact patterns to avoid false positives
        # Pattern: env.resolve(rank, "var_name") or env.resolve(rank, 'var_name')
        escaped_var = re.escape(var_name)
        resolve_patterns = [
            rf'env\.resolve\({re.escape(rank_var)},\s*["\']{escaped_var}["\']\)',
            rf'env\.global_resolve\(["\']{escaped_var}["\']\)',
            rf'env\.global_assign\(["\']{escaped_var}["\']',
            rf'env\.assign_global\(["\']{escaped_var}["\']',
        ]
        if any(re.search(pattern, transformed) for pattern in resolve_patterns):
            continue
        
        # Skip if inside an f-string (already transformed)
        if f"f'" in transformed or 'f"' in transformed:
            # Check if this variable is inside braces in an f-string
            if re.search(rf'f["\'].*?\{{{re.escape(var_name)}', transformed):
                continue
        
        # Skip if being assigned to (left side of =, but not == or !=)
        # Check for assignment operator, not comparison operators
        if '=' in transformed:
            # Check if it's an assignment (=) or comparison (==, !=, <=, >=)
            assignment_match = re.search(rf'\b{re.escape(var_name)}\s*=\s*[^=]', transformed)
            if assignment_match:
                # This is an assignment, skip transforming the variable being assigned to
                continue
        
        # Skip if it's a function call (e.g., "print(...)", "range(...)")
        if f'{var_name}(' in transformed:
            continue
        
        # Determine replacement based on variable type
        # Process variables one at a time, checking if they're already resolved
        pattern = r'\b' + re.escape(var_name) + r'\b'
        
        def replace_var(match):
            # Get the full match context
            full_text = transformed
            start = match.start()
            end = match.end()
            
            # Check if this variable is inside a string literal
            # Parse the text to find string boundaries
            before = full_text[:start]
            after = full_text[end:]
            
            # Track if we're inside a string by parsing backwards
            # We need to find the most recent unescaped quote
            in_string = False
            string_char = None
            i = len(before) - 1
            while i >= 0:
                char = before[i]
                if char in ("'", '"'):
                    # Check if it's escaped
                    if i == 0 or before[i-1] != '\\':
                        # Found an unescaped quote
                        string_char = char
                        # Check if there's a matching closing quote after our position
                        # Look for the next unescaped quote of the same type
                        for j in range(len(after)):
                            if after[j] == string_char and (j == 0 or after[j-1] != '\\'):
                                # We're inside a string literal - don't transform
                                return match.group(0)
                        break
                i -= 1
            
            # Also check if we're inside triple quotes (""" or ''')
            # Check for triple quotes before our position
            if '"""' in before or "'''" in before:
                # Simple heuristic: if we see triple quotes, be conservative
                # and check if we might be inside them
                triple_double = before.rfind('"""')
                triple_single = before.rfind("'''")
                triple_pos = max(triple_double, triple_single)
                if triple_pos != -1:
                    # Check if there's a closing triple quote after our position
                    remaining = full_text[triple_pos:]
                    if triple_double > triple_single:
                        if remaining.count('"""') >= 2:
                            # We might be inside a triple-quoted string
                            return match.group(0)
                    else:
                        if remaining.count("'''") >= 2:
                            # We might be inside a triple-quoted string
                            return match.group(0)
            
            # Check if this variable is already inside a resolve call
            # Look backwards for env.resolve( or env.global_resolve(
            # We need to check if we're inside the string argument of a resolve call
            last_resolve_pos = max(
                before.rfind('env.resolve('),
                before.rfind('env.global_resolve(')
            )
            
            if last_resolve_pos != -1:
                # Check if we're inside this resolve call by counting parentheses
                # Extract the substring from the resolve call to our position
                segment = full_text[last_resolve_pos:end]
                # Count opening and closing parentheses
                open_count = segment.count('(')
                close_count = segment.count(')')
                # If there are more opening than closing, we're inside the call
                if open_count > close_count:
                    # We're inside a resolve call, don't replace
                    return match.group(0)
            
            # Check if this variable is already inside a resolve call's string argument
            # Look for patterns like env.resolve(rank, "var_name") or env.global_resolve("var_name")
            # where var_name is in quotes
            # Simple check: if there's a resolve call before this position and we're inside its parentheses
            resolve_before = max(
                before.rfind('env.resolve('),
                before.rfind('env.global_resolve(')
            )
            if resolve_before != -1:
                # Check if we're inside the parentheses of this resolve call
                # Count parentheses from resolve call to our position
                segment = full_text[resolve_before:start]
                open_parens = segment.count('(')
                close_parens = segment.count(')')
                if open_parens > close_parens:
                    # We're inside a resolve call, check if we're in the string argument
                    # Look for a quote before our position (within the resolve call)
                    # Find the most recent quote before our position
                    quote_pos = -1
                    quote_char = None
                    for i in range(start - 1, resolve_before - 1, -1):
                        if full_text[i] in ('"', "'") and (i == 0 or full_text[i-1] != '\\'):
                            quote_pos = i
                            quote_char = full_text[i]
                            break
                    if quote_pos != -1:
                        # Found a quote, check if it's the opening quote of a string containing our var
                        # Find the closing quote
                        for j in range(quote_pos + 1, min(len(full_text), start + 100)):
                            if full_text[j] == quote_char and full_text[j-1] != '\\':
                                # Found closing quote, check if our variable is in this string
                                string_content = full_text[quote_pos+1:j]
                                if var_name in string_content and start >= quote_pos and start < j:
                                    # Variable is already in a resolve call's string argument
                                    return match.group(0)
                                break
            
            # Determine replacement
            if var_name in r_variant_names:
                return f'env.resolve({rank_var}, "{var_name}")'
            elif var_name in global_names:
                return f'env.global_resolve("{var_name}")'
            else:
                return f'env.global_resolve("{var_name}")'
        
        transformed = re.sub(pattern, replace_var, transformed)
    
    # Restore f-strings from placeholders
    for placeholder, f_string in f_string_placeholders.items():
        transformed = transformed.replace(placeholder, f_string)
    
    # First, replace env.default_resolve() calls
    # Pattern: env.default_resolve("var_name") or env.default_resolve('var_name')
    default_resolve_pattern = r'env\.default_resolve\(["\']([^"\']+)["\']\)'
    
    def replace_default_resolve(match):
        var_name = match.group(1)
        if var_name in r_variant_names:
            return f'env.resolve({rank_var}, "{var_name}")'
        elif var_name in global_names:
            return f'env.global_resolve("{var_name}")'
        else:
            # Unknown variable, assume R-variant
            return f'env.resolve({rank_var}, "{var_name}")'
    
    transformed = re.sub(default_resolve_pattern, replace_default_resolve, transformed)

    # NOTE: The redundant loops for R-variant and global replacement have been removed.
    # The comprehensive pass above (lines 501-655) already handles all variable replacement
    # with proper string literal detection. The removed loops were causing double
    # transformation bugs when variables appeared inside resolve call string arguments.

    return transformed


def transform_code_at_line_level(
    code: str,
    r_variant_names: Set[str],
    global_names: Set[str],
    base_indent: str = "    "
) -> List[str]:
    """
    Transform code at line level: R-variant lines → for loops, non-R-variant lines execute once.
    
    Preserves original indentation structure within code blocks.
    
    Parameters
    ----------
    code : str
        Code block to transform
    r_variant_names : Set[str]
        Set of R-variant names
    global_names : Set[str]
        Set of global variable names
    base_indent : str
        Base indentation for generated code
    
    Returns
    -------
    List[str]
        Transformed code lines
    """
    lines = code.splitlines()
    if not lines:
        return []
    
    transformed_lines = []
    
    # Find minimum indentation (base level)
    min_indent = float('inf')
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
    
    if min_indent == float('inf'):
        min_indent = 0
    
    # Track nested control structures
    indent_stack = []
    # Track if we're in a loop created for an if-elif-else chain
    in_if_chain_loop = False
    
    for line in lines:
        if not line.strip():
            transformed_lines.append("")
            continue
        
        # Calculate relative indentation from minimum
        line_stripped = line.lstrip()
        line_indent = len(line) - len(line_stripped)
        relative_indent = line_indent - min_indent
        
        # Determine if this is a control structure
        is_control = (line_stripped.startswith('if ') or 
                     line_stripped.startswith('elif ') or
                     line_stripped.startswith('else:') or
                     line_stripped.startswith('for ') or
                     line_stripped.startswith('while '))
        
        # Check if line has R-variants
        has_r_variant = detect_r_variant_in_line(line, r_variant_names)
        
        # Transform variable access
        transformed_line = transform_line_to_use_resolve(
            line_stripped, r_variant_names, global_names
        )
        
        # Calculate actual indentation for output
        # relative_indent is in spaces, convert to 4-space increments
        indent_level = max(0, relative_indent // 4)
        actual_indent = base_indent + "    " * indent_level
        
        # Handle control structures
        if is_control:
            # Check for else: first (it doesn't start with 'if' or 'elif')
            if transformed_line.strip().startswith('else:'):
                # For else, if we're in a loop, indent it to match if/elif
                if in_if_chain_loop:
                    # else should be at same level as if/elif (inside the loop)
                    # actual_indent is base_indent (4 spaces), add 4 more for loop body
                    transformed_lines.append(actual_indent + "    " + transformed_line)
                    # Track the else block
                    if ':' in transformed_line:
                        if not indent_stack or line_indent not in indent_stack:
                            indent_stack.append(line_indent + 4)  # Add loop indent
                    continue
            
            # Control structure: preserve as is, but transform variable access inside
            # For if statements, transform the condition
            if transformed_line.strip().startswith('if ') or transformed_line.strip().startswith('elif '):
                # Transform variables in the condition (including computed variables like 'step')
                transformed_line = transform_line_to_use_resolve(
                    transformed_line, r_variant_names, global_names, rank_var="rank"
                )
                # Check if this is the first if in a chain and has R-variants
                # If so, wrap the entire if-elif-else chain in a loop
                if transformed_line.strip().startswith('if ') and detect_r_variant_in_line(transformed_line, r_variant_names):
                    # This is the start of an if-elif-else chain with R-variants
                    # Wrap the entire chain in a loop
                    in_if_chain_loop = True
                    transformed_lines.append(actual_indent + "# R-variant condition: execute for each rank")
                    transformed_lines.append(actual_indent + "for rank in active_ranks:")
                    # Indent the if statement
                    transformed_lines.append(actual_indent + "    " + transformed_line)
                    # Track that we're now in a loop context
                    if ':' in transformed_line:
                        if not indent_stack or line_indent not in indent_stack:
                            indent_stack.append(line_indent + 4)  # Add loop indent
                    continue
                elif transformed_line.strip().startswith('elif '):
                    # For elif, if we're already in a loop (from the if), just indent it
                    if in_if_chain_loop:
                        # We're in a loop, indent the elif
                        transformed_lines.append(actual_indent + "    " + transformed_line)
                        continue
            transformed_lines.append(actual_indent + transformed_line)
            if ':' in transformed_line:
                # Track the base indent for the control structure body
                if not indent_stack or line_indent not in indent_stack:
                    indent_stack.append(line_indent)
        elif has_r_variant:
            # R-variant line: wrap in loop (unless we're already in a loop for if-elif-else chain)
            if in_if_chain_loop:
                # Already in a loop, just indent the line
                transformed_lines.append(actual_indent + "        " + transformed_line)
            else:
                # R-variant line: wrap in loop
                transformed_lines.append(actual_indent + "# R-variant line: execute for each rank")
                transformed_lines.append(actual_indent + "for rank in active_ranks:")
                transformed_lines.append(actual_indent + "    " + transformed_line)
        else:
            # Non-R-variant line: execute once
            # Check if we're inside a control structure - if so, preserve relative indentation
            if indent_stack:
                # We're inside a control structure
                control_indent = indent_stack[-1]
                if line_indent > control_indent:
                    # This is part of the control structure body
                    relative_indent = line_indent - control_indent
                    body_indent = actual_indent + "    " * (relative_indent // 4)
                    # If we're in an if-elif-else loop, add extra indent
                    if in_if_chain_loop:
                        body_indent += "    "
                    transformed_lines.append(body_indent + transformed_line)
                elif line_indent <= control_indent:
                    # We've left the control structure
                    indent_stack.pop()
                    # Check if we've left the if-elif-else chain
                    if line_indent <= indent:
                        in_if_chain_loop = False
                    transformed_lines.append(actual_indent + transformed_line)
                else:
                    transformed_lines.append(actual_indent + transformed_line)
            else:
                # Not in a control structure
                if in_if_chain_loop and line_indent > indent:
                    # Still in the if-elif-else chain body
                    transformed_lines.append(actual_indent + "        " + transformed_line)
                else:
                    transformed_lines.append(actual_indent + transformed_line)
                    if line_indent <= indent:
                        in_if_chain_loop = False
    
    return transformed_lines


def build_decorator_map(decorators: Optional[List[Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a mapping from function name to decorator metadata.
    
    Parameters
    ----------
    decorators : Optional[List[Any]]
        List of FlexDecorator objects from astparser
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping: function_name -> {'type': 'flex_rank'|'flex_group', 'active_ranks': ... or 'active_group': ...}
    """
    if decorators is None:
        return {}
    
    decorator_map = {}
    for decorator in decorators:
        if hasattr(decorator, 'target_name') and decorator.target_name:
            func_name = decorator.target_name
            if decorator.decorator_type == 'flex_rank':
                decorator_map[func_name] = {
                    'type': 'flex_rank',
                    'active_ranks': decorator.active_ranks
                }
            elif decorator.decorator_type == 'flex_group':
                decorator_map[func_name] = {
                    'type': 'flex_group',
                    'active_group': decorator.active_group
                }
    
    return decorator_map


def transform_decorated_function_call(
    line: str,
    decorator_map: Dict[str, Dict[str, Any]],
    indent: str
) -> Optional[str]:
    """
    Transform a decorated function call to SPSD code.
    
    For @flex_rank: Generate a for loop over intersection of active_ranks and decorator's active_ranks
    For @flex_group: Generate code to compute group ranks and execute once for the group
    
    Parameters
    ----------
    line : str
        Original line containing function call
    decorator_map : Dict[str, Dict[str, Any]]
        Mapping from function name to decorator metadata
    indent : str
        Indentation string for the generated code
    
    Returns
    -------
    Optional[str]
        Transformed code (multi-line string) or None if not a decorated function call
    """
    # Pattern to match function calls: func_name(...)
    # This is a simple heuristic - we look for function_name( pattern
    for func_name, decorator_info in decorator_map.items():
        # Check if this line contains a call to the decorated function
        # Pattern: func_name( or func_name (with possible whitespace)
        pattern = r'\b' + re.escape(func_name) + r'\s*\('
        if re.search(pattern, line):
            decorator_type = decorator_info['type']
            
            # Extract the function call part (everything from func_name to the end of the call)
            # Find the position of the function call
            escaped_func_name = re.escape(func_name)
            # Simple pattern to find where the function call starts
            func_pos_match = re.search(escaped_func_name + r'\s*\(', line)
            if func_pos_match:
                call_start = func_pos_match.start()
                # Extract the full function call by finding matching parentheses
                start_pos = func_pos_match.end() - 1  # Position of opening '('
                paren_count = 1
                end_pos = start_pos + 1
                while end_pos < len(line) and paren_count > 0:
                    if line[end_pos] == '(':
                        paren_count += 1
                    elif line[end_pos] == ')':
                        paren_count -= 1
                    end_pos += 1
                
                if paren_count == 0:
                    # Found matching closing paren
                    func_call = line[call_start:end_pos]
                    call_indent = line[:call_start].rstrip()
                else:
                    # Fallback: use simple pattern (no nested parens)
                    simple_match = re.search(escaped_func_name + r'\s*\([^)]*\)', line)
                    if simple_match:
                        func_call = simple_match.group(0)
                        call_indent = line[:simple_match.start()].rstrip()
                    else:
                        func_call = func_name + "()"
                        call_indent = ""
                
                if decorator_type == 'flex_rank':
                    # @flex_rank: Execute once per rank in intersection
                    active_ranks = decorator_info['active_ranks']
                    if active_ranks is None:
                        return None
                    
                    # Convert active_ranks to a list representation
                    if isinstance(active_ranks, (tuple, list)):
                        ranks_str = str(list(active_ranks))
                    else:
                        ranks_str = str(active_ranks)
                    
                    # Generate code:
                    # intersection = [r for r in active_ranks if r in {active_ranks}]
                    # for rank in intersection:
                    #     func_call (with rank passed if needed)
                    
                    result_lines = [
                        f"{indent}# @flex_rank decorated function call: {func_name}",
                        f"{indent}intersection = [r for r in active_ranks if r in {ranks_str}]",
                        f"{indent}if intersection:",
                        f"{indent}    for rank in intersection:",
                    ]
                    
                    # Transform the function call to pass env and rank
                    # Extract function arguments from the call
                    # Find the opening paren after function name
                    func_name_part = func_call.split('(')[0].strip()
                    # Find the matching closing paren by counting depth
                    start_pos = func_call.find('(')
                    if start_pos != -1:
                        depth = 0
                        end_pos = start_pos
                        for i in range(start_pos, len(func_call)):
                            if func_call[i] == '(':
                                depth += 1
                            elif func_call[i] == ')':
                                depth -= 1
                                if depth == 0:
                                    end_pos = i
                                    break
                        
                        if depth == 0:
                            # Extract arguments string
                            existing_args_str = func_call[start_pos + 1:end_pos].strip()
                            
                            # Parse existing arguments (split by comma, but preserve nested calls)
                            existing_args_list = []
                            if existing_args_str:
                                # Split by comma, but skip commas inside parentheses
                                depth = 0
                                current_arg = ""
                                for char in existing_args_str:
                                    if char == '(':
                                        depth += 1
                                        current_arg += char
                                    elif char == ')':
                                        depth -= 1
                                        current_arg += char
                                    elif char == ',' and depth == 0:
                                        if current_arg.strip():
                                            existing_args_list.append(current_arg.strip())
                                        current_arg = ""
                                    else:
                                        current_arg += char
                                if current_arg.strip():
                                    existing_args_list.append(current_arg.strip())
                            
                            # Check if env and rank are already in the arguments (as standalone args)
                            has_env = any(arg.strip() == 'env' for arg in existing_args_list)
                            has_rank = any(arg.strip() == 'rank' for arg in existing_args_list)
                            
                            # Build new argument list: preserve existing args, add missing ones
                            new_args_list = existing_args_list.copy()
                            if not has_env:
                                new_args_list.append('env')
                            if not has_rank:
                                new_args_list.append('rank')
                            
                            new_args = ', '.join(new_args_list)
                            transformed_call = f"{func_name_part}({new_args})"
                        else:
                            # Unmatched parentheses, fallback
                            transformed_call = f"{func_name_part}(env, rank)"
                    else:
                        # No arguments, add env and rank
                        transformed_call = f"{func_name_part}(env, rank)"
                    
                    result_lines.append(f"{indent}        {transformed_call}")
                    
                    return '\n'.join(result_lines)
                
                elif decorator_type == 'flex_group':
                    # @flex_group: Execute once for the group
                    active_group = decorator_info['active_group']
                    if active_group is None:
                        return None
                    
                    # Generate code:
                    # group_ranks = env.get_group_ranks("{active_group}")
                    # group_active_ranks = [r for r in active_ranks if r in group_ranks]
                    # if group_active_ranks:
                    #     func_call (execute once for the group)
                    
                    result_lines = [
                        f"{indent}# @flex_group decorated function call: {func_name}",
                        f"{indent}group_ranks = env.get_group_ranks(\"{active_group}\")",
                        f"{indent}group_active_ranks = [r for r in active_ranks if r in group_ranks]",
                        f"{indent}if group_active_ranks:",
                    ]
                    
                    # The function call executes once for the group (not per rank)
                    # Extract function arguments from the call
                    call_args_match = re.search(r'\(([^)]*)\)', func_call)
                    if call_args_match:
                        existing_args = call_args_match.group(1).strip()
                        # Check if env is already in the arguments
                        if 'env' not in existing_args:
                            if existing_args:
                                new_args = f"env, {existing_args}"
                            else:
                                new_args = "env"
                        else:
                            new_args = existing_args
                        
                        # Reconstruct function call
                        func_name_part = func_call.split('(')[0].strip()
                        transformed_call = f"{func_name_part}({new_args})"
                    else:
                        # No arguments, add env
                        func_name_part = func_call.split('(')[0].strip()
                        transformed_call = f"{func_name_part}(env)"
                    
                    result_lines.append(f"{indent}    {transformed_call}")
                    
                    return '\n'.join(result_lines)
    
    return None


def find_branch_points_in_main(main_code: str, branch_ids: Set[int]) -> List[Tuple[int, int]]:
    """
    Find branch points (mark_cond calls) in main() function.
    
    Returns list of (line_number, branch_id) tuples.
    
    Parameters
    ----------
    main_code : str
        Main function code
    branch_ids : Set[int]
        Set of branch IDs to look for
    
    Returns
    -------
    List[Tuple[int, int]]
        List of (line_number, branch_id) tuples
    """
    lines = main_code.splitlines()
    branch_points = []
    
    for i, line in enumerate(lines, 1):
        for branch_id in branch_ids:
            if f'mark_cond(' in line and f', {branch_id})' in line:
                branch_points.append((i, branch_id))
                break
    
    return sorted(branch_points)


def generate_spsd_code(
    execution_path: List[ExecutionStep],
    source_code: str,
    branch_metadata: Dict[int, any],
    classifications: Dict[int, BranchClassification],
    r_variant_names: Set[str],
    output_path: Path,
    decorators: Optional[List[any]] = None
) -> str:
    """
    Generate SPSD-transformed code with new design:
    - Preserve main() structure
    - Define functions outside main
    - Use explicit active_ranks parameter
    - Transform at line level
    - Use resolve()/global_resolve()
    
    Parameters
    ----------
    execution_path : List[ExecutionStep]
        Sequential execution path
    source_code : str
        Original source code (instrumented)
    branch_metadata : Dict[int, any]
        Branch metadata from astparser
    classifications : Dict[int, BranchClassification]
        Branch classifications
    r_variant_names : Set[str]
        Set of known R-variant variable names
    output_path : Path
        Path to write generated code
    
    Returns
    -------
    str
        Generated SPSD-transformed code
    """
    # Parse AST for code extraction
    try:
        ast_tree = ast.parse(source_code)
    except SyntaxError:
        ast_tree = None
    
    # Extract global variable names (assume world_size is always global)
    global_names = {'world_size'}
    
    # Extract world_size from source code
    world_size = 8  # Default
    world_size_match = re.search(r'world_size\s*=\s*(\d+)', source_code)
    if world_size_match:
        world_size = int(world_size_match.group(1))
    
    # Extract main function
    main_code, main_ast = extract_main_function(source_code)
    if main_code is None:
        raise ValueError("Could not find main() function in source code")
    
    # Build decorator map
    decorator_map = build_decorator_map(decorators)
    
    # Extract decorated function definitions
    decorated_funcs = extract_decorated_functions(source_code, decorator_map)
    
    code_lines = [
        "# SPSD-Transformed Code (New Design)",
        "# Generated by Phase III: Single Sequential Execution Path with Dynamic active_ranks",
        "",
        "# Design:",
        "# - Main function preserved (looks like original format)",
        "# - Branch/arm functions defined outside main",
        "# - Explicit active_ranks parameter",
        "# - Line-level transformation (R-variant lines → loops)",
        "# - Variables resolved when needed using resolve()/global_resolve()",
        "",
        "from pyextend.runtime.simenv import SimEnv",
        "from pyextend.runtime.rvariant import R",
        "from typing import List",
        "import bindlayer",
        "",
    ]
    
    # Extract and add helper function definitions (non-main functions)
    helper_funcs = extract_helper_functions(source_code)
    if helper_funcs:
        code_lines.append("# Helper function definitions (preserved from original)")
        for func_code in helper_funcs:
            code_lines.append(func_code)
            code_lines.append("")

    # Add decorated function definitions (without decorators, since we handle decorator semantics in transformation)
    if decorated_funcs:
        code_lines.append("# Decorated function definitions (decorators removed - handled in transformation)")
        for func_code in decorated_funcs:
            code_lines.append(func_code)
            code_lines.append("")
    
    # Collect all unique branches from execution path
    unique_branches = {}  # branch_id -> {'true': step, 'false': step}
    branch_ids = set()
    
    for step in execution_path:
        if step.branch_id is not None:
            branch_ids.add(step.branch_id)
            if step.branch_id not in unique_branches:
                unique_branches[step.branch_id] = {
                    'true': None,
                    'false': None
                }
            if step.path_type:
                unique_branches[step.branch_id][step.path_type] = step
    
    # If execution path is empty, still generate branch functions for RANK_PARTITIONING branches
    # This handles cases where branches are inside loops or execution path builder returns empty
    if not unique_branches and classifications:
        for branch_id, classification in classifications.items():
            if classification.branch_type.value == 'RANK_PARTITIONING':
                branch_ids.add(branch_id)
                if branch_id not in unique_branches:
                    unique_branches[branch_id] = {
                        'true': None,
                        'false': None
                    }
    
    # Helper functions to find elif branches
    def find_first_elif_branch(parent_id: int) -> Optional[int]:
        """Find the first elif branch that is a child of parent_id."""
        for bid in sorted(branch_ids):
            meta = branch_metadata.get(bid)
            if meta and hasattr(meta, 'branch_type') and meta.branch_type.value == 'elif':
                if meta.parent_lineno == parent_id:
                    return bid
        return None
    
    def find_next_elif_branch(elif_id: int) -> Optional[int]:
        """Find the next elif branch after elif_id in the chain."""
        meta = branch_metadata.get(elif_id)
        if not meta or not hasattr(meta, 'branch_type') or meta.branch_type.value != 'elif':
            return None
        
        # Find the original if branch by following parent_lineno chain
        # parent_lineno for elif points to previous elif/if in chain
        current_id = elif_id
        original_if_id = None
        
        # Follow the parent chain to find the original if
        visited = set()
        while current_id is not None and current_id not in visited:
            visited.add(current_id)
            current_meta = branch_metadata.get(current_id)
            if not current_meta:
                break
            
            if current_meta.branch_type.value == 'if':
                original_if_id = current_id
                break
            
            # Move to parent
            current_id = current_meta.parent_lineno
        
        if original_if_id is None:
            return None
        
        # Find all elif branches that belong to this if chain
        # They all share the same nest_path and their parent chain leads to original_if_id
        elif_branches = []
        for bid in sorted(branch_ids):
            b_meta = branch_metadata.get(bid)
            if (b_meta and hasattr(b_meta, 'branch_type') and 
                b_meta.branch_type.value == 'elif'):
                # Check if this elif belongs to the same chain
                # Follow its parent chain to see if it leads to original_if_id
                check_id = bid
                check_visited = set()
                while check_id is not None and check_id not in check_visited:
                    check_visited.add(check_id)
                    check_meta = branch_metadata.get(check_id)
                    if not check_meta:
                        break
                    if check_id == original_if_id:
                        elif_branches.append((b_meta.lineno, bid))
                        break
                    if check_meta.branch_type.value == 'if':
                        break
                    check_id = check_meta.parent_lineno
        
        # Sort by line number to get the correct order
        elif_branches.sort()
        
        # Find the current elif in the list
        for i, (lineno, bid) in enumerate(elif_branches):
            if bid == elif_id:
                # Found current elif, return the next one
                if i + 1 < len(elif_branches):
                    return elif_branches[i + 1][1]
                break
        
        return None
    
    # Generate branch functions first (they call arm functions)
    branch_functions = {}  # branch_id -> function_name
    function_counter = 0
    
    for branch_id in sorted(unique_branches.keys()):
        function_counter += 1
        func_name = f"_execute_branch_{branch_id}_{function_counter}"
        branch_functions[branch_id] = func_name
        
        code_lines.append(f"def {func_name}(active_ranks: List[int], env: SimEnv):")
        code_lines.append(f"    \"\"\"Branch function for Branch {branch_id}\"\"\"")
        code_lines.append(f"    bindlayer.set_active_ranks(active_ranks)")
        code_lines.append("")
        
        # Get classification to determine which ranks should execute each path
        classification = classifications.get(branch_id)
        if classification:
            # Filter active_ranks based on classification
            true_ranks_set = classification.true_ranks
            false_ranks_set = classification.false_ranks
            
            # Execute true path with filtered ranks
            # We'll generate the arm function name here, but generate the function later
            true_arm_name = f"_execute_arm_{branch_id}_true"
            if true_ranks_set:
                code_lines.append(f"    # True path: execute for ranks {sorted(true_ranks_set)}")
                code_lines.append(f"    true_ranks = [r for r in active_ranks if r in {sorted(true_ranks_set)}]")
                code_lines.append(f"    if true_ranks:")
                code_lines.append(f"        {true_arm_name}(true_ranks, env)")
            
            # Execute false path with filtered ranks
            false_arm_name = f"_execute_arm_{branch_id}_false"
            if false_ranks_set:
                code_lines.append(f"    # False path: execute for ranks {sorted(false_ranks_set)}")
                code_lines.append(f"    false_ranks = [r for r in active_ranks if r in {sorted(false_ranks_set)}]")
                code_lines.append(f"    if false_ranks:")
                code_lines.append(f"        {false_arm_name}(false_ranks, env)")
        else:
            # No classification available, execute for all active_ranks
            true_arm_name = f"_execute_arm_{branch_id}_true"
            false_arm_name = f"_execute_arm_{branch_id}_false"
            code_lines.append(f"    {true_arm_name}(active_ranks, env)")
            code_lines.append(f"    {false_arm_name}(active_ranks, env)")
        
        code_lines.append("")
        code_lines.append("")
    
    # Generate arm functions ordered by calling order
    # Track which arm functions we've generated
    generated_arms = set()
    arm_functions = {}  # (branch_id, path_type) -> function_name
    
    def generate_arm_function(branch_id: int, path_type: str):
        """Generate an arm function if not already generated."""
        if (branch_id, path_type) in generated_arms:
            return arm_functions.get((branch_id, path_type))
        
        generated_arms.add((branch_id, path_type))
        func_name = f"_execute_arm_{branch_id}_{path_type}"
        arm_functions[(branch_id, path_type)] = func_name
        
        code_lines.append(f"def {func_name}(active_ranks: List[int], env: SimEnv):")
        code_lines.append(f"    \"\"\"Arm function for Branch {branch_id} ({path_type} path)\"\"\"")
        code_lines.append(f"    bindlayer.set_active_ranks(active_ranks)")
        code_lines.append("")
        
        # Check if this is a false path that should call the next branch
        if path_type == 'false':
            meta = branch_metadata.get(branch_id)
            if meta and hasattr(meta, 'branch_type'):
                if meta.branch_type.value == 'if':
                    # If branch false path: call first elif branch
                    first_elif = find_first_elif_branch(branch_id)
                    if first_elif is not None:
                        elif_branch_func = branch_functions.get(first_elif)
                        if elif_branch_func:
                            code_lines.append(f"    {elif_branch_func}(active_ranks, env)")
                            code_lines.append("")
                            code_lines.append("")
                            return func_name
                elif meta.branch_type.value == 'elif':
                    # Elif branch false path: call next elif or execute else
                    next_elif = find_next_elif_branch(branch_id)
                    if next_elif is not None:
                        next_elif_func = branch_functions.get(next_elif)
                        if next_elif_func:
                            code_lines.append(f"    {next_elif_func}(active_ranks, env)")
                            code_lines.append("")
                            code_lines.append("")
                            return func_name
                    # No next elif: execute else block
                    # Find the parent if branch to extract else block
                    parent_id = meta.parent_lineno
                    if parent_id is not None:
                        # Extract else block from the original if statement
                        lines = source_code.splitlines()
                        # Find the if statement line
                        if_line_idx = None
                        for i, line in enumerate(lines):
                            if f'mark_cond(' in line and f', {parent_id})' in line:
                                if_line_idx = i
                                break
                        
                        if if_line_idx is not None:
                            # Find the else: block at the same indent level
                            indent = len(lines[if_line_idx]) - len(lines[if_line_idx].lstrip())
                            else_line_idx = None
                            
                            # Skip all elif branches to find the else
                            for j in range(if_line_idx + 1, len(lines)):
                                next_line = lines[j]
                                if not next_line.strip():
                                    continue
                                next_indent = len(next_line) - len(next_line.lstrip())
                                if next_indent == indent and 'else:' in next_line:
                                    else_line_idx = j
                                    break
                            
                            if else_line_idx is not None:
                                # Extract the else block body
                                else_lines = []
                                for k in range(else_line_idx + 1, len(lines)):
                                    body_line = lines[k]
                                    if not body_line.strip():
                                        continue
                                    body_indent = len(body_line) - len(body_line.lstrip())
                                    if body_indent > indent:
                                        # This is part of the else body
                                        clean_line = body_line
                                        if 'mark_cond(' in clean_line:
                                            pattern = r'mark_cond\s*\((.*?),\s*\d+\)'
                                            match = re.search(pattern, clean_line)
                                            if match:
                                                condition = match.group(1)
                                                clean_line = clean_line.replace(match.group(0), condition)
                                        # Normalize indentation
                                        if len(clean_line) >= indent:
                                            clean_line = clean_line[indent:]
                                        else:
                                            clean_line = clean_line.lstrip()
                                        else_lines.append(clean_line)
                                    else:
                                        # Reached end of else block
                                        break
                                
                                if else_lines:
                                    else_block = "\n".join(else_lines)
                                    # Transform the else block code
                                    transformed_lines = transform_code_at_line_level(
                                        else_block, r_variant_names, global_names, base_indent="    "
                                    )
                                    code_lines.extend(transformed_lines)
                                    code_lines.append("")
                                    code_lines.append("")
                                    return func_name
                    
                    # Fallback: if we couldn't extract else, just pass
                    code_lines.append("    pass")
                    code_lines.append("")
                    code_lines.append("")
                    return func_name
        
        # Extract and transform arm code
        step = unique_branches.get(branch_id, {}).get(path_type)
        # Extract arm code even if step is None (branches from classifications, not execution_path)
        arm_code = extract_branch_arm_code(
            source_code, branch_id, path_type,
            branch_metadata, ast_tree
        )
        
        if arm_code.strip() and arm_code.strip() != "pass" and not arm_code.strip().startswith("# No else"):
            transformed_lines = transform_code_at_line_level(
                arm_code, r_variant_names, global_names, base_indent="    "
            )
            # Apply decorator transformation to each line
            final_lines = []
            for line in transformed_lines:
                # Check if this line contains a decorated function call
                if decorator_map:
                    # Extract indent
                    line_indent = len(line) - len(line.lstrip())
                    indent_str = " " * line_indent
                    transformed_decorator = transform_decorated_function_call(
                        line, decorator_map, indent_str
                    )
                    if transformed_decorator:
                        # Replace the line with transformed decorator call
                        final_lines.extend(transformed_decorator.split('\n'))
                    else:
                        final_lines.append(line)
                else:
                    final_lines.append(line)
            code_lines.extend(final_lines)
        else:
            code_lines.append("    pass")
        
        code_lines.append("")
        code_lines.append("")
        return func_name
    
    # Generate arm functions in calling order
    # Start with root branches (not elif, not nested)
    for branch_id in sorted(unique_branches.keys()):
        meta = branch_metadata.get(branch_id)
        if meta and hasattr(meta, 'branch_type') and meta.branch_type.value == 'elif':
            continue  # Skip elif branches for now
        
        # Generate true arm (always generate for RANK_PARTITIONING branches)
        classification = classifications.get(branch_id)
        if classification and classification.branch_type.value == 'RANK_PARTITIONING':
            # Always generate true arm for RANK_PARTITIONING branches
            generate_arm_function(branch_id, 'true')
        elif unique_branches.get(branch_id, {}).get('true'):
            generate_arm_function(branch_id, 'true')
        
        # Generate false arm (may call elif branch)
        # Always generate false arm for if branches (they need to call first elif)
        if meta and hasattr(meta, 'branch_type') and meta.branch_type.value == 'if':
            generate_arm_function(branch_id, 'false')
        elif classification and classification.branch_type.value == 'RANK_PARTITIONING':
            # Always generate false arm for RANK_PARTITIONING branches
            generate_arm_function(branch_id, 'false')
        elif unique_branches.get(branch_id, {}).get('false'):
            generate_arm_function(branch_id, 'false')
    
    # Generate elif branch arms
    for branch_id in sorted(unique_branches.keys()):
        meta = branch_metadata.get(branch_id)
        if meta and hasattr(meta, 'branch_type') and meta.branch_type.value == 'elif':
            # Generate true arm
            if unique_branches.get(branch_id, {}).get('true'):
                generate_arm_function(branch_id, 'true')
            
            # Always generate false arm for elif branches (they need to call next elif or execute else)
            generate_arm_function(branch_id, 'false')
    
    # Generate main function (preserve structure, replace branch points)
    code_lines.append("def main(env: SimEnv):")
    code_lines.append("    \"\"\"Main function (preserved structure)\"\"\"")
    
    # Find branch points in main code
    branch_points = find_branch_points_in_main(main_code, branch_ids)
    branch_point_map = {line_num: branch_id for line_num, branch_id in branch_points}
    
    # Transform main() code: replace branch points with function calls
    main_lines = main_code.splitlines()
    main_body_start = None
    
    # Find where main function body starts (after "def main(env: SimEnv):")
    for i, line in enumerate(main_lines):
        if 'def main' in line:
            # Find the first non-empty line after the function definition
            for j in range(i + 1, len(main_lines)):
                if main_lines[j].strip() and not main_lines[j].strip().startswith('"""'):
                    main_body_start = j
                    break
            break
    
    if main_body_start is None:
        main_body_start = 1
    
    # Add initialization
    code_lines.append("    # Initialize active_ranks")
    code_lines.append("    active_ranks = list(range(env.global_resolve('world_size')))")
    code_lines.append("    bindlayer.set_active_ranks(active_ranks)")
    code_lines.append("")
    code_lines.append("    # Note: Variables like 'rank', 'step', 'loss', 'steps' will be stored as globals")
    code_lines.append("    # when assigned, and resolved using env.global_resolve() when used")
    code_lines.append("")
    
    # Process main() body, replacing branch points and transforming variable access
    i = main_body_start
    loop_stack = []  # Track nested loops: [(loop_start_line, loop_indent, loop_var), ...]
    base_body_indent = None  # Will be set on first iteration
    
    while i < len(main_lines):
        line = main_lines[i]
        line_num = i + 1  # 1-indexed
        original_indent = len(line) - len(line.lstrip())
        
        # Initialize base_body_indent on first iteration
        if base_body_indent is None and line.strip():
            base_body_indent = original_indent
        
        # Check if we've left a loop
        if loop_stack:
            loop_start, loop_indent, loop_var = loop_stack[-1]
            if original_indent <= loop_indent and line.strip() and not line.strip().startswith('#'):
                # We've left the loop
                loop_stack.pop()
        
        # Check if this is a branch point (line with mark_cond)
        if line_num in branch_point_map:
            branch_id = branch_point_map[line_num]
            branch_func = branch_functions.get(branch_id)
            
            if branch_func:
                # Replace branch point with function call
                # Preserve indentation - if we're inside a loop, the branch call should be indented
                relative_indent = original_indent - base_body_indent if base_body_indent is not None and original_indent >= base_body_indent else 0
                output_indent = "    " + "    " * max(0, relative_indent // 4)
                
                code_lines.append(output_indent + f"# Branch {branch_id}: replaced with function call")
                code_lines.append(output_indent + f"{branch_func}(active_ranks, env)")
                
                # Skip the original if statement and its body
                # Find the end of the if/elif/else block
                indent = original_indent
                i += 1
                in_block = False
                while i < len(main_lines):
                    next_line = main_lines[i]
                    if not next_line.strip():
                        i += 1
                        continue
                    next_indent = len(next_line) - len(next_line.lstrip())
                    
                    # Check if we've left the if block
                    if next_indent <= indent and next_line.strip():
                        # Check if it's an elif or else at same level
                        if next_line.strip().startswith('elif') or \
                           (next_line.strip().startswith('else') and ':' in next_line):
                            # Continue in the same block
                            i += 1
                            continue
                        else:
                            # We've left the block
                            break
                    i += 1
                continue
        
        # Transform variable access in the line
        # First, check if this is a for loop - handle specially
        is_for_loop = line.strip().startswith('for ')
        if is_for_loop:
            # Transform: "for step in range(steps):" -> "for step in range(env.global_resolve('steps')):"
            # Extract loop variable and iterable
            for_match = re.match(r'(for\s+)(\w+)(\s+in\s+)(.+)', line.strip())
            if for_match:
                for_prefix = for_match.group(1)
                loop_var = for_match.group(2)
                in_part = for_match.group(3)
                iterable_part = for_match.group(4).rstrip(':')
                
                # Transform the iterable part to use global_resolve for computed variables
                transformed_iterable = transform_line_to_use_resolve(
                    iterable_part, r_variant_names, global_names, rank_var="rank"
                )
                
                # Reconstruct the for loop
                transformed_line = for_prefix + loop_var + in_part + transformed_iterable + ':'
                
                # Preserve indentation
                original_indent = len(line) - len(line.lstrip())
                base_body_indent = len(main_lines[main_body_start]) - len(main_lines[main_body_start].lstrip()) if main_body_start < len(main_lines) else 0
                relative_indent = original_indent - base_body_indent if original_indent >= base_body_indent else 0
                output_indent = "    " + "    " * max(0, relative_indent // 4)
                
                code_lines.append(output_indent + transformed_line)
                # Store loop variable as global at start of loop body
                code_lines.append(output_indent + "    " + f"env.global_assign('{loop_var}', {loop_var})  # Store loop variable as global")
                # Track this loop
                loop_stack.append((i, original_indent, loop_var))
                i += 1
                continue
        
        # Check if this is an if/elif statement - transform variables in condition
        is_if_statement = line.strip().startswith('if ') or line.strip().startswith('elif ')
        if is_if_statement:
            # Remove mark_cond wrapper if present
            transformed_line = line
            if 'mark_cond(' in transformed_line:
                # Pattern: mark_cond(condition, branch_id)
                pattern = r'mark_cond\s*\((.*?),\s*\d+\)'
                match = re.search(pattern, transformed_line)
                if match:
                    condition = match.group(1)
                    transformed_line = transformed_line.replace(match.group(0), condition)
            
            # Transform variables in the condition
            transformed_line = transform_line_to_use_resolve(
                transformed_line, r_variant_names, global_names, rank_var="rank"
            )
            
            # Preserve indentation
            original_indent = len(line) - len(line.lstrip())
            base_body_indent = len(main_lines[main_body_start]) - len(main_lines[main_body_start].lstrip()) if main_body_start < len(main_lines) else 0
            relative_indent = original_indent - base_body_indent if original_indent >= base_body_indent else 0
            output_indent = "    " + "    " * max(0, relative_indent // 4)
            
            code_lines.append(output_indent + transformed_line.lstrip())
            i += 1
            continue
        
        # Check if this is a decorated function call
        if decorator_map:
            original_indent = len(line) - len(line.lstrip())
            base_body_indent = len(main_lines[main_body_start]) - len(main_lines[main_body_start].lstrip()) if main_body_start < len(main_lines) else 0
            relative_indent = original_indent - base_body_indent if original_indent >= base_body_indent else 0
            output_indent = "    " + "    " * max(0, relative_indent // 4)
            
            transformed_decorator_call = transform_decorated_function_call(
                line, decorator_map, output_indent
            )
            if transformed_decorator_call:
                # This is a decorated function call - use the transformed version
                code_lines.extend(transformed_decorator_call.split('\n'))
                i += 1
                continue
        
        # Check if this is an assignment
        is_assignment = '=' in line and not line.strip().startswith('if') and not line.strip().startswith('elif') and not line.strip().startswith('for')
        if is_assignment:
            # Special case: "rank = env.default_resolve('rank')" - this is redundant, skip it
            if 'rank' in line and 'env.default_resolve' in line and 'rank' in r_variant_names:
                # This is redundant in the new design, skip it
                i += 1
                continue
            
            # Transform assignment to use env.global_assign() or env.assign_global()
            transformed_assignment = transform_assignment_to_global_assign(
                line, r_variant_names, global_names
            )
            
            # Skip if None (redundant assignment)
            if transformed_assignment is None:
                i += 1
                continue
            
            # Preserve indentation
            original_indent = len(line) - len(line.lstrip())
            base_body_indent = len(main_lines[main_body_start]) - len(main_lines[main_body_start].lstrip()) if main_body_start < len(main_lines) else 0
            relative_indent = original_indent - base_body_indent if original_indent >= base_body_indent else 0
            output_indent = "    " + "    " * max(0, relative_indent // 4)
            
            # Check if this is an R-variant assignment - if so, wrap in loop
            # Extract variable name from assignment
            parts = line.split('=', 1)
            if len(parts) > 0:
                left_side = parts[0].strip()
                var_name_match = re.match(r'^(\w+)', left_side)
                if var_name_match:
                    var_name = var_name_match.group(1)
                    if var_name in r_variant_names:
                        # R-variant assignment: wrap in loop
                        # Transform right side to use 'rank' directly and resolve other vars
                        # The transformed_assignment already has env.assign('var_name', ...)
                        # We need to ensure the right side uses 'rank' directly for R-variants
                        code_lines.append(output_indent + "# R-variant assignment: execute for each rank")
                        code_lines.append(output_indent + "for rank in active_ranks:")
                        code_lines.append(output_indent + "    " + transformed_assignment.lstrip())
                        i += 1
                        continue
            
            code_lines.append(output_indent + transformed_assignment.lstrip())
            i += 1
            continue
        
        # Handle env.default_resolve() calls - these need special handling
        if 'env.default_resolve' in line:
            # Transform default_resolve to resolve/global_resolve
            # Pattern: env.default_resolve('var_name') or env.default_resolve("var_name")
            default_resolve_pattern = r'env\.default_resolve\(["\']([^"\']+)["\']\)'
            
            def replace_default_resolve(match):
                var_name = match.group(1)
                if var_name in r_variant_names:
                    # R-variant: use resolve(rank, name) - but rank needs to be in a loop
                    return f'env.resolve(rank, "{var_name}")'
                elif var_name in global_names:
                    return f'env.global_resolve("{var_name}")'
                else:
                    # Unknown - assume R-variant
                    return f'env.resolve(rank, "{var_name}")'
            
            transformed_line = re.sub(default_resolve_pattern, replace_default_resolve, line)
            
            # If this line assigns an R-variant, wrap it in a loop
            if '=' in transformed_line:
                # Check if assigning to an R-variant
                parts = transformed_line.split('=')
                if len(parts) > 1:
                    left_var = parts[0].strip()
                    # Check if left side is an R-variant name
                    if any(r_var in left_var for r_var in r_variant_names):
                        # This is an R-variant assignment - wrap in loop
                        statement = transformed_line.lstrip()
                        code_lines.append("    # R-variant assignment: execute for each rank")
                        code_lines.append("    for rank in active_ranks:")
                        code_lines.append("        " + statement)
                        i += 1
                        continue
            
            # Otherwise, transform other variable accesses
            transformed_line = transform_line_to_use_resolve(
                transformed_line, r_variant_names, global_names, rank_var="rank"
            )
        else:
            # No default_resolve - transform other variable accesses
            # Also transform direct attribute access like env.world_size
            if 'env.world_size' in line:
                transformed_line = line.replace('env.world_size', "env.global_resolve('world_size')")
            else:
                # Remove mark_cond if present
                transformed_line = line
                if 'mark_cond(' in transformed_line:
                    # Pattern: mark_cond(condition, branch_id)
                    pattern = r'mark_cond\s*\((.*?),\s*\d+\)'
                    match = re.search(pattern, transformed_line)
                    if match:
                        condition = match.group(1)
                        transformed_line = transformed_line.replace(match.group(0), condition)
                
                transformed_line = transform_line_to_use_resolve(
                    transformed_line, r_variant_names, global_names, rank_var="rank"
                )
        
        # Preserve original indentation for the line
        original_indent = len(line) - len(line.lstrip())
        
        # Find base indent of main function body (first non-empty line)
        if i == main_body_start:
            # First time - calculate base indent
            for k in range(main_body_start, len(main_lines)):
                if main_lines[k].strip():
                    base_body_indent = len(main_lines[k]) - len(main_lines[k].lstrip())
                    break
            else:
                base_body_indent = 0
        
        # Calculate relative indent from base
        relative_indent = original_indent - base_body_indent if original_indent >= base_body_indent else 0
        # Output indent: base (4 spaces for main function) + relative indent
        output_indent = "    " + "    " * max(0, relative_indent // 4)
        
        # Check if line has R-variants (and is not already in a loop context)
        has_r_variant = detect_r_variant_in_line(transformed_line, r_variant_names)
        is_assignment = '=' in transformed_line and not transformed_line.strip().startswith('if') and not transformed_line.strip().startswith('for')
        is_control = (transformed_line.strip().startswith('if ') or 
                     transformed_line.strip().startswith('elif ') or
                     transformed_line.strip().startswith('else:') or
                     transformed_line.strip().startswith('for ') or
                     transformed_line.strip().startswith('while '))
        
        # Skip empty lines and comments
        if not transformed_line.strip() or transformed_line.strip().startswith('#'):
            code_lines.append(output_indent + transformed_line.lstrip())
        elif has_r_variant and 'for rank in' not in transformed_line and not is_assignment and not is_control:
            # R-variant line (not assignment, not control): wrap in loop
            statement = transformed_line.lstrip()
            code_lines.append(output_indent + "# R-variant line: execute for each rank")
            code_lines.append(output_indent + "for rank in active_ranks:")
            code_lines.append(output_indent + "    " + statement)
        else:
            # Non-R-variant line, control structure, or assignment: keep as is with preserved indentation
            code_lines.append(output_indent + transformed_line.lstrip())
        
        i += 1
    
    code_lines.append("")
    
    # Extract __main__ block from source code to preserve group initialization
    main_block_code = None
    try:
        lines = source_code.splitlines()
        in_main_block = False
        main_block_start = None
        for i, line in enumerate(lines):
            if 'if __name__' in line and '__main__' in line:
                in_main_block = True
                main_block_start = i
                break
        
        if main_block_start is not None:
            # Extract the __main__ block
            main_block_lines = []
            indent_level = None
            for i in range(main_block_start, len(lines)):
                line = lines[i]
                if i == main_block_start:
                    main_block_lines.append(line)
                    # Determine indent level
                    indent_level = len(line) - len(line.lstrip())
                elif line.strip() and not line.strip().startswith('#'):
                    # Check if we've left the __main__ block
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= indent_level and not line.strip().startswith('if'):
                        break
                    main_block_lines.append(line)
                else:
                    main_block_lines.append(line)
            
            main_block_code = '\n'.join(main_block_lines)
    except Exception:
        pass
    
    # Generate __main__ block
    if main_block_code:
        # Use extracted __main__ block, but ensure world_size matches
        # Replace world_size assignment if it exists
        main_block_lines = main_block_code.splitlines()
        for i, line in enumerate(main_block_lines):
            if 'world_size' in line and '=' in line and not 'env.' in line:
                # Replace with our world_size
                main_block_lines[i] = f"    world_size = {world_size}  # From original script"
                break
        code_lines.extend(main_block_lines)
    else:
        # Fallback: generate default __main__ block
        code_lines.extend([
            "",
            "",
            "if __name__ == '__main__':",
            f"    world_size = {world_size}  # From original script",
            "    env = SimEnv(world_size=world_size)",
            "    # Initialize R-variants",
            "    for rank in range(world_size):",
            "        env.assign('rank', rank, R(rank))",
            "    env.assign_global('world_size', world_size)",
            "    ",
            "    # Execute main function",
            "    main(env)",
        ])
    
    generated_code = "\n".join(code_lines)
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generated_code, encoding="utf-8")
    
    return generated_code
