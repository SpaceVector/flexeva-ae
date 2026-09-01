"""
Command-line interface for astparser.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from astparser.codegen import DEFAULT_OUTPUT_PATH, generate_code
from astparser.parser import parse_entry_files


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="AST parser for detecting branch points in Python training scripts"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Entry Python files to analyze",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed information about each branch point",
    )
    parser.add_argument(
        "--generate",
        "-g",
        action="store_true",
        help="Generate instrumented code and write to output file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path for generated code (default: {DEFAULT_OUTPUT_PATH})",
    )

    args = parser.parse_args()

    try:
        branches, decorators = parse_entry_files(args.files)

        print(f"Found {len(branches)} branch point(s) and {len(decorators)} decorator(s):\n")

        if branches:
            print("Branch Points:")
            for i, branch in enumerate(branches, 1):
                nested_info = f" (nested under line {branch.parent_lineno})" if branch.is_nested else ""
                print(f"  {i}. {branch.branch_type.value.upper()} at line {branch.lineno}{nested_info}")
                if args.verbose:
                    print(f"     Condition: {branch.condition}")
                    print(f"     Column: {branch.col_offset}")
                    if branch.nest_path:
                        print(f"     Nest path: {branch.nest_path}")
            print()

        if decorators:
            print("Decorators:")
            for i, decorator in enumerate(decorators, 1):
                print(f"  {i}. @{decorator.decorator_type} at line {decorator.lineno} (target: {decorator.target_name})")
                if args.verbose:
                    if decorator.active_ranks is not None:
                        print(f"     active_ranks: {decorator.active_ranks}")
                    if decorator.active_group is not None:
                        print(f"     active_group: {decorator.active_group}")

        if len(branches) == 0 and len(decorators) == 0:
            print("No branch points or decorators found.")

        # Generate code if requested
        if args.generate:
            entry_file_paths = [Path(f) for f in args.files]
            generate_code(entry_file_paths, branches, decorators, args.output)
            print(f"\nGenerated instrumented code to: {args.output}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except SyntaxError as e:
        print(f"Syntax error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

