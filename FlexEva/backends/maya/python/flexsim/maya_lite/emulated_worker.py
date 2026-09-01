#!/usr/bin/env python3
"""
Run one workload rank under the Maya-lite emulated distributed environment.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from flexsim.maya_lite.emulated_dist import install_emulated_distributed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute one blind emulated Maya-lite worker")
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--logical-rank", type=int, required=True)
    parser.add_argument("--logical-world-size", type=int, required=True)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    os.environ["RANK"] = str(args.logical_rank)
    os.environ["WORLD_SIZE"] = str(args.logical_world_size)
    os.environ["LOCAL_RANK"] = str(args.local_rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29631")

    install_emulated_distributed(
        logical_rank=args.logical_rank,
        logical_world_size=args.logical_world_size,
    )

    script_argv = [str(args.script)]
    if args.script_args and args.script_args[0] == "--":
        script_argv.extend(args.script_args[1:])
    else:
        script_argv.extend(args.script_args)

    old_argv = sys.argv[:]
    try:
        sys.argv = script_argv
        runpy.run_path(str(args.script.resolve()), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else 1
    finally:
        sys.argv = old_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
