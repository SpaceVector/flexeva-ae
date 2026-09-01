#!/usr/bin/env python3
"""Keep a node-local SharedEventArena alive around a torchrun worker."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import flexmaya_ras as fm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arena-name", required=True)
    parser.add_argument("--arena-capacity", type=int, required=True)
    parser.add_argument("--events-output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SystemExit("missing worker command")

    args.events_output.parent.mkdir(parents=True, exist_ok=True)
    arena = fm.SharedEventArena.create(args.arena_name, args.arena_capacity, True)
    env = os.environ.copy()
    env["FLEXMAYA_SHM_NAME"] = arena.name()
    env.setdefault("FLEXMAYA_SHARED_ARENA_ONLY", "1")
    completed = subprocess.run(command, env=env, check=False)
    arena.write_binary(str(args.events_output))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
