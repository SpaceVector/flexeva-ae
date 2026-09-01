"""
Capture a CPU-side fake-cuda dry-run into Maya-lite rank_*.jsonl traces.

This approximates Maya's trace-generation / realization phase without using
real GPUs. It is intended as a single-core, preliminary-stage phase-1 path,
not a full reproduction of Maya's multi-core worker scheduling.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Iterable


_REGISTER_PREFIXES = ("__cudaRegister", "__cudaUnregister")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_torchrun() -> Path:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / "bin" / "torchrun"
        if candidate.exists():
            return candidate
    executable = Path(sys.executable).resolve()
    candidate = executable.parent / "torchrun"
    if candidate.exists():
        return candidate
    return Path.home() / "miniconda3" / "envs" / "fakecuda-test" / "bin" / "torchrun"


def _default_python() -> Path:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / "bin" / "python"
        if candidate.exists():
            return candidate
    return Path(sys.executable).resolve()


def _count_interesting_events(path: Path) -> int:
    interesting = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            api = str(payload.get("api", ""))
            if api.startswith(_REGISTER_PREFIXES):
                continue
            interesting += 1
    return interesting


def select_worker_trace_files(trace_files: Iterable[Path], expected_ranks: int) -> list[Path]:
    """Keep the most worker-like traces and drop launcher/control traces."""
    files = list(trace_files)
    if expected_ranks <= 0 or len(files) <= expected_ranks:
        return sorted(files)
    ranked = sorted(
        files,
        key=lambda path: (_count_interesting_events(path), path.stat().st_size),
        reverse=True,
    )
    selected = sorted(ranked[:expected_ranks])
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture fake-cuda dry-run traces into Maya-lite format")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nproc", type=int, default=1, help="Number of worker ranks to launch")
    parser.add_argument("--master-port", type=int, default=29631)
    parser.add_argument("--frun", type=Path, default=_repo_root() / "fake-cuda" / "frun")
    parser.add_argument("--torchrun", type=Path, default=_default_torchrun())
    parser.add_argument("--python-bin", type=Path, default=_default_python())
    parser.add_argument(
        "--pin-core",
        type=int,
        default=0,
        help="Pin this phase-1 capture process to a single CPU core. Use -1 to disable.",
    )
    parser.add_argument(
        "--capture-lock",
        type=Path,
        default=Path.home() / ".cache" / "maya_fake_capture.lock",
        help="Lock file used to serialize access to shared /tmp fake-cuda trace outputs.",
    )
    parser.add_argument("script", type=Path, help="Python workload script to execute")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser


@contextmanager
def _capture_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.pin_core >= 0 and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {args.pin_core})

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frun = args.frun.resolve()
    script = args.script.resolve()
    torchrun = args.torchrun.resolve()
    python_bin = args.python_bin.resolve()
    script_args = args.script_args[1:] if args.script_args and args.script_args[0] == "--" else args.script_args

    with tempfile.TemporaryDirectory(prefix="maya-fake-capture-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        cmd: list[str] = [str(frun)]
        if args.nproc > 1:
            cmd += [str(torchrun), "--master-port", str(args.master_port), f"--nproc_per_node={args.nproc}", str(script)]
        else:
            cmd += [str(python_bin), str(script)]
        cmd += script_args

        env = os.environ.copy()
        env["FAKECUDA_TRACE"] = "1"
        env.pop("FAKECUDA_TRACE_PATH", None)

        with _capture_lock(args.capture_lock):
            cleanup_targets = list(Path("/tmp").glob("fakecuda_trace_*.jsonl"))
            for path in cleanup_targets:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

            start = time.perf_counter()
            proc = subprocess.run(
                cmd,
                cwd=str(_repo_root()),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            elapsed = time.perf_counter() - start

            trace_files = sorted(Path("/tmp").glob("fakecuda_trace_*.jsonl"))
            worker_files = select_worker_trace_files(trace_files, expected_ranks=args.nproc)
            for idx, src in enumerate(worker_files):
                shutil.move(str(src), output_dir / f"rank_{idx}.jsonl")
            for src in trace_files:
                if src.exists():
                    src.unlink()

        (output_dir / "capture_stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (output_dir / "capture_stderr.txt").write_text(proc.stderr, encoding="utf-8")
        (output_dir / "capture_elapsed_seconds.txt").write_text(f"{elapsed:.6f}\n", encoding="utf-8")
        manifest = {
            "mode": "fake_cuda_phase1",
            "nproc": args.nproc,
            "selected_worker_traces": len(worker_files),
            "capture_elapsed_seconds": elapsed,
            "command": cmd,
            "returncode": proc.returncode,
        }
        (output_dir / "capture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if proc.returncode != 0:
            print(proc.stdout, end="", file=sys.stdout)
            print(proc.stderr, end="", file=sys.stderr)
            return proc.returncode
        if not worker_files:
            print("no fake-cuda trace files captured", file=sys.stderr)
            return 2
        print(json.dumps(manifest, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
