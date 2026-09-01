#!/usr/bin/env python3
"""Run the five retained E1 anchors on a real 16-node/128-GPU cluster."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKLOAD_ROOT = ROOT / "script" / "e1" / "workload"
DEFAULT_RESULT_ROOT = ROOT / "result" / "e1" / "real"
DEFAULT_TEMPLATE_LEDGER = ROOT / "result" / "e1" / "trajectory.csv"
ROUND_NAMES = (
    "round_00_baseline",
    "round_01_packed_variable_dispatch",
    "round_02_top1_confident_fallback",
    "round_03_sdpa_attention_recovery",
    "round_04_switch_top1_routing",
)
METRICS = {
    "time": "step_time_s",
    "a2a": "estimated_a2a_bytes",
    "drop": "tokens_dropped",
    "reroute": "tokens_rerouted",
}


def env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-root", type=Path, default=DEFAULT_WORKLOAD_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--template-ledger", type=Path, default=DEFAULT_TEMPLATE_LEDGER)
    parser.add_argument("--nnodes", type=int, default=env_int("FLEXMAYA_NNODES"))
    parser.add_argument("--node-rank", type=int, default=env_int("FLEXMAYA_NODE_RANK"))
    parser.add_argument("--master-addr", default=os.environ.get("FLEXMAYA_MASTER_ADDR"))
    parser.add_argument("--master-port", type=int, default=env_int("FLEXMAYA_MASTER_PORT"))
    parser.add_argument("--dry-run", action="store_true", help="Print the five torchrun commands without executing them.")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def workload_path(workload_root: Path, round_name: str) -> Path:
    return workload_root / round_name / "tests" / "workloads" / "historical_sparse_moe"


def validate_launch(args: argparse.Namespace) -> None:
    if args.nnodes != 16:
        raise ValueError("E1 real mode requires FLEXMAYA_NNODES=16")
    if args.node_rank is None or not 0 <= args.node_rank < args.nnodes:
        raise ValueError("E1 real mode requires FLEXMAYA_NODE_RANK in [0, 15]")
    if not args.master_addr:
        raise ValueError("E1 real mode requires FLEXMAYA_MASTER_ADDR")
    if args.master_port is None or not 1 <= args.master_port <= 65531:
        raise ValueError("E1 real mode requires FLEXMAYA_MASTER_PORT in [1, 65531]")
    for round_name in ROUND_NAMES:
        root = workload_path(args.workload_root, round_name)
        if not (root.parents[1] / "__init__.py").is_file():
            raise FileNotFoundError(f"missing E1 anchor package marker: {root.parents[1] / '__init__.py'}")
        for relative in ("train.py", "config.py", "model.py", "configs/gshard_style_fastloop.json"):
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"missing E1 anchor file: {path}")
        config = json.loads((root / "configs" / "gshard_style_fastloop.json").read_text(encoding="utf-8"))
        if config["runtime"]["nodes"] != 16 or config["runtime"]["gpus_per_node"] != 8:
            raise ValueError(f"E1 anchor must describe a 16-node/eight-GPU launch: {round_name}")
        if config["runtime"]["expert_parallel_size"] != 128 or config["data"]["seed"] != 1234:
            raise ValueError(f"E1 anchor execution contract differs: {round_name}")


def validate_gpus() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("E1 real mode requires the CUDA PyTorch environment from script/setup") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError(f"E1 real mode requires eight visible GPUs per node, found {torch.cuda.device_count()}")


def command_for_round(args: argparse.Namespace, round_id: int, round_name: str) -> tuple[list[str], dict[str, str], Path]:
    root = workload_path(args.workload_root, round_name)
    output = args.result_root / round_name / "benchmark_result.json"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=16",
        "--nproc-per-node=8",
        f"--node-rank={args.node_rank}",
        f"--master-addr={args.master_addr}",
        f"--master-port={args.master_port + round_id}",
        str(root / "train.py"),
        "--config",
        str(root / "configs" / "gshard_style_fastloop.json"),
        "--train-iters",
        "2",
        "--benchmark-warmup-steps",
        "1",
        "--benchmark-measure-steps",
        "1",
        "--output-json",
        str(output),
    ]
    environment = os.environ.copy()
    anchor_root = root.parents[2]
    environment["PYTHONPATH"] = f"{anchor_root}{os.pathsep}{environment['PYTHONPATH']}" if environment.get("PYTHONPATH") else str(anchor_root)
    return command, environment, output


def load_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid E1 benchmark result: {path}")
    summary = payload.get("summary", payload)
    if not isinstance(summary, dict):
        raise ValueError(f"invalid E1 benchmark result: {path}")
    if summary.get("status") != "complete":
        raise ValueError(f"incomplete E1 benchmark result: {path}")
    if int(summary.get("world_size", 0)) != 128 or int(summary.get("rank", -1)) != 0:
        raise ValueError(f"E1 benchmark did not run as global rank 0 of 128: {path}")
    if int(summary.get("benchmark_samples", 0)) != 1:
        raise ValueError(f"E1 benchmark must contain one measured step: {path}")
    for field in METRICS.values():
        value = float(summary[field])
        if not math.isfinite(value) or value < 0.0 or (field in {"step_time_s", "estimated_a2a_bytes"} and value == 0.0):
            raise ValueError(f"invalid E1 {field}: {path}")
    return summary


def format_metric(value: object) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def build_ledger(template_path: Path, result_root: Path) -> list[dict[str, str]]:
    with template_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames or ())
    if len(rows) != len(ROUND_NAMES):
        raise ValueError(f"E1 template ledger requires five rows, found {len(rows)}")

    for row, round_name in zip(rows, ROUND_NAMES, strict=True):
        result_path = result_root / round_name / "benchmark_result.json"
        summary = load_summary(result_path)
        for field in METRICS.values():
            row[field] = format_metric(summary[field])
        row["benchmark_samples"] = str(int(summary["benchmark_samples"]))
        row["world_size"] = str(int(summary["world_size"]))
        row["result_source_file"] = f"{round_name}/benchmark_result.json"
        row["result_source_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
        row["log_source_sha256"] = ""

    baseline = rows[0]
    for row in rows:
        for prefix, field in METRICS.items():
            base = float(baseline[field])
            if base <= 0.0:
                raise ValueError(f"E1 real baseline {field} must be positive")
            reduction = (base - float(row[field])) / base
            row[f"{prefix}_reduction_vs_baseline"] = str(reduction)
            row[f"{prefix}_normalized_improvement"] = str(1.0 + reduction)

    result_root.mkdir(parents=True, exist_ok=True)
    output = result_root / "trajectory.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        result_root = Path(directory)
        values = (
            (10.0, 1000, 100, 50),
            (8.0, 600, 90, 40),
            (11.0, 610, 80, 30),
            (7.0, 500, 70, 20),
            (5.0, 250, 50, 0),
        )
        for round_name, (step_time, a2a, dropped, rerouted) in zip(ROUND_NAMES, values, strict=True):
            path = result_root / round_name / "benchmark_result.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "world_size": 128,
                        "rank": 0,
                        "benchmark_samples": 1,
                        "step_time_s": step_time,
                        "estimated_a2a_bytes": a2a,
                        "tokens_dropped": dropped,
                        "tokens_rerouted": rerouted,
                    }
                ),
                encoding="utf-8",
            )
        rows = build_ledger(DEFAULT_TEMPLATE_LEDGER, result_root)
        from derive_results import derive, validate

        validate(rows, require_historical_outcomes=False)
        derive(rows, result_root)
        assert (result_root / "figure1b.csv").is_file()
        assert (result_root / "figure1c.csv").is_file()
        assert float(rows[-1]["time_normalized_improvement"]) == 1.5
        assert float(rows[-1]["a2a_normalized_improvement"]) == 1.75
        assert rows[-1]["reroute_normalized_improvement"] == "2.0"


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        print("E1 real runner self-test: PASS")
        return 0

    validate_launch(args)
    if not args.dry_run:
        validate_gpus()
    for round_id, round_name in enumerate(ROUND_NAMES):
        command, environment, output = command_for_round(args, round_id, round_name)
        if args.dry_run:
            print(shlex.join(command))
            continue
        if args.node_rank == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.unlink(missing_ok=True)
        print(f"E1 real: running {round_name}", flush=True)
        subprocess.run(command, env=environment, check=True)
        if args.node_rank == 0 and not output.is_file():
            raise FileNotFoundError(f"E1 rank 0 did not write {output}")

    if not args.dry_run and args.node_rank == 0:
        build_ledger(args.template_ledger, args.result_root)
        print(f"wrote {args.result_root / 'trajectory.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
