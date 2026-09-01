#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import flexmaya_ras as fm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FlexMaya RAS GPT V1 validation.")
    parser.add_argument("--total-gpus", type=int, default=16)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=8)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=768)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--micro-batches", type=int, default=64)
    parser.add_argument("--schedule", choices=["1f1b", "gpipe"], default="1f1b")
    parser.add_argument("--pipeline-p2p-mode", choices=["blocking", "async", "batch"], default="blocking")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--arena-capacity", type=int, default=25_000_000)
    parser.add_argument("--synthetic", action="store_true", help="Use a local synthetic GPT-shaped trace.")
    parser.add_argument("--real-fakecuda", action="store_true", help="Launch the existing fake-CUDA workload through frun/proot.")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def _record_kernel(rows: list[object], rank: int, stream: int, ts: int, partition: str, count: int) -> None:
    rows.append(
        fm.make_event(
            "cudaLaunchKernel",
            "kernel_launch",
            rank=rank,
            stream=stream,
            timestamp_ns=ts,
            bytes=0,
            count=count,
            code_partition=partition,
        )
    )


def synthetic_gpt_events(args: argparse.Namespace) -> list[object]:
    rows: list[object] = []
    ts = 100
    for micro_batch in range(args.micro_batches):
        for layer in range(args.num_layers):
            stage = layer % max(args.pp, 1)
            partition = f"layer_{stage:02d}"
            for rank in fm.megatron_pp_stage_active_ranks(args.total_gpus, args.tp, args.pp, stage):
                _record_kernel(rows, rank, 0, ts, partition, args.hidden_size * args.seq_len)
                ts += 10
                _record_kernel(rows, rank, 3, ts, partition, args.hidden_size * args.hidden_size)
                ts += 10
            for group_idx, members in enumerate(
                fm.megatron_tp_groups_for_stage(args.total_gpus, args.tp, args.pp, stage)
            ):
                group = f"tp_allreduce:stage={stage}:layer={layer}:mb={micro_batch}:group={group_idx}"
                for rank in members:
                    rows.append(
                        fm.make_event(
                            "ncclAllReduce",
                            "nccl_collective",
                            rank=rank,
                            stream=3,
                            timestamp_ns=ts,
                            bytes=args.hidden_size * args.seq_len * 2,
                            count=args.hidden_size * args.seq_len,
                            collective_group=group,
                            code_partition=partition,
                        )
                    )
                    ts += 5
    return rows


def run_real_fakecuda(args: argparse.Namespace, arena: object, root: Path, code_partition: str) -> None:
    maya_root = root.parent / "external" / "maya_lite_native_20260428"
    frun = maya_root / "fake-cuda" / "frun"
    proot = root.parent / "external" / "maya-native-source-package-20260514" / "fake-cuda" / "proot"
    workload = maya_root / "tests" / "workloads" / "fake_cuda" / "maya_fig13_megatron.py"
    ext_suffix = next((root / "src" / "flexmaya_ras").glob("_flexmaya_ras*.so"))
    if not frun.exists() or not workload.exists():
        raise FileNotFoundError("fake-CUDA GPT workload is not available; use --synthetic for local validation")
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": f"{root / 'src'}:{maya_root / 'python'}:{env.get('PYTHONPATH', '')}",
            "FAKECUDA_PROOT_BIN": str(proot),
            "FAKECUDA_TARGET_ENV_ROOT": str(Path(sys.executable).resolve().parent.parent),
            "FAKECUDA_FRUN_QUIET": "1",
            "FAKECUDA_SKIP_LDCONFIG": "1",
            "PLAIN_MAYA_TRACE": "1",
            "PLAIN_MAYA_SHM_NAME": arena.name(),
            "PLAIN_MAYA_HOOK_LIBRARY": str(ext_suffix),
            "PLAIN_MAYA_DISABLE_JSONL": "1",
            "FAKECUDA_DISABLE_JSONL": "1",
            "FLEXMAYA_CODE_PARTITION": code_partition,
        }
    )
    master_port = random.randint(42000, 62000)
    command = [
        str(frun),
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={args.total_gpus}",
        f"--master-port={master_port}",
        str(workload),
        "--steps",
        str(args.steps),
        "--global-batch-size",
        str(args.global_batch_size),
        "--seq-len",
        str(args.seq_len),
        "--hidden-size",
        str(args.hidden_size),
        "--num-layers",
        str(args.num_layers),
        "--num-heads",
        str(args.num_heads),
        "--vocab-size",
        str(args.vocab_size),
        "--micro-batches",
        str(args.micro_batches),
        "--schedule",
        args.schedule,
        "--pipeline-p2p-mode",
        args.pipeline_p2p_mode,
        "--dtype",
        args.dtype,
        "--tp",
        str(args.tp),
        "--pp",
        str(args.pp),
        "--dp",
        str(args.dp),
    ]
    subprocess.run(command, cwd=root.parent, env=env, check=True)


def main() -> int:
    args = parse_args()
    if not args.synthetic and not args.real_fakecuda:
        args.synthetic = True
    root = Path(__file__).resolve().parents[1]
    run_name = f"flexmaya_ras_gpt{args.total_gpus}_seq{args.seq_len}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = args.out_dir or (root.parent / "output" / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.real_fakecuda:
        code_partitions = (
            fm.CodePartitionSpec(
                partition_id="gpt_runtime",
                path=__file__,
                active_ranks=tuple(range(args.total_gpus)),
            ),
        )
    else:
        code_partitions = tuple(
            fm.CodePartitionSpec(
                partition_id=f"layer_{stage:02d}",
                path=__file__,
                active_ranks=fm.megatron_pp_stage_active_ranks(args.total_gpus, args.tp, args.pp, stage),
            )
            for stage in range(args.pp)
        )

    spec = fm.FlexMayaWorkloadSpec(
        workload_id="gpt_fig13_v1",
        world_size=args.total_gpus,
        tp=args.tp,
        pp=args.pp,
        dp=args.dp,
        code_partitions=code_partitions,
        rank_group_policy="active_lane_set",
        notes=("GPT Figure-13-style V1 validation",),
    )

    capture_started = time.perf_counter()
    if args.synthetic:
        raw_events = synthetic_gpt_events(args)
    else:
        arena = fm.SharedEventArena.create("flexmaya-gpt-" + uuid.uuid4().hex, args.arena_capacity, True)
        run_real_fakecuda(args, arena, root, code_partitions[0].partition_id)
        raw_events = arena.events()
    capture_s = time.perf_counter() - capture_started

    anchor = fm.init_anchor(spec, raw_events)
    report = {
        "run_name": run_name,
        "route": "flexmaya_ras_cpp_hook_memory" if args.real_fakecuda else "flexmaya_ras_synthetic",
        "capture_seconds": capture_s,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "anchor": anchor.summary,
        "jsonl_files": [str(path) for path in out_dir.glob("rank_*.jsonl")],
    }
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(result_path), **report}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
