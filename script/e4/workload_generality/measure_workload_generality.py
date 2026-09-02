#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import flexmaya_ras as fm


PARTITIONS = ("compute", "matmul", "communication", "memory", "sync", "runtime")


@dataclass(frozen=True)
class Table4Case:
    name: str
    workload: str
    model: str
    optimization: str
    anchor_extra_args: tuple[str, ...]
    candidate_extra_args: tuple[str, ...]
    changed_partitions: tuple[str, ...]


CASES = (
    Table4Case(
        name="resnet_compile",
        workload="ResNet",
        model="resnet",
        optimization=r"\texttt{torch.compile}",
        anchor_extra_args=("--parallel", "ddp", "--cudnn-mode", "disabled"),
        candidate_extra_args=(
            "--parallel",
            "ddp",
            "--cudnn-mode",
            "disabled",
            "--compile",
            "--compile-backend",
            "eager",
        ),
        changed_partitions=("compute", "matmul", "memory"),
    ),
    Table4Case(
        name="bert_fsdp",
        workload="BERT",
        model="bert",
        optimization="FSDP sharding",
        anchor_extra_args=("--parallel", "ddp"),
        candidate_extra_args=("--parallel", "fsdp"),
        changed_partitions=("communication", "sync"),
    ),
    Table4Case(
        name="vit_checkpoint",
        workload="ViT",
        model="vit",
        optimization="activation checkpointing",
        anchor_extra_args=("--parallel", "ddp", "--cudnn-mode", "disabled"),
        candidate_extra_args=("--parallel", "ddp", "--cudnn-mode", "disabled", "--activation-checkpoint"),
        changed_partitions=("compute", "matmul", "memory"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Table 6 workload-generality suite and measure FlexEva/Maya RAS reuse metrics."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--maya-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--proot", type=Path, required=True)
    parser.add_argument(
        "--trace-root",
        type=Path,
        help="Store fresh CASE/{anchor,candidate}/traces outside the result directory.",
    )
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--local-device-count", type=int, default=8)
    parser.add_argument("--preset", choices=["smoke", "table4-lite", "table4-shape"], default="smoke")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--case", action="append", choices=[case.name for case in CASES], default=None)
    parser.add_argument("--reuse-existing-traces", action="store_true")
    parser.add_argument("--keep-raw-traces", action="store_true")
    parser.add_argument("--timing-repeats", type=int, default=11)
    parser.add_argument("--timing-warmups", type=int, default=2)
    args = parser.parse_args()
    if args.world_size <= 0 or args.local_device_count <= 0:
        parser.error("world size and local device count must be positive")
    if args.timing_repeats < 4 or args.timing_warmups < 0:
        parser.error("timing requires at least four repeats and non-negative warmups")
    return args


def timed(fn: Callable[[], object]) -> tuple[float, object]:
    start = time.perf_counter()
    value = fn()
    return time.perf_counter() - start, value


def timing_stats(values: list[float]) -> dict[str, float]:
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return {
        "median_s": statistics.median(values),
        "q1_s": q1,
        "q3_s": q3,
        "iqr_s": q3 - q1,
    }


def dtype_bytes(dtype: str) -> int:
    if dtype == "bf16":
        return 2
    if dtype == "fp32":
        return 4
    return 4


def rank_groups(world_size: int) -> dict[int, list[int]]:
    return {0: list(range(world_size))}


def table4_spec(
    *,
    workload_id: str,
    world_size: int,
    source_paths: dict[str, Path],
) -> fm.FlexMayaWorkloadSpec:
    return fm.FlexMayaWorkloadSpec(
        workload_id=workload_id,
        world_size=world_size,
        tp=1,
        pp=1,
        dp=world_size,
        code_partitions=tuple(
            fm.CodePartitionSpec(
                partition_id=partition,
                path=str(source_paths[partition]),
                active_ranks=tuple(range(world_size)),
            )
            for partition in PARTITIONS
        ),
        rank_group_policy="active_lane_set",
        notes=("table4 workload generality",),
    )


def write_source_partition_files(case_dir: Path, changed_partitions: tuple[str, ...]) -> tuple[dict[str, Path], dict[str, Path]]:
    source_dir = case_dir / "source_partitions"
    source_dir.mkdir(parents=True, exist_ok=True)
    changed = set(changed_partitions)
    anchor_paths: dict[str, Path] = {}
    candidate_paths: dict[str, Path] = {}
    for partition in PARTITIONS:
        common_path = source_dir / f"{partition}_anchor.py"
        common_path.write_text(
            f"# {partition} partition for Table 4 anchor\n"
            f"PARTITION = {partition!r}\n"
            "VERSION = 'anchor'\n",
            encoding="utf-8",
        )
        anchor_paths[partition] = common_path
        if partition in changed:
            candidate_path = source_dir / f"{partition}_candidate.py"
            candidate_path.write_text(
                f"# {partition} partition for Table 4 optimized candidate\n"
                f"PARTITION = {partition!r}\n"
                "VERSION = 'candidate'\n",
                encoding="utf-8",
            )
            candidate_paths[partition] = candidate_path
        else:
            candidate_paths[partition] = common_path
    return anchor_paths, candidate_paths


def run_table4_variant(
    args: argparse.Namespace,
    case: Table4Case,
    *,
    variant: str,
    extra_args: tuple[str, ...],
    variant_dir: Path,
) -> dict[str, object]:
    variant_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = (
        args.trace_root / case.name / variant / "traces"
        if args.trace_root
        else variant_dir / "traces"
    )
    trace_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_existing_traces:
        missing = [
            trace_dir / f"rank_{rank}{suffix}"
            for rank in range(args.world_size)
            for suffix in (".jsonl", "_markers.jsonl")
            if not (trace_dir / f"rank_{rank}{suffix}").exists()
        ]
        if missing:
            raise FileNotFoundError(
                "missing trace files for --reuse-existing-traces: "
                + ", ".join(str(path) for path in missing[:8])
                + (" ..." if len(missing) > 8 else "")
            )
        return {
            "command": [],
            "return_code": 0,
            "elapsed_s": 0.0,
            "stdout": str(variant_dir / "stdout.txt"),
            "stderr": str(variant_dir / "stderr.txt"),
            "trace_dir": str(trace_dir),
            "reused_existing_traces": True,
        }

    wrapper = Path(__file__).resolve().parent / "trace_worker.py"
    workload = Path(__file__).resolve().parents[1] / "workload" / "table4_pytorch" / "models.py"
    frun = args.maya_root / "fake-cuda" / "frun"
    env = os.environ.copy()
    env.pop("FAKECUDA_TRACE", None)
    env.pop("FAKECUDA_TRACE_PATH", None)
    env["FAKECUDA_PROOT_BIN"] = str(args.proot)
    env["FAKECUDA_FRUN_QUIET"] = "1"
    env["FLEXMAYA_TRACE_DIR"] = str(trace_dir)
    env["FLEXMAYA_LOCAL_DEVICE_COUNT"] = str(args.local_device_count)
    env["TABLE4_SCRIPT"] = str(workload)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(__file__).resolve().parents[1] / "workload" / "table4_pytorch"),
            str(args.maya_root / "python"),
            str(args.maya_root / "CppEvent"),
            str(Path(__file__).resolve().parent),
            env.get("PYTHONPATH", ""),
        ]
    )
    command = [
        str(frun),
        str(args.python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={args.world_size}",
        f"--master-port={random.randint(42000, 62000)}",
        str(wrapper),
        "--model",
        case.model,
        "--preset",
        args.preset,
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--dtype",
        args.dtype,
        "--sync-before-step-window",
        "--log-interval",
        "1",
        *extra_args,
    ]
    stdout = variant_dir / "stdout.txt"
    stderr = variant_dir / "stderr.txt"
    start = time.perf_counter()
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        completed = subprocess.run(command, env=env, cwd=variant_dir, stdout=out, stderr=err, text=True)
    return {
        "command": command,
        "return_code": completed.returncode,
        "elapsed_s": time.perf_counter() - start,
        "stdout": str(stdout),
        "stderr": str(stderr),
        "trace_dir": str(trace_dir),
        "variant": variant,
    }


def load_step_window(markers_path: Path) -> tuple[int, int]:
    begins: list[int] = []
    ends: list[int] = []
    with markers_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("label") != "training_step" or int(record.get("step", -1)) != 1:
                continue
            trace_ts = record.get("trace_ts")
            if trace_ts is None:
                continue
            if record.get("kind") == "step_begin":
                begins.append(int(trace_ts))
            elif record.get("kind") == "step_end":
                ends.append(int(trace_ts))
    if not begins or not ends:
        raise RuntimeError(f"missing step markers in {markers_path}")
    return min(begins), max(ends)


def record_count(record: dict[str, object]) -> int:
    api = str(record.get("api", ""))
    if "count" in record:
        return max(int(record.get("count") or 0), 1)
    if api.startswith("cublas"):
        values = [int(record.get(key) or 1) for key in ("m", "n", "k")]
        batch = int(record.get("batch_count") or 1)
        return max(values[0] * values[1] * values[2] * batch, 1)
    grid = int(record.get("grid_x") or 1) * int(record.get("grid_y") or 1) * int(record.get("grid_z") or 1)
    block = int(record.get("block_x") or 1) * int(record.get("block_y") or 1) * int(record.get("block_z") or 1)
    return max(grid * block, 1)


def classify_partition(record: dict[str, object]) -> str:
    api = str(record.get("api", ""))
    api_lower = api.lower()
    kind = str(record.get("type", ""))
    kind_lower = kind.lower()
    if "nccl" in api_lower or kind == "nccl_collective":
        return "communication"
    if api.startswith("cublas") or "gemm" in api_lower or kind == "blas_compute":
        return "matmul"
    if "mem" in api_lower or "memcpy" in api_lower or kind in {"mem_copy", "memcpy"}:
        return "memory"
    if "synchronize" in api_lower or "sync" in api_lower or "wait" in api_lower or "event" in api_lower:
        return "sync"
    if kind == "kernel_launch" or "cudnn" in api_lower:
        return "compute"
    return "runtime"


def raw_event_from_record(
    record: dict[str, object],
    *,
    dtype: str,
    rank: int,
    event_id: int,
) -> object:
    kind = str(record.get("type", ""))
    api = str(record.get("api", ""))
    count = record_count(record)
    bytes_ = int(record.get("bytes") or 0)
    if bytes_ <= 0 and kind == "nccl_collective":
        bytes_ = count * dtype_bytes(dtype)
    collective_group = ""
    if kind == "nccl_collective":
        collective_group = f"{api}:call={int(record.get('call_idx') or 0)}"
    return fm.make_event(
        api,
        kind,
        rank=rank,
        thread_id=int(record.get("tid") or 0),
        stream=int(record.get("stream_id") or 0),
        correlation_id=event_id,
        timestamp_ns=int(record.get("ts") or 0) * 1000,
        duration_hint_us=float(record.get("host_duration_us") or 0.0),
        bytes=bytes_,
        count=count,
        peer_rank=int(record.get("peer") if record.get("peer") is not None else -1),
        collective_group=collective_group,
        code_partition=classify_partition(record),
        blocking=api.endswith("Synchronize"),
    )


def parse_raw_events(args: argparse.Namespace, trace_dir: Path) -> tuple[list[object], dict[str, float]]:
    parse_start = time.perf_counter()
    rows: list[object] = []
    next_id = 1
    for rank in range(args.world_size):
        start_ts, end_ts = load_step_window(trace_dir / f"rank_{rank}_markers.jsonl")
        trace_path = trace_dir / f"rank_{rank}.jsonl"
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                ts = int(record.get("ts") or 0)
                if ts < start_ts or ts > end_ts:
                    continue
                kind = str(record.get("type", ""))
                if kind in {"marker", "other"}:
                    continue
                rows.append(raw_event_from_record(record, dtype=args.dtype, rank=rank, event_id=next_id))
                next_id += 1
    return rows, {"jsonl_parse_s": time.perf_counter() - parse_start}


def build_full(raw_events: list[object]) -> tuple[object, object, dict[str, float]]:
    build_s, trace = timed(lambda: fm.build_trace_ras(raw_events))
    replay_s, replay = timed(lambda: fm.replay_trace_once(trace))
    return trace, replay, {"trace_build_s": build_s, "python_replay_s": replay_s, "total_s": build_s + replay_s}


def build_ras(raw_events: list[object], groups: dict[int, list[int]]) -> tuple[object, object, dict[str, float]]:
    build_s, trace = timed(lambda: fm.build_rank_grouped_trace_ras(raw_events, groups))
    replay_s, replay = timed(lambda: fm.replay_trace_once(trace))
    return trace, replay, {"trace_build_s": build_s, "python_replay_s": replay_s, "total_s": build_s + replay_s}


def event_ids_for_partitions(trace: object, partition_ids: tuple[int, ...]) -> tuple[int, ...]:
    selected = set(partition_ids)
    event_ids: set[int] = set()
    for partition in trace.sync_partitions:
        if int(partition.id) in selected:
            event_ids.update(int(event_id) for event_id in partition.event_ids)
    return tuple(sorted(event_ids))


def selected_refresh(trace: object, partition_ids: tuple[int, ...]) -> tuple[object, object, dict[str, float]]:
    filter_s, selected_trace = timed(lambda: fm.filter_trace_partitions(trace, list(partition_ids)))
    replay_s, replay = timed(lambda: fm.replay_trace_once(selected_trace))
    return selected_trace, replay, {
        "trace_filter_s": filter_s,
        "python_replay_s": replay_s,
        "total_s": filter_s + replay_s,
    }


def measure_core_refresh(
    args: argparse.Namespace,
    anchor: fm.FlexMayaAnchor,
    candidate_spec: fm.FlexMayaWorkloadSpec,
    candidate_trace: object,
) -> dict[str, object]:
    def full_replay() -> tuple[object, dict[str, float]]:
        elapsed_s, replay = timed(lambda: fm.replay_trace_once(candidate_trace))
        return replay, {"total_s": elapsed_s}

    def refresh() -> tuple[object, object, object, dict[str, float]]:
        start = time.perf_counter()
        plan_s, plan = timed(lambda: fm.plan_candidate_refresh(anchor, candidate_spec, candidate_trace))
        selected_trace, replay, phases = selected_refresh(candidate_trace, plan.affected_trace_partitions)
        total_s = time.perf_counter() - start
        return plan, selected_trace, replay, {
            **phases,
            "plan_s": plan_s,
            "unattributed_s": max(total_s - plan_s - phases["total_s"], 0.0),
            "total_s": total_s,
        }

    for _ in range(args.timing_warmups):
        full_replay()
        refresh()

    samples: list[dict[str, object]] = []
    final_full_replay = None
    final_plan = final_selected_trace = final_selected_replay = None
    for repeat in range(1, args.timing_repeats + 1):
        order = ("full_replay", "refresh") if repeat % 2 else ("refresh", "full_replay")
        sample: dict[str, object] = {"repeat": repeat, "order": list(order)}
        for operation in order:
            if operation == "full_replay":
                final_full_replay, sample[operation] = full_replay()
            else:
                final_plan, final_selected_trace, final_selected_replay, sample[operation] = refresh()
        samples.append(sample)

    full_stats = timing_stats([float(row["full_replay"]["total_s"]) for row in samples])
    refresh_stats = timing_stats([float(row["refresh"]["total_s"]) for row in samples])
    return {
        "repeats": args.timing_repeats,
        "warmups": args.timing_warmups,
        "aggregation": "ratio of medians; inclusive-quartile IQR",
        "scope": "same materialized candidate RAS trace: full replay versus plan/filter/selective replay",
        "full_replay": full_stats,
        "refresh": refresh_stats,
        "speedup": full_stats["median_s"] / max(refresh_stats["median_s"], 1.0e-12),
        "samples": samples,
        "final_full_replay": final_full_replay,
        "final_plan": final_plan,
        "final_selected_trace": final_selected_trace,
        "final_selected_replay": final_selected_replay,
    }


def measure_case(args: argparse.Namespace, case: Table4Case) -> dict[str, object]:
    case_dir = args.out_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    anchor_paths, candidate_paths = write_source_partition_files(case_dir, case.changed_partitions)
    anchor_spec = table4_spec(
        workload_id=f"{case.name}_anchor",
        world_size=args.world_size,
        source_paths=anchor_paths,
    )
    candidate_spec = table4_spec(
        workload_id=f"{case.name}_candidate",
        world_size=args.world_size,
        source_paths=candidate_paths,
    )

    anchor_run = run_table4_variant(
        args,
        case,
        variant="anchor",
        extra_args=case.anchor_extra_args,
        variant_dir=case_dir / "anchor",
    )
    if int(anchor_run["return_code"]) != 0:
        return {"case": asdict(case), "anchor_run": anchor_run, "error": "anchor run failed"}

    candidate_run = run_table4_variant(
        args,
        case,
        variant="candidate",
        extra_args=case.candidate_extra_args,
        variant_dir=case_dir / "candidate",
    )
    if int(candidate_run["return_code"]) != 0:
        return {
            "case": asdict(case),
            "anchor_run": anchor_run,
            "candidate_run": candidate_run,
            "error": "candidate run failed",
        }

    anchor_raw, anchor_parse = parse_raw_events(args, Path(anchor_run["trace_dir"]))
    anchor_full_trace, anchor_full_replay, anchor_full_phases = build_full(anchor_raw)
    anchor_ras_trace, anchor_ras_replay, anchor_ras_phases = build_ras(anchor_raw, rank_groups(args.world_size))
    source_hash_s, anchor_source_hashes = timed(lambda: fm.source_hashes(anchor_spec))
    anchor = fm.FlexMayaAnchor(
        spec=anchor_spec,
        source_hashes=anchor_source_hashes,
        trace=anchor_ras_trace,
        feedback=anchor_ras_replay,
        summary={"trace": fm.trace_summary(anchor_ras_trace), "feedback": anchor_ras_replay.to_dict()},
    )

    candidate_raw, candidate_parse = parse_raw_events(args, Path(candidate_run["trace_dir"]))
    candidate_full_trace, candidate_full_replay, candidate_full_phases = build_full(candidate_raw)
    candidate_ras_trace, candidate_ras_replay, candidate_ras_phases = build_ras(
        candidate_raw,
        rank_groups(args.world_size),
    )
    core_timing = measure_core_refresh(args, anchor, candidate_spec, candidate_ras_trace)
    plan = core_timing.pop("final_plan")
    selected_trace = core_timing.pop("final_selected_trace")
    selected_replay = core_timing.pop("final_selected_replay")
    full_replay = core_timing.pop("final_full_replay")
    affected_event_ids = event_ids_for_partitions(candidate_ras_trace, plan.affected_trace_partitions)

    anchor_maya_full_total = anchor_parse["jsonl_parse_s"] + anchor_full_phases["total_s"]
    anchor_init_total = (
        anchor_parse["jsonl_parse_s"] + source_hash_s + anchor_ras_phases["trace_build_s"] + anchor_ras_phases["python_replay_s"]
    )
    candidate_maya_full_total = candidate_parse["jsonl_parse_s"] + candidate_full_phases["total_s"]
    full_replay_median = float(core_timing["full_replay"]["median_s"])
    refresh_median = float(core_timing["refresh"]["median_s"])

    if not args.keep_raw_traces:
        for trace_dir_text in (anchor_run["trace_dir"], candidate_run["trace_dir"]):
            for path in Path(str(trace_dir_text)).glob("rank_*.jsonl"):
                path.unlink(missing_ok=True)

    return {
        "case": asdict(case),
        "preset": args.preset,
        "dtype": args.dtype,
        "world_size": args.world_size,
        "rank_groups": rank_groups(args.world_size),
        "anchor_run": anchor_run,
        "candidate_run": candidate_run,
        "anchor": {
            "raw_events": len(anchor_raw),
            "maya_full": {
                "trace": fm.trace_summary(anchor_full_trace),
                "feedback": anchor_full_replay.to_dict(),
                "phases_s": {**anchor_parse, **anchor_full_phases, "total_s": anchor_maya_full_total},
            },
            "flexeva_init": {
                "source_hash_s": source_hash_s,
                "trace": fm.trace_summary(anchor_ras_trace),
                "feedback": anchor_ras_replay.to_dict(),
                "phases_s": {
                    **anchor_parse,
                    "source_hash_s": source_hash_s,
                    **anchor_ras_phases,
                    "total_s": anchor_init_total,
                },
            },
        },
        "candidate": {
            "raw_events": len(candidate_raw),
            "maya_full": {
                "trace": fm.trace_summary(candidate_full_trace),
                "feedback": candidate_full_replay.to_dict(),
                "phases_s": {**candidate_parse, **candidate_full_phases, "total_s": candidate_maya_full_total},
            },
            "maya_trace_ras": {
                "trace": fm.trace_summary(candidate_ras_trace),
                "feedback": candidate_ras_replay.to_dict(),
                "phases_s": {**candidate_parse, **candidate_ras_phases},
            },
            "full_replay": {
                "feedback": full_replay.to_dict(),
                "timing": core_timing["full_replay"],
            },
            "refresh": {
                "plan": asdict(plan),
                "affected_event_count": len(affected_event_ids),
                "selected_trace": fm.trace_summary(selected_trace),
                "feedback": selected_replay.to_dict(),
                "timing": core_timing["refresh"],
            },
            "core_timing": core_timing,
        },
        "metrics": {
            "init_metric_tinit_over_tb": anchor_init_total / max(anchor_maya_full_total, 1.0e-12),
            "core_refresh_speedup": full_replay_median / max(refresh_median, 1.0e-12),
            "reuse_rate": 1.0 - (len(selected_trace.events) / max(len(candidate_ras_trace.events), 1)),
            "logical_reuse_rate": 1.0
            - (int(selected_trace.logical_event_count) / max(int(candidate_ras_trace.logical_event_count), 1)),
            "candidate_full_replay_median_s": full_replay_median,
            "candidate_refresh_median_s": refresh_median,
            "candidate_maya_full_setup_s_auxiliary": candidate_maya_full_total,
        },
    }


def latex_escape(text: str) -> str:
    return text.replace("&", r"\&")


def write_summary_files(args: argparse.Namespace, results: list[dict[str, object]]) -> None:
    result = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": {
            "metric_version": "table6-core-v2",
            "target": f"{args.world_size}-rank data-parallel target captured through fake-CUDA",
            "capture": "script/e4/workload/table4_pytorch/models.py through fake-CUDA frun",
            "window": "training_step markers",
            "maya_full": "JSONL parse + build_trace_ras + Python replay",
            "init_metric": "FlexEva anchor init over Maya-full for the same anchor workload",
            "core_refresh_metric": "full replay over plan/filter/selective replay from the same materialized candidate RAS trace",
            "aggregation": "ratio of medians over alternating-order repetitions; inclusive-quartile IQR",
            "excluded_from_core_refresh": "capture, JSONL parse, and candidate RAS trace construction on both sides",
            "baseline_attribution": "full RAS replay in our Maya-style backend; not original Maya end-to-end",
            "reuse_rate": "fraction of compact candidate trace events not selected for refresh",
        },
        "results": results,
    }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "workload",
                "optimization",
                "status",
                "world_size",
                "timing_repeats",
                "init_metric_tinit_over_tb",
                "core_refresh_speedup",
                "reuse_rate",
                "logical_reuse_rate",
                "anchor_raw_events",
                "candidate_raw_events",
                "candidate_compact_events",
                "selected_events",
                "full_replay_median_s",
                "full_replay_q1_s",
                "full_replay_q3_s",
                "refresh_median_s",
                "refresh_q1_s",
                "refresh_q3_s",
            ],
        )
        writer.writeheader()
        for row in results:
            case = row["case"]
            metrics = row.get("metrics", {})
            candidate = row.get("candidate", {})
            refresh = candidate.get("refresh", {}) if isinstance(candidate, dict) else {}
            core_timing = candidate.get("core_timing", {}) if isinstance(candidate, dict) else {}
            full_timing = core_timing.get("full_replay", {}) if isinstance(core_timing, dict) else {}
            refresh_timing = core_timing.get("refresh", {}) if isinstance(core_timing, dict) else {}
            writer.writerow(
                {
                    "workload": case["workload"],
                    "optimization": case["optimization"],
                    "status": "ok" if "error" not in row else row["error"],
                    "world_size": row.get("world_size"),
                    "timing_repeats": core_timing.get("repeats") if isinstance(core_timing, dict) else None,
                    "init_metric_tinit_over_tb": metrics.get("init_metric_tinit_over_tb"),
                    "core_refresh_speedup": metrics.get("core_refresh_speedup"),
                    "reuse_rate": metrics.get("reuse_rate"),
                    "logical_reuse_rate": metrics.get("logical_reuse_rate"),
                    "anchor_raw_events": row.get("anchor", {}).get("raw_events") if isinstance(row.get("anchor"), dict) else None,
                    "candidate_raw_events": candidate.get("raw_events") if isinstance(candidate, dict) else None,
                    "candidate_compact_events": candidate.get("maya_trace_ras", {}).get("trace", {}).get("event_count")
                    if isinstance(candidate, dict)
                    else None,
                    "selected_events": refresh.get("selected_trace", {}).get("event_count") if isinstance(refresh, dict) else None,
                    "full_replay_median_s": full_timing.get("median_s"),
                    "full_replay_q1_s": full_timing.get("q1_s"),
                    "full_replay_q3_s": full_timing.get("q3_s"),
                    "refresh_median_s": refresh_timing.get("median_s"),
                    "refresh_q1_s": refresh_timing.get("q1_s"),
                    "refresh_q3_s": refresh_timing.get("q3_s"),
                }
            )

    with (args.out_dir / "timing_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "workload",
                "repeat",
                "order",
                "full_replay_s",
                "refresh_plan_s",
                "refresh_filter_s",
                "refresh_replay_s",
                "refresh_unattributed_s",
                "refresh_total_s",
            ],
        )
        writer.writeheader()
        for row in results:
            if "error" in row:
                continue
            for sample in row["candidate"]["core_timing"]["samples"]:
                refresh = sample["refresh"]
                writer.writerow(
                    {
                        "workload": row["case"]["workload"],
                        "repeat": sample["repeat"],
                        "order": ">".join(sample["order"]),
                        "full_replay_s": sample["full_replay"]["total_s"],
                        "refresh_plan_s": refresh["plan_s"],
                        "refresh_filter_s": refresh["trace_filter_s"],
                        "refresh_replay_s": refresh["python_replay_s"],
                        "refresh_unattributed_s": refresh["unattributed_s"],
                        "refresh_total_s": refresh["total_s"],
                    }
                )

    ok_rows = [row for row in results if "error" not in row]
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        f"  \\caption{{Post-capture core refresh on a {args.world_size}-GPU target.}}",
        r"  \label{tab:gen-workload}",
        r"  \begin{tabularx}{\columnwidth}{@{}>{\raggedright\arraybackslash}p{0.34\columnwidth}X@{}}",
        r"    \toprule",
        r"    Workload & Optimization \\",
        r"    \midrule",
    ]
    for row in ok_rows:
        case = row["case"]
        lines.append(f"    {latex_escape(case['workload'])} & {case['optimization']} \\\\")
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabularx}",
            r"  \vspace{0.5em}",
            r"  \begin{tabularx}{\columnwidth}{@{}>{\raggedright\arraybackslash}p{0.30\columnwidth}ccc@{}}",
            r"    \toprule",
            r"    Workload & Init metric & Core speedup & Reuse rate \\",
            r"    \midrule",
        ]
    )
    for row in ok_rows:
        case = row["case"]
        metrics = row["metrics"]
        lines.append(
            "    "
            f"{latex_escape(case['workload'])} & "
            f"{float(metrics['init_metric_tinit_over_tb']):.3f} & "
            f"{float(metrics['core_refresh_speedup']):.2f}$\\times$ & "
            f"{100.0 * float(metrics['reuse_rate']):.2f}\\% \\\\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabularx}", r"\end{table}", ""])
    (args.out_dir / "table6_workload_table.tex").write_text("\n".join(lines), encoding="utf-8")

    readme = [
        "# Table 6 workload-generality run",
        "",
        f"- preset: `{args.preset}`",
        f"- world size: `{args.world_size}`",
        f"- dtype: `{args.dtype}`",
        f"- timing: median of `{args.timing_repeats}` repetitions after `{args.timing_warmups}` warmups; `summary.csv` also records the IQR.",
        "- paper mapping: Table 6; the workload preset keeps its legacy `table4-lite` name.",
        "- Init metric: `T_init / T_b`, where `T_b` is Maya-full for the anchor workload.",
        "- Core refresh speedup: full replay divided by plan/filter/selective replay from the same materialized candidate RAS trace.",
        "- Capture, JSONL parsing, and candidate trace construction are excluded from both sides; this is not end-to-end Maya speedup.",
        "- Reuse rate: compact candidate trace events not selected for refresh.",
        "",
        "## Results",
        "",
        "| Workload | Optimization | Init metric | Core speedup | Reuse rate |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in ok_rows:
        case = row["case"]
        metrics = row["metrics"]
        readme.append(
            f"| {case['workload']} | {case['optimization']} | "
            f"{float(metrics['init_metric_tinit_over_tb']):.3f} | "
            f"{float(metrics['core_refresh_speedup']):.2f}x | "
            f"{100.0 * float(metrics['reuse_rate']):.2f}% |"
        )
    failed = [row for row in results if "error" in row]
    if failed:
        readme.extend(["", "## Failed cases", ""])
        for row in failed:
            readme.append(f"- {row['case']['workload']}: {row['error']}")
    (args.out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_names = set(args.case or [case.name for case in CASES])
    results = [measure_case(args, case) for case in CASES if case.name in selected_names]
    write_summary_files(args, results)
    print(json.dumps({"result": str(args.out_dir / "result.json"), "summary": str(args.out_dir / "summary.csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
