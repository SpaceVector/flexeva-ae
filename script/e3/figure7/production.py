#!/usr/bin/env python3
"""Paper-aligned Figure 6 Maya-full versus production selective refresh."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import shlex
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import flexmaya_ras as fm

from measure_maya_megatron_fakecuda_similarity import MODELED_KERNEL_NCCL_KINDS as KEY_EVENT_KINDS
from measure_megatron_trace_similarity import MegatronCase


PHASES = (
    "maya_emulation_s",
    "trace_processing_s",
    "trace_ras_compaction_s",
    "code_analysis_s",
    "source_ras_update_s",
    "grounding_s",
    "selective_emulation_s",
    "trace_patch_collation_s",
    "event_simulation_s",
    "feedback_generation_s",
    "unattributed_overhead_s",
)
GPT_OPS = (
    "attention_forward",
    "mlp_forward",
    "attention_backward",
    "mlp_backward",
    "p2p",
    "optimizer_step",
)
MOE_OPS = (
    "attention_forward",
    "router_forward",
    "expert_forward",
    "expert_backward",
    "router_backward",
    "attention_backward",
    "optimizer_step",
    "route_path",
)
SEED_BASE = 6100
FORMAL_REPEATS = 1
FEEDBACK_TOLERANCE = 0.10
MIN_FREE_GIB = 500


@dataclass(frozen=True)
class MoeConfig:
    name: str = "runtime_matched_16rank_routed_moe"
    world_size: int = 16
    ep_size: int = 16
    dp: int = 1
    steps: int = 1
    global_batch_size: int = 128
    seq_len: int = 64
    hidden_size: int = 128
    num_layers: int = 32
    num_heads: int = 4
    vocab_size: int = 32000
    num_experts: int = 16
    top_k: int = 2
    capacity_factor: float = 1.25
    micro_batches: int = 8
    dtype: str = "bf16"


@dataclass(frozen=True)
class Round:
    number: int
    name: str
    label: str
    changed_ops: tuple[str, ...]
    route_experts: tuple[int, int] = (0, 1)
    topology_change: bool = False


GPT_ANCHOR = MegatronCase(
    name="gpt_18p4b_tp2_pp8_dp1",
    parameter_scale="18.4B",
    steps=1,
    global_batch_size=512,
    seq_len=2048,
    hidden_size=6144,
    num_layers=40,
    num_heads=48,
    vocab_size=32000,
    tp=2,
    pp=8,
    dp=1,
    world_size=16,
    micro_batches=512,
    schedule="1f1b",
    dtype="bf16",
)
GPT_R4 = replace(
    GPT_ANCHOR,
    name="gpt_18p4b_tp1_pp8_dp2",
    tp=1,
    dp=2,
    micro_batches=256,
)
GPT_ROUNDS = (
    Round(1, "r1_attention_backward", "Attn", ("attention_backward",)),
    Round(2, "r2_attention_mlp_backward", "Attn+MLP", ("attention_backward", "mlp_backward")),
    Round(
        3,
        "r3_attention_mlp_optimizer",
        "Attn+MLP+Opt",
        ("attention_backward", "mlp_backward", "optimizer_step"),
    ),
    Round(
        4,
        "r4_attention_mlp_optimizer_tp_dp",
        "Attn+MLP+Opt+TP/DP",
        ("attention_backward", "mlp_backward", "optimizer_step"),
        topology_change=True,
    ),
)
MOE_ROUNDS = (
    Round(1, "r1_router_backward", "Router", ("router_backward",)),
    Round(2, "r2_router_attention_backward", "Router+Attn", ("router_backward", "attention_backward")),
    Round(
        3,
        "r3_router_attention_optimizer",
        "Router+Attn+Opt",
        ("router_backward", "attention_backward", "optimizer_step"),
    ),
    Round(
        4,
        "r4_router_attention_optimizer_route",
        "Router+Attn+Opt+Route",
        ("router_backward", "attention_backward", "optimizer_step", "route_path"),
        route_experts=(7, 8),
    ),
)


@dataclass
class Worker:
    workload: str
    system: str
    case_dir: Path
    peer_case_dir: Path
    trace_dir: Path
    peer_trace_dir: Path
    ready_file: Path
    dispatch_file: Path
    feedback_file: Path
    local_command: list[str]
    peer_command: list[str]
    local_process: subprocess.Popen[str]
    peer_process: subprocess.Popen[str]
    local_stdout: object
    local_stderr: object
    peer_stdout: object
    peer_stderr: object
    local_arena: object
    local_binary_path: Path
    peer_binary_path: Path
    spawned_ns: int
    ready_ns: int = 0
    dispatch_ns: int = 0
    process_exit_ns: int = 0
    peak_rss_kib: int = 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "run", "report", "verify", "self-test"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--maya-root", type=Path)
    parser.add_argument("--proot", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--peer-target", default="")
    parser.add_argument("--peer-port", type=int, default=2222)
    parser.add_argument("--peer-repo-root", type=Path)
    parser.add_argument("--peer-python", type=Path)
    parser.add_argument("--peer-maya-root", type=Path)
    parser.add_argument("--peer-proot", type=Path)
    parser.add_argument("--peer-node-root", type=Path)
    parser.add_argument("--peer-work-root", type=Path)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--master-addr", default="")
    parser.add_argument("--master-port-base", type=int, default=45100)
    parser.add_argument("--socket-ifname", default="eth1")
    parser.add_argument("--ready-timeout-s", type=int, default=3600)
    parser.add_argument("--worker-timeout-s", type=int, default=21600)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--probe-round", type=int, choices=range(1, 5), default=1)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def empty_phases() -> dict[str, float]:
    return {name: 0.0 for name in PHASES}


def timed(fn):
    started = time.perf_counter()
    value = fn()
    return time.perf_counter() - started, value



def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_rss_kib(root_pid: int) -> int:
    pending = [root_pid]
    seen: set[int] = set()
    total = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
                    break
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="utf-8")
            pending.extend(int(item) for item in children.split())
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return total


def canonical_json_hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def local_micro_batch(case: MegatronCase) -> int:
    return max(1, math.ceil(case.global_batch_size / (case.dp * case.micro_batches)))


def effective_gpt_batch(case: MegatronCase) -> int:
    return local_micro_batch(case) * case.dp * case.micro_batches


def gpt_case(round_: Round | None) -> MegatronCase:
    return GPT_R4 if round_ is not None and round_.topology_change else GPT_ANCHOR


def config_payload(workload: str, config: object) -> dict[str, object]:
    payload = asdict(config)
    if workload == "gpt":
        payload["local_micro_batch_size"] = local_micro_batch(config)
        payload["effective_global_batch"] = effective_gpt_batch(config)
    else:
        payload["local_micro_batch_size"] = config.global_batch_size // (config.dp * config.micro_batches)
        payload["effective_global_batch"] = payload["local_micro_batch_size"] * config.dp * config.micro_batches
    return payload


def stage_partition(stage: int, op: str) -> str:
    return f"stage_{stage:03d}_{op}"


def write_sources(
    root: Path,
    workload: str,
    config: object,
    rounds: tuple[Round, ...],
) -> tuple[dict[str, Path], dict[int, dict[str, Path]]]:
    source_dir = root / workload / "source_partitions"
    source_dir.mkdir(parents=True, exist_ok=False)
    ops = GPT_OPS if workload == "gpt" else MOE_OPS
    partition_ids = (
        [stage_partition(stage, op) for stage in range(config.pp) for op in ops]
        if workload == "gpt"
        else list(ops)
    )
    anchors: dict[str, Path] = {}
    for partition_id in partition_ids:
        path = source_dir / f"{partition_id}_anchor.py"
        path.write_text(
            f"PARTITION = {partition_id!r}\nVERSION = 'anchor'\n",
            encoding="utf-8",
        )
        anchors[partition_id] = path
    candidates: dict[int, dict[str, Path]] = {}
    for round_ in rounds:
        paths = dict(anchors)
        for partition_id in partition_ids:
            op = partition_id.rsplit("_", 1)[-1] if workload == "gpt" else partition_id
            # GPT op names contain underscores; match the full suffix.
            if workload == "gpt":
                op = next((name for name in ops if partition_id.endswith(f"_{name}")), "")
            if op not in round_.changed_ops:
                continue
            path = source_dir / f"{partition_id}_{round_.name}.py"
            path.write_text(
                f"PARTITION = {partition_id!r}\nVERSION = {round_.name!r}\n",
                encoding="utf-8",
            )
            paths[partition_id] = path
        candidates[round_.number] = paths
    return anchors, candidates


def workload_spec(
    workload: str,
    config: object,
    source_paths: dict[str, Path],
    workload_id: str,
) -> fm.FlexMayaWorkloadSpec:
    if workload == "gpt":
        partitions = tuple(
            fm.CodePartitionSpec(
                stage_partition(stage, op),
                str(source_paths[stage_partition(stage, op)]),
                active_ranks=fm.megatron_pp_stage_active_ranks(
                    config.world_size, config.tp, config.pp, stage
                ),
                requires_grounding=True,
            )
            for stage in range(config.pp)
            for op in GPT_OPS
        )
        return fm.FlexMayaWorkloadSpec(
            workload_id=workload_id,
            world_size=config.world_size,
            tp=config.tp,
            pp=config.pp,
            dp=config.dp,
            code_partitions=partitions,
            rank_group_policy="none",
            notes=("GPT 18.4B Figure 6 production",),
        )
    return fm.FlexMayaWorkloadSpec(
        workload_id=workload_id,
        world_size=config.world_size,
        tp=1,
        pp=1,
        dp=config.dp,
        code_partitions=tuple(
            fm.CodePartitionSpec(
                op,
                str(source_paths[op]),
                active_ranks=tuple(range(config.world_size)),
                requires_grounding=True,
            )
            for op in MOE_OPS
        ),
        rank_group_policy="none",
        notes=("runtime-matched 16-rank Routed-MoE Figure 6 production",),
    )


def write_manifest(
    root: Path,
    workload: str,
    anchor_config: object,
    rounds: tuple[Round, ...],
    anchor_sources: dict[str, Path],
    candidate_sources: dict[int, dict[str, Path]],
) -> Path:
    payload = {
        "schema": "flexeva.figure6.candidate_manifest.v1",
        "workload": workload,
        "description": (
            "GPT 18.4B" if workload == "gpt" else "runtime-matched 16-rank Routed-MoE"
        ),
        "anchor_config": config_payload(workload, anchor_config),
        "anchor_sources": {key: {"path": str(path), "sha256": sha256(path)} for key, path in anchor_sources.items()},
        "rounds": [],
    }
    for round_ in rounds:
        config = gpt_case(round_) if workload == "gpt" else anchor_config
        payload["rounds"].append(
            {
                **asdict(round_),
                "config": config_payload(workload, config),
                "sources": {
                    key: {"path": str(path), "sha256": sha256(path)}
                    for key, path in candidate_sources[round_.number].items()
                },
            }
        )
    path = root / workload / "candidate_manifest.json"
    atomic_json(path, payload)
    return path


def shell_join(command: list[object] | tuple[object, ...]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def peer_ssh(args: argparse.Namespace, remote_command: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-p",
        str(args.peer_port),
        args.peer_target,
        remote_command,
    ]



def shared_arena_capacity(config: object) -> int:
    return max(2_000_000, 8 * int(config.micro_batches) * int(config.num_layers) * 48)


def hook_library(repo_root: Path) -> Path:
    package_dir = repo_root / "FlexEva/flexmaya_ras/src/flexmaya_ras"
    candidates = sorted(package_dir.glob("_flexmaya_ras*.so"))
    require(candidates, f"built FlexMaya hook library is missing under {package_dir}")
    return candidates[0]


def worker_environment(
    args: argparse.Namespace,
    repo_root: Path,
    maya_root: Path,
    proot: Path,
    trace_dir: Path,
    workload: str,
    config: object,
    arena_name: str,
    hook_library_path: str,
    audit_tag: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("FAKECUDA_TRACE", None)
    env.pop("FAKECUDA_TRACE_PATH", None)
    for key in (
        "FLEXMAYA_SHARED_ARENA_ONLY",
        "FLEXMAYA_SHM_NAME",
        "FLEXMAYA_TRACE_MODEL_WINDOW",
        "FLEXMAYA_RAW_AUDIT_DIR",
        "FLEXMAYA_RAW_AUDIT_TAG",
        "FLEXMAYA_PP",
        "PLAIN_MAYA_HOOK_LIBRARY",
    ):
        env.pop(key, None)
    env.update(
        {
            "FAKECUDA_PROOT_BIN": str(proot),
            "FAKECUDA_FRUN_QUIET": "1",
            "FAKECUDA_TRACE_STDIO_BUFFER_BYTES": "0",
            "FLEXSIM_CLUSTER_CPU_AFFINITY": "1",
            "FLEXMAYA_TRACE_DIR": str(trace_dir),
            "FLEXMAYA_RAW_AUDIT_DIR": str(trace_dir),
            "FLEXMAYA_RAW_AUDIT_TAG": audit_tag,
            "FLEXMAYA_LOCAL_DEVICE_COUNT": "8",
            "GLOO_SOCKET_IFNAME": args.socket_ifname,
            "NCCL_IB_DISABLE": "1",
            "NCCL_SOCKET_IFNAME": f"={args.socket_ifname}",
            "PYTHONPATH": os.pathsep.join(
                (
                    str(
                        repo_root
                        / "script/e3/workload"
                        / ("megatron" if workload == "gpt" else "routed_moe")
                    ),
                    str(maya_root / "python"),
                    str(maya_root / "CppEvent"),
                    str(repo_root / "FlexEva/flexmaya_ras/src"),
                    str(repo_root / "script/e3/capture"),
                    str(repo_root / "script/e3/figure7"),
                    env.get("PYTHONPATH", ""),
                )
            ),
        }
    )
    script = repo_root / "script/e3/workload" / (
        "megatron/maya_megatron.py" if workload == "gpt" else "routed_moe/moe_topk.py"
    )
    env["MAYA_MEGATRON_SCRIPT" if workload == "gpt" else "ROUTED_MOE_SCRIPT"] = str(script)
    env.update(
        {
            "FLEXMAYA_SHM_NAME": arena_name,
            "FLEXMAYA_TRACE_MODEL_WINDOW": "0",
            "FLEXMAYA_PP": str(getattr(config, "pp", 1)),
            "PLAIN_MAYA_HOOK_LIBRARY": hook_library_path,
        }
    )
    env["FLEXMAYA_SHARED_ARENA_ONLY"] = (
        "0" if getattr(args, "jsonl_trace", False) else "1"
    )
    return env

def workload_arguments(
    workload: str,
    config: object,
    round_: Round | None,
    system: str,
    ready: Path,
    dispatch: Path,
    feedback: Path,
    timing: Path,
    selected_ops: tuple[str, ...],
    seed: int,
) -> list[str]:
    common = [
        "--steps",
        "1",
        "--warmup-steps",
        "0",
        "--global-batch-size",
        str(config.global_batch_size),
        "--seq-len",
        str(config.seq_len),
        "--hidden-size",
        str(config.hidden_size),
        "--num-layers",
        str(config.num_layers),
        "--num-heads",
        str(config.num_heads),
        "--vocab-size",
        str(config.vocab_size),
        "--micro-batches",
        str(config.micro_batches),
        "--dtype",
        config.dtype,
        "--seed",
        str(seed),
        "--log-interval",
        "1000000",
        "--source-region-markers",
        "--figure6-production",
        "--sync-before-step-window",
        "--validate-effective-global-batch",
    ]
    if workload == "gpt":
        return common + [
            "--tp",
            str(config.tp),
            "--pp",
            str(config.pp),
            "--dp",
            str(config.dp),
            "--schedule",
            config.schedule,
            "--evaluator-executor",
            system,
            "--evaluator-candidate-id",
            round_.name if round_ is not None else "anchor",
            "--evaluator-mutations",
            ",".join(round_.changed_ops) if round_ is not None else "",
            "--evaluator-partitions",
            ",".join(selected_ops),
            "--evaluator-ready-file",
            str(ready),
            "--evaluator-dispatch-file",
            str(dispatch),
            "--evaluator-feedback-output",
            str(feedback),
            "--timing-output",
            str(timing),
        ]
    route = round_.route_experts if round_ is not None else (0, 1)
    return common + [
        "--num-experts",
        str(config.num_experts),
        "--top-k",
        str(config.top_k),
        "--capacity-factor",
        str(config.capacity_factor),
        "--ep-size",
        str(config.ep_size),
        "--dp",
        str(config.dp),
        "--tp",
        "1",
        "--pp",
        "1",
        "--route-experts",
        ",".join(map(str, route)),
        "--route-p2p-probe",
        "--figure6-candidate-id",
        round_.name if round_ is not None else "anchor",
        "--figure6-mutations",
        ",".join(round_.changed_ops) if round_ is not None else "",
        "--evaluator-executor",
        system,
        "--evaluator-partitions",
        ",".join(selected_ops),
        "--evaluator-ready-file",
        str(ready),
        "--evaluator-dispatch-file",
        str(dispatch),
        "--evaluator-feedback-output",
        str(feedback),
        "--timing-output",
        str(timing),
    ]


def torchrun_command(
    *,
    repo_root: Path,
    maya_root: Path,
    python: Path,
    workload: str,
    node_rank: int,
    master_addr: str,
    master_port: int,
    workload_args: list[str],
) -> list[str]:
    wrapper = repo_root / "script/e3/capture" / (
        "maya_megatron_trace_worker.py" if workload == "gpt" else "routed_moe_trace_worker.py"
    )
    return [
        str(maya_root / "fake-cuda/frun"),
        str(python),
        "-m",
        "torch.distributed.run",
        "--nnodes=2",
        f"--node-rank={node_rank}",
        "--nproc-per-node=8",
        f"--master-addr={master_addr}",
        f"--master-port={master_port}",
        str(wrapper),
        *workload_args,
    ]



def spawn_worker(
    args: argparse.Namespace,
    *,
    workload: str,
    system: str,
    config: object,
    round_: Round | None,
    selected_ops: tuple[str, ...],
    seed: int,
    case_dir: Path,
    peer_case_dir: Path,
    port: int,
    peer_run_id: str,
) -> Worker:
    repo_root = args.repo_root.resolve()
    peer_repo_root = args.peer_repo_root
    case_dir.mkdir(parents=True, exist_ok=False)
    trace_dir = case_dir / "traces"
    trace_dir.mkdir()
    peer_trace_dir = peer_case_dir / "traces"
    local_binary_path = trace_dir / "node0_events.bin"
    peer_binary_path = peer_trace_dir / "node1_events.bin"
    arena_name = f"flexeva-figure6-{os.getpid()}-{port}"
    local_arena = fm.SharedEventArena.create(arena_name, shared_arena_capacity(config), True)
    ready = case_dir / "ready.json"
    dispatch = case_dir / "dispatch"
    feedback = case_dir / "executor_feedback.json"
    timing = case_dir / "executor_timing.json"
    local_args = workload_arguments(
        workload, config, round_, system, ready, dispatch, feedback, timing, selected_ops, seed
    )
    peer_args = workload_arguments(
        workload, config, round_, system, ready, dispatch, feedback, timing, selected_ops, seed
    )
    local_command = torchrun_command(
        repo_root=repo_root,
        maya_root=args.maya_root,
        python=args.python,
        workload=workload,
        node_rank=0,
        master_addr=args.master_addr,
        master_port=port,
        workload_args=local_args,
    )
    peer_maya = args.peer_maya_root
    peer_command = torchrun_command(
        repo_root=peer_repo_root,
        maya_root=peer_maya,
        python=args.peer_python,
        workload=workload,
        node_rank=1,
        master_addr=args.master_addr,
        master_port=port,
        workload_args=peer_args,
    )
    local_env = worker_environment(
        args,
        repo_root,
        args.maya_root,
        args.proot,
        trace_dir,
        workload,
        config,
        arena_name,
        str(hook_library(repo_root)),
        "node0",
    )
    peer_env = worker_environment(
        args,
        peer_repo_root,
        peer_maya,
        args.peer_proot,
        peer_trace_dir,
        workload,
        config,
        arena_name,
        str(
            peer_repo_root
            / "FlexEva/flexmaya_ras/src/flexmaya_ras"
            / hook_library(repo_root).name
        ),
        "node1",
    )
    peer_env["FAKECUDA_TARGET_ENV_ROOT"] = str(args.peer_python.parent.parent)
    peer_server = peer_repo_root / "script/e3/server.sh"
    exported = {
        key: value
        for key, value in peer_env.items()
        if key
        in {
            "FAKECUDA_PROOT_BIN",
            "FAKECUDA_FRUN_QUIET",
            "FAKECUDA_TARGET_ENV_ROOT",
            "FAKECUDA_TRACE_STDIO_BUFFER_BYTES",
            "FLEXMAYA_SHARED_ARENA_ONLY",
            "FLEXMAYA_SHM_NAME",
            "FLEXMAYA_TRACE_MODEL_WINDOW",
            "FLEXMAYA_RAW_AUDIT_DIR",
            "FLEXMAYA_RAW_AUDIT_TAG",
            "FLEXMAYA_PP",
            "PLAIN_MAYA_HOOK_LIBRARY",
            "FLEXSIM_CLUSTER_CPU_AFFINITY",
            "FLEXMAYA_TRACE_DIR",
            "FLEXMAYA_LOCAL_DEVICE_COUNT",
            "GLOO_SOCKET_IFNAME",
            "NCCL_IB_DISABLE",
            "NCCL_SOCKET_IFNAME",
            "PYTHONPATH",
            "MAYA_MEGATRON_SCRIPT",
            "ROUTED_MOE_SCRIPT",
        }
    }
    recorded_keys = sorted(set(exported) | {"CUDA_VISIBLE_DEVICES", "CANONICAL_PYTHON", "TMPDIR"})
    atomic_json(
        case_dir / "environment_node0.json",
        {key: local_env.get(key, "") for key in recorded_keys},
    )
    atomic_json(case_dir / "environment_node1.json", dict(sorted(exported.items())))
    peer_runner = peer_repo_root / "script/e3/capture/shared_arena_peer_runner.py"
    peer_runner_command = [
        args.peer_python,
        peer_runner,
        "--arena-name",
        arena_name,
        "--arena-capacity",
        str(shared_arena_capacity(config)),
        "--events-output",
        peer_binary_path,
        "--",
        *peer_command,
    ]
    peer_shell = " ".join(
        (
            f"AE_NODE_ROOT={shlex.quote(str(args.peer_node_root))}",
            f"AE_CANONICAL_PYTHON={shlex.quote(str(args.peer_python))}",
            f"MIN_GPFS_FREE_GIB={MIN_FREE_GIB}",
            shell_join(
                [
                    peer_server,
                    "run",
                    peer_run_id,
                    "8",
                    "--",
                    "/usr/bin/env",
                    *(f"{key}={value}" for key, value in exported.items()),
                    "/usr/bin/time",
                    "-v",
                    *peer_runner_command,
                ]
            ),
        )
    )
    (case_dir / "command_node0.txt").write_text(shell_join(local_command) + "\n", encoding="utf-8")
    (case_dir / "command_node1.txt").write_text(shell_join(peer_runner_command) + "\n", encoding="utf-8")
    local_stdout = (case_dir / "stdout_node0.log").open("w", encoding="utf-8")
    local_stderr = (case_dir / "stderr_node0.log").open("w", encoding="utf-8")
    peer_stdout = (case_dir / "stdout_node1.log").open("w", encoding="utf-8")
    peer_stderr = (case_dir / "stderr_node1.log").open("w", encoding="utf-8")
    peer_process = subprocess.Popen(
        peer_ssh(args, peer_shell),
        stdout=peer_stdout,
        stderr=peer_stderr,
        text=True,
        start_new_session=True,
    )
    time.sleep(2.0)
    local_process = subprocess.Popen(
        local_command,
        cwd=case_dir,
        env=local_env,
        stdout=local_stdout,
        stderr=local_stderr,
        text=True,
        start_new_session=True,
    )
    return Worker(
        workload=workload,
        system=system,
        case_dir=case_dir,
        peer_case_dir=peer_case_dir,
        trace_dir=trace_dir,
        peer_trace_dir=peer_trace_dir,
        ready_file=ready,
        dispatch_file=dispatch,
        feedback_file=feedback,
        local_command=local_command,
        peer_command=peer_command,
        local_process=local_process,
        peer_process=peer_process,
        local_stdout=local_stdout,
        local_stderr=local_stderr,
        peer_stdout=peer_stdout,
        peer_stderr=peer_stderr,
        local_arena=local_arena,
        local_binary_path=local_binary_path,
        peer_binary_path=peer_binary_path,
        spawned_ns=time.time_ns(),
    )

def terminate(worker: Worker) -> None:
    for process in (worker.local_process, worker.peer_process):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def wait_ready(worker: Worker, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    while not worker.ready_file.is_file():
        exits = (worker.local_process.poll(), worker.peer_process.poll())
        if any(code is not None for code in exits):
            terminate(worker)
            raise RuntimeError(f"worker exited before ready: local={exits[0]} peer={exits[1]} case={worker.case_dir}")
        if time.monotonic() >= deadline:
            terminate(worker)
            raise TimeoutError(f"worker did not become ready: {worker.case_dir}")
        time.sleep(0.1)
    worker.ready_ns = time.time_ns()


def dispatch_and_wait(worker: Worker, timeout_s: int) -> float:
    worker.dispatch_ns = time.time_ns()
    started = time.perf_counter()
    worker.dispatch_file.touch(exist_ok=False)
    deadline = time.monotonic() + timeout_s
    while True:
        worker.peak_rss_kib = max(worker.peak_rss_kib, tree_rss_kib(worker.local_process.pid))
        local_exit = worker.local_process.poll()
        peer_exit = worker.peer_process.poll()
        if local_exit is not None and peer_exit is not None:
            break
        if time.monotonic() >= deadline:
            terminate(worker)
            raise TimeoutError(f"worker timed out: {worker.case_dir}")
        time.sleep(0.1)
    worker.process_exit_ns = time.time_ns()
    for handle in (worker.local_stdout, worker.local_stderr, worker.peer_stdout, worker.peer_stderr):
        handle.close()
    if local_exit != 0 or peer_exit != 0:
        raise RuntimeError(
            f"worker failed local={local_exit} peer={peer_exit}; see {worker.case_dir}/stderr_node*.log"
        )
    require(worker.feedback_file.is_file(), f"missing executor feedback: {worker.feedback_file}")
    return time.perf_counter() - started



def collect_peer_traces(args: argparse.Namespace, worker: Worker, world_size: int) -> float:
    started = time.perf_counter()
    paths = binary_paths(worker)
    require(len(set(paths)) == 2, "node compact trace paths must be distinct")
    worker.local_arena.write_binary(str(worker.local_binary_path))
    peer_dir = shlex.quote(str(worker.peer_trace_dir))
    jsonl_pattern = " -o -name 'rank_*.jsonl'" if getattr(args, "jsonl_trace", False) else ""
    peer_tar = (
        f"cd {peer_dir} && "
        "find . -maxdepth 1 -type f "
        f"\\( -name 'node1_events.bin' -o -name 'node1.cudaGetDevice.*.jsonl'{jsonl_pattern} \\) "
        "-printf '%f\\0' | tar --null --files-from=- -cf -"
    )
    archive = subprocess.Popen(
        peer_ssh(args, peer_tar),
        stdout=subprocess.PIPE,
    )
    require(archive.stdout is not None, "cannot read peer compact trace archive")
    try:
        extract = subprocess.run(
            ["tar", "-C", str(worker.trace_dir), "-xf", "-"],
            stdin=archive.stdout,
            check=False,
        )
    finally:
        archive.stdout.close()
    archive_exit = archive.wait()
    require(archive_exit == 0 and extract.returncode == 0, "peer compact trace collection failed")
    require(worker.local_binary_path.is_file() and worker.local_binary_path.stat().st_size > 16, "local compact trace is empty")
    require(worker.trace_dir.joinpath(worker.peer_binary_path.name).is_file(), "peer compact trace is missing")
    if not getattr(args, "jsonl_trace", False):
        require(raw_audit_paths(worker), "cudaGetDevice raw audit files are missing")
    if getattr(args, "jsonl_trace", False):
        rank_paths = [worker.trace_dir / f"rank_{rank}.jsonl" for rank in range(world_size)]
        marker_paths = [worker.trace_dir / f"rank_{rank}_markers.jsonl" for rank in range(world_size)]
        require(all(path.is_file() and path.stat().st_size > 0 for path in rank_paths), "rank JSONL trace is incomplete")
        require(all(path.is_file() and path.stat().st_size > 0 for path in marker_paths), "rank marker trace is incomplete")
    return time.perf_counter() - started

def peak_rss_payload(worker: Worker) -> dict[str, int]:
    peer_peak = 0
    stderr = worker.case_dir / "stderr_node1.log"
    if stderr.is_file():
        for line in stderr.read_text(encoding="utf-8", errors="replace").splitlines():
            if "Maximum resident set size (kbytes):" in line:
                peer_peak = max(peer_peak, int(line.rsplit(":", 1)[1].strip()))
    return {
        "node0_process_tree_kib": worker.peak_rss_kib,
        "coordinator_driver_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "node1_command_max_kib": peer_peak,
    }


def signature_payload(counter: Counter[tuple[object, ...]]) -> dict[str, object]:
    rows = [(list(key), value) for key, value in sorted(counter.items(), key=lambda item: repr(item[0]))]
    collectives = [row for row in rows if row[0][1] == "nccl_collective"]
    return {
        "event_count": sum(counter.values()),
        "digest": canonical_json_hash(rows),
        "collective_participation_digest": canonical_json_hash(collectives),
        "collective_event_count": sum(int(row[1]) for row in collectives),
    }


def key_event_signature(event: object) -> tuple[object, ...]:
    return (
        str(event.api),
        str(event.kind),
        int(event.rank),
        int(event.stream),
        int(event.bytes),
        int(event.count),
        int(event.peer_rank),
        str(event.collective_group).split(":call=", 1)[0],
        str(event.code_partition),
        bool(event.blocking),
    )


def trace_key_signature(
    trace: object,
    partitions: tuple[str, ...] = (),
) -> dict[str, object]:
    selected = set(partitions)
    return signature_payload(
        Counter(
            key_event_signature(event)
            for event in trace.events
            if str(event.kind) in KEY_EVENT_KINDS
            and (not selected or str(event.code_partition) in selected)
        )
    )


def selected_code_partitions(
    plan: fm.FlexMayaRefreshPlan,
    anchor_trace: object,
    candidate_spec: fm.FlexMayaWorkloadSpec,
) -> tuple[str, ...]:
    affected = set(int(item) for item in plan.affected_trace_partitions)
    known = {partition.partition_id for partition in candidate_spec.code_partitions}
    grounded = {
        str(edge.upper_partition)
        for edge in anchor_trace.lineage_edges
        if int(edge.lower_partition) in affected
        and str(edge.edge_kind) == "code_to_trace_partition"
        and str(edge.upper_partition) in known
    }
    source_changed = set(plan.changed_partitions) & known
    selected = tuple(sorted(grounded | source_changed))
    require(selected, "source-RAS selected no executable code partition")
    missing = source_changed - set(selected)
    require(not missing, f"ungrounded changed partitions: {sorted(missing)}")
    return selected


def base_executor_ops(workload: str, partitions: tuple[str, ...]) -> tuple[str, ...]:
    allowed = GPT_OPS if workload == "gpt" else MOE_OPS
    if workload == "moe":
        return tuple(op for op in allowed if op in partitions)
    return tuple(op for op in allowed if any(partition.endswith(f"_{op}") for partition in partitions))


def active_lane_evidence(
    selected: tuple[str, ...],
    spec: fm.FlexMayaWorkloadSpec,
) -> dict[str, list[int]]:
    by_id = {partition.partition_id: list(partition.active_ranks) for partition in spec.code_partitions}
    return {partition: by_id[partition] for partition in selected}



def binary_paths(worker: Worker) -> list[str]:
    paths = [worker.local_binary_path, worker.trace_dir / worker.peer_binary_path.name]
    require(len({path.resolve() for path in paths}) == 2, "node compact trace paths must be distinct")
    return [str(path) for path in paths]


def raw_audit_paths(worker: Worker) -> list[Path]:
    return sorted(worker.trace_dir.glob("*.cudaGetDevice.*.jsonl"))


def raw_cuda_get_device_count(worker: Worker) -> int:
    paths = raw_audit_paths(worker)
    require(paths, "cudaGetDevice raw audit files are missing")
    count = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                record = json.loads(line)
                require(record["api"] == "cudaGetDevice", f"invalid raw audit record: {path}")
                count += int(record.get("count", 1))
    require(count > 0, "cudaGetDevice raw audit is empty")
    return count


def binary_trace_audit(trace: object, worker: Worker, world_size: int) -> dict[str, object]:
    paths = binary_paths(worker)
    ranks = sorted({int(event.rank) for event in trace.events})
    require(ranks == list(range(world_size)), f"compact traces do not cover ranks 0-{world_size - 1}: {ranks}")
    return {
        "trace_source": "SharedEventArena",
        "binary_file_count": 2,
        "binary_paths": paths,
        "binary_paths_distinct": len(set(paths)) == 2,
        "world_size": world_size,
        "rank_coverage": ranks,
        "raw_event_count": len(trace.events),
        "cudaGetDevice_raw_count": raw_cuda_get_device_count(worker),
        "cudaGetDevice_modeled_count": 0,
        "model_excluded_apis": ["cudaGetDevice"],
    }



def process_full_worker(
    args: argparse.Namespace,
    worker: Worker,
    workload: str,
    config: object,
    signature_partitions: tuple[str, ...],
) -> tuple[dict[str, object], object, object]:
    phases = empty_phases()
    phases["maya_emulation_s"] = dispatch_and_wait(worker, args.worker_timeout_s)
    collect_s = collect_peer_traces(args, worker, config.world_size)
    del workload
    build_s, trace = timed(lambda: fm.build_trace_ras_from_binary(binary_paths(worker)))
    phases["trace_processing_s"] = collect_s + build_s
    phases["event_simulation_s"], feedback = timed(lambda: fm.replay_trace_once(trace))
    phases["feedback_generation_s"], feedback_payload = timed(feedback.to_dict)
    executor_feedback = json.loads(worker.feedback_file.read_text(encoding="utf-8"))
    row = {
        "system": "maya-full",
        "executor_feedback": executor_feedback,
        "command_node0": worker.local_command,
        "command_node1": worker.peer_command,
        "logs": {
            "stdout_node0": str(worker.case_dir / "stdout_node0.log"),
            "stderr_node0": str(worker.case_dir / "stderr_node0.log"),
            "stdout_node1": str(worker.case_dir / "stdout_node1.log"),
            "stderr_node1": str(worker.case_dir / "stderr_node1.log"),
            "environment_node0": str(worker.case_dir / "environment_node0.json"),
            "environment_node1": str(worker.case_dir / "environment_node1.json"),
        },
        "trace": binary_trace_audit(trace, worker, config.world_size),
        "selected_signature": trace_key_signature(trace, signature_partitions),
        "output_trace": fm.trace_summary(trace),
        "output_key_signature": trace_key_signature(trace),
        "feedback": feedback_payload,
        "phases_s": phases,
        "peak_rss_kib": peak_rss_payload(worker),
        "fallback": False,
    }
    return row, trace, feedback

def collate_selective_trace(
    anchor_trace: object,
    replacement_trace: object,
    candidate_spec: fm.FlexMayaWorkloadSpec,
    plan: fm.FlexMayaRefreshPlan,
    selected: tuple[str, ...],
) -> tuple[object, dict[str, object]]:
    if not plan.configuration_changed:
        expected = set(selected)
        for name, trace in (("anchor", anchor_trace), ("replacement", replacement_trace)):
            available = {str(edge.upper_partition) for edge in trace.lineage_edges}
            require(expected <= available, f"{name} trace is missing selected code partitions")
        return fm.patch_trace_code_partitions(anchor_trace, replacement_trace, list(selected)), {
            "mode": "chunk_patch",
            "method": "patch anchor chunks from the true selective trace",
        }

    candidate_partitions = {partition.partition_id for partition in candidate_spec.code_partitions}
    require(
        set(selected) == candidate_partitions,
        "topology rebase requires every executable candidate partition",
    )
    replacement_partitions = {
        str(event.code_partition)
        for event in replacement_trace.events
        if str(event.code_partition) in candidate_partitions
    }
    require(
        replacement_partitions == candidate_partitions,
        "topology rebase selective trace is missing executable candidate partitions",
    )
    anchor_ranks = {int(event.rank) for event in anchor_trace.events}
    replacement_ranks = {int(event.rank) for event in replacement_trace.events}
    require(anchor_ranks == replacement_ranks, "topology rebase changed the logical rank set")
    return replacement_trace, {
        "mode": "configuration_rebase",
        "method": "rebase anchor source-RAS state onto the true selective candidate topology trace",
        "reason": "parallel geometry changed the schedule shape; every executable partition was refreshed",
        "full_executable_partition_coverage": True,
        "candidate_partition_count": len(candidate_partitions),
        "anchor_state_inputs": ["source hashes", "lineage edges", "affected trace partitions"],
    }



def process_selective_worker(
    args: argparse.Namespace,
    worker: Worker,
    workload: str,
    config: object,
    anchor: fm.FlexMayaAnchor,
    candidate_spec: fm.FlexMayaWorkloadSpec,
    plan: fm.FlexMayaRefreshPlan,
    selected: tuple[str, ...],
    phases: dict[str, float],
) -> dict[str, object]:
    phases["selective_emulation_s"] = dispatch_and_wait(worker, args.worker_timeout_s)
    collect_s = collect_peer_traces(args, worker, config.world_size)
    del workload
    phases["trace_processing_s"] = collect_s
    phases["source_ras_update_s"], replacement_trace = timed(
        lambda: fm.build_trace_ras_from_binary(binary_paths(worker))
    )
    phases["trace_patch_collation_s"], collated = timed(
        lambda: collate_selective_trace(
            anchor.trace,
            replacement_trace,
            candidate_spec,
            plan,
            selected,
        )
    )
    patched_trace, collation = collated
    phases["event_simulation_s"], feedback = timed(
        lambda: fm.replay_trace_once(patched_trace)
    )
    phases["feedback_generation_s"], feedback_payload = timed(feedback.to_dict)
    executor_feedback = json.loads(worker.feedback_file.read_text(encoding="utf-8"))
    replacement_summary = fm.trace_summary(replacement_trace)
    patched_summary = fm.trace_summary(patched_trace)
    return {
        "system": "flexeva-selective",
        "executor_feedback": executor_feedback,
        "command_node0": worker.local_command,
        "command_node1": worker.peer_command,
        "logs": {
            "stdout_node0": str(worker.case_dir / "stdout_node0.log"),
            "stderr_node0": str(worker.case_dir / "stderr_node0.log"),
            "stdout_node1": str(worker.case_dir / "stdout_node1.log"),
            "stderr_node1": str(worker.case_dir / "stderr_node1.log"),
            "environment_node0": str(worker.case_dir / "environment_node0.json"),
            "environment_node1": str(worker.case_dir / "environment_node1.json"),
        },
        "trace": binary_trace_audit(replacement_trace, worker, config.world_size),
        "selected_signature": trace_key_signature(replacement_trace, selected),
        "replacement_trace": replacement_summary,
        "output_trace": patched_summary,
        "output_key_signature": trace_key_signature(patched_trace),
        "feedback": feedback_payload,
        "source_ras_plan": asdict(plan),
        "executed_code_partitions": list(selected),
        "active_lane_grounding": active_lane_evidence(selected, candidate_spec),
        "anchor_patch": {
            **collation,
            "anchor_event_count": int(anchor.summary["trace"]["event_count"]),
            "replacement_event_count": replacement_summary["event_count"],
            "replacement_selected_event_count": int(
                trace_key_signature(replacement_trace)["event_count"]
            ),
            "patched_event_count": patched_summary["event_count"],
            "full_candidate_trace_generated": False,
        },
        "phases_s": phases,
        "peak_rss_kib": peak_rss_payload(worker),
        "fallback": bool(executor_feedback["fallback"]),
    }

def finish_timing(
    row: dict[str, object],
    worker: Worker,
    started_ns: int,
    started: float,
) -> None:
    wall_s = time.perf_counter() - started
    finished_ns = time.time_ns()
    phases = row["phases_s"]
    attributed = sum(float(phases[name]) for name in PHASES if name != "unattributed_overhead_s")
    phases["unattributed_overhead_s"] = max(wall_s - attributed, 0.0)
    row["timing"] = {
        "started_ns": started_ns,
        "finished_ns": finished_ns,
        "direct_wall_s": wall_s,
        "phase_sum_s": sum(float(phases[name]) for name in PHASES),
        "phase_audit_difference_s": wall_s - sum(float(phases[name]) for name in PHASES),
        "spawned_ns": worker.spawned_ns,
        "ready_ns": worker.ready_ns,
        "dispatch_ns": worker.dispatch_ns,
        "process_exit_ns": worker.process_exit_ns,
        "worker_bootstrap_excluded": True,
        "source": "direct monotonic wall clock; marker timestamps are used only for event attribution",
    }
    atomic_json(worker.case_dir / "evaluation_checkpoint.json", row)


def initialize_fresh_anchor(
    args: argparse.Namespace,
    *,
    workload: str,
    config: object,
    spec: fm.FlexMayaWorkloadSpec,
    case_dir: Path,
    peer_case_dir: Path,
    seed: int,
    port: int,
    peer_run_id: str,
) -> tuple[fm.FlexMayaAnchor, dict[str, object]]:
    worker = spawn_worker(
        args,
        workload=workload,
        system="maya-full",
        config=config,
        round_=None,
        selected_ops=(),
        seed=seed,
        case_dir=case_dir,
        peer_case_dir=peer_case_dir,
        port=port,
        peer_run_id=peer_run_id,
    )
    try:
        wait_ready(worker, args.ready_timeout_s)
        started_ns = time.time_ns()
        started = time.perf_counter()
        row, trace, feedback = process_full_worker(args, worker, workload, config, ())
        anchor = fm.FlexMayaAnchor(
            spec=spec,
            source_hashes=fm.source_hashes(spec),
            trace=trace,
            feedback=feedback,
            summary={
                "kind": "figure6_fresh_anchor",
                "trace": row["output_trace"],
                "feedback": row["feedback"],
            },
        )
        row["source_hashes"] = [asdict(item) for item in anchor.source_hashes]
        finish_timing(row, worker, started_ns, started)
        return anchor, row
    except BaseException:
        terminate(worker)
        raise


def evaluate_candidate(
    args: argparse.Namespace,
    *,
    workload: str,
    config: object,
    round_: Round,
    system: str,
    anchor: fm.FlexMayaAnchor,
    candidate_spec: fm.FlexMayaWorkloadSpec,
    case_dir: Path,
    peer_case_dir: Path,
    seed: int,
    port: int,
    peer_run_id: str,
) -> dict[str, object]:
    expected_source = tuple(
        partition.partition_id
        for partition in candidate_spec.code_partitions
        if partition.partition_id in set(round_.changed_ops)
        or any(partition.partition_id.endswith(f"_{op}") for op in round_.changed_ops)
    )
    analysis_started_ns = time.time_ns()
    analysis_started = time.perf_counter()
    started_ns = analysis_started_ns
    started = analysis_started
    pre_plan = fm.plan_candidate_refresh(
        anchor,
        candidate_spec,
        anchor.trace,
        grounding_satisfied=True,
    )
    pre_selected = selected_code_partitions(pre_plan, anchor.trace, candidate_spec)
    analysis_s = time.perf_counter() - analysis_started
    analysis_finished_ns = time.time_ns()
    require(set(expected_source) <= set(pre_selected), "source mutation is absent from the refresh plan")
    selected_ops = base_executor_ops(workload, pre_selected) if system == "flexeva-selective" else ()
    worker = spawn_worker(
        args,
        workload=workload,
        system=system,
        config=config,
        round_=round_,
        selected_ops=selected_ops,
        seed=seed,
        case_dir=case_dir,
        peer_case_dir=peer_case_dir,
        port=port,
        peer_run_id=peer_run_id,
    )
    try:
        wait_ready(worker, args.ready_timeout_s)
        if system == "maya-full":
            started_ns = time.time_ns()
            started = time.perf_counter()
            row, _, _ = process_full_worker(
                args,
                worker,
                workload,
                config,
                pre_selected,
            )
            row["source_ras_plan"] = asdict(pre_plan)
            row["executed_code_partitions"] = list(pre_selected)
        else:
            phases = empty_phases()
            phases["code_analysis_s"] = analysis_s
            plan = pre_plan
            selected = pre_selected
            require(
                not plan.fallback_reasons or plan.configuration_changed,
                f"source-RAS grounding fell back: {plan.fallback_reasons}",
            )
            row = process_selective_worker(
                args,
                worker,
                workload,
                config,
                anchor,
                candidate_spec,
                plan,
                selected,
                phases,
            )
        row["source_analysis_window"] = {
            "started_ns": analysis_started_ns,
            "finished_ns": analysis_finished_ns,
            "precedes_worker_spawn": True,
            "precedes_dispatch": analysis_finished_ns <= worker.dispatch_ns,
            "analysis_count": 1,
        }
        row.update(
            {
                "round": round_.number,
                "round_name": round_.name,
                "mutation_label": round_.label,
                "candidate_config": config_payload(workload, config),
                "seed": seed,
                "route_experts": list(round_.route_experts),
                "candidate_workers": 1,
                "logical_ranks": config.world_size,
            }
        )
        finish_timing(row, worker, started_ns, started)
        return row
    except BaseException:
        terminate(worker)
        raise


def correctness(
    workload: str,
    maya: dict[str, object],
    flex: dict[str, object],
) -> dict[str, object]:
    maya_feedback = maya["feedback"]
    flex_feedback = flex["feedback"]
    maya_time = float(maya_feedback["total_time_us"])
    flex_time = float(flex_feedback["total_time_us"])
    relative = abs(maya_time - flex_time) / max(abs(maya_time), 1.0)
    maya_executor = maya["executor_feedback"]
    flex_executor = flex["executor_feedback"]
    checks = {
        "same_manifest_config": maya["candidate_config"] == flex["candidate_config"],
        "same_seed": maya["seed"] == flex["seed"],
        "same_route": maya["route_experts"] == flex["route_experts"],
        "selected_key_events_equal": maya["selected_signature"]["digest"]
        == flex["selected_signature"]["digest"],
        "collective_participation_equal": maya["selected_signature"]["collective_participation_digest"]
        == flex["selected_signature"]["collective_participation_digest"],
        "final_key_events_equal": maya["output_key_signature"]["digest"]
        == flex["output_key_signature"]["digest"],
        "backend_completed": (
            not maya_feedback["cycle_detected"]
            and not flex_feedback["cycle_detected"]
            and maya_feedback["completed_events"] == maya_feedback["event_count"]
            and flex_feedback["completed_events"] == flex_feedback["event_count"]
        ),
        "backend_feedback_within_tolerance": relative <= FEEDBACK_TOLERANCE,
        "maya_fresh_full": (
            maya_executor["full_training_step_executed"] is True
            and maya_executor["full_candidate_trace_generated"] is True
        ),
        "flex_true_selective": (
            flex_executor["full_training_step_executed"] is False
            and flex_executor["full_candidate_trace_generated"] is False
            and flex_executor["fallback"] is False
        ),
    }
    if workload == "moe":
        checks["routing_feedback_equal"] = maya_executor["routing"] == flex_executor["routing"]
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "runtime_housekeeping_event_count_delta": (
            int(flex["output_trace"]["event_count"])
            - int(maya["output_trace"]["event_count"])
        ),
        "backend_total_time_relative_difference": relative,
        "backend_total_time_tolerance": FEEDBACK_TOLERANCE,
    }


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "staged-source"


def topology_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "physical_gpus": 16,
        "logical_ranks": 16,
        "node_count": 2,
        "gpus_per_node": 8,
        "gpu_model": "NVIDIA A100-SXM4-80GB",
        "target_topology": "two physical nodes, eight physical A100 GPUs and eight ranks per node",
        "logical_rank_to_physical_gpu": {
            str(rank): {
                "node": "coordinator" if rank < 8 else "peer",
                "local_gpu": rank % 8,
            }
            for rank in range(16)
        },
        "cpu_binding": "FLEXSIM_CLUSTER_CPU_AFFINITY=1",
        "gpu_binding": "torchrun LOCAL_RANK 0..7 on CUDA_VISIBLE_DEVICES=0..7 per node",
        "evaluator_nodes": {
            "coordinator": os.uname().nodename,
            "peer": args.peer_target,
        },
        "socket_interface": args.socket_ifname,
        "shared_gpfs": False,
        "coordination": "hash-identical staged source on each node; peer trace ranks 8..15 streamed to the coordinator",
    }


def source_maps_from_manifest(path: Path) -> tuple[dict[str, Path], dict[int, dict[str, Path]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    anchors = {key: Path(value["path"]) for key, value in payload["anchor_sources"].items()}
    candidates = {
        int(row["number"]): {key: Path(value["path"]) for key, value in row["sources"].items()}
        for row in payload["rounds"]
    }
    for mapping in (anchors, *candidates.values()):
        for path_value in mapping.values():
            require(path_value.is_file(), f"frozen source partition is missing: {path_value}")
    return anchors, candidates


class Ports:
    def __init__(self, first: int) -> None:
        self.value = first

    def take(self) -> int:
        port = self.value
        self.value += 1
        require(self.value < 65000, "master port range exhausted")
        return port


def case_paths(
    args: argparse.Namespace,
    relative: Path,
) -> tuple[Path, Path]:
    return args.out_dir / relative, args.peer_work_root / relative


def run_gate_a(args: argparse.Namespace) -> tuple[dict[str, object], float, int]:
    root = args.out_dir / "gate_a"
    anchor_sources, candidate_sources = write_sources(root, "gpt", GPT_ANCHOR, GPT_ROUNDS)
    manifest = write_manifest(
        root,
        "gpt",
        GPT_ANCHOR,
        GPT_ROUNDS,
        anchor_sources,
        candidate_sources,
    )
    anchor_spec = workload_spec("gpt", GPT_ANCHOR, anchor_sources, "gpt_gate_a_anchor")
    candidate_spec = workload_spec(
        "gpt",
        GPT_ANCHOR,
        candidate_sources[1],
        "gpt_gate_a_r1",
    )
    ports = Ports(args.master_port_base)
    anchor_local, anchor_peer = case_paths(args, Path("gate_a/gpt/anchor"))
    anchor, anchor_row = initialize_fresh_anchor(
        args,
        workload="gpt",
        config=GPT_ANCHOR,
        spec=anchor_spec,
        case_dir=anchor_local,
        peer_case_dir=anchor_peer,
        seed=SEED_BASE,
        port=ports.take(),
        peer_run_id=f"{args.run_id}-ga-anchor",
    )
    maya_local, maya_peer = case_paths(args, Path("gate_a/gpt/maya_r1"))
    maya = evaluate_candidate(
        args,
        workload="gpt",
        config=GPT_ANCHOR,
        round_=GPT_ROUNDS[0],
        system="maya-full",
        anchor=anchor,
        candidate_spec=candidate_spec,
        case_dir=maya_local,
        peer_case_dir=maya_peer,
        seed=SEED_BASE,
        port=ports.take(),
        peer_run_id=f"{args.run_id}-ga-maya-r1",
    )
    flex_local, flex_peer = case_paths(args, Path("gate_a/gpt/flex_r1"))
    flex = evaluate_candidate(
        args,
        workload="gpt",
        config=GPT_ANCHOR,
        round_=GPT_ROUNDS[0],
        system="flexeva-selective",
        anchor=anchor,
        candidate_spec=candidate_spec,
        case_dir=flex_local,
        peer_case_dir=flex_peer,
        seed=SEED_BASE,
        port=ports.take(),
        peer_run_id=f"{args.run_id}-ga-flex-r1",
    )
    check = correctness("gpt", maya, flex)
    require(check["pass"], f"Gate A correctness failed: {check}")
    result = {
        "schema": "flexeva.figure6.gate-a.v1",
        "gate": "A",
        "pass": True,
        "candidate_manifest": str(manifest),
        "candidate_manifest_sha256": sha256(manifest),
        "topology": topology_payload(args),
        "anchor": anchor_row,
        "maya": maya,
        "flexeva": flex,
        "correctness": check,
        "contract": {
            "marker_timing": False,
            "projection": False,
            "synthetic_timing": False,
            "target_shaped_timing": False,
            "full_rerun_fallback": False,
        },
    }
    atomic_json(root / "result.json", result)
    return result, float(maya["timing"]["direct_wall_s"]), int(maya["trace"]["raw_event_count"])


def calibrate_moe(
    args: argparse.Namespace,
    *,
    gpt_wall_s: float,
    gpt_events: int,
) -> Path:
    root = args.out_dir / "calibration"
    root.mkdir(parents=True, exist_ok=False)
    probes: list[dict[str, object]] = []
    config = MoeConfig()
    ports = Ports(args.master_port_base + 20)
    for index in range(1, 7):
        source_root = root / f"probe_{index:02d}_sources"
        anchor_sources, _ = write_sources(source_root, "moe", config, MOE_ROUNDS)
        spec = workload_spec("moe", config, anchor_sources, f"moe_calibration_{index:02d}")
        local, peer = case_paths(args, Path(f"calibration/probe_{index:02d}"))
        _, row = initialize_fresh_anchor(
            args,
            workload="moe",
            config=config,
            spec=spec,
            case_dir=local,
            peer_case_dir=peer,
            seed=SEED_BASE,
            port=ports.take(),
            peer_run_id=f"{args.run_id}-cal-{index:02d}",
        )
        wall_s = float(row["timing"]["direct_wall_s"])
        events = int(row["trace"]["raw_event_count"])
        probe = {
            "probe": index,
            "config": asdict(config),
            "maya_full_direct_wall_s": wall_s,
            "raw_event_count": events,
            "time_ratio_to_gpt": wall_s / gpt_wall_s,
            "event_ratio_to_gpt": events / max(gpt_events, 1),
            "flexeva_executed": False,
            "row": row,
        }
        probes.append(probe)
        atomic_json(root / "calibration_progress.json", {"complete": False, "probes": probes})
        if 0.8 <= probe["time_ratio_to_gpt"] <= 1.25 and 0.5 <= probe["event_ratio_to_gpt"] <= 2.0:
            break
        if probe["time_ratio_to_gpt"] >= 1.25:
            break
        scale = min(4.0, max(1.25, 0.9 / max(float(probe["time_ratio_to_gpt"]), 1e-6)))
        next_layers = int(math.ceil(config.num_layers * scale / 8.0) * 8)
        config = replace(config, num_layers=max(next_layers, config.num_layers + 8))
    satisfying = [
        row
        for row in probes
        if 0.8 <= row["time_ratio_to_gpt"] <= 1.25 and 0.5 <= row["event_ratio_to_gpt"] <= 2.0
    ]
    selected = min(
        satisfying or probes,
        key=(
            (lambda row: (row["config"]["num_layers"] * row["config"]["micro_batches"], row["probe"]))
            if satisfying
            else (lambda row: (abs(row["time_ratio_to_gpt"] - 1.0), row["probe"]))
        ),
    )
    frozen_config = MoeConfig(**selected["config"])
    frozen_root = root / "frozen"
    anchor_sources, candidates = write_sources(frozen_root, "moe", frozen_config, MOE_ROUNDS)
    candidate_manifest = write_manifest(
        frozen_root,
        "moe",
        frozen_config,
        MOE_ROUNDS,
        anchor_sources,
        candidates,
    )
    frozen_at = datetime.now(UTC).isoformat()
    payload = {
        "schema": "flexeva.figure6.moe-calibration.v1",
        "frozen": True,
        "frozen_at": frozen_at,
        "selection_rule": {
            "primary": "0.8 <= T_moe / T_gpt <= 1.25",
            "secondary": "0.5 <= MoE_events / GPT_events <= 2.0",
            "maximum_probes": 6,
            "tie_break": "smallest layers * micro_batches",
            "fallback": "closest Maya-full time after six probes",
        },
        "gpt_reference": {"maya_full_direct_wall_s": gpt_wall_s, "raw_event_count": gpt_events},
        "probes": probes,
        "selected_probe": selected["probe"],
        "selected_config": selected["config"],
        "selected_time_ratio_to_gpt": selected["time_ratio_to_gpt"],
        "selected_event_ratio_to_gpt": selected["event_ratio_to_gpt"],
        "selection_satisfied_both_ranges": bool(selected in satisfying),
        "flexeva_observed_before_freeze": False,
        "candidate_manifest": str(candidate_manifest),
        "candidate_manifest_sha256": sha256(candidate_manifest),
    }
    path = root / "moe_calibration_frozen.json"
    atomic_json(path, payload)
    return path


def expected_candidate_partitions(
    workload: str,
    config: object,
    round_: Round,
) -> tuple[str, ...]:
    if workload == "gpt":
        ops = GPT_OPS if round_.topology_change else round_.changed_ops
        return tuple(stage_partition(stage, op) for stage in range(config.pp) for op in ops)
    return tuple(round_.changed_ops)


def evaluate_maya_fresh(
    args: argparse.Namespace,
    *,
    workload: str,
    config: object,
    round_: Round,
    candidate_spec: fm.FlexMayaWorkloadSpec,
    case_dir: Path,
    peer_case_dir: Path,
    seed: int,
    port: int,
    peer_run_id: str,
) -> dict[str, object]:
    selected = expected_candidate_partitions(workload, config, round_)
    worker = spawn_worker(
        args,
        workload=workload,
        system="maya-full",
        config=config,
        round_=round_,
        selected_ops=(),
        seed=seed,
        case_dir=case_dir,
        peer_case_dir=peer_case_dir,
        port=port,
        peer_run_id=peer_run_id,
    )
    try:
        wait_ready(worker, args.ready_timeout_s)
        started_ns = time.time_ns()
        started = time.perf_counter()
        row, _, _ = process_full_worker(args, worker, workload, config, selected)
        row.update(
            {
                "round": round_.number,
                "round_name": round_.name,
                "mutation_label": round_.label,
                "candidate_config": config_payload(workload, config),
                "candidate_manifest_spec": asdict(candidate_spec),
                "seed": seed,
                "route_experts": list(round_.route_experts),
                "candidate_workers": 1,
                "logical_ranks": config.world_size,
                "source_ras_plan": None,
                "executed_code_partitions": list(selected),
            }
        )
        finish_timing(row, worker, started_ns, started)
        return row
    except BaseException:
        terminate(worker)
        raise


def cumulative_rows(
    maya_rows: list[dict[str, object]],
    flex_rows: list[dict[str, object]],
    anchor_row: dict[str, object],
) -> list[dict[str, object]]:
    baseline = float(maya_rows[0]["timing"]["direct_wall_s"])
    maya_cumulative = 0.0
    flex_cumulative = float(anchor_row["timing"]["direct_wall_s"])
    output = []
    for maya, flex in zip(maya_rows, flex_rows, strict=True):
        maya_cumulative += float(maya["timing"]["direct_wall_s"])
        flex_cumulative += float(flex["timing"]["direct_wall_s"])
        output.append(
            {
                "round": maya["round"],
                "x_label": maya["mutation_label"],
                "maya_full_s": maya["timing"]["direct_wall_s"],
                "flexeva_refresh_s": flex["timing"]["direct_wall_s"],
                "flexeva_anchor_s": anchor_row["timing"]["direct_wall_s"] if maya["round"] == 1 else 0.0,
                "maya_cumulative_s": maya_cumulative,
                "flexeva_cumulative_s": flex_cumulative,
                "maya_normalized": maya_cumulative / baseline,
                "flexeva_normalized": flex_cumulative / baseline,
                "normalization_denominator_maya_r1_s": baseline,
            }
        )
    return output


def summarize_single_run(repetitions: list[dict[str, object]]) -> list[dict[str, object]]:
    require(len(repetitions) == FORMAL_REPEATS == 1, "single-run summary requires one formal run")
    summary = []
    for round_number in range(1, 5):
        for system, field in (("Maya full", "maya_normalized"), ("FlexEva", "flexeva_normalized")):
            value = float(repetitions[0]["cumulative"][round_number - 1][field])
            summary.append(
                {
                    "round": round_number,
                    "x_label": repetitions[0]["cumulative"][round_number - 1]["x_label"],
                    "system": system,
                    "value": value,
                }
            )
    return summary


def run_workload_repeats(
    args: argparse.Namespace,
    *,
    workload: str,
    anchor_config: object,
    rounds: tuple[Round, ...],
    anchor_sources: dict[str, Path],
    candidate_sources: dict[int, dict[str, Path]],
    ports: Ports,
) -> dict[str, object]:
    repetitions = []
    for repeat in range(1, FORMAL_REPEATS + 1):
        seed = SEED_BASE + repeat
        order = ("maya-full", "flexeva-selective")
        anchor_spec = workload_spec(
            workload,
            anchor_config,
            anchor_sources,
            f"{workload}_repeat_{repeat:02d}_anchor",
        )
        anchor = None
        anchor_row = None
        maya_rows: list[dict[str, object]] = []
        flex_rows: list[dict[str, object]] = []
        for system in order:
            if system == "flexeva-selective":
                local, peer = case_paths(args, Path(f"formal/{workload}/repeat_{repeat:02d}/flexeva_anchor"))
                anchor, anchor_row = initialize_fresh_anchor(
                    args,
                    workload=workload,
                    config=anchor_config,
                    spec=anchor_spec,
                    case_dir=local,
                    peer_case_dir=peer,
                    seed=seed,
                    port=ports.take(),
                    peer_run_id=f"{args.run_id}-{workload}-r{repeat}-anchor",
                )
            for round_ in rounds:
                config = gpt_case(round_) if workload == "gpt" else anchor_config
                candidate_spec = workload_spec(
                    workload,
                    config,
                    candidate_sources[round_.number],
                    f"{workload}_repeat_{repeat:02d}_{round_.name}",
                )
                relative = Path(
                    f"formal/{workload}/repeat_{repeat:02d}/{system}/round_{round_.number:02d}"
                )
                local, peer = case_paths(args, relative)
                if system == "maya-full":
                    row = evaluate_maya_fresh(
                        args,
                        workload=workload,
                        config=config,
                        round_=round_,
                        candidate_spec=candidate_spec,
                        case_dir=local,
                        peer_case_dir=peer,
                        seed=seed,
                        port=ports.take(),
                        peer_run_id=f"{args.run_id}-{workload}-r{repeat}-m{round_.number}",
                    )
                    maya_rows.append(row)
                else:
                    require(anchor is not None, "FlexEva anchor was not initialized")
                    row = evaluate_candidate(
                        args,
                        workload=workload,
                        config=config,
                        round_=round_,
                        system="flexeva-selective",
                        anchor=anchor,
                        candidate_spec=candidate_spec,
                        case_dir=local,
                        peer_case_dir=peer,
                        seed=seed,
                        port=ports.take(),
                        peer_run_id=f"{args.run_id}-{workload}-r{repeat}-f{round_.number}",
                    )
                    flex_rows.append(row)
                atomic_json(
                    args.out_dir / "formal_progress.json",
                    {
                        "complete": False,
                        "workload": workload,
                        "repeat": repeat,
                        "system": system,
                        "round": round_.number,
                        "interrupted_results_must_not_be_stitched": True,
                    },
                )
        require(anchor_row is not None, "missing repeat anchor row")
        maya_rows.sort(key=lambda row: int(row["round"]))
        flex_rows.sort(key=lambda row: int(row["round"]))
        checks = [correctness(workload, maya, flex) for maya, flex in zip(maya_rows, flex_rows, strict=True)]
        require(all(check["pass"] for check in checks), f"{workload} repeat {repeat} equivalence failed: {checks}")
        repetitions.append(
            {
                "repeat": repeat,
                "execution_order": list(order),
                "seed": seed,
                "anchor": anchor_row,
                "maya": maya_rows,
                "flexeva": flex_rows,
                "correctness": checks,
                "cumulative": cumulative_rows(maya_rows, flex_rows, anchor_row),
                "interrupted": False,
                "stitched": False,
            }
        )
    return {"repetitions": repetitions, "summary": summarize_single_run(repetitions)}


def validate_frozen_calibration(path: Path) -> tuple[dict[str, object], Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload["schema"] == "flexeva.figure6.moe-calibration.v1", "wrong calibration schema")
    require(payload["frozen"] is True, "MoE calibration is not frozen")
    require(payload["flexeva_observed_before_freeze"] is False, "MoE scale used FlexEva results")
    manifest = Path(payload["candidate_manifest"])
    require(manifest.is_file(), "frozen MoE candidate manifest is missing")
    require(sha256(manifest) == payload["candidate_manifest_sha256"], "frozen MoE manifest hash differs")
    return payload, manifest


def run_formal(args: argparse.Namespace) -> Path:
    require(args.calibration_manifest is not None, "formal run requires --calibration-manifest")
    calibration, moe_manifest = validate_frozen_calibration(args.calibration_manifest)
    gate_result_path = args.calibration_manifest.parent.parent / "gate_a" / "result.json"
    require(gate_result_path.is_file(), "Gate A result is missing")
    gate_result = json.loads(gate_result_path.read_text(encoding="utf-8"))
    require(gate_result["pass"] is True, "Gate A did not pass")
    require(not args.out_dir.exists(), f"output directory already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    calibration_copy = args.out_dir / "moe_calibration_frozen.json"
    calibration_copy.write_bytes(args.calibration_manifest.read_bytes())
    gpt_sources, gpt_candidates = write_sources(args.out_dir / "manifests", "gpt", GPT_ANCHOR, GPT_ROUNDS)
    gpt_manifest = write_manifest(
        args.out_dir / "manifests",
        "gpt",
        GPT_ANCHOR,
        GPT_ROUNDS,
        gpt_sources,
        gpt_candidates,
    )
    moe_sources, moe_candidates = source_maps_from_manifest(moe_manifest)
    moe_config = MoeConfig(**calibration["selected_config"])
    ports = Ports(args.master_port_base + 100)
    gpt = run_workload_repeats(
        args,
        workload="gpt",
        anchor_config=GPT_ANCHOR,
        rounds=GPT_ROUNDS,
        anchor_sources=gpt_sources,
        candidate_sources=gpt_candidates,
        ports=ports,
    )
    moe = run_workload_repeats(
        args,
        workload="moe",
        anchor_config=moe_config,
        rounds=MOE_ROUNDS,
        anchor_sources=moe_sources,
        candidate_sources=moe_candidates,
        ports=ports,
    )
    repo_root = args.repo_root.resolve()
    result = {
        "schema": "flexeva.figure6.production.v1",
        "complete": True,
        "run_id": args.run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "timing_ledger_id": f"{args.run_id}-figure6",
        "gate_a_result": str(gate_result_path),
        "gate_a_result_sha256": sha256(gate_result_path),
        "moe_calibration_manifest": str(calibration_copy),
        "moe_calibration_manifest_sha256": sha256(calibration_copy),
        "moe_calibration": calibration,
        "candidate_manifests": {
            "gpt": {"path": str(gpt_manifest), "sha256": sha256(gpt_manifest)},
            "moe": {"path": str(moe_manifest), "sha256": sha256(moe_manifest)},
        },
        "commits": {
            "ae": git_revision(repo_root),
            "core": git_revision(repo_root / "FlexEva"),
            "peer_staged_from_same_revisions": True,
        },
        "topology": topology_payload(args),
        "candidate_workers": 1,
        "configurations": {
            "gpt_anchor": config_payload("gpt", GPT_ANCHOR),
            "gpt_r4": config_payload("gpt", GPT_R4),
            "moe": config_payload("moe", moe_config),
        },
        "rounds": {
            "gpt": [asdict(round_) for round_ in GPT_ROUNDS],
            "moe": [asdict(round_) for round_ in MOE_ROUNDS],
        },
        "contract": {
            "formal_repeat_count": FORMAL_REPEATS,
            "cross_repeat_aggregation": False,
            "direct_end_to_end_wall_primary": True,
            "marker_timing": False,
            "projection": False,
            "synthetic_timing": False,
            "target_shaped_timing": False,
            "full_rerun_fallback": False,
            "aggregate_ratio_primary": False,
            "interrupted_round_stitching": False,
            "production_binary_trace": True,
        },
        "workloads": {"gpt": gpt, "moe": moe},
    }
    path = args.out_dir / "result.json"
    atomic_json(path, result)
    return path


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    require(rows, f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure6_output_rows(
    result: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    summary: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    plot_rows: list[dict[str, object]] = []
    for workload in ("gpt", "moe"):
        repetition = result["workloads"][workload]["repetitions"][0]
        anchor = repetition["anchor"]
        phase_rows.append(
            {
                "workload": workload,
                "round": 0,
                "system": "FlexEva-anchor",
                **{phase: float(anchor["phases_s"][phase]) for phase in PHASES},
                "direct_wall_s": float(anchor["timing"]["direct_wall_s"]),
            }
        )
        for maya, flex, cumulative in zip(
            repetition["maya"], repetition["flexeva"], repetition["cumulative"], strict=True
        ):
            summary.append(
                {
                    "workload": workload,
                    "round": int(cumulative["round"]),
                    "x_label": cumulative["x_label"],
                    "maya_full_s": float(cumulative["maya_full_s"]),
                    "flexeva_refresh_s": float(cumulative["flexeva_refresh_s"]),
                    "maya_cumulative_s": float(cumulative["maya_cumulative_s"]),
                    "flexeva_cumulative_s": float(cumulative["flexeva_cumulative_s"]),
                    "maya_normalized": float(cumulative["maya_normalized"]),
                    "flexeva_normalized": float(cumulative["flexeva_normalized"]),
                    "normalization_denominator_maya_r1_s": float(
                        cumulative["normalization_denominator_maya_r1_s"]
                    ),
                }
            )
            for system, normalized in (
                ("Maya", cumulative["maya_normalized"]),
                ("FlexEva", cumulative["flexeva_normalized"]),
            ):
                plot_rows.append(
                    {
                        "workload": workload,
                        "round": int(cumulative["round"]),
                        "x_label": cumulative["x_label"],
                        "system": system,
                        "normalized": float(normalized),
                    }
                )
            for system, row in (("Maya-full", maya), ("FlexEva", flex)):
                phase_rows.append(
                    {
                        "workload": workload,
                        "round": int(row["round"]),
                        "system": system,
                        **{phase: float(row["phases_s"][phase]) for phase in PHASES},
                        "direct_wall_s": float(row["timing"]["direct_wall_s"]),
                    }
                )
    return summary, phase_rows, plot_rows


def render_figure6(plot_data: list[dict[str, object]], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), constrained_layout=True)
    colors = {"Maya": "#4C78A8", "FlexEva": "#F58518"}
    for axis, workload, title in zip(
        axes,
        ("gpt", "moe"),
        ("(a) GPT 18.4B", "(b) Runtime-matched Routed-MoE"),
        strict=True,
    ):
        rows = [row for row in plot_data if row["workload"] == workload]
        for system in ("Maya", "FlexEva"):
            series = sorted(
                (row for row in rows if row["system"] == system),
                key=lambda row: int(row["round"]),
            )
            axis.plot(
                [int(row["round"]) for row in series],
                [float(row["normalized"]) for row in series],
                marker="o" if system == "Maya" else "s",
                linewidth=2,
                color=colors[system],
                label=system,
            )
        labels = [
            row["x_label"].replace("+", "+\n")
            for row in sorted(
                (row for row in rows if row["system"] == "Maya"),
                key=lambda row: int(row["round"]),
            )
        ]
        axis.set_xticks(range(1, 5), labels)
        axis.set_title(title)
        axis.set_ylabel("Cumulative evaluator time\n(normalized to Maya R1)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    outputs = [out_dir / "figure6.pdf", out_dir / "figure6.png"]
    for path in outputs:
        figure.savefig(path, dpi=240)
    plt.close(figure)
    return outputs


def write_figure6_outputs(result_path: Path, out_dir: Path) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    require(result["complete"] is True, "cannot plot an incomplete run")
    summary, phases, plot_data = figure6_output_rows(result)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "phase_breakdown.csv", phases)
    write_csv(out_dir / "figure6_plot.csv", plot_data)
    render_figure6(plot_data, out_dir)

def verify_timing(row: dict[str, object]) -> None:
    timing = row["timing"]
    phases = row["phases_s"]
    require(set(phases) == set(PHASES), "phase ledger fields differ")
    require(all(float(value) >= 0.0 for value in phases.values()), "negative phase time")
    if row["system"] == "maya-full":
        require(
            float(phases["trace_ras_compaction_s"]) == 0.0,
            "Maya full includes FlexEva trace-RAS compaction",
        )
    direct = float(timing["direct_wall_s"])
    require(direct > 0.0, "non-positive direct wall time")
    require(
        abs(sum(float(value) for value in phases.values()) - direct)
        <= max(1e-6, direct * 1e-6),
        "phase ledger does not reconcile to direct wall",
    )
    require(timing["source"].startswith("direct monotonic wall clock"), "non-direct timing source")
    require(
        timing["started_ns"]
        <= timing["dispatch_ns"]
        <= timing["process_exit_ns"]
        <= timing["finished_ns"],
        "invalid evaluator timestamps",
    )
    require(
        int(row["peak_rss_kib"]["node0_process_tree_kib"]) > 0
        and int(row["peak_rss_kib"]["node1_command_max_kib"]) > 0,
        "two-node peak RSS evidence is incomplete",
    )


def verify_binary_trace(row: dict[str, object]) -> None:
    audit = row["trace"]
    require(audit["trace_source"] == "SharedEventArena", "formal model did not use SharedEventArena")
    require(audit["binary_file_count"] == 2, "compact trace artifact count differs")
    require(len(audit["binary_paths"]) == 2, "compact trace paths are missing")
    require(audit.get("binary_paths_distinct") is True, "compact trace paths are not distinct")
    require(
        audit.get("rank_coverage") == list(range(int(audit.get("world_size", 0)))),
        "compact traces do not cover the complete rank set",
    )
    require(int(audit["raw_event_count"]) > 0, "empty compact trace")
    require(int(audit["cudaGetDevice_modeled_count"]) == 0, "cudaGetDevice entered modeled input")
    require(all(Path(path).is_file() for path in row["logs"].values()), "candidate log is missing")


def verify_figure6_outputs(result: dict[str, object], result_path: Path) -> None:
    out_dir = result_path.parent
    for name in ("summary.csv", "phase_breakdown.csv", "figure6_plot.csv", "figure6.pdf", "figure6.png"):
        path = out_dir / name
        require(path.is_file() and path.stat().st_size > 0, f"Figure 6 artifact is missing: {path}")
    require(len((out_dir / "summary.csv").read_text(encoding="utf-8").splitlines()) == 9, "summary.csv row count differs")
    require(len((out_dir / "figure6_plot.csv").read_text(encoding="utf-8").splitlines()) == 17, "figure6_plot.csv row count differs")
    info = subprocess.run(["pdfinfo", str(out_dir / "figure6.pdf")], text=True, capture_output=True, check=False)
    require(info.returncode == 0 and "Pages:           1" in info.stdout, "invalid Figure 6 PDF")


def verify_result(path: Path, *, verify_artifacts: bool = True) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    require(result["schema"] == "flexeva.figure6.production.v1" and result["complete"] is True, "formal result is incomplete")
    contract = result["contract"]
    require(contract["production_binary_trace"] is True, "formal trace input contract differs")
    require(contract["formal_repeat_count"] == FORMAL_REPEATS == 1, "formal repeat count differs")
    require(contract["cross_repeat_aggregation"] is False, "cross-repeat aggregation is enabled")
    topology = result["topology"]
    require((topology["physical_gpus"], topology["logical_ranks"], topology["node_count"]) == (16, 16, 2), "formal topology is not 16 physical GPUs")
    require(result["candidate_workers"] == 1, "candidate workers differ")
    require(result["configurations"]["gpt_anchor"] == config_payload("gpt", GPT_ANCHOR), "GPT anchor differs")
    require(result["configurations"]["gpt_r4"] == config_payload("gpt", GPT_R4), "GPT R4 differs")
    require(result["configurations"]["gpt_r4"]["effective_global_batch"] == 512, "GPT R4 batch differs")
    require(result["configurations"]["gpt_r4"]["micro_batches"] == 256, "GPT R4 micro-batches differ")
    for forbidden in ("marker_timing", "projection", "synthetic_timing", "target_shaped_timing", "full_rerun_fallback", "aggregate_ratio_primary", "interrupted_round_stitching"):
        require(contract[forbidden] is False, f"forbidden method enabled: {forbidden}")
    require(contract["direct_end_to_end_wall_primary"] is True, "direct wall is not primary")
    for item in result["candidate_manifests"].values():
        require(sha256(Path(item["path"])) == item["sha256"], "candidate manifest hash differs")
    gate_path = Path(result["gate_a_result"])
    require(sha256(gate_path) == result["gate_a_result_sha256"], "Gate A result hash differs")
    verify_gate_a(gate_path)
    calibration_path = Path(result["moe_calibration_manifest"])
    require(sha256(calibration_path) == result["moe_calibration_manifest_sha256"], "calibration hash differs")
    calibration, _ = validate_frozen_calibration(calibration_path)
    require(result["configurations"]["moe"] == config_payload("moe", MoeConfig(**calibration["selected_config"])), "formal MoE differs from frozen calibration")
    for workload in ("gpt", "moe"):
        section = result["workloads"][workload]
        repetitions = section["repetitions"]
        require(len(repetitions) == 1 and repetitions[0]["repeat"] == 1 and repetitions[0]["execution_order"] == ["maya-full", "flexeva-selective"], "formal run count/order differs")
        repetition = repetitions[0]
        verify_timing(repetition["anchor"])
        verify_binary_trace(repetition["anchor"])
        require(len(repetition["maya"]) == len(repetition["flexeva"]) == 4, "Figure 6 round count differs")
        require(len(repetition["correctness"]) == 4 and all(row["pass"] for row in repetition["correctness"]), "correctness round gate failed")
        for index, (maya, flex, check) in enumerate(zip(repetition["maya"], repetition["flexeva"], repetition["correctness"], strict=True), 1):
            require(maya["round"] == flex["round"] == index, "round order differs")
            require(maya["candidate_config"] == flex["candidate_config"], "Maya/Flex config differs")
            expected = result["configurations"]["gpt_anchor" if index < 4 else "gpt_r4"] if workload == "gpt" else result["configurations"]["moe"]
            require(maya["candidate_config"] == expected, "candidate setting differs")
            verify_timing(maya)
            verify_timing(flex)
            verify_binary_trace(maya)
            verify_binary_trace(flex)
            require(all(float(maya["phases_s"][phase]) == 0.0 for phase in ("trace_ras_compaction_s", "code_analysis_s", "source_ras_update_s", "grounding_s", "selective_emulation_s", "trace_patch_collation_s")), "Maya-full phase gate failed")
            maya_executor = maya["executor_feedback"]
            flex_executor = flex["executor_feedback"]
            require(maya_executor["full_training_step_executed"] and maya_executor["full_candidate_trace_generated"], "Maya path was shortened")
            require(not flex_executor["full_training_step_executed"] and not flex_executor["full_candidate_trace_generated"] and flex_executor["fallback"] is False, "FlexEva full-rerun masquerade")
            require(flex["anchor_patch"]["full_candidate_trace_generated"] is False, "FlexEva generated a full candidate trace")
            require(flex["source_analysis_window"]["precedes_dispatch"] is True, "source analysis followed execution")
            require(check["pass"] is True and all(check["checks"].values()), "correctness check failed")
            if workload == "gpt" and index == 4:
                require(flex["source_ras_plan"]["configuration_changed"] is True, "GPT R4 topology change was not analyzed")
                require(flex["anchor_patch"]["mode"] == "configuration_rebase", "GPT R4 topology patch mode differs")
                require(flex["anchor_patch"]["full_executable_partition_coverage"] is True, "GPT R4 patch coverage is partial")
            else:
                require(flex["anchor_patch"]["mode"] == "chunk_patch", "non-topology patch mode differs")
    require(result["workloads"]["gpt"]["summary"] == summarize_single_run(result["workloads"]["gpt"]["repetitions"]), "GPT summary differs")
    require(result["workloads"]["moe"]["summary"] == summarize_single_run(result["workloads"]["moe"]["repetitions"]), "MoE summary differs")
    if verify_artifacts:
        verify_figure6_outputs(result, path)
    print("Figure 6 production verification: PASS")


def verify_gate_a(path: Path) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    require(result["schema"] == "flexeva.figure6.gate-a.v1" and result["pass"] is True, "Gate A is incomplete")
    require(result["topology"]["physical_gpus"] == 16 and result["topology"]["node_count"] == 2, "Gate A is not two-node/16-physical-GPU")
    require(sha256(Path(result["candidate_manifest"])) == result["candidate_manifest_sha256"], "Gate A manifest hash differs")
    require(result["correctness"]["pass"] is True, "Gate A correctness failed")
    for forbidden, value in result["contract"].items():
        require(value is False, f"Gate A forbidden method enabled: {forbidden}")
    for row in (result["anchor"], result["maya"], result["flexeva"]):
        verify_timing(row)
        verify_binary_trace(row)
    require(result["maya"]["executor_feedback"]["full_training_step_executed"] and result["maya"]["executor_feedback"]["full_candidate_trace_generated"], "Gate A Maya is not full")
    require(not result["flexeva"]["executor_feedback"]["full_training_step_executed"] and not result["flexeva"]["executor_feedback"]["full_candidate_trace_generated"], "Gate A FlexEva is not selective")
    require(result["flexeva"]["fallback"] is False, "Gate A FlexEva fell back")
    require(result["flexeva"]["source_analysis_window"]["precedes_dispatch"] is True, "Gate A analysis followed execution")

def self_test() -> None:
    require(effective_gpt_batch(GPT_ANCHOR) == 512, "GPT anchor batch self-test")
    require(effective_gpt_batch(GPT_R4) == 512 and GPT_R4.micro_batches == 256, "GPT R4 batch self-test")
    require(base_executor_ops("gpt", ("stage_000_attention_backward",)) == ("attention_backward",), "GPT partition mapping self-test")
    require(base_executor_ops("moe", ("router_backward", "route_path")) == ("router_backward", "route_path"), "MoE partition mapping self-test")
    counter = Counter({("api", "kernel_launch", 0, 0, 0, 1, -1, "", "part", False): 2})
    require(signature_payload(counter)["event_count"] == 2, "signature self-test")
    require(
        "kernel_launch" in KEY_EVENT_KINDS and "context_op" not in KEY_EVENT_KINDS,
        "key-event scope self-test",
    )
    trace = fm.build_trace_ras(
        [
            fm.make_event("cudaGetDevice", "context_op", code_partition="part"),
            fm.make_event("kernel", "kernel_launch", code_partition="part"),
        ]
    )
    require(trace_key_signature(trace)["event_count"] == 1, "key-event filtering self-test")
    print("Figure 6 production driver self-test: PASS")


def require_runtime_args(args: argparse.Namespace) -> None:
    required = {
        "--maya-root": args.maya_root,
        "--proot": args.proot,
        "--repo-root": args.repo_root,
        "--peer-repo-root": args.peer_repo_root,
        "--peer-target": args.peer_target,
        "--peer-python": args.peer_python,
        "--peer-maya-root": args.peer_maya_root,
        "--peer-proot": args.peer_proot,
        "--peer-node-root": args.peer_node_root,
        "--peer-work-root": args.peer_work_root,
        "--master-addr": args.master_addr,
        "--run-id": args.run_id,
    }
    missing = [name for name, value in required.items() if value is None or value == ""]
    require(not missing, f"missing runtime arguments: {missing}")
    require(not args.out_dir.exists(), f"output directory already exists: {args.out_dir}")


def main() -> int:
    args = parse_args()
    if args.mode == "self-test":
        self_test()
        return 0
    result_path = args.result or args.out_dir / "result.json"
    if args.mode == "probe":
        require_runtime_args(args)
        args.out_dir.mkdir(parents=True)
        gate, gpt_wall_s, gpt_events = run_gate_a(args)
        verify_gate_a(args.out_dir / "gate_a" / "result.json")
        calibration = calibrate_moe(args, gpt_wall_s=gpt_wall_s, gpt_events=gpt_events)
        print(json.dumps({"gate_a": gate["pass"], "calibration_manifest": str(calibration)}))
        return 0
    if args.mode == "run":
        require_runtime_args(args)
        result_path = run_formal(args)
        write_figure6_outputs(result_path, args.out_dir)
        verify_result(result_path, verify_artifacts=False)
        print(result_path)
        return 0
    if args.mode == "report":
        write_figure6_outputs(result_path, args.out_dir)
    else:
        verify_result(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
