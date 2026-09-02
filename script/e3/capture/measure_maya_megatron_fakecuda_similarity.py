#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import socket
import struct
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import flexmaya_ras as fm

from measure_megatron_trace_similarity import (
    MegatronCase,
    cosine,
    default_cases,
    filtered_weighted_event_counter,
    trace_window_counter,
    weighted_edge_counter,
    weighted_event_counter,
    weighted_jaccard,
)


REPORTED_AUXILIARY_APIS = frozenset(
    {
        "cudaGetDevice",
        "cudaSetDevice",
        "cudaEventCreateWithFlags",
        "cudaEventQuery",
    }
)
MAYA_COMPATIBILITY_ONLY_APIS = REPORTED_AUXILIARY_APIS - {"cudaEventQuery"}
MODELED_KERNEL_NCCL_KINDS = frozenset({"kernel_launch", "blas_compute", "nccl_collective"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run workload/megatron/maya_megatron.py through fake-CUDA and compare Maya/FlexEva traces."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--maya-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--proot", type=Path, required=True)
    parser.add_argument(
        "--local-device-count",
        type=int,
        default=8,
        help="Map worker LOCAL_RANK to RANK %% this count for single-node fake-CUDA logical multi-node runs.",
    )
    parser.add_argument("--case", action="append", choices=[case.name for case in default_cases()])
    parser.add_argument("--keep-raw-traces", action="store_true")
    parser.add_argument(
        "--input-trace-root",
        type=Path,
        help="Read CASE/traces below this directory when reusing existing traces.",
    )
    parser.add_argument(
        "--reuse-existing-traces",
        action="store_true",
        help=(
            "Do not relaunch maya_megatron.py. Reuse rank_*.jsonl and "
            "rank_*_markers.jsonl from INPUT_TRACE_ROOT/CASE/traces, or from "
            "OUT_DIR/CASE/traces when no input root is supplied."
        ),
    )
    return parser.parse_args()


def dtype_bytes(dtype: str) -> int:
    if dtype == "bf16":
        return 2
    if dtype == "fp32":
        return 4
    return 4


def rank_to_pp_rank(rank: int, case: MegatronCase) -> int:
    return rank % case.pp


def stage_spec_for(case: MegatronCase) -> fm.FlexMayaWorkloadSpec:
    return fm.FlexMayaWorkloadSpec(
        workload_id=f"{case.name}_maya_megatron_fakecuda",
        world_size=case.world_size,
        tp=case.tp,
        pp=case.pp,
        dp=case.dp,
        code_partitions=tuple(
            fm.CodePartitionSpec(
                partition_id=f"stage_{stage:03d}",
                path=__file__,
                active_ranks=fm.megatron_pp_stage_active_ranks(case.world_size, case.tp, case.pp, stage),
            )
            for stage in range(case.pp)
        ),
        rank_group_policy="active_lane_set",
        notes=("workload/megatron/maya_megatron.py fake-CUDA capture",),
    )


def _connect_with_retry(address: str, port: int, timeout_s: float) -> socket.socket:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            stream = socket.create_connection((address, port), timeout=min(timeout_s, 10.0))
            stream.settimeout(timeout_s)
            return stream
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out connecting to {address}:{port}")
            time.sleep(0.2)


def _peer_barrier(address: str, port: int, node_rank: int, timeout_s: float) -> None:
    if node_rank == 0:
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((address, port))
            server.listen(1)
            server.settimeout(timeout_s)
            peer, _ = server.accept()
            with peer:
                peer.settimeout(timeout_s)
                peer.recv(1)
                peer.sendall(b"1")
    else:
        with _connect_with_retry(address, port, timeout_s) as peer:
            peer.sendall(b"1")
            peer.recv(1)


def _receive_exact(stream: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.recv(remaining)
        if not chunk:
            raise EOFError("peer trace transfer ended early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def transfer_remote_traces(
    trace_dir: Path,
    world_size: int,
    distributed: dict[str, object],
) -> dict[str, object]:
    if distributed["nnodes"] == 1:
        return {"files": 0, "bytes": 0, "elapsed_s": 0.0}
    address = str(distributed["master_addr"])
    port = int(distributed["control_port"]) + 1
    node_rank = int(distributed["node_rank"])
    timeout_s = float(os.environ.get("FLEXMAYA_PEER_WAIT_S", "7200"))
    started = time.perf_counter()
    files = bytes_transferred = 0
    if node_rank == 0:
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((address, port))
            server.listen(1)
            server.settimeout(timeout_s)
            peer, _ = server.accept()
            with peer:
                peer.settimeout(timeout_s)
                while True:
                    header_size = struct.unpack("!I", _receive_exact(peer, 4))[0]
                    if header_size == 0:
                        break
                    header = json.loads(_receive_exact(peer, header_size))
                    name = Path(str(header["name"])).name
                    size = int(header["size"])
                    with (trace_dir / name).open("wb") as output:
                        remaining = size
                        while remaining:
                            chunk = peer.recv(min(4 * 1024 * 1024, remaining))
                            if not chunk:
                                raise EOFError(f"peer trace {name} ended early")
                            output.write(chunk)
                            remaining -= len(chunk)
                    files += 1
                    bytes_transferred += size
    else:
        with _connect_with_retry(address, port, timeout_s) as peer:
            first_rank = node_rank * int(distributed["nproc_per_node"])
            last_rank = min(first_rank + int(distributed["nproc_per_node"]), world_size)
            for rank in range(first_rank, last_rank):
                for suffix in (".jsonl", "_markers.jsonl"):
                    path = trace_dir / f"rank_{rank}{suffix}"
                    size = path.stat().st_size
                    header = json.dumps({"name": path.name, "size": size}).encode("utf-8")
                    peer.sendall(struct.pack("!I", len(header)) + header)
                    with path.open("rb") as source:
                        while chunk := source.read(4 * 1024 * 1024):
                            peer.sendall(chunk)
                    files += 1
                    bytes_transferred += size
            peer.sendall(struct.pack("!I", 0))
    return {"files": files, "bytes": bytes_transferred, "elapsed_s": time.perf_counter() - started}


def distributed_torchrun_args(world_size: int) -> tuple[list[str], dict[str, object]]:
    nnodes = int(os.environ.get("FLEXMAYA_NNODES", "1"))
    node_rank = int(os.environ.get("FLEXMAYA_NODE_RANK", "0"))
    if nnodes < 1 or not 0 <= node_rank < nnodes or world_size % nnodes:
        raise ValueError(f"invalid distributed launch: world_size={world_size}, nnodes={nnodes}, node_rank={node_rank}")
    nproc_per_node = world_size // nnodes
    if nnodes == 1:
        port = random.randint(42000, 62000)
        return ["--standalone", "--nnodes=1", f"--nproc-per-node={nproc_per_node}", f"--master-port={port}"], {
            "nnodes": 1,
            "node_rank": 0,
            "nproc_per_node": nproc_per_node,
            "master_addr": "127.0.0.1",
            "master_port": port,
        }

    master_addr = os.environ.get("FLEXMAYA_MASTER_ADDR")
    master_port = os.environ.get("FLEXMAYA_MASTER_PORT")
    if not master_addr or not master_port:
        raise ValueError("FLEXMAYA_MASTER_ADDR and FLEXMAYA_MASTER_PORT are required for multi-node launch")
    control_port = int(os.environ.get("FLEXMAYA_CONTROL_PORT", "0"))
    if not control_port:
        raise ValueError("FLEXMAYA_CONTROL_PORT is required for multi-node launch")
    _peer_barrier(master_addr, control_port, node_rank, float(os.environ.get("FLEXMAYA_PEER_WAIT_S", "7200")))
    return [
        f"--nnodes={nnodes}",
        f"--node-rank={node_rank}",
        f"--nproc-per-node={nproc_per_node}",
        f"--master-addr={master_addr}",
        f"--master-port={master_port}",
    ], {
        "nnodes": nnodes,
        "node_rank": node_rank,
        "nproc_per_node": nproc_per_node,
        "master_addr": master_addr,
        "master_port": int(master_port),
        "control_port": control_port,
    }


def run_maya_megatron_case(args: argparse.Namespace, case: MegatronCase, case_dir: Path) -> dict[str, object]:
    input_trace_root = getattr(args, "input_trace_root", None)
    output_trace_root = os.environ.get("FLEXMAYA_TRACE_ROOT", "").strip()
    trace_dir = (
        input_trace_root / case.name / "traces"
        if input_trace_root
        else (Path(output_trace_root) / case_dir.name / "traces" if output_trace_root else case_dir / "traces")
    )
    if args.reuse_existing_traces:
        missing = []
        for rank in range(case.world_size):
            for suffix in (".jsonl", "_markers.jsonl"):
                path = trace_dir / f"rank_{rank}{suffix}"
                if not path.exists():
                    missing.append(str(path))
        if missing:
            raise FileNotFoundError(
                "missing existing trace files for --reuse-existing-traces: "
                + ", ".join(missing[:8])
                + (" ..." if len(missing) > 8 else "")
            )
        return {
            "command": [],
            "return_code": 0,
            "elapsed_s": 0.0,
            "stdout": str(case_dir / "stdout.txt"),
            "stderr": str(case_dir / "stderr.txt"),
            "trace_dir": str(trace_dir),
            "reused_existing_traces": True,
        }

    trace_dir.mkdir(parents=True, exist_ok=True)

    wrapper = Path(__file__).resolve().parent / "maya_megatron_trace_worker.py"
    workload = Path(__file__).resolve().parents[1] / "workload" / "megatron" / "maya_megatron.py"
    frun = args.maya_root / "fake-cuda" / "frun"
    env = os.environ.copy()
    env.pop("FAKECUDA_TRACE", None)
    env.pop("FAKECUDA_TRACE_PATH", None)
    env["FAKECUDA_PROOT_BIN"] = str(args.proot)
    env["FAKECUDA_FRUN_QUIET"] = "1"
    env["FLEXMAYA_TRACE_DIR"] = str(trace_dir)
    env["FLEXMAYA_LOCAL_DEVICE_COUNT"] = str(args.local_device_count)
    env["MAYA_MEGATRON_SCRIPT"] = str(workload)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(__file__).resolve().parents[1] / "workload" / "megatron"),
            str(args.maya_root / "python"),
            str(args.maya_root / "CppEvent"),
            str(Path(__file__).resolve().parent),
            env.get("PYTHONPATH", ""),
        ]
    )
    torchrun_args, distributed = distributed_torchrun_args(case.world_size)
    command = [
        str(frun),
        str(args.python),
        "-m",
        "torch.distributed.run",
        *torchrun_args,
        str(wrapper),
        "--steps",
        str(case.steps),
        "--global-batch-size",
        str(case.global_batch_size),
        "--seq-len",
        str(case.seq_len),
        "--hidden-size",
        str(case.hidden_size),
        "--num-layers",
        str(case.num_layers),
        "--num-heads",
        str(case.num_heads),
        "--vocab-size",
        str(case.vocab_size),
        "--tp",
        str(case.tp),
        "--pp",
        str(case.pp),
        "--dp",
        str(case.dp),
        "--micro-batches",
        str(case.micro_batches),
        "--schedule",
        case.schedule,
        "--dtype",
        case.dtype,
        "--warmup-steps",
        "0",
    ]
    if getattr(args, "sync_before_step_window", False):
        command.append("--sync-before-step-window")
    if getattr(args, "source_region_markers", False):
        command.append("--source-region-markers")
    suffix = f".node{distributed['node_rank']}" if distributed["nnodes"] != 1 else ""
    stdout = case_dir / f"stdout{suffix}.txt"
    stderr = case_dir / f"stderr{suffix}.txt"
    start = time.perf_counter()
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        completed = subprocess.run(command, env=env, cwd=case_dir, stdout=out, stderr=err, text=True)
    elapsed_s = time.perf_counter() - start
    transfer = transfer_remote_traces(trace_dir, case.world_size, distributed) if completed.returncode == 0 else None
    return {
        "command": command,
        "return_code": completed.returncode,
        "elapsed_s": elapsed_s,
        "stdout": str(stdout),
        "stderr": str(stderr),
        "trace_dir": str(trace_dir),
        "distributed": distributed,
        "peer_trace_transfer": transfer,
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


def raw_event_from_record(record: dict[str, object], *, case: MegatronCase, rank: int, event_id: int) -> object:
    kind = str(record.get("type", ""))
    api = str(record.get("api", ""))
    count = record_count(record)
    bytes_ = int(record.get("bytes") or 0)
    if bytes_ <= 0 and kind == "nccl_collective":
        bytes_ = count * dtype_bytes(case.dtype)
    stream = int(record.get("stream_id") or 0)
    pp_rank = rank_to_pp_rank(rank, case)
    collective_group = ""
    if kind == "nccl_collective":
        collective_group = f"{api}:call={int(record.get('call_idx') or 0)}"
    cuda_event_id = int(record.get("event_id") or 0)
    return fm.make_event(
        api,
        kind,
        rank=rank,
        thread_id=int(record.get("tid") or 0),
        stream=stream,
        correlation_id=event_id,
        timestamp_ns=int(record.get("ts") or 0) * 1000,
        duration_hint_us=float(record.get("host_duration_us") or 0.0),
        bytes=bytes_,
        count=count,
        peer_rank=int(record.get("peer") if record.get("peer") is not None else -1),
        event_handle=cuda_event_id if "EventRecord" in api else 0,
        wait_event_handle=cuda_event_id if "WaitEvent" in api else 0,
        collective_group=collective_group,
        code_partition=f"stage_{pp_rank:03d}",
        blocking=api.endswith("Synchronize"),
    )


FIGURE7_SOURCE_OPS = frozenset(
    {"attention_backward", "mlp_backward", "optimizer_step"}
)


def load_source_region_windows(
    markers_path: Path,
    step_start: int,
    step_end: int,
) -> list[tuple[int, int, str]]:
    stacks: dict[str, list[int]] = {label: [] for label in FIGURE7_SOURCE_OPS}
    windows: list[tuple[int, int, str]] = []
    with markers_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            label = str(record.get("label", ""))
            if label not in stacks or record.get("trace_ts") is None:
                continue
            trace_ts = int(record["trace_ts"])
            if record.get("kind") == "region_begin":
                stacks[label].append(trace_ts)
            elif record.get("kind") == "region_end" and stacks[label]:
                begin = stacks[label].pop()
                if trace_ts < step_start or begin > step_end:
                    continue
                windows.append(
                    (max(begin, step_start), min(trace_ts, step_end), label)
                )
    return sorted(windows)


def parse_case_raw_events_with_regions(
    case: MegatronCase,
    trace_dir: Path,
) -> list[object]:
    rows: list[object] = []
    next_id = 1
    for rank in range(case.world_size):
        markers_path = trace_dir / f"rank_{rank}_markers.jsonl"
        start_ts, end_ts = load_step_window(markers_path)
        windows = load_source_region_windows(markers_path, start_ts, end_ts)
        window_index = 0
        with (trace_dir / f"rank_{rank}.jsonl").open(
            "r", encoding="utf-8"
        ) as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                timestamp = int(record.get("ts") or 0)
                if not start_ts <= timestamp <= end_ts:
                    continue
                if str(record.get("type", "")) in {"marker", "other"}:
                    continue
                event = raw_event_from_record(
                    record, case=case, rank=rank, event_id=next_id
                )
                while window_index < len(windows) and timestamp > windows[window_index][1]:
                    window_index += 1
                if window_index < len(windows):
                    begin, end, label = windows[window_index]
                    if begin <= timestamp <= end:
                        event.code_partition = label
                rows.append(event)
                next_id += 1
    return rows


def parse_case_raw_events(
    case: MegatronCase,
    trace_dir: Path,
    hook_audit: dict[str, object] | None = None,
) -> list[object]:
    rows: list[object] = []
    next_id = 1
    window_api_counts: Counter[str] = Counter()
    modeled_api_counts: Counter[str] = Counter()
    file_hashes: dict[str, str] = {}
    for rank in range(case.world_size):
        markers_path = trace_dir / f"rank_{rank}_markers.jsonl"
        start_ts, end_ts = load_step_window(markers_path)
        file_hashes[markers_path.name] = hashlib.sha256(markers_path.read_bytes()).hexdigest()
        trace_path = trace_dir / f"rank_{rank}.jsonl"
        digest = hashlib.sha256()
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                digest.update(line.encode("utf-8"))
                if not line.strip():
                    continue
                record = json.loads(line)
                ts = int(record.get("ts") or 0)
                if ts < start_ts or ts > end_ts:
                    continue
                api = str(record.get("api", ""))
                window_api_counts[api] += 1
                kind = str(record.get("type", ""))
                if kind in {"marker", "other"}:
                    continue
                modeled_api_counts[api] += 1
                rows.append(raw_event_from_record(record, case=case, rank=rank, event_id=next_id))
                next_id += 1
        file_hashes[trace_path.name] = digest.hexdigest()
    if hook_audit is not None:
        combined = hashlib.sha256()
        for name, digest in sorted(file_hashes.items()):
            combined.update(f"{digest}  {name}\n".encode("utf-8"))
        hook_audit.update(
            {
                "trace_file_count": len(file_hashes),
                "trace_file_sha256": file_hashes,
                "combined_sha256": combined.hexdigest(),
                "step_window_hook_records": sum(window_api_counts.values()),
                "step_window_modeled_records": sum(modeled_api_counts.values()),
                "step_window_api_counts": dict(sorted(window_api_counts.items())),
                "modeled_api_counts": dict(sorted(modeled_api_counts.items())),
                "reported_auxiliary_api_counts": {
                    api: modeled_api_counts[api] for api in sorted(REPORTED_AUXILIARY_APIS)
                },
            }
        )
    return rows


def summarize_kind_counts(trace: object, *, use_dedup_weight: bool) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for event in trace.events:
        weight = int(getattr(event, "dedup_weight", 1) or 1) if use_dedup_weight else 1
        counts[str(getattr(event, "kind", ""))] += weight
    return dict(sorted(counts.items()))


def mismatch_api_counts(
    left: Counter[tuple[object, ...]],
    right: Counter[tuple[object, ...]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for side, delta in (("maya_only", left - right), ("flexeva_only", right - left)):
        counts: Counter[str] = Counter()
        for signature, count in delta.items():
            counts[str(signature[0])] += int(count)
        rows.extend({"api": api, "side": side, "count": count} for api, count in sorted(counts.items()))
    return rows


def canonical_lane_order_report(
    raw_events: list[object],
    rank_groups: dict[int, list[int]],
    *,
    excluded_apis: frozenset[str],
) -> dict[str, object]:
    rank_map = {rank: representative for representative, ranks in rank_groups.items() for rank in ranks}
    events_by_rank: dict[int, list[object]] = {}
    for event in raw_events:
        events_by_rank.setdefault(int(event.rank), []).append(event)

    reports: dict[int, dict[str, object]] = {}
    for rank, events in events_by_rank.items():
        host_threads: dict[int, int] = {}
        lane_counts: Counter[tuple[object, ...]] = Counter()
        lane_digests: dict[tuple[object, ...], object] = {}
        event_records: dict[int, tuple[object, ...]] = {}
        dependency_digest = hashlib.sha256()
        dependency_count = 0
        unresolved_dependency_count = 0
        for event in sorted(events, key=lambda item: (int(item.timestamp_ns), int(item.id))):
            if str(event.api) in excluded_apis:
                continue
            kind = str(event.kind)
            stream = int(event.stream)
            is_stream = kind in {"kernel_launch", "blas_compute", "nccl_collective", "mem_copy"} or (
                stream != 0 and kind not in {"host_delay", "host_marker"}
            )
            if is_stream:
                lane = ("stream", stream)
            else:
                thread_id = int(event.thread_id)
                lane = ("host", host_threads.setdefault(thread_id, len(host_threads)))
            peer_rank = int(event.peer_rank)
            collective_group = str(event.collective_group).split("|occ:", 1)[0]
            signature = (
                str(event.api),
                kind,
                str(event.code_partition),
                stream,
                int(event.bytes),
                int(event.count),
                rank_map.get(peer_rank, peer_rank),
                collective_group if kind == "nccl_collective" else "",
                bool(event.blocking),
            )
            digest = lane_digests.setdefault(lane, hashlib.sha256())
            digest.update(json.dumps(signature, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
            lane_counts[lane] += 1
            position = (lane, lane_counts[lane])
            event_handle = int(event.event_handle)
            wait_event_handle = int(event.wait_event_handle)
            if event_handle:
                event_records[event_handle] = position
            if wait_event_handle:
                source_position = event_records.get(wait_event_handle)
                if source_position is None:
                    unresolved_dependency_count += 1
                    source_position = ("unresolved",)
                dependency_digest.update(
                    json.dumps((source_position, position), separators=(",", ":")).encode("utf-8")
                )
                dependency_digest.update(b"\n")
                dependency_count += 1
        reports[rank] = {
            "lanes": {
                lane: (lane_counts[lane], digest.hexdigest()) for lane, digest in sorted(lane_digests.items())
            },
            "event_wait_dependency_count": dependency_count,
            "unresolved_event_wait_dependency_count": unresolved_dependency_count,
            "event_wait_dependency_digest": dependency_digest.hexdigest(),
        }

    comparisons = []
    for representative, ranks in sorted(rank_groups.items()):
        reference = reports.get(representative, {})
        for rank in sorted(ranks):
            comparisons.append(
                {
                    "representative_rank": representative,
                    "rank": rank,
                    "equal": reports.get(rank, {}) == reference,
                    "lane_count": len(reports.get(rank, {}).get("lanes", {})),
                    "event_wait_dependency_count": reports.get(rank, {}).get(
                        "event_wait_dependency_count", 0
                    ),
                    "unresolved_event_wait_dependency_count": reports.get(rank, {}).get(
                        "unresolved_event_wait_dependency_count", 0
                    ),
                    "event_wait_dependency_digest": reports.get(rank, {}).get(
                        "event_wait_dependency_digest", ""
                    ),
                }
            )
    return {
        "excluded_apis": sorted(excluded_apis),
        "all_equal": all(row["equal"] for row in comparisons),
        "comparisons": comparisons,
    }


def feedback_signal_report(maya_feedback: object, flexeva_feedback: object) -> dict[str, object]:
    maya_time = float(maya_feedback.total_time_us)
    flexeva_time = float(flexeva_feedback.total_time_us)
    return {
        "metric": "replay_total_time_us",
        "maya": maya_time,
        "flexeva": flexeva_time,
        "absolute_difference_us": abs(maya_time - flexeva_time),
        "relative_difference": abs(maya_time - flexeva_time) / max(abs(maya_time), 1.0),
        "cycle_free": not bool(maya_feedback.cycle_detected) and not bool(flexeva_feedback.cycle_detected),
        "duration_contract": "use positive captured host-duration hints; otherwise use the local predictor/kind fallback",
        "scope": "local replay diagnostic, not candidate-ranking or closed-loop agent evidence",
    }


def projected_feedback_report(
    maya_trace: object,
    flexeva_trace: object,
    *,
    zero_duration_apis: frozenset[str],
) -> dict[str, object]:
    def replay(trace: object) -> object:
        return fm.replay_trace_once(
            trace,
            predictor=fm.ReplayRandomForestPredictor(fm.ReplayRFConfig(enabled=False)),
            use_duration_hints=False,
            zero_duration_apis=zero_duration_apis,
        )

    report = feedback_signal_report(replay(maya_trace), replay(flexeva_trace))
    report.update(
        {
            "duration_contract": "ignore captured host-duration hints; deterministic kind/size fallback; predictor disabled",
            "zero_duration_apis": sorted(zero_duration_apis),
            "scope": "diagnostic projection, not candidate-ranking or closed-loop agent evidence",
        }
    )
    return report


def configuration_order_report(
    results: list[dict[str, object]],
    signal_field: str = "feedback_signal",
) -> dict[str, object]:
    rows = [row for row in results if signal_field in row]
    if len(rows) < 2:
        return {
            "evaluated": False,
            "reason": "at least two configurations are required",
            "scope": "configuration ordering, not an agent candidate trajectory",
        }
    maya = {str(row["case"]["name"]): float(row[signal_field]["maya"]) for row in rows}
    flexeva = {str(row["case"]["name"]): float(row[signal_field]["flexeva"]) for row in rows}
    names = sorted(maya)
    agreements = 0
    total = 0
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            maya_order = (maya[left] > maya[right]) - (maya[left] < maya[right])
            flexeva_order = (flexeva[left] > flexeva[right]) - (flexeva[left] < flexeva[right])
            agreements += int(maya_order == flexeva_order)
            total += 1
    maya_order = sorted(names, key=lambda name: (maya[name], name))
    flexeva_order = sorted(names, key=lambda name: (flexeva[name], name))
    return {
        "evaluated": True,
        "scope": "configuration ordering, not an agent candidate trajectory",
        "signal_field": signal_field,
        "maya_order_fast_to_slow": maya_order,
        "flexeva_order_fast_to_slow": flexeva_order,
        "exact_order_equal": maya_order == flexeva_order,
        "pairwise_order_agreement": agreements / max(total, 1),
        "pair_count": total,
    }


def compare_case(args: argparse.Namespace, case: MegatronCase) -> dict[str, object]:
    case_dir = args.out_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    run = run_maya_megatron_case(args, case, case_dir)
    if run["return_code"] != 0:
        return {"case": case.__dict__, "run": run, "error": "maya_megatron.py failed"}

    parse_start = time.perf_counter()
    hook_audit: dict[str, object] = {}
    raw_events = parse_case_raw_events(case, Path(run["trace_dir"]), hook_audit)
    parse_s = time.perf_counter() - parse_start
    spec = stage_spec_for(case)
    rank_groups = fm.active_lane_rank_groups(spec)

    maya_start = time.perf_counter()
    maya_trace = fm.build_trace_ras(raw_events)
    maya_build_s = time.perf_counter() - maya_start
    flex_start = time.perf_counter()
    flexeva_trace = fm.build_rank_grouped_trace_ras(raw_events, fm.active_lane_rank_groups(spec))
    flexeva_build_s = time.perf_counter() - flex_start

    maya_replay_start = time.perf_counter()
    maya_feedback = fm.replay_trace_once(maya_trace)
    maya_replay_s = time.perf_counter() - maya_replay_start
    flexeva_replay_start = time.perf_counter()
    flexeva_feedback = fm.replay_trace_once(flexeva_trace)
    flexeva_replay_s = time.perf_counter() - flexeva_replay_start

    maya_weighted = weighted_event_counter(maya_trace, use_dedup_weight=True)
    flexeva_weighted = weighted_event_counter(flexeva_trace, use_dedup_weight=True)
    maya_physical = weighted_event_counter(maya_trace, use_dedup_weight=False)
    flexeva_physical = weighted_event_counter(flexeva_trace, use_dedup_weight=False)
    maya_reported_normalized = filtered_weighted_event_counter(
        maya_trace,
        use_dedup_weight=True,
        excluded_apis=REPORTED_AUXILIARY_APIS,
    )
    flexeva_reported_normalized = filtered_weighted_event_counter(
        flexeva_trace,
        use_dedup_weight=True,
        excluded_apis=REPORTED_AUXILIARY_APIS,
    )
    maya_kernel_nccl = filtered_weighted_event_counter(
        maya_trace,
        use_dedup_weight=True,
        kinds=MODELED_KERNEL_NCCL_KINDS,
    )
    flexeva_kernel_nccl = filtered_weighted_event_counter(
        flexeva_trace,
        use_dedup_weight=True,
        kinds=MODELED_KERNEL_NCCL_KINDS,
    )
    maya_wait_edges = weighted_edge_counter(maya_trace, reason="event_wait")
    flexeva_wait_edges = weighted_edge_counter(flexeva_trace, reason="event_wait")
    maya_windows = trace_window_counter(maya_trace)
    flexeva_windows = trace_window_counter(flexeva_trace)
    mismatches = mismatch_api_counts(maya_weighted, flexeva_weighted)
    mismatch_apis = {str(row["api"]) for row in mismatches}
    run_record = dict(run)
    if args.input_trace_root is not None:
        run_record["trace_dir"] = str(Path(case.name) / "traces")
        run_record["trace_dir_base"] = "TRACE_INPUT_ROOT"
        run_record["stdout"] = str(Path(case.name) / "stdout.txt")
        run_record["stderr"] = str(Path(case.name) / "stderr.txt")
        run_record["log_path_base"] = "OUT_DIR"

    if not args.keep_raw_traces and args.input_trace_root is None:
        for path in Path(run["trace_dir"]).glob("rank_*.jsonl"):
            path.unlink(missing_ok=True)

    return {
        "case": case.__dict__,
        "run": run_record,
        "raw_events": len(raw_events),
        "maya": {
            "trace": fm.trace_summary(maya_trace),
            "feedback": maya_feedback.to_dict(),
            "logical_kind_counts": summarize_kind_counts(maya_trace, use_dedup_weight=True),
            "trace_build_s": maya_build_s,
            "replay_s": maya_replay_s,
        },
        "flexeva": {
            "trace": fm.trace_summary(flexeva_trace),
            "feedback": flexeva_feedback.to_dict(),
            "logical_kind_counts": summarize_kind_counts(flexeva_trace, use_dedup_weight=True),
            "rank_groups_from_active_lane_sets": rank_groups,
            "trace_build_s": flexeva_build_s,
            "replay_s": flexeva_replay_s,
        },
        "phases_s": {"jsonl_parse_s": parse_s},
        "raw_input_contract": {
            "comparison": "one shared fake-CUDA hook capture is transformed by both builders",
            "paired_raw_streams": False,
            "raw_stream_equality_claimed": False,
            "hook_capture": hook_audit,
        },
        "auxiliary_api_audit": {
            "reported_apis": sorted(REPORTED_AUXILIARY_APIS),
            "mismatch_by_api_and_side": mismatches,
            "all_mismatches_within_reported_set": mismatch_apis <= REPORTED_AUXILIARY_APIS,
            "diagnostic_projection_only": (
                "The four-API exclusion reproduces the rebuttal projection. The bundled Maya marks "
                "cudaEventQuery as low-overhead semantic work. The captured-timing replay retains all records; "
                "each diagnostic projection lists the APIs assigned zero duration."
            ),
        },
        "lane_order": canonical_lane_order_report(
            raw_events,
            rank_groups,
            excluded_apis=REPORTED_AUXILIARY_APIS,
        ),
        "feedback_signal": feedback_signal_report(maya_feedback, flexeva_feedback),
        "bundled_maya_visibility_projection": projected_feedback_report(
            maya_trace,
            flexeva_trace,
            zero_duration_apis=MAYA_COMPATIBILITY_ONLY_APIS,
        ),
        "rebuttal_four_api_projection": projected_feedback_report(
            maya_trace,
            flexeva_trace,
            zero_duration_apis=REPORTED_AUXILIARY_APIS,
        ),
        "similarity": {
            "logical_event_coverage": int(flexeva_trace.logical_event_count) / max(int(maya_trace.logical_event_count), 1),
            "logical_event_count_equal": int(maya_trace.logical_event_count) == int(flexeva_trace.logical_event_count),
            "weighted_event_jaccard": weighted_jaccard(maya_weighted, flexeva_weighted),
            "weighted_event_cosine": cosine(maya_weighted, flexeva_weighted),
            "reported_four_api_normalized_event_jaccard": weighted_jaccard(
                maya_reported_normalized,
                flexeva_reported_normalized,
            ),
            "modeled_kernel_nccl_jaccard": weighted_jaccard(maya_kernel_nccl, flexeva_kernel_nccl),
            "event_wait_dependency_jaccard": weighted_jaccard(maya_wait_edges, flexeva_wait_edges),
            "maya_event_wait_logical_edges": sum(maya_wait_edges.values()),
            "flexeva_event_wait_logical_edges": sum(flexeva_wait_edges.values()),
            "trace_window_weighted_jaccard": weighted_jaccard(maya_windows, flexeva_windows),
            "physical_event_jaccard_without_dedup_weights": weighted_jaccard(maya_physical, flexeva_physical),
            "compact_event_ratio_flexeva_vs_maya": len(flexeva_trace.events) / max(len(maya_trace.events), 1),
        },
    }


def main() -> int:
    args = parse_args()
    if args.input_trace_root is not None and not args.reuse_existing_traces:
        raise SystemExit("--input-trace-root requires --reuse-existing-traces")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = default_cases()
    if args.case:
        selected = set(args.case)
        cases = tuple(case for case in cases if case.name in selected)
    results = []
    for case in cases:
        results.append(compare_case(args, case))
        (args.out_dir / "partial-results.json").write_text(
            json.dumps({"results": results}, indent=2) + "\n",
            encoding="utf-8",
        )
    result = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": {
            "capture": "workload/megatron/maya_megatron.py through fake-CUDA frun",
            "window": "training_step markers from FLEXSIM_MAYA_MARKERS_PATH",
            "maya": "Maya-style full trace-RAS with build_trace_ras",
            "flexeva": "FlexEva trace-RAS ablation using active-lane-set grouped compact trace",
            "scope": "trace-RAS compaction over one shared hook capture; not source-RAS selective refresh",
        },
        "results": results,
        "configuration_ordering": configuration_order_report(results),
        "bundled_maya_visibility_configuration_ordering": configuration_order_report(
            results,
            "bundled_maya_visibility_projection",
        ),
        "rebuttal_four_api_configuration_ordering": configuration_order_report(
            results,
            "rebuttal_four_api_projection",
        ),
    }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "parameter_scale",
                "return_code",
                "raw_events",
                "maya_events",
                "flexeva_events",
                "maya_logical_events",
                "flexeva_logical_events",
                "logical_event_coverage",
                "weighted_event_jaccard",
                "weighted_event_cosine",
                "reported_four_api_normalized_event_jaccard",
                "modeled_kernel_nccl_jaccard",
                "event_wait_dependency_jaccard",
                "normalized_lane_order_equal",
                "feedback_relative_difference",
                "bundled_maya_visibility_projection_relative_difference",
                "rebuttal_four_api_projection_relative_difference",
                "physical_event_jaccard_without_dedup_weights",
                "compact_event_ratio_flexeva_vs_maya",
                "elapsed_s",
            ],
        )
        writer.writeheader()
        for row in results:
            case = row["case"]
            similarity = row.get("similarity", {})
            writer.writerow(
                {
                    "case": case["name"],
                    "parameter_scale": case["parameter_scale"],
                    "return_code": row["run"]["return_code"],
                    "raw_events": row.get("raw_events"),
                    "maya_events": row.get("maya", {}).get("trace", {}).get("event_count"),
                    "flexeva_events": row.get("flexeva", {}).get("trace", {}).get("event_count"),
                    "maya_logical_events": row.get("maya", {}).get("trace", {}).get("logical_event_count"),
                    "flexeva_logical_events": row.get("flexeva", {}).get("trace", {}).get("logical_event_count"),
                    "logical_event_coverage": similarity.get("logical_event_coverage"),
                    "weighted_event_jaccard": similarity.get("weighted_event_jaccard"),
                    "weighted_event_cosine": similarity.get("weighted_event_cosine"),
                    "reported_four_api_normalized_event_jaccard": similarity.get(
                        "reported_four_api_normalized_event_jaccard"
                    ),
                    "modeled_kernel_nccl_jaccard": similarity.get("modeled_kernel_nccl_jaccard"),
                    "event_wait_dependency_jaccard": similarity.get("event_wait_dependency_jaccard"),
                    "normalized_lane_order_equal": row.get("lane_order", {}).get("all_equal"),
                    "feedback_relative_difference": row.get("feedback_signal", {}).get("relative_difference"),
                    "bundled_maya_visibility_projection_relative_difference": row.get(
                        "bundled_maya_visibility_projection", {}
                    ).get("relative_difference"),
                    "rebuttal_four_api_projection_relative_difference": row.get(
                        "rebuttal_four_api_projection", {}
                    ).get("relative_difference"),
                    "physical_event_jaccard_without_dedup_weights": similarity.get(
                        "physical_event_jaccard_without_dedup_weights"
                    ),
                    "compact_event_ratio_flexeva_vs_maya": similarity.get("compact_event_ratio_flexeva_vs_maya"),
                    "elapsed_s": row["run"]["elapsed_s"],
                }
            )
    print(json.dumps({"result": str(args.out_dir / "result.json"), "summary": str(args.out_dir / "summary.csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
