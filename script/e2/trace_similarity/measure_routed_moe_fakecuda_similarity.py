#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FLEXEVA_RAS_SRC = ROOT / "FlexEva" / "flexmaya_ras" / "src"
if str(FLEXEVA_RAS_SRC) not in sys.path:
    sys.path.insert(0, str(FLEXEVA_RAS_SRC))

import flexmaya_ras as fm

from trace_metrics import (
    cosine,
    filtered_weighted_event_counter,
    trace_window_counter,
    weighted_edge_counter,
    weighted_jaccard,
)
from measure_maya_megatron_fakecuda_similarity import (
    MAYA_COMPATIBILITY_ONLY_APIS,
    MODELED_KERNEL_NCCL_KINDS,
    REPORTED_AUXILIARY_APIS,
    canonical_lane_order_report,
    feedback_signal_report,
    mismatch_api_counts,
    projected_feedback_report,
)


@dataclass(frozen=True)
class RoutedMoeConfig:
    backend: str
    binary: str
    world_size: int
    ep_size: int
    dp: int
    steps: int
    global_batch_size: int
    seq_len: int
    hidden_size: int
    num_layers: int
    num_heads: int
    vocab_size: int
    num_experts: int
    top_k: int
    capacity_factor: float
    micro_batches: int
    dtype: str


@dataclass(frozen=True)
class RouteCase:
    path_id: int
    name: str
    label: str
    experts: tuple[int, int]


DEFAULT_CONFIG = RoutedMoeConfig(
    backend="ns3",
    binary="extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default",
    world_size=16,
    ep_size=16,
    dp=1,
    steps=1,
    global_batch_size=16,
    seq_len=64,
    hidden_size=128,
    num_layers=2,
    num_heads=4,
    vocab_size=4096,
    num_experts=16,
    top_k=2,
    capacity_factor=1.25,
    micro_batches=1,
    dtype="bf16",
)

ROUTE_CASES = (
    RouteCase(
        path_id=0,
        name="routed_moe_intra_group_0_1",
        label="fixed expert ranks (0,1), intra 8-rank group",
        experts=(0, 1),
    ),
    RouteCase(
        path_id=1,
        name="routed_moe_cross_group_0_8",
        label="fixed expert ranks (0,8), cross 8-rank groups",
        experts=(0, 8),
    ),
    RouteCase(
        path_id=2,
        name="routed_moe_cross_group_0_15",
        label="fixed expert ranks (0,15), cross 8-rank groups",
        experts=(0, 15),
    ),
    RouteCase(
        path_id=3,
        name="routed_moe_boundary_7_8",
        label="fixed expert ranks (7,8), cross group boundary",
        experts=(7, 8),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run workload/routed_moe/moe_topk.py through fake-CUDA and compare Maya/FlexEva traces."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--maya-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--proot", type=Path, required=True)
    parser.add_argument(
        "--trace-root",
        type=Path,
        help="Store fresh CASE/traces outside the result directory.",
    )
    parser.add_argument("--local-device-count", type=int, default=8)
    parser.add_argument("--path-count", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=4200)
    parser.add_argument("--keep-raw-traces", action="store_true")
    parser.add_argument(
        "--reuse-existing-traces",
        action="store_true",
        help=(
            "Do not relaunch moe_topk.py. Reuse rank_*.jsonl and "
            "rank_*_markers.jsonl from INPUT_TRACE_ROOT/CASE/traces, or from "
            "OUT_DIR/CASE/traces when no input root is supplied."
        ),
    )
    parser.add_argument(
        "--input-trace-root",
        type=Path,
        help="Read CASE/traces below this directory when reusing existing traces.",
    )
    parser.add_argument("--sync-before-step-window", action="store_true")
    parser.add_argument(
        "--no-route-p2p-probe",
        action="store_true",
        help="Disable sparse P2P route probes. Enabled by default for path-aware routed-MoE runs.",
    )
    return parser.parse_args()


def dtype_bytes(dtype: str) -> int:
    if dtype == "bf16":
        return 2
    if dtype == "fp32":
        return 4
    return 4


def singleton_rank_groups(world_size: int) -> dict[int, list[int]]:
    return {rank: [rank] for rank in range(world_size)}


def run_routed_moe_path(
    args: argparse.Namespace,
    config: RoutedMoeConfig,
    *,
    route_case: RouteCase,
    seed: int,
    case_dir: Path,
) -> dict[str, object]:
    input_trace_root = getattr(args, "input_trace_root", None)
    trace_dir = (
        input_trace_root / route_case.name / "traces"
        if input_trace_root
        else (
            args.trace_root / route_case.name / "traces"
            if args.trace_root
            else case_dir / "traces"
        )
    )
    if args.reuse_existing_traces:
        missing = []
        for rank in range(config.world_size):
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
            "path_id": route_case.path_id,
            "route_label": route_case.label,
            "route_experts": list(route_case.experts),
            "seed": seed,
            "reused_existing_traces": True,
        }

    trace_dir.mkdir(parents=True, exist_ok=True)

    wrapper = Path(__file__).resolve().parent / "routed_moe_trace_worker.py"
    workload = Path(
        os.environ.get(
            "FLEXMAYA_ROUTED_MOE_SCRIPT",
            Path(__file__).resolve().parents[1] / "workload" / "routed_moe" / "moe_topk.py",
        )
    ).resolve()
    if not workload.is_file():
        raise FileNotFoundError(f"Routed-MoE workload does not exist: {workload}")
    frun = args.maya_root / "fake-cuda" / "frun"
    env = os.environ.copy()
    env.pop("FAKECUDA_TRACE", None)
    env.pop("FAKECUDA_TRACE_PATH", None)
    env["FAKECUDA_PROOT_BIN"] = str(args.proot)
    env["FAKECUDA_FRUN_QUIET"] = "1"
    env["FLEXMAYA_TRACE_DIR"] = str(trace_dir)
    env["FLEXMAYA_LOCAL_DEVICE_COUNT"] = str(args.local_device_count)
    env["ROUTED_MOE_SCRIPT"] = str(workload)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(workload.parent),
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
        f"--nproc-per-node={config.world_size}",
        f"--master-port={random.randint(42000, 62000)}",
        str(wrapper),
        "--steps",
        str(config.steps),
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
        "--micro-batches",
        str(config.micro_batches),
        "--dtype",
        config.dtype,
        "--warmup-steps",
        "0",
        "--seed",
        str(seed),
        "--route-path-id",
        str(route_case.path_id),
        "--route-experts",
        ",".join(str(expert) for expert in route_case.experts),
        "--log-interval",
        "1",
    ]
    if not args.no_route_p2p_probe:
        command.append("--route-p2p-probe")
    if getattr(args, "source_region_markers", False):
        command.append("--source-region-markers")
    if getattr(args, "sync_before_step_window", False):
        command.append("--sync-before-step-window")
    stdout = case_dir / "stdout.txt"
    stderr = case_dir / "stderr.txt"
    start = time.perf_counter()
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        completed = subprocess.run(command, env=env, cwd=case_dir, stdout=out, stderr=err, text=True)
    return {
        "command": command,
        "return_code": completed.returncode,
        "elapsed_s": time.perf_counter() - start,
        "stdout": str(stdout),
        "stderr": str(stderr),
        "trace_dir": str(trace_dir),
        "path_id": route_case.path_id,
        "route_label": route_case.label,
        "route_experts": list(route_case.experts),
        "seed": seed,
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


def raw_event_from_record(record: dict[str, object], *, config: RoutedMoeConfig, rank: int, event_id: int) -> object:
    kind = str(record.get("type", ""))
    api = str(record.get("api", ""))
    count = record_count(record)
    bytes_ = int(record.get("bytes") or 0)
    if bytes_ <= 0 and kind == "nccl_collective":
        bytes_ = count * dtype_bytes(config.dtype)
    collective_group = ""
    if kind == "nccl_collective":
        collective_group = f"{api}:call={int(record.get('call_idx') or 0)}"
    cuda_event_id = int(record.get("event_id") or 0)
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
        event_handle=cuda_event_id if "EventRecord" in api else 0,
        wait_event_handle=cuda_event_id if "WaitEvent" in api else 0,
        collective_group=collective_group,
        code_partition=f"expert_rank_{rank:03d}",
        blocking=api.endswith("Synchronize"),
    )


def parse_case_raw_events(
    config: RoutedMoeConfig,
    trace_dir: Path,
    hook_audit: dict[str, object] | None = None,
) -> list[object]:
    rows: list[object] = []
    next_id = 1
    window_api_counts: Counter[str] = Counter()
    modeled_api_counts: Counter[str] = Counter()
    file_hashes: dict[str, str] = {}
    for rank in range(config.world_size):
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
                rows.append(raw_event_from_record(record, config=config, rank=rank, event_id=next_id))
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


def routed_event_signature(event: object) -> tuple[object, ...]:
    collective_group = str(getattr(event, "collective_group", ""))
    collective_base = collective_group.split("|occ:", 1)[0]
    api = str(getattr(event, "api", ""))
    kind = str(getattr(event, "kind", ""))
    return (
        api,
        kind,
        str(getattr(event, "code_partition", "")),
        int(getattr(event, "rank", -1)),
        int(getattr(event, "peer_rank", -1)),
        int(getattr(event, "stream", 0)),
        int(getattr(event, "bytes", 0)),
        int(getattr(event, "count", 0)),
        collective_base if kind == "nccl_collective" else "",
    )


def weighted_event_counter(trace: object, *, use_dedup_weight: bool) -> Counter[tuple[object, ...]]:
    counter: Counter[tuple[object, ...]] = Counter()
    for event in trace.events:
        weight = int(getattr(event, "dedup_weight", 1) or 1) if use_dedup_weight else 1
        counter[routed_event_signature(event)] += weight
    return counter


def summarize_peer_events(raw_events: list[object]) -> dict[str, object]:
    pair_counts: Counter[tuple[int, int, str]] = Counter()
    for event in raw_events:
        peer = int(getattr(event, "peer_rank", -1))
        rank = int(getattr(event, "rank", -1))
        api = str(getattr(event, "api", ""))
        if peer < 0:
            continue
        pair_counts[(rank, peer, api)] += 1
    return {
        "peer_event_count": sum(pair_counts.values()),
        "unique_peer_pairs": len({(rank, peer) for rank, peer, _api in pair_counts}),
        "sample": [
            {"rank": rank, "peer": peer, "api": api, "count": count}
            for (rank, peer, api), count in sorted(pair_counts.items())[:16]
        ],
    }


def compare_path(args: argparse.Namespace, config: RoutedMoeConfig, *, route_case: RouteCase) -> dict[str, object]:
    seed = args.seed_base + route_case.path_id
    case_name = route_case.name
    case_dir = args.out_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    run = run_routed_moe_path(args, config, route_case=route_case, seed=seed, case_dir=case_dir)
    if run["return_code"] != 0:
        return {"case": case_name, "config": asdict(config), "run": run, "error": "moe_topk.py failed"}

    parse_start = time.perf_counter()
    hook_audit: dict[str, object] = {}
    raw_events = parse_case_raw_events(config, Path(run["trace_dir"]), hook_audit)
    parse_s = time.perf_counter() - parse_start
    peer_summary = summarize_peer_events(raw_events)

    maya_start = time.perf_counter()
    maya_trace = fm.build_trace_ras(raw_events)
    maya_build_s = time.perf_counter() - maya_start

    flex_start = time.perf_counter()
    flexeva_trace = fm.build_rank_grouped_trace_ras(raw_events, singleton_rank_groups(config.world_size))
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
    rank_groups = singleton_rank_groups(config.world_size)
    run_record = dict(run)
    if args.input_trace_root is not None:
        run_record["trace_dir"] = str(Path(route_case.name) / "traces")
        run_record["trace_dir_base"] = "TRACE_INPUT_ROOT"
        run_record["stdout"] = str(Path(route_case.name) / "stdout.txt")
        run_record["stderr"] = str(Path(route_case.name) / "stderr.txt")
        run_record["log_path_base"] = "OUT_DIR"

    if not args.keep_raw_traces and args.input_trace_root is None:
        for path in Path(run["trace_dir"]).glob("rank_*.jsonl"):
            path.unlink(missing_ok=True)

    return {
        "case": case_name,
        "path_id": route_case.path_id,
        "route_label": route_case.label,
        "route_experts": list(route_case.experts),
        "seed": seed,
        "config": asdict(config),
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
            "rank_groups": rank_groups,
            "trace_build_s": flexeva_build_s,
            "replay_s": flexeva_replay_s,
        },
        "phases_s": {"jsonl_parse_s": parse_s},
        "peer_events": peer_summary,
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
    if args.trace_root is not None:
        args.trace_root.mkdir(parents=True, exist_ok=True)
    config = DEFAULT_CONFIG
    route_cases = ROUTE_CASES[: args.path_count]
    results = []
    for route_case in route_cases:
        results.append(compare_path(args, config, route_case=route_case))
        (args.out_dir / "partial-results.json").write_text(
            json.dumps({"results": results}, indent=2) + "\n",
            encoding="utf-8",
        )
    result = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": {
            "capture": "workload/routed_moe/moe_topk.py through fake-CUDA frun",
            "window": "training_step markers from FLEXSIM_MAYA_MARKERS_PATH",
            "maya": "Maya-style full trace-RAS with build_trace_ras",
            "flexeva": "FlexEva trace-RAS ablation with routed-MoE singleton expert-rank active lanes",
            "scope": "trace-RAS transformation over one shared hook capture; not source-RAS selective refresh",
            "path_variation": (
                "explicit forced top-2 expert rank pairs plus sparse P2P route probes; "
                "event signatures include rank and peer_rank"
            ),
        },
        "route_cases": [asdict(route_case) for route_case in route_cases],
        "config": asdict(config),
        "results": results,
    }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "path_id",
                "route_label",
                "route_experts",
                "seed",
                "world_size",
                "ep_size",
                "dp",
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
                "capture_timing_feedback_relative_difference",
                "bundled_maya_visibility_projection_relative_difference",
                "rebuttal_four_api_projection_relative_difference",
                "physical_event_jaccard_without_dedup_weights",
                "compact_event_ratio_flexeva_vs_maya",
                "peer_event_count",
                "unique_peer_pairs",
                "elapsed_s",
            ],
        )
        writer.writeheader()
        for row in results:
            similarity = row.get("similarity", {})
            writer.writerow(
                {
                    "case": row["case"],
                    "path_id": row.get("path_id"),
                    "route_label": row.get("route_label"),
                    "route_experts": json.dumps(row.get("route_experts")),
                    "seed": row.get("seed"),
                    "world_size": row.get("config", {}).get("world_size"),
                    "ep_size": row.get("config", {}).get("ep_size"),
                    "dp": row.get("config", {}).get("dp"),
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
                    "capture_timing_feedback_relative_difference": row.get("feedback_signal", {}).get(
                        "relative_difference"
                    ),
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
                    "peer_event_count": row.get("peer_events", {}).get("peer_event_count"),
                    "unique_peer_pairs": row.get("peer_events", {}).get("unique_peer_pairs"),
                    "elapsed_s": row["run"]["elapsed_s"],
                }
            )
    print(json.dumps({"result": str(args.out_dir / "result.json"), "summary": str(args.out_dir / "summary.csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
