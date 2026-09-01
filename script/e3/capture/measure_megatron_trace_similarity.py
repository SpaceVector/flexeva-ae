#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import flexmaya_ras as fm


@dataclass(frozen=True)
class MegatronCase:
    name: str
    parameter_scale: str
    steps: int
    global_batch_size: int
    seq_len: int
    hidden_size: int
    num_layers: int
    num_heads: int
    vocab_size: int
    tp: int
    pp: int
    dp: int
    world_size: int
    micro_batches: int
    schedule: str
    dtype: str


def default_cases() -> tuple[MegatronCase, ...]:
    return (
        MegatronCase(
            name="megatron_2p7b_8gpu",
            parameter_scale="2.7B",
            steps=1,
            global_batch_size=256,
            seq_len=2048,
            hidden_size=2560,
            num_layers=32,
            num_heads=32,
            vocab_size=32000,
            tp=1,
            pp=8,
            dp=1,
            world_size=8,
            micro_batches=256,
            schedule="1f1b",
            dtype="bf16",
        ),
        MegatronCase(
            name="megatron_18p4b_16gpu",
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
        ),
        MegatronCase(
            name="megatron_2p7b_16gpu_dp2",
            parameter_scale="2.7B",
            steps=1,
            global_batch_size=512,
            seq_len=2048,
            hidden_size=2560,
            num_layers=32,
            num_heads=32,
            vocab_size=32000,
            tp=1,
            pp=8,
            dp=2,
            world_size=16,
            micro_batches=256,
            schedule="1f1b",
            dtype="bf16",
        ),
        MegatronCase(
            name="megatron_18p4b_16gpu_dp2",
            parameter_scale="18.4B",
            steps=1,
            global_batch_size=512,
            seq_len=2048,
            hidden_size=6144,
            num_layers=40,
            num_heads=48,
            vocab_size=32000,
            tp=1,
            pp=8,
            dp=2,
            world_size=16,
            micro_batches=512,
            schedule="1f1b",
            dtype="bf16",
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Maya-vs-FlexEva trace similarity for Megatron cases."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in default_cases()],
        help="Case name to run. Defaults to all configured cases.",
    )
    return parser.parse_args()


def dtype_bytes(dtype: str) -> int:
    if dtype == "fp32":
        return 4
    if dtype == "bf16":
        return 2
    raise ValueError(f"unsupported dtype: {dtype}")


def local_micro_batch_size(case: MegatronCase) -> int:
    return max(1, math.ceil(case.global_batch_size / max(case.dp * case.micro_batches, 1)))


def coords_to_rank(dp_rank: int, tp_rank: int, pp_rank: int, case: MegatronCase) -> int:
    return dp_rank * (case.tp * case.pp) + tp_rank * case.pp + pp_rank


def rank_to_coords(rank: int, case: MegatronCase) -> tuple[int, int, int]:
    pp_rank = rank % case.pp
    tp_rank = (rank // case.pp) % case.tp
    dp_rank = rank // (case.tp * case.pp)
    return dp_rank, tp_rank, pp_rank


def stage_layer_count(case: MegatronCase, pp_rank: int) -> int:
    base = case.num_layers // case.pp
    remainder = case.num_layers % case.pp
    return base + (1 if pp_rank < remainder else 0)


def global_layer_id(case: MegatronCase, pp_rank: int, local_layer: int) -> int:
    return pp_rank + local_layer * case.pp


def model_shape(case: MegatronCase, pp_rank: int) -> dict[str, int]:
    mb = local_micro_batch_size(case)
    elem_bytes = dtype_bytes(case.dtype)
    activation_elems = mb * case.seq_len * case.hidden_size
    hidden = case.hidden_size
    intermediate = hidden * 4
    local_layers = stage_layer_count(case, pp_rank)
    attention_params = (3 * hidden * hidden + hidden * hidden) // case.tp
    mlp_params = (hidden * intermediate + intermediate * hidden) // case.tp
    layer_param_elems = attention_params + mlp_params + 4 * hidden
    layer_param_bytes = layer_param_elems * elem_bytes
    stage_param_bytes = local_layers * layer_param_bytes
    if pp_rank == 0:
        stage_param_bytes += case.vocab_size * hidden * elem_bytes
    if pp_rank == case.pp - 1:
        stage_param_bytes += case.vocab_size * hidden * elem_bytes
    qkv_ops = 2 * activation_elems * (3 * hidden // case.tp)
    attn_score_ops = 2 * mb * (case.num_heads // case.tp) * case.seq_len * case.seq_len
    attn_value_ops = attn_score_ops
    attn_out_ops = 2 * activation_elems * hidden // case.tp
    mlp_up_ops = 2 * activation_elems * intermediate // case.tp
    mlp_down_ops = 2 * activation_elems * intermediate // case.tp
    forward_ops = qkv_ops + attn_score_ops + attn_value_ops + attn_out_ops + mlp_up_ops + mlp_down_ops
    return {
        "activation_bytes": max(activation_elems * elem_bytes, 1),
        "logits_bytes": max(mb * case.seq_len * case.vocab_size * elem_bytes, 1),
        "embedding_bytes": max(case.vocab_size * hidden * elem_bytes, 1),
        "stage_param_bytes": max(stage_param_bytes, 1),
        "forward_ops_per_layer": max(forward_ops, 1),
        "backward_ops_per_layer": max(2 * forward_ops, 1),
        "optimizer_ops": max(stage_param_bytes // elem_bytes * 2, 1),
    }


def spec_for(case: MegatronCase) -> fm.FlexMayaWorkloadSpec:
    return fm.FlexMayaWorkloadSpec(
        workload_id=case.name,
        world_size=case.world_size,
        tp=case.tp,
        pp=case.pp,
        dp=case.dp,
        code_partitions=tuple(
            fm.CodePartitionSpec(
                partition_id=f"layer_{layer:03d}",
                path=__file__,
                active_ranks=fm.megatron_pp_stage_active_ranks(
                    case.world_size, case.tp, case.pp, layer % max(case.pp, 1)
                ),
            )
            for layer in range(case.num_layers)
        ),
        rank_group_policy="active_lane_set",
        notes=("megatron trace similarity", case.parameter_scale),
    )


class EventBuilder:
    def __init__(self, case: MegatronCase) -> None:
        self.case = case
        self.rows: list[object] = []
        self.ts = 100

    def add(
        self,
        api: str,
        kind: str,
        *,
        rank: int,
        stream: int,
        code_partition: str,
        bytes_: int = 0,
        count: int = 0,
        peer_rank: int = -1,
        collective_group: str = "",
        duration_hint_us: float = 0.0,
    ) -> None:
        self.rows.append(
            fm.make_event(
                api,
                kind,
                rank=rank,
                stream=stream,
                timestamp_ns=self.ts,
                duration_hint_us=duration_hint_us,
                bytes=bytes_,
                count=count,
                peer_rank=peer_rank,
                collective_group=collective_group,
                code_partition=code_partition,
            )
        )
        self.ts += 10


def add_tp_all_reduce(
    builder: EventBuilder,
    case: MegatronCase,
    *,
    rank: int,
    dp_rank: int,
    pp_rank: int,
    step: int,
    microbatch: int,
    layer: int,
    op: str,
    bytes_: int,
) -> None:
    if case.tp <= 1:
        return
    group = f"tp:dp{dp_rank}:pp{pp_rank}:step{step}:mb{microbatch}:layer{layer}:{op}"
    builder.add(
        "ncclAllReduce",
        "nccl_collective",
        rank=rank,
        stream=3,
        code_partition=f"layer_{layer:03d}",
        bytes_=bytes_,
        count=max(bytes_ // dtype_bytes(case.dtype), 1),
        collective_group=group,
    )


def add_dp_all_reduce(
    builder: EventBuilder,
    case: MegatronCase,
    *,
    rank: int,
    tp_rank: int,
    pp_rank: int,
    step: int,
    bytes_: int,
) -> None:
    if case.dp <= 1:
        return
    group = f"dp:tp{tp_rank}:pp{pp_rank}:step{step}:grad"
    builder.add(
        "ncclAllReduce",
        "nccl_collective",
        rank=rank,
        stream=3,
        code_partition=f"stage_{pp_rank:03d}_optimizer",
        bytes_=bytes_,
        count=max(bytes_ // dtype_bytes(case.dtype), 1),
        collective_group=group,
    )


def generate_rank_events(builder: EventBuilder, case: MegatronCase, rank: int) -> None:
    dp_rank, tp_rank, pp_rank = rank_to_coords(rank, case)
    shape = model_shape(case, pp_rank)
    warmup = min(case.pp - pp_rank - 1, case.micro_batches)
    queue: list[int] = []

    prev_rank = (
        coords_to_rank(dp_rank, tp_rank, pp_rank - 1, case)
        if pp_rank > 0
        else None
    )
    next_rank = (
        coords_to_rank(dp_rank, tp_rank, pp_rank + 1, case)
        if pp_rank + 1 < case.pp
        else None
    )

    def forward_microbatch(step: int, microbatch: int) -> None:
        if prev_rank is not None:
            builder.add(
                "ncclRecv",
                "mem_copy",
                rank=rank,
                stream=1,
                code_partition=f"stage_{pp_rank:03d}_p2p",
                bytes_=shape["activation_bytes"],
                count=max(shape["activation_bytes"] // dtype_bytes(case.dtype), 1),
                peer_rank=prev_rank,
            )
        elif microbatch == 0:
            builder.add(
                "cudaLaunchKernel",
                "kernel_launch",
                rank=rank,
                stream=0,
                code_partition="embedding",
                bytes_=shape["embedding_bytes"],
                count=local_micro_batch_size(case) * case.seq_len,
            )

        for local_layer in range(stage_layer_count(case, pp_rank)):
            layer = global_layer_id(case, pp_rank, local_layer)
            partition = f"layer_{layer:03d}"
            builder.add(
                "cublasGemmEx",
                "blas_compute",
                rank=rank,
                stream=3,
                code_partition=partition,
                bytes_=shape["activation_bytes"],
                count=shape["forward_ops_per_layer"] // 2,
            )
            add_tp_all_reduce(
                builder,
                case,
                rank=rank,
                dp_rank=dp_rank,
                pp_rank=pp_rank,
                step=step,
                microbatch=microbatch,
                layer=layer,
                op="attention_output",
                bytes_=shape["activation_bytes"],
            )
            builder.add(
                "cublasGemmEx",
                "blas_compute",
                rank=rank,
                stream=3,
                code_partition=partition,
                bytes_=shape["activation_bytes"],
                count=shape["forward_ops_per_layer"] // 2,
            )
            add_tp_all_reduce(
                builder,
                case,
                rank=rank,
                dp_rank=dp_rank,
                pp_rank=pp_rank,
                step=step,
                microbatch=microbatch,
                layer=layer,
                op="mlp_output",
                bytes_=shape["activation_bytes"],
            )

        if next_rank is None:
            builder.add(
                "cudaLaunchKernel",
                "kernel_launch",
                rank=rank,
                stream=0,
                code_partition="loss",
                bytes_=shape["logits_bytes"],
                count=local_micro_batch_size(case) * case.seq_len * case.vocab_size,
            )
        else:
            builder.add(
                "ncclSend",
                "mem_copy",
                rank=rank,
                stream=1,
                code_partition=f"stage_{pp_rank:03d}_p2p",
                bytes_=shape["activation_bytes"],
                count=max(shape["activation_bytes"] // dtype_bytes(case.dtype), 1),
                peer_rank=next_rank,
            )

    def backward_microbatch(step: int, microbatch: int) -> None:
        if next_rank is not None:
            builder.add(
                "ncclRecv",
                "mem_copy",
                rank=rank,
                stream=2,
                code_partition=f"stage_{pp_rank:03d}_p2p",
                bytes_=shape["activation_bytes"],
                count=max(shape["activation_bytes"] // dtype_bytes(case.dtype), 1),
                peer_rank=next_rank,
            )

        for local_layer in reversed(range(stage_layer_count(case, pp_rank))):
            layer = global_layer_id(case, pp_rank, local_layer)
            partition = f"layer_{layer:03d}"
            builder.add(
                "cublasGemmEx",
                "blas_compute",
                rank=rank,
                stream=3,
                code_partition=partition,
                bytes_=shape["activation_bytes"],
                count=shape["backward_ops_per_layer"] // 2,
            )
            builder.add(
                "cublasGemmEx",
                "blas_compute",
                rank=rank,
                stream=3,
                code_partition=partition,
                bytes_=shape["activation_bytes"],
                count=shape["backward_ops_per_layer"] // 2,
            )

        if prev_rank is not None:
            builder.add(
                "ncclSend",
                "mem_copy",
                rank=rank,
                stream=2,
                code_partition=f"stage_{pp_rank:03d}_p2p",
                bytes_=shape["activation_bytes"],
                count=max(shape["activation_bytes"] // dtype_bytes(case.dtype), 1),
                peer_rank=prev_rank,
            )

    for step in range(case.steps):
        if case.schedule == "gpipe":
            for microbatch in range(case.micro_batches):
                forward_microbatch(step, microbatch)
            for microbatch in range(case.micro_batches):
                backward_microbatch(step, microbatch)
        else:
            for microbatch in range(warmup):
                forward_microbatch(step, microbatch)
                queue.append(microbatch)
            for microbatch in range(warmup, case.micro_batches):
                forward_microbatch(step, microbatch)
                queue.append(microbatch)
                backward_microbatch(step, queue.pop(0))
            while queue:
                backward_microbatch(step, queue.pop(0))
        add_dp_all_reduce(
            builder,
            case,
            rank=rank,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            step=step,
            bytes_=shape["stage_param_bytes"],
        )
        builder.add(
            "cudaLaunchKernel",
            "kernel_launch",
            rank=rank,
            stream=0,
            code_partition=f"stage_{pp_rank:03d}_optimizer",
            bytes_=shape["stage_param_bytes"],
            count=shape["optimizer_ops"],
            duration_hint_us=5.0,
        )


def synthetic_megatron_events(case: MegatronCase) -> list[object]:
    if case.world_size != case.tp * case.pp * case.dp:
        raise ValueError(
            f"{case.name}: world_size={case.world_size} does not equal tp*pp*dp={case.tp * case.pp * case.dp}"
        )
    if case.hidden_size % case.num_heads != 0:
        raise ValueError(f"{case.name}: hidden_size must be divisible by num_heads")
    if case.hidden_size % case.tp != 0:
        raise ValueError(f"{case.name}: hidden_size must be divisible by tp")
    if case.num_heads % case.tp != 0:
        raise ValueError(f"{case.name}: num_heads must be divisible by tp")
    builder = EventBuilder(case)
    for rank in range(case.world_size):
        generate_rank_events(builder, case, rank)
    return builder.rows


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return time.perf_counter() - start, value


def event_signature(event: object) -> tuple[object, ...]:
    collective_group = str(getattr(event, "collective_group", ""))
    collective_base = collective_group.split("|occ:", 1)[0]
    api = str(getattr(event, "api", ""))
    kind = str(getattr(event, "kind", ""))
    return (
        api,
        kind,
        str(getattr(event, "code_partition", "")),
        int(getattr(event, "stream", 0)),
        int(getattr(event, "bytes", 0)),
        int(getattr(event, "count", 0)),
        collective_base if kind == "nccl_collective" else "",
    )


def weighted_event_counter(trace: object, *, use_dedup_weight: bool) -> Counter[tuple[object, ...]]:
    counter: Counter[tuple[object, ...]] = Counter()
    for event in trace.events:
        weight = int(getattr(event, "dedup_weight", 1) or 1) if use_dedup_weight else 1
        counter[event_signature(event)] += weight
    return counter


def filtered_weighted_event_counter(
    trace: object,
    *,
    use_dedup_weight: bool,
    kinds: frozenset[str] | None = None,
    excluded_apis: frozenset[str] = frozenset(),
) -> Counter[tuple[object, ...]]:
    counter: Counter[tuple[object, ...]] = Counter()
    for event in trace.events:
        if kinds is not None and str(getattr(event, "kind", "")) not in kinds:
            continue
        if str(getattr(event, "api", "")) in excluded_apis:
            continue
        weight = int(getattr(event, "dedup_weight", 1) or 1) if use_dedup_weight else 1
        counter[event_signature(event)] += weight
    return counter


def weighted_edge_counter(
    trace: object,
    *,
    reason: str | None = None,
) -> Counter[tuple[object, ...]]:
    events = {int(event.id): event for event in trace.events}
    counter: Counter[tuple[object, ...]] = Counter()
    for edge in trace.edges:
        edge_reason = str(edge.reason)
        if reason is not None and edge_reason != reason:
            continue
        source = events.get(int(edge.from_id))
        target = events.get(int(edge.to_id))
        if source is None or target is None:
            continue
        source_weight = int(getattr(source, "dedup_weight", 1) or 1)
        target_weight = int(getattr(target, "dedup_weight", 1) or 1)
        counter[(edge_reason, event_signature(source), event_signature(target))] += max(
            source_weight,
            target_weight,
        )
    return counter


def trace_window_counter(trace: object) -> Counter[tuple[object, ...]]:
    events_by_id = {int(event.id): event for event in trace.events}
    counter: Counter[tuple[object, ...]] = Counter()
    for partition in trace.sync_partitions:
        if str(partition.kind) != "trace_window":
            continue
        event_ids = tuple(int(event_id) for event_id in partition.event_ids)
        signatures = tuple(event_signature(events_by_id[event_id]) for event_id in event_ids if event_id in events_by_id)
        counter[(str(partition.code_partition), signatures)] += max(int(partition.logical_event_count), 1)
    return counter


def weighted_jaccard(left: Counter[tuple[object, ...]], right: Counter[tuple[object, ...]]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    numerator = sum(min(left.get(key, 0), right.get(key, 0)) for key in keys)
    denominator = sum(max(left.get(key, 0), right.get(key, 0)) for key in keys)
    return float(numerator) / float(denominator or 1)


def cosine(left: Counter[tuple[object, ...]], right: Counter[tuple[object, ...]]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    dot = sum(left.get(key, 0) * right.get(key, 0) for key in keys)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return float(dot) / float(left_norm * right_norm or 1.0)


def summarize_kind_counts(trace: object, *, use_dedup_weight: bool) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for event in trace.events:
        weight = int(getattr(event, "dedup_weight", 1) or 1) if use_dedup_weight else 1
        counts[str(getattr(event, "kind", ""))] += weight
    return dict(sorted(counts.items()))


def run_case(case: MegatronCase) -> dict[str, object]:
    spec = spec_for(case)
    event_gen_s, raw_events = timed(lambda: synthetic_megatron_events(case))
    maya_build_s, maya_trace = timed(lambda: fm.build_trace_ras(raw_events))
    flexeva_build_s, flexeva_trace = timed(
        lambda: fm.build_rank_grouped_trace_ras(raw_events, fm.active_lane_rank_groups(spec))
    )
    maya_replay_s, maya_feedback = timed(lambda: fm.replay_trace_once(maya_trace))
    flexeva_replay_s, flexeva_feedback = timed(lambda: fm.replay_trace_once(flexeva_trace))

    maya_weighted = weighted_event_counter(maya_trace, use_dedup_weight=True)
    flexeva_weighted = weighted_event_counter(flexeva_trace, use_dedup_weight=True)
    maya_physical = weighted_event_counter(maya_trace, use_dedup_weight=False)
    flexeva_physical = weighted_event_counter(flexeva_trace, use_dedup_weight=False)
    maya_windows = trace_window_counter(maya_trace)
    flexeva_windows = trace_window_counter(flexeva_trace)

    logical_coverage = int(flexeva_trace.logical_event_count) / max(int(maya_trace.logical_event_count), 1)
    compact_ratio = len(flexeva_trace.events) / max(len(maya_trace.events), 1)
    return {
        "case": asdict(case),
        "raw_events": len(raw_events),
        "maya": {
            "trace": fm.trace_summary(maya_trace),
            "feedback": maya_feedback.to_dict(),
            "logical_kind_counts": summarize_kind_counts(maya_trace, use_dedup_weight=True),
            "phases_s": {
                "event_generation_s": event_gen_s,
                "trace_build_s": maya_build_s,
                "replay_s": maya_replay_s,
            },
        },
        "flexeva": {
            "trace": fm.trace_summary(flexeva_trace),
            "feedback": flexeva_feedback.to_dict(),
            "logical_kind_counts": summarize_kind_counts(flexeva_trace, use_dedup_weight=True),
            "rank_groups_from_active_lane_sets": fm.active_lane_rank_groups(spec),
            "phases_s": {
                "trace_build_s": flexeva_build_s,
                "replay_s": flexeva_replay_s,
            },
        },
        "similarity": {
            "logical_event_coverage": logical_coverage,
            "logical_event_count_equal": int(maya_trace.logical_event_count) == int(flexeva_trace.logical_event_count),
            "weighted_event_jaccard": weighted_jaccard(maya_weighted, flexeva_weighted),
            "weighted_event_cosine": cosine(maya_weighted, flexeva_weighted),
            "trace_window_weighted_jaccard": weighted_jaccard(maya_windows, flexeva_windows),
            "physical_event_jaccard_without_dedup_weights": weighted_jaccard(maya_physical, flexeva_physical),
            "compact_event_ratio_flexeva_vs_maya": compact_ratio,
        },
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = default_cases()
    if args.case:
        selected = set(args.case)
        cases = tuple(case for case in cases if case.name in selected)
    rows = [run_case(case) for case in cases]
    result = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": {
            "maya": "build_trace_ras over all Megatron ranks",
            "flexeva": "active-lane rank grouped compact trace using FlexEva/Maya RAS",
            "primary_similarity": "weighted_event_jaccard and weighted_event_cosine use dedup_weight to compare logical traces",
        },
        "results": rows,
    }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "parameter_scale",
                "world_size",
                "tp",
                "pp",
                "dp",
                "maya_events",
                "flexeva_events",
                "maya_logical_events",
                "flexeva_logical_events",
                "logical_event_coverage",
                "weighted_event_jaccard",
                "weighted_event_cosine",
                "trace_window_weighted_jaccard",
                "physical_event_jaccard_without_dedup_weights",
                "compact_event_ratio_flexeva_vs_maya",
            ],
        )
        writer.writeheader()
        for row in rows:
            case = row["case"]
            similarity = row["similarity"]
            writer.writerow(
                {
                    "case": case["name"],
                    "parameter_scale": case["parameter_scale"],
                    "world_size": case["world_size"],
                    "tp": case["tp"],
                    "pp": case["pp"],
                    "dp": case["dp"],
                    "maya_events": row["maya"]["trace"]["event_count"],
                    "flexeva_events": row["flexeva"]["trace"]["event_count"],
                    "maya_logical_events": row["maya"]["trace"]["logical_event_count"],
                    "flexeva_logical_events": row["flexeva"]["trace"]["logical_event_count"],
                    "logical_event_coverage": similarity["logical_event_coverage"],
                    "weighted_event_jaccard": similarity["weighted_event_jaccard"],
                    "weighted_event_cosine": similarity["weighted_event_cosine"],
                    "trace_window_weighted_jaccard": similarity["trace_window_weighted_jaccard"],
                    "physical_event_jaccard_without_dedup_weights": similarity[
                        "physical_event_jaccard_without_dedup_weights"
                    ],
                    "compact_event_ratio_flexeva_vs_maya": similarity["compact_event_ratio_flexeva_vs_maya"],
                }
            )
    print(json.dumps({"result": str(args.out_dir / "result.json"), "summary": str(args.out_dir / "summary.csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
