#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from cluster_rdma import maybe_apply_cluster_cpu_affinity, maybe_apply_cluster_rdma_affinity
from flexsim.maya_lite.markers import step_window


_DIST_GROUP_TIMEOUT = timedelta(minutes=30)


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected positive integer, got {raw!r}")
    return value


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected nonnegative integer, got {raw!r}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal CUDA event and collective wait-map diagnostic workload"
    )
    parser.add_argument("--steps", type=_positive_int, default=5)
    parser.add_argument("--event-repeats", type=_positive_int, default=5)
    parser.add_argument("--collective-repeats", type=_positive_int, default=5)
    parser.add_argument("--tensor-elements", type=_positive_int, default=1_048_576)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument(
        "--event-record-prework-repeats",
        type=_nonnegative_int,
        default=1,
        help=(
            "Record-stream host-to-device copy batches before cudaEventRecord. "
            "This is a structural queue-shape knob; the gate, not this "
            "default, proves whether waits block in replay."
        ),
    )
    parser.add_argument(
        "--event-record-prework-copy-elements",
        type=_positive_int,
        default=None,
        help=(
            "Optional element count for the record-stream H2D copy tensor used "
            "before cudaEventRecord. Defaults to --tensor-elements. This is a "
            "structural gate knob for making producer work replay-visible "
            "without enlarging collective or wait-stream tensors."
        ),
    )
    parser.add_argument(
        "--event-wait-prework-repeats",
        type=_nonnegative_int,
        default=0,
        help="Optional wait-stream work before cudaStreamWaitEvent; default is an immediate wait.",
    )
    parser.add_argument(
        "--event-postwork-repeats",
        type=_nonnegative_int,
        default=1,
        help=(
            "Wait-stream work batches after cudaStreamWaitEvent. Postwork is "
            "enqueued only after all waits have been issued."
        ),
    )
    parser.add_argument(
        "--sync-before-step-window",
        action="store_true",
        help="Synchronize ranks immediately before each measured step marker.",
    )
    return parser.parse_args()


def runtime_dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def setup_dist() -> tuple[int, int, int, torch.device, str | None, str | None]:
    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    maybe_apply_cluster_cpu_affinity(local_rank)
    rdma_hca, rdma_iface = maybe_apply_cluster_rdma_affinity(local_rank)
    if device.type == "cuda":
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            device_id=device,
            timeout=_DIST_GROUP_TIMEOUT,
        )
    else:
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            timeout=_DIST_GROUP_TIMEOUT,
        )
    return rank, world_size, local_rank, device, rdma_hca, rdma_iface


def synchronize_before_step_window() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _make_event_pool(steps: int, repeats: int) -> list[list[torch.cuda.Event]]:
    return [
        [torch.cuda.Event(blocking=False, enable_timing=False) for _ in range(repeats)]
        for _ in range(steps)
    ]


def run_cuda_event_path(
    *,
    events: list[torch.cuda.Event],
    record_stream: torch.cuda.Stream,
    wait_streams: list[torch.cuda.Stream],
    record_tensor: torch.Tensor,
    record_prework_device_tensor: torch.Tensor,
    wait_tensors: list[torch.Tensor],
    record_prework_host_tensor: torch.Tensor,
    record_prework_repeats: int,
    wait_prework_repeats: int,
    postwork_repeats: int,
) -> None:
    if len(wait_streams) < len(events) or len(wait_tensors) < len(events):
        raise ValueError("wait stream/tensor pools must cover every event repeat")

    # Keep each event record and matching wait adjacent, then defer all
    # wait-stream postwork.  This keeps waits close to producer enqueue time and
    # prevents later waits from hiding behind earlier postwork.
    for repeat_index, event in enumerate(events):
        with torch.cuda.stream(record_stream):
            for _ in range(record_prework_repeats):
                record_prework_device_tensor.copy_(
                    record_prework_host_tensor,
                    non_blocking=True,
                )
            event.record(record_stream)

        wait_stream = wait_streams[repeat_index]
        wait_tensor = wait_tensors[repeat_index]
        with torch.cuda.stream(wait_stream):
            for prework_index in range(wait_prework_repeats):
                wait_tensor.add_(0.25 + repeat_index + 0.001 * prework_index)
                wait_tensor.mul_(0.9999)
        wait_stream.wait_event(event)

    for wait_stream, wait_tensor in zip(wait_streams, wait_tensors):
        with torch.cuda.stream(wait_stream):
            for _ in range(postwork_repeats):
                wait_tensor.add_(record_tensor, alpha=0.03125)
                wait_tensor.mul_(1.0002)


def run_collective_path(
    *,
    rank: int,
    world_size: int,
    repeats: int,
    collective_stream: torch.cuda.Stream,
    collective_tensor: torch.Tensor,
) -> None:
    for repeat_index in range(repeats):
        with torch.cuda.stream(collective_stream):
            collective_tensor.fill_(rank + 1)
            collective_tensor.add_(repeat_index)
            for _ in range(rank):
                collective_tensor.mul_(1.0001)
            dist.all_reduce(collective_tensor)
            collective_tensor.mul_(1.0 / world_size)
            collective_tensor.add_(0.5)


def run_step(
    *,
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    events: list[torch.cuda.Event],
    record_stream: torch.cuda.Stream,
    wait_streams: list[torch.cuda.Stream],
    collective_stream: torch.cuda.Stream,
    record_tensor: torch.Tensor,
    record_prework_device_tensor: torch.Tensor,
    wait_tensors: list[torch.Tensor],
    record_prework_host_tensor: torch.Tensor,
    collective_tensor: torch.Tensor,
) -> None:
    run_cuda_event_path(
        events=events,
        record_stream=record_stream,
        wait_streams=wait_streams,
        record_tensor=record_tensor,
        record_prework_device_tensor=record_prework_device_tensor,
        wait_tensors=wait_tensors,
        record_prework_host_tensor=record_prework_host_tensor,
        record_prework_repeats=args.event_record_prework_repeats,
        wait_prework_repeats=args.event_wait_prework_repeats,
        postwork_repeats=args.event_postwork_repeats,
    )
    run_collective_path(
        rank=rank,
        world_size=world_size,
        repeats=args.collective_repeats,
        collective_stream=collective_stream,
        collective_tensor=collective_tensor,
    )


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device, rdma_hca, rdma_iface = setup_dist()
    try:
        if world_size != 2:
            raise SystemExit(f"expected exactly 2 ranks, got world_size={world_size}")
        if device.type != "cuda":
            raise SystemExit("CUDA is required for the event wait-map diagnostic")

        dtype = runtime_dtype(args.dtype)
        torch.manual_seed(20260507 + rank)
        torch.cuda.manual_seed_all(20260507 + rank)

        record_stream = torch.cuda.Stream(device=device)
        wait_streams = [
            torch.cuda.Stream(device=device) for _ in range(args.event_repeats)
        ]
        collective_stream = torch.cuda.Stream(device=device)
        event_pool = _make_event_pool(args.steps, args.event_repeats)

        record_tensor = torch.full(
            (args.tensor_elements,),
            fill_value=rank + 1,
            device=device,
            dtype=dtype,
        )
        wait_tensors = [
            torch.full_like(record_tensor, fill_value=2 * (rank + 1) + repeat)
            for repeat in range(args.event_repeats)
        ]
        record_prework_elements = (
            args.event_record_prework_copy_elements
            if args.event_record_prework_copy_elements is not None
            else args.tensor_elements
        )
        record_prework_device_tensor = torch.empty(
            (record_prework_elements,),
            device=device,
            dtype=dtype,
        )
        record_prework_host_tensor = torch.full(
            (record_prework_elements,),
            fill_value=0.25 + rank,
            device="cpu",
            dtype=dtype,
        )
        collective_tensor = torch.empty_like(record_tensor)

        if rank == 0:
            print(
                {
                    "workload": "cuda_event_collective_wait_map_minimal",
                    "world_size": world_size,
                    "steps": args.steps,
                    "event_repeats": args.event_repeats,
                    "collective_repeats": args.collective_repeats,
                    "tensor_elements": args.tensor_elements,
                    "dtype": args.dtype,
                    "event_record_prework_repeats": args.event_record_prework_repeats,
                    "event_record_prework_copy_elements": record_prework_elements,
                    "event_wait_prework_repeats": args.event_wait_prework_repeats,
                    "event_postwork_repeats": args.event_postwork_repeats,
                    "event_wait_streams": len(wait_streams),
                    "sync_before_step_window": args.sync_before_step_window,
                    "rdma_hca": rdma_hca,
                    "rdma_iface": rdma_iface,
                }
            )
        print(
            {
                "rank": rank,
                "local_rank": local_rank,
                "device": str(device),
                "rdma_hca": rdma_hca,
                "rdma_iface": rdma_iface,
            }
        )

        dist.barrier()
        torch.cuda.synchronize(device)
        for step in range(1, args.steps + 1):
            if args.sync_before_step_window:
                synchronize_before_step_window()
            with step_window(step):
                run_step(
                    rank=rank,
                    world_size=world_size,
                    args=args,
                    events=event_pool[step - 1],
                    record_stream=record_stream,
                    wait_streams=wait_streams,
                    collective_stream=collective_stream,
                    record_tensor=record_tensor,
                    record_prework_device_tensor=record_prework_device_tensor,
                    wait_tensors=wait_tensors,
                    record_prework_host_tensor=record_prework_host_tensor,
                    collective_tensor=collective_tensor,
                )
                torch.cuda.synchronize(device)

        dist.barrier()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
