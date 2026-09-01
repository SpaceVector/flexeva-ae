#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

from cluster_rdma import maybe_apply_cluster_cpu_affinity, maybe_apply_cluster_rdma_affinity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal NCCL all-reduce smoke workload")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--tensor-mb", type=int, default=32)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--log-interval", type=int, default=1)
    return parser.parse_args()


def runtime_dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def setup_dist() -> tuple[int, int, int, torch.device, str | None, str | None, list[int] | None]:
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

    cpu_affinity = maybe_apply_cluster_cpu_affinity(local_rank)
    rdma_hca, rdma_iface = maybe_apply_cluster_rdma_affinity(local_rank)
    if device.type == "cuda":
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=device)
    else:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    return rank, world_size, local_rank, device, rdma_hca, rdma_iface, cpu_affinity


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device, rdma_hca, rdma_iface, cpu_affinity = setup_dist()
    dtype = runtime_dtype(args.dtype)
    element_size = torch.tensor([], dtype=dtype).element_size()
    numel = max(1, (args.tensor_mb * 1024 * 1024) // element_size)
    tensor = torch.ones(numel, device=device, dtype=dtype) * (rank + 1)

    if rank == 0:
        print(
            {
                "world_size": world_size,
                "tensor_mb": args.tensor_mb,
                "dtype": args.dtype,
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
            "cpu_affinity_count": 0 if cpu_affinity is None else len(cpu_affinity),
        }
    )

    dist.barrier()
    start = time.time()
    for step in range(1, args.steps + 1):
        tensor.fill_(rank + 1)
        dist.all_reduce(tensor)
        expected = float(sum(range(1, world_size + 1)))
        if not torch.allclose(tensor.mean(), torch.tensor(expected, device=device, dtype=dtype), rtol=1e-3, atol=1e-3):
            raise RuntimeError(
                f"rank {rank}: unexpected all_reduce result {tensor.mean().item()} != {expected}"
            )
        if rank == 0 and (step % args.log_interval == 0 or step == args.steps):
            elapsed = time.time() - start
            print(f"step {step}/{args.steps} | elapsed={elapsed:.3f}s | mean={tensor.mean().item():.3f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
