#!/usr/bin/env python3
"""
Paper-facing soft-MoE training workload for real/fake trace capture.

This keeps the workload stable and DDP-safe while matching the current
Maya-lite real-trace conventions:

- optional unmarked warmup step(s)
- measured step wrapped by trace markers
- CUDA synchronize before closing the measured step window by default
- compatibility with the standard supersys real-trace sweep arguments
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from cluster_rdma import maybe_apply_cluster_cpu_affinity, maybe_apply_cluster_rdma_affinity
from flexsim.maya_lite.markers import step_window


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Soft-MoE training with DDP")
    ap.add_argument("--steps", type=int, default=1, help="Measured training steps")
    ap.add_argument("--warmup-steps", type=int, default=1, help="Unmarked warmup steps")
    ap.add_argument("--batch-size", type=int, default=4, help="Per-rank batch size")
    ap.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Optional global batch size; when set, per-rank batch is derived from dp/world size.",
    )
    ap.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    ap.add_argument("--hidden-size", type=int, default=256, help="Hidden dimension")
    ap.add_argument("--num-experts", type=int, default=4, help="Number of experts")
    ap.add_argument("--num-layers", type=int, default=4, help="Number of MoE layers")
    ap.add_argument("--num-heads", type=int, default=4, help="Attention heads")
    ap.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    ap.add_argument("--dp", type=int, default=1, help="Data parallel degree")
    ap.add_argument("--tp", type=int, default=1, help="Tensor parallel degree (must stay 1)")
    ap.add_argument("--pp", type=int, default=1, help="Pipeline parallel degree (must stay 1)")
    ap.add_argument("--micro-batches", type=int, default=1, help="Accepted for sweep compatibility")
    ap.add_argument("--schedule", default="1f1b", help="Accepted for sweep compatibility")
    ap.add_argument(
        "--pipeline-p2p-mode",
        default="blocking",
        help="Accepted for sweep compatibility; unused by the DDP-only soft-MoE path.",
    )
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--log-interval", type=int, default=1, help="Log every N steps")
    ap.add_argument(
        "--no-step-end-synchronize",
        action="store_true",
        help="Do not synchronize CUDA before closing the measured step marker.",
    )
    return ap.parse_args()


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
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size, local_rank, device, rdma_hca, rdma_iface


def runtime_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.dtype == "fp32":
        return torch.float32
    if args.dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {args.dtype}")


def local_batch_size(args: argparse.Namespace, world_size: int) -> int:
    if args.global_batch_size is None:
        return max(1, int(args.batch_size))
    logical_dp = max(int(args.dp), 1)
    return max(1, math.ceil(int(args.global_batch_size) / logical_dp))


def synchronize_completed_iteration(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class Expert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))


class SoftMoELayer(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        intermediate_size = hidden_size * 4
        self.experts = nn.ModuleList(
            [Expert(hidden_size, intermediate_size) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.router(x)
        router_weights = F.softmax(router_logits, dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        output = torch.einsum("bshe,bse->bsh", expert_outputs, router_weights)
        aux_loss = router_weights.mean(dim=(0, 1)).var()
        return output, aux_loss


class MoETransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_experts: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.moe = SoftMoELayer(hidden_size, num_experts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h

        h = self.ln2(x)
        h, aux_loss = self.moe(h)
        x = x + h
        return x, aux_loss


class MoEModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_experts: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.blocks = nn.ModuleList(
            [
                MoETransformerBlock(hidden_size, num_heads, num_experts)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, seq_len = input_ids.shape
        tok_emb = self.token_emb(input_ids)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=input_ids.device))
        x = tok_emb + pos_emb

        total_aux_loss = input_ids.new_zeros((), dtype=torch.float32)
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss = total_aux_loss + aux_loss.float()

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, total_aux_loss / len(self.blocks)


def run_step(
    *,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[float, float]:
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    logits, aux_loss = model(input_ids)
    ce_loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), labels.reshape(-1))
    loss = ce_loss + 0.01 * aux_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(ce_loss.item()), float(aux_loss.item())


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device, rdma_hca, rdma_iface = setup_dist()

    if int(args.tp) != 1 or int(args.pp) != 1:
        raise SystemExit("soft MoE real-trace workload currently supports tp=1 and pp=1 only")
    expected_world_size = max(int(args.dp), 1) * max(int(args.tp), 1) * max(int(args.pp), 1)
    if world_size != expected_world_size:
        raise SystemExit(
            f"world_size={world_size} does not match dp*tp*pp={expected_world_size}"
        )
    if args.dtype == "bf16" and device.type != "cuda":
        raise SystemExit("--dtype bf16 requires CUDA")
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA device does not report bf16 support")

    torch.manual_seed(42 + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + rank)

    batch_size = local_batch_size(args, world_size)
    param_dtype = runtime_dtype(args)
    model = MoEModel(
        vocab_size=int(args.vocab_size),
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        num_experts=int(args.num_experts),
        max_seq_len=int(args.seq_len),
    ).to(device=device, dtype=param_dtype)
    model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(
            {
                "world_size": world_size,
                "dp": args.dp,
                "tp": args.tp,
                "pp": args.pp,
                "global_batch_size": args.global_batch_size,
                "local_batch_size": batch_size,
                "num_experts": args.num_experts,
                "routing": "soft",
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dtype": args.dtype,
                "warmup_steps": args.warmup_steps,
                "step_end_synchronize": not args.no_step_end_synchronize,
                "rdma_hca": rdma_hca,
                "rdma_iface": rdma_iface,
                "params_m": round(num_params / 1e6, 2),
            }
        )

    start_time = time.time()
    for warmup_step in range(1, max(int(args.warmup_steps), 0) + 1):
        run_step(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            seq_len=int(args.seq_len),
            vocab_size=int(args.vocab_size),
            device=device,
        )
        if not args.no_step_end_synchronize:
            synchronize_completed_iteration(device)
        if rank == 0 and warmup_step == max(int(args.warmup_steps), 0):
            print(f"warmup {warmup_step:4d}/{args.warmup_steps} complete")

    for step in range(1, args.steps + 1):
        with step_window(step):
            ce_loss, aux_loss = run_step(
                model=model,
                optimizer=optimizer,
                batch_size=batch_size,
                seq_len=int(args.seq_len),
                vocab_size=int(args.vocab_size),
                device=device,
            )
            if not args.no_step_end_synchronize:
                synchronize_completed_iteration(device)
        if rank == 0 and (step % args.log_interval == 0 or step == args.steps):
            elapsed = time.time() - start_time
            print(
                f"step {step:4d}/{args.steps} | elapsed_s={elapsed:.3f} | "
                f"CE={ce_loss:.4f} | aux={aux_loss:.4f}"
            )

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("Training complete.")


if __name__ == "__main__":
    main()
