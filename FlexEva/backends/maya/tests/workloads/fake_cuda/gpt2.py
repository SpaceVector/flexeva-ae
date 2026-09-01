#!/usr/bin/env python3
import argparse
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

try:
    from transformers import GPT2Config, GPT2LMHeadModel
except Exception as e:
    raise SystemExit(
        "Please install transformers first: pip install transformers"
    )


def parse_args():
    ap = argparse.ArgumentParser(description="Minimal GPT-2 DP training (DDP)")
    ap.add_argument("--steps", type=int, default=100, help="Number of optimizer steps")
    ap.add_argument("--batch-size", type=int, default=4, help="Per-rank batch size")
    ap.add_argument("--seq-len", type=int, default=1024, help="Sequence length (<= 1024 for GPT-2 base)")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate for AdamW")
    ap.add_argument("--seed", type=int, default=1337, help="Base RNG seed")
    ap.add_argument("--log-interval", type=int, default=10, help="Print loss every N steps (rank 0)")
    ap.add_argument("--save-dir", type=str, default="", help="Optional directory to save final checkpoint (rank 0)")
    return ap.parse_args()


def setup_dist():
    if "RANK" not in os.environ:
        # If not using torchrun, default to single process for convenience
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

    rank = int(os.environ["RANK"])  # global rank
    world_size = int(os.environ["WORLD_SIZE"])  # global world size
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=device)
    else:
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    return rank, world_size, local_rank, device


def build_gpt2_model(seq_len: int, device: torch.device) -> GPT2LMHeadModel:
    # Canonical GPT-2 base config (12 layers, 12 heads, 768 hidden) with default vocab 50257.
    # We keep the architecture identical; assert context window compatibility.
    config = GPT2Config()  # defaults: n_layer=12, n_head=12, n_embd=768, n_positions=1024
    if seq_len > config.n_positions:
        raise ValueError(
            f"seq_len={seq_len} exceeds GPT-2 base context window ({config.n_positions})."
        )
    model = GPT2LMHeadModel(config)
    model.to(device)
    return model


def main():
    args = parse_args()
    rank, world_size, local_rank, device = setup_dist()

    # Set (rank-differentiated) seeds for reproducibility and DP diversity
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    model = build_gpt2_model(args.seq_len, device)
    model.train()

    # Wrap with DDP (DP only)
    model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
                find_unused_parameters=False)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Simple training loop with synthetic tokens
    vocab_size = model.module.config.vocab_size if hasattr(model, "module") else model.config.vocab_size
    bsz = args.batch_size
    S = args.seq_len

    if rank == 0:
        print({
            "world_size": world_size,
            "per_rank_batch": bsz,
            "effective_global_batch": bsz * world_size,
            "seq_len": S,
            "model": "gpt2 (base)",
            "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        })

    start_time = time.time()
    for step in range(1, args.steps + 1):
        # Generate synthetic input directly on device (diversified by rank seed)
        input_ids = torch.randint(low=0, high=vocab_size, size=(bsz, S), device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # GPT-2 causal LM loss when labels=input_ids
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss  # scalar

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if rank == 0 and (step % args.log_interval == 0 or step == 1 or step == args.steps):
            elapsed = time.time() - start_time
            tps = (step * bsz * world_size) / max(elapsed, 1e-6)
            print(f"step {step:5d}/{args.steps} | loss {loss.item():.4f} | tokens/s {tps * S:,.0f}")

    # Optional checkpoint (rank 0 only)
    if args.save_dir and rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        # unwrap DDP
        to_save = model.module if hasattr(model, "module") else model
        to_save.save_pretrained(args.save_dir)
        print(f"Saved checkpoint to {args.save_dir}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
