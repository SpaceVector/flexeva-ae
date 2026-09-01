#!/usr/bin/env python3
"""
GPT 3D Parallel training: Data Parallel + Tensor Parallel + Pipeline Parallel.

This demonstrates hybrid parallelism commonly used for large model training.
Uses real DDP and NCCL for all communication.

Parallelism layout (example with 8 GPUs):
    DP=2, TP=2, PP=2

    Stage 0 (layers 0-5):    Stage 1 (layers 6-11):
    ┌─────────────────┐      ┌─────────────────┐
    │ DP0,TP0 │ DP0,TP1│      │ DP0,TP0 │ DP0,TP1│
    │ GPU 0   │ GPU 1  │  →   │ GPU 2   │ GPU 3  │
    ├─────────┼────────┤      ├─────────┼────────┤
    │ DP1,TP0 │ DP1,TP1│      │ DP1,TP0 │ DP1,TP1│
    │ GPU 4   │ GPU 5  │  →   │ GPU 6   │ GPU 7  │
    └─────────────────┘      └─────────────────┘

Run with:
    torchrun --nproc_per_node=4 gpt_3d.py --dp 2 --tp 2 --pp 1 --steps 10

With fake-cuda:
    ./frun torchrun --nproc_per_node=4 test/gpt_3d.py --dp 2 --tp 2 --pp 1
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def parse_args():
    ap = argparse.ArgumentParser(description="GPT 3D Parallel (DP+TP+PP)")
    ap.add_argument("--steps", type=int, default=50, help="Training steps")
    ap.add_argument("--batch-size", type=int, default=4, help="Per-rank batch size")
    ap.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    ap.add_argument("--hidden-size", type=int, default=256, help="Hidden dimension")
    ap.add_argument("--num-layers", type=int, default=8, help="Total transformer layers")
    ap.add_argument("--num-heads", type=int, default=8, help="Attention heads")
    ap.add_argument("--dp", type=int, default=1, help="Data parallel degree")
    ap.add_argument("--tp", type=int, default=1, help="Tensor parallel degree")
    ap.add_argument("--pp", type=int, default=1, help="Pipeline parallel degree")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    return ap.parse_args()


def setup_dist():
    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return rank, world_size, local_rank, device


def get_parallel_ranks(rank: int, dp: int, tp: int, pp: int):
    """
    Compute parallel group membership for a given global rank.
    Layout: [dp_rank, tp_rank, pp_rank] where rank = dp_rank * (tp * pp) + tp_rank * pp + pp_rank
    """
    pp_rank = rank % pp
    tp_rank = (rank // pp) % tp
    dp_rank = rank // (tp * pp)
    return dp_rank, tp_rank, pp_rank


class ColumnParallelLinear(nn.Module):
    """Linear layer with column-wise tensor parallelism."""
    def __init__(self, in_features: int, out_features: int, tp_size: int, tp_rank: int):
        super().__init__()
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        assert out_features % tp_size == 0
        self.local_out_features = out_features // tp_size
        self.linear = nn.Linear(in_features, self.local_out_features, bias=False)

    def forward(self, x):
        return self.linear(x)


class RowParallelLinear(nn.Module):
    """Linear layer with row-wise tensor parallelism."""
    def __init__(self, in_features: int, out_features: int, tp_size: int, tp_rank: int, tp_group=None):
        super().__init__()
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group
        assert in_features % tp_size == 0
        self.local_in_features = in_features // tp_size
        self.linear = nn.Linear(self.local_in_features, out_features, bias=False)

    def forward(self, x):
        out = self.linear(x)
        if self.tp_size > 1 and self.tp_group is not None:
            dist.all_reduce(out, group=self.tp_group)
        return out


class TensorParallelAttention(nn.Module):
    """Multi-head attention with tensor parallelism on heads."""
    def __init__(self, hidden_size: int, num_heads: int, tp_size: int, tp_rank: int, tp_group=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group

        assert num_heads % tp_size == 0
        self.local_heads = num_heads // tp_size
        self.head_dim = hidden_size // num_heads
        self.local_hidden = self.local_heads * self.head_dim

        # QKV projection (column parallel)
        self.qkv = ColumnParallelLinear(hidden_size, 3 * hidden_size, tp_size, tp_rank)
        # Output projection (row parallel)
        self.out_proj = RowParallelLinear(hidden_size, hidden_size, tp_size, tp_rank, tp_group)

    def forward(self, x):
        batch, seq, _ = x.shape

        # QKV (local heads only)
        qkv = self.qkv(x)  # (batch, seq, 3 * local_hidden)
        qkv = qkv.view(batch, seq, 3, self.local_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Transpose for attention: (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.local_hidden)
        out = self.out_proj(out)  # Includes all-reduce for TP

        return out


class TensorParallelMLP(nn.Module):
    """MLP with tensor parallelism."""
    def __init__(self, hidden_size: int, tp_size: int, tp_rank: int, tp_group=None):
        super().__init__()
        intermediate_size = hidden_size * 4
        # Column parallel for first linear (split output)
        self.fc1 = ColumnParallelLinear(hidden_size, intermediate_size, tp_size, tp_rank)
        # Row parallel for second linear (split input, all-reduce output)
        self.fc2 = RowParallelLinear(intermediate_size, hidden_size, tp_size, tp_rank, tp_group)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Transformer block with optional tensor parallelism."""
    def __init__(self, hidden_size: int, num_heads: int, tp_size: int, tp_rank: int, tp_group=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = TensorParallelAttention(hidden_size, num_heads, tp_size, tp_rank, tp_group)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = TensorParallelMLP(hidden_size, tp_size, tp_rank, tp_group)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT3DModel(nn.Module):
    """GPT model with 3D parallelism support."""
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_heads: int,
                 max_seq_len: int, pp_rank: int, pp_size: int, tp_size: int, tp_rank: int, tp_group=None):
        super().__init__()
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.hidden_size = hidden_size

        # Compute layer assignment for pipeline parallelism
        layers_per_stage = num_layers // pp_size
        self.start_layer = pp_rank * layers_per_stage
        self.end_layer = self.start_layer + layers_per_stage

        # Only first stage has embeddings
        if pp_rank == 0:
            self.token_emb = nn.Embedding(vocab_size, hidden_size)
            self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        else:
            self.token_emb = None
            self.pos_emb = None

        # Each stage has its assigned layers
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, tp_size, tp_rank, tp_group)
            for _ in range(layers_per_stage)
        ])

        # Only last stage has output head
        if pp_rank == pp_size - 1:
            self.ln_f = nn.LayerNorm(hidden_size)
            self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        else:
            self.ln_f = None
            self.head = None

    def forward(self, x, input_ids=None):
        """
        Forward pass for this pipeline stage.

        Args:
            x: Hidden states from previous stage (None for first stage)
            input_ids: Token IDs (only used by first stage)
        """
        # First stage: embed input
        if self.pp_rank == 0:
            assert input_ids is not None
            batch, seq = input_ids.shape
            tok_emb = self.token_emb(input_ids)
            pos_emb = self.pos_emb(torch.arange(seq, device=input_ids.device))
            x = tok_emb + pos_emb

        # Process layers for this stage
        for block in self.blocks:
            x = block(x)

        # Last stage: output logits
        if self.pp_rank == self.pp_size - 1:
            x = self.ln_f(x)
            logits = self.head(x)
            return logits

        return x


def main():
    args = parse_args()
    rank, world_size, local_rank, device = setup_dist()

    # Validate parallelism config
    expected_world = args.dp * args.tp * args.pp
    if world_size != expected_world:
        if rank == 0:
            print(f"Warning: world_size={world_size} != dp*tp*pp={expected_world}")
            print(f"Adjusting to dp={world_size}, tp=1, pp=1")
        args.dp = world_size
        args.tp = 1
        args.pp = 1

    dp_rank, tp_rank, pp_rank = get_parallel_ranks(rank, args.dp, args.tp, args.pp)

    # Create process groups for tensor parallelism
    tp_group = None
    if args.tp > 1:
        for dp_r in range(args.dp):
            for pp_r in range(args.pp):
                tp_ranks = [dp_r * args.tp * args.pp + tp_r * args.pp + pp_r for tp_r in range(args.tp)]
                group = dist.new_group(tp_ranks)
                if dp_rank == dp_r and pp_rank == pp_r:
                    tp_group = group

    # Create process groups for data parallelism (DDP will use this)
    dp_group = None
    if args.dp > 1:
        for tp_r in range(args.tp):
            for pp_r in range(args.pp):
                dp_ranks = [dp_r * args.tp * args.pp + tp_r * args.pp + pp_r for dp_r in range(args.dp)]
                group = dist.new_group(dp_ranks)
                if tp_rank == tp_r and pp_rank == pp_r:
                    dp_group = group

    torch.manual_seed(42 + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + rank)

    vocab_size = 32000

    model = GPT3DModel(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
        pp_rank=pp_rank,
        pp_size=args.pp,
        tp_size=args.tp,
        tp_rank=tp_rank,
        tp_group=tp_group,
    ).to(device)

    # Wrap with DDP for data parallelism (within DP group)
    if args.dp > 1 and dp_group is not None:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                    process_group=dp_group)
    else:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    num_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print({
            "world_size": world_size,
            "dp": args.dp,
            "tp": args.tp,
            "pp": args.pp,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "params_m": round(num_params / 1e6, 2),
            "layers_per_stage": args.num_layers // args.pp,
        })

    # Training loop
    start_time = time.time()
    for step in range(1, args.steps + 1):
        # Synthetic data
        input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)
        labels = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)

        # Forward pass (simplified - full PP would use micro-batching)
        if pp_rank == 0:
            hidden = model(x=None, input_ids=input_ids)
        else:
            # In real PP, receive activations from previous stage
            # Here we just create dummy hidden states for demonstration
            hidden = torch.zeros(args.batch_size, args.seq_len, args.hidden_size, device=device)
            hidden = model(x=hidden, input_ids=None)

        # Loss (only last PP stage computes real loss)
        if pp_rank == args.pp - 1:
            loss = F.cross_entropy(hidden.view(-1, vocab_size), labels.view(-1))
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if rank == 0 and (step % args.log_interval == 0 or step == 1 or step == args.steps):
            elapsed = time.time() - start_time
            print(f"step {step:5d}/{args.steps} | loss={loss.item():.4f}")

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        print("Training complete.")


if __name__ == "__main__":
    main()
