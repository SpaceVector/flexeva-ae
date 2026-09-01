#!/usr/bin/env python3
"""
GPT-2 base style PP+DP workload for Maya Figure 13 style capture.

This keeps the worker/rank execution flow close to Maya's paper-facing setup:

- fixed pipeline-parallel and data-parallel decomposition
- one representative worker per pipeline stage under dedup
- explicit microbatch pipeline schedule
- point-to-point pipeline traffic between adjacent stages
- data-parallel gradient synchronization after microbatch accumulation
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import deque
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from flexsim.maya_lite.markers import step_window

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPT-2 base style PP+DP workload for Maya Figure 13"
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=50_257)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=8)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--micro-batches", type=int, default=64)
    parser.add_argument("--schedule", choices=["1f1b", "gpipe"], default="1f1b")
    parser.add_argument(
        "--pipeline-p2p-mode",
        choices=["blocking", "async", "batch"],
        default="blocking",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log-interval", type=int, default=1)
    return parser.parse_args()


def setup_dist() -> tuple[int, int, int, torch.device]:
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


def rank_to_coords(rank: int, dp: int, tp: int, pp: int) -> tuple[int, int, int]:
    del dp
    pp_rank = rank % pp
    tp_rank = (rank // pp) % tp
    dp_rank = rank // (tp * pp)
    return dp_rank, tp_rank, pp_rank


def coords_to_rank(dp_rank: int, tp_rank: int, pp_rank: int, tp: int, pp: int) -> int:
    return dp_rank * (tp * pp) + tp_rank * pp + pp_rank


def stage_layer_count(num_layers: int, pp: int, pp_rank: int) -> int:
    base = num_layers // pp
    remainder = num_layers % pp
    return base + (1 if pp_rank < remainder else 0)


def local_micro_batch_size(global_batch_size: int, dp: int, micro_batches: int) -> int:
    return max(1, math.ceil(global_batch_size / max(dp * micro_batches, 1)))


def build_dp_group(
    *,
    tp_rank: int,
    pp_rank: int,
    dp: int,
    tp: int,
    pp: int,
):
    selected = None
    for current_tp_rank in range(tp):
        for current_pp_rank in range(pp):
            ranks = [
                coords_to_rank(dp_rank, current_tp_rank, current_pp_rank, tp, pp)
                for dp_rank in range(dp)
            ]
            group = dist.new_group(ranks)
            if current_tp_rank == tp_rank and current_pp_rank == pp_rank:
                selected = group
    return selected


class GPT2PipelineStageModel(nn.Module):
    def __init__(
        self,
        *,
        pp_rank: int,
        pp_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        vocab_size: int,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_local_layers = stage_layer_count(num_layers, pp_size, pp_rank)

        if pp_rank == 0:
            self.token_emb = nn.Embedding(vocab_size, hidden_size)
            self.pos_emb = nn.Embedding(max(seq_len, 1024), hidden_size)
            self.drop = nn.Dropout(0.0)
        else:
            self.token_emb = None
            self.pos_emb = None
            self.drop = None

        self.blocks = nn.ModuleList(
            GPT2DecoderBlock(hidden_size, num_heads)
            for _ in range(self.num_local_layers)
        )

        if pp_rank == pp_size - 1:
            self.final_norm = nn.LayerNorm(hidden_size, eps=1e-5)
            self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        else:
            self.final_norm = None
            self.head = None

    def forward_stage(
        self,
        *,
        input_ids: torch.Tensor | None,
        hidden: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.pp_rank == 0:
            assert input_ids is not None
            batch, seq = input_ids.shape
            if seq > self.pos_emb.num_embeddings:
                raise ValueError(
                    f"seq_len={seq} exceeds configured GPT-2 context ({self.pos_emb.num_embeddings})"
                )
            position_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(batch, seq)
            x = self.token_emb(input_ids) + self.pos_emb(position_ids)
            x = self.drop(x)
        else:
            assert hidden is not None
            x = hidden

        for block in self.blocks:
            x = block(x)

        if self.pp_rank == self.pp_size - 1:
            x = self.final_norm(x)
            return self.head(x)
        return x


class GPT2SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.c_attn = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.c_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=2)
        q = q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal_mask = torch.triu(
            torch.ones(seq, seq, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn = attn.masked_fill(causal_mask, -1e4)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.hidden_size)
        return self.c_proj(out)


class GPT2MLP(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        intermediate = hidden_size * 4
        self.c_fc = nn.Linear(hidden_size, intermediate, bias=True)
        self.c_proj = nn.Linear(intermediate, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class GPT2DecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.attn = GPT2SelfAttention(hidden_size, num_heads)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.mlp = GPT2MLP(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class MicrobatchContext:
    microbatch_id: int
    input_activation: torch.Tensor | None
    output_activation: torch.Tensor | None
    loss: torch.Tensor | None


def generate_input_ids(batch: int, seq_len: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch, seq_len), device=device)


def generate_labels(batch: int, seq_len: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch, seq_len), device=device)


def manual_dp_gradient_sync(module: nn.Module, dp_group, dp_size: int) -> None:
    if dp_size <= 1:
        return
    for param in module.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, group=dp_group)
        param.grad.div_(dp_size)


def pipeline_send(
    tensor: torch.Tensor,
    *,
    dst: int,
    mode: str,
) -> None:
    if mode == "blocking":
        dist.send(tensor, dst=dst)
        return
    if mode == "async":
        request = dist.isend(tensor, dst=dst)
        if request is not None:
            request.wait()
        return
    requests = dist.batch_isend_irecv([dist.P2POp(dist.isend, tensor, dst)])
    for request in requests:
        request.wait()


def pipeline_recv(
    tensor: torch.Tensor,
    *,
    src: int,
    mode: str,
) -> None:
    if mode == "blocking":
        dist.recv(tensor, src=src)
        return
    if mode == "async":
        request = dist.irecv(tensor, src=src)
        if request is not None:
            request.wait()
        return
    requests = dist.batch_isend_irecv([dist.P2POp(dist.irecv, tensor, src)])
    for request in requests:
        request.wait()


def run_step_1f1b(
    *,
    model: GPT2PipelineStageModel,
    optimizer: torch.optim.Optimizer,
    dp_rank: int,
    tp_rank: int,
    pp_rank: int,
    args: argparse.Namespace,
    device: torch.device,
    dp_group,
    tp: int,
    pp: int,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    micro_batch = local_micro_batch_size(args.global_batch_size, args.dp, args.micro_batches)
    warmup = min(args.pp - pp_rank - 1, args.micro_batches)
    queue: deque[MicrobatchContext] = deque()
    prev_rank = (
        coords_to_rank(dp_rank, tp_rank, pp_rank - 1, tp, pp)
        if pp_rank > 0
        else None
    )
    next_rank = (
        coords_to_rank(dp_rank, tp_rank, pp_rank + 1, tp, pp)
        if pp_rank + 1 < args.pp
        else None
    )

    def forward_microbatch(microbatch_id: int) -> MicrobatchContext:
        input_activation: torch.Tensor | None = None
        input_ids: torch.Tensor | None = None
        if pp_rank == 0:
            input_ids = generate_input_ids(
                micro_batch,
                args.seq_len,
                args.vocab_size,
                device,
            )
        else:
            input_activation = torch.zeros(
                micro_batch,
                args.seq_len,
                args.hidden_size,
                device=device,
                requires_grad=True,
            )
            assert prev_rank is not None
            pipeline_recv(
                input_activation,
                src=prev_rank,
                mode=args.pipeline_p2p_mode,
            )

        output = model.forward_stage(input_ids=input_ids, hidden=input_activation)
        loss = None
        if pp_rank == args.pp - 1:
            labels = generate_labels(micro_batch, args.seq_len, args.vocab_size, device)
            loss = F.cross_entropy(output.view(-1, args.vocab_size), labels.view(-1))
        elif next_rank is not None:
            pipeline_send(
                output.detach(),
                dst=next_rank,
                mode=args.pipeline_p2p_mode,
            )

        return MicrobatchContext(
            microbatch_id=microbatch_id,
            input_activation=input_activation,
            output_activation=output,
            loss=loss,
        )

    def backward_microbatch(context: MicrobatchContext) -> None:
        if context.loss is not None:
            context.loss.backward()
        else:
            assert context.output_activation is not None
            grad_output = torch.zeros_like(context.output_activation)
            assert next_rank is not None
            pipeline_recv(
                grad_output,
                src=next_rank,
                mode=args.pipeline_p2p_mode,
            )
            context.output_activation.backward(grad_output)

        if prev_rank is not None and context.input_activation is not None:
            grad_input = context.input_activation.grad
            if grad_input is None:
                grad_input = torch.zeros_like(context.input_activation)
            pipeline_send(
                grad_input.detach(),
                dst=prev_rank,
                mode=args.pipeline_p2p_mode,
            )

    if args.schedule == "gpipe":
        for microbatch_id in range(args.micro_batches):
            queue.append(forward_microbatch(microbatch_id))
        while queue:
            backward_microbatch(queue.popleft())
    else:
        for microbatch_id in range(warmup):
            queue.append(forward_microbatch(microbatch_id))

        for microbatch_id in range(warmup, args.micro_batches):
            queue.append(forward_microbatch(microbatch_id))
            backward_microbatch(queue.popleft())

        while queue:
            backward_microbatch(queue.popleft())

    manual_dp_gradient_sync(model, dp_group, args.dp)
    optimizer.step()


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = setup_dist()
    expected_world_size = args.dp * args.tp * args.pp
    if world_size != expected_world_size:
        raise SystemExit(
            f"world_size={world_size} does not match dp*tp*pp={expected_world_size}"
        )
    if args.tp != 1:
        raise SystemExit("gpt2_fig13_pipeline.py currently requires --tp=1")
    if args.hidden_size % args.num_heads != 0:
        raise SystemExit("hidden_size must be divisible by num_heads")

    dp_rank, tp_rank, pp_rank = rank_to_coords(rank, args.dp, args.tp, args.pp)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    dp_group = None
    if args.dp > 1:
        dp_group = build_dp_group(
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp=args.dp,
            tp=args.tp,
            pp=args.pp,
        )

    model = GPT2PipelineStageModel(
        pp_rank=pp_rank,
        pp_size=args.pp,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    micro_batch = local_micro_batch_size(args.global_batch_size, args.dp, args.micro_batches)
    if rank == 0:
        print(
            {
                "world_size": world_size,
                "dp": args.dp,
                "tp": args.tp,
                "pp": args.pp,
                "micro_batches": args.micro_batches,
                "global_batch_size": args.global_batch_size,
                "local_micro_batch_size": micro_batch,
                "schedule": args.schedule,
                "pipeline_p2p_mode": args.pipeline_p2p_mode,
                "seq_len": args.seq_len,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "vocab_size": args.vocab_size,
                "model": "gpt2-base-style-pipeline",
                "stage_layers": [
                    stage_layer_count(args.num_layers, args.pp, index)
                    for index in range(args.pp)
                ],
            }
        )

    start = time.time()
    for step in range(1, args.steps + 1):
        with step_window(step):
            run_step_1f1b(
                model=model,
                optimizer=optimizer,
                dp_rank=dp_rank,
                tp_rank=tp_rank,
                pp_rank=pp_rank,
                args=args,
                device=device,
                dp_group=dp_group,
                tp=args.tp,
                pp=args.pp,
            )
        if rank == 0 and (step % args.log_interval == 0 or step == args.steps):
            elapsed = time.time() - start
            print(
                f"step {step:4d}/{args.steps} | "
                f"elapsed_s={elapsed:.3f} | "
                f"stage={pp_rank} | local_mb={micro_batch}"
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
