#!/usr/bin/env python3
"""
Synthetic Megatron-style workload for Maya Figure 13 reproduction.

This is not intended to be a semantically exact Megatron implementation.
It is a paper-facing trace-shape workload that preserves the important runtime
structure Maya exploits in Figure 13:

- fixed TP / PP / DP decomposition
- one representative worker per pipeline stage
- explicit microbatch pipeline schedule
- tensor-parallel collectives inside each block
- pipeline send/recv traffic between adjacent stages
- data-parallel gradient synchronization after microbatch accumulation
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from cluster_rdma import maybe_apply_cluster_cpu_affinity, maybe_apply_cluster_rdma_affinity
from flexsim.maya_lite.markers import emit_step_marker, step_window


_BOOTSTRAP_DIAG_ENV = "FLEXSIM_MAYA_BOOTSTRAP_DIAG_PATH"
_DIST_GROUP_TIMEOUT = timedelta(hours=2)


def _write_bootstrap_diagnostics(payload: dict[str, object]) -> None:
    raw_path = os.environ.get(_BOOTSTRAP_DIAG_ENV)
    if not raw_path:
        return
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic Megatron workload for Maya Figure 13")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=12_000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--pp", type=int, default=8)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--micro-batches", type=int, default=64)
    parser.add_argument("--schedule", choices=["1f1b", "gpipe"], default="1f1b")
    parser.add_argument(
        "--dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Parameter/activation dtype used by the synthetic workload.",
    )
    parser.add_argument(
        "--pipeline-p2p-mode",
        choices=["blocking", "async", "batch"],
        default="blocking",
        help=(
            "How pipeline point-to-point ops are issued. "
            "batch exercises Megatron-style batch_isend_irecv/P2POp without changing "
            "the high-level step structure."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help=(
            "Unmarked warmup iterations before the paper-facing measured step. "
            "This keeps lazy allocator/cuBLAS/NCCL setup outside the Maya "
            "iteration envelope without using Table 5 tuning knobs."
        ),
    )
    parser.add_argument(
        "--no-step-end-synchronize",
        action="store_true",
        help=(
            "Do not synchronize CUDA before closing a measured step marker. "
            "The default is paper-facing completed-iteration timing."
        ),
    )
    parser.add_argument(
        "--sync-before-step-window",
        action="store_true",
        help=(
            "Synchronize distributed ranks immediately before opening each "
            "measured step marker. This opt-in keeps the existing Figure 13 "
            "default unchanged."
        ),
    )
    parser.add_argument(
        "--source-region-markers",
        action="store_true",
        help="Emit source-level markers around attention/MLP backward and optimizer step regions.",
    )
    parser.add_argument(
        "--evaluator-executor",
        choices=("maya-full", "flexeva-selective"),
        default="maya-full",
    )
    parser.add_argument("--evaluator-partitions", default="")
    parser.add_argument("--evaluator-candidate-id", default="anchor")
    parser.add_argument("--evaluator-mutations", default="")
    parser.add_argument("--evaluator-ready-file", default="")
    parser.add_argument("--evaluator-dispatch-file", default="")
    parser.add_argument("--evaluator-feedback-output", default="")
    parser.add_argument("--timing-output", default="")
    parser.add_argument("--figure6-production", action="store_true")
    parser.add_argument("--validate-effective-global-batch", action="store_true")
    return parser.parse_args()


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
    if device.type == "cuda":
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            device_id=device,
            timeout=timedelta(hours=2),
        )
    else:
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(hours=2),
        )
    return rank, world_size, local_rank, device, rdma_hca, rdma_iface


def runtime_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.dtype == "fp32":
        return torch.float32
    if args.dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {args.dtype}")


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


def build_tp_group(
    *,
    dp_rank: int,
    pp_rank: int,
    dp: int,
    tp: int,
    pp: int,
):
    selected = None
    for current_dp_rank in range(dp):
        for current_pp_rank in range(pp):
            ranks = [
                coords_to_rank(current_dp_rank, tp_rank, current_pp_rank, tp, pp)
                for tp_rank in range(tp)
            ]
            group = dist.new_group(ranks, timeout=_DIST_GROUP_TIMEOUT)
            if current_dp_rank == dp_rank and current_pp_rank == pp_rank:
                selected = group
    return selected


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
            group = dist.new_group(ranks, timeout=_DIST_GROUP_TIMEOUT)
            if current_tp_rank == tp_rank and current_pp_rank == pp_rank:
                selected = group
    return selected


def build_pp_group(
    *,
    dp_rank: int,
    tp_rank: int,
    dp: int,
    tp: int,
    pp: int,
):
    selected = None
    for current_dp_rank in range(dp):
        for current_tp_rank in range(tp):
            ranks = [
                coords_to_rank(current_dp_rank, current_tp_rank, pp_rank, tp, pp)
                for pp_rank in range(pp)
            ]
            group = dist.new_group(ranks, timeout=_DIST_GROUP_TIMEOUT)
            if current_dp_rank == dp_rank and current_tp_rank == tp_rank:
                selected = group
    return selected


def tp_all_reduce(tensor: torch.Tensor, tp_group, tp_size: int) -> torch.Tensor:
    if tp_size > 1:
        dist.all_reduce(tensor, group=tp_group)
    return tensor


def _emit_source_region(kind: str, label: str) -> None:
    if label:
        base = os.environ.get("FLEXMAYA_STAGE_PARTITION", "")
        if kind == "region_begin" and base:
            os.environ["FLEXMAYA_CODE_PARTITION"] = f"{base}_{label}"
        emit_step_marker(kind, label=label)
        if kind == "region_end" and base:
            os.environ["FLEXMAYA_CODE_PARTITION"] = base


@contextmanager
def source_region(label: str, enabled: bool):
    if enabled:
        _emit_source_region("region_begin", label)
    try:
        yield
    finally:
        if enabled:
            _emit_source_region("region_end", label)


def _source_backward_pre_hook(module: nn.Module, _grad_output) -> None:
    if bool(getattr(module, "_flexmaya_source_markers_enabled", False)):
        _emit_source_region("region_begin", str(getattr(module, "_flexmaya_source_partition", "")))


def _source_backward_hook(module: nn.Module, _grad_input, _grad_output) -> None:
    if bool(getattr(module, "_flexmaya_source_markers_enabled", False)):
        _emit_source_region("region_end", str(getattr(module, "_flexmaya_source_partition", "")))


def _source_forward_pre_hook(module: nn.Module, _inputs) -> None:
    if bool(getattr(module, "_flexmaya_source_markers_enabled", False)):
        _emit_source_region("region_begin", str(getattr(module, "_flexmaya_forward_partition", "")))


def _source_forward_hook(module: nn.Module, _inputs, output):
    if bool(getattr(module, "_flexmaya_source_markers_enabled", False)):
        _emit_source_region("region_end", str(getattr(module, "_flexmaya_forward_partition", "")))
    return output


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, tp_size: int) -> None:
        super().__init__()
        assert out_features % tp_size == 0
        self.linear = nn.Linear(in_features, out_features // tp_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RowParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, tp_size: int, tp_group) -> None:
        super().__init__()
        assert in_features % tp_size == 0
        self.linear = nn.Linear(in_features // tp_size, out_features, bias=False)
        self.tp_size = tp_size
        self.tp_group = tp_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        return tp_all_reduce(out, self.tp_group, self.tp_size)


class MegatronAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, tp_size: int, tp_group) -> None:
        super().__init__()
        assert num_heads % tp_size == 0
        self.local_heads = num_heads // tp_size
        self.head_dim = hidden_size // num_heads
        self.local_hidden = self.local_heads * self.head_dim
        self.qkv = ColumnParallelLinear(hidden_size, hidden_size * 3, tp_size)
        self.out_proj = RowParallelLinear(hidden_size, hidden_size, tp_size, tp_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        qkv = self.qkv(x).view(batch, seq, 3, self.local_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.local_hidden)
        return self.out_proj(out)


class MegatronMLP(nn.Module):
    def __init__(self, hidden_size: int, tp_size: int, tp_group) -> None:
        super().__init__()
        intermediate = hidden_size * 4
        self.up = ColumnParallelLinear(hidden_size, intermediate, tp_size)
        self.down = RowParallelLinear(intermediate, hidden_size, tp_size, tp_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class MegatronBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, tp_size: int, tp_group) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = MegatronAttention(hidden_size, num_heads, tp_size, tp_group)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = MegatronMLP(hidden_size, tp_size, tp_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PipelineStageModel(nn.Module):
    def __init__(
        self,
        *,
        pp_rank: int,
        pp_size: int,
        tp_size: int,
        tp_group,
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
            self.pos_emb = nn.Embedding(seq_len, hidden_size)
        else:
            self.token_emb = None
            self.pos_emb = None

        self.blocks = nn.ModuleList(
            MegatronBlock(hidden_size, num_heads, tp_size, tp_group)
            for _ in range(self.num_local_layers)
        )

        if pp_rank == pp_size - 1:
            self.final_norm = nn.LayerNorm(hidden_size)
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
            tok = self.token_emb(input_ids)
            pos = self.pos_emb(torch.arange(seq, device=input_ids.device))
            x = tok + pos
        else:
            assert hidden is not None
            x = hidden

        for block in self.blocks:
            x = block(x)

        if self.pp_rank == self.pp_size - 1:
            x = self.final_norm(x)
            return self.head(x)
        return x


def attach_source_region_markers(
    model: nn.Module,
    enabled: bool,
    *,
    figure6_production: bool = False,
) -> None:
    if not enabled:
        return
    model._flexmaya_source_markers_enabled = True
    for module in model.modules():
        if isinstance(module, MegatronAttention):
            module._flexmaya_source_partition = "attention_backward"
            module._flexmaya_forward_partition = "attention_forward"
            module._flexmaya_source_markers_enabled = True
            if figure6_production:
                module.register_forward_pre_hook(_source_forward_pre_hook)
                module.register_forward_hook(_source_forward_hook)
            module.register_full_backward_pre_hook(_source_backward_pre_hook)
            module.register_full_backward_hook(_source_backward_hook)
        elif isinstance(module, MegatronMLP):
            module._flexmaya_source_partition = "mlp_backward"
            module._flexmaya_forward_partition = "mlp_forward"
            module._flexmaya_source_markers_enabled = True
            if figure6_production:
                module.register_forward_pre_hook(_source_forward_pre_hook)
                module.register_forward_hook(_source_forward_hook)
            module.register_full_backward_pre_hook(_source_backward_pre_hook)
            module.register_full_backward_hook(_source_backward_hook)


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
            # Keep the all-reduce schedule identical across data-parallel peers
            # even if autograd leaves some parameter grads unset on one side.
            param.grad = torch.zeros_like(param)
        dist.all_reduce(param.grad, group=dp_group)
        param.grad.div_(dp_size)


def synchronize_completed_iteration(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def synchronize_before_step_window(host_sync_group) -> None:
    if host_sync_group is None:
        raise RuntimeError("measured step synchronization requires a host-side process group")
    dist.barrier(group=host_sync_group)


EVALUATOR_PARTITIONS = frozenset(
    {
        "attention_forward",
        "mlp_forward",
        "attention_backward",
        "mlp_backward",
        "p2p",
        "optimizer_step",
    }
)


def evaluator_partitions(args: argparse.Namespace) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(item.strip() for item in args.evaluator_partitions.split(",") if item.strip()))
    unknown = sorted(set(selected) - EVALUATOR_PARTITIONS)
    if unknown:
        raise ValueError(f"unknown evaluator partitions: {unknown}")
    return selected


def wait_for_evaluator_dispatch(args: argparse.Namespace, rank: int) -> None:
    if not args.evaluator_dispatch_file:
        return
    dist.barrier()
    if rank == 0:
        if args.evaluator_ready_file:
            ready = Path(args.evaluator_ready_file)
            ready.parent.mkdir(parents=True, exist_ok=True)
            ready.write_text(
                json.dumps({"ready": True, "pid": os.getpid(), "time_ns": time.time_ns()}) + "\n",
                encoding="utf-8",
            )
        while not Path(args.evaluator_dispatch_file).exists():
            time.sleep(0.01)
    dist.barrier()


def write_json(path_text: str, payload: dict[str, object]) -> None:
    if not path_text:
        return
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def pipeline_send(
    tensor: torch.Tensor,
    *,
    group,
    peer_rank: int,
    mode: str,
    markers_enabled: bool = False,
) -> None:
    with source_region("p2p", markers_enabled):
        if mode == "blocking":
            dist.send(tensor, group=group, group_dst=peer_rank)
            return
        if mode == "async":
            request = dist.isend(tensor, group=group, group_dst=peer_rank)
            if request is not None:
                request.wait()
            return
        requests = dist.batch_isend_irecv(
            [dist.P2POp(dist.isend, tensor, group=group, group_peer=peer_rank)]
        )
        for request in requests:
            request.wait()


def pipeline_recv(
    tensor: torch.Tensor,
    *,
    group,
    peer_rank: int,
    mode: str,
    markers_enabled: bool = False,
) -> None:
    with source_region("p2p", markers_enabled):
        if mode == "blocking":
            dist.recv(tensor, group=group, group_src=peer_rank)
            return
        if mode == "async":
            request = dist.irecv(tensor, group=group, group_src=peer_rank)
            if request is not None:
                request.wait()
            return
        requests = dist.batch_isend_irecv(
            [dist.P2POp(dist.irecv, tensor, group=group, group_peer=peer_rank)]
        )
        for request in requests:
            request.wait()


def run_step_1f1b(
    *,
    model: PipelineStageModel,
    optimizer: torch.optim.Optimizer,
    rank: int,
    dp_rank: int,
    tp_rank: int,
    pp_rank: int,
    args: argparse.Namespace,
    device: torch.device,
    activation_dtype: torch.dtype,
    dp_group,
    pp_group,
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
                dtype=activation_dtype,
                requires_grad=True,
            )
            assert prev_rank is not None
            pipeline_recv(
                input_activation,
                group=pp_group,
                peer_rank=pp_rank - 1,
                mode=args.pipeline_p2p_mode,
                markers_enabled=args.source_region_markers and args.figure6_production,
            )

        output = model.forward_stage(input_ids=input_ids, hidden=input_activation)
        loss = None
        if pp_rank == args.pp - 1:
            labels = generate_labels(micro_batch, args.seq_len, args.vocab_size, device)
            loss = F.cross_entropy(output.float().reshape(-1, args.vocab_size), labels.reshape(-1))
        elif next_rank is not None:
            pipeline_send(
                output.detach(),
                group=pp_group,
                peer_rank=pp_rank + 1,
                mode=args.pipeline_p2p_mode,
                markers_enabled=args.source_region_markers and args.figure6_production,
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
                group=pp_group,
                peer_rank=pp_rank + 1,
                mode=args.pipeline_p2p_mode,
                markers_enabled=args.source_region_markers and args.figure6_production,
            )
            context.output_activation.backward(grad_output)

        if prev_rank is not None and context.input_activation is not None:
            grad_input = context.input_activation.grad
            if grad_input is None:
                grad_input = torch.zeros_like(context.input_activation)
            pipeline_send(
                grad_input.detach(),
                group=pp_group,
                peer_rank=pp_rank - 1,
                mode=args.pipeline_p2p_mode,
                markers_enabled=args.source_region_markers and args.figure6_production,
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

    if args.figure6_production:
        with source_region("optimizer_step", args.source_region_markers):
            manual_dp_gradient_sync(model, dp_group, args.dp)
            optimizer.step()
    else:
        manual_dp_gradient_sync(model, dp_group, args.dp)
        with source_region("optimizer_step", args.source_region_markers):
            optimizer.step()


def run_selective_p2p(
    tensor: torch.Tensor,
    *,
    args: argparse.Namespace,
    dp_rank: int,
    tp_rank: int,
    pp_rank: int,
    pp_group,
) -> None:
    previous = coords_to_rank(dp_rank, tp_rank, pp_rank - 1, args.tp, args.pp) if pp_rank else None
    following = (
        coords_to_rank(dp_rank, tp_rank, pp_rank + 1, args.tp, args.pp)
        if pp_rank + 1 < args.pp
        else None
    )
    for _ in range(args.micro_batches):
        if previous is not None:
            pipeline_recv(
                tensor,
                group=pp_group,
                peer_rank=pp_rank - 1,
                mode=args.pipeline_p2p_mode,
                markers_enabled=True,
            )
        if following is not None:
            pipeline_send(
                tensor,
                group=pp_group,
                peer_rank=pp_rank + 1,
                mode=args.pipeline_p2p_mode,
                markers_enabled=True,
            )
    for _ in range(args.micro_batches):
        if following is not None:
            pipeline_recv(
                tensor,
                group=pp_group,
                peer_rank=pp_rank + 1,
                mode=args.pipeline_p2p_mode,
                markers_enabled=True,
            )
        if previous is not None:
            pipeline_send(
                tensor,
                group=pp_group,
                peer_rank=pp_rank - 1,
                mode=args.pipeline_p2p_mode,
                markers_enabled=True,
            )


def run_flexeva_selective(
    *,
    args: argparse.Namespace,
    model: PipelineStageModel,
    optimizer: torch.optim.Optimizer,
    rank: int,
    world_size: int,
    dp_rank: int,
    tp_rank: int,
    pp_rank: int,
    device: torch.device,
    activation_dtype: torch.dtype,
    dp_group,
    pp_group,
) -> None:
    selected = evaluator_partitions(args)
    if not selected:
        raise ValueError("flexeva-selective requires --evaluator-partitions")
    optimizer.zero_grad(set_to_none=True)
    if "optimizer_step" in selected:
        # Selective block replay does not execute the pipeline-edge embedding
        # and output modules. Materialize only those missing gradients before
        # the measured optimizer region so its kernel/NCCL trace matches a
        # full step without adding zeros_like housekeeping to optimizer_step.
        for module in (model.token_emb, model.pos_emb, model.final_norm, model.head):
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.grad = torch.zeros_like(parameter)
    micro_batch = local_micro_batch_size(args.global_batch_size, args.dp, args.micro_batches)
    started = time.perf_counter()
    with step_window(1):
        if "p2p" in selected:
            tensor = torch.zeros(
                micro_batch,
                args.seq_len,
                args.hidden_size,
                device=device,
                dtype=activation_dtype,
            )
            run_selective_p2p(
                tensor,
                args=args,
                dp_rank=dp_rank,
                tp_rank=tp_rank,
                pp_rank=pp_rank,
                pp_group=pp_group,
            )
        attention = bool({"attention_forward", "attention_backward"}.intersection(selected))
        mlp = bool({"mlp_forward", "mlp_backward"}.intersection(selected))
        for _ in range(args.micro_batches):
            x = torch.randn(
                micro_batch,
                args.seq_len,
                args.hidden_size,
                device=device,
                dtype=activation_dtype,
                requires_grad=True,
            )
            for block in model.blocks:
                if attention:
                    output = block.attn(block.ln1(x))
                    if "attention_backward" in selected:
                        output.float().mean().backward()
                if mlp:
                    output = block.mlp(block.ln2(x))
                    if "mlp_backward" in selected:
                        output.float().mean().backward()
        if "optimizer_step" in selected:
            with source_region("optimizer_step", True):
                for parameter in model.parameters():
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                manual_dp_gradient_sync(model, dp_group, args.dp)
                optimizer.step()
        synchronize_completed_iteration(device)
    execute_wall_s = time.perf_counter() - started
    dist.barrier()
    if rank == 0:
        dependencies = sorted(
            set(selected).union(
                {"attention_forward"} if "attention_backward" in selected else set()
            ).union({"mlp_forward"} if "mlp_backward" in selected else set())
        )
        payload = {
            "schema": "flexeva.figure6.executor.v1",
            "executor": "flexeva-selective",
            "full_training_step_executed": False,
            "full_candidate_trace_generated": False,
            "fallback": False,
            "executed_partitions": list(selected),
            "dependency_partitions": dependencies,
            "world_size": world_size,
            "tp": args.tp,
            "pp": args.pp,
            "dp": args.dp,
            "micro_batches": args.micro_batches,
            "local_micro_batch_size": micro_batch,
            "effective_global_batch": micro_batch * args.dp * args.micro_batches,
            "execute_wall_s": execute_wall_s,
            "candidate_id": args.evaluator_candidate_id,
            "candidate_mutations": [item for item in args.evaluator_mutations.split(",") if item],
        }
        write_json(args.evaluator_feedback_output, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    bootstrap_start = time.perf_counter()
    rank, world_size, local_rank, device, rdma_hca, rdma_iface = setup_dist()
    setup_dist_done = time.perf_counter()
    expected_world_size = args.dp * args.tp * args.pp
    if world_size != expected_world_size:
        raise SystemExit(
            f"world_size={world_size} does not match dp*tp*pp={expected_world_size}"
        )
    if args.hidden_size % args.num_heads != 0:
        raise SystemExit("hidden_size must be divisible by num_heads")
    if args.dtype == "bf16" and device.type != "cuda":
        raise SystemExit("--dtype bf16 requires CUDA")
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA device does not report bf16 support")

    dp_rank, tp_rank, pp_rank = rank_to_coords(rank, args.dp, args.tp, args.pp)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    tp_group = None
    tp_group_start = time.perf_counter()
    if args.tp > 1:
        tp_group = build_tp_group(
            dp_rank=dp_rank,
            pp_rank=pp_rank,
            dp=args.dp,
            tp=args.tp,
            pp=args.pp,
        )
    tp_group_done = time.perf_counter()
    dp_group = None
    dp_group_start = time.perf_counter()
    if args.dp > 1:
        dp_group = build_dp_group(
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp=args.dp,
            tp=args.tp,
            pp=args.pp,
        )
    dp_group_done = time.perf_counter()
    pp_group = None
    pp_group_start = time.perf_counter()
    if args.pp > 1:
        pp_group = build_pp_group(
            dp_rank=dp_rank,
            tp_rank=tp_rank,
            dp=args.dp,
            tp=args.tp,
            pp=args.pp,
        )
    pp_group_done = time.perf_counter()

    step_window_sync_group = None
    if args.sync_before_step_window:
        if not dist.is_gloo_available():
            raise RuntimeError("--sync-before-step-window requires the Gloo backend")
        step_window_sync_group = dist.new_group(backend="gloo")

    param_dtype = runtime_dtype(args)
    model_init_start = time.perf_counter()
    model = PipelineStageModel(
        pp_rank=pp_rank,
        pp_size=args.pp,
        tp_size=args.tp,
        tp_group=tp_group,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
    ).to(device=device, dtype=param_dtype)
    attach_source_region_markers(
        model,
        args.source_region_markers,
        figure6_production=args.figure6_production,
    )
    model_init_done = time.perf_counter()
    optimizer_init_start = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer_init_done = time.perf_counter()

    micro_batch = local_micro_batch_size(args.global_batch_size, args.dp, args.micro_batches)
    effective_global_batch = micro_batch * args.dp * args.micro_batches
    if args.validate_effective_global_batch and effective_global_batch != args.global_batch_size:
        raise SystemExit(
            f"effective global batch={effective_global_batch} differs from configured "
            f"global batch={args.global_batch_size}"
        )
    _write_bootstrap_diagnostics(
        {
            "rank": rank,
            "local_rank": local_rank,
            "dp_rank": dp_rank,
            "tp_rank": tp_rank,
            "pp_rank": pp_rank,
            "world_size": world_size,
            "dp": args.dp,
            "tp": args.tp,
            "pp": args.pp,
            "setup_dist_seconds": setup_dist_done - bootstrap_start,
            "tp_group_build_seconds": tp_group_done - tp_group_start,
            "dp_group_build_seconds": dp_group_done - dp_group_start,
            "pp_group_build_seconds": pp_group_done - pp_group_start,
            "model_init_seconds": model_init_done - model_init_start,
            "optimizer_init_seconds": optimizer_init_done - optimizer_init_start,
            "bootstrap_pre_step_seconds": optimizer_init_done - bootstrap_start,
            "micro_batch": micro_batch,
            "dtype": args.dtype,
            "pipeline_p2p_mode": args.pipeline_p2p_mode,
        }
    )
    if rank == 0:
        print(
            {
                "world_size": world_size,
                "dp": args.dp,
                "tp": args.tp,
                "pp": args.pp,
                "micro_batches": args.micro_batches,
                "global_batch_size": args.global_batch_size,
                "effective_global_batch": effective_global_batch,
                "local_micro_batch_size": micro_batch,
                "schedule": args.schedule,
                "dtype": args.dtype,
                "pipeline_p2p_mode": args.pipeline_p2p_mode,
                "rdma_hca": rdma_hca,
                "rdma_iface": rdma_iface,
                "seq_len": args.seq_len,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "warmup_steps": args.warmup_steps,
                "step_end_synchronize": not args.no_step_end_synchronize,
                "sync_before_step_window": args.sync_before_step_window,
                "step_window_sync_backend": "gloo" if step_window_sync_group is not None else None,
            }
        )

    wait_for_evaluator_dispatch(args, rank)
    if args.evaluator_executor == "flexeva-selective":
        run_flexeva_selective(
            args=args,
            model=model,
            optimizer=optimizer,
            rank=rank,
            world_size=world_size,
            dp_rank=dp_rank,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            device=device,
            activation_dtype=param_dtype,
            dp_group=dp_group,
            pp_group=pp_group,
        )
        return

    warmup_and_steps_start = time.time()
    for warmup_step in range(1, max(int(args.warmup_steps), 0) + 1):
        run_step_1f1b(
            model=model,
            optimizer=optimizer,
            rank=rank,
            dp_rank=dp_rank,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            args=args,
            device=device,
            activation_dtype=param_dtype,
            dp_group=dp_group,
            pp_group=pp_group,
            tp=args.tp,
            pp=args.pp,
        )
        if not args.no_step_end_synchronize:
            synchronize_completed_iteration(device)
        if rank == 0 and warmup_step == max(int(args.warmup_steps), 0):
            print(f"warmup {warmup_step:4d}/{args.warmup_steps} complete")

    measured_steps_start = time.time()
    measured_steps: list[float] = []
    for step in range(1, args.steps + 1):
        if args.sync_before_step_window:
            synchronize_before_step_window(step_window_sync_group)
        step_started = time.perf_counter()
        with step_window(step):
            run_step_1f1b(
                model=model,
                optimizer=optimizer,
                rank=rank,
                dp_rank=dp_rank,
                tp_rank=tp_rank,
                pp_rank=pp_rank,
                args=args,
                device=device,
                activation_dtype=param_dtype,
                dp_group=dp_group,
                pp_group=pp_group,
                tp=args.tp,
                pp=args.pp,
            )
            if not args.no_step_end_synchronize:
                synchronize_completed_iteration(device)
        measured_steps.append(time.perf_counter() - step_started)
        if rank == 0 and (step % args.log_interval == 0 or step == args.steps):
            measured_elapsed = time.time() - measured_steps_start
            total_elapsed = time.time() - warmup_and_steps_start
            print(
                f"step {step:4d}/{args.steps} | "
                f"measured_elapsed_s={measured_elapsed:.3f} | "
                f"warmup_plus_steps_elapsed_s={total_elapsed:.3f} | "
                f"stage={pp_rank} | local_mb={micro_batch}"
            )

    dist.barrier()
    if rank == 0:
        total_s = sum(measured_steps)
        timing = {
            "executor": "maya-full",
            "full_training_step_executed": True,
            "full_candidate_trace_generated": True,
            "fallback": False,
            "steps": args.steps,
            "measured_step_s": measured_steps,
            "total_step_s": total_s,
            "world_size": world_size,
            "tp": args.tp,
            "pp": args.pp,
            "dp": args.dp,
            "micro_batches": args.micro_batches,
            "local_micro_batch_size": micro_batch,
            "effective_global_batch": effective_global_batch,
            "candidate_id": args.evaluator_candidate_id,
            "candidate_mutations": [item for item in args.evaluator_mutations.split(",") if item],
        }
        write_json(args.timing_output, timing)
        write_json(
            args.evaluator_feedback_output,
            {
                "schema": "flexeva.figure6.executor.v1",
                **timing,
                "executed_partitions": sorted(EVALUATOR_PARTITIONS),
                "execute_wall_s": total_s,
            },
        )
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
