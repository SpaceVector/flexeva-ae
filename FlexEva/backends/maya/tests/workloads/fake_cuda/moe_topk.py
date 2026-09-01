#!/usr/bin/env python3
"""
Mixture of Experts with Top-K routing and Expert Parallelism.

This workload demonstrates data-dependent control flow that Maya cannot handle:
1. Router computes gating scores from activations (tensor-dependent)
2. Top-k selection creates dynamic dispatch patterns (CPU branches on tensor values)
3. All-to-all communication volume depends on routing decisions
4. Capacity factor drops tokens, creating data-dependent load imbalance

Maya no-ops the compute kernels -> router logits are garbage -> wrong routing
-> wrong communication pattern -> wrong simulation.

FlexSim handles this because SPSD transformation preserves the routing logic.

Run with:
    torchrun --nproc_per_node=2 moe_topk.py --steps 10
    torchrun --nproc_per_node=4 moe_topk.py --steps 10 --num-experts 8

With fake-cuda:
    ./frun torchrun --nproc_per_node=2 test/moe_topk.py --steps 10
"""
import argparse
import contextlib
import os
import time
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from cluster_rdma import maybe_apply_cluster_cpu_affinity, maybe_apply_cluster_rdma_affinity
from flexsim.maya_lite.markers import step_window


_DIST_GROUP_TIMEOUT = timedelta(hours=2)


def is_fakecuda_device(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    if os.environ.get("FAKECUDA_TARGET_ENV_ROOT") or os.environ.get("FAKECUDA_PROOT_BIN"):
        return True
    if not torch.cuda.is_available():
        return False
    index = 0 if device.index is None else device.index
    try:
        return torch.cuda.get_device_name(index) in {"SimGPU", "NVIDIA A100-SXM4-80GB"}
    except Exception:
        return False


def resolve_ddp_ctor():
    ddp_ctor = torch.nn.parallel.DistributedDataParallel
    if getattr(ddp_ctor, "__module__", "").startswith("torch.nn.parallel"):
        try:
            from flexsim.maya_lite.fakecuda_compat import install_fakecuda_ddp_compat

            install_fakecuda_ddp_compat()
            ddp_ctor = torch.nn.parallel.DistributedDataParallel
        except Exception:
            ddp_ctor = DDP
    return ddp_ctor


def wrap_ddp(module: nn.Module, *args, **kwargs):
    return resolve_ddp_ctor()(module, *args, **kwargs)


def compute_cpu_gate_logits(x: torch.Tensor, gate: nn.Linear) -> torch.Tensor:
    gate_weight_cpu = gate.weight.detach().float().cpu()
    gate_bias_cpu = None
    if gate.bias is not None:
        gate_bias_cpu = gate.bias.detach().float().cpu()
    return F.linear(x.detach().float().cpu(), gate_weight_cpu, gate_bias_cpu)


def build_owned_token_list(owned: list[int], expert_token_indices, *, sort_by_density: bool = False):
    selected_list: list[int] = []
    token_counts: dict[int, int] = {}
    first_seen: dict[int, int] = {}
    for expert_id in owned:
        for token_idx in expert_token_indices[expert_id]:
            token_idx = int(token_idx)
            token_counts[token_idx] = token_counts.get(token_idx, 0) + 1
            if token_idx not in first_seen:
                first_seen[token_idx] = len(selected_list)
                selected_list.append(token_idx)
    if sort_by_density:
        selected_list.sort(key=lambda token_idx: (-token_counts[token_idx], first_seen[token_idx], token_idx))
    return selected_list, token_counts


def build_owned_token_tensor(
    owned: list[int],
    expert_token_indices,
    *,
    device: torch.device,
    sort_by_density: bool = False,
):
    selected_list, token_counts = build_owned_token_list(
        owned,
        expert_token_indices,
        sort_by_density=sort_by_density,
    )
    return torch.tensor(selected_list, device=device, dtype=torch.long), token_counts


def infer_dropped_token_list(num_tokens: int, expert_token_indices) -> list[int]:
    assigned = set()
    for token_list in expert_token_indices:
        assigned.update(int(token_idx) for token_idx in token_list)
    return [token_idx for token_idx in range(num_tokens) if token_idx not in assigned]


def build_parallel_groups(rank: int, world_size: int, ep_size: int):
    """Build orthogonal EP and DP groups for the MoE layout.

    EP groups are contiguous blocks of size ``ep_size``.
    DP groups are strided groups that line up identical expert ownership across
    EP blocks. DDP should only run on the DP groups.
    """
    assert world_size % ep_size == 0, "world_size must be divisible by ep_size"
    dp_size = world_size // ep_size

    ep_group = None
    dp_group = None
    rank_in_ep = 0
    rank_in_dp = 0

    for start in range(0, world_size, ep_size):
        ranks_in_group = list(range(start, min(start + ep_size, world_size)))
        group = dist.new_group(ranks_in_group, timeout=_DIST_GROUP_TIMEOUT)
        if rank in ranks_in_group:
            ep_group = group
            rank_in_ep = ranks_in_group.index(rank)

    for offset in range(ep_size):
        ranks_in_group = list(range(offset, world_size, ep_size))
        group = dist.new_group(ranks_in_group, timeout=_DIST_GROUP_TIMEOUT)
        if rank in ranks_in_group:
            dp_group = group
            rank_in_dp = ranks_in_group.index(rank)

    return ep_group, dp_group, dp_size, rank_in_ep, rank_in_dp


def parse_args():
    ap = argparse.ArgumentParser(description="MoE Top-K training with Expert Parallelism")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup-steps", type=int, default=1,
                    help="Unmarked warmup steps before the measured trace window")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--global-batch-size", type=int, default=None,
                    help="Optional global batch size; local batch derives from actual DP size")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--num-heads", type=int, default=0,
                    help="Attention heads (0 derives hidden_size // 64)")
    ap.add_argument("--num-experts", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--top-k", type=int, default=2, help="Top-K experts per token")
    ap.add_argument("--capacity-factor", type=float, default=1.25,
                    help="Expert capacity = factor * (tokens / num_experts)")
    ap.add_argument("--ep-size", type=int, default=0,
                    help="Expert parallel group size (0=world_size)")
    ap.add_argument("--dp", type=int, default=1,
                    help="Accepted for sweep compatibility; actual DP comes from world_size/ep_size")
    ap.add_argument("--tp", type=int, default=1,
                    help="Accepted for sweep compatibility; routed-MoE workload keeps TP=1")
    ap.add_argument("--pp", type=int, default=1,
                    help="Accepted for sweep compatibility; routed-MoE workload keeps PP=1")
    ap.add_argument("--micro-batches", type=int, default=1,
                    help="Number of sequential micro-batches per optimizer step")
    ap.add_argument("--schedule", default="1f1b",
                    help="Accepted for sweep compatibility")
    ap.add_argument("--pipeline-p2p-mode", default="blocking",
                    help="Accepted for sweep compatibility")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--recompute", action="store_true",
                    help="Enable activation recomputation inside transformer blocks")
    ap.add_argument("--expert-layout", choices=["contiguous", "striped"], default="contiguous",
                    help="How experts are assigned to EP ranks")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--aux-weight", type=float, default=0.01,
                    help="Auxiliary load-balancing loss weight")
    ap.add_argument("--no-step-end-synchronize", action="store_true",
                    help="Do not synchronize CUDA before closing the measured step window")
    return ap.parse_args()


def setup_dist():
    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    maybe_apply_cluster_cpu_affinity(local_rank)
    rdma_hca, rdma_iface = maybe_apply_cluster_rdma_affinity(local_rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            device_id=device,
            timeout=timedelta(hours=2),
        )
    else:
        device = torch.device("cpu")
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(hours=2),
        )

    return rank, world_size, local_rank, device, rdma_hca, rdma_iface


def runtime_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dtype_name}")


def autocast_context(device: torch.device, dtype_name: str):
    if device.type == "cuda" and dtype_name == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def local_batch_size(args, actual_dp_size: int) -> int:
    if args.global_batch_size is None:
        return max(1, int(args.batch_size))
    return max(
        1,
        int((int(args.global_batch_size) + max(actual_dp_size, 1) - 1) // max(actual_dp_size, 1)),
    )


def synchronize_completed_iteration(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# Top-K Gating with data-dependent routing
# ---------------------------------------------------------------------------

class TopKGate(nn.Module):
    """
    Top-K gating network with capacity-limited expert dispatch.

    This is the component Maya CANNOT simulate correctly:
    - gate_scores depend on activation values
    - torch.topk() creates data-dependent expert assignments
    - capacity factor drops tokens when experts are overloaded
    - all of this affects downstream communication patterns
    """
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2,
                 capacity_factor: float = 1.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch * seq, hidden)
        Returns:
            dispatch_mask: (num_experts, capacity, batch*seq) - binary assignment
            combine_weights: (batch*seq, num_experts) - gating weights
            aux_loss: load balancing loss
            metadata: dict with routing statistics
        """
        num_tokens = x.shape[0]

        # Compute gating scores (DATA-DEPENDENT - this is the key)
        gate_logits = self.gate(x)  # (num_tokens, num_experts)
        fakecuda_cpu_routing = is_fakecuda_device(x.device) and self.top_k > 1
        if fakecuda_cpu_routing:
            # fake-cuda currently crashes on CUDA topk with k>1. Keep the gate
            # projection on-device for trace realism, but keep the discrete
            # routing state on CPU so `.item()`-driven control flow does not
            # depend on placeholder fake-cuda tensors.
            gate_logits_cpu = compute_cpu_gate_logits(x, self.gate)
            gate_probs_cpu = F.softmax(gate_logits_cpu, dim=-1)
            top_k_gates, top_k_indices = torch.topk(gate_probs_cpu, self.top_k, dim=-1)
            gate_probs = gate_probs_cpu.to(device=x.device, dtype=x.dtype)
        else:
            gate_probs = F.softmax(gate_logits, dim=-1)

            # Top-K selection (DATA-DEPENDENT CPU control flow)
            top_k_gates, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        # top_k_gates: (num_tokens, top_k) - selected expert weights
        # top_k_indices: (num_tokens, top_k) - which experts selected

        # Normalize top-k weights
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=-1, keepdim=True) + 1e-9)

        # Capacity per expert
        capacity = int(self.capacity_factor * num_tokens * self.top_k / self.num_experts)
        capacity = max(capacity, 1)

        # Build dispatch mask: assign tokens to experts with capacity limits
        # THIS IS DATA-DEPENDENT: which tokens go to which experts depends on
        # the routing decisions above
        dispatch_mask = torch.zeros(
            self.num_experts, capacity, num_tokens,
            device=x.device, dtype=x.dtype
        )
        combine_weights = torch.zeros(
            num_tokens, self.num_experts,
            device=x.device, dtype=x.dtype
        )

        # Per-expert position counters (DATA-DEPENDENT CONTROL FLOW)
        expert_counts = (
            torch.zeros(self.num_experts, dtype=torch.long)
            if fakecuda_cpu_routing
            else torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
        )
        expert_token_indices = [[] for _ in range(self.num_experts)] if fakecuda_cpu_routing else None
        tokens_dropped = 0

        for i in range(num_tokens):
            for k in range(self.top_k):
                expert_id = int(top_k_indices[i, k].item())  # DATA-DEPENDENT branch
                pos = int(expert_counts[expert_id].item())

                if pos < capacity:
                    # Token fits in this expert's capacity
                    dispatch_mask[expert_id, pos, i] = 1.0
                    gate_weight = float(top_k_gates[i, k].item()) if fakecuda_cpu_routing else top_k_gates[i, k]
                    combine_weights[i, expert_id] = gate_weight
                    expert_counts[expert_id] += 1
                    if expert_token_indices is not None:
                        expert_token_indices[expert_id].append(i)
                else:
                    # Expert at capacity — token dropped (DATA-DEPENDENT)
                    tokens_dropped += 1

        # Auxiliary load balancing loss (Switch Transformer style)
        # f_i = fraction of tokens routed to expert i
        # P_i = fraction of router probability assigned to expert i
        tokens_per_expert = (
            expert_counts.to(device=x.device, dtype=x.dtype)
            if fakecuda_cpu_routing
            else expert_counts.float()
        )
        f = tokens_per_expert / (num_tokens * self.top_k)
        P = gate_probs.mean(dim=0)
        aux_loss = (f * P).sum() * self.num_experts

        metadata = {
            "expert_counts": expert_counts.tolist(),
            "tokens_dropped": tokens_dropped,
            "capacity": capacity,
            "load_balance_cv": (tokens_per_expert.std() / (tokens_per_expert.mean() + 1e-9)).item(),
        }
        if expert_token_indices is not None:
            metadata["expert_token_indices"] = expert_token_indices

        return dispatch_mask, combine_weights, aux_loss, metadata


class TopKMoELayer(nn.Module):
    """MoE layer with top-K routing and optional expert parallelism."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2,
                 capacity_factor: float = 1.25):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.gate = TopKGate(hidden_size, num_experts, top_k, capacity_factor)

        intermediate_size = hidden_size * 4
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size, bias=False),
                nn.GELU(),
                nn.Linear(intermediate_size, hidden_size, bias=False),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)  # (B*S, H)

        # Route tokens to experts (DATA-DEPENDENT)
        dispatch_mask, combine_weights, aux_loss, metadata = self.gate(x_flat)
        expert_counts_list = list(metadata.get("expert_counts", []))

        # Dispatch and compute per-expert
        # dispatch_mask: (E, C, B*S), x_flat: (B*S, H)
        expert_outputs = torch.zeros_like(x_flat)

        for expert_id in range(self.num_experts):
            # Gather tokens for this expert
            # expert_input = dispatch_mask[expert_id] @ x_flat  # (C, H)
            mask = dispatch_mask[expert_id]  # (C, B*S)
            expert_input = torch.matmul(mask, x_flat)  # (C, H)

            # Only compute if expert has tokens (DATA-DEPENDENT SKIP)
            if expert_id < len(expert_counts_list) and int(expert_counts_list[expert_id]) > 0:
                expert_out = self.experts[expert_id](expert_input)  # (C, H)
                # Scatter back: transpose mask and multiply
                expert_outputs += torch.matmul(mask.t(), expert_out)  # (B*S, H)

        # Weight by combine_weights
        # combine_weights: (B*S, E) - but we already dispatched per-expert
        # The combine_weights were already applied in the dispatch_mask construction
        output = expert_outputs.view(batch_size, seq_len, hidden)

        return output, aux_loss, metadata


class ExpertParallelMoELayer(nn.Module):
    """
    MoE layer with expert parallelism via all-to-all communication.

    Each rank owns a subset of experts. Tokens are routed across ranks
    via all-to-all. Communication volume is DATA-DEPENDENT (varies by
    routing decisions).

    Maya cannot simulate this because:
    1. Router output determines all-to-all send counts
    2. All-to-all is not a fixed-size collective
    3. Load imbalance across ranks depends on routing
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2,
                 capacity_factor: float = 1.25, ep_group=None, ep_size: int = 1,
                 expert_layout: str = "contiguous"):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.ep_group = ep_group
        self.ep_size = ep_size
        self.expert_layout = expert_layout

        # Each rank owns num_experts // ep_size local experts
        self.local_experts = num_experts // ep_size
        self.gate = TopKGate(hidden_size, num_experts, top_k, capacity_factor)
        self.expert_owner = self._build_expert_owner()

        intermediate_size = hidden_size * 4
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size, bias=False),
                nn.GELU(),
                nn.Linear(intermediate_size, hidden_size, bias=False),
            )
            for _ in range(self.local_experts)
        ])
        self.rank_in_ep = dist.get_rank(self.ep_group) if self.ep_group is not None and self.ep_size > 1 else 0
        self.local_global_experts = [
            expert_id for expert_id, owner in enumerate(self.expert_owner) if owner == self.rank_in_ep
        ]

    def _build_expert_owner(self) -> list[int]:
        if self.expert_layout == "striped":
            return [expert_id % self.ep_size for expert_id in range(self.num_experts)]
        return [expert_id // self.local_experts for expert_id in range(self.num_experts)]

    def forward(self, x):
        batch_size, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        num_tokens = x_flat.shape[0]

        # Route (DATA-DEPENDENT)
        dispatch_mask, combine_weights, aux_loss, metadata = self.gate(x_flat)
        expert_counts_list = list(metadata.get("expert_counts", []))
        expert_token_indices = metadata.get("expert_token_indices")

        if self.ep_group is not None and self.ep_size > 1:
            # Expert parallelism: all-to-all dispatch
            # Compute per-rank token counts (DATA-DEPENDENT COMMUNICATION)
            expert_counts = torch.tensor(metadata["expert_counts"], device=x.device)

            # Tokens destined for local experts vs remote
            send_counts = []
            for r in range(self.ep_size):
                owned = [idx for idx, owner in enumerate(self.expert_owner) if owner == r]
                if expert_token_indices is not None and owned:
                    count = sum(int(expert_counts_list[idx]) for idx in owned if idx < len(expert_counts_list))
                else:
                    count = expert_counts[owned].sum().item() if owned else 0
                send_counts.append(count)

            # All-to-all to exchange token counts (so receivers know what's coming)
            send_counts_tensor = torch.tensor(send_counts, device=x.device, dtype=torch.long)
            recv_counts_tensor = torch.zeros_like(send_counts_tensor)
            dist.all_to_all_single(recv_counts_tensor, send_counts_tensor,
                                   group=self.ep_group)

            # All-to-all to exchange actual tokens
            # Build send buffer: gather tokens by destination rank
            # (Simplified: use padded fixed-size buffers for fake-cuda compatibility)
            max_tokens = int(num_tokens * 1.5)  # padded
            send_buf = torch.zeros(self.ep_size, max_tokens, hidden, device=x.device)
            recv_buf = torch.zeros(self.ep_size, max_tokens, hidden, device=x.device)

            # Pack send buffer (DATA-DEPENDENT packing)
            for r in range(self.ep_size):
                owned = [idx for idx, owner in enumerate(self.expert_owner) if owner == r]
                # Gather tokens routed to experts on rank r. When fake-cuda
                # routing metadata is available, use that CPU-side assignment
                # directly instead of reducing placeholder device tensors.
                if expert_token_indices is not None and owned:
                    selected, _token_counts = build_owned_token_tensor(
                        owned,
                        expert_token_indices,
                        device=x.device,
                    )
                else:
                    r_mask = dispatch_mask[owned].sum(dim=0).sum(dim=0) if owned else torch.zeros(num_tokens, device=x.device)
                    selected = (r_mask > 0).nonzero(as_tuple=True)[0]
                n = min(len(selected), max_tokens)
                if n > 0:
                    send_buf[r, :n] = x_flat[selected[:n]]

            # All-to-all token exchange
            dist.all_to_all(
                list(recv_buf.unbind(0)),
                list(send_buf.unbind(0)),
                group=self.ep_group,
            )

            # Process local experts on received tokens
            expert_outputs_local = torch.zeros_like(recv_buf[self.rank_in_ep])
            for local_idx, global_idx in enumerate(self.local_global_experts):
                mask = dispatch_mask[global_idx]  # (C, B*S)
                expert_input = torch.matmul(mask, x_flat)
                if global_idx < len(expert_counts_list) and int(expert_counts_list[global_idx]) > 0:
                    expert_out = self.experts[local_idx](expert_input)
                    expert_outputs_local[:expert_out.shape[0]] += expert_out

            # All-to-all to send results back (reverse direction)
            result_send = torch.zeros_like(send_buf)
            result_recv = torch.zeros_like(recv_buf)
            result_send[self.rank_in_ep] = expert_outputs_local

            dist.all_to_all(
                list(result_recv.unbind(0)),
                list(result_send.unbind(0)),
                group=self.ep_group,
            )

            # Combine results
            output = result_recv.sum(dim=0)[:num_tokens]
        else:
            # No expert parallelism — local dispatch
            output = torch.zeros_like(x_flat)
            for expert_id in range(self.num_experts):
                mask = dispatch_mask[expert_id]
                expert_input = torch.matmul(mask, x_flat)
                if expert_id < len(expert_counts_list) and int(expert_counts_list[expert_id]) > 0:
                    expert_out = self.experts[expert_id](expert_input)
                    output += torch.matmul(mask.t(), expert_out)

        output = output.view(batch_size, seq_len, hidden)
        return output, aux_loss, metadata


class MoETransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num_experts, top_k,
                 capacity_factor, ep_group=None, ep_size=1, recompute=False,
                 expert_layout="contiguous"):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.recompute = recompute
        self.num_experts = num_experts

        if ep_group is not None and ep_size > 1:
            self.moe = ExpertParallelMoELayer(
                hidden_size, num_experts, top_k, capacity_factor,
                ep_group=ep_group, ep_size=ep_size, expert_layout=expert_layout)
        else:
            self.moe = TopKMoELayer(
                hidden_size, num_experts, top_k, capacity_factor)

    def _forward_impl(self, x):
        # Keep layer norm numerically stable and dtype-consistent under bf16
        # mixed precision by normalizing in fp32, then casting back.
        h = self.ln1(x.float()).to(dtype=x.dtype)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h

        h = self.ln2(x.float()).to(dtype=x.dtype)
        h, aux_loss, metadata = self.moe(h)
        x = x + h
        return x, aux_loss, metadata

    def forward(self, x):
        if self.recompute and self.training:
            def _recompute_fn(inp):
                out, aux, _ = self._forward_impl(inp)
                return out, aux

            # fake-cuda keeps recompute semantics, but the non-reentrant
            # checkpoint backend is materially more stable for routed MoE
            # control flow under placeholder device execution.
            use_reentrant = False if is_fakecuda_device(x.device) else True
            x, aux_loss = checkpoint(_recompute_fn, x, use_reentrant=use_reentrant)
            metadata = {
                "expert_counts": [0 for _ in range(self.num_experts)],
                "tokens_dropped": 0,
                "capacity": 0,
                "load_balance_cv": 0.0,
                "recompute": True,
            }
            return x, aux_loss, metadata
        return self._forward_impl(x)


class MoETopKModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads,
                 num_experts, max_seq_len, top_k, capacity_factor,
                 ep_group=None, ep_size=1, micro_batches=1, recompute=False,
                 expert_layout="contiguous"):
        super().__init__()
        self.micro_batches = max(1, micro_batches)
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)

        self.blocks = nn.ModuleList([
            MoETransformerBlock(
                hidden_size, num_heads, num_experts, top_k, capacity_factor,
                ep_group=ep_group, ep_size=ep_size, recompute=recompute,
                expert_layout=expert_layout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        tok_emb = self.token_emb(input_ids)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=input_ids.device))
        x = tok_emb + pos_emb

        total_aux = 0.0
        all_metadata = []
        for block in self.blocks:
            x, aux, meta = block(x)
            total_aux += aux
            all_metadata.append(meta)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, total_aux / len(self.blocks), all_metadata


def main():
    args = parse_args()
    rank, world_size, local_rank, device, rdma_hca, rdma_iface = setup_dist()
    fakecuda = is_fakecuda_device(device)

    torch.manual_seed(42 + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + rank)

    if args.tp != 1 or args.pp != 1:
        raise ValueError("routed-MoE real-trace workload currently requires tp=1 and pp=1")

    # Expert/data parallel groups
    ep_size = args.ep_size if args.ep_size > 0 else world_size
    ep_size = min(ep_size, world_size)
    assert args.num_experts % ep_size == 0, \
        f"num_experts ({args.num_experts}) must be divisible by ep_size ({ep_size})"
    assert world_size % ep_size == 0, \
        f"world_size ({world_size}) must be divisible by ep_size ({ep_size})"

    ep_group, dp_group, dp_size, rank_in_ep, _rank_in_dp = build_parallel_groups(
        rank, world_size, ep_size
    )

    vocab_size = int(args.vocab_size)
    num_heads = int(args.num_heads) if int(args.num_heads) > 0 else max(1, args.hidden_size // 64)
    activation_dtype = runtime_dtype(args.dtype)

    model = MoETopKModel(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=num_heads,
        num_experts=args.num_experts,
        max_seq_len=args.seq_len,
        top_k=args.top_k,
        capacity_factor=args.capacity_factor,
        ep_group=ep_group,
        ep_size=ep_size,
        micro_batches=args.micro_batches,
        recompute=args.recompute,
        expert_layout=args.expert_layout,
    ).to(device=device)

    # DDP should run only across true DP replicas. For EP-only jobs (dp_size=1),
    # there is nothing to synchronize.
    if dp_size > 1:
        model = wrap_ddp(model, device_ids=[local_rank] if device.type == "cuda" else None,
                         output_device=local_rank if device.type == "cuda" else None,
                         process_group=dp_group,
                         broadcast_buffers=False,
                         find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batch_size = local_batch_size(args, dp_size)

    num_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print({
            "world_size": world_size,
            "ep_size": ep_size,
            "dp_size": dp_size,
            "configured_dp": args.dp,
            "rank_in_ep": rank_in_ep,
            "num_experts": args.num_experts,
            "local_experts": args.num_experts // ep_size,
            "top_k": args.top_k,
            "capacity_factor": args.capacity_factor,
            "micro_batches": args.micro_batches,
            "recompute": args.recompute,
            "expert_layout": args.expert_layout,
            "wrapper_type": type(model).__name__,
            "dtype": args.dtype,
            "activation_dtype": str(activation_dtype),
            "global_batch_size": args.global_batch_size,
            "local_batch_size": batch_size,
            "rdma_hca": rdma_hca,
            "rdma_iface": rdma_iface,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "params_m": round(num_params / 1e6, 2),
        })

    # Training loop
    def run_single_step(step: int):
        input_ids = torch.randint(0, vocab_size,
                                  (batch_size, args.seq_len), device=device)
        labels = torch.randint(0, vocab_size,
                               (batch_size, args.seq_len), device=device)
        optimizer.zero_grad(set_to_none=True)
        micro_inputs = torch.chunk(input_ids, args.micro_batches, dim=0)
        micro_labels = torch.chunk(labels, args.micro_batches, dim=0)
        ce_loss = None
        aux_loss = None
        layer_metadata = None
        for chunk_ids, chunk_labels in zip(micro_inputs, micro_labels):
            with autocast_context(device, args.dtype):
                logits, aux_loss_chunk, layer_metadata = model(chunk_ids)
            ce_loss_chunk = F.cross_entropy(logits.view(-1, vocab_size), chunk_labels.view(-1))
            loss = (ce_loss_chunk + args.aux_weight * aux_loss_chunk) / max(len(micro_inputs), 1)
            loss.backward()
            ce_loss = ce_loss_chunk
            aux_loss = aux_loss_chunk
        optimizer.step()

        if rank == 0 and (step % args.log_interval == 0 or step == 1):
            # Report routing statistics
            meta = layer_metadata[0]  # first layer
            cv = meta["load_balance_cv"]
            dropped = meta["tokens_dropped"]
            counts = meta["expert_counts"]
            print(f"step {step:4d}/{args.steps} | "
                  f"CE={ce_loss.item():.4f} aux={aux_loss.item():.4f} | "
                  f"dropped={dropped} cv={cv:.3f} counts={counts}")
        return ce_loss, aux_loss, layer_metadata

    for warmup_step in range(1, max(int(args.warmup_steps), 0) + 1):
        run_single_step(warmup_step)
        if not args.no_step_end_synchronize:
            synchronize_completed_iteration(device)
        if rank == 0 and warmup_step == max(int(args.warmup_steps), 0):
            print(f"warmup {warmup_step:4d}/{args.warmup_steps} complete")

    for step in range(1, args.steps + 1):
        with step_window(step):
            run_single_step(step)
            if not args.no_step_end_synchronize:
                synchronize_completed_iteration(device)

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("Training complete.")


if __name__ == "__main__":
    main()
