#!/usr/bin/env python3
"""Shared runner for a reviewable routed-MoE sibling family.

This family is intentionally built as a small set of nearby code/runtime-semantic
siblings around the existing `moe_topk.py` workload, so a human can review the
exact code differences instead of only reading config tables.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


_BASE_PATH = Path(__file__).resolve().parents[1] / "moe_topk.py"
_SPEC = importlib.util.spec_from_file_location("flexsim_moe_topk_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"failed to load base workload from {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)
BASE = _BASE


BASE_ARG_DEFAULTS = {
    "steps": 50,
    "batch_size": 4,
    "seq_len": 128,
    "hidden_size": 256,
    "num_experts": 4,
    "num_layers": 4,
    "top_k": 2,
    "capacity_factor": 1.25,
    "ep_size": 0,
    "micro_batches": 1,
    "recompute": False,
    "expert_layout": "contiguous",
    "lr": 1e-4,
    "log_interval": 10,
    "aux_weight": 0.01,
}


@dataclass(frozen=True)
class FamilyVariant:
    variant_id: str
    description: str
    default_overrides: dict[str, Any]
    gate_cls: type[nn.Module] | None = None
    ep_layer_cls: type[nn.Module] | None = None


def _apply_variant_defaults(args, overrides: dict[str, Any]) -> None:
    argv = sys.argv[1:]
    for key, value in overrides.items():
        flag = f"--{key.replace('_', '-')}"
        explicitly_set = any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)
        if explicitly_set:
            continue
        current = getattr(args, key)
        if current == BASE_ARG_DEFAULTS[key]:
            setattr(args, key, value)


class VariantTopKMoELayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        *,
        top_k: int,
        capacity_factor: float,
        gate_cls: type[nn.Module],
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.gate = gate_cls(hidden_size, num_experts, top_k, capacity_factor)

        intermediate_size = hidden_size * 4
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, intermediate_size, bias=False),
                    nn.GELU(),
                    nn.Linear(intermediate_size, hidden_size, bias=False),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):
        batch_size, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        dispatch_mask, combine_weights, aux_loss, metadata = self.gate(x_flat)
        expert_counts_list = list(metadata.get("expert_counts", []))

        expert_outputs = torch.zeros_like(x_flat)
        for expert_id in range(self.num_experts):
            mask = dispatch_mask[expert_id]
            expert_input = torch.matmul(mask, x_flat)
            if expert_id < len(expert_counts_list) and int(expert_counts_list[expert_id]) > 0:
                expert_out = self.experts[expert_id](expert_input)
                expert_outputs += torch.matmul(mask.t(), expert_out)

        output = expert_outputs.view(batch_size, seq_len, hidden)
        return output, aux_loss, metadata


class VariantMoETransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_experts,
        top_k,
        capacity_factor,
        *,
        ep_group=None,
        ep_size=1,
        recompute=False,
        expert_layout="contiguous",
        gate_cls=None,
        ep_layer_cls=None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.recompute = recompute
        self.num_experts = num_experts

        gate_cls = gate_cls or _BASE.TopKGate
        if ep_group is not None and ep_size > 1:
            layer_cls = ep_layer_cls or _BASE.ExpertParallelMoELayer
            self.moe = layer_cls(
                hidden_size,
                num_experts,
                top_k,
                capacity_factor,
                ep_group=ep_group,
                ep_size=ep_size,
                expert_layout=expert_layout,
            )
        else:
            self.moe = VariantTopKMoELayer(
                hidden_size,
                num_experts,
                top_k=top_k,
                capacity_factor=capacity_factor,
                gate_cls=gate_cls,
            )

    def _forward_impl(self, x):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h

        h = self.ln2(x)
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
            use_reentrant = False if _BASE.is_fakecuda_device(x.device) else True
            x, aux_loss = _BASE.checkpoint(_recompute_fn, x, use_reentrant=use_reentrant)
            metadata = {
                "expert_counts": [0 for _ in range(self.num_experts)],
                "tokens_dropped": 0,
                "capacity": 0,
                "load_balance_cv": 0.0,
                "recompute": True,
            }
            return x, aux_loss, metadata
        return self._forward_impl(x)


class VariantMoETopKModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size,
        hidden_size,
        num_layers,
        num_heads,
        num_experts,
        max_seq_len,
        top_k,
        capacity_factor,
        ep_group=None,
        ep_size=1,
        micro_batches=1,
        recompute=False,
        expert_layout="contiguous",
        gate_cls=None,
        ep_layer_cls=None,
    ):
        super().__init__()
        self.micro_batches = max(1, micro_batches)
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.blocks = nn.ModuleList(
            [
                VariantMoETransformerBlock(
                    hidden_size,
                    num_heads,
                    num_experts,
                    top_k,
                    capacity_factor,
                    ep_group=ep_group,
                    ep_size=ep_size,
                    recompute=recompute,
                    expert_layout=expert_layout,
                    gate_cls=gate_cls,
                    ep_layer_cls=ep_layer_cls,
                )
                for _ in range(num_layers)
            ]
        )
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


def run_variant(variant: FamilyVariant) -> None:
    args = _BASE.parse_args()
    _apply_variant_defaults(args, variant.default_overrides)

    setup_result = _BASE.setup_dist()
    rank, world_size, local_rank, device = setup_result[:4]
    if _BASE.is_fakecuda_device(device) and args.recompute:
        args.recompute = False

    torch.manual_seed(42 + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + rank)

    ep_size = args.ep_size if args.ep_size > 0 else world_size
    ep_size = min(ep_size, world_size)
    assert args.num_experts % ep_size == 0
    assert world_size % ep_size == 0
    ep_group, dp_group, dp_size, rank_in_ep, _rank_in_dp = _BASE.build_parallel_groups(
        rank,
        world_size,
        ep_size,
    )

    vocab_size = 32000
    num_heads = max(1, args.hidden_size // 64)

    model = VariantMoETopKModel(
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
        gate_cls=variant.gate_cls,
        ep_layer_cls=variant.ep_layer_cls,
    ).to(device)

    if dp_size > 1:
        model = _BASE.wrap_ddp(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            process_group=dp_group,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_params = sum(p.numel() for p in model.parameters())
    wrapper_type = type(model).__name__

    if rank == 0:
        print(
            {
                "variant_id": variant.variant_id,
                "variant_description": variant.description,
                "world_size": world_size,
                "ep_size": ep_size,
                "dp_size": dp_size,
                "rank_in_ep": rank_in_ep,
                "num_experts": args.num_experts,
                "top_k": args.top_k,
                "capacity_factor": args.capacity_factor,
                "micro_batches": args.micro_batches,
                "recompute": args.recompute,
                "expert_layout": args.expert_layout,
                "wrapper_type": wrapper_type,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "params_m": round(num_params / 1e6, 2),
            }
        )

    def run_single_step(step: int):
        input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)
        labels = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)
        optimizer.zero_grad(set_to_none=True)
        micro_inputs = torch.chunk(input_ids, args.micro_batches, dim=0)
        micro_labels = torch.chunk(labels, args.micro_batches, dim=0)
        ce_loss = None
        aux_loss = None
        layer_metadata = None
        for chunk_ids, chunk_labels in zip(micro_inputs, micro_labels):
            logits, aux_loss_chunk, layer_metadata = model(chunk_ids)
            ce_loss_chunk = F.cross_entropy(logits.view(-1, vocab_size), chunk_labels.view(-1))
            loss = (ce_loss_chunk + args.aux_weight * aux_loss_chunk) / max(len(micro_inputs), 1)
            loss.backward()
            ce_loss = ce_loss_chunk
            aux_loss = aux_loss_chunk
        optimizer.step()

        if rank == 0 and (step % args.log_interval == 0 or step == 1):
            meta = layer_metadata[0]
            cv = meta.get("load_balance_cv", 0.0)
            dropped = meta.get("tokens_dropped", 0)
            counts = meta.get("expert_counts", [])
            print(
                f"{variant.variant_id} step {step:4d}/{args.steps} | "
                f"CE={ce_loss.item():.4f} aux={aux_loss.item():.4f} | "
                f"dropped={dropped} cv={cv:.3f} counts={counts}"
            )
        return ce_loss, aux_loss, layer_metadata

    for warmup_step in range(1, max(int(args.warmup_steps), 0) + 1):
        run_single_step(warmup_step)
        if not args.no_step_end_synchronize:
            _BASE.synchronize_completed_iteration(device)
        if rank == 0 and warmup_step == max(int(args.warmup_steps), 0):
            print(f"warmup {warmup_step:4d}/{args.warmup_steps} complete")

    for step in range(1, args.steps + 1):
        with _BASE.step_window(step):
            run_single_step(step)
            if not args.no_step_end_synchronize:
                _BASE.synchronize_completed_iteration(device)

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f"Variant {variant.variant_id} complete.")
