#!/usr/bin/env python3
"""Sibling with locality-aware near-tie routing instead of global locality bias."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn

from common import BASE, FamilyVariant, run_variant


class NearTieLocalityTopKGate(nn.Module):
    """A drop-in gate that only prefers local experts under near-tie conditions."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, capacity_factor: float = 1.25):
        from local_expert_bias import LocalExpertBiasTopKGate

        super().__init__()
        self._delegate = LocalExpertBiasTopKGate(hidden_size, num_experts, top_k, capacity_factor)
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.margin = 0.10

    def forward(self, x):
        num_tokens = x.shape[0]
        gate_logits = self._delegate.gate(x)

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            ep_size = dist.get_world_size()
            if self.num_experts % ep_size == 0:
                local_experts = self.num_experts // ep_size
                rank = dist.get_rank()
                start = rank * local_experts
                end = start + local_experts
                local_logits = gate_logits[:, start:end]
                global_max = gate_logits.max(dim=-1, keepdim=True).values
                near_tie = local_logits >= (global_max - self.margin)
                gate_logits[:, start:end] = gate_logits[:, start:end] + near_tie.to(gate_logits.dtype) * self._delegate.local_bias

        fakecuda_cpu_routing = BASE.is_fakecuda_device(x.device) and self.top_k > 1
        if fakecuda_cpu_routing:
            gate_logits_cpu = BASE.compute_cpu_gate_logits(x, self._delegate.gate)
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                ep_size = dist.get_world_size()
                if self.num_experts % ep_size == 0:
                    local_experts = self.num_experts // ep_size
                    rank = dist.get_rank()
                    start = rank * local_experts
                    end = start + local_experts
                    local_logits = gate_logits_cpu[:, start:end]
                    global_max = gate_logits_cpu.max(dim=-1, keepdim=True).values
                    near_tie = local_logits >= (global_max - self.margin)
                    gate_logits_cpu[:, start:end] = gate_logits_cpu[:, start:end] + near_tie.to(gate_logits_cpu.dtype) * self._delegate.local_bias
            gate_probs_cpu = F.softmax(gate_logits_cpu, dim=-1)
            top_k_gates, top_k_indices = torch.topk(gate_probs_cpu, self.top_k, dim=-1)
            gate_probs = gate_probs_cpu.to(device=x.device, dtype=x.dtype)
        else:
            gate_probs = F.softmax(gate_logits, dim=-1)
            top_k_gates, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=-1, keepdim=True) + 1e-9)

        capacity = int(self.capacity_factor * num_tokens * self.top_k / self.num_experts)
        capacity = max(capacity, 1)
        dispatch_mask = torch.zeros(self.num_experts, capacity, num_tokens, device=x.device, dtype=x.dtype)
        combine_weights = torch.zeros(num_tokens, self.num_experts, device=x.device, dtype=x.dtype)
        expert_counts = (
            torch.zeros(self.num_experts, dtype=torch.long)
            if fakecuda_cpu_routing
            else torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
        )
        expert_token_indices = [[] for _ in range(self.num_experts)] if fakecuda_cpu_routing else None
        tokens_dropped = 0

        for i in range(num_tokens):
            for k in range(self.top_k):
                expert_id = int(top_k_indices[i, k].item())
                pos = int(expert_counts[expert_id].item())
                if pos < capacity:
                    dispatch_mask[expert_id, pos, i] = 1.0
                    gate_weight = float(top_k_gates[i, k].item()) if fakecuda_cpu_routing else top_k_gates[i, k]
                    combine_weights[i, expert_id] = gate_weight
                    expert_counts[expert_id] += 1
                    if expert_token_indices is not None:
                        expert_token_indices[expert_id].append(i)
                else:
                    tokens_dropped += 1

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
            "near_tie_margin": self.margin,
            "locality_bias": self._delegate.local_bias,
        }
        if expert_token_indices is not None:
            metadata["expert_token_indices"] = expert_token_indices
        return dispatch_mask, combine_weights, aux_loss, metadata


if __name__ == "__main__":
    run_variant(
        FamilyVariant(
            variant_id="near_tie_locality_router",
            description="Code-level sibling: only apply locality preference when local and remote expert scores are near tied.",
            default_overrides={
                "batch_size": 2,
                "seq_len": 64,
                "hidden_size": 128,
                "num_layers": 2,
                "num_experts": 8,
                "top_k": 2,
                "capacity_factor": 1.25,
                "ep_size": 2,
                "micro_batches": 2,
                "recompute": True,
                "expert_layout": "contiguous",
                "log_interval": 1,
            },
            gate_cls=NearTieLocalityTopKGate,
        )
    )
