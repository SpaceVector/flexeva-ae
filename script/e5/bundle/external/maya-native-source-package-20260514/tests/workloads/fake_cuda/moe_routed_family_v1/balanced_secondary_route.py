#!/usr/bin/env python3
"""Sibling that prefers the secondary selected expert when it is less loaded."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from common import BASE, FamilyVariant, run_variant


class BalancedSecondaryRouteTopKGate(BASE.TopKGate):
    """Keep routing local to the selected pair, but avoid the hotter one."""

    def forward(self, x):
        num_tokens = x.shape[0]
        gate_logits = self.gate(x)
        fakecuda_cpu_routing = BASE.is_fakecuda_device(x.device) and self.top_k > 1
        if fakecuda_cpu_routing:
            gate_logits_cpu = BASE.compute_cpu_gate_logits(x, self.gate)
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
        secondary_promotions = 0
        tokens_dropped = 0

        for i in range(num_tokens):
            selected = [int(expert_id) for expert_id in top_k_indices[i].tolist()]
            weight_by_expert = {
                expert_id: float(top_k_gates[i, k].item()) if fakecuda_cpu_routing else top_k_gates[i, k]
                for k, expert_id in enumerate(selected)
            }
            preferred = selected
            if len(selected) >= 2:
                first, second = selected[0], selected[1]
                if int(expert_counts[first].item()) > int(expert_counts[second].item()) + 1:
                    preferred = [second, first]
                    secondary_promotions += 1
            for rank, expert_id in enumerate(preferred):
                pos = int(expert_counts[expert_id].item())
                if pos < capacity:
                    dispatch_mask[expert_id, pos, i] = 1.0
                    combine_weights[i, expert_id] = weight_by_expert[expert_id]
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
            "secondary_promotions": secondary_promotions,
        }
        if expert_token_indices is not None:
            metadata["expert_token_indices"] = expert_token_indices
        return dispatch_mask, combine_weights, aux_loss, metadata


if __name__ == "__main__":
    run_variant(
        FamilyVariant(
            variant_id="balanced_secondary_route",
            description="Code-level sibling: prefer the second selected expert when it is less loaded than the primary.",
            default_overrides={
                "batch_size": 2,
                "seq_len": 64,
                "hidden_size": 128,
                "num_layers": 2,
                "num_experts": 8,
                "top_k": 2,
                "capacity_factor": 1.1,
                "ep_size": 2,
                "micro_batches": 2,
                "recompute": True,
                "expert_layout": "contiguous",
                "log_interval": 1,
            },
            gate_cls=BalancedSecondaryRouteTopKGate,
        )
    )
