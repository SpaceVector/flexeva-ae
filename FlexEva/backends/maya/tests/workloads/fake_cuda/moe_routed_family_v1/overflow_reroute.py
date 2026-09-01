#!/usr/bin/env python3
"""Sibling with a local code change in overflow handling."""

from __future__ import annotations

import torch

from common import BASE, FamilyVariant, run_variant


class OverflowRerouteTopKGate(BASE.TopKGate):
    """Reroute overflow tokens to the currently lightest expert instead of dropping.

    This is a real code-level sibling:
    - the anchor drops overflow tokens when all selected experts are full
    - this variant performs a bounded reroute to the lightest expert with free capacity
    """

    def forward(self, x):
        dispatch_mask, combine_weights, aux_loss, metadata = super().forward(x)
        fakecuda_cpu_routing = BASE.is_fakecuda_device(x.device) and self.top_k > 1
        expert_counts = (
            torch.tensor(metadata["expert_counts"], dtype=torch.long)
            if fakecuda_cpu_routing
            else torch.tensor(metadata["expert_counts"], dtype=torch.long, device=x.device)
        )
        expert_token_indices = metadata.get("expert_token_indices")
        if fakecuda_cpu_routing and expert_token_indices is None:
            expert_token_indices = [[] for _ in range(self.num_experts)]
        capacity = int(metadata["capacity"])
        rerouted_tokens = 0

        if capacity <= 0:
            return dispatch_mask, combine_weights, aux_loss, metadata

        if expert_token_indices is not None:
            dropped_tokens = BASE.infer_dropped_token_list(dispatch_mask.shape[-1], expert_token_indices)
        else:
            dropped_mask = combine_weights.sum(dim=-1) == 0
            dropped_tokens = (dropped_mask.nonzero(as_tuple=True)[0]).tolist()
        for token_idx in dropped_tokens:
            lightest = int(torch.argmin(expert_counts).item())
            pos = int(expert_counts[lightest].item())
            if pos >= capacity:
                continue
            dispatch_mask[lightest, pos, token_idx] = 1.0
            combine_weights[token_idx, lightest] = 1.0
            expert_counts[lightest] += 1
            if expert_token_indices is not None:
                expert_token_indices[lightest].append(int(token_idx))
            rerouted_tokens += 1

        metadata = dict(metadata)
        metadata["expert_counts"] = expert_counts.tolist()
        if expert_token_indices is not None:
            metadata["expert_token_indices"] = expert_token_indices
        metadata["rerouted_tokens"] = rerouted_tokens
        metadata["tokens_dropped"] = max(int(metadata["tokens_dropped"]) - rerouted_tokens, 0)
        return dispatch_mask, combine_weights, aux_loss, metadata


if __name__ == "__main__":
    run_variant(
        FamilyVariant(
            variant_id="overflow_reroute",
            description="Code-level sibling: overflow tokens are rerouted to the lightest expert instead of being dropped.",
            default_overrides={
                "batch_size": 2,
                "seq_len": 64,
                "hidden_size": 128,
                "num_layers": 2,
                "num_experts": 8,
                "top_k": 2,
                "capacity_factor": 1.0,
                "ep_size": 2,
                "micro_batches": 2,
                "recompute": True,
                "expert_layout": "contiguous",
                "log_interval": 1,
            },
            gate_cls=OverflowRerouteTopKGate,
        )
    )
