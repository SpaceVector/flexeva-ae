#!/usr/bin/env python3
"""Anchor sibling for routed-MoE family v1."""

from __future__ import annotations

from common import FamilyVariant, run_variant


if __name__ == "__main__":
    run_variant(
        FamilyVariant(
            variant_id="anchor_baseline",
            description="Anchor routed-MoE sibling with contiguous expert ownership and recompute barrier.",
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
        )
    )
