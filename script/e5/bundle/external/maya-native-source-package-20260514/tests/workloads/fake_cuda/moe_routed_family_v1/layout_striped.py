#!/usr/bin/env python3
"""Sibling that changes only expert ownership layout."""

from __future__ import annotations

from common import FamilyVariant, run_variant


if __name__ == "__main__":
    run_variant(
        FamilyVariant(
            variant_id="layout_striped",
            description="Same routed-MoE program as anchor, but striped expert ownership changes routing targets and A2A balance.",
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
                "expert_layout": "striped",
                "log_interval": 1,
            },
        )
    )
