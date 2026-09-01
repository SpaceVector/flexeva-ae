#!/usr/bin/env python3
"""Sibling that changes the microbatch staging window."""

from __future__ import annotations

from common import FamilyVariant, run_variant


if __name__ == "__main__":
    run_variant(
        FamilyVariant(
            variant_id="staggered_microbatch_overlap",
            description="Runtime-control sibling: use more microbatches to create a different overlap and synchronization window.",
            default_overrides={
                "batch_size": 2,
                "seq_len": 64,
                "hidden_size": 128,
                "num_layers": 2,
                "num_experts": 8,
                "top_k": 2,
                "capacity_factor": 1.25,
                "ep_size": 2,
                "micro_batches": 4,
                "recompute": False,
                "expert_layout": "contiguous",
                "log_interval": 1,
            },
        )
    )
