# Routed MoE Family v1

This directory contains small Routed-MoE variants used to test routing, layout,
and dispatch changes. `anchor_baseline.py` is the baseline; each remaining
entry changes one policy or code path.

## Variants

- [`anchor_baseline.py`](anchor_baseline.py): baseline program and runtime
  configuration.
- [`layout_striped.py`](layout_striped.py): striped expert ownership.
- [`recompute_eager_sync.py`](recompute_eager_sync.py): eager synchronization
  without recomputation.
- [`overflow_reroute.py`](overflow_reroute.py): reroutes overflow tokens.
- [`locality_first_dispatch.py`](locality_first_dispatch.py): packs
  destination buffers by routing density.
- [`local_backup_reroute.py`](local_backup_reroute.py): prefers local backup
  experts before remote fallback.
- [`balanced_secondary_route.py`](balanced_secondary_route.py): selects the
  secondary expert when the primary is more heavily loaded.
- [`staged_priority_dispatch.py`](staged_priority_dispatch.py): splits
  expert-parallel dispatch into two stages.
- [`staggered_microbatch_overlap.py`](staggered_microbatch_overlap.py): changes
  microbatch staging and communication overlap.
- [`local_expert_bias.py`](local_expert_bias.py): adds a small bias toward
  local experts.
- [`near_tie_locality_router.py`](near_tie_locality_router.py): uses locality
  to break near ties.
- [`local_remote_staged_dispatch.py`](local_remote_staged_dispatch.py): sends
  local-owner traffic before remote-owner traffic.

Shared code is in [`common.py`](common.py), and
[`family_manifest.json`](family_manifest.json) lists the variants.

## Run

From this directory:

```bash
torchrun --nproc_per_node=2 anchor_baseline.py --steps 2
torchrun --nproc_per_node=2 overflow_reroute.py --steps 2
```
