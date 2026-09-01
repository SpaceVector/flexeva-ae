# Routed MoE Family v1

This directory defines a **reviewable sibling family** around the existing
[moe_topk.py](../moe_topk.py) workload.

The point is not to maximize family size. The point is to make the **code
differences explicit**:

- some siblings change only runtime-control settings
- some siblings introduce small local code-path differences that an agent might
  plausibly synthesize

## Family Shape

- [anchor_baseline.py](anchor_baseline.py)
  Anchor program and runtime configuration.
- [layout_striped.py](layout_striped.py)
  Changes expert ownership layout only.
- [recompute_eager_sync.py](recompute_eager_sync.py)
  Changes loop/runtime policy only.
- [overflow_reroute.py](overflow_reroute.py)
  Changes gate overflow handling in code.
- [locality_first_dispatch.py](locality_first_dispatch.py)
  Changes all-to-all packing order in code.
- [local_backup_reroute.py](local_backup_reroute.py)
  Changes overflow recovery to prefer local experts before remote fallback.
- [balanced_secondary_route.py](balanced_secondary_route.py)
  Changes routing to use the second selected expert when the primary is already hotter.
- [staged_priority_dispatch.py](staged_priority_dispatch.py)
  Splits expert-parallel dispatch into two staged bursts instead of one monolithic A2A.
- [staggered_microbatch_overlap.py](staggered_microbatch_overlap.py)
  Changes loop-level staging to create a different communication overlap window.
- [local_expert_bias.py](local_expert_bias.py)
  Changes router logits to mildly prefer current-rank experts when scores are close.
- [near_tie_locality_router.py](near_tie_locality_router.py)
  Uses locality-aware tie-breaking only when local and remote expert scores are near tied.
- [local_remote_staged_dispatch.py](local_remote_staged_dispatch.py)
  Exchanges local-owner traffic before remote-owner traffic to change locality and critical path.

Shared runner and hook points:
- [common.py](common.py)

Manifest:
- [family_manifest.json](family_manifest.json)

## Review Guidance

Start from:
1. [anchor_baseline.py](anchor_baseline.py)
2. [overflow_reroute.py](overflow_reroute.py)
3. [local_backup_reroute.py](local_backup_reroute.py)
4. [locality_first_dispatch.py](locality_first_dispatch.py)

The two code-semantic siblings are the most important for the agent-code story.

## Run Example

From the family directory:

```bash
torchrun --nproc_per_node=2 anchor_baseline.py --steps 2
torchrun --nproc_per_node=2 overflow_reroute.py --steps 2
```
