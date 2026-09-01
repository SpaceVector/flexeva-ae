# Routed MoE Workload Family v2

This directory reframes the routed-MoE assets as a **workload family**, not a
flat variant list.

The paper-facing unit here is:

- one user-defined routed-MoE training requirement,
- one first-pass anchor implementation,
- one enhanced Maya-like evaluator that stays close to real behavior,
- and multiple nearby candidate programs generated to target concrete system
  bottlenecks such as load imbalance or exposed communication critical paths.

The candidate programs are still bounded local siblings, but the family is no
longer "sweep a few knobs and compare." The intended closed loop is:

1. fix a workload requirement and cluster budget,
2. choose an anchor implementation,
3. build an enhanced Maya anchor state, mostly through semantic dry-run,
4. diagnose the current bottleneck in simulation,
5. pick the current optimization goal,
6. spend a candidate budget on multiple semantic alternatives in parallel,
7. evaluate them in the simulator and rerun only uncertain cases,
8. decide whether the next round should keep the same goal or switch.

## Live Example Loop

```text
User workload requirement
  |
  |  (model size, step shape, cluster budget)
  v
+----------------------+
| Anchor implementation|
+----------------------+
  |
  | enhanced Maya semantic dry-run
  | SPMD -> SPSD critical path
  v
+----------------------+
| Anchor semantic state|
| sema + trace prior   |
| + portable structure |
+----------------------+
  |
  | least-required hardware grounding
  | only when actual values and real
  | hardware jointly determine
  | behavior beyond dry-run
  v
+----------------------+
| Cluster adaptation   |
| small hardware subset|
| + timing/topology    |
+----------------------+
  |
  | simulated bottleneck diagnosis
  | load-skew? comm path?
  | locality/topology?
  v
+----------------------+
| Candidate batch      |
| bounded parallel set |
| of nearby code/rctl  |
| mutations            |
+----------------------+
  |
  | enhanced Maya evaluation
  | reuse anchor where possible
  | escalate only uncertain cases
  v
+----------------------+
| Accept next anchor   |
| or reject candidates |
+----------------------+
  |
  | same bottleneck remains?
  | or switch goal next round?
  v
 next optimization round
```

## Family Components

- [workload_family.json](workload_family.json)
  User requirement, simulator role, sparse real-cluster role, optimization
  goals, and candidate definitions.
- [family.py](family.py)
  Helper API for selecting round candidates and materializing simulator round
  plans from the manifest.
- [playbook.json](playbook.json)
  Branching optimization playbook with operator library, curated combinations,
  evolving anchors, and synthetic negatives for evaluator testing.
- [playbook.py](playbook.py)
  Parser/helper API for the 128-GPU playbook layer.
- `simulate_moe_workload_family_live_example.py`
  Historical driver from the pre-migration workspace; not present in this checkout.

Candidate entrypoints are reused from:
- [moe_routed_family_v1](../moe_routed_family_v1)

## Current Candidate Semantics

Anchor:
- `anchor_baseline`: standard top-k routed MoE with capacity-based dropping.

Straggler / load-skew candidates:
- `overflow_reroute`: overflow tokens are rerouted to the lightest expert
  instead of being dropped.
- `layout_striped`: expert ownership changes from contiguous to striped.
- `local_backup_reroute`: dropped tokens first spill to local experts before
  global fallback, favoring hot-spot relief with locality awareness.
- `balanced_secondary_route`: when the primary selected expert is already
  hotter than the secondary one, route to the secondary expert instead.

Communication critical-path candidates:
- `locality_first_dispatch`: destination-rank buffers are packed by routing
  density rather than token order.
- `recompute_eager_sync`: removes recompute and collapses to eager
  single-microbatch synchronization.
- `staged_priority_dispatch`: expert-parallel all-to-all traffic is split into
  two priority stages instead of one monolithic burst.
- `staggered_microbatch_overlap`: increases microbatch staging to change the
  overlap window and communication burst shape.

Topology / locality candidates:
- `local_expert_bias`: routing slightly favors experts already local to the
  current EP rank when scores are close.
- `near_tie_locality_router`: locality-aware tie-breaking that only activates
  when local and remote expert scores are near-tied.
- `local_remote_staged_dispatch`: local-owner traffic is staged before remote
  traffic instead of mixing both in one exchange.

## Expected Use

For a given round:

```python
from tests.workloads.fake_cuda.moe_workload_family_v2.family import (
    load_workload_family,
    select_round_candidates,
)

family = load_workload_family()
round_spec = family.round_specs["round_load_skew_r1"]
candidates = select_round_candidates(family, goal_id=round_spec.goal_id, budget=round_spec.parallel_budget)
```

Then evaluate those candidates inside the enhanced Maya loop, and only use real
hardware for:

- least-required grounding of value-sensitive / hardware-sensitive regions,
- calibration or spot validation,
- and escalation of uncertain winners.

The supporting tempcloud collector was a pre-migration script and is not present
in this checkout.

## Playbook Layer

The live example is not just a flat family of scripts. It also has a
**branching playbook** for evaluator testing on the intended 128-GPU design
target:

- start from a plain anchor,
- curate likely-working single operators,
- compose a bounded set of plausible multi-operator candidates,
- mix in synthetic definite negatives,
- promote the best candidate to the next anchor,
- and allow multiple possible optimization paths with different round counts.

This playbook layer is where we evaluate:
- how many rounds are needed,
- whether the evaluator wastes rounds on bad branches,
- and whether it can recover a good optimization trajectory under budget.

## Boundary

The paper contribution here is the evaluator, not the candidate generator.

- The executor / code agent is responsible for preserving workload intent and
  emitting candidate metadata or sidecars.
- FlexSim consumes that metadata as one possible input, together with the
  anchor and observed execution structure.
- The hard FlexSim problem is:
  - what remains reusable,
  - what must be regenerated,
  - and what least-required hardware subset is necessary when dry-run
    semantics are insufficient.
