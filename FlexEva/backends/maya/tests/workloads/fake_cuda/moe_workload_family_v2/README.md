# Routed MoE Workload Family v2

This directory groups an anchor and related Routed-MoE candidates used by the
evaluator tests. The candidates keep the same training shape while changing
load balancing, communication scheduling, or locality policy.

## Files

- [`family.py`](family.py): manifest data types and candidate selection helpers.
- [`playbook.json`](playbook.json): predefined candidate combinations used by
  the tests.
- [`playbook.py`](playbook.py): playbook parser.
- [`moe_routed_family_v1`](../moe_routed_family_v1): candidate entry points.

## Candidate groups

Load balancing:

- `overflow_reroute`
- `layout_striped`
- `local_backup_reroute`
- `balanced_secondary_route`

Communication scheduling:

- `locality_first_dispatch`
- `recompute_eager_sync`
- `staged_priority_dispatch`
- `staggered_microbatch_overlap`

Locality:

- `local_expert_bias`
- `near_tie_locality_router`
- `local_remote_staged_dispatch`

These files test evaluator reuse and refresh across related programs. They do
not implement candidate generation or production optimization policy.
