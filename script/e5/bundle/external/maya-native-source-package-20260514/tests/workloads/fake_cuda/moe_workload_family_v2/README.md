# Routed MoE Workload Family v2

This directory groups an anchor and related Routed-MoE candidates used by the
evaluator tests. The candidates keep the same training shape while changing
load balancing, communication scheduling, or locality policy.

## Files

- [`workload_family.json`](workload_family.json): workload configuration,
  optimization goals, rounds, and candidate definitions.
- [`family.py`](family.py): manifest parser and candidate selection helpers.
- [`playbook.json`](playbook.json): predefined candidate combinations used by
  the tests.
- [`playbook.py`](playbook.py): playbook parser.
- [`moe_routed_family_v1`](../moe_routed_family_v1): candidate entry points.

The historical `simulate_moe_workload_family_live_example.py` driver is not
included in this checkout.

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

## Select candidates

```python
from tests.workloads.fake_cuda.moe_workload_family_v2.family import (
    load_workload_family,
    select_round_candidates,
)

family = load_workload_family()
round_spec = family.round_specs["round_load_skew_r1"]
candidates = select_round_candidates(
    family,
    goal_id=round_spec.goal_id,
    budget=round_spec.parallel_budget,
)
```

These files test evaluator reuse and refresh across related programs. They do
not implement candidate generation or production optimization policy.
