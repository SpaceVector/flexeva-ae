# Resilient Anchor State (frozen E5 source)

This self-contained package stores the resilient anchor state used by the
submitted-value Table 8 audit. It models anchor code, semantics, runtime
values, traces, grounding, and selective refresh plans.

The E5 runner uses the package through `src/paper_resilient_anchor_state/` and
checks the load-skew adapter with
`tests/test_maya_v2_load_skew_case.py`. Historical workload manifests retain
their original paths; the adapter maps them to the bundled source before use.

Run the package tests from this directory:

```bash
python3 -m pytest -q
```
