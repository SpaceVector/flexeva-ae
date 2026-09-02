# Resilient Anchor State (E5 source snapshot)

This package provides the resilient anchor state used by E5 Table 8. It models
anchor code, runtime values, traces, and selective refresh plans.

The E5 runner uses the package through `src/paper_resilient_anchor_state/` and
checks the load-skew adapter with
`tests/test_maya_v2_load_skew_case.py`. Historical workload manifests keep
their original paths; the adapter maps them to the bundled source before use.

Run the package tests from this directory:

```bash
python3 -m pytest -q
```
