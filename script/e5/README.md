# E5 implementation

`collect_real_routes.py`, `measure_paper_aligned.py`,
`measure_case_speedup.py`, and `run_paper_aligned.sh` implement the primary
32-distinct-candidate peak-RSS and paired per-round speedup paths. `bundle/`
retains the frozen evaluator sources for the separate submitted-value audit.
Both contracts are documented in [`E5.md`](E5.md).

Run from the repository root:

```bash
script/run_e5 audit
script/run_e5
script/run_e5 verify
script/run_e5 paper-self-test
script/run_e5 paper
script/run_e5 paper-verify
```
