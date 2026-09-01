# E4 trace output

Table 6 generates one FakeCUDA trace tree per workload and variant:

```text
trace/e4/table6/<case>/{anchor,candidate}/traces/
  rank_<n>.jsonl
  rank_<n>_markers.jsonl
```

These files are produced by `script/run_e4`; they are not precomputed inputs.
Table 7 is an ASTRA-Sim backend experiment, so its simulation traces remain
with the generated backend results under `result/e4/generated/table7/`.
