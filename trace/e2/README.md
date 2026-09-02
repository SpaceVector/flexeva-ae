# E2 generated traces

`script/run_e2` creates the current Table 4 and Figure 5 traces below
run-specific directories:

```text
trace/e2/table4/<run-id>/
trace/e2/figure5/<run-id>/
```

Table 4 records one trace and marker file per logical rank. Figure 5 freshly
captures the 8/16-GPU real paths and generates evaluator traces for all scales.
The supplied 32/64/128-GPU real traces remain under `large-cluster/e2/`.
