# E2 generated traces

`script/run_e2` creates Table 4 traces below a run-specific directory:

```text
trace/e2/table4/<run-id>/
```

Table 4 records one trace and marker file per logical rank. The default Figure
5 route reads supplied inputs and creates no new native traces. The optional
native mode writes its 8/16-GPU traces below:

```text
trace/e2/figure5/<run-id>/native/
```

The supplied 32/64/128-GPU traces remain under `large-cluster/e2/`.
