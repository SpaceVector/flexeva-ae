# E2 trace output

`script/run_e2 table4` generates Table 4 FakeCUDA traces here. It creates one
directory per workload and case:

```text
trace/e2/table4/gpt/<case>/traces/
trace/e2/table4/routed-moe/<case>/traces/
```

Each case contains one `rank_<n>.jsonl` FakeCUDA capture and one
`rank_<n>_markers.jsonl` step-marker file per logical rank. These files are
runtime output, not checked-in input data. The generated summary and derived
CSV are written under `result/e2/generated_table4/` and
`result/e2/table4_from_trace.csv`.

To replay a supplied trace tree instead of launching the workload, use the
driver's `--reuse-existing-traces --input-trace-root PATH` options.

`script/run_e2 figure5` keeps a separate tree:

```text
trace/e2/figure5/gpt/{8,16}/real/
trace/e2/figure5/gpt/{8,16,32,64,128}/emulated/
trace/e2/figure5/moe/<case>/real/
trace/e2/figure5/moe/<case>/emulated/
```

Only the 8/16-GPU GPT and 16-GPU MoE `real/` directories are produced by
native CUDA/NCCL execution. The 32/64/128 real traces remain links under
`large-cluster/e2`; the local `emulated/` directories are evaluator inputs,
not substitutes for real execution.
