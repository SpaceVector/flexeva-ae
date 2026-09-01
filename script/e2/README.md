# E2 implementation

The E2 implementation is split by responsibility:

```text
script/e2/
├── trace_similarity/  # launch, capture, parse, and compare fresh traces
├── workload/          # GPT and Routed-MoE workloads executed by torchrun
├── collect_table4.py  # combine fresh driver summaries into Table 4 CSV
├── collect_figure5.py # enforce native/large-trace provenance and build CSVs
├── run_figure5.sh     # 8/16-GPU native capture + 32/64/128 trace replay
├── validate_results.py # validate retained and freshly generated results
└── plot_figure5.py    # render generated Figure 5 CSVs
```

Use [`script/run_e2`](../run_e2) once on node 0 as the normal Figure 5
entrypoint; the shared two-node launcher starts the guarded peer through SSH
and returns its native traces before node 0 builds the final data and plot.
`script/run_e2 table4` selects the independent single-node Table 4 path. The
retained `result/e2/figure5*.csv` files are references only; fresh Figure 5
data is stored below `result/e2/generated_figure5/<run-id>/`.
