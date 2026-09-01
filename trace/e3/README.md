# E3 trace output

Figure 6 writes fresh two-node rank traces here:

```text
trace/e3/figure6/gpt/<capture>/traces/rank_*.jsonl
trace/e3/figure6/moe/<capture>/traces/rank_*.jsonl
```

Each capture contains 16 rank files and 16 marker files. Node 1 transfers its
eight-rank files to node 0 before analysis; no precomputed raw trace is used.

Figure 7 preserves its binary and JSONL traces beside each guarded run under
`result/e3/generated/figure7/<run-id>/formal/`. Keeping these traces with the
candidate manifest and timing checkpoints prevents result files from being
mixed across guarded runs.
