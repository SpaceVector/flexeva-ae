# E2 scripts

`script/run_e2` is the reviewer entry point. It generates Table 4 and uses the
trace-based Figure 5 route by default. See [`E2.md`](E2.md) for the optional
8/16-GPU native mode and output paths.

## Complete pipeline exercised in the Table 4 experiment

Each GPT configuration and Routed-MoE route has an independent anchor/candidate
pair: AdamW with `foreach=False, fused=False` versus default AdamW. Model dimensions,
parallel geometry, routing, seeds and inputs remain fixed.

The candidate capture records only the optimizer step and trailing work; it still
executes the prefix to establish runtime state. After checking the stream/event
bindings, the evaluator reuses the anchor's prefix graph and replay checkpoints
to build and replay only the new suffix, including both diagnostic projections.
This path supports the optimizer tail with unchanged rank groups and no new
collectives. All captures use blocking NCCL waits and single-threaded autograd.

An independent full candidate capture runs after candidate feedback is published.
Each case checks weighted Jaccard, kernel/NCCL events, event-wait dependencies and
lane order before the next case starts. Capture directories are `anchor/traces/`,
`selective/traces/` and `traces/` (reference). Per-case `candidate.json` and the
comparison `result.json` record the pair, event reuse and incremental work counts;
`table4.csv` contains the eight comparison rows.
