# E1 large-cluster data

E1's historical trajectory was measured on 128 GPUs. The current AE server
environment is limited to 16 GPUs, so the native cases are not rerun.

On the evaluation servers, `historical_sparse_moe` is a link to
`/c20250205/ymx/historical_sparse_moe`. The default `script/run_e1` trace mode
validates all five 128-rank trace rounds before deriving the retained ledger
and Figure 1. Set `E1_TRACE_ROOT` only when the same trace tree is mounted
elsewhere. The optional `script/run_e1 real` mode does not read this link.
