# E1 historical inputs

The reviewer server has 16 GPUs, so E1's five 128-GPU rounds cannot be run
there. `historical_sparse_moe` links to the raw CUDA/NCCL trace mount on the
evaluation server. `trajectory.csv` records the corresponding benchmark,
patch, and log fingerprints used for the paper figure.

`script/run_e1` first validates all 128 ranks and measured-step windows. It
then checks that trace timing has the same improvement direction as the
historical benchmark and that the normalized A2A difference is at most 0.02.
Only after those checks pass does it use the ledger to generate Figure 1.

The trace does not encode code-line counts or token drop/reroute values, so
those fields cannot be reconstructed from the CUDA/NCCL events alone.

Set `E1_TRACE_ROOT` when the same trace tree is mounted elsewhere. Set
`E1_LEDGER` only when validating an equivalent historical ledger.
