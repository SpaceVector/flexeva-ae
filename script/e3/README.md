# E3 implementation

The experiment implementations are grouped by paper figure:

```text
script/e3/
├── capture/        # shared FakeCUDA, marker, and SharedEventArena helpers
├── figure6/        # GPT/MoE collection, summary, plotting, and two-node runner
├── figure7/        # production timing, breakdown, plotting, and runner
├── figure8/        # paper-scale collection, accounting, and guarded runner
├── workload/       # exact Megatron and Routed-MoE workloads
├── server.sh       # guarded per-node entrypoint used by Figures 5--8
└── server_guard.sh # environment, GPU-idle, and provenance gates
```

`script/run_e3` defaults to Figure 6; Figures 7 and 8 require explicit modes.
Configure the shared peer settings, then run every Figure 6--8 command only on
node 0. The coordinator launcher starts and guards node 1 through SSH; the
existing capture paths return peer traces before node 0 validates and plots
the final data. See [`E3.md`](../../E3.md) for the commands and checks.
