# E4 implementation

E4 covers the two generality tables:

```text
script/e4/
├── workload_generality/  # Table 6: PyTorch workload capture and RAS refresh
├── workload/             # Table 6 model workload implementation
└── backend/              # Table 7: ASTRA-Sim build, inputs, and runner
```

Use [`script/run_e4`](../run_e4). Table 6 raw FakeCUDA traces are written to
`trace/e4/table6/`; Table 7 simulation artifacts and CSV summaries are written
to `result/e4/generated/table7/`.
