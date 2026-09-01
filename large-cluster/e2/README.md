# E2 large-cluster traces

Figure 5(a)'s 32-, 64-, and 128-GPU points are replayed from the existing
real-GPU traces linked here:

```text
gpt-32  -> /c20250205/ymx/megatron-lm/fig13_query_nonblocking_32gpu_20260418/gpus_32
gpt-64  -> /c20250205/ymx/megatron-lm/fig13_controlplane_default_freshreal64_20260418/gpus_64
gpt-128 -> /c20250205/ymx/megatron-lm/fig13_freshpair_measure_directproot_markertrace_128gpu_20260419/gpus_128
```

The links deliberately fail closed when the supplied trace mount is absent.
Do not replace them with retained CSV values or small-machine runs. The 8- and
16-GPU points are fresh native CUDA/NCCL captures and therefore live under
[`trace/e2/figure5/`](../../trace/e2/README.md), not here.

The independently trained estimator used by the evaluator is included as
`estimator.json`. `FIGURE5_ESTIMATOR_MODEL` can select another independent
model when reproducing the experiment in a different environment.
