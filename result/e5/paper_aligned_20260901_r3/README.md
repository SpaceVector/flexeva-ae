# E5 paper-scale server result

Guarded server run `e5-paper-scale-rounds-20260901-r3` completed with exit code
0 in about 47 wall-minutes. Top-level and build stderr were empty. The run
freshly captured one anchor and 32 distinct route candidates at 16 logical
ranks and EP=8, producing 10,378,875,140 bytes of raw trace/marker data.

The paper-scale reference model used 64 micro-batches, 64 layers, sequence
length 256, hidden size 512, world size 16, EP=8, and DP=2. Table 8 contains
three isolated repeats for every evaluator/K cell:

| Evaluator | K=1 | K=8 | K=32 | Marginal RSS/candidate |
| --- | ---: | ---: | ---: | ---: |
| Maya-full | 0.62 GiB | 2.93 GiB | 10.73 GiB | 334.03 MiB |
| Maya-trace-RAS | 0.27 GiB | 0.47 GiB | 1.14 GiB | 28.95 MiB |
| FlexEva | 0.27 GiB | 0.27 GiB | 0.67 GiB | 13.39 MiB |

The speed experiment measured every candidate as a paired round three times.
The primary ratio excludes one-time anchor initialization:

```text
speedup_i = Maya-full_i / FlexEva-refresh_i
```

Across the 32 round medians, the mean is 13.104x, median 13.095x, range
12.730--13.350x, IQR 0.140x, and population CV 0.94%. The ratio of the 32
round-median totals is 13.096x and is secondary. Including the measured 1.961 s
anchor once gives a cumulative round-32 speedup of 11.394x. All 96 paired
samples have zero feedback-relative error.

`memory_measurements.csv` and `memory_summary.csv` contain the RSS inputs and
aggregates. `speed_samples.csv`, `speed_per_round.csv`, and `speed_summary.csv`
contain paired timings, round medians, and the final distribution. The two
integrity JSON files record independent verifier decisions.

## Evidence boundary

This is a paper-scale reference-method result, not a recovered production
end-to-end reproduction. Fresh FakeCUDA traces ground route identity and
lineage at a reduced capture shape; paper-scale states use the recovered
abstract event model. The original “different parallelism” candidate manifest
and an independent production selective executor are unavailable. Therefore
the stable 13.10x result does not reproduce or validate the submitted 2.48x.
The full raw traces remain server-only.
