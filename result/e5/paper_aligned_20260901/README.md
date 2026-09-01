# E5 retained result

This directory contains the compact output of the September 1 Table 8 run.
The experiment captured one anchor and 32 distinct forced-route Routed-MoE
candidates at 16 logical ranks (EP=8), then measured K=1, 8, and 32 resident
candidate states in fresh child processes. Each evaluator/K pair has three
repeats.

`table8.csv` reports the median process-lifetime peak RSS delta.
`table8_retained_diagnostic.csv` reports post-construction `VmRSS` after GC and
`malloc_trim`. Raw measurements, hashes, run identifiers, and verification
status are kept in the adjacent CSV, JSON, and SHA-256 files.

| Evaluator | K=1 | K=8 | K=32 | Marginal peak RSS/candidate |
| --- | ---: | ---: | ---: | ---: |
| Maya-full | 0.62 GiB | 2.93 GiB | 10.73 GiB | 334.03 MiB |
| Maya-trace-RAS | 0.27 GiB | 0.46 GiB | 1.14 GiB | 28.92 MiB |
| FlexEva | 0.27 GiB | 0.27 GiB | 0.67 GiB | 13.44 MiB |

At K=32, FlexEva uses 93.7% less peak RSS than Maya-full and 41.0% less than
Maya-trace-RAS. The measured 11.28x Maya-full/FlexEva ratio covers state
construction only, not the paper's end-to-end case study.

## Measurement scope

Fresh FakeCUDA traces establish route identity and lineage using a reduced
2-micro-batch, 2-layer, sequence-64, hidden-128 capture. Resident states use
the paper shape: 64 micro-batches, 64 layers, sequence length 256, hidden size
512, EP group size 8, and 16 logical ranks. The run therefore measures
paper-shape state memory, not full paper-shape raw-trace capture.
