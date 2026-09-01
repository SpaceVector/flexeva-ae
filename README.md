# FlexEva Artifact Evaluation

FlexEva incrementally evaluates changes to distributed-training workloads. It
reuses an evaluated anchor, identifies the affected source and trace
partitions, refreshes those partitions when possible, and replays the resulting
compact trace.

This repository contains the artifact implementation, experiment drivers,
compact result ledgers, and paper plots. FlexEva is included directly under
`FlexEva/`; no submodule initialization is required.

## Repository contents

| Path | Contents |
| --- | --- |
| [`FlexEva/`](FlexEva/README.md) | FlexEva core, the independent Maya-style evaluator, FakeCUDA, and tests |
| `script/` | Setup, validation, plotting, and fresh-run entry points |
| `result/` | Checked-in result ledgers and locations for fresh outputs |
| `plot/` | Checked-in Figures 1, 5, 6(a), 6(b), and 8 |
| `trace/` | Runtime trace locations; raw generated rank traces are not committed |
| `large-cluster/` | Links to externally mounted 32/64/128-GPU traces |

The checked-in evidence covers:

| Experiment | Paper output | Detailed guide |
| --- | --- | --- |
| E1 | Figure 1(b)--(c) | [`script/e1/E1.md`](script/e1/E1.md) |
| E2 | Table 4 and Figure 5 | [`script/e2/E2.md`](script/e2/E2.md) |
| E3 | Figures 6 and 8 retained; Figure 7 from a fresh run | [`script/e3/E3.md`](script/e3/E3.md) |
| E4 | Tables 6 and 7 | [`script/e4/E4.md`](script/e4/E4.md) |
| E5 | Table 8 and Section 7.5 per-round speedup | [`script/e5/E5.md`](script/e5/E5.md) |

## Audit the checked-in results

The retained-result audit is CPU-only and does not require `script/setup` or
the external raw traces:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
script/run_all
```

A successful audit ends with:

```text
AE retained-result audit: PASS (E1-E5)
```

`script/run_all` validates the compact ledgers; it does not launch GPU jobs or
regenerate every PDF.

## Prepare a server for fresh runs

Fresh GPU experiments use the environment enforced by `script/check_setup`:

- Linux x86-64 and a checkout located on GPFS, with at least 20 GiB free for
  setup;
- eight NVIDIA A100-SXM4-80GB GPUs with full NV12 topology per node;
- Python 3.12.13 with development headers and a shared library;
- PyTorch 2.8.0+cu128 and CUDA toolkit 12.8;
- g++ 11.4.x, CMake 3.22.1, `git`, `make`, `protoc`, `mpicxx`, and
  `nvidia-smi`.

A normal desktop checkout can run the retained audit, but the strict fresh-run
check will reject a non-GPFS filesystem or a different GPU topology. Deploy the
same revision below the experiment filesystem on every participating node,
then run:

```bash
export PYTHON_BIN=/path/to/python3.12
script/setup
```

`script/setup` installs the Python dependencies and builds the FlexEva C++
extension, FakeCUDA, CppEvent, the CUDA/NCCL/cuBLAS interposition wrappers, and
PRoot. It finishes by running `script/check_setup`; success is reported as
`AE setup: PASS`.

The distributed runners enforce larger defaults: 500 GiB for E2 Figure 5 and
E3 Figures 6--7, and 50 GiB for E3 Figure 8. Their `*_MIN_FREE_GIB`
environment variables expose the corresponding checks.

Useful overrides are:

```bash
export CUDA_HOME=/usr/local/cuda
export JOBS=8
export MIN_GPFS_FREE_GIB=20
```

## Two-node experiments

E2 Figure 5 and E3 Figures 6--8 use two nodes with eight GPUs each. Both
checkouts must be at the same revision and pass `script/setup`. From node 0,
configure non-interactive SSH access to node 1:

```bash
export PYTHON_BIN=/path/to/node-0/python
export AE_NODE_ROOT=/path/to/node-0/experiment-filesystem
export FLEXMAYA_MASTER_ADDR=<node-0-address>
export FLEXMAYA_PEER_TARGET=<user>@<node-1-address>
export FLEXMAYA_PEER_PORT=22
export FLEXMAYA_PEER_REPO_ROOT=/path/to/node-1/flexeva-ae
export FLEXMAYA_PEER_NODE_ROOT=/path/to/node-1/experiment-filesystem
export FLEXMAYA_PEER_PYTHON=/path/to/node-1/python
```

Check the local guard and shared launcher before a long run:

```bash
script/e3/server.sh self-test
python3 script/lib/two_node.py self-test
```

Each experiment guide lists its run ID and rendezvous/control port variables.

## Experiment entry points

| Experiment | Fresh-run command or modes |
| --- | --- |
| E1 | `script/run_e1` validates supplied 128-GPU traces; `script/run_e1 real` launches the five anchors on 16 nodes |
| E2 | `FIGURE5_RUN_ID=<id> script/run_e2` runs Figure 5; `script/run_e2 table4` runs Table 4 |
| E3 | `FIGURE6_RUN_ID=<id> script/run_e3` runs Figure 6; `script/run_e3 figure7 <mode>` and `script/run_e3 figure8 <mode>` select Figures 7 and 8 |
| E4 | `script/run_e4` runs Table 6 followed by the Table 7 ASTRA-Sim backend cases |
| E5 | `script/run_e5 paper-self-test`; guarded `script/e3/server.sh run <id> 8 -- script/run_e5 paper`; `E5_RESULT_ROOT=<path> script/run_e5 paper-verify` |

Figure 7 modes are `self-test`, `probe`, `run`, `report`, and `verify`.
Figure 8 modes are `self-test`, `run`, and `verify`. E5 also retains
`audit`, `run`, and `verify` for the archived submitted-value workflow.
Figure 7 and Figure 8 production runs require `FIGURE7_RUN_ID` and
`FIGURE8_RUN_ID`, respectively. E4 downloads the pinned ASTRA-Sim source into
`.deps/` the first time Table 7 is built.

Fresh runners write below `result/<experiment>/generated/` and `trace/` and
generally require a new run ID or an empty output directory.

## External trace inputs

The four links under `large-cluster/` point to traces mounted on the original
evaluation servers and may be broken in a normal clone. They are not required
for `script/run_all`, but they are required by the default E1 trace workflow
and the 32/64/128-GPU points in E2 Figure 5.

Use these overrides when the same data is mounted elsewhere:

```bash
export E1_TRACE_ROOT=/path/to/historical_sparse_moe
export FIGURE5_LARGE_CLUSTER_ROOT=/path/to/figure5-large-cluster
export FIGURE5_ESTIMATOR_MODEL=/path/to/independent-estimator.json
```

The Figure 5 large-cluster root must contain the `gpt-32`, `gpt-64`, and
`gpt-128` trace trees. The experiment intentionally fails instead of replacing
missing real traces with retained CSV values or smaller synthetic runs.
