# FlexEva Artifact Evaluation

FlexEva incrementally evaluates changes to distributed-training workloads. It
reuses an evaluated anchor, identifies the affected source and trace
partitions, refreshes those partitions when possible, and replays the resulting
compact trace.

This repository contains the artifact implementation, experiment drivers,
compact result ledgers, and paper plots. FlexEva is included directly under
`FlexEva/`; no submodule initialization is required.

## Reviewer workflow

### 1. Prepare both nodes

Check out the same revision on node 0 and node 1. From the repository root,
run the following command once on each node:

```bash
export PYTHON_BIN=/path/to/python3.12
script/setup
```

`script/setup` installs all Python dependencies, builds the native components,
and checks the required software, GPU topology, and filesystem. A successful
setup ends with `AE setup: PASS`. The command prepares its local node; continue
only after it has passed on both nodes.

### 2. Configure node 0

Set the following variables only on node 0 and configure non-interactive SSH
access to node 1:

```bash
export PYTHON_BIN=/path/to/node-0/python
export AE_NODE_ROOT=/path/to/node-0/experiment-filesystem
export FLEXMAYA_MASTER_ADDR=<node-0-address>
export FLEXMAYA_MASTER_PORT=29500
export FLEXMAYA_CONTROL_PORT=29600
export FLEXMAYA_PEER_TARGET=<user>@<node-1-address>
export FLEXMAYA_PEER_PORT=22
export FLEXMAYA_PEER_REPO_ROOT=/path/to/node-1/flexeva-ae
export FLEXMAYA_PEER_NODE_ROOT=/path/to/node-1/experiment-filesystem
export FLEXMAYA_PEER_PYTHON=/path/to/node-1/python
```

Make sure the supplied large-cluster trace links resolve, or set the overrides
described under [External trace inputs](#external-trace-inputs).

### 3. Run from node 0

Start the complete reproduction once, on node 0:

```bash
script/run_all
```

Do not start a matching command manually on node 1. `script/run_all` invokes
each experiment runner on node 0. When an experiment needs both nodes, that
runner opens SSH to node 1, starts the peer under the server guard, waits for
both nodes, and returns the peer output to node 0 automatically.

`script/run_all` assigns one timestamped base run ID and runs E1 through E5 in
paper order. It regenerates Table 4, Figures 1 and 5--8, Tables 6--8, and the
E5 per-round speedup result. It stops at the first failed experiment and
finishes with:

```text
AE full reproduction: PASS (<run-id>)
```

E1 and the 32/64/128-GPU Figure 5 points use the supplied real trace inputs;
the remaining paths launch their documented fresh measurements. Generated
data is written below `result/<experiment>/generated/`, `trace/`, and `plot/`.

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

## Environment requirements

Fresh GPU experiments use the environment enforced by `script/check_setup`:

- Linux x86-64 and a checkout located on GPFS, with at least 20 GiB free for
  setup;
- eight NVIDIA A100-SXM4-80GB GPUs with full NV12 topology per node;
- Python 3.12.13 with development headers and a shared library;
- PyTorch 2.8.0+cu128 and CUDA toolkit 12.8;
- g++ 11.4.x, CMake 3.22.1, `git`, `make`, `protoc`, `mpicxx`, and
  `nvidia-smi`.

A normal desktop checkout can run the retained audit, but `script/setup` and
the full reproduction reject a non-GPFS filesystem or a different GPU
topology.

The distributed runners enforce larger defaults: 500 GiB for E2 Figure 5 and
E3 Figures 6--7, and 50 GiB for E3 Figure 8. Their `*_MIN_FREE_GIB`
environment variables expose the corresponding checks.

Useful overrides are:

```bash
export CUDA_HOME=/usr/local/cuda
export JOBS=8
export MIN_GPFS_FREE_GIB=20
```

## Automatic two-node execution

The automatic peer launch applies to both `script/run_all` and the individual
experiment runners. E2 Figure 5 and E3 Figures 6--8 connect to node 1 when
their command is started on node 0. Setup remains explicit: `script/setup`
must already have passed on both nodes before any experiment starts.

## Individual experiment runners

Except for the separate E1 `real` path, run each command below once on node 0.
The runner selects local or automatic two-node execution as required.

| Experiment | Command or modes | Execution |
| --- | --- | --- |
| E1 | `script/run_e1`; `script/run_e1 real` is the separate 16-node path | Node 0 for the supplied-trace workflow |
| E2 | `FIGURE5_RUN_ID=<id> script/run_e2` for Figure 5; `script/run_e2 table4` for Table 4 | Figure 5 uses nodes 0 and 1 automatically; Table 4 stays on node 0 |
| E3 | `FIGURE6_RUN_ID=<id> script/run_e3`; `FIGURE7_RUN_ID=<id> script/run_e3 figure7 run`; `FIGURE8_RUN_ID=<id> script/run_e3 figure8 run` | Fresh Figure 6--8 runs use nodes 0 and 1 automatically; checks stay on node 0 |
| E4 | `script/run_e4` | Node 0 |
| E5 | `script/run_e5 paper-self-test`; guarded `script/e3/server.sh run <id> 8 -- script/run_e5 paper` | Node 0 |

Figure 7 modes are `self-test`, `probe`, `run`, `report`, and `verify`.
Figure 8 modes are `self-test`, `run`, and `verify`. E5 also retains
`audit`, `run`, and `verify` for the archived submitted-value workflow.
Figure 7 and Figure 8 production runs require `FIGURE7_RUN_ID` and
`FIGURE8_RUN_ID`, respectively. E4 downloads the pinned ASTRA-Sim source into
`.deps/` the first time Table 7 is built.

Fresh runners write below `result/<experiment>/generated/` and `trace/` and
generally require a new run ID or an empty output directory.

## Audit retained results only

To validate the checked-in ledgers without launching the full reproduction,
install `requirements.txt` in any Python environment and run:

```bash
script/run_all audit
```

A successful audit ends with `AE retained-result audit: PASS (E1-E5)`.

## External trace inputs

The four links under `large-cluster/` point to traces mounted on the original
evaluation servers and may be broken in a normal clone. The full
`script/run_all` workflow requires them for E1 and the 32/64/128-GPU points in
E2 Figure 5; `script/run_all audit` does not require them.

Use these overrides when the same data is mounted elsewhere:

```bash
export E1_TRACE_ROOT=/path/to/historical_sparse_moe
export FIGURE5_LARGE_CLUSTER_ROOT=/path/to/figure5-large-cluster
export FIGURE5_ESTIMATOR_MODEL=/path/to/independent-estimator.json
```

The Figure 5 large-cluster root must contain the `gpt-32`, `gpt-64`, and
`gpt-128` trace trees. The experiment intentionally fails instead of replacing
missing real traces with retained CSV values or smaller synthetic runs.
