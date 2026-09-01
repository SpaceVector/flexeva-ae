# FlexEva Artifact Evaluation

FlexEva incrementally evaluates changes to distributed-training workloads. It
reuses an evaluated anchor, identifies the affected source and trace
partitions, refreshes those partitions when possible, and replays the resulting
compact trace.

This repository contains the artifact implementation, experiment drivers,
compact result ledgers, and paper plots. FlexEva is included directly under
`FlexEva/`; no submodule initialization is required.

For reviewers in mainland China, the repository is also mirrored at
[`https://gitee.com/space-line-vector/flexeva-ae`](https://gitee.com/space-line-vector/flexeva-ae).
Setup uses this mirror if it needs to create the peer checkout.

## Primary reviewer entry

```bash
ssh -p 18405 ae_reviewer@182.92.117.22
```

## Reviewer workflow

The supplied AE environment already provides mutual SSH access between the two
nodes. Setup derives and stores the checkout, experiment-filesystem, and
rendezvous settings. Reviewers do not need to enter or record any node-specific
values.

After login, enter the coordinator checkout and prepare both nodes:

```bash
cd flexeva-ae
script/setup
```

`script/setup` reads the preconfigured AE environment, discovers the unique
peer, creates its checkout when missing, verifies the exact commit, and
performs the same uv-only installation on both nodes. No system Python
installation is required. Continue only after the final line is
`AE two-node setup: PASS`.

Then reproduce all paper results with one command on the same coordinator:

```bash
script/run_all
```

Do not start a matching command on the second node. When an experiment needs
both nodes, its runner starts the peer through the existing AE connection,
waits for both nodes, and returns the peer output automatically.

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
| `large-cluster/` | External trace links and the independent Figure 5 estimator |

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
- `curl` or `wget`, plus network access for uv 0.11.7, its managed Python
  3.12.13 distribution, and Python wheels;
- PyTorch 2.8.0+cu128 and CUDA toolkit 12.8;
- g++ 11.4.x, CMake 3.22.1, `git`, `make`, `protoc`, `mpicxx`, `ssh`, and
  `nvidia-smi`.

uv provides Python 3.12.13, its development headers, and its shared library;
the system Python version is ignored. Setup uses the Astral release service for
the checksum-pinned uv binary, npmmirror for managed Python, and Aliyun mirrors
for PyPI and CUDA 12.8 wheels. The full setup check rejects a non-GPFS
filesystem or a different GPU topology.

The distributed runners enforce larger defaults: 500 GiB for E2 Figure 5 and
E3 Figures 6--7, and 50 GiB for E3 Figure 8. Their `*_MIN_FREE_GIB`
environment variables expose the corresponding checks.

Useful overrides are:

```bash
export CUDA_HOME=/usr/local/cuda
export JOBS=8
export MIN_GPFS_FREE_GIB=20
```

To use the upstream services instead, set `UV_DEFAULT_INDEX`,
`UV_PYTHON_INSTALL_MIRROR`, and `TORCH_INDEX_URL` before `script/setup`.

## Automatic two-node execution

The automatic peer launch applies to both `script/run_all` and the individual
experiment runners. E2 Figure 5 and E3 Figures 6--8 connect to the prepared
peer automatically. The single `script/setup` command on the coordinator
prepares both nodes before any experiment starts.

## Individual experiment runners

Except for the separate E1 `real` path, run each command below once on the
coordinator. The runner selects local or automatic two-node execution as
required.

| Experiment | Command or modes | Execution |
| --- | --- | --- |
| E1 | `script/run_e1`; `script/run_e1 real` is the separate 16-node path | Coordinator for the supplied-trace workflow |
| E2 | `FIGURE5_RUN_ID=<id> script/run_e2` for Figure 5; `script/run_e2 table4` for Table 4 | Figure 5 adds the peer automatically; Table 4 stays on the coordinator |
| E3 | `FIGURE6_RUN_ID=<id> script/run_e3`; `FIGURE7_RUN_ID=<id> script/run_e3 figure7 run`; `FIGURE8_RUN_ID=<id> script/run_e3 figure8 run` | Fresh Figure 6--8 runs add the peer automatically; checks stay on the coordinator |
| E4 | `script/run_e4` | Coordinator |
| E5 | `script/run_e5 paper-self-test`; guarded `script/e3/server.sh run <id> 8 -- script/run_e5 paper` | Coordinator |

Figure 7 modes are `self-test`, `probe`, `run`, `report`, and `verify`.
Figure 8 modes are `self-test`, `run`, and `verify`. E5 also retains
`audit`, `run`, and `verify` for the archived submitted-value workflow.
Figure 7 and Figure 8 production runs require `FIGURE7_RUN_ID` and
`FIGURE8_RUN_ID`, respectively. E4 downloads the pinned ASTRA-Sim source into
`.deps/` the first time Table 7 is built.

Fresh runners write below `result/<experiment>/generated/` and `trace/` and
generally require a new run ID or an empty output directory.

## Audit retained results only

After `script/setup`, validate the checked-in ledgers without launching the
full reproduction with:

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
