# FlexEva Artifact Evaluation

FlexEva incrementally evaluates changes to distributed-training workloads. It
reuses an evaluated anchor, identifies the affected source and trace
partitions, refreshes those partitions when possible, and replays the resulting
compact trace.

## Artifact overview

This artifact provides the FlexEva implementation, experiment drivers, input
descriptions, result processing, and plotting code needed to reproduce the
paper's main results. FlexEva is included directly under `FlexEva/`; no
submodule initialization is required.

| Experiment | Paper output | Detailed guide |
| --- | --- | --- |
| E1 | Figure 1(b)--(c) | [`script/e1/E1.md`](script/e1/E1.md) |
| E2 | Table 4 and Figure 5 | [`script/e2/E2.md`](script/e2/E2.md) |
| E3 | Figures 6--8 | [`script/e3/E3.md`](script/e3/E3.md) |
| E4 | Tables 6 and 7 | [`script/e4/E4.md`](script/e4/E4.md) |
| E5 | Table 8 and Section 7.5 per-round speedup | [`script/e5/E5.md`](script/e5/E5.md) |

## Server access

Access to the provided AE server is authorized by SSH public key. Before
connecting:

1. Contact the authors through the official artifact-evaluation discussion
   channel.
2. Submit the SSH public key that should be authorized for the reviewer
   account. Send only the public key (normally a `.pub` file), never a private
   key.
3. Wait for confirmation that the key has been installed.

After access is confirmed, connect to the coordinator with:

```bash
ssh -p 18405 ae_reviewer@182.92.117.22
```

The second node is already connected to the coordinator through the private AE
environment. Reviewers do not need its address, account, repository path, or
rendezvous settings.

## Repository access

The coordinator normally provides the checkout in the login directory:

```bash
cd flexeva-ae
```

If a fresh checkout is needed, the primary repository is:

```bash
git clone https://github.com/SpaceVector/flexeva-ae.git
cd flexeva-ae
```

Connectivity from the provided AE server to GitHub can be unstable. If a
GitHub clone or pull fails, use the Gitee mirror:

```bash
git clone https://gitee.com/space-line-vector/flexeva-ae.git
cd flexeva-ae
```

For an existing checkout, switch its remote and update it with:

```bash
git remote set-url origin https://gitee.com/space-line-vector/flexeva-ae.git
git pull --ff-only
```

Both repositories publish the same `main` commit. `script/setup` also verifies
that the coordinator and peer use the exact same commit.

## Environment setup

Run setup once from the coordinator checkout:

```bash
script/setup
```

`script/setup` discovers the preconfigured peer, creates or safely
fast-forwards its checkout when needed, and prepares both nodes. It installs
the checksum-pinned uv binary, uses uv to provision Python 3.12.13 and all
Python packages, builds the native components, and runs the complete setup
check on each node. It does not depend on the system Python version.

The default download sources are the Astral release service for uv, npmmirror
for managed Python, and Aliyun mirrors for PyPI and CUDA 12.8 wheels. A
successful setup ends with:

```text
AE two-node setup: PASS
```

Do not start `script/setup` separately on the peer.

## Full reproduction

After setup succeeds, reproduce E1 through E5 from the coordinator with:

```bash
script/run_all
```

Do not start a matching command on the peer. Each distributed runner starts
the peer automatically, waits for both nodes, and returns the peer output to
the coordinator. The workflow stops at the first failed experiment and ends
successfully with:

```text
AE full reproduction: PASS (<run-id>)
```

The command assigns one timestamped run ID and regenerates Table 4, Figures 1
and 5--8, Tables 6--8, and the E5 per-round speedup result. E1 and the
32/64/128-GPU Figure 5 points use the supplied real trace inputs; the remaining
paths launch their documented measurements.

Generated data is written below `result/<experiment>/generated/`, `trace/`,
and `plot/`. Fresh runs generally require a new run ID or an empty output
directory.

## Individual experiment runners

Run these commands once on the coordinator. The runner selects single-node or
automatic two-node execution as required.

| Experiment | Command or modes | Execution |
| --- | --- | --- |
| E1 | `script/run_e1`; `script/run_e1 real` is the separate 16-node path | Coordinator for the supplied-trace workflow |
| E2 | `FIGURE5_RUN_ID=<id> script/run_e2` for Figure 5; `script/run_e2 table4` for Table 4 | Figure 5 adds the peer automatically; Table 4 stays on the coordinator |
| E3 | `FIGURE6_RUN_ID=<id> script/run_e3`; `FIGURE7_RUN_ID=<id> script/run_e3 figure7 run`; `FIGURE8_RUN_ID=<id> script/run_e3 figure8 run` | Figures 6--8 add the peer automatically |
| E4 | `script/run_e4` | Coordinator |
| E5 | `script/run_e5 paper-self-test`; guarded `script/e3/server.sh run <id> 8 -- script/run_e5 paper` | Coordinator |

Figure 7 modes are `self-test`, `probe`, `run`, `report`, and `verify`.
Figure 8 modes are `self-test`, `run`, and `verify`. Figure 7 and Figure 8
production runs require `FIGURE7_RUN_ID` and `FIGURE8_RUN_ID`, respectively.
E4 downloads the pinned ASTRA-Sim source into `.deps/` the first time Table 7
is built.

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

The distributed runners enforce larger free-space defaults: 500 GiB for E2
Figure 5 and E3 Figures 6--7, and 50 GiB for E3 Figure 8. Their
`*_MIN_FREE_GIB` environment variables expose the corresponding checks.

Useful overrides are:

```bash
export CUDA_HOME=/usr/local/cuda
export JOBS=8
export MIN_GPFS_FREE_GIB=20
```

To use upstream package services instead of the default mirrors, set
`UV_DEFAULT_INDEX`, `UV_PYTHON_INSTALL_MIRROR`, and `TORCH_INDEX_URL` before
`script/setup`.

## Repository layout

| Path | Contents |
| --- | --- |
| [`FlexEva/`](FlexEva/README.md) | FlexEva core, the independent Maya-style evaluator, FakeCUDA, and tests |
| `script/` | Setup, validation, plotting, and experiment entry points |
| `result/` | Experiment result tables and generated output locations |
| `plot/` | Paper figures and generated plot outputs |
| `trace/` | Runtime trace locations |
| `large-cluster/` | External trace links and the independent Figure 5 estimator |

## External trace inputs

The four links under `large-cluster/` point to traces mounted on the provided
evaluation servers. They may be broken in a normal clone. The full
`script/run_all` workflow requires them for E1 and the 32/64/128-GPU points in
E2 Figure 5.

Use these overrides when the same data is mounted elsewhere:

```bash
export E1_TRACE_ROOT=/path/to/historical_sparse_moe
export FIGURE5_LARGE_CLUSTER_ROOT=/path/to/figure5-large-cluster
export FIGURE5_ESTIMATOR_MODEL=/path/to/independent-estimator.json
```

The Figure 5 large-cluster root must contain the `gpt-32`, `gpt-64`, and
`gpt-128` trace trees. The experiment fails when required real traces are
missing; it does not substitute synthetic traces.
