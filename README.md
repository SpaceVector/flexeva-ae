# FlexEva Artifact Evaluation

FlexEva incrementally evaluates distributed-training changes by reusing an
evaluated anchor, refreshing affected source and trace partitions, and replaying
the resulting compact trace. The implementation is included directly under
`FlexEva/`; this repository has no submodules.

| Experiment | Reproduced output | Guide |
| --- | --- | --- |
| E1 | Figure 1(b)--(c) | [`script/e1/E1.md`](script/e1/E1.md) |
| E2 | Table 4 and Figure 5 | [`script/e2/E2.md`](script/e2/E2.md) |
| E3 | Figures 6--8 | [`script/e3/E3.md`](script/e3/E3.md) |
| E4 | Tables 6 and 7 | [`script/e4/E4.md`](script/e4/E4.md) |
| E5 | Table 8 and per-round speedup | [`script/e5/E5.md`](script/e5/E5.md) |

## Reviewer server access

Access is authorized by SSH public key. Contact the authors through the
artifact-evaluation discussion channel and send the public key to be installed
for the reviewer account. Send only the public key, never the private key.

After the authors confirm installation, connect to the coordinator:

```bash
ssh -p 18405 ae_reviewer@182.92.117.22
```

The peer node is already connected inside the evaluation environment. Its
address and launch settings are discovered automatically.

## Repository checkout

The coordinator normally already contains the checkout:

```bash
cd ~/flexeva-ae
```

For a fresh checkout, use GitHub:

```bash
git clone https://github.com/SpaceVector/flexeva-ae.git
cd flexeva-ae
```

Connectivity from the evaluation server to GitHub can be unstable. If cloning
or pulling from GitHub fails, use the Gitee mirror:

```bash
git clone https://gitee.com/space-line-vector/flexeva-ae.git
cd flexeva-ae
```

For an existing checkout:

```bash
git remote set-url origin https://gitee.com/space-line-vector/flexeva-ae.git
git pull --ff-only
```

GitHub and Gitee publish the same `main` commit.

### Multiple reviewers sharing one account

Use a separate fresh checkout for each reviewer, for example
`~/reviewer-a/flexeva-ae` and `~/reviewer-b/flexeva-ae`. Run `script/setup` in
each checkout; do not copy another checkout's `.deps/` or `.venv/`.
Setup uses the same home-relative checkout path on the peer, so these two
checkouts also use separate directories on node 1. Explicit or previously
saved `FLEXMAYA_PEER_REPO_ROOT` settings are preserved. For a checkout outside
the home directory, set this variable to a dedicated absolute peer path before
setup.

Guarded runs store their logs, caches, and status in that checkout's
`result/server-runs/<run-id>/`. Reusing an ID in the same checkout is rejected
without modifying the existing run; the same ID in different checkouts does
not share results. Existing data under `~/eurosys27-ae/runs/` is left untouched.
The GPU lock remains shared at `~/eurosys27-ae/.locks/gpu-run.lock`; do not
delete it to bypass a busy run. Coordinate setup and experiment time slots:
separate directories do not provide separate GPUs, ports, or CPU resources,
and not every experiment entry point uses the GPU guard.

## 1. Prepare both nodes

Run setup once on the coordinator:

```bash
script/setup
```

The command discovers the peer, synchronizes its checkout to the coordinator
commit, and prepares both nodes. It installs the pinned uv release, uses uv to
install Python 3.12.13 and the Python packages, builds native components, and
runs the setup checks. The system Python version is not used.

The default sources are npmmirror for uv-managed Python, Aliyun for PyPI and
CUDA wheels, and the configured source mirrors for native dependencies. To use
upstream Python package services, set `UV_DEFAULT_INDEX`,
`UV_PYTHON_INSTALL_MIRROR`, and `TORCH_INDEX_URL` before setup.

Successful setup ends with:

```text
AE two-node setup: PASS
```

Do not run `script/setup` separately on the peer.

## 2. Reproduce E1--E5

Run the complete workflow once on the coordinator:

```bash
script/run_all
```

`script/run_all` calls the five experiment entry points in order:
`script/run_e1`, `script/run_e2`, `script/run_e3`, `script/run_e4`, and
`script/run_e5`. Each entry point performs its required trace validation or
measurements, followed by table construction and plotting. Distributed entries
start and wait for the peer automatically; do not start a matching command
there.

Approximate wall-clock times on the provided reviewer server, after setup, are:

| Entry point | Approximate time |
| --- | ---: |
| `script/run_e1` | less than 1 minute |
| `script/run_e2` | about 1.5 hours |
| `script/run_e3` | about 2 hours |
| `script/run_e4` | about 5.5 hours |
| `script/run_e5` | about 35 minutes |
| `script/run_all` | about 9.5 hours |

Reserve roughly 10 hours for `script/run_all` to allow for system load and the
first ASTRA-Sim build. These estimates exclude `script/setup`. Table 7 cases
and Table 8 memory cells run once. Table 6 replay timing and E5 per-round
speedup retain short repetitions for their wall-clock medians.

The workflow stops at the first failure. Success ends with:

```text
AE full reproduction: PASS (<run-id>)
```

### Resource usage

Approximate planning budgets for the default workflow on the provided reviewer
server, based on existing runs and the execution pipeline with headroom, are:

| Entry point | Approximate host RAM per node | Approximate output storage |
| --- | ---: | ---: |
| `script/run_e1` | about 2 GiB | less than 1 GiB |
| `script/run_e2` | about 64 GiB | about 20 GiB |
| `script/run_e3` | about 64 GiB | about 130 GiB |
| `script/run_e4` | about 32 GiB | about 20 GiB |
| `script/run_e5` | about 32 GiB | about 20 GiB |
| `script/run_all` | about 64 GiB | about 200 GiB |

RAM budgets apply to each active node. Output storage is the combined total
with raw traces retained on both nodes. Since `run_all` executes experiments
sequentially, its RAM budget follows the largest stage while storage accumulates.

The Python environment uses about 7 GiB per node, and about 471 GiB of
supplied traces are already mounted on shared storage. Installation,
supplied inputs, and build/download caches are excluded from the table.

## Input and output policy

Generated result tables and PDFs are not tracked. They are created by the
experiment entry points and ignored by Git.

The supplied E1 inputs contain five 128-GPU raw trace sets and one historical
trajectory ledger whose rows carry the original benchmark, patch, and log
SHA-256 fingerprints.

E1 cannot run its original 16-node job on the reviewer server. It validates
the raw trace coverage, Time direction, and A2A trajectory before reconstructing
Figure 1 from the fingerprinted ledger. It does not use FakeCUDA or the current
16 GPUs to replace the original Drop/Reroute measurements. Every E3--E5
measurement is generated during the current run.

Generated artifacts are written below:

```text
result/e1/generated/<run-id>/
result/e2/generated/table4/<run-id>/
result/e2/generated/figure5/<run-id>/
result/e3/generated/{figure6,figure7,figure8}/<run-id>/
result/e4/generated/<run-id>/
result/e5/generated/<run-id>/
result/server-runs/<run-id>/
trace/
plot/
```

E5's full capture stays in `result/server-runs/<run-id>/` because it is large;
its final Table 8 and speedup tables are copied to `result/e5/generated/`.

The supplied trace links under `large-cluster/` target mounts on the evaluation
server and may be broken in another clone. Equivalent mounts can be selected
with:

```bash
export E1_TRACE_ROOT=/path/to/historical_sparse_moe
```

## Environment requirements

The supplied two-node server provides 64 physical CPU cores (128 hardware
threads) and approximately 1.8 TiB of RAM per node. Both nodes share GPFS
storage and communicate through SSH and sockets.

The setup check verifies:

- Linux x86-64;
- eight NVIDIA A100-SXM4-80GB GPUs per node with full NV12 topology;
- PyTorch 2.8.0+cu128 and CUDA toolkit 12.8;
- g++ 11.4.x, CMake 3.22.1, `git`, `make`, `protoc`, `mpicxx`, and `ssh`;
- free space on the shared filesystem.

Useful setup overrides are:

```bash
export CUDA_HOME=/usr/local/cuda
export JOBS=8
export MIN_GPFS_FREE_GIB=20
```

## Repository layout

| Path | Contents |
| --- | --- |
| [`FlexEva/`](FlexEva/README.md) | Implements incremental evaluation, including the Maya backend, FakeCUDA capture, trace replay, and tests. |
| `script/` | Provides environment setup, E1--E5 experiment drivers, result validation, and plotting; implementations are grouped under `script/eN/`. |
| `large-cluster/` | Supplies the historical traces, trajectory ledger, and source measurements used by E1 and the default Figure 5 reconstruction. |
| `result/` | Stores generated result tables, measurement summaries, and validation records; `result/server-runs/` also holds guarded-run logs, caches, and E5 captures. |
| `trace/` | Stores newly captured raw traces used to construct and validate experiment results. |
| `plot/` | Stores the paper figures generated from the result tables by the plotting scripts. |

## Citation

If you use FlexEva in your research, please cite our paper:

```bibtex
@unpublished{flexeva2027eurosys,
  title  = {{Lightweight Evaluation for Agentic ML Workload Optimization with Resilient Anchor State}},
  author = {Yan, Muxi and Wu, Yinjie and Tang, Bo and Wang, Xiaoting},
  year   = {2027},
  note   = {Submitted to EuroSys 2027}
}
```
