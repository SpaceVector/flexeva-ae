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

`script/run_all` is a thin aggregate of the five experiment entry points:
`script/run_e1`, `script/run_e2`, `script/run_e3`, `script/run_e4`, and
`script/run_e5`. Each entry point performs its required trace validation or
measurements, followed by table construction and plotting. Distributed entries
start and wait for the peer automatically; do not start a matching command
there.

The workflow stops at the first failure. Success ends with:

```text
AE full reproduction: PASS (<run-id>)
```

## Input and output policy

Generated result tables and PDFs are not tracked. They are created by the
experiment entry points and ignored by Git.

The supplied large-scale inputs are:

- E1: five 128-GPU raw trace sets and one historical trajectory ledger whose
  rows carry the original benchmark, patch, and log SHA-256 fingerprints;
- E2: the 32-, 64-, and 128-GPU traces used by Figure 5.

E1 cannot run its original 16-node job on the reviewer server. It validates
the raw trace coverage, Time direction, and A2A trajectory before reconstructing
Figure 1 from the fingerprinted ledger. It does not use FakeCUDA or the current
16 GPUs to replace the original Drop/Reroute measurements. E2's 8- and 16-GPU
traces and every E3--E5 measurement are generated during the current run.

Generated artifacts are written below:

```text
result/e1/generated/<run-id>/
result/e2/generated/{table4,figure5}/<run-id>/
result/e3/generated/{figure6,figure7,figure8}/<run-id>/
result/e4/generated/<run-id>/
result/e5/generated/<run-id>/
trace/
plot/
```

E5's full capture stays in the guarded server run directory because it is
large; its final Table 8 and speedup tables are copied to the result directory
above.

The supplied trace links under `large-cluster/` target mounts on the evaluation
server and may be broken in another clone. Equivalent mounts can be selected
with:

```bash
export E1_TRACE_ROOT=/path/to/historical_sparse_moe
export FIGURE5_LARGE_CLUSTER_ROOT=/path/to/figure5-large-cluster
export FIGURE5_ESTIMATOR_MODEL=/path/to/independent-estimator.json
```

## Environment requirements

The setup check enforces the reviewer-server contract:

- Linux x86-64 on the configured shared filesystem;
- two nodes, each with eight NVIDIA A100-SXM4-80GB GPUs and full NV12 topology;
- PyTorch 2.8.0+cu128 and CUDA toolkit 12.8;
- g++ 11.4.x, CMake 3.22.1, `git`, `make`, `protoc`, `mpicxx`, and `ssh`;
- sufficient shared storage for raw trace capture.

Useful setup overrides are:

```bash
export CUDA_HOME=/usr/local/cuda
export JOBS=8
export MIN_GPFS_FREE_GIB=20
```

## Repository layout

| Path | Contents |
| --- | --- |
| [`FlexEva/`](FlexEva/README.md) | FlexEva core, Maya-style evaluator, FakeCUDA, and tests |
| `script/` | Setup and E1--E5 experiment entry points |
| `large-cluster/` | Links to the supplied large-scale raw traces |
| `result/` | Created result tables and measurement summaries |
| `trace/` | Created raw traces from runnable scales |
| `plot/` | Created paper figures |
