# FlexEva Core

This directory contains the FlexEva implementation used by the artifact
experiments. Paper workloads, experiment orchestration, retained measurements,
and generated figures live in the repository root.

## Components

- `flexmaya_ras/`: resilient anchor state, partition metadata, selective
  refresh, trace compaction, replay, and the C++ extension.
- `backends/maya/`: the independently authored Maya-style evaluator,
  fake-CUDA runtime, and source-analysis implementation. It is not original
  Maya source.

## Build and test

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
make PYTHON=.venv/bin/python test
```

FlexEva is included directly in the artifact repository, so one checkout is
sufficient for setup and reproduction.
