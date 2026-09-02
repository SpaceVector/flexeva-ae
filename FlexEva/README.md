# FlexEva Core

This directory contains the FlexEva implementation used by the artifact
experiments. Paper workloads, experiment orchestration, and generated outputs
live in the repository root.

FlexEva incrementally evaluates changes to distributed-training workloads. It
reuses a previously evaluated anchor, identifies the code and trace partitions
affected by a candidate, refreshes those partitions when possible, and replays
the resulting compact trace. This avoids repeating the full evaluation for
every candidate while preserving a full-refresh path for configuration changes.

## Components

- `flexmaya_ras/`: resilient anchor state, partition metadata, selective
  refresh, trace compaction, replay, and the C++ extension.
- `backends/maya/`: the independently authored Maya-style evaluator,
  fake-CUDA runtime, and source-analysis implementation. It is not original
  Maya source.

## Build and test

```bash
script/setup
make -C FlexEva PYTHON=../.venv/bin/python test
```

FlexEva is included directly in the artifact repository, so one checkout is
sufficient for setup and reproduction.
