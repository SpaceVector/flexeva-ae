# Independent Maya-style evaluator and source analyzer

This directory contains the authors' independent Maya-style evaluator used by
the FlexMaya artifact. It is not original Maya source. It also contains the
imported paper source-analysis path; machine-local runbooks, raw traces, and
private environment records are excluded.

Important subdirectories:

- `python/astparser/`: Python AST branch discovery and instrumentation.
- `python/pyextend/`: rank-taint dry-run runtime and branch signatures.
- `python/flexsim/anchor_reuse/`: selected CFG, active-rank, logic-slicing, and
  four-slot resilient-anchor-state modules; see
  `SOURCE_ANALYSIS_PROVENANCE.md`.
- `python/flexsim/maya_lite/`: independent Maya-style evaluator.
- `cpp/`: fake-CUDA and C++ event/runtime source.
- `fake-cuda/`: launcher wrapper used by experiment scripts. PRoot and
  environment-specific launch policy are supplied by the consuming project.
- `gpu_estimator/`: runtime estimator bundle used by the evaluator.
- `tests/`: portable source-level tests and workload fixtures.

From the core repository root, run `make test` for the CPU-only source-analysis
probe and FlexEva RAS tests. `SOURCE_ANALYSIS_FILES.sha256` verifies the
imported source subset before the test runs.

The AE's native-GPU capture path builds the checked-in real CUDA/NCCL/cuBLAS
interposers with `scripts/build_real_wrapper_stack.sh`; generated objects and
Python extensions stay below this checkout's ignored `build/` directories.
