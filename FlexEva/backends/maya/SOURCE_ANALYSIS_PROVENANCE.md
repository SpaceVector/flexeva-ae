# Source-analysis implementation

This directory contains the source-analysis code used by the artifact. It was
imported from the authors' 2026-08-22 workspace snapshot at commit
`4fb3488b93f433de770f0cf64879011a3b6e30c7`. `python/astparser/` was tracked in
that snapshot; `python/flexsim/anchor_reuse/` came from the accompanying
working tree. `SOURCE_ANALYSIS_FILES.sha256` records the imported file set.

## Components

- `python/astparser/`: Python branch discovery and `mark_cond`
  instrumentation using the standard-library `ast` module.
- `python/pyextend/`: rank-varying values, dynamic taint propagation, branch
  signatures, semantic dependencies, and dry-run execution.
- `python/flexsim/anchor_reuse/`: control regions, active-rank partitions,
  source slicing, trace lineage, and resilient-anchor-state assembly.
- `cpp/lowlevel_interface/tools/scan_fakes.cpp`: libclang-based extraction of
  fake-wrapper signatures.

## Scope

The Python frontend uses `ast`; the C++ control-flow path uses source lines and
regular expressions rather than Clang CFG or def-use analysis. Dynamic
rank-taint propagation supplies the dependency evidence used by the runnable
path; `pyextend.runtime.rankcontext.add_dependency()` is not populated by a
call site in this subset.

The portable probe builds code, semantic, runtime-value, and trace slots from
synthetic source and trace input. Runtime-value distributions come from dry-run
branch signatures, and trace lineage uses normalized windows plus operator
evidence. These checks cover the object model and data flow, not
workload-level soundness or candidate-ranking equivalence.
