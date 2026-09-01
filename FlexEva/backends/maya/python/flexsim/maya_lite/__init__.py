"""
Maya-lite baseline package.

This package implements a claim-faithful, low-level baseline inspired by the
Maya paper architecture:

1. load raw per-rank backend traces
2. collate low-level traces
3. annotate low-level operations with durations
4. replay the annotated trace to obtain end-to-end metrics

The package is intentionally isolated from SPSD semantic recovery so it can
serve as a fair black-box baseline.
"""

from .annotate import (
    AnnotationTimingRecorder,
    annotate_collated_trace,
    collective_group_duration_summary,
    estimate_low_level_event_us,
)
from .capture_fake import select_worker_trace_files
from .collate import collate_trace_bundle
from .evaluate import evaluate_candidate_set, evaluate_trace_directory
from .fig13_contract import (
    FIG13_ALLOWED_STEP_NCCL_APIS,
    inspect_fig13_step_contract_bundle,
    inspect_fig13_step_contract,
    validate_fig13_step_contract_bundle,
    validate_fig13_step_contract,
)
from .fig13_validation import compare_fig13_step_trace_dirs
from .io import (
    dedup_identical_rank_traces,
    dedup_pattern_rank_traces,
    inspect_trace_directory,
    iter_rank_trace_events,
    load_trace_directory,
    materialize_profiled_rank_traces,
)
from .planner import (
    plan_identity_rank_groups,
    plan_megatron_pipeline_stage_groups,
    plan_profiled_rank_groups,
    profiled_ranks_for_groups,
)
from .replay import (
    export_replay_edge_diagnostics,
    pair_seq_collective_ablation_predicate,
    replay_annotated_trace,
)
from .schema import (
    AnnotatedEvent,
    AnnotatedTrace,
    CandidateEvaluation,
    CollectiveGroup,
    CollatedEvent,
    CollatedTrace,
    FidelityWindow,
    RankReplayMetrics,
    RankTrace,
    ReplayResult,
    SimulatedEvent,
    TraceBundle,
    TraceDirectorySummary,
    TraceEvent,
    TraceSource,
)
from .stage_timing import StageTimingBreakdown, benchmark_trace_bundle, benchmark_trace_directory

__all__ = [
    "annotate_collated_trace",
    "AnnotationTimingRecorder",
    "AnnotatedEvent",
    "AnnotatedTrace",
    "benchmark_trace_directory",
    "benchmark_trace_bundle",
    "CandidateEvaluation",
    "collective_group_duration_summary",
    "CollectiveGroup",
    "CollatedEvent",
    "CollatedTrace",
    "evaluate_candidate_set",
    "evaluate_trace_directory",
    "FIG13_ALLOWED_STEP_NCCL_APIS",
    "FidelityWindow",
    "compare_fig13_step_trace_dirs",
    "RankReplayMetrics",
    "estimate_low_level_event_us",
    "inspect_fig13_step_contract",
    "inspect_fig13_step_contract_bundle",
    "RankTrace",
    "ReplayResult",
    "SimulatedEvent",
    "select_worker_trace_files",
    "TraceBundle",
    "TraceDirectorySummary",
    "TraceEvent",
    "TraceSource",
    "validate_fig13_step_contract",
    "validate_fig13_step_contract_bundle",
    "replay_annotated_trace",
    "export_replay_edge_diagnostics",
    "pair_seq_collective_ablation_predicate",
    "collate_trace_bundle",
    "dedup_identical_rank_traces",
    "dedup_pattern_rank_traces",
    "inspect_trace_directory",
    "iter_rank_trace_events",
    "load_trace_directory",
    "materialize_profiled_rank_traces",
    "plan_identity_rank_groups",
    "plan_megatron_pipeline_stage_groups",
    "plan_profiled_rank_groups",
    "profiled_ranks_for_groups",
    "StageTimingBreakdown",
]
