"""
Figure 13 workload-specific trace-shape checks.

These checks are intentionally narrow. They validate the current synthetic
Megatron-style Figure 13 workload and should not be treated as generic Maya-lite
trace requirements.
"""

from __future__ import annotations

from pathlib import Path

from .io import load_trace_directory
from .schema import TraceBundle


FIG13_ALLOWED_STEP_NCCL_COMM_APIS = frozenset({"ncclAllReduce", "ncclSend", "ncclRecv"})
FIG13_ALLOWED_STEP_NCCL_CONTROL_APIS = frozenset(
    {
        "ncclCommGetAsyncError",
        "ncclCommInitRankConfig",
        "ncclGetUniqueId",
        "ncclGetVersion",
        "ncclGroupEnd",
        "ncclGroupStart",
    }
)
FIG13_ALLOWED_STEP_NCCL_APIS = frozenset(
    FIG13_ALLOWED_STEP_NCCL_COMM_APIS | FIG13_ALLOWED_STEP_NCCL_CONTROL_APIS
)


def inspect_fig13_step_contract(
    trace_dir: Path,
    *,
    max_events_per_rank: int | None = None,
) -> dict[str, object]:
    bundle = load_trace_directory(
        trace_dir,
        max_events_per_rank=max_events_per_rank,
        trace_window="auto",
    )
    return inspect_fig13_step_contract_bundle(bundle)


def inspect_fig13_step_contract_bundle(bundle: TraceBundle) -> dict[str, object]:
    observed_nccl_apis = sorted(
        {
            event.api
            for rank_trace in bundle.rank_traces
            for event in rank_trace.events
            if event.api.startswith("nccl")
        }
    )
    unexpected_nccl_apis = sorted(
        api for api in observed_nccl_apis if api not in FIG13_ALLOWED_STEP_NCCL_APIS
    )
    return {
        "trace_window": bundle.trace_window,
        "paper_valid_step_window_rank_count": sum(
            1
            for fidelity_window in bundle.fidelity_windows.values()
            if fidelity_window.is_paper_valid_step_window
        ),
        "step_window_sources": sorted(
            {fidelity_window.source for fidelity_window in bundle.fidelity_windows.values()}
        ),
        "observed_nccl_apis": observed_nccl_apis,
        "unexpected_nccl_apis": unexpected_nccl_apis,
        "total_events": bundle.total_events,
    }


def validate_fig13_step_contract(
    trace_dir: Path,
    *,
    max_events_per_rank: int | None = None,
) -> dict[str, object]:
    summary = inspect_fig13_step_contract(trace_dir, max_events_per_rank=max_events_per_rank)
    return _validate_fig13_step_contract_summary(summary)


def validate_fig13_step_contract_bundle(bundle: TraceBundle) -> dict[str, object]:
    summary = inspect_fig13_step_contract_bundle(bundle)
    return _validate_fig13_step_contract_summary(summary)


def _validate_fig13_step_contract_summary(summary: dict[str, object]) -> dict[str, object]:
    if summary["trace_window"] != "step":
        raise ValueError(
            "Figure 13 contract validation requires explicit step-window traces; "
            f"got trace_window={summary['trace_window']!r}"
        )
    if int(summary.get("paper_valid_step_window_rank_count", 0)) <= 0:
        raise ValueError(
            "Figure 13 contract validation requires paper-valid step-window metadata"
        )
    unexpected = summary["unexpected_nccl_apis"]
    if unexpected:
        raise ValueError(
            "Figure 13 step trace contains unexpected NCCL APIs: "
            + ", ".join(str(api) for api in unexpected)
        )
    return summary
