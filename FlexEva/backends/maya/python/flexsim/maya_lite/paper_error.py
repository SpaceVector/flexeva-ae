"""Paper-aligned prediction-error helpers."""

from __future__ import annotations

from dataclasses import dataclass

from flexsim.maya_lite.schema import TraceBundle

MAX_RANK_DURATION_SKEW_RATIO = 0.05
MAX_GLOBAL_MAKESPAN_RATIO_SINGLE_CLOCK = 1.05


@dataclass(frozen=True, slots=True)
class PaperActualRuntime:
    actual_per_iteration_runtime_us: float
    basis: str
    rank_count: int
    critical_rank_runtime_us: float
    global_makespan_us: float
    rank0_runtime_us: float | None
    basis_reason: str
    global_makespan_warning: str | None
    validation: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "actual_per_iteration_runtime_us": self.actual_per_iteration_runtime_us,
            "basis": self.basis,
            "rank_count": self.rank_count,
            "critical_rank_runtime_us": self.critical_rank_runtime_us,
            "global_makespan_us": self.global_makespan_us,
            "rank0_runtime_us": self.rank0_runtime_us,
            "basis_reason": self.basis_reason,
            "global_makespan_warning": self.global_makespan_warning,
            "validation": self.validation,
        }


def _clock_domain_count(real_bundle: TraceBundle) -> int:
    machines = {str(v) for v in real_bundle.rank_host_machines.values() if str(v)}
    return len(machines) if machines else 1


def actual_runtime_from_real_trace(
    real_bundle: TraceBundle,
    *,
    basis: str = "critical_rank_completed_iteration_runtime",
) -> PaperActualRuntime:
    windows = {
        int(rank): window
        for rank, window in real_bundle.fidelity_windows.items()
        if window.is_paper_valid_step_window
    }
    if not windows:
        raise ValueError("real trace has no paper-valid step windows for actual runtime")
    if len(windows) != int(real_bundle.world_size):
        raise ValueError(
            f"paper actual runtime requires one valid step window per rank: "
            f"valid={len(windows)}, world_size={real_bundle.world_size}"
        )
    bad_step_counts = {
        rank: window.extras.get("step_count")
        for rank, window in windows.items()
        if window.extras.get("step_count") not in (None, 1)
    }
    if bad_step_counts:
        raise ValueError(f"paper actual runtime requires one measured step per rank: {bad_step_counts}")

    durations = {rank: float(window.end_ts - window.start_ts) for rank, window in windows.items()}
    critical = max(durations.values())
    minimum = min(durations.values())
    duration_skew_ratio = (critical - minimum) / critical if critical > 0 else 0.0
    if duration_skew_ratio > MAX_RANK_DURATION_SKEW_RATIO:
        raise ValueError(
            "per-rank completed-iteration durations are not synchronized enough "
            f"for critical-rank paper actual basis: skew_ratio={duration_skew_ratio:.6f}, "
            f"limit={MAX_RANK_DURATION_SKEW_RATIO:.6f}"
        )

    global_makespan = float(max(w.end_ts for w in windows.values()) - min(w.start_ts for w in windows.values()))
    global_ratio = global_makespan / critical if critical > 0 else 0.0
    clock_domains = _clock_domain_count(real_bundle)
    if clock_domains <= 1 and global_ratio > MAX_GLOBAL_MAKESPAN_RATIO_SINGLE_CLOCK:
        raise ValueError(
            "single-clock trace has global makespan too far from critical-rank duration: "
            f"ratio={global_ratio:.6f}, limit={MAX_GLOBAL_MAKESPAN_RATIO_SINGLE_CLOCK:.6f}"
        )

    rank0 = durations.get(0)
    validation = {
        "valid_step_window_count": len(windows),
        "world_size": int(real_bundle.world_size),
        "rank_duration_min_us": minimum,
        "rank_duration_max_us": critical,
        "rank_duration_skew_ratio": duration_skew_ratio,
        "rank_duration_skew_limit": MAX_RANK_DURATION_SKEW_RATIO,
        "global_makespan_ratio_to_critical": global_ratio,
        "clock_domain_count": clock_domains,
        "global_makespan_used_for_default": False,
    }
    basis_reason = (
        "validated critical-rank duration across per-rank completed-iteration step windows; "
        "all ranks have one paper-valid step and bounded duration skew"
    )
    global_makespan_warning = None
    if clock_domains > 1:
        global_makespan_warning = (
            "global_step_makespan spans multiple host clock domains and may include clock skew; "
            "it is reported for diagnostics, not used as the default paper actual runtime"
        )

    if basis == "critical_rank_completed_iteration_runtime":
        actual = critical
    elif basis == "global_step_makespan":
        actual = global_makespan
    elif basis == "rank0_completed_iteration_runtime":
        if rank0 is None:
            raise ValueError("rank0 step window missing from real trace")
        actual = rank0
    else:
        raise ValueError(f"unsupported paper actual runtime basis: {basis}")
    return PaperActualRuntime(
        actual_per_iteration_runtime_us=actual,
        basis=basis,
        rank_count=len(windows),
        critical_rank_runtime_us=critical,
        global_makespan_us=global_makespan,
        rank0_runtime_us=rank0,
        basis_reason=basis_reason,
        global_makespan_warning=global_makespan_warning,
        validation=validation,
    )


def paper_absolute_error_pct(predicted_us: float, actual_us: float) -> float:
    if actual_us <= 0:
        raise ValueError(f"actual runtime must be positive, got {actual_us}")
    return 100.0 * abs(float(predicted_us) - float(actual_us)) / float(actual_us)
