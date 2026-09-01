"""Maya-style host API delay profiles.

Maya models host-side overheads from wall-clock measurements collected during
emulation/profiling.  This module builds a small per-API profile from traced
``host_duration_us`` observations, respecting the trace window selected by the
trace loader/manifest, and exposes dispatch-only host delays for fake traces
whose stubs return too quickly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median

from .schema import CollatedEvent, TraceBundle, TraceEvent, TraceSource


_HOST_PROFILE_APIS = {
    "cublasCreate_v2",
    "cublasSetStream_v2",
    "cudaGetDevice",
    "cudaGetDeviceCount",
    "cudaGetLastError",
    "cudaSetDevice",
    "cudaStreamCreate",
    "cudaEventCreateWithFlags",
    "cudaEventDestroy",
    "cudaEventQuery",
    "cudaEventRecord",
    "cudaStreamWaitEvent",
    "cudaLaunchKernel",
    "cudaMalloc",
    "cudaMemcpyAsync",
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "ncclAllReduce",
    "ncclRecv",
    "ncclSend",
    "ncclGroupStart",
    "ncclGroupEnd",
    "ncclCommGetAsyncError",
    "ncclCommInitRank",
    "ncclCommInitRankConfig",
    "ncclCommSplit",
    "ncclCommDestroy",
    "ncclGetUniqueId",
    "ncclGetVersion",
}


def _float_field(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _record_host_duration_us(event: TraceEvent | CollatedEvent) -> float | None:
    observed = _float_field(event.extras.get("host_duration_us"))
    if observed is not None:
        return max(observed, 0.0)
    end_ts = _float_field(event.extras.get("end_ts"))
    if end_ts is None:
        return None
    return max(float(end_ts) - float(event.ts), 0.0)


@dataclass(frozen=True)
class HostDelayProfileEntry:
    api: str
    count: int
    mean_us: float
    p50_us: float
    p95_us: float

    def select(self, percentile: str) -> float:
        if percentile == "mean":
            return self.mean_us
        if percentile == "p95":
            return self.p95_us
        return self.p50_us


@dataclass(frozen=True)
class HostDelayProfile:
    entries: dict[str, HostDelayProfileEntry] = field(default_factory=dict)
    percentile: str = "mean"

    @classmethod
    def fit_from_bundle(
        cls,
        bundle: TraceBundle,
        *,
        percentile: str = "mean",
        min_count: int = 1,
    ) -> "HostDelayProfile":
        samples: dict[str, list[float]] = {}
        for rank_trace in bundle.rank_traces:
            active_window = None
            if bundle.trace_window == "step" and rank_trace.rank in bundle.step_windows:
                active_window = bundle.step_windows[rank_trace.rank]
            for event in rank_trace.events:
                if active_window is not None:
                    start_ts, end_ts = active_window
                    if int(event.ts) < int(start_ts) or int(event.ts) > int(end_ts):
                        continue
                if event.api not in _HOST_PROFILE_APIS:
                    continue
                observed = _record_host_duration_us(event)
                if observed is None or observed <= 0.0:
                    continue
                samples.setdefault(event.api, []).append(float(observed))
        entries: dict[str, HostDelayProfileEntry] = {}
        for api, values in samples.items():
            if len(values) < min_count:
                continue
            ordered = sorted(values)
            p95_index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
            entries[api] = HostDelayProfileEntry(
                api=api,
                count=len(ordered),
                mean_us=float(mean(ordered)),
                p50_us=float(median(ordered)),
                p95_us=float(ordered[p95_index]),
            )
        return cls(entries=entries, percentile=percentile)

    def dispatch_duration_us(self, event: CollatedEvent) -> float | None:
        # Only fill in missing/under-modeled fake host overhead.  Real traces and
        # explicit observations remain authoritative.
        if event.source is not TraceSource.FAKE:
            return None
        entry = self.entries.get(event.api)
        if entry is None:
            return None
        profiled = max(float(entry.select(self.percentile)), 0.0)
        observed = _record_host_duration_us(event) or 0.0
        if profiled <= observed:
            return None
        return profiled


@dataclass(frozen=True)
class HostGapProfileEntry:
    key: tuple[str, str]
    count: int
    mean_us: float
    p50_us: float
    p95_us: float

    def select(self, percentile: str) -> float:
        if percentile == "mean":
            return self.mean_us
        if percentile == "p95":
            return self.p95_us
        return self.p50_us


@dataclass(frozen=True)
class HostGapProfile:
    """Transition-level Maya hostDelay profile fitted from collated real traces."""

    entries: dict[tuple[str, str], HostGapProfileEntry] = field(default_factory=dict)
    api_entries: dict[str, HostGapProfileEntry] = field(default_factory=dict)
    rank_entries: dict[tuple[int, str, str], HostGapProfileEntry] = field(default_factory=dict)
    rank_api_entries: dict[tuple[int, str], HostGapProfileEntry] = field(default_factory=dict)
    percentile: str = "mean"

    @classmethod
    def fit_from_collated_events(
        cls,
        events: dict[int, tuple[CollatedEvent, ...]],
        *,
        percentile: str = "mean",
        min_count: int = 1,
    ) -> "HostGapProfile":
        by_id = {event.id: event for rank_events in events.values() for event in rank_events}
        pair_samples: dict[tuple[str, str], list[float]] = {}
        api_samples: dict[str, list[float]] = {}
        rank_pair_samples: dict[tuple[int, str, str], list[float]] = {}
        rank_api_samples: dict[tuple[int, str], list[float]] = {}
        for rank_events in events.values():
            sorted_events = sorted(rank_events, key=lambda event: (event.ts, event.ordinal))
            next_materialized_api_by_index: list[str | None] = [None] * len(sorted_events)
            next_materialized_api: str | None = None
            for reverse_index in range(len(sorted_events) - 1, -1, -1):
                next_materialized_api_by_index[reverse_index] = next_materialized_api
                candidate = sorted_events[reverse_index]
                if candidate.api != "__hostDelay__" and candidate.op_type != "host_delay":
                    next_materialized_api = candidate.api
            for index, event in enumerate(sorted_events):
                if event.api != "__hostDelay__" and event.op_type != "host_delay":
                    continue
                observed = _float_field(event.extras.get("observed_gap_us"))
                if observed is None or observed <= 0.0:
                    continue
                prev_event = by_id.get(str(event.prev_event_id or ""))
                next_api = next_materialized_api_by_index[index]
                if prev_event is None or next_api is None:
                    continue
                key = (prev_event.api, next_api)
                pair_samples.setdefault(key, []).append(float(observed))
                api_samples.setdefault(next_api, []).append(float(observed))
                rank_pair_samples.setdefault((event.rank, prev_event.api, next_api), []).append(float(observed))
                rank_api_samples.setdefault((event.rank, next_api), []).append(float(observed))

        def make_entry(key: tuple[str, str], values: list[float]) -> HostGapProfileEntry:
            ordered = sorted(values)
            p95_index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
            return HostGapProfileEntry(
                key=key,
                count=len(ordered),
                mean_us=float(mean(ordered)),
                p50_us=float(median(ordered)),
                p95_us=float(ordered[p95_index]),
            )

        return cls(
            entries={
                key: make_entry(key, values)
                for key, values in pair_samples.items()
                if len(values) >= min_count
            },
            api_entries={
                api: make_entry(("*", api), values)
                for api, values in api_samples.items()
                if len(values) >= min_count
            },
            rank_entries={
                key: make_entry((key[1], key[2]), values)
                for key, values in rank_pair_samples.items()
                if len(values) >= min_count
            },
            rank_api_entries={
                key: make_entry(("*", key[1]), values)
                for key, values in rank_api_samples.items()
                if len(values) >= min_count
            },
            percentile=percentile,
        )

    def profiled_gap_us(
        self,
        previous_api: str | None,
        next_api: str | None,
        *,
        rank: int | None = None,
    ) -> float | None:
        if previous_api is None or next_api is None:
            return None
        entry = None
        if rank is not None:
            entry = self.rank_entries.get((rank, previous_api, next_api)) or self.rank_api_entries.get((rank, next_api))
        entry = entry or self.entries.get((previous_api, next_api)) or self.api_entries.get(next_api)
        if entry is None:
            return None
        return max(float(entry.select(self.percentile)), 0.0)
