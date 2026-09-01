"""
Low-level trace loading utilities for Maya-lite.

These helpers only understand raw per-rank JSONL traces and intentionally avoid
any semantic augmentation from the SPSD pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Mapping

from .communicators import (
    build_emulated_communicator_id,
    recover_communicator_topology_from_events,
)
from .filters import (
    is_ignorable_dedup_api,
    is_ignorable_setup_api,
    is_teardown_api,
)
from .markers import (
    TRACE_MARKER_API,
    load_step_markers,
    resolve_step_window_from_markers,
    resolve_step_window_from_trace_markers,
)
from .schema import (
    FidelityWindow,
    RankTrace,
    TraceBundle,
    TraceDirectorySummary,
    TraceEvent,
    TraceSource,
)

_RANK_TRACE_RE = re.compile(r"rank_(\d+)\.jsonl$")
_RANK_TRACE_OR_RAW_RE = re.compile(r"rank_(\d+)(?:\.raw)?\.jsonl$")
_CAPTURE_MANIFEST = "capture_manifest.json"
_PAPER_VALID_FIDELITY_WINDOW_SOURCES = frozenset(
    {"manifest", "trace_markers", "workload_heuristic"}
)
# Historical manifests may still carry older source spellings; normalize them
# on load without re-exposing those names downstream.
_LEGACY_FIDELITY_WINDOW_SOURCE_ALIASES = {
    "markers": "trace_markers",
    "marker_sidecar": "trace_markers",
}
_LEGACY_DIAGNOSTIC_SOURCE_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_helper_tail_extended", "helper_tail"),
    ("_tail_extended", "tail_extended"),
)
_HOST_MACHINE_ID_FALLBACK_PREFIX = "legacy_pid"
_TRACE_BUNDLE_CACHE_VERSION = "trace_bundle_cache_v7"
_TRACE_BUNDLE_CACHE_DIRNAME = ".maya_lite_cache"
_TRACE_BUNDLE_CACHE_ENV = "FLEXSIM_MAYA_LITE_TRACE_BUNDLE_CACHE"


def _coerce_event_end_ts(raw_end_ts: object, fallback_ts: int) -> int:
    if raw_end_ts in (None, ""):
        return int(fallback_ts)
    try:
        resolved = int(float(raw_end_ts))
    except (TypeError, ValueError):
        return int(fallback_ts)
    return max(resolved, int(fallback_ts))


def _effective_event_end_ts(event: TraceEvent) -> int:
    return _coerce_event_end_ts(event.extras.get("end_ts"), event.ts)


def _normalize_host_machine_id(value: object | None) -> str | None:
    if value in (None, ""):
        return None
    resolved = str(value).strip()
    return resolved or None


def _fallback_host_machine_id_for_event(event: TraceEvent) -> str:
    return f"{_HOST_MACHINE_ID_FALLBACK_PREFIX}:{int(event.pid)}"


def _normalize_host_dispatch_queue_id(value: object | None) -> str | None:
    if value in (None, ""):
        return None
    resolved = str(value).strip()
    return resolved or None


def _default_host_dispatch_queue_id(*, rank: int, host_machine_id: str) -> str:
    # Maya's dispatch queue represents the host execution context that drives
    # device work.  In torchrun-style paper workloads a physical host runs
    # multiple rank worker processes, so physical host topology and dispatch
    # queue identity must stay distinct.
    return f"{host_machine_id}:rank:{int(rank)}"


def _canonical_host_dispatch_queue_id(
    value: object | None,
    *,
    rank: int,
    host_machine_id: str,
    dispatch_scope: str | None,
) -> str:
    resolved = _normalize_host_dispatch_queue_id(value)
    if resolved is not None:
        return resolved
    return _default_host_dispatch_queue_id(
        rank=rank,
        host_machine_id=host_machine_id,
    )


def _fallback_host_dispatch_queue_id_for_event(
    event: TraceEvent,
    *,
    host_machine_id: str,
) -> str:
    return _default_host_dispatch_queue_id(
        rank=event.rank,
        host_machine_id=host_machine_id,
    )


def _manifest_rank_host_machines(manifest: dict | None) -> dict[int, str]:
    if manifest is None:
        return {}

    resolved: dict[int, str] = {}
    raw_rank_host_machines = manifest.get("rank_host_machines", {})
    if isinstance(raw_rank_host_machines, Mapping):
        for raw_rank, raw_host_machine_id in raw_rank_host_machines.items():
            try:
                rank = int(raw_rank)
            except (TypeError, ValueError):
                continue
            host_machine_id = _normalize_host_machine_id(raw_host_machine_id)
            if host_machine_id is None:
                continue
            resolved[rank] = host_machine_id
    if resolved:
        return resolved

    raw_profiled_ranks = manifest.get("profiled_ranks", ())
    if not isinstance(raw_profiled_ranks, list):
        raw_profiled_ranks = ()
    profiled_ranks: list[int] = []
    for raw_rank in raw_profiled_ranks:
        try:
            profiled_ranks.append(int(raw_rank))
        except (TypeError, ValueError):
            continue
    if not profiled_ranks:
        return resolved

    local_device_span = manifest.get("local_device_span")
    try:
        resolved_local_device_span = max(int(local_device_span), 1)
    except (TypeError, ValueError):
        launched_workers = manifest.get("launched_workers", ())
        observed_local_ranks: list[int] = []
        if isinstance(launched_workers, list):
            for worker in launched_workers:
                if not isinstance(worker, Mapping):
                    continue
                try:
                    observed_local_ranks.append(int(worker.get("local_rank")))
                except (TypeError, ValueError):
                    continue
        resolved_local_device_span = (max(observed_local_ranks) + 1) if observed_local_ranks else 1

    for rank in profiled_ranks:
        resolved[rank] = f"logical_host_{rank // resolved_local_device_span}"
    return resolved


def _manifest_rank_host_dispatch_queues(
    manifest: dict | None,
    *,
    rank_host_machines: Mapping[int, str],
    dispatch_scope: str | None,
) -> dict[int, str]:
    resolved: dict[int, str] = {}
    if manifest is not None:
        raw_rank_host_dispatch_queues = manifest.get("rank_host_dispatch_queues", {})
        if isinstance(raw_rank_host_dispatch_queues, Mapping):
            for raw_rank, raw_dispatch_queue_id in raw_rank_host_dispatch_queues.items():
                try:
                    rank = int(raw_rank)
                except (TypeError, ValueError):
                    continue
                host_machine_id = rank_host_machines.get(rank)
                if host_machine_id is None:
                    dispatch_queue_id = _normalize_host_dispatch_queue_id(raw_dispatch_queue_id)
                    if dispatch_queue_id is None:
                        continue
                    resolved[rank] = dispatch_queue_id
                    continue
                resolved[rank] = _canonical_host_dispatch_queue_id(
                    raw_dispatch_queue_id,
                    rank=rank,
                    host_machine_id=host_machine_id,
                    dispatch_scope=dispatch_scope,
                )
    for rank, host_machine_id in sorted(rank_host_machines.items()):
        resolved.setdefault(
            int(rank),
            _default_host_dispatch_queue_id(
                rank=int(rank),
                host_machine_id=str(host_machine_id),
            ),
        )
    return resolved


def _normalize_dispatch_scope(value: object | None) -> str | None:
    if value in (None, ""):
        return None
    resolved = str(value).strip().lower()
    if resolved in {"thread", "process", "host_machine"}:
        return resolved
    return None


def _manifest_host_timing_dispatch_scope_resolved(
    manifest: dict | None,
) -> str | None:
    if manifest is None:
        return None
    resolved = _normalize_dispatch_scope(manifest.get("host_timing_dispatch_scope_resolved"))
    if resolved is not None:
        return resolved
    resolved = _normalize_dispatch_scope(manifest.get("host_timing_dispatch_scope"))
    if resolved is not None:
        return resolved
    line_family = str(manifest.get("host_timing_line_family") or "").strip().lower()
    paper_alignment_line = str(manifest.get("host_timing_paper_alignment_line") or "").strip().lower()
    if line_family == "disabled" or paper_alignment_line == "disabled":
        return "host_machine"
    return None


def _event_with_host_identity(
    event: TraceEvent,
    *,
    host_machine_id: str,
    host_dispatch_queue_id: str,
) -> TraceEvent:
    extras = dict(event.extras)
    extras["host_machine_id"] = host_machine_id
    extras["host_dispatch_queue_id"] = host_dispatch_queue_id
    return TraceEvent(
        rank=event.rank,
        ordinal=event.ordinal,
        source=event.source,
        ts=event.ts,
        pid=event.pid,
        tid=event.tid,
        module=event.module,
        api=event.api,
        op_type=event.op_type,
        extras=extras,
    )


def _normalize_fidelity_window_source(
    source: str | None,
) -> tuple[str, dict[str, object], bool]:
    normalized = str(source or "").strip().lower()
    extras: dict[str, object] = {}
    force_non_paper_valid = False
    for suffix, diagnostic_extension in _LEGACY_DIAGNOSTIC_SOURCE_SUFFIXES:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            extras["diagnostic_extension"] = diagnostic_extension
            force_non_paper_valid = True
            break
    normalized = _LEGACY_FIDELITY_WINDOW_SOURCE_ALIASES.get(normalized, normalized)
    return normalized, extras, force_non_paper_valid


def _canonicalize_fidelity_window_source(source: str | None) -> str:
    normalized, _, _ = _normalize_fidelity_window_source(source)
    return normalized


def is_paper_valid_fidelity_window_source(source: str | None) -> bool:
    return _canonicalize_fidelity_window_source(source) in _PAPER_VALID_FIDELITY_WINDOW_SOURCES


def fidelity_window_from_payload(
    payload: Mapping[str, Any] | None,
    *,
    default_source: str | None = None,
) -> FidelityWindow | None:
    if not isinstance(payload, Mapping):
        return None
    start_ts = payload.get("start_ts")
    end_ts = payload.get("end_ts")
    if start_ts in (None, "") or end_ts in (None, ""):
        return None
    try:
        resolved_start_ts = int(start_ts)
        resolved_end_ts = int(end_ts)
    except (TypeError, ValueError):
        return None
    if resolved_end_ts < resolved_start_ts:
        return None

    payload_extras = {
        str(key): value
        for key, value in payload.items()
        if key not in {"start_ts", "end_ts", "source", "is_paper_valid_step_window"}
    }
    resolved_source, source_extras, force_non_paper_valid = _normalize_fidelity_window_source(
        str(payload.get("source") or default_source or "manifest")
    )
    source_allows_paper_valid = (
        is_paper_valid_fidelity_window_source(resolved_source)
        and not force_non_paper_valid
    )
    is_paper_valid = payload.get("is_paper_valid_step_window")
    if is_paper_valid in (None, ""):
        resolved_is_paper_valid = source_allows_paper_valid
    else:
        resolved_is_paper_valid = bool(is_paper_valid) and source_allows_paper_valid

    extras = {**source_extras, **payload_extras}
    return FidelityWindow(
        start_ts=resolved_start_ts,
        end_ts=resolved_end_ts,
        source=resolved_source,
        is_paper_valid_step_window=resolved_is_paper_valid,
        extras=extras,
    )


def infer_trace_source(trace_dir: str | Path) -> TraceSource:
    path = Path(trace_dir)
    lowered = {part.lower() for part in path.parts}
    if "real" in lowered:
        return TraceSource.REAL
    if "remote_traces" in lowered:
        return TraceSource.REAL
    if "fake" in lowered:
        return TraceSource.FAKE
    return TraceSource.UNKNOWN


def _infer_trace_source_from_manifest(manifest: dict[str, object] | None) -> TraceSource:
    if not isinstance(manifest, dict):
        return TraceSource.UNKNOWN
    mode = str(manifest.get("mode") or "").strip().lower()
    if mode.startswith("emulated"):
        return TraceSource.FAKE
    if mode.startswith("real"):
        return TraceSource.REAL

    host_timing_line = str(manifest.get("host_timing_paper_alignment_line") or "").strip().lower()
    host_timing_family = str(manifest.get("host_timing_line_family") or "").strip().lower()
    host_timing_mode = str(manifest.get("host_timing_mode") or "").strip().lower()
    if host_timing_line == "disabled" or host_timing_family == "disabled":
        return TraceSource.REAL
    if host_timing_mode in {"trace", "measure", "sleep"}:
        return TraceSource.FAKE
    return TraceSource.UNKNOWN


def _rank_from_path(path: Path) -> int:
    match = _RANK_TRACE_OR_RAW_RE.match(path.name)
    if not match:
        raise ValueError(
            f"expected rank_*.jsonl or rank_*.raw.jsonl trace file, got {path}"
        )
    return int(match.group(1))


def list_rank_trace_files(trace_dir: str | Path) -> tuple[Path, ...]:
    path = Path(trace_dir)
    files = tuple(
        sorted(
            candidate
            for candidate in path.glob("rank_*.jsonl")
            if _RANK_TRACE_RE.match(candidate.name)
        )
    )
    if not files:
        raise FileNotFoundError(f"no rank_*.jsonl files found in {path}")
    return files


def load_capture_manifest(trace_dir: str | Path) -> dict | None:
    path = Path(trace_dir) / _CAPTURE_MANIFEST
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def estimate_rank_trace_window(
    trace_file: str | Path,
    *,
    source: TraceSource | None = None,
    strict_json: bool = False,
) -> dict[str, int | str] | None:
    min_ts: int | None = None
    max_ts: int | None = None

    for event in iter_rank_trace_events(trace_file, source=source, strict_json=strict_json):
        if min_ts is None or event.ts < min_ts:
            min_ts = event.ts
        event_end_ts = _effective_event_end_ts(event)
        if max_ts is None or event_end_ts > max_ts:
            max_ts = event_end_ts

    if min_ts is None or max_ts is None:
        return None
    return {
        "start_ts": int(min_ts),
        "end_ts": int(max_ts),
        "source": "boundary_fallback",
    }


def _is_active_trace_timing_event(event: TraceEvent) -> bool:
    if is_ignorable_setup_api(event.api) or is_teardown_api(event.api):
        return False
    return event.op_type in {
        "kernel_launch",
        "blas_compute",
        "nccl_collective",
        "mem_copy",
    }


def _estimate_rank_trace_active_window(
    trace_file: str | Path,
    *,
    source: TraceSource | None = None,
    strict_json: bool = False,
) -> dict[str, int | str] | None:
    min_ts: int | None = None
    max_ts: int | None = None

    for event in iter_rank_trace_events(trace_file, source=source, strict_json=strict_json):
        if not _is_active_trace_timing_event(event):
            continue
        if min_ts is None or event.ts < min_ts:
            min_ts = event.ts
        event_end_ts = _effective_event_end_ts(event)
        if max_ts is None or event_end_ts > max_ts:
            max_ts = event_end_ts

    if min_ts is None or max_ts is None:
        return None
    return {
        "start_ts": int(min_ts),
        "end_ts": int(max_ts),
        "source": "boundary_fallback",
    }


def estimate_rank_trace_active_seconds(
    trace_file: str | Path,
    *,
    source: TraceSource | None = None,
    strict_json: bool = False,
) -> float | None:
    window = _estimate_rank_trace_active_window(
        trace_file,
        source=source,
        strict_json=strict_json,
    )
    if window is None:
        return None
    return max(int(window["end_ts"]) - int(window["start_ts"]), 0) / 1_000_000.0


def _manifest_fidelity_windows(
    manifest: dict | None,
    *,
    prefer_explicit_fidelity_windows: bool = False,
) -> dict[int, FidelityWindow]:
    if manifest is None:
        return {}
    raw_step_windows = manifest.get("step_windows", {})
    raw_fidelity_windows = manifest.get("fidelity_windows")

    resolved_step_windows: dict[int, FidelityWindow] = {}
    if isinstance(raw_step_windows, dict):
        for rank, payload in raw_step_windows.items():
            fidelity_window = fidelity_window_from_payload(
                payload,
                default_source="manifest",
            )
            if fidelity_window is None:
                continue
            resolved_step_windows[int(rank)] = fidelity_window

    resolved_fidelity_windows: dict[int, FidelityWindow] = {}
    if isinstance(raw_fidelity_windows, dict):
        for rank, payload in raw_fidelity_windows.items():
            fidelity_window = fidelity_window_from_payload(payload)
            if fidelity_window is None:
                continue
            resolved_fidelity_windows[int(rank)] = fidelity_window

    if not resolved_fidelity_windows:
        return dict(resolved_step_windows)
    if prefer_explicit_fidelity_windows:
        return dict(resolved_fidelity_windows)

    resolved: dict[int, FidelityWindow] = dict(resolved_fidelity_windows)
    for rank, step_window in resolved_step_windows.items():
        diagnostic_window = resolved_fidelity_windows.get(rank)
        if diagnostic_window is None:
            resolved[rank] = step_window
            continue
        if not step_window.is_paper_valid_step_window:
            continue

        merged_extras = dict(step_window.extras)
        if (
            diagnostic_window.start_ts != step_window.start_ts
            or diagnostic_window.end_ts != step_window.end_ts
            or diagnostic_window.source != step_window.source
            or diagnostic_window.extras
        ):
            diagnostic_payload = diagnostic_window.to_dict()
            diagnostic_payload.pop("is_paper_valid_step_window", None)
            merged_extras["diagnostic_fidelity_window"] = diagnostic_payload
        resolved[rank] = FidelityWindow(
            start_ts=step_window.start_ts,
            end_ts=step_window.end_ts,
            source=step_window.source,
            is_paper_valid_step_window=True,
            extras=merged_extras,
        )
    return resolved


def _rank_marker_path(trace_file: Path) -> Path:
    rank = _rank_from_path(trace_file)
    return trace_file.with_name(f"rank_{rank}.markers.jsonl")


def _cache_file_fingerprint(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": path.name, "exists": False}
    stat = path.stat()
    return {
        "path": path.name,
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _trace_bundle_cache_path(
    trace_dir: Path,
    *,
    rank_files: tuple[Path, ...],
    manifest_path: Path,
    trace_window: str,
    max_events_per_rank: int | None,
    step_pre_padding_us: int,
    step_post_padding_us: int,
    strict_json: bool,
) -> Path:
    payload = {
        "version": _TRACE_BUNDLE_CACHE_VERSION,
        "trace_window": trace_window,
        "max_events_per_rank": max_events_per_rank,
        "step_pre_padding_us": int(step_pre_padding_us),
        "step_post_padding_us": int(step_post_padding_us),
        "strict_json": bool(strict_json),
        "manifest": _cache_file_fingerprint(manifest_path),
        "rank_files": [_cache_file_fingerprint(path) for path in rank_files],
        "marker_files": [
            _cache_file_fingerprint(_rank_marker_path(path))
            for path in rank_files
        ],
    }
    cache_key = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_dir = trace_dir / _TRACE_BUNDLE_CACHE_DIRNAME
    return cache_dir / f"trace_bundle_{cache_key}.pkl"


def _trace_bundle_cache_enabled() -> bool:
    raw = os.environ.get(_TRACE_BUNDLE_CACHE_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_trace_bundle_cache(cache_path: Path) -> TraceBundle | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError):
        return None
    if not isinstance(cached, TraceBundle):
        return None
    return cached


def _store_trace_bundle_cache(cache_path: Path, bundle: TraceBundle) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)


def resolve_fidelity_window(
    trace_file: Path,
    *,
    source: TraceSource,
    manifest_window: FidelityWindow | None,
    strict_json: bool,
    prefer_manifest_window: bool = False,
) -> FidelityWindow | None:
    if prefer_manifest_window and manifest_window is not None:
        return manifest_window

    if manifest_window is not None and manifest_window.is_paper_valid_step_window:
        return manifest_window

    trace_marker_window = resolve_step_window_from_trace_markers(trace_file)
    if trace_marker_window is not None:
        return fidelity_window_from_payload(
            {
                **trace_marker_window,
                "source": "trace_markers",
                "is_paper_valid_step_window": True,
            }
        )

    marker_records = load_step_markers(_rank_marker_path(trace_file))
    marker_window = resolve_step_window_from_markers(marker_records, source=source)
    if marker_window is not None:
        return fidelity_window_from_payload(
            {
                **marker_window,
                "source": "trace_markers",
                "is_paper_valid_step_window": True,
            }
        )

    estimated_window = _estimate_rank_trace_active_window(
        trace_file,
        source=source,
        strict_json=strict_json,
    )
    if estimated_window is None:
        estimated_window = estimate_rank_trace_window(
            trace_file,
            source=source,
            strict_json=strict_json,
        )
    if manifest_window is not None:
        return manifest_window
    if estimated_window is None:
        return None
    return fidelity_window_from_payload(
        {
            **estimated_window,
            "is_paper_valid_step_window": False,
        }
    )


def _manifest_communicators(manifest: dict | None) -> dict[str, tuple[int, ...]]:
    if manifest is None:
        return {}
    raw = manifest.get("communicators", {})
    if not isinstance(raw, dict):
        return {}
    resolved: dict[str, tuple[int, ...]] = {}
    for comm_id, payload in raw.items():
        if isinstance(payload, Mapping):
            members = payload.get("members")
        else:
            members = payload
        if not isinstance(members, list):
            continue
        resolved[str(comm_id)] = tuple(int(member) for member in members)
    return resolved


def _manifest_communicator_aliases(
    manifest: dict | None,
) -> dict[tuple[int, str], str]:
    if manifest is None:
        return {}
    raw = manifest.get("communicator_aliases", {})
    if not isinstance(raw, dict):
        return {}
    resolved: dict[tuple[int, str], str] = {}
    for raw_rank, payload in raw.items():
        if not isinstance(payload, Mapping):
            continue
        try:
            rank = int(raw_rank)
        except (TypeError, ValueError):
            continue
        for local_comm_id, canonical_comm_id in payload.items():
            local_comm_id_str = str(local_comm_id).strip()
            canonical_comm_id_str = str(canonical_comm_id).strip()
            if not local_comm_id_str or not canonical_comm_id_str:
                continue
            resolved[(rank, local_comm_id_str)] = canonical_comm_id_str
    return resolved


def _trace_communicator_recovery(
    trace_dir: Path,
    *,
    source: TraceSource,
    strict_json: bool = False,
) -> tuple[dict[str, tuple[int, ...]], dict[tuple[int, str], str]]:
    recovery = recover_communicator_topology_from_events(
        event
        for trace_file in list_rank_trace_files(trace_dir)
        for event in iter_rank_trace_events(trace_file, source=source, strict_json=strict_json)
    )
    return dict(recovery.memberships), dict(recovery.local_comm_aliases)


def _merge_communicator_memberships(
    *,
    manifest_memberships: dict[str, tuple[int, ...]],
    trace_memberships: dict[str, tuple[int, ...]],
) -> dict[str, tuple[int, ...]]:
    merged = dict(manifest_memberships)
    for comm_id, trace_members in trace_memberships.items():
        manifest_members = manifest_memberships.get(comm_id)
        if manifest_members is not None and manifest_members != trace_members:
            raise ValueError(
                "trace-derived communicator topology disagrees with manifest "
                f"for {comm_id}: {trace_members} != {manifest_members}"
            )
        merged[comm_id] = trace_members
    return merged


def _merge_communicator_aliases(
    *,
    manifest_aliases: dict[tuple[int, str], str],
    trace_aliases: dict[tuple[int, str], str],
) -> dict[tuple[int, str], str]:
    merged = dict(manifest_aliases)
    for key, trace_canonical_comm_id in trace_aliases.items():
        manifest_canonical_comm_id = merged.get(key)
        if (
            manifest_canonical_comm_id is not None
            and manifest_canonical_comm_id != trace_canonical_comm_id
        ):
            raise ValueError(
                "trace-derived communicator alias disagrees with manifest "
                f"for {key}: {trace_canonical_comm_id} != {manifest_canonical_comm_id}"
            )
        merged[key] = trace_canonical_comm_id
    return merged


def _rewrite_event_comm_id(
    event: TraceEvent,
    *,
    communicator_aliases: Mapping[tuple[int, str], str],
) -> TraceEvent:
    local_comm_id = str(event.extras.get("comm_id", "")).strip()
    if not local_comm_id:
        return event
    canonical_comm_id = communicator_aliases.get((event.rank, local_comm_id))
    if canonical_comm_id is None or canonical_comm_id == local_comm_id:
        return event
    extras = dict(event.extras)
    extras["comm_id"] = canonical_comm_id
    extras["local_comm_id"] = local_comm_id
    return TraceEvent(
        rank=event.rank,
        ordinal=event.ordinal,
        source=event.source,
        ts=event.ts,
        pid=event.pid,
        tid=event.tid,
        module=event.module,
        api=event.api,
        op_type=event.op_type,
        extras=extras,
    )


def _rewrite_rank_trace_comm_ids(
    rank_trace: RankTrace,
    *,
    communicator_aliases: Mapping[tuple[int, str], str],
) -> RankTrace:
    rewritten_events = tuple(
        _rewrite_event_comm_id(event, communicator_aliases=communicator_aliases)
        for event in rank_trace.events
    )
    if rewritten_events == rank_trace.events:
        return rank_trace
    return RankTrace(
        rank=rank_trace.rank,
        path=rank_trace.path,
        source=rank_trace.source,
        events=rewritten_events,
    )


def iter_rank_trace_events(
    trace_file: str | Path,
    *,
    source: TraceSource | None = None,
    strict_json: bool = False,
) -> Iterator[TraceEvent]:
    path = Path(trace_file)
    rank = _rank_from_path(path)
    resolved_source = source or infer_trace_source(path.parent)

    with path.open() as handle:
        for ordinal, line in enumerate(handle):
            payload = line.strip()
            if not payload:
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError as exc:
                if not strict_json:
                    warnings.warn(
                        f"skipping invalid JSON in {path}:{ordinal + 1}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                raise ValueError(f"invalid JSON in {path}:{ordinal + 1}") from exc
            yield TraceEvent.from_json_record(
                record,
                rank=rank,
                ordinal=ordinal,
                source=resolved_source,
            )


def _recommended_trace_io_workers(rank_file_count: int) -> int:
    if rank_file_count <= 1:
        return 1
    return min(rank_file_count, 8)


def _load_rank_trace(
    trace_file: Path,
    *,
    source: TraceSource,
    strict_json: bool,
    active_window: tuple[int, int] | None,
    step_pre_padding_us: int,
    step_post_padding_us: int,
    max_events_per_rank: int | None,
    initial_host_machine_id: str | None,
    initial_host_dispatch_queue_id: str | None,
    host_timing_dispatch_scope_resolved: str | None,
) -> tuple[RankTrace, str | None, str | None]:
    rank = _rank_from_path(trace_file)
    events: list[TraceEvent] = []
    budget_events = 0
    resolved_host_machine_id = initial_host_machine_id
    resolved_host_dispatch_queue_id = initial_host_dispatch_queue_id

    for event in iter_rank_trace_events(trace_file, source=source, strict_json=strict_json):
        if event.api == TRACE_MARKER_API:
            continue
        host_machine_id = resolved_host_machine_id
        if host_machine_id is None:
            host_machine_id = _normalize_host_machine_id(event.extras.get("host_machine_id"))
            if host_machine_id is None:
                host_machine_id = _fallback_host_machine_id_for_event(event)
            resolved_host_machine_id = host_machine_id
        host_dispatch_queue_id = resolved_host_dispatch_queue_id
        if host_dispatch_queue_id is None:
            host_dispatch_queue_id = _canonical_host_dispatch_queue_id(
                event.extras.get("host_dispatch_queue_id")
                if "host_dispatch_queue_id" in event.extras
                else None,
                rank=event.rank,
                host_machine_id=host_machine_id,
                dispatch_scope=host_timing_dispatch_scope_resolved,
            )
            resolved_host_dispatch_queue_id = host_dispatch_queue_id
        if (
            _normalize_host_machine_id(event.extras.get("host_machine_id")) != host_machine_id
            or _normalize_host_dispatch_queue_id(event.extras.get("host_dispatch_queue_id"))
            != host_dispatch_queue_id
        ):
            event = _event_with_host_identity(
                event,
                host_machine_id=host_machine_id,
                host_dispatch_queue_id=host_dispatch_queue_id,
            )
        if active_window is not None:
            start_ts = active_window[0] - max(int(step_pre_padding_us), 0)
            end_ts = active_window[1] + max(int(step_post_padding_us), 0)
            event_end_ts = _effective_event_end_ts(event)
            if event.ts > end_ts or event_end_ts < start_ts:
                continue
        events.append(event)
        if max_events_per_rank is None:
            continue
        if not is_ignorable_setup_api(event.api):
            budget_events += 1
        if budget_events >= max_events_per_rank:
            break

    return (
        RankTrace(
            rank=rank,
            path=trace_file,
            source=source,
            events=tuple(events),
        ),
        resolved_host_machine_id,
        resolved_host_dispatch_queue_id,
    )


def load_trace_directory(
    trace_dir: str | Path,
    *,
    max_events_per_rank: int | None = None,
    trace_window: str = "auto",
    step_pre_padding_us: int = 0,
    step_post_padding_us: int = 0,
    strict_json: bool = False,
) -> TraceBundle:
    path = Path(trace_dir)
    source = infer_trace_source(path)
    manifest_path = path / _CAPTURE_MANIFEST
    manifest = load_capture_manifest(path)
    if source is TraceSource.UNKNOWN:
        source = _infer_trace_source_from_manifest(manifest)
    if trace_window not in {"auto", "full", "step", "trace"}:
        raise ValueError(
            "trace_window must be one of 'auto', 'full', 'step', or 'trace', "
            f"got {trace_window!r}"
        )
    rank_files = list_rank_trace_files(path)
    cache_path: Path | None = None
    if _trace_bundle_cache_enabled():
        cache_path = _trace_bundle_cache_path(
            path,
            rank_files=rank_files,
            manifest_path=manifest_path,
            trace_window=trace_window,
            max_events_per_rank=max_events_per_rank,
            step_pre_padding_us=step_pre_padding_us,
            step_post_padding_us=step_post_padding_us,
            strict_json=strict_json,
        )
        cached_bundle = _load_trace_bundle_cache(cache_path)
        if cached_bundle is not None:
            return cached_bundle
    fidelity_windows = _manifest_fidelity_windows(
        manifest,
        prefer_explicit_fidelity_windows=trace_window == "trace",
    )
    rank_host_machines = _manifest_rank_host_machines(manifest)
    host_timing_dispatch_scope_resolved = _manifest_host_timing_dispatch_scope_resolved(manifest)
    rank_host_dispatch_queues = _manifest_rank_host_dispatch_queues(
        manifest,
        rank_host_machines=rank_host_machines,
        dispatch_scope=host_timing_dispatch_scope_resolved,
    )
    resolved_fidelity_windows: dict[int, FidelityWindow] = {}
    io_workers = _recommended_trace_io_workers(len(rank_files))

    if io_workers > 1:
        with ThreadPoolExecutor(max_workers=io_workers) as executor:
            fidelity_futures = {
                rank: executor.submit(
                    resolve_fidelity_window,
                    trace_file,
                    source=source,
                    manifest_window=fidelity_windows.get(rank),
                    strict_json=strict_json,
                    prefer_manifest_window=trace_window == "trace",
                )
                for trace_file in rank_files
                for rank in (_rank_from_path(trace_file),)
            }
            for rank, future in sorted(fidelity_futures.items()):
                resolved_fidelity_window = future.result()
                if resolved_fidelity_window is not None:
                    resolved_fidelity_windows[rank] = resolved_fidelity_window
    else:
        for trace_file in rank_files:
            rank = _rank_from_path(trace_file)
            resolved_fidelity_window = resolve_fidelity_window(
                trace_file,
                source=source,
                manifest_window=fidelity_windows.get(rank),
                strict_json=strict_json,
                prefer_manifest_window=trace_window == "trace",
            )
            if resolved_fidelity_window is not None:
                resolved_fidelity_windows[rank] = resolved_fidelity_window

    active_trace_window_mode: str | None = None
    active_step_windows: dict[int, tuple[int, int]] = {}
    if trace_window == "step":
        missing_ranks = sorted(
            rank
            for rank in (_rank_from_path(trace_file) for trace_file in rank_files)
            if rank not in resolved_fidelity_windows
        )
        if missing_ranks:
            raise ValueError(
                "trace_window='step' requires an explicit fidelity window for every rank; "
                f"missing ranks: {missing_ranks}"
            )
        invalid_ranks = sorted(
            rank
            for rank, fidelity_window in resolved_fidelity_windows.items()
            if not fidelity_window.is_paper_valid_step_window
        )
        if invalid_ranks:
            invalid_sources = {
                rank: resolved_fidelity_windows[rank].source for rank in invalid_ranks
            }
            raise ValueError(
                "trace_window='step' requires paper-valid step windows; "
                f"invalid ranks/sources: {invalid_sources}"
            )
        active_trace_window_mode = "step"
    elif trace_window == "trace":
        missing_ranks = sorted(
            rank
            for rank in (_rank_from_path(trace_file) for trace_file in rank_files)
            if rank not in resolved_fidelity_windows
        )
        if missing_ranks:
            raise ValueError(
                "trace_window='trace' requires a trace fidelity window for every rank; "
                f"missing ranks: {missing_ranks}"
            )
        active_trace_window_mode = "trace"
    elif trace_window == "auto":
        if bool(rank_files) and all(
            resolved_fidelity_windows.get(_rank_from_path(trace_file)) is not None
            and resolved_fidelity_windows[_rank_from_path(trace_file)].is_paper_valid_step_window
            for trace_file in rank_files
        ):
            active_trace_window_mode = "step"

    if active_trace_window_mode == "step":
        active_step_windows = {
            rank: (fidelity_window.start_ts, fidelity_window.end_ts)
            for rank, fidelity_window in resolved_fidelity_windows.items()
            if fidelity_window.is_paper_valid_step_window
        }
        active_trace_windows = dict(active_step_windows)
    elif active_trace_window_mode == "trace":
        active_trace_windows = {
            rank: (fidelity_window.start_ts, fidelity_window.end_ts)
            for rank, fidelity_window in resolved_fidelity_windows.items()
        }
    else:
        active_trace_windows = {}

    rank_traces: list[RankTrace] = []
    if io_workers > 1:
        with ThreadPoolExecutor(max_workers=io_workers) as executor:
            trace_futures = {
                rank: executor.submit(
                    _load_rank_trace,
                    trace_file,
                    source=source,
                    strict_json=strict_json,
                    active_window=active_trace_windows.get(rank),
                    step_pre_padding_us=step_pre_padding_us,
                    step_post_padding_us=step_post_padding_us,
                    max_events_per_rank=max_events_per_rank,
                    initial_host_machine_id=rank_host_machines.get(rank),
                    initial_host_dispatch_queue_id=rank_host_dispatch_queues.get(rank),
                    host_timing_dispatch_scope_resolved=host_timing_dispatch_scope_resolved,
                )
                for trace_file in rank_files
                for rank in (_rank_from_path(trace_file),)
            }
            for rank, future in sorted(trace_futures.items()):
                (
                    rank_trace,
                    resolved_host_machine_id,
                    resolved_host_dispatch_queue_id,
                ) = future.result()
                rank_traces.append(rank_trace)
                if resolved_host_machine_id is not None:
                    rank_host_machines[rank] = resolved_host_machine_id
                if resolved_host_dispatch_queue_id is not None:
                    rank_host_dispatch_queues[rank] = resolved_host_dispatch_queue_id
    else:
        for trace_file in rank_files:
            rank = _rank_from_path(trace_file)
            (
                rank_trace,
                resolved_host_machine_id,
                resolved_host_dispatch_queue_id,
            ) = _load_rank_trace(
                trace_file,
                source=source,
                strict_json=strict_json,
                active_window=active_trace_windows.get(rank),
                step_pre_padding_us=step_pre_padding_us,
                step_post_padding_us=step_post_padding_us,
                max_events_per_rank=max_events_per_rank,
                initial_host_machine_id=rank_host_machines.get(rank),
                initial_host_dispatch_queue_id=rank_host_dispatch_queues.get(rank),
                host_timing_dispatch_scope_resolved=host_timing_dispatch_scope_resolved,
            )
            rank_traces.append(rank_trace)
            if resolved_host_machine_id is not None:
                rank_host_machines[rank] = resolved_host_machine_id
            if resolved_host_dispatch_queue_id is not None:
                rank_host_dispatch_queues[rank] = resolved_host_dispatch_queue_id

    rank_traces.sort(key=lambda trace: trace.rank)
    original_world_size = None
    profiled_rank_groups: dict[int, tuple[int, ...]] = {}
    manifest_communicator_memberships: dict[str, tuple[int, ...]] = {}
    manifest_communicator_aliases: dict[tuple[int, str], str] = {}
    if manifest is not None:
        raw_world_size = manifest.get("original_world_size")
        if raw_world_size not in (None, ""):
            original_world_size = int(raw_world_size)
        raw_groups = manifest.get("profiled_rank_groups", {})
        profiled_rank_groups = {
            int(rank): tuple(int(member) for member in members)
            for rank, members in raw_groups.items()
        }
        manifest_communicator_memberships = _manifest_communicators(manifest)
        manifest_communicator_aliases = _manifest_communicator_aliases(manifest)

    need_trace_communicator_recovery = (
        not manifest_communicator_memberships
        or not manifest_communicator_aliases
    )
    use_trace_communicator_aliases = (
        not manifest_communicator_memberships
        or (source is TraceSource.REAL and not manifest_communicator_aliases)
    )
    if need_trace_communicator_recovery:
        trace_communicator_memberships, trace_communicator_aliases = _trace_communicator_recovery(
            path,
            source=source,
            strict_json=strict_json,
        )
        if not use_trace_communicator_aliases:
            trace_communicator_aliases = {}
    else:
        trace_communicator_memberships = {}
        trace_communicator_aliases = {}
    communicator_aliases = _merge_communicator_aliases(
        manifest_aliases=manifest_communicator_aliases,
        trace_aliases=trace_communicator_aliases,
    )
    if communicator_aliases:
        rank_traces = [
            _rewrite_rank_trace_comm_ids(
                rank_trace,
                communicator_aliases=communicator_aliases,
            )
            for rank_trace in rank_traces
        ]

    communicator_memberships = _merge_communicator_memberships(
        manifest_memberships=manifest_communicator_memberships,
        trace_memberships=trace_communicator_memberships,
    )

    bundle = TraceBundle(
        trace_dir=path,
        source=source,
        rank_traces=tuple(rank_traces),
        original_world_size=original_world_size,
        profiled_rank_groups=profiled_rank_groups,
        rank_host_machines=dict(rank_host_machines),
        rank_host_dispatch_queues=dict(rank_host_dispatch_queues),
        communicator_memberships=communicator_memberships,
        host_timing_dispatch_scope_resolved=host_timing_dispatch_scope_resolved,
        step_windows=dict(active_step_windows),
        fidelity_windows=dict(resolved_fidelity_windows),
        trace_window=active_trace_window_mode or "full",
    )
    if cache_path is not None:
        _store_trace_bundle_cache(cache_path, bundle)
    return bundle


def _clone_trace_event(event: TraceEvent, *, rank: int) -> TraceEvent:
    return TraceEvent(
        rank=rank,
        ordinal=event.ordinal,
        source=event.source,
        ts=event.ts,
        pid=event.pid,
        tid=event.tid,
        module=event.module,
        api=event.api,
        op_type=event.op_type,
        extras=dict(event.extras),
    )


def _materialized_members_for_variant(
    members: tuple[int, ...],
    profiled_rank_groups: dict[int, tuple[int, ...]],
    *,
    variant_index: int,
) -> tuple[int, ...]:
    if not members:
        return members
    representatives = set(profiled_rank_groups)
    if not all(member in representatives for member in members):
        return members

    remapped: list[int] = []
    for representative in members:
        materialized_members = profiled_rank_groups[representative]
        if variant_index >= len(materialized_members):
            return members
        remapped.append(int(materialized_members[variant_index]))
    return tuple(remapped)


def _clone_trace_event_for_materialized_rank(
    event: TraceEvent,
    *,
    rank: int,
    variant_index: int,
    communicator_memberships: dict[str, tuple[int, ...]],
    materialized_communicator_memberships: dict[str, tuple[int, ...]],
    profiled_rank_groups: dict[int, tuple[int, ...]],
    rank_host_machines: dict[int, str],
    rank_host_dispatch_queues: dict[int, str],
) -> TraceEvent:
    extras = dict(event.extras)
    host_machine_id = rank_host_machines.get(int(rank))
    if host_machine_id is not None:
        extras["host_machine_id"] = host_machine_id
    host_dispatch_queue_id = rank_host_dispatch_queues.get(int(rank))
    if host_dispatch_queue_id is not None:
        extras["host_dispatch_queue_id"] = host_dispatch_queue_id
    comm_id = extras.get("comm_id")
    if comm_id not in (None, ""):
        members = communicator_memberships.get(str(comm_id))
        if members:
            remapped_members = _materialized_members_for_variant(
                members,
                profiled_rank_groups,
                variant_index=variant_index,
            )
            if remapped_members != members:
                remapped_comm_id = build_emulated_communicator_id(remapped_members)
                extras["comm_id"] = remapped_comm_id
                materialized_communicator_memberships[remapped_comm_id] = remapped_members
    return TraceEvent(
        rank=rank,
        ordinal=event.ordinal,
        source=event.source,
        ts=event.ts,
        pid=event.pid,
        tid=event.tid,
        module=event.module,
        api=event.api,
        op_type=event.op_type,
        extras=extras,
    )


def _validate_profiled_rank_groups(
    bundle: TraceBundle,
    groups: dict[int, tuple[int, ...]],
) -> None:
    missing_representatives = sorted(set(groups) - set(bundle.rank_ids()))
    if missing_representatives:
        raise ValueError(
            "profiled_rank_groups references representatives with no captured "
            f"trace file: {missing_representatives}"
        )

    members_seen: set[int] = set()
    overlapping_members: set[int] = set()
    for members in groups.values():
        for member in members:
            if member in members_seen:
                overlapping_members.add(member)
            members_seen.add(member)
    if overlapping_members:
        raise ValueError(
            "profiled_rank_groups contains overlapping logical ranks: "
            f"{sorted(overlapping_members)}"
        )

    if bundle.original_world_size is not None and len(members_seen) != bundle.original_world_size:
        raise ValueError(
            "profiled_rank_groups does not cover the declared original_world_size: "
            f"expected {bundle.original_world_size}, got {len(members_seen)}"
        )


def materialize_profiled_rank_traces(bundle: TraceBundle) -> TraceBundle:
    """
    Expand representative traces back to one logical rank trace per original rank.

    This is useful for Figure 13 style experiments where phase 1 captures only a
    few representative workers but later stages should model the full logical
    world size.
    """
    groups = dict(bundle.profiled_rank_groups)
    if not groups:
        return bundle
    if bundle.logical_rank_materialized:
        return bundle

    _validate_profiled_rank_groups(bundle, groups)
    representative_traces = {rank_trace.rank: rank_trace for rank_trace in bundle.rank_traces}
    materialized: list[RankTrace] = []
    materialized_communicator_memberships = dict(bundle.communicator_memberships)
    materialized_rank_host_machines = {
        int(member): host_machine_id
        for representative, members in sorted(groups.items())
        for member in members
        for host_machine_id in (
            bundle.rank_host_machines.get(
                int(member),
                bundle.rank_host_machines.get(int(representative), f"logical_host_unknown:{member}"),
            ),
        )
    }
    materialized_rank_host_dispatch_queues: dict[int, str] = {}
    materialized_dispatch_scope = bundle.host_timing_dispatch_scope_resolved
    for representative, members in sorted(groups.items()):
        for member in members:
            member = int(member)
            explicit_dispatch_queue = bundle.rank_host_dispatch_queues.get(member)
            if explicit_dispatch_queue is None and member == int(representative):
                explicit_dispatch_queue = bundle.rank_host_dispatch_queues.get(int(representative))
            host_machine_id = materialized_rank_host_machines.get(
                member,
                f"logical_host_unknown:{member}",
            )
            materialized_rank_host_dispatch_queues[member] = _canonical_host_dispatch_queue_id(
                explicit_dispatch_queue,
                rank=member,
                host_machine_id=host_machine_id,
                dispatch_scope=materialized_dispatch_scope,
            )
    for representative, members in sorted(groups.items()):
        trace = representative_traces[representative]
        for member in members:
            variant_index = members.index(member)
            materialized.append(
                RankTrace(
                    rank=member,
                    path=bundle.trace_dir / f"rank_{member}.jsonl",
                    source=trace.source,
                    events=tuple(
                        _clone_trace_event_for_materialized_rank(
                            event,
                            rank=member,
                            variant_index=variant_index,
                            communicator_memberships=bundle.communicator_memberships,
                            materialized_communicator_memberships=materialized_communicator_memberships,
                            profiled_rank_groups=groups,
                            rank_host_machines=materialized_rank_host_machines,
                            rank_host_dispatch_queues=materialized_rank_host_dispatch_queues,
                        )
                        for event in trace.events
                    ),
                )
            )

    materialized.sort(key=lambda trace: trace.rank)
    return TraceBundle(
        trace_dir=bundle.trace_dir,
        source=bundle.source,
        rank_traces=tuple(materialized),
        original_world_size=bundle.world_size,
        captured_world_size=bundle.profiled_world_size,
        profiled_rank_groups=groups,
        rank_host_machines=materialized_rank_host_machines,
        rank_host_dispatch_queues=materialized_rank_host_dispatch_queues,
        communicator_memberships=materialized_communicator_memberships,
        host_timing_dispatch_scope_resolved=bundle.host_timing_dispatch_scope_resolved,
        step_windows={
            int(member): tuple(bundle.step_windows[trace.rank])
            for trace in bundle.rank_traces
            if trace.rank in bundle.step_windows
            for member in groups.get(trace.rank, (trace.rank,))
        },
        fidelity_windows={
            int(member): bundle.fidelity_windows[trace.rank]
            for trace in bundle.rank_traces
            if trace.rank in bundle.fidelity_windows
            for member in groups.get(trace.rank, (trace.rank,))
        },
        logical_rank_materialized=True,
        trace_window=bundle.trace_window,
    )


_SIGNATURE_EXTRA_KEYS = (
    "api",
    "type",
    "bytes",
    "size",
    "count",
    "numel",
    "kind",
    "datatype",
    "dtype_code",
    "op",
    "collective",
    "m",
    "n",
    "k",
    "lda",
    "ldb",
    "ldc",
    "batch_count",
    "batchCount",
    "stride_a",
    "stride_b",
    "stride_c",
    "strideA",
    "strideB",
    "strideC",
    "stridea",
    "strideb",
    "stridec",
    "algorithm",
    "algo",
    "top_k",
    "ep_size",
    "capacity_factor",
    "world_size",
)


def _signature_for_event(event: TraceEvent) -> tuple[object, ...]:
    meaningful_extras = tuple(
        (key, event.extras[key]) for key in _SIGNATURE_EXTRA_KEYS if key in event.extras and key != "type"
    )
    return (event.api, event.op_type, meaningful_extras)


def _rank_trace_signature(rank_trace: RankTrace) -> tuple[object, ...]:
    return tuple(_signature_for_event(event) for event in rank_trace.events)


_PATTERN_HASH_BASE = 1_000_003
_PATTERN_HASH_MASK = (1 << 64) - 1


def _safe_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except Exception:
        return None


def _normalized_root_role(
    root: object,
    *,
    members: tuple[int, ...] | None,
) -> object:
    root_int = _safe_int(root)
    if root_int is None:
        return root
    if not members:
        return root_int
    if root_int in members:
        return members.index(root_int)
    if 0 <= root_int < len(members):
        return root_int
    return root_int


def _pattern_signature_for_event(
    event: TraceEvent,
    *,
    communicator_memberships: dict[str, tuple[int, ...]],
) -> tuple[object, ...] | None:
    if is_ignorable_dedup_api(event.api, event.op_type):
        return None

    meaningful_extras: list[tuple[str, object]] = [
        (key, event.extras[key])
        for key in _SIGNATURE_EXTRA_KEYS
        if key in event.extras and key != "type"
    ]
    if event.op_type != "nccl_collective":
        return (event.api, event.op_type, tuple(meaningful_extras))

    comm_id = str(event.extras.get("comm_id", ""))
    members = communicator_memberships.get(comm_id)
    comm_size = _safe_int(event.extras.get("nranks"))
    if comm_size is None and members:
        comm_size = len(members)
    if comm_size is not None:
        meaningful_extras.append(("nranks", comm_size))

    if event.api in {"ncclSend", "ncclRecv"} and "peer" in event.extras:
        meaningful_extras.append(("peer_role", _safe_int(event.extras.get("peer"))))
    if "root" in event.extras:
        meaningful_extras.append(
            (
                "root_role",
                _normalized_root_role(
                    event.extras.get("root"),
                    members=members,
                ),
            )
        )

    return (event.api, event.op_type, tuple(meaningful_extras))


def _rank_trace_pattern_tokens(
    rank_trace: RankTrace,
    *,
    communicator_memberships: dict[str, tuple[int, ...]],
) -> tuple[tuple[object, ...], ...]:
    tokens: list[tuple[object, ...]] = []
    for event in rank_trace.events:
        token = _pattern_signature_for_event(
            event,
            communicator_memberships=communicator_memberships,
        )
        if token is not None:
            tokens.append(token)
    return tuple(tokens)


def pattern_tokens_for_rank_trace(
    rank_trace: RankTrace,
    *,
    communicator_memberships: dict[str, tuple[int, ...]],
) -> tuple[tuple[object, ...], ...]:
    return _rank_trace_pattern_tokens(
        rank_trace,
        communicator_memberships=communicator_memberships,
    )


def _rolling_hashes(
    tokens: tuple[tuple[object, ...], ...],
    *,
    window_size: int,
) -> tuple[int, ...]:
    if not tokens:
        return ()

    encoded = [
        int.from_bytes(
            hashlib.blake2b(
                repr(token).encode("utf-8"),
                digest_size=8,
            ).digest(),
            byteorder="big",
            signed=False,
        )
        for token in tokens
    ]
    width = max(1, min(int(window_size), len(encoded)))
    if width == len(encoded):
        digest = 0
        for value in encoded:
            digest = ((digest * _PATTERN_HASH_BASE) + value + 1) & _PATTERN_HASH_MASK
        return (digest,)

    highest_power = 1
    for _ in range(width - 1):
        highest_power = (highest_power * _PATTERN_HASH_BASE) & _PATTERN_HASH_MASK

    current = 0
    for value in encoded[:width]:
        current = ((current * _PATTERN_HASH_BASE) + value + 1) & _PATTERN_HASH_MASK

    hashes = [current]
    for index in range(width, len(encoded)):
        outgoing = encoded[index - width] + 1
        current = (current - ((outgoing * highest_power) & _PATTERN_HASH_MASK)) & _PATTERN_HASH_MASK
        current = ((current * _PATTERN_HASH_BASE) + encoded[index] + 1) & _PATTERN_HASH_MASK
        hashes.append(current)
    return tuple(hashes)


def _rank_trace_pattern_fingerprint(
    rank_trace: RankTrace,
    *,
    communicator_memberships: dict[str, tuple[int, ...]],
    window_size: int,
) -> tuple[int, int, tuple[int, ...]]:
    tokens = _rank_trace_pattern_tokens(
        rank_trace,
        communicator_memberships=communicator_memberships,
    )
    return (
        len(tokens),
        max(1, min(int(window_size), len(tokens) if tokens else 1)),
        _rolling_hashes(tokens, window_size=window_size),
    )


def pattern_fingerprint_for_rank_trace(
    rank_trace: RankTrace,
    *,
    communicator_memberships: dict[str, tuple[int, ...]],
    window_size: int = 16,
) -> tuple[int, int, tuple[int, ...]]:
    return _rank_trace_pattern_fingerprint(
        rank_trace,
        communicator_memberships=communicator_memberships,
        window_size=window_size,
    )


def _build_deduped_trace_bundle(
    bundle: TraceBundle,
    *,
    groups: dict[object, list[RankTrace]],
) -> TraceBundle:
    deduped_rank_traces: list[RankTrace] = []
    profiled_rank_groups: dict[int, tuple[int, ...]] = {}
    for traces in groups.values():
        traces = sorted(traces, key=lambda item: item.rank)
        representative = traces[0]
        deduped_rank_traces.append(representative)
        logical_members: list[int] = []
        seen_members: set[int] = set()
        for trace in traces:
            members = bundle.profiled_rank_groups.get(trace.rank, (trace.rank,))
            for member in members:
                member = int(member)
                if member in seen_members:
                    continue
                seen_members.add(member)
                logical_members.append(member)
        profiled_rank_groups[representative.rank] = tuple(sorted(logical_members))

    deduped_rank_traces.sort(key=lambda trace: trace.rank)
    return TraceBundle(
        trace_dir=bundle.trace_dir,
        source=bundle.source,
        rank_traces=tuple(deduped_rank_traces),
        original_world_size=bundle.world_size,
        captured_world_size=len(deduped_rank_traces),
        profiled_rank_groups=profiled_rank_groups,
        rank_host_machines={
            int(trace.rank): bundle.rank_host_machines[trace.rank]
            for trace in deduped_rank_traces
            if trace.rank in bundle.rank_host_machines
        },
        rank_host_dispatch_queues={
            int(trace.rank): bundle.rank_host_dispatch_queues[trace.rank]
            for trace in deduped_rank_traces
            if trace.rank in bundle.rank_host_dispatch_queues
        },
        communicator_memberships=dict(bundle.communicator_memberships),
        host_timing_dispatch_scope_resolved=bundle.host_timing_dispatch_scope_resolved,
        step_windows={
            int(trace.rank): tuple(bundle.step_windows[trace.rank])
            for trace in deduped_rank_traces
            if trace.rank in bundle.step_windows
        },
        fidelity_windows={
            int(trace.rank): bundle.fidelity_windows[trace.rank]
            for trace in deduped_rank_traces
            if trace.rank in bundle.fidelity_windows
        },
        logical_rank_materialized=False,
        trace_window=bundle.trace_window,
    )


def dedup_identical_rank_traces(bundle: TraceBundle) -> TraceBundle:
    """Collapse byte-for-byte identical low-level rank traces to one representative."""
    groups: dict[tuple[object, ...], list[RankTrace]] = defaultdict(list)
    for rank_trace in bundle.rank_traces:
        groups[_rank_trace_signature(rank_trace)].append(rank_trace)
    return _build_deduped_trace_bundle(bundle, groups=groups)


def dedup_pattern_rank_traces(
    bundle: TraceBundle,
    *,
    window_size: int = 16,
) -> TraceBundle:
    """
    Collapse semantically redundant rank traces using post-capture pattern hashes.

    This mirrors the paper's worker-dedup spirit more closely than
    dedup_identical_rank_traces(): rank-local identifiers such as communicator IDs,
    call indices, timestamps, and process IDs are ignored, and candidate duplicate
    traces are first bucketed by a rolling-hash fingerprint before exact normalized
    pattern confirmation.
    """
    hashed_groups: dict[tuple[int, int, tuple[int, ...]], list[tuple[RankTrace, tuple[tuple[object, ...], ...]]]] = defaultdict(list)
    for rank_trace in bundle.rank_traces:
        tokens = _rank_trace_pattern_tokens(
            rank_trace,
            communicator_memberships=bundle.communicator_memberships,
        )
        fingerprint = _rank_trace_pattern_fingerprint(
            rank_trace,
            communicator_memberships=bundle.communicator_memberships,
            window_size=window_size,
        )
        hashed_groups[fingerprint].append((rank_trace, tokens))

    groups: dict[tuple[tuple[object, ...], ...], list[RankTrace]] = defaultdict(list)
    for candidates in hashed_groups.values():
        for rank_trace, tokens in candidates:
            groups[tokens].append(rank_trace)

    return _build_deduped_trace_bundle(bundle, groups=groups)


def inspect_trace_directory(
    trace_dir: str | Path,
    *,
    sample_events_per_rank: int = 16,
    strict_json: bool = False,
) -> TraceDirectorySummary:
    path = Path(trace_dir)
    source = infer_trace_source(path)
    observed_keys: set[str] = set()
    observed_types: set[str] = set()
    sample_event_counts: dict[int, int] = defaultdict(int)
    rank_files = list_rank_trace_files(path)

    for trace_file in rank_files:
        rank = _rank_from_path(trace_file)
        with trace_file.open() as handle:
            for line_number, line in enumerate(handle):
                payload = line.strip()
                if not payload:
                    continue
                try:
                    record = json.loads(payload)
                except json.JSONDecodeError as exc:
                    if not strict_json:
                        warnings.warn(
                            f"skipping invalid JSON in {trace_file}:{line_number + 1}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        continue
                    raise ValueError(f"invalid JSON in {trace_file}:{line_number + 1}") from exc
                TraceEvent.from_json_record(
                    record,
                    rank=rank,
                    ordinal=line_number,
                    source=source,
                )
                observed_keys.update(str(key) for key in record.keys())
                observed_types.add(str(record["type"]))
                sample_event_counts[rank] += 1
                if sample_event_counts[rank] >= sample_events_per_rank:
                    break

    rank_ids = tuple(sorted(_rank_from_path(path) for path in rank_files))
    return TraceDirectorySummary(
        trace_dir=path,
        source=source,
        rank_files=rank_files,
        rank_ids=rank_ids,
        sample_event_counts=dict(sample_event_counts),
        observed_keys=tuple(sorted(observed_keys)),
        observed_types=tuple(sorted(observed_types)),
    )
