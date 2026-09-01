"""
Python-side step markers for Maya-lite capture segmentation.

These markers are optional. When present, capture paths can translate them into
per-rank trace timestamp windows and compare only the actual training-step
region rather than bootstrap or teardown noise.
"""

from __future__ import annotations

import ctypes
import json
import os
import time
from functools import lru_cache
from contextlib import contextmanager
from pathlib import Path

from .schema import TraceSource


_ENV_MARKERS_PATH = "FLEXSIM_MAYA_MARKERS_PATH"
TRACE_MARKER_API = "mayaStepMarker"
TRACE_MARKER_TYPE = "marker"


@lru_cache(maxsize=1)
def _resolve_trace_marker_helper():
    candidates: list[object] = [None, "libcudart.so.12", "libcudart.so"]
    for candidate in candidates:
        try:
            library = ctypes.CDLL(None if candidate is None else str(candidate))
        except OSError:
            continue
        try:
            function = getattr(library, "fakecudaTraceMarker")
        except AttributeError:
            continue
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_longlong]
        function.restype = ctypes.c_longlong
        return function
    return None


def _emit_trace_marker(kind: str, *, step: int | None, label: str) -> int | None:
    if os.environ.get("FAKECUDA_TRACE") != "1":
        return None
    helper = _resolve_trace_marker_helper()
    if helper is None:
        return None
    emitted_trace_ts = int(
        helper(
            kind.encode("utf-8"),
            label.encode("utf-8"),
            int(step if step is not None else -1),
        )
    )
    if emitted_trace_ts < 0:
        return None
    return emitted_trace_ts


def _marker_trace_ts_us(record: dict[str, object]) -> int | None:
    for key in ("trace_ts", "trace_ts_us"):
        raw = record.get(key)
        if raw in (None, ""):
            continue
        return int(raw)
    return None


def _resolve_marker_window_from_trace_timestamps(
    begins: list[dict[str, object]],
    ends: list[dict[str, object]],
    *,
    occurrence: int | None = None,
    step: int | None = None,
) -> dict[str, int | str] | None:
    if not begins or not ends:
        return None
    if step is not None:
        begins = [record for record in begins if int(record.get("step", -1)) == int(step)]
        ends = [record for record in ends if int(record.get("step", -1)) == int(step)]
    if not begins or not ends:
        return None
    selected_begin = begins[0]
    selected_end = ends[-1]
    if occurrence is not None:
        if occurrence <= 0:
            raise ValueError(f"occurrence must be positive, got {occurrence}")
        if occurrence > len(begins) or occurrence > len(ends):
            return None
        selected_begin = begins[occurrence - 1]
        selected_end = ends[occurrence - 1]
    start_ts = _marker_trace_ts_us(selected_begin)
    end_ts = _marker_trace_ts_us(selected_end)
    if start_ts is None or end_ts is None:
        return None
    if end_ts < start_ts:
        return None
    resolved: dict[str, int | str] = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "source": "trace_markers",
        "step_count": min(len(begins), len(ends)),
    }
    if occurrence is not None:
        resolved["occurrence"] = occurrence
    if step is not None:
        resolved["step"] = int(step)
    return resolved


def resolve_step_window_from_marker_trace_timestamps(
    marker_records: list[dict[str, object]],
    *,
    label: str = "training_step",
) -> dict[str, int | str] | None:
    begins = [
        record for record in marker_records
        if record.get("kind") == "step_begin" and record.get("label", label) == label
    ]
    ends = [
        record for record in marker_records
        if record.get("kind") == "step_end" and record.get("label", label) == label
    ]
    return _resolve_marker_window_from_trace_timestamps(begins, ends)


def resolve_indexed_step_window_from_marker_trace_timestamps(
    marker_records: list[dict[str, object]],
    *,
    occurrence: int = 1,
    step: int | None = None,
    label: str = "training_step",
) -> dict[str, int | str] | None:
    begins = [
        record for record in marker_records
        if record.get("kind") == "step_begin" and record.get("label", label) == label
    ]
    ends = [
        record for record in marker_records
        if record.get("kind") == "step_end" and record.get("label", label) == label
    ]
    return _resolve_marker_window_from_trace_timestamps(
        begins,
        ends,
        occurrence=occurrence,
        step=step,
    )


def _emit_step_marker_payload(kind: str, *, step: int | None, label: str) -> dict[str, object]:
    emitted_trace_ts = _emit_trace_marker(kind, step=step, label=label)
    payload: dict[str, object] = {
        "kind": kind,
        "label": label,
        "pid": os.getpid(),
        "realtime_ns": time.time_ns(),
        "monotonic_ns": time.perf_counter_ns(),
        "trace_marker_emitted": emitted_trace_ts is not None,
    }
    if step is not None:
        payload["step"] = int(step)
    if emitted_trace_ts is not None:
        payload["trace_ts"] = emitted_trace_ts
    return payload


def markers_path_from_env() -> Path | None:
    raw = os.environ.get(_ENV_MARKERS_PATH)
    if not raw:
        return None
    return Path(raw)


def emit_step_marker(kind: str, *, step: int | None = None, label: str = "training_step") -> None:
    path = markers_path_from_env()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _emit_step_marker_payload(kind, step=step, label=label)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def mark_step_begin(step: int | None = None, *, label: str = "training_step") -> None:
    emit_step_marker("step_begin", step=step, label=label)


def mark_step_end(step: int | None = None, *, label: str = "training_step") -> None:
    emit_step_marker("step_end", step=step, label=label)


@contextmanager
def step_window(step: int | None = None, *, label: str = "training_step"):
    mark_step_begin(step, label=label)
    try:
        yield
    finally:
        mark_step_end(step, label=label)


def load_step_markers(path: str | Path) -> list[dict[str, object]]:
    marker_path = Path(path)
    if not marker_path.exists():
        return []
    records: list[dict[str, object]] = []
    with marker_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            try:
                raw = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict):
                continue
            records.append(raw)
    return records


def resolve_step_window_from_trace_markers(
    trace_path: str | Path,
    *,
    label: str = "training_step",
) -> dict[str, int | str] | None:
    path = Path(trace_path)
    if not path.exists():
        return None
    begins: list[dict[str, object]] = []
    ends: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            if record.get("api") != TRACE_MARKER_API or record.get("type") != TRACE_MARKER_TYPE:
                continue
            if record.get("label", label) != label:
                continue
            if record.get("kind") == "step_begin":
                begins.append(record)
            elif record.get("kind") == "step_end":
                ends.append(record)
    if not begins or not ends:
        return None
    start_ts = begins[0].get("ts")
    end_ts = ends[-1].get("ts")
    if start_ts in (None, "") or end_ts in (None, ""):
        return None
    start_ts = int(start_ts)
    end_ts = int(end_ts)
    if end_ts < start_ts:
        return None
    return {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "source": "trace_markers",
        "step_count": min(len(begins), len(ends)),
    }


def resolve_indexed_step_window_from_trace_markers(
    trace_path: str | Path,
    *,
    occurrence: int = 1,
    step: int | None = None,
    label: str = "training_step",
) -> dict[str, int | str] | None:
    if occurrence <= 0:
        raise ValueError(f"occurrence must be positive, got {occurrence}")

    path = Path(trace_path)
    if not path.exists():
        return None
    begins: list[dict[str, object]] = []
    ends: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            if record.get("api") != TRACE_MARKER_API or record.get("type") != TRACE_MARKER_TYPE:
                continue
            if record.get("label", label) != label:
                continue
            if record.get("kind") == "step_begin":
                begins.append(record)
            elif record.get("kind") == "step_end":
                ends.append(record)
    if step is not None:
        begins = [record for record in begins if int(record.get("step", -1)) == int(step)]
        ends = [record for record in ends if int(record.get("step", -1)) == int(step)]
    if not begins or not ends:
        return None
    if occurrence > len(begins) or occurrence > len(ends):
        return None

    begin = begins[occurrence - 1]
    end = ends[occurrence - 1]
    start_ts = begin.get("ts")
    end_ts = end.get("ts")
    if start_ts in (None, "") or end_ts in (None, ""):
        return None
    start_ts = int(start_ts)
    end_ts = int(end_ts)
    if end_ts < start_ts:
        return None
    resolved: dict[str, int | str] = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "source": "trace_markers",
        "step_count": min(len(begins), len(ends)),
        "occurrence": occurrence,
    }
    if step is not None:
        resolved["step"] = int(step)
    return resolved


def resolve_step_window_from_markers(
    marker_records: list[dict[str, object]],
    *,
    source: TraceSource,
    label: str = "training_step",
) -> dict[str, int | str] | None:
    begins = [
        record for record in marker_records
        if record.get("kind") == "step_begin" and record.get("label", label) == label
    ]
    ends = [
        record for record in marker_records
        if record.get("kind") == "step_end" and record.get("label", label) == label
    ]
    if not begins or not ends:
        return None
    clock_key = "monotonic_ns" if source is TraceSource.REAL else "realtime_ns"
    start_ns = begins[0].get(clock_key)
    end_ns = ends[-1].get(clock_key)
    if start_ns in (None, "") or end_ns in (None, ""):
        return None
    start_ts = int(int(start_ns) / 1_000)
    end_ts = int(int(end_ns) / 1_000)
    if end_ts < start_ts:
        return None
    return {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "source": "trace_markers",
        "step_count": min(len(begins), len(ends)),
    }


def completed_step_count_from_markers(
    marker_records: list[dict[str, object]],
    *,
    label: str = "training_step",
) -> int:
    begins = [
        record for record in marker_records
        if record.get("kind") == "step_begin" and record.get("label", label) == label
    ]
    ends = [
        record for record in marker_records
        if record.get("kind") == "step_end" and record.get("label", label) == label
    ]
    return min(len(begins), len(ends))


def resolve_indexed_step_window_from_markers(
    marker_records: list[dict[str, object]],
    *,
    source: TraceSource,
    occurrence: int = 1,
    step: int | None = None,
    label: str = "training_step",
) -> dict[str, int | str] | None:
    if occurrence <= 0:
        raise ValueError(f"occurrence must be positive, got {occurrence}")

    begins = [
        record for record in marker_records
        if record.get("kind") == "step_begin" and record.get("label", label) == label
    ]
    ends = [
        record for record in marker_records
        if record.get("kind") == "step_end" and record.get("label", label) == label
    ]
    if step is not None:
        begins = [record for record in begins if int(record.get("step", -1)) == int(step)]
        ends = [record for record in ends if int(record.get("step", -1)) == int(step)]
    if not begins or not ends:
        return None
    if occurrence > len(begins) or occurrence > len(ends):
        return None

    begin = begins[occurrence - 1]
    end = ends[occurrence - 1]
    clock_key = "monotonic_ns" if source is TraceSource.REAL else "realtime_ns"
    start_ns = begin.get(clock_key)
    end_ns = end.get(clock_key)
    if start_ns in (None, "") or end_ns in (None, ""):
        return None
    start_ts = int(int(start_ns) / 1_000)
    end_ts = int(int(end_ns) / 1_000)
    if end_ts < start_ts:
        return None
    resolved: dict[str, int | str] = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "source": "trace_markers",
        "step_count": min(len(begins), len(ends)),
        "occurrence": occurrence,
    }
    if step is not None:
        resolved["step"] = int(step)
    return resolved
