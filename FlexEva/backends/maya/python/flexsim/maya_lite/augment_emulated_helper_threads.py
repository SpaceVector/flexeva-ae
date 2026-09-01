#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from flexsim.maya_lite.io import (
    fidelity_window_from_payload,
    list_rank_trace_files,
    load_capture_manifest,
)

HELPER_THREAD_AUGMENTATION_STATUS_NOT_REQUIRED = "not_required"
HELPER_THREAD_AUGMENTATION_STATUS_MISSING_SUMMARY_DIR = "missing_summary_dir"
HELPER_THREAD_AUGMENTATION_STATUS_SUMMARY_DIR_NOT_FOUND = "summary_dir_not_found"
HELPER_THREAD_AUGMENTATION_STATUS_COMPLETED = "completed"
HELPER_THREAD_AUGMENTATION_STATUS_FAILED = "failed"

# Keep helper augmentation aligned with the Emulator stage in Figure 6:
# missing host/control-plane helper threads should be repaired in the raw
# emulated trace, not compensated later in Predictor/Simulator.
# When synthetic helper events extend beyond the paper-valid step window, we
# record the wider diagnostic envelope in fidelity_windows without mutating the
# canonical step_windows contract.
_SYNTHETIC_HELPER_START_APIS = {
    "ncclCommGetAsyncError",
    "ncclGetVersion",
    "cudaSetDevice",
    "cudaGetDevice",
    "cudaGetLastError",
}


def _guess_module(api: str) -> str:
    if api.startswith("nccl"):
        return "libnccl.so.2"
    return "libcudart.so.12"


def _guess_type(api: str) -> str:
    if api in {"cudaSetDevice", "cudaGetDevice"}:
        return "context_op"
    return "other"


def _load_helper_templates(summary_path: Path) -> list[dict]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    by_rank = payload.get("helper_thread_templates_by_rank", {})
    if not isinstance(by_rank, dict):
        return []
    rank = summary_path.stem.split("_")[-1]
    templates = by_rank.get(str(int(rank)), [])
    if not isinstance(templates, list):
        return []
    return [
        template
        for template in templates
        if str(template.get("first_api", "")) in _SYNTHETIC_HELPER_START_APIS
    ]


def _trace_rank(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])


def _manifest_path(trace_dir: Path) -> Path:
    return trace_dir / "capture_manifest.json"


def _write_manifest(trace_dir: Path, manifest: dict[str, object]) -> None:
    _manifest_path(trace_dir).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _helper_thread_augmentation_payload(
    *,
    expected: bool,
    status: str,
    summary_dir: Path | None,
    injected_by_rank: dict[int, int] | None = None,
    step_window_extensions_by_rank: dict[str, dict[str, int]] | None = None,
    error: str | None = None,
) -> dict[str, object]:
    normalized_injected_by_rank = {
        str(int(rank)): int(count)
        for rank, count in sorted((injected_by_rank or {}).items())
    }
    normalized_step_window_extensions = {
        str(rank): {
            "start_ts": int(payload["start_ts"]),
            "end_ts": int(payload["end_ts"]),
        }
        for rank, payload in sorted((step_window_extensions_by_rank or {}).items())
    }
    payload: dict[str, object] = {
        "expected": bool(expected),
        "status": str(status),
        "embedded_in_emulator_artifact": bool(
            status in {
                HELPER_THREAD_AUGMENTATION_STATUS_NOT_REQUIRED,
                HELPER_THREAD_AUGMENTATION_STATUS_COMPLETED,
            }
        ),
        "summary_dir": str(summary_dir.resolve()) if summary_dir is not None else None,
        "checked_rank_count": len(normalized_injected_by_rank),
        "injected_by_rank": normalized_injected_by_rank,
        "total_injected_events": int(sum(normalized_injected_by_rank.values())),
        "step_window_extensions_by_rank": normalized_step_window_extensions,
    }
    if error:
        payload["error"] = str(error)
    return payload


def record_helper_thread_augmentation_status(
    trace_dir: Path,
    *,
    expected: bool,
    status: str,
    summary_dir: Path | None,
    injected_by_rank: dict[int, int] | None = None,
    step_window_extensions_by_rank: dict[str, dict[str, int]] | None = None,
    error: str | None = None,
) -> dict[str, object]:
    trace_dir = trace_dir.resolve()
    manifest = load_capture_manifest(trace_dir) or {}
    payload = _helper_thread_augmentation_payload(
        expected=expected,
        status=status,
        summary_dir=summary_dir,
        injected_by_rank=injected_by_rank,
        step_window_extensions_by_rank=step_window_extensions_by_rank,
        error=error,
    )
    manifest["helper_thread_augmentation"] = payload
    _write_manifest(trace_dir, manifest)
    return payload


def _load_step_windows(trace_dir: Path) -> dict[int, tuple[int, int]]:
    manifest = load_capture_manifest(trace_dir)
    if not manifest:
        return {}
    raw = manifest.get("step_windows", {})
    if not isinstance(raw, dict):
        raw = manifest.get("fidelity_windows", {})
    if not isinstance(raw, dict):
        return {}
    resolved: dict[int, tuple[int, int]] = {}
    for raw_rank, payload in raw.items():
        fidelity_window = fidelity_window_from_payload(payload, default_source="manifest")
        if fidelity_window is None or not fidelity_window.is_paper_valid_step_window:
            continue
        try:
            rank = int(raw_rank)
        except Exception:
            continue
        resolved[rank] = (fidelity_window.start_ts, fidelity_window.end_ts)
    return resolved


def _load_trace_events(trace_path: Path) -> list[dict]:
    events: list[dict] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            events.append(json.loads(payload))
    return events


def _existing_helper_thread_start_counts(events: list[dict]) -> dict[str, int]:
    first_by_tid: dict[int, tuple[int, str]] = {}
    for event in events:
        try:
            tid = int(event["tid"])
            ts = int(event["ts"])
            api = str(event["api"])
        except Exception:
            continue
        current = first_by_tid.get(tid)
        if current is None or ts < current[0]:
            first_by_tid[tid] = (ts, api)
    counts: dict[str, int] = {}
    for _, api in sorted(first_by_tid.values(), key=lambda item: (item[0], item[1])):
        if api not in _SYNTHETIC_HELPER_START_APIS:
            continue
        counts[api] = counts.get(api, 0) + 1
    return counts


def augment_trace_file(
    trace_path: Path,
    *,
    summary_path: Path,
    step_window: tuple[int, int] | None,
) -> dict[str, int | None]:
    templates = _load_helper_templates(summary_path)
    if not templates:
        return {
            "injected_events": 0,
            "synthetic_min_ts": None,
            "synthetic_max_ts": None,
        }

    events = _load_trace_events(trace_path)
    if not events:
        return {
            "injected_events": 0,
            "synthetic_min_ts": None,
            "synthetic_max_ts": None,
        }

    base_ts = step_window[0] if step_window is not None else min(int(event["ts"]) for event in events)
    pid = int(events[0]["pid"])
    max_tid = max(int(event["tid"]) for event in events)
    existing_counts = _existing_helper_thread_start_counts(events)
    consumed_counts: dict[str, int] = {}
    synthetic_events: list[dict] = []

    for template in sorted(templates, key=lambda item: (int(item["start_offset_us"]), int(item["end_offset_us"]))):
        first_api = str(template["first_api"])
        already_present = existing_counts.get(first_api, 0)
        consumed = consumed_counts.get(first_api, 0)
        if consumed < already_present:
            consumed_counts[first_api] = consumed + 1
            continue

        max_tid += 1
        start_ts = int(base_ts + int(template["start_offset_us"]))
        end_ts = int(base_ts + int(template["end_offset_us"]))
        api_sequence = [str(api) for api in template.get("api_sequence", []) if str(api)]
        if api_sequence:
            seq_len = len(api_sequence)
            if seq_len == 1:
                event_timestamps = [start_ts]
            else:
                span = max(end_ts - start_ts, 0)
                event_timestamps = [
                    int(start_ts + round((span * index) / max(seq_len - 1, 1)))
                    for index in range(seq_len)
                ]
            for index, (api, ts) in enumerate(zip(api_sequence, event_timestamps, strict=False)):
                synthetic_events.append(
                    {
                        "ts": ts,
                        "pid": pid,
                        "tid": max_tid,
                        "mod": _guess_module(api),
                        "api": api,
                        "type": _guess_type(api),
                        "synthetic_helper_thread": True,
                        "helper_thread_source_tid": int(template.get("source_tid", -1)),
                        "helper_thread_role": "sequence",
                        "helper_thread_event_index": index,
                        "helper_thread_event_count": seq_len,
                    }
                )
        else:
            dominant_api = str(template.get("dominant_api") or first_api)
            synthetic_events.append(
                {
                    "ts": start_ts,
                    "pid": pid,
                    "tid": max_tid,
                    "mod": _guess_module(first_api),
                    "api": first_api,
                    "type": _guess_type(first_api),
                    "synthetic_helper_thread": True,
                    "helper_thread_source_tid": int(template.get("source_tid", -1)),
                    "helper_thread_role": "start",
                }
            )
            if end_ts > start_ts:
                synthetic_events.append(
                    {
                        "ts": end_ts,
                        "pid": pid,
                        "tid": max_tid,
                        "mod": _guess_module(dominant_api),
                        "api": dominant_api,
                        "type": _guess_type(dominant_api),
                        "synthetic_helper_thread": True,
                        "helper_thread_source_tid": int(template.get("source_tid", -1)),
                        "helper_thread_role": "end",
                    }
                )
        consumed_counts[first_api] = consumed + 1

    all_events = events + synthetic_events
    all_events.sort(key=lambda event: (int(event["ts"]), int(event["tid"]), str(event.get("api", ""))))
    if synthetic_events:
        trace_path.write_text(
            "\n".join(json.dumps(event, separators=(",", ":")) for event in all_events) + "\n",
            encoding="utf-8",
        )

    synthetic_timestamps = [
        int(event["ts"])
        for event in all_events
        if bool(event.get("synthetic_helper_thread")) and "ts" in event
    ]
    return {
        "injected_events": len(synthetic_events),
        "synthetic_min_ts": min(synthetic_timestamps) if synthetic_timestamps else None,
        "synthetic_max_ts": max(synthetic_timestamps) if synthetic_timestamps else None,
    }


def augment_trace_directory(
    trace_dir: Path,
    *,
    summary_dir: Path,
    target_ranks: set[int] | None = None,
    summary_json: Path | None = None,
) -> dict[str, object]:
    trace_dir = trace_dir.resolve()
    summary_dir = summary_dir.resolve()
    manifest = load_capture_manifest(trace_dir) or {}
    raw_step_windows = manifest.get("step_windows", {})
    raw_fidelity_windows = manifest.get("fidelity_windows")
    if not isinstance(raw_fidelity_windows, dict):
        raw_fidelity_windows = {}
    step_windows = _load_step_windows(trace_dir)
    injected_by_rank: dict[int, int] = {}
    step_window_extensions_by_rank: dict[str, dict[str, int]] = {}

    for trace_path in list_rank_trace_files(trace_dir):
        rank = _trace_rank(trace_path)
        if target_ranks is not None and rank not in target_ranks:
            continue
        summary_path = summary_dir / f"rank_{rank}.json"
        if not summary_path.exists():
            continue
        augment_summary = augment_trace_file(
            trace_path,
            summary_path=summary_path,
            step_window=step_windows.get(rank),
        )
        injected_by_rank[rank] = int(augment_summary["injected_events"] or 0)
        synthetic_min_ts = augment_summary.get("synthetic_min_ts")
        synthetic_max_ts = augment_summary.get("synthetic_max_ts")
        if rank not in step_windows:
            continue
        rank_key = str(rank)
        raw_window = raw_step_windows.get(rank_key)
        fidelity_window = None
        if isinstance(raw_fidelity_windows, dict):
            fidelity_window = raw_fidelity_windows.get(rank_key)
        if not isinstance(fidelity_window, dict):
            fidelity_window = raw_window if isinstance(raw_window, dict) else None
        if not isinstance(fidelity_window, dict):
            continue
        current_start = int(fidelity_window.get("start_ts", step_windows[rank][0]))
        current_end = int(fidelity_window.get("end_ts", step_windows[rank][1]))
        updated_start = current_start
        updated_end = current_end
        if synthetic_min_ts is not None:
            updated_start = min(updated_start, int(synthetic_min_ts))
        if synthetic_max_ts is not None:
            updated_end = max(updated_end, int(synthetic_max_ts))
        if updated_start != current_start or updated_end != current_end:
            if not isinstance(manifest.get("fidelity_windows"), dict):
                manifest["fidelity_windows"] = raw_fidelity_windows
            canonical_source = str(fidelity_window.get("source") or "manifest")
            if canonical_source in {"markers", "marker_sidecar"}:
                canonical_source = "trace_markers"
            raw_fidelity_windows[rank_key] = {
                **fidelity_window,
                "start_ts": updated_start,
                "end_ts": updated_end,
                "source": canonical_source,
                "is_paper_valid_step_window": False,
                "diagnostic_extension": "helper_tail",
                "diagnostic_only": True,
            }
            step_window_extensions_by_rank[rank_key] = {
                "start_ts": updated_start,
                "end_ts": updated_end,
            }

    payload = _helper_thread_augmentation_payload(
        expected=True,
        status=HELPER_THREAD_AUGMENTATION_STATUS_COMPLETED,
        summary_dir=summary_dir,
        injected_by_rank=injected_by_rank,
        step_window_extensions_by_rank=step_window_extensions_by_rank,
    )
    manifest["helper_thread_augmentation"] = payload
    _write_manifest(trace_dir, manifest)

    if summary_json is not None:
        summary_json.write_text(
            json.dumps(
                {
                    "trace_dir": str(trace_dir),
                    "helper_thread_augmentation": payload,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Augment emulated traces with helper-thread start/end events derived from host timing summaries.")
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--rank", action="append", type=int, dest="ranks", default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    augment_trace_directory(
        args.trace_dir,
        summary_dir=args.summary_dir,
        target_ranks=set(args.ranks) if args.ranks else None,
        summary_json=args.summary_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
