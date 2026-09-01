"""Merge per-node capture manifests after node-local trace materialization."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

_HOST_TIMING_LINE_DISABLED_PAYLOAD: dict[str, object] = {
    "host_timing_paper_alignment_line": "disabled",
    "host_timing_line_family": "disabled",
    "host_timing_line_contract_version": "phase4_v1",
    "host_timing_profile_backed": False,
    "host_timing_paper_alignment_ready": False,
}


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    payload = json.loads(raw)
    return payload if isinstance(payload, dict) else {}


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp_name = handle.name
    os.replace(tmp_name, path)


def _merge_unique_ints(existing: object, incoming: object) -> list[int]:
    values: set[int] = set()
    for source in (existing, incoming):
        if not isinstance(source, (list, tuple)):
            continue
        for value in source:
            try:
                values.add(int(value))
            except (TypeError, ValueError):
                continue
    return sorted(values)


def _merge_dict_of_lists(
    target: dict[str, object],
    key: str,
    incoming: dict[str, object],
) -> None:
    merged = target.get(key)
    if not isinstance(merged, dict):
        merged = {}
    raw_incoming = incoming.get(key)
    if isinstance(raw_incoming, dict):
        for item_key, item_value in raw_incoming.items():
            if isinstance(item_value, (list, tuple)):
                existing = merged.get(str(item_key))
                merged[str(item_key)] = _merge_unique_ints(existing, item_value)
    target[key] = merged


def _merge_shallow_dict(
    target: dict[str, object],
    key: str,
    incoming: dict[str, object],
) -> None:
    merged = target.get(key)
    if not isinstance(merged, dict):
        merged = {}
    raw_incoming = incoming.get(key)
    if isinstance(raw_incoming, dict):
        for item_key, item_value in raw_incoming.items():
            merged[str(item_key)] = item_value
    target[key] = merged


def _merge_nested_shallow_dict(
    target: dict[str, object],
    key: str,
    incoming: dict[str, object],
) -> None:
    merged = target.get(key)
    if not isinstance(merged, dict):
        merged = {}
    raw_incoming = incoming.get(key)
    if isinstance(raw_incoming, dict):
        for outer_key, outer_value in raw_incoming.items():
            existing_inner = merged.get(str(outer_key))
            if not isinstance(existing_inner, dict):
                existing_inner = {}
            if isinstance(outer_value, dict):
                for inner_key, inner_value in outer_value.items():
                    existing_inner[str(inner_key)] = inner_value
            merged[str(outer_key)] = existing_inner
    target[key] = merged


def merge_capture_manifests(
    trace_dir: Path,
    *,
    partial_glob: str = "capture_manifest.*.json",
    output_name: str = "capture_manifest.json",
) -> dict[str, object]:
    trace_dir = Path(trace_dir)
    partials = sorted(trace_dir.glob(partial_glob))
    output_path = trace_dir / output_name
    if not partials and not output_path.exists():
        raise FileNotFoundError(
            f"no capture manifests found in {trace_dir} using {partial_glob!r}"
        )

    merged = _read_json(output_path)
    for partial in partials:
        payload = _read_json(partial)
        if not payload:
            continue
        if payload.get("original_world_size") is not None:
            merged["original_world_size"] = payload.get("original_world_size")
        merged["profiled_ranks"] = _merge_unique_ints(
            merged.get("profiled_ranks"),
            payload.get("profiled_ranks"),
        )
        _merge_dict_of_lists(merged, "profiled_rank_groups", payload)
        _merge_shallow_dict(merged, "rank_host_machines", payload)
        _merge_shallow_dict(merged, "rank_host_dispatch_queues", payload)
        _merge_shallow_dict(merged, "step_windows", payload)
        _merge_shallow_dict(merged, "fidelity_windows", payload)
        _merge_shallow_dict(merged, "communicators", payload)
        _merge_nested_shallow_dict(merged, "communicator_aliases", payload)

    for key, value in _HOST_TIMING_LINE_DISABLED_PAYLOAD.items():
        merged[key] = value
    merged["capture_manifest_materialized_from_partials"] = len(partials)
    _atomic_write_json(output_path, merged)
    return {
        "partial_manifest_count": len(partials),
        "profiled_rank_count": len(merged.get("profiled_ranks") or []),
        "rank_host_machine_count": len(merged.get("rank_host_machines") or {}),
        "step_window_count": len(merged.get("step_windows") or {}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge node-local capture_manifest.*.json files"
    )
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--partials-glob", default="capture_manifest.*.json")
    parser.add_argument("--output-name", default="capture_manifest.json")
    args = parser.parse_args(argv)
    summary = merge_capture_manifests(
        args.trace_dir,
        partial_glob=args.partials_glob,
        output_name=args.output_name,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
