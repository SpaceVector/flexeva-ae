"""Backfill real-trace communicator topology after all rank traces exist."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from .communicators import recover_communicator_topology_from_events
from .io import iter_rank_trace_events, list_rank_trace_files
from .schema import TraceSource


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


def backfill_real_communicators(trace_dir: Path) -> dict[str, int]:
    trace_dir = Path(trace_dir)
    manifest_path = trace_dir / "capture_manifest.json"
    manifest: dict[str, object]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8") or "{}")
    else:
        manifest = {}

    rank_files = list_rank_trace_files(trace_dir)
    recovery = recover_communicator_topology_from_events(
        event
        for rank_file in rank_files
        for event in iter_rank_trace_events(rank_file, source=TraceSource.REAL)
    )

    manifest["communicators"] = {
        str(comm_id): {"members": [int(member) for member in members]}
        for comm_id, members in sorted(recovery.memberships.items())
    }
    aliases_by_rank: dict[int, dict[str, str]] = {}
    for (rank, local_comm_id), canonical_comm_id in sorted(
        recovery.local_comm_aliases.items()
    ):
        aliases_by_rank.setdefault(int(rank), {})[str(local_comm_id)] = str(
            canonical_comm_id
        )
    manifest["communicator_aliases"] = {
        str(rank): dict(sorted(alias_map.items()))
        for rank, alias_map in sorted(aliases_by_rank.items())
    }
    _atomic_write_json(manifest_path, manifest)

    cache_dir = trace_dir / ".maya_lite_cache"
    if cache_dir.exists():
        for cache_file in cache_dir.glob("trace_bundle_*.pkl"):
            cache_file.unlink(missing_ok=True)

    return {
        "rank_file_count": len(rank_files),
        "communicator_count": len(recovery.memberships),
        "communicator_alias_count": len(recovery.local_comm_aliases),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill real-trace communicator topology into capture_manifest.json"
    )
    parser.add_argument("trace_dir", type=Path)
    args = parser.parse_args(argv)
    summary = backfill_real_communicators(args.trace_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
