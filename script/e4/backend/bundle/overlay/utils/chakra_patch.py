#!/usr/bin/env python3
"""Materialize Chakra ET workload candidates from a compact patch file."""

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PATCH_SCHEMA_VERSION = "chakra-patch-v1"
MANIFEST_SCHEMA_VERSION = "chakra-patch-manifest-v1"
SUPPORTED_SET_FIELDS = frozenset(("name", "duration_micros", "attrs"))


class ChakraPatchError(Exception):
    """Raised when a Chakra patch cannot be applied."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a complete Chakra ET candidate prefix from a baseline "
            "prefix and a small JSON patch."
        )
    )
    parser.add_argument("baseline_workload_prefix")
    parser.add_argument("candidate_workload_prefix")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--patch", required=True)
    parser.add_argument(
        "--ranks",
        required=True,
        help="rank count, comma-separated ranks, or ranges such as 0-3,8",
    )
    parser.add_argument(
        "--manifest",
        help=(
            "manifest output path; defaults to "
            "<candidate_workload_prefix>.manifest.json"
        ),
    )
    return parser.parse_args()


def parse_rank_spec(value: str) -> List[int]:
    value = value.strip()
    if not value:
        raise ChakraPatchError("--ranks must not be empty")
    if value.isdigit():
        count = int(value)
        if count <= 0:
            raise ChakraPatchError("--ranks count must be positive")
        return list(range(count))

    ranks = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            if not start_text.isdigit() or not end_text.isdigit():
                raise ChakraPatchError(f"invalid rank range {part!r}")
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ChakraPatchError(f"invalid descending rank range {part!r}")
            ranks.update(range(start, end + 1))
        else:
            if not part.isdigit():
                raise ChakraPatchError(f"invalid rank {part!r}")
            ranks.add(int(part))
    if not ranks:
        raise ChakraPatchError("--ranks did not name any ranks")
    return sorted(ranks)


def configure_pythonpath(repo_root: Path) -> None:
    chakra_root = repo_root / "extern" / "graph_frontend"
    proto_dir = chakra_root / "chakra" / "schema" / "protobuf"
    for path in (repo_root, chakra_root, proto_dir):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def ensure_chakra_bindings(repo_root: Path) -> None:
    configure_pythonpath(repo_root)
    try:
        spec = importlib.util.find_spec("chakra.schema.protobuf.et_def_pb2")
    except ModuleNotFoundError:
        spec = None
    if spec is not None:
        return
    generated = (
        repo_root
        / "extern"
        / "graph_frontend"
        / "chakra"
        / "schema"
        / "protobuf"
        / "et_def_pb2.py"
    )
    if generated.exists():
        return
    raise ChakraPatchError(
        "Chakra protobuf Python bindings were not found. Run "
        "./build/astra_analytical/build.sh first or set PYTHONPATH to include "
        "et_def_pb2.py and its protobuf dependencies."
    )


def load_json(path: Path, description: str) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as json_file:
            value = json.load(json_file)
    except FileNotFoundError as exc:
        raise ChakraPatchError(f"{description} {path}: file does not exist") from exc
    except json.JSONDecodeError as exc:
        raise ChakraPatchError(
            f"{description} {path}:{exc.lineno}:{exc.colno}: malformed JSON: "
            f"{exc.msg}"
        ) from exc
    except OSError as exc:
        raise ChakraPatchError(
            f"{description} {path}: unable to read: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ChakraPatchError(f"{description} {path}: JSON root must be an object")
    return value


def rank_et_path(prefix: Path, rank: int) -> Path:
    return prefix.with_name(f"{prefix.name}.{rank}.et")


def load_patch(path: Path) -> Dict[str, Any]:
    patch = load_json(path, "patch")
    if patch.get("schema_version") != PATCH_SCHEMA_VERSION:
        raise ChakraPatchError(
            f"patch {path}: expected schema_version {PATCH_SCHEMA_VERSION!r}"
        )
    patches = patch.get("patches")
    if not isinstance(patches, list):
        raise ChakraPatchError(f"patch {path}: missing patches list")
    for index, entry in enumerate(patches):
        if not isinstance(entry, dict):
            raise ChakraPatchError(f"patch {path}: patches[{index}] must be an object")
        rank = entry.get("rank")
        node_id = entry.get("node_id")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ChakraPatchError(
                f"patch {path}: patches[{index}].rank must be a non-negative int"
            )
        if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id < 0:
            raise ChakraPatchError(
                f"patch {path}: patches[{index}].node_id must be a non-negative int"
            )
        set_fields = entry.get("set", {})
        if not isinstance(set_fields, dict):
            raise ChakraPatchError(f"patch {path}: patches[{index}].set must be an object")
        unknown_set_fields = sorted(set(set_fields) - SUPPORTED_SET_FIELDS)
        if unknown_set_fields:
            raise ChakraPatchError(
                f"patch {path}: patches[{index}].set has unsupported fields: "
                f"{unknown_set_fields}"
            )
        if "attrs" in set_fields and not isinstance(set_fields["attrs"], dict):
            raise ChakraPatchError(
                f"patch {path}: patches[{index}].set.attrs must be an object"
            )
        name_suffix = entry.get("name_suffix", "")
        if not isinstance(name_suffix, str):
            raise ChakraPatchError(
                f"patch {path}: patches[{index}].name_suffix must be a string"
            )
    return patch


def patch_entries_by_rank(patch: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    by_rank: Dict[int, List[Dict[str, Any]]] = {}
    for entry in patch["patches"]:
        by_rank.setdefault(int(entry["rank"]), []).append(entry)
    return by_rank


def read_chakra_et(path: Path) -> Tuple[Any, List[Any]]:
    from chakra.schema.protobuf.et_def_pb2 import (  # pylint: disable=import-error
        GlobalMetadata,
        Node as ChakraNode,
    )
    from chakra.src.third_party.utils.protolib import (  # pylint: disable=import-error
        decodeMessage as decode_message,
        openFileRd as open_file_rd,
    )

    if not path.exists():
        raise ChakraPatchError(f"ET file does not exist: {path}")
    trace = None
    try:
        trace = open_file_rd(str(path))
        metadata = GlobalMetadata()
        if not decode_message(trace, metadata):
            raise ChakraPatchError(f"{path}: missing GlobalMetadata message")
        nodes = []
        while True:
            node = ChakraNode()
            if not decode_message(trace, node):
                break
            nodes.append(node)
        return metadata, nodes
    finally:
        if trace is not None:
            trace.close()


def write_chakra_et(path: Path, metadata: Any, nodes: Iterable[Any]) -> None:
    from chakra.src.third_party.utils.protolib import (  # pylint: disable=import-error
        encodeMessage as encode_message,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("wb") as et_file:
            encode_message(et_file, metadata)
            for node in nodes:
                encode_message(et_file, node)
    except OSError as exc:
        raise ChakraPatchError(f"{path}: unable to write ET file: {exc}") from exc


def _set_list_attr(attr: Any, field_name: str, values: List[Any]) -> None:
    list_value = getattr(attr, field_name)
    del list_value.values[:]
    list_value.values.extend(values)


def set_attr_value(attr: Any, value: Any) -> None:
    current = attr.WhichOneof("value")
    if current is not None:
        attr.ClearField(current)
    if isinstance(value, bool):
        attr.bool_val = value
    elif isinstance(value, int):
        attr.int64_val = value
    elif isinstance(value, float):
        attr.double_val = value
    elif isinstance(value, str):
        attr.string_val = value
    elif isinstance(value, list):
        if all(isinstance(item, bool) for item in value):
            _set_list_attr(attr, "bool_list", value)
        elif all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            _set_list_attr(attr, "int64_list", value)
        elif all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            _set_list_attr(attr, "double_list", [float(item) for item in value])
        elif all(isinstance(item, str) for item in value):
            _set_list_attr(attr, "string_list", value)
        else:
            raise ChakraPatchError(f"unsupported attribute list value: {value!r}")
    else:
        raise ChakraPatchError(f"unsupported attribute value: {value!r}")


def set_node_attr(node: Any, name: str, value: Any) -> None:
    if not name:
        raise ChakraPatchError("attribute name must not be empty")
    for attr in node.attr:
        if attr.name == name:
            set_attr_value(attr, value)
            return
    attr = node.attr.add()
    attr.name = name
    set_attr_value(attr, value)


def apply_patch_to_nodes(nodes: List[Any], entries: List[Dict[str, Any]]) -> List[Any]:
    patched_nodes = []
    by_id = {}
    for node in nodes:
        copied = node.__class__()
        copied.CopyFrom(node)
        patched_nodes.append(copied)
        by_id[int(copied.id)] = copied

    for entry in entries:
        node_id = int(entry["node_id"])
        if node_id not in by_id:
            raise ChakraPatchError(
                f"rank {entry['rank']} node {node_id}: node_id not found"
            )
        node = by_id[node_id]
        set_fields = entry.get("set", {})
        if "name" in set_fields:
            if not isinstance(set_fields["name"], str):
                raise ChakraPatchError(f"rank {entry['rank']} node {node_id}: set.name must be a string")
            node.name = set_fields["name"]
        if "duration_micros" in set_fields:
            duration = set_fields["duration_micros"]
            if isinstance(duration, bool) or not isinstance(duration, int) or duration < 0:
                raise ChakraPatchError(
                    f"rank {entry['rank']} node {node_id}: "
                    "set.duration_micros must be a non-negative int"
                )
            node.duration_micros = duration
        attrs = set_fields.get("attrs", {})
        if attrs:
            if not isinstance(attrs, dict):
                raise ChakraPatchError(
                    f"rank {entry['rank']} node {node_id}: set.attrs must be an object"
                )
            for name, value in attrs.items():
                set_node_attr(node, str(name), value)
        name_suffix = entry.get("name_suffix", "")
        if name_suffix:
            node.name = f"{node.name}{name_suffix}"
    return patched_nodes


def materialize_candidate(
    baseline_prefix: Path,
    candidate_prefix: Path,
    ranks: List[int],
    patch: Dict[str, Any],
) -> Dict[str, Any]:
    for rank in ranks:
        baseline_path = rank_et_path(baseline_prefix, rank)
        candidate_path = rank_et_path(candidate_prefix, rank)
        if baseline_path.resolve(strict=False) == candidate_path.resolve(
            strict=False
        ):
            raise ChakraPatchError(
                "candidate prefix must not refer to the same ET files as the "
                f"baseline prefix; rank {rank} would overwrite "
                f"{baseline_path}"
            )

    entries_by_rank = patch_entries_by_rank(patch)
    unknown_ranks = sorted(set(entries_by_rank) - set(ranks))
    if unknown_ranks:
        raise ChakraPatchError(
            f"patch references ranks outside --ranks: {unknown_ranks}"
        )

    manifest_entries = []
    for rank in ranks:
        baseline_path = rank_et_path(baseline_prefix, rank)
        candidate_path = rank_et_path(candidate_prefix, rank)
        rank_entries = entries_by_rank.get(rank, [])
        if rank_entries:
            metadata, nodes = read_chakra_et(baseline_path)
            patched_nodes = apply_patch_to_nodes(nodes, rank_entries)
            write_chakra_et(candidate_path, metadata, patched_nodes)
            action = "patched"
        else:
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if candidate_path.exists() or candidate_path.is_symlink():
                    candidate_path.unlink()
                os.link(baseline_path, candidate_path)
                action = "hardlinked"
            except OSError:
                shutil.copy2(baseline_path, candidate_path)
                action = "copied"
        manifest_entries.append(
            {
                "rank": rank,
                "action": action,
                "baseline_et": str(baseline_path),
                "candidate_et": str(candidate_path),
                "patch_count": len(rank_entries),
            }
        )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "patch_schema_version": patch.get("schema_version"),
        "baseline_workload_prefix": str(baseline_prefix),
        "candidate_workload_prefix": str(candidate_prefix),
        "ranks": ranks,
        "rank_files": manifest_entries,
    }


def write_json(value: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(value, output_file, indent=2)
            output_file.write("\n")
    except OSError as exc:
        raise ChakraPatchError(f"{output_path}: unable to write: {exc}") from exc


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    try:
        ensure_chakra_bindings(repo_root)
        patch = load_patch(Path(args.patch))
        manifest = materialize_candidate(
            Path(args.baseline_workload_prefix),
            Path(args.candidate_workload_prefix),
            parse_rank_spec(args.ranks),
            patch,
        )
        manifest_path = (
            Path(args.manifest)
            if args.manifest
            else Path(f"{args.candidate_workload_prefix}.manifest.json")
        )
        write_json(manifest, manifest_path)
    except ChakraPatchError as exc:
        print(f"chakra_patch.py: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
