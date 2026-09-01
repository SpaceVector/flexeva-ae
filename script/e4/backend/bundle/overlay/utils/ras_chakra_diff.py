#!/usr/bin/env python3
"""Build a RAS reuse plan directly from baseline/candidate Chakra ET files."""

import argparse
import collections
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set


ANCHOR_SCHEMA_VERSION = "ras-anchor-v1"
REPLAY_CACHE_SCHEMA_VERSION = "ras-replay-cache-v1"
WORKLOAD_INDEX_SCHEMA_VERSION = "ras-workload-index-v1"
CHANGED_EVENTS_SCHEMA_VERSION = "native-maya-changed-events-v1"
WORKLOAD_ISSUE_KIND = "chakra_node_issue"
CHAKRA_DATA_DEP_KIND = "ChakraDataDep"
CHAKRA_CTRL_DEP_KIND = "ChakraCtrlDep"
COMM_NODE_TYPES = {5, 6, 7}
COMM_COLL_NODE = 7
DEFAULT_INVOLVED_DIM = [True, True, True, True]
WORKLOAD_ATTRS = {
    "is_cpu_op",
    "num_ops",
    "tensor_size",
    "comm_type",
    "comm_size",
    "comm_src",
    "comm_dst",
    "comm_tag",
}
COLLECTIVE_PAYLOAD_FIELDS = (
    "collective_group_key",
    "collective_node_id",
    "collective_comm_type",
    "collective_comm_size",
    "collective_comm_priority",
    "collective_pg_name",
    "collective_involved_dim",
    "collective_group_ranks",
)
NON_SEMANTIC_PAYLOAD_FIELDS = frozenset(
    (
        "tick",
        "execution_time",
        "finish_reason",
        "cached_duration",
        "reuse_partition_id",
        "event_id",
        "message_id",
        "chunk_id",
        "dataset_id",
    )
)


class ChakraDiffError(Exception):
    """Raised when a Chakra ET diff cannot be built."""


@dataclass(frozen=True)
class ChakraNodeRecord:
    rank: int
    seq: int
    node: Any
    payload: Dict[str, Any]
    partition_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline/candidate Chakra ET workload prefixes and emit a "
            "workload-only ras-diff-v1 reuse plan."
        )
    )
    parser.add_argument("baseline_workload_prefix")
    parser.add_argument("candidate_workload_prefix")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument(
        "--ranks",
        required=True,
        help="rank count, comma-separated ranks, or ranges such as 0-3,8",
    )
    parser.add_argument(
        "--comm-group-configuration",
        default="empty",
        help="optional ASTRA-sim communicator group JSON file",
    )
    parser.add_argument("--baseline-replay-cache", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument(
        "--baseline-index",
        help=(
            "ras-workload-index-v1 JSON produced from the baseline ET. When "
            "used with --changed-events, refresh avoids scanning baseline and "
            "candidate ET files."
        ),
    )
    parser.add_argument(
        "--write-baseline-index",
        help="write a ras-workload-index-v1 JSON for reuse by later refreshes",
    )
    parser.add_argument(
        "--changed-events",
        help=(
            "native-maya-changed-events-v1 JSON containing the changed "
            "candidate events for local RAS refresh."
        ),
    )
    parser.add_argument(
        "--candidate-patch",
        help=(
            "chakra-patch-v1 JSON to apply over baseline ET files when "
            "building the candidate anchor, without reading materialized "
            "candidate rank ET files"
        ),
    )
    parser.add_argument(
        "--compact-reuse-plan",
        "--changed-only-reuse-plan",
        dest="compact_reuse_plan",
        action="store_true",
        help=(
            "omit materialized reusable partitions and let replay treat cache "
            "entries as reusable unless rerun, refresh, or removed"
        ),
    )
    return parser.parse_args()


def parse_rank_spec(value: str) -> List[int]:
    value = value.strip()
    if not value:
        raise ChakraDiffError("--ranks must not be empty")
    if value.isdigit():
        count = int(value)
        if count <= 0:
            raise ChakraDiffError("--ranks count must be positive")
        return list(range(count))

    ranks: Set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            if not start_text.isdigit() or not end_text.isdigit():
                raise ChakraDiffError(f"invalid rank range {part!r}")
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ChakraDiffError(f"invalid descending rank range {part!r}")
            ranks.update(range(start, end + 1))
        else:
            if not part.isdigit():
                raise ChakraDiffError(f"invalid rank {part!r}")
            ranks.add(int(part))
    if not ranks:
        raise ChakraDiffError("--ranks did not name any ranks")
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
    raise ChakraDiffError(
        "Chakra protobuf Python bindings were not found. Run "
        "./build/astra_analytical/build.sh first or set PYTHONPATH to include "
        "et_def_pb2.py and its protobuf dependencies."
    )


def load_json(path: Path, description: str) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as json_file:
            value = json.load(json_file)
    except FileNotFoundError as exc:
        raise ChakraDiffError(f"{description} {path}: file does not exist") from exc
    except json.JSONDecodeError as exc:
        raise ChakraDiffError(
            f"{description} {path}:{exc.lineno}:{exc.colno}: malformed JSON: "
            f"{exc.msg}"
        ) from exc
    except OSError as exc:
        raise ChakraDiffError(f"{description} {path}: unable to read: {exc}") from exc
    if not isinstance(value, dict):
        raise ChakraDiffError(f"{description} {path}: JSON root must be an object")
    return value


def load_replay_cache(path: Path) -> Dict[str, Any]:
    cache = load_json(path, "replay cache")
    if cache.get("schema_version") != REPLAY_CACHE_SCHEMA_VERSION:
        raise ChakraDiffError(
            f"replay cache {path}: expected schema_version "
            f"{REPLAY_CACHE_SCHEMA_VERSION!r}"
        )
    context = cache.get("context_fingerprint")
    if not isinstance(context, str) or not context:
        raise ChakraDiffError(
            f"replay cache {path}: missing non-empty context_fingerprint"
        )
    workload_partitions = cache.get("workload_partitions")
    if not isinstance(workload_partitions, dict):
        raise ChakraDiffError(f"replay cache {path}: missing workload_partitions")
    return cache


def load_comm_groups(path_text: str) -> Dict[str, List[int]]:
    if not path_text or "empty" in path_text:
        return {}
    path = Path(path_text)
    root = load_json(path, "comm group configuration")
    groups: Dict[str, List[int]] = {}
    for key, value in root.items():
        if not isinstance(value, list) or any(
            not isinstance(rank, int) or rank < 0 for rank in value
        ):
            raise ChakraDiffError(
                f"comm group configuration {path}: group {key!r} must be a "
                "list of non-negative integer ranks"
            )
        groups[str(key)] = list(value)
    return groups


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def insertion_order_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def semantic_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: semantic_payload(item)
            for key, item in value.items()
            if key not in NON_SEMANTIC_PAYLOAD_FIELDS
        }
    if isinstance(value, list):
        return [semantic_payload(item) for item in value]
    return value


def event_id(rank: int, node_id: int) -> str:
    return f"rank:{rank}/chakra:{node_id}"


def attr_value(attr: Any) -> Any:
    kind = attr.WhichOneof("value")
    if kind is None:
        return None
    value = getattr(attr, kind)
    if kind.endswith("_list"):
        return list(value.values)
    if kind == "bytes_val":
        return bytes(value).decode("utf-8", errors="replace")
    if kind == "bytes_list":
        return [bytes(item).decode("utf-8", errors="replace") for item in value.values]
    return value


def node_attrs(node: Any) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    for attr in node.attr:
        if not attr.name:
            continue
        attrs[attr.name] = attr_value(attr)
    return attrs


def int_attr(attrs: Dict[str, Any], name: str, default: int = 0) -> int:
    value = attrs.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ChakraDiffError(f"attribute {name!r} must be an integer")
    return int(value)


def string_attr(attrs: Dict[str, Any], name: str, default: str = "") -> str:
    value = attrs.get(name, default)
    if not isinstance(value, str):
        raise ChakraDiffError(f"attribute {name!r} must be a string")
    return value


def bool_list_attr(
    attrs: Dict[str, Any], name: str, default: List[bool]
) -> List[bool]:
    value = attrs.get(name, default)
    if (
        not isinstance(value, list)
        or any(not isinstance(item, bool) for item in value)
    ):
        raise ChakraDiffError(f"attribute {name!r} must be a bool list")
    return list(value)


def collective_group_ranks(
    attrs: Dict[str, Any], all_ranks: List[int], comm_groups: Dict[str, List[int]]
) -> List[int]:
    pg_name = string_attr(attrs, "pg_name", "")
    if pg_name == "" or pg_name == "0":
        return list(all_ranks)
    if pg_name not in comm_groups:
        raise ChakraDiffError(f"communicator group {pg_name!r} not found")
    return list(comm_groups[pg_name])


def collective_payload(
    node: Any,
    attrs: Dict[str, Any],
    all_ranks: List[int],
    comm_groups: Dict[str, List[int]],
) -> Dict[str, Any]:
    comm_type = int_attr(attrs, "comm_type")
    comm_size = int_attr(attrs, "comm_size")
    comm_priority = int_attr(attrs, "comm_priority", 0)
    pg_name = string_attr(attrs, "pg_name", "")
    involved_dim = bool_list_attr(attrs, "involved_dim", DEFAULT_INVOLVED_DIM)
    group_ranks = collective_group_ranks(attrs, all_ranks, comm_groups)
    key = {
        "node_id": int(node.id),
        "comm_type": comm_type,
        "comm_size": comm_size,
        "comm_priority": comm_priority,
        "pg_name": pg_name,
        "involved_dim": involved_dim,
        "group_ranks": group_ranks,
    }
    return {
        "collective_group_key": insertion_order_json(key),
        "collective_node_id": int(node.id),
        "collective_comm_type": comm_type,
        "collective_comm_size": comm_size,
        "collective_comm_priority": comm_priority,
        "collective_pg_name": pg_name,
        "collective_involved_dim": involved_dim,
        "collective_group_ranks": group_ranks,
    }


def workload_payload(
    rank: int,
    node: Any,
    all_ranks: List[int],
    comm_groups: Dict[str, List[int]],
) -> Dict[str, Any]:
    attrs = node_attrs(node)
    node_id = int(node.id)
    payload: Dict[str, Any] = {}
    if int(node.type) == COMM_COLL_NODE:
        payload.update(collective_payload(node, attrs, all_ranks, comm_groups))
    payload.update(
        {
            "rank": rank,
            "node_id": node_id,
            "node_name": str(node.name),
            "node_type": int(node.type),
            "event_id": event_id(rank, node_id),
            "runtime_micros": int(node.duration_micros),
        }
    )

    data_deps = [int(dep) for dep in node.data_deps]
    if data_deps:
        payload["data_deps"] = data_deps
        payload["data_dep_event_ids"] = [event_id(rank, dep) for dep in data_deps]
    ctrl_deps = [int(dep) for dep in node.ctrl_deps]
    if ctrl_deps:
        payload["ctrl_deps"] = ctrl_deps
        payload["ctrl_dep_event_ids"] = [event_id(rank, dep) for dep in ctrl_deps]

    for name in sorted(WORKLOAD_ATTRS):
        is_compute_attr = name in {"is_cpu_op", "num_ops", "tensor_size"}
        if name in attrs and (is_compute_attr or int(node.type) in COMM_NODE_TYPES):
            payload[name] = attrs[name]
    return payload


def read_rank_nodes(
    prefix: Path,
    rank: int,
    all_ranks: List[int],
    comm_groups: Dict[str, List[int]],
) -> List[ChakraNodeRecord]:
    from chakra.schema.protobuf.et_def_pb2 import (  # pylint: disable=import-error
        GlobalMetadata,
    )
    from chakra.schema.protobuf.et_def_pb2 import (  # pylint: disable=import-error
        Node as ChakraNode,
    )
    from chakra.src.third_party.utils.protolib import (  # pylint: disable=import-error
        decodeMessage as decode_message,
    )
    from chakra.src.third_party.utils.protolib import (  # pylint: disable=import-error
        openFileRd as open_file_rd,
    )

    path = prefix.with_name(f"{prefix.name}.{rank}.et")
    if not path.exists():
        raise ChakraDiffError(f"rank {rank} ET file does not exist: {path}")

    records: List[ChakraNodeRecord] = []
    trace = None
    try:
        trace = open_file_rd(str(path))
        metadata = GlobalMetadata()
        if not decode_message(trace, metadata):
            raise ChakraDiffError(f"{path}: missing GlobalMetadata message")
        seq = 0
        while True:
            node = ChakraNode()
            if not decode_message(trace, node):
                break
            seq += 1
            records.append(
                chakra_node_record(rank, seq, node, all_ranks, comm_groups)
            )
    finally:
        if trace is not None:
            trace.close()
    return records


def chakra_node_record(
    rank: int,
    seq: int,
    node: Any,
    all_ranks: List[int],
    comm_groups: Dict[str, List[int]],
) -> ChakraNodeRecord:
    partition_id = event_id(rank, int(node.id))
    return ChakraNodeRecord(
        rank=rank,
        seq=seq,
        node=node,
        payload=workload_payload(rank, node, all_ranks, comm_groups),
        partition_id=partition_id,
    )


def rank_node_records(
    rank: int,
    nodes: List[Any],
    all_ranks: List[int],
    comm_groups: Dict[str, List[int]],
) -> List[ChakraNodeRecord]:
    return [
        chakra_node_record(rank, seq, node, all_ranks, comm_groups)
        for seq, node in enumerate(nodes, start=1)
    ]


def build_partitions(records: List[ChakraNodeRecord]) -> List[Dict[str, Any]]:
    partitions = []
    for record in sorted(records, key=lambda item: (item.rank, int(item.node.id))):
        partitions.append(
            build_partition_from_payload(
                rank=record.rank,
                seq=record.seq,
                partition_id=record.partition_id,
                payload=record.payload,
            )
        )
    return partitions


def build_partition_from_payload(
    *,
    rank: int,
    seq: int,
    partition_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    lane = f"rank:{rank}"
    hash_input = [
        {
            "kind": WORKLOAD_ISSUE_KIND,
            "lane": lane,
            "payload": semantic_payload(payload),
        }
    ]
    return {
        "id": partition_id,
        "layer": "workload",
        "lane": lane,
        "kinds": [WORKLOAD_ISSUE_KIND],
        "event_seq": [seq],
        "event_ids": [partition_id],
        "events": [
            {
                "seq": seq,
                "kind": WORKLOAD_ISSUE_KIND,
                "lane": lane,
                "payload": payload,
            }
        ],
        "content_hash": stable_hash(hash_input),
    }


def build_explicit_dependency_edges(
    records: List[ChakraNodeRecord],
) -> List[Dict[str, Any]]:
    data_dep_edges: List[Dict[str, Any]] = []
    known_partitions = {record.partition_id for record in records}
    edge_no = 0
    for record in sorted(records, key=lambda item: (item.rank, item.seq)):
        for field, edge_kind in (
            ("data_dep_event_ids", CHAKRA_DATA_DEP_KIND),
            ("ctrl_dep_event_ids", CHAKRA_CTRL_DEP_KIND),
        ):
            for dep_event_id in record.payload.get(field, []):
                if dep_event_id not in known_partitions:
                    raise ChakraDiffError(
                        f"{record.partition_id}: dependency {dep_event_id} "
                        "does not exist in the workload"
                    )
                if dep_event_id == record.partition_id:
                    continue
                edge_no += 1
                data_dep_edges.append(
                    {
                        "id": f"data_dep:{edge_no}",
                        "kind": edge_kind,
                        "layer": "workload",
                        "lane": f"rank:{record.rank}",
                        "src": dep_event_id,
                        "dst": record.partition_id,
                        "src_ref": {
                            "layer": "workload",
                            "event_id": dep_event_id,
                        },
                        "dst_event_seq": record.seq,
                    }
                )
    return data_dep_edges


def build_summary(
    records: List[ChakraNodeRecord],
    partitions: List[Dict[str, Any]],
    data_dep_edges: List[Dict[str, Any]],
) -> Dict[str, Any]:
    kind_counts: DefaultDict[str, int] = collections.defaultdict(int)
    for _record in records:
        kind_counts[WORKLOAD_ISSUE_KIND] += 1
    return {
        "events": len(records),
        "trace_edges": 0,
        "partitions": len(partitions),
        "data_dep_edges": len(data_dep_edges),
        "al_edges": 0,
        "unresolved_edges": 0,
        "layers": {
            "workload": {
                "events": len(records),
                "partitions": len(partitions),
                "kinds": dict(sorted(kind_counts.items())),
            }
        },
        "trace_edge_kinds": {},
    }


def build_workload_anchor(
    prefix: Path,
    ranks: List[int],
    context_fingerprint: str,
    comm_groups: Dict[str, List[int]],
) -> Dict[str, Any]:
    records: List[ChakraNodeRecord] = []
    for rank in ranks:
        records.extend(read_rank_nodes(prefix, rank, ranks, comm_groups))
    partitions = build_partitions(records)
    data_dep_edges = build_explicit_dependency_edges(records)
    return {
        "schema_version": ANCHOR_SCHEMA_VERSION,
        "source_trace": str(prefix),
        "context_fingerprint": context_fingerprint,
        "summary": build_summary(records, partitions, data_dep_edges),
        "partitions": partitions,
        "data_dep_edges": data_dep_edges,
        "al_edges": [],
        "unresolved_edges": [],
    }


def _edge_sort_key(edge: Dict[str, Any]) -> tuple[str, str, str]:
    return (edge.get("src", ""), edge.get("dst", ""), edge.get("id", ""))


def build_workload_index(
    baseline_prefix: Path,
    ranks: List[int],
    context_fingerprint: str,
    comm_groups: Dict[str, List[int]],
) -> Dict[str, Any]:
    anchor = build_workload_anchor(
        baseline_prefix, ranks, context_fingerprint, comm_groups
    )
    partitions_by_id = {
        partition["id"]: partition for partition in anchor["partitions"]
    }
    outgoing_edges_by_src: DefaultDict[str, List[Dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    incoming_edges_by_dst: DefaultDict[str, List[Dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for edge in anchor["data_dep_edges"]:
        edge_ref = dict(edge)
        outgoing_edges_by_src[edge["src"]].append(edge_ref)
        incoming_edges_by_dst[edge["dst"]].append(edge_ref)

    for edge_list in outgoing_edges_by_src.values():
        edge_list.sort(key=_edge_sort_key)
    for edge_list in incoming_edges_by_dst.values():
        edge_list.sort(key=_edge_sort_key)

    return {
        "schema_version": WORKLOAD_INDEX_SCHEMA_VERSION,
        "source_trace": str(baseline_prefix),
        "context_fingerprint": context_fingerprint,
        "ranks": ranks,
        "summary": anchor["summary"],
        "partition_ids": sorted(partitions_by_id),
        "partitions_by_id": partitions_by_id,
        "data_dep_edges": anchor["data_dep_edges"],
        "outgoing_edges_by_src": dict(outgoing_edges_by_src),
        "incoming_edges_by_dst": dict(incoming_edges_by_dst),
    }


def load_workload_index(path: Path) -> Dict[str, Any]:
    index = load_json(path, "workload index")
    if index.get("schema_version") != WORKLOAD_INDEX_SCHEMA_VERSION:
        raise ChakraDiffError(
            f"workload index {path}: expected schema_version "
            f"{WORKLOAD_INDEX_SCHEMA_VERSION!r}"
        )
    for key in (
        "context_fingerprint",
        "partition_ids",
        "partitions_by_id",
        "outgoing_edges_by_src",
        "incoming_edges_by_dst",
    ):
        if key not in index:
            raise ChakraDiffError(f"workload index {path}: missing {key!r}")
    if not isinstance(index["context_fingerprint"], str):
        raise ChakraDiffError(
            f"workload index {path}: context_fingerprint must be a string"
        )
    if not isinstance(index["partition_ids"], list) or any(
        not isinstance(partition_id, str)
        for partition_id in index["partition_ids"]
    ):
        raise ChakraDiffError(
            f"workload index {path}: partition_ids must be a string list"
        )
    if not isinstance(index["partitions_by_id"], dict):
        raise ChakraDiffError(
            f"workload index {path}: partitions_by_id must be an object"
        )
    if set(index["partition_ids"]) != set(index["partitions_by_id"]):
        raise ChakraDiffError(
            f"workload index {path}: partition_ids do not match partitions_by_id"
        )
    return index


def load_changed_events(path: Path) -> Dict[str, Any]:
    root = load_json(path, "changed events")
    if root.get("schema_version") != CHANGED_EVENTS_SCHEMA_VERSION:
        raise ChakraDiffError(
            f"changed events {path}: expected schema_version "
            f"{CHANGED_EVENTS_SCHEMA_VERSION!r}"
        )
    events = root.get("events")
    if not isinstance(events, list):
        raise ChakraDiffError(f"changed events {path}: events must be a list")
    return root


def validate_workload_index_rank_coverage(
    workload_index: Dict[str, Any], ranks: List[int]
) -> None:
    indexed_ranks = workload_index.get("ranks")
    if not isinstance(indexed_ranks, list) or any(
        isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
        for rank in indexed_ranks
    ):
        raise ChakraDiffError(
            "baseline workload index is missing a valid rank coverage list"
        )
    selected_ranks = sorted(set(ranks))
    indexed_rank_set = sorted(set(indexed_ranks))
    if indexed_rank_set != selected_ranks:
        raise ChakraDiffError(
            "baseline workload index ranks do not match --ranks; rebuild the "
            "index for this rank set. "
            f"index={indexed_rank_set}, selected={selected_ranks}"
        )


def build_patched_workload_anchor(
    baseline_prefix: Path,
    ranks: List[int],
    context_fingerprint: str,
    comm_groups: Dict[str, List[int]],
    candidate_patch: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        import utils.chakra_patch as chakra_patch  # pylint: disable=import-error
    except ImportError as exc:
        raise ChakraDiffError(f"unable to import chakra_patch.py: {exc}") from exc

    entries_by_rank = chakra_patch.patch_entries_by_rank(candidate_patch)
    unknown_ranks = sorted(set(entries_by_rank) - set(ranks))
    if unknown_ranks:
        raise ChakraDiffError(
            f"candidate patch references ranks outside --ranks: {unknown_ranks}"
        )

    records: List[ChakraNodeRecord] = []
    for rank in ranks:
        path = chakra_patch.rank_et_path(baseline_prefix, rank)
        try:
            _metadata, nodes = chakra_patch.read_chakra_et(path)
            patched_nodes = chakra_patch.apply_patch_to_nodes(
                nodes, entries_by_rank.get(rank, [])
            )
        except chakra_patch.ChakraPatchError as exc:
            raise ChakraDiffError(str(exc)) from exc
        records.extend(rank_node_records(rank, patched_nodes, ranks, comm_groups))

    partitions = build_partitions(records)
    data_dep_edges = build_explicit_dependency_edges(records)
    return {
        "schema_version": ANCHOR_SCHEMA_VERSION,
        "source_trace": str(baseline_prefix),
        "context_fingerprint": context_fingerprint,
        "summary": build_summary(records, partitions, data_dep_edges),
        "partitions": partitions,
        "data_dep_edges": data_dep_edges,
        "al_edges": [],
        "unresolved_edges": [],
    }


def validate_reusable_partitions_are_cached(
    diff: Dict[str, Any], replay_cache: Dict[str, Any]
) -> None:
    workload_partitions = replay_cache["workload_partitions"]
    missing = [
        partition["id"]
        for partition in diff["reuse_plan"].get("reusable_partitions", [])
        if (
            isinstance(partition, dict)
            and partition.get("id") not in workload_partitions
        )
    ]
    if missing:
        raise ChakraDiffError(
            "ET diff marked partitions reusable that are absent from the "
            f"baseline replay cache: {', '.join(sorted(missing))}"
        )


def replay_cache_ranks(replay_cache: Dict[str, Any]) -> List[int]:
    ranks = set()
    for partition_id, partition in replay_cache["workload_partitions"].items():
        rank = partition.get("rank")
        if not isinstance(rank, int):
            raise ChakraDiffError(
                "baseline replay cache has workload partition "
                f"{partition_id} without an integer rank"
            )
        ranks.add(rank)
    return sorted(ranks)


def validate_compact_plan_rank_coverage(
    ranks: List[int], replay_cache: Dict[str, Any]
) -> None:
    cached_ranks = replay_cache_ranks(replay_cache)
    selected_ranks = sorted(set(ranks))
    if selected_ranks != cached_ranks:
        raise ChakraDiffError(
            "--compact-reuse-plan requires diff coverage for every rank in "
            "the replay cache; use a full-rank --ranks value or disable "
            "compact mode. "
            f"selected={selected_ranks}, cache={cached_ranks}"
        )


def _require_changed_event_int(
    event: Dict[str, Any], field: str, index: int
) -> int:
    value = event.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ChakraDiffError(
            f"changed events entry {index}: {field!r} must be a non-negative integer"
        )
    return int(value)


def _require_changed_event_str(
    event: Dict[str, Any], field: str, index: int
) -> str:
    value = event.get(field)
    if not isinstance(value, str) or not value:
        raise ChakraDiffError(
            f"changed events entry {index}: {field!r} must be a non-empty string"
        )
    return value


def _changed_event_payload(event: Dict[str, Any], index: int) -> Dict[str, Any]:
    rank = _require_changed_event_int(event, "rank", index)
    node_id = _require_changed_event_int(event, "node_id", index)
    node_name = _require_changed_event_str(event, "node_name", index)
    node_type = _require_changed_event_int(event, "node_type", index)
    duration = _require_changed_event_int(event, "duration_micros", index)
    attrs = event.get("attrs", {})
    if not isinstance(attrs, dict):
        raise ChakraDiffError(
            f"changed events entry {index}: 'attrs' must be an object"
        )
    data_deps = event.get("data_deps", [])
    if not isinstance(data_deps, list) or any(
        isinstance(dep, bool) or not isinstance(dep, int) or dep < 0
        for dep in data_deps
    ):
        raise ChakraDiffError(
            f"changed events entry {index}: 'data_deps' must be a non-negative integer list"
        )

    payload: Dict[str, Any] = {
        "rank": rank,
        "node_id": node_id,
        "node_name": node_name,
        "node_type": node_type,
        "event_id": event_id(rank, node_id),
        "runtime_micros": duration,
    }
    if data_deps:
        payload["data_deps"] = list(data_deps)
        payload["data_dep_event_ids"] = [
            event_id(rank, int(dep)) for dep in data_deps
        ]
    for name in sorted(WORKLOAD_ATTRS):
        if name in attrs:
            payload[name] = attrs[name]
    _copy_collective_payload_fields(event, payload, index)
    return payload


def _copy_collective_payload_fields(
    source: Dict[str, Any], payload: Dict[str, Any], index: int
) -> None:
    for name in COLLECTIVE_PAYLOAD_FIELDS:
        if name not in source:
            continue
        value = source[name]
        if name in {
            "collective_group_key",
            "collective_pg_name",
        }:
            if not isinstance(value, str):
                raise ChakraDiffError(
                    f"changed events entry {index}: {name!r} must be a string"
                )
            payload[name] = value
        elif name == "collective_involved_dim":
            if (
                not isinstance(value, list)
                or any(not isinstance(item, bool) for item in value)
            ):
                raise ChakraDiffError(
                    f"changed events entry {index}: {name!r} must be a bool list"
                )
            payload[name] = list(value)
        elif name == "collective_group_ranks":
            if (
                not isinstance(value, list)
                or any(
                    isinstance(item, bool) or not isinstance(item, int) or item < 0
                    for item in value
                )
            ):
                raise ChakraDiffError(
                    f"changed events entry {index}: {name!r} must be a "
                    "non-negative integer list"
                )
            payload[name] = list(value)
        else:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ChakraDiffError(
                    f"changed events entry {index}: {name!r} must be a "
                    "non-negative integer"
                )
            payload[name] = int(value)


def _inherit_baseline_collective_payload(
    baseline_payload: Dict[str, Any], candidate_payload: Dict[str, Any]
) -> None:
    if int(candidate_payload.get("node_type", -1)) != COMM_COLL_NODE:
        return
    for name in COLLECTIVE_PAYLOAD_FIELDS:
        if name not in candidate_payload and name in baseline_payload:
            candidate_payload[name] = baseline_payload[name]


def _validate_changed_event_fast_path_payload(
    event: Dict[str, Any],
    baseline_payload: Dict[str, Any],
    candidate_payload: Dict[str, Any],
    index: int,
) -> None:
    baseline_node_type = baseline_payload.get("node_type")
    candidate_node_type = candidate_payload.get("node_type")
    if (
        isinstance(baseline_node_type, bool)
        or not isinstance(baseline_node_type, int)
        or baseline_node_type != candidate_node_type
    ):
        raise ChakraDiffError(
            f"changed events entry {index}: node_type must match the "
            "baseline partition"
        )
    if baseline_node_type in COMM_NODE_TYPES:
        raise ChakraDiffError(
            f"changed events entry {index}: local refresh fast path does not "
            "support communication or collective node mutation; use full ET "
            "diff or a Chakra patch for communication changes"
        )
    attrs = event.get("attrs", {})
    if isinstance(attrs, dict):
        for field in ("comm_type", "comm_size", "comm_src", "comm_dst", "comm_tag"):
            if field in attrs:
                raise ChakraDiffError(
                    f"changed events entry {index}: local refresh fast path "
                    f"does not support communication attribute {field!r}"
                )
    for field in COLLECTIVE_PAYLOAD_FIELDS:
        if field in event:
            raise ChakraDiffError(
                f"changed events entry {index}: local refresh fast path does "
                f"not support collective field {field!r}"
            )


def _partition_seq(partition: Dict[str, Any]) -> int:
    event_seq = partition.get("event_seq")
    if not isinstance(event_seq, list) or len(event_seq) != 1:
        raise ChakraDiffError(
            f"partition {partition.get('id', '<unknown>')!r}: expected one event_seq"
        )
    seq = event_seq[0]
    if isinstance(seq, bool) or not isinstance(seq, int) or seq <= 0:
        raise ChakraDiffError(
            f"partition {partition.get('id', '<unknown>')!r}: invalid event_seq"
        )
    return int(seq)


def _partition_payload(partition: Dict[str, Any]) -> Dict[str, Any]:
    events = partition.get("events")
    if not isinstance(events, list) or len(events) != 1:
        raise ChakraDiffError(
            f"partition {partition.get('id', '<unknown>')!r}: expected one event"
        )
    payload = events[0].get("payload") if isinstance(events[0], dict) else None
    if not isinstance(payload, dict):
        raise ChakraDiffError(
            f"partition {partition.get('id', '<unknown>')!r}: missing event payload"
        )
    return payload


def _edge_signature(edge: Dict[str, Any]) -> tuple[str, str, str]:
    return (edge.get("kind", ""), edge.get("src", ""), edge.get("dst", ""))


def _local_dependency_edges(
    partition_id: str,
    rank: int,
    seq: int,
    payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    edge_no = 0
    for field, edge_kind in (
        ("data_dep_event_ids", CHAKRA_DATA_DEP_KIND),
        ("ctrl_dep_event_ids", CHAKRA_CTRL_DEP_KIND),
    ):
        for dep_event_id in payload.get(field, []):
            if dep_event_id == partition_id:
                continue
            edge_no += 1
            edges.append(
                {
                    "id": f"local_dep:{rank}:{seq}:{edge_no}",
                    "kind": edge_kind,
                    "layer": "workload",
                    "lane": f"rank:{rank}",
                    "src": dep_event_id,
                    "dst": partition_id,
                    "src_ref": {
                        "layer": "workload",
                        "event_id": dep_event_id,
                    },
                    "dst_event_seq": seq,
                }
            )
    return edges


def _local_changed_edges(
    baseline_edges: List[Dict[str, Any]],
    candidate_edges: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    baseline_by_signature = {
        _edge_signature(edge): edge for edge in baseline_edges
    }
    candidate_by_signature = {
        _edge_signature(edge): edge for edge in candidate_edges
    }
    baseline_keys = set(baseline_by_signature)
    candidate_keys = set(candidate_by_signature)
    added = [
        dict(candidate_by_signature[key])
        for key in sorted(candidate_keys - baseline_keys)
    ]
    removed = [
        dict(baseline_by_signature[key])
        for key in sorted(baseline_keys - candidate_keys)
    ]
    return {"added": added, "removed": removed, "modified": []}


def _partition_ref(partition: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: partition[key]
        for key in ("id", "layer", "lane", "kinds")
        if key in partition
    }


def _collective_group_key(partition: Dict[str, Any]) -> Optional[str]:
    try:
        payload = _partition_payload(partition)
    except ChakraDiffError:
        return None
    group_key = payload.get("collective_group_key")
    if not isinstance(group_key, str) or not group_key:
        return None
    return group_key


def _collective_members_by_group(
    candidate_partitions: Dict[str, Dict[str, Any]]
) -> Dict[str, List[str]]:
    members: DefaultDict[str, List[str]] = collections.defaultdict(list)
    for partition_id, partition in candidate_partitions.items():
        group_key = _collective_group_key(partition)
        if group_key is not None:
            members[group_key].append(partition_id)
    return {
        group_key: sorted(partition_ids)
        for group_key, partition_ids in sorted(members.items())
    }


def _group_aware_propagate(
    candidate_partitions: Dict[str, Dict[str, Any]],
    adjacency: Dict[str, List[tuple[str, Dict[str, Any]]]],
    seed_ids: Set[str],
) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    collective_members = _collective_members_by_group(candidate_partitions)
    queue = sorted(
        partition_id
        for partition_id in seed_ids
        if partition_id in candidate_partitions
    )
    visited: Set[str] = set(queue)
    affected_edges_by_type: DefaultDict[str, List[Dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    seen_edges: DefaultDict[str, Set[tuple[str, str, str]]] = (
        collections.defaultdict(set)
    )

    def enqueue_collective_members(partition_id: str) -> None:
        group_key = _collective_group_key(candidate_partitions[partition_id])
        if group_key is None:
            return
        for member_id in collective_members.get(group_key, []):
            if member_id not in visited:
                visited.add(member_id)
                queue.append(member_id)

    cursor = 0
    while cursor < len(queue):
        src = queue[cursor]
        cursor += 1

        enqueue_collective_members(src)

        for edge_type, edge in sorted(
            adjacency.get(src, []),
            key=lambda item: (item[0], _edge_sort_key(item[1])),
        ):
            signature = _edge_signature(edge)
            if signature not in seen_edges[edge_type]:
                seen_edges[edge_type].add(signature)
                affected_edges_by_type[edge_type].append(edge)
            dst = edge["dst"]
            if dst in candidate_partitions and dst not in visited:
                visited.add(dst)
                queue.append(dst)

    affected_by_layer: DefaultDict[str, List[Dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for partition_id in sorted(visited):
        partition = candidate_partitions[partition_id]
        layer = str(partition.get("layer", "unknown"))
        affected_by_layer[layer].append(_partition_ref(partition))

    return (
        {
            layer: affected_by_layer[layer]
            for layer in sorted(affected_by_layer)
        },
        {
            edge_type: sorted(
                affected_edges_by_type.get(edge_type, []), key=_edge_sort_key
            )
            for edge_type in ("data_dep", "al")
        },
    )


def _propagate_from_index(
    index: Dict[str, Any],
    candidate_partitions: Dict[str, Dict[str, Any]],
    seed_ids: Set[str],
    changed_edges: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    outgoing = index["outgoing_edges_by_src"]
    adjacency: Dict[str, List[tuple[str, Dict[str, Any]]]] = {
        src: [("data_dep", dict(edge)) for edge in edges]
        for src, edges in outgoing.items()
        if isinstance(edges, list)
    }
    for edge in changed_edges["data_dep"]["removed"]:
        src = edge["src"]
        signature = _edge_signature(edge)
        adjacency[src] = [
            existing
            for existing in adjacency.get(src, [])
            if _edge_signature(existing[1]) != signature
        ]
    for edge in changed_edges["data_dep"]["added"]:
        adjacency.setdefault(edge["src"], []).append(("data_dep", edge))

    return _group_aware_propagate(candidate_partitions, adjacency, seed_ids)


def build_chakra_diff(
    baseline_prefix: Path,
    candidate_prefix: Path,
    ranks: List[int],
    replay_cache: Dict[str, Any],
    comm_groups: Dict[str, List[int]],
    compact_reuse_plan: bool = False,
    candidate_patch: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import utils.ras_diff as ras_diff  # pylint: disable=import-error

    if compact_reuse_plan:
        validate_compact_plan_rank_coverage(ranks, replay_cache)

    context_fingerprint = replay_cache["context_fingerprint"]
    baseline_anchor = build_workload_anchor(
        baseline_prefix, ranks, context_fingerprint, comm_groups
    )
    if candidate_patch is None:
        candidate_anchor = build_workload_anchor(
            candidate_prefix, ranks, context_fingerprint, comm_groups
        )
        candidate_source = str(candidate_prefix)
        diff_source = "chakra-et"
    else:
        candidate_anchor = build_patched_workload_anchor(
            baseline_prefix,
            ranks,
            context_fingerprint,
            comm_groups,
            candidate_patch,
        )
        candidate_source = f"{baseline_prefix} + candidate patch"
        diff_source = "chakra-et-overlay-patch"
    diff = ras_diff.build_diff(
        str(baseline_prefix),
        candidate_source,
        baseline_anchor,
        candidate_anchor,
        compact_reuse_plan=compact_reuse_plan,
    )
    diff["chakra_workload_diff"] = {
        "source": diff_source,
        "ranks": ranks,
        "workload_only": True,
        "baseline_replay_cache_context_fingerprint": context_fingerprint,
        "candidate_patch": candidate_patch is not None,
    }
    validate_reusable_partitions_are_cached(diff, replay_cache)
    return diff


def build_chakra_diff_from_index(
    *,
    baseline_prefix: Path,
    candidate_prefix: Path,
    ranks: List[int],
    replay_cache: Dict[str, Any],
    workload_index: Dict[str, Any],
    changed_events: Dict[str, Any],
    compact_reuse_plan: bool = False,
) -> Dict[str, Any]:
    import utils.ras_diff as ras_diff  # pylint: disable=import-error

    if compact_reuse_plan:
        validate_compact_plan_rank_coverage(ranks, replay_cache)
    context_fingerprint = replay_cache["context_fingerprint"]
    if workload_index["context_fingerprint"] != context_fingerprint:
        raise ChakraDiffError(
            "baseline workload index context_fingerprint does not match "
            "the replay cache"
        )
    validate_workload_index_rank_coverage(workload_index, ranks)

    baseline_partitions = workload_index["partitions_by_id"]
    candidate_partitions = dict(baseline_partitions)
    modified_partitions: List[Dict[str, Any]] = []
    data_dep_changes = {"added": [], "removed": [], "modified": []}
    changed_ids: Set[str] = set()

    for index, event in enumerate(changed_events["events"]):
        if not isinstance(event, dict):
            raise ChakraDiffError(
                f"changed events entry {index}: event must be an object"
            )
        rank = _require_changed_event_int(event, "rank", index)
        if rank not in ranks:
            raise ChakraDiffError(
                f"changed events entry {index}: rank {rank} is outside --ranks"
            )
        node_id = _require_changed_event_int(event, "node_id", index)
        partition_id = event_id(rank, node_id)
        if partition_id not in baseline_partitions:
            raise ChakraDiffError(
                f"changed events entry {index}: baseline index has no "
                f"partition {partition_id!r}; local refresh only supports "
                "existing mutable-region events"
            )
        baseline_partition = baseline_partitions[partition_id]
        baseline_payload = _partition_payload(baseline_partition)
        baseline_duration = event.get("baseline_duration_micros")
        if baseline_duration is not None and (
            isinstance(baseline_duration, bool)
            or not isinstance(baseline_duration, int)
            or baseline_duration != baseline_payload.get("runtime_micros")
        ):
            raise ChakraDiffError(
                f"changed events entry {index}: baseline_duration_micros "
                "does not match the baseline index"
            )

        candidate_payload = _changed_event_payload(event, index)
        _validate_changed_event_fast_path_payload(
            event, baseline_payload, candidate_payload, index
        )
        _inherit_baseline_collective_payload(baseline_payload, candidate_payload)
        seq = _partition_seq(baseline_partition)
        candidate_partition = build_partition_from_payload(
            rank=rank,
            seq=seq,
            partition_id=partition_id,
            payload=candidate_payload,
        )
        candidate_partitions[partition_id] = candidate_partition
        if (
            candidate_partition["content_hash"]
            == baseline_partition["content_hash"]
        ):
            continue

        changed_ids.add(partition_id)
        modified_partitions.append(
            {
                "id": partition_id,
                "layer": "workload",
                "lane": f"rank:{rank}",
                "kinds": [WORKLOAD_ISSUE_KIND],
                "baseline_content_hash": baseline_partition["content_hash"],
                "candidate_content_hash": candidate_partition["content_hash"],
            }
        )

        baseline_edges = workload_index["incoming_edges_by_dst"].get(
            partition_id, []
        )
        candidate_edges = _local_dependency_edges(
            partition_id, rank, seq, candidate_payload
        )
        local_changes = _local_changed_edges(baseline_edges, candidate_edges)
        data_dep_changes["added"].extend(local_changes["added"])
        data_dep_changes["removed"].extend(local_changes["removed"])

    changed_partitions = {
        "added": [],
        "removed": [],
        "modified": sorted(modified_partitions, key=lambda item: item["id"]),
    }
    changed_edges = {
        "data_dep": {
            "added": sorted(data_dep_changes["added"], key=_edge_sort_key),
            "removed": sorted(data_dep_changes["removed"], key=_edge_sort_key),
            "modified": [],
        },
        "al": {"added": [], "removed": [], "modified": []},
    }
    seed_ids = set(changed_ids)
    for edge_changes in changed_edges.values():
        for change_type in ("added", "removed", "modified"):
            for edge in edge_changes[change_type]:
                candidate_edge = edge.get("candidate", edge)
                for endpoint in ("src", "dst"):
                    partition_id = candidate_edge[endpoint]
                    if partition_id in candidate_partitions:
                        seed_ids.add(partition_id)

    affected_partitions, affected_edges = _propagate_from_index(
        workload_index, candidate_partitions, seed_ids, changed_edges
    )
    reuse_plan = ras_diff.build_reuse_plan(
        baseline_partitions,
        candidate_partitions,
        changed_partitions,
        affected_partitions,
        changed_edges,
    )
    if compact_reuse_plan:
        ras_diff.apply_compact_reuse_plan_policy(reuse_plan)
    summary = ras_diff.build_summary(
        changed_partitions,
        changed_edges,
        affected_partitions,
        affected_edges,
        reuse_plan,
    )
    summary["context_fingerprint_mismatch"] = False

    diff = {
        "schema_version": "ras-diff-v1",
        "baseline_anchor": {
            "path": str(baseline_prefix),
            "schema_version": ANCHOR_SCHEMA_VERSION,
            "source_trace": workload_index.get("source_trace", str(baseline_prefix)),
            "context_fingerprint": context_fingerprint,
            "summary": workload_index.get("summary", {}),
        },
        "candidate_anchor": {
            "path": str(candidate_prefix),
            "schema_version": ANCHOR_SCHEMA_VERSION,
            "source_trace": str(candidate_prefix),
            "context_fingerprint": context_fingerprint,
        },
        "context_fingerprint_mismatch": False,
        "summary": summary,
        "changed_partitions": changed_partitions,
        "changed_edges": changed_edges,
        "affected_partitions": affected_partitions,
        "affected_edges": affected_edges,
        "reuse_plan": reuse_plan,
        "chakra_workload_diff": {
            "source": "baseline-index+changed-events",
            "ranks": ranks,
            "workload_only": True,
            "baseline_replay_cache_context_fingerprint": context_fingerprint,
            "baseline_index": True,
            "changed_events": len(changed_events["events"]),
        },
    }
    validate_reusable_partitions_are_cached(diff, replay_cache)
    return diff


def write_json(value: Dict[str, Any], output_path: Path) -> None:
    output_dir = output_path.parent
    if str(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(value, output_file, indent=2)
            output_file.write("\n")
    except OSError as exc:
        raise ChakraDiffError(f"{output_path}: unable to write output: {exc}") from exc


def print_summary(diff: Dict[str, Any]) -> None:
    summary = diff["summary"]
    reuse_counts = diff["reuse_plan"]["counts"]
    print("RAS Chakra ET diff summary")
    print(f"  changed workload partitions: {summary['total_changed_partitions']}")
    print(f"  changed dependency edges: {summary['total_changed_edges']}")
    print(
        "  reuse plan: "
        f"rerun={reuse_counts['rerun_partitions']}, "
        f"refresh={reuse_counts['refresh_partitions']}, "
        f"reusable={reuse_counts['reusable_partitions']}"
    )
    by_layer = reuse_counts.get("reusable_partitions_by_layer", {})
    print(
        "  reusable workload partitions: "
        f"{by_layer.get('workload', 0)}"
    )


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    try:
        ranks = parse_rank_spec(args.ranks)
        replay_cache = load_replay_cache(Path(args.baseline_replay_cache))
        comm_groups = load_comm_groups(args.comm_group_configuration)
        workload_index = None
        if args.baseline_index:
            workload_index = load_workload_index(Path(args.baseline_index))
        if args.write_baseline_index and workload_index is None:
            ensure_chakra_bindings(repo_root)
            workload_index = build_workload_index(
                Path(args.baseline_workload_prefix),
                ranks,
                replay_cache["context_fingerprint"],
                comm_groups,
            )
        if args.write_baseline_index:
            if workload_index is None:
                raise ChakraDiffError(
                    "--write-baseline-index needs either --baseline-index or "
                    "readable baseline ET files"
                )
            write_json(workload_index, Path(args.write_baseline_index))

        candidate_patch = None
        if args.candidate_patch:
            if args.changed_events:
                raise ChakraDiffError(
                    "--candidate-patch cannot be combined with --changed-events"
                )
            try:
                import utils.chakra_patch as chakra_patch  # pylint: disable=import-error

                candidate_patch = chakra_patch.load_patch(Path(args.candidate_patch))
            except ImportError as exc:
                raise ChakraDiffError(
                    f"unable to import chakra_patch.py: {exc}"
                ) from exc
            except chakra_patch.ChakraPatchError as exc:
                raise ChakraDiffError(str(exc)) from exc
        if args.changed_events:
            if workload_index is None:
                raise ChakraDiffError(
                    "--changed-events requires --baseline-index; build it once "
                    "with --write-baseline-index during RAS initialization"
                )
            changed_events = load_changed_events(Path(args.changed_events))
            diff = build_chakra_diff_from_index(
                baseline_prefix=Path(args.baseline_workload_prefix),
                candidate_prefix=Path(args.candidate_workload_prefix),
                ranks=ranks,
                replay_cache=replay_cache,
                workload_index=workload_index,
                changed_events=changed_events,
                compact_reuse_plan=args.compact_reuse_plan,
            )
        else:
            ensure_chakra_bindings(repo_root)
            diff = build_chakra_diff(
                Path(args.baseline_workload_prefix),
                Path(args.candidate_workload_prefix),
                ranks,
                replay_cache,
                comm_groups,
                compact_reuse_plan=args.compact_reuse_plan,
                candidate_patch=candidate_patch,
            )
        write_json(diff, Path(args.output))
    except ChakraDiffError as exc:
        print(f"ras_chakra_diff.py: error: {exc}", file=sys.stderr)
        return 1

    if args.summary:
        print_summary(diff)
    return 0


if __name__ == "__main__":
    sys.exit(main())
