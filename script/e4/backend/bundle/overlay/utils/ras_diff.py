#!/usr/bin/env python3
"""Compare two offline RAS anchor files and emit a refresh plan."""

import argparse
import collections
import json
import os
import sys
from typing import Any, DefaultDict, Dict, Iterable, List, Set, Tuple


SCHEMA_VERSION = "ras-diff-v1"
REQUIRED_TOP_LEVEL_KEYS = (
    "schema_version",
    "partitions",
    "data_dep_edges",
    "al_edges",
)
NON_SEMANTIC_EDGE_PAYLOAD_FIELDS = frozenset(
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
        "src_id",
        "dst_id",
    )
)


class DiffError(Exception):
    """Raised when anchors cannot be compared."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two RAS anchor JSON files and emit changed partitions plus "
            "downstream DataDep/AL propagation."
        )
    )
    parser.add_argument("baseline_anchor", help="baseline anchor JSON path")
    parser.add_argument("candidate_anchor", help="candidate anchor JSON path")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="output diff JSON path",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print change counts and affected partition counts by layer",
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


def load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as json_file:
            value = json.load(json_file)
    except FileNotFoundError as exc:
        raise DiffError(f"{path}: file does not exist") from exc
    except json.JSONDecodeError as exc:
        raise DiffError(
            f"{path}:{exc.lineno}:{exc.colno}: malformed JSON: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise DiffError(f"{path}: unable to read file: {exc}") from exc

    if not isinstance(value, dict):
        raise DiffError(f"{path}: anchor JSON must be an object")
    return value


def require_list(anchor: Dict[str, Any], path: str, key: str) -> List[Any]:
    value = anchor.get(key)
    if not isinstance(value, list):
        raise DiffError(f"{path}: top-level key '{key}' must be a list")
    return value


def require_str(
    value: Dict[str, Any], field: str, path: str, context: str
) -> str:
    field_value = value.get(field)
    if not isinstance(field_value, str) or not field_value:
        raise DiffError(f"{path}: {context} missing non-empty string field '{field}'")
    return field_value


def validate_anchor(path: str) -> Dict[str, Any]:
    anchor = load_json(path)
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in anchor:
            raise DiffError(f"{path}: missing required top-level key '{key}'")
    if not isinstance(anchor["schema_version"], str):
        raise DiffError(f"{path}: top-level key 'schema_version' must be a string")
    context_fingerprint = anchor.get("context_fingerprint")
    if context_fingerprint is not None and not isinstance(context_fingerprint, str):
        raise DiffError(
            f"{path}: top-level key 'context_fingerprint' must be a string or null"
        )

    partitions = require_list(anchor, path, "partitions")
    seen_partition_ids: Set[str] = set()
    for index, partition in enumerate(partitions):
        context = f"partitions[{index}]"
        if not isinstance(partition, dict):
            raise DiffError(f"{path}: {context} must be an object")
        partition_id = require_str(partition, "id", path, context)
        require_str(partition, "content_hash", path, context)
        if partition_id in seen_partition_ids:
            raise DiffError(f"{path}: duplicate partition id '{partition_id}'")
        seen_partition_ids.add(partition_id)

    for key in ("data_dep_edges", "al_edges"):
        edges = require_list(anchor, path, key)
        for index, edge in enumerate(edges):
            context = f"{key}[{index}]"
            if not isinstance(edge, dict):
                raise DiffError(f"{path}: {context} must be an object")
            require_str(edge, "src", path, context)
            require_str(edge, "dst", path, context)

    return anchor


def partition_index(anchor: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {partition["id"]: partition for partition in anchor["partitions"]}


def sorted_ids(values: Iterable[str]) -> List[str]:
    return sorted(values)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def partition_ref(partition: Dict[str, Any]) -> Dict[str, Any]:
    ref: Dict[str, Any] = {"id": partition["id"]}
    for key in ("layer", "lane", "kinds"):
        if key in partition:
            ref[key] = partition[key]
    return ref


def partition_payload(partition: Dict[str, Any]) -> Dict[str, Any]:
    events = partition.get("events")
    if not isinstance(events, list) or len(events) != 1:
        return {}
    event = events[0]
    if not isinstance(event, dict):
        return {}
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return {}
    return payload


def collective_group_key(partition: Dict[str, Any]) -> str:
    value = partition_payload(partition).get("collective_group_key")
    if not isinstance(value, str):
        return ""
    return value


def build_collective_members_by_group(
    candidate_partitions: Dict[str, Dict[str, Any]],
    baseline_partitions: Dict[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    members: DefaultDict[str, Set[str]] = collections.defaultdict(set)
    for partition_id, partition in candidate_partitions.items():
        group_key = collective_group_key(partition)
        if group_key:
            members[group_key].add(partition_id)
    for partition_id, partition in baseline_partitions.items():
        if partition_id not in candidate_partitions:
            continue
        group_key = collective_group_key(partition)
        if group_key:
            members[group_key].add(partition_id)
    return {
        group_key: sorted(partition_ids)
        for group_key, partition_ids in sorted(members.items())
    }


def partition_collective_group_keys(
    partition_id: str,
    candidate_partitions: Dict[str, Dict[str, Any]],
    baseline_partitions: Dict[str, Dict[str, Any]],
) -> List[str]:
    group_keys: Set[str] = set()
    candidate = candidate_partitions.get(partition_id)
    if candidate is not None:
        group_key = collective_group_key(candidate)
        if group_key:
            group_keys.add(group_key)
    baseline = baseline_partitions.get(partition_id)
    if baseline is not None:
        group_key = collective_group_key(baseline)
        if group_key:
            group_keys.add(group_key)
    return sorted(group_keys)


def changed_partition_ref(
    partition: Dict[str, Any], baseline_hash: str = "", candidate_hash: str = ""
) -> Dict[str, Any]:
    ref = partition_ref(partition)
    if baseline_hash:
        ref["baseline_content_hash"] = baseline_hash
    if candidate_hash:
        ref["candidate_content_hash"] = candidate_hash
    elif "content_hash" in partition:
        ref["content_hash"] = partition["content_hash"]
    return ref


def build_changed_partitions(
    baseline_partitions: Dict[str, Dict[str, Any]],
    candidate_partitions: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    baseline_ids = set(baseline_partitions)
    candidate_ids = set(candidate_partitions)

    added = [
        changed_partition_ref(candidate_partitions[partition_id])
        for partition_id in sorted_ids(candidate_ids - baseline_ids)
    ]
    removed = [
        changed_partition_ref(baseline_partitions[partition_id])
        for partition_id in sorted_ids(baseline_ids - candidate_ids)
    ]
    modified: List[Dict[str, Any]] = []
    for partition_id in sorted_ids(baseline_ids & candidate_ids):
        baseline_hash = baseline_partitions[partition_id]["content_hash"]
        candidate_hash = candidate_partitions[partition_id]["content_hash"]
        if baseline_hash != candidate_hash:
            modified.append(
                changed_partition_ref(
                    candidate_partitions[partition_id],
                    baseline_hash=baseline_hash,
                    candidate_hash=candidate_hash,
                )
            )

    return {"added": added, "removed": removed, "modified": modified}


def edge_sort_key(edge: Dict[str, Any]) -> Tuple[str, str, str]:
    return (edge.get("src", ""), edge.get("dst", ""), edge.get("id", ""))


def semantic_edge_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: semantic_edge_payload(item)
            for key, item in value.items()
            if key not in NON_SEMANTIC_EDGE_PAYLOAD_FIELDS
        }
    if isinstance(value, list):
        return [semantic_edge_payload(item) for item in value]
    return value


def edge_stable_key(edge: Dict[str, Any], edge_type: str) -> Tuple[Any, ...]:
    key = [
        "semantic",
        edge.get("kind", ""),
        edge.get("src", ""),
        edge.get("dst", ""),
    ]
    if edge_type != "al":
        for field in ("layer", "lane"):
            if field in edge:
                key.extend((field, edge[field]))
    return tuple(key)


def semantic_edge(edge: Dict[str, Any]) -> Dict[str, Any]:
    projected: Dict[str, Any] = {
        "kind": edge.get("kind", ""),
        "src": edge.get("src", ""),
        "dst": edge.get("dst", ""),
        "payload": semantic_edge_payload(edge.get("payload", {})),
    }
    for field in ("layer", "lane"):
        if field in edge:
            projected[field] = edge[field]
    return projected


def edge_ref(edge: Dict[str, Any]) -> Dict[str, Any]:
    ref: Dict[str, Any] = {
        "src": edge["src"],
        "dst": edge["dst"],
    }
    for key in (
        "id",
        "kind",
        "layer",
        "lane",
        "src_event_seq",
        "dst_event_seq",
        "edge_seq",
        "src_ref",
        "dst_ref",
    ):
        if key in edge:
            ref[key] = edge[key]
    return ref


def edge_change_ref(
    edge: Dict[str, Any], baseline_edge: Dict[str, Any] = None
) -> Dict[str, Any]:
    ref = edge_ref(edge)
    if baseline_edge is not None:
        ref["baseline"] = edge_ref(baseline_edge)
        ref["candidate"] = edge_ref(edge)
    return ref


def edge_index(
    edges: List[Dict[str, Any]], edge_type: str
) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    indexed: DefaultDict[Tuple[Any, ...], List[Dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for edge in edges:
        indexed[edge_stable_key(edge, edge_type)].append(edge)
    return {
        key: sorted(
            indexed[key],
            key=lambda edge: (canonical_json(semantic_edge(edge)), edge_sort_key(edge)),
        )
        for key in indexed
    }


def consume_matching_semantic_edges(
    baseline_edges: List[Dict[str, Any]],
    candidate_edges: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    baseline_remaining = list(baseline_edges)
    candidate_remaining: List[Dict[str, Any]] = []
    baseline_signatures = [
        canonical_json(semantic_edge(edge)) for edge in baseline_remaining
    ]

    for candidate_edge in candidate_edges:
        candidate_signature = canonical_json(semantic_edge(candidate_edge))
        try:
            match_index = baseline_signatures.index(candidate_signature)
        except ValueError:
            candidate_remaining.append(candidate_edge)
            continue
        del baseline_remaining[match_index]
        del baseline_signatures[match_index]

    return baseline_remaining, candidate_remaining


def build_changed_edges(
    baseline: Dict[str, Any], candidate: Dict[str, Any]
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    changed_edges: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for edge_type, edge_key in (
        ("data_dep", "data_dep_edges"),
        ("al", "al_edges"),
    ):
        baseline_edges = edge_index(baseline[edge_key], edge_type)
        candidate_edges = edge_index(candidate[edge_key], edge_type)
        baseline_keys = set(baseline_edges)
        candidate_keys = set(candidate_edges)

        added = [
            edge_ref(edge)
            for key in sorted(candidate_keys - baseline_keys)
            for edge in candidate_edges[key]
        ]
        removed = [
            edge_ref(edge)
            for key in sorted(baseline_keys - candidate_keys)
            for edge in baseline_edges[key]
        ]
        modified: List[Dict[str, Any]] = []
        for key in sorted(baseline_keys & candidate_keys):
            remaining_baseline, remaining_candidate = consume_matching_semantic_edges(
                baseline_edges[key], candidate_edges[key]
            )
            for baseline_edge, candidate_edge in zip(
                remaining_baseline, remaining_candidate
            ):
                modified.append(edge_change_ref(candidate_edge, baseline_edge))
            if len(remaining_candidate) > len(remaining_baseline):
                added.extend(
                    edge_ref(edge)
                    for edge in remaining_candidate[len(remaining_baseline) :]
                )
            elif len(remaining_baseline) > len(remaining_candidate):
                removed.extend(
                    edge_ref(edge)
                    for edge in remaining_baseline[len(remaining_candidate) :]
                )

        changed_edges[edge_type] = {
            "added": added,
            "removed": removed,
            "modified": modified,
        }
    return changed_edges


def build_adjacency(
    anchor: Dict[str, Any]
) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    adjacency: DefaultDict[str, List[Tuple[str, Dict[str, Any]]]] = (
        collections.defaultdict(list)
    )
    for edge_type, edge_key in (
        ("data_dep", "data_dep_edges"),
        ("al", "al_edges"),
    ):
        for edge in anchor[edge_key]:
            adjacency[edge["src"]].append((edge_type, edge))

    for src in list(adjacency):
        adjacency[src].sort(key=lambda item: (item[0], edge_sort_key(item[1])))
    return dict(adjacency)


def propagate(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    changed_partitions: Dict[str, List[Dict[str, Any]]],
    changed_edges: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    baseline_partitions = partition_index(baseline)
    candidate_partitions = partition_index(candidate)
    collective_members_by_group = build_collective_members_by_group(
        candidate_partitions, baseline_partitions
    )
    adjacency = build_adjacency(candidate)
    seed_ids = {
        partition["id"]
        for change_type in ("added", "removed", "modified")
        for partition in changed_partitions[change_type]
        if partition["id"] in candidate_partitions
    }

    def seed_collective_group_members(partition_id: str) -> None:
        for group_key in partition_collective_group_keys(
            partition_id, candidate_partitions, baseline_partitions
        ):
            seed_ids.update(collective_members_by_group.get(group_key, []))

    for change_type in ("added", "removed", "modified"):
        for partition in changed_partitions[change_type]:
            seed_collective_group_members(partition["id"])

    for edge_changes in changed_edges.values():
        for change_type in ("added", "modified"):
            for edge in edge_changes[change_type]:
                candidate_edge = edge.get("candidate", edge)
                for endpoint in ("src", "dst"):
                    partition_id = candidate_edge[endpoint]
                    if partition_id in candidate_partitions:
                        seed_ids.add(partition_id)
                    seed_collective_group_members(partition_id)
                baseline_edge = edge.get("baseline")
                if isinstance(baseline_edge, dict):
                    for endpoint in ("src", "dst"):
                        partition_id = baseline_edge[endpoint]
                        if partition_id in candidate_partitions:
                            seed_ids.add(partition_id)
                        seed_collective_group_members(partition_id)
        for edge in edge_changes["removed"]:
            for endpoint in ("src", "dst"):
                partition_id = edge[endpoint]
                if partition_id in candidate_partitions:
                    seed_ids.add(partition_id)
                seed_collective_group_members(partition_id)

    visited: Set[str] = set()
    queue: List[str] = []
    for partition_id in sorted_ids(seed_ids):
        visited.add(partition_id)
        queue.append(partition_id)

    affected_edge_keys: Dict[str, Set[Tuple[str, str, str]]] = {
        "data_dep": set(),
        "al": set(),
    }
    affected_edges_by_type: Dict[str, List[Dict[str, Any]]] = {
        "data_dep": [],
        "al": [],
    }

    cursor = 0
    while cursor < len(queue):
        src = queue[cursor]
        cursor += 1
        for group_key in partition_collective_group_keys(
            src, candidate_partitions, baseline_partitions
        ):
            for partition_id in collective_members_by_group.get(group_key, []):
                if partition_id not in visited:
                    visited.add(partition_id)
                    queue.append(partition_id)
        for edge_type, edge in adjacency.get(src, []):
            key = edge_sort_key(edge)
            if key not in affected_edge_keys[edge_type]:
                affected_edge_keys[edge_type].add(key)
                affected_edges_by_type[edge_type].append(edge_ref(edge))
            dst = edge["dst"]
            if dst in candidate_partitions and dst not in visited:
                visited.add(dst)
                queue.append(dst)

    affected_by_layer: DefaultDict[str, List[Dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for partition_id in sorted_ids(visited):
        partition = candidate_partitions[partition_id]
        layer = str(partition.get("layer", "unknown"))
        affected_by_layer[layer].append(partition_ref(partition))

    affected_partitions = {
        layer: affected_by_layer[layer] for layer in sorted(affected_by_layer)
    }
    affected_edges = {
        edge_type: sorted(edges, key=edge_sort_key)
        for edge_type, edges in affected_edges_by_type.items()
    }
    return affected_partitions, affected_edges


def flatten_partitions(
    partitions_by_layer: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for layer in sorted(partitions_by_layer):
        flattened.extend(partitions_by_layer[layer])
    return sorted(flattened, key=lambda partition: partition["id"])


def group_partition_refs_by_layer(
    partitions: Iterable[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    by_layer: DefaultDict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for partition in partitions:
        layer = str(partition.get("layer", "unknown"))
        by_layer[layer].append(partition_ref(partition))
    return {
        layer: sorted(by_layer[layer], key=lambda partition: partition["id"])
        for layer in sorted(by_layer)
    }


def count_partitions_by_layer(
    partitions_by_layer: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, int]:
    return {
        layer: len(partitions)
        for layer, partitions in sorted(partitions_by_layer.items())
    }


def edge_endpoint_ids(
    changed_edges: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> Dict[str, List[str]]:
    candidate_ids: Set[str] = set()
    removed_ids: Set[str] = set()
    for edge_changes in changed_edges.values():
        for change_type in ("added", "modified"):
            for edge in edge_changes[change_type]:
                candidate_edge = edge.get("candidate", edge)
                candidate_ids.add(candidate_edge["src"])
                candidate_ids.add(candidate_edge["dst"])
                baseline_edge = edge.get("baseline")
                if isinstance(baseline_edge, dict):
                    removed_ids.add(baseline_edge["src"])
                    removed_ids.add(baseline_edge["dst"])
        for edge in edge_changes["removed"]:
            removed_ids.add(edge["src"])
            removed_ids.add(edge["dst"])
    return {
        "candidate_changed_edge_endpoint_ids": sorted_ids(candidate_ids),
        "removed_edge_endpoint_ids": sorted_ids(removed_ids),
    }


def build_reuse_plan(
    baseline_partitions: Dict[str, Dict[str, Any]],
    candidate_partitions: Dict[str, Dict[str, Any]],
    changed_partitions: Dict[str, List[Dict[str, Any]]],
    affected_partitions: Dict[str, List[Dict[str, Any]]],
    changed_edges: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> Dict[str, Any]:
    rerun_ids = {
        partition["id"]
        for change_type in ("added", "modified")
        for partition in changed_partitions[change_type]
        if partition["id"] in candidate_partitions
    }
    removed_ids = {
        partition["id"]
        for partition in changed_partitions["removed"]
        if partition["id"] in baseline_partitions
    }
    affected_ids = {
        partition["id"] for partition in flatten_partitions(affected_partitions)
    }
    refresh_ids = affected_ids - rerun_ids
    reusable_ids = set(candidate_partitions) - affected_ids

    rerun_by_layer = group_partition_refs_by_layer(
        candidate_partitions[partition_id] for partition_id in rerun_ids
    )
    refresh_by_layer = group_partition_refs_by_layer(
        candidate_partitions[partition_id] for partition_id in refresh_ids
    )
    reusable_by_layer = group_partition_refs_by_layer(
        candidate_partitions[partition_id] for partition_id in reusable_ids
    )
    removed_by_layer = group_partition_refs_by_layer(
        baseline_partitions[partition_id] for partition_id in removed_ids
    )
    edge_endpoint_summary = edge_endpoint_ids(changed_edges)

    return {
        "candidate_partitions_total": len(candidate_partitions),
        "rerun_partitions": flatten_partitions(rerun_by_layer),
        "rerun_partitions_by_layer": rerun_by_layer,
        "refresh_partitions": flatten_partitions(refresh_by_layer),
        "refresh_partitions_by_layer": refresh_by_layer,
        "reusable_partitions": flatten_partitions(reusable_by_layer),
        "reusable_partitions_by_layer": reusable_by_layer,
        "removed_baseline_partitions": flatten_partitions(removed_by_layer),
        "removed_baseline_partitions_by_layer": removed_by_layer,
        "changed_edge_endpoint_ids": edge_endpoint_summary[
            "candidate_changed_edge_endpoint_ids"
        ],
        "removed_edge_endpoint_ids": edge_endpoint_summary[
            "removed_edge_endpoint_ids"
        ],
        "counts": {
            "rerun_partitions": len(rerun_ids),
            "refresh_partitions": len(refresh_ids),
            "reusable_partitions": len(reusable_ids),
            "removed_baseline_partitions": len(removed_ids),
            "rerun_partitions_by_layer": count_partitions_by_layer(rerun_by_layer),
            "refresh_partitions_by_layer": count_partitions_by_layer(
                refresh_by_layer
            ),
            "reusable_partitions_by_layer": count_partitions_by_layer(
                reusable_by_layer
            ),
            "removed_baseline_partitions_by_layer": count_partitions_by_layer(
                removed_by_layer
            ),
            "changed_edge_endpoint_ids": len(
                edge_endpoint_summary["candidate_changed_edge_endpoint_ids"]
            ),
            "removed_edge_endpoint_ids": len(
                edge_endpoint_summary["removed_edge_endpoint_ids"]
            ),
        },
    }


def all_edges(anchor: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "data_dep": sorted(
            (edge_ref(edge) for edge in anchor["data_dep_edges"]),
            key=edge_sort_key,
        ),
        "al": sorted(
            (edge_ref(edge) for edge in anchor["al_edges"]),
            key=edge_sort_key,
        ),
    }


def build_context_mismatch_reuse_plan(
    baseline_partitions: Dict[str, Dict[str, Any]],
    candidate_partitions: Dict[str, Dict[str, Any]],
    changed_partitions: Dict[str, List[Dict[str, Any]]],
    changed_edges: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> Dict[str, Any]:
    rerun_by_layer = group_partition_refs_by_layer(candidate_partitions.values())
    empty_by_layer: Dict[str, List[Dict[str, Any]]] = {}
    removed_ids = {
        partition["id"]
        for partition in changed_partitions["removed"]
        if partition["id"] in baseline_partitions
    }
    removed_by_layer = group_partition_refs_by_layer(
        baseline_partitions[partition_id] for partition_id in removed_ids
    )
    edge_endpoint_summary = edge_endpoint_ids(changed_edges)

    return {
        "candidate_partitions_total": len(candidate_partitions),
        "rerun_partitions": flatten_partitions(rerun_by_layer),
        "rerun_partitions_by_layer": rerun_by_layer,
        "refresh_partitions": [],
        "refresh_partitions_by_layer": empty_by_layer,
        "reusable_partitions": [],
        "reusable_partitions_by_layer": empty_by_layer,
        "removed_baseline_partitions": flatten_partitions(removed_by_layer),
        "removed_baseline_partitions_by_layer": removed_by_layer,
        "changed_edge_endpoint_ids": edge_endpoint_summary[
            "candidate_changed_edge_endpoint_ids"
        ],
        "removed_edge_endpoint_ids": edge_endpoint_summary[
            "removed_edge_endpoint_ids"
        ],
        "context_fingerprint_mismatch": True,
        "counts": {
            "rerun_partitions": len(candidate_partitions),
            "refresh_partitions": 0,
            "reusable_partitions": 0,
            "removed_baseline_partitions": len(removed_ids),
            "rerun_partitions_by_layer": count_partitions_by_layer(rerun_by_layer),
            "refresh_partitions_by_layer": {},
            "reusable_partitions_by_layer": {},
            "removed_baseline_partitions_by_layer": count_partitions_by_layer(
                removed_by_layer
            ),
            "changed_edge_endpoint_ids": len(
                edge_endpoint_summary["candidate_changed_edge_endpoint_ids"]
            ),
            "removed_edge_endpoint_ids": len(
                edge_endpoint_summary["removed_edge_endpoint_ids"]
            ),
        },
    }


def apply_compact_reuse_plan_policy(reuse_plan: Dict[str, Any]) -> None:
    reuse_plan["default_reuse_policy"] = "cache_minus_rerun_refresh_removed"
    reuse_plan["reusable_partitions"] = []
    reuse_plan["reusable_partitions_by_layer"] = {}


def anchor_ref(path: str, anchor: Dict[str, Any]) -> Dict[str, Any]:
    ref: Dict[str, Any] = {
        "path": path,
        "schema_version": anchor["schema_version"],
    }
    if "context_fingerprint" in anchor:
        ref["context_fingerprint"] = anchor["context_fingerprint"]
    if "source_trace" in anchor:
        ref["source_trace"] = anchor["source_trace"]
    summary = anchor.get("summary")
    if isinstance(summary, dict):
        ref["summary"] = {
            key: summary[key]
            for key in (
                "events",
                "trace_edges",
                "partitions",
                "data_dep_edges",
                "al_edges",
                "unresolved_edges",
            )
            if key in summary
        }
    return ref


def build_summary(
    changed_partitions: Dict[str, List[Dict[str, Any]]],
    changed_edges: Dict[str, Dict[str, List[Dict[str, Any]]]],
    affected_partitions: Dict[str, List[Dict[str, Any]]],
    affected_edges: Dict[str, List[Dict[str, Any]]],
    reuse_plan: Dict[str, Any],
) -> Dict[str, Any]:
    affected_by_layer = {
        layer: len(partitions)
        for layer, partitions in sorted(affected_partitions.items())
    }
    return {
        "changed_partitions": {
            change_type: len(changed_partitions[change_type])
            for change_type in ("added", "removed", "modified")
        },
        "total_changed_partitions": sum(
            len(changed_partitions[change_type])
            for change_type in ("added", "removed", "modified")
        ),
        "changed_edges": {
            edge_type: {
                change_type: len(changed_edges[edge_type][change_type])
                for change_type in ("added", "removed", "modified")
            }
            for edge_type in ("data_dep", "al")
        },
        "total_changed_edges": sum(
            len(changed_edges[edge_type][change_type])
            for edge_type in ("data_dep", "al")
            for change_type in ("added", "removed", "modified")
        ),
        "affected_partitions": sum(affected_by_layer.values()),
        "affected_partitions_by_layer": affected_by_layer,
        "affected_edges": {
            edge_type: len(affected_edges[edge_type])
            for edge_type in ("data_dep", "al")
        },
        "rerun_partitions": reuse_plan["counts"]["rerun_partitions"],
        "rerun_partitions_by_layer": reuse_plan["counts"][
            "rerun_partitions_by_layer"
        ],
        "refresh_partitions": reuse_plan["counts"]["refresh_partitions"],
        "refresh_partitions_by_layer": reuse_plan["counts"][
            "refresh_partitions_by_layer"
        ],
        "reusable_partitions": reuse_plan["counts"]["reusable_partitions"],
        "reusable_partitions_by_layer": reuse_plan["counts"][
            "reusable_partitions_by_layer"
        ],
    }


def build_diff(
    baseline_path: str,
    candidate_path: str,
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    compact_reuse_plan: bool = False,
) -> Dict[str, Any]:
    baseline_partitions = partition_index(baseline)
    candidate_partitions = partition_index(candidate)
    changed_partitions = build_changed_partitions(
        baseline_partitions, candidate_partitions
    )
    changed_edges = build_changed_edges(baseline, candidate)
    affected_partitions, affected_edges = propagate(
        baseline, candidate, changed_partitions, changed_edges
    )
    baseline_context = baseline.get("context_fingerprint")
    candidate_context = candidate.get("context_fingerprint")
    context_mismatch = baseline_context != candidate_context
    if context_mismatch:
        affected_partitions = group_partition_refs_by_layer(
            candidate_partitions.values()
        )
        affected_edges = all_edges(candidate)
        reuse_plan = build_context_mismatch_reuse_plan(
            baseline_partitions,
            candidate_partitions,
            changed_partitions,
            changed_edges,
        )
    else:
        reuse_plan = build_reuse_plan(
            baseline_partitions,
            candidate_partitions,
            changed_partitions,
            affected_partitions,
            changed_edges,
        )
    if compact_reuse_plan and not context_mismatch:
        apply_compact_reuse_plan_policy(reuse_plan)
    summary = build_summary(
        changed_partitions,
        changed_edges,
        affected_partitions,
        affected_edges,
        reuse_plan,
    )
    summary["context_fingerprint_mismatch"] = context_mismatch
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_anchor": anchor_ref(baseline_path, baseline),
        "candidate_anchor": anchor_ref(candidate_path, candidate),
        "context_fingerprint_mismatch": context_mismatch,
        "summary": summary,
        "changed_partitions": changed_partitions,
        "changed_edges": changed_edges,
        "affected_partitions": affected_partitions,
        "affected_edges": affected_edges,
        "reuse_plan": reuse_plan,
    }


def write_diff(diff: Dict[str, Any], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(diff, output_file, indent=2)
            output_file.write("\n")
    except OSError as exc:
        raise DiffError(f"{output_path}: unable to write output: {exc}") from exc


def print_summary(diff: Dict[str, Any]) -> None:
    summary = diff["summary"]
    changed = summary["changed_partitions"]
    changed_edges = summary["changed_edges"]
    affected_edges = summary["affected_edges"]
    print("RAS diff summary")
    print(
        "  changed partitions: "
        f"added={changed['added']}, "
        f"removed={changed['removed']}, "
        f"modified={changed['modified']}, "
        f"total={summary['total_changed_partitions']}"
    )
    print(
        "  changed edges: "
        "data_dep="
        f"added={changed_edges['data_dep']['added']}, "
        f"removed={changed_edges['data_dep']['removed']}, "
        f"modified={changed_edges['data_dep']['modified']}; "
        "al="
        f"added={changed_edges['al']['added']}, "
        f"removed={changed_edges['al']['removed']}, "
        f"modified={changed_edges['al']['modified']}, "
        f"total={summary['total_changed_edges']}"
    )
    print(
        "  affected edges: "
        f"data_dep={affected_edges['data_dep']}, al={affected_edges['al']}"
    )
    print("  affected partitions by layer:")
    if summary["affected_partitions_by_layer"]:
        for layer, count in summary["affected_partitions_by_layer"].items():
            print(f"    {layer}: {count}")
    else:
        print("    none: 0")
    print(
        "  reuse plan: "
        f"rerun={summary['rerun_partitions']}, "
        f"refresh={summary['refresh_partitions']}, "
        f"reusable={summary['reusable_partitions']}"
    )
    if summary.get("context_fingerprint_mismatch"):
        print("  context fingerprint mismatch: reuse disabled")
    print("  reusable partitions by layer:")
    if summary["reusable_partitions_by_layer"]:
        for layer, count in summary["reusable_partitions_by_layer"].items():
            print(f"    {layer}: {count}")
    else:
        print("    none: 0")


def main() -> int:
    args = parse_args()
    try:
        baseline = validate_anchor(args.baseline_anchor)
        candidate = validate_anchor(args.candidate_anchor)
        diff = build_diff(
            args.baseline_anchor,
            args.candidate_anchor,
            baseline,
            candidate,
            compact_reuse_plan=args.compact_reuse_plan,
        )
        write_diff(diff, args.output)
    except DiffError as exc:
        print(f"ras_diff.py: error: {exc}", file=sys.stderr)
        return 1

    if args.summary:
        print_summary(diff)
    return 0


if __name__ == "__main__":
    sys.exit(main())
