#!/usr/bin/env python3
"""Build a RAS replay cache from a full ASTRA-sim RAS JSONL trace."""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


SCHEMA_VERSION = "ras-replay-cache-v1"
CONTEXT_LAYER = "context"
CONTEXT_KIND = "run_context"
ISSUE_KIND = "chakra_node_issue"
FINISH_KIND = "chakra_node_finish"


class TraceError(Exception):
    """Raised when a trace cannot be converted to a replay cache."""


@dataclass
class NodeTiming:
    event_id: str
    issue_tick: Optional[int] = None
    finish_tick: Optional[int] = None
    node_id: Optional[int] = None
    rank: Optional[int] = None
    collective_group_key: Optional[str] = None
    collective_group_ranks: Optional[List[int]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a RAS replay cache from Chakra issue/finish events."
    )
    parser.add_argument("trace", help="input baseline RAS JSONL trace")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="output replay cache JSON path",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print completed and missing workload partition counts",
    )
    return parser.parse_args()


def require_payload(record: Dict[str, Any], path: str, line_no: int) -> Dict[str, Any]:
    payload = record.get("payload")
    if not isinstance(payload, dict):
        raise TraceError(f"{path}:{line_no}: missing or non-object field 'payload'")
    return payload


def require_int(
    payload: Dict[str, Any], field: str, path: str, line_no: int
) -> int:
    value = payload.get(field)
    if not isinstance(value, int):
        raise TraceError(
            f"{path}:{line_no}: missing or non-integer payload field '{field}'"
        )
    return value


def require_event_id(payload: Dict[str, Any], path: str, line_no: int) -> str:
    event_id = payload.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise TraceError(
            f"{path}:{line_no}: missing or non-empty string payload field 'event_id'"
        )
    return event_id


def update_context_fingerprint(
    current: Optional[str], payload: Dict[str, Any], path: str, line_no: int
) -> str:
    value = payload.get("context_fingerprint")
    if not isinstance(value, str) or not value:
        raise TraceError(
            f"{path}:{line_no}: context event missing non-empty string "
            "payload field 'context_fingerprint'"
        )
    if current is not None and current != value:
        raise TraceError(
            f"{path}:{line_no}: conflicting context_fingerprint values "
            f"{current!r} and {value!r}"
        )
    return value


def parse_workload_event(
    timings: Dict[str, NodeTiming],
    kind: str,
    payload: Dict[str, Any],
    path: str,
    line_no: int,
) -> None:
    event_id = require_event_id(payload, path, line_no)
    tick = require_int(payload, "tick", path, line_no)
    rank = require_int(payload, "rank", path, line_no)
    node_id = require_int(payload, "node_id", path, line_no)

    timing = timings.setdefault(event_id, NodeTiming(event_id=event_id))
    if timing.rank is None:
        timing.rank = rank
    elif timing.rank != rank:
        raise TraceError(
            f"{path}:{line_no}: event_id {event_id!r} has inconsistent rank"
        )
    if timing.node_id is None:
        timing.node_id = node_id
    elif timing.node_id != node_id:
        raise TraceError(
            f"{path}:{line_no}: event_id {event_id!r} has inconsistent node_id"
        )
    if "collective_group_key" in payload:
        collective_group_key = payload.get("collective_group_key")
        if not isinstance(collective_group_key, str) or not collective_group_key:
            raise TraceError(
                f"{path}:{line_no}: event_id {event_id!r} has invalid "
                "collective_group_key"
            )
        if timing.collective_group_key is None:
            timing.collective_group_key = collective_group_key
        elif timing.collective_group_key != collective_group_key:
            raise TraceError(
                f"{path}:{line_no}: event_id {event_id!r} has conflicting "
                "collective_group_key values"
            )
    if "collective_group_ranks" in payload:
        collective_group_ranks = payload.get("collective_group_ranks")
        if (
            not isinstance(collective_group_ranks, list)
            or not collective_group_ranks
            or any(
                not isinstance(rank, int) or rank < 0
                for rank in collective_group_ranks
            )
        ):
            raise TraceError(
                f"{path}:{line_no}: event_id {event_id!r} has invalid "
                "collective_group_ranks"
            )
        if timing.collective_group_ranks is None:
            timing.collective_group_ranks = list(collective_group_ranks)
        elif timing.collective_group_ranks != collective_group_ranks:
            raise TraceError(
                f"{path}:{line_no}: event_id {event_id!r} has conflicting "
                "collective_group_ranks values"
            )

    if kind == ISSUE_KIND:
        if timing.issue_tick is None:
            timing.issue_tick = tick
    elif kind == FINISH_KIND:
        timing.finish_tick = tick


def read_trace(path: str) -> Tuple[Dict[str, NodeTiming], Optional[str]]:
    timings: Dict[str, NodeTiming] = {}
    context_fingerprint: Optional[str] = None
    try:
        with open(path, "r", encoding="utf-8") as trace_file:
            for line_no, raw_line in enumerate(trace_file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TraceError(
                        f"{path}:{line_no}: malformed JSON: {exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise TraceError(f"{path}:{line_no}: JSON line is not an object")

                record_type = record.get("type")
                if record_type not in {"event", "edge"}:
                    raise TraceError(
                        f"{path}:{line_no}: unsupported record type {record_type!r}"
                    )
                if record_type != "event":
                    continue

                layer = record.get("layer")
                kind = record.get("kind")
                if not isinstance(layer, str):
                    raise TraceError(
                        f"{path}:{line_no}: missing or non-string field 'layer'"
                    )
                if not isinstance(kind, str):
                    raise TraceError(
                        f"{path}:{line_no}: missing or non-string field 'kind'"
                    )
                if layer == "workload" and kind in {ISSUE_KIND, FINISH_KIND}:
                    parse_workload_event(
                        timings,
                        kind,
                        require_payload(record, path, line_no),
                        path,
                        line_no,
                    )
                elif layer == CONTEXT_LAYER and kind == CONTEXT_KIND:
                    context_fingerprint = update_context_fingerprint(
                        context_fingerprint,
                        require_payload(record, path, line_no),
                        path,
                        line_no,
                    )
    except FileNotFoundError as exc:
        raise TraceError(f"{path}: file does not exist") from exc
    except OSError as exc:
        raise TraceError(f"{path}: unable to read file: {exc}") from exc
    return timings, context_fingerprint


def build_cache(trace_path: str) -> Dict[str, Any]:
    timings, context_fingerprint = read_trace(trace_path)
    if context_fingerprint is None:
        raise TraceError(
            f"{trace_path}: missing required non-empty context_fingerprint"
        )
    workload_partitions: Dict[str, Dict[str, Any]] = {}
    missing_finish = 0

    for event_id in sorted(timings):
        timing = timings[event_id]
        if timing.issue_tick is None:
            continue
        if timing.finish_tick is None:
            missing_finish += 1
            continue
        duration = timing.finish_tick - timing.issue_tick
        if duration < 0:
            raise TraceError(
                f"{trace_path}: event_id {event_id!r} has finish before issue"
            )
        if timing.node_id is None or timing.rank is None:
            raise TraceError(f"{trace_path}: event_id {event_id!r} is incomplete")
        partition: Dict[str, Any] = {
            "duration": duration,
            "issue_tick": timing.issue_tick,
            "finish_tick": timing.finish_tick,
            "node_id": timing.node_id,
            "rank": timing.rank,
        }
        if timing.collective_group_key is not None:
            partition["collective_group_key"] = timing.collective_group_key
        if timing.collective_group_ranks is not None:
            partition["collective_group_ranks"] = timing.collective_group_ranks
        workload_partitions[event_id] = partition

    return {
        "schema_version": SCHEMA_VERSION,
        "source_trace": trace_path,
        "context_fingerprint": context_fingerprint,
        "summary": {
            "workload_partitions": len(workload_partitions),
            "missing_finish": missing_finish,
        },
        "workload_partitions": workload_partitions,
    }


def write_cache(cache: Dict[str, Any], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(cache, output_file, indent=2)
            output_file.write("\n")
    except OSError as exc:
        raise TraceError(f"{output_path}: unable to write output: {exc}") from exc


def print_summary(summary: Dict[str, int]) -> None:
    print("RAS replay cache summary")
    print(f"  workload_partitions: {summary['workload_partitions']}")
    print(f"  missing_finish: {summary['missing_finish']}")


def main() -> int:
    args = parse_args()
    try:
        cache = build_cache(args.trace)
        write_cache(cache, args.output)
    except TraceError as exc:
        print(f"ras_replay_cache.py: error: {exc}", file=sys.stderr)
        return 1

    if args.summary:
        print_summary(cache["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
