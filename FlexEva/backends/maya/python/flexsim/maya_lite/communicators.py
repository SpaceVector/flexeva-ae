"""
Deterministic communicator identifiers for Maya-lite emulation.

These helpers let emulated process groups carry stable communicator IDs across
independent worker processes, while still allowing us to recover ordered global
memberships from trace-contained communicator lifecycle events when possible.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .schema import TraceEvent

_INLINE_PREFIX = "flexsim-members:"
_HASH_PREFIX = "flexsim-sha1:"
_REAL_INLINE_PREFIX = "flexsim-real-comm:"
_REAL_HASH_PREFIX = "flexsim-real-sha1:"
_MAX_INLINE_BYTES = 120
_COMM_INIT_APIS = {
    "ncclCommInitRank",
    "ncclCommInitRankConfig",
}
_COMM_SPLIT_API = "ncclCommSplit"
_P2P_APIS = {"ncclSend", "ncclRecv"}


@dataclass(frozen=True)
class _CommUsageRecord:
    global_rank: int
    local_comm_id: str
    host_machine_id: str | None
    ops: tuple[tuple[str, str | None, str | None, str | None, str | None], ...]


def build_emulated_communicator_id(ranks: Sequence[int]) -> str:
    normalized = tuple(int(rank) for rank in ranks)
    members = ",".join(str(rank) for rank in normalized)
    inline = f"{_INLINE_PREFIX}{members}"
    if len(inline.encode("utf-8")) <= _MAX_INLINE_BYTES:
        return inline
    digest = hashlib.sha1(members.encode("utf-8")).hexdigest()[:20]
    return f"{_HASH_PREFIX}{digest}"


def parse_emulated_communicator_id(comm_id: str | None) -> tuple[int, ...] | None:
    if not comm_id or not comm_id.startswith(_INLINE_PREFIX):
        return None
    payload = comm_id[len(_INLINE_PREFIX):]
    if not payload:
        return tuple()
    try:
        return tuple(int(rank) for rank in payload.split(",") if rank)
    except ValueError:
        return None


def build_real_communicator_id(
    members: Sequence[int],
    *,
    api: str,
    occurrence_index: int,
    group_index: int,
    nranks: int,
    world_size: int | None,
) -> str:
    normalized_members = tuple(int(member) for member in members)
    members_payload = ",".join(str(member) for member in normalized_members)
    world_size_payload = "" if world_size is None else str(int(world_size))
    payload = (
        f"{api}|occ={int(occurrence_index)}|group={int(group_index)}|"
        f"nranks={int(nranks)}|world={world_size_payload}|members={members_payload}"
    )
    inline = f"{_REAL_INLINE_PREFIX}{payload}"
    if len(inline.encode("utf-8")) <= _MAX_INLINE_BYTES:
        return inline
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]
    return f"{_REAL_HASH_PREFIX}{digest}"


@dataclass(frozen=True)
class CommunicatorRecovery:
    memberships: dict[str, tuple[int, ...]] = field(default_factory=dict)
    local_comm_aliases: dict[tuple[int, str], str] = field(default_factory=dict)


@dataclass(frozen=True)
class _InitRecord:
    global_rank: int
    local_rank: int
    nranks: int
    world_size: int | None
    api: str
    occurrence_index: int
    local_comm_id: str
    ts: int


@dataclass(frozen=True)
class _SplitRecord:
    global_rank: int
    parent_comm_id: str
    new_comm_id: str
    color: int
    key: int
    occurrence_index: int
    world_size: int | None
    ts: int


def _direct_comm_id_recovery_from_init_records(
    records: Sequence[_InitRecord],
) -> CommunicatorRecovery:
    members_by_comm: dict[str, dict[int, int]] = defaultdict(dict)
    expected_size_by_comm: dict[str, int] = {}
    aliases: dict[tuple[int, str], str] = {}

    for record in records:
        comm_id = record.local_comm_id
        # Raw real-wrapper communicator handles are process-local numeric
        # addresses. They are not safe to merge across ranks by comm_id alone.
        if comm_id.isdigit():
            continue
        members_by_comm[comm_id][record.local_rank] = int(record.global_rank)
        expected_size_by_comm[comm_id] = record.nranks

    resolved: dict[str, tuple[int, ...]] = {}
    for comm_id, members_by_local_rank in members_by_comm.items():
        expected_size = expected_size_by_comm.get(comm_id)
        if expected_size is None:
            continue
        if len(members_by_local_rank) != expected_size:
            continue
        if set(members_by_local_rank) != set(range(expected_size)):
            continue
        members = tuple(
            int(members_by_local_rank[local_rank])
            for local_rank in range(expected_size)
        )
        resolved[comm_id] = members
        for local_rank, global_rank in members_by_local_rank.items():
            aliases[(int(global_rank), comm_id)] = comm_id
    return CommunicatorRecovery(memberships=resolved, local_comm_aliases=aliases)


def _synthetic_split_recovery_from_split_records(
    records: Sequence[_SplitRecord],
    *,
    existing_aliases: dict[tuple[int, str], str],
    existing_memberships: dict[str, tuple[int, ...]],
) -> CommunicatorRecovery:
    grouped_records: dict[tuple[object, ...], list[_SplitRecord]] = defaultdict(list)
    for record in records:
        new_key = (record.global_rank, record.new_comm_id)
        if new_key in existing_aliases:
            continue
        parent_key = existing_aliases.get(
            (record.global_rank, record.parent_comm_id),
            record.parent_comm_id,
        )
        parent_members = existing_memberships.get(parent_key)
        if (
            parent_members is None
            and record.parent_comm_id.isdigit()
            and record.world_size is not None
        ):
            # Real NCCL handles are process-local numeric addresses. When the
            # parent world communicator has not yet been recovered, grouping by
            # those raw handles collapses CommSplit groups to singleton aliases.
            # CommSplit color/key is the trace-visible topology signal within
            # the same world, so use it as a conservative unresolved-parent
            # fallback.
            group_key = (
                "unresolved_numeric_parent",
                record.world_size,
                record.occurrence_index,
                record.color,
            )
        else:
            group_key = ("parent", parent_key, record.occurrence_index, record.color)
        grouped_records[group_key].append(record)

    def _split_unresolved_numeric_parent_records(
        signature_records: Sequence[_SplitRecord],
    ) -> list[list[_SplitRecord]]:
        records_by_key: dict[int, list[_SplitRecord]] = defaultdict(list)
        for record in signature_records:
            records_by_key[record.key].append(record)
        if not records_by_key:
            return []
        sorted_keys = sorted(records_by_key)
        for key in sorted_keys:
            records_by_key[key].sort(
                key=lambda record: (record.ts, record.global_rank, record.new_comm_id)
            )
        key_counts = {len(records_by_key[key]) for key in sorted_keys}
        if len(key_counts) == 1:
            duplicate_count = key_counts.pop()
            if duplicate_count == 1:
                return [list(signature_records)]
            # Duplicate split keys mean the broad unresolved-parent fallback is
            # observing more than one parent namespace. Pair records by their
            # stable per-key occurrence instead of synthesizing one oversized
            # communicator.
            return [
                [records_by_key[key][index] for key in sorted_keys]
                for index in range(duplicate_count)
            ]
        # Uneven duplicate keys are ambiguous. Materialize singleton aliases so
        # later collective-sequence fallback cannot over-merge all participants.
        return [
            [record]
            for record in sorted(
                signature_records,
                key=lambda record: (
                    record.ts,
                    record.global_rank,
                    record.key,
                    record.new_comm_id,
                ),
            )
        ]

    resolved: dict[str, tuple[int, ...]] = {}
    aliases: dict[tuple[int, str], str] = {}
    group_index = 0

    for signature, signature_records in sorted(
        grouped_records.items(), key=lambda item: item[0]
    ):
        partitioned_records: list[list[_SplitRecord]]
        recovered_world_size: int | None
        occurrence_index: int
        if signature and signature[0] == "unresolved_numeric_parent":
            _, world_size, occurrence_index, _color = signature
            recovered_world_size = int(world_size) if world_size is not None else None
            partitioned_records = _split_unresolved_numeric_parent_records(signature_records)
        else:
            _, parent_key, occurrence_index, _color = signature
            parent_members = existing_memberships.get(parent_key, ())
            recovered_world_size = len(parent_members) if parent_members else None
            partitioned_records = [list(signature_records)]

        for partition_records in partitioned_records:
            if not partition_records:
                continue
            ordered_records = sorted(
                partition_records,
                key=lambda record: (record.key, record.global_rank, record.ts, record.new_comm_id),
            )
            members = tuple(int(record.global_rank) for record in ordered_records)
            synthetic_comm_id = build_real_communicator_id(
                members,
                api="ncclCommSplit",
                occurrence_index=occurrence_index,
                group_index=group_index,
                nranks=len(members),
                world_size=recovered_world_size,
            )
            group_index += 1
            resolved[synthetic_comm_id] = members
            for record in ordered_records:
                aliases[(record.global_rank, record.new_comm_id)] = synthetic_comm_id

    return CommunicatorRecovery(memberships=resolved, local_comm_aliases=aliases)

def _synthetic_real_comm_recovery_from_init_records(
    records: Sequence[_InitRecord],
    *,
    existing_aliases: dict[tuple[int, str], str],
) -> CommunicatorRecovery:
    grouped_records: dict[tuple[str, int, int, int | None], list[_InitRecord]] = defaultdict(list)
    for record in records:
        if (record.global_rank, record.local_comm_id) in existing_aliases:
            continue
        grouped_records[
            (record.api, record.occurrence_index, record.nranks, record.world_size)
        ].append(record)

    resolved: dict[str, tuple[int, ...]] = {}
    aliases: dict[tuple[int, str], str] = {}

    for signature, signature_records in grouped_records.items():
        api, occurrence_index, nranks, world_size = signature
        if nranks <= 0 or len(signature_records) < nranks:
            continue
        if len(signature_records) % nranks != 0:
            continue

        records_by_local_rank: dict[int, list[_InitRecord]] = defaultdict(list)
        invalid_local_rank = False
        for record in signature_records:
            if record.local_rank < 0 or record.local_rank >= nranks:
                invalid_local_rank = True
                break
            records_by_local_rank[record.local_rank].append(record)
        if invalid_local_rank:
            continue
        if set(records_by_local_rank) != set(range(nranks)):
            continue

        group_count = len(signature_records) // nranks
        if any(len(records_by_local_rank[local_rank]) != group_count for local_rank in range(nranks)):
            continue

        for local_rank in range(nranks):
            records_by_local_rank[local_rank].sort(
                key=lambda record: (record.global_rank, record.ts, record.local_comm_id)
            )

        for group_index in range(group_count):
            group_records = [
                records_by_local_rank[local_rank][group_index]
                for local_rank in range(nranks)
            ]
            members = tuple(int(record.global_rank) for record in group_records)
            synthetic_comm_id = build_real_communicator_id(
                members,
                api=api,
                occurrence_index=occurrence_index,
                group_index=group_index,
                nranks=nranks,
                world_size=world_size,
            )
            resolved[synthetic_comm_id] = members
            for record in group_records:
                aliases[(record.global_rank, record.local_comm_id)] = synthetic_comm_id

    return CommunicatorRecovery(memberships=resolved, local_comm_aliases=aliases)


def _p2p_component_recovery_from_usage_records(
    usage_records: Sequence[_CommUsageRecord],
    *,
    existing_aliases: dict[tuple[int, str], str],
    ranks_by_host_machine: dict[str, tuple[int, ...]],
) -> CommunicatorRecovery:
    usage_by_key = {
        (record.global_rank, record.local_comm_id): record for record in usage_records
    }
    adjacency_by_host: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    keys_by_host_rank: dict[str, dict[int, set[tuple[int, str]]]] = defaultdict(lambda: defaultdict(set))

    for record in usage_records:
        key = (record.global_rank, record.local_comm_id)
        if key in existing_aliases:
            continue
        if record.host_machine_id is None:
            continue
        if not any(api in _P2P_APIS for api, *_ in record.ops):
            continue
        host_members = ranks_by_host_machine.get(record.host_machine_id, ())
        if not host_members:
            continue
        keys_by_host_rank[record.host_machine_id][record.global_rank].add(key)
        for api, peer, *_ in record.ops:
            if api not in _P2P_APIS or peer in (None, ""):
                continue
            try:
                peer_index = int(str(peer))
            except (TypeError, ValueError):
                continue
            if peer_index < 0 or peer_index >= len(host_members):
                continue
            peer_rank = int(host_members[peer_index])
            adjacency_by_host[record.host_machine_id][record.global_rank].add(peer_rank)
            adjacency_by_host[record.host_machine_id][peer_rank].add(record.global_rank)

    resolved: dict[str, tuple[int, ...]] = {}
    aliases: dict[tuple[int, str], str] = {}

    for host_machine_id, adjacency in adjacency_by_host.items():
        visited: set[int] = set()
        component_index = 0
        for rank in sorted(adjacency):
            if rank in visited:
                continue
            stack = [rank]
            component: set[int] = set()
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                stack.extend(
                    neighbor
                    for neighbor in sorted(adjacency.get(current, ()))
                    if neighbor not in visited
                )
            if len(component) <= 1:
                continue
            members = tuple(sorted(component))
            canonical_comm_id = build_real_communicator_id(
                members,
                api="ncclP2PHostComponent",
                occurrence_index=0,
                group_index=component_index,
                nranks=len(members),
                world_size=len(ranks_by_host_machine.get(host_machine_id, ())),
            )
            component_index += 1
            resolved[canonical_comm_id] = members
            for member in members:
                for key in keys_by_host_rank[host_machine_id].get(member, ()):
                    usage_record = usage_by_key.get(key)
                    if usage_record is None:
                        continue
                    if any(api in _P2P_APIS for api, *_ in usage_record.ops):
                        aliases[key] = canonical_comm_id

    return CommunicatorRecovery(memberships=resolved, local_comm_aliases=aliases)


def _synthetic_p2p_pair_recovery_from_init_records(
    records: Sequence[_InitRecord],
    usage_records: Sequence[_CommUsageRecord],
    *,
    existing_aliases: dict[tuple[int, str], str],
) -> CommunicatorRecovery:
    p2p_comm_keys = {
        (record.global_rank, record.local_comm_id)
        for record in usage_records
        if any(api in _P2P_APIS for api, *_ in record.ops)
    }
    grouped_records: dict[tuple[str, int, int | None], dict[int, list[_InitRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        key = (record.global_rank, record.local_comm_id)
        if key in existing_aliases or key not in p2p_comm_keys:
            continue
        if record.nranks != 2 or record.local_rank not in {0, 1}:
            continue
        grouped_records[(record.api, record.nranks, record.world_size)][
            record.local_rank
        ].append(record)

    resolved: dict[str, tuple[int, ...]] = {}
    aliases: dict[tuple[int, str], str] = {}
    for (api, nranks, world_size), records_by_local_rank in grouped_records.items():
        local_zero_records = list(records_by_local_rank.get(0, ()))
        local_one_records = list(records_by_local_rank.get(1, ()))
        if not local_zero_records or not local_one_records:
            continue
        local_zero_records.sort(key=lambda record: (record.ts, record.global_rank, record.local_comm_id))
        local_one_records.sort(key=lambda record: (record.ts, record.global_rank, record.local_comm_id))
        for group_index, (rank_zero_record, rank_one_record) in enumerate(
            zip(local_zero_records, local_one_records)
        ):
            members = (int(rank_zero_record.global_rank), int(rank_one_record.global_rank))
            synthetic_comm_id = build_real_communicator_id(
                members,
                api=f"{api}P2PPair",
                occurrence_index=0,
                group_index=group_index,
                nranks=nranks,
                world_size=world_size,
            )
            resolved[synthetic_comm_id] = members
            aliases[(rank_zero_record.global_rank, rank_zero_record.local_comm_id)] = synthetic_comm_id
            aliases[(rank_one_record.global_rank, rank_one_record.local_comm_id)] = synthetic_comm_id

    return CommunicatorRecovery(memberships=resolved, local_comm_aliases=aliases)


def _collective_fingerprint_recovery_from_usage_records(
    usage_records: Sequence[_CommUsageRecord],
    *,
    existing_aliases: dict[tuple[int, str], str],
) -> CommunicatorRecovery:
    grouped_records: dict[
        tuple[tuple[str, str | None, str | None, str | None, str | None], ...],
        list[_CommUsageRecord],
    ] = defaultdict(list)
    for record in usage_records:
        key = (record.global_rank, record.local_comm_id)
        if key in existing_aliases:
            continue
        if not record.ops:
            continue
        if any(api in _P2P_APIS for api, *_ in record.ops):
            continue
        grouped_records[record.ops].append(record)

    resolved: dict[str, tuple[int, ...]] = {}
    aliases: dict[tuple[int, str], str] = {}
    group_index = 0

    for records in grouped_records.values():
        members = tuple(sorted({int(record.global_rank) for record in records}))
        if len(members) <= 1:
            continue
        canonical_comm_id = build_real_communicator_id(
            members,
            api="ncclCollectiveSequence",
            occurrence_index=0,
            group_index=group_index,
            nranks=len(members),
            world_size=None,
        )
        group_index += 1
        resolved[canonical_comm_id] = members
        for record in records:
            aliases[(record.global_rank, record.local_comm_id)] = canonical_comm_id

    return CommunicatorRecovery(memberships=resolved, local_comm_aliases=aliases)


def recover_communicator_topology_from_events(
    events: Iterable[TraceEvent],
) -> CommunicatorRecovery:
    init_occurrence_by_rank: dict[int, int] = defaultdict(int)
    split_occurrence_by_rank_parent: dict[tuple[int, str], int] = defaultdict(int)
    init_records: list[_InitRecord] = []
    split_records: list[_SplitRecord] = []
    ranks_by_host_machine: dict[str, set[int]] = defaultdict(set)
    usage_by_rank_comm: dict[
        tuple[int, str],
        dict[str, object],
    ] = {}

    for event in events:
        host_machine_id_raw = event.extras.get("host_machine_id")
        host_machine_id = str(host_machine_id_raw).strip() if host_machine_id_raw not in (None, "") else None
        if host_machine_id is not None:
            ranks_by_host_machine[host_machine_id].add(int(event.rank))
        local_comm_id = str(event.extras.get("comm_id", "")).strip()
        if local_comm_id:
            usage_record = usage_by_rank_comm.setdefault(
                (int(event.rank), local_comm_id),
                {
                    "global_rank": int(event.rank),
                    "local_comm_id": local_comm_id,
                    "host_machine_id": host_machine_id,
                    "ops": [],
                },
            )
            if usage_record.get("host_machine_id") is None and host_machine_id is not None:
                usage_record["host_machine_id"] = host_machine_id
            if event.op_type == "nccl_collective":
                usage_record["ops"].append(
                    (
                        str(event.api),
                        None if event.extras.get("peer") in (None, "") else str(event.extras.get("peer")),
                        None if event.extras.get("count") in (None, "") else str(event.extras.get("count")),
                        None if event.extras.get("datatype") in (None, "") else str(event.extras.get("datatype")),
                        None if event.extras.get("op") in (None, "") else str(event.extras.get("op")),
                    )
                )

        if event.api == _COMM_SPLIT_API:
            parent_comm_id = str(
                event.extras.get("parent_comm_id", local_comm_id)
            ).strip()
            new_comm_id = str(event.extras.get("new_comm_id", "")).strip()
            if not parent_comm_id or not new_comm_id:
                continue
            try:
                color = int(str(event.extras.get("color")))
                key = int(str(event.extras.get("key")))
            except (TypeError, ValueError):
                continue
            try:
                world_size = (
                    int(str(event.extras.get("world_size")))
                    if event.extras.get("world_size") not in (None, "")
                    else None
                )
            except (TypeError, ValueError):
                world_size = None
            # NCCL_SPLIT_NOCOLOR creates no new participating communicator.
            if color < 0:
                continue
            split_occurrence_key = (int(event.rank), parent_comm_id)
            occurrence_index = split_occurrence_by_rank_parent[split_occurrence_key]
            split_occurrence_by_rank_parent[split_occurrence_key] += 1
            split_records.append(
                _SplitRecord(
                    global_rank=int(event.rank),
                    parent_comm_id=parent_comm_id,
                    new_comm_id=new_comm_id,
                    color=color,
                    key=key,
                    occurrence_index=occurrence_index,
                    world_size=world_size,
                    ts=int(event.ts),
                )
            )

        if event.api not in _COMM_INIT_APIS:
            continue
        if not local_comm_id:
            continue
        local_rank_raw = event.extras.get("rank")
        nranks_raw = event.extras.get("nranks")
        world_size_raw = event.extras.get("world_size")
        try:
            local_rank = int(str(local_rank_raw))
            nranks = int(str(nranks_raw))
        except (TypeError, ValueError):
            continue
        if nranks <= 0 or local_rank < 0 or local_rank >= nranks:
            continue
        try:
            world_size = int(str(world_size_raw)) if world_size_raw not in (None, "") else None
        except (TypeError, ValueError):
            world_size = None
        occurrence_index = init_occurrence_by_rank[event.rank]
        init_occurrence_by_rank[event.rank] += 1
        init_records.append(
            _InitRecord(
                global_rank=int(event.rank),
                local_rank=local_rank,
                nranks=nranks,
                world_size=world_size,
                api=event.api,
                occurrence_index=occurrence_index,
                local_comm_id=local_comm_id,
                ts=int(event.ts),
            )
        )

    direct_recovery = _direct_comm_id_recovery_from_init_records(init_records)
    synthetic_recovery = _synthetic_real_comm_recovery_from_init_records(
        init_records,
        existing_aliases=direct_recovery.local_comm_aliases,
    )
    split_recovery = _synthetic_split_recovery_from_split_records(
        split_records,
        existing_aliases={
            **direct_recovery.local_comm_aliases,
            **synthetic_recovery.local_comm_aliases,
        },
        existing_memberships={
            **direct_recovery.memberships,
            **synthetic_recovery.memberships,
        },
    )
    usage_records = tuple(
        _CommUsageRecord(
            global_rank=int(payload["global_rank"]),
            local_comm_id=str(payload["local_comm_id"]),
            host_machine_id=(
                str(payload["host_machine_id"])
                if payload.get("host_machine_id") not in (None, "")
                else None
            ),
            ops=tuple(payload["ops"]),
        )
        for payload in usage_by_rank_comm.values()
    )
    ranks_by_host_machine_resolved = {
        host_machine_id: tuple(sorted(ranks))
        for host_machine_id, ranks in ranks_by_host_machine.items()
    }
    p2p_pair_recovery = _synthetic_p2p_pair_recovery_from_init_records(
        init_records,
        usage_records,
        existing_aliases={
            **direct_recovery.local_comm_aliases,
            **synthetic_recovery.local_comm_aliases,
            **split_recovery.local_comm_aliases,
        },
    )
    p2p_recovery = _p2p_component_recovery_from_usage_records(
        usage_records,
        existing_aliases={
            **direct_recovery.local_comm_aliases,
            **synthetic_recovery.local_comm_aliases,
            **split_recovery.local_comm_aliases,
            **p2p_pair_recovery.local_comm_aliases,
        },
        ranks_by_host_machine=ranks_by_host_machine_resolved,
    )
    collective_fingerprint_recovery = _collective_fingerprint_recovery_from_usage_records(
        usage_records,
        existing_aliases={
            **direct_recovery.local_comm_aliases,
            **synthetic_recovery.local_comm_aliases,
            **split_recovery.local_comm_aliases,
            **p2p_pair_recovery.local_comm_aliases,
            **p2p_recovery.local_comm_aliases,
        },
    )

    memberships = dict(direct_recovery.memberships)
    memberships.update(synthetic_recovery.memberships)
    memberships.update(split_recovery.memberships)
    memberships.update(p2p_pair_recovery.memberships)
    memberships.update(p2p_recovery.memberships)
    memberships.update(collective_fingerprint_recovery.memberships)
    local_comm_aliases = dict(direct_recovery.local_comm_aliases)
    local_comm_aliases.update(synthetic_recovery.local_comm_aliases)
    local_comm_aliases.update(split_recovery.local_comm_aliases)
    local_comm_aliases.update(p2p_pair_recovery.local_comm_aliases)
    local_comm_aliases.update(p2p_recovery.local_comm_aliases)
    local_comm_aliases.update(collective_fingerprint_recovery.local_comm_aliases)
    return CommunicatorRecovery(
        memberships=memberships,
        local_comm_aliases=local_comm_aliases,
    )


def infer_communicator_memberships_from_events(
    events: Iterable[TraceEvent],
) -> dict[str, tuple[int, ...]]:
    """Recover communicator memberships directly from trace lifecycle events.

    Maya's communicator reconstruction is trace-driven. For traces that contain
    communicator init lifecycle events, we can recover the ordered global member
    list without relying on a topology sidecar.
    """

    return recover_communicator_topology_from_events(events).memberships
