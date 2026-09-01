from flexsim.maya_lite.communicators import recover_communicator_topology_from_events
from flexsim.maya_lite.schema import TraceEvent, TraceSource


def _event(
    *,
    rank: int,
    ordinal: int,
    api: str,
    op_type: str,
    ts: int,
    extras: dict[str, object],
) -> TraceEvent:
    return TraceEvent(
        rank=rank,
        ordinal=ordinal,
        source=TraceSource.REAL,
        ts=ts,
        pid=rank,
        tid=rank,
        module="libnccl.so",
        api=api,
        op_type=op_type,
        extras=extras,
    )


def test_real_recovery_keeps_nranks_two_p2p_communicators_as_pairs():
    events = [
        _event(
            rank=0,
            ordinal=0,
            api="ncclCommInitRankConfig",
            op_type="other",
            ts=100,
            extras={"comm_id": "r0_pair_a", "rank": "0", "nranks": "2", "world_size": "2"},
        ),
        _event(
            rank=1,
            ordinal=0,
            api="ncclCommInitRankConfig",
            op_type="other",
            ts=101,
            extras={"comm_id": "r1_pair_a", "rank": "1", "nranks": "2", "world_size": "2"},
        ),
        _event(
            rank=8,
            ordinal=0,
            api="ncclCommInitRankConfig",
            op_type="other",
            ts=200,
            extras={"comm_id": "r8_pair_a", "rank": "0", "nranks": "2", "world_size": "2"},
        ),
        _event(
            rank=9,
            ordinal=0,
            api="ncclCommInitRankConfig",
            op_type="other",
            ts=201,
            extras={"comm_id": "r9_pair_a", "rank": "1", "nranks": "2", "world_size": "2"},
        ),
        _event(
            rank=0,
            ordinal=1,
            api="ncclSend",
            op_type="nccl_collective",
            ts=110,
            extras={"comm_id": "r0_pair_a", "peer": "1", "count": "8", "datatype": "9"},
        ),
        _event(
            rank=1,
            ordinal=1,
            api="ncclRecv",
            op_type="nccl_collective",
            ts=111,
            extras={"comm_id": "r1_pair_a", "peer": "0", "count": "8", "datatype": "9"},
        ),
        _event(
            rank=8,
            ordinal=1,
            api="ncclSend",
            op_type="nccl_collective",
            ts=210,
            extras={"comm_id": "r8_pair_a", "peer": "1", "count": "8", "datatype": "9"},
        ),
        _event(
            rank=9,
            ordinal=1,
            api="ncclRecv",
            op_type="nccl_collective",
            ts=211,
            extras={"comm_id": "r9_pair_a", "peer": "0", "count": "8", "datatype": "9"},
        ),
    ]

    recovery = recover_communicator_topology_from_events(events)

    recovered_pairs = set(recovery.memberships.values())
    assert (0, 1) in recovered_pairs
    assert (8, 9) in recovered_pairs
    assert all(len(members) == 2 for members in recovered_pairs)


def test_real_recovery_uses_nccl_comm_split_for_collective_topology():
    events: list[TraceEvent] = []
    for rank in range(4):
        events.append(
            _event(
                rank=rank,
                ordinal=0,
                api="ncclCommInitRankConfig",
                op_type="other",
                ts=100 + rank,
                extras={
                    "comm_id": f"world_{rank}",
                    "rank": str(rank),
                    "nranks": "4",
                    "world_size": "4",
                },
            )
        )
        events.append(
            _event(
                rank=rank,
                ordinal=1,
                api="ncclCommSplit",
                op_type="other",
                ts=200 + rank,
                extras={
                    "comm_id": f"world_{rank}",
                    "parent_comm_id": f"world_{rank}",
                    "new_comm_id": f"tp_{rank}",
                    "color": str(rank % 2),
                    "key": str(rank // 2),
                },
            )
        )
        events.append(
            _event(
                rank=rank,
                ordinal=2,
                api="ncclAllReduce",
                op_type="nccl_collective",
                ts=300 + rank,
                extras={
                    "comm_id": f"tp_{rank}",
                    "count": "8",
                    "datatype": "9",
                    "op": "0",
                },
            )
        )

    recovery = recover_communicator_topology_from_events(events)

    recovered_groups = set(recovery.memberships.values())
    assert (0, 2) in recovered_groups
    assert (1, 3) in recovered_groups
    assert recovery.local_comm_aliases[(0, "tp_0")] == recovery.local_comm_aliases[(2, "tp_2")]
    assert recovery.local_comm_aliases[(1, "tp_1")] == recovery.local_comm_aliases[(3, "tp_3")]


def test_real_recovery_groups_numeric_parent_comm_split_by_world_color():
    events: list[TraceEvent] = []
    split_specs = {
        0: ("1000", "2000", "17", "0"),
        8: ("9000", "9008", "17", "1"),
        1: ("1001", "2001", "23", "0"),
        9: ("9001", "9009", "23", "1"),
    }
    for rank, (parent_comm, new_comm, color, key) in split_specs.items():
        events.append(
            _event(
                rank=rank,
                ordinal=0,
                api="ncclCommSplit",
                op_type="other",
                ts=100 + rank,
                extras={
                    "comm_id": parent_comm,
                    "parent_comm_id": parent_comm,
                    "new_comm_id": new_comm,
                    "color": color,
                    "key": key,
                    "world_size": "16",
                },
            )
        )
        events.append(
            _event(
                rank=rank,
                ordinal=1,
                api="ncclAllReduce",
                op_type="nccl_collective",
                ts=200 + rank,
                extras={
                    "comm_id": new_comm,
                    "count": "8",
                    "datatype": "9",
                    "op": "0",
                },
            )
        )

    recovery = recover_communicator_topology_from_events(events)

    recovered_groups = set(recovery.memberships.values())
    assert (0, 8) in recovered_groups
    assert (1, 9) in recovered_groups
    assert recovery.local_comm_aliases[(0, "2000")] == recovery.local_comm_aliases[(8, "9008")]
    assert recovery.local_comm_aliases[(1, "2001")] == recovery.local_comm_aliases[(9, "9009")]


def test_real_recovery_does_not_overmerge_duplicate_key_numeric_parent_splits():
    events: list[TraceEvent] = []
    split_specs = {
        0: ("1000", "2000", "17", "0"),
        2: ("9000", "9002", "17", "1"),
        1: ("1001", "2001", "17", "0"),
        3: ("9001", "9003", "17", "1"),
    }
    for rank, (parent_comm, new_comm, color, key) in split_specs.items():
        events.append(
            _event(
                rank=rank,
                ordinal=0,
                api="ncclCommSplit",
                op_type="other",
                ts=100 + rank,
                extras={
                    "comm_id": parent_comm,
                    "parent_comm_id": parent_comm,
                    "new_comm_id": new_comm,
                    "color": color,
                    "key": key,
                    "world_size": "4",
                },
            )
        )
        events.append(
            _event(
                rank=rank,
                ordinal=1,
                api="ncclAllReduce",
                op_type="nccl_collective",
                ts=200 + rank,
                extras={
                    "comm_id": new_comm,
                    "count": "8",
                    "datatype": "9",
                    "op": "0",
                },
            )
        )

    recovery = recover_communicator_topology_from_events(events)

    recovered_groups = set(recovery.memberships.values())
    assert (0, 2) in recovered_groups
    assert (1, 3) in recovered_groups
    assert not any(set(members) == {0, 1, 2, 3} for members in recovered_groups)
    assert recovery.local_comm_aliases[(0, "2000")] == recovery.local_comm_aliases[(2, "9002")]
    assert recovery.local_comm_aliases[(1, "2001")] == recovery.local_comm_aliases[(3, "9003")]
    assert recovery.local_comm_aliases[(0, "2000")] != recovery.local_comm_aliases[(1, "2001")]
