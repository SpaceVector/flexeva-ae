from pathlib import Path

import pytest

from flexsim.estimator import Estimator, TraceSignatureTimingProvider
from flexsim.maya_lite import (
    CollectiveGroup,
    TraceSource,
    annotate_collated_trace,
    collective_group_duration_summary,
    collate_trace_bundle,
    estimate_low_level_event_us,
    load_trace_directory,
)
from flexsim.maya_lite.annotate import is_ignorable_setup_event
from flexsim.maya_lite.annotate import AnnotationTimingRecorder
from flexsim.maya_lite.annotate import export_predicted_provider_rows
from flexsim.maya_lite.filters import is_low_overhead_api
from flexsim.maya_lite.schema import CollatedEvent, CollatedTrace


@pytest.fixture(scope="module")
def estimator():
    trace_dir = Path("paper/traces/real/e1")
    if not trace_dir.exists():
        pytest.skip(f"trace dir not available: {trace_dir}")
    return Estimator.fit_from_traces(str(trace_dir), max_files=2)


def test_estimate_low_level_event_uses_estimator(estimator: Estimator):
    trace_dir = Path("paper/traces/fake/e3")
    if not trace_dir.exists():
        pytest.skip(f"trace dir not available: {trace_dir}")

    bundle = load_trace_directory(trace_dir, max_events_per_rank=32)
    collated = collate_trace_bundle(bundle)
    event = collated.global_events[0]

    duration_us = estimate_low_level_event_us(event, estimator)
    assert duration_us >= 0.0


def test_annotate_collated_trace_preserves_structure(estimator: Estimator):
    trace_dir = Path("paper/traces/fake/e3")
    if not trace_dir.exists():
        pytest.skip(f"trace dir not available: {trace_dir}")

    bundle = load_trace_directory(trace_dir, max_events_per_rank=128)
    collated = collate_trace_bundle(bundle)
    annotated = annotate_collated_trace(collated, estimator)

    assert annotated.total_events == collated.total_events
    assert annotated.world_size == collated.world_size
    assert set(annotated.collective_groups) == set(collated.collective_groups)

    first_event = annotated.global_events[0]
    assert first_event.duration_us >= 0.0
    assert first_event.id == collated.global_events[0].id
    assert first_event.prev_event_id == collated.global_events[0].prev_event_id

    assert any(event.duration_source == "ignored_setup" for event in annotated.global_events)
    assert any(
        event.duration_source.startswith("estimator_")
        for event in annotated.global_events
    )


def test_annotate_collated_trace_parallel_matches_serial():
    estimator = Estimator()
    event_a0 = CollatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.REAL,
        ts=0,
        pid=1,
        tid=1,
        module="host.dispatch",
        api="__hostDelay__",
        op_type="host_delay",
        extras={"observed_gap_us": 5},
    )
    event_a1 = CollatedEvent(
        id="r0:e1",
        rank=0,
        ordinal=1,
        source=TraceSource.REAL,
        ts=5,
        pid=1,
        tid=1,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
    )
    event_b0 = CollatedEvent(
        id="r1:e0",
        rank=1,
        ordinal=0,
        source=TraceSource.REAL,
        ts=0,
        pid=2,
        tid=2,
        module="host.dispatch",
        api="__hostDelay__",
        op_type="host_delay",
        extras={"observed_gap_us": 7},
    )
    event_b1 = CollatedEvent(
        id="r1:e1",
        rank=1,
        ordinal=1,
        source=TraceSource.REAL,
        ts=7,
        pid=2,
        tid=2,
        module="libcudart.so.12",
        api="cudaDeviceSynchronize",
        op_type="stream_op",
    )
    collated = CollatedTrace(
        trace_dir=Path("/tmp/parallel-annotate"),
        source=TraceSource.REAL,
        rank_events={0: (event_a0, event_a1), 1: (event_b0, event_b1)},
        global_events=(event_a0, event_b0, event_a1, event_b1),
        collective_groups={},
    )

    serial = annotate_collated_trace(collated, estimator, parallel_workers=1)
    parallel = annotate_collated_trace(collated, estimator, parallel_workers=2)

    assert [event.id for event in parallel.global_events] == [event.id for event in serial.global_events]
    assert [event.duration_us for event in parallel.global_events] == [event.duration_us for event in serial.global_events]
    assert [event.duration_source for event in parallel.global_events] == [
        event.duration_source for event in serial.global_events
    ]


def test_annotation_timing_recorder_separates_estimator_work_from_pass_through() -> None:
    estimator = Estimator()
    host_delay = CollatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.REAL,
        ts=0,
        pid=1,
        tid=1,
        module="host.dispatch",
        api="__hostDelay__",
        op_type="host_delay",
        extras={"observed_gap_us": 5},
    )
    modeled = CollatedEvent(
        id="r0:e1",
        rank=0,
        ordinal=1,
        source=TraceSource.REAL,
        ts=5,
        pid=1,
        tid=1,
        module="libcustom.so",
        api="customRuntimeOp",
        op_type="other",
    )
    collated = CollatedTrace(
        trace_dir=Path("/tmp/annotation-timing"),
        source=TraceSource.REAL,
        rank_events={0: (host_delay, modeled)},
        global_events=(host_delay, modeled),
        collective_groups={},
    )

    timing_recorder = AnnotationTimingRecorder()
    annotate_collated_trace(collated, estimator, timing_recorder=timing_recorder)
    summary = timing_recorder.summary(total_annotation_seconds=1.0)

    assert summary["basis"] == "runtime_estimation_only"
    assert summary["annotated_event_count"] == 2
    assert summary["runtime_estimation_event_count"] == 1
    assert summary["pass_through_event_count"] == 1
    assert summary["runtime_estimation_wall_seconds"] >= 0.0
    assert summary["total_annotation_seconds"] == 1.0
    assert summary["duration_source_counts"]["estimator_global_fallback"] == 1
    assert summary["duration_source_counts_by_api"]["customRuntimeOp"]["estimator_global_fallback"] == 1


def test_annotation_timing_recorder_tracks_collective_group_provider_basis() -> None:
    timing_recorder = AnnotationTimingRecorder()

    timing_recorder.record_collective_group_duration_basis("group_provider:test_provider")
    summary = timing_recorder.summary(total_annotation_seconds=0.0)

    assert summary["collective_group_duration_basis_counts"] == {
        "group_provider:test_provider": 1
    }


def test_setup_events_are_zero_cost_and_marked():
    event = CollatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.REAL,
        ts=1,
        pid=1,
        tid=1,
        module="libcudart.so.12",
        api="__cudaRegisterFunction",
        op_type="other",
    )
    estimator = Estimator()

    assert is_ignorable_setup_event(event)
    assert estimate_low_level_event_us(event, estimator) == 0.0


def test_host_delay_event_uses_observed_gap_directly():
    event = CollatedEvent(
        id="r0:h1",
        rank=0,
        ordinal=1,
        source=TraceSource.REAL,
        ts=10,
        pid=1,
        tid=1,
        module="host.dispatch",
        api="__hostDelay__",
        op_type="host_delay",
        extras={"observed_gap_us": 37},
    )
    estimator = Estimator()

    assert estimate_low_level_event_us(event, estimator) == 37.0


def test_control_plane_wrapper_duration_is_opt_in():
    estimator = Estimator()
    event = CollatedEvent(
        id="r0:e_cp",
        rank=0,
        ordinal=0,
        source=TraceSource.FAKE,
        ts=10,
        pid=1,
        tid=1,
        module="libcublas.so.12",
        api="cublasSetStream_v2",
        op_type="other",
        extras={"host_duration_us": 7.5},
    )

    default_duration, default_source = (
        estimate_low_level_event_us(event, estimator),
        annotate_collated_trace(
            CollatedTrace(
                trace_dir=Path("/tmp/control-plane-default"),
                source=TraceSource.FAKE,
                rank_events={0: (event,)},
                global_events=(event,),
                collective_groups={},
            ),
            estimator,
        ).global_events[0].duration_source,
    )
    opted_in = annotate_collated_trace(
        CollatedTrace(
            trace_dir=Path("/tmp/control-plane-optin"),
            source=TraceSource.FAKE,
            rank_events={0: (event,)},
            global_events=(event,),
            collective_groups={},
        ),
        estimator,
        use_observed_control_plane_wrapper_durations=True,
    ).global_events[0]

    assert default_duration != 7.5
    assert default_source != "observed_wrapper_duration"
    assert opted_in.duration_us == 7.5
    assert opted_in.duration_source == "observed_wrapper_duration"


def test_semantic_wrapper_duration_is_opt_in() -> None:
    estimator = Estimator()
    event = CollatedEvent(
        id="r0:e_sem",
        rank=0,
        ordinal=0,
        source=TraceSource.FAKE,
        ts=10,
        pid=1,
        tid=1,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={
            "host_duration_us": 7.5,
            "wrapper_runtime_contract": "direct_runtime",
        },
    )

    default_annotated = annotate_collated_trace(
        CollatedTrace(
            trace_dir=Path("/tmp/semantic-default"),
            source=TraceSource.FAKE,
            rank_events={0: (event,)},
            global_events=(event,),
            collective_groups={},
        ),
        estimator,
        allow_kernel_launch_heuristic_fallback=True,
    ).global_events[0]
    opted_in = annotate_collated_trace(
        CollatedTrace(
            trace_dir=Path("/tmp/semantic-optin"),
            source=TraceSource.FAKE,
            rank_events={0: (event,)},
            global_events=(event,),
            collective_groups={},
        ),
        estimator,
        allow_kernel_launch_heuristic_fallback=True,
        use_observed_semantic_wrapper_durations=True,
    ).global_events[0]

    assert default_annotated.duration_us != 7.5
    assert default_annotated.duration_source != "observed_wrapper_duration"
    assert opted_in.duration_us == 7.5
    assert opted_in.duration_source == "observed_wrapper_duration"


def test_semantic_wrapper_duration_opt_in_requires_direct_runtime_contract() -> None:
    estimator = Estimator()
    event = CollatedEvent(
        id="r0:e_sem_missing_contract",
        rank=0,
        ordinal=0,
        source=TraceSource.FAKE,
        ts=10,
        pid=1,
        tid=1,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={"host_duration_us": 7.5},
    )

    annotated = annotate_collated_trace(
        CollatedTrace(
            trace_dir=Path("/tmp/semantic-contract-missing"),
            source=TraceSource.FAKE,
            rank_events={0: (event,)},
            global_events=(event,),
            collective_groups={},
        ),
        estimator,
        allow_kernel_launch_heuristic_fallback=True,
        use_observed_semantic_wrapper_durations=True,
    ).global_events[0]

    assert annotated.duration_us != 7.5
    assert annotated.duration_source != "observed_wrapper_duration"


def test_semantic_wrapper_duration_opt_in_can_fall_back_to_end_ts_delta() -> None:
    estimator = Estimator()
    event = CollatedEvent(
        id="r0:e_sem_endts",
        rank=0,
        ordinal=0,
        source=TraceSource.FAKE,
        ts=10,
        pid=1,
        tid=1,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={
            "end_ts": 18,
            "wrapper_runtime_contract": "direct_runtime",
        },
    )

    annotated = annotate_collated_trace(
        CollatedTrace(
            trace_dir=Path("/tmp/semantic-endts-optin"),
            source=TraceSource.FAKE,
            rank_events={0: (event,)},
            global_events=(event,),
            collective_groups={},
        ),
        estimator,
        allow_kernel_launch_heuristic_fallback=True,
        use_observed_semantic_wrapper_durations=True,
    ).global_events[0]

    assert annotated.duration_us == 8.0
    assert annotated.duration_source == "observed_wrapper_duration"


def test_semantic_wrapper_duration_opt_in_accepts_explicit_direct_runtime_field_without_contract() -> None:
    estimator = Estimator()
    event = CollatedEvent(
        id="r0:e_sem_explicit",
        rank=0,
        ordinal=0,
        source=TraceSource.FAKE,
        ts=10,
        pid=1,
        tid=1,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={
            "direct_runtime_us": 14.0,
        },
    )

    annotated = annotate_collated_trace(
        CollatedTrace(
            trace_dir=Path("/tmp/semantic-explicit-optin"),
            source=TraceSource.FAKE,
            rank_events={0: (event,)},
            global_events=(event,),
            collective_groups={},
        ),
        estimator,
        allow_kernel_launch_heuristic_fallback=True,
        use_observed_semantic_wrapper_durations=True,
    ).global_events[0]

    assert annotated.duration_us == 14.0
    assert annotated.duration_source == "observed_wrapper_duration"


def test_annotate_attaches_group_level_collective_duration_metadata() -> None:
    estimator = Estimator()
    event_rank0 = CollatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.FAKE,
        ts=10,
        pid=1,
        tid=1,
        module="libnccl.so.2",
        api="ncclAllReduce",
        op_type="nccl_collective",
        extras={
            "host_duration_us": "13.0",
            "wrapper_runtime_contract": "direct_runtime",
        },
        collective_group_id="ncclAllReduce#0",
    )
    event_rank1 = CollatedEvent(
        id="r1:e0",
        rank=1,
        ordinal=0,
        source=TraceSource.FAKE,
        ts=12,
        pid=2,
        tid=2,
        module="libnccl.so.2",
        api="ncclAllReduce",
        op_type="nccl_collective",
        extras={
            "host_duration_us": "17.0",
            "wrapper_runtime_contract": "direct_runtime",
        },
        collective_group_id="ncclAllReduce#0",
    )
    collated = CollatedTrace(
        trace_dir=Path("/tmp/collective-group-metadata"),
        source=TraceSource.FAKE,
        rank_events={0: (event_rank0,), 1: (event_rank1,)},
        global_events=(event_rank0, event_rank1),
        collective_groups={
            "ncclAllReduce#0": CollectiveGroup(
                id="ncclAllReduce#0",
                api="ncclAllReduce",
                op_type="nccl_collective",
                ranks=(0, 1),
                event_ids=("r0:e0", "r1:e0"),
            )
        },
    )

    annotated = annotate_collated_trace(
        collated,
        estimator,
        use_observed_semantic_wrapper_durations=True,
    )

    collective_events = [event for event in annotated.global_events if event.collective_group_id]
    assert [event.duration_us for event in collective_events] == [13.0, 17.0]
    assert [event.duration_source for event in collective_events] == [
        "observed_wrapper_duration",
        "observed_wrapper_duration",
    ]
    assert all(event.extras["collective_group_duration_us"] == 17.0 for event in collective_events)
    assert all(
        event.extras["collective_group_duration_basis"] == "max_member_duration"
        for event in collective_events
    )


def test_annotate_prefers_group_signature_duration_for_observed_collectives() -> None:
    estimator = Estimator(
        providers=[
            TraceSignatureTimingProvider.fit(
                [
                    {
                        "api": "ncclAllReduce",
                        "type": "nccl_collective",
                        "module": "libnccl.so.2",
                        "collective": "allreduce",
                        "collective_api": "ncclAllReduce",
                        "count": 1024,
                        "datatype": 7,
                        "world_size": 8,
                        "communicator_size": 2,
                        "participant_count": 2,
                    }
                ],
                [80.0],
            )
        ]
    )
    event_rank0 = CollatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.REAL,
        ts=10,
        pid=1,
        tid=1,
        module="libnccl.so.2",
        api="ncclAllReduce",
        op_type="nccl_collective",
        extras={
            "collective": "allreduce",
            "collective_api": "ncclAllReduce",
            "count": "1024",
            "datatype": "7",
            "communicator_size": "2",
            "participant_count": "2",
            "host_duration_us": "13.0",
            "wrapper_runtime_contract": "direct_runtime",
        },
        collective_group_id="ncclAllReduce#0",
    )
    event_rank1 = CollatedEvent(
        id="r1:e0",
        rank=1,
        ordinal=0,
        source=TraceSource.REAL,
        ts=12,
        pid=2,
        tid=2,
        module="libnccl.so.2",
        api="ncclAllReduce",
        op_type="nccl_collective",
        extras={
            "collective": "allreduce",
            "collective_api": "ncclAllReduce",
            "count": "1024",
            "datatype": "7",
            "communicator_size": "2",
            "participant_count": "2",
            "host_duration_us": "17.0",
            "wrapper_runtime_contract": "direct_runtime",
        },
        collective_group_id="ncclAllReduce#0",
    )
    collated = CollatedTrace(
        trace_dir=Path("/tmp/collective-group-observed-provider"),
        source=TraceSource.REAL,
        rank_events={0: (event_rank0,), 1: (event_rank1,)},
        global_events=(event_rank0, event_rank1),
        collective_groups={
            "ncclAllReduce#0": CollectiveGroup(
                id="ncclAllReduce#0",
                api="ncclAllReduce",
                op_type="nccl_collective",
                ranks=(0, 1),
                event_ids=("r0:e0", "r1:e0"),
            )
        },
        original_world_size=8,
    )

    annotated = annotate_collated_trace(
        collated,
        estimator,
        use_observed_semantic_wrapper_durations=True,
    )

    collective_events = [event for event in annotated.global_events if event.collective_group_id]
    assert [event.duration_source for event in collective_events] == [
        "observed_wrapper_duration",
        "observed_wrapper_duration",
    ]
    assert all(event.extras["collective_group_duration_us"] == 80.0 for event in collective_events)
    assert all(
        event.extras["collective_group_duration_basis"] == "group_provider:trace_signature_stats"
        for event in collective_events
    )


def test_annotate_prefers_group_signature_duration_for_estimator_collectives() -> None:
    estimator = Estimator(
        providers=[
            TraceSignatureTimingProvider.fit(
                [
                    {
                        "api": "ncclAllReduce",
                        "type": "nccl_collective",
                        "module": "libnccl.so.2",
                        "collective": "allreduce",
                        "collective_api": "ncclAllReduce",
                        "count": 1024,
                        "datatype": 7,
                        "world_size": 8,
                        "communicator_size": 4,
                        "participant_count": 4,
                    }
                ],
                [80.0],
            )
        ]
    )
    event_rank0 = CollatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.FAKE,
        ts=10,
        pid=1,
        tid=1,
        module="libnccl.so.2",
        api="ncclAllReduce",
        op_type="nccl_collective",
        extras={
            "collective": "allreduce",
            "collective_api": "ncclAllReduce",
            "count": "1024",
            "datatype": "7",
            "communicator_size": "4",
            "participant_count": "4",
        },
        collective_group_id="ncclAllReduce#0",
    )
    event_rank1 = CollatedEvent(
        id="r1:e0",
        rank=1,
        ordinal=0,
        source=TraceSource.FAKE,
        ts=12,
        pid=2,
        tid=2,
        module="libnccl.so.2",
        api="ncclAllReduce",
        op_type="nccl_collective",
        extras={
            "collective": "allreduce",
            "collective_api": "ncclAllReduce",
            "count": "1024",
            "datatype": "7",
            "communicator_size": "4",
            "participant_count": "4",
        },
        collective_group_id="ncclAllReduce#0",
    )
    collated = CollatedTrace(
        trace_dir=Path("/tmp/collective-group-provider"),
        source=TraceSource.FAKE,
        rank_events={0: (event_rank0,), 1: (event_rank1,)},
        global_events=(event_rank0, event_rank1),
        collective_groups={
            "ncclAllReduce#0": CollectiveGroup(
                id="ncclAllReduce#0",
                api="ncclAllReduce",
                op_type="nccl_collective",
                ranks=(0, 1),
                event_ids=("r0:e0", "r1:e0"),
            )
        },
        original_world_size=8,
    )

    annotated = annotate_collated_trace(collated, estimator)

    collective_events = [event for event in annotated.global_events if event.collective_group_id]
    assert all(event.duration_us == 80.0 for event in collective_events)
    assert all(event.extras["collective_group_duration_us"] == 80.0 for event in collective_events)
    assert all(
        event.extras["collective_group_duration_basis"] == "group_provider:trace_signature_stats"
        for event in collective_events
    )


def test_annotate_normalizes_p2p_group_runtime_payload_and_preserves_member_api() -> None:
    class P2PGroupProvider:
        name = "p2p_group_provider"
        supports_collective_group_timing = True

        def __init__(self) -> None:
            self.seen: list[dict[str, object]] = []

        def estimate_us(self, event, percentile="p50"):
            del percentile
            self.seen.append(dict(event))
            if (
                event.get("api") == "ncclP2P"
                and event.get("collective") == "p2p"
                and event.get("collective_api") == "ncclP2P"
                and event.get("world_size") == 2
                and event.get("communicator_size") == 2
                and event.get("participant_count") == 2
            ):
                return 55.0
            if event.get("api") in {"ncclSend", "ncclRecv"}:
                return 1.0
            return None

    provider = P2PGroupProvider()
    estimator = Estimator(providers=[provider])
    event_rank0 = CollatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.FAKE,
        ts=10,
        pid=1,
        tid=1,
        module="libnccl.so.2",
        api="ncclRecv",
        op_type="nccl_collective",
        extras={
            "collective": "recv",
            "collective_api": "ncclRecv",
            "count": "4096",
            "datatype": "7",
            "communicator_size": "2",
            "participant_count": "1",
        },
        collective_group_id="ncclRecv|comm:p2p|call:0",
    )
    event_rank1 = CollatedEvent(
        id="r1:e0",
        rank=1,
        ordinal=0,
        source=TraceSource.FAKE,
        ts=12,
        pid=2,
        tid=2,
        module="libnccl.so.2",
        api="ncclSend",
        op_type="nccl_collective",
        extras={
            "collective": "send",
            "collective_api": "ncclSend",
            "count": "4096",
            "datatype": "7",
            "communicator_size": "2",
            "participant_count": "1",
        },
        collective_group_id="ncclRecv|comm:p2p|call:0",
    )
    collated = CollatedTrace(
        trace_dir=Path("/tmp/p2p-group-runtime-payload"),
        source=TraceSource.FAKE,
        rank_events={0: (event_rank0,), 1: (event_rank1,)},
        global_events=(event_rank0, event_rank1),
        collective_groups={
            "ncclRecv|comm:p2p|call:0": CollectiveGroup(
                id="ncclRecv|comm:p2p|call:0",
                api="ncclP2P",
                op_type="nccl_collective",
                ranks=(0, 1),
                event_ids=("r0:e0", "r1:e0"),
            )
        },
        original_world_size=16,
    )

    annotated = annotate_collated_trace(collated, estimator)

    group_payloads = [payload for payload in provider.seen if payload.get("api") == "ncclP2P"]
    assert len(group_payloads) == 1
    assert group_payloads[0]["collective"] == "p2p"
    assert group_payloads[0]["collective_api"] == "ncclP2P"
    assert group_payloads[0]["world_size"] == 2
    assert group_payloads[0]["communicator_size"] == 2
    assert group_payloads[0]["participant_count"] == 2
    assert group_payloads[0]["trace_world_size"] == 16
    assert group_payloads[0]["member_api"] == "ncclRecv"
    collective_events = [event for event in annotated.global_events if event.collective_group_id]
    assert all(event.extras["collective_group_duration_us"] == 55.0 for event in collective_events)
    assert all(event.extras["collective_group_runtime_api"] == "ncclP2P" for event in collective_events)
    assert [event.extras["member_api"] for event in collective_events] == ["ncclRecv", "ncclSend"]


def test_annotate_preserves_global_event_order_when_attaching_collective_group_duration() -> None:
    estimator = Estimator()
    event_rank0 = CollatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.FAKE,
        ts=10,
        pid=1,
        tid=1,
        module="libnccl.so.2",
        api="ncclAllReduce",
        op_type="nccl_collective",
        collective_group_id="ncclAllReduce#0",
    )
    host_event = CollatedEvent(
        id="r0:e1",
        rank=0,
        ordinal=1,
        source=TraceSource.FAKE,
        ts=15,
        pid=1,
        tid=1,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
    )
    event_rank1 = CollatedEvent(
        id="r1:e0",
        rank=1,
        ordinal=0,
        source=TraceSource.FAKE,
        ts=20,
        pid=2,
        tid=2,
        module="libnccl.so.2",
        api="ncclAllReduce",
        op_type="nccl_collective",
        collective_group_id="ncclAllReduce#0",
    )
    collated = CollatedTrace(
        trace_dir=Path("/tmp/collective-group-order"),
        source=TraceSource.FAKE,
        rank_events={0: (event_rank0, host_event), 1: (event_rank1,)},
        global_events=(event_rank0, host_event, event_rank1),
        collective_groups={
            "ncclAllReduce#0": CollectiveGroup(
                id="ncclAllReduce#0",
                api="ncclAllReduce",
                op_type="nccl_collective",
                ranks=(0, 1),
                event_ids=("r0:e0", "r1:e0"),
            )
        },
    )

    annotated = annotate_collated_trace(collated, estimator)

    assert [event.id for event in annotated.global_events] == ["r0:e0", "r0:e1", "r1:e0"]
    assert annotated.global_events[0].extras["collective_group_duration_us"] == 0.0
    assert "collective_group_duration_us" not in annotated.global_events[1].extras
    assert annotated.global_events[2].extras["collective_group_duration_us"] == 0.0


def test_collective_group_duration_summary_reports_group_basis_counts() -> None:
    estimator = Estimator()
    event_rank0 = CollatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.FAKE,
        ts=10,
        pid=1,
        tid=1,
        module="libnccl.so.2",
        api="ncclAllReduce",
        op_type="nccl_collective",
        extras={
            "host_duration_us": "13.0",
            "wrapper_runtime_contract": "direct_runtime",
        },
        collective_group_id="ncclAllReduce#0",
    )
    event_rank1 = CollatedEvent(
        id="r1:e0",
        rank=1,
        ordinal=0,
        source=TraceSource.FAKE,
        ts=12,
        pid=2,
        tid=2,
        module="libnccl.so.2",
        api="ncclAllReduce",
        op_type="nccl_collective",
        extras={
            "host_duration_us": "17.0",
            "wrapper_runtime_contract": "direct_runtime",
        },
        collective_group_id="ncclAllReduce#0",
    )
    annotated = annotate_collated_trace(
        CollatedTrace(
            trace_dir=Path("/tmp/collective-group-summary"),
            source=TraceSource.FAKE,
            rank_events={0: (event_rank0,), 1: (event_rank1,)},
            global_events=(event_rank0, event_rank1),
            collective_groups={
                "ncclAllReduce#0": CollectiveGroup(
                    id="ncclAllReduce#0",
                    api="ncclAllReduce",
                    op_type="nccl_collective",
                    ranks=(0, 1),
                    event_ids=("r0:e0", "r1:e0"),
                )
            },
        ),
        estimator,
        use_observed_semantic_wrapper_durations=True,
    )

    summary = collective_group_duration_summary(annotated)

    assert summary["collective_group_count"] == 1
    assert summary["collective_group_with_duration_metadata_count"] == 1
    assert summary["collective_event_with_duration_metadata_count"] == 2
    assert summary["collective_group_duration_basis_counts"] == {
        "max_member_duration": 1
    }
    assert summary["duration_source_counts"] == {
        "observed_wrapper_duration": 2,
    }
    assert summary["strict_runtime_signal_duration_source_counts"] == {
        "observed_wrapper_duration": 2,
    }
    assert summary["strict_runtime_signal_wrapper_timing_contract_counts"] == {
        "direct_runtime": 2,
    }
    assert summary["strict_runtime_signal_event_count_by_api"] == {
        "ncclAllReduce": 2,
    }
    assert summary["strict_runtime_signal_duration_source_counts_by_api"] == {
        "ncclAllReduce": {"observed_wrapper_duration": 2},
    }
    assert summary["strict_runtime_signal_wrapper_timing_contract_counts_by_api"] == {
        "ncclAllReduce": {"direct_runtime": 2},
    }
    assert summary["event_with_wrapper_timing_field_count"] == 2
    assert summary["event_with_explicit_direct_runtime_field_count"] == 0
    assert summary["event_with_direct_wrapper_runtime_count"] == 2
    assert summary["strict_runtime_signal_event_count"] == 2
    assert summary["strict_runtime_signal_event_with_wrapper_timing_field_count"] == 2
    assert summary["strict_runtime_signal_event_with_explicit_direct_runtime_field_count"] == 0
    assert summary["strict_runtime_signal_event_with_direct_wrapper_runtime_count"] == 2
    assert summary["strict_runtime_signal_event_with_direct_runtime_contract_count"] == 2
    assert summary["strict_runtime_signal_observed_wrapper_duration_count"] == 2
    assert summary["strict_runtime_signal_observed_wrapper_duration_dispatch_only_count"] == 0
    assert summary["strict_runtime_signal_observed_wrapper_duration_direct_runtime_count"] == 2


def test_collective_group_duration_summary_separates_dispatch_only_wrapper_timing_from_direct_runtime() -> None:
    estimator = Estimator()
    event = CollatedEvent(
        id="r0:e_dispatch_only",
        rank=0,
        ordinal=0,
        source=TraceSource.FAKE,
        ts=10,
        pid=1,
        tid=1,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={
            "host_duration_us": 9.0,
            "wrapper_runtime_contract": "dispatch_only",
        },
    )

    annotated = annotate_collated_trace(
        CollatedTrace(
            trace_dir=Path("/tmp/dispatch-only-summary"),
            source=TraceSource.FAKE,
            rank_events={0: (event,)},
            global_events=(event,),
            collective_groups={},
        ),
        estimator,
        allow_kernel_launch_heuristic_fallback=True,
        use_observed_semantic_wrapper_durations=True,
    )
    summary = collective_group_duration_summary(annotated)

    assert summary["event_with_wrapper_timing_field_count"] == 1
    assert summary["event_with_explicit_direct_runtime_field_count"] == 0
    assert summary["event_with_direct_wrapper_runtime_count"] == 0
    assert summary["strict_runtime_signal_event_with_wrapper_timing_field_count"] == 1
    assert summary["strict_runtime_signal_event_with_explicit_direct_runtime_field_count"] == 0
    assert summary["strict_runtime_signal_event_with_direct_wrapper_runtime_count"] == 0
    assert summary["strict_runtime_signal_wrapper_timing_contract_counts"] == {
        "dispatch_only": 1,
    }
    assert summary["strict_runtime_signal_event_count_by_api"] == {
        "cudaLaunchKernel": 1,
    }
    assert summary["strict_runtime_signal_duration_source_counts_by_api"] == {
        "cudaLaunchKernel": summary["strict_runtime_signal_duration_source_counts"],
    }
    assert summary["strict_runtime_signal_wrapper_timing_contract_counts_by_api"] == {
        "cudaLaunchKernel": {"dispatch_only": 1},
    }
    assert summary["strict_runtime_signal_event_with_direct_runtime_contract_count"] == 0
    assert summary["strict_runtime_signal_observed_wrapper_duration_count"] == 0
    assert summary["strict_runtime_signal_observed_wrapper_duration_dispatch_only_count"] == 0
    assert summary["strict_runtime_signal_observed_wrapper_duration_direct_runtime_count"] == 0


def test_collective_group_duration_summary_tracks_explicit_direct_runtime_fields() -> None:
    estimator = Estimator()
    event = CollatedEvent(
        id="r0:e_explicit_direct_runtime",
        rank=0,
        ordinal=0,
        source=TraceSource.FAKE,
        ts=10,
        pid=1,
        tid=1,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={
            "direct_runtime_us": 9.0,
        },
    )

    annotated = annotate_collated_trace(
        CollatedTrace(
            trace_dir=Path("/tmp/explicit-direct-runtime-summary"),
            source=TraceSource.FAKE,
            rank_events={0: (event,)},
            global_events=(event,),
            collective_groups={},
        ),
        estimator,
        allow_kernel_launch_heuristic_fallback=True,
        use_observed_semantic_wrapper_durations=True,
    )
    summary = collective_group_duration_summary(annotated)

    assert summary["event_with_wrapper_timing_field_count"] == 1
    assert summary["event_with_explicit_direct_runtime_field_count"] == 1
    assert summary["event_with_direct_wrapper_runtime_count"] == 1
    assert summary["strict_runtime_signal_event_with_wrapper_timing_field_count"] == 1
    assert summary["strict_runtime_signal_event_with_explicit_direct_runtime_field_count"] == 1
    assert summary["strict_runtime_signal_event_with_direct_wrapper_runtime_count"] == 1
    assert summary["strict_runtime_signal_wrapper_timing_contract_counts"] == {
        "direct_runtime": 1,
    }
    assert summary["strict_runtime_signal_event_count_by_api"] == {
        "cudaLaunchKernel": 1,
    }
    assert summary["strict_runtime_signal_duration_source_counts_by_api"] == {
        "cudaLaunchKernel": summary["strict_runtime_signal_duration_source_counts"],
    }
    assert summary["strict_runtime_signal_wrapper_timing_contract_counts_by_api"] == {
        "cudaLaunchKernel": {"direct_runtime": 1},
    }
    assert summary["strict_runtime_signal_event_with_direct_runtime_contract_count"] == 1
    assert summary["strict_runtime_signal_observed_wrapper_duration_count"] == 1
    assert summary["strict_runtime_signal_observed_wrapper_duration_dispatch_only_count"] == 0
    assert summary["strict_runtime_signal_observed_wrapper_duration_direct_runtime_count"] == 1


class _PayloadCheckingProvider:
    name = "payload_check"

    def __init__(self):
        self.seen = None

    def estimate_us(self, event, percentile="p50"):
        del percentile
        self.seen = dict(event)
        return 7.0


class _ProviderCounterpartProvider:
    name = "counterpart_provider"

    def estimate_us(self, event, percentile="p50"):
        del event, percentile
        return 9.0


def test_annotate_adds_provider_counterpart_duration_source_metadata():
    estimator = Estimator(providers=[_ProviderCounterpartProvider()])
    event = CollatedEvent(
        id="r0:e1",
        rank=0,
        ordinal=1,
        source=TraceSource.FAKE,
        ts=10,
        pid=1,
        tid=2,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={
            "kernel": "kernel_a",
            "launch_grid": "2x3x4",
            "launch_block": "5x6x7",
            "launch_shared_mem_bytes": 128,
            "launch_stream_id": "9",
        },
    )
    collated = CollatedTrace(
        trace_dir=Path("/tmp/manual"),
        source=TraceSource.FAKE,
        rank_events={0: (event,)},
        global_events=(event,),
        collective_groups={},
    )

    annotated = annotate_collated_trace(collated, estimator)
    annotated_event = annotated.global_events[0]

    assert annotated_event.duration_source == "estimator_provider:counterpart_provider"
    assert annotated_event.extras["provider_duration_source_expected"] == "estimator_provider:counterpart_provider"
    assert annotated_event.extras["material_api"] == "cudaLaunchKernel"
    assert annotated_event.extras["material_signature"] == (
        "kernel=kernel_a;grid=2x3x4;block=5x6x7;shared_mem=128;stream=9"
    )
    assert annotated_event.extras["provider_counterpart_key"] == (
        "rank=0|api=cudaLaunchKernel|ordinal=1|"
        "kernel=kernel_a;grid=2x3x4;block=5x6x7;shared_mem=128;stream=9"
    )


def test_annotate_keeps_empty_material_signature_inputs_for_unavailable_non_gemm_signature():
    estimator = Estimator(providers=[_ProviderCounterpartProvider()])
    event = CollatedEvent(
        id="r0:e1",
        rank=0,
        ordinal=1,
        source=TraceSource.FAKE,
        ts=10,
        pid=1,
        tid=2,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={},
    )
    collated = CollatedTrace(
        trace_dir=Path("/tmp/manual-unavailable-signature"),
        source=TraceSource.FAKE,
        rank_events={0: (event,)},
        global_events=(event,),
        collective_groups={},
    )

    annotated_event = annotate_collated_trace(collated, estimator).global_events[0]

    assert annotated_event.duration_source == "estimator_provider:counterpart_provider"
    assert "material_signature" not in annotated_event.extras
    assert annotated_event.extras["material_signature_inputs"] == {}
    assert annotated_event.extras["provider_counterpart_key"] == (
        "rank=0|api=cudaLaunchKernel|ordinal=1|signature:unavailable"
    )


def test_annotate_passes_low_level_context_to_event_providers():
    provider = _PayloadCheckingProvider()
    estimator = Estimator(providers=[provider])
    event = CollatedEvent(
        id="r1:e3",
        rank=1,
        ordinal=3,
        source=TraceSource.REAL,
        ts=10,
        pid=21,
        tid=22,
        module="libnccl.so.2",
        api="ncclAllReduce",
        op_type="nccl_collective",
        collective_group_id="ncclAllReduce#0",
    )

    duration = estimate_low_level_event_us(
        event,
        estimator,
        prev_api="cudaLaunchKernel",
        world_size=4,
    )

    assert duration == 7.0
    assert provider.seen is not None
    assert provider.seen["module"] == "libnccl.so.2"
    assert provider.seen["rank"] == 1
    assert provider.seen["world_size"] == 4
    assert provider.seen["prev_api"] == "cudaLaunchKernel"
    assert provider.seen["collective_group_id"] == "ncclAllReduce#0"


def test_annotate_marks_host_delay_source():
    estimator = Estimator()
    event = CollatedEvent(
        id="r0:h2",
        rank=0,
        ordinal=2,
        source=TraceSource.REAL,
        ts=20,
        pid=1,
        tid=1,
        module="host.dispatch",
        api="__hostDelay__",
        op_type="host_delay",
        extras={"observed_gap_us": 19},
    )

    manual = annotate_collated_trace(
        CollatedTrace(
            trace_dir=Path("/tmp/manual"),
            source=TraceSource.REAL,
            rank_events={0: (event,)},
            global_events=(event,),
            collective_groups={},
        ),
        estimator,
    )
    assert manual.global_events[0].duration_us == 19.0
    assert manual.global_events[0].duration_source == "observed_host_delay"


def test_estimate_low_level_event_treats_cuda_launch_kernel_as_low_overhead():
    provider = _PayloadCheckingProvider()
    estimator = Estimator(providers=[provider])
    event = CollatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.REAL,
        ts=1,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={"kernel_id": "k1"},
    )

    duration = estimate_low_level_event_us(
        event,
        estimator,
        allow_kernel_launch_heuristic_fallback=True,
    )

    assert duration == 11500.0
    assert provider.seen is None


def test_estimate_low_level_event_can_disable_cuda_launch_kernel_heuristic_fallback():
    provider = _PayloadCheckingProvider()
    estimator = Estimator(providers=[provider])
    event = CollatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.REAL,
        ts=1,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={"kernel_id": "k1"},
    )

    duration = estimate_low_level_event_us(
        event,
        estimator,
        allow_kernel_launch_heuristic_fallback=False,
    )

    assert duration == 7.0
    assert provider.seen is not None


def test_estimate_low_level_event_defaults_to_no_cuda_launch_kernel_heuristic_fallback():
    provider = _PayloadCheckingProvider()
    estimator = Estimator(providers=[provider])
    event = CollatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.REAL,
        ts=1,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={"kernel_id": "k1"},
    )

    duration = estimate_low_level_event_us(event, estimator)

    assert duration == 7.0
    assert provider.seen is not None


def test_estimate_low_level_event_rejects_weak_kernel_type_fallback_by_default():
    estimator = Estimator()
    estimator._type_stats["kernel_launch"] = {
        "p50": 9.0,
        "mean": 9.0,
        "p95": 9.0,
        "count": 1,
    }
    event = CollatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.REAL,
        ts=1,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={"kernel_id": "k1"},
    )

    with pytest.raises(RuntimeError, match="Strict runtime estimation required for cudaLaunchKernel"):
        estimate_low_level_event_us(event, estimator)


def test_estimate_low_level_event_can_opt_in_to_weak_kernel_type_fallback():
    estimator = Estimator()
    estimator._type_stats["kernel_launch"] = {
        "p50": 9.0,
        "mean": 9.0,
        "p95": 9.0,
        "count": 1,
    }
    event = CollatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.REAL,
        ts=1,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={"kernel_id": "k1"},
    )

    duration = estimate_low_level_event_us(
        event,
        estimator,
        allow_weak_runtime_fallback=True,
    )

    assert duration == 9.0


def test_estimate_low_level_event_allows_api_specific_kernel_stats_without_opt_in():
    estimator = Estimator()
    estimator._stats[("cudaLaunchKernel", "kernel_launch")] = {
        "p50": 13.0,
        "mean": 13.0,
        "p95": 13.0,
        "count": 1,
    }
    event = CollatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.REAL,
        ts=1,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={"kernel_id": "k1"},
    )

    duration = estimate_low_level_event_us(event, estimator)

    assert duration == 13.0


def test_estimate_low_level_event_rejects_weak_blas_type_fallback_by_default():
    estimator = Estimator()
    estimator._type_stats["blas_compute"] = {
        "p50": 21.0,
        "mean": 21.0,
        "p95": 21.0,
        "count": 1,
    }
    event = CollatedEvent(
        id="r0:e1",
        rank=0,
        ordinal=1,
        source=TraceSource.REAL,
        ts=2,
        pid=2,
        tid=3,
        module="libcublas.so.12",
        api="cublasGemmEx",
        op_type="blas_compute",
        extras={"m": "16", "n": "16", "k": "16"},
    )

    with pytest.raises(RuntimeError, match="Strict runtime estimation required for cublasGemmEx"):
        estimate_low_level_event_us(event, estimator)


def test_estimate_low_level_event_allows_api_specific_blas_stats_without_opt_in():
    estimator = Estimator()
    estimator._stats[("cublasGemmEx", "blas_compute")] = {
        "p50": 22.0,
        "mean": 22.0,
        "p95": 22.0,
        "count": 1,
    }
    event = CollatedEvent(
        id="r0:e1",
        rank=0,
        ordinal=1,
        source=TraceSource.REAL,
        ts=2,
        pid=2,
        tid=3,
        module="libcublas.so.12",
        api="cublasGemmEx",
        op_type="blas_compute",
        extras={"m": "16", "n": "16", "k": "16"},
    )

    duration = estimate_low_level_event_us(event, estimator)

    assert duration == 22.0


def test_async_observer_event_gets_actual_device_runtime_ledger_metadata():
    estimator = Estimator()
    estimator._stats[("cublasGemmEx", "blas_compute")] = {
        "p50": 22.0,
        "mean": 22.0,
        "p95": 22.0,
        "count": 1,
    }
    event = CollatedEvent(
        id="r3:e7",
        rank=3,
        ordinal=7,
        source=TraceSource.REAL,
        ts=20,
        pid=2,
        tid=3,
        module="libcublas.so.12",
        api="cublasGemmEx",
        op_type="blas_compute",
        extras={
            "observed_runtime_us": "12.5",
            "wrapper_runtime_contract": "async_runtime",
            "runtime_observation_source": "capture_real_cuda_event",
            "m": "16",
            "n": "32",
            "k": "64",
            "stream_id": "9",
        },
    )
    collated = CollatedTrace(
        trace_dir=Path("/tmp/actual-device-runtime"),
        source=TraceSource.REAL,
        rank_events={3: (event,)},
        global_events=(event,),
        collective_groups={},
    )

    annotated = annotate_collated_trace(collated, estimator)
    extras = annotated.global_events[0].extras

    assert extras["actual_device_runtime_us"] == 12.5
    assert extras["actual_runtime_measurement_kind"] == "existing_cpp_event_async_runtime_observer"
    assert extras["material_api"] == "cublasGemmEx"
    assert "material_signature" in extras
    assert "m=16" in extras["material_signature"]
    assert "n=32" in extras["material_signature"]
    assert "k=64" in extras["material_signature"]
    assert "provider_counterpart_key" in extras
    assert extras["provider_counterpart_key"].startswith("rank=3|api=cublasGemmEx|ordinal=7|")
    assert extras["actual_counterpart_rank"] == 3
    assert extras["stream"] == "9"
    assert extras["actual_counterpart_window"] == "event"
    assert annotated.global_events[0].duration_us == 12.5
    assert annotated.global_events[0].duration_source == "observed_wrapper_duration"


def test_non_async_observed_runtime_does_not_get_actual_device_runtime_metadata():
    estimator = Estimator()
    event = CollatedEvent(
        id="r0:e1",
        rank=0,
        ordinal=1,
        source=TraceSource.REAL,
        ts=2,
        pid=2,
        tid=3,
        module="libcublas.so.12",
        api="cublasGemmEx",
        op_type="blas_compute",
        extras={
            "observed_runtime_us": "12.5",
            "wrapper_runtime_contract": "dispatch_only",
            "runtime_observation_source": "capture_real_cuda_event",
        },
    )
    collated = CollatedTrace(
        trace_dir=Path("/tmp/non-async-actual-device-runtime"),
        source=TraceSource.REAL,
        rank_events={0: (event,)},
        global_events=(event,),
        collective_groups={},
    )

    annotated = annotate_collated_trace(collated, estimator, allow_weak_runtime_fallback=True)

    assert "actual_device_runtime_us" not in annotated.global_events[0].extras
    assert "actual_runtime_measurement_kind" not in annotated.global_events[0].extras


def test_actual_device_runtime_metadata_does_not_change_provider_duration_selection():
    estimator = Estimator()
    estimator._stats[("cublasGemmEx", "blas_compute")] = {
        "p50": 22.0,
        "mean": 22.0,
        "p95": 22.0,
        "count": 1,
    }
    event = CollatedEvent(
        id="r0:e1",
        rank=0,
        ordinal=1,
        source=TraceSource.REAL,
        ts=2,
        pid=2,
        tid=3,
        module="libcublas.so.12",
        api="cublasGemmEx",
        op_type="blas_compute",
        extras={
            "actual_device_runtime_us": "9999.0",
            "actual_runtime_measurement_kind": "existing_cpp_event_async_runtime_observer",
            "m": "16",
            "n": "16",
            "k": "16",
        },
    )

    duration = estimate_low_level_event_us(event, estimator)

    assert duration == 22.0


def test_estimate_low_level_event_rejects_weak_collective_global_fallback_by_default():
    estimator = Estimator()
    event = CollatedEvent(
        id="r0:e2",
        rank=0,
        ordinal=2,
        source=TraceSource.REAL,
        ts=3,
        pid=2,
        tid=3,
        module="libnccl.so.2",
        api="ncclAllReduce",
        op_type="nccl_collective",
        extras={"count": "8", "datatype": "7"},
    )

    with pytest.raises(RuntimeError, match="Strict runtime estimation required for ncclAllReduce"):
        estimate_low_level_event_us(event, estimator)


def test_estimate_low_level_event_can_opt_in_to_weak_collective_global_fallback():
    estimator = Estimator()
    event = CollatedEvent(
        id="r0:e2",
        rank=0,
        ordinal=2,
        source=TraceSource.REAL,
        ts=3,
        pid=2,
        tid=3,
        module="libnccl.so.2",
        api="ncclAllReduce",
        op_type="nccl_collective",
        extras={"count": "8", "datatype": "7"},
    )

    duration = estimate_low_level_event_us(
        event,
        estimator,
        allow_weak_runtime_fallback=True,
    )

    assert duration == 1.0


def test_cuda_event_record_with_flags_is_treated_as_low_overhead():
    provider = _PayloadCheckingProvider()
    estimator = Estimator(providers=[provider])
    event = CollatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.REAL,
        ts=1,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaEventRecordWithFlags",
        op_type="stream_op",
        extras={"event_id": "evt-1", "stream_id": "s0", "flags": "0"},
    )

    assert is_low_overhead_api(event.api)
    assert estimate_low_level_event_us(event, estimator) == 1.0
    assert provider.seen is None


def test_cublas_set_stream_is_treated_as_low_overhead_even_with_estimator_stats():
    estimator = Estimator()
    estimator._stats[("cublasSetStream_v2", "stream_op")] = {
        "p50": 5.0,
        "mean": 5.0,
        "p95": 5.0,
        "count": 1,
    }
    event = CollatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.REAL,
        ts=1,
        pid=2,
        tid=3,
        module="libcublas.so.12",
        api="cublasSetStream_v2",
        op_type="stream_op",
        extras={"handle_id": "7", "stream_id": "s0"},
    )

    assert is_low_overhead_api(event.api)
    assert estimate_low_level_event_us(event, estimator) == 1.0


def test_cuda_event_query_is_low_overhead_event_wait_polling_api():
    estimator = Estimator()
    estimator._type_stats["stream_op"] = {
        "p50": 5.0,
        "mean": 5.0,
        "p95": 5.0,
        "count": 1,
    }
    event = CollatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.REAL,
        ts=1,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaEventQuery",
        op_type="stream_op",
        extras={"event_id": "evt-1"},
    )

    assert not is_ignorable_setup_event(event)
    assert is_low_overhead_api(event.api)
    assert estimate_low_level_event_us(event, estimator) == 1.0


def test_estimate_low_level_event_scales_down_kernel_fallback_for_fine_grained_launch_regime():
    estimator = Estimator()
    event = CollatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.REAL,
        ts=1,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={"kernel_id": "k1", "rank_kernel_launch_count": "4642"},
    )

    duration = estimate_low_level_event_us(
        event,
        estimator,
        allow_kernel_launch_heuristic_fallback=True,
    )

    assert duration == 1500.0


def test_annotate_collated_trace_preserves_rank_kernel_launch_count_context() -> None:
    estimator = Estimator()
    kernel0 = CollatedEvent(
        id="r0:e0",
        rank=0,
        ordinal=0,
        source=TraceSource.REAL,
        ts=1,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={"kernel_id": "k0"},
    )
    sync = CollatedEvent(
        id="r0:e1",
        rank=0,
        ordinal=1,
        source=TraceSource.REAL,
        ts=2,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaDeviceSynchronize",
        op_type="stream_op",
    )
    kernel1 = CollatedEvent(
        id="r0:e2",
        rank=0,
        ordinal=2,
        source=TraceSource.REAL,
        ts=3,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={"kernel_id": "k1"},
    )
    collated = CollatedTrace(
        trace_dir=Path("/tmp/rank-kernel-launch-count"),
        source=TraceSource.REAL,
        rank_events={0: (kernel0, sync, kernel1)},
        global_events=(kernel0, sync, kernel1),
        collective_groups={},
    )

    annotated = annotate_collated_trace(
        collated,
        estimator,
        allow_kernel_launch_heuristic_fallback=True,
    )

    assert [
        event.extras.get("rank_kernel_launch_count")
        for event in annotated.global_events
    ] == ["2", "2", "2"]


def test_default_estimator_is_not_calibrated_but_provider_backed_estimator_is():
    assert Estimator().is_calibrated() is False
    assert Estimator(providers=[_PayloadCheckingProvider()]).is_calibrated() is True


def test_estimate_low_level_event_routes_sync_ops_through_estimator():
    provider = _PayloadCheckingProvider()
    estimator = Estimator(providers=[provider])
    event = CollatedEvent(
        id="r0:e1",
        rank=0,
        ordinal=1,
        source=TraceSource.REAL,
        ts=2,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaStreamSynchronize",
        op_type="stream_op",
        extras={"stream_id": "s1"},
    )

    duration = estimate_low_level_event_us(event, estimator)

    assert is_low_overhead_api(event.api) is False
    assert duration == 7.0
    assert provider.seen is not None


def test_estimate_low_level_event_prefers_observed_wrapper_duration_for_blocking_sync():
    provider = _PayloadCheckingProvider()
    estimator = Estimator(providers=[provider])
    event = CollatedEvent(
        id="r0:e1",
        rank=0,
        ordinal=1,
        source=TraceSource.REAL,
        ts=2,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaStreamSynchronize",
        op_type="stream_op",
        extras={"stream_id": "s1", "host_duration_us": "42.5"},
    )

    annotated = annotate_collated_trace(
        CollatedTrace(
            trace_dir=Path("/tmp/observed-wrapper-sync"),
            source=TraceSource.REAL,
            rank_events={0: (event,)},
            global_events=(event,),
            collective_groups={},
        ),
        estimator,
    )

    assert annotated.global_events[0].duration_us == 42.5
    assert annotated.global_events[0].duration_source == "observed_wrapper_duration"
    assert provider.seen is None


def test_estimate_low_level_event_prefers_observed_wrapper_duration_for_nccl_comm_query():
    provider = _PayloadCheckingProvider()
    estimator = Estimator(providers=[provider])
    event = CollatedEvent(
        id="r0:e_nccl_query",
        rank=0,
        ordinal=1,
        source=TraceSource.REAL,
        ts=2,
        pid=2,
        tid=3,
        module="libnccl.so.2",
        api="ncclCommCount",
        op_type="other",
        extras={"comm_id": "comm-a", "host_duration_us": "31.5"},
    )

    annotated = annotate_collated_trace(
        CollatedTrace(
            trace_dir=Path("/tmp/observed-wrapper-nccl-query"),
            source=TraceSource.REAL,
            rank_events={0: (event,)},
            global_events=(event,),
            collective_groups={},
        ),
        estimator,
    )

    assert annotated.global_events[0].duration_us == 31.5
    assert annotated.global_events[0].duration_source == "observed_wrapper_duration"
    assert provider.seen is None


def test_estimate_low_level_event_zeroes_stream_create_setup():
    provider = _PayloadCheckingProvider()
    estimator = Estimator(providers=[provider])
    event = CollatedEvent(
        id="r0:e2",
        rank=0,
        ordinal=2,
        source=TraceSource.REAL,
        ts=3,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaStreamCreateWithPriority",
        op_type="stream_op",
    )

    duration = estimate_low_level_event_us(event, estimator)

    assert duration == 0.0
    assert provider.seen is None


def test_estimate_low_level_event_keeps_setup_api_ignored_even_with_host_duration():
    provider = _PayloadCheckingProvider()
    estimator = Estimator(providers=[provider])
    event = CollatedEvent(
        id="r0:e2",
        rank=0,
        ordinal=2,
        source=TraceSource.REAL,
        ts=3,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaGetDevice",
        op_type="context_op",
        extras={"host_duration_us": "19.0"},
    )

    annotated = annotate_collated_trace(
        CollatedTrace(
            trace_dir=Path("/tmp/ignored-setup-with-host-duration"),
            source=TraceSource.REAL,
            rank_events={0: (event,)},
            global_events=(event,),
            collective_groups={},
        ),
        estimator,
    )

    assert annotated.global_events[0].duration_us == 0.0
    assert annotated.global_events[0].duration_source == "ignored_setup"
    assert provider.seen is None


def test_estimate_low_level_event_uses_size_aware_memcpy_model():
    provider = _PayloadCheckingProvider()
    estimator = Estimator(providers=[provider])
    event = CollatedEvent(
        id="r0:e3",
        rank=0,
        ordinal=3,
        source=TraceSource.REAL,
        ts=4,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaMemcpyAsync",
        op_type="mem_copy",
        extras={"bytes": "3072", "kind": "3"},
    )

    duration = estimate_low_level_event_us(event, estimator)

    assert 3.0 <= duration < 4.0
    assert provider.seen is None


def test_estimate_low_level_event_prefers_observed_wrapper_duration_for_blocking_memcpy():
    provider = _PayloadCheckingProvider()
    estimator = Estimator(providers=[provider])
    event = CollatedEvent(
        id="r0:e3",
        rank=0,
        ordinal=3,
        source=TraceSource.REAL,
        ts=4,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaMemcpy",
        op_type="mem_copy",
        extras={"bytes": "3072", "kind": "1", "host_duration_us": "88.0"},
    )

    annotated = annotate_collated_trace(
        CollatedTrace(
            trace_dir=Path("/tmp/observed-wrapper-memcpy"),
            source=TraceSource.REAL,
            rank_events={0: (event,)},
            global_events=(event,),
            collective_groups={},
        ),
        estimator,
    )

    assert annotated.global_events[0].duration_us == 88.0
    assert annotated.global_events[0].duration_source == "observed_wrapper_duration"
    assert provider.seen is None


def test_estimate_low_level_event_uses_size_aware_malloc_model():
    provider = _PayloadCheckingProvider()
    estimator = Estimator(providers=[provider])
    event = CollatedEvent(
        id="r0:e4",
        rank=0,
        ordinal=4,
        source=TraceSource.REAL,
        ts=5,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaMalloc",
        op_type="mem_alloc",
        extras={"bytes": "20971520"},
    )

    duration = estimate_low_level_event_us(event, estimator)

    assert 25.0 < duration < 30.0
    assert provider.seen is None


def test_estimate_low_level_event_keeps_async_memcpy_on_model_path_even_with_host_duration():
    provider = _PayloadCheckingProvider()
    estimator = Estimator(providers=[provider])
    event = CollatedEvent(
        id="r0:e4",
        rank=0,
        ordinal=4,
        source=TraceSource.REAL,
        ts=5,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaMemcpyAsync",
        op_type="mem_copy",
        extras={"bytes": "3072", "kind": "3", "host_duration_us": "88.0"},
    )

    annotated = annotate_collated_trace(
        CollatedTrace(
            trace_dir=Path("/tmp/async-memcpy-model-path"),
            source=TraceSource.REAL,
            rank_events={0: (event,)},
            global_events=(event,),
            collective_groups={},
        ),
        estimator,
    )

    assert annotated.global_events[0].duration_us > 0.0
    assert annotated.global_events[0].duration_us != 88.0
    assert annotated.global_events[0].duration_source == "heuristic_memory_model"
    assert provider.seen is None


def test_estimate_low_level_event_keeps_async_memcpy_on_model_path_with_explicit_dispatch_only_contract():
    provider = _PayloadCheckingProvider()
    estimator = Estimator(providers=[provider])
    event = CollatedEvent(
        id="r0:e4_dispatch_only",
        rank=0,
        ordinal=4,
        source=TraceSource.REAL,
        ts=5,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaMemcpyAsync",
        op_type="mem_copy",
        extras={
            "bytes": "3072",
            "kind": "3",
            "host_duration_us": "88.0",
            "wrapper_runtime_contract": "dispatch_only",
        },
    )

    annotated = annotate_collated_trace(
        CollatedTrace(
            trace_dir=Path("/tmp/async-memcpy-dispatch-only-model-path"),
            source=TraceSource.REAL,
            rank_events={0: (event,)},
            global_events=(event,),
            collective_groups={},
        ),
        estimator,
    )

    assert annotated.global_events[0].duration_us > 0.0
    assert annotated.global_events[0].duration_us != 88.0
    assert annotated.global_events[0].duration_source == "heuristic_memory_model"
    assert provider.seen is None


def test_estimate_low_level_event_keeps_async_alloc_on_model_path_with_explicit_dispatch_only_contract():
    provider = _PayloadCheckingProvider()
    estimator = Estimator(providers=[provider])
    event = CollatedEvent(
        id="r0:e_async_alloc_dispatch_only",
        rank=0,
        ordinal=5,
        source=TraceSource.REAL,
        ts=6,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaMallocAsync",
        op_type="mem_alloc",
        extras={
            "bytes": "20971520",
            "host_duration_us": "88.0",
            "wrapper_runtime_contract": "dispatch_only",
            "stream_id": "5",
        },
    )

    annotated = annotate_collated_trace(
        CollatedTrace(
            trace_dir=Path("/tmp/async-alloc-dispatch-only-model-path"),
            source=TraceSource.REAL,
            rank_events={0: (event,)},
            global_events=(event,),
            collective_groups={},
        ),
        estimator,
    )

    assert annotated.global_events[0].duration_us > 0.0
    assert annotated.global_events[0].duration_us != 88.0
    assert annotated.global_events[0].duration_source == "heuristic_memory_model"
    assert provider.seen is None


def test_estimate_low_level_event_keeps_event_record_low_overhead_with_explicit_dispatch_only_contract():
    provider = _PayloadCheckingProvider()
    estimator = Estimator(providers=[provider])
    event = CollatedEvent(
        id="r0:e_event_record_dispatch_only",
        rank=0,
        ordinal=6,
        source=TraceSource.REAL,
        ts=7,
        pid=2,
        tid=3,
        module="libcudart.so.12",
        api="cudaEventRecord",
        op_type="stream_op",
        extras={
            "event_id": "9",
            "stream_id": "5",
            "host_duration_us": "17.0",
            "wrapper_runtime_contract": "dispatch_only",
        },
    )

    annotated = annotate_collated_trace(
        CollatedTrace(
            trace_dir=Path("/tmp/event-record-dispatch-only-low-overhead"),
            source=TraceSource.REAL,
            rank_events={0: (event,)},
            global_events=(event,),
            collective_groups={},
        ),
        estimator,
    )

    assert annotated.global_events[0].duration_us == 1.0
    assert annotated.global_events[0].duration_source == "low_overhead_api"
    assert provider.seen is None


def test_estimate_low_level_event_keeps_cublas_set_stream_low_overhead_with_explicit_dispatch_only_contract():
    provider = _PayloadCheckingProvider()
    estimator = Estimator(providers=[provider])
    event = CollatedEvent(
        id="r0:e_cublas_set_stream_dispatch_only",
        rank=0,
        ordinal=7,
        source=TraceSource.REAL,
        ts=8,
        pid=2,
        tid=3,
        module="libcublas.so.12",
        api="cublasSetStream_v2",
        op_type="stream_op",
        extras={
            "handle_id": "17",
            "stream_id": "5",
            "host_duration_us": "19.0",
            "wrapper_runtime_contract": "dispatch_only",
        },
    )

    annotated = annotate_collated_trace(
        CollatedTrace(
            trace_dir=Path("/tmp/cublas-set-stream-dispatch-only-low-overhead"),
            source=TraceSource.REAL,
            rank_events={0: (event,)},
            global_events=(event,),
            collective_groups={},
        ),
        estimator,
    )

    assert annotated.global_events[0].duration_us == 1.0
    assert annotated.global_events[0].duration_source == "low_overhead_api"
    assert provider.seen is None


def test_export_predicted_provider_rows_includes_join_metadata():
    estimator = Estimator(providers=[_ProviderCounterpartProvider()])
    event = CollatedEvent(
        id="r0:e1",
        rank=0,
        ordinal=1,
        source=TraceSource.FAKE,
        ts=10,
        pid=1,
        tid=2,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={
            "kernel": "kernel_a",
            "launch_grid": "2x3x4",
            "launch_block": "5x6x7",
            "launch_shared_mem_bytes": 128,
            "launch_stream_id": "9",
            "paper_valid_step_window_id": "step0",
            "window_start_us": "10.0",
            "window_end_us": "20.0",
        },
    )
    collated = CollatedTrace(
        trace_dir=Path("/tmp/provider-export"),
        source=TraceSource.FAKE,
        rank_events={0: (event,)},
        global_events=(event,),
        collective_groups={},
    )

    annotated = annotate_collated_trace(collated, estimator)
    before = (annotated.global_events[0].duration_us, annotated.global_events[0].duration_source)
    rows = export_predicted_provider_rows(annotated)
    after = (annotated.global_events[0].duration_us, annotated.global_events[0].duration_source)

    assert before == after == (9.0, "estimator_provider:counterpart_provider")
    assert len(rows) == 1
    row = rows[0]
    assert row["duration_us"] == 9.0
    assert row["duration_source"] == "estimator_provider:counterpart_provider"
    assert row["provider_duration_source_expected"] == "estimator_provider:counterpart_provider"
    assert row["rank"] == 0
    assert row["ordinal"] == 1
    assert row["event_id"] == "r0:e1"
    assert row["api"] == "cudaLaunchKernel"
    assert row["op_type"] == "kernel_launch"
    assert row["stream"] == "9"
    assert row["material_api"] == "cudaLaunchKernel"
    assert row["material_signature"] == "kernel=kernel_a;grid=2x3x4;block=5x6x7;shared_mem=128;stream=9"
    assert row["provider_counterpart_key"].startswith("rank=0|api=cudaLaunchKernel|ordinal=1|")
    assert row["paper_valid_step_window_id"] == "step0"
    assert row["window_start_us"] == 10.0
    assert row["window_end_us"] == 20.0
    assert row["provider_ordinal_within_rank_window_api_signature"] == 0


def test_export_predicted_provider_rows_increments_provider_ordinal():
    estimator = Estimator(providers=[_ProviderCounterpartProvider()])
    events = tuple(
        CollatedEvent(
            id=f"r0:e{i}",
            rank=0,
            ordinal=i,
            source=TraceSource.FAKE,
            ts=10 + i,
            pid=1,
            tid=2,
            module="libcublas.so.12",
            api="cublasGemmEx",
            op_type="blas_compute",
            extras={"m": "16", "n": "32", "k": "64", "stream_id": "3", "paper_valid_step_window_id": "w0"},
        )
        for i in (1, 2)
    )
    collated = CollatedTrace(
        trace_dir=Path("/tmp/provider-export-ordinal"),
        source=TraceSource.FAKE,
        rank_events={0: events},
        global_events=events,
        collective_groups={},
    )

    rows = export_predicted_provider_rows(annotate_collated_trace(collated, estimator))

    assert [row["provider_ordinal_within_rank_window_api_signature"] for row in rows] == [0, 1]
    assert [row["ordinal"] for row in rows] == [1, 2]


def test_export_predicted_provider_rows_marks_unavailable_signature():
    estimator = Estimator(providers=[_ProviderCounterpartProvider()])
    event = CollatedEvent(
        id="r1:e5",
        rank=1,
        ordinal=5,
        source=TraceSource.FAKE,
        ts=10,
        pid=1,
        tid=2,
        module="libcudart.so.12",
        api="cudaLaunchKernel",
        op_type="kernel_launch",
        extras={},
    )
    collated = CollatedTrace(
        trace_dir=Path("/tmp/provider-export-unavailable"),
        source=TraceSource.FAKE,
        rank_events={1: (event,)},
        global_events=(event,),
        collective_groups={},
    )

    rows = export_predicted_provider_rows(annotate_collated_trace(collated, estimator))

    assert rows[0]["material_signature"] == "signature:unavailable"
    assert rows[0]["provider_counterpart_key"].endswith("signature:unavailable")
    assert rows[0]["paper_valid_step_window_id"] is None
    assert rows[0]["window_start_us"] is None
    assert rows[0]["window_end_us"] is None


def test_export_predicted_provider_rows_does_not_change_duration_selection():
    estimator = Estimator(providers=[_ProviderCounterpartProvider()])
    event = CollatedEvent(
        id="r2:e4",
        rank=2,
        ordinal=4,
        source=TraceSource.FAKE,
        ts=10,
        pid=1,
        tid=2,
        module="libcublas.so.12",
        api="cublasGemmEx",
        op_type="blas_compute",
        extras={"m": "8", "n": "8", "k": "8", "stream_id": "0"},
    )
    collated = CollatedTrace(
        trace_dir=Path("/tmp/provider-export-no-change"),
        source=TraceSource.FAKE,
        rank_events={2: (event,)},
        global_events=(event,),
        collective_groups={},
    )

    annotated = annotate_collated_trace(collated, estimator)
    original_event = annotated.global_events[0]
    original_duration = original_event.duration_us
    original_source = original_event.duration_source
    original_extras = dict(original_event.extras)

    rows = export_predicted_provider_rows(annotated)

    assert len(rows) == 1
    assert annotated.global_events[0].duration_us == original_duration == 9.0
    assert annotated.global_events[0].duration_source == original_source == "estimator_provider:counterpart_provider"
    assert annotated.global_events[0].extras == original_extras


def _single_event_trace(event: CollatedEvent) -> CollatedTrace:
    return CollatedTrace(
        trace_dir=Path("/tmp/cublas-material-signature"),
        source=event.source,
        rank_events={event.rank: (event,)},
        global_events=(event,),
        collective_groups={},
    )


def test_cublas_gemmex_material_signature_canonicalizes_compute_type_without_stream():
    estimator = Estimator(providers=[_ProviderCounterpartProvider()])
    predicted = CollatedEvent(
        id="r0:gemm",
        rank=0,
        ordinal=1,
        source=TraceSource.FAKE,
        ts=10,
        pid=1,
        tid=2,
        module="libcublas.so.12",
        api="cublasGemmEx",
        op_type="blas_compute",
        extras={
            "m": 768,
            "n": 3072,
            "k": 512,
            "lda": 512,
            "ldb": 512,
            "ldc": 768,
            "computeType": 68,
            "algo": 99,
            "stream_id": 0,
            "transa": 1,
            "transb": 0,
            "material_signature": "stream=0;m=768;n=3072;k=512",
        },
    )
    actual = CollatedEvent(
        id="r0:actual-gemm",
        rank=0,
        ordinal=2,
        source=TraceSource.REAL,
        ts=11,
        pid=1,
        tid=2,
        module="libcublas.so.12",
        api="cublasGemmEx",
        op_type="blas_compute",
        extras={
            "observed_runtime_us": "12.5",
            "wrapper_runtime_contract": "async_runtime",
            "runtime_observation_source": "capture_real_cuda_event",
            "m": "768",
            "n": "3072",
            "k": "512",
            "lda": "512",
            "ldb": "512",
            "ldc": "768",
            "computeType": "68",
            "algorithm": "99",
            "transa": "1",
            "transb": "0",
            "material_signature": "m=768;n=3072;k=512;computeType=68",
        },
    )

    predicted_extras = annotate_collated_trace(_single_event_trace(predicted), estimator).global_events[0].extras
    actual_extras = annotate_collated_trace(_single_event_trace(actual), estimator).global_events[0].extras

    assert predicted_extras["material_signature"] == actual_extras["material_signature"]
    assert predicted_extras["material_signature"] == (
        "m=768;n=3072;k=512;lda=512;ldb=512;ldc=768;"
        "compute_type=68;transa=1;transb=0;algorithm=99"
    )
    assert "algo=" not in predicted_extras["material_signature"]
    assert "stream=" not in predicted_extras["material_signature"]
    assert predicted_extras["material_signature_inputs"]["compute_type"] == "68"
    assert predicted_extras["material_signature_inputs"]["algorithm"] == "99"
    assert "stream_id" not in predicted_extras["material_signature_inputs"]
    assert predicted_extras["raw_stream_id"] == "0"
    assert predicted_extras["canonical_stream_id"] == "0"


def test_cublas_strided_batched_signature_collapses_aliases_and_strides():
    estimator = Estimator(providers=[_ProviderCounterpartProvider()])
    event = CollatedEvent(
        id="r0:strided",
        rank=0,
        ordinal=3,
        source=TraceSource.FAKE,
        ts=10,
        pid=1,
        tid=2,
        module="libcublas.so.12",
        api="cublasGemmStridedBatchedEx",
        op_type="blas_compute",
        extras={
            "m": 256,
            "n": 256,
            "k": 64,
            "lda": 256,
            "ldb": 64,
            "ldc": 256,
            "batch_count": 48,
            "batchCount": "48",
            "strideA": "16384",
            "strideB": 16384,
            "strideC": 65536,
            "computeType": 68,
            "algo": 23,
            "stream_id": "0",
            "transa": 0,
            "transb": 0,
        },
    )

    extras = annotate_collated_trace(_single_event_trace(event), estimator).global_events[0].extras
    signature = extras["material_signature"]

    assert signature == (
        "m=256;n=256;k=64;lda=256;ldb=64;ldc=256;batch_count=48;"
        "stride_a=16384;stride_b=16384;stride_c=65536;"
        "compute_type=68;transa=0;transb=0;algorithm=23"
    )
    assert signature.count("batch_count=") == 1
    assert "batchCount=" not in signature
    assert "strideA=" not in signature
    assert "algo=" not in signature
    assert extras["material_signature_inputs"]["stride_a"] == "16384"
    assert extras["material_signature_inputs"]["stride_b"] == "16384"
    assert extras["material_signature_inputs"]["stride_c"] == "65536"
    assert extras["material_signature_inputs"]["algorithm"] == "23"


def test_cublas_provider_ordinal_namespace_keeps_stream_separate_from_signature():
    estimator = Estimator(providers=[_ProviderCounterpartProvider()])
    events = tuple(
        CollatedEvent(
            id=f"r0:gemm-stream-{stream}",
            rank=0,
            ordinal=stream,
            source=TraceSource.FAKE,
            ts=10 + stream,
            pid=1,
            tid=2,
            module="libcublas.so.12",
            api="cublasGemmEx",
            op_type="blas_compute",
            extras={
                "m": 16,
                "n": 32,
                "k": 64,
                "computeType": 68,
                "stream_id": stream,
                "paper_valid_step_window_id": "w0",
            },
        )
        for stream in (0, 1)
    )
    collated = CollatedTrace(
        trace_dir=Path("/tmp/cublas-provider-stream-ordinal"),
        source=TraceSource.FAKE,
        rank_events={0: events},
        global_events=events,
        collective_groups={},
    )

    rows = export_predicted_provider_rows(annotate_collated_trace(collated, estimator))

    assert rows[0]["material_signature"] == rows[1]["material_signature"]
    assert "stream=" not in rows[0]["material_signature"]
    assert [row["stream"] for row in rows] == ["0", "1"]
    assert [row["provider_ordinal_within_rank_window_api_signature"] for row in rows] == [0, 0]


def test_provider_row_full_join_dataset_cublas_wrappers_use_canonical_gemm_signature():
    from paper.maya_lite.provider_row_full_join_dataset import (
        JOIN_FIELDS,
        cublas_stream_from_handle_state,
        material_signature_from_record,
        material_signature_inputs_from_record,
        ordinalize,
        update_cublas_handle_stream_state,
    )

    assert "stream" in JOIN_FIELDS
    predicted = {
        "api": "cublasGemmEx",
        "m": 16,
        "n": 32,
        "k": 64,
        "computeType": 68,
        "algo": 7,
        "stream_id": 0,
    }
    actual = {
        "api": "cublasGemmEx",
        "m": "16",
        "n": "32",
        "k": "64",
        "computeType": "68",
        "algorithm": "7",
    }
    assert material_signature_from_record(predicted) == material_signature_from_record(actual)
    assert material_signature_inputs_from_record(predicted)["compute_type"] == "68"
    assert material_signature_inputs_from_record(predicted)["algorithm"] == "7"
    assert "stream_id" not in material_signature_inputs_from_record(predicted)

    rows = [
        {"rank": 0, "paper_valid_step_window_id": "w0", "stream": "0", "api": "cublasGemmEx", "material_signature": material_signature_from_record(predicted)},
        {"rank": 0, "paper_valid_step_window_id": "w0", "stream": "1", "api": "cublasGemmEx", "material_signature": material_signature_from_record(predicted)},
    ]
    ordinalize(rows)
    assert [row["provider_ordinal_within_rank_window_api_signature"] for row in rows] == [0, 0]

    handle_streams: dict[str, str] = {}
    update_cublas_handle_stream_state(
        {"api": "cublasCreate_v2", "handle_id": "h0"},
        handle_streams,
    )
    assert cublas_stream_from_handle_state(
        {"api": "cublasGemmEx", "handle_id": "h0"},
        handle_streams,
    ) == "0"

    update_cublas_handle_stream_state(
        {"api": "cublasSetStream_v2", "handle_id": "h0", "stream_id": "7"},
        handle_streams,
    )
    assert cublas_stream_from_handle_state(
        {"api": "cublasGemmEx", "handle_id": "h0"},
        handle_streams,
    ) == "7"
    assert cublas_stream_from_handle_state(
        {"api": "cublasGemmEx", "handle_id": "missing"},
        handle_streams,
    ) is None

    update_cublas_handle_stream_state(
        {"api": "cublasDestroy_v2", "handle_id": "h0"},
        handle_streams,
    )
    assert cublas_stream_from_handle_state(
        {"api": "cublasGemmEx", "handle_id": "h0"},
        handle_streams,
    ) is None
