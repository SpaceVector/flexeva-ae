"""Regression checks for Table 4 selected capture and incremental feedback."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "FlexEva/flexmaya_ras/src"))

import flexmaya_ras as fm
from flexmaya_ras._flexmaya_ras import TraceSuffixBuilder
from flexmaya_ras.simulator import replay_trace_segment
from anchor_pipeline import OPTIMIZER, rebase_suffix, run_pipeline, join_graph, GRAPH_FIELDS


def test_selected_generation_precedes_independent_reference(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("optimizer = " + OPTIMIZER + "\n")
    args = SimpleNamespace(out_dir=tmp_path / "out", trace_root=tmp_path / "traces",
                           input_trace_root=None, reuse_existing_traces=False)
    prefix = fm.make_event("cudaEventRecord", "stream_op", rank=0, stream=1,
                           event_handle=7, timestamp_ns=10, code_partition="stage_000")
    old_tail = fm.make_event("old_kernel", "kernel_launch", rank=0, stream=1,
                             timestamp_ns=20, code_partition="stage_000_optimizer_step")
    new_tail = fm.make_event("cudaStreamWaitEvent", "stream_op", rank=0, stream=2,
                             wait_event_handle=7, timestamp_ns=2,
                             code_partition="stage_000_optimizer_step")
    calls = []

    def launch(options, output):
        phase = options.table4_phase
        calls.append(phase)
        if phase == "reference":
            published = json.loads((args.out_dir / "case/candidate.json").read_text())
            assert published["independent_reference"] is False
            assert published["reused_events"] == published["refreshed_events"] == 1
            assert published["candidate_feedback"]["completed_events"] == 2
        directory = options.capture_trace_dir
        directory.mkdir(parents=True)
        (directory / "rank_0_refresh.json").write_text(json.dumps(
            {"phase": phase, "entry_resources": "stream=1,event=7"}))
        return {"trace_dir": str(directory), "return_code": 0}

    def parse(directory, audit):
        if directory.parent.name == "anchor":
            return [prefix, old_tail]
        if directory.parent.name == "selective":
            return [new_tail]
        # Deliberately different reference: it must not repair the candidate.
        return [fm.make_event("wrong_reference", "kernel_launch", rank=0)]

    policy = frozenset({"cudaEventRecord"})
    _, reference, candidate, feedback, pipeline, _ = run_pipeline(
        args, "case", source, 1, {0: [0]}, launch, parse, projection_policies=(policy,))
    assert calls == ["anchor", "selective", "reference"]
    assert [event.api for event in candidate.events] == ["cudaEventRecord", "cudaStreamWaitEvent"]
    assert reference[0].api == "wrong_reference"
    assert any(edge.reason == "event_wait" for edge in candidate.edges)
    assert feedback.completed_events == 2 and not feedback.cycle_detected
    assert pipeline["selective_capture"] and pipeline["independent_reference"]
    assert prefix.timestamp_ns == 10
    with pytest.raises(ValueError, match="unselected"):
        rebase_suffix({0: 10}, [prefix, new_tail], 1)
    assert pipeline["candidate_graph_events_built"] == feedback.replayed_events == 1
    assert pipeline["prefix_graph_events_reused"] == feedback.reused_events == 1
    from measure_maya_megatron_fakecuda_similarity import projected_feedback_report
    with monkeypatch.context() as patch:
        def reject_full_replay(*args, **kwargs):
            raise AssertionError("candidate projection used full replay")
        patch.setattr(fm, "replay_trace_once", reject_full_replay)
        projection = projected_feedback_report(candidate, candidate, zero_duration_apis=policy)
    assert projection["relative_difference"] == 0
    assert candidate.projection_feedback[policy].replayed_events == 1


def test_suffix_continuation_reuses_prefix():
    def event(api, kind, rank, stream, ts, code, duration=1, **kw):
        return fm.make_event(api, kind, rank=rank, stream=stream, thread_id=rank + 1,
                             timestamp_ns=ts, code_partition=code, duration_hint_us=duration, **kw)
    prefix_raw = [
        event("copy", "mem_copy", 0, 1, 1, "prefix", 10),
        event("cudaEventRecord", "stream_op", 0, 1, 2, "prefix", event_handle=7),
        event("ncclAllReduce", "nccl_collective", 0, 1, 3, "prefix", 5, collective_group="all"),
        event("copy", "mem_copy", 1, 2, 1, "prefix", 30),
        event("cudaEventRecord", "stream_op", 1, 2, 2, "prefix", event_handle=9),
        event("ncclAllReduce", "nccl_collective", 1, 2, 3, "prefix", 5, collective_group="all"),
    ]
    suffix = [
        event("cudaStreamWaitEvent", "stream_op", 0, 3, 4, "optimizer", wait_event_handle=7),
        event("kernel", "kernel_launch", 0, 3, 5, "optimizer", 100),
        event("cudaEventRecord", "stream_op", 0, 3, 6, "optimizer", event_handle=8),
        event("cudaStreamWaitEvent", "stream_op", 0, 1, 7, "optimizer", wait_event_handle=8),
        event("kernel", "kernel_launch", 0, 1, 8, "optimizer", 20),
        event("cudaStreamWaitEvent", "stream_op", 1, 3, 4, "optimizer", wait_event_handle=9),
        event("kernel", "kernel_launch", 1, 3, 5, "optimizer", 3),
    ]
    builder = TraceSuffixBuilder(prefix_raw, {0: [0], 1: [1]})
    prefix_native = builder.prefix
    prefix = SimpleNamespace(
        **{key: tuple(getattr(prefix_native, key)) for key in GRAPH_FIELDS},
        dedup_groups=prefix_native.dedup_groups, logical_event_count=prefix_native.logical_event_count,
    )
    checkpoint = replay_trace_segment(prefix_native)
    original_finish = dict(checkpoint.finish_time)
    delta = builder.append(suffix)
    resumed = replay_trace_segment(delta, checkpoint=checkpoint)
    joined = join_graph(prefix, delta)
    full = fm.build_rank_grouped_trace_ras(prefix_raw + suffix, {0: [0], 1: [1]})
    expected = replay_trace_segment(full)

    def identities(trace):
        return {e.id: (e.rank, e.timestamp_ns, e.api) for e in trace.events}
    def finishes(trace, state):
        return {identities(trace)[key]: value for key, value in state.finish_time.items()}
    def edges(trace):
        ids = identities(trace)
        return {(ids[e.from_id], ids[e.to_id], e.reason) for e in trace.edges}
    def windows(trace):
        ids = identities(trace)
        return {(p.kind, p.code_partition, tuple(ids[key] for key in p.event_ids))
                for p in trace.sync_partitions}

    assert resumed.report.replayed_events == len(suffix)
    assert resumed.report.reused_events == len(prefix_raw)
    assert resumed.report.total_time_us == expected.report.total_time_us
    assert finishes(joined, resumed) == finishes(full, expected)
    assert edges(joined) == edges(full)
    assert windows(joined) == windows(full)
    assert next(iter(joined.events)) is prefix.events[0]
    policy = frozenset({"cudaEventRecord"})
    saved_projection = replay_trace_segment(
        prefix_native, use_duration_hints=False, zero_duration_apis=policy)
    projected = replay_trace_segment(
        delta, checkpoint=saved_projection, use_duration_hints=False,
        zero_duration_apis=policy)
    full_projection = replay_trace_segment(
        full, use_duration_hints=False, zero_duration_apis=policy)
    assert finishes(joined, projected) == finishes(full, full_projection)
    # A second candidate branches from the same immutable frontier.
    suffix[1].duration_hint_us = 200
    second = replay_trace_segment(builder.append(suffix), checkpoint=checkpoint)
    assert second.report.total_time_us == resumed.report.total_time_us + 100
    assert dict(checkpoint.finish_time) == original_finish
    for invalid in ([prefix_raw[0]], [event("ncclAllReduce", "nccl_collective", 0, 1, 9, "optimizer")]):
        with pytest.raises(ValueError):
            builder.append(invalid)
