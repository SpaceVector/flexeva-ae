from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import flexmaya_ras as fm


def test_shared_arena_builds_trace_without_jsonl():
    arena = fm.SharedEventArena.create("flexeva-test-" + uuid.uuid4().hex, 16, True)
    arena.append(
        fm.make_event(
            "cudaLaunchKernel",
            "kernel_launch",
            rank=0,
            stream=7,
            timestamp_ns=10,
            code_partition="attention",
        )
    )
    arena.append(
        fm.make_event(
            "cudaStreamSynchronize",
            "stream_op",
            rank=0,
            stream=7,
            timestamp_ns=20,
            blocking=True,
            code_partition="attention",
        )
    )

    trace = arena.build_trace_ras()
    assert [event.api for event in trace.events] == ["cudaLaunchKernel", "cudaStreamSynchronize"]
    assert any(lane.kind == "stream" and lane.stream == 7 for lane in trace.lanes)


def test_local_arena_builds_full_and_rank_grouped_traces_without_python_list():
    arena = fm.EventArena()
    for rank in range(2):
        arena.append(
            fm.make_event(
                "cudaLaunchKernel",
                "kernel_launch",
                rank=rank,
                stream=7,
                timestamp_ns=10 + rank,
                code_partition="attention",
            )
        )

    full = arena.build_trace_ras()
    compact = arena.build_rank_grouped_trace_ras({0: [0, 1]})

    assert arena.size() == 2
    assert len(full.events) == 2
    assert compact.deduplicated
    assert compact.logical_event_count == 2


def test_rank_grouped_trace_preserves_logical_event_count():
    raw = []
    for rank in range(4):
        raw.append(
            fm.make_event(
                "cudaLaunchKernel",
                "kernel_launch",
                rank=rank,
                stream=0,
                timestamp_ns=rank,
                code_partition="layer",
            )
        )
        raw.append(
            fm.make_event(
                "ncclAllReduce",
                "nccl_collective",
                rank=rank,
                stream=3,
                timestamp_ns=10 + rank,
                collective_group="tp",
                bytes=4096,
                code_partition="layer",
            )
        )

    full = fm.build_trace_ras(raw)
    compact = fm.build_rank_grouped_trace_ras(raw, {0: [0, 2], 1: [1, 3]})
    report = fm.replay_trace_once(compact)

    assert compact.deduplicated
    assert compact.logical_event_count == full.logical_event_count
    assert report.completed_events == len(compact.events)
    assert not report.cycle_detected


def test_active_lane_groups_follow_pipeline_stages():
    assert fm.megatron_pp_stage_active_ranks(16, 2, 8, 0) == (0, 8)
    assert fm.megatron_pp_stage_active_ranks(16, 2, 8, 7) == (7, 15)
    assert fm.megatron_tp_groups_for_stage(16, 2, 8, 0) == ((0, 8),)


def test_source_change_refreshes_only_edited_partition(tmp_path: Path):
    source = tmp_path / "model.py"
    source.write_text("first_a\nfirst_b\nsecond_a\nsecond_b\n", encoding="utf-8")
    spec = fm.FlexMayaWorkloadSpec(
        workload_id="partition-test",
        world_size=1,
        code_partitions=(
            fm.CodePartitionSpec("first", str(source), start_line=1, end_line=2, active_ranks=(0,)),
            fm.CodePartitionSpec("second", str(source), start_line=3, end_line=4, active_ranks=(0,)),
        ),
        rank_group_policy="none",
    )
    raw = [
        fm.make_event("first", "kernel_launch", rank=0, stream=0, timestamp_ns=1, code_partition="first"),
        fm.make_event("second", "kernel_launch", rank=0, stream=0, timestamp_ns=2, code_partition="second"),
    ]
    anchor = fm.init_anchor(spec, raw)

    source.write_text("first_a\nfirst_b\nsecond_a\nsecond_changed\n", encoding="utf-8")
    report = fm.evaluate_candidate(anchor, spec, raw)

    assert report.refresh_plan.changed_partitions == ("second",)


def test_patch_trace_replaces_only_selected_chunks():
    def event(api: str, partition: str, timestamp: int):
        return fm.make_event(
            api,
            "kernel_launch",
            rank=0,
            stream=7,
            timestamp_ns=timestamp,
            duration_hint_us=float(timestamp),
            code_partition=partition,
        )

    anchor = fm.build_trace_ras(
        [
            event("old-router-1", "router", 1),
            event("keep-1", "attention", 2),
            event("old-expert", "expert", 3),
            event("keep-2", "attention", 4),
            event("old-router-2", "router", 5),
        ]
    )
    delta = fm.build_trace_ras(
        [
            event("new-router-1", "router", 1),
            event("new-expert", "expert", 2),
            event("new-router-2", "router", 3),
        ]
    )

    patched = fm.patch_trace_code_partitions(anchor, delta, ["router", "expert"])

    assert [row.api for row in patched.events] == [
        "new-router-1",
        "keep-1",
        "new-expert",
        "keep-2",
        "new-router-2",
    ]
    assert fm.replay_trace_once(patched).cycle_detected is False


def test_patch_trace_uses_non_selected_events_as_chunk_boundaries():
    def event(api: str, partition: str, timestamp: int):
        return fm.make_event(
            api,
            "kernel_launch",
            rank=0,
            stream=7,
            timestamp_ns=timestamp,
            code_partition=partition,
        )

    anchor = fm.build_trace_ras(
        [
            event("old-backward-1", "attention_backward", 1),
            event("keep", "mlp_backward", 2),
            event("old-backward-2", "attention_backward", 3),
        ]
    )
    selective = fm.build_trace_ras(
        [
            event("new-backward-1", "attention_backward", 1),
            event("dependency", "attention_forward", 2),
            event("new-backward-2", "attention_backward", 3),
        ]
    )

    patched = fm.patch_trace_code_partitions(anchor, selective, ["attention_backward"])

    assert [row.api for row in patched.events] == ["new-backward-1", "keep", "new-backward-2"]


def test_real_grounding_clears_only_the_placeholder_fallback(tmp_path: Path):
    source = tmp_path / "partition.py"
    source.write_text("VERSION = 1\n", encoding="utf-8")
    spec = fm.FlexMayaWorkloadSpec(
        workload_id="grounded",
        world_size=1,
        code_partitions=(
            fm.CodePartitionSpec(
                "router",
                str(source),
                active_ranks=(0,),
                requires_grounding=True,
            ),
        ),
        rank_group_policy="none",
    )
    raw = [fm.make_event("old", "kernel_launch", code_partition="router")]
    anchor = fm.init_anchor(spec, raw)
    source.write_text("VERSION = 2\n", encoding="utf-8")

    plan = fm.plan_candidate_refresh(
        anchor,
        spec,
        anchor.trace,
        grounding_satisfied=True,
    )

    assert plan.changed_partitions == ("router",)
    assert plan.fallback_reasons == ()


def test_synthetic_example_runs_without_rank_jsonl(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "run_gpt_v1.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--synthetic",
            "--total-gpus",
            "4",
            "--tp",
            "2",
            "--pp",
            "2",
            "--dp",
            "1",
            "--num-layers",
            "2",
            "--micro-batches",
            "2",
            "--seq-len",
            "16",
            "--hidden-size",
            "32",
            "--out-dir",
            str(tmp_path),
        ],
        check=True,
        cwd=root,
        env=env,
    )
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["jsonl_files"] == []
    assert result["anchor"]["feedback"]["cycle_detected"] is False
