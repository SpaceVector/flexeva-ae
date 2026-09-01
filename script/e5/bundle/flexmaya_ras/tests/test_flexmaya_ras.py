from __future__ import annotations

from dataclasses import asdict
import json
import subprocess
import sys
import uuid
from pathlib import Path

import flexmaya_ras as fm


def test_shared_arena_records_without_jsonl_pipeline():
    arena = fm.SharedEventArena.create("flexmaya-test-" + uuid.uuid4().hex, 16, True)
    arena.append(
        fm.make_event(
            "cudaLaunchKernel",
            "kernel_launch",
            rank=0,
            stream=7,
            timestamp_ns=10,
            count=32,
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
    assert arena.size() == 2
    assert [event.api for event in trace.events] == ["cudaLaunchKernel", "cudaStreamSynchronize"]
    assert any(lane.kind == "stream" and lane.stream == 7 for lane in trace.lanes)
    assert "flexsim.maya_lite.collate" not in sys.modules


def test_lanes_are_host_stream_and_sync_is_partition_metadata():
    raw = [
        fm.make_event("cudaLaunchKernel", "kernel_launch", rank=0, stream=5, timestamp_ns=1),
        fm.make_event(
            "ncclAllReduce",
            "nccl_collective",
            rank=0,
            stream=5,
            timestamp_ns=2,
            collective_group="allreduce",
        ),
        fm.make_event(
            "ncclAllReduce",
            "nccl_collective",
            rank=1,
            stream=5,
            timestamp_ns=2,
            collective_group="allreduce",
        ),
        fm.make_event("cudaStreamSynchronize", "stream_op", rank=0, stream=5, timestamp_ns=3, blocking=True),
    ]

    trace = fm.build_trace_ras(raw)
    assert {lane.kind for lane in trace.lanes} == {"stream"}
    assert not any(lane.kind == "collective" for lane in trace.lanes)
    assert any(partition.kind == "collective" for partition in trace.sync_partitions)
    assert any(partition.kind == "stream_sync" for partition in trace.sync_partitions)


def test_planned_dedup_compact_collective_replay_uses_logical_weights():
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
    full_report = fm.replay_trace_once(full)
    compact_report = fm.replay_trace_once(compact)

    assert compact.deduplicated
    assert len(compact.events) == 4
    assert compact.logical_event_count == full.logical_event_count
    assert compact_report.completed_events == len(compact.events)
    assert not compact_report.cycle_detected


def test_selected_cxx_build_matches_filtered_rank_grouped_trace():
    raw = []
    for rank in range(4):
        raw.append(
            fm.make_event(
                "cudaLaunchKernel",
                "kernel_launch",
                rank=rank,
                stream=0,
                timestamp_ns=rank * 10 + 1,
                duration_hint_us=2.0,
                code_partition="routing",
            )
        )
        raw.append(
            fm.make_event(
                "cublasGemmEx",
                "blas_compute",
                rank=rank,
                stream=4,
                timestamp_ns=rank * 10 + 2,
                duration_hint_us=5.0,
                code_partition="expert_compute",
            )
        )

    groups = {0: [0, 2], 1: [1, 3]}
    full = fm.build_rank_grouped_trace_ras(raw, groups)
    routing_windows = [
        int(partition.id)
        for partition in full.sync_partitions
        if str(partition.kind) == "trace_window" and str(partition.code_partition) == "routing"
    ]
    filtered = fm.filter_trace_partitions(full, routing_windows)
    selected = fm.build_rank_grouped_selected_trace_ras(raw, groups, [0, 1], ["routing"])

    def signature(trace: object) -> list[tuple[str, str, int, str, int]]:
        return [
            (str(event.api), str(event.kind), int(event.rank), str(event.code_partition), int(event.dedup_weight))
            for event in trace.events
        ]

    assert signature(selected) == signature(filtered)
    assert selected.logical_event_count == filtered.logical_event_count
    assert fm.replay_trace_once(selected).total_time_us == fm.replay_trace_once(filtered).total_time_us


def test_megatron_stage_active_lane_sets_are_not_per_rank_event_copies():
    assert fm.megatron_pp_stage_active_ranks(16, 2, 8, 0) == (0, 8)
    assert fm.megatron_pp_stage_active_ranks(16, 2, 8, 7) == (7, 15)
    assert fm.megatron_pp_stage_active_ranks(16, 4, 4, 1) == (1, 5, 9, 13)
    assert fm.megatron_tp_groups_for_stage(16, 2, 8, 0) == ((0, 8),)
    spec = fm.FlexMayaWorkloadSpec(
        workload_id="unit",
        world_size=16,
        tp=2,
        pp=8,
        dp=1,
        code_partitions=tuple(
            fm.CodePartitionSpec(f"layer_{stage}", __file__, active_ranks=fm.megatron_pp_stage_active_ranks(16, 2, 8, stage))
            for stage in range(8)
        ),
    )
    assert spec.rank_group_policy == "active_lane_set"
    assert fm.active_lane_rank_groups(spec) == {
        0: [0, 8],
        1: [1, 9],
        2: [2, 10],
        3: [3, 11],
        4: [4, 12],
        5: [5, 13],
        6: [6, 14],
        7: [7, 15],
    }


def test_anchor_context_is_serializable(tmp_path: Path):
    src = tmp_path / "model.py"
    src.write_text("def route(x):\n    return x\n", encoding="utf-8")
    spec = fm.FlexMayaWorkloadSpec(
        workload_id="ctx",
        world_size=4,
        code_partitions=(
            fm.CodePartitionSpec("routing_g0", str(src), active_ranks=(0, 2)),
            fm.CodePartitionSpec("routing_g1", str(src), active_ranks=(1, 3)),
        ),
    )
    raw = [
        fm.make_event("cudaLaunchKernel", "kernel_launch", rank=0, stream=0, timestamp_ns=1, code_partition="routing_g0"),
        fm.make_event("cudaLaunchKernel", "kernel_launch", rank=1, stream=0, timestamp_ns=2, code_partition="routing_g1"),
    ]

    anchor = fm.init_anchor(spec, raw)
    context = fm.anchor_context(anchor)
    payload = {
        "spec": asdict(context.spec),
        "source_hashes": [asdict(item) for item in context.source_hashes],
        "rank_groups": {str(rep): list(ranks) for rep, ranks in context.rank_groups.items()},
        "summary": dict(context.summary),
    }

    encoded = json.dumps(payload, sort_keys=True)
    assert '"rank_groups"' in encoded
    assert context.rank_groups == {0: (0, 2), 1: (1, 3)}


def test_code_partition_lineage_and_refresh_plan(tmp_path: Path):
    anchor_src = tmp_path / "model.py"
    anchor_src.write_text("def layer(x):\n    return x + 1\n", encoding="utf-8")
    candidate_src = tmp_path / "model_candidate.py"
    candidate_src.write_text("def layer(x):\n    return x + 2\n", encoding="utf-8")
    anchor_spec = fm.FlexMayaWorkloadSpec(
        workload_id="unit",
        world_size=2,
        tp=1,
        pp=1,
        dp=2,
        code_partitions=(
            fm.CodePartitionSpec("layer", str(anchor_src), start_line=1, end_line=2, active_ranks=(0, 1)),
        ),
        rank_group_policy="none",
    )
    candidate_spec = fm.FlexMayaWorkloadSpec(
        workload_id="unit",
        world_size=2,
        tp=1,
        pp=1,
        dp=2,
        code_partitions=(
            fm.CodePartitionSpec("layer", str(candidate_src), start_line=1, end_line=2, active_ranks=(0, 1)),
        ),
        rank_group_policy="none",
    )
    raw = [
        fm.make_event("cudaLaunchKernel", "kernel_launch", rank=0, stream=1, timestamp_ns=1, code_partition="layer"),
        fm.make_event("cudaLaunchKernel", "kernel_launch", rank=1, stream=1, timestamp_ns=2, code_partition="layer"),
    ]

    anchor = fm.init_anchor(anchor_spec, raw)
    report = fm.evaluate_candidate(anchor, candidate_spec, raw)

    assert trace_lineage_upper_partitions(anchor.trace) == {"layer"}
    assert trace_lineage_lower_kinds(anchor.trace) == {"trace_window"}
    assert trace_lineage_lower_partitions(anchor.trace) <= trace_partition_ids(anchor.trace)
    assert report.refresh_plan.changed_partitions == ("layer",)
    assert report.refresh_plan.affected_trace_partitions == tuple(sorted(trace_lineage_lower_partitions(anchor.trace)))
    assert report.refresh_plan.affected_trace_partition_count == len(anchor.trace.lineage_edges)
    assert report.summary["trace"]["lineage_edge_count"] == 2


def test_lineage_points_to_partition_not_event_id():
    raw = [
        fm.make_event(
            "cudaStreamSynchronize",
            "stream_op",
            rank=0,
            stream=7,
            timestamp_ns=1,
            blocking=True,
            code_partition="layer",
        ),
    ]

    trace = fm.build_trace_ras(raw)
    lineage = list(trace.lineage_edges)

    assert len(lineage) == 1
    assert lineage[0].lower_partition_kind == "trace_window"
    assert int(lineage[0].lower_partition) in trace_partition_ids(trace)
    assert int(lineage[0].lower_partition) != int(trace.events[0].id)


def test_hook_default_code_partition_creates_lineage(monkeypatch):
    monkeypatch.setenv("FLEXMAYA_CODE_PARTITION", "runtime")
    fm.clear_hook_events()
    fm.record_hook_api("cudaLaunchKernel", "kernel_launch", rank=0, stream=1, timestamp_ns=1)
    fm.record_hook_api("ncclAllReduce", "nccl_collective", rank=0, stream=1, timestamp_ns=2)

    trace = fm.build_trace_ras(fm.hook_events())

    assert trace_lineage_upper_partitions(trace) == {"runtime"}
    assert trace_lineage_lower_kinds(trace) == {"trace_window"}
    assert trace_lineage_lower_partitions(trace) <= trace_partition_ids(trace)


def test_gpt_v1_synthetic_runner_writes_no_rank_jsonl(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_gpt_v1.py"
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
        cwd=Path(__file__).resolve().parents[1],
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["jsonl_files"] == []
    assert result["anchor"]["trace"]["dedup_group_count"] == 2
    assert result["anchor"]["feedback"]["cycle_detected"] is False


def test_moe_v2_case_variants_keep_base_candidate_semantics():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import run_moe_v2_matrix as matrix

    candidate = {
        "candidate_id": "layout_striped",
        "change_surface": "routing",
        "entry": __file__,
        "semantic_diffs": (),
    }
    variants = matrix.expand_candidate_variants([candidate], 2)
    assert variants[0]["candidate_id"] == "layout_striped__v00"
    assert variants[0]["base_candidate_id"] == "layout_striped"
    assert "memory_payload" in matrix.selected_code_partitions(variants[0])

    args = {
        "world_size": 4,
        "ep_group_size": 2,
        "micro_batches": 1,
        "layers": 1,
        "seq_len": 16,
        "hidden_size": 32,
    }
    groups = matrix.rank_groups(args["world_size"], args["ep_group_size"])
    anchor = {"candidate_id": "anchor_baseline", "entry": __file__, "change_surface": "anchor"}
    anchor_payload = matrix.build_anchor(args, anchor, groups)
    flex = matrix.evaluate_flexeva_selected_candidate(
        variants[0],
        args,
        groups,
        float(anchor_payload["anchor"].feedback.total_time_us),
    )

    full_raw = matrix.synthetic_moe_events(
        args,
        candidate_id=variants[0]["candidate_id"],
        base_candidate_id=variants[0]["base_candidate_id"],
        ranks=range(args["world_size"]),
    )
    assert "maya_full" not in flex
    assert flex["raw_event_count"] < len(full_raw)
    assert flex["predicted_candidate_total_runtime_us"] > 0


def test_parallel_strategy_change_selects_config_dependent_trace_windows(tmp_path: Path):
    src = tmp_path / "model.py"
    src.write_text("def layer(x):\n    return x\n", encoding="utf-8")
    anchor_spec = fm.FlexMayaWorkloadSpec(
        workload_id="unit",
        world_size=2,
        tp=2,
        pp=1,
        dp=1,
        code_partitions=(fm.CodePartitionSpec("layer", str(src), active_ranks=(0, 1)),),
        rank_group_policy="none",
    )
    candidate_spec = fm.FlexMayaWorkloadSpec(
        workload_id="unit",
        world_size=2,
        tp=1,
        pp=1,
        dp=2,
        code_partitions=(fm.CodePartitionSpec("layer", str(src), active_ranks=(0, 1)),),
        rank_group_policy="none",
    )
    raw = [
        fm.make_event("cudaLaunchKernel", "kernel_launch", rank=0, stream=1, timestamp_ns=1, code_partition="layer"),
        fm.make_event("cudaLaunchKernel", "kernel_launch", rank=1, stream=1, timestamp_ns=2, code_partition="layer"),
        fm.make_event("ncclAllReduce", "nccl_collective", rank=0, stream=1, timestamp_ns=3, collective_group="g", code_partition="layer"),
        fm.make_event("ncclAllReduce", "nccl_collective", rank=1, stream=1, timestamp_ns=4, collective_group="g", code_partition="layer"),
    ]
    anchor = fm.init_anchor(anchor_spec, raw)
    report = fm.evaluate_candidate(anchor, candidate_spec, raw)

    assert report.refresh_plan.configuration_changed
    assert report.refresh_plan.changed_partitions == ("__parallel_strategy__",)
    assert report.refresh_plan.refresh_scope == "config_trace_partitions"
    assert 0 < report.refresh_plan.affected_trace_partition_count < trace_window_count(report.trace)
    selected = set(report.refresh_plan.affected_trace_partitions)
    event_by_id = {int(event.id): event for event in report.trace.events}
    selected_windows = [
        partition
        for partition in report.trace.sync_partitions
        if int(partition.id) in selected and str(partition.kind) == "trace_window"
    ]
    reused_windows = [
        partition
        for partition in report.trace.sync_partitions
        if int(partition.id) not in selected and str(partition.kind) == "trace_window"
    ]
    assert all(
        any(str(event_by_id[int(event_id)].kind) == "nccl_collective" for event_id in partition.event_ids)
        for partition in selected_windows
    )
    assert any(
        any(str(event_by_id[int(event_id)].kind) == "kernel_launch" for event_id in partition.event_ids)
        for partition in reused_windows
    )
    assert report.feedback is report.selected_feedback
    assert 0 < len(report.selected_trace.events) < len(report.trace.events)
    assert {str(event.kind) for event in report.selected_trace.events} == {"nccl_collective"}


def test_config_reuse_measurement_script_reports_benefit(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "measure_config_reuse.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--world-size",
            "4",
            "--anchor-tp",
            "2",
            "--anchor-pp",
            "2",
            "--anchor-dp",
            "1",
            "--candidate",
            "1,4,1",
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
        cwd=Path(__file__).resolve().parents[1],
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    row = result["candidates"][0]
    assert row["refresh_plan"]["configuration_changed"] is True
    assert row["maya_full"]["trace"]["deduplicated"] is False
    assert row["maya_trace_ras"]["trace"]["deduplicated"] is True
    assert row["maya_dedup"]["trace"]["deduplicated"] is True
    assert row["flexeva_refresh"]["trace"]["deduplicated"] is True
    assert row["flexeva_refresh"]["diagnostic_full_feedback"] is None
    assert row["flexeva_selected_event_replay"]["selected_trace"]["event_count"] <= row["flexeva_refresh"]["trace"]["event_count"]
    assert row["benefit"]["raw_event_reduction"] >= 0.0
    assert row["benefit"]["selected_event_reduction_vs_refresh_trace"] >= 0.0
    assert row["benefit"]["flexeva_speedup_vs_maya_trace_ras"] > 0.0
    assert row["benefit"]["flexeva_selected_event_speedup_vs_maya_trace_ras"] > 0.0
    assert row["benefit"]["flexeva_speedup_vs_maya_dedup"] > 0.0
    assert row["benefit"]["candidate_speedup_vs_cold_full"] > 0.0


def test_moe_v2_matrix_requires_oracle(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_moe_v2_matrix.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--world-size",
            "4",
            "--ep-group-size",
            "2",
            "--layers",
            "1",
            "--micro-batches",
            "1",
            "--out-dir",
            str(tmp_path),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "--oracle-results" in result.stderr


def test_moe_v2_matrix_parallel_with_oracle(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_moe_v2_matrix.py"
    oracle = tmp_path / "oracle.json"
    oracle.write_text(
        json.dumps(
            {
                "case_id": "maya_v2_load_skew_r1",
                "round_id": "round_load_skew_r1",
                "metric": "per_step_runtime_us",
                "candidates": {
                    "anchor_baseline": {"status": "complete", "runtime_us": 1000.0},
                    "overflow_reroute": {"status": "complete", "runtime_us": 920.0},
                    "layout_striped": {"status": "complete", "runtime_us": 880.0},
                    "local_backup_reroute": {"status": "complete", "runtime_us": 850.0},
                    "balanced_secondary_route": {"status": "complete", "runtime_us": 900.0},
                },
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--world-size",
            "4",
            "--ep-group-size",
            "2",
            "--workers",
            "2",
            "--layers",
            "1",
            "--micro-batches",
            "1",
            "--seq-len",
            "8",
            "--hidden-size",
            "16",
            "--oracle-results",
            str(oracle),
            "--out-dir",
            str(tmp_path / "matrix"),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    result = json.loads((tmp_path / "matrix" / "matrix.json").read_text(encoding="utf-8"))

    assert result["designs"]["maya_full"].startswith("full synthetic hook capture")
    assert len(result["candidates"]) == 4
    assert result["oracle_fidelity"]["oracle_coverage"] == 1.0
    assert all(row["reuse"]["raw_event_reuse_rate"] > 0.0 for row in result["candidates"])
    assert (tmp_path / "matrix" / "summary.csv").exists()


def test_moe_v2_case_study_runner_writes_variant_outputs(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_moe_v2_case_study.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--world-size",
            "4",
            "--ep-group-size",
            "2",
            "--layers",
            "1",
            "--micro-batches",
            "1",
            "--seq-len",
            "8",
            "--hidden-size",
            "16",
            "--candidate-limit",
            "1",
            "--variants-per-candidate",
            "1",
            "--maya-trace-ras-limit",
            "1",
            "--memory-candidate-counts",
            "1",
            "--out-dir",
            str(tmp_path / "case"),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    out_dir = tmp_path / "case"
    result = json.loads((out_dir / "matrix.json").read_text(encoding="utf-8"))
    row = result["candidates"][0]

    assert row["candidate_id"] == "overflow_reroute__v00"
    assert row["base_candidate_id"] == "overflow_reroute"
    assert "dispatch_collective" in row["selected_code_partitions"]
    assert result["designs"]["maya_full"].startswith("measured exactly once")
    assert row["maya_trace_ras"] is not None
    assert (out_dir / "run_manifest.json").exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "memory_scaling.json").exists()
    assert (out_dir / "memory_scaling.csv").exists()
    assert (out_dir / "case_study_tables.md").exists()


def test_candidate_memory_script_writes_scaling_report(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "measure_candidate_memory.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--world-size",
            "4",
            "--ep-group-size",
            "2",
            "--layers",
            "1",
            "--micro-batches",
            "1",
            "--seq-len",
            "8",
            "--hidden-size",
            "16",
            "--candidate-counts",
            "1,2",
            "--out-dir",
            str(tmp_path / "memory"),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    result = json.loads((tmp_path / "memory" / "memory_scaling.json").read_text(encoding="utf-8"))

    assert result["configuration"]["measurement"].startswith("fresh subprocess")
    assert {row["mode"] for row in result["rows"]} == {"maya_full", "maya_trace_ras", "flexeva_selected"}
    assert result["summary"]["max_count"] == 2


def trace_lineage_upper_partitions(trace: object) -> set[str]:
    return {str(edge.upper_partition) for edge in trace.lineage_edges}


def trace_lineage_lower_kinds(trace: object) -> set[str]:
    return {str(edge.lower_partition_kind) for edge in trace.lineage_edges}


def trace_lineage_lower_partitions(trace: object) -> set[int]:
    return {int(edge.lower_partition) for edge in trace.lineage_edges}


def trace_partition_ids(trace: object) -> set[int]:
    return {int(partition.id) for partition in trace.sync_partitions}


def trace_window_count(trace: object) -> int:
    return sum(1 for partition in trace.sync_partitions if str(partition.kind) == "trace_window")
