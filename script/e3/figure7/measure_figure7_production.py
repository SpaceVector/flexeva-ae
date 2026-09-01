#!/usr/bin/env python3
"""Paper-setting Figure 7 on the reusable two-node production path."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import flexmaya_ras as fm

import production
from measure_maya_megatron_fakecuda_similarity import parse_case_raw_events_with_regions
from measure_megatron_trace_similarity import MegatronCase, event_signature


ANCHOR = MegatronCase(
    name="figure7_megatron_2p7b_tp1_pp8_dp2",
    parameter_scale="2.7B",
    steps=1,
    global_batch_size=512,
    seq_len=2048,
    hidden_size=2560,
    num_layers=32,
    num_heads=32,
    vocab_size=32000,
    tp=1,
    pp=8,
    dp=2,
    world_size=16,
    micro_batches=256,
    schedule="1f1b",
    dtype="bf16",
)
R4 = MegatronCase(
    **{
        **asdict(ANCHOR),
        "name": "figure7_megatron_2p7b_tp2_pp8_dp1",
        "tp": 2,
        "dp": 1,
        "micro_batches": 512,
    }
)
ROUNDS = (
    production.Round(1, "r1_attention_backward", "Attn", ("attention_backward",)),
    production.Round(
        2,
        "r2_attention_mlp_backward",
        "Attn+MLP",
        ("attention_backward", "mlp_backward"),
    ),
    production.Round(
        3,
        "r3_attention_mlp_optimizer",
        "Attn+MLP+Opt",
        ("attention_backward", "mlp_backward", "optimizer_step"),
    ),
    production.Round(
        4,
        "r4_attention_mlp_optimizer_tp_dp",
        "Attn+MLP+Opt+TP/DP",
        ("attention_backward", "mlp_backward", "optimizer_step"),
        topology_change=True,
    ),
)
SYSTEMS = (
    "Maya-style full",
    "Maya-style + FlexEva trace-RAS",
    "FlexEva cumulative",
)
PAPER_PHASES = (
    "maya_emulation_s",
    "trace_ras_compaction_s",
    "source_ras_update_s",
    "selective_emulation_s",
    "trace_patch_collation_s",
    "event_simulation_s",
)
PLOT_COMPONENTS = (
    ("maya_emulation_s", "Maya emu.", "#009E73", "////"),
    ("trace_ras_compaction_s", "Trace-RAS", "#CC79A7", "\\\\\\\\"),
    ("source_ras_update_s", "RAS update", "#E69F00", ""),
    ("selective_emulation_s", "Selective emu.", "#D55E00", "xxxx"),
    ("trace_patch_collation_s", "Trace patch", "#56B4E9", "----"),
    ("event_simulation_s", "Event sim.", "#F0E442", "xx"),
)


def process_wall_s(row: dict[str, object]) -> float:
    timing = row["timing"]
    elapsed_s = (int(timing["process_exit_ns"]) - int(timing["spawned_ns"])) / 1e9
    production.require(elapsed_s > 0.0, "full emulation process wall is invalid")
    return elapsed_s


def anchor_paper_phases(row: dict[str, object]) -> dict[str, float]:
    phases = row["phases_s"]
    return {
        "maya_emulation_s": process_wall_s(row),
        "trace_ras_compaction_s": float(phases["trace_ras_compaction_s"]),
        "source_ras_update_s": float(phases["code_analysis_s"])
        + float(phases["source_ras_update_s"]),
        "selective_emulation_s": 0.0,
        "trace_patch_collation_s": float(phases["trace_processing_s"]),
        "event_simulation_s": float(phases["event_simulation_s"])
        + float(phases["feedback_generation_s"]),
    }


def maya_paper_phases(row: dict[str, object]) -> dict[str, float]:
    phases = row["phases_s"]
    return {
        "maya_emulation_s": process_wall_s(row),
        "trace_ras_compaction_s": 0.0,
        "source_ras_update_s": 0.0,
        "selective_emulation_s": 0.0,
        "trace_patch_collation_s": float(phases["trace_processing_s"]),
        "event_simulation_s": float(phases["event_simulation_s"])
        + float(phases["feedback_generation_s"]),
    }


def hybrid_paper_phases(
    maya_row: dict[str, object], hybrid_row: dict[str, object]
) -> dict[str, float]:
    phases = hybrid_row["phases_s"]
    return {
        "maya_emulation_s": process_wall_s(maya_row),
        "trace_ras_compaction_s": float(phases["trace_ras_compaction_s"]),
        "source_ras_update_s": 0.0,
        "selective_emulation_s": 0.0,
        "trace_patch_collation_s": float(phases["trace_processing_s"]),
        "event_simulation_s": float(phases["event_simulation_s"])
        + float(phases["feedback_generation_s"]),
    }


def flexeva_paper_phases(row: dict[str, object]) -> dict[str, float]:
    phases = row["phases_s"]
    return {
        "maya_emulation_s": 0.0,
        "trace_ras_compaction_s": 0.0,
        "source_ras_update_s": float(phases["code_analysis_s"])
        + float(phases["source_ras_update_s"]),
        "selective_emulation_s": (
            process_wall_s(row)
            if bool(row["configuration_full_refresh"])
            else float(phases["selective_emulation_s"])
        ),
        "trace_patch_collation_s": float(phases["trace_processing_s"])
        + float(phases["trace_patch_collation_s"]),
        "event_simulation_s": float(phases["event_simulation_s"])
        + float(phases["feedback_generation_s"]),
    }


def config_for(round_: production.Round) -> MegatronCase:
    return R4 if round_.topology_change else ANCHOR


def spec_for(
    case: MegatronCase,
    sources: dict[str, Path],
    workload_id: str,
) -> fm.FlexMayaWorkloadSpec:
    return fm.FlexMayaWorkloadSpec(
        workload_id=workload_id,
        world_size=case.world_size,
        tp=case.tp,
        pp=case.pp,
        dp=case.dp,
        code_partitions=tuple(
            fm.CodePartitionSpec(
                production.stage_partition(stage, op),
                str(sources[production.stage_partition(stage, op)]),
                active_ranks=fm.megatron_pp_stage_active_ranks(
                    case.world_size,
                    case.tp,
                    case.pp,
                    stage,
                ),
                requires_grounding=True,
            )
            for stage in range(case.pp)
            for op in production.GPT_OPS
        ),
        rank_group_policy="active_lane_set",
        notes=("Figure 7 Megatron 2.7B two-node production",),
    )


def worker_record(worker: production.Worker) -> dict[str, object]:
    return {
        "command_node0": worker.local_command,
        "command_node1": worker.peer_command,
        "logs": {
            "stdout_node0": str(worker.case_dir / "stdout_node0.log"),
            "stderr_node0": str(worker.case_dir / "stderr_node0.log"),
            "stdout_node1": str(worker.case_dir / "stdout_node1.log"),
            "stderr_node1": str(worker.case_dir / "stderr_node1.log"),
            "environment_node0": str(worker.case_dir / "environment_node0.json"),
            "environment_node1": str(worker.case_dir / "environment_node1.json"),
        },
        "peak_rss_kib": production.peak_rss_payload(worker),
    }


def full_audit(
    trace: object,
    worker: production.Worker,
    world_size: int,
    raw_cuda_get_device_count: int,
) -> dict[str, object]:
    ranks = sorted({int(event.rank) for event in trace.events})
    production.require(
        ranks == list(range(world_size)),
        f"full trace does not cover ranks 0-{world_size - 1}: {ranks}",
    )
    return {
        **fm.trace_summary(trace),
        "trace_source": "rank JSONL",
        "jsonl_file_count": world_size,
        "jsonl_paths": jsonl_paths(worker, world_size),
        "binary_audit_source": "SharedEventArena",
        "binary_file_count": 2,
        "binary_paths": production.binary_paths(worker),
        "world_size": world_size,
        "rank_coverage": ranks,
        "raw_event_count": len(trace.events),
        "cudaGetDevice_raw_count": raw_cuda_get_device_count,
        "cudaGetDevice_modeled_count": 0,
        "model_excluded_apis": ["cudaGetDevice"],
    }


def binary_full_audit(
    trace: object,
    worker: production.Worker,
    world_size: int,
) -> dict[str, object]:
    return {
        **fm.trace_summary(trace),
        **production.binary_trace_audit(trace, worker, world_size),
    }


def grouped_audit(
    trace: object,
    *,
    binary_paths: list[str],
    jsonl_paths: list[str],
    raw_cuda_get_device_count: int,
    world_size: int,
) -> dict[str, object]:
    coverage = {int(event.rank) for event in trace.events}
    for group in trace.dedup_groups:
        coverage.update(int(rank) for rank in group.ranks)
    ranks = sorted(coverage)
    production.require(
        ranks == list(range(world_size)),
        f"grouped trace does not cover ranks 0-{world_size - 1}: {ranks}",
    )
    summary = fm.trace_summary(trace)
    production.require(summary["deduplicated"] is True, "trace-RAS output is not grouped")
    return {
        **summary,
        "trace_source": "rank JSONL",
        "binary_audit_source": "SharedEventArena",
        "binary_file_count": len(binary_paths),
        "binary_paths": binary_paths,
        "binary_paths_distinct": len(set(binary_paths)) == len(binary_paths),
        "jsonl_file_count": len(jsonl_paths),
        "jsonl_paths": jsonl_paths,
        "world_size": world_size,
        "rank_coverage": ranks,
        "raw_event_count": int(trace.logical_event_count),
        "cudaGetDevice_raw_count": raw_cuda_get_device_count,
        "cudaGetDevice_modeled_count": 0,
    }


def logical_counter(trace: object, partitions: tuple[str, ...] = ()) -> Counter:
    selected = set(partitions)
    counter: Counter = Counter()
    for event in trace.events:
        if str(event.kind) not in production.KEY_EVENT_KINDS:
            continue
        if selected and str(event.code_partition) not in selected:
            continue
        counter[event_signature(event)] += int(getattr(event, "dedup_weight", 1) or 1)
    return counter


def counter_digest(counter: Counter) -> str:
    rows = [(list(key), value) for key, value in sorted(counter.items(), key=lambda row: repr(row[0]))]
    return production.canonical_json_hash(rows)


def grouped_trace(
    paths: list[str],
    spec: fm.FlexMayaWorkloadSpec,
) -> tuple[float, object]:
    groups = fm.active_lane_rank_groups(spec)
    return production.timed(
        lambda: fm.build_rank_grouped_trace_ras_from_binary(paths, groups)
    )


def jsonl_paths(worker: production.Worker, world_size: int) -> list[str]:
    paths = [worker.trace_dir / f"rank_{rank}.jsonl" for rank in range(world_size)]
    production.require(
        all(path.is_file() and path.stat().st_size > 0 for path in paths),
        "paper Figure 7 rank JSONL trace is incomplete",
    )
    return [str(path) for path in paths]


def parse_jsonl(
    worker: production.Worker,
    case: MegatronCase,
) -> tuple[float, list[object], int]:
    jsonl_paths(worker, case.world_size)

    def parse() -> tuple[list[object], int]:
        raw = parse_case_raw_events_with_regions(case, worker.trace_dir)
        for event in raw:
            partition = str(event.code_partition)
            if partition in production.GPT_OPS:
                event.code_partition = (
                    f"stage_{int(event.rank) % case.pp:03d}_{partition}"
                )
        cuda_get_device_count = sum(str(event.api) == "cudaGetDevice" for event in raw)
        return [event for event in raw if str(event.api) != "cudaGetDevice"], cuda_get_device_count

    elapsed_s, parsed = production.timed(parse)
    raw_events, cuda_get_device_count = parsed
    return elapsed_s, raw_events, cuda_get_device_count


def grouped_raw_trace(
    raw_events: list[object],
    spec: fm.FlexMayaWorkloadSpec,
) -> tuple[float, object]:
    groups = fm.active_lane_rank_groups(spec)
    return production.timed(lambda: fm.build_rank_grouped_trace_ras(raw_events, groups))


def collate_candidate_trace(
    anchor_trace: object,
    replacement_trace: object,
    candidate_spec: fm.FlexMayaWorkloadSpec,
    plan: fm.FlexMayaRefreshPlan,
    selected: tuple[str, ...],
) -> tuple[object, dict[str, object]]:
    if not plan.configuration_changed:
        return production.collate_selective_trace(
            anchor_trace, replacement_trace, candidate_spec, plan, selected
        )
    candidate_partitions = {partition.partition_id for partition in candidate_spec.code_partitions}
    production.require(
        set(selected) == candidate_partitions,
        "topology rebase did not select every executable partition",
    )
    return replacement_trace, {
        "mode": "configuration_rebase",
        "method": "replace the old-topology anchor trace with the complete candidate-topology trace",
        "full_executable_partition_coverage": True,
        "candidate_partition_count": len(candidate_partitions),
    }


def evaluate_maya(
    args,
    *,
    case: MegatronCase,
    round_: production.Round,
    candidate_spec: fm.FlexMayaWorkloadSpec,
    case_dir: Path,
    peer_case_dir: Path,
    seed: int,
    port: int,
) -> tuple[dict[str, object], object, object, list[object]]:
    args.jsonl_trace = True
    worker = production.spawn_worker(
        args,
        workload="gpt",
        system="maya-full",
        config=case,
        round_=round_,
        selected_ops=(),
        seed=seed,
        case_dir=case_dir,
        peer_case_dir=peer_case_dir,
        port=port,
        peer_run_id=f"{args.run_id}-maya-r{round_.number}",
    )
    try:
        production.wait_ready(worker, args.ready_timeout_s)
        started_ns = time.time_ns()
        started = time.perf_counter()
        phases = production.empty_phases()
        phases["maya_emulation_s"] = production.dispatch_and_wait(
            worker, args.worker_timeout_s
        )
        collect_s = production.collect_peer_traces(args, worker, case.world_size)
        parse_s, raw_events, cuda_get_device_count = parse_jsonl(worker, case)
        build_s, trace = production.timed(lambda: fm.build_trace_ras(raw_events))
        phases["trace_processing_s"] = collect_s + parse_s + build_s
        phases["event_simulation_s"], feedback = production.timed(
            lambda: fm.replay_trace_once(trace)
        )
        phases["feedback_generation_s"], feedback_payload = production.timed(
            feedback.to_dict
        )
        row = {
            "system": "maya-full",
            "round": round_.number,
            "round_name": round_.name,
            "mutation_label": round_.label,
            "candidate_config": production.config_payload("gpt", case),
            "candidate_manifest_spec": asdict(candidate_spec),
            "seed": seed,
            "executor_feedback": json.loads(
                worker.feedback_file.read_text(encoding="utf-8")
            ),
            "feedback": feedback_payload,
            "phases_s": phases,
            "phase_detail_s": {
                "trace_collection_s": collect_s,
                "jsonl_parse_s": parse_s,
                "ordinary_full_trace_build_s": build_s,
            },
            **worker_record(worker),
        }
        production.finish_timing(row, worker, started_ns, started)
        binary_trace = fm.build_trace_ras_from_binary(production.binary_paths(worker))
        full_counter = logical_counter(trace)
        binary_counter = logical_counter(binary_trace)
        row.update(
            {
                "trace": full_audit(
                    trace, worker, case.world_size, cuda_get_device_count
                ),
                "output_trace": fm.trace_summary(trace),
                "output_key_signature": {
                    "digest": counter_digest(full_counter),
                    "event_count": sum(full_counter.values()),
                },
                "binary_key_signature": {
                    "digest": counter_digest(binary_counter),
                    "event_count": sum(binary_counter.values()),
                    "trace_source": "SharedEventArena sibling from the same Maya execution",
                    "excluded_from_paper_timing": True,
                },
            }
        )
        production.atomic_json(worker.case_dir / "evaluation_checkpoint.json", row)
        return row, trace, binary_trace, raw_events
    except BaseException:
        production.terminate(worker)
        raise


def evaluate_hybrid(
    maya: dict[str, object],
    maya_trace: object,
    raw_events: list[object],
    candidate_spec: fm.FlexMayaWorkloadSpec,
) -> dict[str, object]:
    paths = list(maya["trace"]["binary_paths"])
    build_s, trace = grouped_raw_trace(raw_events, candidate_spec)
    replay_s, feedback = production.timed(lambda: fm.replay_trace_once(trace))
    feedback_s, feedback_payload = production.timed(feedback.to_dict)
    phases = production.empty_phases()
    phases["maya_emulation_s"] = float(maya["phases_s"]["maya_emulation_s"])
    phases["trace_processing_s"] = float(
        maya["phase_detail_s"]["trace_collection_s"]
    ) + float(maya["phase_detail_s"]["jsonl_parse_s"])
    phases["trace_ras_compaction_s"] = build_s
    phases["event_simulation_s"] = replay_s
    phases["feedback_generation_s"] = feedback_s
    phases["unattributed_overhead_s"] = float(
        maya["phases_s"]["unattributed_overhead_s"]
    )
    full_counter = logical_counter(maya_trace)
    grouped_counter = logical_counter(trace)
    production.require(
        full_counter == grouped_counter,
        "hybrid trace-RAS changed the logical kernel/NCCL multiset",
    )
    total_s = sum(float(value) for value in phases.values())
    return {
        "system": "maya-trace-ras",
        "round": maya["round"],
        "round_name": maya["round_name"],
        "mutation_label": maya["mutation_label"],
        "candidate_config": maya["candidate_config"],
        "input_trace": maya["trace"],
        "output_trace": grouped_audit(
            trace,
            binary_paths=paths,
            jsonl_paths=list(maya["trace"]["jsonl_paths"]),
            raw_cuda_get_device_count=int(maya["trace"]["cudaGetDevice_raw_count"]),
            world_size=int(maya["candidate_config"]["world_size"]),
        ),
        "feedback": feedback_payload,
        "logical_equivalence": {
            "pass": True,
            "kernel_nccl_digest": counter_digest(grouped_counter),
            "logical_event_count_equal": int(trace.logical_event_count)
            == int(maya_trace.logical_event_count),
        },
        "phases_s": phases,
        "timing": {
            "direct_wall_s": total_s,
            "phase_sum_s": total_s,
            "source": "paired component accounting over the identical Maya-full raw capture",
            "independent_candidate_execution": False,
        },
    }


def initialize_anchor(
    args,
    *,
    spec: fm.FlexMayaWorkloadSpec,
    case_dir: Path,
    peer_case_dir: Path,
    seed: int,
    port: int,
) -> tuple[fm.FlexMayaAnchor, dict[str, object]]:
    args.jsonl_trace = True
    worker = production.spawn_worker(
        args,
        workload="gpt",
        system="maya-full",
        config=ANCHOR,
        round_=None,
        selected_ops=(),
        seed=seed,
        case_dir=case_dir,
        peer_case_dir=peer_case_dir,
        port=port,
        peer_run_id=f"{args.run_id}-anchor",
    )
    try:
        production.wait_ready(worker, args.ready_timeout_s)
        started_ns = time.time_ns()
        started = time.perf_counter()
        phases = production.empty_phases()
        phases["maya_emulation_s"] = production.dispatch_and_wait(
            worker, args.worker_timeout_s
        )
        collect_s = production.collect_peer_traces(
            args, worker, ANCHOR.world_size
        )
        parse_s, raw_events, cuda_get_device_count = parse_jsonl(worker, ANCHOR)
        full_build_s, trace = production.timed(lambda: fm.build_trace_ras(raw_events))
        phases["trace_processing_s"] = collect_s + parse_s + full_build_s
        grouped_build_s, grouped = grouped_raw_trace(raw_events, spec)
        phases["trace_ras_compaction_s"] = grouped_build_s
        phases["event_simulation_s"], feedback = production.timed(
            lambda: fm.replay_trace_once(trace)
        )
        phases["feedback_generation_s"], feedback_payload = production.timed(
            feedback.to_dict
        )
        source_hash_s, source_hashes = production.timed(lambda: fm.source_hashes(spec))
        phases["code_analysis_s"] = source_hash_s
        row = {
            "system": "flexeva-anchor",
            "candidate_config": production.config_payload("gpt", ANCHOR),
            "executor_feedback": json.loads(
                worker.feedback_file.read_text(encoding="utf-8")
            ),
            "feedback": feedback_payload,
            "source_hashes": [asdict(item) for item in source_hashes],
            "phases_s": phases,
            **worker_record(worker),
        }
        production.finish_timing(row, worker, started_ns, started)
        trace_audit = full_audit(
            trace, worker, ANCHOR.world_size, cuda_get_device_count
        )
        row["trace"] = trace_audit
        row["trace_ras_diagnostic"] = grouped_audit(
            grouped,
            binary_paths=production.binary_paths(worker),
            jsonl_paths=jsonl_paths(worker, ANCHOR.world_size),
            raw_cuda_get_device_count=cuda_get_device_count,
            world_size=ANCHOR.world_size,
        )
        row["phase_detail_s"] = {
            "trace_collection_s": collect_s,
            "jsonl_parse_s": parse_s,
            "ordinary_full_trace_build_s": full_build_s,
        }
        production.atomic_json(worker.case_dir / "evaluation_checkpoint.json", row)
        anchor = fm.FlexMayaAnchor(
            spec=spec,
            source_hashes=source_hashes,
            trace=trace,
            feedback=feedback,
            summary={"kind": "figure7_production_anchor", "trace": trace_audit},
        )
        return anchor, row
    except BaseException:
        production.terminate(worker)
        raise


def evaluate_flexeva(
    args,
    *,
    case: MegatronCase,
    round_: production.Round,
    anchor: fm.FlexMayaAnchor,
    candidate_spec: fm.FlexMayaWorkloadSpec,
    maya: dict[str, object],
    maya_trace: object,
    maya_binary_trace: object,
    case_dir: Path,
    peer_case_dir: Path,
    seed: int,
    port: int,
) -> dict[str, object]:
    analysis_started_ns = time.time_ns()
    analysis_started = time.perf_counter()
    plan = fm.plan_candidate_refresh(
        anchor,
        candidate_spec,
        anchor.trace,
        grounding_satisfied=True,
    )
    selected = (
        tuple(partition.partition_id for partition in candidate_spec.code_partitions)
        if plan.configuration_changed
        else production.selected_code_partitions(plan, anchor.trace, candidate_spec)
    )
    analysis_s = time.perf_counter() - analysis_started
    analysis_finished_ns = time.time_ns()
    selected_ops = production.base_executor_ops("gpt", selected)
    executor = "maya-full" if plan.configuration_changed else "flexeva-selective"
    args.jsonl_trace = False
    worker = production.spawn_worker(
        args,
        workload="gpt",
        system=executor,
        config=case,
        round_=round_,
        selected_ops=selected_ops,
        seed=seed,
        case_dir=case_dir,
        peer_case_dir=peer_case_dir,
        port=port,
        peer_run_id=f"{args.run_id}-flex-r{round_.number}",
    )
    try:
        production.wait_ready(worker, args.ready_timeout_s)
        started_ns = analysis_started_ns
        # Count source analysis, but exclude the same worker bootstrap that Maya
        # excludes by starting its direct timing after the ready gate.
        started = time.perf_counter() - analysis_s
        phases = production.empty_phases()
        phases["code_analysis_s"] = analysis_s
        phases["selective_emulation_s"] = production.dispatch_and_wait(
            worker, args.worker_timeout_s
        )
        collect_s = production.collect_peer_traces(
            args, worker, case.world_size
        )
        phases["trace_processing_s"] = collect_s
        build_s, trace = production.timed(
            lambda: fm.build_trace_ras_from_binary(production.binary_paths(worker))
        )
        phases["source_ras_update_s"] = build_s
        phases["trace_patch_collation_s"], collated = production.timed(
            lambda: collate_candidate_trace(
                anchor.trace,
                trace,
                candidate_spec,
                plan,
                selected,
            )
        )
        patched_trace, collation = collated
        phases["event_simulation_s"], feedback = production.timed(
            lambda: fm.replay_trace_once(patched_trace)
        )
        phases["feedback_generation_s"], feedback_payload = production.timed(
            feedback.to_dict
        )
        row = {
            "system": "flexeva-refresh",
            "round": round_.number,
            "round_name": round_.name,
            "mutation_label": round_.label,
            "candidate_config": production.config_payload("gpt", case),
            "seed": seed,
            "executor_feedback": json.loads(
                worker.feedback_file.read_text(encoding="utf-8")
            ),
            "feedback": feedback_payload,
            "source_ras_plan": asdict(plan),
            "executed_code_partitions": list(selected),
            "configuration_full_refresh": bool(plan.configuration_changed),
            "source_analysis_window": {
                "started_ns": analysis_started_ns,
                "finished_ns": analysis_finished_ns,
                "precedes_dispatch": analysis_finished_ns <= worker.dispatch_ns,
            },
            "phases_s": phases,
            **worker_record(worker),
        }
        production.finish_timing(row, worker, started_ns, started)
        expected = logical_counter(maya_binary_trace, selected)
        observed = logical_counter(trace, selected)
        production.require(
            expected == observed,
            f"round {round_.number} selective kernel/NCCL events differ from Maya-full",
        )
        expected_surface = "SharedEventArena" if plan.configuration_changed else "rank JSONL"
        expected_full = logical_counter(
            maya_binary_trace if plan.configuration_changed else maya_trace
        )
        observed_full = logical_counter(patched_trace)
        production.require(
            expected_full == observed_full,
            f"round {round_.number} patched full trace differs from Maya-full",
        )
        maya_time = float(maya["feedback"]["total_time_us"])
        feedback_relative = abs(float(feedback_payload["total_time_us"]) - maya_time) / max(
            abs(maya_time), 1.0
        )
        production.require(
            feedback_relative <= production.FEEDBACK_TOLERANCE,
            f"round {round_.number} patched feedback differs from Maya-full",
        )
        replacement_summary = fm.trace_summary(trace)
        patched_summary = fm.trace_summary(patched_trace)
        row.update(
            {
                "trace": binary_full_audit(trace, worker, case.world_size),
                "replacement_trace": replacement_summary,
                "output_trace": patched_summary,
                "output_key_signature": {
                    "digest": counter_digest(observed_full),
                    "event_count": sum(observed_full.values()),
                },
                "logical_selected_equivalence": {
                    "pass": True,
                    "kernel_nccl_digest": counter_digest(observed),
                    "event_count": sum(observed.values()),
                },
                "patched_full_equivalence": {
                    "pass": True,
                    "comparison_trace_source": expected_surface,
                    "expected_digest": counter_digest(expected_full),
                    "observed_digest": counter_digest(observed_full),
                    "kernel_nccl_digest": counter_digest(observed_full),
                    "event_count": sum(observed_full.values()),
                    "feedback_relative_difference": feedback_relative,
                    "feedback_tolerance": production.FEEDBACK_TOLERANCE,
                },
                "anchor_patch": {
                    **collation,
                    "anchor_event_count": int(anchor.trace.logical_event_count),
                    "replacement_event_count": int(trace.logical_event_count),
                    "patched_event_count": int(patched_trace.logical_event_count),
                    "full_candidate_trace_generated": bool(plan.configuration_changed),
                },
                "phase_detail_s": {
                    "trace_collection_s": collect_s,
                    "binary_trace_build_s": build_s,
                },
            }
        )
        production.atomic_json(worker.case_dir / "evaluation_checkpoint.json", row)
        return row
    except BaseException:
        production.terminate(worker)
        raise


def add_phases(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {
        phase: float(left.get(phase, 0.0)) + float(right.get(phase, 0.0))
        for phase in PAPER_PHASES
    }


def breakdown_rows(
    anchor: dict[str, object],
    maya: list[dict[str, object]],
    hybrid: list[dict[str, object]],
    flexeva: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_round: list[dict[str, object]] = []
    cumulative: list[dict[str, object]] = []
    running = {
        "Maya-style full": {phase: 0.0 for phase in PAPER_PHASES},
        "Maya-style + FlexEva trace-RAS": {phase: 0.0 for phase in PAPER_PHASES},
        "FlexEva cumulative": anchor_paper_phases(anchor),
    }
    for maya_row, hybrid_row, flex_row in zip(maya, hybrid, flexeva, strict=True):
        round_number = int(maya_row["round"])
        round_phases = {
            "Maya-style full": maya_paper_phases(maya_row),
            "Maya-style + FlexEva trace-RAS": hybrid_paper_phases(
                maya_row, hybrid_row
            ),
            "FlexEva refresh": flexeva_paper_phases(flex_row),
        }
        for system, phases in round_phases.items():
            per_round.append(
                {
                    "round": round_number,
                    "x_label": maya_row["mutation_label"],
                    "system": system,
                    **phases,
                    "total_s": sum(phases.values()),
                }
            )
        for system, phase_source in (
            ("Maya-style full", "Maya-style full"),
            ("Maya-style + FlexEva trace-RAS", "Maya-style + FlexEva trace-RAS"),
            ("FlexEva cumulative", "FlexEva refresh"),
        ):
            running[system] = add_phases(running[system], round_phases[phase_source])
            cumulative.append(
                {
                    "round": round_number,
                    "x_label": maya_row["mutation_label"],
                    "system": system,
                    **running[system],
                    "total_s": sum(running[system].values()),
                }
            )
    return per_round, cumulative


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(path: Path) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    out_dir = path.parent
    write_csv(out_dir / "breakdown_per_round_wide.csv", result["per_round"])
    write_csv(out_dir / "breakdown_cumulative_wide.csv", result["cumulative"])
    lines = [
        "# Figure 7 two-node production run",
        "",
        "Megatron 2.7B, 16 physical A100 GPUs on two nodes, GBS 512.",
        "Full emulation uses spawned-to-exit process wall time. Trace patch follows the paper boundary: fresh trace collation for full paths and collection plus patch/collation for refresh paths.",
        "",
        "| Round | Maya-style full (s) | Maya-style + Trace-RAS (s) | FlexEva cumulative (s) |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for round_number in (int(row["number"]) for row in result["rounds"]):
        rows = {
            row["system"]: row
            for row in result["cumulative"]
            if int(row["round"]) == round_number
        }
        lines.append(
            f"| {round_number} | {rows[SYSTEMS[0]]['total_s']:.3f} | "
            f"{rows[SYSTEMS[1]]['total_s']:.3f} | {rows[SYSTEMS[2]]['total_s']:.3f} |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    render_figure(result["cumulative"], out_dir)


def reaccount_result(source: Path, out_dir: Path) -> Path:
    result = json.loads(source.read_text(encoding="utf-8"))
    per_round, cumulative = breakdown_rows(
        result["anchor"], result["maya"], result["hybrid"], result["flexeva"]
    )
    result["per_round"] = per_round
    result["cumulative"] = cumulative
    result["contract"].update(
        {
            "paper_components": [label for _, label, _, _ in PLOT_COMPONENTS],
            "full_emulation_process_wall": True,
            "paper_trace_patch_boundary": True,
        }
    )
    result["contract"].pop("plotted_components", None)
    result["contract"].pop("trace_processing_explicit", None)
    result["accounting"] = {
        "reaccounted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_result": str(source.resolve()),
        "source_result_sha256": production.sha256(source),
        "raw_measurements_modified": False,
        "full_emulation_boundary": "process_exit_ns - spawned_ns",
        "trace_patch": "fresh collation for Maya/anchor; collection plus patch/collation for FlexEva refresh",
    }
    manifest_source = Path(result["candidate_manifest"])
    if not manifest_source.is_file():
        manifest_source = source.parent / "candidate_manifest.json"
    production.require(manifest_source.is_file(), "candidate manifest is unavailable for re-accounting")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_output = out_dir / "candidate_manifest.json"
    if manifest_source.resolve() != manifest_output.resolve():
        copyfile(manifest_source, manifest_output)
    result["candidate_manifest"] = str(manifest_output.resolve())
    output = out_dir / "result.json"
    production.atomic_json(output, result)
    write_outputs(output)
    return output


def render_figure(rows: list[dict[str, object]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rounds = sorted({int(row["round"]) for row in rows})
    by_key = {(int(row["round"]), row["system"]): row for row in rows}
    baselines = {round_: float(by_key[(round_, SYSTEMS[0])]["total_s"]) for round_ in rounds}
    x = np.arange(len(rounds))
    width = 0.22
    offsets = {SYSTEMS[0]: -width, SYSTEMS[1]: 0.0, SYSTEMS[2]: width}
    figure, axis = plt.subplots(figsize=(6.2, 3.0), constrained_layout=True)
    handles = []
    labels = []
    for phase_index, (phase, label, color, hatch) in enumerate(PLOT_COMPONENTS):
        handle = None
        for system in SYSTEMS:
            heights = [float(by_key[(round_, system)][phase]) / baselines[round_] for round_ in rounds]
            bottoms = [
                sum(
                    float(by_key[(round_, system)][previous[0]]) / baselines[round_]
                    for previous in PLOT_COMPONENTS[:phase_index]
                )
                for round_ in rounds
            ]
            bars = axis.bar(
                x + offsets[system],
                heights,
                width,
                bottom=bottoms,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                hatch=hatch,
            )
            handle = handle or bars[0]
        if max(
            float(by_key[(round_, system)][phase]) / baselines[round_]
            for round_ in rounds
            for system in SYSTEMS
        ) >= 0.005:
            handles.append(handle)
            labels.append(label)
    axis.set_ylabel("Norm. Runtime")
    axis.set_xlabel("Cumulative Round")
    axis.set_xticks(x, [f"R{round_}" for round_ in rounds])
    max_total = max(
        float(by_key[(round_, system)]["total_s"]) / baselines[round_]
        for round_ in rounds
        for system in SYSTEMS
    )
    axis.set_ylim(0.0, max(1.1, max_total * 1.05))
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.1f}x")
    axis.axhline(1.0, color="#AAAAAA", linestyle="--", linewidth=0.6)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(handles, labels, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    for output in (out_dir / "figure7.pdf", out_dir / "figure7.png"):
        figure.savefig(output, dpi=240)
    plt.close(figure)


def manifest(
    root: Path,
    anchors: dict[str, Path],
    candidates: dict[int, dict[str, Path]],
) -> Path:
    payload = {
        "schema": "flexeva.figure7.candidate-manifest.v1",
        "anchor_config": production.config_payload("gpt", ANCHOR),
        "anchor_sources": {
            key: {"path": str(value), "sha256": production.sha256(value)}
            for key, value in anchors.items()
        },
        "rounds": [
            {
                **asdict(round_),
                "config": production.config_payload("gpt", config_for(round_)),
                "sources": {
                    key: {"path": str(value), "sha256": production.sha256(value)}
                    for key, value in candidates[round_.number].items()
                },
            }
            for round_ in ROUNDS
        ],
    }
    path = root / "gpt" / "candidate_manifest.json"
    production.atomic_json(path, payload)
    return path


def run(args, *, selected_rounds: tuple[production.Round, ...]) -> Path:
    args.out_dir.mkdir(parents=True)
    sources_root = args.out_dir / "manifests"
    anchors, candidates = production.write_sources(
        sources_root, "gpt", ANCHOR, ROUNDS
    )
    manifest_path = manifest(sources_root, anchors, candidates)
    ports = production.Ports(args.master_port_base)
    seed = production.SEED_BASE
    anchor_spec = spec_for(ANCHOR, anchors, "figure7_anchor")
    local, peer = production.case_paths(args, Path("formal/anchor"))
    anchor, anchor_row = initialize_anchor(
        args,
        spec=anchor_spec,
        case_dir=local,
        peer_case_dir=peer,
        seed=seed,
        port=ports.take(),
    )
    maya_rows = []
    hybrid_rows = []
    flex_rows = []
    for round_ in selected_rounds:
        case = config_for(round_)
        candidate_spec = spec_for(
            case,
            candidates[round_.number],
            f"figure7_{round_.name}",
        )
        local, peer = production.case_paths(args, Path(f"formal/maya/round_{round_.number:02d}"))
        maya, maya_trace, maya_binary_trace, raw_events = evaluate_maya(
            args,
            case=case,
            round_=round_,
            candidate_spec=candidate_spec,
            case_dir=local,
            peer_case_dir=peer,
            seed=seed,
            port=ports.take(),
        )
        hybrid = evaluate_hybrid(maya, maya_trace, raw_events, candidate_spec)
        local, peer = production.case_paths(args, Path(f"formal/flexeva/round_{round_.number:02d}"))
        flex = evaluate_flexeva(
            args,
            case=case,
            round_=round_,
            anchor=anchor,
            candidate_spec=candidate_spec,
            maya=maya,
            maya_trace=maya_trace,
            maya_binary_trace=maya_binary_trace,
            case_dir=local,
            peer_case_dir=peer,
            seed=seed,
            port=ports.take(),
        )
        maya_rows.append(maya)
        hybrid_rows.append(hybrid)
        flex_rows.append(flex)
        production.atomic_json(
            args.out_dir / "progress.json",
            {"complete": False, "last_completed_round": round_.number},
        )
    per_round, cumulative = breakdown_rows(
        anchor_row, maya_rows, hybrid_rows, flex_rows
    )
    result = {
        "schema": "flexeva.figure7.production.v1",
        "complete": selected_rounds == ROUNDS,
        "run_id": args.run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "candidate_manifest": str(manifest_path),
        "candidate_manifest_sha256": production.sha256(manifest_path),
        "topology": production.topology_payload(args),
        "configurations": {
            "anchor": production.config_payload("gpt", ANCHOR),
            "r4": production.config_payload("gpt", R4),
        },
        "rounds": [asdict(round_) for round_ in selected_rounds],
        "accounting": {
            "raw_measurements_modified": False,
            "full_emulation_boundary": "process_exit_ns - spawned_ns",
            "trace_patch": "fresh collation for Maya/anchor; collection plus patch/collation for FlexEva refresh",
        },
        "contract": {
            "paper_setting": True,
            "two_nodes": True,
            "physical_gpus": 16,
            "logical_ranks": 16,
            "global_batch_size": 512,
            "single_run": True,
            "maya_trace_ras_author_ablation": True,
            "hybrid_uses_identical_maya_raw_capture": True,
            "configuration_change_full_refresh": True,
            "paper_components": [label for _, label, _, _ in PLOT_COMPONENTS],
            "full_emulation_process_wall": True,
            "paper_trace_patch_boundary": True,
            "maya_rank_jsonl_trace": True,
            "flexeva_shared_event_arena": True,
            "flexeva_replays_patched_full_trace": True,
            "audit_overhead_plotted": False,
            "marker_timing_primary": False,
            "projection": False,
            "interrupted_round_stitching": False,
        },
        "anchor": anchor_row,
        "maya": maya_rows,
        "hybrid": hybrid_rows,
        "flexeva": flex_rows,
        "per_round": per_round,
        "cumulative": cumulative,
    }
    path = args.out_dir / "result.json"
    production.atomic_json(path, result)
    write_outputs(path)
    return path


def verify_flexeva_row(
    maya: dict[str, object],
    flex: dict[str, object],
    *,
    round_number: int,
) -> None:
    production.require(
        flex["logical_selected_equivalence"]["pass"] is True,
        "FlexEva selected trace differs from Maya-full",
    )
    production.require(
        flex["source_analysis_window"]["precedes_dispatch"] is True,
        "analysis followed dispatch",
    )
    production.require(
        float(flex["phases_s"]["trace_patch_collation_s"]) > 0.0,
        "FlexEva omitted trace patch",
    )
    expected_mode = "configuration_rebase" if round_number == 4 else "chunk_patch"
    production.require(
        flex["anchor_patch"]["mode"] == expected_mode,
        "FlexEva anchor patch mode differs",
    )
    patched = flex["patched_full_equivalence"]
    production.require(
        patched["pass"] is True
        and patched["expected_digest"] == flex["output_key_signature"]["digest"]
        and patched["observed_digest"] == flex["output_key_signature"]["digest"],
        "FlexEva patched full trace differs from Maya-full",
    )
    maya_time = float(maya["feedback"]["total_time_us"])
    flex_time = float(flex["feedback"]["total_time_us"])
    relative = abs(flex_time - maya_time) / max(abs(maya_time), 1.0)
    production.require(
        relative <= production.FEEDBACK_TOLERANCE,
        "FlexEva patched feedback differs from Maya-full",
    )
    executor = flex.get("executor_feedback", {})
    if executor:
        if round_number == 4:
            production.require(
                executor["full_training_step_executed"] is True
                and executor["full_candidate_trace_generated"] is True,
                "Figure 7 R4 did not perform the required full configuration refresh",
            )
        else:
            production.require(
                executor["full_training_step_executed"] is False
                and executor["full_candidate_trace_generated"] is False
                and executor["fallback"] is False,
                "FlexEva refresh was not selective",
            )


def verify(path: Path, *, require_complete: bool) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    production.require(result["schema"] == "flexeva.figure7.production.v1", "wrong schema")
    production.require(result["complete"] is require_complete, "completion state differs")
    production.require(
        (result["topology"]["physical_gpus"], result["topology"]["logical_ranks"], result["topology"]["node_count"])
        == (16, 16, 2),
        "Figure 7 is not a two-node/16-physical-GPU run",
    )
    production.require(result["configurations"]["anchor"] == production.config_payload("gpt", ANCHOR), "anchor differs")
    production.require(result["configurations"]["r4"] == production.config_payload("gpt", R4), "R4 differs")
    production.require(
        result["contract"]["paper_components"]
        == [label for _, label, _, _ in PLOT_COMPONENTS],
        "Figure 7 component contract differs",
    )
    production.require(result["contract"]["full_emulation_process_wall"] is True, "full emulation excludes process wall")
    production.require(result["contract"]["paper_trace_patch_boundary"] is True, "Trace patch boundary differs")
    production.require(result["contract"]["maya_rank_jsonl_trace"] is True, "Figure 7 Maya omitted rank JSONL trace")
    production.require(result["contract"]["flexeva_shared_event_arena"] is True, "Figure 7 FlexEva omitted incremental arena trace")
    production.require(result["contract"]["flexeva_replays_patched_full_trace"] is True, "Figure 7 omitted patched replay")
    production.require(result["contract"]["audit_overhead_plotted"] is False, "Figure 7 plots audit overhead")
    production.require(production.sha256(Path(result["candidate_manifest"])) == result["candidate_manifest_sha256"], "manifest hash differs")
    for row in result["maya"]:
        production.verify_timing(row)
        production.require(row["trace"]["trace_source"] == "rank JSONL", "Maya did not use the paper full-trace path")
        production.require(float(row["phases_s"]["trace_ras_compaction_s"]) == 0.0, "Maya contains Trace-RAS")
        production.require(
            float(row["phases_s"]["unattributed_overhead_s"])
            <= float(row["timing"]["direct_wall_s"]) * 0.5,
            "Maya unattributed overhead dominates direct timing",
        )
    for row in result["hybrid"]:
        production.require(row["logical_equivalence"]["pass"] is True, "hybrid logical equivalence failed")
        production.require(float(row["phases_s"]["trace_ras_compaction_s"]) > 0.0, "hybrid omitted Trace-RAS")
    for maya, row in zip(result["maya"], result["flexeva"], strict=True):
        production.verify_timing(row)
        production.require(row["trace"]["trace_source"] == "SharedEventArena", "FlexEva did not use the incremental trace path")
        verify_flexeva_row(maya, row, round_number=int(row["round"]))
        production.require(
            float(row["phases_s"]["unattributed_overhead_s"])
            <= float(row["timing"]["direct_wall_s"]) * 0.5,
            "FlexEva unattributed overhead dominates direct timing",
        )
        if int(row["round"]) == 4:
            production.require(row["configuration_full_refresh"] is True, "R4 did not use full refresh")
    expected_per_round, expected_cumulative = breakdown_rows(
        result["anchor"], result["maya"], result["hybrid"], result["flexeva"]
    )
    production.require(result["per_round"] == expected_per_round, "per-round accounting differs")
    production.require(result["cumulative"] == expected_cumulative, "cumulative accounting differs")
    plotted = {(int(row["round"]), row["system"]): row for row in result["per_round"]}
    anchor_plot = anchor_paper_phases(result["anchor"])
    production.require(anchor_plot["maya_emulation_s"] == process_wall_s(result["anchor"]), "anchor process wall differs")
    production.require(
        anchor_plot["trace_patch_collation_s"] == float(result["anchor"]["phases_s"]["trace_processing_s"]),
        "anchor Trace patch differs from fresh collation",
    )
    for maya, hybrid, flex in zip(result["maya"], result["hybrid"], result["flexeva"], strict=True):
        round_number = int(maya["round"])
        maya_plot = plotted[(round_number, "Maya-style full")]
        hybrid_plot = plotted[(round_number, "Maya-style + FlexEva trace-RAS")]
        flex_plot = plotted[(round_number, "FlexEva refresh")]
        detail = maya["phase_detail_s"]
        maya_collation_s = sum(
            float(detail[name])
            for name in ("trace_collection_s", "jsonl_parse_s", "ordinary_full_trace_build_s")
        )
        hybrid_collation_s = float(detail["trace_collection_s"]) + float(detail["jsonl_parse_s"])
        production.require(maya_plot["maya_emulation_s"] == process_wall_s(maya), "Maya process wall differs")
        production.require(maya_plot["trace_ras_compaction_s"] == 0.0, "Maya contains Trace-RAS")
        production.require(maya_plot["trace_patch_collation_s"] == maya_collation_s, "Maya fresh collation differs")
        production.require(hybrid_plot["maya_emulation_s"] == process_wall_s(maya), "hybrid Maya process wall differs")
        production.require(hybrid_plot["trace_patch_collation_s"] == hybrid_collation_s, "hybrid fresh collation differs")
        production.require(
            hybrid_plot["trace_ras_compaction_s"] == float(hybrid["phases_s"]["trace_ras_compaction_s"]),
            "hybrid active-lane compaction differs",
        )
        production.require(
            flex_plot["trace_patch_collation_s"]
            == float(flex["phases_s"]["trace_processing_s"])
            + float(flex["phases_s"]["trace_patch_collation_s"]),
            "FlexEva collection/patch boundary differs",
        )
        if round_number == 4:
            production.require(
                flex_plot["selective_emulation_s"] == process_wall_s(flex),
                "R4 full refresh excludes process wall",
            )
    for row in result["per_round"] + result["cumulative"]:
        production.require(
            set(row) == {"round", "x_label", "system", *PAPER_PHASES, "total_s"},
            "paper breakdown fields differ",
        )
        total = sum(float(row[phase]) for phase in PAPER_PHASES)
        production.require(abs(total - float(row["total_s"])) <= max(1e-6, total * 1e-9), "breakdown does not sum")
    for name in ("breakdown_per_round_wide.csv", "breakdown_cumulative_wide.csv", "README.md", "figure7.pdf", "figure7.png"):
        artifact = path.parent / name
        production.require(artifact.is_file() and artifact.stat().st_size > 0, f"missing artifact: {name}")
    print("Figure 7 production verification: PASS")


def self_test() -> None:
    production.require(production.effective_gpt_batch(ANCHOR) == 512, "anchor batch")
    production.require(production.effective_gpt_batch(R4) == 512, "R4 batch")
    production.require((ANCHOR.tp, ANCHOR.pp, ANCHOR.dp) == (1, 8, 2), "anchor topology")
    production.require((R4.tp, R4.pp, R4.dp) == (2, 8, 1), "R4 topology")
    raw = [
        fm.make_event("kernel", "kernel_launch", rank=rank, stream=0, code_partition="stage_000_attention_backward")
        for rank in range(2)
    ]
    full = fm.build_trace_ras(raw)
    grouped = fm.build_rank_grouped_trace_ras(raw, {0: [0, 1]})
    production.require(logical_counter(full) == logical_counter(grouped), "weighted grouping")

    phases = production.empty_phases()
    phases.update(
        {
            "trace_processing_s": 2.0,
            "trace_ras_compaction_s": 3.0,
            "code_analysis_s": 4.0,
            "source_ras_update_s": 5.0,
            "selective_emulation_s": 6.0,
            "trace_patch_collation_s": 7.0,
            "event_simulation_s": 8.0,
            "feedback_generation_s": 9.0,
        }
    )
    row = {
        "phases_s": phases,
        "timing": {"spawned_ns": 1_000_000_000, "process_exit_ns": 4_000_000_000},
        "phase_detail_s": {
            "trace_collection_s": 0.5,
            "jsonl_parse_s": 0.5,
            "ordinary_full_trace_build_s": 1.0,
        },
    }
    production.require(
        maya_paper_phases(row)
        == {
            "maya_emulation_s": 3.0,
            "trace_ras_compaction_s": 0.0,
            "source_ras_update_s": 0.0,
            "selective_emulation_s": 0.0,
            "trace_patch_collation_s": 2.0,
            "event_simulation_s": 17.0,
        },
        "paper phase accounting",
    )

    with TemporaryDirectory() as directory:
        trace_dir = Path(directory)
        (trace_dir / "rank_0_markers.jsonl").write_text(
            "\n".join(
                json.dumps(record)
                for record in (
                    {"kind": "step_begin", "label": "training_step", "step": 1, "trace_ts": 10},
                    {"kind": "region_begin", "label": "attention_backward", "trace_ts": 11},
                    {"kind": "region_end", "label": "attention_backward", "trace_ts": 19},
                    {"kind": "step_end", "label": "training_step", "step": 1, "trace_ts": 20},
                )
            )
            + "\n",
            encoding="utf-8",
        )
        (trace_dir / "rank_0.jsonl").write_text(
            json.dumps(
                {
                    "api": "cudaLaunchKernel",
                    "type": "kernel_launch",
                    "ts": 15,
                    "grid_x": 1,
                    "block_x": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        _, parsed, _ = parse_jsonl(
            SimpleNamespace(trace_dir=trace_dir),
            SimpleNamespace(world_size=1, pp=1, dtype="bf16"),
        )
        production.require(
            parsed[0].code_partition == "stage_000_attention_backward",
            "source-region partition parsing",
        )
    print("Figure 7 production driver self-test: PASS")


def main() -> int:
    args = production.parse_args()
    if args.mode == "self-test":
        self_test()
        return 0
    result_path = args.result or args.out_dir / "result.json"
    if args.mode in {"probe", "run"}:
        production.require_runtime_args(args)
        selected_rounds = (
            (ROUNDS[args.probe_round - 1],) if args.mode == "probe" else ROUNDS
        )
        result_path = run(args, selected_rounds=selected_rounds)
        verify(result_path, require_complete=args.mode == "run")
        print(result_path)
        return 0
    if args.mode == "report":
        result_path = reaccount_result(result_path, args.out_dir)
    verify(result_path, require_complete=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
