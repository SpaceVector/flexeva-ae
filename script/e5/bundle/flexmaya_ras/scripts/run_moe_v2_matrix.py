#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping

import flexmaya_ras as fm


ROOT = Path(__file__).resolve().parents[1]
NEW_SIM_ROOT = ROOT.parent
PRAS_SRC = NEW_SIM_ROOT / "paper_resilient_anchor_state" / "src"
if PRAS_SRC.exists() and str(PRAS_SRC) not in sys.path:
    sys.path.insert(0, str(PRAS_SRC))

from paper_resilient_anchor_state.maya_v2_load_skew_case import (  # noqa: E402
    MAYA_V2_LOAD_SKEW_CASE_ID,
    MAYA_V2_LOAD_SKEW_ROUND_ID,
    build_maya_v2_load_skew_case,
)


CODE_PARTITIONS = ("routing", "memory_payload", "dispatch_collective", "expert_compute", "sync")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Maya v2 MoE load-skew matrix with separated Maya and FlexEva evaluators."
    )
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--ep-group-size", type=int, default=8)
    parser.add_argument("--micro-batches", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--oracle-results", type=Path, default=None)
    parser.add_argument("--allow-missing-oracle", action="store_true")
    parser.add_argument("--error-threshold-pct", type=float, default=10.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return time.perf_counter() - start, value


def rank_groups(world_size: int, ep_group_size: int) -> dict[int, list[int]]:
    if world_size <= 0 or ep_group_size <= 0:
        raise ValueError("world-size and ep-group-size must be positive")
    groups: dict[int, list[int]] = {}
    for start in range(0, world_size, ep_group_size):
        members = list(range(start, min(start + ep_group_size, world_size)))
        groups[members[0]] = members
    return groups


def candidate_profiles() -> dict[str, dict[str, float]]:
    return {
        "anchor_baseline": {
            "routing": 1.00,
            "memory_payload": 1.00,
            "dispatch_collective": 1.00,
            "expert_compute": 1.00,
        },
        "overflow_reroute": {
            "routing": 1.08,
            "memory_payload": 1.00,
            "dispatch_collective": 0.86,
            "expert_compute": 1.00,
        },
        "layout_striped": {
            "routing": 1.00,
            "memory_payload": 0.92,
            "dispatch_collective": 0.78,
            "expert_compute": 1.03,
        },
        "local_backup_reroute": {
            "routing": 1.12,
            "memory_payload": 1.04,
            "dispatch_collective": 0.72,
            "expert_compute": 1.02,
        },
        "balanced_secondary_route": {
            "routing": 1.05,
            "memory_payload": 1.00,
            "dispatch_collective": 0.82,
            "expert_compute": 1.01,
        },
    }


def base_candidate_id(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("base_candidate_id") or candidate["candidate_id"])


def candidate_variant_scale(candidate: Mapping[str, Any]) -> float:
    return float(candidate.get("variant_scale", 1.0))


def expand_candidate_variants(candidates: Iterable[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    base = [dict(candidate) for candidate in candidates]
    if count <= 0:
        raise ValueError("candidate count must be positive")
    if not base:
        raise ValueError("at least one base candidate is required")
    variants: list[dict[str, Any]] = []
    for index in range(count):
        source = dict(base[index % len(base)])
        base_id = base_candidate_id(source)
        family_round = index // len(base)
        source["base_candidate_id"] = base_id
        source["candidate_id"] = f"{base_id}__v{family_round:02d}"
        source["variant_index"] = index
        # Small deterministic perturbation keeps the candidate matrix ordered
        # without inventing new semantic families.
        source["variant_scale"] = 1.0 + ((family_round % 5) - 2) * 0.015
        source["semantic_diffs"] = tuple(source.get("semantic_diffs", ())) + (
            f"case_study_variant_scale={source['variant_scale']:.3f}",
        )
        variants.append(source)
    return variants


def selected_code_partitions(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    candidate_id = base_candidate_id(candidate)
    selected = {"routing", "sync"}
    if candidate_id in {
        "overflow_reroute",
        "layout_striped",
        "local_backup_reroute",
        "balanced_secondary_route",
    }:
        selected.add("dispatch_collective")
    if candidate_id in {"layout_striped", "local_backup_reroute"}:
        selected.add("expert_compute")
        selected.add("memory_payload")
    return tuple(partition for partition in CODE_PARTITIONS if partition in selected)


def spec_for_candidate(candidate: Mapping[str, Any], args: Mapping[str, Any]) -> fm.FlexMayaWorkloadSpec:
    return fm.FlexMayaWorkloadSpec(
        workload_id=str(candidate["candidate_id"]),
        world_size=int(args["world_size"]),
        included_files=(str(candidate["entry"]),),
        code_partitions=tuple(
            fm.CodePartitionSpec(
                partition_id=partition,
                path=str(candidate["entry"]),
                active_ranks=tuple(range(int(args["world_size"]))),
            )
            for partition in CODE_PARTITIONS
        ),
        rank_group_policy="none",
        notes=("maya_v2_moe_load_skew_matrix",),
    )


def synthetic_moe_events(
    args: Mapping[str, Any],
    *,
    candidate_id: str,
    ranks: Iterable[int],
    capture_code_partitions: Iterable[str] | None = None,
    base_candidate_id: str | None = None,
    profile_scale: float = 1.0,
) -> list[object]:
    profile_id = base_candidate_id or candidate_id
    base_profile = candidate_profiles().get(profile_id, candidate_profiles()["anchor_baseline"])
    profile = {key: float(value) * float(profile_scale) for key, value in base_profile.items()}
    capture = None if capture_code_partitions is None else set(capture_code_partitions)
    rows: list[object] = []
    ts = 100
    correlation = 1
    hidden = int(args["hidden_size"])
    seq_len = int(args["seq_len"])
    tokens = max(seq_len * hidden, 1)
    expert_work = max(hidden * hidden, 1)
    ranks = tuple(int(rank) for rank in ranks)

    def emit(partition: str, event: object) -> None:
        nonlocal correlation
        if capture is not None and partition not in capture:
            return
        event.correlation_id = correlation
        correlation += 1
        rows.append(event)

    def advance(duration_us: float) -> None:
        nonlocal ts
        ts += max(int(duration_us * 1000), 1)

    for micro_batch in range(int(args["micro_batches"])):
        for layer in range(int(args["layers"])):
            group_name = f"moe-dispatch:mb{micro_batch}:layer{layer}"
            for rank in ranks:
                routing_us = max(tokens / 8192.0, 1.0) * profile["routing"]
                emit(
                    "routing",
                    fm.make_event(
                        "cudaLaunchKernel",
                        "kernel_launch",
                        rank=rank,
                        stream=0,
                        timestamp_ns=ts,
                        duration_hint_us=routing_us,
                        count=tokens,
                        code_partition="routing",
                    ),
                )
                advance(routing_us)

                memory_us = max(tokens / 16384.0, 0.5) * profile["memory_payload"]
                emit(
                    "memory_payload",
                    fm.make_event(
                        "cudaMemcpyAsync",
                        "mem_copy",
                        rank=rank,
                        stream=1,
                        timestamp_ns=ts,
                        duration_hint_us=memory_us,
                        bytes=tokens * 2,
                        count=tokens,
                        code_partition="memory_payload",
                    ),
                )
                advance(memory_us)

                dispatch_us = max(tokens / 4096.0, 2.0) * profile["dispatch_collective"]
                emit(
                    "dispatch_collective",
                    fm.make_event(
                        "ncclAllToAll",
                        "nccl_collective",
                        rank=rank,
                        stream=3,
                        timestamp_ns=ts,
                        duration_hint_us=dispatch_us,
                        bytes=tokens * 2,
                        count=tokens,
                        collective_group=group_name,
                        code_partition="dispatch_collective",
                    ),
                )
                advance(dispatch_us)

                expert_us = max(expert_work / 32768.0, 3.0) * profile["expert_compute"]
                emit(
                    "expert_compute",
                    fm.make_event(
                        "cublasGemmEx",
                        "blas_compute",
                        rank=rank,
                        stream=4,
                        timestamp_ns=ts,
                        duration_hint_us=expert_us,
                        count=expert_work,
                        code_partition="expert_compute",
                    ),
                )
                advance(expert_us)

                sync_us = 0.25
                emit(
                    "sync",
                    fm.make_event(
                        "cudaStreamSynchronize",
                        "stream_op",
                        rank=rank,
                        stream=4,
                        timestamp_ns=ts,
                        duration_hint_us=sync_us,
                        code_partition="sync",
                        blocking=True,
                    ),
                )
                advance(sync_us)
    return rows


def build_anchor(args: Mapping[str, Any], anchor_candidate: Mapping[str, Any], groups: Mapping[int, list[int]]) -> dict[str, Any]:
    spec = spec_for_candidate(anchor_candidate, args)
    raw_s, raw = timed(
        lambda: synthetic_moe_events(args, candidate_id=str(anchor_candidate["candidate_id"]), ranks=range(int(args["world_size"])))
    )
    build_s, trace = timed(lambda: fm.build_rank_grouped_trace_ras(raw, groups))
    replay_s, feedback = timed(lambda: fm.replay_trace_once(trace))
    anchor = fm.FlexMayaAnchor(
        spec=spec,
        source_hashes=fm.source_hashes(spec),
        trace=trace,
        feedback=feedback,
        summary={
            "kind": "moe_v2_anchor",
            "candidate_id": str(anchor_candidate["candidate_id"]),
            "phases_s": {
                "synthetic_hook_capture_s": raw_s,
                "cpp_collation_s": build_s,
                "python_replay_s": replay_s,
            },
            "raw_event_count": len(raw),
            "trace": fm.trace_summary(trace),
            "feedback": feedback.to_dict(),
        },
    )
    context = fm.anchor_context(anchor, rank_groups=groups)
    return {
        "anchor": anchor,
        "context": context,
        "summary": {
            "candidate_id": str(anchor_candidate["candidate_id"]),
            "source_hashes": [asdict(item) for item in anchor.source_hashes],
            "rank_groups": {str(rep): list(ranks) for rep, ranks in context.rank_groups.items()},
            "phases_s": anchor.summary["phases_s"],
            "raw_event_count": len(raw),
            "trace": fm.trace_summary(trace),
            "feedback": feedback.to_dict(),
        },
    }


def build_and_replay_full(raw: list[object]) -> tuple[object, object, dict[str, float]]:
    phases: dict[str, float] = {}
    phases["cpp_collation_s"], trace = timed(lambda: fm.build_trace_ras(raw))
    phases["python_replay_s"], replay = timed(lambda: fm.replay_trace_once(trace))
    return trace, replay, phases


def build_and_replay_rank_grouped(raw: list[object], groups: Mapping[int, list[int]]) -> tuple[object, object, dict[str, float]]:
    phases: dict[str, float] = {}
    phases["cpp_collation_s"], trace = timed(lambda: fm.build_rank_grouped_trace_ras(raw, groups))
    phases["python_replay_s"], replay = timed(lambda: fm.replay_trace_once(trace))
    return trace, replay, phases


def evaluate_maya_trace_ras_candidate(
    candidate: Mapping[str, Any],
    args: Mapping[str, Any],
    groups: Mapping[int, list[int]],
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    profile_id = base_candidate_id(candidate)
    profile_scale = candidate_variant_scale(candidate)
    gen_s, raw = timed(
        lambda: synthetic_moe_events(
            args,
            candidate_id=candidate_id,
            base_candidate_id=profile_id,
            profile_scale=profile_scale,
            ranks=range(int(args["world_size"])),
        )
    )
    trace, replay, phases = build_and_replay_rank_grouped(raw, groups)
    return {
        "phases_s": {"synthetic_hook_capture_s": gen_s, **phases},
        "total_s": gen_s + sum(phases.values()),
        "raw_event_count": len(raw),
        "trace": fm.trace_summary(trace),
        "feedback": replay.to_dict(),
    }


def evaluate_flexeva_selected_candidate(
    candidate: Mapping[str, Any],
    args: Mapping[str, Any],
    groups: Mapping[int, list[int]],
    anchor_total_runtime_us: float,
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    profile_id = base_candidate_id(candidate)
    profile_scale = candidate_variant_scale(candidate)
    spec = spec_for_candidate(candidate, args)
    representatives = tuple(sorted(int(rep) for rep in groups))
    selected_partitions = selected_code_partitions(candidate)

    selected_gen_s, selected_raw = timed(
        lambda: synthetic_moe_events(
            args,
            candidate_id=candidate_id,
            base_candidate_id=profile_id,
            profile_scale=profile_scale,
            ranks=representatives,
            capture_code_partitions=selected_partitions,
        )
    )
    selected_build_s, selected_trace = timed(
        lambda: fm.build_selected_trace(
            spec,
            selected_raw,
            selected_ranks=representatives,
            selected_code_partitions=selected_partitions,
            rank_groups=groups,
        )
    )
    selected_replay_s, selected_feedback = timed(lambda: fm.replay_trace_once(selected_trace))

    anchor_selected_gen_s, anchor_selected_raw = timed(
        lambda: synthetic_moe_events(
            args,
            candidate_id="anchor_baseline",
            ranks=representatives,
            capture_code_partitions=selected_partitions,
        )
    )
    anchor_selected_build_s, anchor_selected_trace = timed(
        lambda: fm.build_selected_trace(
            spec,
            anchor_selected_raw,
            selected_ranks=representatives,
            selected_code_partitions=selected_partitions,
            rank_groups=groups,
        )
    )
    anchor_selected_replay_s, anchor_selected_feedback = timed(lambda: fm.replay_trace_once(anchor_selected_trace))
    predicted_total_us = (
        float(anchor_total_runtime_us)
        - float(anchor_selected_feedback.total_time_us)
        + float(selected_feedback.total_time_us)
    )

    return {
        "phases_s": {
            "selected_hook_capture_s": selected_gen_s,
            "cpp_selected_collation_s": selected_build_s,
            "python_replay_s": selected_replay_s,
            "anchor_selected_hook_capture_s": anchor_selected_gen_s,
            "anchor_selected_cpp_collation_s": anchor_selected_build_s,
            "anchor_selected_python_replay_s": anchor_selected_replay_s,
        },
        "total_s": selected_gen_s + selected_build_s + selected_replay_s,
        "raw_event_count": len(selected_raw),
        "trace": fm.trace_summary(selected_trace),
        "feedback": selected_feedback.to_dict(),
        "anchor_selected_trace": fm.trace_summary(anchor_selected_trace),
        "anchor_selected_feedback": anchor_selected_feedback.to_dict(),
        "predicted_candidate_total_runtime_us": predicted_total_us,
        "selected_code_partitions": list(selected_partitions),
        "representative_ranks": list(representatives),
    }


def evaluate_candidate_worker(payload: tuple[dict[str, Any], dict[str, Any], dict[int, list[int]], float]) -> dict[str, Any]:
    candidate, args, groups, anchor_total_runtime_us = payload
    candidate_id = str(candidate["candidate_id"])
    profile_id = base_candidate_id(candidate)
    profile_scale = candidate_variant_scale(candidate)
    representatives = tuple(sorted(int(rep) for rep in groups))
    selected_partitions = selected_code_partitions(candidate)

    full_gen_s, full_raw = timed(
        lambda: synthetic_moe_events(
            args,
            candidate_id=candidate_id,
            base_candidate_id=profile_id,
            profile_scale=profile_scale,
            ranks=range(int(args["world_size"])),
        )
    )
    maya_full_trace, maya_full_replay, maya_full_phases = build_and_replay_full(full_raw)
    maya_trace_ras_trace, maya_trace_ras_replay, maya_trace_ras_phases = build_and_replay_rank_grouped(full_raw, groups)
    flexeva_selected = evaluate_flexeva_selected_candidate(candidate, args, groups, anchor_total_runtime_us)

    maya_full_total_s = full_gen_s + sum(maya_full_phases.values())
    maya_trace_ras_total_s = full_gen_s + sum(maya_trace_ras_phases.values())
    flexeva_total_s = float(flexeva_selected["total_s"])
    full_raw_count = len(full_raw)
    selected_raw_count = int(flexeva_selected["raw_event_count"])
    maya_trace_event_count = len(maya_trace_ras_trace.events)
    selected_trace_event_count = int(flexeva_selected["trace"]["event_count"])
    return {
        "candidate_id": candidate_id,
        "base_candidate_id": profile_id,
        "variant_scale": profile_scale,
        "change_surface": str(candidate["change_surface"]),
        "entry": str(candidate["entry"]),
        "semantic_diffs": list(candidate.get("semantic_diffs", ())),
        "selected_code_partitions": list(selected_partitions),
        "representative_ranks": list(representatives),
        "maya_full": {
            "phases_s": {"synthetic_hook_capture_s": full_gen_s, **maya_full_phases},
            "total_s": maya_full_total_s,
            "raw_event_count": full_raw_count,
            "trace": fm.trace_summary(maya_full_trace),
            "feedback": maya_full_replay.to_dict(),
        },
        "maya_trace_ras": {
            "phases_s": {"synthetic_hook_capture_s": full_gen_s, **maya_trace_ras_phases},
            "total_s": maya_trace_ras_total_s,
            "trace": fm.trace_summary(maya_trace_ras_trace),
            "feedback": maya_trace_ras_replay.to_dict(),
        },
        "flexeva_selected": flexeva_selected,
        "reuse": {
            "raw_event_reuse_rate": 1.0 - (selected_raw_count / max(full_raw_count, 1)),
            "trace_event_reuse_rate_vs_maya_trace_ras": 1.0 - (
                selected_trace_event_count / max(maya_trace_event_count, 1)
            ),
            "rank_capture_reduction_rate": 1.0 - (len(representatives) / max(int(args["world_size"]), 1)),
        },
        "speedup": {
            "flexeva_vs_maya_full_wall": maya_full_total_s / max(flexeva_total_s, 1e-12),
            "flexeva_vs_maya_trace_ras_wall": maya_trace_ras_total_s / max(flexeva_total_s, 1e-12),
        },
    }


def load_oracle(path: Path | None, required_candidate_ids: Iterable[str], allow_missing: bool) -> dict[str, float | None]:
    required = tuple(dict.fromkeys(str(item) for item in required_candidate_ids))
    if path is None:
        if allow_missing:
            return {candidate_id: None for candidate_id in required}
        raise SystemExit("--oracle-results is required for paper-facing fidelity mode; pass --allow-missing-oracle for dry runs")
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", payload)
    oracle: dict[str, float | None] = {}
    missing: list[str] = []
    for candidate_id in required:
        row = candidates.get(candidate_id)
        runtime = None
        if isinstance(row, Mapping):
            if str(row.get("status", "complete")) != "complete":
                runtime = None
            else:
                runtime = row.get("runtime_us", row.get("per_step_runtime_us"))
        elif row is not None:
            runtime = row
        if runtime is None:
            missing.append(candidate_id)
            oracle[candidate_id] = None
        else:
            oracle[candidate_id] = float(runtime)
    if missing and not allow_missing:
        raise SystemExit("oracle missing complete runtime_us for: " + ", ".join(missing))
    return oracle


def fidelity_report(rows: list[dict[str, Any]], oracle: Mapping[str, float | None], threshold_pct: float) -> dict[str, Any]:
    per_candidate: list[dict[str, Any]] = []
    observed = 0
    errors: list[float] = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        oracle_us = oracle.get(candidate_id)
        predicted_us = float(row["flexeva_selected"]["predicted_candidate_total_runtime_us"])
        error_pct = None
        if oracle_us is not None and oracle_us != 0:
            observed += 1
            error_pct = abs(predicted_us - float(oracle_us)) / abs(float(oracle_us)) * 100.0
            errors.append(error_pct)
        per_candidate.append(
            {
                "candidate_id": candidate_id,
                "predicted_runtime_us": predicted_us,
                "oracle_runtime_us": oracle_us,
                "absolute_error_pct": error_pct,
                "uncertain": error_pct is None or error_pct > threshold_pct,
            }
        )
    predicted_order = [item["candidate_id"] for item in sorted(per_candidate, key=lambda item: item["predicted_runtime_us"])]
    oracle_known = [item for item in per_candidate if item["oracle_runtime_us"] is not None]
    oracle_order = [item["candidate_id"] for item in sorted(oracle_known, key=lambda item: float(item["oracle_runtime_us"]))]
    return {
        "oracle_coverage": observed / max(len(rows), 1),
        "mean_absolute_error_pct": sum(errors) / max(len(errors), 1) if errors else None,
        "max_absolute_error_pct": max(errors) if errors else None,
        "predicted_winner": predicted_order[0] if predicted_order else None,
        "oracle_winner": oracle_order[0] if oracle_order else None,
        "winner_match": bool(predicted_order and oracle_order and predicted_order[0] == oracle_order[0]),
        "ranking_match": predicted_order[: len(oracle_order)] == oracle_order if oracle_order else None,
        "uncertainty_flags": [
            item["candidate_id"]
            for item in per_candidate
            if item["absolute_error_pct"] is None or item["absolute_error_pct"] > threshold_pct
        ],
        "per_candidate": per_candidate,
    }


def write_outputs(out_dir: Path, result: Mapping[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matrix.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "oracle_fidelity.json").write_text(
        json.dumps(result["oracle_fidelity"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "predicted_runtime_us",
                "oracle_runtime_us",
                "absolute_error_pct",
                "raw_event_reuse_rate",
                "trace_event_reuse_rate_vs_maya_trace_ras",
                "flexeva_vs_maya_trace_ras_wall",
            ],
        )
        writer.writeheader()
        fidelity_by_id = {row["candidate_id"]: row for row in result["oracle_fidelity"]["per_candidate"]}
        for row in result["candidates"]:
            fidelity = fidelity_by_id[str(row["candidate_id"])]
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "predicted_runtime_us": fidelity["predicted_runtime_us"],
                    "oracle_runtime_us": fidelity["oracle_runtime_us"],
                    "absolute_error_pct": fidelity["absolute_error_pct"],
                    "raw_event_reuse_rate": row["reuse"]["raw_event_reuse_rate"],
                    "trace_event_reuse_rate_vs_maya_trace_ras": row["reuse"]["trace_event_reuse_rate_vs_maya_trace_ras"],
                    "flexeva_vs_maya_trace_ras_wall": row["speedup"]["flexeva_vs_maya_trace_ras_wall"],
                }
            )
    (out_dir / "result.json").write_text(json.dumps({"matrix_json": str(out_dir / "matrix.json")}, indent=2), encoding="utf-8")


def main() -> None:
    args_ns = parse_args()
    if args_ns.out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args_ns.out_dir = ROOT / "output" / f"moe_v2_matrix_{stamp}"
    args = vars(args_ns).copy()
    args["out_dir"] = str(args_ns.out_dir)
    args["oracle_results"] = None if args_ns.oracle_results is None else str(args_ns.oracle_results)

    case = build_maya_v2_load_skew_case()
    all_candidates = [asdict(case.anchor)] + [
        asdict(candidate)
        for candidate in case.candidates
        if candidate.candidate_id != case.anchor.candidate_id
    ]
    matrix_candidates = [candidate for candidate in all_candidates if candidate["candidate_id"] != case.anchor.candidate_id]
    oracle = load_oracle(
        args_ns.oracle_results,
        [candidate["candidate_id"] for candidate in all_candidates],
        args_ns.allow_missing_oracle,
    )
    groups = rank_groups(args_ns.world_size, args_ns.ep_group_size)
    anchor_payload = build_anchor(args, all_candidates[0], groups)
    anchor_total_runtime_us = float(anchor_payload["anchor"].feedback.total_time_us)

    worker_payloads = [
        (candidate, args, groups, anchor_total_runtime_us)
        for candidate in matrix_candidates
    ]
    if args_ns.workers <= 1:
        rows = [evaluate_candidate_worker(payload) for payload in worker_payloads]
    else:
        with ProcessPoolExecutor(max_workers=args_ns.workers) as executor:
            rows = list(executor.map(evaluate_candidate_worker, worker_payloads))

    rows.sort(key=lambda item: str(item["candidate_id"]))
    fidelity = fidelity_report(rows, oracle, args_ns.error_threshold_pct)
    result = {
        "case_id": MAYA_V2_LOAD_SKEW_CASE_ID,
        "round_id": MAYA_V2_LOAD_SKEW_ROUND_ID,
        "mode": "maya_v2_moe_load_skew_matrix",
        "configuration": {
            "world_size": args_ns.world_size,
            "ep_group_size": args_ns.ep_group_size,
            "micro_batches": args_ns.micro_batches,
            "layers": args_ns.layers,
            "seq_len": args_ns.seq_len,
            "hidden_size": args_ns.hidden_size,
            "workers": args_ns.workers,
            "oracle_results": None if args_ns.oracle_results is None else str(args_ns.oracle_results),
            "allow_missing_oracle": args_ns.allow_missing_oracle,
        },
        "designs": {
            "maya_full": "full synthetic hook capture, full C++ collation, one Python replay",
            "maya_trace_ras": "full hook capture, rank-grouped Maya trace RAS, one Python replay",
            "flexeva_selected": "representative hook capture for changed code partitions, selected C++ placement, one Python replay plus anchor delta feedback",
        },
        "rank_groups": {str(rep): members for rep, members in groups.items()},
        "anchor": anchor_payload["summary"],
        "candidates": rows,
        "oracle": {candidate_id: runtime for candidate_id, runtime in oracle.items()},
        "oracle_fidelity": fidelity,
    }
    write_outputs(args_ns.out_dir, result)
    print(json.dumps({"matrix_json": str(args_ns.out_dir / "matrix.json"), "oracle_fidelity": fidelity}, sort_keys=True))


if __name__ == "__main__":
    main()
