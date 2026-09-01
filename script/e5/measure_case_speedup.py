#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
from pathlib import Path
from statistics import mean, median, pstdev, quantiles
import subprocess
import sys
import time

import measure_paper_aligned as memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure the paper-scale E5 speedup one candidate round at a time.")
    sub = parser.add_subparsers(dest="action", required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--manifest", type=Path, required=True)
    probe.add_argument("--round", type=int, required=True)
    probe.add_argument("--repeat", type=int, required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--repeats", type=int, default=3)
    verify = sub.add_parser("verify")
    verify.add_argument("--result", type=Path, required=True)
    sub.add_parser("self-test")
    return parser.parse_args()


def timed(function):
    started = time.perf_counter()
    value = function()
    return time.perf_counter() - started, value


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("speedups must be finite and positive")
    q1, _, q3 = quantiles(values, n=4, method="inclusive") if len(values) > 1 else (values[0],) * 3
    average = mean(values)
    deviation = pstdev(values)
    return {
        "count": len(values),
        "mean": average,
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "population_stddev": deviation,
        "population_cv": deviation / average,
    }


def anchor_state(root: Path, manifest: dict, groups: dict[int, list[int]]) -> tuple[object, dict]:
    anchor = manifest["anchor"]
    anchor_spec = memory.spec(anchor["candidate_id"], memory.source_paths(root, anchor))
    raw_s, raw = timed(lambda: memory.events(anchor, ranks=range(16)))
    build_s, trace = timed(lambda: memory.fm.build_rank_grouped_trace_ras(raw, groups))
    replay_s, feedback = timed(lambda: memory.replay(trace))
    hash_s, hashes = timed(lambda: memory.fm.source_hashes(anchor_spec))
    state = memory.fm.FlexMayaAnchor(
        spec=anchor_spec,
        source_hashes=hashes,
        trace=trace,
        feedback=feedback,
        summary={"kind": "paper-scale E5 anchor"},
    )
    return state, {
        "event_generation_s": raw_s,
        "trace_build_s": build_s,
        "replay_s": replay_s,
        "source_hash_s": hash_s,
        "total_s": raw_s + build_s + replay_s + hash_s,
        "raw_event_count": len(raw),
        "trace": memory.fm.trace_summary(trace),
        "feedback": feedback.to_dict(),
    }


def maya_full(row: dict) -> dict:
    started = time.perf_counter()
    generation_s, raw = timed(lambda: memory.events(row, ranks=range(16)))
    build_s, trace = timed(lambda: memory.fm.build_trace_ras(raw))
    replay_s, feedback = timed(lambda: memory.replay(trace))
    return {
        "phases_s": {
            "full_event_generation_s": generation_s,
            "full_trace_build_s": build_s,
            "full_replay_s": replay_s,
            "total_s": time.perf_counter() - started,
        },
        "raw_event_count": len(raw),
        "trace": memory.fm.trace_summary(trace),
        "feedback": feedback.to_dict(),
    }


def flexeva_refresh(
    root: Path,
    anchor: object,
    anchor_row: dict,
    row: dict,
    groups: dict[int, list[int]],
) -> tuple[dict, float]:
    candidate_spec = memory.spec(row["candidate_id"], memory.source_paths(root, row))
    plan_s, plan = timed(lambda: memory.fm.plan_candidate_refresh(anchor, candidate_spec, anchor.trace))
    selected = tuple(partition for partition in memory.PARTITIONS if partition in set(plan.changed_partitions))
    if selected != ("routing", "dispatch_collective"):
        raise RuntimeError(f"unexpected source-RAS refresh scope: {selected}")
    representatives = tuple(sorted(groups))

    # Algorithm 1 prepares this baseline partition contribution once. It is
    # recorded as anchor initialization, never as a per-round refresh cost.
    cache_gen_s, cache_raw = timed(lambda: memory.events(anchor_row, ranks=representatives, selected=selected))
    cache_build_s, cache_trace = timed(
        lambda: memory.fm.build_selected_trace(
            candidate_spec,
            cache_raw,
            selected_ranks=representatives,
            selected_code_partitions=selected,
            rank_groups=groups,
        )
    )
    cache_replay_s, cache_feedback = timed(lambda: memory.replay(cache_trace))
    cache_s = cache_gen_s + cache_build_s + cache_replay_s

    started = time.perf_counter()
    generation_s, raw = timed(lambda: memory.events(row, ranks=representatives, selected=selected))
    build_s, trace = timed(
        lambda: memory.fm.build_selected_trace(
            candidate_spec,
            raw,
            selected_ranks=representatives,
            selected_code_partitions=selected,
            rank_groups=groups,
        )
    )
    replay_s, feedback = timed(lambda: memory.replay(trace))
    total_s = plan_s + (time.perf_counter() - started)
    predicted_total_us = (
        float(anchor.feedback.total_time_us)
        - float(cache_feedback.total_time_us)
        + float(feedback.total_time_us)
    )
    return {
        "phases_s": {
            "source_diff_al_plan_s": plan_s,
            "selective_event_generation_s": generation_s,
            "selected_trace_build_s": build_s,
            "selected_replay_s": replay_s,
            "total_s": total_s,
        },
        "plan": {
            "changed_partitions": list(plan.changed_partitions),
            "affected_rank_groups": list(plan.affected_rank_groups),
            "affected_trace_partition_count": plan.affected_trace_partition_count,
            "configuration_changed": plan.configuration_changed,
            "refresh_scope": plan.refresh_scope,
            "fallback_reasons": list(plan.fallback_reasons),
        },
        "selected_code_partitions": list(selected),
        "selected_ranks": list(representatives),
        "raw_event_count": len(raw),
        "trace": memory.fm.trace_summary(trace),
        "selected_feedback": feedback.to_dict(),
        "predicted_candidate_total_runtime_us": predicted_total_us,
        "anchor_selected_cache": {
            "event_generation_s": cache_gen_s,
            "trace_build_s": cache_build_s,
            "replay_s": cache_replay_s,
            "total_s": cache_s,
        },
    }, cache_s


def run_probe(args: argparse.Namespace) -> int:
    root, manifest = memory.load_manifest(args.manifest)
    if not 1 <= args.round <= 32 or args.repeat < 1:
        raise ValueError("--round must be in [1, 32] and --repeat must be positive")
    row = list(manifest["candidates"])[args.round - 1]
    groups = memory.matrix.rank_groups(16, 8)
    anchor, anchor_result = anchor_state(root, manifest, groups)

    if (args.round + args.repeat) % 2:
        flex, cache_s = flexeva_refresh(root, anchor, manifest["anchor"], row, groups)
        memory.trim_heap()
        maya = maya_full(row)
        order = "flexeva_then_maya"
    else:
        maya = maya_full(row)
        memory.trim_heap()
        flex, cache_s = flexeva_refresh(root, anchor, manifest["anchor"], row, groups)
        order = "maya_then_flexeva"

    maya_s = float(maya["phases_s"]["total_s"])
    flex_s = float(flex["phases_s"]["total_s"])
    maya_feedback_us = float(maya["feedback"]["total_time_us"])
    predicted_us = float(flex["predicted_candidate_total_runtime_us"])
    result = {
        "round": args.round,
        "repeat": args.repeat,
        "candidate_id": row["candidate_id"],
        "route_experts": row["route_experts"],
        "execution_order": order,
        "host": {"hostname": platform.node(), "python": platform.python_version()},
        "paper_config": memory.PAPER_CONFIG,
        "anchor": {**anchor_result, "selected_cache_s": cache_s},
        "maya_full": maya,
        "flexeva_refresh": flex,
        "speedup": maya_s / flex_s,
        "feedback_relative_error": abs(predicted_us - maya_feedback_us) / max(abs(maya_feedback_us), 1.0e-12),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def round_rows(samples: list[dict]) -> list[dict]:
    rows = []
    for round_index in range(1, 33):
        selected = [row for row in samples if int(row["round"]) == round_index]
        if not selected:
            raise ValueError(f"missing round {round_index}")
        maya_values = [float(row["maya_full"]["phases_s"]["total_s"]) for row in selected]
        flex_values = [float(row["flexeva_refresh"]["phases_s"]["total_s"]) for row in selected]
        speedups = [float(row["speedup"]) for row in selected]
        rows.append(
            {
                "round": round_index,
                "candidate_id": selected[0]["candidate_id"],
                "route_experts": "-".join(str(value) for value in selected[0]["route_experts"]),
                "maya_full_s_median": median(maya_values),
                "flexeva_refresh_s_median": median(flex_values),
                "speedup_median": median(speedups),
                "speedup_min": min(speedups),
                "speedup_max": max(speedups),
                "feedback_relative_error_max": max(float(row["feedback_relative_error"]) for row in selected),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_all(args: argparse.Namespace) -> int:
    if args.repeats < 3:
        raise ValueError("paper-scale speedup requires at least three paired repeats")
    _, manifest = memory.load_manifest(args.manifest)
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    logs = out / "logs"
    logs.mkdir(parents=True)
    samples = []
    for repeat in range(1, args.repeats + 1):
        for round_index in range(1, 33):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "probe",
                "--manifest",
                str(args.manifest.resolve()),
                "--round",
                str(round_index),
                "--repeat",
                str(repeat),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": "0"})
            stem = f"round{round_index:02d}-repeat{repeat:02d}"
            (logs / f"{stem}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (logs / f"{stem}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode:
                raise RuntimeError(f"speed probe failed: {stem}; see {logs / (stem + '.stderr.txt')}")
            samples.append(json.loads(completed.stdout.strip().splitlines()[-1]))
            print(json.dumps({"repeat": repeat, "round": round_index}), flush=True)

    rounds = round_rows(samples)
    speedups = [float(row["speedup_median"]) for row in rounds]
    maya_total = sum(float(row["maya_full_s_median"]) for row in rounds)
    flex_total = sum(float(row["flexeva_refresh_s_median"]) for row in rounds)
    anchor_s = median(float(row["anchor"]["total_s"]) + float(row["anchor"]["selected_cache_s"]) for row in samples)
    summary = {
        "per_round_speedup": distribution(speedups),
        "ratio_of_round_median_totals": maya_total / flex_total,
        "maya_round_median_total_s": maya_total,
        "flexeva_round_median_total_s": flex_total,
        "one_time_anchor_init_s_median": anchor_s,
        "cumulative_speedup_with_one_anchor_at_round_32": maya_total / (anchor_s + flex_total),
        "amortized_anchor_per_round_s": anchor_s / 32.0,
    }
    result = {
        "schema": "e5.paper_scale_per_round_speedup.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {"hostname": platform.node(), "platform": platform.platform(), "python": platform.python_version()},
        "paper_config": memory.PAPER_CONFIG,
        "candidate_manifest": str(args.manifest.resolve()),
        "candidate_count": 32,
        "repeats": args.repeats,
        "method": {
            "primary_formula": "speedup_i = Maya-full_i / FlexEva-refresh_i; report the distribution of 32 round medians",
            "maya_full": "full paper-shape event generation, full trace construction, and full replay",
            "flexeva_refresh": "source diff and AL plan, selected affected source partitions, one representative per EP rank group, selected trace construction, and selected replay",
            "anchor": "Algorithm-1 anchor state and cached baseline partition contribution; excluded from the primary per-round metric",
            "execution": "each paired candidate/repeat is isolated in a fresh subprocess; Maya/FlexEva order alternates",
            "candidate_population": "32 distinct ordered route mutations grounded by fresh FakeCUDA capture; fixed world=16, EP=8, DP=2",
            "different_parallelism_manifest_recovered": False,
            "full_paper_shape_raw_fakecuda_capture": False,
            "scope": "paper-scale reference-method measurement, not the unrecovered production 16-GPU end-to-end timing contract",
        },
        "summary": summary,
        "rounds": rounds,
        "samples": samples,
        "grounding_capture": manifest["grounding_capture"],
    }
    memory.write_json(out / "result.json", result)
    write_csv(out / "per_round.csv", rounds)
    write_csv(
        out / "samples.csv",
        [
            {
                "round": row["round"],
                "repeat": row["repeat"],
                "candidate_id": row["candidate_id"],
                "route_experts": "-".join(str(value) for value in row["route_experts"]),
                "execution_order": row["execution_order"],
                "maya_full_s": row["maya_full"]["phases_s"]["total_s"],
                "flexeva_refresh_s": row["flexeva_refresh"]["phases_s"]["total_s"],
                "speedup": row["speedup"],
                "feedback_relative_error": row["feedback_relative_error"],
            }
            for row in samples
        ],
    )
    write_csv(out / "summary.csv", [{"candidate_rounds": 32, **summary["per_round_speedup"], **{key: value for key, value in summary.items() if key != "per_round_speedup"}}])
    print(json.dumps({"result": str(out / "result.json"), "per_round": str(out / "per_round.csv")}, sort_keys=True))
    return 0


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-10, abs_tol=1.0e-10)


def verify(args: argparse.Namespace) -> int:
    result_path = args.result.resolve()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "e5.paper_scale_per_round_speedup.v1":
        raise AssertionError("unexpected speedup schema")
    if payload.get("paper_config") != memory.PAPER_CONFIG or int(payload.get("candidate_count", 0)) != 32:
        raise AssertionError("paper shape or candidate count differs")
    repeats = int(payload.get("repeats", 0))
    rounds = payload["rounds"]
    samples = payload["samples"]
    if repeats < 3 or len(rounds) != 32 or len(samples) != 32 * repeats:
        raise AssertionError("paired round/repeat population is incomplete")
    if [int(row["round"]) for row in rounds] != list(range(1, 33)):
        raise AssertionError("round order differs")
    for row in samples:
        maya_s = float(row["maya_full"]["phases_s"]["total_s"])
        flex_s = float(row["flexeva_refresh"]["phases_s"]["total_s"])
        if min(maya_s, flex_s) <= 0.0 or not close(float(row["speedup"]), maya_s / flex_s):
            raise AssertionError("paired speedup accounting differs")
        if row["flexeva_refresh"]["selected_code_partitions"] != ["routing", "dispatch_collective"]:
            raise AssertionError("FlexEva refresh scope differs")
        if row["flexeva_refresh"]["selected_ranks"] != [0, 8]:
            raise AssertionError("active-lane representative set differs")
        if float(row["feedback_relative_error"]) > 1.0e-9:
            raise AssertionError("anchor-delta feedback does not match Maya-full")
    recomputed = distribution([float(row["speedup_median"]) for row in rounds])
    for key, value in recomputed.items():
        if not close(float(payload["summary"]["per_round_speedup"][key]), float(value)):
            raise AssertionError(f"speedup distribution mismatch: {key}")
    if len(list(csv.DictReader((result_path.parent / "per_round.csv").open(encoding="utf-8")))) != 32:
        raise AssertionError("per-round CSV is incomplete")
    integrity = {
        "status": "PASS",
        "paper_shape": True,
        "candidate_rounds": 32,
        "paired_repeats": repeats,
        "primary_metric": "per-round Maya-full / FlexEva-refresh",
        "anchor_excluded": True,
        "feedback_equivalent": True,
        "production_end_to_end_contract_recovered": False,
        "limitation": "reference-method abstract events are grounded by reduced-shape FakeCUDA captures; the original different-parallelism manifest and production selective executor are unavailable",
    }
    memory.write_json(result_path.parent / "integrity.json", integrity)
    print("E5 paper-scale per-round speedup verification: PASS")
    return 0


def self_test() -> int:
    values = [2.0, 2.5, 3.0, 3.5]
    stats = distribution(values)
    assert stats["count"] == 4 and stats["mean"] == 2.75 and stats["median"] == 2.75
    rows = [
        {
            "round": round_index,
            "candidate_id": f"c{round_index}",
            "route_experts": [0, round_index],
            "maya_full": {"phases_s": {"total_s": 4.0}},
            "flexeva_refresh": {"phases_s": {"total_s": 2.0}},
            "speedup": 2.0,
            "feedback_relative_error": 0.0,
        }
        for round_index in range(1, 33)
    ]
    assert len(round_rows(rows)) == 32
    print("E5 paper-scale per-round speedup self-test: PASS")
    return 0


def main() -> int:
    args = parse_args()
    if args.action == "probe":
        return run_probe(args)
    if args.action == "run":
        return run_all(args)
    if args.action == "verify":
        return verify(args)
    return self_test()


if __name__ == "__main__":
    raise SystemExit(main())
