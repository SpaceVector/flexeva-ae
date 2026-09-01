#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from statistics import median, quantiles


ROOT = Path(__file__).resolve().parents[2]
BUNDLE_SCRIPTS = ROOT / "script" / "e5" / "bundle" / "flexmaya_ras" / "scripts"
if str(BUNDLE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BUNDLE_SCRIPTS))

import flexmaya_ras as fm  # noqa: E402
import run_moe_v2_matrix as matrix  # noqa: E402


SYSTEMS = ("maya_full", "maya_trace_ras", "flexeva")
LABELS = {"maya_full": "Maya-full", "maya_trace_ras": "Maya-trace-RAS", "flexeva": "FlexEva"}
KS = (1, 8, 32)
PARTITIONS = ("routing", "memory_payload", "dispatch_collective", "expert_compute", "sync")
SELECTED_PARTITIONS = ("routing", "dispatch_collective", "sync")
PAPER_CONFIG = {
    "world_size": 16,
    "ep_group_size": 8,
    "micro_batches": 64,
    "layers": 64,
    "seq_len": 256,
    "hidden_size": 512,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and measure paper-aligned E5 candidate-state memory.")
    sub = parser.add_subparsers(dest="action", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--grounding-manifest", type=Path, required=True)
    collect.add_argument("--out-dir", type=Path, required=True)
    measure = sub.add_parser("measure")
    measure.add_argument("--manifest", type=Path, required=True)
    measure.add_argument("--out-dir", type=Path, required=True)
    measure.add_argument("--repeats", type=int, default=3)
    probe = sub.add_parser("probe")
    probe.add_argument("--manifest", type=Path, required=True)
    probe.add_argument("--system", choices=SYSTEMS, required=True)
    probe.add_argument("--k", type=int, choices=KS, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--result", type=Path, required=True)
    sub.add_parser("self-test")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def collect_sources(args: argparse.Namespace) -> int:
    grounding_path = args.grounding_manifest.resolve()
    grounding = json.loads(grounding_path.read_text(encoding="utf-8"))
    candidates = list(grounding.get("candidates", ()))
    route_pairs = [tuple(row["route"]["experts"]) for row in candidates]
    if grounding.get("status") != "complete" or len(candidates) != 32 or len(set(route_pairs)) != 32:
        raise ValueError("grounding capture must contain 32 complete, distinct route candidates")
    if int(grounding["config"]["world_size"]) != 16 or int(grounding["config"]["ep_size"]) != 8:
        raise ValueError("grounding capture must use 16 logical ranks with EP=8")

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    anchor_dir = out / "anchor"
    anchor_dir.mkdir(exist_ok=True)
    anchor_paths = {}
    for partition in PARTITIONS:
        path = anchor_dir / f"{partition}.py"
        path.write_text(f"# E5 anchor partition\nPARTITION = {partition!r}\n", encoding="utf-8")
        anchor_paths[partition] = path

    captured = []
    for index, (row, experts) in enumerate(zip(candidates, route_pairs, strict=True)):
        candidate_dir = out / "candidates" / str(row["candidate_id"])
        candidate_dir.mkdir(parents=True, exist_ok=True)
        source_paths = dict(anchor_paths)
        for partition in ("routing", "dispatch_collective"):
            path = candidate_dir / f"{partition}.py"
            path.write_text(
                "# Distinct E5 forced-route mutation\n"
                f"PARTITION = {partition!r}\nROUTE_EXPERTS = {experts!r}\nVARIANT = {index}\n",
                encoding="utf-8",
            )
            source_paths[partition] = path
        fingerprint = row["run"].get("trace_fingerprint", {})
        if int(fingerprint.get("file_count", 0)) != 32 or int(fingerprint.get("total_bytes", 0)) <= 0:
            raise ValueError(f"candidate grounding fingerprint is incomplete: {row['candidate_id']}")
        captured.append(
            {
                "candidate_id": row["candidate_id"],
                "parent": "anchor_route_0_1",
                "route_experts": list(experts),
                "source_paths": {name: str(path.relative_to(out)) for name, path in source_paths.items()},
                "source_sha256": {name: sha256(path) for name, path in source_paths.items()},
                "grounding_trace_fingerprint": fingerprint,
            }
        )

    payload = {
        "status": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "paper_config": PAPER_CONFIG,
        "candidate_count": 32,
        "candidate_parallelism": "K candidate states resident in one process; isolated subprocess per evaluator/K/repeat",
        "state_input": "paper-shape abstract events grounded by fresh 16-rank FakeCUDA route captures",
        "grounding_capture": {
            "manifest": str(grounding_path),
            "manifest_sha256": sha256(grounding_path),
            "config": grounding["config"],
            "total_trace_bytes": sum(int(row["run"]["trace_fingerprint"]["total_bytes"]) for row in candidates),
        },
        "anchor": {
            "candidate_id": "anchor_route_0_1",
            "route_experts": [0, 1],
            "source_paths": {name: str(path.relative_to(out)) for name, path in anchor_paths.items()},
        },
        "candidates": captured,
    }
    manifest = out / "candidate_manifest.json"
    write_json(manifest, payload)
    print(json.dumps({"candidate_manifest": str(manifest), "candidate_count": 32}, sort_keys=True))
    return 0


def load_manifest(path: Path) -> tuple[Path, dict]:
    manifest_path = path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("paper_config") != PAPER_CONFIG:
        raise ValueError("candidate manifest is incomplete or not paper-aligned")
    if int(payload.get("candidate_count", 0)) != 32:
        raise ValueError("paper-aligned E5 requires 32 candidates")
    return manifest_path.parent, payload


def source_paths(root: Path, row: dict) -> dict[str, Path]:
    return {name: root / value for name, value in row["source_paths"].items()}


def spec(candidate_id: str, paths: dict[str, Path]) -> fm.FlexMayaWorkloadSpec:
    return fm.FlexMayaWorkloadSpec(
        workload_id=candidate_id,
        world_size=PAPER_CONFIG["world_size"],
        code_partitions=tuple(
            fm.CodePartitionSpec(partition_id=name, path=str(paths[name]), active_ranks=tuple(range(16)))
            for name in PARTITIONS
        ),
        rank_group_policy="none",
        notes=("paper Table 8 routed-MoE abstract state",),
    )


def events(row: dict, *, ranks: range | tuple[int, ...], selected: tuple[str, ...] | None = None) -> list[object]:
    left, right = (int(value) for value in row["route_experts"])
    rows = matrix.synthetic_moe_events(
        PAPER_CONFIG,
        candidate_id=str(row["candidate_id"]),
        ranks=ranks,
        capture_code_partitions=selected,
        base_candidate_id="anchor_baseline",
        profile_scale=1.0,
    )
    route_scale = 1.0 + (left * 8 + right) / 1000.0
    for event in rows:
        if event.code_partition in {"routing", "dispatch_collective"}:
            event.duration_hint_us *= route_scale
    return rows


def current_rss_mib() -> float:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    raise RuntimeError("Linux VmRSS is unavailable")


def peak_rss_mib() -> float:
    if platform.system() != "Linux":
        raise RuntimeError("E5 peak RSS is defined only on Linux")
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def trim_heap() -> None:
    gc.collect()
    trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
    if trim is not None:
        trim(0)


def replay(trace: object) -> object:
    feedback = fm.replay_trace_once(trace)
    if feedback.cycle_detected:
        raise RuntimeError(f"trace replay cycle: {feedback.pending_summary}")
    return feedback


def run_probe(args: argparse.Namespace) -> int:
    root, manifest = load_manifest(args.manifest)
    candidates = list(manifest["candidates"])[: args.k]
    groups = matrix.rank_groups(16, 8)
    representatives = tuple(sorted(groups))
    resident: list[object] = []
    trim_heap()
    baseline_current = current_rss_mib()
    baseline_peak = peak_rss_mib()
    started = time.perf_counter()

    if args.system == "flexeva":
        anchor = manifest["anchor"]
        anchor_raw = events(anchor, ranks=range(16))
        anchor_trace = fm.build_rank_grouped_trace_ras(anchor_raw, groups)
        anchor_feedback = replay(anchor_trace)
        resident.append(("anchor", spec(anchor["candidate_id"], source_paths(root, anchor)), anchor_trace, anchor_feedback))
        del anchor_raw, anchor_trace, anchor_feedback
        trim_heap()

    input_events = 0
    resident_events = 0
    for row in candidates:
        candidate_spec = spec(row["candidate_id"], source_paths(root, row))
        if args.system == "maya_full":
            raw = events(row, ranks=range(16))
            trace = fm.build_trace_ras(raw)
            feedback = replay(trace)
            resident.append((candidate_spec, raw, trace, feedback))
            input_events += len(raw)
            resident_events += len(trace.events)
        elif args.system == "maya_trace_ras":
            raw = events(row, ranks=range(16))
            trace = fm.build_rank_grouped_trace_ras(raw, groups)
            feedback = replay(trace)
            input_events += len(raw)
            resident_events += len(trace.events)
            resident.append((candidate_spec, trace, feedback))
            del raw, trace, feedback
            trim_heap()
        else:
            raw = events(row, ranks=representatives, selected=SELECTED_PARTITIONS)
            trace = fm.build_selected_trace(
                candidate_spec,
                raw,
                selected_ranks=representatives,
                selected_code_partitions=SELECTED_PARTITIONS,
                rank_groups=groups,
            )
            feedback = replay(trace)
            input_events += len(raw)
            resident_events += len(trace.events)
            resident.append((candidate_spec, fm.source_hashes(candidate_spec), SELECTED_PARTITIONS, trace, feedback))
            del raw, trace, feedback
            trim_heap()

    trim_heap()
    peak = peak_rss_mib()
    retained = current_rss_mib()
    print(
        json.dumps(
            {
                "system": args.system,
                "system_label": LABELS[args.system],
                "k": args.k,
                "candidate_ids": [row["candidate_id"] for row in candidates],
                "baseline_rss_mib": baseline_current,
                "retained_rss_mib": retained,
                "retained_rss_delta_mib": max(retained - baseline_current, 0.0),
                "baseline_peak_rss_mib": baseline_peak,
                "peak_rss_mib": peak,
                "peak_rss_delta_mib": max(peak - baseline_peak, 0.0),
                "resident_items": len(resident),
                "input_events": input_events,
                "resident_trace_events": resident_events,
                "wall_time_s": time.perf_counter() - started,
            },
            sort_keys=True,
        )
    )
    return 0


def quartiles(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    values = quantiles(values, n=4, method="inclusive")
    return values[0], values[2]


def summarize(measurements: list[dict]) -> list[dict]:
    rows = []
    for system in SYSTEMS:
        row: dict[str, object] = {"system": system, "system_label": LABELS[system]}
        for k in KS:
            selected = [item for item in measurements if item["system"] == system and int(item["k"]) == k]
            for metric in ("peak_rss_delta_mib", "retained_rss_delta_mib", "wall_time_s"):
                values = [float(item[metric]) for item in selected]
                row[f"k{k}_{metric}_median"] = median(values)
                if metric == "peak_rss_delta_mib":
                    row[f"k{k}_{metric}_q1"], row[f"k{k}_{metric}_q3"] = quartiles(values)
        for metric in ("peak_rss_delta_mib", "retained_rss_delta_mib"):
            row[f"k32_incremental_{metric}_per_candidate"] = (
                float(row[f"k32_{metric}_median"]) - float(row[f"k1_{metric}_median"])
            ) / 31.0
        rows.append(row)
    return rows


def write_table(path: Path, summary: list[dict], metric: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Evaluator", "K=1 (GiB)", "K=8 (GiB)", "K=32 (GiB)", "Marginal delta (MiB/cand.)"))
        for row in summary:
            writer.writerow(
                (
                    row["system_label"],
                    f"{float(row[f'k1_{metric}_median']) / 1024:.2f}",
                    f"{float(row[f'k8_{metric}_median']) / 1024:.2f}",
                    f"{float(row[f'k32_{metric}_median']) / 1024:.2f}",
                    f"{float(row[f'k32_incremental_{metric}_per_candidate']):.2f}",
                )
            )


def run_measure(args: argparse.Namespace) -> int:
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    _, manifest = load_manifest(args.manifest)
    out = args.out_dir.resolve()
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    measurements = []
    for repeat in range(1, args.repeats + 1):
        for system in SYSTEMS:
            for k in KS:
                command = [sys.executable, str(Path(__file__).resolve()), "probe", "--manifest", str(args.manifest.resolve()), "--system", system, "--k", str(k)]
                completed = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": "0"})
                stem = f"{system}-k{k:02d}-repeat{repeat:02d}"
                (logs / f"{stem}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
                (logs / f"{stem}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
                if completed.returncode:
                    raise RuntimeError(f"probe failed: {stem}; see {logs / (stem + '.stderr.txt')}")
                row = json.loads(completed.stdout.strip().splitlines()[-1])
                row["repeat"] = repeat
                measurements.append(row)
                print(json.dumps({"repeat": repeat, "system": system, "k": k}), flush=True)

    summary = summarize(measurements)
    by_system = {row["system"]: row for row in summary}
    flex_peak = float(by_system["flexeva"]["k32_peak_rss_delta_mib_median"])
    result = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {"hostname": platform.node(), "platform": platform.platform(), "python": platform.python_version()},
        "paper_config": PAPER_CONFIG,
        "method": {
            "primary_metric": "process-lifetime peak RSS delta from Linux ru_maxrss",
            "secondary_metric": "post-construction retained VmRSS delta after GC and malloc_trim",
            "candidate_parallelism": "K distinct candidate states resident in one child process",
            "multiprocess": "fresh child process per evaluator/K/repeat",
            "candidate_inputs": manifest["state_input"],
            "marginal_formula": "(median RSS K=32 - median RSS K=1) / 31",
        },
        "candidate_manifest": str(args.manifest.resolve()),
        "candidate_count": 32,
        "ks": list(KS),
        "repeats": args.repeats,
        "measurements": measurements,
        "summary": summary,
        "comparisons": {
            "peak_flexeva_reduction_vs_maya_full_k32": 1.0 - flex_peak / float(by_system["maya_full"]["k32_peak_rss_delta_mib_median"]),
            "peak_flexeva_reduction_vs_maya_trace_ras_k32": 1.0 - flex_peak / float(by_system["maya_trace_ras"]["k32_peak_rss_delta_mib_median"]),
            "k32_wall_speedup_maya_full_over_flexeva": float(by_system["maya_full"]["k32_wall_time_s_median"]) / float(by_system["flexeva"]["k32_wall_time_s_median"]),
        },
    }
    write_json(out / "result.json", result)
    with (out / "measurements.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [key for key in measurements[0] if key != "candidate_ids"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in measurements)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["system", "system_label"] + sorted({key for row in summary for key in row if key not in {"system", "system_label"}})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    write_table(out / "table8.csv", summary, "peak_rss_delta_mib")
    write_table(out / "table8_retained_diagnostic.csv", summary, "retained_rss_delta_mib")
    print(json.dumps({"result": str(out / "result.json"), "table8": str(out / "table8.csv")}, sort_keys=True))
    return 0


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-9)


def run_verify(args: argparse.Namespace) -> int:
    result_path = args.result.resolve()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    manifest_path = Path(payload["candidate_manifest"])
    if not manifest_path.is_file():
        manifest_path = result_path.parent.parent / "collection" / "candidate_manifest.json"
    root, manifest = load_manifest(manifest_path)
    candidates = list(manifest["candidates"])
    if len(candidates) != 32 or len({row["candidate_id"] for row in candidates}) != 32:
        raise AssertionError("candidate identities are not distinct")
    if payload["paper_config"] != PAPER_CONFIG or tuple(payload["ks"]) != KS or int(payload["repeats"]) < 3:
        raise AssertionError("paper configuration, K set, or repeat count differs")
    measurements = payload["measurements"]
    if len(measurements) != len(SYSTEMS) * len(KS) * int(payload["repeats"]):
        raise AssertionError("measurement cell count differs")
    for row in measurements:
        k = int(row["k"])
        if row["candidate_ids"] != [item["candidate_id"] for item in candidates[:k]]:
            raise AssertionError("probe did not retain the ordered distinct candidate prefix")
        if float(row["peak_rss_mib"]) < float(row["retained_rss_mib"]) or min(float(row["peak_rss_delta_mib"]), float(row["retained_rss_delta_mib"])) <= 0:
            raise AssertionError("RSS accounting differs")
    recomputed = {row["system"]: row for row in summarize(measurements)}
    stored = {row["system"]: row for row in payload["summary"]}
    for system in SYSTEMS:
        peaks = []
        for k in KS:
            key = f"k{k}_peak_rss_delta_mib_median"
            if not close(float(recomputed[system][key]), float(stored[system][key])):
                raise AssertionError(f"summary mismatch: {system} K={k}")
            peaks.append(float(stored[system][key]))
        # ponytail: tolerate sub-MiB allocator noise; tighten only if repeated
        # server measurements show a larger stable resolution floor.
        if peaks[-1] <= peaks[0] or any(later + 1.0 < earlier for earlier, later in zip(peaks, peaks[1:])):
            raise AssertionError(f"peak RSS materially decreases: {system}")
    table_path = result_path.parent / "table8.csv"
    if len(list(csv.DictReader(table_path.open(encoding="utf-8")))) != 3:
        raise AssertionError("Table 8 output must contain three evaluator rows")
    integrity = {
        "status": "PASS",
        "paper_shape": True,
        "distinct_candidates": 32,
        "fresh_route_grounding": manifest["grounding_capture"],
        "primary_metric": "peak_rss_delta_mib",
        "peak_monotonic_tolerance_mib": 1.0,
        "full_paper_shape_raw_trace_capture": False,
        "limitation": "paper-shape states use the recovered abstract event model; fresh FakeCUDA traces ground route/lineage at a reduced capture shape",
    }
    write_json(result_path.parent / "integrity.json", integrity)
    print("E5 paper-aligned verification: PASS")
    return 0


def self_test() -> int:
    rows = [
        {
            "system": system,
            "k": k,
            "peak_rss_delta_mib": base + k * slope,
            "retained_rss_delta_mib": base / 2 + k * slope / 2,
            "wall_time_s": k,
        }
        for system, base, slope in zip(SYSTEMS, (100.0, 80.0, 60.0), (10.0, 5.0, 2.0), strict=True)
        for k in KS
        for _ in range(3)
    ]
    summary = {row["system"]: row for row in summarize(rows)}
    assert summary["maya_full"]["k32_incremental_peak_rss_delta_mib_per_candidate"] == 10.0
    assert summary["flexeva"]["k32_incremental_peak_rss_delta_mib_per_candidate"] == 2.0
    print("E5 paper-aligned driver self-test: PASS")
    return 0


def main() -> int:
    args = parse_args()
    if args.action == "collect":
        return collect_sources(args)
    if args.action == "measure":
        return run_measure(args)
    if args.action == "probe":
        return run_probe(args)
    if args.action == "verify":
        return run_verify(args)
    return self_test()


if __name__ == "__main__":
    raise SystemExit(main())
