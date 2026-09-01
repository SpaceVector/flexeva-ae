#!/usr/bin/env python3
"""Paired full/RAS ASTRA-Sim measurements for paper Table 7."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ANCHOR_ROOT = REPO_ROOT / "inputs/table7_anchor_20260901"
PHASE_RE = re.compile(
    r"ASTRA-sim RAS simulation phase\[[^]]+\]: wall_seconds=([0-9.eE+-]+) simulated_ns=([0-9]+)"
)

WORKLOADS = {
    "gpt": {
        "label": "GPT-3 2.7B",
        "world_size": 16,
        "generator": "examples/workload/native-maya/gen_astra_chakra.py",
        "baseline_name": "2p7b",
        "comm_name": "2p7b_comm_groups.json",
        "logical_topology": "examples/network/ns3/sample_16nodes_1D.json",
        "ns3_context_stub": "0\n",
        "generator_args": [
            "--steps", "1", "--global-batch-size", "512", "--seq-len", "2048",
            "--hidden-size", "2560", "--num-layers", "32", "--num-heads", "32",
            "--vocab-size", "32000", "--tp", "1", "--pp", "8", "--dp", "2",
            "--micro-batches", "256", "--schedule", "1f1b", "--dtype", "bf16",
            "--compute-us", "10", "--optimizer-us", "5",
        ],
    },
    "moe": {
        "label": "Routed-MoE",
        "world_size": 16,
        "generator": "examples/workload/routed-moe/gen_astra_chakra.py",
        "baseline_name": "baseline_chakra",
        "comm_name": "comm_groups.json",
        "logical_topology": "examples/network/ns3/sample_16nodes_1D.json",
        "ns3_context_stub": "",
        "generator_args": [
            "--steps", "1", "--global-batch-size", "16", "--seq-len", "64",
            "--hidden-size", "128", "--num-layers", "2", "--num-heads", "4",
            "--vocab-size", "4096", "--num-experts", "16", "--top-k", "2",
            "--capacity-factor", "1.25", "--ep-size", "16", "--dp", "1",
            "--micro-batches", "1", "--dtype", "bf16", "--compute-us", "10",
            "--optimizer-us", "5",
        ],
    },
}

CASES = {
    "gpt_attention_backward": {
        "workload": "gpt", "label": "attention backward", "op": "attention_backward",
        "match": {"ranks": [0], "pp_ranks": [0], "steps": [0], "microbatches": [255], "layers": [0]},
        "duration_micros": 20, "counts": (1, 4, 80413),
    },
    "gpt_mlp_backward": {
        "workload": "gpt", "label": "MLP backward", "op": "mlp_backward",
        "match": {"ranks": [0], "pp_ranks": [0], "steps": [0], "microbatches": [255], "layers": [0]},
        "duration_micros": 20, "counts": (1, 5, 80412),
    },
    "gpt_optimizer_step": {
        "workload": "gpt", "label": "optimizer step", "op": "optimizer_step",
        "match": {"ranks": [0], "steps": [0]},
        "duration_micros": 20, "counts": (1, 0, 80417),
    },
    "moe_optimizer_step": {
        "workload": "moe", "label": "optimizer step", "op": "optimizer_step",
        "match": {"ranks": [0], "steps": [0]},
        "duration_micros": 80, "counts": (1, 0, 415),
    },
    "moe_attention_backward": {
        "workload": "moe", "label": "attention backward", "op": "attention_backward",
        "match": {"ranks": [0], "steps": [0], "microbatches": [0], "layers": [0]},
        "duration_micros": 80, "counts": (1, 1, 414),
    },
    "moe_router_backward": {
        "workload": "moe", "label": "router backward", "op": "router_backward",
        "match": {"ranks": [0], "steps": [0], "microbatches": [0], "layers": [0]},
        "duration_micros": 80, "counts": (1, 2, 413),
    },
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workload_fingerprint(prefix: Path, world_size: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    total_bytes = 0
    for rank in range(world_size):
        path = prefix.with_name(f"{prefix.name}.{rank}.et")
        data_hash = bytes.fromhex(sha256(path))
        total_bytes += path.stat().st_size
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(data_hash)
    return total_bytes, digest.hexdigest()


def run_logged(command: list[str], stdout: Path, stderr: Path, *, cwd: Path, env: dict[str, str] | None = None, timeout_s: float | None = None) -> dict[str, object]:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        completed = subprocess.run(command, cwd=cwd, env=env, text=True, stdout=out, stderr=err, timeout=timeout_s)
    record = {
        "command": command,
        "cwd": str(cwd),
        "return_code": completed.returncode,
        "elapsed_s": time.time() - started,
        "stdout": str(stdout),
        "stderr": str(stderr),
    }
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}); see {stderr}: {' '.join(command)}")
    return record


def parse_phase(stdout: Path) -> dict[str, float | int]:
    text = re.sub(
        r"\[\d{4}-\d{2}-\d{2} [^\n]*\n",
        "",
        stdout.read_text(encoding="utf-8"),
    )
    matches = PHASE_RE.findall(text)
    if len(matches) != 1:
        raise ValueError(f"expected one ASTRA-Sim phase record in {stdout}, found {len(matches)}")
    return {"wall_seconds": float(matches[0][0]), "simulated_ns": int(matches[0][1])}


def parse_replay_evidence(trace: Path, stderr: Path, replay_cache: Path, world_size: int) -> dict[str, object]:
    expected_context = json.loads(replay_cache.read_text(encoding="utf-8"))["context_fingerprint"]
    observed_context = None
    counts: dict[str, int] = {}
    for line in trace.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        kind = event.get("kind")
        if isinstance(kind, str):
            counts[kind] = counts.get(kind, 0) + 1
        if kind == "run_context":
            observed_context = event.get("payload", {}).get("context_fingerprint")
    error_text = stderr.read_text(encoding="utf-8")
    if "RAS replay disabled" in error_text:
        raise RuntimeError(f"RAS replay was disabled; see {stderr}")
    if observed_context != expected_context:
        raise RuntimeError(f"RAS context mismatch: expected {expected_context}, observed {observed_context}")
    if counts.get("ras_incremental_init") != world_size or counts.get("ras_incremental_complete") != world_size:
        raise RuntimeError(f"RAS incremental execution was not active for all {world_size} ranks")
    return {
        "enabled": True,
        "context_fingerprint": observed_context,
        "incremental_init_events": counts["ras_incremental_init"],
        "incremental_complete_events": counts["ras_incremental_complete"],
    }


def quartiles(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, q3


def make_network_config(repo_root: Path, run_dir: Path, context_stub: str) -> Path:
    output = run_dir / "ns3_output"
    output.mkdir(parents=True, exist_ok=True)
    (output / "flow.txt").write_text(context_stub, encoding="ascii")
    (output / "trace.txt").write_text(context_stub, encoding="ascii")
    paths = {
        "TOPOLOGY_FILE": repo_root / "extern/network_backend/ns-3/scratch/topology/128_nodes_16_switch_topology.txt",
        "FLOW_FILE": output / "flow.txt",
        "TRACE_FILE": output / "trace.txt",
        "TRACE_OUTPUT_FILE": output / "mix.tr",
        "FCT_OUTPUT_FILE": output / "fct.txt",
        "PFC_OUTPUT_FILE": output / "pfc.txt",
        "QLEN_MON_FILE": output / "qlen.txt",
    }
    template = repo_root / "extern/network_backend/ns-3/scratch/config/config_clos.txt"
    rendered: list[bytes] = []
    lines = template.read_bytes().split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    for line in lines:
        tokens = line.split()
        key = tokens[0].decode() if tokens else ""
        rendered.append(f"{key} {paths[key]}\n".encode() if key in paths else line + b"\n")
    config = run_dir / "config_clos.txt"
    config.write_bytes(b"".join(rendered))
    return config


def generator_command(repo_root: Path, spec: dict[str, object], prefix: Path, comm_groups: Path, mutation: Path | None, changed_events: Path | None) -> list[str]:
    command = [
        sys.executable,
        str(repo_root / str(spec["generator"])),
        "--repo-root", str(repo_root),
        "--output-prefix", str(prefix),
        "--comm-group-output", str(comm_groups),
        *[str(value) for value in spec["generator_args"]],
    ]
    if mutation is not None:
        command.extend(["--mutation-config", str(mutation)])
    if changed_events is not None:
        command.extend(["--mutation-events-output", str(changed_events)])
    return command


def run_astra(repo_root: Path, binary: Path, workload_prefix: Path, comm_groups: Path, logical_topology: Path, run_dir: Path, world_size: int, context_stub: str, replay_cache: Path | None = None, reuse_plan: Path | None = None, timeout_s: float | None = None) -> dict[str, object]:
    config = make_network_config(repo_root, run_dir, context_stub)
    trace = run_dir / "events.ras.jsonl"
    command = [
        str(binary),
        f"--workload-configuration={workload_prefix}",
        f"--system-configuration={repo_root / 'examples/system/native_collectives/Ring_4chunks.json'}",
        f"--network-configuration={config}",
        f"--remote-memory-configuration={repo_root / 'examples/remote_memory/analytical/no_memory_expansion.json'}",
        f"--logical-topology-configuration={logical_topology}",
        f"--comm-group-configuration={comm_groups}",
    ]
    env = dict(os.environ)
    env["ASTRA_SIM_RAS_TRACE"] = str(trace)
    env.pop("ASTRA_SIM_RAS_REPLAY_CACHE", None)
    env.pop("ASTRA_SIM_RAS_REUSE_PLAN", None)
    if replay_cache is not None and reuse_plan is not None:
        env["ASTRA_SIM_RAS_REPLAY_CACHE"] = str(replay_cache)
        env["ASTRA_SIM_RAS_REUSE_PLAN"] = str(reuse_plan)
    record = run_logged(command, run_dir / "stdout.txt", run_dir / "stderr.txt", cwd=binary.parent, env=env, timeout_s=timeout_s)
    record["phase"] = parse_phase(run_dir / "stdout.txt")
    record["trace"] = {"path": str(trace), "bytes": trace.stat().st_size, "sha256": sha256(trace)}
    if replay_cache is not None:
        record["replay"] = parse_replay_evidence(trace, run_dir / "stderr.txt", replay_cache, world_size)
    return record


def prepare_workload(repo_root: Path, anchor_root: Path, out_dir: Path, workload: str, logs: Path, binary: Path, rebuild_anchor: bool) -> dict[str, object]:
    spec = WORKLOADS[workload]
    manifest = json.loads((anchor_root / "manifest.json").read_text(encoding="utf-8"))["anchors"][workload]
    if manifest["world_size"] != spec["world_size"]:
        raise ValueError(f"{workload} world size differs from the persisted anchor input")
    persisted_replay_cache = anchor_root / workload / "baseline.replay_cache.json"
    if sha256(persisted_replay_cache) != manifest["replay_cache_sha256"]:
        raise ValueError(f"{workload} replay-cache hash mismatch")
    cache = json.loads(persisted_replay_cache.read_text(encoding="utf-8"))
    if cache["context_fingerprint"] != manifest["context_fingerprint"] or len(cache["workload_partitions"]) != manifest["partition_count"]:
        raise ValueError(f"{workload} replay-cache metadata mismatch")

    workload_dir = out_dir / workload / "workload"
    prefix = workload_dir / str(spec["baseline_name"])
    comm_groups = workload_dir / str(spec["comm_name"])
    run_logged(
        generator_command(repo_root, spec, prefix, comm_groups, None, None),
        logs / f"generate-{workload}-baseline.stdout.txt",
        logs / f"generate-{workload}-baseline.stderr.txt",
        cwd=repo_root,
    )
    total_bytes, fingerprint = workload_fingerprint(prefix, int(spec["world_size"]))
    if total_bytes != manifest["generated_workload_bytes"] or fingerprint != manifest["generated_workload_fingerprint"]:
        raise ValueError(f"{workload} generated baseline differs from the persisted anchor input")
    if sha256(comm_groups) != manifest["comm_groups_sha256"]:
        raise ValueError(f"{workload} communicator groups differ from the persisted anchor input")

    replay_cache = persisted_replay_cache
    anchor_record: dict[str, object] = {
        "mode": "persisted",
        "replay_cache": str(replay_cache),
        "replay_cache_sha256": sha256(replay_cache),
        "context_fingerprint": cache["context_fingerprint"],
    }
    if rebuild_anchor:
        anchor_dir = out_dir / workload / "anchor_rebuild"
        full = run_astra(
            repo_root, binary, prefix, comm_groups,
            repo_root / str(spec["logical_topology"]), anchor_dir / "full",
            int(spec["world_size"]), str(spec["ns3_context_stub"]),
        )
        print(json.dumps({"workload": workload, "mode": "anchor_full", "phase": full["phase"]}), flush=True)
        replay_cache = anchor_dir / "baseline.replay_cache.json"
        cache_build = run_logged(
            [sys.executable, str(repo_root / "utils/ras_replay_cache.py"), full["trace"]["path"], "-o", str(replay_cache), "--summary"],
            logs / f"rebuild-{workload}-anchor.stdout.txt",
            logs / f"rebuild-{workload}-anchor.stderr.txt",
            cwd=repo_root,
        )
        cache = json.loads(replay_cache.read_text(encoding="utf-8"))
        if len(cache["workload_partitions"]) != manifest["partition_count"] or cache["summary"]["missing_finish"] != 0:
            raise ValueError(f"{workload} rebuilt replay cache is incomplete")
        anchor_record = {
            "mode": "fresh_full_simulation",
            "full": full,
            "cache_build": cache_build,
            "replay_cache": str(replay_cache),
            "replay_cache_sha256": sha256(replay_cache),
            "context_fingerprint": cache["context_fingerprint"],
        }

    index = out_dir / workload / "baseline.workload_index.json"
    self_diff = out_dir / workload / "baseline.self.diff.json"
    diff_command = [
        sys.executable, str(repo_root / "utils/ras_chakra_diff.py"), str(prefix), str(prefix),
        "--repo-root", str(repo_root), "--ranks", str(spec["world_size"]),
        "--comm-group-configuration", str(comm_groups), "--baseline-replay-cache", str(replay_cache),
        "--write-baseline-index", str(index), "--compact-reuse-plan", "-o", str(self_diff), "--summary",
    ]
    run_logged(diff_command, logs / f"index-{workload}.stdout.txt", logs / f"index-{workload}.stderr.txt", cwd=repo_root)
    return {
        "prefix": prefix,
        "comm_groups": comm_groups,
        "replay_cache": replay_cache,
        "baseline_index": index,
        "logical_topology": repo_root / str(spec["logical_topology"]),
        "ns3_context_stub": str(spec["ns3_context_stub"]),
        "self_diff": self_diff,
        "world_size": int(spec["world_size"]),
        "fingerprint": fingerprint,
        "bytes": total_bytes,
        "anchor": anchor_record,
    }


def build_candidate(repo_root: Path, out_dir: Path, case_name: str, prepared: dict[str, object], logs: Path) -> dict[str, object]:
    case = CASES[case_name]
    spec = WORKLOADS[str(case["workload"])]
    case_dir = out_dir / case_name
    mutation = case_dir / "mutation.json"
    changed_events = case_dir / "changed_events.json"
    match = dict(case["match"])
    match["ops"] = [case["op"]]
    write_json(mutation, {"schema_version": "native-maya-mutation-v1", "rules": [{"match": match, "set": {"duration_micros": case["duration_micros"]}}]})
    prefix = case_dir / "workload" / f"candidate_{case_name}"
    comm_groups = case_dir / "workload" / str(spec["comm_name"])
    run_logged(
        generator_command(repo_root, spec, prefix, comm_groups, mutation, changed_events),
        logs / f"generate-{case_name}.stdout.txt", logs / f"generate-{case_name}.stderr.txt", cwd=repo_root,
    )
    events = json.loads(changed_events.read_text(encoding="utf-8"))["events"]
    if len(events) != 1 or events[0]["context"]["op"] != case["op"]:
        raise ValueError(f"{case_name} did not generate exactly its requested source mutation")
    if sha256(comm_groups) != sha256(Path(prepared["comm_groups"])):
        raise ValueError(f"{case_name} changed communicator groups")
    changed_ranks = [
        rank for rank in range(int(prepared["world_size"]))
        if sha256(prefix.with_name(f"{prefix.name}.{rank}.et"))
        != sha256(Path(prepared["prefix"]).with_name(f"{Path(prepared['prefix']).name}.{rank}.et"))
    ]
    if changed_ranks != [0]:
        raise ValueError(f"{case_name} changed unexpected rank ET files: {changed_ranks}")

    diff = case_dir / "reuse_plan.json"
    command = [
        sys.executable, str(repo_root / "utils/ras_chakra_diff.py"), str(prepared["prefix"]), str(prefix),
        "--repo-root", str(repo_root), "--ranks", str(prepared["world_size"]),
        "--comm-group-configuration", str(comm_groups), "--baseline-replay-cache", str(prepared["replay_cache"]),
        "--baseline-index", str(prepared["baseline_index"]), "--changed-events", str(changed_events),
        "--compact-reuse-plan", "-o", str(diff), "--summary",
    ]
    run_logged(command, logs / f"diff-{case_name}.stdout.txt", logs / f"diff-{case_name}.stderr.txt", cwd=repo_root)
    plan = json.loads(diff.read_text(encoding="utf-8"))
    counts = plan["reuse_plan"]["counts"]
    actual = (counts["rerun_partitions"], counts["refresh_partitions"], counts["reusable_partitions"])
    if actual != tuple(case["counts"]):
        raise ValueError(f"{case_name} reuse counts changed: {actual}")
    return {"prefix": prefix, "comm_groups": comm_groups, "changed_events": changed_events, "reuse_plan": diff, "counts": counts}


def summarize_case(case_name: str, measurements: list[dict[str, object]], counts: dict[str, int]) -> dict[str, object]:
    case = CASES[case_name]
    full = [float(row["full"]["phase"]["wall_seconds"]) for row in measurements]
    refresh = [float(row["refresh"]["phase"]["wall_seconds"]) for row in measurements]
    full_q1, full_q3 = quartiles(full)
    refresh_q1, refresh_q3 = quartiles(refresh)
    total = counts["rerun_partitions"] + counts["refresh_partitions"] + counts["reusable_partitions"] + counts.get("removed_baseline_partitions", 0)
    full_median = statistics.median(full)
    refresh_median = statistics.median(refresh)
    return {
        "case": case_name,
        "workload": WORKLOADS[str(case["workload"])]["label"],
        "mutation": case["label"],
        "world_size": WORKLOADS[str(case["workload"])]["world_size"],
        "repeats": len(measurements),
        "rerun_partitions": counts["rerun_partitions"],
        "refresh_partitions": counts["refresh_partitions"],
        "reusable_partitions": counts["reusable_partitions"],
        "total_partitions": total,
        "partition_reuse_rate": counts["reusable_partitions"] / total,
        "full_phase_median_s": full_median,
        "full_phase_q1_s": full_q1,
        "full_phase_q3_s": full_q3,
        "refresh_phase_median_s": refresh_median,
        "refresh_phase_q1_s": refresh_q1,
        "refresh_phase_q3_s": refresh_q3,
        "phase_speedup": full_median / refresh_median,
    }


def verify_payload(payload: dict[str, object], require_all: bool) -> None:
    results = {row["case"]: row for row in payload["results"]}
    if require_all and set(results) != set(CASES):
        raise AssertionError("Table 7 requires all six cases")
    for case_name, row in results.items():
        if case_name not in CASES:
            raise AssertionError(f"unexpected Table 7 case: {case_name}")
        expected_counts = tuple(CASES[case_name]["counts"])
        actual_counts = (row["rerun_partitions"], row["refresh_partitions"], row["reusable_partitions"])
        if actual_counts != expected_counts:
            raise AssertionError(f"reuse counts differ: {case_name}")
        if row["total_partitions"] != sum(actual_counts):
            raise AssertionError(f"partition accounting differs: {case_name}")
        if abs(row["partition_reuse_rate"] - row["reusable_partitions"] / row["total_partitions"]) > 1.0e-12:
            raise AssertionError(f"reuse arithmetic differs: {case_name}")
        samples = payload["measurements"][case_name]
        if len(samples) != row["repeats"] or not samples:
            raise AssertionError(f"missing repetitions: {case_name}")
        for sample in samples:
            if sample["full"]["return_code"] != 0 or sample["refresh"]["return_code"] != 0:
                raise AssertionError(f"failed ASTRA-Sim process: {case_name}")
            if sample["full"]["phase"]["simulated_ns"] != sample["refresh"]["phase"]["simulated_ns"]:
                raise AssertionError(f"full/refresh output mismatch: {case_name}")
            replay = sample["refresh"].get("replay", {})
            if not replay.get("enabled") or replay.get("incremental_init_events") != row["world_size"] or replay.get("incremental_complete_events") != row["world_size"]:
                raise AssertionError(f"RAS replay evidence is missing: {case_name}")
        full_median = statistics.median(float(sample["full"]["phase"]["wall_seconds"]) for sample in samples)
        refresh_median = statistics.median(float(sample["refresh"]["phase"]["wall_seconds"]) for sample in samples)
        if abs(full_median / refresh_median - row["phase_speedup"]) > 1.0e-9:
            raise AssertionError(f"speedup arithmetic differs: {case_name}")


def run_experiment(args: argparse.Namespace) -> int:
    # The server's protoc 3.12 binding predates its Python protobuf runtime.
    # This affects only untimed ET generation/diff; ASTRA-Sim uses C++ protobuf.
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    repo_root = args.repo_root.resolve()
    existing_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(repo_root) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    binary = args.binary.resolve()
    anchor_root = args.anchor_root.resolve()
    out_dir = args.out_dir.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"ASTRA-Sim ns-3 binary is unavailable: {binary}")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    selected = args.case or list(CASES)
    out_dir.mkdir(parents=True, exist_ok=False)
    logs = out_dir / "logs"
    logs.mkdir()
    selected_workloads = list(dict.fromkeys(str(CASES[name]["workload"]) for name in selected))
    rebuild_anchors = set(args.rebuild_anchor)
    if not rebuild_anchors.issubset(selected_workloads):
        raise ValueError("--rebuild-anchor must name a selected workload")
    prepared = {
        workload: prepare_workload(repo_root, anchor_root, out_dir, workload, logs, binary, workload in rebuild_anchors)
        for workload in selected_workloads
    }
    for workload, state in prepared.items():
        probe = run_astra(
            repo_root, binary, Path(state["prefix"]), Path(state["comm_groups"]),
            Path(state["logical_topology"]), out_dir / workload / "replay_preflight",
            int(state["world_size"]), str(state["ns3_context_stub"]),
            Path(state["replay_cache"]), Path(state["self_diff"]), timeout_s=30.0,
        )
        print(json.dumps({"workload": workload, "mode": "replay_preflight", "phase": probe["phase"]}), flush=True)
    measurements: dict[str, list[dict[str, object]]] = {}
    results: list[dict[str, object]] = []
    for case_name in selected:
        case = CASES[case_name]
        state = prepared[str(case["workload"])]
        candidate = build_candidate(repo_root, out_dir, case_name, state, logs)
        case_samples: list[dict[str, object]] = []
        for repeat in range(1, args.repeats + 1):
            runs: dict[str, object] = {}
            order = ("refresh", "full") if repeat % 2 else ("full", "refresh")
            for mode in order:
                run_dir = out_dir / case_name / f"repeat-{repeat:02d}" / mode
                runs[mode] = run_astra(
                    repo_root, binary, Path(candidate["prefix"]), Path(candidate["comm_groups"]),
                    Path(state["logical_topology"]), run_dir, int(state["world_size"]), str(state["ns3_context_stub"]),
                    Path(state["replay_cache"]) if mode == "refresh" else None,
                    Path(candidate["reuse_plan"]) if mode == "refresh" else None,
                )
                print(json.dumps({"case": case_name, "repeat": repeat, "mode": mode, "phase": runs[mode]["phase"]}), flush=True)
            if runs["full"]["phase"]["simulated_ns"] != runs["refresh"]["phase"]["simulated_ns"]:
                raise RuntimeError(f"{case_name} full and refresh simulated outputs differ")
            case_samples.append({"repeat": repeat, "order": list(order), **runs})
        measurements[case_name] = case_samples
        results.append(summarize_case(case_name, case_samples, candidate["counts"]))

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {"hostname": platform.node(), "platform": platform.platform(), "python": platform.python_version()},
        "method": {
            "timing": "ASTRA-Sim ns-3 simulation phase; excludes process and backend initialization",
            "pairing": "full and RAS refresh consume the same regenerated candidate Chakra ET prefix",
            "output_gate": "full and refresh simulated_ns must match for every repetition",
            "reuse_rate": "reusable workload partitions / total workload partitions",
            "submitted_reuse_rate_correction": "submitted percentages were 1 - 1/speedup and are not reuse measurements",
            "python_protobuf_implementation": os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"],
        },
        "source": {
            "repo_root": str(repo_root), "binary": str(binary), "binary_sha256": sha256(binary),
            "driver_sha256": sha256(Path(__file__).resolve()), "anchor_manifest": str(anchor_root / "manifest.json"),
            "anchors": {workload: state["anchor"] for workload, state in prepared.items()},
        },
        "selected_cases": selected,
        "repeats": args.repeats,
        "measurements": measurements,
        "results": results,
    }
    verify_payload(payload, require_all=set(selected) == set(CASES))
    write_json(out_dir / "result.json", payload)
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Table 7 verification: PASS ({len(results)} cases)")
    return 0


def self_test() -> int:
    assert WORKLOADS["gpt"]["world_size"] == 16
    assert WORKLOADS["gpt"]["ns3_context_stub"] == "0\n" and WORKLOADS["moe"]["ns3_context_stub"] == ""
    sample = "ASTRA-sim RAS simulation phase[single]: wall_seconds=2.5 simulated_ns=42\n"
    match = PHASE_RE.search(sample)
    assert match and float(match.group(1)) == 2.5 and int(match.group(2)) == 42
    interleaved = (
        "ASTRA-sim RAS simulation phase[[2026-08-31 20:41:17.283] [statistics] [info] noise\n"
        "single[2026-08-31 20:41:17.283] [statistics] [info] noise\n"
        "]: wall_seconds=[2026-08-31 20:41:17.283] [statistics] [info] noise\n"
        "0.000482047[2026-08-31 20:41:17.283] [statistics] [info] noise\n"
        " simulated_ns=0[2026-08-31 20:41:17.283] [statistics] [info] noise\n"
    )
    with tempfile.TemporaryDirectory() as directory:
        stdout = Path(directory) / "stdout.txt"
        stdout.write_text(interleaved, encoding="utf-8")
        assert parse_phase(stdout) == {"wall_seconds": 0.000482047, "simulated_ns": 0}
    counts = {"rerun_partitions": 1, "refresh_partitions": 1, "reusable_partitions": 8}
    runs = [{"full": {"phase": {"wall_seconds": 4.0}}, "refresh": {"phase": {"wall_seconds": 2.0}}}]
    row = summarize_case("gpt_attention_backward", runs, counts)
    assert row["partition_reuse_rate"] == 0.8 and row["phase_speedup"] == 2.0
    print("Table 7 driver self-test: PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    run.add_argument("--anchor-root", type=Path, default=ANCHOR_ROOT)
    run.add_argument("--binary", type=Path, default=REPO_ROOT / "extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default")
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--case", action="append", choices=tuple(CASES))
    run.add_argument("--rebuild-anchor", action="append", choices=tuple(WORKLOADS), default=[])
    verify = sub.add_parser("verify")
    verify.add_argument("--result", type=Path, required=True)
    verify.add_argument("--require-all", action="store_true")
    sub.add_parser("self-test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "self-test":
        return self_test()
    if args.action == "verify":
        payload = json.loads(args.result.read_text(encoding="utf-8"))
        verify_payload(payload, args.require_all)
        print("Table 7 verification: PASS")
        return 0
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
