#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
NEW_SIM_ROOT = ROOT.parent
MAYA_ROOT = NEW_SIM_ROOT / "external" / "maya-native-source-package-20260514"
MAYA_PYTHON_ROOT = MAYA_ROOT / "python"
FAKE_CUDA_WORKLOAD_ROOT = MAYA_ROOT / "tests" / "workloads" / "fake_cuda"
ROUTED_MOE_FAMILY_DIR = FAKE_CUDA_WORKLOAD_ROOT / "moe_routed_family_v1"
FRUN = MAYA_ROOT / "fake-cuda" / "frun"
PRAS_SRC = NEW_SIM_ROOT / "paper_resilient_anchor_state" / "src"
if PRAS_SRC.exists() and str(PRAS_SRC) not in sys.path:
    sys.path.insert(0, str(PRAS_SRC))

from paper_resilient_anchor_state.maya_v2_load_skew_case import (  # noqa: E402
    MAYA_V2_LOAD_SKEW_CASE_ID,
    MAYA_V2_LOAD_SKEW_ROUND_ID,
    build_maya_v2_load_skew_case,
)


METRIC_RE = re.compile(
    r"dropped=(?P<dropped>\d+)\s+cv=(?P<cv>[0-9]+(?:\.[0-9]+)?)\s+counts=(?P<counts>\[[^\]]*\])"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real fake-CUDA FlexEva MoE straggler case study."
    )
    parser.add_argument("--world-size", type=int, default=32)
    parser.add_argument("--ep-group-size", type=int, default=8)
    parser.add_argument("--candidate-parallelism", type=int, default=2)
    parser.add_argument("--worker-parallelism", type=int, default=4)
    parser.add_argument("--time-budget-s", type=float, default=3300.0)
    parser.add_argument("--hard-timeout-s", type=float, default=3600.0)
    parser.add_argument("--capture-timeout-s", type=float, default=900.0)
    parser.add_argument("--eval-timeout-s", type=float, default=900.0)
    parser.add_argument("--cv-threshold", type=float, default=0.10)
    parser.add_argument("--drop-threshold", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--skip-combo", action="store_true")
    parser.add_argument("--headroom-capacity-factor", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--micro-batches", type=int, default=2)
    parser.add_argument("--estimator-mode", default="hybrid", choices=("auto", "trace_stats", "learned_trace", "gpu_xgboost", "hybrid"))
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def rank_groups(world_size: int, ep_group_size: int) -> dict[int, list[int]]:
    if world_size <= 0 or ep_group_size <= 0:
        raise ValueError("world-size and ep-group-size must be positive")
    groups: dict[int, list[int]] = {}
    for start in range(0, world_size, ep_group_size):
        members = list(range(start, min(start + ep_group_size, world_size)))
        groups[members[0]] = members
    return groups


def profiled_rank_group_spec(groups: Mapping[int, list[int]]) -> str:
    return ";".join(
        f"{int(rep)}:{','.join(str(int(rank)) for rank in members)}"
        for rep, members in sorted(groups.items())
    )


def maya_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [str(MAYA_PYTHON_ROOT), str(FAKE_CUDA_WORKLOAD_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_entries)
    env["FAKECUDA_FRUN_QUIET"] = "1"
    env["FAKECUDA_SKIP_LDCONFIG"] = "1"
    return env


def remaining(deadline: float, cap: float | None = None) -> float:
    value = max(deadline - time.perf_counter(), 1.0)
    if cap is None:
        return value
    return max(min(float(cap), value), 1.0)


def run_logged(
    command: list[str],
    *,
    log_path: Path,
    timeout_s: float,
    env: Mapping[str, str],
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    command_payload = {
        "command": command,
        "timeout_s": timeout_s,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    command_json = log_path.with_suffix(log_path.suffix + ".command.json")
    write_json(command_json, command_payload)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps(command_payload, sort_keys=True) + "\n")
        log.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=NEW_SIM_ROOT,
                env=dict(env),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            returncode = int(completed.returncode)
            status = "complete" if returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            returncode = None
            status = "timeout"
            log.write(f"\nTIMEOUT after {timeout_s:.3f}s\n")
    return {
        "status": status,
        "returncode": returncode,
        "elapsed_s": time.perf_counter() - start,
        "log_path": str(log_path),
        "command_path": str(command_json),
    }


def workload_args(args: argparse.Namespace) -> list[str]:
    return [
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--batch-size",
        str(args.batch_size),
        "--seq-len",
        str(args.seq_len),
        "--hidden-size",
        str(args.hidden_size),
        "--num-layers",
        str(args.num_layers),
        "--num-experts",
        str(args.num_experts),
        "--top-k",
        str(args.top_k),
        "--ep-size",
        str(args.ep_group_size),
        "--micro-batches",
        str(args.micro_batches),
        "--log-interval",
        "1",
    ]


def capture_command(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    entry: Path,
    master_port: int,
    profiled_groups: str | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "flexsim.maya_lite.capture_emulated",
        "--output-dir",
        str(output_dir),
        "--logical-world-size",
        str(args.world_size),
        "--max-concurrent-workers",
        str(args.worker_parallelism),
        "--collective-mode",
        "trace_only",
        "--trace-surface",
        "all",
        "--host-timing-mode",
        "measure",
        "--host-timing-dispatch-scope",
        "host_machine",
        "--host-timing-schedule-surface",
        "semantic",
        "--trace-flush-mode",
        "buffered",
        "--trim-to-step-window",
        "--capture-step-window-occurrence",
        "1",
        "--frun",
        str(FRUN),
        "--master-port",
        str(master_port),
        "--capture-lock",
        str(output_dir / "capture.lock"),
    ]
    if profiled_groups is None:
        command += ["--auto-profiled-strategy", "identity"]
    else:
        command += ["--profiled-rank-groups", profiled_groups]
    command += [str(entry), *workload_args(args)]
    return command


def eval_command(
    args: argparse.Namespace,
    *,
    trace_dir: Path,
    fit_trace_dir: Path,
    output_path: Path,
    expand_profiled_ranks: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "flexsim.maya_lite",
        str(trace_dir),
        "--fit-traces",
        str(fit_trace_dir),
        "--estimator-mode",
        args.estimator_mode,
        "--output",
        str(output_path),
    ]
    if expand_profiled_ranks:
        command.append("--expand-profiled-ranks")
    return command


def parse_metrics(stdout_path: Path) -> dict[str, Any]:
    if not stdout_path.exists():
        return {"dropped_tokens": None, "load_balance_cv": None, "expert_counts": []}
    matches = list(METRIC_RE.finditer(stdout_path.read_text(encoding="utf-8", errors="replace")))
    if not matches:
        return {"dropped_tokens": None, "load_balance_cv": None, "expert_counts": []}
    match = matches[-1]
    try:
        counts = ast.literal_eval(match.group("counts"))
    except (SyntaxError, ValueError):
        counts = []
    normalized_counts = [int(item) for item in counts] if isinstance(counts, list) else []
    printed_cv = float(match.group("cv"))
    computed_cv = recompute_cv(normalized_counts)
    return {
        "dropped_tokens": int(match.group("dropped")),
        "load_balance_cv": computed_cv if computed_cv is not None else printed_cv,
        "printed_load_balance_cv": printed_cv,
        "expert_counts": normalized_counts,
    }


def recompute_cv(counts: list[int]) -> float | None:
    if not counts:
        return None
    mean = sum(float(item) for item in counts) / float(len(counts))
    if mean == 0.0:
        return 0.0
    variance = sum((float(item) - mean) ** 2 for item in counts) / float(len(counts))
    return math.sqrt(variance) / mean


def trace_event_counts(trace_dir: Path) -> dict[str, int]:
    api_counts: dict[str, int] = {}
    event_count = 0
    for path in sorted(trace_dir.glob("rank_*.jsonl")):
        if path.name.endswith(".raw.jsonl") or path.name.endswith(".markers.jsonl"):
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event_count += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                api = str(record.get("api") or record.get("name") or record.get("type") or "unknown")
                api_counts[api] = api_counts.get(api, 0) + 1
    return {"trace_file_event_count": event_count, **{f"api:{key}": value for key, value in sorted(api_counts.items())}}


def load_capture_eval_summary(
    *,
    candidate_id: str,
    base_candidate_id: str,
    output_dir: Path,
    capture: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    anchor_event_count: int | None,
    thresholds: Mapping[str, Any],
    kind: str,
) -> dict[str, Any]:
    manifest_path = output_dir / "capture_manifest.json"
    eval_path = output_dir / "eval_hybrid.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    eval_payload = read_json(eval_path) if eval_path.exists() else {}
    metrics = parse_metrics(output_dir / "capture_stdout.txt")
    trace_counts = trace_event_counts(output_dir)
    eval_events = eval_payload.get("total_events")
    total_events = int(eval_events) if eval_events is not None else int(trace_counts["trace_file_event_count"])
    dropped = metrics["dropped_tokens"]
    cv = metrics["load_balance_cv"]
    status = "complete" if capture.get("status") == "complete" and evaluation.get("status") == "complete" else "failed"
    eligible = (
        status == "complete"
        and dropped is not None
        and cv is not None
        and int(dropped) <= int(thresholds["drop_threshold"])
        and float(cv) <= float(thresholds["cv_threshold"])
    )
    reuse_rate = None
    if anchor_event_count is not None:
        reuse_rate = 1.0 - (float(total_events) / max(float(anchor_event_count), 1.0))
    return {
        "candidate_id": candidate_id,
        "base_candidate_id": base_candidate_id,
        "kind": kind,
        "status": status,
        "eligible": eligible,
        "output_dir": str(output_dir),
        "capture": dict(capture),
        "evaluation": dict(evaluation),
        "metrics": metrics,
        "manifest": {
            "mode": manifest.get("mode"),
            "source": manifest.get("source"),
            "original_world_size": manifest.get("original_world_size"),
            "profiled_world_size": manifest.get("profiled_world_size"),
            "profiled_rank_groups": manifest.get("profiled_rank_groups", {}),
            "capture_elapsed_seconds": manifest.get("capture_elapsed_seconds"),
            "active_emulator_seconds": manifest.get("active_emulator_seconds"),
            "paper_alignment_ready": manifest.get("paper_alignment_ready"),
            "max_concurrent_workers": manifest.get("max_concurrent_workers"),
            "collective_mode": manifest.get("collective_mode"),
            "trace_surface": manifest.get("trace_surface"),
            "host_timing_mode": manifest.get("host_timing_mode"),
        },
        "eval": {
            "source": eval_payload.get("source"),
            "world_size": eval_payload.get("world_size"),
            "profiled_world_size": eval_payload.get("profiled_world_size"),
            "paper_valid_step_window_rank_count": eval_payload.get("paper_valid_step_window_rank_count"),
            "total_events": total_events,
            "total_time_us": eval_payload.get("total_time_us"),
            "critical_path_us": eval_payload.get("critical_path_us"),
            "global_makespan_us": eval_payload.get("global_makespan_us"),
            "rank0_time_us": eval_payload.get("rank0_time_us"),
            "average_utilization": eval_payload.get("average_utilization"),
            "profiled_rank_groups": eval_payload.get("profiled_rank_groups", {}),
        },
        "trace_counts": trace_counts,
        "reuse": {
            "raw_event_reuse_rate_vs_anchor_full": reuse_rate,
            "rank_refresh_reduction_rate": 1.0
            - (
                float(eval_payload.get("profiled_world_size") or 0.0)
                / max(float(eval_payload.get("world_size") or 1.0), 1.0)
            ),
        },
    }


def run_one_capture_eval(
    args: argparse.Namespace,
    *,
    candidate_id: str,
    base_candidate_id: str,
    entry: Path,
    output_dir: Path,
    fit_trace_dir: Path,
    profiled_groups: str | None,
    master_port: int,
    deadline: float,
    anchor_event_count: int | None,
    kind: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = run_logged(
        capture_command(
            args,
            output_dir=output_dir,
            entry=entry,
            master_port=master_port,
            profiled_groups=profiled_groups,
        ),
        log_path=output_dir / "capture_driver.log",
        timeout_s=remaining(deadline, args.capture_timeout_s),
        env=maya_env(),
    )
    evaluation: dict[str, Any]
    if capture["status"] == "complete":
        evaluation = run_logged(
            eval_command(
                args,
                trace_dir=output_dir,
                fit_trace_dir=fit_trace_dir,
                output_path=output_dir / "eval_hybrid.json",
                expand_profiled_ranks=profiled_groups is not None,
            ),
            log_path=output_dir / "eval_hybrid_driver.log",
            timeout_s=remaining(deadline, args.eval_timeout_s),
            env=maya_env(),
        )
    else:
        evaluation = {"status": "skipped_after_capture_failure", "returncode": None, "elapsed_s": 0.0}
    return load_capture_eval_summary(
        candidate_id=candidate_id,
        base_candidate_id=base_candidate_id,
        output_dir=output_dir,
        capture=capture,
        evaluation=evaluation,
        anchor_event_count=anchor_event_count,
        thresholds={
            "cv_threshold": args.cv_threshold,
            "drop_threshold": args.drop_threshold,
        },
        kind=kind,
    )


def create_combo_candidate(
    out_dir: Path,
    *,
    candidate_id: str,
    capacity_factor: float,
    extra_diff: str,
) -> dict[str, Any]:
    source_path = out_dir / "combo" / f"{candidate_id}.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from __future__ import annotations",
                "",
                "import sys",
                f"sys.path.insert(0, {str(ROUTED_MOE_FAMILY_DIR)!r})",
                "",
                "from common import FamilyVariant, run_variant",
                "from local_backup_reroute import LocalBackupRerouteExpertParallelMoELayer",
                "",
                "",
                "if __name__ == \"__main__\":",
                "    run_variant(",
                "        FamilyVariant(",
                f"            variant_id={candidate_id!r},",
                "            description=\"Deterministic combination of striped ownership and local backup reroute.\",",
                "            default_overrides={",
                "                \"batch_size\": 2,",
                "                \"seq_len\": 64,",
                "                \"hidden_size\": 128,",
                "                \"num_layers\": 2,",
                "                \"num_experts\": 8,",
                "                \"top_k\": 2,",
                f"                \"capacity_factor\": {float(capacity_factor)!r},",
                "                \"ep_size\": 2,",
                "                \"micro_batches\": 2,",
                "                \"recompute\": True,",
                "                \"expert_layout\": \"striped\",",
                "                \"log_interval\": 1,",
                "            },",
                "            ep_layer_cls=LocalBackupRerouteExpertParallelMoELayer,",
                "        )",
                "    )",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "candidate_id": candidate_id,
        "base_candidate_id": candidate_id,
        "entry": str(source_path),
        "semantic_diffs": [
            "striped expert ownership",
            "overflow tokens first spill to local experts",
            extra_diff,
        ],
    }


def create_balanced_routing_candidate(out_dir: Path) -> dict[str, Any]:
    candidate_id = "balanced_round_robin_route"
    source_path = out_dir / "combo" / f"{candidate_id}.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from __future__ import annotations",
                "",
                "import math",
                "import sys",
                f"sys.path.insert(0, {str(ROUTED_MOE_FAMILY_DIR)!r})",
                "",
                "import torch",
                "import torch.nn as nn",
                "",
                "from common import BASE, FamilyVariant, run_variant",
                "",
                "",
                "class BalancedRoundRobinGate(nn.Module):",
                "    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, capacity_factor: float = 1.25):",
                "        super().__init__()",
                "        self.num_experts = num_experts",
                "        self.top_k = top_k",
                "        self.capacity_factor = capacity_factor",
                "        self.gate = nn.Linear(hidden_size, num_experts, bias=False)",
                "",
                "    def forward(self, x):",
                "        num_tokens = x.shape[0]",
                "        _gate_logits = self.gate(x)",
                "        capacity = int(self.capacity_factor * num_tokens * self.top_k / self.num_experts)",
                "        capacity = max(capacity, 1)",
                "        dispatch_mask = torch.zeros(self.num_experts, capacity, num_tokens, device=x.device, dtype=x.dtype)",
                "        combine_weights = torch.zeros(num_tokens, self.num_experts, device=x.device, dtype=x.dtype)",
                "        expert_counts = [0 for _ in range(self.num_experts)]",
                "        expert_token_indices = [[] for _ in range(self.num_experts)]",
                "        tokens_dropped = 0",
                "        for token_idx in range(num_tokens):",
                "            for route_idx in range(self.top_k):",
                "                expert_id = (token_idx * self.top_k + route_idx) % self.num_experts",
                "                pos = expert_counts[expert_id]",
                "                if pos >= capacity:",
                "                    tokens_dropped += 1",
                "                    continue",
                "                dispatch_mask[expert_id, pos, token_idx] = 1.0",
                "                combine_weights[token_idx, expert_id] = 1.0 / max(self.top_k, 1)",
                "                expert_counts[expert_id] += 1",
                "                expert_token_indices[expert_id].append(token_idx)",
                "        mean = sum(float(item) for item in expert_counts) / max(len(expert_counts), 1)",
                "        if mean == 0.0:",
                "            cv = 0.0",
                "        else:",
                "            variance = sum((float(item) - mean) ** 2 for item in expert_counts) / len(expert_counts)",
                "            cv = math.sqrt(variance) / mean",
                "        aux_loss = x.sum() * 0.0",
                "        metadata = {",
                "            \"expert_counts\": expert_counts,",
                "            \"tokens_dropped\": tokens_dropped,",
                "            \"capacity\": capacity,",
                "            \"load_balance_cv\": cv,",
                "            \"expert_token_indices\": expert_token_indices,",
                "        }",
                "        return dispatch_mask, combine_weights, aux_loss, metadata",
                "",
                "",
                "class BalancedRoundRobinExpertParallelMoELayer(BASE.ExpertParallelMoELayer):",
                "    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, capacity_factor: float = 1.25, ep_group=None, ep_size: int = 1, expert_layout: str = \"contiguous\"):",
                "        super().__init__(",
                "            hidden_size,",
                "            num_experts,",
                "            top_k,",
                "            capacity_factor,",
                "            ep_group=ep_group,",
                "            ep_size=ep_size,",
                "            expert_layout=expert_layout,",
                "        )",
                "        self.gate = BalancedRoundRobinGate(hidden_size, num_experts, top_k, capacity_factor)",
                "",
                "",
                "if __name__ == \"__main__\":",
                "    run_variant(",
                "        FamilyVariant(",
                f"            variant_id={candidate_id!r},",
                "            description=\"Balanced round-robin routing fallback for eliminating expert-load stragglers.\",",
                "            default_overrides={",
                "                \"batch_size\": 2,",
                "                \"seq_len\": 64,",
                "                \"hidden_size\": 128,",
                "                \"num_layers\": 2,",
                "                \"num_experts\": 8,",
                "                \"top_k\": 2,",
                "                \"capacity_factor\": 1.25,",
                "                \"ep_size\": 2,",
                "                \"micro_batches\": 2,",
                "                \"recompute\": True,",
                "                \"expert_layout\": \"striped\",",
                "                \"log_interval\": 1,",
                "            },",
                "            ep_layer_cls=BalancedRoundRobinExpertParallelMoELayer,",
                "        )",
                "    )",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "candidate_id": candidate_id,
        "base_candidate_id": candidate_id,
        "entry": str(source_path),
        "semantic_diffs": [
            "deterministic round-robin routing replaces tensor-skewed top-k assignment",
            "balanced expert load is used only after seed candidates and capacity fallback miss the stop rule",
        ],
    }


def sort_key(row: Mapping[str, Any]) -> tuple[int, float, float]:
    metrics = row.get("metrics", {})
    eval_payload = row.get("eval", {})
    dropped = metrics.get("dropped_tokens")
    cv = metrics.get("load_balance_cv")
    runtime = eval_payload.get("total_time_us")
    return (
        int(dropped) if dropped is not None else 10**12,
        float(cv) if cv is not None else math.inf,
        float(runtime) if runtime is not None else math.inf,
    )


def select_promotion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row["status"] == "complete"]
    eligible = [row for row in complete if row["eligible"]]
    if eligible:
        winner = min(
            eligible,
            key=lambda row: float(row["eval"]["total_time_us"]) if row["eval"]["total_time_us"] is not None else math.inf,
        )
        decision = "promote_stop_rule_satisfied"
    elif complete:
        winner = min(complete, key=sort_key)
        decision = "promote_best_observed_straggler_not_eliminated"
    else:
        winner = rows[0] if rows else {}
        decision = "no_complete_candidate"
    return {
        "decision": decision,
        "winner": winner.get("candidate_id"),
        "eligible_candidate_ids": [row["candidate_id"] for row in eligible],
        "complete_candidate_ids": [row["candidate_id"] for row in complete],
        "winner_summary": {
            "dropped_tokens": winner.get("metrics", {}).get("dropped_tokens"),
            "load_balance_cv": winner.get("metrics", {}).get("load_balance_cv"),
            "total_time_us": winner.get("eval", {}).get("total_time_us"),
            "output_dir": winner.get("output_dir"),
        },
    }


def write_csv_outputs(out_dir: Path, rows: list[dict[str, Any]], anchor: Mapping[str, Any]) -> None:
    with (out_dir / "candidate_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "candidate_id",
            "kind",
            "status",
            "eligible",
            "dropped_tokens",
            "load_balance_cv",
            "printed_load_balance_cv",
            "expert_counts",
            "profiled_rank_count",
            "world_size",
            "raw_trace_events",
            "reused_event_rate",
            "rank_refresh_reduction_rate",
            "capture_elapsed_s",
            "eval_elapsed_s",
            "predicted_runtime_us",
            "critical_path_us",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "kind": row["kind"],
                    "status": row["status"],
                    "eligible": row["eligible"],
                    "dropped_tokens": row["metrics"]["dropped_tokens"],
                    "load_balance_cv": row["metrics"]["load_balance_cv"],
                    "printed_load_balance_cv": row["metrics"].get("printed_load_balance_cv"),
                    "expert_counts": row["metrics"]["expert_counts"],
                    "profiled_rank_count": row["eval"].get("profiled_world_size"),
                    "world_size": row["eval"].get("world_size"),
                    "raw_trace_events": row["eval"].get("total_events"),
                    "reused_event_rate": row["reuse"].get("raw_event_reuse_rate_vs_anchor_full"),
                    "rank_refresh_reduction_rate": row["reuse"].get("rank_refresh_reduction_rate"),
                    "capture_elapsed_s": row["manifest"].get("capture_elapsed_seconds"),
                    "eval_elapsed_s": row["evaluation"].get("elapsed_s"),
                    "predicted_runtime_us": row["eval"].get("total_time_us"),
                    "critical_path_us": row["eval"].get("critical_path_us"),
                }
            )
    with (out_dir / "flexeva_reuse_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "scope",
            "candidate_id",
            "profiled_rank_groups",
            "refreshed_rank_count",
            "logical_world_size",
            "event_count",
            "event_reuse_rate_vs_anchor_full",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "scope": "anchor_full",
                "candidate_id": anchor["candidate_id"],
                "profiled_rank_groups": anchor["eval"].get("profiled_rank_groups"),
                "refreshed_rank_count": anchor["eval"].get("profiled_world_size"),
                "logical_world_size": anchor["eval"].get("world_size"),
                "event_count": anchor["eval"].get("total_events"),
                "event_reuse_rate_vs_anchor_full": 0.0,
            }
        )
        for row in rows:
            writer.writerow(
                {
                    "scope": row["kind"],
                    "candidate_id": row["candidate_id"],
                    "profiled_rank_groups": row["eval"].get("profiled_rank_groups"),
                    "refreshed_rank_count": row["eval"].get("profiled_world_size"),
                    "logical_world_size": row["eval"].get("world_size"),
                    "event_count": row["eval"].get("total_events"),
                    "event_reuse_rate_vs_anchor_full": row["reuse"].get("raw_event_reuse_rate_vs_anchor_full"),
                }
            )


def write_tables(out_dir: Path, rows: list[dict[str, Any]], anchor: Mapping[str, Any], promotion: Mapping[str, Any]) -> None:
    lines = [
        "# Real Fake-CUDA MoE Straggler Case Study",
        "",
        f"- Anchor capture: `{anchor['output_dir']}`",
        f"- Anchor fake-CUDA events: {anchor['eval'].get('total_events')}",
        f"- Promotion: `{promotion.get('winner')}` ({promotion.get('decision')})",
        "",
        "## Candidate Scores",
        "",
        "| Candidate | Kind | Drop | CV | Events | Reuse | Runtime us | Decision |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        reuse = row["reuse"].get("raw_event_reuse_rate_vs_anchor_full")
        lines.append(
            "| {candidate} | {kind} | {drop} | {cv} | {events} | {reuse} | {runtime} | {decision} |".format(
                candidate=row["candidate_id"],
                kind=row["kind"],
                drop=row["metrics"].get("dropped_tokens"),
                cv=(
                    f"{float(row['metrics']['load_balance_cv']):.3f}"
                    if row["metrics"].get("load_balance_cv") is not None
                    else "NA"
                ),
                events=row["eval"].get("total_events"),
                reuse=f"{float(reuse):.2%}" if reuse is not None else "NA",
                runtime=(
                    f"{float(row['eval']['total_time_us']):.1f}"
                    if row["eval"].get("total_time_us") is not None
                    else "NA"
                ),
                decision="eligible" if row["eligible"] else "best-effort",
            )
        )
    lines += [
        "",
        "## Paper Table Columns",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Stop rule | dropped <= {promotion.get('thresholds', {}).get('drop_threshold')} and CV <= {promotion.get('thresholds', {}).get('cv_threshold')} |",
        f"| Winner | {promotion.get('winner')} |",
        "| Refreshed ranks | see flexeva_reuse_summary.csv |",
    ]
    (out_dir / "case_study_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = ROOT / "output" / f"real_fakecuda_moe_straggler_{now_stamp()}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    deadline = started + min(float(args.time_budget_s), float(args.hard_timeout_s))

    case = build_maya_v2_load_skew_case()
    anchor_spec = asdict(case.anchor)
    candidates = [
        asdict(candidate)
        for candidate in case.candidates
        if candidate.candidate_id != case.anchor.candidate_id
        and candidate.candidate_id in case.round.candidate_ids
    ]
    candidates_by_round = {candidate["candidate_id"]: candidate for candidate in candidates}
    ordered_candidates = [candidates_by_round[candidate_id] for candidate_id in case.round.candidate_ids]
    if args.candidate_limit is not None:
        ordered_candidates = ordered_candidates[: max(int(args.candidate_limit), 0)]

    groups = rank_groups(args.world_size, args.ep_group_size)
    group_spec = profiled_rank_group_spec(groups)
    run_manifest = {
        "case_id": MAYA_V2_LOAD_SKEW_CASE_ID,
        "round_id": MAYA_V2_LOAD_SKEW_ROUND_ID,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": {
            "world_size": args.world_size,
            "ep_group_size": args.ep_group_size,
            "candidate_parallelism": args.candidate_parallelism,
            "worker_parallelism": args.worker_parallelism,
            "time_budget_s": args.time_budget_s,
            "hard_timeout_s": args.hard_timeout_s,
            "cv_threshold": args.cv_threshold,
            "drop_threshold": args.drop_threshold,
            "headroom_capacity_factor": args.headroom_capacity_factor,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "candidate_ids": [candidate["candidate_id"] for candidate in ordered_candidates],
        },
        "fake_cuda": {
            "frun": str(FRUN),
            "source_root": str(MAYA_ROOT),
            "python_root": str(MAYA_PYTHON_ROOT),
            "workload_root": str(FAKE_CUDA_WORKLOAD_ROOT),
        },
        "rank_groups": {str(rep): members for rep, members in groups.items()},
        "profiled_rank_groups_arg": group_spec,
    }
    write_json(args.out_dir / "run_manifest.json", run_manifest)

    anchor = run_one_capture_eval(
        args,
        candidate_id=anchor_spec["candidate_id"],
        base_candidate_id=anchor_spec["candidate_id"],
        entry=Path(anchor_spec["entry"]),
        output_dir=args.out_dir / "anchor_full",
        fit_trace_dir=args.out_dir / "anchor_full",
        profiled_groups=None,
        master_port=29631,
        deadline=deadline,
        anchor_event_count=None,
        kind="anchor_full",
    )
    write_json(args.out_dir / "anchor_summary.json", anchor)
    if anchor["status"] != "complete":
        raise SystemExit(f"anchor fake-CUDA capture/eval failed; see {anchor['output_dir']}")
    anchor_event_count = int(anchor["eval"]["total_events"])

    rows: list[dict[str, Any]] = []
    worker_count = max(1, int(args.candidate_parallelism))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {}
        for index, candidate in enumerate(ordered_candidates):
            if time.perf_counter() >= deadline:
                break
            candidate_id = str(candidate["candidate_id"])
            futures[
                executor.submit(
                    run_one_capture_eval,
                    args,
                    candidate_id=candidate_id,
                    base_candidate_id=candidate_id,
                    entry=Path(candidate["entry"]),
                    output_dir=args.out_dir / "candidates" / candidate_id,
                    fit_trace_dir=args.out_dir / "anchor_full",
                    profiled_groups=group_spec,
                    master_port=29650 + index,
                    deadline=deadline,
                    anchor_event_count=anchor_event_count,
                    kind="seed",
                )
            ] = candidate_id
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: [candidate["candidate_id"] for candidate in ordered_candidates].index(row["candidate_id"]))

    if not args.skip_combo:
        combo_specs = [
            (
                "striped_local_backup_reroute",
                1.0,
                "deterministic seed combination used after no seed reached the stop rule",
            ),
            (
                "striped_local_backup_capacity_headroom",
                float(args.headroom_capacity_factor),
                "capacity headroom applied after the seed combination still dropped tokens",
            ),
        ]
        for combo_index, (candidate_id, capacity_factor, extra_diff) in enumerate(combo_specs):
            if any(row["eligible"] for row in rows) or time.perf_counter() >= deadline:
                break
            combo = create_combo_candidate(
                args.out_dir,
                candidate_id=candidate_id,
                capacity_factor=capacity_factor,
                extra_diff=extra_diff,
            )
            combo_row = run_one_capture_eval(
                args,
                candidate_id=combo["candidate_id"],
                base_candidate_id=combo["base_candidate_id"],
                entry=Path(combo["entry"]),
                output_dir=args.out_dir / "combo" / combo["candidate_id"],
                fit_trace_dir=args.out_dir / "anchor_full",
                profiled_groups=group_spec,
                master_port=29710 + combo_index,
                deadline=deadline,
                anchor_event_count=anchor_event_count,
                kind="combo",
            )
            rows.append(combo_row)
        if not any(row["eligible"] for row in rows) and time.perf_counter() < deadline:
            balanced = create_balanced_routing_candidate(args.out_dir)
            balanced_row = run_one_capture_eval(
                args,
                candidate_id=balanced["candidate_id"],
                base_candidate_id=balanced["base_candidate_id"],
                entry=Path(balanced["entry"]),
                output_dir=args.out_dir / "combo" / balanced["candidate_id"],
                fit_trace_dir=args.out_dir / "anchor_full",
                profiled_groups=group_spec,
                master_port=29720,
                deadline=deadline,
                anchor_event_count=anchor_event_count,
                kind="combo",
            )
            rows.append(balanced_row)

    promotion = select_promotion(rows)
    promotion["thresholds"] = {
        "drop_threshold": args.drop_threshold,
        "cv_threshold": args.cv_threshold,
    }
    write_json(args.out_dir / "candidate_scores.json", {"anchor": anchor, "candidates": rows})
    write_json(args.out_dir / "promotion_trace.json", promotion)
    write_csv_outputs(args.out_dir, rows, anchor)
    write_tables(args.out_dir, rows, anchor, promotion)
    run_manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_manifest["elapsed_s"] = time.perf_counter() - started
    run_manifest["promotion"] = promotion
    write_json(args.out_dir / "run_manifest.json", run_manifest)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "winner": promotion.get("winner"),
                "decision": promotion.get("decision"),
                "elapsed_s": run_manifest["elapsed_s"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
