from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


MAYA_CASE_EXPERIMENT_ID = "flexeva_maya_round03_to_round04_sparse_moe"

DEFAULT_MAYA_CASE_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "historical_sparse_moe_round04_tempcloud16"
)

REQUIRED_FAKE_TRACE_EVENT_TYPES = (
    "kernel_launch",
    "nccl_collective",
    "blas_compute",
    "mem_copy",
    "stream_op",
    "host_delay",
)

CUDA_OR_DRIVER_EVENT_TYPES = frozenset(
    {
        "kernel_launch",
        "mem_copy",
        "mem_alloc",
        "mem_free",
        "stream_op",
        "context_op",
    }
)


@dataclass(frozen=True)
class FakeCudaTraceSummary:
    trace_dir: str
    rank_count: int
    total_events: int
    event_type_counts: Mapping[str, int]
    module_counts: Mapping[str, int]
    rank_event_counts: Mapping[str, int]
    has_cuda_or_driver_events: bool
    has_nccl_events: bool
    has_blas_events: bool
    has_host_delay: bool
    raw_trace_files: tuple[str, ...]
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class SelectiveRefreshAccounting:
    logical_world_size: int
    ep_group_size: int
    anchor_rounds: int
    candidate_rounds: int
    candidate_refreshed_groups: int
    cold_maya_rank_trace_units: int
    flexeva_rank_trace_units: int
    saved_rank_trace_units: int
    reduction_vs_cold: float
    notes: tuple[str, ...]


def estimate_maya_case_selective_refresh(
    *,
    world_size: int = 16,
    ep_group_size: int = 8,
    candidate_rounds: int = 1,
) -> SelectiveRefreshAccounting:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if ep_group_size <= 0:
        raise ValueError("ep_group_size must be positive")
    if candidate_rounds < 0:
        raise ValueError("candidate_rounds must be non-negative")

    anchor_rounds = 1
    candidate_refreshed_groups = math.ceil(world_size / ep_group_size)
    cold_units = world_size * (anchor_rounds + candidate_rounds)
    flexeva_units = world_size * anchor_rounds + candidate_refreshed_groups * candidate_rounds
    saved_units = cold_units - flexeva_units
    reduction = saved_units / cold_units if cold_units else 0.0

    return SelectiveRefreshAccounting(
        logical_world_size=world_size,
        ep_group_size=ep_group_size,
        anchor_rounds=anchor_rounds,
        candidate_rounds=candidate_rounds,
        candidate_refreshed_groups=candidate_refreshed_groups,
        cold_maya_rank_trace_units=cold_units,
        flexeva_rank_trace_units=flexeva_units,
        saved_rank_trace_units=saved_units,
        reduction_vs_cold=reduction,
        notes=(
            "cold Maya baseline regenerates every rank trace for anchor and candidate",
            "FlexEva keeps anchor rank traces and refreshes one representative per EP group",
        ),
    )


def summarize_fake_cuda_trace_dir(trace_dir: Path | str) -> FakeCudaTraceSummary:
    root = Path(trace_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    manifest_path = root / "capture_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}
    trace_files = _jsonl_trace_files(root)
    if not trace_files:
        raise ValueError(f"no jsonl trace files found under {root}")

    event_type_counts: dict[str, int] = {}
    module_counts: dict[str, int] = {}
    rank_event_counts: dict[str, int] = {}
    total_events = 0
    has_cuda_or_driver_events = False
    has_nccl_events = False
    has_blas_events = False
    has_host_delay = False

    for trace_file in trace_files:
        rank_key = _rank_key(trace_file)
        rank_total = 0
        for record in _iter_jsonl_records(trace_file):
            event_type = _event_type(record)
            module = _module_name(record)
            api = _api_name(record)
            rank_total += 1
            total_events += 1
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            module_counts[module] = module_counts.get(module, 0) + 1
            lower_probe = f"{module} {api} {record.get('name', '')}".lower()
            has_cuda_or_driver_events = (
                has_cuda_or_driver_events
                or event_type in CUDA_OR_DRIVER_EVENT_TYPES
                or "cuda" in lower_probe
                or "cudart" in lower_probe
                or "libcuda" in lower_probe
            )
            has_nccl_events = (
                has_nccl_events
                or event_type == "nccl_collective"
                or "nccl" in lower_probe
            )
            has_blas_events = (
                has_blas_events
                or event_type == "blas_compute"
                or "cublas" in lower_probe
                or "cublaslt" in lower_probe
            )
            has_host_delay = has_host_delay or _is_host_delay(record, event_type)
        rank_event_counts[rank_key] = rank_event_counts.get(rank_key, 0) + rank_total

    raw_trace_files = tuple(str(path.relative_to(root)) for path in trace_files)
    return FakeCudaTraceSummary(
        trace_dir=str(root),
        rank_count=len(rank_event_counts),
        total_events=total_events,
        event_type_counts=dict(sorted(event_type_counts.items())),
        module_counts=dict(sorted(module_counts.items())),
        rank_event_counts=dict(sorted(rank_event_counts.items())),
        has_cuda_or_driver_events=has_cuda_or_driver_events,
        has_nccl_events=has_nccl_events,
        has_blas_events=has_blas_events,
        has_host_delay=has_host_delay,
        raw_trace_files=raw_trace_files,
        manifest=manifest,
    )


def build_maya_case_experiment_manifest(
    *,
    fixture_root: Path | str = DEFAULT_MAYA_CASE_FIXTURE_ROOT,
    trace_dir: Path | str | None = None,
    world_size: int = 16,
    ep_group_size: int = 8,
) -> dict[str, Any]:
    root = Path(fixture_root)
    ras_expected = _load_json(root / "ras_expected_layers.json")
    execution_substrate = _load_json(root / "execution_substrate.json")
    round03_doc = _load_json(root / "round03_anchor_summary.json")
    round04_doc = _load_json(root / "round04_candidate_summary.json")
    round03 = round03_doc["summary"]
    round04 = round04_doc["summary"]

    a0 = _layer_by_id(ras_expected, "A0_code")
    a1 = _layer_by_id(ras_expected, "A1_code_mutation")
    a2 = _layer_by_id(ras_expected, "A2_semantic")
    a3 = _layer_by_id(ras_expected, "A3_runtime_config")

    trace_section: dict[str, Any] = {
        "mode": "fake_cuda_proot",
        "required_event_types": list(REQUIRED_FAKE_TRACE_EVENT_TYPES),
        "must_not_use": ["real CUDA", "real NCCL", "real cuBLAS", "LD_PRELOAD"],
        "unsupported_behavior_policy": "report_explicitly",
    }
    if trace_dir is not None:
        trace_section["summary"] = asdict(summarize_fake_cuda_trace_dir(trace_dir))

    accounting = estimate_maya_case_selective_refresh(
        world_size=world_size,
        ep_group_size=ep_group_size,
        candidate_rounds=1,
    )

    round03_a2a = float(round03["estimated_a2a_bytes"])
    round04_a2a = float(round04["estimated_a2a_bytes"])
    round03_rerouted = float(round03["tokens_rerouted"])
    round04_rerouted = float(round04["tokens_rerouted"])

    return {
        "schema_version": 1,
        "experiment_id": MAYA_CASE_EXPERIMENT_ID,
        "primary_claim": "selective_trace_refresh_preserves_oracle_decision_signal",
        "scope": {
            "anchor_id": ras_expected["anchor_id"],
            "candidate_id": ras_expected["candidate_id"],
            "workload_family": "historical_sparse_moe",
            "fixture_root": str(root),
            "world_size": world_size,
            "ep_group_size": ep_group_size,
        },
        "pipeline": [
            "source_diff",
            "code_analysis_ras",
            "grounding_targets",
            "fake_cuda_maya_trace_ras",
            "selective_trace_refresh_report",
            "native_tempcloud_oracle_comparison",
        ],
        "source_scope": {
            "included_files": list(a0.get("files", ())),
            "changed_files": list(a1.get("changed_files", ())),
            "blackbox_libraries": ["PyTorch", "CUDA", "NCCL", "cuBLAS", "cuDNN", "driver"],
        },
        "frontend_analysis": {
            "method": "build_CFG_then_check_control_predicates",
            "predicate_dependency_rule": (
                "track dependencies to rank/configuration values and to runtime emission returns"
            ),
            "static_lane_placement": "allowed only for rank/configuration-dependent predicates",
            "runtime_dependent_predicates": "become grounding targets",
            "expected_semantic_points": list(a2.get("expected_points", ())),
            "expected_runtime_config": dict(a3.get("expected_summary", {})),
        },
        "grounding_targets": [
            {
                "target": "moe_routing_predicate_distribution",
                "reason": "token routing depends on runtime tensor values",
                "evidence": "native minimal run or backend-observed predicate feature distribution",
            },
            {
                "target": "dispatch_footprint",
                "reason": "all-to-all traffic depends on runtime routing output",
                "evidence": "estimated_a2a_bytes from oracle summaries",
            },
        ],
        "maya_trace": trace_section,
        "native_oracle": {
            "execution_target": execution_substrate["execution_target"],
            "world_size": execution_substrate["world_size"],
            "nodes": execution_substrate["nodes"],
            "gpus_per_node": execution_substrate["gpus_per_node"],
            "required_ib_hcas": list(execution_substrate["required_ib_hcas"]),
            "required_nccl_env": dict(execution_substrate["required_nccl_env"]),
            "validation_requirements": list(execution_substrate["validation_requirements"]),
            "round03_summary_path": "round03_anchor_summary.json",
            "round04_summary_path": "round04_candidate_summary.json",
        },
        "semantic_signal": {
            "round03_name": round03_doc["round_name"],
            "round04_name": round04_doc["round_name"],
            "round03_status": round03["status"],
            "round04_status": round04["status"],
            "round03_tokens_rerouted": round03_rerouted,
            "round04_tokens_rerouted": round04_rerouted,
            "round03_estimated_a2a_bytes": round03_a2a,
            "round04_estimated_a2a_bytes": round04_a2a,
            "estimated_a2a_reduction_bytes": round03_a2a - round04_a2a,
            "decision_signal_preserved": (
                round03_rerouted > 0
                and round04_rerouted == 0
                and round04_a2a < round03_a2a
                and round03["status"] == "complete"
                and round04["status"] == "complete"
            ),
        },
        "selective_refresh": asdict(accounting),
        "acceptance": [
            "round03_tokens_rerouted_gt_zero",
            "round04_tokens_rerouted_eq_zero",
            "round04_estimated_a2a_lt_round03",
            "fake_cuda_trace_summary_nonempty",
            "unsupported_behavior_reported_explicitly",
        ],
        "out_of_scope": [
            "arbitrary_pytorch_cuda_heap_checkpoint_resume",
            "full_five_round_historical_sparse_moe_evaluation",
            "astra_sim_adapter_implementation",
            "latex_section_rewrite",
        ],
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _layer_by_id(ras_expected: Mapping[str, Any], layer_id: str) -> Mapping[str, Any]:
    for layer in ras_expected.get("layers", ()):
        if layer.get("layer_id") == layer_id:
            return layer
    raise KeyError(layer_id)


def _jsonl_trace_files(root: Path) -> tuple[Path, ...]:
    rank_files = tuple(sorted(root.glob("rank_*.jsonl")))
    if rank_files:
        return rank_files
    return tuple(sorted(path for path in root.glob("*.jsonl") if not path.name.startswith(".")))


def _iter_jsonl_records(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{lineno}") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"trace record in {path}:{lineno} is not an object")
            yield record


def _event_type(record: Mapping[str, Any]) -> str:
    raw_type = record.get("type") or record.get("op_type") or record.get("event_type")
    if raw_type:
        return _normalize_event_type(str(raw_type))

    api = _api_name(record).lower()
    module = _module_name(record).lower()
    name = str(record.get("name", "")).lower()
    probe = f"{api} {module} {name}"
    if "__hostdelay__" in probe or "host_delay" in probe:
        return "host_delay"
    if "nccl" in probe:
        return "nccl_collective"
    if "cublas" in probe or "cublaslt" in probe:
        return "blas_compute"
    if "launchkernel" in probe or "kernel" in probe:
        return "kernel_launch"
    if "memcpy" in probe:
        return "mem_copy"
    if "malloc" in probe or "alloc" in probe:
        return "mem_alloc"
    if "free" in probe:
        return "mem_free"
    if "stream" in probe or "event" in probe:
        return "stream_op"
    if "cuda" in probe or "driver" in probe:
        return "context_op"
    return "unknown"


def _normalize_event_type(raw_type: str) -> str:
    normalized = raw_type.strip().lower()
    aliases = {
        "cuda_launch": "kernel_launch",
        "cuda_kernel": "kernel_launch",
        "kernel": "kernel_launch",
        "collective": "nccl_collective",
        "nccl": "nccl_collective",
        "gemm": "blas_compute",
        "blas": "blas_compute",
        "copy": "mem_copy",
        "memcpy": "mem_copy",
        "stream": "stream_op",
        "cuda_stream": "stream_op",
        "host": "host_delay",
        "delay": "host_delay",
    }
    return aliases.get(normalized, normalized)


def _api_name(record: Mapping[str, Any]) -> str:
    return str(record.get("api") or record.get("name") or record.get("symbol") or "")


def _module_name(record: Mapping[str, Any]) -> str:
    module = record.get("mod") or record.get("module") or record.get("library")
    return str(module) if module else "<unknown>"


def _rank_key(trace_file: Path) -> str:
    match = re.search(r"rank[_-]?(\d+)", trace_file.stem)
    if match:
        return f"rank:{int(match.group(1))}"
    return trace_file.stem


def _is_host_delay(record: Mapping[str, Any], event_type: str) -> bool:
    if event_type == "host_delay":
        return True
    probe = " ".join(
        str(record.get(key, ""))
        for key in ("api", "name", "semantic_role", "role", "region")
    ).lower()
    return "__hostdelay__" in probe or "host_delay" in probe
