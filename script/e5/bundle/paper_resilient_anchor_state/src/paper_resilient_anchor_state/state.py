"""Standalone resilient-anchor-state reference implementation.

This package is the paper-facing extraction of the four anchor-state slots used
by the partial rerun evaluator:

- anchor code
- anchor semantic
- anchor runtime values
- anchor trace

It is intentionally self-contained. The implementation depends only on the
Python standard library and does not import the larger `test-sim` codebase.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


class ProgramLogicKind(str, Enum):
    RANK_PARTITION = "rank_partition"
    REGION_ACTIVATION = "region_activation"
    STAGE_ORDER = "stage_order"
    UNKNOWN = "unknown"


class RegionKind(str, Enum):
    DISPATCH = "dispatch"
    EXPERT_COMPUTE = "expert_compute"
    COLLECTIVE = "collective"
    OVERLAP = "overlap"
    MEMORY = "memory"
    OTHER = "other"


class DependencyType(str, Enum):
    NARROW = "narrow"
    WIDE = "wide"


class RuntimeValuePointKind(str, Enum):
    BRANCH = "branch"
    COUNT = "count"
    BUCKET = "bucket"
    STAGE = "stage"


@dataclass(frozen=True)
class RuntimeValuePoint:
    name: str
    kind: RuntimeValuePointKind
    description: str
    required_fields: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkloadSemanticContract:
    workload_family: str
    runtime_value_points: tuple[RuntimeValuePoint, ...]
    notes: tuple[str, ...] = ()


def default_contract_for_workload_family(workload_family: str) -> WorkloadSemanticContract | None:
    lowered = workload_family.lower()
    if "moe" in lowered:
        return WorkloadSemanticContract(
            workload_family="routed_moe",
            runtime_value_points=(
                RuntimeValuePoint(
                    name="route_decision",
                    kind=RuntimeValuePointKind.BRANCH,
                    description="Token routing choice and fanout bucket.",
                    required_fields=("routing.top_k", "routing.dispatch_footprint"),
                ),
                RuntimeValuePoint(
                    name="expert_load",
                    kind=RuntimeValuePointKind.COUNT,
                    description="Per-expert or per-rank load summary.",
                ),
                RuntimeValuePoint(
                    name="overflow_state",
                    kind=RuntimeValuePointKind.COUNT,
                    description="Overflow / drop / reroute summary.",
                ),
                RuntimeValuePoint(
                    name="remote_dispatch",
                    kind=RuntimeValuePointKind.BUCKET,
                    description="Local-vs-remote dispatch pressure summary.",
                ),
                RuntimeValuePoint(
                    name="collective_stage",
                    kind=RuntimeValuePointKind.STAGE,
                    description="Collective and overlap stage ordering summary.",
                ),
            ),
            notes=("Reference workload semantic contract for routed-MoE anchor states.",),
        )
    if "gpt" in lowered or "megatron" in lowered:
        return WorkloadSemanticContract(
            workload_family="gpt_pipeline",
            runtime_value_points=(
                RuntimeValuePoint(
                    name="pipeline_stage_exit_sync",
                    kind=RuntimeValuePointKind.BRANCH,
                    description="Stage-exit tensor-parallel synchronization branch.",
                ),
                RuntimeValuePoint(
                    name="pipeline_p2p_mode",
                    kind=RuntimeValuePointKind.STAGE,
                    description="Pipeline send/recv schedule mode.",
                ),
                RuntimeValuePoint(
                    name="microbatch_cadence",
                    kind=RuntimeValuePointKind.COUNT,
                    description="Microbatch count and cadence summary.",
                ),
            ),
            notes=("Reference workload semantic contract for GPT-style pipeline traces.",),
        )
    return None


@dataclass(frozen=True)
class LogicScopeSpec:
    scope_id: str
    selected_paths: tuple[str, ...]
    selected_functions: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProgramLogicPoint:
    name: str
    kind: ProgramLogicKind
    value: Any
    source: str
    source_path: str | None = None
    lineno: int | None = None
    end_lineno: int | None = None
    branch_ids: tuple[int, ...] = ()
    source_fields: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProgramLogicCarrier:
    source: str
    points: tuple[ProgramLogicPoint, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperatorStub:
    stub_id: str
    emission_signature: str
    site_signature: str
    callee_name: str
    source_path: str
    lineno: int
    boundary_kind: str = "opaque_call"
    branch_ids: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReturnSummary:
    return_kind: str
    logic_observable: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SideEffectSummary:
    mutates_state: bool = False
    advances_group_state: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BoundaryContextCapsule:
    capsule_id: str
    stub_id: str
    emission_signature: str
    site_signature: str
    callee_name: str
    source_path: str
    lineno: int
    boundary_kind: str
    branch_ids: tuple[int, ...] = ()
    positional_arg_kinds: tuple[str, ...] = ()
    keyword_arg_names: tuple[str, ...] = ()
    return_summary: ReturnSummary | None = None
    side_effect_summary: SideEffectSummary | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DryRunProgramLogicCapture:
    code_path: str
    world_size: int
    branch_signatures: dict[int, tuple[tuple[int, bool, bool], ...]]
    semantic_summaries: dict[int, dict[str, dict[str, Any]]]
    program_logic: ProgramLogicCarrier
    logic_scope: LogicScopeSpec
    operator_stubs: tuple[OperatorStub, ...] = ()
    boundary_capsules: tuple[BoundaryContextCapsule, ...] = ()
    control_region_ids: tuple[str, ...] = ()
    logic_slice_ids: tuple[str, ...] = ()
    logic_slice_edges: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class CodeFileSnapshot:
    path: str
    sha1: str
    line_count: int


@dataclass(frozen=True)
class CodeMutationHunk:
    before_path: str
    after_path: str
    before_start_line: int
    before_end_line: int
    after_start_line: int
    after_end_line: int
    before_lines: tuple[str, ...]
    after_lines: tuple[str, ...]
    summary: str


@dataclass(frozen=True)
class CodeMutationRound:
    round_index: int
    before_path: str
    after_path: str
    before_sha1: str
    after_sha1: str
    hunk_count: int
    round_id: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnchorCodeState:
    baseline_files: tuple[CodeFileSnapshot, ...]
    mutation_hunks: tuple[CodeMutationHunk, ...] = ()
    mutation_rounds: tuple[CodeMutationRound, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticMutationSummary:
    changed_logic_point_names: tuple[str, ...]
    added_stub_ids: tuple[str, ...]
    removed_stub_ids: tuple[str, ...]
    changed_capsule_ids: tuple[str, ...]
    incompatible_capsule_ids: tuple[str, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnchorSemanticState:
    source: str
    logic_scope: LogicScopeSpec | None
    program_logic_carrier: ProgramLogicCarrier
    operator_stubs: tuple[OperatorStub, ...] = ()
    boundary_capsules: tuple[BoundaryContextCapsule, ...] = ()
    control_region_count: int = 0
    logic_slice_count: int = 0
    logic_slice_edge_count: int = 0
    mutation: SemanticMutationSummary | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeValueDistribution:
    point_name: str
    source_kind: str
    sample_count: int
    observed_values: tuple[str, ...]
    distribution: dict[str, float]
    supporting_fields: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeValueMutationSummary:
    changed_point_names: tuple[str, ...]
    added_point_names: tuple[str, ...]
    removed_point_names: tuple[str, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnchorRuntimeValueState:
    workload_family: str
    contract: WorkloadSemanticContract | None
    distributions: tuple[RuntimeValueDistribution, ...]
    mutation: RuntimeValueMutationSummary | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnchorRegion:
    region_id: str
    region_kind: RegionKind
    order_index: int
    timing_share: float
    start_window: int | None = None
    end_window: int | None = None
    lane_id: str | None = None
    semantic_role: str | None = None
    notes: tuple[str, ...] = ()
    value_sensitive: bool = False
    hardware_sensitive: bool = False
    criticality_slack: float | None = None
    dependency_type: DependencyType = DependencyType.NARROW


@dataclass(frozen=True)
class AnchorWitness:
    anchor_candidate_id: str
    workload_family: str
    world_size: int
    regions: tuple[AnchorRegion, ...]
    artifacts: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def critical_region_ids(self, *, slack_threshold: float = 0.10) -> tuple[str, ...]:
        return tuple(
            region.region_id
            for region in self.regions
            if region.criticality_slack is not None and region.criticality_slack <= slack_threshold
        )


@dataclass(frozen=True)
class TraceRankSummary:
    rank: int
    event_count: int
    start_ts: int | None
    end_ts: int | None
    span_us: int | None
    dominant_mods: tuple[str, ...] = ()


@dataclass(frozen=True)
class TraceMutationSummary:
    candidate_trace_dir: str
    total_event_delta: int
    per_rank_event_delta: dict[int, int]
    changed_region_positions: tuple[int, ...]
    added_region_signatures: tuple[str, ...]
    removed_region_signatures: tuple[str, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TraceStepWindow:
    start_ts: int | None
    end_ts: int | None
    source: str | None
    step_count: int | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnchorTraceState:
    trace_dir: str
    world_size: int
    witness: AnchorWitness
    rank_summaries: tuple[TraceRankSummary, ...]
    region_signatures: tuple[str, ...]
    step_window: TraceStepWindow | None = None
    step_window_source: str | None = None
    lane_ids: tuple[str, ...] = ()
    mutation: TraceMutationSummary | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResilientAnchorState:
    anchor_candidate_id: str
    workload_family: str
    world_size: int
    code: AnchorCodeState
    semantic: AnchorSemanticState
    runtime_values: AnchorRuntimeValueState
    trace: AnchorTraceState
    notes: tuple[str, ...] = ()


def _stable_repr(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, (list, tuple)):
        return json.dumps(value)
    return repr(value)


def _distribution(values: Iterable[str]) -> dict[str, float]:
    values = [str(item) for item in values]
    if not values:
        return {}
    counts = Counter(values)
    total = float(sum(counts.values()))
    return {key: counts[key] / total for key in sorted(counts)}


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _snapshot(path: str | Path) -> CodeFileSnapshot:
    resolved = Path(path).resolve()
    text = _read_text(resolved)
    return CodeFileSnapshot(
        path=str(resolved),
        sha1=_sha1_text(text),
        line_count=len(text.splitlines()),
    )


def _path_sha1(path: str | Path) -> str:
    return _sha1_text(_read_text(path))


def _summarize_hunk(before_lines: list[str], after_lines: list[str]) -> str:
    before_text = " ".join(line.strip() for line in before_lines if line.strip())
    after_text = " ".join(line.strip() for line in after_lines if line.strip())
    if before_text and after_text:
        return f"{before_text} -> {after_text}"
    if before_text:
        return f"remove {before_text}"
    if after_text:
        return f"add {after_text}"
    return "empty change"


def _build_mutation_hunks(before_path: str | Path, after_path: str | Path) -> tuple[CodeMutationHunk, ...]:
    resolved_before = Path(before_path).resolve()
    resolved_after = Path(after_path).resolve()
    before_lines = _read_text(resolved_before).splitlines()
    after_lines = _read_text(resolved_after).splitlines()
    matcher = SequenceMatcher(a=before_lines, b=after_lines)
    hunks: list[CodeMutationHunk] = []
    for tag, before_start, before_end, after_start, after_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        before_slice = tuple(before_lines[before_start:before_end])
        after_slice = tuple(after_lines[after_start:after_end])
        hunks.append(
            CodeMutationHunk(
                before_path=str(resolved_before),
                after_path=str(resolved_after),
                before_start_line=before_start + 1,
                before_end_line=max(before_end, before_start + 1),
                after_start_line=after_start + 1,
                after_end_line=max(after_end, after_start + 1),
                before_lines=before_slice,
                after_lines=after_slice,
                summary=_summarize_hunk(list(before_slice), list(after_slice)),
            )
        )
    return tuple(hunks)


def build_anchor_code_state(
    *,
    anchor_code_paths: Iterable[str | Path] = (),
    code_mutation_pairs: Iterable[tuple[str | Path, str | Path]] = (),
) -> AnchorCodeState:
    baseline_files = tuple(_snapshot(path) for path in anchor_code_paths)
    mutation_hunks: list[CodeMutationHunk] = []
    mutation_rounds: list[CodeMutationRound] = []
    for round_index, (before_path, after_path) in enumerate(code_mutation_pairs):
        hunks = _build_mutation_hunks(before_path, after_path)
        mutation_hunks.extend(hunks)
        resolved_before = Path(before_path).resolve()
        resolved_after = Path(after_path).resolve()
        mutation_rounds.append(
            CodeMutationRound(
                round_index=round_index,
                before_path=str(resolved_before),
                after_path=str(resolved_after),
                before_sha1=_path_sha1(resolved_before),
                after_sha1=_path_sha1(resolved_after),
                hunk_count=len(hunks),
                round_id=f"round_{round_index:02d}",
                notes=("mutation round derived from ordered code_mutation_pairs",),
            )
        )
    notes: list[str] = []
    if not baseline_files:
        notes.append("no baseline code files were attached")
    if not mutation_hunks:
        notes.append("no explicit code mutation hunks were attached")
    return AnchorCodeState(
        baseline_files=baseline_files,
        mutation_hunks=tuple(mutation_hunks),
        mutation_rounds=tuple(mutation_rounds),
        notes=tuple(notes),
    )


def _program_logic_deltas(
    before: ProgramLogicCarrier | None,
    after: ProgramLogicCarrier | None,
) -> tuple[str, ...]:
    before_map = {point.name: point.value for point in before.points} if before else {}
    after_map = {point.name: point.value for point in after.points} if after else {}
    changed = [
        name
        for name in sorted(set(before_map) | set(after_map))
        if before_map.get(name) != after_map.get(name)
    ]
    return tuple(changed)


def _compare_operator_stubs(
    before: tuple[OperatorStub, ...],
    after: tuple[OperatorStub, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    before_keys = {(stub.emission_signature, stub.site_signature, stub.callee_name): stub.stub_id for stub in before}
    after_keys = {(stub.emission_signature, stub.site_signature, stub.callee_name): stub.stub_id for stub in after}
    added = tuple(
        after_keys[key]
        for key in sorted(set(after_keys) - set(before_keys))
    )
    removed = tuple(
        before_keys[key]
        for key in sorted(set(before_keys) - set(after_keys))
    )
    return added, removed


def _capsule_contract_fields(capsule: BoundaryContextCapsule) -> tuple[Any, ...]:
    return (
        capsule.callee_name,
        capsule.boundary_kind,
        capsule.branch_ids,
        capsule.positional_arg_kinds,
        capsule.keyword_arg_names,
        capsule.return_summary.return_kind if capsule.return_summary else "none",
        capsule.return_summary.logic_observable if capsule.return_summary else False,
        capsule.side_effect_summary.mutates_state if capsule.side_effect_summary else False,
        capsule.side_effect_summary.advances_group_state if capsule.side_effect_summary else False,
    )


def _compare_capsules(
    before: tuple[BoundaryContextCapsule, ...],
    after: tuple[BoundaryContextCapsule, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    before_map = {capsule.capsule_id: _capsule_contract_fields(capsule) for capsule in before}
    after_map = {capsule.capsule_id: _capsule_contract_fields(capsule) for capsule in after}
    changed: list[str] = []
    incompatible: list[str] = []
    for capsule_id in sorted(set(before_map) & set(after_map)):
        if before_map[capsule_id] == after_map[capsule_id]:
            continue
        changed.append(capsule_id)
        incompatible.append(capsule_id)
    return tuple(changed), tuple(incompatible)


def build_anchor_semantic_state(
    *,
    anchor_capture: DryRunProgramLogicCapture | None = None,
    anchor_program_logic: ProgramLogicCarrier | None = None,
    candidate_capture: DryRunProgramLogicCapture | None = None,
) -> AnchorSemanticState:
    if anchor_capture is None and anchor_program_logic is None:
        raise ValueError("anchor_capture or anchor_program_logic is required")
    carrier = anchor_capture.program_logic if anchor_capture is not None else anchor_program_logic
    assert carrier is not None
    mutation = None
    notes: list[str] = []
    if anchor_capture is not None:
        notes.append("semantic state is backed by dry-run capture and logic slices")
    else:
        notes.append("semantic state is backed only by a program-logic carrier")
    if candidate_capture is not None:
        changed_logic = _program_logic_deltas(carrier, candidate_capture.program_logic)
        added_stubs, removed_stubs = _compare_operator_stubs(
            anchor_capture.operator_stubs if anchor_capture is not None else (),
            candidate_capture.operator_stubs,
        )
        changed_capsules, incompatible_capsules = _compare_capsules(
            anchor_capture.boundary_capsules if anchor_capture is not None else (),
            candidate_capture.boundary_capsules,
        )
        if changed_logic or added_stubs or removed_stubs or changed_capsules or incompatible_capsules:
            mutation = SemanticMutationSummary(
                changed_logic_point_names=changed_logic,
                added_stub_ids=added_stubs,
                removed_stub_ids=removed_stubs,
                changed_capsule_ids=changed_capsules,
                incompatible_capsule_ids=incompatible_capsules,
                notes=("semantic mutation is computed from logic, stub, and capsule deltas",),
            )
    return AnchorSemanticState(
        source="dryrun_capture" if anchor_capture is not None else "program_logic_carrier",
        logic_scope=anchor_capture.logic_scope if anchor_capture is not None else None,
        program_logic_carrier=carrier,
        operator_stubs=anchor_capture.operator_stubs if anchor_capture is not None else (),
        boundary_capsules=anchor_capture.boundary_capsules if anchor_capture is not None else (),
        control_region_count=len(anchor_capture.control_region_ids) if anchor_capture is not None else 0,
        logic_slice_count=len(anchor_capture.logic_slice_ids) if anchor_capture is not None else 0,
        logic_slice_edge_count=len(anchor_capture.logic_slice_edges) if anchor_capture is not None else 0,
        mutation=mutation,
        notes=tuple(notes),
    )


def _branch_distributions(capture: DryRunProgramLogicCapture) -> tuple[RuntimeValueDistribution, ...]:
    rows: dict[int, list[str]] = {}
    notes: dict[int, list[str]] = {}
    for signature in capture.branch_signatures.values():
        for branch_id, outcome, is_rtainted in signature:
            rows.setdefault(int(branch_id), []).append("taken" if outcome else "not_taken")
            if is_rtainted:
                notes.setdefault(int(branch_id), []).append("rank-tainted branch")
    return tuple(
        RuntimeValueDistribution(
            point_name=f"branch_{branch_id}",
            source_kind="branch_signature",
            sample_count=len(values),
            observed_values=tuple(values),
            distribution=_distribution(values),
            notes=tuple(dict.fromkeys(notes.get(branch_id, ()))),
        )
        for branch_id, values in sorted(rows.items())
    )


def _summary_distributions(capture: DryRunProgramLogicCapture) -> tuple[RuntimeValueDistribution, ...]:
    grouped_values: dict[str, list[str]] = {}
    grouped_fields: dict[str, list[str]] = {}
    grouped_notes: dict[str, list[str]] = {}
    for summary in capture.semantic_summaries.values():
        for name, payload in summary.items():
            grouped_values.setdefault(str(name), []).append(_stable_repr(payload.get("value")))
            grouped_fields.setdefault(str(name), []).extend(str(item) for item in payload.get("dependencies", ()))
            if payload.get("r_tainted"):
                grouped_notes.setdefault(str(name), []).append("rank-tainted summary")
            branch_ids = tuple(int(item) for item in payload.get("branch_ids", ()))
            if branch_ids:
                grouped_notes.setdefault(str(name), []).append(
                    "branch_ids=" + ",".join(str(item) for item in branch_ids)
                )
    return tuple(
        RuntimeValueDistribution(
            point_name=name,
            source_kind="semantic_summary",
            sample_count=len(values),
            observed_values=tuple(values),
            distribution=_distribution(values),
            supporting_fields=tuple(dict.fromkeys(grouped_fields.get(name, ()))),
            notes=tuple(dict.fromkeys(grouped_notes.get(name, ()))),
        )
        for name, values in sorted(grouped_values.items())
    )


def _runtime_mutation(
    before: tuple[RuntimeValueDistribution, ...],
    after: tuple[RuntimeValueDistribution, ...],
) -> RuntimeValueMutationSummary | None:
    before_map = {item.point_name: item.distribution for item in before}
    after_map = {item.point_name: item.distribution for item in after}
    changed = tuple(
        name
        for name in sorted(set(before_map) & set(after_map))
        if before_map[name] != after_map[name]
    )
    added = tuple(sorted(set(after_map) - set(before_map)))
    removed = tuple(sorted(set(before_map) - set(after_map)))
    if not changed and not added and not removed:
        return None
    return RuntimeValueMutationSummary(
        changed_point_names=changed,
        added_point_names=added,
        removed_point_names=removed,
        notes=("runtime-value mutation is computed from observed distribution changes",),
    )


def build_anchor_runtime_value_state(
    *,
    workload_family: str,
    anchor_capture: DryRunProgramLogicCapture | None = None,
    candidate_capture: DryRunProgramLogicCapture | None = None,
) -> AnchorRuntimeValueState:
    notes: list[str] = []
    distributions: tuple[RuntimeValueDistribution, ...] = ()
    mutation = None
    if anchor_capture is None:
        notes.append("no dry-run capture is attached; runtime-value state has no observed distributions")
    else:
        distributions = tuple(
            sorted(_branch_distributions(anchor_capture) + _summary_distributions(anchor_capture), key=lambda item: item.point_name)
        )
        notes.append("runtime-value state is derived from dry-run branch signatures and semantic summaries")
        if candidate_capture is not None:
            candidate_distributions = tuple(
                sorted(_branch_distributions(candidate_capture) + _summary_distributions(candidate_capture), key=lambda item: item.point_name)
            )
            mutation = _runtime_mutation(distributions, candidate_distributions)
    return AnchorRuntimeValueState(
        workload_family=workload_family,
        contract=default_contract_for_workload_family(workload_family),
        distributions=distributions,
        mutation=mutation,
        notes=tuple(notes),
    )


def _event_category(event: Mapping[str, Any]) -> str:
    event_type = str(event.get("type", ""))
    mod = str(event.get("mod", ""))
    if event_type == "nccl_collective" or mod.startswith("libnccl"):
        return "nccl"
    if event_type == "blas_compute" or mod.startswith("libcublas"):
        return "blas"
    if event_type == "kernel_launch":
        return "kernel"
    if event_type == "mem_copy":
        return "memcpy"
    if event_type == "stream_op":
        return "stream"
    if event_type == "mem_alloc":
        return "alloc"
    if event_type == "context_op":
        return "context"
    return "other"


def _dominant_meaningful_category(counter: Counter[str]) -> str:
    meaningful = {key: value for key, value in counter.items() if key not in {"context", "other"}}
    if meaningful:
        return max(sorted(meaningful), key=lambda key: meaningful[key])
    if counter:
        return max(sorted(counter), key=lambda key: counter[key])
    return "empty"


def _region_kind_for_category(category: str) -> RegionKind:
    if category == "nccl":
        return RegionKind.COLLECTIVE
    if category in {"blas", "kernel"}:
        return RegionKind.EXPERT_COMPUTE
    if category in {"memcpy", "alloc"}:
        return RegionKind.MEMORY
    if category == "stream":
        return RegionKind.OVERLAP
    return RegionKind.OTHER


def _world_size_from_manifest(trace_dir: Path) -> int:
    manifest_path = trace_dir / "capture_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = payload.get("original_world_size")
        if value is not None:
            return int(value)
    rank_files = sorted(trace_dir.glob("rank_*.jsonl"))
    if rank_files:
        first_line = next(
            line for line in rank_files[0].read_text(encoding="utf-8").splitlines() if line.strip()
        )
        maybe_world = json.loads(first_line).get("world_size")
        if maybe_world is not None:
            return int(maybe_world)
    return len(rank_files)


def _note_field(notes: tuple[str, ...], key: str) -> str | None:
    prefix = f"{key}="
    for note in notes:
        if note.startswith(prefix):
            return note[len(prefix):]
    return None


def _region_with_first_class_metadata(region: AnchorRegion) -> AnchorRegion:
    lane_id = region.lane_id or _note_field(region.notes, "lane")
    semantic_role = region.semantic_role or _note_field(region.notes, "semantic_role")
    if lane_id == region.lane_id and semantic_role == region.semantic_role:
        return region
    return replace(region, lane_id=lane_id, semantic_role=semantic_role)


def _witness_with_first_class_metadata(witness: AnchorWitness) -> AnchorWitness:
    regions = tuple(_region_with_first_class_metadata(region) for region in witness.regions)
    if regions == witness.regions:
        return witness
    return replace(witness, regions=regions)


def build_anchor_witness_from_trace_dir(
    trace_dir: str | Path,
    *,
    anchor_candidate_id: str,
    workload_family: str,
    window_count: int = 24,
) -> AnchorWitness:
    trace_dir = Path(trace_dir)
    rank_files = sorted(trace_dir.glob("rank_*.jsonl"))
    if not rank_files:
        raise FileNotFoundError(f"no rank traces found in {trace_dir}")
    window_counts = [Counter() for _ in range(window_count)]
    window_share = [0.0 for _ in range(window_count)]
    rank_count = 0
    for rank_file in rank_files:
        events = [
            json.loads(line)
            for line in rank_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not events:
            continue
        ts0 = int(events[0]["ts"])
        ts1 = int(events[-1]["ts"])
        span = max(ts1 - ts0, 1)
        total_events = max(len(events), 1)
        per_counts = [Counter() for _ in range(window_count)]
        min_ts: list[int | None] = [None for _ in range(window_count)]
        max_ts: list[int | None] = [None for _ in range(window_count)]
        for event in events:
            position = (int(event["ts"]) - ts0) / span
            idx = min(window_count - 1, int(position * window_count))
            per_counts[idx][_event_category(event)] += 1
            event_ts = int(event["ts"])
            min_ts[idx] = event_ts if min_ts[idx] is None else min(min_ts[idx], event_ts)
            max_ts[idx] = event_ts if max_ts[idx] is None else max(max_ts[idx], event_ts)
        for idx in range(window_count):
            window_counts[idx].update(per_counts[idx])
            event_fraction = sum(per_counts[idx].values()) / total_events
            if min_ts[idx] is not None and max_ts[idx] is not None:
                duration_fraction = (max_ts[idx] - min_ts[idx]) / span
                window_share[idx] += duration_fraction if duration_fraction > 0.0 else event_fraction
            else:
                window_share[idx] += event_fraction
        rank_count += 1
    if rank_count == 0:
        raise ValueError(f"trace directory contains no non-empty rank traces: {trace_dir}")
    normalized_window_share = [value / rank_count for value in window_share]
    total_share = sum(normalized_window_share)
    normalized_window_share = [
        value / total_share if total_share else 0.0 for value in normalized_window_share
    ]
    merged_windows: list[tuple[int, int, RegionKind, float, Counter[str]]] = []
    for idx, counter in enumerate(window_counts):
        share = normalized_window_share[idx]
        if share <= 0.0:
            continue
        category = _dominant_meaningful_category(counter)
        region_kind = _region_kind_for_category(category)
        if merged_windows and merged_windows[-1][2] == region_kind:
            start, _, kind, prev_share, prev_counter = merged_windows[-1]
            prev_counter.update(counter)
            merged_windows[-1] = (start, idx, kind, prev_share + share, prev_counter)
        else:
            merged_windows.append((idx, idx, region_kind, share, Counter(counter)))
    max_share = max((share for _, _, _, share, _ in merged_windows), default=0.0)
    regions = []
    for order_index, (start_idx, end_idx, region_kind, share, counter) in enumerate(merged_windows):
        category = _dominant_meaningful_category(counter)
        regions.append(
            AnchorRegion(
                region_id=f"{region_kind.value}_{order_index:02d}",
                region_kind=region_kind,
                order_index=order_index,
                timing_share=share,
                start_window=start_idx,
                end_window=end_idx,
                lane_id=None,
                notes=(
                    f"windows={start_idx}-{end_idx}",
                    f"dominant={category}",
                ),
                value_sensitive=region_kind in {RegionKind.DISPATCH, RegionKind.EXPERT_COMPUTE, RegionKind.OVERLAP},
                hardware_sensitive=region_kind in {RegionKind.COLLECTIVE, RegionKind.OVERLAP, RegionKind.MEMORY},
                criticality_slack=max(max_share - share, 0.0),
                dependency_type=(
                    DependencyType.WIDE
                    if region_kind in {RegionKind.DISPATCH, RegionKind.COLLECTIVE, RegionKind.OVERLAP}
                    else DependencyType.NARROW
                ),
            )
        )
    artifacts = tuple(str(path) for path in ([trace_dir / "capture_manifest.json"] + rank_files))
    return AnchorWitness(
        anchor_candidate_id=anchor_candidate_id,
        workload_family=workload_family,
        world_size=_world_size_from_manifest(trace_dir),
        regions=tuple(regions),
        artifacts=artifacts,
        notes=("built from normalized multi-rank trace windows",),
    )


def _trace_rank_summaries(trace_dir: str | Path) -> tuple[TraceRankSummary, ...]:
    resolved = Path(trace_dir)
    rows: list[TraceRankSummary] = []
    for rank_file in sorted(resolved.glob("rank_*.jsonl")):
        rank = int(rank_file.stem.split("_")[1])
        events = [
            json.loads(line)
            for line in rank_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not events:
            rows.append(TraceRankSummary(rank=rank, event_count=0, start_ts=None, end_ts=None, span_us=None))
            continue
        mod_counter = Counter(str(event.get("mod", "")) for event in events)
        rows.append(
            TraceRankSummary(
                rank=rank,
                event_count=len(events),
                start_ts=int(events[0]["ts"]),
                end_ts=int(events[-1]["ts"]),
                span_us=int(events[-1]["ts"]) - int(events[0]["ts"]),
                dominant_mods=tuple(name for name, _ in mod_counter.most_common(3)),
            )
        )
    return tuple(rows)


def _region_signature(region: AnchorRegion) -> str:
    return (
        f"{region.region_kind.value}:"
        f"{region.start_window}-{region.end_window}:"
        f"{region.lane_id or 'any-lane'}:"
        f"{region.dependency_type.value}:"
        f"{region.timing_share:.6f}"
    )


def _window_from_manifest(trace_dir: Path) -> TraceStepWindow | None:
    manifest_path = trace_dir / "capture_manifest.json"
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = payload.get("step_window_source")
    step_count: int | None = None
    start_ts: int | None = None
    end_ts: int | None = None

    step_window_us = payload.get("step_window_us")
    if isinstance(step_window_us, list) and len(step_window_us) == 2:
        start_ts = int(step_window_us[0])
        end_ts = int(step_window_us[1])

    windows = payload.get("step_windows") or payload.get("fidelity_windows")
    if isinstance(windows, dict) and windows:
        starts: list[int] = []
        ends: list[int] = []
        sources: list[str] = []
        counts: list[int] = []
        for item in windows.values():
            if not isinstance(item, dict):
                continue
            if item.get("start_ts") is not None:
                starts.append(int(item["start_ts"]))
            if item.get("end_ts") is not None:
                ends.append(int(item["end_ts"]))
            if item.get("source") is not None:
                sources.append(str(item["source"]))
            if item.get("step_count") is not None:
                counts.append(int(item["step_count"]))
        if starts:
            start_ts = min(starts)
        if ends:
            end_ts = max(ends)
        if source is None and sources:
            source = sources[0] if len(set(sources)) == 1 else "mixed"
        if counts:
            step_count = max(counts)

    if source is None:
        source = payload.get("trace_window")
    if start_ts is None and end_ts is None and source is None:
        return None
    return TraceStepWindow(
        start_ts=start_ts,
        end_ts=end_ts,
        source=None if source is None else str(source),
        step_count=step_count,
        notes=("step window derived from capture_manifest.json",),
    )


def build_anchor_trace_state(
    *,
    trace_dir: str | Path,
    anchor_candidate_id: str,
    workload_family: str,
    witness: AnchorWitness | None = None,
    candidate_trace_dir: str | Path | None = None,
) -> AnchorTraceState:
    resolved_trace_dir = Path(trace_dir).resolve()
    resolved_witness = witness or build_anchor_witness_from_trace_dir(
        resolved_trace_dir,
        anchor_candidate_id=anchor_candidate_id,
        workload_family=workload_family,
    )
    resolved_witness = _witness_with_first_class_metadata(resolved_witness)
    rank_summaries = _trace_rank_summaries(resolved_trace_dir)
    region_signatures = tuple(_region_signature(region) for region in resolved_witness.regions)
    step_window = _window_from_manifest(resolved_trace_dir)
    lane_ids = tuple(
        dict.fromkeys(region.lane_id for region in resolved_witness.regions if region.lane_id)
    )
    mutation = None
    notes = ("trace state is backed by witness windows and raw rank-level event summaries",)
    if candidate_trace_dir is not None:
        resolved_candidate_dir = Path(candidate_trace_dir).resolve()
        candidate_witness = build_anchor_witness_from_trace_dir(
            resolved_candidate_dir,
            anchor_candidate_id=Path(resolved_candidate_dir).parent.name or "candidate",
            workload_family=workload_family,
        )
        candidate_witness = _witness_with_first_class_metadata(candidate_witness)
        candidate_rank_summaries = _trace_rank_summaries(resolved_candidate_dir)
        candidate_signatures = tuple(_region_signature(region) for region in candidate_witness.regions)
        changed_positions = [
            idx
            for idx in range(min(len(region_signatures), len(candidate_signatures)))
            if region_signatures[idx] != candidate_signatures[idx]
        ]
        anchor_total = sum(item.event_count for item in rank_summaries)
        candidate_total = sum(item.event_count for item in candidate_rank_summaries)
        per_rank = {
            item.rank: next(candidate.event_count for candidate in candidate_rank_summaries if candidate.rank == item.rank) - item.event_count
            for item in rank_summaries
            if any(candidate.rank == item.rank for candidate in candidate_rank_summaries)
        }
        mutation = TraceMutationSummary(
            candidate_trace_dir=str(resolved_candidate_dir),
            total_event_delta=candidate_total - anchor_total,
            per_rank_event_delta=per_rank,
            changed_region_positions=tuple(changed_positions),
            added_region_signatures=tuple(candidate_signatures[len(region_signatures):]),
            removed_region_signatures=tuple(region_signatures[len(candidate_signatures):]),
            notes=("trace mutation is computed from region-shape and raw event-count deltas",),
        )
    return AnchorTraceState(
        trace_dir=str(resolved_trace_dir),
        world_size=resolved_witness.world_size,
        witness=resolved_witness,
        rank_summaries=rank_summaries,
        region_signatures=region_signatures,
        step_window=step_window,
        step_window_source=None if step_window is None else step_window.source,
        lane_ids=lane_ids,
        mutation=mutation,
        notes=notes,
    )


def build_resilient_anchor_state(
    *,
    anchor_candidate_id: str,
    workload_family: str,
    trace_dir: str | Path,
    anchor_capture: DryRunProgramLogicCapture | None = None,
    anchor_program_logic: ProgramLogicCarrier | None = None,
    witness: AnchorWitness | None = None,
    anchor_code_paths: Iterable[str | Path] = (),
    code_mutation_pairs: Iterable[tuple[str | Path, str | Path]] = (),
    candidate_capture: DryRunProgramLogicCapture | None = None,
    candidate_trace_dir: str | Path | None = None,
) -> ResilientAnchorState:
    code_state = build_anchor_code_state(
        anchor_code_paths=anchor_code_paths,
        code_mutation_pairs=code_mutation_pairs,
    )
    semantic_state = build_anchor_semantic_state(
        anchor_capture=anchor_capture,
        anchor_program_logic=anchor_program_logic,
        candidate_capture=candidate_capture,
    )
    runtime_value_state = build_anchor_runtime_value_state(
        workload_family=workload_family,
        anchor_capture=anchor_capture,
        candidate_capture=candidate_capture,
    )
    trace_state = build_anchor_trace_state(
        trace_dir=trace_dir,
        anchor_candidate_id=anchor_candidate_id,
        workload_family=workload_family,
        witness=witness,
        candidate_trace_dir=candidate_trace_dir,
    )
    notes: list[str] = []
    if not code_state.baseline_files:
        notes.append("code slot is incomplete")
    if anchor_capture is None:
        notes.append("semantic and runtime slots are not dry-run grounded")
    if not runtime_value_state.distributions:
        notes.append("runtime-value slot lacks observed distributions")
    return ResilientAnchorState(
        anchor_candidate_id=anchor_candidate_id,
        workload_family=workload_family,
        world_size=trace_state.world_size,
        code=code_state,
        semantic=semantic_state,
        runtime_values=runtime_value_state,
        trace=trace_state,
        notes=tuple(notes),
    )


def resilient_anchor_state_summary(state: ResilientAnchorState) -> dict[str, Any]:
    return {
        "anchor_candidate_id": state.anchor_candidate_id,
        "workload_family": state.workload_family,
        "world_size": state.world_size,
        "notes": list(state.notes),
        "code": {
            "baseline_files": [
                {
                    "path": item.path,
                    "sha1": item.sha1,
                    "line_count": item.line_count,
                }
                for item in state.code.baseline_files
            ],
            "mutation_hunks": [
                {
                    "before_path": item.before_path,
                    "after_path": item.after_path,
                    "before_lines": [item.before_start_line, item.before_end_line],
                    "after_lines": [item.after_start_line, item.after_end_line],
                    "summary": item.summary,
                }
                for item in state.code.mutation_hunks
            ],
            "notes": list(state.code.notes),
        },
        "semantic": {
            "source": state.semantic.source,
            "logic_scope_id": None if state.semantic.logic_scope is None else state.semantic.logic_scope.scope_id,
            "program_logic_points": [point.name for point in state.semantic.program_logic_carrier.points],
            "operator_stub_count": len(state.semantic.operator_stubs),
            "boundary_capsule_count": len(state.semantic.boundary_capsules),
            "control_region_count": state.semantic.control_region_count,
            "logic_slice_count": state.semantic.logic_slice_count,
            "logic_slice_edge_count": state.semantic.logic_slice_edge_count,
            "mutation": None if state.semantic.mutation is None else {
                "changed_logic_point_names": list(state.semantic.mutation.changed_logic_point_names),
                "added_stub_ids": list(state.semantic.mutation.added_stub_ids),
                "removed_stub_ids": list(state.semantic.mutation.removed_stub_ids),
                "changed_capsule_ids": list(state.semantic.mutation.changed_capsule_ids),
                "incompatible_capsule_ids": list(state.semantic.mutation.incompatible_capsule_ids),
                "notes": list(state.semantic.mutation.notes),
            },
            "notes": list(state.semantic.notes),
        },
        "runtime_values": {
            "contract": None if state.runtime_values.contract is None else state.runtime_values.contract.workload_family,
            "distributions": [
                {
                    "point_name": item.point_name,
                    "source_kind": item.source_kind,
                    "sample_count": item.sample_count,
                    "distribution": item.distribution,
                    "supporting_fields": list(item.supporting_fields),
                    "notes": list(item.notes),
                }
                for item in state.runtime_values.distributions
            ],
            "mutation": None if state.runtime_values.mutation is None else {
                "changed_point_names": list(state.runtime_values.mutation.changed_point_names),
                "added_point_names": list(state.runtime_values.mutation.added_point_names),
                "removed_point_names": list(state.runtime_values.mutation.removed_point_names),
                "notes": list(state.runtime_values.mutation.notes),
            },
            "notes": list(state.runtime_values.notes),
        },
        "trace": {
            "trace_dir": state.trace.trace_dir,
            "critical_region_ids": list(state.trace.witness.critical_region_ids()),
            "region_signatures": list(state.trace.region_signatures),
            "rank_summaries": [
                {
                    "rank": item.rank,
                    "event_count": item.event_count,
                    "span_us": item.span_us,
                    "dominant_mods": list(item.dominant_mods),
                }
                for item in state.trace.rank_summaries
            ],
            "mutation": None if state.trace.mutation is None else {
                "candidate_trace_dir": state.trace.mutation.candidate_trace_dir,
                "total_event_delta": state.trace.mutation.total_event_delta,
                "per_rank_event_delta": state.trace.mutation.per_rank_event_delta,
                "changed_region_positions": list(state.trace.mutation.changed_region_positions),
                "added_region_signatures": list(state.trace.mutation.added_region_signatures),
                "removed_region_signatures": list(state.trace.mutation.removed_region_signatures),
                "notes": list(state.trace.mutation.notes),
            },
            "notes": list(state.trace.notes),
        },
    }
