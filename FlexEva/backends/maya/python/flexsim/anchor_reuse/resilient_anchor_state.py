"""Explicit resilient-anchor-state aggregation for paper-facing review.

This layer does not replace the existing runtime/orchestration code. It
collects the currently scattered anchor-reuse artifacts into the four paper
slots that matter for review:

- anchor code
- anchor semantic
- anchor runtime values
- anchor trace

The current implementation is intentionally conservative:

- it wraps existing objects instead of redefining them,
- it keeps mutation tracking as explicit deltas around the anchor,
- and it records gaps when a slot is only partially grounded today.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .dryrun_bridge import (
    BoundaryContextCapsule,
    DryRunProgramLogicCapture,
    LogicScopeSpec,
    compare_boundary_contracts,
)
from .lineage import AnchorLineage, build_anchor_lineage
from .operator_evidence import compare_operator_stubs
from .program_logic import ProgramLogicCarrier, program_logic_deltas_from_carriers
from .schema import AnchorWitness
from .semantic_basis import WorkloadSemanticContract, default_contract_for_workload_family
from .source_diff import build_source_diff_hint
from .trace_builder import build_witness_from_micro_trace_dir


def _stable_repr(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, (list, tuple)):
        return json.dumps(value)
    return repr(value)


def _distribution(values: Iterable[str]) -> dict[str, float]:
    normalized = [str(item) for item in values]
    if not normalized:
        return {}
    counts = Counter(normalized)
    total = float(sum(counts.values()))
    return {key: counts[key] / total for key in sorted(counts)}


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _path_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


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
class AnchorCodeState:
    baseline_files: tuple[CodeFileSnapshot, ...]
    mutation_hunks: tuple[CodeMutationHunk, ...] = ()
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
    operator_stubs: tuple[Any, ...] = ()
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
    candidate_segment_count: int
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnchorTraceState:
    trace_dir: str
    world_size: int
    witness: AnchorWitness
    lineage: AnchorLineage
    rank_summaries: tuple[TraceRankSummary, ...]
    region_signatures: tuple[str, ...]
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


def _build_code_file_snapshot(path: str | Path) -> CodeFileSnapshot:
    resolved = Path(path).resolve()
    text = _path_text(resolved)
    return CodeFileSnapshot(
        path=str(resolved),
        sha1=_sha1_text(text),
        line_count=len(text.splitlines()),
    )


def _summarize_hunk(
    before_lines: list[str],
    after_lines: list[str],
) -> str:
    before_text = " ".join(line.strip() for line in before_lines if line.strip())
    after_text = " ".join(line.strip() for line in after_lines if line.strip())
    if before_text and after_text:
        return f"{before_text} -> {after_text}"
    if before_text:
        return f"remove {before_text}"
    if after_text:
        return f"add {after_text}"
    return "empty change"


def _build_code_mutation_hunks(
    before_path: str | Path,
    after_path: str | Path,
) -> tuple[CodeMutationHunk, ...]:
    resolved_before = Path(before_path).resolve()
    resolved_after = Path(after_path).resolve()
    before_lines = _path_text(resolved_before).splitlines()
    after_lines = _path_text(resolved_after).splitlines()
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
    baseline_files = tuple(_build_code_file_snapshot(path) for path in anchor_code_paths)
    mutation_hunks: list[CodeMutationHunk] = []
    for before_path, after_path in code_mutation_pairs:
        mutation_hunks.extend(_build_code_mutation_hunks(before_path, after_path))
    notes: list[str] = []
    if not baseline_files:
        notes.append("no baseline code files were attached to the anchor state")
    if not mutation_hunks:
        notes.append("no explicit code mutation hunks were attached")
    return AnchorCodeState(
        baseline_files=baseline_files,
        mutation_hunks=tuple(mutation_hunks),
        notes=tuple(notes),
    )


def _capsule_key(capsule: BoundaryContextCapsule) -> str:
    if capsule.site_signature:
        return f"site:{capsule.site_signature}"
    if capsule.stub_id:
        return f"stub:{capsule.stub_id}"
    return f"capsule:{capsule.capsule_id}"


def _build_semantic_mutation(
    before_capture: DryRunProgramLogicCapture | None,
    after_capture: DryRunProgramLogicCapture | None,
    before_carrier: ProgramLogicCarrier | None,
    after_carrier: ProgramLogicCarrier | None,
) -> SemanticMutationSummary | None:
    logic_deltas = program_logic_deltas_from_carriers(before_carrier, after_carrier)
    changed_logic_point_names = tuple(delta.point.name for delta in logic_deltas)
    notes: list[str] = []

    added_stub_ids: tuple[str, ...] = ()
    removed_stub_ids: tuple[str, ...] = ()
    changed_capsule_ids: tuple[str, ...] = ()
    incompatible_capsule_ids: tuple[str, ...] = ()

    if before_capture is not None and after_capture is not None:
        source_diff_hint = build_source_diff_hint(before_capture, after_capture)
        if source_diff_hint is not None:
            notes.extend(source_diff_hint.notes)
        stub_deltas = compare_operator_stubs(
            before_capture.operator_stubs,
            after_capture.operator_stubs,
            source_diff_hint=source_diff_hint,
        )
        added_stub_ids = tuple(
            delta.stub.stub_id for delta in stub_deltas if delta.change == "added"
        )
        removed_stub_ids = tuple(
            delta.stub.stub_id for delta in stub_deltas if delta.change == "removed"
        )

        before_capsules = {
            _capsule_key(capsule): capsule for capsule in before_capture.boundary_capsules
        }
        after_capsules = {
            _capsule_key(capsule): capsule for capsule in after_capture.boundary_capsules
        }
        changed_capsules: list[str] = []
        incompatible_capsules: list[str] = []
        for key in sorted(set(before_capsules) & set(after_capsules)):
            comparison = compare_boundary_contracts(before_capsules[key], after_capsules[key])
            if comparison.compatible:
                continue
            changed_capsules.append(after_capsules[key].capsule_id)
            if comparison.required_preservation_requirements:
                incompatible_capsules.append(after_capsules[key].capsule_id)
        changed_capsule_ids = tuple(changed_capsules)
        incompatible_capsule_ids = tuple(incompatible_capsules)

    if not (
        changed_logic_point_names
        or added_stub_ids
        or removed_stub_ids
        or changed_capsule_ids
        or incompatible_capsule_ids
    ):
        return None

    return SemanticMutationSummary(
        changed_logic_point_names=changed_logic_point_names,
        added_stub_ids=added_stub_ids,
        removed_stub_ids=removed_stub_ids,
        changed_capsule_ids=changed_capsule_ids,
        incompatible_capsule_ids=incompatible_capsule_ids,
        notes=tuple(notes),
    )


def build_anchor_semantic_state(
    *,
    anchor_capture: DryRunProgramLogicCapture | None = None,
    anchor_program_logic: ProgramLogicCarrier | None = None,
    candidate_capture: DryRunProgramLogicCapture | None = None,
) -> AnchorSemanticState:
    if anchor_capture is None and anchor_program_logic is None:
        raise ValueError("anchor_capture or anchor_program_logic is required")
    program_logic_carrier = (
        anchor_capture.program_logic if anchor_capture is not None else anchor_program_logic
    )
    assert program_logic_carrier is not None
    mutation = _build_semantic_mutation(
        anchor_capture,
        candidate_capture,
        program_logic_carrier,
        candidate_capture.program_logic if candidate_capture is not None else None,
    )
    notes: list[str] = []
    if anchor_capture is None:
        notes.append("semantic state is anchored only by a helper program-logic carrier, not a dry-run capture")
    if anchor_capture is not None:
        notes.append("semantic state is backed by dry-run capture plus CFG/control-region slices")
    return AnchorSemanticState(
        source="dryrun_capture" if anchor_capture is not None else "program_logic_carrier",
        logic_scope=anchor_capture.logic_scope if anchor_capture is not None else None,
        program_logic_carrier=program_logic_carrier,
        operator_stubs=anchor_capture.operator_stubs if anchor_capture is not None else (),
        boundary_capsules=anchor_capture.boundary_capsules if anchor_capture is not None else (),
        control_region_count=(
            len(anchor_capture.control_region_tree.regions)
            if anchor_capture is not None and anchor_capture.control_region_tree is not None
            else 0
        ),
        logic_slice_count=(
            len(anchor_capture.logic_state_store.slices)
            if anchor_capture is not None and anchor_capture.logic_state_store is not None
            else 0
        ),
        logic_slice_edge_count=(
            len(anchor_capture.logic_slice_graph.edges)
            if anchor_capture is not None and anchor_capture.logic_slice_graph is not None
            else 0
        ),
        mutation=mutation,
        notes=tuple(notes),
    )


def _branch_runtime_distributions(
    capture: DryRunProgramLogicCapture,
) -> tuple[RuntimeValueDistribution, ...]:
    branch_rows: dict[int, list[str]] = {}
    branch_notes: dict[int, list[str]] = {}
    for signature in capture.branch_signatures.values():
        for branch_id, outcome, is_rtainted in signature:
            branch_rows.setdefault(int(branch_id), []).append(
                "taken" if bool(outcome) else "not_taken"
            )
            if is_rtainted:
                branch_notes.setdefault(int(branch_id), []).append("rank-tainted branch")
    distributions: list[RuntimeValueDistribution] = []
    for branch_id in sorted(branch_rows):
        values = tuple(branch_rows[branch_id])
        distributions.append(
            RuntimeValueDistribution(
                point_name=f"branch_{branch_id}",
                source_kind="branch_signature",
                sample_count=len(values),
                observed_values=values,
                distribution=_distribution(values),
                notes=tuple(dict.fromkeys(branch_notes.get(branch_id, ()))),
            )
        )
    return tuple(distributions)


def _semantic_summary_runtime_distributions(
    capture: DryRunProgramLogicCapture,
) -> tuple[RuntimeValueDistribution, ...]:
    grouped_values: dict[str, list[str]] = {}
    grouped_fields: dict[str, list[str]] = {}
    grouped_notes: dict[str, list[str]] = {}
    for summary in capture.semantic_summaries.values():
        for name, payload in summary.items():
            grouped_values.setdefault(str(name), []).append(_stable_repr(payload.get("value")))
            grouped_fields.setdefault(str(name), []).extend(
                str(item) for item in payload.get("dependencies", ())
            )
            branch_ids = tuple(int(item) for item in payload.get("branch_ids", ()))
            if branch_ids:
                grouped_notes.setdefault(str(name), []).append(
                    "branch_ids=" + ",".join(str(item) for item in branch_ids)
                )
            if payload.get("r_tainted"):
                grouped_notes.setdefault(str(name), []).append("rank-tainted summary")
    distributions: list[RuntimeValueDistribution] = []
    for name in sorted(grouped_values):
        values = tuple(grouped_values[name])
        distributions.append(
            RuntimeValueDistribution(
                point_name=name,
                source_kind="semantic_summary",
                sample_count=len(values),
                observed_values=values,
                distribution=_distribution(values),
                supporting_fields=tuple(dict.fromkeys(grouped_fields.get(name, ()))),
                notes=tuple(dict.fromkeys(grouped_notes.get(name, ()))),
            )
        )
    return tuple(distributions)


def _runtime_mutation(
    before: tuple[RuntimeValueDistribution, ...],
    after: tuple[RuntimeValueDistribution, ...],
) -> RuntimeValueMutationSummary | None:
    before_map = {item.point_name: item.distribution for item in before}
    after_map = {item.point_name: item.distribution for item in after}
    changed = [
        name
        for name in sorted(set(before_map) & set(after_map))
        if before_map[name] != after_map[name]
    ]
    added = sorted(set(after_map) - set(before_map))
    removed = sorted(set(before_map) - set(after_map))
    if not changed and not added and not removed:
        return None
    return RuntimeValueMutationSummary(
        changed_point_names=tuple(changed),
        added_point_names=tuple(added),
        removed_point_names=tuple(removed),
        notes=("runtime-value state currently comes from dry-run observations rather than actual-run persisted samples",),
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
    if anchor_capture is not None:
        branch_points = _branch_runtime_distributions(anchor_capture)
        summary_points = _semantic_summary_runtime_distributions(anchor_capture)
        distributions = tuple(sorted(branch_points + summary_points, key=lambda item: item.point_name))
        notes.append("runtime-value state is derived from dry-run branch signatures and semantic summaries")
        if candidate_capture is not None:
            candidate_distributions = tuple(
                sorted(
                    _branch_runtime_distributions(candidate_capture)
                    + _semantic_summary_runtime_distributions(candidate_capture),
                    key=lambda item: item.point_name,
                )
            )
            mutation = _runtime_mutation(distributions, candidate_distributions)
    else:
        notes.append("no dry-run capture is attached; runtime-value state has no observed distributions yet")
    contract = default_contract_for_workload_family(workload_family)
    if contract is None:
        notes.append(f"no workload semantic contract is registered for family {workload_family}")
    return AnchorRuntimeValueState(
        workload_family=workload_family,
        contract=contract,
        distributions=distributions,
        mutation=mutation,
        notes=tuple(notes),
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
            rows.append(
                TraceRankSummary(
                    rank=rank,
                    event_count=0,
                    start_ts=None,
                    end_ts=None,
                    span_us=None,
                )
            )
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


def _region_signature(region: Any) -> str:
    return (
        f"{region.region_kind.value}:"
        f"{region.start_window}-{region.end_window}:"
        f"{region.dependency_type.value}:"
        f"{region.timing_share:.6f}"
    )


def build_anchor_trace_state(
    *,
    trace_dir: str | Path,
    anchor_candidate_id: str,
    workload_family: str,
    witness: AnchorWitness | None = None,
    lineage: AnchorLineage | None = None,
    candidate_trace_dir: str | Path | None = None,
) -> AnchorTraceState:
    resolved_trace_dir = Path(trace_dir).resolve()
    resolved_witness = witness or build_witness_from_micro_trace_dir(
        resolved_trace_dir,
        anchor_candidate_id=anchor_candidate_id,
        workload_family=workload_family,
    )
    resolved_lineage = lineage or build_anchor_lineage(
        resolved_witness,
        anchor_trace_dir=resolved_trace_dir,
        max_events_per_rank=None,
        replay_max_events_per_rank=500,
    )
    rank_summaries = _trace_rank_summaries(resolved_trace_dir)
    region_signatures = tuple(_region_signature(region) for region in resolved_witness.regions)

    mutation = None
    notes = ["trace state is backed by the existing anchor witness and anchor lineage"]
    if candidate_trace_dir is not None:
        resolved_candidate_dir = Path(candidate_trace_dir).resolve()
        candidate_witness = build_witness_from_micro_trace_dir(
            resolved_candidate_dir,
            anchor_candidate_id=Path(resolved_candidate_dir).parent.name,
            workload_family=workload_family,
        )
        candidate_lineage = build_anchor_lineage(
            candidate_witness,
            anchor_trace_dir=resolved_candidate_dir,
            max_events_per_rank=None,
            replay_max_events_per_rank=500,
        )
        candidate_regions = tuple(_region_signature(region) for region in candidate_witness.regions)
        changed_positions = []
        for idx in range(min(len(region_signatures), len(candidate_regions))):
            if region_signatures[idx] != candidate_regions[idx]:
                changed_positions.append(idx)
        added_signatures = candidate_regions[len(region_signatures) :]
        removed_signatures = region_signatures[len(candidate_regions) :]
        anchor_event_total = sum(item.event_count for item in rank_summaries)
        candidate_rank_summaries = _trace_rank_summaries(resolved_candidate_dir)
        candidate_event_total = sum(item.event_count for item in candidate_rank_summaries)
        per_rank_event_delta = {
            item.rank: next(
                candidate.event_count
                for candidate in candidate_rank_summaries
                if candidate.rank == item.rank
            )
            - item.event_count
            for item in rank_summaries
            if any(candidate.rank == item.rank for candidate in candidate_rank_summaries)
        }
        mutation = TraceMutationSummary(
            candidate_trace_dir=str(resolved_candidate_dir),
            total_event_delta=candidate_event_total - anchor_event_total,
            per_rank_event_delta=per_rank_event_delta,
            changed_region_positions=tuple(changed_positions),
            added_region_signatures=tuple(added_signatures),
            removed_region_signatures=tuple(removed_signatures),
            candidate_segment_count=len(candidate_lineage.segment_bundle.segments),
            notes=("trace mutation compares the collated window-region shape and raw per-rank event counts",),
        )

    return AnchorTraceState(
        trace_dir=str(resolved_trace_dir),
        world_size=resolved_witness.world_size,
        witness=resolved_witness,
        lineage=resolved_lineage,
        rank_summaries=rank_summaries,
        region_signatures=region_signatures,
        mutation=mutation,
        notes=tuple(notes),
    )


def build_resilient_anchor_state(
    *,
    anchor_candidate_id: str,
    workload_family: str,
    trace_dir: str | Path,
    anchor_capture: DryRunProgramLogicCapture | None = None,
    anchor_program_logic: ProgramLogicCarrier | None = None,
    witness: AnchorWitness | None = None,
    lineage: AnchorLineage | None = None,
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
        lineage=lineage,
        candidate_trace_dir=candidate_trace_dir,
    )
    notes: list[str] = []
    if anchor_capture is None:
        notes.append("semantic/runtime slots are not fully dry-run grounded for this anchor")
    if not code_state.baseline_files:
        notes.append("code slot is incomplete")
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
            "logic_scope_id": None
            if state.semantic.logic_scope is None
            else state.semantic.logic_scope.scope_id,
            "program_logic_points": [point.name for point in state.semantic.program_logic_carrier.points],
            "operator_stub_count": len(state.semantic.operator_stubs),
            "boundary_capsule_count": len(state.semantic.boundary_capsules),
            "control_region_count": state.semantic.control_region_count,
            "logic_slice_count": state.semantic.logic_slice_count,
            "logic_slice_edge_count": state.semantic.logic_slice_edge_count,
            "mutation": None
            if state.semantic.mutation is None
            else {
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
            "contract": None
            if state.runtime_values.contract is None
            else state.runtime_values.contract.workload_family,
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
            "mutation": None
            if state.runtime_values.mutation is None
            else {
                "changed_point_names": list(state.runtime_values.mutation.changed_point_names),
                "added_point_names": list(state.runtime_values.mutation.added_point_names),
                "removed_point_names": list(state.runtime_values.mutation.removed_point_names),
                "notes": list(state.runtime_values.mutation.notes),
            },
            "notes": list(state.runtime_values.notes),
        },
        "trace": {
            "trace_dir": state.trace.trace_dir,
            "region_signatures": list(state.trace.region_signatures),
            "segment_count": len(state.trace.lineage.segment_bundle.segments),
            "rank_summaries": [
                {
                    "rank": item.rank,
                    "event_count": item.event_count,
                    "span_us": item.span_us,
                    "dominant_mods": list(item.dominant_mods),
                }
                for item in state.trace.rank_summaries
            ],
            "mutation": None
            if state.trace.mutation is None
            else {
                "candidate_trace_dir": state.trace.mutation.candidate_trace_dir,
                "total_event_delta": state.trace.mutation.total_event_delta,
                "per_rank_event_delta": state.trace.mutation.per_rank_event_delta,
                "changed_region_positions": list(state.trace.mutation.changed_region_positions),
                "added_region_signatures": list(state.trace.mutation.added_region_signatures),
                "removed_region_signatures": list(state.trace.mutation.removed_region_signatures),
                "candidate_segment_count": state.trace.mutation.candidate_segment_count,
                "notes": list(state.trace.mutation.notes),
            },
            "notes": list(state.trace.notes),
        },
    }
