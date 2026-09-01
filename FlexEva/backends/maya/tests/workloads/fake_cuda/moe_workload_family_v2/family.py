#!/usr/bin/env python3
"""Helpers for the routed-MoE workload family definition."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


_ROOT = Path(__file__).resolve().parent
_MANIFEST = _ROOT / "workload_family.json"


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    parent: str | None
    goal_ids: tuple[str, ...]
    change_surface: str
    entry: str
    semantic_diffs: tuple[str, ...]


@dataclass(frozen=True)
class RoundSpec:
    round_id: str
    goal_id: str
    parallel_budget: int
    candidate_order: tuple[str, ...]


@dataclass(frozen=True)
class LiveExampleSpec:
    loop_engine: str
    summary: str
    real_cluster_summary: str
    real_cluster_uses: tuple[str, ...]


@dataclass(frozen=True)
class WorkloadFamily:
    family_id: str
    workload_kind: str
    anchor_candidate_id: str
    live_example: LiveExampleSpec
    candidates: dict[str, CandidateSpec]
    round_specs: dict[str, RoundSpec]
    raw_manifest: dict


def load_workload_family(path: Path | None = None) -> WorkloadFamily:
    manifest_path = path or _MANIFEST
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = {
        item["candidate_id"]: CandidateSpec(
            candidate_id=item["candidate_id"],
            parent=item.get("parent"),
            goal_ids=tuple(item["goal_ids"]),
            change_surface=item["change_surface"],
            entry=item["entry"],
            semantic_diffs=tuple(item["semantic_diffs"]),
        )
        for item in payload["candidates"]
    }
    round_specs = {
        item["round_id"]: RoundSpec(
            round_id=item["round_id"],
            goal_id=item["goal_id"],
            parallel_budget=int(item["parallel_budget"]),
            candidate_order=tuple(item["candidate_order"]),
        )
        for item in payload["round_specs"]
    }
    anchor_candidate_id = payload["anchor"]["candidate_id"]
    live_example_payload = payload["live_example"]
    return WorkloadFamily(
        family_id=payload["family_id"],
        workload_kind=payload["workload_kind"],
        anchor_candidate_id=anchor_candidate_id,
        live_example=LiveExampleSpec(
            loop_engine=live_example_payload["loop_engine"],
            summary=live_example_payload["summary"],
            real_cluster_summary=live_example_payload["real_cluster_role"]["summary"],
            real_cluster_uses=tuple(live_example_payload["real_cluster_role"]["uses"]),
        ),
        candidates=candidates,
        round_specs=round_specs,
        raw_manifest=payload,
    )


def select_round_candidates(
    family: WorkloadFamily,
    *,
    goal_id: str,
    budget: int,
    include_anchor: bool = False,
) -> list[CandidateSpec]:
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for round_spec in family.round_specs.values():
        if round_spec.goal_id != goal_id:
            continue
        for candidate_id in round_spec.candidate_order:
            if candidate_id not in seen and candidate_id in family.candidates:
                ordered_ids.append(candidate_id)
                seen.add(candidate_id)

    for candidate in sorted(family.candidates.values(), key=lambda item: item.candidate_id):
        if goal_id in candidate.goal_ids and candidate.candidate_id not in seen:
            ordered_ids.append(candidate.candidate_id)
            seen.add(candidate.candidate_id)

    ordered = [family.candidates[candidate_id] for candidate_id in ordered_ids]

    selected = ordered[: max(0, budget)]
    if include_anchor:
        anchor = family.candidates[family.anchor_candidate_id]
        return [anchor, *selected]
    return selected


def select_round_spec_candidates(
    family: WorkloadFamily,
    round_id: str,
    *,
    include_anchor: bool = True,
) -> list[CandidateSpec]:
    round_spec = family.round_specs[round_id]
    selected: list[CandidateSpec] = []
    if include_anchor:
        selected.append(family.candidates[family.anchor_candidate_id])
    for candidate_id in round_spec.candidate_order[: round_spec.parallel_budget]:
        selected.append(family.candidates[candidate_id])
    return selected


def build_simulation_round_plan(
    family: WorkloadFamily,
    round_id: str,
    *,
    current_anchor_id: str | None = None,
) -> dict:
    round_spec = family.round_specs[round_id]
    anchor_candidate_id = current_anchor_id or family.anchor_candidate_id
    candidates = select_round_spec_candidates(family, round_id, include_anchor=False)
    return {
        "family_id": family.family_id,
        "loop_engine": family.live_example.loop_engine,
        "anchor_candidate_id": anchor_candidate_id,
        "round_id": round_spec.round_id,
        "goal_id": round_spec.goal_id,
        "parallel_budget": round_spec.parallel_budget,
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "real_cluster_role": {
            "summary": family.live_example.real_cluster_summary,
            "uses": list(family.live_example.real_cluster_uses),
        },
    }


__all__ = [
    "CandidateSpec",
    "LiveExampleSpec",
    "RoundSpec",
    "WorkloadFamily",
    "build_simulation_round_plan",
    "load_workload_family",
    "select_round_candidates",
    "select_round_spec_candidates",
]
