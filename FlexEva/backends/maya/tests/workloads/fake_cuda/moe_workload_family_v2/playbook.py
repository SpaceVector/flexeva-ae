#!/usr/bin/env python3
"""Helpers for the 128-GPU routed-MoE branching optimization playbook."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


_ROOT = Path(__file__).resolve().parent
_PLAYBOOK = _ROOT / "playbook.json"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    goal_ids: tuple[str, ...]
    status: str
    source_family: str | None
    backed_by_candidate_id: str | None
    summary: str


@dataclass(frozen=True)
class CandidateCombination:
    combination_id: str
    base_anchor_id: str
    goal_id: str
    operator_ids: tuple[str, ...]
    status: str
    expected_round_delta: str


@dataclass(frozen=True)
class AnchorVersion:
    anchor_id: str
    parent_anchor_id: str | None
    applied_operator_ids: tuple[str, ...]
    dominant_goal: str
    summary: str


@dataclass(frozen=True)
class RoundBranch:
    round_id: str
    input_anchor_id: str
    goal_id: str
    candidate_combination_ids: tuple[str, ...]
    promotion_options: tuple[str, ...]


@dataclass(frozen=True)
class Playbook:
    playbook_id: str
    family_id: str
    operator_library: dict[str, OperatorSpec]
    candidate_combinations: dict[str, CandidateCombination]
    anchors: dict[str, AnchorVersion]
    round_branches: dict[str, RoundBranch]
    raw_payload: dict


def load_playbook(path: Path | None = None) -> Playbook:
    payload = json.loads((path or _PLAYBOOK).read_text(encoding="utf-8"))
    operators = {
        item["operator_id"]: OperatorSpec(
            operator_id=item["operator_id"],
            goal_ids=tuple(item["goal_ids"]),
            status=item["status"],
            source_family=item.get("source_family"),
            backed_by_candidate_id=item.get("backed_by_candidate_id"),
            summary=item["summary"],
        )
        for item in payload["operator_library"]
    }
    combinations = {
        item["combination_id"]: CandidateCombination(
            combination_id=item["combination_id"],
            base_anchor_id=item["base_anchor_id"],
            goal_id=item["goal_id"],
            operator_ids=tuple(item["operator_ids"]),
            status=item["status"],
            expected_round_delta=item["expected_round_delta"],
        )
        for item in payload["candidate_combinations"]
    }
    anchors = {
        item["anchor_id"]: AnchorVersion(
            anchor_id=item["anchor_id"],
            parent_anchor_id=item.get("parent_anchor_id"),
            applied_operator_ids=tuple(item["applied_operator_ids"]),
            dominant_goal=item["dominant_goal"],
            summary=item["summary"],
        )
        for item in payload["anchors"]
    }
    branches = {
        item["round_id"]: RoundBranch(
            round_id=item["round_id"],
            input_anchor_id=item["input_anchor_id"],
            goal_id=item["goal_id"],
            candidate_combination_ids=tuple(item["candidate_combination_ids"]),
            promotion_options=tuple(item["promotion_options"]),
        )
        for item in payload["round_branches"]
    }
    return Playbook(
        playbook_id=payload["playbook_id"],
        family_id=payload["family_id"],
        operator_library=operators,
        candidate_combinations=combinations,
        anchors=anchors,
        round_branches=branches,
        raw_payload=payload,
    )


def select_combination_batch(playbook: Playbook, round_id: str) -> list[CandidateCombination]:
    branch = playbook.round_branches[round_id]
    return [playbook.candidate_combinations[item_id] for item_id in branch.candidate_combination_ids]


def list_may_work_combinations(playbook: Playbook, goal_id: str) -> list[CandidateCombination]:
    combos = [
        combo
        for combo in playbook.candidate_combinations.values()
        if combo.goal_id == goal_id and combo.status == "may_work"
    ]
    combos.sort(key=lambda item: item.combination_id)
    return combos


__all__ = [
    "AnchorVersion",
    "CandidateCombination",
    "OperatorSpec",
    "Playbook",
    "RoundBranch",
    "list_may_work_combinations",
    "load_playbook",
    "select_combination_batch",
]
