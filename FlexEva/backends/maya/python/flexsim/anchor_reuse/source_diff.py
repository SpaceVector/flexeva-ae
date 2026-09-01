"""Source-level helper signals for repeated-emission correspondence.

These helpers are intentionally advisory. They are used only to disambiguate
repeated source-level emissions when existing explicit signatures remain
ambiguous; they are not part of the core correctness proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dryrun_bridge import DryRunProgramLogicCapture, OperatorStub


@dataclass(frozen=True)
class SourceDiffHint:
    preferred_site_signature_by_after_stub_id: dict[str, str]
    ambiguous_after_stub_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def preferred_site_signature(self, stub_id: str) -> str | None:
        return self.preferred_site_signature_by_after_stub_id.get(str(stub_id))

    def is_ambiguous(self, stub_id: str) -> bool:
        return str(stub_id) in set(self.ambiguous_after_stub_ids)


def _normalize_source_line(line: str) -> str:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return ""
    return " ".join(stripped.split())


def _is_call_like_source_line(line: str) -> bool:
    normalized = _normalize_source_line(line)
    if not normalized:
        return False
    return normalized.endswith("()") and "=" not in normalized


def _is_assignment_like_source_line(line: str) -> bool:
    normalized = _normalize_source_line(line)
    if not normalized:
        return False
    if _is_call_like_source_line(line):
        return False
    return "=" in normalized and "==" not in normalized and "!=" not in normalized


def _is_control_header_line(line: str) -> bool:
    normalized = _normalize_source_line(line)
    if not normalized:
        return False
    return normalized.endswith(":")


def _context_signature_for_stub(
    code_path: str | Path,
    stub: OperatorStub,
    *,
    window: int = 1,
) -> str:
    source_path = Path(stub.source_path or code_path)
    lines = source_path.read_text(encoding="utf-8").splitlines()
    index = max(int(stub.lineno) - 1, 0)
    if index >= len(lines):
        index = max(len(lines) - 1, 0)
    previous_context: list[str] = []
    cursor = index - 1
    while cursor >= 0 and len(previous_context) < window:
        line = lines[cursor]
        normalized = _normalize_source_line(line)
        cursor -= 1
        if not normalized or not _is_assignment_like_source_line(line):
            continue
        previous_context.append(normalized)
    next_context: list[str] = []
    cursor = index + 1
    while cursor < len(lines) and len(next_context) < window:
        line = lines[cursor]
        normalized = _normalize_source_line(line)
        cursor += 1
        if not normalized or not _is_assignment_like_source_line(line):
            continue
        next_context.append(normalized)
    return "|".join(
        (
            f"prev={'/'.join(reversed(previous_context)) if previous_context else '<start>'}",
            f"next={'/'.join(next_context) if next_context else '<end>'}",
        )
    )


def _broad_context_signature_for_stub(
    code_path: str | Path,
    stub: OperatorStub,
    *,
    window: int = 1,
) -> str:
    source_path = Path(stub.source_path or code_path)
    lines = source_path.read_text(encoding="utf-8").splitlines()
    index = max(int(stub.lineno) - 1, 0)
    if index >= len(lines):
        index = max(len(lines) - 1, 0)
    previous_context: list[str] = []
    cursor = index - 1
    while cursor >= 0 and len(previous_context) < window:
        normalized = _normalize_source_line(lines[cursor])
        cursor -= 1
        if not normalized or _is_control_header_line(normalized):
            continue
        previous_context.append(normalized)
    next_context: list[str] = []
    cursor = index + 1
    while cursor < len(lines) and len(next_context) < window:
        normalized = _normalize_source_line(lines[cursor])
        cursor += 1
        if not normalized or _is_control_header_line(normalized):
            continue
        next_context.append(normalized)
    return "|".join(
        (
            f"prev={'/'.join(reversed(previous_context)) if previous_context else '<start>'}",
            f"next={'/'.join(next_context) if next_context else '<end>'}",
        )
    )


def build_source_diff_hint(
    before_capture: DryRunProgramLogicCapture | None,
    after_capture: DryRunProgramLogicCapture | None,
) -> SourceDiffHint | None:
    if before_capture is None or after_capture is None:
        return None
    before_by_signature: dict[str, list[tuple[OperatorStub, str]]] = {}
    after_by_signature: dict[str, list[tuple[OperatorStub, str]]] = {}
    for stub in before_capture.operator_stubs:
        before_by_signature.setdefault(stub.emission_signature, []).append(
            (stub, _context_signature_for_stub(before_capture.code_path, stub))
        )
    for stub in after_capture.operator_stubs:
        after_by_signature.setdefault(stub.emission_signature, []).append(
            (stub, _context_signature_for_stub(after_capture.code_path, stub))
        )

    preferred: dict[str, str] = {}
    ambiguous: list[str] = []
    notes: list[str] = []
    for emission_signature, after_items in after_by_signature.items():
        before_items = before_by_signature.get(emission_signature)
        if not before_items:
            continue
        before_context_to_sites: dict[str, list[str]] = {}
        for stub, context_signature in before_items:
            before_context_to_sites.setdefault(context_signature, []).append(stub.site_signature)
        after_context_to_ids: dict[str, list[str]] = {}
        for stub, context_signature in after_items:
            after_context_to_ids.setdefault(context_signature, []).append(stub.stub_id)
        for context_signature, after_stub_ids in after_context_to_ids.items():
            before_sites = before_context_to_sites.get(context_signature, [])
            if len(before_sites) != 1 or len(after_stub_ids) != 1:
                continue
            preferred[after_stub_ids[0]] = before_sites[0]
            notes.append(
                f"source_diff_context_match emission={emission_signature} after={after_stub_ids[0]} -> {before_sites[0]}"
            )
        preferred_after_ids = {
            stub_id
            for stub_id in (stub.stub_id for stub, _ in after_items)
            if stub_id in preferred
        }
        preferred_before_sites = {
            site_signature
            for site_signature in (stub.site_signature for stub, _ in before_items)
            if site_signature in preferred.values()
        }
        unmatched_after_stubs = [
            stub for stub, _ in after_items if stub.stub_id not in preferred_after_ids
        ]
        unmatched_before_stubs = [
            stub for stub, _ in before_items if stub.site_signature not in preferred_before_sites
        ]
        family_has_partial_unique_match = bool(preferred_after_ids) and (
            bool(unmatched_after_stubs) or bool(unmatched_before_stubs)
        )
        before_structure_signatures = {
            stub.structure_signature
            for stub, _ in before_items
            if stub.structure_signature
        }
        after_structure_signatures = {
            stub.structure_signature
            for stub, _ in after_items
            if stub.structure_signature
        }
        before_block_signatures = {
            stub.block_signature
            for stub, _ in before_items
            if stub.block_signature
        }
        after_block_signatures = {
            stub.block_signature
            for stub, _ in after_items
            if stub.block_signature
        }
        family_is_undifferentiated = (
            len(before_structure_signatures) <= 1
            and len(after_structure_signatures) <= 1
            and len(before_block_signatures) <= 1
            and len(after_block_signatures) <= 1
        )
        before_broad_contexts = sorted(
            _broad_context_signature_for_stub(before_capture.code_path, stub)
            for stub, _ in before_items
        )
        after_broad_contexts = sorted(
            _broad_context_signature_for_stub(after_capture.code_path, stub)
            for stub, _ in after_items
        )
        family_has_changed_broad_context = before_broad_contexts != after_broad_contexts
        if len(before_items) != len(after_items) or (
            family_has_partial_unique_match
            and unmatched_after_stubs
            and unmatched_before_stubs
        ) or (
            not preferred_after_ids
            and family_is_undifferentiated
            and family_has_changed_broad_context
        ):
            for stub in unmatched_after_stubs:
                ambiguous.append(stub.stub_id)
                notes.append(
                    f"source_diff_context_ambiguous emission={emission_signature} after={stub.stub_id}"
                )
    if not preferred and not ambiguous:
        return None
    return SourceDiffHint(
        preferred_site_signature_by_after_stub_id=preferred,
        ambiguous_after_stub_ids=tuple(dict.fromkeys(ambiguous)),
        notes=tuple(notes),
    )
