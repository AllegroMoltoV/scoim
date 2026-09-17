"""Private typed model contracts for phase-4 placement-specific foregrounds."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from .generation_script_validation import check_generation_script_document
from .phase3_model_contracts import HarmonicPlan
from .profile_capabilities import solo_piano_3m_v2_capabilities
from .score_ir import ScoreHarmony, ScoreNote
from .score_operation_preflight import build_score_operation_plan
from .validation import IssueCode, ValidationIssue


@dataclass(frozen=True, slots=True)
class ForegroundComparisonSource:
    kind: str
    source_material_placement_id: str
    relation_id: str | None = None
    description: str | None = None
    preserve: tuple[str, ...] = ()
    change: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ForegroundOperation:
    operation_id: str
    material_placement_id: str
    section_id: str
    score_unit_id: str
    material_id: str
    transition_id: str | None
    comparison_sources: tuple[ForegroundComparisonSource, ...]


def foreground_prompt(
    document: Mapping[str, object],
    plan: HarmonicPlan,
    operation: ForegroundOperation,
    *,
    harmonies_by_score_unit: Mapping[str, tuple[ScoreHarmony, ...]],
    accepted_notes_by_material_placement: Mapping[str, tuple[ScoreNote, ...]],
    existing_unit_notes: tuple[ScoreNote, ...] = (),
) -> str:
    """Build one bounded foreground request from validated upstream values."""
    script = cast(Mapping[str, object], document["script"])
    sections = cast(Mapping[str, Mapping[str, object]], script["sections"])
    materials = cast(Mapping[str, Mapping[str, object]], script["materials"])
    comparisons: list[dict[str, object]] = []
    for source in operation.comparison_sources:
        notes = accepted_notes_by_material_placement.get(source.source_material_placement_id)
        if notes is None:
            raise ValueError("a foreground comparison source has not been accepted")
        serialized_notes = _serialized_notes(notes)
        if source.kind == "transition_source":
            terminal = max(note.at_units + note.duration_units for note in notes)
            serialized_notes = _serialized_notes(
                tuple(note for note in notes if note.at_units + note.duration_units == terminal)
            )
        elif source.kind == "transition_target":
            initial = min(note.at_units for note in notes)
            serialized_notes = _serialized_notes(
                tuple(note for note in notes if note.at_units == initial)
            )
        comparisons.append(
            {
                "kind": source.kind,
                "source_material_placement_id": source.source_material_placement_id,
                "relation_id": source.relation_id,
                "description": source.description,
                "preserve": list(source.preserve),
                "change": list(source.change),
                "notes": serialized_notes,
            }
        )
    harmonies = harmonies_by_score_unit.get(operation.score_unit_id)
    if harmonies is None:
        raise ValueError("the foreground target has no shared harmony")
    intent = next(
        (item for item in plan.section_intents if item.score_unit_id == operation.score_unit_id),
        None,
    )
    if intent is None:
        raise ValueError("the foreground target has no section intent")
    target: dict[str, object] = {
        "section_description": sections[operation.section_id]["description"],
        "material_description": materials[operation.material_id]["description"],
        "length_units": plan.length_units_by_score_unit[operation.score_unit_id],
        "harmonic_intent": intent.harmonic_intent,
        "shared_harmony": [
            {
                "at_units": harmony.at_units,
                "duration_units": harmony.duration_units,
                "root_pitch_class": harmony.root_pitch_class,
                "quality": harmony.quality,
            }
            for harmony in harmonies
        ],
        "transition": operation.transition_id is not None,
        "comparison_contexts": comparisons,
    }
    occupancy_instruction = ""
    if existing_unit_notes:
        target["occupied_notes"] = _serialized_notes(
            tuple(
                sorted(
                    existing_unit_notes,
                    key=lambda note: (
                        note.at_units,
                        note.duration_units,
                        note.pitch,
                        note.voice,
                    ),
                )
            )
        )
        occupancy_instruction = (
            "occupied_notesは変更禁止の採用済み前景です。候補と同じ音高かつ同じ声部で、"
            "両方の半開区間が重なる音符を作らないでください。境界が接するだけの場合と、"
            "異なる音高または異なる声部の同時発音は許されます。"
        )
    context = {
        "whole_piece": {
            "tonal_center": plan.piece_plan.tonal_center,
            "mode": plan.piece_plan.mode,
            "overall_harmonic_story": plan.overall_harmonic_story,
        },
        "target": target,
    }
    return (
        "対象配置の前景音符を作ってください。同じマテリアルの再利用または変奏では、"
        "比較元の特徴を保ちながら完全な同一コピーにしないでください。遷移では前後の"
        "境界を自然につないでください。"
        f"{occupancy_instruction}"
        "ID、JSON path、伴奏、演奏表現は返さず、指定された"
        "Schemaだけに従ってください。\n\n"
        f"入力: {json.dumps(context, ensure_ascii=False, sort_keys=True)}\n"
    )


def _serialized_notes(notes: tuple[ScoreNote, ...]) -> list[dict[str, object]]:
    return [
        {
            "at_units": note.at_units,
            "duration_units": note.duration_units,
            "pitch": note.pitch,
            "voice": note.voice,
        }
        for note in notes
    ]


def foreground_response_schema() -> dict[str, object]:
    """Return the private transfer schema for one placement foreground."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["notes"],
        "properties": {
            "notes": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["at_units", "duration_units", "pitch", "voice"],
                    "properties": {
                        "at_units": {"type": "integer", "minimum": 0},
                        "duration_units": {"type": "integer", "minimum": 1},
                        "pitch": {"type": "integer", "minimum": 21, "maximum": 108},
                        "voice": {"type": "string", "enum": ["upper", "lower"]},
                    },
                },
            }
        },
    }


def check_foreground_response(
    response: Mapping[str, object],
    *,
    length_units: int,
    comparison_notes: tuple[tuple[ScoreNote, ...], ...] = (),
    existing_unit_notes: tuple[ScoreNote, ...] = (),
) -> tuple[ValidationIssue, ...]:
    """Check one schema-valid foreground before assigning identifiers."""
    raw_notes = cast(list[Mapping[str, object]], response["notes"])
    values = tuple(
        (
            cast(int, note["at_units"]),
            cast(int, note["duration_units"]),
            cast(int, note["pitch"]),
            cast(str, note["voice"]),
        )
        for note in raw_notes
    )
    issues: list[ValidationIssue] = []
    for index, (at_units, duration_units, _pitch, _voice) in enumerate(values):
        if at_units + duration_units > length_units:
            issues.append(
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "A foreground note exceeds its score unit",
                    f"/notes/{index}",
                )
            )
    if len(values) != len(set(values)):
        issues.append(
            ValidationIssue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "Foreground notes contain an exact duplicate",
                "/notes",
            )
        )
    candidate_notes = tuple(
        ScoreNote("", at_units, duration_units, pitch, voice)
        for at_units, duration_units, pitch, voice in values
    )
    for right_index, right in enumerate(candidate_notes):
        for left in candidate_notes[:right_index]:
            if _same_pitch_voice_overlap(left, right):
                issues.append(
                    ValidationIssue(
                        IssueCode.MODEL_OUTPUT_INVALID,
                        (
                            "Foreground candidate overlaps another candidate: "
                            f"candidate={_interval(right)}, "
                            f"other_candidate={_interval(left)}, "
                            f"pitch={right.pitch}, voice={right.voice}"
                        ),
                        f"/notes/{right_index}",
                    )
                )
    for candidate_index, candidate in enumerate(candidate_notes):
        for occupied in existing_unit_notes:
            if _same_pitch_voice_overlap(candidate, occupied):
                issues.append(
                    ValidationIssue(
                        IssueCode.MODEL_OUTPUT_INVALID,
                        (
                            "Foreground candidate overlaps an occupied note: "
                            f"candidate={_interval(candidate)}, "
                            f"occupied={_interval(occupied)}, "
                            f"pitch={candidate.pitch}, voice={candidate.voice}"
                        ),
                        f"/notes/{candidate_index}",
                    )
                )
    normalized = _normalized_notes(candidate_notes)
    if any(normalized == _normalized_notes(source) for source in comparison_notes):
        issues.append(
            ValidationIssue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "A reused or varied foreground must not be an exact copy",
                "/notes",
            )
        )
    return tuple(issues)


def build_foreground_notes(
    material_placement_id: str, response: Mapping[str, object]
) -> tuple[ScoreNote, ...]:
    """Assign stable note identifiers to one validated positional response."""
    return tuple(
        ScoreNote(
            score_note_id=f"score-note-{material_placement_id}-{index:03d}",
            at_units=cast(int, note["at_units"]),
            duration_units=cast(int, note["duration_units"]),
            pitch=cast(int, note["pitch"]),
            voice=cast(str, note["voice"]),
        )
        for index, note in enumerate(cast(list[Mapping[str, object]], response["notes"]), start=1)
    )


def _normalized_notes(notes: tuple[ScoreNote, ...]) -> tuple[tuple[int, int, int, str], ...]:
    return tuple(
        sorted((note.at_units, note.duration_units, note.pitch, note.voice) for note in notes)
    )


def _same_pitch_voice_overlap(left: ScoreNote, right: ScoreNote) -> bool:
    return (
        left.pitch == right.pitch
        and left.voice == right.voice
        and left.at_units < right.at_units + right.duration_units
        and right.at_units < left.at_units + left.duration_units
    )


def _interval(note: ScoreNote) -> str:
    return f"[{note.at_units},{note.at_units + note.duration_units})"


def build_foreground_operations(
    document: Mapping[str, object],
) -> tuple[ForegroundOperation, ...]:
    """Register every foreground placement after its required source placements."""
    validation = check_generation_script_document(document)
    if not validation.valid:
        raise ValueError("the generation script must pass validation before phase 4")
    script = cast(Mapping[str, object], document["script"])
    placements = cast(Mapping[str, Mapping[str, object]], script["material_placements"])
    plan = build_score_operation_plan(document, solo_piano_3m_v2_capabilities())
    if plan.issues:
        issue = plan.issues[0]
        raise ValueError(f"{issue.code.value} at {issue.path}: {issue.message}")
    operations: list[ForegroundOperation] = []
    for planned in plan.foreground_operations:
        placement = placements[planned.material_placement_id]
        section_id = cast(str, placement["section_id"])
        operations.append(
            ForegroundOperation(
                operation_id=f"foreground-{planned.material_placement_id}",
                material_placement_id=planned.material_placement_id,
                section_id=section_id,
                score_unit_id=f"score-unit-{section_id}",
                material_id=cast(str, placement["material_id"]),
                transition_id=planned.transition_id,
                comparison_sources=tuple(
                    ForegroundComparisonSource(
                        source.kind,
                        source.source_material_placement_id,
                        source.relation_id,
                        source.description,
                        source.preserve,
                        source.change,
                    )
                    for source in planned.comparison_sources
                ),
            )
        )
    return tuple(operations)
