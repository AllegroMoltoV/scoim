"""Private model contracts for phase-5 placement-specific accompaniment."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from llm_musical_composer.performance_pipeline import HARMONY_INTERVALS

from .generation_script_validation import check_generation_script_document
from .phase3_model_contracts import HarmonicPlan
from .profile_capabilities import solo_piano_3m_v2_capabilities
from .score_ir import ScoreHarmony, ScoreNote
from .score_operation_preflight import build_score_operation_plan
from .validation import IssueCode, ValidationIssue

DEGREES = ("root", "third", "fifth", "seventh")
REGISTER_ZONES = ("bass", "low", "middle", "high")
VOICES = ("upper", "lower")
ARTICULATIONS = ("normal", "staccato", "tenuto", "accent")
_DEGREE_INDEX = {"root": 0, "third": 1, "fifth": 2, "seventh": 3}


@dataclass(frozen=True, slots=True)
class AccompanimentComparisonSource:
    kind: str
    source_material_placement_id: str


@dataclass(frozen=True, slots=True)
class AccompanimentOperation:
    operation_id: str
    material_placement_id: str
    section_id: str
    score_unit_id: str
    material_id: str
    comparison_source: AccompanimentComparisonSource | None


def build_accompaniment_operations(
    document: Mapping[str, object],
) -> tuple[AccompanimentOperation, ...]:
    """Register ordinary accompaniment placements in stable performance order."""
    validation = check_generation_script_document(document)
    if not validation.valid:
        raise ValueError("the generation script must pass validation before phase 5")
    script = cast(Mapping[str, object], document["script"])
    placements = cast(Mapping[str, Mapping[str, object]], script["material_placements"])
    plan = build_score_operation_plan(document, solo_piano_3m_v2_capabilities())
    if plan.issues:
        issue = plan.issues[0]
        raise ValueError(f"{issue.code.value} at {issue.path}: {issue.message}")
    operations: list[AccompanimentOperation] = []
    for planned in plan.accompaniment_operations:
        placement = placements[planned.material_placement_id]
        section_id = cast(str, placement["section_id"])
        comparison_source = (
            AccompanimentComparisonSource(
                planned.comparison_sources[0].kind,
                planned.comparison_sources[0].source_material_placement_id,
            )
            if planned.comparison_sources
            else None
        )
        operations.append(
            AccompanimentOperation(
                operation_id=f"accompaniment-{planned.material_placement_id}",
                material_placement_id=planned.material_placement_id,
                section_id=section_id,
                score_unit_id=f"score-unit-{section_id}",
                material_id=cast(str, placement["material_id"]),
                comparison_source=comparison_source,
            )
        )
    return tuple(operations)


def accompaniment_response_schema() -> dict[str, object]:
    """Return the private transfer schema for one accompaniment placement."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["events"],
        "properties": {
            "events": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "at_units",
                        "preferred_duration_units",
                        "degree",
                        "preferred_register_zone",
                        "voice",
                        "articulations",
                    ],
                    "properties": {
                        "at_units": {"type": "integer", "minimum": 0},
                        "preferred_duration_units": {"type": "integer", "minimum": 1},
                        "degree": {"type": "string", "enum": list(DEGREES)},
                        "preferred_register_zone": {
                            "type": "string",
                            "enum": list(REGISTER_ZONES),
                        },
                        "voice": {"type": "string", "enum": list(VOICES)},
                        "articulations": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "enum": list(ARTICULATIONS)},
                        },
                    },
                },
            }
        },
    }


def check_accompaniment_response(
    response: Mapping[str, object],
    *,
    length_units: int,
    harmonies: tuple[ScoreHarmony, ...],
) -> tuple[ValidationIssue, ...]:
    """Check a schema-valid symbolic accompaniment before pitch placement."""
    events = cast(list[Mapping[str, object]], response["events"])
    issues: list[ValidationIssue] = []
    normalized: list[tuple[object, ...]] = []
    for index, event in enumerate(events):
        at_units = cast(int, event["at_units"])
        duration_units = cast(int, event["preferred_duration_units"])
        end_units = at_units + duration_units
        if end_units > length_units:
            issues.append(
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "An accompaniment event exceeds its score unit",
                    f"/events/{index}",
                )
            )
        if not any(
            harmony.at_units <= at_units and end_units <= harmony.at_units + harmony.duration_units
            for harmony in harmonies
        ):
            issues.append(
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "An accompaniment event must fit within one shared harmony",
                    f"/events/{index}",
                )
            )
        starting_harmony = next(
            (
                harmony
                for harmony in harmonies
                if harmony.at_units <= at_units < harmony.at_units + harmony.duration_units
            ),
            None,
        )
        degree_index = _DEGREE_INDEX[cast(str, event["degree"])]
        if starting_harmony is not None and degree_index >= len(
            HARMONY_INTERVALS[starting_harmony.quality]
        ):
            issues.append(
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "The requested degree is unavailable in the starting shared harmony",
                    f"/events/{index}/degree",
                )
            )
        articulations = tuple(cast(list[str], event["articulations"]))
        if len(articulations) != len(set(articulations)):
            issues.append(
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "Accompaniment event articulations must be unique",
                    f"/events/{index}/articulations",
                )
            )
        normalized.append(
            (
                at_units,
                duration_units,
                event["degree"],
                event["preferred_register_zone"],
                event["voice"],
                articulations,
            )
        )
    if len(normalized) != len(set(normalized)):
        issues.append(
            ValidationIssue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "Accompaniment events contain an exact duplicate",
                "/events",
            )
        )
    return tuple(issues)


def accompaniment_prompt(
    document: Mapping[str, object],
    plan: HarmonicPlan,
    operation: AccompanimentOperation,
    *,
    harmonies_by_score_unit: Mapping[str, tuple[ScoreHarmony, ...]],
    foreground_notes: tuple[ScoreNote, ...],
    accepted_responses: Mapping[str, Mapping[str, object]],
    accepted_notes: Mapping[str, tuple[ScoreNote, ...]],
) -> str:
    """Build one bounded accompaniment request from validated upstream values."""
    script = cast(Mapping[str, object], document["script"])
    sections = cast(Mapping[str, Mapping[str, object]], script["sections"])
    materials = cast(Mapping[str, Mapping[str, object]], script["materials"])
    harmonies = harmonies_by_score_unit[operation.score_unit_id]
    intent = next(
        item for item in plan.section_intents if item.score_unit_id == operation.score_unit_id
    )
    comparison: dict[str, object] | None = None
    if operation.comparison_source is not None:
        source_id = operation.comparison_source.source_material_placement_id
        if source_id not in accepted_responses or source_id not in accepted_notes:
            raise ValueError("an accompaniment comparison source has not been accepted")
        comparison = {
            "kind": operation.comparison_source.kind,
            "events": accepted_responses[source_id]["events"],
            "notes": [_serialized_note(note) for note in accepted_notes[source_id]],
        }
    context = {
        "whole_piece": {
            "tonal_center": plan.piece_plan.tonal_center,
            "mode": plan.piece_plan.mode,
            "overall_harmonic_story": plan.overall_harmonic_story,
        },
        "target": {
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
                    "available_degrees": list(DEGREES[: len(HARMONY_INTERVALS[harmony.quality])]),
                }
                for harmony in harmonies
            ],
            "foreground_notes": [_serialized_note(note) for note in foreground_notes],
            "comparison_context": comparison,
        },
        "allowed_values": {
            "degree": list(DEGREES),
            "preferred_register_zone": list(REGISTER_ZONES),
            "voice": list(VOICES),
            "articulations": list(ARTICULATIONS),
        },
    }
    return (
        "対象配置のピアノ伴奏を記号イベントとして作ってください。低域では根音と完全五度を"
        "優先してください。同じマテリアルの再利用では比較元を生かしつつ完全コピーにしないで"
        "ください。ID、絶対音高、JSON path、前景の変更は返さず、指定されたSchemaだけに従って"
        "ください。\n\n"
        f"入力: {json.dumps(context, ensure_ascii=False, sort_keys=True)}\n"
    )


def _serialized_note(note: ScoreNote) -> dict[str, object]:
    return {
        "at_units": note.at_units,
        "duration_units": note.duration_units,
        "pitch": note.pitch,
        "voice": note.voice,
    }
