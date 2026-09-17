"""Deterministic conversion from typed model choices to the existing piano IR."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import jsonpointer
from jsonschema import Draft202012Validator

from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    PiecePlan,
    ScoreDirection,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    ordered_leaf_schedule,
    performance_time_map,
    validate_pipeline,
    validate_score_spec,
)

from .projection import PlanChoice, ProjectionTarget, project_solo_piano_3m
from .validation import IssueCode, ValidationIssue

_DIVISIONS = 12
_UNITS_PER_WEIGHT = 6
_TARGET_DURATION_MS = 180_000


def _object_schema(properties: Mapping[str, object]) -> dict[str, object]:
    ordered = {key: properties[key] for key in sorted(properties)}
    return {
        "type": "object",
        "properties": ordered,
        "required": list(ordered),
        "additionalProperties": False,
    }


def _nullable_string(values: tuple[str, ...]) -> dict[str, object]:
    return {
        "anyOf": [
            {"type": "string", "enum": list(values)},
            {"type": "null"},
        ]
    }


def _section_shape(document: Mapping[str, object]) -> tuple[tuple[str, ...], dict[str, int]]:
    script = cast(Mapping[str, object], document["script"])
    sections = cast(Mapping[str, Mapping[str, object]], script["sections"])
    children: dict[str, list[str]] = {section_id: [] for section_id in sections}
    for section_id, section in sections.items():
        parent_id = section["parent_section_id"]
        if isinstance(parent_id, str):
            children[parent_id].append(section_id)
    for child_ids in children.values():
        child_ids.sort(key=lambda section_id: cast(int, sections[section_id]["order"]))
    ordered: list[str] = []
    depths: dict[str, int] = {}

    def visit(section_id: str, depth: int) -> None:
        ordered.append(section_id)
        depths[section_id] = depth
        for child_id in children[section_id]:
            visit(child_id, depth + 1)

    visit(cast(str, script["root_section_id"]), 0)
    return tuple(ordered), depths


def build_typed_response_schema(document: Mapping[str, object]) -> dict[str, object]:
    """Build the strict model-response schema for one approved script."""

    script = cast(Mapping[str, object], document["script"])
    sections = cast(Mapping[str, Mapping[str, object]], script["sections"])
    material_ids = sorted(cast(Mapping[str, object], script["materials"]))
    section_order, depths = _section_shape(document)

    harmonic_focus = _object_schema(
        {
            section_id: {
                "anyOf": [
                    {"type": "integer", "minimum": 0, "maximum": 11},
                    {"type": "null"},
                ]
            }
            for section_id in section_order
        }
    )
    contrast_properties: dict[str, object] = {}
    for index, section_id in enumerate(section_order):
        if sections[section_id]["role"] != "contrast":
            continue
        candidates = [
            prior_id for prior_id in section_order[:index] if depths[prior_id] == depths[section_id]
        ]
        choices: list[dict[str, object]] = []
        if candidates:
            choices.append(
                {
                    "type": "integer",
                    "enum": list(range(len(candidates))),
                }
            )
        choices.append({"type": "null"})
        contrast_properties[section_id] = {"anyOf": choices}

    note_schema = _object_schema(
        {
            "at_units": {"type": "integer", "minimum": 0},
            "duration_units": {"type": "integer", "minimum": 1},
            "pitch": {"type": "integer", "minimum": 21, "maximum": 108},
            "voice": {"type": "string", "enum": ["upper", "lower"]},
            "tie": _nullable_string(("start", "continue", "stop")),
            "articulations": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["normal", "staccato", "tenuto", "accent"],
                },
            },
        }
    )
    harmony_schema = _object_schema(
        {
            "at_units": {"type": "integer", "minimum": 0},
            "duration_units": {"type": "integer", "minimum": 1},
            "root_pitch_class": {"type": "integer", "minimum": 0, "maximum": 11},
            "quality": {
                "type": "string",
                "enum": ["major", "minor", "diminished", "major-seventh"],
            },
        }
    )
    direction_schema = _object_schema(
        {
            "at_units": {"type": "integer", "minimum": 0},
            "kind": {"type": "string", "enum": ["dynamic", "breath"]},
            "value": {
                "type": "string",
                "enum": ["pp", "p", "mp", "mf", "f", "ff", "light", "full"],
            },
        }
    )
    material_schema = _object_schema(
        {
            "foreground_voice": _nullable_string(("upper", "lower")),
            "notes": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/$defs/note"},
            },
            "harmonies": {"type": "array", "items": {"$ref": "#/$defs/harmony"}},
            "directions": {
                "type": "array",
                "items": {"$ref": "#/$defs/direction"},
            },
        }
    )
    performance_schema = _object_schema(
        {
            "timing_profile": _nullable_string(("neutral", "savor", "flow", "build", "release")),
            "timing_amount": _nullable_string(("subtle", "moderate")),
            "dynamics_profile": _nullable_string(("steady", "shape", "build", "release")),
            "articulation_profile": _nullable_string(("score", "legato", "light")),
            "coordination_profile": _nullable_string(("score", "rolled", "aligned")),
            "pedal_profile": _nullable_string(("none", "phrase_legato", "harmony_legato", "clear")),
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://scoim.dev/schemas/typed-realization-response-2.schema.json",
        "title": "SCoIM typed realization response",
        "$defs": {
            "direction": direction_schema,
            "harmony": harmony_schema,
            "material": material_schema,
            "note": note_schema,
            "performance": performance_schema,
        },
        **_object_schema(
            {
                "plan_choice": _object_schema(
                    {
                        "tonal_center": {"type": "integer", "minimum": 0, "maximum": 11},
                        "mode": {"type": "string", "enum": ["major", "minor"]},
                        "harmonic_focus_by_section": harmonic_focus,
                        "contrasts_with_by_section": _object_schema(contrast_properties),
                    }
                ),
                "score_materials": _object_schema(
                    {material_id: {"$ref": "#/$defs/material"} for material_id in material_ids}
                ),
                "node_performances": _object_schema(
                    {section_id: {"$ref": "#/$defs/performance"} for section_id in section_order}
                ),
            }
        ),
    }


def validate_typed_model_response(
    document: Mapping[str, object],
    response: object,
    *,
    path_prefix: str = "/runner/response",
) -> tuple[ValidationIssue, ...]:
    """Validate one model response and retain a concrete response pointer."""

    schema = build_typed_response_schema(document)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(response),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return ()
    error = errors[0]
    parts = [str(part) for part in error.absolute_path]
    if error.validator == "additionalProperties" and isinstance(error.instance, Mapping):
        properties = cast(Mapping[str, object], error.schema.get("properties", {}))
        unexpected = sorted(set(error.instance) - set(properties))
        if unexpected:
            parts.append(str(unexpected[0]))
    elif error.validator == "required" and isinstance(error.instance, Mapping):
        required = cast(list[str], error.validator_value)
        missing = sorted(set(required) - set(error.instance))
        if missing:
            parts.append(missing[0])
    pointer = "".join(f"/{jsonpointer.escape(part)}" for part in parts)
    return (
        ValidationIssue(
            IssueCode.MODEL_OUTPUT_INVALID,
            error.message,
            f"{path_prefix}{pointer}",
        ),
    )


def _contrast_candidates(document: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    script = cast(Mapping[str, object], document["script"])
    sections = cast(Mapping[str, Mapping[str, object]], script["sections"])
    section_order, depths = _section_shape(document)
    return {
        section_id: tuple(
            prior_id for prior_id in section_order[:index] if depths[prior_id] == depths[section_id]
        )
        for index, section_id in enumerate(section_order)
        if sections[section_id]["role"] == "contrast"
    }


def normalize_typed_model_response(
    document: Mapping[str, object], response: Mapping[str, object]
) -> dict[str, object]:
    """Normalize a schema-validated model response into frozen Schema 2."""

    plan_choice = cast(Mapping[str, object], response["plan_choice"])
    section_order, _ = _section_shape(document)
    raw_focus = cast(Mapping[str, int | None], plan_choice["harmonic_focus_by_section"])
    raw_contrasts = cast(Mapping[str, int | None], plan_choice["contrasts_with_by_section"])
    score_materials = cast(Mapping[str, object], response["score_materials"])
    node_performances = cast(Mapping[str, object], response["node_performances"])
    return {
        "schema_version": 2,
        "profile": "solo_piano_3m_v1",
        "plan_choice": {
            "tonal_center": plan_choice["tonal_center"],
            "mode": plan_choice["mode"],
            "harmonic_focus_by_section": {
                section_id: raw_focus[section_id] for section_id in section_order
            },
            "contrasts_with_by_section": {
                section_id: raw_contrasts[section_id]
                for section_id in _contrast_candidates(document)
            },
        },
        "score_materials": {
            material_id: copy.deepcopy(score_materials[material_id])
            for material_id in sorted(score_materials)
        },
        "node_performances": {
            section_id: copy.deepcopy(node_performances[section_id]) for section_id in section_order
        },
    }


class TypedRealizationError(ValueError):
    """A typed response cannot be converted without changing its meaning."""

    def __init__(self, issue: ValidationIssue) -> None:
        super().__init__(issue.message)
        self.issue = issue


@dataclass(frozen=True, slots=True)
class TypedIR:
    """Existing piano IR reconstructed from one normalized typed response."""

    plan: PiecePlan
    score: ScoreSpec
    performance: PerformanceSpec
    targets: tuple[ProjectionTarget, ...]


def ensure_all_script_materials_are_placed(plan: PiecePlan, material_ids: set[str]) -> None:
    """Reject script materials whose score length cannot be derived from a leaf."""

    placed = {node.score_material_id for node in plan.nodes if node.score_material_id is not None}
    unplaced = sorted(material_ids - placed)
    if unplaced:
        material_id = unplaced[0]
        raise TypedRealizationError(
            ValidationIssue(
                IssueCode.UNREPRESENTABLE,
                "A score material has no placed leaf from which to derive its length",
                f"/script/materials/{material_id}",
            )
        )


def ensure_plan_resource_limit(plan: PiecePlan) -> None:
    """Reject plans whose expanded unit grid exceeds the fixed resource limit."""

    total_units = sum(
        node.duration_weight * _UNITS_PER_WEIGHT
        for node in plan.nodes
        if node.duration_weight is not None
    )
    if total_units > _TARGET_DURATION_MS:
        raise TypedRealizationError(
            ValidationIssue(
                IssueCode.UNREPRESENTABLE,
                "The projected score grid exceeds the solo-piano resource limit",
                "/script/sections",
            )
        )


def material_lengths_for_plan(plan: PiecePlan) -> dict[str, int]:
    """Return deterministic score lengths for materials placed in leaves."""

    return {
        node.score_material_id: node.duration_weight * _UNITS_PER_WEIGHT
        for node in plan.nodes
        if node.score_material_id is not None and node.duration_weight is not None
    }


def preflight_typed_realization(document: Mapping[str, object]) -> PiecePlan:
    """Check script-only transfer constraints before spending a model call."""

    projection = project_solo_piano_3m(
        document,
        plan_choice=PlanChoice(
            tonal_center=0,
            mode="major",
            harmonic_focus_by_section={},
            contrasts_with_by_section={},
        ),
    )
    if not projection.projected or projection.piece_plan is None:
        raise TypedRealizationError(projection.issues[0])
    plan = projection.piece_plan
    script = cast(Mapping[str, object], document["script"])
    ensure_all_script_materials_are_placed(
        plan,
        set(cast(Mapping[str, object], script["materials"])),
    )
    ensure_plan_resource_limit(plan)
    return plan


def ensure_distinct_event_boundaries(
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
) -> None:
    """Reject distinct score boundaries that collapse to one integer millisecond."""

    time_map = performance_time_map(plan, score, performance)
    leaves, _ = ordered_leaf_schedule(plan, score)
    materials = {material.material_id: material for material in score.materials}
    boundaries: list[tuple[int, str]] = []
    for leaf, occurrence_start, _ in leaves:
        assert leaf.score_material_id is not None
        material = materials[leaf.score_material_id]
        prefix = f"/runner/response/score_materials/{material.material_id}"
        for index, note in enumerate(material.notes):
            path = f"{prefix}/notes/{index}"
            boundaries.extend(
                (
                    (occurrence_start + note.at_units, path),
                    (occurrence_start + note.at_units + note.duration_units, path),
                )
            )
        for index, harmony in enumerate(material.harmonies):
            path = f"{prefix}/harmonies/{index}"
            boundaries.extend(
                (
                    (occurrence_start + harmony.at_units, path),
                    (occurrence_start + harmony.at_units + harmony.duration_units, path),
                )
            )
        for index, direction in enumerate(material.directions):
            boundaries.append(
                (
                    occurrence_start + direction.at_units,
                    f"{prefix}/directions/{index}",
                )
            )

    unit_by_ms: dict[int, int] = {}
    for unit, path in sorted(boundaries):
        at_ms = time_map[unit]
        previous_unit = unit_by_ms.get(at_ms)
        if previous_unit is not None and previous_unit != unit:
            raise TypedRealizationError(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    "Distinct score event boundaries collapse to the same integer millisecond",
                    path,
                )
            )
        unit_by_ms[at_ms] = unit


def _material_derivations(document: Mapping[str, object]) -> dict[str, str]:
    script = cast(Mapping[str, object], document["script"])
    variations = cast(Mapping[str, Mapping[str, object]], script["variations"])
    result: dict[str, str] = {}
    for variation in variations.values():
        source = cast(Mapping[str, object], variation["source"])
        target = cast(Mapping[str, object], variation["target"])
        if source["type"] == "material":
            result[cast(str, target["id"])] = cast(str, source["id"])
    return result


def build_typed_ir(
    document: Mapping[str, object], frozen_response: Mapping[str, object]
) -> TypedIR:
    """Convert frozen Schema 2 values into all existing piano IR stages."""

    raw_choice = cast(Mapping[str, object], frozen_response["plan_choice"])
    raw_focus = cast(Mapping[str, int | None], raw_choice["harmonic_focus_by_section"])
    raw_contrasts = cast(Mapping[str, int | None], raw_choice["contrasts_with_by_section"])
    contrast_candidates = _contrast_candidates(document)
    projection = project_solo_piano_3m(
        document,
        plan_choice=PlanChoice(
            tonal_center=cast(int, raw_choice["tonal_center"]),
            mode=cast(str, raw_choice["mode"]),
            harmonic_focus_by_section=cast(
                Mapping[str, int],
                {section_id: value for section_id, value in raw_focus.items() if value is not None},
            ),
            contrasts_with_by_section=cast(
                Mapping[str, str],
                {
                    section_id: contrast_candidates[section_id][choice]
                    for section_id, choice in raw_contrasts.items()
                    if choice is not None
                },
            ),
        ),
    )
    if not projection.projected or projection.piece_plan is None:
        raise TypedRealizationError(projection.issues[0])
    plan = projection.piece_plan
    script = cast(Mapping[str, object], document["script"])
    material_ids = set(cast(Mapping[str, object], script["materials"]))
    ensure_all_script_materials_are_placed(plan, material_ids)
    ensure_plan_resource_limit(plan)
    score = build_score_spec_from_typed_response(
        plan,
        cast(Mapping[str, object], frozen_response["score_materials"]),
        material_derivations=_material_derivations(document),
    )
    performance = build_performance_spec_from_typed_response(
        plan,
        cast(Mapping[str, object], frozen_response["node_performances"]),
    )
    validate_pipeline(plan, score, performance)
    ensure_distinct_event_boundaries(plan, score, performance)
    return TypedIR(
        plan=plan,
        score=score,
        performance=performance,
        targets=projection.targets,
    )


def _material_order(
    material_ids: set[str], material_derivations: Mapping[str, str]
) -> tuple[str, ...]:
    """Return a stable source-before-derived order."""

    remaining = set(material_ids)
    ordered: list[str] = []
    while remaining:
        ready = sorted(
            material_id
            for material_id in remaining
            if material_derivations.get(material_id) not in remaining
        )
        if not ready:
            raise ValueError("score material derivations contain a cycle")
        ordered.extend(ready)
        remaining.difference_update(ready)
    return tuple(ordered)


def build_score_spec_from_typed_response(
    plan: PiecePlan,
    score_materials: Mapping[str, object],
    *,
    material_derivations: Mapping[str, str],
) -> ScoreSpec:
    """Build a ScoreSpec while deriving all identifiers and material lengths."""

    length_by_material = material_lengths_for_plan(plan)

    missing = sorted(set(length_by_material) - set(score_materials))
    unexpected = sorted(set(score_materials) - set(length_by_material))
    if missing or unexpected:
        material_id = (missing or unexpected)[0]
        raise TypedRealizationError(
            ValidationIssue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "The typed score material keys do not match the projected material keys",
                f"/runner/response/score_materials/{material_id}",
            )
        )

    materials: list[ScoreMaterial] = []
    for material_id in _material_order(set(score_materials), material_derivations):
        raw = cast(Mapping[str, object], score_materials[material_id])
        length_units = length_by_material[material_id]
        raw_notes = cast(list[Mapping[str, object]], raw["notes"])
        raw_harmonies = cast(list[Mapping[str, object]], raw["harmonies"])
        foreground_voice = cast(str | None, raw["foreground_voice"])
        if bool(raw_harmonies) != (foreground_voice is not None):
            raise TypedRealizationError(
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "Typed harmonies and foreground voice must be declared together",
                    f"/runner/response/score_materials/{material_id}/foreground_voice",
                )
            )
        for index, item in enumerate(raw_notes):
            at_units = cast(int, item["at_units"])
            duration_units = cast(int, item["duration_units"])
            if at_units < 0 or duration_units <= 0 or at_units + duration_units > length_units:
                raise TypedRealizationError(
                    ValidationIssue(
                        IssueCode.MODEL_OUTPUT_INVALID,
                        "A typed note is outside its fixed material length",
                        f"/runner/response/score_materials/{material_id}/notes/{index}",
                    )
                )
            articulations = cast(list[str], item["articulations"])
            seen_articulations: set[str] = set()
            for articulation_index, articulation in enumerate(articulations):
                if articulation in seen_articulations:
                    raise TypedRealizationError(
                        ValidationIssue(
                            IssueCode.MODEL_OUTPUT_INVALID,
                            "A typed note cannot repeat an articulation",
                            f"/runner/response/score_materials/{material_id}/notes/"
                            f"{index}/articulations/{articulation_index}",
                        )
                    )
                seen_articulations.add(articulation)
        notes_by_pitch_and_voice: dict[tuple[int, str], list[tuple[int, Mapping[str, object]]]] = {}
        for index, item in enumerate(raw_notes):
            key = (cast(int, item["pitch"]), cast(str, item["voice"]))
            notes_by_pitch_and_voice.setdefault(key, []).append((index, item))
        for same_pitch in notes_by_pitch_and_voice.values():
            ordered = sorted(same_pitch, key=lambda pair: (pair[1]["at_units"], pair[0]))
            previous_end = -1
            for index, item in ordered:
                at_units = cast(int, item["at_units"])
                if at_units < previous_end:
                    raise TypedRealizationError(
                        ValidationIssue(
                            IssueCode.MODEL_OUTPUT_INVALID,
                            "The same typed pitch cannot overlap within one material",
                            f"/runner/response/score_materials/{material_id}/notes/{index}",
                        )
                    )
                previous_end = at_units + cast(int, item["duration_units"])
        notes = tuple(
            ScoreNote(
                event_id=f"{material_id}-note-{index:03d}",
                at_units=cast(int, item["at_units"]),
                duration_units=cast(int, item["duration_units"]),
                pitch=cast(int, item["pitch"]),
                voice=cast(str, item["voice"]),
                tie=cast(str | None, item["tie"]),
                articulations=tuple(cast(list[str], item["articulations"])),
            )
            for index, item in enumerate(raw_notes)
        )
        harmony_cursor = 0
        for index, item in enumerate(raw_harmonies):
            at_units = cast(int, item["at_units"])
            duration_units = cast(int, item["duration_units"])
            if at_units != harmony_cursor or at_units + duration_units > length_units:
                raise TypedRealizationError(
                    ValidationIssue(
                        IssueCode.MODEL_OUTPUT_INVALID,
                        "Typed harmonies must cover a material contiguously",
                        f"/runner/response/score_materials/{material_id}/harmonies/{index}",
                    )
                )
            harmony_cursor += duration_units
        if raw_harmonies and harmony_cursor != length_units:
            raise TypedRealizationError(
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "Typed harmonies must fill the fixed material length",
                    f"/runner/response/score_materials/{material_id}/harmonies/"
                    f"{len(raw_harmonies) - 1}",
                )
            )
        harmonies = tuple(
            ScoreHarmony(
                harmony_id=f"{material_id}-harmony-{index:03d}",
                at_units=cast(int, item["at_units"]),
                duration_units=cast(int, item["duration_units"]),
                root_pitch_class=cast(int, item["root_pitch_class"]),
                quality=cast(str, item["quality"]),
            )
            for index, item in enumerate(raw_harmonies)
        )
        raw_directions = cast(list[Mapping[str, object]], raw["directions"])
        for index, item in enumerate(raw_directions):
            at_units = cast(int, item["at_units"])
            kind = cast(str, item["kind"])
            value = cast(str, item["value"])
            if at_units > length_units:
                raise TypedRealizationError(
                    ValidationIssue(
                        IssueCode.MODEL_OUTPUT_INVALID,
                        "A typed direction is outside its fixed material length",
                        f"/runner/response/score_materials/{material_id}/directions/"
                        f"{index}/at_units",
                    )
                )
            allowed_values = (
                {"pp", "p", "mp", "mf", "f", "ff"} if kind == "dynamic" else {"light", "full"}
            )
            if value not in allowed_values:
                raise TypedRealizationError(
                    ValidationIssue(
                        IssueCode.MODEL_OUTPUT_INVALID,
                        "A typed direction value does not match its kind",
                        f"/runner/response/score_materials/{material_id}/directions/{index}/value",
                    )
                )
        directions = tuple(
            ScoreDirection(
                direction_id=f"{material_id}-direction-{index:03d}",
                at_units=cast(int, item["at_units"]),
                kind=cast(str, item["kind"]),
                value=cast(str, item["value"]),
            )
            for index, item in enumerate(raw_directions)
        )
        materials.append(
            ScoreMaterial(
                material_id=material_id,
                length_units=length_units,
                notes=notes,
                derived_from=material_derivations.get(material_id),
                directions=directions,
                harmonies=harmonies,
                foreground_voice=foreground_voice,
            )
        )
    score = ScoreSpec(
        score_id=f"{plan.plan_id}-score",
        divisions=_DIVISIONS,
        materials=tuple(materials),
    )
    validate_score_spec(plan, score)
    return score


def build_performance_spec_from_typed_response(
    plan: PiecePlan,
    node_performances: Mapping[str, object],
) -> PerformanceSpec:
    """Build a PerformanceSpec while keeping runner-owned values deterministic."""

    items = []
    for node in plan.nodes:
        raw = cast(Mapping[str, object], node_performances[node.node_id])
        if raw["timing_profile"] is None and raw["timing_amount"] is not None:
            raise TypedRealizationError(
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "A typed timing amount requires a timing profile",
                    f"/runner/response/node_performances/{node.node_id}/timing_amount",
                )
            )
        items.append(
            NodePerformance(
                node_id=node.node_id,
                timing_profile=cast(str | None, raw["timing_profile"]),
                timing_amount=cast(str | None, raw["timing_amount"]),
                dynamics_profile=cast(str | None, raw["dynamics_profile"]),
                articulation_profile=cast(str | None, raw["articulation_profile"]),
                coordination_profile=cast(str | None, raw["coordination_profile"]),
                pedal_profile=cast(str | None, raw["pedal_profile"]),
            )
        )
    return PerformanceSpec(
        performance_id=f"{plan.plan_id}-performance",
        target_duration_ms=_TARGET_DURATION_MS,
        default_velocity=64,
        timing_budget_id="narrative-v2",
        node_performances=tuple(items),
        key_release_percent=100,
        velocity_policy_id="foreground-accompaniment-harmony-shape-v1",
    )
