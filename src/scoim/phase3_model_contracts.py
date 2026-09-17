"""Private typed model contracts for phase-3 planning and shared harmony."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from .projection_ledger import ProjectionLedgerEntry, validate_projection_ledger
from .score_ir import PiecePlan, ScoreHarmony
from .score_projection import PlanChoice, build_piece_plan
from .validation import IssueCode, ValidationIssue


@dataclass(frozen=True, slots=True)
class SectionHarmonicIntent:
    section_id: str
    score_unit_id: str
    harmonic_intent: str
    connection_from_previous: str


@dataclass(frozen=True, slots=True)
class HarmonicPlan:
    piece_plan: PiecePlan
    overall_harmonic_story: str
    section_intents: tuple[SectionHarmonicIntent, ...]
    divisions: int
    length_units_by_score_unit: dict[str, int]
    projection_ledger: tuple[ProjectionLedgerEntry, ...]


def ordered_leaf_section_ids(document: Mapping[str, object]) -> tuple[str, ...]:
    """Return leaf section identifiers in the script's recursive performance order."""
    script = cast(dict[str, object], document["script"])
    sections = cast(dict[str, dict[str, object]], script["sections"])
    children: dict[str, list[str]] = {}
    for section_id, section in sections.items():
        parent_id = section["parent_section_id"]
        if isinstance(parent_id, str):
            children.setdefault(parent_id, []).append(section_id)
    for child_ids in children.values():
        child_ids.sort(key=lambda section_id: cast(int, sections[section_id]["order"]))
    ordered_leaves: list[str] = []

    def visit(section_id: str) -> None:
        child_ids = children.get(section_id, ())
        if not child_ids:
            ordered_leaves.append(section_id)
            return
        for child_id in child_ids:
            visit(child_id)

    visit(cast(str, script["root_section_id"]))
    return tuple(ordered_leaves)


def overall_plan_response_schema(*, leaf_count: int) -> dict[str, object]:
    """Return the positional overall-plan transfer schema for one script."""
    intent = {
        "type": "object",
        "additionalProperties": False,
        "required": ["harmonic_intent", "connection_from_previous"],
        "properties": {
            "harmonic_intent": {"type": "string", "minLength": 1},
            "connection_from_previous": {"type": "string", "minLength": 1},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "tonal_center",
            "mode",
            "overall_harmonic_story",
            "section_harmonic_intents",
        ],
        "properties": {
            "tonal_center": {"type": "integer", "minimum": 0, "maximum": 11},
            "mode": {"type": "string", "enum": ["major", "minor"]},
            "overall_harmonic_story": {"type": "string", "minLength": 1},
            "section_harmonic_intents": {
                "type": "array",
                "minItems": leaf_count,
                "maxItems": leaf_count,
                "items": intent,
            },
        },
    }


def overall_plan_prompt(document: Mapping[str, object]) -> str:
    """Ask for positional tonal and harmonic intent without generated identifiers."""
    script = cast(dict[str, object], document["script"])
    sections = cast(dict[str, dict[str, object]], script["sections"])
    leaf_sections = [
        {
            "section_id": section_id,
            "role": sections[section_id]["role"],
            "description": sections[section_id]["description"],
            "relative_length": sections[section_id]["relative_length"],
        }
        for section_id in ordered_leaf_section_ids(document)
    ]
    context = {
        "title": script["title"],
        "brief": script["brief"],
        "leaf_sections_in_performance_order": leaf_sections,
        "materials": script["materials"],
        "material_placements": script["material_placements"],
        "variation_relations": script["script_element_variation_relations"],
        "material_placement_transitions": script["material_placement_transitions"],
    }
    return (
        "曲全体の調性と和声の流れを設計してください。区分IDは応答へ返さず、"
        "section_harmonic_intentsを入力の子なし区分と同じ順、同じ件数で返してください。"
        "指定されたSchemaだけに従ってください。\n\n"
        f"入力: {json.dumps(context, ensure_ascii=False, sort_keys=False)}\n"
    )


def build_harmonic_plan(
    document: Mapping[str, object],
    response: Mapping[str, object],
    *,
    divisions: int,
    units_per_duration_weight: int,
) -> HarmonicPlan:
    """Bind positional model intent to deterministic plan nodes and time capacity."""
    piece_plan, ledger = build_piece_plan(
        document,
        PlanChoice(
            tonal_center=cast(int, response["tonal_center"]),
            mode=cast(str, response["mode"]),
        ),
    )
    leaf_nodes = tuple(node for node in piece_plan.nodes if node.score_unit_id is not None)
    response_intents = cast(list[dict[str, object]], response["section_harmonic_intents"])
    if len(response_intents) != len(leaf_nodes):
        raise ValueError("section harmonic intents must exactly cover leaf sections")
    intents = tuple(
        SectionHarmonicIntent(
            section_id=node.section_id,
            score_unit_id=cast(str, node.score_unit_id),
            harmonic_intent=cast(str, intent["harmonic_intent"]),
            connection_from_previous=cast(str, intent["connection_from_previous"]),
        )
        for node, intent in zip(leaf_nodes, response_intents, strict=True)
    )
    lengths: dict[str, int] = {}
    for node in leaf_nodes:
        raw_length = cast(float, node.duration_weight) * units_per_duration_weight
        if not float(raw_length).is_integer() or raw_length <= 0:
            raise ValueError("duration weight cannot be represented by the configured unit grid")
        lengths[cast(str, node.score_unit_id)] = int(raw_length)
    score_unit_ledger = tuple(
        ProjectionLedgerEntry(
            source_kind="plan_node",
            source_id=node.section_id,
            target_kind="score_unit",
            target_id=cast(str, node.score_unit_id),
            target_stage="harmonic_plan",
            verification="direct_id_equality",
            status="passed",
            evidence=(f"plan_node_id={node.section_id}; score_unit_id={node.score_unit_id}"),
        )
        for node in leaf_nodes
    )
    completed_ledger = ledger + score_unit_ledger
    validate_projection_ledger(completed_ledger)
    return HarmonicPlan(
        piece_plan=piece_plan,
        overall_harmonic_story=cast(str, response["overall_harmonic_story"]),
        section_intents=intents,
        divisions=divisions,
        length_units_by_score_unit=lengths,
        projection_ledger=completed_ledger,
    )


def harmony_response_schema() -> dict[str, object]:
    """Return the private transfer schema for one score unit's shared harmony."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["harmonies"],
        "properties": {
            "harmonies": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["duration_units", "root_pitch_class", "quality"],
                    "properties": {
                        "duration_units": {"type": "integer", "minimum": 1},
                        "root_pitch_class": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 11,
                        },
                        "quality": {
                            "type": "string",
                            "enum": [
                                "major",
                                "minor",
                                "diminished",
                                "major-seventh",
                            ],
                        },
                    },
                },
            }
        },
    }


def build_score_harmonies(
    score_unit_id: str, response: Mapping[str, object]
) -> tuple[ScoreHarmony, ...]:
    """Assign stable identifiers and cumulative positions to validated harmonies."""
    cursor = 0
    harmonies: list[ScoreHarmony] = []
    for index, item in enumerate(cast(list[dict[str, object]], response["harmonies"]), start=1):
        duration = cast(int, item["duration_units"])
        harmonies.append(
            ScoreHarmony(
                harmony_id=f"harmony-{score_unit_id}-{index:03d}",
                at_units=cursor,
                duration_units=duration,
                root_pitch_class=cast(int, item["root_pitch_class"]),
                quality=cast(str, item["quality"]),
            )
        )
        cursor += duration
    return tuple(harmonies)


def harmony_projection_entries(
    score_unit_id: str, harmonies: tuple[ScoreHarmony, ...]
) -> tuple[ProjectionLedgerEntry, ...]:
    """Record the adjacent projection from one score unit to its harmonies."""
    entries = tuple(
        ProjectionLedgerEntry(
            source_kind="score_unit",
            source_id=score_unit_id,
            target_kind="score_harmony",
            target_id=harmony.harmony_id,
            target_stage="shared_harmony",
            verification="time_coverage_and_source_equality",
            status="passed",
            evidence=(
                f"score_unit_id={score_unit_id}; at_units={harmony.at_units}; "
                f"duration_units={harmony.duration_units}"
            ),
        )
        for harmony in harmonies
    )
    validate_projection_ledger(entries)
    return entries


def check_harmony_response(
    response: Mapping[str, object],
    *,
    length_units: int,
    required_final_tonic: tuple[int, str] | None = None,
) -> tuple[ValidationIssue, ...]:
    """Check time coverage and the optional whole-piece terminal condition."""
    harmonies = cast(list[dict[str, object]], response["harmonies"])
    issues: list[ValidationIssue] = []
    total = sum(cast(int, harmony["duration_units"]) for harmony in harmonies)
    if total != length_units:
        issues.append(
            ValidationIssue(
                IssueCode.SEMANTIC_INVALID,
                "Harmonies must exactly cover the score unit length",
                "/harmonies",
            )
        )
    if required_final_tonic is not None and harmonies:
        last = harmonies[-1]
        required_root, required_quality = required_final_tonic
        if last["root_pitch_class"] != required_root or last["quality"] != required_quality:
            issues.append(
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    "The final harmony must be the whole-piece tonic",
                    f"/harmonies/{len(harmonies) - 1}",
                )
            )
    return tuple(issues)


def harmony_prompt(
    document: Mapping[str, object],
    plan: HarmonicPlan,
    *,
    intent_index: int,
    previous_final_harmony: Mapping[str, object] | None,
) -> str:
    """Build one bounded harmony request from validated neighboring context."""
    script = cast(dict[str, object], document["script"])
    sections = cast(dict[str, dict[str, object]], script["sections"])
    materials = cast(dict[str, dict[str, object]], script["materials"])
    placements = cast(dict[str, dict[str, object]], script["material_placements"])
    relations = cast(dict[str, dict[str, object]], script["script_element_variation_relations"])
    target_intent = plan.section_intents[intent_index]
    target_placements: dict[str, object] = {}
    target_placement_ids: set[str] = set()
    for placement_id, placement in placements.items():
        if placement["section_id"] != target_intent.section_id:
            continue
        target_placement_ids.add(placement_id)
        material_id = cast(str, placement["material_id"])
        target_placements[placement_id] = {
            **placement,
            "material_description": materials[material_id]["description"],
        }
    target_relations: dict[str, object] = {}
    for relation_id, relation in relations.items():
        source = cast(dict[str, object], relation["source"])
        target = cast(dict[str, object], relation["target"])
        if source["id"] in target_placement_ids or target["id"] in target_placement_ids:
            target_relations[relation_id] = relation
    next_intent = (
        plan.section_intents[intent_index + 1].harmonic_intent
        if intent_index + 1 < len(plan.section_intents)
        else None
    )
    context = {
        "whole_piece": {
            "tonal_center": plan.piece_plan.tonal_center,
            "mode": plan.piece_plan.mode,
            "overall_harmonic_story": plan.overall_harmonic_story,
        },
        "target": {
            "section_id": target_intent.section_id,
            "section_description": sections[target_intent.section_id]["description"],
            "score_unit_id": target_intent.score_unit_id,
            "length_units": plan.length_units_by_score_unit[target_intent.score_unit_id],
            "harmonic_intent": target_intent.harmonic_intent,
            "connection_from_previous": target_intent.connection_from_previous,
            "material_placements": target_placements,
            "variation_relations": target_relations,
        },
        "previous_final_harmony": (
            dict(previous_final_harmony) if previous_final_harmony is not None else None
        ),
        "next_harmonic_intent": next_intent,
        "allowed_qualities": ["major", "minor", "diminished", "major-seventh"],
    }
    return (
        "対象の楽譜生成単位を完全に覆う共有和声を提案してください。"
        "IDや開始位置は返さず、指定されたSchemaだけに従ってください。\n\n"
        f"入力: {json.dumps(context, ensure_ascii=False, sort_keys=True)}\n"
    )
