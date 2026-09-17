"""Finite model-response contracts for solo-piano phase 7."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import cast

from jsonschema import Draft202012Validator

from llm_musical_composer.run_state import sha256_json

from .performance_ir import (
    PerformanceIrValidationError,
    PerformanceSpec,
    SectionPerformance,
    validate_performance_spec,
)
from .profile_capabilities import GenerationProfileCapabilities
from .projection_ledger import ProjectionLedgerEntry, validate_projection_ledger
from .score_ir import PiecePlan, ScoreSpec
from .validation import IssueCode, ValidationIssue

_PERFORMANCE_FIELDS = (
    "timing_profile",
    "timing_amount",
    "dynamics_profile",
    "articulation_profile",
    "coordination_profile",
    "pedal_profile",
)


@dataclass(frozen=True, slots=True)
class PerformanceChoiceBuildResult:
    valid: bool
    performance: PerformanceSpec | None
    projection_ledger: tuple[ProjectionLedgerEntry, ...]
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True, slots=True)
class PerformanceChoiceOperation:
    operation_id: str
    target_section_id: str
    direction_ids: tuple[str, ...]
    comparison_section_ids: tuple[str, ...]


def build_performance_choice_operations(
    document: Mapping[str, object], plan: PiecePlan
) -> tuple[PerformanceChoiceOperation, ...]:
    """Order directed sections so every directed comparison source is settled first."""
    directions_by_target = _directions_by_target(document)
    script = cast(Mapping[str, object], document["script"])
    setup = cast(Mapping[str, object], script["performance_setup"])
    directions = cast(Mapping[str, Mapping[str, object]], setup["performance_directions"])
    section_order = {node.section_id: index for index, node in enumerate(plan.nodes)}
    directed_sections = set(directions_by_target)
    comparisons_by_target: dict[str, set[str]] = {
        section_id: set() for section_id in directed_sections
    }
    for section_id, grouped in directions_by_target.items():
        for direction_id, _ in grouped:
            relative_to = directions[direction_id].get("relative_to")
            if relative_to is not None:
                comparisons_by_target[section_id].add(
                    cast(str, cast(Mapping[str, object], relative_to)["id"])
                )

    remaining = set(directed_sections)
    settled: set[str] = set()
    ordered_targets: list[str] = []
    while remaining:
        ready = sorted(
            (
                section_id
                for section_id in remaining
                if (comparisons_by_target[section_id] & directed_sections) <= settled
            ),
            key=section_order.__getitem__,
        )
        if not ready:
            raise ValueError("performance direction comparisons cannot be ordered")
        ordered_targets.extend(ready)
        settled.update(ready)
        remaining.difference_update(ready)
    return tuple(
        PerformanceChoiceOperation(
            operation_id=f"phase7-performance-{index:03d}",
            target_section_id=section_id,
            direction_ids=tuple(
                direction_id for direction_id, _ in directions_by_target[section_id]
            ),
            comparison_section_ids=tuple(
                sorted(comparisons_by_target[section_id], key=section_order.__getitem__)
            ),
        )
        for index, section_id in enumerate(ordered_targets, start=1)
    )


def performance_choices_prompt(
    document: Mapping[str, object],
    plan: PiecePlan,
    score: ScoreSpec,
    operation: PerformanceChoiceOperation,
    accepted_performances: Mapping[str, Mapping[str, object]],
    capabilities: GenerationProfileCapabilities,
) -> str:
    """Describe one performance target without exposing arbitrary numeric controls."""
    script = cast(Mapping[str, object], document["script"])
    setup = cast(Mapping[str, object], script["performance_setup"])
    directions = cast(Mapping[str, Mapping[str, object]], setup["performance_directions"])
    operation_directions = [directions[item] for item in operation.direction_ids]
    aspect_ids = {
        aspect
        for direction in operation_directions
        for aspect in cast(Sequence[str], direction["performance_aspects"])
    }
    context = {
        "target": {
            "description": cast(Mapping[str, object], script["sections"])[
                operation.target_section_id
            ]["description"],
            "score_summary": _score_summary(plan, score, operation.target_section_id),
        },
        "performance_directions": [
            {
                "direction_id": direction_id,
                "description": directions[direction_id]["description"],
                "performance_aspects": directions[direction_id]["performance_aspects"],
                "relative_to": directions[direction_id].get("relative_to"),
            }
            for direction_id in operation.direction_ids
        ],
        "settled_comparisons": [
            {
                "section_id": section_id,
                "performance": accepted_performances.get(section_id),
            }
            for section_id in operation.comparison_section_ids
        ],
        "performance_choice_vocabulary_version": (
            capabilities.performance_choice_vocabulary_version
        ),
        "performance_choice_vocabulary": [
            {
                "aspect_id": aspect.aspect_id,
                "description": aspect.description,
                "fields": [
                    {
                        "field_name": field.field_name,
                        "choices": [
                            {"value": choice.value, "description": choice.description}
                            for choice in field.choices
                        ],
                    }
                    for field in aspect.fields
                ],
            }
            for aspect in capabilities.performance_aspects
            if aspect.aspect_id in aspect_ids
        ],
    }
    return (
        "完成した楽譜を変えず、演奏指示を実現する演奏方法を選んでください。"
        "機械が読む値は入力の演奏選択語彙からだけ選び、任意のミリ秒、velocity、"
        "ペダル値、音符、音高、和音は返さないでください。指定されていない演奏要素の"
        "fieldはnullにしてください。各演奏指示を、扱ったものか扱わなかったものの"
        "どちらか一方へ一度だけ含めてください。対象区分IDは返さず、Schemaどおりの"
        "位置で返してください。\n\n"
        "演奏選択語彙と入力: "
        f"{json.dumps(context, ensure_ascii=False, sort_keys=True)}\n"
    )


def performance_operation_immutable_input(
    document: Mapping[str, object],
    plan: PiecePlan,
    score: ScoreSpec,
    operation: PerformanceChoiceOperation,
    capabilities: GenerationProfileCapabilities,
) -> dict[str, object]:
    """Return the stable input identity shared by execution and offline replay."""
    return {
        "script_sha256": sha256_json(document),
        "piece_plan_sha256": sha256_json(asdict(plan)),
        "score_spec_sha256": sha256_json(asdict(score)),
        "target_section_id": operation.target_section_id,
        "comparison_section_ids": list(operation.comparison_section_ids),
        "performance_choice_vocabulary_version": (
            capabilities.performance_choice_vocabulary_version
        ),
    }


def build_performance_choices(
    document: Mapping[str, object],
    plan: PiecePlan,
    target_section_ids: Sequence[str],
    response: Mapping[str, object],
    capabilities: GenerationProfileCapabilities,
    *,
    performance_id: str,
) -> PerformanceChoiceBuildResult:
    """Bind one positional model response directly to the v2 performance IR."""
    schema_errors = sorted(
        Draft202012Validator(
            performance_choices_response_schema(document, target_section_ids, capabilities)
        ).iter_errors(response),
        key=lambda error: (tuple(str(part) for part in error.absolute_path), error.message),
    )
    if schema_errors:
        return PerformanceChoiceBuildResult(
            False,
            None,
            (),
            tuple(
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    error.message,
                    "".join(f"/{_escape_pointer_token(part)}" for part in error.absolute_path),
                )
                for error in schema_errors
            ),
        )
    semantic_issues = _check_response_semantics(
        document,
        target_section_ids,
        cast(Mapping[str, object], response),
        capabilities,
    )
    if semantic_issues:
        return PerformanceChoiceBuildResult(False, None, (), semantic_issues)

    performances = tuple(
        SectionPerformance(
            section_id=section_id,
            **{field: item[field] for field in _PERFORMANCE_FIELDS},
        )
        for section_id, item in zip(
            target_section_ids,
            cast(Sequence[dict[str, object]], response["performances"]),
            strict=True,
        )
    )
    setup = cast(
        Mapping[str, object],
        cast(Mapping[str, object], document["script"])["performance_setup"],
    )
    performance = PerformanceSpec(
        performance_id=performance_id,
        target_duration_ms=round(float(setup["target_duration_seconds"]) * 1000),
        default_velocity=64,
        timing_budget_id="subtle-v1",
        section_performances=performances,
        velocity_policy_id=min(capabilities.velocity_policy_ids),
    )
    try:
        validate_performance_spec(plan, performance, capabilities)
    except PerformanceIrValidationError as error:
        return PerformanceChoiceBuildResult(
            False,
            None,
            (),
            (
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    str(error),
                    "/performances",
                ),
            ),
        )

    ledger = _performance_direction_ledger(document, target_section_ids, response)
    validate_projection_ledger(ledger)
    return PerformanceChoiceBuildResult(True, performance, ledger, ())


def performance_choices_response_schema(
    document: Mapping[str, object],
    target_section_ids: Sequence[str],
    capabilities: GenerationProfileCapabilities,
) -> dict[str, object]:
    """Build a positional response schema from each target's directed aspects."""
    directions_by_target = _directions_by_target(document)
    target_schemas = [
        _target_response_schema(directions_by_target.get(section_id, ()), capabilities)
        for section_id in target_section_ids
    ]
    item_schema = target_schemas[0] if len(target_schemas) == 1 else {"anyOf": target_schemas}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["performances"],
        "properties": {
            "performances": {
                "type": "array",
                "minItems": len(target_schemas),
                "maxItems": len(target_schemas),
                "items": item_schema,
            }
        },
    }


def _directions_by_target(
    document: Mapping[str, object],
) -> dict[str, tuple[tuple[str, tuple[str, ...]], ...]]:
    script = cast(Mapping[str, object], document["script"])
    setup = cast(Mapping[str, object], script["performance_setup"])
    directions = cast(Mapping[str, Mapping[str, object]], setup["performance_directions"])
    grouped: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    for direction_id, direction in sorted(directions.items()):
        target = cast(Mapping[str, object], direction["target"])
        if target["type"] != "section":
            continue
        section_id = cast(str, target["id"])
        aspects = tuple(cast(Sequence[str], direction["performance_aspects"]))
        grouped.setdefault(section_id, []).append((direction_id, aspects))
    return {section_id: tuple(values) for section_id, values in grouped.items()}


def _target_response_schema(
    directions: Sequence[tuple[str, tuple[str, ...]]],
    capabilities: GenerationProfileCapabilities,
) -> dict[str, object]:
    direction_ids = [direction_id for direction_id, _ in directions]
    allowed_fields = {
        field
        for _, aspects in directions
        for aspect in aspects
        for field in capabilities.performance_fields_for_aspect(aspect)
    }
    properties: dict[str, object] = {
        field: _field_schema(field, field in allowed_fields, capabilities)
        for field in _PERFORMANCE_FIELDS
    }
    properties.update(
        {
            "handled_directions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["direction_id", "field_names"],
                    "properties": {
                        "direction_id": {"enum": direction_ids},
                        "field_names": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"enum": sorted(allowed_fields)},
                        },
                    },
                },
            },
            "unhandled_directions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["direction_id", "reason"],
                    "properties": {
                        "direction_id": {"enum": direction_ids},
                        "reason": {"type": "string", "minLength": 1},
                    },
                },
            },
        }
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            *_PERFORMANCE_FIELDS,
            "handled_directions",
            "unhandled_directions",
        ],
        "properties": properties,
    }


def _field_schema(
    field_name: str,
    allowed: bool,
    capabilities: GenerationProfileCapabilities,
) -> dict[str, object]:
    if not allowed:
        return {"type": "null"}
    return {
        "anyOf": [
            {"type": "null"},
            {"enum": list(capabilities.performance_choices_for_field(field_name))},
        ]
    }


def _score_summary(
    plan: PiecePlan, score: ScoreSpec, target_section_id: str
) -> list[dict[str, object]]:
    children: dict[str, list[str]] = {}
    for node in plan.nodes:
        if node.parent_section_id is not None:
            children.setdefault(node.parent_section_id, []).append(node.section_id)
    descendants: set[str] = set()
    pending = [target_section_id]
    while pending:
        section_id = pending.pop()
        if section_id in descendants:
            continue
        descendants.add(section_id)
        pending.extend(children.get(section_id, ()))
    return [
        {
            "source_section_id": unit.source_section_id,
            "length_units": unit.length_units,
            "harmony_count": len(unit.harmonies),
            "note_count": sum(len(layer.notes) for layer in unit.score_unit_layers),
            "voices": sorted(
                {note.voice for layer in unit.score_unit_layers for note in layer.notes}
            ),
        }
        for unit in score.score_units
        if unit.source_section_id in descendants
    ]


def _performance_direction_ledger(
    document: Mapping[str, object],
    target_section_ids: Sequence[str],
    response: Mapping[str, object],
) -> tuple[ProjectionLedgerEntry, ...]:
    script = cast(Mapping[str, object], document["script"])
    setup = cast(Mapping[str, object], script["performance_setup"])
    directions = cast(Mapping[str, Mapping[str, object]], setup["performance_directions"])
    entries: list[ProjectionLedgerEntry] = []
    for section_id, item in zip(
        target_section_ids,
        cast(Sequence[Mapping[str, object]], response["performances"]),
        strict=True,
    ):
        for handled in cast(Sequence[Mapping[str, object]], item["handled_directions"]):
            direction_id = cast(str, handled["direction_id"])
            fields = cast(Sequence[str], handled["field_names"])
            selected = ",".join(f"{field}={item[field]}" for field in fields)
            entries.append(
                ProjectionLedgerEntry(
                    "performance_direction",
                    direction_id,
                    "section_performance",
                    section_id,
                    "performance_spec",
                    "natural_language_claim",
                    "unverified",
                    f"{directions[direction_id]['description']}: {selected}",
                )
            )
        for unhandled in cast(Sequence[Mapping[str, object]], item["unhandled_directions"]):
            direction_id = cast(str, unhandled["direction_id"])
            entries.append(
                ProjectionLedgerEntry(
                    "performance_direction",
                    direction_id,
                    "section_performance",
                    section_id,
                    "performance_spec",
                    "natural_language_claim",
                    "unverified",
                    f"{directions[direction_id]['description']}: "
                    f"unhandled because {unhandled['reason']}",
                )
            )
    return tuple(entries)


def _check_response_semantics(
    document: Mapping[str, object],
    target_section_ids: Sequence[str],
    response: Mapping[str, object],
    capabilities: GenerationProfileCapabilities,
) -> tuple[ValidationIssue, ...]:
    directions_by_target = _directions_by_target(document)
    issues: list[ValidationIssue] = []
    for index, (section_id, item) in enumerate(
        zip(
            target_section_ids,
            cast(Sequence[Mapping[str, object]], response["performances"]),
            strict=True,
        )
    ):
        direction_aspects = dict(directions_by_target[section_id])
        expected = list(direction_aspects)
        actual = [
            cast(str, outcome["direction_id"])
            for collection in ("handled_directions", "unhandled_directions")
            for outcome in cast(Sequence[Mapping[str, object]], item[collection])
        ]
        if sorted(actual) != sorted(expected) or len(actual) != len(set(actual)):
            issues.append(
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "Every performance direction must be accounted for exactly once",
                    f"/performances/{index}",
                )
            )
            continue
        claimed_fields: set[str] = set()
        for handled_index, outcome in enumerate(
            cast(Sequence[Mapping[str, object]], item["handled_directions"])
        ):
            direction_id = cast(str, outcome["direction_id"])
            allowed_fields = {
                field
                for aspect in direction_aspects[direction_id]
                for field in capabilities.performance_fields_for_aspect(aspect)
            }
            field_names = set(cast(Sequence[str], outcome["field_names"]))
            if not field_names <= allowed_fields or any(
                item[field] is None for field in field_names
            ):
                issues.append(
                    ValidationIssue(
                        IssueCode.MODEL_OUTPUT_INVALID,
                        "Handled fields must be non-null and belong to the direction aspects",
                        f"/performances/{index}/handled_directions/{handled_index}/field_names",
                    )
                )
            claimed_fields.update(field_names)
        selected_fields = {field for field in _PERFORMANCE_FIELDS if item[field] is not None}
        if selected_fields - claimed_fields:
            issues.append(
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "Every selected performance field must handle a direction",
                    f"/performances/{index}",
                )
            )
    return tuple(issues)


def _escape_pointer_token(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")
