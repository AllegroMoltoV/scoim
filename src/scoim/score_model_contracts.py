"""Typed joint score requests and deterministic, atomic candidate evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import cast

from llm_musical_composer.run_state import sha256_json

from .phase3_model_contracts import HarmonicPlan
from .phase4_model_contracts import (
    build_foreground_notes,
    check_foreground_response,
    foreground_response_schema,
)
from .phase5_model_contracts import accompaniment_response_schema, check_accompaniment_response
from .projection_ledger import ProjectionLedgerEntry, validate_projection_ledger
from .score_generation_context import ancestor_sections
from .score_group_pitch_placement import (
    SCORE_GROUP_SEARCH_LIMIT,
    ScoreGroupPitchPlacementError,
    place_score_group_accompaniment_events,
)
from .score_ir import ScoreHarmony, ScoreNote
from .score_work_plan import ScoreComparison, ScoreWorkOperation, section_leaf_ids
from .terminal_boundary import (
    TerminalBoundary,
    TerminalBoundaryError,
    available_terminal_boundary,
    final_role_placement_ids,
    final_score_unit_id,
    terminal_boundary_role,
)
from .validation import IssueCode, ValidationIssue


@dataclass(frozen=True, slots=True)
class ScoreGroupCandidateResult:
    issues: tuple[ValidationIssue, ...] = ()
    notes_by_placement: dict[str, tuple[ScoreNote, ...]] = field(default_factory=dict)
    responses_by_placement: dict[str, dict[str, object]] = field(default_factory=dict)
    projection_ledger: tuple[ProjectionLedgerEntry, ...] = ()
    comparison_evidence: tuple[dict[str, object], ...] = ()
    evaluated_candidate_count_by_score_unit: dict[str, int] = field(default_factory=dict)
    terminal_boundary: TerminalBoundary | None = None


def _placements(document: Mapping[str, object]) -> Mapping[str, Mapping[str, object]]:
    return cast(
        Mapping[str, Mapping[str, object]],
        cast(Mapping[str, object], document["script"])["material_placements"],
    )


def _role_ids(
    document: Mapping[str, object], operation: ScoreWorkOperation, role: str
) -> tuple[str, ...]:
    placements = _placements(document)
    return tuple(key for key in operation.material_placement_ids if placements[key]["role"] == role)


def _comparisons(
    operation: ScoreWorkOperation, comparisons: Sequence[ScoreComparison]
) -> tuple[ScoreComparison, ...]:
    owned = set(operation.material_placement_ids)
    return tuple(item for item in comparisons if owned.intersection(item.target_placement_ids))


def score_group_response_schema(
    document: Mapping[str, object], operation: ScoreWorkOperation
) -> dict[str, object]:
    """Bind each role's positional response to exactly the owned placements."""
    properties: dict[str, object] = {}
    for key, role, schema in (
        ("foregrounds", "foreground", foreground_response_schema()),
        ("accompaniments", "accompaniment", accompaniment_response_schema()),
    ):
        count = len(_role_ids(document, operation, role))
        properties[key] = {
            "type": "array",
            "minItems": count,
            "maxItems": count,
            "items": schema,
        }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["foregrounds", "accompaniments"],
        "properties": properties,
    }


def _note_values(notes: tuple[ScoreNote, ...]) -> list[dict[str, object]]:
    return [
        {
            "at_units": note.at_units,
            "duration_units": note.duration_units,
            "pitch": note.pitch,
            "voice": note.voice,
            "articulations": list(note.articulations),
            "tie": note.tie,
        }
        for note in notes
    ]


def _boundary_notes(notes: tuple[ScoreNote, ...], kind: str) -> tuple[ScoreNote, ...]:
    if not notes:
        raise ValueError("a comparison placement must contain notes")
    if kind == "transition_source":
        end = max(note.at_units + note.duration_units for note in notes)
        return tuple(note for note in notes if note.at_units + note.duration_units == end)
    if kind == "transition_target":
        start = min(note.at_units for note in notes)
        return tuple(note for note in notes if note.at_units == start)
    return notes


def score_group_prompt(
    document: Mapping[str, object],
    harmonic_plan: HarmonicPlan,
    operation: ScoreWorkOperation,
    comparisons: Sequence[ScoreComparison],
    *,
    harmonies_by_score_unit: Mapping[str, tuple[ScoreHarmony, ...]],
    accepted_notes_by_placement: Mapping[str, tuple[ScoreNote, ...]],
    accepted_responses_by_placement: Mapping[str, Mapping[str, object]],
    shared_terminal: bool = True,
) -> str:
    """Distinguish immutable external scores from jointly proposed internal references."""
    script = cast(Mapping[str, object], document["script"])
    placements = _placements(document)
    sections = cast(Mapping[str, Mapping[str, object]], script["sections"])
    materials = cast(Mapping[str, Mapping[str, object]], script["materials"])
    owned = set(operation.material_placement_ids)
    relevant = _comparisons(operation, comparisons)
    intents = {item.score_unit_id: item for item in harmonic_plan.section_intents}

    def placement_context(key: str, *, source: bool = False, kind: str = "") -> dict[str, object]:
        placement = placements[key]
        section_id = cast(str, placement["section_id"])
        unit_id = f"score-unit-{section_id}"
        material_id = cast(str, placement["material_id"])
        context: dict[str, object] = {
            "material_placement_id": key,
            "section_id": section_id,
            "role": placement["role"],
            "material_id": material_id,
            "material_description": materials[material_id]["description"],
            "section_description": sections[section_id]["description"],
            "ancestor_sections": ancestor_sections(document, section_id),
            "length_units": harmonic_plan.length_units_by_score_unit[unit_id],
            "harmonic_intent": intents[unit_id].harmonic_intent,
            "connection_from_previous": intents[unit_id].connection_from_previous,
            "shared_harmony": [asdict(value) for value in harmonies_by_score_unit[unit_id]],
            "reference_state": "joint_candidate" if key in owned else "accepted",
        }
        if key in owned:
            role_ids = _role_ids(document, operation, cast(str, placement["role"]))
            context["response_array"] = (
                "foregrounds" if placement["role"] == "foreground" else "accompaniments"
            )
            context["response_index"] = role_ids.index(key)
            if source and kind.startswith("transition_"):
                context["boundary_selection"] = (
                    "last_ending_notes" if kind == "transition_source" else "first_starting_notes"
                )
        elif source:
            if key not in accepted_notes_by_placement:
                raise ValueError(f"a score comparison source has not been accepted: {key}")
            notes = accepted_notes_by_placement[key]
            context["notes"] = _note_values(notes)
            context["notes_sha256"] = sha256_json(context["notes"])
            if kind.startswith("transition_"):
                context["boundary_notes"] = _note_values(_boundary_notes(notes, kind))
            if placement["role"] == "accompaniment":
                response = accepted_responses_by_placement.get(key)
                if response is None or "events" not in response:
                    raise ValueError(f"an accompaniment comparison response is missing: {key}")
                context["events"] = response["events"]
        return context

    def section_context(
        section_id: str, ids: tuple[str, ...], *, source: bool
    ) -> dict[str, object]:
        leaves: list[dict[str, object]] = []
        offset = 0
        for leaf in section_leaf_ids(document, section_id):
            unit_id = f"score-unit-{leaf}"
            length = harmonic_plan.length_units_by_score_unit[unit_id]
            leaves.append(
                {
                    "section_id": leaf,
                    "offset_units": offset,
                    "length_units": length,
                    "description": sections[leaf]["description"],
                    "shared_harmony": [asdict(value) for value in harmonies_by_score_unit[unit_id]],
                    "placements": [
                        placement_context(key, source=source)
                        for key in ids
                        if placements[key]["section_id"] == leaf
                    ],
                }
            )
            offset += length
        return {"section_id": section_id, "length_units": offset, "leaves": leaves}

    contexts: list[dict[str, object]] = []
    for comparison in relevant:
        context = asdict(comparison)
        if comparison.kind == "section_variation":
            if comparison.source_section_id is None or comparison.target_section_id is None:
                raise ValueError("a section comparison requires both section IDs")
            context["source_range"] = section_context(
                comparison.source_section_id, comparison.source_placement_ids, source=True
            )
            context["target_range"] = section_context(
                comparison.target_section_id, comparison.target_placement_ids, source=False
            )
        else:
            context["source_placements"] = [
                placement_context(key, source=True, kind=comparison.kind)
                for key in comparison.source_placement_ids
            ]
        context["target_placements_in_operation"] = [
            key for key in comparison.target_placement_ids if key in owned
        ]
        contexts.append(context)
    target_units = {placements[key]["section_id"] for key in owned}
    transitions = cast(Mapping[str, Mapping[str, object]], script["material_placement_transitions"])
    transition_ids = sorted(
        {
            item.relation_id
            for item in relevant
            if item.kind.startswith("transition_") and item.relation_id is not None
        }
    )
    context = {
        "operation_id": operation.operation_id,
        "whole_piece": {
            "title": script["title"],
            "brief": script["brief"],
            "tonal_center": harmonic_plan.piece_plan.tonal_center,
            "mode": harmonic_plan.piece_plan.mode,
            "overall_harmonic_story": harmonic_plan.overall_harmonic_story,
            "divisions": harmonic_plan.divisions,
            "total_score_units": harmonic_plan.total_score_units,
        },
        "foregrounds": [
            placement_context(key) for key in _role_ids(document, operation, "foreground")
        ],
        "accompaniments": [
            placement_context(key) for key in _role_ids(document, operation, "accompaniment")
        ],
        "comparisons": contexts,
        "transitions": [{"transition_id": key, **transitions[key]} for key in transition_ids],
        "occupied_notes": [
            {
                "material_placement_id": key,
                "section_id": placements[key]["section_id"],
                "role": placements[key]["role"],
                "notes": _note_values(notes),
            }
            for key, notes in sorted(accepted_notes_by_placement.items())
            if placements[key]["section_id"] in target_units and key not in owned
        ],
    }
    if shared_terminal:
        plan = harmonic_plan.piece_plan
        final_id = final_score_unit_id(plan)
        role = terminal_boundary_role(document, plan)
        owner_ids = final_role_placement_ids(document, plan, role)
        support_ids = final_role_placement_ids(document, plan, "accompaniment")
        boundary = available_terminal_boundary(
            document, plan, harmonies_by_score_unit, accepted_notes_by_placement
        )
        context["shared_terminal"] = {
            "score_unit_id": final_id,
            "final_harmony": asdict(harmonies_by_score_unit[final_id][-1]),
            "owner_role": role,
            "owner_placements": [
                {
                    "material_placement_id": key,
                    "reference_state": "joint_candidate"
                    if key in owned
                    else ("accepted" if key in accepted_notes_by_placement else "pending"),
                }
                for key in owner_ids
            ],
            "support_placement_id": support_ids[-1] if support_ids else None,
            "accepted_boundary": asdict(boundary) if boundary else None,
            "requirement": (
                "終端担当の全配置の最後の打鍵を最後の共有和声内に置き、その打鍵時点で鳴る"
                "音符の最も遅い終了位置を共有終端にする。最後の単位では、その最後の打鍵より"
                "後に打ち始めず、共有終端を越える音価を選ばない。支持担当伴奏は最後の打鍵を"
                "含み共有終端まで続くイベントを持つ。同じ応答の前景から決まる値も守る。"
                "前景がなければ全伴奏から導出する。群外の確定音符を変更しない。"
            ),
        }
    return (
        "共同生成群の全配置を一つの候補として提案してください。foregroundsとaccompanimentsは"
        "入力の役割別配置順と同じ件数にしてください。前景は具体音符、伴奏は和音度数と希望音域を"
        "持つ記号イベントで返し、IDや演奏表現は返さないでください。伴奏の具体音高はPythonが決めます。"
        "acceptedの比較元とoccupied_notesは変更禁止です。joint_candidateはこの同じ応答で一緒に"
        "提案する未確定の対象です。内部の伴奏の最終具体音高はまだ決まっていません。"
        "区分変奏は全範囲で保持・変更要求を満たし、完全コピーを避けてください。区分の一部や"
        "前景を保ってほかを変えて構いません。区分変奏の対象へ暗黙再利用の個別変更を重ねて"
        "強制しません。明示された素材・配置変奏と、区分変奏のない暗黙再利用では完全コピーを"
        "避けてください。遷移は指定された前後の境界へ自然につないでください。"
        "同じ区分の前景では同音高かつ同声部の区間を重ねず、伴奏では同鍵重複と低域の密集を"
        "避けてください。区分ごとの時間と共有和声を守り、指定Schemaだけに従ってください。\n\n"
        f"入力: {json.dumps(context, ensure_ascii=False, sort_keys=True)}\n"
    )


def _prefixed_issues(
    issues: tuple[ValidationIssue, ...], prefix: str
) -> tuple[ValidationIssue, ...]:
    return tuple(ValidationIssue(item.code, item.message, prefix + item.path) for item in issues)


def _fingerprint(notes: tuple[ScoreNote, ...], offset: int = 0) -> list[tuple[object, ...]]:
    return [
        (
            offset + note.at_units,
            note.duration_units,
            note.pitch,
            note.voice,
            tuple(sorted(value for value in note.articulations if value != "normal")),
            note.tie or "",
        )
        for note in notes
    ]


def _section_fingerprint(
    document: Mapping[str, object],
    harmonic_plan: HarmonicPlan,
    section_id: str,
    notes: Mapping[str, tuple[ScoreNote, ...]],
) -> dict[str, object]:
    placements = _placements(document)
    values: list[tuple[object, ...]] = []
    offset = 0
    for leaf in section_leaf_ids(document, section_id):
        for key, placement in placements.items():
            if placement["section_id"] == leaf:
                values.extend(_fingerprint(notes[key], offset))
        offset += harmonic_plan.length_units_by_score_unit[f"score-unit-{leaf}"]
    return {"length_units": offset, "notes": sorted(values)}


def evaluate_score_group_candidate(
    document: Mapping[str, object],
    harmonic_plan: HarmonicPlan,
    operation: ScoreWorkOperation,
    comparisons: Sequence[ScoreComparison],
    response: Mapping[str, object],
    *,
    harmonies_by_score_unit: Mapping[str, tuple[ScoreHarmony, ...]],
    accepted_notes_by_placement: Mapping[str, tuple[ScoreNote, ...]],
    search_limit: int = SCORE_GROUP_SEARCH_LIMIT,
    shared_terminal: bool = True,
) -> ScoreGroupCandidateResult:
    """Evaluate a schema-valid response without mutating any accepted score value."""
    placements = _placements(document)
    foreground_ids = _role_ids(document, operation, "foreground")
    accompaniment_ids = _role_ids(document, operation, "accompaniment")
    raw_foregrounds = cast(list[Mapping[str, object]], response["foregrounds"])
    raw_accompaniments = cast(list[Mapping[str, object]], response["accompaniments"])
    if len(raw_foregrounds) != len(foreground_ids) or len(raw_accompaniments) != len(
        accompaniment_ids
    ):
        return ScoreGroupCandidateResult(
            (
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "Response arrays must exactly cover the score group",
                    "",
                ),
            )
        )
    owned = set(operation.material_placement_ids)
    if owned.intersection(accepted_notes_by_placement):
        raise ValueError("an owned score placement is already accepted")
    responses = {
        key: dict(value)
        for ids, values in (
            (foreground_ids, raw_foregrounds),
            (accompaniment_ids, raw_accompaniments),
        )
        for key, value in zip(ids, values, strict=True)
    }
    unit_by_placement = {key: f"score-unit-{placements[key]['section_id']}" for key in placements}
    existing_by_unit: dict[str, list[ScoreNote]] = {}
    for key, notes in accepted_notes_by_placement.items():
        existing_by_unit.setdefault(unit_by_placement[key], []).extend(notes)
    candidate_notes: dict[str, tuple[ScoreNote, ...]] = {}
    for index, key in enumerate(foreground_ids):
        unit_id = unit_by_placement[key]
        issues = check_foreground_response(
            responses[key],
            length_units=harmonic_plan.length_units_by_score_unit[unit_id],
            existing_unit_notes=tuple(existing_by_unit.get(unit_id, ())),
        )
        if issues:
            return ScoreGroupCandidateResult(_prefixed_issues(issues, f"/foregrounds/{index}"))
        candidate_notes[key] = build_foreground_notes(key, responses[key])
        existing_by_unit.setdefault(unit_id, []).extend(candidate_notes[key])
    boundary = None
    if shared_terminal:
        try:
            boundary = available_terminal_boundary(
                document,
                harmonic_plan.piece_plan,
                harmonies_by_score_unit,
                {**accepted_notes_by_placement, **candidate_notes},
            )
        except TerminalBoundaryError as error:
            return ScoreGroupCandidateResult(
                (ValidationIssue(IssueCode.MODEL_OUTPUT_INVALID, str(error), "/foregrounds"),)
            )
    for index, key in enumerate(accompaniment_ids):
        unit_id = unit_by_placement[key]
        issues = check_accompaniment_response(
            responses[key],
            length_units=harmonic_plan.length_units_by_score_unit[unit_id],
            harmonies=harmonies_by_score_unit[unit_id],
        )
        if issues:
            return ScoreGroupCandidateResult(_prefixed_issues(issues, f"/accompaniments/{index}"))
    try:
        placed = place_score_group_accompaniment_events(
            {key: responses[key] for key in accompaniment_ids},
            harmonies_by_score_unit,
            unit_by_placement,
            existing_notes_by_score_unit={
                key: tuple(values) for key, values in existing_by_unit.items()
            },
            search_limit=search_limit,
        )
    except ScoreGroupPitchPlacementError as error:
        return ScoreGroupCandidateResult(
            (
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    f"{error.code}: {error}; score_unit_id={error.score_unit_id}",
                    "/accompaniments",
                ),
            )
        )
    candidate_notes.update(placed.notes_by_placement)
    combined = {**accepted_notes_by_placement, **candidate_notes}
    if shared_terminal:
        try:
            boundary = available_terminal_boundary(
                document, harmonic_plan.piece_plan, harmonies_by_score_unit, combined
            )
        except TerminalBoundaryError as error:
            return ScoreGroupCandidateResult(
                (ValidationIssue(IssueCode.MODEL_OUTPUT_INVALID, str(error), "/accompaniments"),)
            )
    relevant = _comparisons(operation, comparisons)
    section_targets = {
        key
        for item in relevant
        if item.kind == "section_variation"
        for key in item.target_placement_ids
    }
    evidence: list[dict[str, object]] = []
    issues: list[ValidationIssue] = []
    for comparison in relevant:
        target_ids = tuple(key for key in comparison.target_placement_ids if key in owned)
        if any(key not in combined for key in comparison.source_placement_ids):
            raise ValueError("a score comparison source is missing from accepted or joint values")
        check_copy = not comparison.kind.startswith("transition_") and not (
            comparison.kind == "implicit_reuse" and set(target_ids) <= section_targets
        )
        if comparison.kind == "section_variation":
            if set(target_ids) != set(comparison.target_placement_ids):
                raise ValueError("a section variation target must be owned by one atomic group")
            if comparison.source_section_id is None or comparison.target_section_id is None:
                raise ValueError("a section comparison requires both section IDs")
            source_values = _section_fingerprint(
                document, harmonic_plan, comparison.source_section_id, combined
            )
            target_values = _section_fingerprint(
                document, harmonic_plan, comparison.target_section_id, combined
            )
            exact_copy = source_values == target_values
        else:
            source_values = {
                key: sorted(_fingerprint(combined[key])) for key in comparison.source_placement_ids
            }
            target_values = {key: sorted(_fingerprint(combined[key])) for key in target_ids}
            exact_copy = any(
                source_values[source] == target_values[target]
                for source in comparison.source_placement_ids
                for target in target_ids
            )
        if check_copy and exact_copy:
            issues.append(
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "A varied or reused score must not be an exact copy: "
                    f"kind={comparison.kind}, relation_id={comparison.relation_id}; "
                    f"source={json.dumps(source_values, ensure_ascii=False)}; "
                    f"target={json.dumps(target_values, ensure_ascii=False)}",
                    "/foregrounds" if not accompaniment_ids else "/accompaniments",
                )
            )
        evidence.append(
            {
                "comparison_kind": comparison.kind,
                "relation_id": comparison.relation_id,
                "source_placement_ids": list(comparison.source_placement_ids),
                "target_placement_ids": list(target_ids),
                "source_section_id": comparison.source_section_id,
                "target_section_id": comparison.target_section_id,
                "description": comparison.description,
                "preserve": list(comparison.preserve),
                "change": list(comparison.change),
                "external_source_placement_ids": [
                    key for key in comparison.source_placement_ids if key not in owned
                ],
                "internal_source_placement_ids": [
                    key for key in comparison.source_placement_ids if key in owned
                ],
                "source_notes_sha256": sha256_json(source_values),
                "target_notes_sha256": sha256_json(target_values),
                "copy_check": "failed"
                if check_copy and exact_copy
                else ("passed" if check_copy else "not_required"),
                "semantic_status": "unverified",
            }
        )
    if issues:
        first = issues[0]
        issues[0] = ValidationIssue(
            first.code,
            first.message
            + "; realized_joint_candidate="
            + json.dumps(
                {key: _note_values(notes) for key, notes in candidate_notes.items()},
                ensure_ascii=False,
                sort_keys=True,
            ),
            first.path,
        )
        return ScoreGroupCandidateResult(
            issues=tuple(issues),
            notes_by_placement=candidate_notes,
            responses_by_placement=responses,
            comparison_evidence=tuple(evidence),
            evaluated_candidate_count_by_score_unit=placed.evaluated_candidate_count_by_score_unit,
        )
    ledger = _projection_entries(
        operation,
        placements,
        candidate_notes,
        evidence,
        placed.evaluated_candidate_count_by_score_unit,
    )
    return ScoreGroupCandidateResult(
        notes_by_placement=candidate_notes,
        responses_by_placement=responses,
        projection_ledger=ledger,
        terminal_boundary=boundary,
        comparison_evidence=tuple(evidence),
        evaluated_candidate_count_by_score_unit=placed.evaluated_candidate_count_by_score_unit,
    )


def _projection_entries(
    operation: ScoreWorkOperation,
    placements: Mapping[str, Mapping[str, object]],
    notes_by_placement: Mapping[str, tuple[ScoreNote, ...]],
    evidence: list[dict[str, object]],
    counts: Mapping[str, int],
) -> tuple[ProjectionLedgerEntry, ...]:
    entries: list[ProjectionLedgerEntry] = []
    for key in operation.material_placement_ids:
        layer_id = f"score-unit-layer-{key}"
        role = cast(str, placements[key]["role"])
        unit_id = f"score-unit-{placements[key]['section_id']}"
        entries.append(
            ProjectionLedgerEntry(
                "material_placement",
                key,
                "score_unit_layer",
                layer_id,
                role,
                "direct_id_equality",
                "passed",
                f"material_placement_id={key}; score_unit_layer_id={layer_id}",
            )
        )
        for note in notes_by_placement[key]:
            entries.append(
                ProjectionLedgerEntry(
                    "score_unit_layer",
                    layer_id,
                    "score_note",
                    note.score_note_id,
                    role,
                    "time_bounds_and_source_equality"
                    if role == "foreground"
                    else "joint_harmony_register_placement",
                    "passed",
                    f"score_unit_layer_id={layer_id}; at_units={note.at_units}; "
                    f"duration_units={note.duration_units}; "
                    f"evaluated_candidates={counts.get(unit_id, 0)}",
                )
            )
    for index, item in enumerate(evidence):
        target_id = f"{operation.operation_id}-comparison-{index:03d}"
        source_kind = "script_element_variation_relation"
        if cast(str, item["comparison_kind"]).startswith("transition_"):
            source_kind = "material_placement_transition"
        elif item["relation_id"] is None:
            source_kind = "material_placement"
        source_id = (
            cast(str | None, item["relation_id"])
            or cast(list[str], item["source_placement_ids"])[0]
        )
        entries.append(
            ProjectionLedgerEntry(
                source_kind,
                source_id,
                "score_operation_comparison",
                target_id,
                "score",
                "comparison_scope_and_candidate_equality",
                "passed",
                json.dumps(item, ensure_ascii=False, sort_keys=True),
            )
        )
        entries.append(
            ProjectionLedgerEntry(
                source_kind,
                source_id,
                "score_operation_creative_goal",
                target_id,
                "score",
                "natural_language_claim",
                "unverified",
                json.dumps(
                    {key: item[key] for key in ("description", "preserve", "change")},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
        )
    result = tuple(entries)
    validate_projection_ledger(result)
    return result
