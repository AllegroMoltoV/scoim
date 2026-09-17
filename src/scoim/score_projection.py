"""Deterministic projection from validated scripts into the v2 score boundary."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from typing import cast

from .generation_script_validation import check_generation_script_document
from .profile_capabilities import (
    check_generation_profile_document,
    solo_piano_3m_v2_capabilities,
)
from .projection_ledger import (
    ProjectionLedgerEntry,
    validate_expected_projection_targets,
    validate_projection_ledger,
)
from .score_ir import (
    PiecePlan,
    PlanNode,
    ScoreDirection,
    ScoreHarmony,
    ScoreNote,
    ScoreSpec,
    ScoreUnit,
    ScoreUnitLayer,
    validate_piece_plan,
    validate_score_ir,
)


class ScoreProjectionError(ValueError):
    """A validated source cannot be projected without losing its contract."""


@dataclass(frozen=True, slots=True)
class PlanChoice:
    tonal_center: int
    mode: str


def build_piece_plan(
    document: Mapping[str, object], choice: PlanChoice
) -> tuple[PiecePlan, tuple[ProjectionLedgerEntry, ...]]:
    """Build the phase-3 structural plan without inventing musical content."""
    validation = check_generation_script_document(document)
    if not validation.valid:
        raise ScoreProjectionError("the generation script must pass validation before projection")
    if isinstance(choice.tonal_center, bool) or not 0 <= choice.tonal_center <= 11:
        raise ScoreProjectionError("tonal center must be a pitch class from 0 to 11")
    if choice.mode not in {"major", "minor"}:
        raise ScoreProjectionError("mode must be major or minor")

    script = cast(dict[str, object], document["script"])
    capabilities = solo_piano_3m_v2_capabilities()
    profile_check = check_generation_profile_document(document, capabilities)
    if not profile_check.valid:
        raise ScoreProjectionError(profile_check.issues[0].message)
    sections = cast(dict[str, dict[str, object]], script["sections"])
    children: dict[str, list[str]] = {}
    for section_id, section in sections.items():
        parent_id = section["parent_section_id"]
        if isinstance(parent_id, str):
            children.setdefault(parent_id, []).append(section_id)
    for child_ids in children.values():
        child_ids.sort(key=lambda section_id: cast(int, sections[section_id]["order"]))

    ordered_section_ids: list[str] = []

    def visit(section_id: str) -> None:
        ordered_section_ids.append(section_id)
        for child_id in children.get(section_id, ()):
            visit(child_id)

    root_section_id = cast(str, script["root_section_id"])
    visit(root_section_id)
    leaf_ids = [section_id for section_id in ordered_section_ids if section_id not in children]
    relative_lengths = {
        section_id: Fraction(str(sections[section_id]["relative_length"]))
        for section_id in leaf_ids
    }
    denominator = math.lcm(*(length.denominator for length in relative_lengths.values()))
    unscaled = {
        section_id: length.numerator * (denominator // length.denominator)
        for section_id, length in relative_lengths.items()
    }
    divisor = reduce(math.gcd, unscaled.values())
    duration_weights = {section_id: value // divisor for section_id, value in unscaled.items()}
    nodes = tuple(
        PlanNode(
            section_id=section_id,
            parent_section_id=cast(str | None, sections[section_id]["parent_section_id"]),
            order=cast(int, sections[section_id]["order"]),
            duration_weight=duration_weights.get(section_id),
            score_unit_id=(f"score-unit-{section_id}" if section_id in duration_weights else None),
        )
        for section_id in ordered_section_ids
    )
    plan = PiecePlan(
        plan_id=cast(str, document["document_id"]),
        title=cast(str, script["title"]),
        tonal_center=choice.tonal_center,
        mode=choice.mode,
        root_section_id=root_section_id,
        nodes=nodes,
    )
    validate_piece_plan(plan)
    ledger = tuple(
        ProjectionLedgerEntry(
            source_kind="section",
            source_id=node.section_id,
            target_kind="plan_node",
            target_id=node.section_id,
            target_stage="piece_plan",
            verification="direct_id_equality",
            status="passed",
            evidence=f"section_id={node.section_id}",
        )
        for node in nodes
    )
    validate_projection_ledger(ledger)
    return plan, ledger


def build_score_spec(
    document: Mapping[str, object],
    plan: PiecePlan,
    *,
    score_id: str,
    divisions: int,
    length_units_by_score_unit: Mapping[str, int],
    harmonies_by_score_unit: Mapping[str, tuple[ScoreHarmony, ...]],
    directions_by_score_unit: Mapping[str, tuple[ScoreDirection, ...]],
    notes_by_material_placement: Mapping[str, tuple[ScoreNote, ...]],
    cumulative_projection_ledger: Sequence[ProjectionLedgerEntry],
) -> tuple[ScoreSpec, tuple[ProjectionLedgerEntry, ...]]:
    """Build phase-6 score data from already validated stage values."""
    validation = check_generation_script_document(document)
    if not validation.valid:
        raise ScoreProjectionError("the generation script must pass validation before projection")
    validate_piece_plan(plan)
    expected_plan, _ = build_piece_plan(
        document,
        PlanChoice(tonal_center=plan.tonal_center, mode=plan.mode),
    )
    if plan != expected_plan:
        raise ScoreProjectionError("piece plan does not match the script source content")
    script = cast(dict[str, object], document["script"])
    placements = cast(dict[str, dict[str, object]], script["material_placements"])
    if set(notes_by_material_placement) != set(placements):
        raise ScoreProjectionError("placement values must exactly cover material placements")
    expected_unit_ids = {
        node.score_unit_id for node in plan.nodes if node.score_unit_id is not None
    }
    if any(
        set(values) != expected_unit_ids
        for values in (
            length_units_by_score_unit,
            harmonies_by_score_unit,
            directions_by_score_unit,
        )
    ):
        raise ScoreProjectionError("score-unit values must exactly cover the planned units")
    placement_ids_by_section: dict[str, list[str]] = {}
    for placement_id, placement in placements.items():
        section_id = cast(str, placement["section_id"])
        placement_ids_by_section.setdefault(section_id, []).append(placement_id)
    for placement_ids in placement_ids_by_section.values():
        placement_ids.sort()

    existing_ledger = tuple(cumulative_projection_ledger)
    validate_projection_ledger(existing_ledger)
    validate_expected_projection_targets(
        existing_ledger,
        {
            "plan_node": frozenset(node.section_id for node in plan.nodes),
            "score_unit": frozenset(cast(str, unit_id) for unit_id in expected_unit_ids),
            "score_unit_layer": frozenset(
                f"score-unit-layer-{placement_id}" for placement_id in placements
            ),
            "score_note": frozenset(
                note.score_note_id
                for notes in notes_by_material_placement.values()
                for note in notes
            ),
        },
    )

    score_units: list[ScoreUnit] = []
    ledger: list[ProjectionLedgerEntry] = []
    for node in plan.nodes:
        if node.score_unit_id is None:
            continue
        unit_id = node.score_unit_id
        layers = tuple(
            ScoreUnitLayer(
                score_unit_layer_id=f"score-unit-layer-{placement_id}",
                source_material_placement_id=placement_id,
                notes=notes_by_material_placement[placement_id],
            )
            for placement_id in placement_ids_by_section.get(node.section_id, ())
        )
        score_units.append(
            ScoreUnit(
                score_unit_id=unit_id,
                source_section_id=node.section_id,
                length_units=length_units_by_score_unit[unit_id],
                harmonies=harmonies_by_score_unit[unit_id],
                directions=directions_by_score_unit[unit_id],
                score_unit_layers=layers,
            )
        )
    score = ScoreSpec(score_id, divisions, tuple(score_units))
    validate_score_ir(plan, score)
    relations = cast(dict[str, dict[str, object]], script["script_element_variation_relations"])
    for relation_id, relation in sorted(relations.items()):
        source = cast(dict[str, object], relation["source"])
        target = cast(dict[str, object], relation["target"])
        source_text = f"{source['type']}:{source['id']}"
        target_text = f"{target['type']}:{target['id']}"
        ledger.append(
            ProjectionLedgerEntry(
                source_kind="script_element_variation_relation",
                source_id=relation_id,
                target_kind="score_generation_constraint",
                target_id=relation_id,
                target_stage="score_spec",
                verification="typed_relation_equality",
                status="passed",
                evidence=f"{source_text}->{target_text}",
            )
        )
    transitions = cast(dict[str, dict[str, object]], script["material_placement_transitions"])
    for transition_id, transition in sorted(transitions.items()):
        source_id = cast(str, transition["source_material_placement_id"])
        transition_placement_id = cast(str, transition["transition_material_placement_id"])
        target_id = cast(str, transition["target_material_placement_id"])
        ledger.append(
            ProjectionLedgerEntry(
                source_kind="material_placement_transition",
                source_id=transition_id,
                target_kind="score_generation_constraint",
                target_id=transition_id,
                target_stage="score_spec",
                verification="typed_relation_equality",
                status="passed",
                evidence=f"{source_id}->{transition_placement_id}->{target_id}",
            )
        )
    completed_ledger = tuple(ledger)
    validate_projection_ledger(completed_ledger)
    validate_projection_ledger((*existing_ledger, *completed_ledger))
    validate_expected_projection_targets(
        (*existing_ledger, *completed_ledger),
        {"score_generation_constraint": frozenset((*relations.keys(), *transitions.keys()))},
    )
    return score, completed_ledger
