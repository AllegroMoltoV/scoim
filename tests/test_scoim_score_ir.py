from dataclasses import fields

import pytest

from scoim.score_ir import (
    PiecePlan,
    PlanNode,
    ScoreHarmony,
    ScoreIrValidationError,
    ScoreNote,
    ScoreSpec,
    ScoreUnit,
    ScoreUnitLayer,
    validate_score_ir,
)


def _valid_ir() -> tuple[PiecePlan, ScoreSpec]:
    plan = PiecePlan(
        plan_id="plan-001",
        title="小さな曲",
        tonal_center=0,
        mode="major",
        root_section_id="whole",
        nodes=(
            PlanNode("whole", None, 0),
            PlanNode("statement", "whole", 0, duration_weight=1, score_unit_id="unit-001"),
        ),
    )
    score = ScoreSpec(
        score_id="score-001",
        divisions=12,
        score_units=(
            ScoreUnit(
                score_unit_id="unit-001",
                source_section_id="statement",
                length_units=48,
                harmonies=(ScoreHarmony("harmony-001", 0, 48, 0, "major"),),
                directions=(),
                score_unit_layers=(
                    ScoreUnitLayer(
                        score_unit_layer_id="layer-theme",
                        source_material_placement_id="theme-first",
                        notes=(
                            ScoreNote("note-upper", 0, 12, 72, "upper"),
                            ScoreNote("note-lower", 12, 12, 48, "lower"),
                        ),
                    ),
                    ScoreUnitLayer(
                        score_unit_layer_id="layer-support",
                        source_material_placement_id="support-first",
                        notes=(ScoreNote("note-support", 6, 36, 43, "lower"),),
                    ),
                ),
            ),
        ),
    )
    return plan, score


def test_score_ir_allows_two_voices_and_different_layer_spans_in_one_unit() -> None:
    plan, score = _valid_ir()

    validate_score_ir(plan, score)


def test_score_unit_layer_stores_only_its_adjacent_source_reference() -> None:
    assert {field.name for field in fields(ScoreUnitLayer)} == {
        "score_unit_layer_id",
        "source_material_placement_id",
        "notes",
    }


def test_score_ir_rejects_a_score_unit_on_a_branch_node() -> None:
    plan, score = _valid_ir()
    invalid_plan = PiecePlan(
        plan_id=plan.plan_id,
        title=plan.title,
        tonal_center=plan.tonal_center,
        mode=plan.mode,
        root_section_id=plan.root_section_id,
        nodes=(
            PlanNode("whole", None, 0, duration_weight=1, score_unit_id="unit-root"),
            plan.nodes[1],
        ),
    )

    with pytest.raises(ScoreIrValidationError, match="branch node"):
        validate_score_ir(invalid_plan, score)


def test_score_ir_requires_each_leaf_to_own_one_score_unit() -> None:
    plan, score = _valid_ir()
    invalid_plan = PiecePlan(
        plan_id=plan.plan_id,
        title=plan.title,
        tonal_center=plan.tonal_center,
        mode=plan.mode,
        root_section_id=plan.root_section_id,
        nodes=(plan.nodes[0], PlanNode("statement", "whole", 0, duration_weight=1)),
    )

    with pytest.raises(ScoreIrValidationError, match="leaf node"):
        validate_score_ir(invalid_plan, score)


def test_score_ir_requires_a_positive_leaf_duration_weight() -> None:
    plan, score = _valid_ir()
    invalid_plan = PiecePlan(
        plan_id=plan.plan_id,
        title=plan.title,
        tonal_center=plan.tonal_center,
        mode=plan.mode,
        root_section_id=plan.root_section_id,
        nodes=(
            plan.nodes[0],
            PlanNode("statement", "whole", 0, duration_weight=0, score_unit_id="unit-001"),
        ),
    )

    with pytest.raises(ScoreIrValidationError, match="positive duration"):
        validate_score_ir(invalid_plan, score)


def test_score_ir_rejects_duplicate_section_nodes() -> None:
    plan, score = _valid_ir()
    invalid_plan = PiecePlan(
        plan_id=plan.plan_id,
        title=plan.title,
        tonal_center=plan.tonal_center,
        mode=plan.mode,
        root_section_id=plan.root_section_id,
        nodes=(*plan.nodes, plan.nodes[1]),
    )

    with pytest.raises(ScoreIrValidationError, match="section ID"):
        validate_score_ir(invalid_plan, score)


def test_score_ir_requires_the_designated_root_node() -> None:
    plan, score = _valid_ir()
    invalid_plan = PiecePlan(
        plan_id=plan.plan_id,
        title=plan.title,
        tonal_center=plan.tonal_center,
        mode=plan.mode,
        root_section_id="missing",
        nodes=plan.nodes,
    )

    with pytest.raises(ScoreIrValidationError, match="root section"):
        validate_score_ir(invalid_plan, score)


def test_score_ir_requires_exactly_one_parentless_root() -> None:
    plan, score = _valid_ir()
    invalid_plan = PiecePlan(
        plan_id=plan.plan_id,
        title=plan.title,
        tonal_center=plan.tonal_center,
        mode=plan.mode,
        root_section_id=plan.root_section_id,
        nodes=(plan.nodes[0], PlanNode("statement", None, 0, 1, "unit-001")),
    )

    with pytest.raises(ScoreIrValidationError, match="parentless root"):
        validate_score_ir(invalid_plan, score)


def test_score_ir_rejects_a_missing_parent_section() -> None:
    plan, score = _valid_ir()
    invalid_plan = PiecePlan(
        plan_id=plan.plan_id,
        title=plan.title,
        tonal_center=plan.tonal_center,
        mode=plan.mode,
        root_section_id=plan.root_section_id,
        nodes=(
            plan.nodes[0],
            PlanNode("statement", "missing", 0, duration_weight=1, score_unit_id="unit-001"),
        ),
    )

    with pytest.raises(ScoreIrValidationError, match="parent section"):
        validate_score_ir(invalid_plan, score)


def test_score_ir_requires_consecutive_sibling_order() -> None:
    plan, score = _valid_ir()
    invalid_plan = PiecePlan(
        plan_id=plan.plan_id,
        title=plan.title,
        tonal_center=plan.tonal_center,
        mode=plan.mode,
        root_section_id=plan.root_section_id,
        nodes=(
            *plan.nodes,
            PlanNode("return", "whole", 2, duration_weight=1, score_unit_id="unit-002"),
        ),
    )

    with pytest.raises(ScoreIrValidationError, match="sibling order"):
        validate_score_ir(invalid_plan, score)


def test_score_ir_rejects_a_disconnected_section_cycle() -> None:
    plan, score = _valid_ir()
    invalid_plan = PiecePlan(
        plan_id=plan.plan_id,
        title=plan.title,
        tonal_center=plan.tonal_center,
        mode=plan.mode,
        root_section_id=plan.root_section_id,
        nodes=(
            *plan.nodes,
            PlanNode("cycle-a", "cycle-b", 0),
            PlanNode("cycle-b", "cycle-a", 0),
        ),
    )

    with pytest.raises(ScoreIrValidationError, match="reachable"):
        validate_score_ir(invalid_plan, score)


def test_score_ir_requires_one_score_unit_for_each_leaf() -> None:
    plan, score = _valid_ir()
    invalid_score = ScoreSpec(score.score_id, score.divisions, ())

    with pytest.raises(ScoreIrValidationError, match="score units must match"):
        validate_score_ir(plan, invalid_score)


def test_score_ir_rejects_one_score_unit_shared_by_two_leaf_sections() -> None:
    plan, score = _valid_ir()
    invalid_plan = PiecePlan(
        plan.plan_id,
        plan.title,
        plan.tonal_center,
        plan.mode,
        plan.root_section_id,
        (
            *plan.nodes,
            PlanNode("return", "whole", 1, duration_weight=1, score_unit_id="unit-001"),
        ),
    )

    with pytest.raises(ScoreIrValidationError, match="one score unit"):
        validate_score_ir(invalid_plan, score)


def test_score_ir_requires_at_least_one_layer_per_score_unit() -> None:
    plan, score = _valid_ir()
    unit = score.score_units[0]
    invalid_score = ScoreSpec(
        score.score_id,
        score.divisions,
        (
            ScoreUnit(
                unit.score_unit_id,
                unit.source_section_id,
                unit.length_units,
                unit.harmonies,
                unit.directions,
                (),
            ),
        ),
    )

    with pytest.raises(ScoreIrValidationError, match="at least one layer"):
        validate_score_ir(plan, invalid_score)


def test_score_ir_rejects_two_layers_for_one_material_placement() -> None:
    plan, score = _valid_ir()
    unit = score.score_units[0]
    duplicate = ScoreUnitLayer(
        "layer-duplicate",
        unit.score_unit_layers[0].source_material_placement_id,
        (ScoreNote("note-duplicate", 24, 12, 74, "upper"),),
    )
    invalid_score = ScoreSpec(
        score.score_id,
        score.divisions,
        (
            ScoreUnit(
                unit.score_unit_id,
                unit.source_section_id,
                unit.length_units,
                unit.harmonies,
                unit.directions,
                (*unit.score_unit_layers, duplicate),
            ),
        ),
    )

    with pytest.raises(ScoreIrValidationError, match="material placement"):
        validate_score_ir(plan, invalid_score)


def test_score_ir_rejects_a_note_id_reused_by_another_layer() -> None:
    plan, score = _valid_ir()
    unit = score.score_units[0]
    support = unit.score_unit_layers[1]
    duplicate_note_layer = ScoreUnitLayer(
        support.score_unit_layer_id,
        support.source_material_placement_id,
        (ScoreNote("note-upper", 6, 36, 43, "lower"),),
    )
    invalid_score = ScoreSpec(
        score.score_id,
        score.divisions,
        (
            ScoreUnit(
                unit.score_unit_id,
                unit.source_section_id,
                unit.length_units,
                unit.harmonies,
                unit.directions,
                (unit.score_unit_layers[0], duplicate_note_layer),
            ),
        ),
    )

    with pytest.raises(ScoreIrValidationError, match="score note ID"):
        validate_score_ir(plan, invalid_score)


def test_score_ir_requires_at_least_one_note_per_layer() -> None:
    plan, score = _valid_ir()
    unit = score.score_units[0]
    theme = unit.score_unit_layers[0]
    empty_theme = ScoreUnitLayer(
        theme.score_unit_layer_id,
        theme.source_material_placement_id,
        (),
    )
    invalid_score = ScoreSpec(
        score.score_id,
        score.divisions,
        (
            ScoreUnit(
                unit.score_unit_id,
                unit.source_section_id,
                unit.length_units,
                unit.harmonies,
                unit.directions,
                (empty_theme, unit.score_unit_layers[1]),
            ),
        ),
    )

    with pytest.raises(ScoreIrValidationError, match="at least one note"):
        validate_score_ir(plan, invalid_score)


def test_score_ir_requires_globally_unique_layer_ids() -> None:
    plan, score = _valid_ir()
    unit = score.score_units[0]
    support = unit.score_unit_layers[1]
    duplicate_id_layer = ScoreUnitLayer(
        unit.score_unit_layers[0].score_unit_layer_id,
        support.source_material_placement_id,
        support.notes,
    )
    invalid_score = ScoreSpec(
        score.score_id,
        score.divisions,
        (
            ScoreUnit(
                unit.score_unit_id,
                unit.source_section_id,
                unit.length_units,
                unit.harmonies,
                unit.directions,
                (unit.score_unit_layers[0], duplicate_id_layer),
            ),
        ),
    )

    with pytest.raises(ScoreIrValidationError, match="layer IDs"):
        validate_score_ir(plan, invalid_score)


def test_score_ir_rejects_a_note_outside_the_two_piano_voices() -> None:
    plan, score = _valid_ir()
    unit = score.score_units[0]
    theme = unit.score_unit_layers[0]
    invalid_theme = ScoreUnitLayer(
        theme.score_unit_layer_id,
        theme.source_material_placement_id,
        (ScoreNote("note-invalid", 0, 12, 72, "middle"),),
    )
    invalid_score = ScoreSpec(
        score.score_id,
        score.divisions,
        (
            ScoreUnit(
                unit.score_unit_id,
                unit.source_section_id,
                unit.length_units,
                unit.harmonies,
                unit.directions,
                (invalid_theme, unit.score_unit_layers[1]),
            ),
        ),
    )

    with pytest.raises(ScoreIrValidationError, match="upper or lower"):
        validate_score_ir(plan, invalid_score)


def test_score_ir_rejects_a_note_past_the_score_unit_end() -> None:
    plan, score = _valid_ir()
    unit = score.score_units[0]
    theme = unit.score_unit_layers[0]
    invalid_theme = ScoreUnitLayer(
        theme.score_unit_layer_id,
        theme.source_material_placement_id,
        (ScoreNote("note-too-long", 40, 12, 72, "upper"),),
    )
    invalid_score = ScoreSpec(
        score.score_id,
        score.divisions,
        (
            ScoreUnit(
                unit.score_unit_id,
                unit.source_section_id,
                unit.length_units,
                unit.harmonies,
                unit.directions,
                (invalid_theme, unit.score_unit_layers[1]),
            ),
        ),
    )

    with pytest.raises(ScoreIrValidationError, match="score unit bounds"):
        validate_score_ir(plan, invalid_score)


def test_score_ir_requires_harmony_to_cover_the_whole_score_unit() -> None:
    plan, score = _valid_ir()
    unit = score.score_units[0]
    invalid_score = ScoreSpec(
        score.score_id,
        score.divisions,
        (
            ScoreUnit(
                unit.score_unit_id,
                unit.source_section_id,
                unit.length_units,
                (ScoreHarmony("harmony-short", 0, 24, 0, "major"),),
                unit.directions,
                unit.score_unit_layers,
            ),
        ),
    )

    with pytest.raises(ScoreIrValidationError, match="harmony must cover"):
        validate_score_ir(plan, invalid_score)


def test_score_ir_requires_positive_time_units() -> None:
    plan, score = _valid_ir()
    invalid_score = ScoreSpec(score.score_id, 0, score.score_units)

    with pytest.raises(ScoreIrValidationError, match="positive divisions"):
        validate_score_ir(plan, invalid_score)


def test_score_ir_requires_a_positive_score_unit_length() -> None:
    plan, score = _valid_ir()
    unit = score.score_units[0]
    invalid_score = ScoreSpec(
        score.score_id,
        score.divisions,
        (
            ScoreUnit(
                unit.score_unit_id,
                unit.source_section_id,
                0,
                (),
                unit.directions,
                unit.score_unit_layers,
            ),
        ),
    )

    with pytest.raises(ScoreIrValidationError, match="positive length"):
        validate_score_ir(plan, invalid_score)
