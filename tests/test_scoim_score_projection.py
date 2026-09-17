from dataclasses import replace

import pytest

from scoim.projection_ledger import ProjectionLedgerEntry, ProjectionLedgerValidationError
from scoim.score_ir import ScoreHarmony, ScoreNote
from scoim.score_projection import (
    PlanChoice,
    ScoreProjectionError,
    build_piece_plan,
    build_score_spec,
)


def _validated_script() -> dict[str, object]:
    return {
        "document_type": "script",
        "schema_version": "0.3.0",
        "document_id": "composition-001",
        "revision": 1,
        "status": "validated",
        "source_flow": {
            "document_id": "flow-001",
            "revision": 1,
            "content_sha256": "a" * 64,
        },
        "script": {
            "title": "小さな曲",
            "brief": "前景と伴奏を併用する。",
            "performance_setup": {
                "instrumentation": "solo_piano",
                "target_duration_seconds": 180,
                "performance_directions": {},
            },
            "root_section_id": "whole",
            "sections": {
                "whole": {
                    "parent_section_id": None,
                    "order": 0,
                    "role": "whole",
                    "description": "曲全体",
                },
                "statement": {
                    "parent_section_id": "whole",
                    "order": 0,
                    "role": "statement",
                    "relative_length": 1,
                    "description": "主題を提示する。",
                },
            },
            "materials": {
                "theme": {"description": "中心となる短い動き。"},
                "support": {"description": "主題を支える動き。"},
            },
            "material_placements": {
                "theme-first": {
                    "section_id": "statement",
                    "material_id": "theme",
                    "role": "foreground",
                },
                "support-first": {
                    "section_id": "statement",
                    "material_id": "support",
                    "role": "accompaniment",
                },
            },
            "script_element_variation_relations": {},
            "material_placement_transitions": {},
            "performance_direction_comparison_requirements": {},
        },
    }


def _with_return_variation(document: dict[str, object]) -> dict[str, object]:
    document["script"]["sections"]["return"] = {
        "parent_section_id": "whole",
        "order": 1,
        "role": "return",
        "relative_length": 1,
        "description": "主題を再提示する。",
    }
    document["script"]["material_placements"]["theme-return"] = {
        "section_id": "return",
        "material_id": "theme",
        "role": "foreground",
    }
    document["script"]["script_element_variation_relations"]["theme-varied"] = {
        "source": {"type": "material_placement", "id": "theme-first"},
        "target": {"type": "material_placement", "id": "theme-return"},
        "preserve": ["冒頭の輪郭"],
        "change": ["リズム"],
        "description": "再提示ではリズムを変える。",
    }
    return document


def _with_transition(document: dict[str, object]) -> dict[str, object]:
    sections = document["script"]["sections"]
    sections["bridge"] = {
        "parent_section_id": "whole",
        "order": 1,
        "role": "transition",
        "relative_length": 0.25,
        "description": "再提示へ移る。",
    }
    sections["return"] = {
        "parent_section_id": "whole",
        "order": 2,
        "role": "return",
        "relative_length": 1,
        "description": "主題を再提示する。",
    }
    document["script"]["materials"]["bridge"] = {"description": "短い遷移。"}
    placements = document["script"]["material_placements"]
    placements["bridge-only"] = {
        "section_id": "bridge",
        "material_id": "bridge",
        "role": "foreground",
    }
    placements["theme-return"] = {
        "section_id": "return",
        "material_id": "theme",
        "role": "foreground",
    }
    document["script"]["material_placement_transitions"]["to-return"] = {
        "source_material_placement_id": "theme-first",
        "transition_material_placement_id": "bridge-only",
        "target_material_placement_id": "theme-return",
        "description": "主題から再提示へ移る。",
    }
    return document


def _upstream_score_ledger(plan, notes_by_placement):
    unit_ids = tuple(node.score_unit_id for node in plan.nodes if node.score_unit_id is not None)
    targets = {
        "plan_node": tuple(node.section_id for node in plan.nodes),
        "score_unit": unit_ids,
        "score_unit_layer": tuple(
            f"score-unit-layer-{placement_id}" for placement_id in notes_by_placement
        ),
        "score_note": tuple(
            note.score_note_id for notes in notes_by_placement.values() for note in notes
        ),
    }
    return tuple(
        ProjectionLedgerEntry(
            "phase",
            target_id,
            target_kind,
            target_id,
            "upstream",
            "direct_id_equality",
            "passed",
            f"{target_kind}_id={target_id}",
        )
        for target_kind, target_ids in targets.items()
        for target_id in target_ids
    )


def test_build_piece_plan_maps_each_section_and_assigns_one_unit_to_each_leaf() -> None:
    plan, ledger = build_piece_plan(
        _validated_script(),
        PlanChoice(tonal_center=0, mode="major"),
    )

    assert [(node.section_id, node.score_unit_id) for node in plan.nodes] == [
        ("whole", None),
        ("statement", "score-unit-statement"),
    ]
    assert {(entry.source_kind, entry.source_id, entry.target_kind) for entry in ledger} == {
        ("section", "whole", "plan_node"),
        ("section", "statement", "plan_node"),
    }
    assert all(entry.status == "passed" and entry.evidence for entry in ledger)


def test_build_score_spec_maps_each_placement_to_one_separate_layer() -> None:
    document = _validated_script()
    plan, _ = build_piece_plan(document, PlanChoice(tonal_center=0, mode="major"))
    notes_by_placement = {
        "theme-first": (ScoreNote("note-theme", 0, 24, 72, "upper"),),
        "support-first": (ScoreNote("note-support", 0, 48, 48, "lower"),),
    }

    score, ledger = build_score_spec(
        document,
        plan,
        score_id="score-001",
        divisions=12,
        length_units_by_score_unit={"score-unit-statement": 48},
        harmonies_by_score_unit={
            "score-unit-statement": (ScoreHarmony("harmony-001", 0, 48, 0, "major"),)
        },
        directions_by_score_unit={"score-unit-statement": ()},
        notes_by_material_placement=notes_by_placement,
        cumulative_projection_ledger=_upstream_score_ledger(plan, notes_by_placement),
    )

    layers = score.score_units[0].score_unit_layers
    assert [layer.source_material_placement_id for layer in layers] == [
        "support-first",
        "theme-first",
    ]
    assert ledger == ()


def test_build_piece_plan_rejects_a_role_not_supported_by_the_profile() -> None:
    document = _validated_script()
    document["script"]["material_placements"]["theme-first"]["role"] = "counterline"

    with pytest.raises(ScoreProjectionError, match="material placement role"):
        build_piece_plan(document, PlanChoice(tonal_center=0, mode="major"))


def test_build_piece_plan_requires_a_material_placement_in_each_leaf() -> None:
    document = _validated_script()
    document["script"]["material_placements"] = {}

    with pytest.raises(ScoreProjectionError, match="placement count"):
        build_piece_plan(document, PlanChoice(tonal_center=0, mode="major"))


def test_build_score_spec_requires_values_for_every_material_placement() -> None:
    document = _validated_script()
    plan, _ = build_piece_plan(document, PlanChoice(tonal_center=0, mode="major"))

    with pytest.raises(ScoreProjectionError, match="placement values"):
        build_score_spec(
            document,
            plan,
            score_id="score-001",
            divisions=12,
            length_units_by_score_unit={"score-unit-statement": 48},
            harmonies_by_score_unit={
                "score-unit-statement": (ScoreHarmony("harmony-001", 0, 48, 0, "major"),)
            },
            directions_by_score_unit={"score-unit-statement": ()},
            notes_by_material_placement={
                "theme-first": (ScoreNote("note-theme", 0, 24, 72, "upper"),)
            },
            cumulative_projection_ledger=(),
        )


def test_build_piece_plan_rejects_placement_level_performance_directions() -> None:
    document = _validated_script()
    document["script"]["performance_setup"]["performance_directions"]["theme-gentle"] = {
        "target": {"type": "material_placement", "id": "theme-first"},
        "description": "主題だけを穏やかに演奏する。",
    }

    with pytest.raises(ScoreProjectionError, match="performance direction target"):
        build_piece_plan(document, PlanChoice(tonal_center=0, mode="major"))


def test_build_score_spec_requires_each_stage_to_cover_the_planned_units() -> None:
    document = _validated_script()
    plan, _ = build_piece_plan(document, PlanChoice(tonal_center=0, mode="major"))

    with pytest.raises(ScoreProjectionError, match="score-unit values"):
        build_score_spec(
            document,
            plan,
            score_id="score-001",
            divisions=12,
            length_units_by_score_unit={},
            harmonies_by_score_unit={"score-unit-statement": ()},
            directions_by_score_unit={"score-unit-statement": ()},
            notes_by_material_placement={
                "theme-first": (ScoreNote("note-theme", 0, 24, 72, "upper"),),
                "support-first": (ScoreNote("note-support", 0, 48, 48, "lower"),),
            },
            cumulative_projection_ledger=(),
        )


def test_build_piece_plan_requires_a_validated_script() -> None:
    document = _validated_script()
    document["status"] = "draft"

    with pytest.raises(ScoreProjectionError, match="pass validation"):
        build_piece_plan(document, PlanChoice(tonal_center=0, mode="major"))


def test_build_piece_plan_rejects_an_invalid_tonal_center() -> None:
    with pytest.raises(ScoreProjectionError, match="pitch class"):
        build_piece_plan(_validated_script(), PlanChoice(tonal_center=12, mode="major"))


def test_build_piece_plan_requires_the_profile_instrumentation_and_duration() -> None:
    document = _validated_script()
    document["script"]["performance_setup"]["target_duration_seconds"] = 120

    with pytest.raises(ScoreProjectionError, match="generation profile"):
        build_piece_plan(document, PlanChoice(tonal_center=0, mode="major"))


def test_build_score_spec_preserves_a_placement_variation_as_a_typed_constraint() -> None:
    document = _with_return_variation(_validated_script())
    plan, _ = build_piece_plan(document, PlanChoice(tonal_center=0, mode="major"))
    unit_ids = ("score-unit-statement", "score-unit-return")
    notes_by_placement = {
        "theme-first": (ScoreNote("note-theme", 0, 24, 72, "upper"),),
        "support-first": (ScoreNote("note-support", 0, 48, 48, "lower"),),
        "theme-return": (ScoreNote("note-return", 0, 24, 74, "upper"),),
    }

    score, ledger = build_score_spec(
        document,
        plan,
        score_id="score-001",
        divisions=12,
        length_units_by_score_unit={unit_id: 48 for unit_id in unit_ids},
        harmonies_by_score_unit={
            unit_id: (ScoreHarmony(f"harmony-{unit_id}", 0, 48, 0, "major"),)
            for unit_id in unit_ids
        },
        directions_by_score_unit={unit_id: () for unit_id in unit_ids},
        notes_by_material_placement=notes_by_placement,
        cumulative_projection_ledger=_upstream_score_ledger(plan, notes_by_placement),
    )

    relation = next(entry for entry in ledger if entry.source_id == "theme-varied")
    theme_layer_ids = {
        layer.score_unit_layer_id
        for unit in score.score_units
        for layer in unit.score_unit_layers
        if layer.source_material_placement_id in {"theme-first", "theme-return"}
    }
    assert theme_layer_ids == {
        "score-unit-layer-theme-first",
        "score-unit-layer-theme-return",
    }
    assert relation.source_kind == "script_element_variation_relation"
    assert relation.target_kind == "score_generation_constraint"
    assert relation.status == "passed"
    assert relation.evidence == ("material_placement:theme-first->material_placement:theme-return")


def test_build_score_spec_preserves_a_material_placement_transition() -> None:
    document = _with_transition(_validated_script())
    plan, _ = build_piece_plan(document, PlanChoice(tonal_center=0, mode="major"))
    unit_ids = (
        "score-unit-statement",
        "score-unit-bridge",
        "score-unit-return",
    )
    placement_ids = ("theme-first", "support-first", "bridge-only", "theme-return")
    notes_by_placement = {
        placement_id: (ScoreNote(f"note-{placement_id}", 0, 24, 60, "upper"),)
        for placement_id in placement_ids
    }

    _, ledger = build_score_spec(
        document,
        plan,
        score_id="score-001",
        divisions=12,
        length_units_by_score_unit={unit_id: 48 for unit_id in unit_ids},
        harmonies_by_score_unit={
            unit_id: (ScoreHarmony(f"harmony-{unit_id}", 0, 48, 0, "major"),)
            for unit_id in unit_ids
        },
        directions_by_score_unit={unit_id: () for unit_id in unit_ids},
        notes_by_material_placement=notes_by_placement,
        cumulative_projection_ledger=_upstream_score_ledger(plan, notes_by_placement),
    )

    transition = next(entry for entry in ledger if entry.source_id == "to-return")
    assert transition.source_kind == "material_placement_transition"
    assert transition.target_kind == "score_generation_constraint"
    assert transition.status == "passed"
    assert transition.evidence == "theme-first->bridge-only->theme-return"


def test_build_score_spec_returns_only_new_relation_projection_evidence() -> None:
    document = _with_return_variation(_validated_script())
    plan, _ = build_piece_plan(document, PlanChoice(tonal_center=0, mode="major"))
    unit_ids = ("score-unit-statement", "score-unit-return")
    notes_by_placement = {
        "theme-first": (ScoreNote("note-theme", 0, 24, 72, "upper"),),
        "support-first": (ScoreNote("note-support", 0, 48, 48, "lower"),),
        "theme-return": (ScoreNote("note-return", 0, 24, 74, "upper"),),
    }
    existing = _upstream_score_ledger(plan, notes_by_placement)

    _, new_evidence = build_score_spec(
        document,
        plan,
        score_id="score-001",
        divisions=12,
        length_units_by_score_unit={unit_id: 48 for unit_id in unit_ids},
        harmonies_by_score_unit={
            unit_id: (ScoreHarmony(f"harmony-{unit_id}", 0, 48, 0, "major"),)
            for unit_id in unit_ids
        },
        directions_by_score_unit={unit_id: () for unit_id in unit_ids},
        notes_by_material_placement=notes_by_placement,
        cumulative_projection_ledger=existing,
    )

    assert [entry.source_id for entry in new_evidence] == ["theme-varied"]
    assert {entry.target_kind for entry in new_evidence} == {"score_generation_constraint"}


def test_build_score_spec_requires_existing_evidence_for_every_score_note() -> None:
    document = _validated_script()
    plan, _ = build_piece_plan(document, PlanChoice(tonal_center=0, mode="major"))
    notes_by_placement = {
        "theme-first": (ScoreNote("note-theme", 0, 24, 72, "upper"),),
        "support-first": (ScoreNote("note-support", 0, 48, 48, "lower"),),
    }
    incomplete_ledger = tuple(
        entry
        for entry in _upstream_score_ledger(plan, notes_by_placement)
        if entry.target_id != "note-support"
    )

    with pytest.raises(
        ProjectionLedgerValidationError,
        match="missing projection target: score_note:note-support",
    ):
        build_score_spec(
            document,
            plan,
            score_id="score-001",
            divisions=12,
            length_units_by_score_unit={"score-unit-statement": 48},
            harmonies_by_score_unit={
                "score-unit-statement": (ScoreHarmony("harmony-001", 0, 48, 0, "major"),)
            },
            directions_by_score_unit={"score-unit-statement": ()},
            notes_by_material_placement=notes_by_placement,
            cumulative_projection_ledger=incomplete_ledger,
        )


def test_build_score_spec_rejects_a_plan_from_different_source_content() -> None:
    document = _validated_script()
    plan, _ = build_piece_plan(document, PlanChoice(tonal_center=0, mode="major"))

    with pytest.raises(ScoreProjectionError, match="does not match the script"):
        build_score_spec(
            document,
            replace(plan, title="別の曲"),
            score_id="score-001",
            divisions=12,
            length_units_by_score_unit={"score-unit-statement": 48},
            harmonies_by_score_unit={
                "score-unit-statement": (ScoreHarmony("harmony-001", 0, 48, 0, "major"),)
            },
            directions_by_score_unit={"score-unit-statement": ()},
            notes_by_material_placement={
                "theme-first": (ScoreNote("note-theme", 0, 24, 72, "upper"),),
                "support-first": (ScoreNote("note-support", 0, 48, 48, "lower"),),
            },
            cumulative_projection_ledger=(),
        )
