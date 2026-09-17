import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from scoim.phase3_model_contracts import HarmonicPlan, SectionHarmonicIntent
from scoim.phase4_model_contracts import (
    build_foreground_notes,
    build_foreground_operations,
    check_foreground_response,
    foreground_prompt,
    foreground_response_schema,
)
from scoim.score_ir import ScoreHarmony, ScoreNote
from scoim.score_projection import PlanChoice, build_piece_plan

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "scoim" / "score-unit-layer-vertical"
COLLISION_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "scoim" / "phase4-collisions"


def _basic_script() -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / "basic-validated-script.json").read_text(encoding="utf-8"))


def _basic_plan(document: dict[str, object]) -> HarmonicPlan:
    plan, ledger = build_piece_plan(document, PlanChoice(0, "major"))
    leaf_nodes = tuple(node for node in plan.nodes if node.score_unit_id is not None)
    lengths = {
        node.score_unit_id: int(node.duration_weight * 48)
        for node in leaf_nodes
        if node.score_unit_id is not None and node.duration_weight is not None
    }
    return HarmonicPlan(
        piece_plan=plan,
        overall_harmonic_story="ハ長調から主和音へ戻る。",
        section_intents=tuple(
            SectionHarmonicIntent(
                node.section_id,
                node.score_unit_id,
                "共有和声に沿う。",
                "前区分からつなぐ。",
            )
            for node in leaf_nodes
            if node.score_unit_id is not None
        ),
        divisions=12,
        length_units_by_score_unit=lengths,
        projection_ledger=ledger,
    )


def test_registers_each_foreground_placement_in_dependency_order() -> None:
    operations = build_foreground_operations(_basic_script())

    assert tuple(operation.material_placement_id for operation in operations) == (
        "theme-first",
        "theme-return",
        "ending-only",
        "bridge-only",
    )
    bridge = operations[-1]
    assert bridge.transition_id == "to-return"
    assert tuple(source.kind for source in bridge.comparison_sources) == (
        "transition_source",
        "transition_target",
    )


def test_builds_stable_score_notes_from_a_valid_foreground_response() -> None:
    response = {
        "notes": [
            {"at_units": 0, "duration_units": 12, "pitch": 72, "voice": "upper"},
            {"at_units": 12, "duration_units": 24, "pitch": 48, "voice": "lower"},
        ]
    }

    assert check_foreground_response(response, length_units=48) == ()
    notes = build_foreground_notes("theme-first", response)

    assert tuple(note.score_note_id for note in notes) == (
        "score-note-theme-first-001",
        "score-note-theme-first-002",
    )
    assert {note.voice for note in notes} == {"upper", "lower"}


def test_foreground_schema_rejects_model_owned_identifiers() -> None:
    response = {
        "notes": [
            {
                "score_note_id": "model-owned",
                "at_units": 0,
                "duration_units": 12,
                "pitch": 72,
                "voice": "upper",
            }
        ]
    }

    errors = tuple(Draft202012Validator(foreground_response_schema()).iter_errors(response))

    assert len(errors) == 1
    assert errors[0].validator == "additionalProperties"


def test_prompt_includes_the_accepted_variation_source() -> None:
    document = _basic_script()
    plan = _basic_plan(document)
    operation = next(
        item
        for item in build_foreground_operations(document)
        if item.material_placement_id == "theme-return"
    )
    harmonies = {
        score_unit_id: (ScoreHarmony(f"h-{score_unit_id}", 0, length, 0, "major"),)
        for score_unit_id, length in plan.length_units_by_score_unit.items()
    }
    source_note = ScoreNote("source-note", 0, 12, 72, "upper")

    prompt = foreground_prompt(
        document,
        plan,
        operation,
        harmonies_by_score_unit=harmonies,
        accepted_notes_by_material_placement={"theme-first": (source_note,)},
    )

    assert '"kind": "material_placement_variation"' in prompt
    assert '"source_material_placement_id": "theme-first"' in prompt
    assert '"pitch": 72' in prompt
    assert '"preserve": ["冒頭の輪郭"]' in prompt


def test_prompt_exposes_existing_score_unit_occupancy_and_collision_rule() -> None:
    document = _basic_script()
    plan = _basic_plan(document)
    operation = next(
        item
        for item in build_foreground_operations(document)
        if item.material_placement_id == "theme-return"
    )
    harmonies = {
        score_unit_id: (ScoreHarmony(f"h-{score_unit_id}", 0, length, 0, "major"),)
        for score_unit_id, length in plan.length_units_by_score_unit.items()
    }
    source_note = ScoreNote("source-note", 0, 12, 72, "upper")
    occupied_notes = (
        ScoreNote("occupied-later", 12, 6, 67, "lower"),
        ScoreNote("occupied-first", 0, 12, 60, "lower"),
    )

    prompt = foreground_prompt(
        document,
        plan,
        operation,
        harmonies_by_score_unit=harmonies,
        accepted_notes_by_material_placement={"theme-first": (source_note,)},
        existing_unit_notes=occupied_notes,
    )

    context = json.loads(prompt.split("入力: ", 1)[1])
    assert context["target"]["occupied_notes"] == [
        {"at_units": 0, "duration_units": 12, "pitch": 60, "voice": "lower"},
        {"at_units": 12, "duration_units": 6, "pitch": 67, "voice": "lower"},
    ]
    assert "occupied_notesは変更禁止" in prompt
    assert "同じ音高かつ同じ声部" in prompt
    assert "境界が接するだけ" in prompt
    assert "異なる音高または異なる声部" in prompt


def test_prompt_rejects_a_comparison_source_that_has_not_been_accepted() -> None:
    document = _basic_script()
    plan = _basic_plan(document)
    operation = next(
        item
        for item in build_foreground_operations(document)
        if item.material_placement_id == "theme-return"
    )
    harmonies = {
        score_unit_id: (ScoreHarmony(f"h-{score_unit_id}", 0, length, 0, "major"),)
        for score_unit_id, length in plan.length_units_by_score_unit.items()
    }

    with pytest.raises(ValueError, match=r"^a foreground comparison source has not been accepted$"):
        foreground_prompt(
            document,
            plan,
            operation,
            harmonies_by_score_unit=harmonies,
            accepted_notes_by_material_placement={},
        )


def test_prompt_rejects_a_target_without_shared_harmony() -> None:
    document = _basic_script()
    plan = _basic_plan(document)
    operation = build_foreground_operations(document)[0]

    with pytest.raises(ValueError, match=r"^the foreground target has no shared harmony$"):
        foreground_prompt(
            document,
            plan,
            operation,
            harmonies_by_score_unit={},
            accepted_notes_by_material_placement={},
        )


def test_prompt_rejects_a_target_without_a_section_intent() -> None:
    document = _basic_script()
    plan = _basic_plan(document)
    operation = build_foreground_operations(document)[0]
    plan = replace(
        plan,
        section_intents=tuple(
            intent
            for intent in plan.section_intents
            if intent.score_unit_id != operation.score_unit_id
        ),
    )
    harmonies = {
        score_unit_id: (ScoreHarmony(f"h-{score_unit_id}", 0, length, 0, "major"),)
        for score_unit_id, length in plan.length_units_by_score_unit.items()
    }

    with pytest.raises(ValueError, match=r"^the foreground target has no section intent$"):
        foreground_prompt(
            document,
            plan,
            operation,
            harmonies_by_score_unit=harmonies,
            accepted_notes_by_material_placement={},
        )


def test_rejects_an_exact_copy_and_a_collision_with_an_accepted_layer() -> None:
    source = (
        ScoreNote("source", 0, 12, 72, "upper"),
        ScoreNote("source-2", 12, 12, 74, "upper"),
    )
    response = {
        "notes": [
            {"at_units": 0, "duration_units": 12, "pitch": 72, "voice": "upper"},
            {"at_units": 12, "duration_units": 12, "pitch": 74, "voice": "upper"},
        ]
    }

    issues = check_foreground_response(
        response,
        length_units=48,
        comparison_notes=(source,),
        existing_unit_notes=(ScoreNote("other-layer", 6, 12, 72, "upper"),),
    )

    assert {issue.message for issue in issues} == {
        "Foreground candidate overlaps an occupied note: "
        "candidate=[0,12), occupied=[6,18), pitch=72, voice=upper",
        "A reused or varied foreground must not be an exact copy",
    }


def test_reports_an_exact_duplicate_within_one_foreground_response() -> None:
    note = {"at_units": 0, "duration_units": 12, "pitch": 72, "voice": "upper"}

    issues = check_foreground_response(
        {"notes": [note, note.copy()]},
        length_units=48,
    )

    assert "Foreground notes contain an exact duplicate" in {issue.message for issue in issues}


def test_reports_candidate_to_candidate_collision_with_both_intervals() -> None:
    response = {
        "notes": [
            {"at_units": 0, "duration_units": 8, "pitch": 72, "voice": "upper"},
            {"at_units": 6, "duration_units": 6, "pitch": 72, "voice": "upper"},
        ]
    }

    issues = check_foreground_response(response, length_units=48)

    collision = next(issue for issue in issues if "another candidate" in issue.message)
    assert collision.path == "/notes/1"
    assert "candidate=[6,12)" in collision.message
    assert "other_candidate=[0,8)" in collision.message
    assert "pitch=72" in collision.message
    assert "voice=upper" in collision.message


def test_reports_every_candidate_to_occupied_collision_by_candidate_index() -> None:
    response = {
        "notes": [
            {"at_units": 2, "duration_units": 8, "pitch": 60, "voice": "lower"},
            {"at_units": 12, "duration_units": 4, "pitch": 60, "voice": "lower"},
            {"at_units": 2, "duration_units": 8, "pitch": 61, "voice": "lower"},
            {"at_units": 2, "duration_units": 8, "pitch": 60, "voice": "upper"},
        ]
    }
    occupied = (
        ScoreNote("first", 0, 4, 60, "lower"),
        ScoreNote("second", 8, 4, 60, "lower"),
    )

    issues = check_foreground_response(
        response,
        length_units=48,
        existing_unit_notes=occupied,
    )

    collisions = [issue for issue in issues if "an occupied note" in issue.message]
    assert [issue.path for issue in collisions] == ["/notes/0", "/notes/0"]
    assert {issue.message for issue in collisions} == {
        "Foreground candidate overlaps an occupied note: "
        "candidate=[2,10), occupied=[0,4), pitch=60, voice=lower",
        "Foreground candidate overlaps an occupied note: "
        "candidate=[2,10), occupied=[8,12), pitch=60, voice=lower",
    }


@pytest.mark.parametrize(
    ("case_id", "expected_count"),
    (("long-shadow", 15), ("quiet-tension", 21)),
)
def test_reproduces_observed_phase4_collision_pairs(
    case_id: str,
    expected_count: int,
) -> None:
    fixture = json.loads((COLLISION_FIXTURE_ROOT / f"{case_id}.json").read_text(encoding="utf-8"))
    occupied = tuple(
        ScoreNote(
            f"occupied-{index}",
            note["at_units"],
            note["duration_units"],
            note["pitch"],
            note["voice"],
        )
        for index, note in enumerate(fixture["occupied_notes"])
    )

    issues = check_foreground_response(
        fixture["response"],
        length_units=fixture["length_units"],
        existing_unit_notes=occupied,
    )

    actual = [
        {"path": issue.path, "message": issue.message}
        for issue in issues
        if "an occupied note" in issue.message
    ]
    expected = []
    for collision in fixture["expected_collisions"]:
        candidate = collision["candidate"]
        occupied_note = collision["occupied"]
        candidate_end = candidate["at_units"] + candidate["duration_units"]
        occupied_end = occupied_note["at_units"] + occupied_note["duration_units"]
        expected.append(
            {
                "path": f"/notes/{collision['candidate_index']}",
                "message": (
                    "Foreground candidate overlaps an occupied note: "
                    f"candidate=[{candidate['at_units']},{candidate_end}), "
                    f"occupied=[{occupied_note['at_units']},{occupied_end}), "
                    f"pitch={candidate['pitch']}, voice={candidate['voice']}"
                ),
            }
        )
    assert len(actual) == expected_count
    assert actual == expected


def test_keeps_material_variation_and_implicit_reuse_as_separate_contexts() -> None:
    document = _basic_script()
    script = document["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    placements = script["material_placements"]
    materials = script["materials"]
    relations = script["script_element_variation_relations"]
    assert all(isinstance(value, dict) for value in (sections, placements, materials, relations))
    sections["variation-later"] = {
        "parent_section_id": "whole",
        "order": 3,
        "role": "variation",
        "relative_length": 0.5,
        "description": "変奏をもう一度使う。",
    }
    sections["release"]["order"] = 4
    materials["varied-theme"] = {"description": "主題から派生した旋律。"}
    placements["theme-return"]["material_id"] = "varied-theme"
    placements["varied-later"] = {
        "section_id": "variation-later",
        "material_id": "varied-theme",
        "role": "foreground",
    }
    relations.clear()
    relations["material-varied"] = {
        "source": {"type": "material", "id": "theme"},
        "target": {"type": "material", "id": "varied-theme"},
        "preserve": ["輪郭"],
        "change": ["リズム"],
        "description": "主題を変奏する。",
    }

    operation = next(
        item
        for item in build_foreground_operations(document)
        if item.material_placement_id == "varied-later"
    )

    assert tuple(source.kind for source in operation.comparison_sources) == (
        "material_variation",
        "implicit_reuse",
    )


def test_rejects_a_transition_whose_boundary_is_not_foreground() -> None:
    document = _basic_script()
    script = document["script"]
    assert isinstance(script, dict)
    transitions = script["material_placement_transitions"]
    assert isinstance(transitions, dict)
    transitions["to-return"]["source_material_placement_id"] = "support-first"

    with pytest.raises(
        ValueError, match=r"^unrepresentable at /script/material_placement_transitions/to-return"
    ):
        build_foreground_operations(document)


def test_rejects_a_variation_source_that_is_itself_a_transition() -> None:
    document = _basic_script()
    script = document["script"]
    assert isinstance(script, dict)
    relations = script["script_element_variation_relations"]
    assert isinstance(relations, dict)
    relations["theme-varied"]["source"]["id"] = "bridge-only"

    with pytest.raises(
        ValueError,
        match=(
            r"^unrepresentable at /script/script_element_variation_relations/"
            r"theme-varied/source/id"
        ),
    ):
        build_foreground_operations(document)


def test_rejects_an_accompaniment_relation_before_registering_foregrounds() -> None:
    document = _basic_script()
    script = document["script"]
    assert isinstance(script, dict)
    relations = script["script_element_variation_relations"]
    assert isinstance(relations, dict)
    relations["theme-varied"]["target"]["id"] = "support-return"

    with pytest.raises(
        ValueError,
        match=r"^unrepresentable at /script/script_element_variation_relations/theme-varied",
    ):
        build_foreground_operations(document)


def test_rejects_a_material_variation_source_without_a_foreground_placement() -> None:
    document = _basic_script()
    script = document["script"]
    assert isinstance(script, dict)
    materials = script["materials"]
    relations = script["script_element_variation_relations"]
    assert isinstance(materials, dict)
    assert isinstance(relations, dict)
    materials["unused-source"] = {"description": "配置されない変奏元。"}
    relations["ending-from-unused"] = {
        "source": {"type": "material", "id": "unused-source"},
        "target": {"type": "material", "id": "ending"},
        "preserve": ["輪郭"],
        "change": ["終止"],
        "description": "配置されない素材から終止を変奏する。",
    }

    with pytest.raises(
        ValueError,
        match=(
            r"^unrepresentable at /script/script_element_variation_relations/"
            r"ending-from-unused/source/id"
        ),
    ):
        build_foreground_operations(document)


def test_rejects_cyclic_foreground_variation_dependencies() -> None:
    document = _basic_script()
    script = document["script"]
    assert isinstance(script, dict)
    relations = script["script_element_variation_relations"]
    assert isinstance(relations, dict)
    relations["theme-varied-back"] = {
        "source": {"type": "material_placement", "id": "theme-return"},
        "target": {"type": "material_placement", "id": "theme-first"},
        "preserve": ["輪郭"],
        "change": ["リズム"],
        "description": "戻りから冒頭へ循環する。",
    }

    with pytest.raises(
        ValueError, match="the generation script must pass validation before phase 4"
    ):
        build_foreground_operations(document)
