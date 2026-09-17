import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from scoim.phase3_model_contracts import HarmonicPlan, SectionHarmonicIntent
from scoim.phase5_model_contracts import (
    accompaniment_prompt,
    accompaniment_response_schema,
    build_accompaniment_operations,
    check_accompaniment_response,
)
from scoim.score_ir import ScoreHarmony, ScoreNote
from scoim.score_projection import PlanChoice, build_piece_plan

_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "scoim"
    / "score-unit-layer-vertical"
    / "basic-validated-script.json"
)


def _document() -> dict[str, object]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _plan(document: dict[str, object]) -> HarmonicPlan:
    piece_plan, ledger = build_piece_plan(document, PlanChoice(0, "major"))
    leaves = tuple(node for node in piece_plan.nodes if node.score_unit_id is not None)
    lengths = {
        node.score_unit_id: int(node.duration_weight * 48)
        for node in leaves
        if node.score_unit_id is not None and node.duration_weight is not None
    }
    return HarmonicPlan(
        piece_plan,
        "主調から少し離れて戻る。",
        tuple(
            SectionHarmonicIntent(
                node.section_id,
                node.score_unit_id,
                "共有和声に沿う。",
                "前区分からつなぐ。",
            )
            for node in leaves
            if node.score_unit_id is not None
        ),
        12,
        lengths,
        ledger,
    )


def test_phase5_registers_every_accompaniment_with_implicit_reuse() -> None:
    operations = build_accompaniment_operations(_document())

    assert [operation.material_placement_id for operation in operations] == [
        "support-first",
        "support-return",
    ]
    assert operations[0].comparison_source is None
    assert operations[1].comparison_source is not None
    assert operations[1].comparison_source.source_material_placement_id == "support-first"


def test_phase5_rejects_an_explicit_relation_involving_accompaniment() -> None:
    document = _document()
    script = document["script"]
    assert isinstance(script, dict)
    relations = script["script_element_variation_relations"]
    assert isinstance(relations, dict)
    relations["support-varied"] = {
        "source": {"type": "material_placement", "id": "support-first"},
        "target": {"type": "material_placement", "id": "support-return"},
        "preserve": ["低音域"],
        "change": ["リズム"],
        "description": "伴奏を変奏する。",
    }

    with pytest.raises(ValueError, match="unrepresentable"):
        build_accompaniment_operations(document)


def test_phase5_rejects_an_event_crossing_a_shared_harmony_boundary() -> None:
    response = {
        "events": [
            {
                "at_units": 11,
                "preferred_duration_units": 2,
                "degree": "root",
                "preferred_register_zone": "low",
                "voice": "lower",
                "articulations": ["normal"],
            }
        ]
    }
    harmonies = (
        ScoreHarmony("h1", 0, 12, 0, "major"),
        ScoreHarmony("h2", 12, 12, 7, "major"),
    )

    issues = check_accompaniment_response(response, length_units=24, harmonies=harmonies)

    assert [issue.message for issue in issues] == [
        "An accompaniment event must fit within one shared harmony"
    ]


def test_phase5_reports_all_static_event_problems_before_pitch_search() -> None:
    response = {
        "events": [
            {
                "at_units": 11,
                "preferred_duration_units": 2,
                "degree": "seventh",
                "preferred_register_zone": "middle",
                "voice": "upper",
                "articulations": ["normal"],
            }
        ]
    }
    harmonies = (
        ScoreHarmony("h1", 0, 12, 0, "major"),
        ScoreHarmony("h2", 12, 12, 7, "major"),
    )

    issues = check_accompaniment_response(response, length_units=24, harmonies=harmonies)

    assert {issue.message for issue in issues} == {
        "An accompaniment event must fit within one shared harmony",
        "The requested degree is unavailable in the starting shared harmony",
    }


def test_phase5_schema_rejects_model_owned_pitch_and_identifier() -> None:
    response = {
        "events": [
            {
                "score_note_id": "model-owned",
                "pitch": 48,
                "at_units": 0,
                "preferred_duration_units": 6,
                "degree": "root",
                "preferred_register_zone": "bass",
                "voice": "lower",
                "articulations": ["normal"],
            }
        ]
    }

    errors = tuple(Draft202012Validator(accompaniment_response_schema()).iter_errors(response))

    assert len(errors) == 1
    assert errors[0].validator == "additionalProperties"


def test_phase5_checks_duplicate_articulations_outside_the_model_schema() -> None:
    schema = accompaniment_response_schema()
    events = schema["properties"]["events"]
    articulations = events["items"]["properties"]["articulations"]
    assert "uniqueItems" not in articulations
    response = {
        "events": [
            {
                "at_units": 0,
                "preferred_duration_units": 6,
                "degree": "root",
                "preferred_register_zone": "bass",
                "voice": "lower",
                "articulations": ["accent", "accent"],
            }
        ]
    }

    issues = check_accompaniment_response(
        response,
        length_units=12,
        harmonies=(ScoreHarmony("h", 0, 12, 0, "major"),),
    )

    assert [issue.message for issue in issues] == [
        "Accompaniment event articulations must be unique"
    ]


def test_phase5_leaves_realized_copy_detection_to_pitch_placement() -> None:
    response = {
        "events": [
            {
                "at_units": 0,
                "preferred_duration_units": 6,
                "degree": "root",
                "preferred_register_zone": "bass",
                "voice": "lower",
                "articulations": ["normal"],
            }
        ]
    }

    issues = check_accompaniment_response(
        response,
        length_units=12,
        harmonies=(ScoreHarmony("h", 0, 12, 0, "major"),),
    )

    assert issues == ()


def test_phase5_prompt_contains_only_the_accepted_reuse_context() -> None:
    document = _document()
    plan = _plan(document)
    operation = build_accompaniment_operations(document)[1]
    harmonies = {
        score_unit_id: (ScoreHarmony(f"h-{score_unit_id}", 0, length, 0, "major"),)
        for score_unit_id, length in plan.length_units_by_score_unit.items()
    }
    response = {
        "events": [
            {
                "at_units": 0,
                "preferred_duration_units": 6,
                "degree": "root",
                "preferred_register_zone": "bass",
                "voice": "lower",
                "articulations": ["normal"],
            }
        ]
    }

    prompt = accompaniment_prompt(
        document,
        plan,
        operation,
        harmonies_by_score_unit=harmonies,
        foreground_notes=(ScoreNote("foreground", 0, 6, 72, "upper"),),
        accepted_responses={"support-first": response},
        accepted_notes={"support-first": (ScoreNote("support", 0, 6, 36, "lower"),)},
    )

    assert '"kind": "implicit_reuse"' in prompt
    assert '"pitch": 36' in prompt
    assert '"pitch": 72' in prompt


def test_phase5_prompt_lists_the_degrees_available_in_each_harmony() -> None:
    document = _document()
    plan = _plan(document)
    operation = build_accompaniment_operations(document)[0]
    length = plan.length_units_by_score_unit[operation.score_unit_id]
    harmonies = {
        operation.score_unit_id: (
            ScoreHarmony("h-major", 0, length // 2, 0, "major"),
            ScoreHarmony(
                "h-major-seventh",
                length // 2,
                length - length // 2,
                5,
                "major-seventh",
            ),
        )
    }

    prompt = accompaniment_prompt(
        document,
        plan,
        operation,
        harmonies_by_score_unit=harmonies,
        foreground_notes=(),
        accepted_responses={},
        accepted_notes={},
    )
    context = json.loads(prompt.split("入力: ", maxsplit=1)[1])

    assert context["target"]["shared_harmony"] == [
        {
            "at_units": 0,
            "available_degrees": ["root", "third", "fifth"],
            "duration_units": length // 2,
            "quality": "major",
            "root_pitch_class": 0,
        },
        {
            "at_units": length // 2,
            "available_degrees": ["root", "third", "fifth", "seventh"],
            "duration_units": length - length // 2,
            "quality": "major-seventh",
            "root_pitch_class": 5,
        },
    ]
