import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from scoim.phase3_model_contracts import build_harmonic_plan, ordered_leaf_section_ids
from scoim.score_ir import ScoreHarmony, ScoreNote
from scoim.score_model_contracts import (
    evaluate_score_group_candidate,
    score_group_prompt,
    score_group_response_schema,
)
from scoim.score_work_plan import ScoreComparison, ScoreWorkOperation

FIXTURE = (
    Path(__file__).parent / "fixtures/scoim/score-unit-layer-vertical/basic-validated-script.json"
)


def _setup(document=None):
    document = document or json.loads(FIXTURE.read_text(encoding="utf-8"))
    plan = build_harmonic_plan(
        document,
        {
            "mode": "major",
            "total_score_units": 120,
            "overall_harmonic_story": "主和音へ戻る。",
            "section_harmonic_intents": [
                {
                    "harmonic_intent": "共有和声に沿う。",
                    "connection_from_previous": "自然につなぐ。",
                }
                for _ in ordered_leaf_section_ids(document)
            ],
        },
        tonal_center=0,
        divisions=12,
    )
    harmonies = {
        key: (ScoreHarmony(f"h-{key}", 0, length, 0, "major"),)
        for key, length in plan.length_units_by_score_unit.items()
    }
    return document, plan, harmonies


def _foreground(pitch=72, at=0, duration=12):
    return {
        "notes": [{"at_units": at, "duration_units": duration, "pitch": pitch, "voice": "upper"}]
    }


def _accompaniment(degree="root"):
    return {
        "events": [
            {
                "at_units": 0,
                "preferred_duration_units": 12,
                "degree": degree,
                "preferred_register_zone": "low",
                "voice": "lower",
                "articulations": ["normal"],
            }
        ]
    }


def _section_comparison(
    source=("theme-first", "support-first"), target=("theme-return", "support-return")
):
    return ScoreComparison(
        "section_variation",
        "section-varied",
        source,
        target,
        "輪郭を保って変奏する。",
        ("輪郭",),
        ("伴奏",),
        "statement",
        "return",
    )


def _evaluate(setup, operation, comparisons, response, accepted=None):
    document, plan, harmonies = setup
    return evaluate_score_group_candidate(
        document,
        plan,
        operation,
        comparisons,
        response,
        harmonies_by_score_unit=harmonies,
        accepted_notes_by_placement=accepted or {},
    )


def test_response_schema_binds_role_counts_and_forbids_model_owned_ids():
    document, _, _ = _setup()
    operation = ScoreWorkOperation("op", ("theme-first", "support-first"), ())
    validator = Draft202012Validator(score_group_response_schema(document, operation))
    response = {"foregrounds": [_foreground()], "accompaniments": [_accompaniment()]}
    assert not tuple(validator.iter_errors(response))
    response["foregrounds"][0]["material_placement_id"] = "model-id"
    assert any(e.validator == "additionalProperties" for e in validator.iter_errors(response))
    response["foregrounds"] = []
    assert any(e.validator == "minItems" for e in validator.iter_errors(response))
    solo_schema = score_group_response_schema(
        document, ScoreWorkOperation("solo", ("bridge-only",), ())
    )
    assert solo_schema["properties"]["accompaniments"]["maxItems"] == 0


def test_prompt_delivers_full_section_range_and_distinguishes_internal_from_external():
    document, _, _ = _setup()
    script = document["script"]
    script["sections"]["source-span"] = {
        "parent_section_id": "whole",
        "order": 0,
        "role": "statement",
        "description": "元の範囲。",
    }
    script["sections"]["statement"]["parent_section_id"] = "source-span"
    script["sections"]["bridge"]["parent_section_id"] = "source-span"
    script["sections"]["return"]["order"] = 1
    script["sections"]["release"]["order"] = 2
    setup = _setup(document)
    document, plan, harmonies = setup
    operation = ScoreWorkOperation("joint", ("bridge-only", "theme-return", "support-return"), ())
    comparison = ScoreComparison(
        "section_variation",
        "range",
        ("theme-first", "support-first", "bridge-only"),
        ("theme-return", "support-return"),
        "範囲を変奏する。",
        ("動機",),
        ("長さ",),
        "source-span",
        "return",
    )
    accepted = {
        "theme-first": (ScoreNote("f", 0, 12, 72, "upper"),),
        "support-first": (ScoreNote("a", 0, 12, 48, "lower"),),
    }
    prompt = score_group_prompt(
        document,
        plan,
        operation,
        (comparison,),
        harmonies_by_score_unit=harmonies,
        accepted_notes_by_placement=accepted,
        accepted_responses_by_placement={"support-first": _accompaniment()},
    )
    context = json.loads(prompt.split("入力: ", 1)[1])
    source = context["comparisons"][0]["source_range"]
    assert [leaf["section_id"] for leaf in source["leaves"]] == ["statement", "bridge"]
    assert (
        source["leaves"][1]["offset_units"]
        == plan.length_units_by_score_unit["score-unit-statement"]
    )
    assert len(context["comparisons"][0]["target_range"]["leaves"]) == 1
    external = source["leaves"][0]["placements"][1]
    assert external["reference_state"] == "accepted"
    assert external["notes"][0]["pitch"] == 48
    assert external["events"] == _accompaniment()["events"]
    assert len(external["notes_sha256"]) == 64
    internal = source["leaves"][1]["placements"][0]
    assert internal["reference_state"] == "joint_candidate"
    assert internal["response_array"] == "foregrounds" and internal["response_index"] == 0
    assert "notes" not in internal and "notes_sha256" not in internal
    assert context["comparisons"][0]["preserve"] == ["動機"]


def test_section_copy_rejection_is_atomic_but_partial_retention_is_allowed():
    setup = _setup()
    op = ScoreWorkOperation("target", ("theme-return", "support-return"), ())
    section = _section_comparison()
    reuse = ScoreComparison("implicit_reuse", None, ("theme-first",), ("theme-return",))
    accepted = {
        "theme-first": (ScoreNote("f", 0, 12, 72, "upper"),),
        "support-first": (ScoreNote("a", 0, 12, 48, "lower"),),
    }
    response = {"foregrounds": [_foreground()], "accompaniments": [_accompaniment()]}
    copied = _evaluate(setup, op, (section, reuse), response, accepted)
    assert len(copied.issues) == 1
    assert copied.comparison_evidence[0]["copy_check"] == "failed"
    assert copied.comparison_evidence[1]["copy_check"] == "not_required"
    assert "source=" in copied.issues[0].message and "target=" in copied.issues[0].message
    assert "realized_joint_candidate=" in copied.issues[0].message
    assert '"support-return"' in copied.issues[0].message
    response["accompaniments"] = [_accompaniment("third")]
    changed = _evaluate(setup, op, (section, reuse), response, accepted)
    assert not changed.issues
    assert changed.notes_by_placement["theme-return"][0].pitch == 72
    assert changed.notes_by_placement["support-return"][0].pitch % 12 == 4
    assert changed.comparison_evidence[0]["semantic_status"] == "unverified"
    assert any(e.status == "unverified" for e in changed.projection_ledger)
    explicit = ScoreComparison(
        "material_placement_variation", "explicit", ("theme-first",), ("theme-return",)
    )
    assert _evaluate(setup, op, (section, explicit), response, accepted).issues


@pytest.mark.parametrize("change_role", [False, True])
def test_section_copy_ignores_placement_role_and_detects_role_removal(change_role):
    document, _, _ = _setup()
    placements = document["script"]["material_placements"]
    del placements["support-return"]
    if change_role:
        del placements["theme-first"]
        document["script"]["material_placement_transitions"] = {}
        document["script"]["script_element_variation_relations"] = {}
        placements["theme-return"]["role"] = "foreground"
        source_ids = ("support-first",)
        accepted = {"support-first": (ScoreNote("old", 0, 12, 72, "upper"),)}
    else:
        source_ids = ("theme-first", "support-first")
        accepted = {
            "theme-first": (ScoreNote("old-f", 0, 12, 72, "upper"),),
            "support-first": (ScoreNote("old-a", 0, 12, 48, "lower"),),
        }
    result = _evaluate(
        _setup(document),
        ScoreWorkOperation("target", ("theme-return",), ()),
        (_section_comparison(source_ids, ("theme-return",)),),
        {"foregrounds": [_foreground()], "accompaniments": []},
        accepted,
    )
    assert bool(result.issues) == change_role


def test_section_copy_uses_global_offsets_instead_of_matching_child_positions():
    document, _, _ = _setup()
    script = document["script"]
    script["material_placement_transitions"] = {}
    script["script_element_variation_relations"] = {}
    del script["sections"]["return"]["relative_length"]
    for index in range(2):
        script["sections"][f"return-{index}"] = {
            "parent_section_id": "return",
            "order": index,
            "role": "phrase",
            "relative_length": 0.5,
            "description": "変奏の子区分。",
        }
    placements = script["material_placements"]
    del placements["support-first"]
    del placements["support-return"]
    placements["theme-return"]["section_id"] = "return-0"
    placements["theme-return-2"] = {
        "section_id": "return-1",
        "material_id": "theme",
        "role": "foreground",
    }
    setup = _setup(document)
    half = setup[1].length_units_by_score_unit["score-unit-return-0"]
    source = (ScoreNote("s1", 0, 12, 72, "upper"), ScoreNote("s2", half, 12, 74, "upper"))
    target_ids = ("theme-return", "theme-return-2")
    result = _evaluate(
        setup,
        ScoreWorkOperation("target", target_ids, ()),
        (_section_comparison(("theme-first",), target_ids),),
        {"foregrounds": [_foreground(), _foreground(74)], "accompaniments": []},
        {"theme-first": source},
    )
    assert len(result.issues) == 1
    assert (
        result.comparison_evidence[0]["source_notes_sha256"]
        == result.comparison_evidence[0]["target_notes_sha256"]
    )


def test_joint_foregrounds_collide_before_accompaniment_placement():
    document, _, _ = _setup()
    document["script"]["material_placements"]["second-foreground"] = {
        "material_id": "ending",
        "section_id": "statement",
        "role": "foreground",
    }
    op = ScoreWorkOperation("joint", ("theme-first", "second-foreground", "support-first"), ())
    result = _evaluate(
        _setup(document),
        op,
        (),
        {
            "foregrounds": [_foreground(duration=12), _foreground(at=6, duration=6)],
            "accompaniments": [_accompaniment()],
        },
    )
    assert result.issues[0].path == "/foregrounds/1/notes/0"
    assert "candidate=[6,12)" in result.issues[0].message
    assert not result.notes_by_placement


def test_mixed_role_joint_source_evidence_is_computed_after_pitch_placement():
    setup = _setup()
    operation = ScoreWorkOperation(
        "joint", ("theme-first", "support-first", "theme-return", "support-return"), ()
    )
    result = _evaluate(
        setup,
        operation,
        (_section_comparison(),),
        {
            "foregrounds": [_foreground(), _foreground(74)],
            "accompaniments": [_accompaniment(), _accompaniment("third")],
        },
    )
    assert not result.issues
    evidence = result.comparison_evidence[0]
    assert evidence["internal_source_placement_ids"] == ["theme-first", "support-first"]
    assert evidence["external_source_placement_ids"] == []
    assert evidence["source_notes_sha256"] != evidence["target_notes_sha256"]
    assert set(result.notes_by_placement) == set(operation.material_placement_ids)


def test_transition_prompt_selects_external_boundary_and_labels_internal_boundary():
    document, plan, harmonies = _setup()
    op = ScoreWorkOperation("joint", ("bridge-only", "theme-return"), ())
    comparisons = (
        ScoreComparison("transition_source", "to-return", ("theme-first",), ("bridge-only",)),
        ScoreComparison("transition_target", "to-return", ("theme-return",), ("bridge-only",)),
    )
    prompt = score_group_prompt(
        document,
        plan,
        op,
        comparisons,
        harmonies_by_score_unit=harmonies,
        accepted_notes_by_placement={
            "theme-first": (
                ScoreNote("first", 0, 12, 72, "upper"),
                ScoreNote("last", 12, 12, 74, "upper"),
            )
        },
        accepted_responses_by_placement={},
    )
    context = json.loads(prompt.split("入力: ", 1)[1])
    external = context["comparisons"][0]["source_placements"][0]
    assert [n["pitch"] for n in external["notes"]] == [72, 74]
    assert [n["pitch"] for n in external["boundary_notes"]] == [74]
    assert (
        context["comparisons"][1]["source_placements"][0]["boundary_selection"]
        == "first_starting_notes"
    )
    assert len(context["transitions"]) == 1


def test_missing_accepted_source_is_not_presented_as_an_internal_candidate():
    document, plan, harmonies = _setup()
    with pytest.raises(ValueError, match="has not been accepted"):
        score_group_prompt(
            document,
            plan,
            ScoreWorkOperation("target", ("theme-return", "support-return"), ()),
            (_section_comparison(),),
            harmonies_by_score_unit=harmonies,
            accepted_notes_by_placement={},
            accepted_responses_by_placement={},
        )
