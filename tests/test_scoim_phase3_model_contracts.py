import json
from pathlib import Path

from jsonschema import Draft202012Validator

from scoim.phase3_model_contracts import (
    build_harmonic_plan,
    build_score_harmonies,
    check_harmony_response,
    harmony_projection_entries,
    harmony_prompt,
    harmony_response_schema,
    overall_plan_prompt,
    overall_plan_response_schema,
)

_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "scoim"
    / "score-unit-layer-vertical"
    / "basic-validated-script.json"
)


def _document() -> dict[str, object]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _overall_response() -> dict[str, object]:
    return {
        "tonal_center": 0,
        "mode": "major",
        "overall_harmonic_story": "安定した響きから少し離れ、主調へ戻る。",
        "section_harmonic_intents": [
            {
                "harmonic_intent": "主調を明確に示す。",
                "connection_from_previous": "静かに始める。",
            },
            {
                "harmonic_intent": "緊張を少し高める。",
                "connection_from_previous": "前の響きを受けて動き出す。",
            },
            {
                "harmonic_intent": "主題を違う角度から支える。",
                "connection_from_previous": "緊張を解きながら戻る。",
            },
            {
                "harmonic_intent": "主調へ着地する。",
                "connection_from_previous": "余韻を保って閉じる。",
            },
        ],
    }


def test_overall_plan_schema_uses_position_and_does_not_accept_final_ids() -> None:
    schema = overall_plan_response_schema(leaf_count=4)
    valid = _overall_response()

    assert list(Draft202012Validator(schema).iter_errors(valid)) == []
    valid["section_harmonic_intents"][0]["section_id"] = "statement"
    errors = list(Draft202012Validator(schema).iter_errors(valid))

    assert len(errors) == 1
    assert errors[0].validator == "additionalProperties"


def test_overall_plan_is_bound_to_leaf_sections_in_performance_order() -> None:
    result = build_harmonic_plan(
        _document(),
        _overall_response(),
        divisions=12,
        units_per_duration_weight=12,
    )

    assert result.piece_plan.tonal_center == 0
    assert result.piece_plan.mode == "major"
    assert [intent.section_id for intent in result.section_intents] == [
        "statement",
        "bridge",
        "return",
        "release",
    ]
    assert [intent.score_unit_id for intent in result.section_intents] == [
        "score-unit-statement",
        "score-unit-bridge",
        "score-unit-return",
        "score-unit-release",
    ]
    assert result.length_units_by_score_unit == {
        "score-unit-statement": 48,
        "score-unit-bridge": 12,
        "score-unit-return": 48,
        "score-unit-release": 12,
    }
    assert {
        entry.target_id for entry in result.projection_ledger if entry.target_kind == "score_unit"
    } == set(result.length_units_by_score_unit)


def test_overall_plan_prompt_lists_every_leaf_section_in_performance_order() -> None:
    prompt = overall_plan_prompt(_document())

    positions = [prompt.index(name) for name in ("statement", "bridge", "return", "release")]
    assert positions == sorted(positions)
    assert "主題を提示する" in prompt
    assert "主音で終止する" in prompt


def test_harmony_schema_omits_ids_and_python_assigns_stable_ids() -> None:
    schema = harmony_response_schema()
    response = {
        "harmonies": [
            {"duration_units": 8, "root_pitch_class": 7, "quality": "major"},
            {"duration_units": 4, "root_pitch_class": 0, "quality": "major"},
        ]
    }

    assert list(Draft202012Validator(schema).iter_errors(response)) == []
    harmonies = build_score_harmonies("score-unit-release", response)

    assert [harmony.harmony_id for harmony in harmonies] == [
        "harmony-score-unit-release-001",
        "harmony-score-unit-release-002",
    ]
    assert [harmony.at_units for harmony in harmonies] == [0, 8]
    ledger = harmony_projection_entries("score-unit-release", harmonies)
    assert [entry.target_id for entry in ledger] == [
        "harmony-score-unit-release-001",
        "harmony-score-unit-release-002",
    ]
    assert all(entry.source_id == "score-unit-release" for entry in ledger)


def test_harmony_check_rejects_incomplete_score_unit_coverage() -> None:
    response = {
        "harmonies": [
            {"duration_units": 8, "root_pitch_class": 7, "quality": "major"},
        ]
    }

    issues = check_harmony_response(response, length_units=12)

    assert [issue.path for issue in issues] == ["/harmonies"]
    assert "exactly cover" in issues[0].message


def test_harmony_check_requires_only_the_final_unit_to_end_on_the_tonic() -> None:
    response = {
        "harmonies": [
            {"duration_units": 12, "root_pitch_class": 7, "quality": "major"},
        ]
    }

    non_final_issues = check_harmony_response(response, length_units=12)
    final_issues = check_harmony_response(
        response,
        length_units=12,
        required_final_tonic=(0, "major"),
    )

    assert non_final_issues == ()
    assert [issue.path for issue in final_issues] == ["/harmonies/0"]


def test_harmony_prompt_contains_target_materials_variation_and_neighbor_context() -> None:
    document = _document()
    plan = build_harmonic_plan(
        document,
        _overall_response(),
        divisions=12,
        units_per_duration_weight=12,
    )

    prompt = harmony_prompt(
        document,
        plan,
        intent_index=2,
        previous_final_harmony={"root_pitch_class": 7, "quality": "major"},
    )

    assert "theme-return" in prompt
    assert "support-return" in prompt
    assert "中心となる短い旋律" in prompt
    assert "theme-varied" in prompt
    assert "冒頭の輪郭" in prompt
    assert "主調へ着地する" in prompt
    assert '"root_pitch_class": 7' in prompt
