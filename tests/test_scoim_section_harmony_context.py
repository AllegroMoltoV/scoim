import json

import pytest
from test_scoim_score_work_plan import unequal_sections

from scoim.phase3_model_contracts import build_harmonic_plan, harmony_prompt
from scoim.score_ir import ScoreHarmony
from scoim.section_harmony_context import section_harmony_contexts


def test_harmony_request_carries_all_source_leaves_and_applicable_ancestor_relations():
    document = unequal_sections()
    response = {
        "mode": "major",
        "total_score_units": 120,
        "overall_harmonic_story": "Return",
        "section_harmonic_intents": [
            {"harmonic_intent": "Tonic", "connection_from_previous": "Continue"} for _ in range(6)
        ],
    }
    plan = build_harmonic_plan(document, response, tonal_center=0, divisions=12)
    harmonies = {
        f"score-unit-{key}": (
            ScoreHarmony(
                f"harmony-{key}",
                0,
                plan.length_units_by_score_unit[f"score-unit-{key}"],
                0,
                "major",
            ),
        )
        for key in ("statement", "bridge")
    }
    prompt = harmony_prompt(
        document,
        plan,
        intent_index=4,
        previous_final_harmony=None,
        accepted_harmonies_by_score_unit=harmonies,
    )
    context = json.loads(prompt.split("入力: ", 1)[1])
    comparisons = context["target"]["section_comparison_contexts"]
    assert {value["relation_id"] for value in comparisons} == {"nested", "theme-varied"}
    full = next(value for value in comparisons if value["relation_id"] == "theme-varied")
    assert [leaf["section_id"] for leaf in full["source"]["leaves"]] == ["statement", "bridge"]
    assert (
        full["source"]["leaves"][1]["offset_units"]
        == plan.length_units_by_score_unit["score-unit-statement"]
    )
    assert all(leaf["harmonies"] for leaf in full["source"]["leaves"])
    assert len(full["target"]["leaves"]) == 3
    assert (
        full["relation"] == document["script"]["script_element_variation_relations"]["theme-varied"]
    )
    with pytest.raises(ValueError, match="not accepted"):
        section_harmony_contexts(
            document,
            "return-2",
            length_units_by_score_unit=plan.length_units_by_score_unit,
            accepted_harmonies_by_score_unit={},
        )


def test_unrelated_harmony_target_does_not_receive_the_section_comparison():
    assert (
        section_harmony_contexts(
            unequal_sections(),
            "release",
            length_units_by_score_unit={},
            accepted_harmonies_by_score_unit={},
        )
        == []
    )
