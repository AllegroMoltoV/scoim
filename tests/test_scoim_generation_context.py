import copy
import json
from pathlib import Path

import pytest
from test_scoim_phase3_model_contracts import _document, _overall_response
from test_scoim_phase4_realization import (
    SequencedRunner,
    _phase3_inputs,
    _phase4_responses,
)

from scoim.phase3_model_contracts import (
    build_harmonic_plan,
    harmony_prompt,
    overall_plan_prompt,
)
from scoim.phase4_model_contracts import build_foreground_operations, foreground_prompt
from scoim.phase4_realization import Phase4Request, realize_phase4
from scoim.phase5_model_contracts import accompaniment_prompt, build_accompaniment_operations
from scoim.score_ir import ScoreHarmony, ScoreNote


def _nested_document() -> dict:
    document = _document()
    sections = document["script"]["sections"]
    sections["whole"]["description"] = "全曲を通して夜明け前の静けさを保つ。"
    sections["first-scene"] = {
        "parent_section_id": "whole",
        "order": 0,
        "role": "opening-scene",
        "description": "冒頭の場面全体で低い音域を中心にする。",
    }
    sections["statement"]["parent_section_id"] = "first-scene"
    sections["statement"]["description"] = "短い問いかけを置く。"
    sections["last-scene"] = {
        "parent_section_id": "whole",
        "order": 3,
        "role": "closing-scene",
        "description": "最後の場面だけで響きを明るく広げる。",
    }
    sections["release"]["parent_section_id"] = "last-scene"
    sections["release"]["order"] = 0
    return document


def _context(prompt: str) -> dict:
    return json.loads(prompt.split("入力: ", 1)[1])


def _section_context(document: dict, section_id: str) -> dict:
    section = document["script"]["sections"][section_id]
    return {
        "section_id": section_id,
        **{key: section[key] for key in ("parent_section_id", "order", "role", "description")},
    }


def _local_prompt(document: dict, kind: str) -> str:
    plan = build_harmonic_plan(document, _overall_response(), tonal_center=0, divisions=12)
    harmonies = {
        unit_id: (ScoreHarmony(f"h-{unit_id}", 0, length, 0, "major"),)
        for unit_id, length in plan.length_units_by_score_unit.items()
    }
    if kind == "harmony":
        return harmony_prompt(document, plan, intent_index=0, previous_final_harmony=None)
    if kind == "foreground":
        operation = build_foreground_operations(document)[0]
        return foreground_prompt(
            document,
            plan,
            operation,
            harmonies_by_score_unit=harmonies,
            accepted_notes_by_material_placement={},
        )
    operation = build_accompaniment_operations(document)[0]
    return accompaniment_prompt(
        document,
        plan,
        operation,
        harmonies_by_score_unit=harmonies,
        foreground_notes=(),
        accepted_responses={},
        accepted_notes={},
    )


def test_overall_context_preserves_all_section_descriptions_and_hierarchy_order() -> None:
    document = _nested_document()
    script = document["script"]
    script["sections"] = dict(reversed(list(script["sections"].items())))

    context = _context(overall_plan_prompt(document, tonal_center=0))

    assert context["sections_in_hierarchy_order"] == [
        _section_context(document, section_id)
        for section_id in (
            "whole",
            "first-scene",
            "statement",
            "bridge",
            "return",
            "last-scene",
            "release",
        )
    ]
    assert [item["section_id"] for item in context["leaf_sections_in_performance_order"]] == [
        "statement",
        "bridge",
        "return",
        "release",
    ]


@pytest.mark.parametrize("kind", ("harmony", "foreground", "accompaniment"))
def test_local_context_delivers_only_target_ancestors_in_root_to_parent_order(kind: str) -> None:
    document = _nested_document()
    prompt = _local_prompt(document, kind)
    target = _context(prompt)["target"]

    assert target["ancestor_sections"] == [
        _section_context(document, "whole"),
        _section_context(document, "first-scene"),
    ]
    assert (
        target["section_description"] == document["script"]["sections"]["statement"]["description"]
    )
    assert document["script"]["sections"]["last-scene"]["description"] not in prompt
    if kind == "foreground":
        assert target["transition_context"] is None


@pytest.mark.parametrize("kind", ("harmony", "foreground", "accompaniment"))
@pytest.mark.parametrize("ancestor_id", ("whole", "first-scene"))
def test_ancestor_edit_changes_local_request_but_unrelated_branch_edit_does_not(
    kind: str, ancestor_id: str
) -> None:
    original = _nested_document()
    changed_ancestor = copy.deepcopy(original)
    changed_ancestor["script"]["sections"][ancestor_id]["description"] = "細かな上行形を重ねる。"
    changed_other_branch = copy.deepcopy(original)
    changed_other_branch["script"]["sections"]["last-scene"]["description"] = "静かに閉じる。"

    original_prompt = _local_prompt(original, kind)

    assert _local_prompt(changed_ancestor, kind) != original_prompt
    assert _local_prompt(changed_other_branch, kind) == original_prompt


def test_transition_context_delivers_original_relation_and_keeps_boundary_notes_only() -> None:
    document = _nested_document()
    relation = document["script"]["material_placement_transitions"]["to-return"]
    relation["description"] = "上行をいったん止め、次の主題へ静かにつなぐ。"
    plan = build_harmonic_plan(document, _overall_response(), tonal_center=0, divisions=12)
    operation = build_foreground_operations(document)[-1]
    source = (
        ScoreNote("earlier", 0, 6, 55, "lower"),
        ScoreNote("last-upper", 36, 12, 71, "upper"),
        ScoreNote("last-lower", 42, 6, 59, "lower"),
    )
    destination = (
        ScoreNote("first-upper", 0, 12, 72, "upper"),
        ScoreNote("first-lower", 0, 6, 60, "lower"),
        ScoreNote("later", 12, 6, 74, "upper"),
    )

    prompt = foreground_prompt(
        document,
        plan,
        operation,
        harmonies_by_score_unit={
            operation.score_unit_id: (ScoreHarmony("bridge-harmony", 0, 12, 7, "major"),)
        },
        accepted_notes_by_material_placement={
            "theme-first": source,
            "theme-return": destination,
        },
    )
    target = _context(prompt)["target"]

    assert target["transition_context"] == {"transition_id": "to-return", **relation}
    comparisons = target["comparison_contexts"]
    assert [item["kind"] for item in comparisons] == ["transition_source", "transition_target"]
    assert comparisons[0]["notes"] == [
        {"at_units": 36, "duration_units": 12, "pitch": 71, "voice": "upper"},
        {"at_units": 42, "duration_units": 6, "pitch": 59, "voice": "lower"},
    ]
    assert comparisons[1]["notes"] == [
        {"at_units": 0, "duration_units": 12, "pitch": 72, "voice": "upper"},
        {"at_units": 0, "duration_units": 6, "pitch": 60, "voice": "lower"},
    ]


def test_content_repairs_retain_original_ancestor_and_transition_context(tmp_path: Path) -> None:
    document = _nested_document()
    state, ledger = _phase3_inputs(tmp_path, document)
    responses = _phase4_responses()
    runner = SequencedRunner([{}, *responses[:3], {}, responses[3]])

    result = realize_phase4(Phase4Request(document, state, ledger), runner, tmp_path / "phase4")

    assert result.realized
    assert len(runner.prompts) == 6
    for original_index, repair_index in ((0, 1), (4, 5)):
        repair = _context(runner.prompts[repair_index])
        assert repair["original_prompt"] == runner.prompts[original_index]
    first_target = _context(_context(runner.prompts[1])["original_prompt"])["target"]
    assert first_target["ancestor_sections"] == [
        _section_context(document, "whole"),
        _section_context(document, "first-scene"),
    ]
    transition_target = _context(_context(runner.prompts[5])["original_prompt"])["target"]
    assert transition_target["transition_context"] == {
        "transition_id": "to-return",
        **document["script"]["material_placement_transitions"]["to-return"],
    }
