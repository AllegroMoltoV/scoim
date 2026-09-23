import copy
import json
from pathlib import Path

import pytest

from scoim.score_work_plan import build_score_work_plan, section_leaf_ids


def document():
    path = (
        Path(__file__).parent
        / "fixtures/scoim/score-unit-layer-vertical/basic-validated-script.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value["script"]["script_element_variation_relations"]["theme-varied"].update(
        source={"type": "section", "id": "statement"},
        target={"type": "section", "id": "return"},
    )
    return value


def operation_for(plan, placement_id):
    return next(
        operation
        for operation in plan.operations
        if placement_id in operation.material_placement_ids
    )


def test_source_accompaniment_is_ready_before_target_foreground_and_target_is_atomic():
    plan = build_score_work_plan(document())
    assert not plan.issues
    source = operation_for(plan, "support-first")
    target = operation_for(plan, "theme-return")
    assert set(target.material_placement_ids) == {"theme-return", "support-return"}
    assert source.operation_id in target.dependency_operation_ids
    assert plan.operations.index(source) < plan.operations.index(target)
    assert sorted(
        key for operation in plan.operations for key in operation.material_placement_ids
    ) == sorted(document()["script"]["material_placements"])


@pytest.mark.parametrize("removed", ["theme-first", "support-return"])
def test_section_variation_allows_role_addition_and_removal(removed):
    value = document()
    del value["script"]["material_placements"][removed]
    value["script"]["material_placement_transitions"] = {}
    plan = build_score_work_plan(value)
    assert not plan.issues
    comparison = next(item for item in plan.comparisons if item.kind == "section_variation")
    assert "support-first" in comparison.source_placement_ids
    assert "theme-return" in comparison.target_placement_ids
    assert removed not in comparison.source_placement_ids + comparison.target_placement_ids


def test_transition_cycle_is_one_complete_candidate_instead_of_partial_order():
    value = document()
    value["script"]["script_element_variation_relations"]["theme-varied"]["source"]["id"] = "bridge"
    plan = build_score_work_plan(value)
    assert not plan.issues
    assert set(operation_for(plan, "bridge-only").material_placement_ids) == {
        "bridge-only",
        "theme-return",
        "support-return",
    }
    assert (
        len({key for operation in plan.operations for key in operation.material_placement_ids}) == 6
    )


def test_material_and_section_cycle_keeps_every_role_in_the_same_candidate():
    value = document()
    script = value["script"]
    script["script_element_variation_relations"]["theme-varied"]["source"]["id"] = "bridge"
    script["materials"]["return-theme"] = {"description": "Returning theme"}
    script["material_placements"]["theme-return"]["material_id"] = "return-theme"
    script["script_element_variation_relations"]["back-reference"] = {
        "source": {"type": "material", "id": "return-theme"},
        "target": {"type": "material", "id": "theme"},
        "description": "Related contours",
        "preserve": ["contour"],
        "change": ["rhythm"],
    }
    plan = build_score_work_plan(value)
    assert not plan.issues
    assert set(operation_for(plan, "support-first").material_placement_ids) == {
        "theme-first",
        "support-first",
        "bridge-only",
        "theme-return",
        "support-return",
    }


def unequal_sections():
    value = document()
    script = value["script"]
    script["material_placement_transitions"] = {}
    sections = script["sections"]
    sections["opening"] = {
        "parent_section_id": "whole",
        "order": 0,
        "role": "scene",
        "description": "Opening",
    }
    sections["statement"]["parent_section_id"] = "opening"
    sections["bridge"]["parent_section_id"] = "opening"
    sections["return"]["order"] = 1
    del sections["return"]["relative_length"]
    sections["release"]["order"] = 2
    for index in range(3):
        sections[f"return-{index}"] = {
            "parent_section_id": "return",
            "order": index,
            "role": "phrase",
            "relative_length": index + 1,
            "description": "A returning phrase",
        }
    placements = script["material_placements"]
    placements["theme-return"]["section_id"] = "return-0"
    placements["support-return"]["section_id"] = "return-0"
    placements["extra-theme"] = {
        "section_id": "return-1",
        "material_id": "theme",
        "role": "foreground",
    }
    placements["extra-support"] = {
        "section_id": "return-2",
        "material_id": "support",
        "role": "accompaniment",
    }
    script["script_element_variation_relations"]["theme-varied"]["source"]["id"] = "opening"
    script["script_element_variation_relations"]["nested"] = {
        "source": {"type": "section", "id": "statement"},
        "target": {"type": "section", "id": "return-2"},
        "description": "Nested relation",
        "preserve": ["harmony"],
        "change": ["texture"],
    }
    return value


def test_different_leaf_counts_and_nested_targets_preserve_complete_ranges():
    value = unequal_sections()
    plan = build_score_work_plan(value)
    assert not plan.issues
    assert len(section_leaf_ids(value, "opening")) == 2
    assert len(section_leaf_ids(value, "return")) == 3
    target = operation_for(plan, "extra-support")
    assert set(target.material_placement_ids) == {
        "theme-return",
        "support-return",
        "extra-theme",
        "extra-support",
    }
    assert {item.relation_id for item in plan.comparisons if item.kind == "section_variation"} == {
        "theme-varied",
        "nested",
    }
    reordered = copy.deepcopy(value)
    for key in (
        "sections",
        "materials",
        "material_placements",
        "script_element_variation_relations",
    ):
        reordered["script"][key] = dict(reversed(list(reordered["script"][key].items())))
    assert build_score_work_plan(reordered) == plan


def test_explicit_placement_role_crossing_remains_a_separate_unsupported_capability():
    value = document()
    relation = value["script"]["script_element_variation_relations"]["theme-varied"]
    relation["source"] = {"type": "material_placement", "id": "support-first"}
    relation["target"] = {"type": "material_placement", "id": "theme-return"}
    plan = build_score_work_plan(value)
    assert len(plan.issues) == 1
    assert plan.issues[0].code.value == "unrepresentable"
    assert not plan.operations
