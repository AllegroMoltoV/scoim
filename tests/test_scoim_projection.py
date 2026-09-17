import copy

import jsonpointer

from scoim.operations import approve
from scoim.projection import PlanChoice, project_solo_piano_3m
from scoim.validation import IssueCode, check, content_sha256


def _approved_document(*, instrumentation: str = "solo_piano", duration: int = 180):
    draft = {
        "schema_version": "0.1.0",
        "document_id": "evening_return",
        "revision": 1,
        "status": "draft",
        "approval": None,
        "script": {
            "title": "夕暮れの回帰",
            "brief": "静かに始まり、主題へ戻って閉じる。",
            "performance_setup": {
                "instrumentation": instrumentation,
                "target_duration_seconds": duration,
                "performance_directions": {},
            },
            "root_section_id": "whole",
            "sections": {
                "whole": {
                    "parent_section_id": None,
                    "order": 0,
                    "role": "whole",
                    "description": "曲全体。",
                },
                "statement": {
                    "parent_section_id": "whole",
                    "order": 0,
                    "role": "statement",
                    "relative_length": 1,
                    "description": "主題を示す。",
                },
            },
            "materials": {
                "theme": {"kind": "theme", "description": "中心主題。"},
            },
            "placements": {
                "theme_first": {
                    "section_id": "statement",
                    "material_id": "theme",
                }
            },
            "variations": {},
            "requirements": {},
            "transitions": {},
        },
    }
    result = approve(draft)
    assert result.document is not None
    return result.document


def _plan_choice() -> PlanChoice:
    return PlanChoice(
        tonal_center=4,
        mode="minor",
        harmonic_focus_by_section={"whole": 4, "statement": 4},
    )


def _reapprove(document: dict[str, object]) -> dict[str, object]:
    draft = copy.deepcopy(document)
    draft["status"] = "draft"
    draft["approval"] = None
    script = draft["script"]
    assert isinstance(script, dict)
    script.setdefault("requirements", {})
    result = approve(draft)
    assert result.document is not None
    return result.document


def _approved_relational_document() -> dict[str, object]:
    document = _approved_document()
    draft = copy.deepcopy(document)
    draft["status"] = "draft"
    draft["approval"] = None
    script = draft["script"]
    assert isinstance(script, dict)
    script["sections"] = {
        "whole": {
            "parent_section_id": None,
            "order": 0,
            "role": "whole",
            "description": "曲全体。",
        },
        "statement": {
            "parent_section_id": "whole",
            "order": 0,
            "role": "statement",
            "relative_length": 4,
            "description": "主題を示す。",
        },
        "bridge": {
            "parent_section_id": "whole",
            "order": 1,
            "role": "transition",
            "relative_length": 1,
            "description": "対照部へつなぐ。",
        },
        "contrast": {
            "parent_section_id": "whole",
            "order": 2,
            "role": "contrast",
            "relative_length": 4,
            "description": "景色を開く。",
        },
        "return": {
            "parent_section_id": "whole",
            "order": 3,
            "role": "return",
            "relative_length": 4,
            "description": "主題へ戻る。",
        },
        "release": {
            "parent_section_id": "whole",
            "order": 4,
            "role": "release",
            "relative_length": 2,
            "description": "余韻を残して閉じる。",
        },
    }
    script["materials"] = {
        "theme": {"kind": "theme", "description": "夕暮れの中心主題。"},
        "connector": {"kind": "transition", "description": "短いつなぎ。"},
        "contrast": {"kind": "contrast", "description": "広がる対照素材。"},
        "cadence": {"kind": "ending", "description": "静かな終止素材。"},
    }
    script["placements"] = {
        "theme_first": {"section_id": "statement", "material_id": "theme"},
        "bridge_ab": {"section_id": "bridge", "material_id": "connector"},
        "contrast_first": {"section_id": "contrast", "material_id": "contrast"},
        "theme_return": {"section_id": "return", "material_id": "theme"},
        "cadence_last": {"section_id": "release", "material_id": "cadence"},
    }
    script["variations"] = {
        "return_variation": {
            "source": {"type": "section", "id": "statement"},
            "target": {"type": "section", "id": "return"},
            "preserve": ["主題の特徴"],
            "change": ["時間の揺れ"],
            "description": "主題を揃えて戻す。",
        }
    }
    script["transitions"] = {
        "statement_to_contrast": {
            "from_placement_id": "theme_first",
            "connector_placement_id": "bridge_ab",
            "to_placement_id": "contrast_first",
            "description": "主題から対照部へ滑らかにつなぐ。",
        }
    }
    setup = script["performance_setup"]
    assert isinstance(setup, dict)
    setup["performance_directions"] = {
        "quiet_opening": {
            "target": {"type": "section", "id": "statement"},
            "description": "強さを控え、時間を少し揺らす。",
        },
        "clear_return": {
            "target": {"type": "section", "id": "return"},
            "relative_to": {"type": "section", "id": "statement"},
            "description": "冒頭より打鍵を揃えて簡潔に進める。",
        },
    }
    result = approve(draft)
    assert result.document is not None
    legacy_approved = result.document
    legacy_script = legacy_approved["script"]
    assert isinstance(legacy_script, dict)
    legacy_script.pop("requirements")
    legacy_approved["approval"] = {"content_sha256": content_sha256(legacy_approved)}
    return legacy_approved


def _approved_relational_with_requirement() -> dict[str, object]:
    document = _approved_relational_document()
    draft = copy.deepcopy(document)
    draft["status"] = "draft"
    draft["approval"] = None
    draft["revision"] = 2
    script = draft["script"]
    assert isinstance(script, dict)
    script["requirements"] = {
        "return_more_aligned": {
            "performance_direction_id": "clear_return",
            "feature": "onset_alignment",
            "relation": "more",
        }
    }
    result = approve(draft)
    assert result.document is not None
    return result.document


def test_solo_piano_projection_rejects_an_unsupported_instrumentation() -> None:
    document = _approved_document(instrumentation="string_quartet")

    assert check(document).valid
    result = project_solo_piano_3m(document)

    assert not result.projected
    assert result.piece_plan is None
    assert result.issues[0].code is IssueCode.UNSUPPORTED_PROFILE
    assert result.issues[0].path == "/script/performance_setup/instrumentation"


def test_solo_piano_projection_rejects_an_unsupported_duration() -> None:
    document = _approved_document(duration=30)

    assert check(document).valid
    result = project_solo_piano_3m(document)

    assert not result.projected
    assert result.piece_plan is None
    assert result.issues[0].code is IssueCode.UNSUPPORTED_PROFILE
    assert result.issues[0].path == "/script/performance_setup/target_duration_seconds"


def test_solo_piano_projection_rejects_a_draft() -> None:
    approved = _approved_document()
    document = dict(approved)
    document["status"] = "draft"
    document["approval"] = None

    assert check(document).valid
    result = project_solo_piano_3m(document)

    assert not result.projected
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == "/status"


def test_solo_piano_projection_maps_the_section_tree_and_material_placement() -> None:
    document = _approved_document()

    result = project_solo_piano_3m(
        document,
        plan_choice=_plan_choice(),
    )

    assert result.projected
    assert result.issues == ()
    assert result.piece_plan is not None
    assert result.piece_plan.plan_id == "evening_return"
    assert result.piece_plan.title == "夕暮れの回帰"
    assert result.piece_plan.tonal_center == 4
    assert result.piece_plan.mode == "minor"
    assert result.piece_plan.ending_intent == "tonic"
    assert result.piece_plan.root_node_id == "whole"
    assert result.piece_plan.nodes[0].node_id == "whole"
    assert result.piece_plan.nodes[0].harmonic_focus == 4
    assert result.piece_plan.nodes[1].node_id == "statement"
    assert result.piece_plan.nodes[1].parent_id == "whole"
    assert result.piece_plan.nodes[1].duration_weight == 1
    assert result.piece_plan.nodes[1].score_material_id == "theme"
    targets = {target.source_path: target for target in result.targets}
    assert targets["/script/title"].generation_stages == ("piece_plan",)
    assert targets["/script/title"].verification == "direct_equality"
    assert targets["/script/brief"].generation_stages == (
        "piece_plan",
        "score_spec",
        "performance_spec",
    )
    assert targets["/script/brief"].verification == "translated_mechanical_claim"
    assert targets["/script/sections/statement/role"].value == "statement"
    assert targets["/script/sections/statement/description"].generation_stages == (
        "piece_plan",
        "score_spec",
        "performance_spec",
    )
    assert targets["/script/materials/theme/description"].generation_stages == ("score_spec",)
    assert targets["/script/placements/theme_first/material_id"].value == "theme"
    assert targets["/script/performance_setup/target_duration_seconds"].value == 180


def test_solo_piano_projection_adds_a_resolved_typed_requirement_target() -> None:
    document = _approved_relational_with_requirement()
    plan_choice = PlanChoice(
        tonal_center=4,
        mode="minor",
        harmonic_focus_by_section={
            "whole": 4,
            "statement": 4,
            "bridge": 5,
            "contrast": 7,
            "return": 4,
            "release": 4,
        },
        contrasts_with_by_section={"contrast": "statement"},
    )

    result = project_solo_piano_3m(document, plan_choice=plan_choice)

    assert result.projected is True
    target = next(
        item
        for item in result.targets
        if item.source_path == "/script/requirements/return_more_aligned"
    )
    assert target.generation_stages == ("rendered_performance",)
    assert target.verification == "typed_requirement"
    assert target.value == {
        "requirement_id": "return_more_aligned",
        "performance_direction_id": "clear_return",
        "target_section_id": "return",
        "reference_section_id": "statement",
        "feature": "onset_alignment",
        "relation": "more",
    }


def test_changing_a_section_description_changes_only_its_projection_target() -> None:
    original = _approved_document()
    changed = copy.deepcopy(original)
    script = changed["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    assert isinstance(sections, dict)
    statement = sections["statement"]
    assert isinstance(statement, dict)
    statement["description"] = "主題をためらいながら示す。"
    changed = _reapprove(changed)

    before = project_solo_piano_3m(original, plan_choice=_plan_choice())
    after = project_solo_piano_3m(changed, plan_choice=_plan_choice())

    assert before.piece_plan == after.piece_plan
    before_targets = {target.source_path: target.value for target in before.targets}
    after_targets = {target.source_path: target.value for target in after.targets}
    changed_paths = {path for path in before_targets if before_targets[path] != after_targets[path]}
    assert changed_paths == {"/script/sections/statement/description"}


def test_solo_piano_projection_preserves_fractional_leaf_length_ratios() -> None:
    document = _approved_document()
    changed = copy.deepcopy(document)
    script = changed["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    materials = script["materials"]
    placements = script["placements"]
    assert isinstance(sections, dict)
    assert isinstance(materials, dict)
    assert isinstance(placements, dict)
    statement = sections["statement"]
    assert isinstance(statement, dict)
    statement["relative_length"] = 0.5
    sections["release"] = {
        "parent_section_id": "whole",
        "order": 1,
        "role": "release",
        "relative_length": 0.75,
        "description": "余韻を残して閉じる。",
    }
    materials["release_theme"] = {
        "kind": "ending",
        "description": "終止専用素材。",
    }
    placements["theme_release"] = {
        "section_id": "release",
        "material_id": "release_theme",
    }
    changed = _reapprove(changed)

    result = project_solo_piano_3m(changed, plan_choice=_plan_choice())

    assert result.projected
    assert result.piece_plan is not None
    assert [node.duration_weight for node in result.piece_plan.nodes[1:]] == [2, 3]


def test_solo_piano_projection_rejects_reused_material_with_different_lengths() -> None:
    document = _approved_relational_document()
    changed = copy.deepcopy(document)
    script = changed["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    assert isinstance(sections, dict)
    returning = sections["return"]
    assert isinstance(returning, dict)
    returning["relative_length"] = 3
    changed = _reapprove(changed)

    result = project_solo_piano_3m(
        changed,
        plan_choice=PlanChoice(9, "minor", {}),
    )

    assert not result.projected
    assert result.issues[0].code is IssueCode.UNREPRESENTABLE
    assert result.issues[0].path == "/script/sections/return/relative_length"


def test_solo_piano_projection_rejects_section_variation_between_different_materials() -> None:
    document = _approved_relational_document()
    changed = copy.deepcopy(document)
    script = changed["script"]
    assert isinstance(script, dict)
    placements = script["placements"]
    assert isinstance(placements, dict)
    returning = placements["theme_return"]
    assert isinstance(returning, dict)
    returning["material_id"] = "contrast"
    changed = _reapprove(changed)

    result = project_solo_piano_3m(
        changed,
        plan_choice=PlanChoice(9, "minor", {}),
    )

    assert not result.projected
    assert result.issues[0].code is IssueCode.UNREPRESENTABLE
    assert result.issues[0].path == "/script/variations/return_variation/target"


def test_solo_piano_projection_rejects_multiple_placements_in_one_leaf() -> None:
    document = _approved_document()
    changed = copy.deepcopy(document)
    script = changed["script"]
    assert isinstance(script, dict)
    placements = script["placements"]
    assert isinstance(placements, dict)
    placements["theme_layer"] = {
        "section_id": "statement",
        "material_id": "theme",
    }
    changed = _reapprove(changed)

    assert check(changed).valid
    result = project_solo_piano_3m(changed, plan_choice=_plan_choice())

    assert not result.projected
    assert result.issues[0].code is IssueCode.UNREPRESENTABLE
    assert result.issues[0].path == "/script/sections/statement"


def test_solo_piano_projection_rejects_a_leaf_without_a_placement() -> None:
    document = _approved_document()
    changed = copy.deepcopy(document)
    script = changed["script"]
    assert isinstance(script, dict)
    script["placements"] = {}
    changed = _reapprove(changed)

    assert check(changed).valid
    result = project_solo_piano_3m(changed, plan_choice=_plan_choice())

    assert not result.projected
    assert result.issues[0].code is IssueCode.UNREPRESENTABLE
    assert result.issues[0].path == "/script/sections/statement"


def test_solo_piano_projection_rejects_a_role_outside_the_profile_vocabulary() -> None:
    document = _approved_document()
    changed = copy.deepcopy(document)
    script = changed["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    assert isinstance(sections, dict)
    statement = sections["statement"]
    assert isinstance(statement, dict)
    statement["role"] = "chorus"
    changed = _reapprove(changed)

    assert check(changed).valid
    result = project_solo_piano_3m(changed, plan_choice=_plan_choice())

    assert not result.projected
    assert result.issues[0].code is IssueCode.UNREPRESENTABLE
    assert result.issues[0].path == "/script/sections/statement/role"


def test_solo_piano_projection_maps_a_section_variation_to_derived_from() -> None:
    document = _approved_document()
    changed = copy.deepcopy(document)
    script = changed["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    placements = script["placements"]
    assert isinstance(sections, dict)
    assert isinstance(placements, dict)
    sections["return"] = {
        "parent_section_id": "whole",
        "order": 1,
        "role": "return",
        "relative_length": 1,
        "description": "主題を簡潔に戻す。",
    }
    placements["theme_return"] = {
        "section_id": "return",
        "material_id": "theme",
    }
    script["variations"] = {
        "return_variation": {
            "source": {"type": "section", "id": "statement"},
            "target": {"type": "section", "id": "return"},
            "preserve": ["主題の特徴"],
            "change": ["時間の揺れ"],
            "description": "同じ主題を、より揃えて戻す。",
        }
    }
    changed = _reapprove(changed)

    result = project_solo_piano_3m(changed, plan_choice=_plan_choice())

    assert result.projected
    assert result.piece_plan is not None
    return_node = next(node for node in result.piece_plan.nodes if node.node_id == "return")
    assert return_node.derived_from == "statement"


def test_solo_piano_projection_maps_a_placement_variation_to_derived_from() -> None:
    document = _approved_relational_document()
    changed = copy.deepcopy(document)
    script = changed["script"]
    assert isinstance(script, dict)
    variations = script["variations"]
    assert isinstance(variations, dict)
    variation = variations["return_variation"]
    assert isinstance(variation, dict)
    variation["source"] = {"type": "placement", "id": "theme_first"}
    variation["target"] = {"type": "placement", "id": "theme_return"}
    changed = _reapprove(changed)

    result = project_solo_piano_3m(
        changed,
        plan_choice=PlanChoice(
            tonal_center=4,
            mode="minor",
            harmonic_focus_by_section={},
        ),
    )

    assert result.projected
    assert result.piece_plan is not None
    return_node = next(node for node in result.piece_plan.nodes if node.node_id == "return")
    assert return_node.derived_from == "statement"


def test_solo_piano_projection_maps_a_frozen_contrast_choice() -> None:
    document = _approved_relational_document()
    choice = PlanChoice(
        tonal_center=9,
        mode="minor",
        harmonic_focus_by_section={},
        contrasts_with_by_section={"contrast": "statement"},
    )

    result = project_solo_piano_3m(document, plan_choice=choice)

    assert result.projected
    assert result.piece_plan is not None
    contrast = next(node for node in result.piece_plan.nodes if node.node_id == "contrast")
    assert contrast.contrasts_with == "statement"


def test_solo_piano_projection_rejects_an_unknown_contrast_section() -> None:
    document = _approved_relational_document()
    choice = PlanChoice(
        tonal_center=9,
        mode="minor",
        harmonic_focus_by_section={},
        contrasts_with_by_section={"unknown": "statement"},
    )

    result = project_solo_piano_3m(document, plan_choice=choice)

    assert not result.projected
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/plan_choice/contrasts_with_by_section/unknown"


def test_solo_piano_projection_rejects_a_choice_for_a_noncontrast_section() -> None:
    document = _approved_relational_document()
    choice = PlanChoice(
        tonal_center=9,
        mode="minor",
        harmonic_focus_by_section={},
        contrasts_with_by_section={"return": "statement"},
    )

    result = project_solo_piano_3m(document, plan_choice=choice)

    assert not result.projected
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/plan_choice/contrasts_with_by_section/return"


def test_solo_piano_projection_rejects_a_later_contrast_source() -> None:
    document = _approved_relational_document()
    choice = PlanChoice(
        tonal_center=9,
        mode="minor",
        harmonic_focus_by_section={},
        contrasts_with_by_section={"contrast": "return"},
    )

    result = project_solo_piano_3m(document, plan_choice=choice)

    assert not result.projected
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/plan_choice/contrasts_with_by_section/contrast"


def test_solo_piano_projection_rejects_a_contrast_source_at_another_depth() -> None:
    document = _approved_relational_document()
    choice = PlanChoice(
        tonal_center=9,
        mode="minor",
        harmonic_focus_by_section={},
        contrasts_with_by_section={"contrast": "whole"},
    )

    result = project_solo_piano_3m(document, plan_choice=choice)

    assert not result.projected
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/plan_choice/contrasts_with_by_section/contrast"


def test_solo_piano_projection_rejects_conflicting_variation_sources() -> None:
    document = _approved_relational_document()
    changed = copy.deepcopy(document)
    script = changed["script"]
    assert isinstance(script, dict)
    variations = script["variations"]
    assert isinstance(variations, dict)
    variations["conflicting_return"] = {
        "source": {"type": "placement", "id": "contrast_first"},
        "target": {"type": "placement", "id": "theme_return"},
        "preserve": ["輪郭"],
        "change": ["和声"],
        "description": "対照素材からも派生させる。",
    }
    changed = _reapprove(changed)

    result = project_solo_piano_3m(
        changed,
        plan_choice=PlanChoice(
            tonal_center=4,
            mode="minor",
            harmonic_focus_by_section={},
        ),
    )

    assert not result.projected
    assert result.issues[0].code is IssueCode.UNREPRESENTABLE
    assert result.issues[0].path.endswith("/target")


def test_solo_piano_projection_rejects_a_plan_choice_for_an_unknown_section() -> None:
    document = _approved_document()
    choice = PlanChoice(
        tonal_center=4,
        mode="minor",
        harmonic_focus_by_section={"whole": 4, "statement": 4, "unknown": 7},
    )

    result = project_solo_piano_3m(document, plan_choice=choice)

    assert not result.projected
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/plan_choice/harmonic_focus_by_section/unknown"


def test_solo_piano_projection_rejects_an_invalid_plan_choice_mode() -> None:
    document = _approved_document()
    choice = PlanChoice(
        tonal_center=4,
        mode="dorian",
        harmonic_focus_by_section={"whole": 4, "statement": 4},
    )

    result = project_solo_piano_3m(document, plan_choice=choice)

    assert not result.projected
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/plan_choice/mode"


def test_solo_piano_projection_rejects_an_invalid_tonal_center() -> None:
    document = _approved_document()
    choice = PlanChoice(
        tonal_center=12,
        mode="minor",
        harmonic_focus_by_section={"whole": 4, "statement": 4},
    )

    result = project_solo_piano_3m(document, plan_choice=choice)

    assert not result.projected
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/plan_choice/tonal_center"


def test_solo_piano_projection_rejects_an_invalid_harmonic_focus() -> None:
    document = _approved_document()
    choice = PlanChoice(
        tonal_center=4,
        mode="minor",
        harmonic_focus_by_section={"whole": 4, "statement": 12},
    )

    result = project_solo_piano_3m(document, plan_choice=choice)

    assert not result.projected
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/plan_choice/harmonic_focus_by_section/statement"


def test_solo_piano_projection_reports_a_missing_frozen_plan_choice() -> None:
    document = _approved_document()

    result = project_solo_piano_3m(document)

    assert not result.projected
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/plan_choice"


def test_solo_piano_projection_rejects_a_return_without_a_variation_source() -> None:
    document = _approved_document()
    changed = copy.deepcopy(document)
    script = changed["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    assert isinstance(sections, dict)
    statement = sections["statement"]
    assert isinstance(statement, dict)
    statement["role"] = "return"
    changed = _reapprove(changed)

    assert check(changed).valid
    result = project_solo_piano_3m(changed, plan_choice=_plan_choice())

    assert not result.projected
    assert result.issues[0].code is IssueCode.UNREPRESENTABLE
    assert result.issues[0].path == "/script/sections/statement/role"


def test_relational_script_projects_every_intent_category_without_cross_talk() -> None:
    original = _approved_relational_document()
    choice = PlanChoice(4, "minor", {})
    before = project_solo_piano_3m(original, plan_choice=choice)
    assert before.projected

    changed_fields = {
        "/script/materials/theme/description": "問いかける中心主題。",
        "/script/variations/return_variation/change": ["打鍵の揃い方"],
        "/script/transitions/statement_to_contrast/description": "短く区切って対照部へ移る。",
        (
            "/script/performance_setup/performance_directions/clear_return/description"
        ): "冒頭より軽く、縦を揃えて進める。",
    }
    before_targets = {target.source_path: target.value for target in before.targets}
    for path, value in changed_fields.items():
        changed = copy.deepcopy(original)
        changed["status"] = "draft"
        changed["approval"] = None
        jsonpointer.set_pointer(changed, path, value, inplace=True)
        changed = _reapprove(changed)

        after = project_solo_piano_3m(changed, plan_choice=choice)
        assert after.projected
        assert after.piece_plan == before.piece_plan
        after_targets = {target.source_path: target.value for target in after.targets}
        changed_paths = {
            target_path
            for target_path in before_targets
            if before_targets[target_path] != after_targets[target_path]
        }
        assert changed_paths == {path}


def test_solo_piano_projection_rejects_a_derivation_across_section_depths() -> None:
    document = _approved_document()
    changed = copy.deepcopy(document)
    script = changed["script"]
    assert isinstance(script, dict)
    sections = script["sections"]
    placements = script["placements"]
    assert isinstance(sections, dict)
    assert isinstance(placements, dict)
    sections["return_group"] = {
        "parent_section_id": "whole",
        "order": 1,
        "role": "variation",
        "description": "主題を別の粒度で扱う。",
    }
    sections["return_detail"] = {
        "parent_section_id": "return_group",
        "order": 0,
        "role": "variation",
        "relative_length": 1,
        "description": "細部を変える。",
    }
    placements["theme_detail"] = {
        "section_id": "return_detail",
        "material_id": "theme",
    }
    script["variations"] = {
        "depth_mismatch": {
            "source": {"type": "section", "id": "statement"},
            "target": {"type": "section", "id": "return_detail"},
            "preserve": ["主題"],
            "change": ["細部"],
            "description": "別の階層へ派生する。",
        }
    }
    changed = _reapprove(changed)

    assert check(changed).valid
    result = project_solo_piano_3m(changed, plan_choice=_plan_choice())

    assert not result.projected
    assert result.issues[0].code is IssueCode.UNREPRESENTABLE
    assert result.issues[0].path == "/script/variations/depth_mismatch/target"
