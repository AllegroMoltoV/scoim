import copy

from scoim.script_0_3_validation import (
    check_script_0_3_document,
    script_0_3_content_sha256,
)
from scoim.validation import IssueCode


def _valid_document() -> dict[str, object]:
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
            "brief": "前景と伴奏を同じ区分で併用する。",
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


def _with_bridge_and_return(document: dict[str, object]) -> dict[str, object]:
    sections = document["script"]["sections"]
    sections["bridge"] = {
        "parent_section_id": "whole",
        "order": 1,
        "role": "transition",
        "relative_length": 0.25,
        "description": "次の区分へ移る。",
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
    return document


def _add_comparative_direction(
    document: dict[str, object],
    *,
    direction_id: str,
    target_type: str,
    target_id: str,
    relative_id: str,
) -> None:
    document["script"]["performance_setup"]["performance_directions"][direction_id] = {
        "target": {"type": target_type, "id": target_id},
        "relative_to": {"type": target_type, "id": relative_id},
        "description": "比較できる演奏指示。",
    }


def _add_comparison_requirement(
    document: dict[str, object],
    *,
    requirement_id: str,
    direction_id: str,
    feature: str = "loudness",
    relation: str = "more",
) -> None:
    document["script"]["performance_direction_comparison_requirements"][requirement_id] = {
        "performance_direction_id": direction_id,
        "feature": feature,
        "relation": relation,
    }


def test_check_script_0_3_document_accepts_explicit_placement_roles() -> None:
    result = check_script_0_3_document(_valid_document())

    assert result.valid is True
    assert result.issues == ()


def test_script_0_3_content_hash_ignores_validation_state() -> None:
    document = _valid_document()
    expected = script_0_3_content_sha256(document)
    document["status"] = "another-state"

    assert script_0_3_content_sha256(document) == expected


def test_check_script_0_3_document_requires_a_placement_role() -> None:
    document = copy.deepcopy(_valid_document())
    del document["script"]["material_placements"]["theme-first"]["role"]

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID


def test_check_script_0_3_document_accepts_a_profile_defined_role_identifier() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["material_placements"]["theme-first"]["role"] = "counterline"

    result = check_script_0_3_document(document)

    assert result.valid is True


def test_check_script_0_3_document_rejects_material_kind() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["materials"]["theme"]["kind"] = "theme"

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID


def test_check_script_0_3_document_rejects_placement_order() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["material_placements"]["theme-first"]["order"] = 0

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID


def test_check_script_0_3_document_rejects_legacy_collection_names() -> None:
    document = copy.deepcopy(_valid_document())
    script = document["script"]
    script["placements"] = script.pop("material_placements")

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any(issue.code is IssueCode.SCHEMA_INVALID for issue in result.issues)


def test_check_script_0_3_document_rejects_a_missing_material_reference() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["material_placements"]["theme-first"]["material_id"] = "missing"

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/material_placements/theme-first/material_id"
        for issue in result.issues
    )


def test_check_script_0_3_document_rejects_a_missing_parent_section() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["sections"]["statement"]["parent_section_id"] = "missing"

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/sections/statement/parent_section_id"
        for issue in result.issues
    )


def test_check_script_0_3_document_rejects_a_missing_variation_target() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["script_element_variation_relations"]["theme-change"] = {
        "source": {"type": "material", "id": "theme"},
        "target": {"type": "material", "id": "missing"},
        "preserve": ["冒頭の輪郭"],
        "change": ["リズム"],
        "description": "主題を動かす。",
    }

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/script_element_variation_relations/theme-change/target/id"
        for issue in result.issues
    )


def test_check_script_0_3_document_rejects_a_missing_transition_placement() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["material_placement_transitions"]["to-return"] = {
        "source_material_placement_id": "theme-first",
        "transition_material_placement_id": "support-first",
        "target_material_placement_id": "missing",
        "description": "再提示へ移る。",
    }

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path
        == "/script/material_placement_transitions/to-return/target_material_placement_id"
        for issue in result.issues
    )


def test_check_script_0_3_document_rejects_a_missing_performance_target() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["performance_setup"]["performance_directions"]["quiet"] = {
        "target": {"type": "section", "id": "missing"},
        "description": "静かに演奏する。",
    }

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/performance_setup/performance_directions/quiet/target/id"
        for issue in result.issues
    )


def test_check_script_0_3_document_rejects_a_missing_comparison_direction() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["performance_direction_comparison_requirements"]["louder"] = {
        "performance_direction_id": "missing",
        "feature": "loudness",
        "relation": "more",
    }

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path
        == ("/script/performance_direction_comparison_requirements/louder/performance_direction_id")
        for issue in result.issues
    )


def test_check_script_0_3_document_rejects_a_reversed_transition() -> None:
    document = _with_bridge_and_return(copy.deepcopy(_valid_document()))
    document["script"]["material_placement_transitions"]["reversed"] = {
        "source_material_placement_id": "theme-return",
        "transition_material_placement_id": "bridge-only",
        "target_material_placement_id": "theme-first",
        "description": "逆順の遷移。",
    }

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path
        == ("/script/material_placement_transitions/reversed/transition_material_placement_id")
        for issue in result.issues
    )


def test_check_script_0_3_document_rejects_a_reversed_section_variation() -> None:
    document = _with_bridge_and_return(copy.deepcopy(_valid_document()))
    document["script"]["script_element_variation_relations"]["reversed"] = {
        "source": {"type": "section", "id": "return"},
        "target": {"type": "section", "id": "statement"},
        "preserve": ["主題"],
        "change": ["演奏"],
        "description": "逆順の変奏。",
    }

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/script_element_variation_relations/reversed/target/id"
        for issue in result.issues
    )


def test_check_script_0_3_document_rejects_an_unsupported_version() -> None:
    document = copy.deepcopy(_valid_document())
    document["schema_version"] = "0.4.0"

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.UNSUPPORTED_SCHEMA_VERSION


def test_check_script_0_3_document_rejects_nonconsecutive_sibling_order() -> None:
    document = _with_bridge_and_return(copy.deepcopy(_valid_document()))
    document["script"]["sections"]["bridge"]["order"] = 2

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID and issue.path == "/script/sections/bridge/order"
        for issue in result.issues
    )


def test_check_script_0_3_document_rejects_a_missing_root_section() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["root_section_id"] = "missing"

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any(issue.path == "/script/root_section_id" for issue in result.issues)


def test_check_script_0_3_document_rejects_a_root_with_a_parent() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["sections"]["whole"]["parent_section_id"] = "statement"

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any(issue.path == "/script/sections/whole/parent_section_id" for issue in result.issues)


def test_check_script_0_3_document_rejects_a_second_parentless_section() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["sections"]["statement"]["parent_section_id"] = None

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any(
        issue.path == "/script/sections/statement/parent_section_id" for issue in result.issues
    )


def test_check_script_0_3_document_rejects_length_on_a_branch_section() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["sections"]["whole"]["relative_length"] = 1

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any(issue.path == "/script/sections/whole/relative_length" for issue in result.issues)


def test_check_script_0_3_document_rejects_a_placement_on_a_branch_section() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["material_placements"]["theme-first"]["section_id"] = "whole"

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/material_placements/theme-first/section_id"
        for issue in result.issues
    )


def test_check_script_0_3_document_rejects_a_missing_placement_section() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["material_placements"]["theme-first"]["section_id"] = "missing"

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any(
        issue.path == "/script/material_placements/theme-first/section_id"
        for issue in result.issues
    )


def test_check_script_0_3_document_rejects_mixed_variation_reference_types() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["script_element_variation_relations"]["mixed"] = {
        "source": {"type": "material", "id": "theme"},
        "target": {"type": "section", "id": "statement"},
        "preserve": ["輪郭"],
        "change": ["リズム"],
        "description": "異なる種類を変奏として結ぶ。",
    }

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any("same type" in issue.message for issue in result.issues)


def test_check_script_0_3_document_rejects_a_self_variation() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["script_element_variation_relations"]["self"] = {
        "source": {"type": "material", "id": "theme"},
        "target": {"type": "material", "id": "theme"},
        "preserve": ["輪郭"],
        "change": ["リズム"],
        "description": "同じ要素を変奏先にする。",
    }

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any("different IDs" in issue.message for issue in result.issues)


def test_check_script_0_3_document_rejects_duplicate_variation_targets() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["materials"].update(
        {
            "theme-return": {"description": "主題の再現。"},
            "other": {"description": "別の素材。"},
        }
    )
    relations = document["script"]["script_element_variation_relations"]
    relations["from-theme"] = {
        "source": {"type": "material", "id": "theme"},
        "target": {"type": "material", "id": "theme-return"},
        "preserve": ["輪郭"],
        "change": ["リズム"],
        "description": "主題から変える。",
    }
    relations["from-other"] = {
        "source": {"type": "material", "id": "other"},
        "target": {"type": "material", "id": "theme-return"},
        "preserve": ["輪郭"],
        "change": ["リズム"],
        "description": "別素材から同じ先へ変える。",
    }

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any("target must be unique" in issue.message for issue in result.issues)


def test_check_script_0_3_document_rejects_overlapping_variation_instructions() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["materials"]["theme-return"] = {"description": "主題の再現。"}
    document["script"]["script_element_variation_relations"]["theme-change"] = {
        "source": {"type": "material", "id": "theme"},
        "target": {"type": "material", "id": "theme-return"},
        "preserve": ["輪郭"],
        "change": ["輪郭"],
        "description": "同じ特徴を維持と変更の両方に指定する。",
    }

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/script_element_variation_relations/theme-change/change/0"
        for issue in result.issues
    )


def test_check_script_0_3_document_rejects_a_material_variation_cycle() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["materials"]["theme-return"] = {"description": "主題の再現。"}
    relations = document["script"]["script_element_variation_relations"]
    relations["outward"] = {
        "source": {"type": "material", "id": "theme"},
        "target": {"type": "material", "id": "theme-return"},
        "preserve": ["輪郭"],
        "change": ["リズム"],
        "description": "主題を変える。",
    }
    relations["backward"] = {
        "source": {"type": "material", "id": "theme-return"},
        "target": {"type": "material", "id": "theme"},
        "preserve": ["輪郭"],
        "change": ["リズム"],
        "description": "変奏元へ戻す。",
    }

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any("cycle" in issue.message for issue in result.issues)


def test_check_script_0_3_document_rejects_a_shared_transition_section() -> None:
    document = _with_bridge_and_return(copy.deepcopy(_valid_document()))
    document["script"]["material_placements"]["support-bridge"] = {
        "section_id": "bridge",
        "material_id": "support",
        "role": "accompaniment",
    }
    document["script"]["material_placement_transitions"]["to-return"] = {
        "source_material_placement_id": "theme-first",
        "transition_material_placement_id": "bridge-only",
        "target_material_placement_id": "theme-return",
        "description": "専用でない区分を遷移に使う。",
    }

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path
        == ("/script/material_placement_transitions/to-return/transition_material_placement_id")
        for issue in result.issues
    )


def test_check_script_0_3_document_rejects_reused_transition_placements() -> None:
    document = _with_bridge_and_return(copy.deepcopy(_valid_document()))
    document["script"]["material_placement_transitions"]["reused"] = {
        "source_material_placement_id": "theme-first",
        "transition_material_placement_id": "bridge-only",
        "target_material_placement_id": "bridge-only",
        "description": "遷移と遷移先に同じ配置を使う。",
    }

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any("must be distinct" in issue.message for issue in result.issues)


def test_check_script_0_3_document_rejects_a_noncomparative_required_direction() -> None:
    document = copy.deepcopy(_valid_document())
    document["script"]["performance_setup"]["performance_directions"]["louder"] = {
        "target": {"type": "section", "id": "statement"},
        "description": "より強く演奏する。",
    }
    _add_comparison_requirement(
        document,
        requirement_id="louder-required",
        direction_id="louder",
    )

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any("comparative performance direction" in issue.message for issue in result.issues)


def test_check_script_0_3_document_rejects_mixed_comparison_target_types() -> None:
    document = copy.deepcopy(_valid_document())
    directions = document["script"]["performance_setup"]["performance_directions"]
    directions["mixed"] = {
        "target": {"type": "section", "id": "statement"},
        "relative_to": {"type": "material_placement", "id": "theme-first"},
        "description": "異なる種類を比較する。",
    }
    _add_comparison_requirement(
        document,
        requirement_id="mixed-required",
        direction_id="mixed",
    )

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any("same target type" in issue.message for issue in result.issues)


def test_check_script_0_3_document_rejects_ancestor_descendant_comparison() -> None:
    document = copy.deepcopy(_valid_document())
    _add_comparative_direction(
        document,
        direction_id="child-louder",
        target_type="section",
        target_id="statement",
        relative_id="whole",
    )
    _add_comparison_requirement(
        document,
        requirement_id="child-louder-required",
        direction_id="child-louder",
    )

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any("ancestor and descendant" in issue.message for issue in result.issues)


def test_check_script_0_3_document_rejects_a_comparison_cycle() -> None:
    document = _with_bridge_and_return(copy.deepcopy(_valid_document()))
    _add_comparative_direction(
        document,
        direction_id="return-louder",
        target_type="section",
        target_id="return",
        relative_id="statement",
    )
    _add_comparative_direction(
        document,
        direction_id="statement-louder",
        target_type="section",
        target_id="statement",
        relative_id="return",
    )
    _add_comparison_requirement(
        document,
        requirement_id="return-louder-required",
        direction_id="return-louder",
    )
    _add_comparison_requirement(
        document,
        requirement_id="statement-louder-required",
        direction_id="statement-louder",
    )

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any("comparison cycle" in issue.message for issue in result.issues)


def test_check_script_0_3_document_rejects_the_same_comparison_target_twice() -> None:
    document = copy.deepcopy(_valid_document())
    _add_comparative_direction(
        document,
        direction_id="same-target",
        target_type="section",
        target_id="statement",
        relative_id="statement",
    )
    _add_comparison_requirement(
        document,
        requirement_id="same-target-required",
        direction_id="same-target",
    )

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any("different targets" in issue.message for issue in result.issues)


def test_check_script_0_3_document_rejects_duplicate_direction_feature_requirements() -> None:
    document = _with_bridge_and_return(copy.deepcopy(_valid_document()))
    _add_comparative_direction(
        document,
        direction_id="return-louder",
        target_type="section",
        target_id="return",
        relative_id="statement",
    )
    _add_comparison_requirement(
        document,
        requirement_id="first",
        direction_id="return-louder",
    )
    _add_comparison_requirement(
        document,
        requirement_id="second",
        direction_id="return-louder",
    )

    result = check_script_0_3_document(document)

    assert result.valid is False
    assert any("can only be required once" in issue.message for issue in result.issues)
