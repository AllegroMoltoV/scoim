from copy import deepcopy

from scoim.query import show
from scoim.validation import IssueCode


def _document_with_repeated_material() -> dict[str, object]:
    return {
        "schema_version": "0.1.0",
        "document_id": "repeated_theme",
        "revision": 1,
        "status": "draft",
        "approval": None,
        "script": {
            "title": "主題の再提示",
            "brief": "同じ主題を二度提示する。",
            "performance_setup": {
                "instrumentation": "solo_piano",
                "target_duration_seconds": 30,
                "performance_directions": {
                    "linger": {
                        "target": {"type": "section", "id": "z_first"},
                        "description": "少し間を取る。",
                    },
                    "clear_first": {
                        "target": {"type": "placement", "id": "theme_first"},
                        "relative_to": {"type": "placement", "id": "theme_return"},
                        "description": "旋律を明確にする。",
                    },
                },
            },
            "root_section_id": "whole",
            "sections": {
                "whole": {
                    "parent_section_id": None,
                    "order": 0,
                    "role": "whole",
                    "description": "曲全体",
                },
                "z_first": {
                    "parent_section_id": "whole",
                    "order": 0,
                    "role": "statement",
                    "relative_length": 1,
                    "description": "最初の提示。",
                },
                "a_return": {
                    "parent_section_id": "whole",
                    "order": 2,
                    "role": "return",
                    "relative_length": 1,
                    "description": "主題の再提示。",
                },
                "bridge": {
                    "parent_section_id": "whole",
                    "order": 1,
                    "role": "transition",
                    "relative_length": 0.25,
                    "description": "再提示へつなぐ。",
                },
            },
            "materials": {
                "theme": {"kind": "theme", "description": "中心素材。"},
                "theme_varied": {"kind": "theme", "description": "変奏した中心素材。"},
                "bridge": {"kind": "transition", "description": "短いつなぎ素材。"},
            },
            "placements": {
                "theme_first": {"section_id": "z_first", "material_id": "theme"},
                "theme_return": {"section_id": "a_return", "material_id": "theme"},
                "bridge_fill": {"section_id": "bridge", "material_id": "bridge"},
            },
            "variations": {
                "section_return": {
                    "source": {"type": "section", "id": "z_first"},
                    "target": {"type": "section", "id": "a_return"},
                    "preserve": ["中心素材"],
                    "change": ["提示の明確さ"],
                    "description": "主題を明確に再提示する。",
                },
                "material_return": {
                    "source": {"type": "material", "id": "theme"},
                    "target": {"type": "material", "id": "theme_varied"},
                    "preserve": ["中心となる音型"],
                    "change": ["終わり方"],
                    "description": "中心素材の終わり方を変える。",
                },
            },
            "requirements": {},
            "transitions": {
                "to_return": {
                    "from_placement_id": "theme_first",
                    "connector_placement_id": "bridge_fill",
                    "to_placement_id": "theme_return",
                    "description": "主題の再提示へつなぐ。",
                }
            },
        },
    }


def _document_with_requirement() -> dict[str, object]:
    document = _document_with_repeated_material()
    script = document["script"]
    assert isinstance(script, dict)
    setup = script["performance_setup"]
    assert isinstance(setup, dict)
    directions = setup["performance_directions"]
    assert isinstance(directions, dict)
    directions["clear_return"] = {
        "target": {"type": "section", "id": "a_return"},
        "relative_to": {"type": "section", "id": "z_first"},
        "description": "再提示では縦の線をそろえる。",
    }
    requirements = script["requirements"]
    assert isinstance(requirements, dict)
    requirements["return_more_aligned"] = {
        "performance_direction_id": "clear_return",
        "feature": "onset_alignment",
        "relation": "more",
    }
    return document


def test_show_returns_a_material_and_its_placements_without_mutating_the_document() -> None:
    document = _document_with_repeated_material()
    original = deepcopy(document)

    result = show(document, "material", "theme")

    assert result.shown is True
    assert result.issues == ()
    assert result.item is not None
    assert result.item.type == "material"
    assert result.item.id == "theme"
    assert result.item.value == {"kind": "theme", "description": "中心素材。"}
    assert [
        (relation.role, relation.item.type, relation.item.id) for relation in result.relations
    ] == [
        ("placement", "placement", "theme_first"),
        ("placement", "placement", "theme_return"),
        ("variation_as_source", "variation", "material_return"),
    ]
    assert document == original


def test_show_returns_section_children_in_performance_order() -> None:
    document = _document_with_repeated_material()

    result = show(document, "section", "whole")

    assert result.shown is True
    assert result.item is not None
    assert result.item.type == "section"
    assert result.item.id == "whole"
    assert [
        (relation.role, relation.item.type, relation.item.id) for relation in result.relations
    ] == [
        ("child", "section", "z_first"),
        ("child", "section", "bridge"),
        ("child", "section", "a_return"),
    ]


def test_show_returns_all_direct_relations_for_a_leaf_section() -> None:
    document = _document_with_repeated_material()

    result = show(document, "section", "z_first")

    assert result.shown is True
    assert [
        (relation.role, relation.item.type, relation.item.id) for relation in result.relations
    ] == [
        ("parent", "section", "whole"),
        ("placement", "placement", "theme_first"),
        ("variation_as_source", "variation", "section_return"),
        ("performance_direction", "performance_direction", "linger"),
    ]


def test_show_returns_all_direct_relations_for_a_placement() -> None:
    document = _document_with_repeated_material()

    result = show(document, "placement", "theme_first")

    assert result.shown is True
    assert [
        (relation.role, relation.item.type, relation.item.id) for relation in result.relations
    ] == [
        ("section", "section", "z_first"),
        ("material", "material", "theme"),
        ("transition_from", "transition", "to_return"),
        ("performance_direction", "performance_direction", "clear_first"),
    ]


def test_show_returns_the_source_and_target_of_a_variation() -> None:
    document = _document_with_repeated_material()

    result = show(document, "variation", "material_return")

    assert result.shown is True
    assert [
        (relation.role, relation.item.type, relation.item.id) for relation in result.relations
    ] == [
        ("source", "material", "theme"),
        ("target", "material", "theme_varied"),
    ]


def test_show_returns_the_three_placements_of_a_transition() -> None:
    document = _document_with_repeated_material()

    result = show(document, "transition", "to_return")

    assert result.shown is True
    assert [
        (relation.role, relation.item.type, relation.item.id) for relation in result.relations
    ] == [
        ("from", "placement", "theme_first"),
        ("connector", "placement", "bridge_fill"),
        ("to", "placement", "theme_return"),
    ]


def test_show_returns_the_target_and_comparison_target_of_a_performance_direction() -> None:
    document = _document_with_repeated_material()

    result = show(document, "performance_direction", "clear_first")

    assert result.shown is True
    assert [
        (relation.role, relation.item.type, relation.item.id) for relation in result.relations
    ] == [
        ("target", "placement", "theme_first"),
        ("relative_to", "placement", "theme_return"),
    ]


def test_show_returns_a_requirement_and_its_resolved_relations() -> None:
    result = show(_document_with_requirement(), "requirement", "return_more_aligned")

    assert result.shown is True
    assert result.item is not None
    assert result.item.value == {
        "performance_direction_id": "clear_return",
        "feature": "onset_alignment",
        "relation": "more",
    }
    assert [
        (relation.role, relation.item.type, relation.item.id) for relation in result.relations
    ] == [
        ("performance_direction", "performance_direction", "clear_return"),
        ("target", "section", "a_return"),
        ("reference", "section", "z_first"),
    ]


def test_show_returns_requirements_linked_to_a_performance_direction() -> None:
    result = show(_document_with_requirement(), "performance_direction", "clear_return")

    assert result.shown is True
    assert [
        (relation.role, relation.item.type, relation.item.id) for relation in result.relations
    ] == [
        ("target", "section", "a_return"),
        ("relative_to", "section", "z_first"),
        ("requirement", "requirement", "return_more_aligned"),
    ]


def test_show_returns_requirement_roles_from_both_compared_sections() -> None:
    document = _document_with_requirement()

    target = show(document, "section", "a_return")
    reference = show(document, "section", "z_first")

    assert target.shown is True
    assert ("requirement_as_target", "requirement", "return_more_aligned") in [
        (relation.role, relation.item.type, relation.item.id) for relation in target.relations
    ]
    assert reference.shown is True
    assert ("requirement_as_reference", "requirement", "return_more_aligned") in [
        (relation.role, relation.item.type, relation.item.id) for relation in reference.relations
    ]


def test_show_rejects_an_invalid_document_before_querying_it() -> None:
    document = _document_with_repeated_material()
    script = document["script"]
    assert isinstance(script, dict)
    placements = script["placements"]
    assert isinstance(placements, dict)
    placement = placements["theme_first"]
    assert isinstance(placement, dict)
    placement["section_id"] = "missing"

    result = show(document, "material", "theme")

    assert result.shown is False
    assert result.item is None
    assert result.relations == ()
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == "/script/placements/theme_first/section_id"


def test_show_rejects_an_unknown_item_type() -> None:
    document = _document_with_repeated_material()

    result = show(document, "note", "theme")

    assert result.shown is False
    assert result.item is None
    assert result.relations == ()
    assert len(result.issues) == 1
    assert result.issues[0].code is IssueCode.QUERY_INVALID
    assert result.issues[0].path == "/item_type"


def test_show_reports_a_missing_item_without_raising() -> None:
    document = _document_with_repeated_material()

    result = show(document, "material", "missing")

    assert result.shown is False
    assert result.item is None
    assert result.relations == ()
    assert len(result.issues) == 1
    assert result.issues[0].code is IssueCode.NOT_FOUND
    assert result.issues[0].path == "/script/materials/missing"
