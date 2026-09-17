import copy

from scoim.profile_capabilities import (
    check_generation_profile_document,
    solo_piano_3m_v2_capabilities,
)
from scoim.script_0_4_validation import check_script_0_4_document
from scoim.validation import IssueCode


def _valid_document() -> dict[str, object]:
    return {
        "document_type": "script",
        "schema_version": "0.4.0",
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
                "performance_directions": {
                    "opening": {
                        "target": {"type": "section", "id": "statement"},
                        "performance_aspects": ["timing", "dynamics"],
                        "description": "少し溜めながら、音量に穏やかな山を作る。",
                    }
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
        },
    }


def test_script_0_4_requires_nonempty_unique_performance_aspects() -> None:
    document = _valid_document()

    result = check_script_0_4_document(document)

    assert result.valid is True
    assert result.issues == ()

    for invalid_aspects in ([], ["timing", "timing"]):
        invalid = copy.deepcopy(document)
        invalid["script"]["performance_setup"]["performance_directions"]["opening"][
            "performance_aspects"
        ] = invalid_aspects

        invalid_result = check_script_0_4_document(invalid)

        assert invalid_result.valid is False
        assert invalid_result.issues[0].code is IssueCode.SCHEMA_INVALID


def test_script_0_4_rejects_the_removed_comparison_requirements() -> None:
    document = _valid_document()
    document["script"]["performance_direction_comparison_requirements"] = {}

    result = check_script_0_4_document(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID


def test_profile_check_rejects_an_unsupported_performance_aspect() -> None:
    document = _valid_document()
    document["script"]["performance_setup"]["performance_directions"]["opening"][
        "performance_aspects"
    ] = ["breathing"]

    schema_result = check_script_0_4_document(document)
    profile_result = check_generation_profile_document(document, solo_piano_3m_v2_capabilities())

    assert schema_result.valid is True
    assert profile_result.valid is False
    assert any(
        issue.code is IssueCode.UNREPRESENTABLE and issue.path.endswith("/performance_aspects/0")
        for issue in profile_result.issues
    )
