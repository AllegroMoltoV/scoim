from scoim.operations import apply_patch, approve
from scoim.validation import IssueCode, check, content_sha256


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
                "first": {
                    "parent_section_id": "whole",
                    "order": 0,
                    "role": "statement",
                    "relative_length": 1,
                    "description": "最初の提示。",
                },
                "return": {
                    "parent_section_id": "whole",
                    "order": 1,
                    "role": "return",
                    "relative_length": 1,
                    "description": "主題の再提示。",
                },
            },
            "materials": {
                "theme": {"kind": "theme", "description": "中心素材。"},
                "theme_varied": {"kind": "theme", "description": "変奏した中心素材。"},
            },
            "placements": {
                "theme_first": {"section_id": "first", "material_id": "theme"},
                "theme_return": {"section_id": "return", "material_id": "theme"},
            },
            "variations": {},
            "requirements": {},
            "transitions": {},
        },
    }


def _legacy_approved_document() -> dict[str, object]:
    document = _document_with_repeated_material()
    script = document["script"]
    assert isinstance(script, dict)
    del script["requirements"]
    document["status"] = "approved"
    document["approval"] = {"content_sha256": content_sha256(document)}
    return document


def test_apply_patch_changes_only_one_placement_and_increments_revision() -> None:
    document = _document_with_repeated_material()
    operations = [
        {"op": "test", "path": "/revision", "value": 1},
        {"op": "replace", "path": "/revision", "value": 2},
        {
            "op": "replace",
            "path": "/script/placements/theme_return/material_id",
            "value": "theme_varied",
        },
    ]

    result = apply_patch(document, operations)

    assert result.applied is True
    assert result.issues == ()
    assert result.document is not None
    assert result.document["revision"] == 2
    updated_script = result.document["script"]
    original_script = document["script"]
    assert isinstance(updated_script, dict)
    assert isinstance(original_script, dict)
    assert updated_script["placements"] == {
        "theme_first": {"section_id": "first", "material_id": "theme"},
        "theme_return": {"section_id": "return", "material_id": "theme_varied"},
    }
    assert original_script["placements"] == {
        "theme_first": {"section_id": "first", "material_id": "theme"},
        "theme_return": {"section_id": "return", "material_id": "theme"},
    }


def test_apply_patch_rejects_a_stale_revision() -> None:
    document = _document_with_repeated_material()
    operations = [
        {"op": "test", "path": "/revision", "value": 0},
        {"op": "replace", "path": "/revision", "value": 2},
    ]

    result = apply_patch(document, operations)

    assert result.applied is False
    assert result.document is None
    assert len(result.issues) == 1
    assert result.issues[0].code is IssueCode.STALE_REVISION
    assert result.issues[0].path == "/revision"


def test_apply_patch_rejects_an_unchanged_revision() -> None:
    document = _document_with_repeated_material()
    operations = [
        {"op": "test", "path": "/revision", "value": 1},
        {
            "op": "replace",
            "path": "/script/placements/theme_return/material_id",
            "value": "theme_varied",
        },
    ]

    result = apply_patch(document, operations)

    assert result.applied is False
    assert result.document is None
    assert len(result.issues) == 1
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == "/revision"


def test_apply_patch_rejects_a_revision_increment_greater_than_one() -> None:
    document = _document_with_repeated_material()
    operations = [
        {"op": "test", "path": "/revision", "value": 1},
        {"op": "replace", "path": "/revision", "value": 3},
    ]

    result = apply_patch(document, operations)

    assert result.applied is False
    assert result.document is None
    assert len(result.issues) == 1
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == "/revision"


def test_approve_freezes_the_content_hash_without_changing_revision() -> None:
    document = _document_with_repeated_material()

    result = approve(document)

    assert result.approved is True
    assert result.issues == ()
    assert result.document is not None
    assert result.document["revision"] == 1
    assert result.document["status"] == "approved"
    approval = result.document["approval"]
    assert isinstance(approval, dict)
    assert len(approval["content_sha256"]) == 64
    assert check(result.document).valid is True
    assert document["status"] == "draft"
    assert document["approval"] is None


def test_apply_patch_rejects_an_approved_document_as_immutable() -> None:
    approval_result = approve(_document_with_repeated_material())
    assert approval_result.document is not None
    operations = [
        {"op": "test", "path": "/revision", "value": 1},
        {"op": "replace", "path": "/revision", "value": 2},
    ]

    result = apply_patch(approval_result.document, operations)

    assert result.applied is False
    assert result.document is None
    assert len(result.issues) == 1
    assert result.issues[0].code is IssueCode.IMMUTABLE_APPROVED
    assert result.issues[0].path == "/status"


def test_apply_patch_creates_a_new_draft_from_an_approved_document() -> None:
    approval_result = approve(_document_with_repeated_material())
    assert approval_result.document is not None
    approved_document = approval_result.document
    operations = [
        {"op": "test", "path": "/revision", "value": 1},
        {"op": "replace", "path": "/revision", "value": 2},
        {
            "op": "replace",
            "path": "/script/placements/theme_return/material_id",
            "value": "theme_varied",
        },
    ]

    result = apply_patch(approved_document, operations, new_draft=True)

    assert result.applied is True
    assert result.document is not None
    assert result.document["revision"] == 2
    assert result.document["status"] == "draft"
    assert result.document["approval"] is None
    assert approved_document["revision"] == 1
    assert approved_document["status"] == "approved"
    assert approved_document["approval"] is not None


def test_new_draft_from_a_legacy_approved_document_requires_requirements() -> None:
    result = apply_patch(
        _legacy_approved_document(),
        [
            {"op": "test", "path": "/revision", "value": 1},
            {"op": "replace", "path": "/revision", "value": 2},
        ],
        new_draft=True,
    )

    assert result.applied is False
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == "/script/requirements"


def test_new_draft_from_a_legacy_approved_document_can_add_requirements() -> None:
    result = apply_patch(
        _legacy_approved_document(),
        [
            {"op": "test", "path": "/revision", "value": 1},
            {"op": "replace", "path": "/revision", "value": 2},
            {"op": "add", "path": "/script/requirements", "value": {}},
        ],
        new_draft=True,
    )

    assert result.applied is True
    assert result.document is not None
    assert result.document["script"]["requirements"] == {}


def test_apply_patch_rejects_new_draft_for_an_existing_draft() -> None:
    document = _document_with_repeated_material()
    operations = [
        {"op": "test", "path": "/revision", "value": 1},
        {"op": "replace", "path": "/revision", "value": 2},
    ]

    result = apply_patch(document, operations, new_draft=True)

    assert result.applied is False
    assert result.document is None
    assert len(result.issues) == 1
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == "/status"


def test_apply_patch_reports_an_unusable_patch_without_changing_the_document() -> None:
    document = _document_with_repeated_material()
    operations = [
        {"op": "test", "path": "/revision", "value": 1},
        {"op": "replace", "path": "/revision", "value": 2},
        {"op": "replace", "path": "/script/materials/missing/description", "value": "変更"},
    ]

    result = apply_patch(document, operations)

    assert result.applied is False
    assert result.document is None
    assert len(result.issues) == 1
    assert result.issues[0].code is IssueCode.PATCH_INVALID
    assert result.issues[0].path == "/script/materials/missing/description"
    assert document["revision"] == 1
