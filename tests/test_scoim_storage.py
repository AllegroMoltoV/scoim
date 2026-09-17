import json
from pathlib import Path

from scoim.operations import apply_patch, approve
from scoim.query import show
from scoim.storage import save_document
from scoim.validation import IssueCode, check


def _valid_document() -> dict[str, object]:
    return {
        "schema_version": "0.1.0",
        "document_id": "small_song",
        "revision": 1,
        "status": "draft",
        "approval": None,
        "script": {
            "title": "小さな曲",
            "brief": "一つの素材を短く提示する。",
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
                "statement": {
                    "parent_section_id": "whole",
                    "order": 0,
                    "role": "statement",
                    "relative_length": 1,
                    "description": "素材を提示する。",
                },
            },
            "materials": {
                "theme": {"kind": "theme", "description": "中心素材。"},
            },
            "placements": {
                "theme_first": {"section_id": "statement", "material_id": "theme"},
            },
            "variations": {},
            "requirements": {},
            "transitions": {},
        },
    }


def test_save_document_creates_a_valid_utf8_json_file_atomically(tmp_path: Path) -> None:
    document = _valid_document()
    target = tmp_path / "nested" / "script.json"

    result = save_document(target, document, expected_revision=None)

    assert result.saved is True
    assert result.issues == ()
    content = target.read_bytes()
    assert content.endswith(b"\n")
    assert b"\r\n" not in content
    assert json.loads(content) == document
    assert list(target.parent.glob(".script.json.*.tmp")) == []


def test_save_document_rejects_a_stale_expected_revision(tmp_path: Path) -> None:
    target = tmp_path / "script.json"
    document = _valid_document()
    first_result = save_document(target, document, expected_revision=None)
    assert first_result.saved is True
    original = target.read_bytes()
    updated = _valid_document()
    updated["revision"] = 2

    result = save_document(target, updated, expected_revision=0)

    assert result.saved is False
    assert len(result.issues) == 1
    assert result.issues[0].code is IssueCode.STORAGE_CONFLICT
    assert result.issues[0].path == "/revision"
    assert target.read_bytes() == original


def test_save_document_updates_a_matching_draft_revision(tmp_path: Path) -> None:
    target = tmp_path / "script.json"
    document = _valid_document()
    assert save_document(target, document, expected_revision=None).saved is True
    updated = _valid_document()
    updated["revision"] = 2

    result = save_document(target, updated, expected_revision=1)

    assert result.saved is True
    assert json.loads(target.read_bytes())["revision"] == 2


def test_save_document_rejects_a_different_document_id_at_the_same_path(tmp_path: Path) -> None:
    target = tmp_path / "script.json"
    document = _valid_document()
    assert save_document(target, document, expected_revision=None).saved is True
    original = target.read_bytes()
    other_document = _valid_document()
    other_document["document_id"] = "other_song"

    result = save_document(target, other_document, expected_revision=1)

    assert result.saved is False
    assert result.issues[0].code is IssueCode.STORAGE_CONFLICT
    assert result.issues[0].path == "/document_id"
    assert target.read_bytes() == original


def test_save_document_does_not_overwrite_an_approved_snapshot(tmp_path: Path) -> None:
    target = tmp_path / "approved.json"
    approval = approve(_valid_document())
    assert approval.document is not None
    assert save_document(target, approval.document, expected_revision=None).saved is True
    original = target.read_bytes()
    new_draft = _valid_document()
    new_draft["revision"] = 2

    result = save_document(target, new_draft, expected_revision=1)

    assert result.saved is False
    assert result.issues[0].code is IssueCode.IMMUTABLE_APPROVED
    assert result.issues[0].path == "/status"
    assert target.read_bytes() == original


def test_save_document_preserves_the_old_bytes_when_replacement_fails(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "script.json"
    document = _valid_document()
    assert save_document(target, document, expected_revision=None).saved is True
    original = target.read_bytes()
    updated = _valid_document()
    updated["revision"] = 2
    monkeypatch.setattr(
        "scoim.storage.os.replace",
        lambda *args: (_ for _ in ()).throw(OSError("replace failed")),
    )

    result = save_document(target, updated, expected_revision=1)

    assert result.saved is False
    assert result.issues[0].code is IssueCode.STORAGE_ERROR
    assert result.issues[0].path == ""
    assert target.read_bytes() == original
    assert list(tmp_path.glob(".script.json.*.tmp")) == []


def test_save_document_does_not_replace_a_file_with_an_invalid_document(tmp_path: Path) -> None:
    target = tmp_path / "script.json"
    document = _valid_document()
    assert save_document(target, document, expected_revision=None).saved is True
    original = target.read_bytes()
    invalid = _valid_document()
    invalid["revision"] = "two"

    result = save_document(target, invalid, expected_revision=1)

    assert result.saved is False
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID
    assert target.read_bytes() == original


def test_save_document_does_not_replace_malformed_stored_json(tmp_path: Path) -> None:
    target = tmp_path / "script.json"
    target.write_bytes(b"not-json\r\n")
    original = target.read_bytes()

    result = save_document(target, _valid_document(), expected_revision=1)

    assert result.saved is False
    assert result.issues[0].code is IssueCode.STORAGE_ERROR
    assert target.read_bytes() == original


def test_non_piano_instrumentation_remains_opaque_across_core_operations(
    tmp_path: Path,
) -> None:
    document = _valid_document()
    script = document["script"]
    assert isinstance(script, dict)
    performance_setup = script["performance_setup"]
    assert isinstance(performance_setup, dict)
    performance_setup["instrumentation"] = "string_quartet"

    assert check(document).valid is True
    shown = show(document, "section", "whole")
    assert shown.shown is True
    patch_result = apply_patch(
        document,
        [
            {"op": "test", "path": "/revision", "value": 1},
            {"op": "replace", "path": "/revision", "value": 2},
            {
                "op": "replace",
                "path": "/script/brief",
                "value": "弦楽四重奏で一つの素材を短く提示する。",
            },
        ],
    )
    assert patch_result.applied is True
    assert patch_result.document is not None
    patched_script = patch_result.document["script"]
    assert isinstance(patched_script, dict)
    patched_setup = patched_script["performance_setup"]
    assert isinstance(patched_setup, dict)
    assert patched_setup["instrumentation"] == "string_quartet"

    target = tmp_path / "string-quartet.json"
    saved = save_document(target, patch_result.document, expected_revision=None)

    assert saved.saved is True
    assert json.loads(target.read_bytes()) == patch_result.document
