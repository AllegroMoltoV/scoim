import copy
import json
from pathlib import Path

from scoim.script_validation import check_script_document, script_content_sha256
from scoim.validation import IssueCode

_FIXTURE = Path("tests/fixtures/scoim/fixed-aba/approved-script.json")


def _valid_script_document() -> dict[str, object]:
    legacy = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return {
        "document_type": "script",
        "schema_version": "0.2.0",
        "document_id": "composition-001",
        "revision": 1,
        "status": "validated",
        "source_flow": {
            "document_id": "flow-001",
            "revision": 2,
            "content_sha256": "a" * 64,
        },
        "script": copy.deepcopy(legacy["script"]),
    }


def test_check_script_document_accepts_a_validated_document() -> None:
    document = _valid_script_document()

    result = check_script_document(document)

    assert result.valid is True
    assert result.issues == ()
    assert len(script_content_sha256(document)) == 64


def test_check_script_document_rejects_a_nonvalidated_state() -> None:
    document = _valid_script_document()
    document["status"] = "approved"

    result = check_script_document(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID
    assert result.issues[0].path == "/status"


def test_check_script_document_rejects_broken_inner_references() -> None:
    document = _valid_script_document()
    document["script"]["placements"]["theme_first"]["material_id"] = "missing"

    result = check_script_document(document)

    assert result.valid is False
    assert any(
        issue.code is IssueCode.SEMANTIC_INVALID
        and issue.path == "/script/placements/theme_first/material_id"
        for issue in result.issues
    )


def test_check_script_document_rejects_an_unsupported_version() -> None:
    document = _valid_script_document()
    document["schema_version"] = "0.3.0"

    result = check_script_document(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.UNSUPPORTED_SCHEMA_VERSION
