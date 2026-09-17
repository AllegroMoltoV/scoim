import json
from pathlib import Path

from scoim.flow_operations import approve_flow
from scoim.storage import save_flow_document
from scoim.validation import IssueCode


def _flow() -> dict[str, object]:
    return {
        "document_type": "flow",
        "schema_version": "0.1.0",
        "document_id": "stored-flow",
        "revision": 1,
        "status": "draft",
        "approval": None,
        "title": "保存する曲",
        "overall_flow": "静かに始まり、広がって戻る。",
        "instrumentation": "solo_piano",
        "target_duration": 180,
        "scenes": [
            {
                "scene_id": "scene-001",
                "name": "始まり",
                "length_class": "medium",
                "heard_as": "静かに始まる。",
                "relation_to_previous": "曲の始まり。",
                "transition_to_next": "少し広がる。",
            }
        ],
    }


def test_save_flow_document_creates_a_valid_flow_file(tmp_path: Path) -> None:
    target = tmp_path / "flow.json"
    document = _flow()

    result = save_flow_document(target, document, expected_revision=None)

    assert result.saved is True
    assert result.issues == ()
    assert json.loads(target.read_bytes()) == document


def test_save_flow_document_replaces_a_matching_draft_revision(tmp_path: Path) -> None:
    target = tmp_path / "flow.json"
    document = _flow()
    assert save_flow_document(target, document, expected_revision=None).saved is True
    updated = _flow()
    updated["revision"] = 2

    result = save_flow_document(target, updated, expected_revision=1)

    assert result.saved is True
    assert json.loads(target.read_bytes())["revision"] == 2


def test_save_flow_document_rejects_a_different_document_at_the_same_path(
    tmp_path: Path,
) -> None:
    target = tmp_path / "flow.json"
    assert save_flow_document(target, _flow(), expected_revision=None).saved is True
    other = _flow()
    other["document_id"] = "other-flow"

    result = save_flow_document(target, other, expected_revision=1)

    assert result.saved is False
    assert result.issues[0].code is IssueCode.STORAGE_CONFLICT
    assert json.loads(target.read_bytes())["document_id"] == "stored-flow"


def test_save_flow_document_does_not_replace_an_approved_flow(tmp_path: Path) -> None:
    target = tmp_path / "flow.json"
    approved = approve_flow(_flow()).document
    assert approved is not None
    assert save_flow_document(target, approved, expected_revision=None).saved is True
    changed = _flow()
    changed["revision"] = 2

    result = save_flow_document(target, changed, expected_revision=1)

    assert result.saved is False
    assert result.issues[0].code is IssueCode.IMMUTABLE_APPROVED
    assert json.loads(target.read_bytes())["status"] == "approved"
