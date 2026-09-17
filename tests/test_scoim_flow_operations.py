from scoim.flow_operations import apply_flow_patch, approve_flow
from scoim.validation import IssueCode


def _draft_flow() -> dict[str, object]:
    return {
        "document_type": "flow",
        "schema_version": "0.1.0",
        "document_id": "flow-001",
        "revision": 1,
        "status": "draft",
        "approval": None,
        "title": "窓辺の午後",
        "overall_flow": "明るく始まり、穏やかに戻る。",
        "instrumentation": "solo_piano",
        "target_duration": 180,
        "scenes": [
            {
                "scene_id": "scene-001",
                "name": "始まり",
                "length_class": "short",
                "heard_as": "軽やかに聞こえる。",
                "relation_to_previous": "曲の始まり。",
                "transition_to_next": "少し広がる。",
            }
        ],
    }


def test_apply_flow_patch_changes_one_field_and_increments_revision() -> None:
    original = _draft_flow()

    result = apply_flow_patch(
        original,
        [
            {"op": "test", "path": "/revision", "value": 1},
            {"op": "replace", "path": "/revision", "value": 2},
            {"op": "replace", "path": "/scenes/0/name", "value": "静かな始まり"},
        ],
    )

    assert result.applied is True
    assert result.document is not None
    assert result.document["revision"] == 2
    assert result.document["scenes"][0]["name"] == "静かな始まり"
    assert original["revision"] == 1
    assert original["scenes"][0]["name"] == "始まり"


def test_apply_flow_patch_requires_the_current_revision_test_first() -> None:
    result = apply_flow_patch(
        _draft_flow(),
        [{"op": "replace", "path": "/revision", "value": 2}],
    )

    assert result.applied is False
    assert result.issues[0].code is IssueCode.STALE_REVISION
    assert result.issues[0].path == "/revision"


def test_apply_flow_patch_requires_an_increment_of_exactly_one() -> None:
    result = apply_flow_patch(
        _draft_flow(),
        [
            {"op": "test", "path": "/revision", "value": 1},
            {"op": "replace", "path": "/revision", "value": 3},
        ],
    )

    assert result.applied is False
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == "/revision"


def test_approve_flow_freezes_the_content_without_mutating_the_draft() -> None:
    draft = _draft_flow()

    result = approve_flow(draft)

    assert result.approved is True
    assert result.document is not None
    assert result.document["status"] == "approved"
    assert result.document["revision"] == 1
    assert result.document["approval"] is not None
    assert draft["status"] == "draft"
    assert draft["approval"] is None


def test_apply_flow_patch_rejects_overwriting_an_approved_flow() -> None:
    approved = approve_flow(_draft_flow()).document
    assert approved is not None

    result = apply_flow_patch(
        approved,
        [
            {"op": "test", "path": "/revision", "value": 1},
            {"op": "replace", "path": "/revision", "value": 2},
        ],
    )

    assert result.applied is False
    assert result.issues[0].code is IssueCode.IMMUTABLE_APPROVED


def test_apply_flow_patch_can_create_and_reapprove_a_new_revision() -> None:
    approved = approve_flow(_draft_flow()).document
    assert approved is not None

    changed = apply_flow_patch(
        approved,
        [
            {"op": "test", "path": "/revision", "value": 1},
            {"op": "replace", "path": "/revision", "value": 2},
            {"op": "replace", "path": "/scenes/0/name", "value": "静かな始まり"},
        ],
        new_draft=True,
    )

    assert changed.applied is True
    assert changed.document is not None
    assert changed.document["status"] == "draft"
    assert changed.document["approval"] is None
    reapproved = approve_flow(changed.document)
    assert reapproved.approved is True
    assert reapproved.document is not None
    assert reapproved.document["revision"] == 2


def test_approve_flow_rejects_an_invalid_draft() -> None:
    draft = _draft_flow()
    draft["title"] = ""

    result = approve_flow(draft)

    assert result.approved is False
    assert result.document is None
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID
