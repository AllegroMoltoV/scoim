from scoim.flow_validation import check_flow, flow_content_sha256
from scoim.validation import IssueCode


def _minimal_flow() -> dict[str, object]:
    return {
        "document_type": "flow",
        "schema_version": "0.1.0",
        "document_id": "flow-001",
        "revision": 1,
        "status": "draft",
        "approval": None,
        "title": "窓辺の午後",
        "overall_flow": "明るく始まり、少し陰ってから穏やかに戻る。",
        "instrumentation": "solo_piano",
        "target_duration": 180,
        "scenes": [
            {
                "scene_id": "scene-001",
                "name": "やわらかな始まり",
                "length_class": "short",
                "heard_as": "軽やかな音が現れる。",
                "relation_to_previous": "曲の始まり。",
                "transition_to_next": "動きを少し広げる。",
            }
        ],
    }


def test_check_flow_accepts_a_minimal_draft() -> None:
    result = check_flow(_minimal_flow())

    assert result.valid is True
    assert result.issues == ()


def test_check_flow_rejects_an_unknown_field() -> None:
    document = _minimal_flow()
    document["unknown"] = True

    result = check_flow(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID
    assert result.issues[0].path == ""


def test_check_flow_rejects_an_empty_scene_list() -> None:
    document = _minimal_flow()
    document["scenes"] = []

    result = check_flow(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID
    assert result.issues[0].path == "/scenes"


def test_check_flow_rejects_a_duplicate_scene_id() -> None:
    document = _minimal_flow()
    scenes = document["scenes"]
    assert isinstance(scenes, list)
    duplicate = dict(scenes[0])
    duplicate["name"] = "別の場面"
    scenes.append(duplicate)

    result = check_flow(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == "/scenes/1/scene_id"


def test_check_flow_rejects_approval_metadata_on_a_draft() -> None:
    document = _minimal_flow()
    document["approval"] = {"content_sha256": "0" * 64}

    result = check_flow(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == "/approval"


def test_check_flow_rejects_an_approved_flow_with_the_wrong_hash() -> None:
    document = _minimal_flow()
    document["status"] = "approved"
    document["approval"] = {"content_sha256": "0" * 64}

    result = check_flow(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == "/approval/content_sha256"


def test_check_flow_rejects_an_approved_flow_without_approval_metadata() -> None:
    document = _minimal_flow()
    document["status"] = "approved"

    result = check_flow(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == "/approval"


def test_check_flow_accepts_an_approved_flow_with_the_matching_hash() -> None:
    document = _minimal_flow()
    expected_hash = flow_content_sha256(document)
    document["status"] = "approved"
    document["approval"] = {"content_sha256": expected_hash}

    result = check_flow(document)

    assert result.valid is True
    assert flow_content_sha256(document) == expected_hash


def test_check_flow_reports_an_unsupported_version_before_schema_validation() -> None:
    document = _minimal_flow()
    document["schema_version"] = "9.9.9"

    result = check_flow(document)

    assert result.valid is False
    assert result.issues[0].code is IssueCode.UNSUPPORTED_SCHEMA_VERSION
    assert result.issues[0].path == "/schema_version"
