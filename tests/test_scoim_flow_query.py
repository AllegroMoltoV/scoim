from scoim.flow_query import show_flow
from scoim.validation import IssueCode


def _flow() -> dict[str, object]:
    return {
        "document_type": "flow",
        "schema_version": "0.1.0",
        "document_id": "query-flow",
        "revision": 1,
        "status": "draft",
        "approval": None,
        "title": "三つの場面",
        "overall_flow": "始まり、広がり、戻る。",
        "instrumentation": "solo_piano",
        "target_duration": 180,
        "scenes": [
            {
                "scene_id": "scene-001",
                "name": "始まり",
                "length_class": "short",
                "heard_as": "静かに始まる。",
                "relation_to_previous": "曲の始まり。",
                "transition_to_next": "少し広がる。",
            },
            {
                "scene_id": "scene-002",
                "name": "広がり",
                "length_class": "long",
                "heard_as": "大きく広がる。",
                "relation_to_previous": "最初の音楽を発展させる。",
                "transition_to_next": "力を抜いて戻る。",
            },
            {
                "scene_id": "scene-003",
                "name": "戻り",
                "length_class": "medium",
                "heard_as": "落ち着いて終わる。",
                "relation_to_previous": "最初の音楽が形を変えて戻る。",
                "transition_to_next": "余韻を残して終わる。",
            },
        ],
    }


def test_show_flow_returns_the_scene_position_and_immediate_neighbors() -> None:
    document = _flow()

    result = show_flow(document, "scene-002")

    assert result.shown is True
    assert result.scene == document["scenes"][1]
    assert result.scene is not document["scenes"][1]
    assert result.position == 2
    assert result.previous_scene == document["scenes"][0]
    assert result.next_scene == document["scenes"][2]
    assert result.issues == ()


def test_show_flow_uses_none_for_a_neighbor_outside_the_scene_list() -> None:
    document = _flow()

    first = show_flow(document, "scene-001")
    last = show_flow(document, "scene-003")

    assert first.previous_scene is None
    assert first.next_scene == document["scenes"][1]
    assert last.previous_scene == document["scenes"][1]
    assert last.next_scene is None


def test_show_flow_returns_the_flow_validation_failure() -> None:
    document = _flow()
    document["schema_version"] = "9.9.9"

    result = show_flow(document, "scene-001")

    assert result.shown is False
    assert result.scene is None
    assert result.issues[0].code is IssueCode.UNSUPPORTED_SCHEMA_VERSION


def test_show_flow_returns_a_typed_issue_for_an_unknown_scene() -> None:
    result = show_flow(_flow(), "scene-999")

    assert result.shown is False
    assert result.scene is None
    assert result.issues[0].code is IssueCode.NOT_FOUND
    assert result.issues[0].path == "/scene_id"
