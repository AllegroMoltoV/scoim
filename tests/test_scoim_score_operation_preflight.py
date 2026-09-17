import json
from pathlib import Path

from scoim.profile_capabilities import solo_piano_3m_v2_capabilities
from scoim.score_operation_preflight import build_score_operation_plan
from scoim.validation import IssueCode

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "scoim" / "score-unit-layer-vertical"


def _basic_script() -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / "basic-validated-script.json").read_text(encoding="utf-8"))


def test_reports_foreground_to_accompaniment_variation_as_unrepresentable() -> None:
    document = _basic_script()
    relations = document["script"]["script_element_variation_relations"]
    relations["theme-varied"]["target"]["id"] = "support-return"

    plan = build_score_operation_plan(document, solo_piano_3m_v2_capabilities())

    assert [(issue.code, issue.path) for issue in plan.issues] == [
        (
            IssueCode.UNREPRESENTABLE,
            "/script/script_element_variation_relations/theme-varied",
        )
    ]
    assert plan.issues[0].message == (
        "Explicit variation is unsupported by the generation profile: "
        "source=material_placement:theme-first (role=foreground), "
        "target=material_placement:support-return (role=accompaniment)"
    )


def test_reports_accompaniment_transition_as_unrepresentable() -> None:
    document = _basic_script()
    transition = document["script"]["material_placement_transitions"]["to-return"]
    transition["source_material_placement_id"] = "support-first"

    plan = build_score_operation_plan(document, solo_piano_3m_v2_capabilities())

    assert [(issue.code, issue.path) for issue in plan.issues] == [
        (
            IssueCode.UNREPRESENTABLE,
            "/script/material_placement_transitions/to-return",
        )
    ]
    assert plan.issues[0].message == (
        "Transition is unsupported by the generation profile: "
        "source=support-first (role=accompaniment), "
        "connector=bridge-only (role=foreground), "
        "target=theme-return (role=foreground)"
    )


def test_reports_a_transition_connector_used_as_variation_source() -> None:
    document = _basic_script()
    relation = document["script"]["script_element_variation_relations"]["theme-varied"]
    relation["source"]["id"] = "bridge-only"

    plan = build_score_operation_plan(document, solo_piano_3m_v2_capabilities())

    assert [(issue.code, issue.path) for issue in plan.issues] == [
        (
            IssueCode.UNREPRESENTABLE,
            "/script/script_element_variation_relations/theme-varied/source/id",
        )
    ]
    assert "source=material_placement:bridge-only (role=foreground)" in plan.issues[0].message
    assert "target=material_placement:theme-return (role=foreground)" in plan.issues[0].message


def test_reports_material_variation_without_a_foreground_source() -> None:
    document = _basic_script()
    script = document["script"]
    script["materials"]["unused-source"] = {"description": "配置されない変奏元。"}
    script["script_element_variation_relations"]["theme-varied"] = {
        "source": {"type": "material", "id": "unused-source"},
        "target": {"type": "material", "id": "theme"},
        "preserve": ["輪郭"],
        "change": ["リズム"],
        "description": "配置されないマテリアルから主題を変奏する。",
    }

    plan = build_score_operation_plan(document, solo_piano_3m_v2_capabilities())

    assert [(issue.code, issue.path) for issue in plan.issues] == [
        (
            IssueCode.UNREPRESENTABLE,
            "/script/script_element_variation_relations/theme-varied/source/id",
        )
    ]


def test_plans_supported_foreground_and_accompaniment_operations() -> None:
    plan = build_score_operation_plan(_basic_script(), solo_piano_3m_v2_capabilities())

    assert plan.issues == ()
    assert [operation.material_placement_id for operation in plan.foreground_operations] == [
        "theme-first",
        "theme-return",
        "ending-only",
        "bridge-only",
    ]
    assert [
        [source.kind for source in operation.comparison_sources]
        for operation in plan.foreground_operations
    ] == [[], ["material_placement_variation"], [], ["transition_source", "transition_target"]]
    assert [operation.material_placement_id for operation in plan.accompaniment_operations] == [
        "support-first",
        "support-return",
    ]
    assert [
        [source.kind for source in operation.comparison_sources]
        for operation in plan.accompaniment_operations
    ] == [[], ["implicit_reuse"]]


def test_reports_every_unsupported_relation_in_stable_order() -> None:
    document = _basic_script()
    script = document["script"]
    relations = script["script_element_variation_relations"]
    transitions = script["material_placement_transitions"]
    relations["z-accompaniment"] = {
        "source": {"type": "material_placement", "id": "support-first"},
        "target": {"type": "material_placement", "id": "support-return"},
        "preserve": ["低音域"],
        "change": ["リズム"],
        "description": "伴奏を変奏する。",
    }
    relations["a-cross-role"] = {
        "source": {"type": "material_placement", "id": "theme-first"},
        "target": {"type": "material_placement", "id": "support-return"},
        "preserve": ["輪郭"],
        "change": ["役割"],
        "description": "前景を伴奏へ変奏する。",
    }
    transitions["z-accompaniment-transition"] = {
        "source_material_placement_id": "support-first",
        "transition_material_placement_id": "bridge-only",
        "target_material_placement_id": "theme-return",
        "description": "伴奏から前景へ遷移する。",
    }

    plan = build_score_operation_plan(document, solo_piano_3m_v2_capabilities())

    assert [issue.path for issue in plan.issues] == [
        "/script/script_element_variation_relations/a-cross-role",
        "/script/script_element_variation_relations/z-accompaniment",
        "/script/material_placement_transitions/z-accompaniment-transition",
    ]
