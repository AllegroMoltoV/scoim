import copy
import json
from pathlib import Path

from scoim.projection import (
    PlanChoice,
    prepare_solo_piano_3m_structure,
    project_solo_piano_3m,
)
from scoim.solo_piano_performance import performance_operation_schedule_for_nodes


def _validated_script() -> dict[str, object]:
    legacy = json.loads(
        Path("tests/fixtures/scoim/fixed-aba/approved-script.json").read_text(encoding="utf-8")
    )
    return {
        "document_type": "script",
        "schema_version": "0.2.0",
        "document_id": "composition-001",
        "revision": 1,
        "status": "validated",
        "source_flow": {
            "document_id": "flow-001",
            "revision": 1,
            "content_sha256": "a" * 64,
        },
        "script": copy.deepcopy(legacy["script"]),
    }


def test_project_solo_piano_accepts_a_validated_script_document() -> None:
    result = project_solo_piano_3m(
        _validated_script(),
        plan_choice=PlanChoice(tonal_center=0, mode="major", harmonic_focus_by_section={}),
    )

    assert result.projected is True
    assert result.piece_plan is not None
    assert result.piece_plan.plan_id == "composition-001"


def test_performance_schedule_groups_absolute_and_comparative_directions_by_node() -> None:
    document = _validated_script()
    directions = document["script"]["performance_setup"]["performance_directions"]
    directions["return_emphasis"] = {
        "target": {"type": "placement", "id": "theme_return"},
        "description": "戻った主題を少し強調する。",
    }
    projection = project_solo_piano_3m(
        document,
        plan_choice=PlanChoice(tonal_center=0, mode="major", harmonic_focus_by_section={}),
    )
    assert projection.piece_plan is not None

    schedule = performance_operation_schedule_for_nodes(document, projection.piece_plan.nodes)

    assert schedule.absolute_node_ids == ("statement",)
    assert schedule.comparative_waves == (("return",),)
    return_group = next(group for group in schedule.direction_groups if group.node_id == "return")
    assert return_group.direction_ids == ("clear_return", "return_emphasis")


def test_profile_check_reports_independent_return_and_ending_problems_together() -> None:
    document = _validated_script()
    document["script"]["variations"] = {}
    document["script"]["materials"]["cadence"]["kind"] = "theme"

    result = prepare_solo_piano_3m_structure(document)

    assert result.prepared is False
    assert result.checks_complete is True
    assert {issue.path for issue in result.issues} >= {
        "/script/sections/return/role",
        "/script/materials",
    }
