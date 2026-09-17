import copy

from jsonschema import Draft202012Validator

from scoim.performance_ir import PerformanceSpec, SectionPerformance
from scoim.phase7_performance_contracts import (
    build_performance_choice_operations,
    build_performance_choices,
    performance_choices_response_schema,
)
from scoim.profile_capabilities import solo_piano_3m_v2_capabilities
from scoim.score_ir import PiecePlan, PlanNode


def _script() -> dict[str, object]:
    return {
        "document_type": "script",
        "schema_version": "0.4.0",
        "document_id": "composition-001",
        "revision": 1,
        "status": "validated",
        "source_flow": {"document_id": "flow-001", "revision": 1, "content_sha256": "a" * 64},
        "script": {
            "title": "朝の即興曲",
            "brief": "静かな冒頭から少しずつ前へ進む。",
            "performance_setup": {
                "instrumentation": "solo_piano",
                "target_duration_seconds": 180,
                "performance_directions": {
                    "performance-direction-001": {
                        "target": {"type": "section", "id": "section-001"},
                        "performance_aspects": ["timing"],
                        "description": "句の終わりだけ少し溜める。",
                    }
                },
            },
            "materials": {},
            "sections": {},
            "material_placements": {},
            "script_element_variation_relations": {},
            "material_placement_transitions": {},
        },
    }


def test_phase7_schema_accepts_only_directed_fields_and_registered_choices() -> None:
    schema = performance_choices_response_schema(
        _script(), ("section-001",), solo_piano_3m_v2_capabilities()
    )
    validator = Draft202012Validator(schema)
    valid = {
        "performances": [
            {
                "timing_profile": "savor",
                "timing_amount": "subtle",
                "dynamics_profile": None,
                "articulation_profile": None,
                "coordination_profile": None,
                "pedal_profile": None,
                "handled_directions": [
                    {
                        "direction_id": "performance-direction-001",
                        "field_names": ["timing_profile", "timing_amount"],
                    }
                ],
                "unhandled_directions": [],
            }
        ]
    }

    assert list(validator.iter_errors(valid)) == []

    undirected_field = {"performances": [{**valid["performances"][0], "pedal_profile": "none"}]}
    assert list(validator.iter_errors(undirected_field))

    unknown_choice = {"performances": [{**valid["performances"][0], "timing_profile": "rubato"}]}
    assert list(validator.iter_errors(unknown_choice))


def test_phase7_schema_uses_only_codex_supported_array_keywords() -> None:
    schema = performance_choices_response_schema(
        _script(), ("section-001",), solo_piano_3m_v2_capabilities()
    )

    performances = schema["properties"]["performances"]
    performance = performances["items"]
    handled = performance["properties"]["handled_directions"]["items"]

    assert "prefixItems" not in performances
    assert isinstance(performance, dict)
    assert "uniqueItems" not in handled["properties"]["field_names"]


def test_phase7_builds_v2_performance_and_keeps_artistic_intent_unverified() -> None:
    document = _script()
    document["script"]["performance_setup"]["target_duration_seconds"] = 180.0
    plan = PiecePlan(
        "plan-001",
        "朝の即興曲",
        0,
        "major",
        "section-001",
        (PlanNode("section-001", None, 0, 1, "score-unit-001"),),
    )
    response = {
        "performances": [
            {
                "timing_profile": "savor",
                "timing_amount": "subtle",
                "dynamics_profile": None,
                "articulation_profile": None,
                "coordination_profile": None,
                "pedal_profile": None,
                "handled_directions": [
                    {
                        "direction_id": "performance-direction-001",
                        "field_names": ["timing_profile", "timing_amount"],
                    }
                ],
                "unhandled_directions": [],
            }
        ]
    }

    result = build_performance_choices(
        document,
        plan,
        ("section-001",),
        response,
        solo_piano_3m_v2_capabilities(),
        performance_id="performance-001",
    )

    assert result.valid is True
    assert isinstance(result.performance.target_duration_ms, int)
    assert result.performance == PerformanceSpec(
        performance_id="performance-001",
        target_duration_ms=180_000,
        default_velocity=64,
        timing_budget_id="subtle-v1",
        section_performances=(
            SectionPerformance(
                section_id="section-001",
                timing_profile="savor",
                timing_amount="subtle",
            ),
        ),
    )
    assert len(result.projection_ledger) == 1
    assert result.projection_ledger[0].source_id == "performance-direction-001"
    assert result.projection_ledger[0].status == "unverified"
    assert "timing_profile=savor" in result.projection_ledger[0].evidence


def test_phase7_requires_each_direction_to_be_accounted_for_once() -> None:
    plan = PiecePlan(
        "plan-001",
        "朝の即興曲",
        0,
        "major",
        "section-001",
        (PlanNode("section-001", None, 0, 1, "score-unit-001"),),
    )
    response = {
        "performances": [
            {
                "timing_profile": "savor",
                "timing_amount": "subtle",
                "dynamics_profile": None,
                "articulation_profile": None,
                "coordination_profile": None,
                "pedal_profile": None,
                "handled_directions": [],
                "unhandled_directions": [],
            }
        ]
    }

    result = build_performance_choices(
        _script(),
        plan,
        ("section-001",),
        response,
        solo_piano_3m_v2_capabilities(),
        performance_id="performance-001",
    )

    assert result.valid is False
    assert result.performance is None
    assert result.issues[0].path == "/performances/0"
    assert "exactly once" in result.issues[0].message


def test_phase7_handled_fields_must_belong_to_their_direction_aspects() -> None:
    document = copy.deepcopy(_script())
    directions = document["script"]["performance_setup"]["performance_directions"]
    directions["performance-direction-002"] = {
        "target": {"type": "section", "id": "section-001"},
        "performance_aspects": ["pedal"],
        "description": "フレーズのまとまりで響きをつなぐ。",
    }
    plan = PiecePlan(
        "plan-001",
        "朝の即興曲",
        0,
        "major",
        "section-001",
        (PlanNode("section-001", None, 0, 1, "score-unit-001"),),
    )
    response = {
        "performances": [
            {
                "timing_profile": "savor",
                "timing_amount": "subtle",
                "dynamics_profile": None,
                "articulation_profile": None,
                "coordination_profile": None,
                "pedal_profile": "phrase_legato",
                "handled_directions": [
                    {
                        "direction_id": "performance-direction-001",
                        "field_names": ["pedal_profile"],
                    }
                ],
                "unhandled_directions": [
                    {
                        "direction_id": "performance-direction-002",
                        "reason": "この指示には対応しなかった。",
                    }
                ],
            }
        ]
    }

    result = build_performance_choices(
        document,
        plan,
        ("section-001",),
        response,
        solo_piano_3m_v2_capabilities(),
        performance_id="performance-001",
    )

    assert result.valid is False
    assert result.issues[0].path.endswith("/handled_directions/0/field_names")


def test_phase7_orders_a_comparison_after_its_source_section() -> None:
    document = copy.deepcopy(_script())
    script = document["script"]
    script["performance_setup"]["performance_directions"]["performance-direction-002"] = {
        "target": {"type": "section", "id": "section-002"},
        "relative_to": {"type": "section", "id": "section-001"},
        "performance_aspects": ["timing"],
        "description": "最初の区分より流れよく進める。",
    }
    plan = PiecePlan(
        "plan-001",
        "朝の即興曲",
        0,
        "major",
        "whole",
        (
            PlanNode("whole", None, 0),
            PlanNode("section-001", "whole", 0, 1, "score-unit-001"),
            PlanNode("section-002", "whole", 1, 1, "score-unit-002"),
        ),
    )

    operations = build_performance_choice_operations(document, plan)

    assert tuple(operation.target_section_id for operation in operations) == (
        "section-001",
        "section-002",
    )
    assert operations[1].comparison_section_ids == ("section-001",)
