from dataclasses import replace

import pytest

from llm_musical_composer.performance_pipeline import (
    PerformedNote,
    PiecePlan,
    PlanNode,
    RenderedPerformance,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
)
from scoim.requirements import ResolvedRequirement
from scoim.solo_piano_requirements import evaluate_requirement


def _example() -> tuple[PiecePlan, ScoreSpec, RenderedPerformance]:
    plan = PiecePlan(
        plan_id="comparison",
        title="comparison",
        tonal_center=0,
        mode="major",
        root_node_id="whole",
        ending_intent="tonic",
        nodes=(
            PlanNode("whole", None, 0, "whole"),
            PlanNode(
                "opening",
                "whole",
                0,
                "statement",
                duration_weight=1,
                score_material_id="theme",
            ),
            PlanNode(
                "return",
                "whole",
                1,
                "return",
                derived_from="opening",
                duration_weight=1,
                score_material_id="theme",
            ),
        ),
    )
    score = ScoreSpec(
        score_id="score",
        divisions=4,
        materials=(
            ScoreMaterial(
                material_id="theme",
                length_units=4,
                notes=(
                    ScoreNote("upper", 0, 4, 72, "upper"),
                    ScoreNote("lower", 0, 4, 48, "lower"),
                ),
            ),
        ),
    )
    rendered = RenderedPerformance(
        performance_id="performance",
        title="comparison",
        duration_ms=2_000,
        notes=(
            PerformedNote("opening:upper", "opening", 0, 900, 72, 70, "upper"),
            PerformedNote("opening:lower", "opening", 30, 900, 48, 70, "lower"),
            PerformedNote("return:upper", "return", 1_000, 900, 72, 90, "upper"),
            PerformedNote("return:lower", "return", 1_000, 900, 48, 90, "lower"),
        ),
        pedals=(),
        harmonies=(),
        node_intervals=(
            ("whole", 0, 2_000),
            ("opening", 0, 1_000),
            ("return", 1_000, 2_000),
        ),
        lineage=("comparison", "score", "performance"),
    )
    return plan, score, rendered


def _requirement(feature: str = "onset_alignment", relation: str = "more") -> ResolvedRequirement:
    return ResolvedRequirement(
        requirement_id="comparison_requirement",
        source_path="/script/requirements/comparison_requirement",
        performance_direction_id="comparison_direction",
        target_section_id="return",
        reference_section_id="opening",
        feature=feature,
        relation=relation,
    )


def test_evaluate_requirement_passes_when_the_return_is_more_aligned() -> None:
    plan, score, rendered = _example()

    result = evaluate_requirement(plan, score, rendered, _requirement())

    assert result.status == "passed"
    assert result.observed_relation == "more"
    assert result.target.value == 0.0
    assert result.reference.value == 30.0
    assert result.target.sample_count == 1
    assert result.target.unit == "ms_mean_spread"


def test_evaluate_requirement_passes_for_a_less_request_in_the_observed_direction() -> None:
    plan, score, rendered = _example()
    requirement = replace(
        _requirement(feature="loudness", relation="less"),
        target_section_id="opening",
        reference_section_id="return",
    )

    result = evaluate_requirement(plan, score, rendered, requirement)

    assert result.status == "passed"
    assert result.observed_relation == "less"
    assert result.target.value == 70.0
    assert result.reference.value == 90.0


def test_evaluate_requirement_fails_when_a_less_request_observes_more() -> None:
    plan, score, rendered = _example()

    result = evaluate_requirement(
        plan, score, rendered, _requirement(feature="loudness", relation="less")
    )

    assert result.status == "failed"
    assert result.observed_relation == "more"


def test_evaluate_requirement_fails_when_a_directional_request_is_equal() -> None:
    plan, score, rendered = _example()
    equal = replace(
        rendered,
        notes=tuple(replace(note, velocity=70) for note in rendered.notes),
    )

    result = evaluate_requirement(
        plan, score, equal, _requirement(feature="loudness", relation="less")
    )

    assert result.status == "failed"
    assert result.observed_relation == "equal"


def test_evaluate_requirement_preserves_missing_alignment() -> None:
    plan, score, rendered = _example()
    material = score.materials[0]
    staggered = replace(
        material,
        notes=(material.notes[0], replace(material.notes[1], at_units=1)),
    )

    result = evaluate_requirement(
        plan, replace(score, materials=(staggered,)), rendered, _requirement()
    )

    assert result.status == "missing"
    assert result.observed_relation == "missing"
    assert result.target.reason == "no_score_simultaneous_multivoice_onset"


def test_evaluate_requirement_reports_a_missing_rendered_note() -> None:
    plan, score, rendered = _example()
    incomplete = replace(
        rendered,
        notes=tuple(note for note in rendered.notes if note.event_id != "return:lower"),
    )

    result = evaluate_requirement(plan, score, incomplete, _requirement())

    assert result.status == "missing"
    assert result.target.reason == "rendered_note_missing"
    assert result.target.sample_count == 0


def test_evaluate_requirement_preserves_missing_loudness() -> None:
    plan, score, rendered = _example()
    silent = replace(
        rendered,
        notes=tuple(note for note in rendered.notes if note.occurrence_node_id != "return"),
    )

    result = evaluate_requirement(plan, score, silent, _requirement(feature="loudness"))

    assert result.status == "missing"
    assert result.target.reason == "no_rendered_notes"
    assert result.target.sample_count == 0


def test_evaluate_requirement_rejects_an_unknown_feature() -> None:
    plan, score, rendered = _example()

    with pytest.raises(ValueError, match="unsupported feature"):
        evaluate_requirement(plan, score, rendered, _requirement(feature="tempo"))
