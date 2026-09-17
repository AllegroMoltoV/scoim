from __future__ import annotations

from dataclasses import replace

import pytest

from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    PiecePlan,
    PlanNode,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    render_performance,
)
from llm_musical_composer.recurrence_quality import (
    RecurrenceQualityError,
    analyze_foreground_dissonance,
    analyze_foreground_variation,
    analyze_material_harmony,
    analyze_rendered_boundary,
    analyze_rendered_foreground_dissonance,
    analyze_rendered_harmony,
    analyze_transition_connection,
)


def _resolved_non_chord_score() -> ScoreSpec:
    return ScoreSpec(
        "resolved-non-chord",
        4,
        (
            ScoreMaterial(
                "phrase",
                8,
                (
                    ScoreNote("u0", 0, 2, 60, "upper"),
                    ScoreNote("u1", 2, 2, 61, "upper"),
                    ScoreNote("u2", 4, 2, 60, "upper"),
                    ScoreNote("u3", 6, 2, 64, "upper"),
                    ScoreNote("l0", 0, 4, 48, "lower"),
                    ScoreNote("l1", 4, 4, 55, "lower"),
                ),
                harmonies=(ScoreHarmony("h0", 0, 8, 0, "major"),),
                foreground_voice="upper",
            ),
        ),
    )


def test_material_harmony_counts_resolved_foreground_non_chord_tones() -> None:
    score = _resolved_non_chord_score()

    assessment = analyze_material_harmony(
        score,
        "phrase",
        low_pitch_boundary=48,
        minimum_low_spacing_semitones=7,
    )

    assert assessment.foreground_non_chord_tones == 1
    assert assessment.resolved_foreground_non_chord_tones == 1
    assert assessment.unresolved_foreground_non_chord_tones == 0
    assert assessment.passes


def test_foreground_dissonance_classifies_passing_neighbor_and_unsupported() -> None:
    score = ScoreSpec(
        "functional-dissonance",
        4,
        (
            ScoreMaterial(
                "passing",
                8,
                (
                    ScoreNote("passing-u0", 0, 2, 60, "upper"),
                    ScoreNote("passing-u1", 2, 2, 62, "upper"),
                    ScoreNote("passing-u2", 4, 2, 64, "upper"),
                    ScoreNote("passing-l0", 0, 8, 48, "lower"),
                ),
                harmonies=(ScoreHarmony("passing-h", 0, 8, 0, "major"),),
                foreground_voice="upper",
            ),
            ScoreMaterial(
                "neighbor",
                8,
                (
                    ScoreNote("neighbor-u0", 0, 2, 64, "upper"),
                    ScoreNote("neighbor-u1", 2, 2, 65, "upper"),
                    ScoreNote("neighbor-u2", 4, 2, 64, "upper"),
                    ScoreNote("neighbor-l0", 0, 8, 48, "lower"),
                ),
                harmonies=(ScoreHarmony("neighbor-h", 0, 8, 0, "major"),),
                foreground_voice="upper",
            ),
            ScoreMaterial(
                "unsupported",
                8,
                (
                    ScoreNote("unsupported-u0", 0, 2, 60, "upper"),
                    ScoreNote("unsupported-u1", 2, 2, 71, "upper"),
                    ScoreNote("unsupported-u2", 4, 2, 72, "upper"),
                    ScoreNote("unsupported-l0", 0, 8, 48, "lower"),
                ),
                harmonies=(ScoreHarmony("unsupported-h", 0, 8, 0, "major"),),
                foreground_voice="upper",
            ),
        ),
    )

    passing = analyze_foreground_dissonance(score, "passing")
    neighbor = analyze_foreground_dissonance(score, "neighbor")
    unsupported = analyze_foreground_dissonance(score, "unsupported")

    assert (passing.passing_tones, passing.neighbor_tones, passing.unsupported_tones) == (1, 0, 0)
    assert passing.tones[0].event_id == "passing-u1"
    assert passing.tones[0].approach_semitones == 2
    assert passing.tones[0].departure_semitones == 2
    assert (
        neighbor.passing_tones,
        neighbor.neighbor_tones,
        neighbor.unsupported_tones,
    ) == (0, 1, 0)
    assert neighbor.tones[0].approach_semitones == 1
    assert neighbor.tones[0].departure_semitones == -1
    assert unsupported.unsupported_tones == 1
    assert unsupported.tones[0].approach_semitones == 11
    assert unsupported.tones[0].departure_semitones == 1


def test_foreground_dissonance_rejects_missing_context_and_preserves_legacy_counts() -> None:
    score = _resolved_non_chord_score()
    material = score.materials[0]
    changed = replace(
        score,
        materials=(
            replace(
                material,
                notes=(
                    ScoreNote("edge-start", 0, 2, 61, "upper"),
                    ScoreNote("edge-middle", 2, 3, 60, "upper"),
                    ScoreNote("edge-end", 6, 2, 61, "upper"),
                    ScoreNote("edge-lower", 0, 8, 48, "lower"),
                ),
            ),
        ),
    )

    detailed = analyze_foreground_dissonance(changed, "phrase")
    legacy = analyze_material_harmony(
        score,
        "phrase",
        low_pitch_boundary=48,
        minimum_low_spacing_semitones=7,
    )

    assert detailed.passing_tones == 0
    assert detailed.neighbor_tones == 0
    assert detailed.unsupported_tones == 2
    assert legacy.foreground_non_chord_tones == 1
    assert legacy.resolved_foreground_non_chord_tones == 1
    assert legacy.unresolved_foreground_non_chord_tones == 0


def test_major_seventh_chord_tone_is_not_foreground_dissonance() -> None:
    score = ScoreSpec(
        "major-seventh",
        4,
        (
            ScoreMaterial(
                "phrase",
                8,
                (
                    ScoreNote("u0", 0, 2, 71, "upper"),
                    ScoreNote("u1", 2, 2, 72, "upper"),
                    ScoreNote("u2", 4, 2, 71, "upper"),
                    ScoreNote("l0", 0, 8, 48, "lower"),
                ),
                harmonies=(ScoreHarmony("h0", 0, 8, 0, "major-seventh"),),
                foreground_voice="upper",
            ),
        ),
    )

    assessment = analyze_foreground_dissonance(score, "phrase")

    assert assessment.tones == ()
    assert assessment.unsupported_tones == 0


def test_rendered_dissonance_reports_pedal_overlap_at_resolution() -> None:
    plan, score, performance = _pedal_boundary_fixture("harmony_legato")
    left = score.materials[0]
    passing_left = replace(
        left,
        notes=(
            ScoreNote("left-u0", 0, 1, 60, "upper"),
            ScoreNote("left-u1", 1, 1, 62, "upper"),
            ScoreNote("left-u2", 2, 2, 64, "upper"),
            ScoreNote("left-l", 0, 4, 48, "lower"),
        ),
    )
    changed_score = replace(score, materials=(passing_left, score.materials[1]))
    rendered = render_performance(plan, changed_score, performance)

    assessment = analyze_rendered_foreground_dissonance(plan, changed_score, rendered)

    assert assessment.pedal_overlap_count == 1
    assert assessment.maximum_overlap_ms > 0
    assert assessment.overlaps[0].event_id == "left-u1"


def test_material_harmony_rejects_a_non_chord_tone_that_is_too_long() -> None:
    score = _resolved_non_chord_score()
    material = score.materials[0]
    score = replace(
        score,
        materials=(
            replace(
                material,
                notes=tuple(
                    replace(note, duration_units=3) if note.event_id == "u1" else note
                    for note in material.notes
                ),
            ),
        ),
    )

    assessment = analyze_material_harmony(
        score,
        "phrase",
        low_pitch_boundary=48,
        minimum_low_spacing_semitones=7,
    )

    assert assessment.foreground_non_chord_tones == 1
    assert assessment.resolved_foreground_non_chord_tones == 0
    assert assessment.unresolved_foreground_non_chord_tones == 1
    assert not assessment.passes


def _variation_score() -> ScoreSpec:
    harmony = (ScoreHarmony("source-h", 0, 12, 7, "major"),)
    source_lower = (55, 57, 59, 60, 62, 59)
    target_lower = (55, 57, 59, 60, 59, 62)
    return ScoreSpec(
        "variation",
        4,
        (
            ScoreMaterial(
                "source",
                12,
                tuple(
                    ScoreNote(f"source-l{index}", index * 2, 2, pitch, "lower")
                    for index, pitch in enumerate(source_lower)
                ),
                harmonies=harmony,
                foreground_voice="lower",
            ),
            ScoreMaterial(
                "target",
                12,
                tuple(
                    ScoreNote(f"target-l{index}", index * 2, 2, pitch, "lower")
                    for index, pitch in enumerate(target_lower)
                ),
                derived_from="source",
                harmonies=(replace(harmony[0], harmony_id="target-h"),),
                foreground_voice="lower",
            ),
        ),
    )


def test_foreground_variation_uses_the_declared_lower_foreground() -> None:
    score = _variation_score()

    varied = analyze_foreground_variation(score, "source", "target")
    copied = analyze_foreground_variation(score, "source", "source")

    assert varied.voice == "lower"
    assert varied.pitch_difference_count == 2
    assert varied.foreground_motif_head_preserved
    assert not varied.exact_surface_copy
    assert varied.passes
    assert copied.exact_surface_copy
    assert not copied.passes


def test_foreground_variation_rejects_missing_or_incompatible_foregrounds() -> None:
    score = _variation_score()
    with pytest.raises(RecurrenceQualityError, match="does not exist"):
        analyze_foreground_variation(score, "source", "missing")

    target = replace(score.materials[1], foreground_voice="upper")
    with pytest.raises(RecurrenceQualityError, match="same foreground"):
        analyze_foreground_variation(
            replace(score, materials=(score.materials[0], target)),
            "source",
            "target",
        )


def _transition_fixture() -> tuple[PiecePlan, ScoreSpec]:
    plan = PiecePlan(
        "transition",
        "接続",
        9,
        "minor",
        "root",
        "tonic",
        (
            PlanNode("root", None, 0, "whole"),
            PlanNode("source", "root", 0, "statement"),
            PlanNode(
                "source-leaf",
                "source",
                0,
                "statement",
                duration_weight=1,
                score_material_id="source",
            ),
            PlanNode(
                "bridge",
                "root",
                1,
                "transition",
                duration_weight=1,
                score_material_id="bridge",
            ),
            PlanNode("target", "root", 2, "contrast"),
            PlanNode(
                "target-leaf",
                "target",
                0,
                "statement",
                duration_weight=1,
                score_material_id="target",
            ),
        ),
    )
    score = ScoreSpec(
        "transition",
        4,
        (
            ScoreMaterial(
                "source",
                4,
                (
                    ScoreNote("source-u0", 0, 2, 69, "upper"),
                    ScoreNote("source-u1", 2, 2, 71, "upper"),
                ),
            ),
            ScoreMaterial(
                "bridge",
                12,
                tuple(
                    ScoreNote(f"bridge-u{index}", index * 2, 2, pitch, "upper")
                    for index, pitch in enumerate((72, 76, 79, 74, 71, 74))
                ),
            ),
            ScoreMaterial("target", 4, (ScoreNote("target-u0", 0, 4, 72, "upper"),)),
        ),
    )
    return plan, score


def test_transition_connection_checks_both_boundaries_and_internal_motion() -> None:
    plan, score = _transition_fixture()

    assessment = analyze_transition_connection(
        plan,
        score,
        source_node_id="source",
        transition_node_id="bridge",
        target_node_id="target",
        voice="upper",
        minimum_transition_attacks=6,
        maximum_boundary_leap=2,
        maximum_internal_leap=5,
    )

    assert assessment.source_to_transition_semitones == 1
    assert assessment.transition_to_target_semitones == 2
    assert assessment.maximum_internal_leap == 5
    assert not assessment.same_pitch_restrike
    assert assessment.passes


def test_transition_connection_rejects_invalid_contracts_and_restrikes() -> None:
    plan, score = _transition_fixture()
    arguments = {
        "source_node_id": "source",
        "transition_node_id": "bridge",
        "target_node_id": "target",
        "voice": "upper",
        "minimum_transition_attacks": 6,
        "maximum_boundary_leap": 2,
        "maximum_internal_leap": 5,
    }
    with pytest.raises(RecurrenceQualityError, match="voice"):
        analyze_transition_connection(plan, score, **{**arguments, "voice": "middle"})
    with pytest.raises(RecurrenceQualityError, match="thresholds"):
        analyze_transition_connection(
            plan,
            score,
            **{**arguments, "minimum_transition_attacks": 1},
        )
    with pytest.raises(RecurrenceQualityError, match="does not exist"):
        analyze_transition_connection(
            plan,
            score,
            **{**arguments, "target_node_id": "missing"},
        )
    with pytest.raises(RecurrenceQualityError, match="selected voice"):
        analyze_transition_connection(plan, score, **{**arguments, "voice": "lower"})

    target = replace(
        score.materials[2],
        notes=(replace(score.materials[2].notes[0], pitch=74),),
    )
    assessment = analyze_transition_connection(
        plan,
        replace(score, materials=(*score.materials[:2], target)),
        **arguments,
    )
    assert assessment.same_pitch_restrike
    assert not assessment.passes


def _pedal_boundary_fixture(profile: str) -> tuple[PiecePlan, ScoreSpec, PerformanceSpec]:
    plan = PiecePlan(
        "pedal-boundary",
        "踏み替え",
        0,
        "major",
        "root",
        "tonic",
        (
            PlanNode("root", None, 0, "whole"),
            PlanNode(
                "left",
                "root",
                0,
                "statement",
                duration_weight=1,
                score_material_id="left",
            ),
            PlanNode(
                "right",
                "root",
                1,
                "contrast",
                duration_weight=1,
                score_material_id="right",
            ),
        ),
    )
    score = ScoreSpec(
        "pedal-boundary",
        4,
        (
            ScoreMaterial(
                "left",
                4,
                (
                    ScoreNote("left-u", 0, 4, 60, "upper"),
                    ScoreNote("left-l", 0, 4, 48, "lower"),
                ),
                harmonies=(ScoreHarmony("left-h", 0, 4, 0, "major"),),
                foreground_voice="upper",
            ),
            ScoreMaterial(
                "right",
                4,
                (
                    ScoreNote("right-u", 0, 4, 62, "upper"),
                    ScoreNote("right-l", 0, 4, 50, "lower"),
                ),
                harmonies=(ScoreHarmony("right-h", 0, 4, 2, "major"),),
                foreground_voice="upper",
            ),
        ),
    )
    performance = PerformanceSpec(
        "pedal-boundary",
        8_000,
        64,
        "subtle-v1",
        (NodePerformance("root", pedal_profile=profile),),
    )
    return plan, score, performance


def test_harmony_legato_repedals_at_the_material_boundary_without_a_dry_gap() -> None:
    plan, score, performance = _pedal_boundary_fixture("harmony_legato")
    rendered = render_performance(plan, score, performance)

    boundary = next(item[1] for item in rendered.node_intervals if item[0] == "right")
    events = tuple((item.at_ms, item.value) for item in rendered.pedals if item.at_ms == boundary)
    assessment = analyze_rendered_boundary(
        plan,
        rendered,
        transition_node_id="left",
        target_node_id="right",
        voice="upper",
        maximum_pedal_gap_ms=0,
    )

    assert events == ((boundary, 0), (boundary, 127))
    assert assessment.pedal_gap_ms == 0
    assert assessment.passes
    assert analyze_rendered_harmony(plan, score, rendered).passes


def test_rendered_harmony_detects_carryover_between_occurrences() -> None:
    plan, score, performance = _pedal_boundary_fixture("phrase_legato")
    rendered = render_performance(plan, score, performance)

    assessment = analyze_rendered_harmony(plan, score, rendered)

    assert assessment.pedal_carryover_violations == 1
    assert not assessment.passes


def test_rendered_boundary_rejects_invalid_contracts() -> None:
    plan, score, performance = _pedal_boundary_fixture("harmony_legato")
    rendered = render_performance(plan, score, performance)
    arguments = {
        "transition_node_id": "left",
        "target_node_id": "right",
        "voice": "upper",
        "maximum_pedal_gap_ms": 0,
    }
    with pytest.raises(RecurrenceQualityError, match="parameters"):
        analyze_rendered_boundary(
            plan,
            rendered,
            **{**arguments, "maximum_pedal_gap_ms": -1},
        )
    with pytest.raises(RecurrenceQualityError, match="does not exist"):
        analyze_rendered_boundary(
            plan,
            rendered,
            **{**arguments, "target_node_id": "missing"},
        )
