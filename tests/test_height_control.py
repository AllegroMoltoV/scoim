from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from llm_musical_composer.brightness_control import height_descriptor
from llm_musical_composer.control_reference_baseline import normalize_value
from llm_musical_composer.generic_pipeline_quality import (
    evaluate_generic_pipeline_quality,
)
from llm_musical_composer.height_control import (
    SCHEME_ID,
    _new_or_worsened_voice_collisions,
    _open_pitch_classes,
    _validate_ending_material,
    build_height_candidate,
    build_height_variant,
    candidate_transpositions,
    denormalize_height,
    resolve_height,
    resolve_measured_height,
)
from llm_musical_composer.performance_pipeline import (
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    render_performance,
    validate_pipeline,
)
from llm_musical_composer.pipeline_dsl import (
    parse_performance_spec,
    parse_piece_plan,
    parse_score_spec,
)

ROOT = Path(__file__).parents[1]
INPUT_DIR = ROOT / ".appendix" / "multiscale-calibration-run-v8" / "inputs"
V18_INPUT_DIR = (
    ROOT
    / ".appendix"
    / "supported-normal-generation-v18-accompaniment-velocity"
    / "runs"
    / "whole-score"
)
SUMMARY_PATH = ROOT / ".appendix" / "control-reference-baseline-v3" / "summary.json"


def _inputs():
    return (
        parse_piece_plan((INPUT_DIR / "piece-plan.music.py").read_text(encoding="utf-8")),
        parse_score_spec((INPUT_DIR / "score.music.py").read_text(encoding="utf-8")),
        parse_performance_spec((INPUT_DIR / "performance.music.py").read_text(encoding="utf-8")),
    )


def _height_range() -> tuple[float, float]:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    axis = summary["axes"]["高さ"]
    return axis["minimum"], axis["maximum"]


def _v18_inputs():
    return (
        parse_piece_plan(
            (V18_INPUT_DIR / "inputs" / "piece-plan.dsl").read_text(encoding="utf-8")
        ),
        parse_score_spec(
            (V18_INPUT_DIR / "outputs" / "score-spec.dsl").read_text(encoding="utf-8")
        ),
        parse_performance_spec(
            (V18_INPUT_DIR / "outputs" / "performance-spec.dsl").read_text(
                encoding="utf-8"
            )
        ),
    )


def _non_pitch_note(note):
    return (
        note.event_id,
        note.at_units,
        note.duration_units,
        note.voice,
        note.tie,
        note.articulations,
    )


def test_denormalize_height_uses_corpus_endpoints() -> None:
    minimum, maximum = _height_range()

    assert denormalize_height(-1.0, minimum=minimum, maximum=maximum) == minimum
    assert denormalize_height(0.0, minimum=minimum, maximum=maximum) == pytest.approx(
        (minimum + maximum) / 2
    )
    assert denormalize_height(1.0, minimum=minimum, maximum=maximum) == maximum
    with pytest.raises(ValueError, match=r"between -1\.0 and 1\.0"):
        denormalize_height(1.01, minimum=minimum, maximum=maximum)


def test_low_candidate_revoices_color_tones_without_removing_notes() -> None:
    plan, score, performance = _inputs()

    candidate = build_height_candidate(plan, score, performance, semitones=-19)
    rendered = render_performance(candidate.plan, candidate.score, candidate.performance)

    assert candidate.scheme_id == SCHEME_ID
    assert candidate.performance == performance
    assert candidate.diagnostics.revoiced_note_count == 56
    assert candidate.diagnostics.lifted_note_count == 3
    assert sum(len(item.notes) for item in candidate.score.materials) == sum(
        len(item.notes) for item in score.materials
    )
    assert height_descriptor(rendered)["mean"] == pytest.approx(45.60546282)


def test_zero_candidate_preserves_v18_music_content() -> None:
    plan, score, performance = _v18_inputs()

    candidate = build_height_candidate(plan, score, performance, semitones=0)

    assert replace(candidate.plan, plan_id=plan.plan_id) == plan
    assert replace(candidate.score, score_id=score.score_id) == score
    assert candidate.performance == performance
    assert candidate.diagnostics.revoiced_note_count == 0
    assert candidate.diagnostics.lifted_note_count == 0


@pytest.mark.parametrize("semitones", [1, 4])
def test_upward_v18_candidate_is_uniform_transposition(semitones: int) -> None:
    plan, score, performance = _v18_inputs()

    candidate = build_height_candidate(
        plan, score, performance, semitones=semitones
    )

    assert candidate.diagnostics.revoiced_note_count == 0
    assert candidate.diagnostics.lifted_note_count == 0
    for base_material, target_material in zip(
        score.materials, candidate.score.materials, strict=True
    ):
        for base_note, target_note in zip(
            base_material.notes, target_material.notes, strict=True
        ):
            assert target_note.pitch == base_note.pitch + semitones


def test_downward_v18_candidate_repairs_register_and_keeps_generic_quality() -> None:
    plan, score, performance = _v18_inputs()

    candidate = build_height_candidate(plan, score, performance, semitones=-1)
    rendered = render_performance(
        candidate.plan, candidate.score, candidate.performance
    )

    assert candidate.diagnostics.revoiced_note_count > 0
    assert evaluate_generic_pipeline_quality(
        candidate.plan, candidate.score, candidate.performance, rendered
    )["passes"]


def test_measured_resolution_uses_revoiced_mean_instead_of_nominal_shift() -> None:
    plan, score, performance = _inputs()
    minimum, maximum = _height_range()
    means = {}
    for semitones in candidate_transpositions(score):
        candidate = build_height_candidate(
            plan, score, performance, semitones=semitones
        )
        means[semitones] = height_descriptor(
            render_performance(candidate.plan, candidate.score, candidate.performance)
        )["mean"]

    low = resolve_measured_height(
        -1.0, candidate_means=means, minimum=minimum, maximum=maximum
    )
    middle = resolve_measured_height(
        0.0, candidate_means=means, minimum=minimum, maximum=maximum
    )
    high = resolve_measured_height(
        1.0, candidate_means=means, minimum=minimum, maximum=maximum
    )

    assert [low.semitones, middle.semitones, high.semitones] == [-19, -6, 10]
    assert [low.achieved_mean, middle.achieved_mean, high.achieved_mean] == pytest.approx(
        [45.60546282, 58.59939302, 74.56297420]
    )
    assert low.status == "unreachable"


def test_diminished_chord_does_not_treat_diminished_fifth_as_perfect_fifth() -> None:
    note = ScoreNote("lower", 0, 4, 30, "lower")
    diminished = ScoreMaterial(
        "diminished",
        4,
        (note,),
        harmonies=(ScoreHarmony("h", 0, 4, 0, "diminished"),),
        foreground_voice="upper",
    )
    major = replace(
        diminished,
        material_id="major",
        harmonies=(ScoreHarmony("h", 0, 4, 0, "major"),),
    )

    assert _open_pitch_classes(diminished, note) == {0}
    assert _open_pitch_classes(major, note) == {0, 7}


def test_voice_collision_check_only_rejects_new_or_deeper_collision() -> None:
    lower = ScoreNote("lower", 0, 4, 40, "lower")
    upper = ScoreNote("upper", 0, 4, 60, "upper")
    base = ScoreMaterial("material", 4, (lower, upper))

    assert _new_or_worsened_voice_collisions(base, base) == ()
    narrower_but_ordered = replace(base, notes=(replace(lower, pitch=50), upper))
    assert _new_or_worsened_voice_collisions(base, narrower_but_ordered) == ()

    newly_colliding = replace(base, notes=(replace(lower, pitch=60), upper))
    assert _new_or_worsened_voice_collisions(base, newly_colliding) == (
        ("lower", "upper", 20, 0),
    )

    base_collision = replace(base, notes=(replace(lower, pitch=60), upper))
    assert _new_or_worsened_voice_collisions(base_collision, base_collision) == ()
    deeper_collision = replace(base, notes=(replace(lower, pitch=61), upper))
    assert _new_or_worsened_voice_collisions(base_collision, deeper_collision) == (
        ("lower", "upper", 0, -1),
    )


def test_unharmonized_ending_is_validated_without_automatic_revoicing() -> None:
    valid = ScoreMaterial(
        "ending",
        4,
        (
            ScoreNote("lower", 0, 4, 40, "lower"),
            ScoreNote("upper", 0, 4, 60, "upper"),
        ),
    )
    _validate_ending_material(valid)

    invalid = replace(
        valid,
        notes=(valid.notes[0], replace(valid.notes[1], pitch=40)),
    )
    with pytest.raises(ValueError, match="ending voice collision"):
        _validate_ending_material(invalid)


def test_v8_resolution_separates_quantization_from_unreachable_range() -> None:
    plan, score, performance = _inputs()
    rendered = render_performance(plan, score, performance)
    current_mean = height_descriptor(rendered)["mean"]
    minimum, maximum = _height_range()

    low = resolve_height(
        -1.0,
        score=score,
        current_mean=current_mean,
        minimum=minimum,
        maximum=maximum,
    )
    middle = resolve_height(
        0.0,
        score=score,
        current_mean=current_mean,
        minimum=minimum,
        maximum=maximum,
    )
    high = resolve_height(
        1.0,
        score=score,
        current_mean=current_mean,
        minimum=minimum,
        maximum=maximum,
    )

    assert [low.semitones, middle.semitones, high.semitones] == [-19, -6, 10]
    assert [low.status, middle.status, high.status] == [
        "unreachable",
        "quantized",
        "quantized",
    ]
    assert low.target_error_semitones > 1
    assert 0 < middle.target_error_semitones <= 1
    assert 0 < high.target_error_semitones <= 1
    assert low.achieved_normalized == pytest.approx(
        normalize_value(low.achieved_mean, minimum=minimum, maximum=maximum)
    )
    assert high.achieved_normalized <= 1.0
    assert low.safe_normalized_range[0] == pytest.approx(low.achieved_normalized)
    assert high.safe_normalized_range[1] == pytest.approx(high.achieved_normalized)


def test_transposition_moves_all_pitch_declarations_modulo_twelve() -> None:
    plan, score, performance = _inputs()
    rendered = render_performance(plan, score, performance)
    minimum, maximum = _height_range()
    resolution = resolve_height(
        0.0,
        score=score,
        current_mean=height_descriptor(rendered)["mean"],
        minimum=minimum,
        maximum=maximum,
    )

    variant = build_height_variant(plan, score, performance, resolution, purpose="candidate")

    assert variant.scheme_id == SCHEME_ID
    assert variant.performance == performance
    assert variant.plan.tonal_center == (plan.tonal_center + resolution.semitones) % 12
    for base_node, target_node in zip(plan.nodes, variant.plan.nodes, strict=True):
        expected = (
            None
            if base_node.harmonic_focus is None
            else (base_node.harmonic_focus + resolution.semitones) % 12
        )
        assert target_node.harmonic_focus == expected
        assert replace(target_node, harmonic_focus=base_node.harmonic_focus) == base_node
    for base_material, target_material in zip(
        score.materials, variant.score.materials, strict=True
    ):
        assert base_material.material_id == target_material.material_id
        assert base_material.length_units == target_material.length_units
        assert base_material.derived_from == target_material.derived_from
        assert base_material.directions == target_material.directions
        assert base_material.foreground_voice == target_material.foreground_voice
        for base_note, target_note in zip(
            base_material.notes, target_material.notes, strict=True
        ):
            assert target_note.pitch - base_note.pitch == resolution.semitones
            assert _non_pitch_note(target_note) == _non_pitch_note(base_note)
        for base_harmony, target_harmony in zip(
            base_material.harmonies, target_material.harmonies, strict=True
        ):
            assert target_harmony.root_pitch_class == (
                base_harmony.root_pitch_class + resolution.semitones
            ) % 12
            assert replace(
                target_harmony, root_pitch_class=base_harmony.root_pitch_class
            ) == base_harmony
    validate_pipeline(variant.plan, variant.score, variant.performance)


def test_height_distribution_moves_as_a_whole_and_other_performance_data_stays_fixed() -> None:
    plan, score, performance = _inputs()
    base_rendered = render_performance(plan, score, performance)
    base_height = height_descriptor(base_rendered)
    minimum, maximum = _height_range()
    resolutions = [
        resolve_height(
            requested,
            score=score,
            current_mean=base_height["mean"],
            minimum=minimum,
            maximum=maximum,
        )
        for requested in (-1.0, 0.0, 1.0)
    ]
    variants = [
        build_height_variant(
            plan,
            score,
            performance,
            resolution,
            purpose="diagnostic" if resolution.status == "unreachable" else "candidate",
        )
        for resolution in resolutions
    ]
    rendered = [
        render_performance(item.plan, item.score, item.performance) for item in variants
    ]
    descriptors = [height_descriptor(item) for item in rendered]

    assert [item["mean"] for item in descriptors] == sorted(
        item["mean"] for item in descriptors
    )
    for resolution, descriptor in zip(resolutions, descriptors, strict=True):
        for key in ("mean", "median", "p10", "p90"):
            assert descriptor[key] == pytest.approx(
                base_height[key] + resolution.semitones
            )
        for voice, value in base_height["voice_means"].items():
            assert descriptor["voice_means"][voice] == pytest.approx(
                value + resolution.semitones
            )
    for item in rendered:
        assert item.duration_ms == base_rendered.duration_ms
        assert item.pedals == base_rendered.pedals
        assert item.node_intervals == base_rendered.node_intervals
        assert [
            (note.at_ms, note.duration_ms, note.velocity, note.voice)
            for note in item.notes
        ] == [
            (note.at_ms, note.duration_ms, note.velocity, note.voice)
            for note in base_rendered.notes
        ]


def test_unreachable_resolution_requires_diagnostic_purpose() -> None:
    plan, score, performance = _inputs()
    rendered = render_performance(plan, score, performance)
    minimum, maximum = _height_range()
    resolution = resolve_height(
        -1.0,
        score=score,
        current_mean=height_descriptor(rendered)["mean"],
        minimum=minimum,
        maximum=maximum,
    )

    with pytest.raises(ValueError, match="diagnostic"):
        build_height_variant(plan, score, performance, resolution, purpose="candidate")
    diagnostic = build_height_variant(
        plan, score, performance, resolution, purpose="diagnostic"
    )
    assert diagnostic.purpose == "diagnostic_only"
