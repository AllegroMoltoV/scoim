from __future__ import annotations

import math

import pytest

from llm_musical_composer.attack_frequency_control import (
    MeasuredFrequencyCandidate,
    build_high_frequency_candidate,
    build_low_frequency_candidate,
    denormalize_frequency,
    resolve_measured_frequency,
    scale_score_resolution,
)
from llm_musical_composer.performance_pipeline import (
    ScoreDirection,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
)


def _note(event_id: str, at: int, pitch: int, voice: str, duration: int = 1):
    return ScoreNote(event_id, at, duration, pitch, voice)


def _score() -> ScoreSpec:
    notes = tuple(
        _note(f"u-{at}", at, 60 + (at % 3) * 2, "upper") for at in range(0, 12, 2)
    ) + tuple(_note(f"l-{at}", at, 48 + (at % 3) * 4, "lower") for at in range(1, 12, 2))
    material = ScoreMaterial(
        "theme",
        12,
        notes,
        directions=(ScoreDirection("d", 3, "dynamic", "mp"),),
        harmonies=(
            ScoreHarmony("h1", 0, 6, 0, "major"),
            ScoreHarmony("h2", 6, 6, 5, "major"),
        ),
        foreground_voice="upper",
    )
    return ScoreSpec("score", 4, (material,))


def test_scale_score_resolution_scales_all_unit_fields() -> None:
    scaled = scale_score_resolution(_score(), factor=3)
    material = scaled.materials[0]
    assert scaled.divisions == 12
    assert material.length_units == 36
    assert material.notes[1].at_units == 6
    assert material.notes[1].duration_units == 3
    assert material.directions[0].at_units == 9
    assert [(item.at_units, item.duration_units) for item in material.harmonies] == [
        (0, 18),
        (18, 18),
    ]


def test_low_candidate_aligns_accompaniment_onsets_and_preserves_notes() -> None:
    scaled = scale_score_resolution(_score(), factor=3)
    stage1 = build_low_frequency_candidate(scaled, stage=1)
    stage2 = build_low_frequency_candidate(scaled, stage=2)
    base_onsets = {note.at_units for note in scaled.materials[0].notes}
    stage1_onsets = {note.at_units for note in stage1.score.materials[0].notes}
    stage2_notes = stage2.score.materials[0].notes

    assert stage1_onsets < base_onsets
    assert all(onset not in stage1_onsets for onset in stage1.changed_onsets["theme"])
    assert len(stage2_notes) == len(scaled.materials[0].notes)
    assert sorted(note.pitch for note in stage2_notes) == sorted(
        note.pitch for note in scaled.materials[0].notes
    )
    assert {note.voice for note in stage2_notes} == {"upper", "lower"}


def test_protected_material_is_not_thinned() -> None:
    score = _score()
    protected = ScoreMaterial(
        "ending",
        score.materials[0].length_units,
        score.materials[0].notes,
        directions=score.materials[0].directions,
        harmonies=score.materials[0].harmonies,
        foreground_voice="upper",
    )
    source = ScoreSpec("score", 4, (protected,))
    candidate = build_low_frequency_candidate(source, stage=2)
    assert candidate.score.materials == source.materials


def test_high_candidate_adds_only_chord_tones_without_same_pitch_overlap() -> None:
    scaled = scale_score_resolution(_score(), factor=3)
    candidate = build_high_frequency_candidate(scaled, stage=3)
    material = candidate.score.materials[0]
    added = [note for note in material.notes if note.event_id.startswith("freq+")]

    assert added
    for note in added:
        harmony = next(
            item
            for item in material.harmonies
            if item.at_units <= note.at_units < item.at_units + item.duration_units
        )
        assert (note.pitch - harmony.root_pitch_class) % 12 in {0, 4, 7}
        assert note.pitch >= 48
        assert note.voice == "lower"
    for index, left in enumerate(material.notes):
        for right in material.notes[index + 1 :]:
            if left.voice == right.voice and left.pitch == right.pitch:
                assert not (
                    left.at_units < right.at_units + right.duration_units
                    and right.at_units < left.at_units + left.duration_units
                )


def test_high_stages_add_monotonically_more_notes() -> None:
    scaled = scale_score_resolution(_score(), factor=3)
    counts = [
        build_high_frequency_candidate(scaled, stage=stage).added_note_count for stage in (1, 2, 3)
    ]
    assert counts == sorted(counts)
    assert len(set(counts)) == 3


def test_denormalize_frequency_uses_corpus_endpoints() -> None:
    assert denormalize_frequency(-1.0, minimum=1.0, maximum=9.0) == 1.0
    assert denormalize_frequency(0.0, minimum=1.0, maximum=9.0) == 5.0
    assert denormalize_frequency(1.0, minimum=1.0, maximum=9.0) == 9.0


@pytest.mark.parametrize("value", [-1.1, 1.1, math.nan, True, "0"])
def test_denormalize_frequency_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        denormalize_frequency(value, minimum=1.0, maximum=9.0)


def test_resolution_reports_unreachable_separately_from_safety() -> None:
    result = resolve_measured_frequency(
        1.0,
        candidates=(
            MeasuredFrequencyCandidate("low", 2.0, True, 3),
            MeasuredFrequencyCandidate("high", 7.0, True, 8),
            MeasuredFrequencyCandidate("unsafe", 9.0, False, 0),
        ),
        minimum=1.0,
        maximum=9.0,
    )
    assert result.candidate_id == "high"
    assert result.status == "unreachable"
    assert result.safe_raw_range == (2.0, 7.0)


def test_resolution_tie_breaks_by_change_count_then_id() -> None:
    result = resolve_measured_frequency(
        0.0,
        candidates=(
            MeasuredFrequencyCandidate("z", 5.0, True, 3),
            MeasuredFrequencyCandidate("b", 5.0, True, 2),
            MeasuredFrequencyCandidate("a", 5.0, True, 2),
        ),
        minimum=1.0,
        maximum=9.0,
    )
    assert result.candidate_id == "a"
    assert result.status == "exact"
