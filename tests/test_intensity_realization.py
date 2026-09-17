from __future__ import annotations

from dataclasses import replace

import pytest

from llm_musical_composer.composition_ir import Material, Note
from llm_musical_composer.intensity_realization import extract_intensity_realization


def _note(
    event_id: str,
    at_ms: int,
    duration_ms: int,
    pitch: int,
    velocity: int,
    voice: str,
) -> Note:
    return Note(event_id, at_ms, duration_ms, pitch, velocity, voice)


def _material() -> Material:
    return Material(
        material_id="contrast",
        duration_ms=4000,
        notes=(
            _note("u1", 0, 400, 72, 50, "upper"),
            _note("u2", 1000, 250, 72, 80, "upper"),
            _note("u3", 2000, 400, 72, 50, "upper"),
            _note("u4", 3000, 900, 76, 50, "upper"),
            _note("l1", 0, 1500, 36, 60, "lower"),
            _note("l2", 1000, 1500, 36, 75, "lower"),
            _note("l3", 2000, 1500, 36, 60, "lower"),
            _note("l4", 3000, 900, 43, 60, "lower"),
        ),
    )


def test_extracts_separate_realization_descriptors_without_a_total_score() -> None:
    report = extract_intensity_realization(_material())

    assert report["status"] == "measured"
    assert report["hold"] == {
        "duration_ms": 4000,
        "note_count": 8,
        "global_onset_count": 4,
    }
    assert report["timing"]["median_voice_local_ioi_ms"] == 1000.0
    assert report["timing"]["median_consecutive_ioi_change_log2"] == 0.0
    assert report["accent"]["velocity_p90_minus_median"] == 16.5
    assert report["articulation"]["median_duration_to_next_onset_ratio"] == 0.95
    assert report["repetition"]["consecutive_pitch_set_repeat_ratio"] == pytest.approx(2 / 3)
    assert report["voice_coupling"]["synchronous_onset_ratio"] == 1.0
    assert report["voice_coupling"]["pitch_class_coupling_ratio"] == 0.75
    assert report["bass"]["low_bass_sounding_time_ratio"] == 0.975
    assert report["bass"]["low_bass_velocity_median"] == 60.0
    assert "score" not in report
    assert "target" not in report
    assert "passes" not in report


def test_time_stretch_changes_ioi_but_not_scale_free_descriptors() -> None:
    original = _material()
    stretched = replace(
        original,
        duration_ms=8000,
        notes=tuple(
            replace(note, at_ms=note.at_ms * 2, duration_ms=note.duration_ms * 2)
            for note in original.notes
        ),
    )

    before = extract_intensity_realization(original)
    after = extract_intensity_realization(stretched)

    assert after["timing"]["median_voice_local_ioi_ms"] == 2000.0
    for group, metric in (
        ("timing", "median_consecutive_ioi_change_log2"),
        ("articulation", "median_duration_to_next_onset_ratio"),
        ("repetition", "consecutive_pitch_set_repeat_ratio"),
        ("voice_coupling", "synchronous_onset_ratio"),
        ("voice_coupling", "pitch_class_coupling_ratio"),
        ("bass", "low_bass_sounding_time_ratio"),
    ):
        assert after[group][metric] == before[group][metric]


def test_uniform_velocity_offset_does_not_fake_accent_contrast() -> None:
    original = _material()
    louder = replace(
        original,
        notes=tuple(replace(note, velocity=note.velocity + 10) for note in original.notes),
    )

    before = extract_intensity_realization(original)
    after = extract_intensity_realization(louder)

    assert after["accent"]["velocity_median"] == before["accent"]["velocity_median"] + 10
    assert (
        after["accent"]["velocity_p90_minus_median"]
        == before["accent"]["velocity_p90_minus_median"]
    )


def test_missing_temporal_or_voice_evidence_is_not_changed_to_zero() -> None:
    material = Material(
        material_id="one-note",
        duration_ms=1000,
        notes=(_note("u1", 100, 500, 72, 60, "upper"),),
    )

    report = extract_intensity_realization(material)

    assert report["status"] == "measured_with_unavailable_metrics"
    assert report["timing"]["median_voice_local_ioi_ms"] is None
    assert report["voice_coupling"]["synchronous_onset_ratio"] is None
    assert report["bass"]["low_bass_sounding_time_ratio"] is None
    assert set(report["unavailable_metrics"]) >= {
        "median_voice_local_ioi_ms",
        "synchronous_onset_ratio",
        "low_bass_sounding_time_ratio",
    }

    asynchronous = Material(
        material_id="asynchronous",
        duration_ms=1000,
        notes=(
            _note("u1", 100, 200, 72, 60, "upper"),
            _note("l1", 500, 200, 48, 60, "lower"),
        ),
    )
    asynchronous_report = extract_intensity_realization(asynchronous)

    assert asynchronous_report["voice_coupling"]["synchronous_onset_ratio"] == 0.0
    assert asynchronous_report["voice_coupling"]["pitch_class_coupling_ratio"] is None
    assert "pitch_class_coupling_ratio" in asynchronous_report["unavailable_metrics"]


def test_invalid_material_is_rejected_instead_of_reported_as_silence() -> None:
    invalid = replace(_material(), duration_ms=0)

    with pytest.raises(ValueError, match="duration_ms must be positive"):
        extract_intensity_realization(invalid)
