from __future__ import annotations

import math

from llm_musical_composer.pilot_features import (
    FeatureStatus,
    NoteEvent,
    build_reference_profile,
    build_reference_profile_from_directory,
    evaluate_features,
    extract_features,
    run_minimal_controls,
)


def notes(scale: float = 1.0) -> list[NoteEvent]:
    pitches = [60, 64, 62, 67, 65, 69, 67, 72]
    durations = [600, 400, 800, 300, 700, 500, 900, 600]
    onsets = [0, 500, 1000, 1400, 2100, 2500, 3200, 3800]
    return [
        NoteEvent(
            pitch=pitch,
            onset_ms=round(onset * scale),
            duration_ms=round(duration * scale),
            velocity=70,
        )
        for pitch, onset, duration in zip(pitches, onsets, durations, strict=True)
    ]


def test_relative_timing_is_invariant_to_global_time_stretch() -> None:
    original = extract_features(notes())
    stretched = extract_features(notes(1.75))

    assert original["relative_timing"] == stretched["relative_timing"]


def test_profile_keeps_axes_separate_and_does_not_create_total_loss() -> None:
    feature_sets = [extract_features(notes()), extract_features(notes(1.3))]
    profile = build_reference_profile(feature_sets, source_count=2)
    evaluation = evaluate_features(feature_sets[0], profile)

    assert set(evaluation["axes"]) == {"pitch_order", "relative_timing", "texture"}
    assert "total_loss" not in evaluation
    assert evaluation["inside_axis_count"] == 3

    filtered = {**profile, "axes": {"texture": profile["axes"]["texture"]}}
    filtered_evaluation = evaluate_features(feature_sets[0], filtered)
    assert set(filtered_evaluation["axes"]) == {"texture"}


def test_minimal_controls_detect_time_invariance_order_and_material_reuse() -> None:
    result = run_minimal_controls(notes())

    assert result["time_stretch"]["status"] == FeatureStatus.PASS.value
    assert result["order_disruption"]["status"] == FeatureStatus.PASS.value
    assert result["material_reuse"]["status"] == FeatureStatus.PASS.value


def test_controls_report_unable_when_there_are_too_few_notes() -> None:
    result = run_minimal_controls(notes()[:2])

    assert result["order_disruption"]["status"] == FeatureStatus.UNABLE.value


def test_build_profile_from_directory_excludes_named_and_records_unreadable(tmp_path) -> None:
    import mido

    for name in ("kept.mid", "rut.mid"):
        midi = mido.MidiFile(type=0, ticks_per_beat=500)
        track = mido.MidiTrack()
        track.append(mido.Message("note_on", note=60, velocity=70, time=0))
        track.append(mido.Message("note_off", note=60, velocity=0, time=500))
        track.append(mido.Message("note_on", note=64, velocity=70, time=0))
        track.append(mido.Message("note_off", note=64, velocity=0, time=500))
        midi.tracks.append(track)
        midi.save(tmp_path / name)
    (tmp_path / "broken.mid").write_bytes(b"not-midi")

    profile = build_reference_profile_from_directory(
        tmp_path, excluded_names=frozenset({"rut.mid"})
    )

    assert profile["source_count"] == 1
    assert profile["source_files"] == ["kept.mid"]
    assert profile["excluded_files"] == ["rut.mid"]
    assert profile["unable_to_investigate"][0]["name"] == "broken.mid"


def test_empty_reference_profile_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="at least one"):
        build_reference_profile([], source_count=0)


def test_evaluation_keeps_unavailable_metric_distinct_from_zero() -> None:
    profile = {
        "axes": {
            "activity": {
                "high_activity_candidate_count": {"p25": 1.0, "median": 1.0, "p75": 2.0},
                "first_high_activity_position": {
                    "p25": 0.25,
                    "median": 0.5,
                    "p75": 0.75,
                },
            }
        }
    }

    evaluation = evaluate_features(
        {
            "activity": {
                "high_activity_candidate_count": 0.0,
                "first_high_activity_position": None,
            }
        },
        profile,
    )

    activity = evaluation["axes"]["activity"]
    assert activity["inside_iqr"] is False
    assert activity["metrics"]["high_activity_candidate_count"]["value"] == 0.0
    unavailable = activity["metrics"]["first_high_activity_position"]
    assert unavailable["value"] is None
    assert unavailable["status"] == FeatureStatus.UNABLE.value
    assert not math.isnan(activity["normalized_distance"])
