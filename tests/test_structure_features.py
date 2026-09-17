from __future__ import annotations

import pytest

from llm_musical_composer.pilot_features import NoteEvent
from llm_musical_composer.structure_features import (
    build_structure_reference_profile,
    extract_structure_features,
    run_structure_controls,
)


def section(
    start: int,
    pitches: list[int],
    *,
    velocity: int = 64,
    spacing: int = 200,
    duration: int = 300,
) -> list[NoteEvent]:
    return [
        NoteEvent(pitch, start + index * spacing, duration, velocity)
        for index, pitch in enumerate(pitches)
    ]


def test_structure_features_are_invariant_to_global_time_stretch() -> None:
    notes = section(0, [60, 64, 62, 67]) + section(1_000, [65, 69, 67, 72])
    stretched = [
        NoteEvent(note.pitch, note.onset_ms * 2, note.duration_ms * 2, note.velocity)
        for note in notes
    ]

    assert extract_structure_features(notes, window_count=4) == extract_structure_features(
        stretched, window_count=4
    )


def test_interval_similarity_detects_transposed_recurrence_and_repetition_break() -> None:
    a = section(0, [60, 64, 62, 67])
    b = section(1_000, [60, 61, 66, 63])
    transposed_a = section(2_000, [67, 71, 69, 74])
    c = section(2_000, [60, 66, 61, 70])

    aba = extract_structure_features(a + b + transposed_a, window_count=3)
    abc = extract_structure_features(a + b + c, window_count=3)

    assert aba["similarity_matrices"]["pitch_interval"][0][2] == 1.0
    assert (
        aba["similarity_matrices"]["pitch_interval"][0][2]
        > abc["similarity_matrices"]["pitch_interval"][0][2]
    )
    assert aba["repetition"]["pitch_interval"]["delayed_pairs"]


def test_reordering_sections_changes_activity_position_or_boundary_evidence() -> None:
    quiet = section(0, [60, 64], velocity=40, spacing=350)
    busy = section(1_000, [48, 55, 60, 64, 67, 72], velocity=100, spacing=100)
    original = extract_structure_features(quiet + busy, window_count=2)
    reordered_notes = [
        *section(0, [48, 55, 60, 64, 67, 72], velocity=100, spacing=100),
        *section(1_000, [60, 64], velocity=40, spacing=350),
    ]
    reordered = extract_structure_features(reordered_notes, window_count=2)

    assert (
        original["activity"]["velocity"]["peak_positions"]
        != reordered["activity"]["velocity"]["peak_positions"]
    )
    assert original["novelty"] != reordered["novelty"]


def test_flattening_velocity_only_changes_velocity_activity_series() -> None:
    notes = section(0, [60, 64], velocity=30) + section(1_000, [65, 69], velocity=100)
    flat = [NoteEvent(note.pitch, note.onset_ms, note.duration_ms, 65) for note in notes]

    original = extract_structure_features(notes, window_count=2)["activity"]
    flattened = extract_structure_features(flat, window_count=2)["activity"]

    assert flattened["velocity"]["range"] == 0.0
    assert original["velocity"]["range"] > 0.0
    for name in ("onset_density", "polyphony", "pitch_median"):
        assert flattened[name] == original[name]


def test_empty_input_is_reported_as_unable_instead_of_zero_music() -> None:
    result = extract_structure_features([], window_count=16)

    assert result["status"] == "unable_to_investigate"
    assert result["windows"] == []


def test_declared_material_reuse_is_compared_with_event_similarity() -> None:
    notes = (
        section(0, [60, 64, 62, 67])
        + section(1_000, [60, 61, 66, 63])
        + section(2_000, [67, 71, 69, 74])
    )

    result = extract_structure_features(
        notes, window_count=3, declared_material_ids=["A", "B", "A"]
    )

    comparison = result["declared_form_comparison"]
    assert comparison[0]["same_declared_material"] is True
    assert comparison[0]["pitch_interval_similarity"] == 1.0


def test_structure_reference_profile_keeps_axes_separate() -> None:
    first = extract_structure_features(
        section(0, [60, 64]) + section(1_000, [67, 72]), window_count=2
    )
    second = extract_structure_features(
        section(0, [48, 55, 60]) + section(1_000, [65, 69]), window_count=2
    )

    profile = build_structure_reference_profile([first, second])

    assert "total_loss" not in profile
    assert profile["source_count"] == 2
    assert set(profile["axes"]) == {"boundaries", "repetition", "activity"}


def test_all_structure_controls_pass_on_deliberate_counterexamples() -> None:
    controls = run_structure_controls()

    assert {item["status"] for item in controls.values()} == {"pass"}


def test_structure_feature_argument_and_profile_failures_are_explicit() -> None:
    with pytest.raises(ValueError, match="window_count"):
        extract_structure_features(section(0, [60, 64]), window_count=1)
    with pytest.raises(ValueError, match="declared_material_ids"):
        extract_structure_features(
            section(0, [60, 64]), window_count=2, declared_material_ids=["A"]
        )
    with pytest.raises(ValueError, match="usable"):
        build_structure_reference_profile([{"status": "unable_to_investigate"}])


def test_sparse_windows_keep_similarity_and_profile_unavailable_states_explicit() -> None:
    sparse = extract_structure_features(
        [NoteEvent(60, 0, 100, 60), NoteEvent(64, 3_900, 100, 70)], window_count=4
    )
    profile = build_structure_reference_profile([sparse])

    assert sparse["similarity_matrices"]["pitch_interval"][0][2] == 0.0
    assert (
        profile["axes"]["repetition"]["pitch_interval"]["median_recurrence_distance"]["status"]
        == "unable_to_investigate"
    )


def test_onset_order_and_half_window_phase_are_explicit_coordinate_options() -> None:
    notes = section(0, [60, 64, 62, 67]) + section(2_000, [65, 69, 67, 72])

    onset_order = extract_structure_features(notes, window_count=4, coordinate="onset_order")
    shifted = extract_structure_features(
        notes,
        window_count=4,
        coordinate="elapsed_time",
        phase_fraction=0.5,
    )

    assert onset_order["coordinate"] == "onset_order"
    assert len(onset_order["windows"]) == 4
    assert shifted["phase_fraction"] == 0.5
    assert len(shifted["windows"]) == 3


def test_structure_coordinate_options_reject_unknown_values() -> None:
    with pytest.raises(ValueError, match="coordinate"):
        extract_structure_features(section(0, [60, 64]), coordinate="beats")
    with pytest.raises(ValueError, match="phase_fraction"):
        extract_structure_features(section(0, [60, 64]), phase_fraction=1.0)
