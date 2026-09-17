from __future__ import annotations

import pytest

from llm_musical_composer.ending_closure import (
    build_closure_profile,
    evaluate_ending_discrimination,
    extract_closure_features,
    run_closure_controls,
)
from llm_musical_composer.pilot_features import NoteEvent


def ending_example(offset: int = 0) -> list[NoteEvent]:
    notes: list[NoteEvent] = []
    for onset in range(0, 7_000, 250):
        notes.append(NoteEvent(60 + (onset // 250) % 5, offset + onset, 180, 75))
    for onset in (7_200, 7_900, 8_700):
        notes.append(NoteEvent(60, offset + onset, 900, 52))
    return notes


def test_appended_silence_is_diagnostic_only() -> None:
    notes = ending_example()
    sounding_end = max(note.onset_ms + note.duration_ms for note in notes)

    without_silence = extract_closure_features(notes, smf_end_ms=sounding_end)
    with_silence = extract_closure_features(notes, smf_end_ms=sounding_end + 5_000)

    assert without_silence["comparison_metrics"] == with_silence["comparison_metrics"]
    assert (
        without_silence["diagnostics"]["terminal_silence_ratio"]
        < with_silence["diagnostics"]["terminal_silence_ratio"]
    )


def test_closure_profile_keeps_metrics_separate_without_total_loss() -> None:
    features = [extract_closure_features(ending_example(index * 10)) for index in range(4)]

    profile = build_closure_profile(features)

    assert "total_loss" not in profile
    assert profile["source_count"] == 4
    assert profile["metrics"]["onset_density_ratio"]["available_count"] == 4


def test_real_endings_can_be_distinguished_from_busy_middle_segments() -> None:
    corpus = [ending_example(index * 10) for index in range(12)]

    result = evaluate_ending_discrimination(corpus, bootstrap_samples=400, seed=7)

    assert result["status"] == "pass"
    assert result["actual_win_rate"] == 1.0
    assert result["confidence_interval_95"]["lower"] > 0.5
    assert result["metric_win_counts"]["onset_density_ratio"]["actual_better"] > 0


def test_ending_discrimination_reports_unable_for_too_few_songs() -> None:
    result = evaluate_ending_discrimination([ending_example()], bootstrap_samples=20, seed=1)

    assert result["status"] == "unable_to_investigate"


def test_terminal_silence_control_passes_without_using_silence_as_improvement() -> None:
    result = run_closure_controls(ending_example())

    assert result["terminal_silence_only"]["status"] == "pass"


def test_closure_failure_states_are_not_converted_to_zero() -> None:
    empty = extract_closure_features([])
    assert empty["status"] == "unable_to_investigate"
    assert run_closure_controls([])["terminal_silence_only"]["status"] == ("unable_to_investigate")
    with pytest.raises(ValueError, match="target interval"):
        extract_closure_features(ending_example(), target_interval=(0.05, 0.2))
    with pytest.raises(ValueError, match="at least one"):
        build_closure_profile([])


def test_uniform_music_fails_the_real_ending_discrimination_gate() -> None:
    uniform = [NoteEvent(60 + index % 3, index * 200, 150, 70) for index in range(50)]

    result = evaluate_ending_discrimination(
        [uniform for _ in range(8)], bootstrap_samples=100, seed=2
    )

    assert result["status"] == "fail"
    assert result["confidence_interval_95"]["upper"] <= 0.5
    assert result["comparisons"][0]["required_better_metrics"] == 2
