from __future__ import annotations

import json
from pathlib import Path

import mido
import pytest

from llm_musical_composer.activity_selection import (
    build_activity_reference_profile,
    build_activity_reference_profile_from_directory,
    extract_activity_features,
    main,
    replay_activity_selection,
)
from llm_musical_composer.music_dsl import parse_composition
from llm_musical_composer.pilot_features import NoteEvent, build_reference_profile
from llm_musical_composer.smf_render import render_composition
from tests.test_music_dsl import ENDING_SOURCE


def _section(start: int, pitches: list[int], *, velocity: int, spacing: int) -> list[NoteEvent]:
    return [
        NoteEvent(pitch, start + index * spacing, 300, velocity)
        for index, pitch in enumerate(pitches)
    ]


def _write_simple_midi(path: Path, pitches: tuple[int, ...] = (60, 64, 67, 72)) -> None:
    midi = mido.MidiFile(type=0, ticks_per_beat=500)
    track = mido.MidiTrack()
    for pitch in pitches:
        track.append(mido.Message("note_on", note=pitch, velocity=70, time=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=500))
    midi.tracks.append(track)
    midi.save(path)


def test_extract_activity_features_reports_count_and_first_position() -> None:
    notes = (
        _section(0, [60, 64], velocity=40, spacing=350)
        + _section(1_000, [48, 55, 60, 64, 67, 72], velocity=100, spacing=100)
        + _section(2_000, [60, 64], velocity=40, spacing=350)
        + _section(3_000, [65, 69], velocity=40, spacing=350)
    )

    features = extract_activity_features(notes, window_count=4)

    assert features["activity"]["high_activity_candidate_count"] == 1.0
    assert features["activity"]["first_high_activity_position"] == 0.375


def test_no_high_activity_keeps_zero_count_and_unavailable_position_separate() -> None:
    notes = [
        NoteEvent(60, 0, 300, 64),
        NoteEvent(60, 1_000, 300, 64),
        NoteEvent(60, 2_000, 300, 64),
        NoteEvent(60, 3_000, 300, 64),
    ]

    features = extract_activity_features(notes, window_count=4)

    assert features["activity"]["high_activity_candidate_count"] == 0.0
    assert features["activity"]["first_high_activity_position"] is None


def test_activity_reference_profile_counts_available_positions_separately() -> None:
    profile = build_activity_reference_profile(
        [
            {
                "activity": {
                    "high_activity_candidate_count": 0.0,
                    "first_high_activity_position": None,
                }
            },
            {
                "activity": {
                    "high_activity_candidate_count": 2.0,
                    "first_high_activity_position": 0.4,
                }
            },
        ]
    )

    assert profile["source_count"] == 2
    count = profile["axes"]["activity"]["high_activity_candidate_count"]
    position = profile["axes"]["activity"]["first_high_activity_position"]
    assert count["available_count"] == 2
    assert position["available_count"] == 1


def test_activity_directory_profile_records_unreadable_and_excluded_files(tmp_path) -> None:
    _write_simple_midi(tmp_path / "kept.mid")
    _write_simple_midi(tmp_path / "rut.mid")
    (tmp_path / "broken.mid").write_bytes(b"not-midi")

    profile = build_activity_reference_profile_from_directory(
        tmp_path, excluded_names=frozenset({"rut.mid"})
    )

    assert profile["source_count"] == 1
    assert profile["source_files"] == ["kept.mid"]
    assert profile["excluded_files"] == ["rut.mid"]
    assert profile["unable_to_investigate"][0]["name"] == "broken.mid"


def test_activity_profile_requires_usable_source() -> None:
    with pytest.raises(ValueError, match="usable"):
        build_activity_reference_profile([])


def test_activity_extraction_rejects_empty_notes() -> None:
    with pytest.raises(ValueError, match="complete pitched notes"):
        extract_activity_features([])


def test_replay_uses_saved_candidates_without_api_calls(tmp_path) -> None:
    source_dir = tmp_path / "source"
    candidate_dir = tmp_path / "run" / "candidates"
    source_dir.mkdir()
    candidate_dir.mkdir(parents=True)
    _write_simple_midi(source_dir / "reference.mid")
    composition = parse_composition(ENDING_SOURCE)
    for number in range(1, 4):
        render_composition(composition, candidate_dir / f"candidate-{number}.mid")
    base_profile = build_reference_profile(
        [
            {
                "texture": {
                    "pitch_range": 4.0,
                    "maximum_polyphony": 1.0,
                    "notes_per_normalized_span": 1.0,
                }
            }
        ],
        source_count=1,
    )
    base_profile_path = tmp_path / "base-profile.json"
    base_profile_path.write_text(json.dumps(base_profile), encoding="utf-8")

    output_dir = tmp_path / "replay"
    result = replay_activity_selection(
        source_dir=source_dir,
        base_profile_path=base_profile_path,
        candidate_dir=candidate_dir,
        output_dir=output_dir,
    )

    assert result["api_call_count"] == 0
    assert len(result["candidates"]) == 3
    assert result["selected_candidate"] == "candidate-1"
    assert (output_dir / "activity-reference-profile.json").is_file()
    assert (output_dir / "candidate-evaluations.json").is_file()


@pytest.mark.parametrize(("selected", "expected"), [("candidate-1", 0), (None, 2)])
def test_main_returns_status_from_replay(monkeypatch, selected, expected) -> None:
    monkeypatch.setattr(
        "llm_musical_composer.activity_selection.replay_activity_selection",
        lambda **kwargs: {"selected_candidate": selected},
    )

    assert main([]) == expected
