from __future__ import annotations

from dataclasses import replace

import pytest

from llm_musical_composer.composition_ir import Material, Note
from llm_musical_composer.material_development import (
    DevelopmentEvent,
    build_material_development_profile,
    build_material_development_reference_profile_from_directory,
    evaluate_material_development,
    extract_material_development,
    extract_reference_windows,
)
from llm_musical_composer.smf_notes import SmfNote


def events(scale: float = 1.0) -> list[DevelopmentEvent]:
    values = [
        (0, 500),
        (0, 700),
        (0, 900),
        (700, 350),
        (1700, 600),
        (1700, 800),
        (2500, 400),
        (4100, 750),
        (4100, 950),
        (4100, 1150),
        (6000, 500),
    ]
    return [
        DevelopmentEvent(round(onset * scale), round(duration * scale))
        for onset, duration in values
    ]


def material(material_id: str, source: list[DevelopmentEvent]) -> Material:
    notes = tuple(
        Note(
            event_id=f"{material_id}-{index}",
            at_ms=event.onset_ms,
            duration_ms=event.duration_ms,
            pitch=48 + index,
            velocity=70,
        )
        for index, event in enumerate(source)
    )
    return Material(material_id=material_id, duration_ms=8000, notes=notes)


def test_development_features_are_invariant_to_global_time_stretch() -> None:
    original = extract_material_development(events(), span_ms=8000)
    stretched = extract_material_development(events(1.75), span_ms=14000)

    assert original == stretched


def test_empty_and_invalid_inputs_remain_distinct() -> None:
    empty = extract_material_development([], span_ms=8000)

    assert empty == {
        "duration_change_rate": 0.0,
        "ioi_change_rate": 0.0,
        "attack_size_change_rate": 0.0,
    }
    with pytest.raises(ValueError, match="span_ms"):
        extract_material_development([], span_ms=0)
    with pytest.raises(ValueError, match="at least one"):
        build_material_development_profile([], source_count=0)

    profile = build_material_development_profile([empty], source_count=1)
    unavailable = evaluate_material_development((), profile)
    assert unavailable["status"] == "unable_to_investigate"
    assert unavailable["worst_material_id"] is None
    assert extract_reference_windows([]) == []
    assert extract_reference_windows([SmfNote(60, 0, 8000, 70)]) == []


def test_each_uniformity_control_reduces_only_the_target_observation() -> None:
    original_events = events()
    original = extract_material_development(original_events, span_ms=8000)
    equal_duration = extract_material_development(
        [replace(event, duration_ms=500) for event in original_events], span_ms=8000
    )

    onset_map = {
        onset: index * 1000
        for index, onset in enumerate(sorted({event.onset_ms for event in events()}))
    }
    equal_ioi = extract_material_development(
        [replace(event, onset_ms=onset_map[event.onset_ms]) for event in original_events],
        span_ms=8000,
    )
    single_attacks = extract_material_development(
        [replace(event, onset_ms=index * 600) for index, event in enumerate(original_events)],
        span_ms=8000,
    )

    assert original["duration_change_rate"] > equal_duration["duration_change_rate"] == 0
    assert original["ioi_change_rate"] > equal_ioi["ioi_change_rate"] == 0
    assert original["attack_size_change_rate"] > single_attacks["attack_size_change_rate"] == 0


def test_only_joint_lower_quartile_is_extreme_uniformity() -> None:
    varied = extract_material_development(events(), span_ms=8000)
    profile = build_material_development_profile(
        [
            varied,
            {key: value * 0.8 for key, value in varied.items()},
            {key: value * 1.2 for key, value in varied.items()},
            {key: value * 1.4 for key, value in varied.items()},
        ],
        source_count=4,
    )
    uniform = material(
        "uniform",
        [DevelopmentEvent(index * 1000, 500) for index in range(7)],
    )
    duration_only = material(
        "duration-only",
        [replace(event, duration_ms=500) for event in events()],
    )

    report = evaluate_material_development((uniform, duration_only), profile)

    by_id = {item["material_id"]: item for item in report["materials"]}
    assert by_id["uniform"]["extreme_uniformity"] is True
    assert set(by_id["uniform"]["lower_quartile_metrics"]) == {
        "duration_change_rate",
        "ioi_change_rate",
        "attack_size_change_rate",
    }
    assert by_id["duration-only"]["extreme_uniformity"] is False
    assert report["extreme_material_count"] == 1
    assert report["worst_material_id"] == "uniform"


def _write_midi(path, onsets: list[int]) -> None:
    import mido

    midi = mido.MidiFile(type=0, ticks_per_beat=500)
    track = mido.MidiTrack()
    absolute_events: list[tuple[int, int, mido.Message]] = []
    for index, onset in enumerate(onsets):
        pitch = 48 + index
        absolute_events.append((onset, 1, mido.Message("note_on", note=pitch, velocity=70, time=0)))
        absolute_events.append(
            (onset + 300 + index * 20, 0, mido.Message("note_off", note=pitch, time=0))
        )
    previous = 0
    for absolute, _, message in sorted(absolute_events, key=lambda item: (item[0], item[1])):
        message.time = absolute - previous
        previous = absolute
        track.append(message)
    midi.tracks.append(track)
    midi.save(path)


def test_reference_directory_distinguishes_excluded_short_and_unreadable_files(tmp_path) -> None:
    long_onsets = [0, 600, 1500, 2300, 4000, 5200, 7000, 9300, 11000, 13000, 15000]
    _write_midi(tmp_path / "kept.mid", long_onsets)
    _write_midi(tmp_path / "rut.mid", long_onsets)
    _write_midi(tmp_path / "short.mid", [0, 500])
    (tmp_path / "broken.mid").write_bytes(b"not-midi")

    profile = build_material_development_reference_profile_from_directory(
        tmp_path, excluded_names=frozenset({"rut.mid"})
    )

    assert profile["examined_source_count"] == 3
    assert profile["source_count"] == 1
    assert profile["window_count"] > 0
    assert profile["source_files"] == ["kept.mid"]
    assert profile["insufficient_window_files"] == ["short.mid"]
    assert profile["unable_to_investigate"][0]["name"] == "broken.mid"
    assert profile["excluded_files"] == ["rut.mid"]
