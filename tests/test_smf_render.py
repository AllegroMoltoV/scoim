from __future__ import annotations

import hashlib
from dataclasses import replace

import mido
import pytest

from llm_musical_composer.composition_ir import Material, Note, Use
from llm_musical_composer.music_dsl import parse_composition
from llm_musical_composer.smf_render import render_composition, tonic_ending_pitches
from tests.test_music_dsl import CONTRACT_ENDING_SOURCE, ENDING_SOURCE, PARTS_SOURCE, VALID_SOURCE


def test_render_is_deterministic_and_round_trips(tmp_path) -> None:
    composition = parse_composition(VALID_SOURCE)
    first = tmp_path / "first.mid"
    second = tmp_path / "second.mid"

    first_result = render_composition(composition, first)
    second_result = render_composition(composition, second)

    assert (
        hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    )
    midi = mido.MidiFile(first)
    assert midi.ticks_per_beat == 500
    assert first_result.duration_ms == 32000
    assert second_result.note_count == 3
    messages = list(mido.merge_tracks(midi.tracks))
    assert any(message.type == "program_change" and message.program == 0 for message in messages)
    assert sum(message.type == "note_on" and message.velocity > 0 for message in messages) == 3


def test_parts_and_equivalent_flat_form_render_identically(tmp_path) -> None:
    flat = parse_composition(VALID_SOURCE)
    hierarchical = parse_composition(PARTS_SOURCE)
    flat_path = tmp_path / "flat.mid"
    hierarchical_path = tmp_path / "hierarchical.mid"

    render_composition(flat, flat_path)
    render_composition(hierarchical, hierarchical_path)

    assert flat_path.read_bytes() == hierarchical_path.read_bytes()


def test_render_adds_deterministic_minor_tonic_hold_and_closes_events(tmp_path) -> None:
    composition = parse_composition(ENDING_SOURCE)
    path = tmp_path / "ending.mid"

    result = render_composition(composition, path)

    midi = mido.MidiFile(path)
    absolute = 0
    events: list[tuple[int, mido.Message]] = []
    for message in midi.tracks[1]:
        absolute += message.time
        events.append((absolute, message))
    ending_pitches = {50, 53, 57, 62}
    ending_on = [
        (tick, message.note)
        for tick, message in events
        if message.type == "note_on" and message.velocity > 0 and message.note in ending_pitches
    ]
    ending_off = [
        (tick, message.note)
        for tick, message in events
        if message.type == "note_off" and message.note in ending_pitches
    ]
    pedal_off_ticks = [
        tick
        for tick, message in events
        if message.type == "control_change" and message.control == 64 and message.value == 0
    ]

    assert ending_on == [(32250, 50), (32250, 53), (32250, 57), (32250, 62)]
    assert ending_off == [(36000, 50), (36000, 53), (36000, 57), (36000, 62)]
    assert pedal_off_ticks == [32000, 36000]
    assert result.duration_ms == 36000
    assert result.note_count == 7


def test_tonic_ending_resolves_each_voice_by_nearest_motion() -> None:
    composition = parse_composition(ENDING_SOURCE)
    closing = Material(
        material_id="closing",
        duration_ms=30_000,
        notes=(
            Note("lower", 29_200, 500, 38, 70, "lower"),
            Note("upper", 29_200, 500, 50, 74, "upper"),
        ),
    )
    composition = replace(composition, form=(Use("closing"),), materials=(closing,))

    pitches = tonic_ending_pitches(composition)

    assert pitches == (38, 41, 45, 50)
    assert pitches[-1] == 50
    assert {pitch % 12 for pitch in pitches} == {2, 5, 9}


def test_tonic_ending_uses_last_onset_as_context_without_voice_labels() -> None:
    composition = parse_composition(ENDING_SOURCE)

    assert tonic_ending_pitches(composition) == (50, 53, 57, 62)


def test_tonic_ending_uses_neutral_anchors_for_an_empty_body() -> None:
    composition = parse_composition(ENDING_SOURCE)
    empty = Material(material_id="empty", duration_ms=30_000, notes=())
    composition = replace(composition, form=(Use("empty"),), materials=(empty,))

    assert tonic_ending_pitches(composition) == (50, 53, 57, 62)


def test_tonic_ending_rejects_missing_or_invalid_tonal_center() -> None:
    composition = parse_composition(ENDING_SOURCE)

    with pytest.raises(ValueError, match="requires tonal_center"):
        tonic_ending_pitches(replace(composition, tonal_center=None))
    with pytest.raises(ValueError, match="no tonic pitch"):
        tonic_ending_pitches(replace(composition, tonal_center=99))


def test_tonic_ending_uses_outer_pitches_when_final_attack_has_doubled_voices() -> None:
    composition = parse_composition(ENDING_SOURCE)
    closing = Material(
        material_id="closing",
        duration_ms=30_000,
        notes=(
            Note("lower-high", 29_200, 500, 50, 70, "lower"),
            Note("lower-low", 29_200, 500, 38, 70, "lower"),
            Note("upper-low", 29_200, 500, 50, 74, "upper"),
            Note("upper-high", 29_200, 500, 57, 74, "upper"),
        ),
    )
    composition = replace(composition, form=(Use("closing"),), materials=(closing,))

    assert tonic_ending_pitches(composition) == (38, 53, 57, 62)


def test_render_preserves_non_latin_title_as_utf8_meta_text(tmp_path) -> None:
    composition = parse_composition(ENDING_SOURCE.replace('title="pilot"', 'title="終止試験"'))
    path = tmp_path / "unicode-title.mid"

    render_composition(composition, path)

    midi = mido.MidiFile(path, charset="utf-8")
    assert midi.tracks[0][0].type == "track_name"
    assert midi.tracks[0][0].name == "終止試験"


def test_section_contract_metadata_does_not_change_rendered_events(tmp_path) -> None:
    contracted = parse_composition(CONTRACT_ENDING_SOURCE, require_section_contract=True)
    without_contract = replace(
        contracted,
        form=tuple(
            replace(section, role=None, energy=None, attack_style=None)
            for section in contracted.form
        ),
    )
    contracted_path = tmp_path / "contracted.mid"
    legacy_path = tmp_path / "legacy.mid"

    render_composition(contracted, contracted_path)
    render_composition(without_contract, legacy_path)

    assert contracted_path.read_bytes() == legacy_path.read_bytes()


def test_render_preserves_binary_sustain_contract_at_exact_times(tmp_path) -> None:
    composition = parse_composition(CONTRACT_ENDING_SOURCE, require_section_contract=True)
    path = tmp_path / "sustain.mid"

    render_composition(composition, path)

    midi = mido.MidiFile(path)
    absolute = 0
    pedal_events = []
    for message in midi.tracks[1]:
        absolute += message.time
        if message.type == "control_change" and message.control == 64:
            pedal_events.append((absolute, message.value))
    assert pedal_events == [
        (0, 127),
        (4_500, 0),
        (4_700, 127),
        (13_700, 0),
        (14_000, 127),
        (18_500, 0),
        (18_700, 127),
        (27_700, 0),
        (28_000, 127),
        (32_500, 0),
        (32_700, 127),
        (41_700, 0),
        (42_000, 0),
        (46_000, 0),
    ]
