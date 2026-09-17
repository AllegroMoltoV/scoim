from __future__ import annotations

from pathlib import Path

import mido

from llm_musical_composer.reference_decomposition import (
    build_observed_performance,
    load_observed_smf,
    observed_smf_differences,
    write_observed_smf,
)


def _write_full_event_fixture(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("track_name", name="conductor", time=0))
    conductor.append(mido.MetaMessage("set_tempo", tempo=600_000, time=0))
    conductor.append(mido.MetaMessage("time_signature", numerator=3, denominator=4, time=0))
    conductor.append(mido.MetaMessage("end_of_track", time=960))
    performance = mido.MidiTrack()
    performance.append(mido.MetaMessage("track_name", name="piano", time=0))
    performance.append(mido.Message("program_change", channel=3, program=0, time=0))
    performance.append(mido.Message("control_change", channel=3, control=1, value=17, time=0))
    performance.append(mido.Message("control_change", channel=3, control=64, value=127, time=0))
    performance.append(mido.Message("pitchwheel", channel=3, pitch=31, time=0))
    performance.append(mido.Message("sysex", data=(1, 2, 3), time=0))
    performance.append(mido.Message("note_on", channel=3, note=60, velocity=71, time=0))
    performance.append(mido.Message("note_off", channel=3, note=60, velocity=19, time=480))
    performance.append(mido.Message("control_change", channel=3, control=64, value=0, time=0))
    performance.append(mido.MetaMessage("end_of_track", time=480))
    midi.tracks.extend((conductor, performance))
    midi.save(path)


def test_observed_smf_roundtrip_preserves_all_event_fields(tmp_path: Path) -> None:
    source = tmp_path / "source.mid"
    regenerated = tmp_path / "regenerated.mid"
    _write_full_event_fixture(source)

    observed = load_observed_smf(source)
    write_observed_smf(observed, regenerated)
    reloaded = load_observed_smf(regenerated)

    assert observed.midi_type == 1
    assert observed.ticks_per_beat == 480
    assert observed_smf_differences(observed, reloaded) == ()
    assert {event.message_type for event in observed.events} >= {
        "note_on",
        "note_off",
        "control_change",
        "program_change",
        "pitchwheel",
        "sysex",
        "set_tempo",
        "time_signature",
    }
    note_off = next(event for event in observed.events if event.message_type == "note_off")
    assert dict(note_off.fields)["velocity"] == 19
    assert dict(note_off.fields)["channel"] == 3


def test_performance_projection_keeps_track_local_crossing_notes_separate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "crossing.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    first = mido.MidiTrack()
    first.append(mido.Message("note_on", channel=0, note=60, velocity=80, time=0))
    first.append(mido.Message("note_off", channel=0, note=60, velocity=10, time=480))
    second = mido.MidiTrack()
    second.append(mido.Message("note_on", channel=0, note=60, velocity=70, time=240))
    second.append(mido.Message("note_off", channel=0, note=60, velocity=20, time=120))
    midi.tracks.extend((first, second))
    midi.save(source)

    performance = build_observed_performance(load_observed_smf(source))

    assert len(performance.notes) == 2
    by_track = {note.track: note for note in performance.notes}
    assert by_track[0].onset_tick == 0
    assert by_track[0].offset_tick == 480
    assert by_track[0].note_off_velocity == 10
    assert by_track[1].onset_tick == 240
    assert by_track[1].offset_tick == 360
    assert by_track[1].note_off_velocity == 20
    assert performance.note_matching_status == "assessed"


def test_overlapping_same_track_note_is_reported_as_ambiguous(tmp_path: Path) -> None:
    source = tmp_path / "ambiguous.mid"
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", channel=0, note=60, velocity=80, time=0))
    track.append(mido.Message("note_on", channel=0, note=60, velocity=70, time=120))
    track.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=120))
    track.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=120))
    midi.tracks.append(track)
    midi.save(source)

    performance = build_observed_performance(load_observed_smf(source))

    assert len(performance.notes) == 2
    assert performance.note_matching_status == "ambiguous"
    assert performance.ambiguous_note_count == 2


def test_attack_groups_use_first_attack_without_chain_merging(tmp_path: Path) -> None:
    source = tmp_path / "attacks.mid"
    midi = mido.MidiFile(type=0, ticks_per_beat=1000)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=1_000_000, time=0))
    for delta, pitch in ((0, 60), (20, 64), (20, 67)):
        track.append(mido.Message("note_on", note=pitch, velocity=80, time=delta))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=0))
    midi.tracks.append(track)
    midi.save(source)

    performance = build_observed_performance(load_observed_smf(source))

    assert [group.onset_us for group in performance.attack_groups] == [0, 40_000]
    assert [len(group.note_on_event_ids) for group in performance.attack_groups] == [2, 1]
    assert performance.attack_group_window_us == 30_000


def test_roundtrip_difference_reports_track_and_field_changes(tmp_path: Path) -> None:
    source = tmp_path / "source.mid"
    _write_full_event_fixture(source)
    observed = load_observed_smf(source)
    changed_event = next(event for event in observed.events if event.message_type == "note_on")
    changed_fields = tuple(
        (key, 72 if key == "note" else value) for key, value in changed_event.fields
    )
    changed = type(observed)(
        midi_type=observed.midi_type,
        ticks_per_beat=observed.ticks_per_beat,
        track_count=observed.track_count,
        events=tuple(
            type(event)(
                event_id=event.event_id,
                track=event.track,
                event_index=event.event_index,
                absolute_tick=event.absolute_tick,
                is_meta=event.is_meta,
                message_type=event.message_type,
                fields=changed_fields if event.event_id == changed_event.event_id else event.fields,
            )
            for event in observed.events
        ),
        source_sha256=observed.source_sha256,
        ledger_sha256=observed.ledger_sha256,
    )

    differences = observed_smf_differences(observed, changed)

    assert len(differences) == 1
    assert differences[0].event_id == changed_event.event_id
    assert "fields" in differences[0].changed_fields


def test_roundtrip_difference_detects_timing_change_and_pedal_removal(tmp_path: Path) -> None:
    source = tmp_path / "source.mid"
    _write_full_event_fixture(source)
    observed = load_observed_smf(source)
    pedal = next(
        event
        for event in observed.events
        if event.message_type == "control_change" and event.field("control") == 64
    )
    timing_changed = type(observed)(
        midi_type=observed.midi_type,
        ticks_per_beat=observed.ticks_per_beat,
        track_count=observed.track_count,
        events=tuple(
            type(event)(
                event_id=event.event_id,
                track=event.track,
                event_index=event.event_index,
                absolute_tick=event.absolute_tick + 1
                if event.event_id == pedal.event_id
                else event.absolute_tick,
                is_meta=event.is_meta,
                message_type=event.message_type,
                fields=event.fields,
            )
            for event in observed.events
        ),
        source_sha256=observed.source_sha256,
        ledger_sha256=observed.ledger_sha256,
    )
    without_pedal = type(observed)(
        midi_type=observed.midi_type,
        ticks_per_beat=observed.ticks_per_beat,
        track_count=observed.track_count,
        events=tuple(
            event
            for event in observed.events
            if not (event.message_type == "control_change" and event.field("control") == 64)
        ),
        source_sha256=observed.source_sha256,
        ledger_sha256=observed.ledger_sha256,
    )

    timing_differences = observed_smf_differences(observed, timing_changed)
    pedal_differences = observed_smf_differences(observed, without_pedal)

    assert any("absolute_tick" in difference.changed_fields for difference in timing_differences)
    assert any(
        "missing_event" in difference.changed_fields or "event_id" in difference.changed_fields
        for difference in pedal_differences
    )
