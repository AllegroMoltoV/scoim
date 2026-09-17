from __future__ import annotations

import mido

from llm_musical_composer.smf_notes import collect_events, load_smf_notes, match_notes


def test_collect_and_match_notes_across_tracks(tmp_path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=500)
    first = mido.MidiTrack()
    first.append(mido.Message("note_on", channel=0, note=60, velocity=70, time=0))
    first.append(mido.MetaMessage("end_of_track", time=500))
    second = mido.MidiTrack()
    second.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=500))
    second.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.extend([first, second])
    path = tmp_path / "cross-track.mid"
    midi.save(path)

    reread = mido.MidiFile(path)
    events, reports = collect_events(reread)
    matched, anomalies = match_notes(events, 500)

    assert len(reports) == 2
    assert matched[0]["duration_ticks"] == 500
    assert anomalies["track_local_unmatched_note_off"] == 1
    notes = load_smf_notes(path)
    assert notes[0].onset_ms == 0
    assert notes[0].duration_ms == 500


def test_load_smf_notes_honors_tempo_and_excludes_drums(tmp_path) -> None:
    midi = mido.MidiFile(type=0, ticks_per_beat=500)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=1_000_000, time=0))
    track.append(mido.Message("note_on", channel=0, note=64, velocity=80, time=0))
    track.append(mido.Message("note_on", channel=9, note=36, velocity=90, time=0))
    track.append(mido.Message("note_off", channel=0, note=64, velocity=0, time=500))
    track.append(mido.Message("note_off", channel=9, note=36, velocity=0, time=0))
    midi.tracks.append(track)
    path = tmp_path / "tempo.mid"
    midi.save(path)

    notes = load_smf_notes(path)

    assert len(notes) == 1
    assert notes[0].duration_ms == 1000
