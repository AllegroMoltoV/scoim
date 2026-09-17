"""Shared deterministic writer for neutral score diagnostic SMFs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mido

from .score_ir import ScoreNote

TICKS_PER_BEAT = 960
TEMPO = 500_000


@dataclass(frozen=True, slots=True)
class NeutralPreviewSegment:
    length_units: int
    notes: tuple[ScoreNote, ...]


def write_neutral_score_preview(
    segments: Sequence[NeutralPreviewSegment],
    target_seconds: float,
    output_path: Path,
    *,
    meta_track_name: str,
    note_track_name: str,
) -> Path:
    """Write score notes with neutral velocity and no performance expression."""
    total_ticks, expected_events = _preview_geometry(segments, target_seconds)
    events = [
        (
            tick,
            priority,
            mido.Message(kind, note=note, velocity=velocity, channel=channel),
        )
        for tick, priority, kind, note, velocity, channel in expected_events
    ]

    midi = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name=meta_track_name, time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=TEMPO, time=0))
    meta.append(mido.MetaMessage("end_of_track", time=total_ticks))
    midi.tracks.append(meta)
    piano = mido.MidiTrack()
    piano.append(mido.MetaMessage("track_name", name=note_track_name, time=0))
    piano.append(mido.Message("program_change", channel=0, program=0, time=0))
    piano.append(mido.Message("program_change", channel=1, program=0, time=0))
    previous_tick = 0
    for tick, _priority, message in events:
        piano.append(message.copy(time=tick - previous_tick))
        previous_tick = tick
    piano.append(mido.MetaMessage("end_of_track", time=max(0, total_ticks - previous_tick)))
    midi.tracks.append(piano)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(filename=str(output_path))
    return output_path


def check_neutral_score_preview(
    segments: Sequence[NeutralPreviewSegment],
    target_seconds: float,
    preview_path: Path,
) -> None:
    """Require a saved preview to contain exactly the deterministic neutral events."""
    total_ticks, expected_events = _preview_geometry(segments, target_seconds)
    midi = mido.MidiFile(preview_path)
    if midi.type != 1 or midi.ticks_per_beat != TICKS_PER_BEAT:
        raise ValueError("score preview MIDI header does not match the neutral format")

    actual_events: list[tuple[int, int, str, int, int, int]] = []
    tempo_events: list[tuple[int, int]] = []
    track_end_ticks: list[int] = []
    for track in midi.tracks:
        absolute_tick = 0
        for message in track:
            absolute_tick += message.time
            if message.type in {"note_on", "note_off"}:
                priority = 0 if message.type == "note_off" else 1
                actual_events.append(
                    (
                        absolute_tick,
                        priority,
                        message.type,
                        message.note,
                        message.velocity,
                        message.channel,
                    )
                )
            elif message.type == "set_tempo":
                tempo_events.append((absolute_tick, message.tempo))
            elif message.type == "control_change":
                raise ValueError("score preview must not contain control changes")
        track_end_ticks.append(absolute_tick)
    actual_events.sort(key=lambda item: (item[0], item[1], item[5], item[3], item[4]))
    if tuple(actual_events) != expected_events:
        raise ValueError("score preview note events do not match the score")
    if tempo_events != [(0, TEMPO)]:
        raise ValueError("score preview tempo events do not match the neutral format")
    if not track_end_ticks or any(tick != total_ticks for tick in track_end_ticks):
        raise ValueError("score preview duration does not match the target duration")


def _preview_geometry(
    segments: Sequence[NeutralPreviewSegment],
    target_seconds: float,
) -> tuple[int, tuple[tuple[int, int, str, int, int, int], ...]]:
    total_units = sum(segment.length_units for segment in segments)
    if total_units <= 0:
        raise ValueError("a score preview requires positive total duration")
    total_ticks = round(target_seconds * 1_000_000 * TICKS_PER_BEAT / TEMPO)
    if total_ticks <= 0:
        raise ValueError("a score preview requires positive target duration")
    events: list[tuple[int, int, str, int, int, int]] = []
    segment_offset = 0
    for segment in segments:
        if segment.length_units <= 0:
            raise ValueError("a score preview segment requires positive duration")
        for note in segment.notes:
            start = round((segment_offset + note.at_units) * total_ticks / total_units)
            end = round(
                (segment_offset + note.at_units + note.duration_units) * total_ticks / total_units
            )
            channel = 0 if note.voice == "upper" else 1
            events.append((start, 1, "note_on", note.pitch, 64, channel))
            events.append((max(start + 1, end), 0, "note_off", note.pitch, 0, channel))
        segment_offset += segment.length_units
    events.sort(key=lambda item: (item[0], item[1], item[5], item[3], item[4]))
    return total_ticks, tuple(events)
