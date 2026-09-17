"""型付き内部表現を決定的なピアノ SMF へ変換する。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mido

from llm_musical_composer.composition_ir import Composition

TICKS_PER_BEAT = 500
TEMPO = 500_000
ENDING_BREATH_MS = 250
ENDING_VELOCITY = 56


@dataclass(frozen=True)
class RenderResult:
    path: Path
    duration_ms: int
    note_count: int


def _append_absolute_events(
    track: mido.MidiTrack, events: list[tuple[int, int, mido.Message]]
) -> None:
    previous_tick = 0
    for tick, _, message in sorted(events, key=lambda item: (item[0], item[1])):
        message.time = tick - previous_tick
        track.append(message)
        previous_tick = tick


def _nearest_pitch_class(reference: int, pitch_class: int, *, minimum: int, maximum: int) -> int:
    candidates = [pitch for pitch in range(minimum, maximum + 1) if pitch % 12 == pitch_class]
    if not candidates:
        raise ValueError("no tonic pitch is available in the requested piano range")
    return min(candidates, key=lambda pitch: (abs(pitch - reference), pitch))


def _closing_voice_anchors(composition: Composition) -> tuple[int, int]:
    events: list[tuple[int, int, str | None]] = []
    offset = 0
    materials = composition.material_by_id
    for section in composition.form:
        material = materials[section.material_id]
        events.extend((offset + note.at_ms, note.pitch, note.voice) for note in material.notes)
        offset += material.duration_ms
    if not events:
        return 48, 60

    last_onset = max(onset for onset, _, _ in events)
    last_pitches = [pitch for onset, pitch, _ in events if onset == last_onset]
    upper_events = [(onset, pitch) for onset, pitch, voice in events if voice == "upper"]
    lower_events = [(onset, pitch) for onset, pitch, voice in events if voice == "lower"]
    upper = max(
        upper_events,
        default=(last_onset, max(last_pitches)),
        key=lambda item: (item[0], item[1]),
    )
    lower = max(
        lower_events,
        default=(last_onset, min(last_pitches)),
        key=lambda item: (item[0], -item[1]),
    )
    return lower[1], upper[1]


def tonic_ending_pitches(composition: Composition) -> tuple[int, int, int, int]:
    """直前の上下声から最小移動となる、根音を外声に置いた主和音を返す。"""
    if composition.tonal_center is None or composition.mode is None:
        raise ValueError("tonic ending requires tonal_center and mode")
    lower_anchor, upper_anchor = _closing_voice_anchors(composition)
    tonic = composition.tonal_center
    soprano_root = _nearest_pitch_class(upper_anchor, tonic, minimum=45, maximum=84)
    bass_root = _nearest_pitch_class(
        lower_anchor,
        tonic,
        minimum=max(21, soprano_root - 24),
        maximum=soprano_root - 12,
    )
    third_below_root = 8 if composition.mode == "major" else 9
    return bass_root, soprano_root - third_below_root, soprano_root - 5, soprano_root


def render_composition(composition: Composition, output_path: Path) -> RenderResult:
    """1 tick を 1 ms として SMF を書き出し、再読込する。"""
    midi = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT, charset="utf-8")
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name=composition.title, time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=TEMPO, time=0))
    meta.append(mido.MetaMessage("end_of_track", time=composition.duration_ms))
    midi.tracks.append(meta)

    piano = mido.MidiTrack()
    piano.append(mido.MetaMessage("track_name", name="Piano", time=0))
    piano.append(mido.Message("program_change", channel=0, program=0, time=0))
    events: list[tuple[int, int, mido.Message]] = []
    offset = 0
    materials = composition.material_by_id
    for section in composition.form:
        material = materials[section.material_id]
        for note in material.notes:
            onset = offset + note.at_ms
            ending = onset + note.duration_ms
            events.append(
                (
                    onset,
                    2,
                    mido.Message("note_on", channel=0, note=note.pitch, velocity=note.velocity),
                )
            )
            events.append(
                (ending, 0, mido.Message("note_off", channel=0, note=note.pitch, velocity=0))
            )
        for pedal in material.pedals:
            events.append(
                (
                    offset + pedal.at_ms,
                    1,
                    mido.Message("control_change", channel=0, control=64, value=pedal.value),
                )
            )
        offset += material.duration_ms
    if composition.ending is not None:
        assert composition.tonal_center is not None
        assert composition.mode is not None
        events.append(
            (
                composition.body_duration_ms,
                1,
                mido.Message("control_change", channel=0, control=64, value=0),
            )
        )
        ending_onset = composition.body_duration_ms + ENDING_BREATH_MS
        for pitch in tonic_ending_pitches(composition):
            events.append(
                (
                    ending_onset,
                    2,
                    mido.Message("note_on", channel=0, note=pitch, velocity=ENDING_VELOCITY),
                )
            )
            events.append(
                (
                    composition.duration_ms,
                    0,
                    mido.Message("note_off", channel=0, note=pitch, velocity=0),
                )
            )
    events.append(
        (composition.duration_ms, 1, mido.Message("control_change", channel=0, control=64, value=0))
    )
    _append_absolute_events(piano, events)
    piano.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(piano)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output_path)
    reread = mido.MidiFile(output_path, charset="utf-8")
    if reread.ticks_per_beat != TICKS_PER_BEAT:
        raise ValueError("rendered SMF did not preserve ticks_per_beat")
    return RenderResult(output_path, composition.duration_ms, composition.note_count)
