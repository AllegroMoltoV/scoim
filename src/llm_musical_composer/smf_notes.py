"""SMF のイベント収集、ノート対応、実時間変換を共通化する。"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mido


@dataclass(frozen=True)
class SmfNote:
    pitch: int
    onset_ms: int
    duration_ms: int
    velocity: int


def _issue(code: str, message: str) -> dict[str, Any]:
    return {"code": code, "message": message}


def collect_events(midi: mido.MidiFile) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    track_reports: list[dict[str, Any]] = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        names: list[str] = []
        end_of_track_indices: list[int] = []
        for event_index, message in enumerate(track):
            tick += message.time
            events.append(
                {
                    "tick": tick,
                    "track": track_index,
                    "event_index": event_index,
                    "message": message,
                }
            )
            if message.type == "track_name":
                names.append(message.name)
            if message.type == "end_of_track":
                end_of_track_indices.append(event_index)
        issues: list[dict[str, Any]] = []
        if not end_of_track_indices:
            issues.append(_issue("missing_end_of_track", "Track has no end_of_track event."))
        if len(end_of_track_indices) > 1:
            issues.append(
                _issue(
                    "multiple_end_of_track",
                    f"Track has {len(end_of_track_indices)} end_of_track events.",
                )
            )
        if end_of_track_indices and end_of_track_indices[0] != len(track) - 1:
            issues.append(_issue("data_after_end_of_track", "Events follow end_of_track."))
        track_reports.append(
            {
                "track": track_index,
                "event_count": len(track),
                "last_tick": tick,
                "names": names,
                "issues": issues,
            }
        )
    events.sort(key=lambda event: (event["tick"], event["track"], event["event_index"]))
    return events, track_reports


def match_notes(
    events: list[dict[str, Any]], final_tick: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    active_global: dict[tuple[int, int], deque[int]] = defaultdict(deque)
    active_local: dict[tuple[int, int, int], deque[int]] = defaultdict(deque)
    notes: list[dict[str, Any]] = []
    overlapping = 0
    unmatched_global = 0
    unmatched_local = 0

    for event in events:
        message = event["message"]
        is_note_on = message.type == "note_on" and message.velocity > 0
        is_note_off = message.type == "note_off" or (
            message.type == "note_on" and message.velocity == 0
        )
        if not (is_note_on or is_note_off):
            continue
        global_key = (message.channel, message.note)
        local_key = (event["track"], message.channel, message.note)
        if is_note_on:
            if active_global[global_key]:
                overlapping += 1
            note_index = len(notes)
            notes.append(
                {
                    "track": event["track"],
                    "channel": message.channel,
                    "pitch": message.note,
                    "velocity": message.velocity,
                    "onset_tick": event["tick"],
                    "offset_tick": None,
                    "duration_ticks": None,
                }
            )
            active_global[global_key].append(note_index)
            active_local[local_key].append(note_index)
            continue
        if active_local[local_key]:
            active_local[local_key].popleft()
        else:
            unmatched_local += 1
        if active_global[global_key]:
            note_index = active_global[global_key].popleft()
            notes[note_index]["offset_tick"] = event["tick"]
            notes[note_index]["duration_ticks"] = event["tick"] - notes[note_index]["onset_tick"]
        else:
            unmatched_global += 1

    dangling_global = sum(len(indices) for indices in active_global.values())
    dangling_local = sum(len(indices) for indices in active_local.values())
    for indices in active_global.values():
        for note_index in indices:
            notes[note_index]["effective_end_tick"] = final_tick + 1
    for note in notes:
        note.setdefault("effective_end_tick", note["offset_tick"])
    return notes, {
        "overlapping_note_on": overlapping,
        "unmatched_note_off": unmatched_global,
        "track_local_unmatched_note_off": unmatched_local,
        "dangling_note_on": dangling_global,
        "track_local_dangling_note_on": dangling_local,
    }


def maximum_polyphony(notes: Iterable[dict[str, Any]], final_tick: int) -> int:
    changes: list[tuple[int, int]] = []
    for note in notes:
        end_tick = note["effective_end_tick"]
        if end_tick is None:
            end_tick = final_tick
        changes.append((note["onset_tick"], 1))
        changes.append((end_tick, -1))
    current = 0
    maximum = 0
    for _, delta in sorted(changes, key=lambda item: (item[0], item[1])):
        current += delta
        maximum = max(maximum, current)
    return maximum


def _tick_to_seconds_map(midi: mido.MidiFile, ticks: set[int]) -> dict[int, float]:
    wanted = sorted(ticks)
    tempo_events: list[tuple[int, int]] = []
    events, _ = collect_events(midi)
    for event in events:
        if event["message"].type == "set_tempo":
            tempo_events.append((event["tick"], event["message"].tempo))
    result: dict[int, float] = {}
    tempo = 500_000
    previous_tick = 0
    seconds = 0.0
    event_index = 0
    for tick in wanted:
        while event_index < len(tempo_events) and tempo_events[event_index][0] <= tick:
            event_tick, new_tempo = tempo_events[event_index]
            seconds += mido.tick2second(event_tick - previous_tick, midi.ticks_per_beat, tempo)
            previous_tick = event_tick
            tempo = new_tempo
            event_index += 1
        result[tick] = seconds + mido.tick2second(tick - previous_tick, midi.ticks_per_beat, tempo)
    return result


def load_smf_notes(path: Path) -> list[SmfNote]:
    """SMF から完結した非打楽器ノートをミリ秒単位で読む。"""
    midi = mido.MidiFile(path)
    if midi.type == 2:
        raise ValueError("SMF format 2 has no single global timeline")
    events, reports = collect_events(midi)
    final_tick = max((report["last_tick"] for report in reports), default=0)
    notes, _ = match_notes(events, final_tick)
    usable = [
        note
        for note in notes
        if note["channel"] != 9
        and note["offset_tick"] is not None
        and note["offset_tick"] > note["onset_tick"]
    ]
    ticks = {int(note[key]) for note in usable for key in ("onset_tick", "offset_tick")}
    seconds = _tick_to_seconds_map(midi, ticks)
    return [
        SmfNote(
            pitch=note["pitch"],
            onset_ms=round(seconds[note["onset_tick"]] * 1000),
            duration_ms=max(
                1,
                round((seconds[note["offset_tick"]] - seconds[note["onset_tick"]]) * 1000),
            ),
            velocity=note["velocity"],
        )
        for note in usable
    ]
