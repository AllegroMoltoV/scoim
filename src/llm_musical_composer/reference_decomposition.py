"""参照 SMF の生イベントと演奏向け派生情報を分離する。"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import mido


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ObservedSmfEvent:
    event_id: str
    track: int
    event_index: int
    absolute_tick: int
    is_meta: bool
    message_type: str
    fields: tuple[tuple[str, Any], ...]

    def field(self, name: str, default: Any = None) -> Any:
        return dict(self.fields).get(name, default)


@dataclass(frozen=True)
class ObservedSmf:
    midi_type: int
    ticks_per_beat: int
    track_count: int
    events: tuple[ObservedSmfEvent, ...]
    source_sha256: str
    ledger_sha256: str


@dataclass(frozen=True)
class ObservedSmfDifference:
    event_id: str
    changed_fields: tuple[str, ...]


@dataclass(frozen=True)
class ObservedEventTime:
    event_id: str
    at_us: int


@dataclass(frozen=True)
class ObservedNote:
    note_on_event_id: str
    note_off_event_id: str
    track: int
    channel: int
    pitch: int
    velocity: int
    note_off_velocity: int
    onset_tick: int
    offset_tick: int
    onset_us: int
    offset_us: int
    matching_status: str


@dataclass(frozen=True)
class ObservedAttackGroup:
    onset_us: int
    note_on_event_ids: tuple[str, ...]


@dataclass(frozen=True)
class ObservedPerformance:
    event_times: tuple[ObservedEventTime, ...]
    notes: tuple[ObservedNote, ...]
    attack_groups: tuple[ObservedAttackGroup, ...]
    attack_group_window_us: int
    note_matching_status: str
    ambiguous_note_count: int
    unmatched_note_off_count: int
    dangling_note_on_count: int


def _ledger_payload(
    *,
    midi_type: int,
    ticks_per_beat: int,
    track_count: int,
    events: tuple[ObservedSmfEvent, ...],
) -> dict[str, Any]:
    return {
        "midi_type": midi_type,
        "ticks_per_beat": ticks_per_beat,
        "track_count": track_count,
        "events": [
            {
                "event_id": event.event_id,
                "track": event.track,
                "event_index": event.event_index,
                "absolute_tick": event.absolute_tick,
                "is_meta": event.is_meta,
                "message_type": event.message_type,
                "fields": {key: _thaw(value) for key, value in event.fields},
            }
            for event in events
        ],
    }


def _ledger_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_observed_smf(path: Path) -> ObservedSmf:
    """SMF を解釈前の決定的な event ledger へ変換する。"""
    path = Path(path)
    midi = mido.MidiFile(path)
    events: list[ObservedSmfEvent] = []
    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        for event_index, message in enumerate(track):
            absolute_tick += int(message.time)
            message_dict = message.dict()
            message_dict.pop("time", None)
            message_dict.pop("type", None)
            events.append(
                ObservedSmfEvent(
                    event_id=f"track-{track_index}-event-{event_index}",
                    track=track_index,
                    event_index=event_index,
                    absolute_tick=absolute_tick,
                    is_meta=message.is_meta,
                    message_type=message.type,
                    fields=tuple(
                        (key, _freeze(value)) for key, value in sorted(message_dict.items())
                    ),
                )
            )
    frozen_events = tuple(events)
    payload = _ledger_payload(
        midi_type=midi.type,
        ticks_per_beat=midi.ticks_per_beat,
        track_count=len(midi.tracks),
        events=frozen_events,
    )
    return ObservedSmf(
        midi_type=midi.type,
        ticks_per_beat=midi.ticks_per_beat,
        track_count=len(midi.tracks),
        events=frozen_events,
        source_sha256=_sha256_file(path),
        ledger_sha256=_ledger_sha256(payload),
    )


def write_observed_smf(observed: ObservedSmf, output_path: Path) -> None:
    """ObservedSmf の保持イベントから診断用 SMF を出力する。"""
    if observed.track_count <= 0:
        raise ValueError("ObservedSmf must contain at least one track")
    midi = mido.MidiFile(type=observed.midi_type, ticks_per_beat=observed.ticks_per_beat)
    midi.tracks.extend(mido.MidiTrack() for _ in range(observed.track_count))
    grouped: dict[int, list[ObservedSmfEvent]] = defaultdict(list)
    for event in observed.events:
        if not 0 <= event.track < observed.track_count:
            raise ValueError(f"event track is outside track_count: {event.event_id}")
        grouped[event.track].append(event)
    for track_index in range(observed.track_count):
        previous_tick = 0
        ordered = sorted(grouped[track_index], key=lambda item: item.event_index)
        for expected_index, event in enumerate(ordered):
            if event.event_index != expected_index:
                raise ValueError(f"event indices must be consecutive: track {track_index}")
            if event.absolute_tick < previous_tick:
                raise ValueError(f"event ticks must be non-decreasing: {event.event_id}")
            values = {key: _thaw(value) for key, value in event.fields}
            values["type"] = event.message_type
            values["time"] = event.absolute_tick - previous_tick
            message = (
                mido.MetaMessage.from_dict(values)
                if event.is_meta
                else mido.Message.from_dict(values)
            )
            midi.tracks[track_index].append(message)
            previous_tick = event.absolute_tick
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output_path)


def observed_smf_differences(
    first: ObservedSmf, second: ObservedSmf
) -> tuple[ObservedSmfDifference, ...]:
    """二つの event ledger の意味差を返す。"""
    differences: list[ObservedSmfDifference] = []
    header_changes = tuple(
        name
        for name in ("midi_type", "ticks_per_beat", "track_count")
        if getattr(first, name) != getattr(second, name)
    )
    if header_changes:
        differences.append(ObservedSmfDifference("__header__", header_changes))
    maximum = max(len(first.events), len(second.events))
    for index in range(maximum):
        if index >= len(first.events):
            differences.append(
                ObservedSmfDifference(second.events[index].event_id, ("unexpected_event",))
            )
            continue
        if index >= len(second.events):
            differences.append(
                ObservedSmfDifference(first.events[index].event_id, ("missing_event",))
            )
            continue
        left = first.events[index]
        right = second.events[index]
        changed = tuple(
            name
            for name in (
                "event_id",
                "track",
                "event_index",
                "absolute_tick",
                "is_meta",
                "message_type",
                "fields",
            )
            if getattr(left, name) != getattr(right, name)
        )
        if changed:
            differences.append(ObservedSmfDifference(left.event_id, changed))
    return tuple(differences)


def _rounded_fraction(value: Fraction) -> int:
    numerator = value.numerator
    denominator = value.denominator
    return (2 * numerator + denominator) // (2 * denominator)


def _event_times(observed: ObservedSmf) -> dict[str, int]:
    by_tick: dict[int, list[ObservedSmfEvent]] = defaultdict(list)
    for event in observed.events:
        by_tick[event.absolute_tick].append(event)
    tempo = 500_000
    previous_tick = 0
    elapsed = Fraction(0)
    result: dict[str, int] = {}
    for tick in sorted(by_tick):
        elapsed += Fraction((tick - previous_tick) * tempo, observed.ticks_per_beat)
        at_us = _rounded_fraction(elapsed)
        ordered = sorted(by_tick[tick], key=lambda item: (item.track, item.event_index))
        for event in ordered:
            result[event.event_id] = at_us
        for event in ordered:
            if event.message_type == "set_tempo":
                tempo = int(event.field("tempo"))
        previous_tick = tick
    return result


def _attack_groups(
    attacks: list[tuple[int, str]], window_us: int
) -> tuple[ObservedAttackGroup, ...]:
    groups: list[ObservedAttackGroup] = []
    index = 0
    ordered = sorted(attacks, key=lambda item: (item[0], item[1]))
    while index < len(ordered):
        onset = ordered[index][0]
        ids: list[str] = []
        while index < len(ordered) and ordered[index][0] - onset <= window_us:
            ids.append(ordered[index][1])
            index += 1
        groups.append(ObservedAttackGroup(onset, tuple(ids)))
    return tuple(groups)


def build_observed_performance(
    observed: ObservedSmf, *, attack_group_window_us: int = 30_000
) -> ObservedPerformance:
    """ObservedSmf から時間付きnote対応候補と発音群ビューを作る。"""
    if attack_group_window_us < 0:
        raise ValueError("attack_group_window_us must be non-negative")
    times = _event_times(observed)
    ordered = sorted(
        observed.events,
        key=lambda item: (item.absolute_tick, item.track, item.event_index),
    )
    active: dict[tuple[int, int, int], deque[ObservedSmfEvent]] = defaultdict(deque)
    ambiguous_ids: set[str] = set()
    note_pairs: list[tuple[ObservedSmfEvent, ObservedSmfEvent]] = []
    unmatched_note_off_count = 0
    attacks: list[tuple[int, str]] = []
    for event in ordered:
        channel = event.field("channel")
        pitch = event.field("note")
        is_note_on = event.message_type == "note_on" and int(event.field("velocity", 0)) > 0
        is_note_off = event.message_type == "note_off" or (
            event.message_type == "note_on" and int(event.field("velocity", 0)) == 0
        )
        if not (is_note_on or is_note_off):
            continue
        key = (event.track, int(channel), int(pitch))
        if is_note_on:
            if active[key]:
                ambiguous_ids.update(item.event_id for item in active[key])
                ambiguous_ids.add(event.event_id)
            active[key].append(event)
            attacks.append((times[event.event_id], event.event_id))
            continue
        if not active[key]:
            unmatched_note_off_count += 1
            continue
        note_on = active[key].popleft()
        note_pairs.append((note_on, event))
    dangling_note_on_count = sum(len(items) for items in active.values())
    notes = tuple(
        ObservedNote(
            note_on_event_id=note_on.event_id,
            note_off_event_id=note_off.event_id,
            track=note_on.track,
            channel=int(note_on.field("channel")),
            pitch=int(note_on.field("note")),
            velocity=int(note_on.field("velocity")),
            note_off_velocity=int(note_off.field("velocity", 0)),
            onset_tick=note_on.absolute_tick,
            offset_tick=note_off.absolute_tick,
            onset_us=times[note_on.event_id],
            offset_us=times[note_off.event_id],
            matching_status=("ambiguous" if note_on.event_id in ambiguous_ids else "assessed"),
        )
        for note_on, note_off in note_pairs
    )
    if ambiguous_ids:
        matching_status = "ambiguous"
    elif unmatched_note_off_count or dangling_note_on_count:
        matching_status = "unable_to_investigate"
    else:
        matching_status = "assessed"
    return ObservedPerformance(
        event_times=tuple(
            ObservedEventTime(event.event_id, times[event.event_id]) for event in ordered
        ),
        notes=notes,
        attack_groups=_attack_groups(attacks, attack_group_window_us),
        attack_group_window_us=attack_group_window_us,
        note_matching_status=matching_status,
        ambiguous_note_count=len(ambiguous_ids),
        unmatched_note_off_count=unmatched_note_off_count,
        dangling_note_on_count=dangling_note_on_count,
    )
