"""Deterministically place phase-5 accompaniment intents on piano keys."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import cast

from llm_musical_composer.performance_pipeline import HARMONY_INTERVALS
from llm_musical_composer.piano_texture_register_placement import REGISTER_ZONES

from .score_ir import ScoreHarmony, ScoreNote

_DEGREE_INDEX = {"root": 0, "third": 1, "fifth": 2, "seventh": 3}
_PIANO_LOW = min(zone[0] for zone in REGISTER_ZONES.values())
_PIANO_HIGH = max(zone[1] for zone in REGISTER_ZONES.values())


class Phase5PitchPlacementError(ValueError):
    """A symbolic accompaniment could not be placed under deterministic constraints."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class Phase5PitchPlacementResult:
    notes: tuple[ScoreNote, ...]
    evaluated_candidate_count: int


@dataclass(frozen=True, slots=True)
class _Event:
    original_index: int
    at_units: int
    preferred_duration_units: int
    degree: str
    preferred_register_zone: str
    voice: str
    articulations: tuple[str, ...]


def place_accompaniment_events(
    material_placement_id: str,
    score_unit_id: str,
    response: Mapping[str, object],
    harmonies: tuple[ScoreHarmony, ...],
    *,
    existing_notes_by_score_unit: Mapping[str, tuple[ScoreNote, ...]],
    search_limit: int = 100_000,
) -> Phase5PitchPlacementResult:
    """Choose concrete pitches without modifying notes accepted by earlier phases."""
    if search_limit <= 0:
        raise ValueError("search_limit must be positive")
    events = tuple(
        sorted(
            (
                _Event(
                    original_index=index,
                    at_units=cast(int, value["at_units"]),
                    preferred_duration_units=cast(int, value["preferred_duration_units"]),
                    degree=cast(str, value["degree"]),
                    preferred_register_zone=cast(str, value["preferred_register_zone"]),
                    voice=cast(str, value["voice"]),
                    articulations=tuple(cast(list[str], value["articulations"])),
                )
                for index, value in enumerate(cast(list[Mapping[str, object]], response["events"]))
            ),
            key=lambda event: (event.at_units, event.original_index),
        )
    )
    existing = tuple(existing_notes_by_score_unit.get(score_unit_id, ()))
    evaluated = 0
    limit_reached = False

    def search(
        candidate_lists: tuple[tuple[int, ...], ...],
        index: int,
        placed: tuple[ScoreNote, ...],
    ) -> tuple[ScoreNote, ...] | None:
        nonlocal evaluated, limit_reached
        if index == len(events):
            return placed
        event = events[index]
        ordered_pitches = sorted(
            candidate_lists[index],
            key=lambda pitch: _candidate_rank(pitch, event, placed, existing),
        )
        for pitch in ordered_pitches:
            if evaluated >= search_limit:
                limit_reached = True
                return None
            evaluated += 1
            shortened = _shorten_own_rearticulation(placed, pitch, event.at_units)
            note = ScoreNote(
                score_note_id=(
                    f"score-note-{material_placement_id}-{event.original_index + 1:03d}"
                ),
                at_units=event.at_units,
                duration_units=event.preferred_duration_units,
                pitch=pitch,
                voice=event.voice,
                articulations=event.articulations,
            )
            extended = (*shortened, note)
            if not _constraints_hold(existing, extended):
                continue
            result = search(candidate_lists, index + 1, extended)
            if result is not None:
                return result
            if limit_reached:
                return None
        return None

    preferred_candidates = tuple(
        _candidate_pitches(event, harmonies, preferred_zone_only=True) for event in events
    )
    solution = (
        None
        if any(not candidates for candidates in preferred_candidates)
        else search(preferred_candidates, 0, ())
    )
    if solution is None and not limit_reached:
        all_candidates = tuple(
            _candidate_pitches(event, harmonies, preferred_zone_only=False) for event in events
        )
        solution = (
            None
            if any(not candidates for candidates in all_candidates)
            else search(all_candidates, 0, ())
        )
    if solution is None:
        code = "search_limit_reached" if limit_reached else "constraint_unplaceable"
        raise Phase5PitchPlacementError(code, "the accompaniment pitch search did not succeed")
    notes = tuple(sorted(solution, key=lambda note: note.score_note_id))
    return Phase5PitchPlacementResult(notes, evaluated)


def _candidate_pitches(
    event: _Event,
    harmonies: tuple[ScoreHarmony, ...],
    *,
    preferred_zone_only: bool,
) -> tuple[int, ...]:
    harmony = next(
        (
            value
            for value in harmonies
            if value.at_units <= event.at_units
            and event.at_units + event.preferred_duration_units
            <= value.at_units + value.duration_units
        ),
        None,
    )
    if harmony is None:
        return ()
    intervals = HARMONY_INTERVALS.get(harmony.quality, ())
    degree_index = _DEGREE_INDEX.get(event.degree)
    if degree_index is None or degree_index >= len(intervals):
        return ()
    pitch_class = (harmony.root_pitch_class + intervals[degree_index]) % 12
    if preferred_zone_only:
        low, high, _center = REGISTER_ZONES[event.preferred_register_zone]
    else:
        low, high = _PIANO_LOW, _PIANO_HIGH
    return tuple(pitch for pitch in range(low, high + 1) if pitch % 12 == pitch_class)


def _candidate_rank(
    pitch: int,
    event: _Event,
    placed: tuple[ScoreNote, ...],
    existing: tuple[ScoreNote, ...],
) -> tuple[int, int, int, int, int]:
    earlier_same_voice = [
        note
        for note in (*existing, *placed)
        if note.voice == event.voice and note.at_units <= event.at_units
    ]
    movement = (
        abs(pitch - max(earlier_same_voice, key=lambda note: note.at_units).pitch)
        if earlier_same_voice
        else 0
    )
    low, high, center = REGISTER_ZONES[event.preferred_register_zone]
    outside_preferred_zone = int(not low <= pitch <= high)
    low_degree_penalty = int(pitch < 48 and event.degree not in {"root", "fifth"})
    return outside_preferred_zone, movement, abs(pitch - center), low_degree_penalty, pitch


def _shorten_own_rearticulation(
    placed: tuple[ScoreNote, ...], pitch: int, at_units: int
) -> tuple[ScoreNote, ...]:
    updated = list(placed)
    for index, note in enumerate(updated):
        if note.pitch == pitch and note.at_units < at_units < note.at_units + note.duration_units:
            updated[index] = replace(note, duration_units=at_units - note.at_units)
    return tuple(updated)


def _constraints_hold(existing: tuple[ScoreNote, ...], candidates: tuple[ScoreNote, ...]) -> bool:
    combined = (*existing, *candidates)
    existing_count = len(existing)
    for left_index, left in enumerate(combined):
        for right_index in range(left_index + 1, len(combined)):
            if left_index < existing_count and right_index < existing_count:
                continue
            right = combined[right_index]
            if not _overlap(left, right):
                continue
            if left.pitch == right.pitch:
                return False
            low, high = sorted((left.pitch, right.pitch))
            if low < 48 and high - low < 7:
                return False
    return True


def _overlap(left: ScoreNote, right: ScoreNote) -> bool:
    return (
        left.at_units < right.at_units + right.duration_units
        and right.at_units < left.at_units + left.duration_units
    )
