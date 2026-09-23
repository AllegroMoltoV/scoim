"""Place all mutable accompaniment layers of a score group together."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import cast

from .phase5_pitch_placement import (
    Phase5PitchPlacementError,
    _candidate_pitches,
    _candidate_rank,
    _constraints_hold,
    _Event,
)
from .score_ir import ScoreHarmony, ScoreNote

SCORE_GROUP_SEARCH_LIMIT = 100_000


class ScoreGroupPitchPlacementError(Phase5PitchPlacementError):
    """A joint search failed for one identified score unit."""

    def __init__(self, code: str, score_unit_id: str) -> None:
        super().__init__(code, f"joint accompaniment search failed for {score_unit_id}")
        self.score_unit_id = score_unit_id


@dataclass(frozen=True, slots=True)
class ScoreGroupPitchPlacementResult:
    notes_by_placement: dict[str, tuple[ScoreNote, ...]]
    evaluated_candidate_count_by_score_unit: dict[str, int]


@dataclass(frozen=True, slots=True)
class _GroupEvent:
    placement_id: str
    event: _Event


def place_score_group_accompaniment_events(
    requests_by_placement: Mapping[str, Mapping[str, object]],
    harmonies_by_score_unit: Mapping[str, tuple[ScoreHarmony, ...]],
    score_unit_id_by_placement: Mapping[str, str],
    *,
    existing_notes_by_score_unit: Mapping[str, tuple[ScoreNote, ...]],
    search_limit: int = SCORE_GROUP_SEARCH_LIMIT,
) -> ScoreGroupPitchPlacementResult:
    """Backtrack across mutable placements, retaining fixed foregrounds and degree choices."""
    if search_limit <= 0:
        raise ValueError("search_limit must be positive")
    by_unit: dict[str, list[_GroupEvent]] = {}
    for placement_id in sorted(requests_by_placement):
        unit_id = score_unit_id_by_placement[placement_id]
        raw_events = cast(list[Mapping[str, object]], requests_by_placement[placement_id]["events"])
        for index, value in enumerate(raw_events):
            by_unit.setdefault(unit_id, []).append(
                _GroupEvent(
                    placement_id,
                    _Event(
                        original_index=index,
                        at_units=cast(int, value["at_units"]),
                        preferred_duration_units=cast(int, value["preferred_duration_units"]),
                        degree=cast(str, value["degree"]),
                        preferred_register_zone=cast(str, value["preferred_register_zone"]),
                        voice=cast(str, value["voice"]),
                        articulations=tuple(cast(list[str], value["articulations"])),
                    ),
                )
            )
    notes: dict[str, list[ScoreNote]] = {placement_id: [] for placement_id in requests_by_placement}
    counts: dict[str, int] = {}
    for unit_id in sorted(by_unit):
        events = tuple(
            sorted(
                by_unit[unit_id],
                key=lambda item: (
                    item.event.at_units,
                    item.placement_id,
                    item.event.original_index,
                ),
            )
        )
        placed, count = _place_unit(
            unit_id,
            events,
            harmonies_by_score_unit[unit_id],
            tuple(existing_notes_by_score_unit.get(unit_id, ())),
            search_limit,
        )
        counts[unit_id] = count
        for placement_id, note in placed:
            notes[placement_id].append(note)
    return ScoreGroupPitchPlacementResult(
        {
            placement_id: tuple(
                sorted(values, key=lambda note: int(note.score_note_id.rsplit("-", 1)[1]))
            )
            for placement_id, values in notes.items()
        },
        counts,
    )


def _place_unit(
    unit_id: str,
    events: tuple[_GroupEvent, ...],
    harmonies: tuple[ScoreHarmony, ...],
    existing: tuple[ScoreNote, ...],
    search_limit: int,
) -> tuple[tuple[tuple[str, ScoreNote], ...], int]:
    evaluated = 0
    limit_reached = False

    def search(candidates: tuple[tuple[int, ...], ...]) -> tuple[tuple[str, ScoreNote], ...] | None:
        nonlocal evaluated, limit_reached
        if not events:
            return ()

        def ranked(index: int, placed: tuple[tuple[str, ScoreNote], ...]) -> Iterator[int]:
            return iter(
                sorted(
                    candidates[index],
                    key=lambda pitch: _candidate_rank(
                        pitch, events[index].event, tuple(note for _, note in placed), existing
                    ),
                )
            )

        # Explicit frames avoid making the musical group size depend on Python's recursion limit.
        stack: list[tuple[int, tuple[tuple[str, ScoreNote], ...], Iterator[int]]] = [
            (0, (), ranked(0, ()))
        ]
        while stack:
            index, placed, choices = stack[-1]
            pitch = next(choices, None)
            if pitch is None:
                stack.pop()
                continue
            if evaluated >= search_limit:
                limit_reached = True
                return None
            evaluated += 1
            item = events[index]
            event = item.event
            shortened = tuple(
                (
                    placement_id,
                    replace(note, duration_units=event.at_units - note.at_units)
                    if placement_id == item.placement_id
                    and note.pitch == pitch
                    and note.at_units < event.at_units < note.at_units + note.duration_units
                    else note,
                )
                for placement_id, note in placed
            )
            note = ScoreNote(
                score_note_id=f"score-note-{item.placement_id}-{event.original_index + 1:03d}",
                at_units=event.at_units,
                duration_units=event.preferred_duration_units,
                pitch=pitch,
                voice=event.voice,
                articulations=event.articulations,
            )
            extended = (*shortened, (item.placement_id, note))
            if not _constraints_hold(existing, tuple(value for _, value in extended)):
                continue
            if index + 1 == len(events):
                return extended
            stack.append((index + 1, extended, ranked(index + 1, extended)))
        return None

    preferred = tuple(
        _candidate_pitches(item.event, harmonies, preferred_zone_only=True) for item in events
    )
    solution = None if any(not values for values in preferred) else search(preferred)
    if solution is None and not limit_reached:
        all_pitches = tuple(
            _candidate_pitches(item.event, harmonies, preferred_zone_only=False) for item in events
        )
        solution = None if any(not values for values in all_pitches) else search(all_pitches)
    if solution is None:
        code = "search_limit_reached" if limit_reached else "constraint_unplaceable"
        raise ScoreGroupPitchPlacementError(code, unit_id)
    return solution, evaluated
