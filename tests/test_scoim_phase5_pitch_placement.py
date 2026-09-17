import pytest

from scoim.phase5_pitch_placement import (
    Phase5PitchPlacementError,
    place_accompaniment_events,
)
from scoim.score_ir import ScoreHarmony, ScoreNote


def _event(
    *,
    at_units: int = 0,
    duration_units: int = 12,
    degree: str = "root",
    zone: str = "low",
    voice: str = "lower",
) -> dict[str, object]:
    return {
        "at_units": at_units,
        "preferred_duration_units": duration_units,
        "degree": degree,
        "preferred_register_zone": zone,
        "voice": voice,
        "articulations": ["normal"],
    }


def test_phase5_places_the_requested_degree_in_the_requested_zone() -> None:
    result = place_accompaniment_events(
        "support-first",
        "score-unit-statement",
        {"events": [_event(degree="third")]},
        (ScoreHarmony("h1", 0, 48, 0, "major"),),
        existing_notes_by_score_unit={},
    )

    assert len(result.notes) == 1
    assert result.notes[0].pitch % 12 == 4
    assert 36 <= result.notes[0].pitch <= 59


def test_phase5_can_shorten_its_own_note_for_a_same_key_rearticulation() -> None:
    result = place_accompaniment_events(
        "support-first",
        "score-unit-statement",
        {
            "events": [
                _event(at_units=0, duration_units=12),
                _event(at_units=6, duration_units=6),
            ]
        },
        (ScoreHarmony("h1", 0, 48, 0, "major"),),
        existing_notes_by_score_unit={},
    )

    assert [note.duration_units for note in result.notes] == [6, 6]
    assert result.notes[0].pitch == result.notes[1].pitch


def test_phase5_does_not_treat_another_score_units_local_time_as_a_collision() -> None:
    result = place_accompaniment_events(
        "support-first",
        "score-unit-statement",
        {"events": [_event()]},
        (ScoreHarmony("h1", 0, 48, 0, "major"),),
        existing_notes_by_score_unit={
            "score-unit-return": (ScoreNote("other", 0, 12, 48, "lower"),)
        },
    )

    assert result.notes[0].pitch == 48


def test_phase5_can_leave_the_preferred_zone_when_that_is_required() -> None:
    blockers = tuple(ScoreNote(f"n-{pitch}", 0, 12, pitch, "lower") for pitch in (36, 48))

    result = place_accompaniment_events(
        "support-first",
        "score-unit-statement",
        {"events": [_event()]},
        (ScoreHarmony("h1", 0, 48, 0, "major"),),
        existing_notes_by_score_unit={"score-unit-statement": blockers},
    )

    assert result.notes[0].pitch % 12 == 0
    assert not 36 <= result.notes[0].pitch <= 59


def test_phase5_reports_a_typed_result_when_the_degree_is_blocked_everywhere() -> None:
    blockers = tuple(ScoreNote(f"n-{pitch}", 0, 12, pitch, "lower") for pitch in range(24, 109, 12))

    with pytest.raises(Phase5PitchPlacementError) as caught:
        place_accompaniment_events(
            "support-first",
            "score-unit-statement",
            {"events": [_event()]},
            (ScoreHarmony("h1", 0, 48, 0, "major"),),
            existing_notes_by_score_unit={"score-unit-statement": blockers},
        )

    assert caught.value.code == "constraint_unplaceable"


def test_phase5_prefers_the_requested_zone_over_a_nearer_previous_pitch() -> None:
    previous = (ScoreNote("previous", 0, 6, 72, "lower"),)

    result = place_accompaniment_events(
        "support-first",
        "score-unit-statement",
        {"events": [_event(at_units=12)]},
        (ScoreHarmony("h1", 0, 48, 0, "major"),),
        existing_notes_by_score_unit={"score-unit-statement": previous},
    )

    assert result.notes[0].pitch == 48


def test_phase5_avoids_a_muddy_low_interval() -> None:
    result = place_accompaniment_events(
        "support-first",
        "score-unit-statement",
        {
            "events": [
                _event(degree="root", zone="bass"),
                _event(degree="third", zone="bass", voice="upper"),
            ]
        },
        (ScoreHarmony("h1", 0, 48, 0, "major"),),
        existing_notes_by_score_unit={},
    )

    low, high = sorted(note.pitch for note in result.notes)
    assert low < 48
    assert high - low >= 7


def test_phase5_reports_when_the_search_trial_limit_is_reached() -> None:
    blockers = (ScoreNote("block", 0, 12, 48, "upper"),)

    with pytest.raises(Phase5PitchPlacementError) as caught:
        place_accompaniment_events(
            "support-first",
            "score-unit-statement",
            {"events": [_event()]},
            (ScoreHarmony("h1", 0, 48, 0, "major"),),
            existing_notes_by_score_unit={"score-unit-statement": blockers},
            search_limit=1,
        )

    assert caught.value.code == "search_limit_reached"


def test_phase5_pitch_placement_is_deterministic() -> None:
    arguments = (
        "support-first",
        "score-unit-statement",
        {"events": [_event(), _event(at_units=12, degree="fifth")]},
        (ScoreHarmony("h1", 0, 48, 0, "major"),),
    )

    first = place_accompaniment_events(*arguments, existing_notes_by_score_unit={})
    second = place_accompaniment_events(*arguments, existing_notes_by_score_unit={})

    assert first == second
