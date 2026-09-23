import pytest

from scoim.score_group_pitch_placement import (
    ScoreGroupPitchPlacementError,
    place_score_group_accompaniment_events,
)
from scoim.score_ir import ScoreHarmony, ScoreNote


def _event(at=0, duration=12, degree="root", zone="low"):
    return {
        "at_units": at,
        "preferred_duration_units": duration,
        "degree": degree,
        "preferred_register_zone": zone,
        "voice": "lower",
        "articulations": ["normal"],
    }


def _place(requests, existing=(), *, limit=100_000):
    return place_score_group_accompaniment_events(
        requests,
        {"unit": (ScoreHarmony("h", 0, 48, 0, "major"),)},
        dict.fromkeys(requests, "unit"),
        existing_notes_by_score_unit={"unit": existing},
        search_limit=limit,
    )


def test_joint_search_backtracks_into_an_earlier_layer_without_changing_events():
    blockers = tuple(ScoreNote(f"fixed-{p}", 0, 20, p, "upper") for p in (24, 36, 48, 84, 96, 108))
    blockers += (ScoreNote("fixed-late", 10, 10, 60, "upper"),)
    requests = {
        "a": {"events": [_event(duration=10, zone="high")]},
        "b": {"events": [_event(duration=20, zone="high")]},
    }
    result = _place(requests, blockers)
    assert result.notes_by_placement["a"][0].pitch == 60
    assert result.notes_by_placement["b"][0].pitch == 72
    assert result.notes_by_placement["a"][0].duration_units == 10
    assert result.notes_by_placement["b"][0].duration_units == 20
    assert result == _place(dict(reversed(tuple(requests.items()))), blockers)
    assert result.evaluated_candidate_count_by_score_unit["unit"] > 2


def test_rearticulation_only_shortens_notes_in_its_own_placement():
    own = _place({"a": {"events": [_event(duration=12), _event(at=6, duration=6)]}})
    assert [n.duration_units for n in own.notes_by_placement["a"]] == [6, 6]
    other = _place(
        {
            "a": {"events": [_event(duration=12)]},
            "b": {"events": [_event(at=6, duration=6)]},
        }
    )
    assert other.notes_by_placement["a"][0].duration_units == 12
    assert other.notes_by_placement["a"][0].pitch != other.notes_by_placement["b"][0].pitch


def test_joint_search_preserves_degree_low_spacing_and_only_then_widens_register():
    result = _place(
        {
            "a": {"events": [_event(degree="root", zone="bass")]},
            "b": {"events": [_event(degree="third", zone="bass")]},
        }
    )
    a, b = (result.notes_by_placement[key][0] for key in ("a", "b"))
    assert a.pitch % 12 == 0 and b.pitch % 12 == 4
    assert abs(a.pitch - b.pitch) >= 7
    blocked = tuple(ScoreNote(f"p-{p}", 0, 12, p, "upper") for p in (36, 48))
    wide = _place({"a": {"events": [_event()]}}, blocked)
    assert wide.notes_by_placement["a"][0].pitch % 12 == 0
    assert not 36 <= wide.notes_by_placement["a"][0].pitch <= 59


@pytest.mark.parametrize(
    "limit,code", [(1, "search_limit_reached"), (100_000, "constraint_unplaceable")]
)
def test_joint_search_distinguishes_budget_exhaustion_from_proven_failure(limit, code):
    blocked = tuple(ScoreNote(f"p-{p}", 0, 12, p, "upper") for p in range(24, 109, 12))
    with pytest.raises(ScoreGroupPitchPlacementError) as caught:
        _place({"a": {"events": [_event()]}}, blocked, limit=limit)
    assert caught.value.code == code
    assert caught.value.score_unit_id == "unit"


def test_separate_units_do_not_share_occupancy_or_search_budget():
    result = place_score_group_accompaniment_events(
        {"a": {"events": [_event()]}, "b": {"events": [_event()]}},
        {unit: (ScoreHarmony(f"h-{unit}", 0, 48, 0, "major"),) for unit in ("u1", "u2")},
        {"a": "u1", "b": "u2"},
        existing_notes_by_score_unit={},
        search_limit=1,
    )
    assert result.notes_by_placement["a"][0].pitch == result.notes_by_placement["b"][0].pitch
    assert result.evaluated_candidate_count_by_score_unit == {"u1": 1, "u2": 1}
