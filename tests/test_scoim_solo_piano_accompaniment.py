from types import SimpleNamespace

import pytest

import scoim.solo_piano_accompaniment as accompaniment_module
from llm_musical_composer.performance_pipeline import ScoreNote
from scoim.realization_workspace import HarmonyChord, MelodyNote, MelodyValue
from scoim.solo_piano_accompaniment import (
    AccompanimentRequestEvent,
    place_workspace_accompaniment,
)


def test_places_workspace_accompaniment_from_coarse_preferences() -> None:
    result = place_workspace_accompaniment(
        material_key="theme",
        tonal_center=0,
        mode="major",
        harmonies=(HarmonyChord(8, 0, "major"),),
        melody=MelodyValue("upper", (MelodyNote(0, 8, 72),)),
        requests=(
            AccompanimentRequestEvent(0, 8, "root", "bass", ()),
            AccompanimentRequestEvent(0, 8, "fifth", "low", ()),
        ),
    )

    assert result.status == "placed"
    assert result.value is not None
    assert tuple(note.pitch for note in result.value.notes) == (36, 43)
    assert tuple(note.preferred_register_zone for note in result.value.notes) == (
        "bass",
        "low",
    )
    assert tuple(note.preferred_duration_units for note in result.value.notes) == (8, 8)
    assert tuple(note.duration_units for note in result.value.notes) == (8, 8)


def test_keeps_preferred_duration_when_rearticulation_shortens_realized_note() -> None:
    result = place_workspace_accompaniment(
        material_key="theme",
        tonal_center=2,
        mode="major",
        harmonies=(HarmonyChord(32, 2, "major"),),
        melody=MelodyValue(
            "upper",
            (
                MelodyNote(0, 8, 74),
                MelodyNote(8, 8, 76),
                MelodyNote(16, 8, 72),
                MelodyNote(24, 8, 74),
            ),
        ),
        requests=(
            AccompanimentRequestEvent(0, 24, "root", "bass", ()),
            AccompanimentRequestEvent(8, 16, "root", "bass", ()),
            AccompanimentRequestEvent(16, 2, "root", "bass", ()),
            AccompanimentRequestEvent(20, 4, "root", "bass", ()),
        ),
    )

    assert result.status == "placed"
    assert result.value is not None
    assert tuple(note.preferred_duration_units for note in result.value.notes) == (24, 16, 2, 4)
    assert tuple(note.duration_units for note in result.value.notes) == (24, 8, 2, 4)


def test_rejects_a_placer_result_that_changes_the_requested_onset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        accompaniment_module,
        "place_piano_texture_v8",
        lambda *args, **kwargs: SimpleNamespace(
            status="placed",
            notes=(ScoreNote("workspace-a-000", 1, 3, 36, "lower"),),
            reason=None,
            total_candidate_evaluation_count=1,
        ),
    )

    with pytest.raises(ValueError, match="onset"):
        place_workspace_accompaniment(
            material_key="theme",
            tonal_center=0,
            mode="major",
            harmonies=(HarmonyChord(4, 0, "major"),),
            melody=MelodyValue("upper", (MelodyNote(0, 4, 72),)),
            requests=(AccompanimentRequestEvent(0, 4, "root", "bass", ()),),
        )


@pytest.mark.parametrize(
    ("notes", "message"),
    [
        ((), "event set"),
        ((ScoreNote("workspace-a-000", 0, 4, 36, "upper"),), "voice"),
        (
            (ScoreNote("workspace-a-000", 0, 4, 36, "lower", articulations=("accent",)),),
            "articulations",
        ),
        ((ScoreNote("workspace-a-000", 0, 5, 36, "lower"),), "duration"),
        ((ScoreNote("workspace-a-000", 0, 4, 37, "lower"),), "degree"),
    ],
)
def test_rejects_other_placer_changes(
    monkeypatch: pytest.MonkeyPatch,
    notes: tuple[ScoreNote, ...],
    message: str,
) -> None:
    monkeypatch.setattr(
        accompaniment_module,
        "place_piano_texture_v8",
        lambda *args, **kwargs: SimpleNamespace(
            status="placed",
            notes=notes,
            reason=None,
            total_candidate_evaluation_count=1,
        ),
    )

    with pytest.raises(ValueError, match=message):
        place_workspace_accompaniment(
            material_key="theme",
            tonal_center=0,
            mode="major",
            harmonies=(HarmonyChord(4, 0, "major"),),
            melody=MelodyValue("upper", (MelodyNote(0, 4, 72),)),
            requests=(AccompanimentRequestEvent(0, 4, "root", "bass", ()),),
        )


def test_returns_a_typed_unplaced_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        accompaniment_module,
        "place_piano_texture_v8",
        lambda *args, **kwargs: SimpleNamespace(
            status="search_unplaceable",
            notes=(),
            reason="no joint placement",
            total_candidate_evaluation_count=3,
        ),
    )

    result = place_workspace_accompaniment(
        material_key="theme",
        tonal_center=0,
        mode="major",
        harmonies=(HarmonyChord(4, 0, "major"),),
        melody=MelodyValue("upper", (MelodyNote(0, 4, 72),)),
        requests=(AccompanimentRequestEvent(0, 4, "root", "bass", ()),),
    )

    assert result.status == "search_unplaceable"
    assert result.value is None
    assert result.reason == "no joint placement"
