from __future__ import annotations

from dataclasses import replace

import pytest

from llm_musical_composer import song_change_arrangement
from llm_musical_composer.composition_ir import (
    Composition,
    Material,
    Note,
    Part,
    Phrase,
    TonicEnding,
    Use,
)
from llm_musical_composer.song_change_arrangement import (
    SongChangeArrangementError,
    _contrast_slots,
    _distance_components,
    derive_arrangement_triplet,
)


def _material(material_id: str, upper: tuple[int, ...], lower: tuple[int, ...]) -> Material:
    notes = []
    for index, pitch in enumerate(upper):
        notes.append(Note(f"{material_id}-u-{index}", index * 600, 350, pitch, 70 + index, "upper"))
    for index, pitch in enumerate(lower):
        notes.append(Note(f"{material_id}-l-{index}", index * 600, 700, pitch, 60 + index, "lower"))
    return Material(material_id, 2_400, tuple(notes))


def _composition() -> Composition:
    materials = (
        _material("S1", (60, 62, 64, 65), (48, 50, 52, 53)),
        _material("S2", (72, 69, 67, 64), (55, 52, 50, 47)),
        _material("S3", (65, 65, 67, 65), (41, 48, 41, 48)),
        _material("C1", (60, 62, 64, 65), (48, 50, 52, 53)),
        _material("C2", (65, 65, 67, 65), (41, 48, 41, 48)),
        _material("C3", (72, 69, 67, 64), (55, 52, 50, 47)),
    )
    return Composition(
        title="arrangement",
        tonal_center=0,
        mode="major",
        ending=TonicEnding(4_000),
        form=tuple(
            Use(material_id, role=role, energy=2)
            for material_id, role in (
                ("S1", "opening"),
                ("C1", "contrast"),
                ("S2", "opening"),
                ("C2", "contrast"),
                ("S3", "opening"),
                ("C3", "contrast"),
            )
        ),
        materials=materials,
        parts=(Part("P1", "opening", 2, 0, 6),),
        phrases=(
            Phrase("S-P1", "P1", "statement", None, 0, 1),
            Phrase("C-P1", "P1", "contrast", None, 1, 2),
            Phrase("S-P2", "P1", "statement", None, 2, 3),
            Phrase("C-P2", "P1", "contrast", None, 3, 4),
            Phrase("S-P3", "P1", "statement", None, 4, 5),
            Phrase("C-P3", "P1", "contrast", None, 5, 6),
        ),
    )


def test_arrangement_triplet_changes_only_contrast_placement() -> None:
    center = _composition()

    triplet = derive_arrangement_triplet(center)

    assert triplet.center == center
    assert triplet.selection["low_objective"] < triplet.selection["center_low_objective"]
    assert triplet.selection["high_objective"] > triplet.selection["center_high_objective"]
    assert triplet.selection["low_order"] != triplet.selection["high_order"]
    assert triplet.low.note_count == triplet.center.note_count == triplet.high.note_count
    assert (
        triplet.selection["hold"]["low"]["velocity_median"]
        == triplet.selection["hold"]["high"]["velocity_median"]
    )
    assert [triplet.low.form[index].material_id for index in (0, 2, 4)] == ["S1", "S2", "S3"]
    assert all(
        triplet.low.form[index].material_id.startswith("song-change-low-") for index in (1, 3, 5)
    )


def test_contrast_selection_rejects_missing_statement_and_too_few_slots() -> None:
    composition = _composition()
    contrast_only = replace(
        composition,
        phrases=(replace(composition.phrases[1], start_use_index=0, end_use_index=1),),
    )
    with pytest.raises(SongChangeArrangementError, match="no prior statement"):
        _contrast_slots(contrast_only)

    one_pair = replace(composition, form=composition.form[:2], phrases=composition.phrases[:2])
    with pytest.raises(SongChangeArrangementError, match="at least three"):
        _contrast_slots(one_pair)


def test_contrast_selection_caps_a_long_phrase_at_eight_slots() -> None:
    composition = _composition()
    form = (
        composition.form[0],
        *(Use("C1", role="contrast", energy=2) for _ in range(10)),
    )
    phrases = (
        replace(composition.phrases[0], start_use_index=0, end_use_index=1),
        replace(composition.phrases[1], start_use_index=1, end_use_index=11),
    )

    slots = _contrast_slots(replace(composition, form=form, phrases=phrases))

    assert len(slots) == 8
    assert [slot.use_index for slot in slots] == list(range(1, 9))


def test_arrangement_rejects_unmeasurable_contrast() -> None:
    empty = Material("empty", 2_400, ())

    with pytest.raises(SongChangeArrangementError, match="unavailable metrics"):
        _distance_components(empty, empty)


def test_arrangement_rejects_indistinguishable_low_and_high_assignments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        song_change_arrangement,
        "permutations",
        lambda values: iter(((0, 1, 2), (1, 0, 2))),
    )

    with pytest.raises(SongChangeArrangementError, match="cannot be separated"):
        derive_arrangement_triplet(_composition())


@pytest.mark.parametrize(
    ("key_name", "message"),
    [
        ("_low_key", "not closer than the center"),
        ("_high_key", "not farther than the center"),
    ],
)
def test_arrangement_requires_both_objective_directions(
    key_name: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(song_change_arrangement, key_name, lambda distances: (0.0, 0.0, 0.0))

    with pytest.raises(SongChangeArrangementError, match=message):
        derive_arrangement_triplet(_composition())
