from __future__ import annotations

from dataclasses import replace

import pytest

from llm_musical_composer.composition_ir import Material, Note, Pedal
from llm_musical_composer.intensity_patch import (
    IntensityPatchError,
    MaterialPatch,
    NoteClone,
    NoteEdit,
    apply_material_patch,
    parse_material_patch_batch,
    plan_material_intensity_patch,
)


def _material() -> Material:
    return Material(
        "A",
        2_000,
        (
            Note("bass-1", 0, 600, 48, 70, "lower"),
            Note("melody-1", 0, 400, 60, 90, "upper"),
            Note("bass-2", 500, 600, 50, 72, "lower"),
            Note("melody-2", 1_000, 400, 62, 92, "upper"),
        ),
        (Pedal("p1", 0, 127), Pedal("p2", 1_900, 0)),
    )


def test_patch_dsl_parses_only_compact_note_operations() -> None:
    batch_id, patches = parse_material_patch_batch(
        """material_patch_batch("batch-1", patches=[
            material_patch("A", remove_ids=["bass-2"], edits=[
                note_edit("bass-1", at_ms=250, duration_ms=500),
            ], additions=[
                note_clone("melody-1", "melody-1-copy", at_ms=750,
                           duration_ms=200, pitch=60),
            ], velocity_offset=-4),
        ])"""
    )

    assert batch_id == "batch-1"
    assert patches == (
        MaterialPatch(
            "A",
            ("bass-2",),
            (NoteEdit("bass-1", 250, 500),),
            (NoteClone("melody-1", "melody-1-copy", 750, 200, 60),),
            -4,
        ),
    )


def test_patch_application_is_deterministic_and_keeps_metadata() -> None:
    material = _material()
    patch = MaterialPatch(
        "A",
        ("bass-2",),
        (NoteEdit("bass-1", 250, 500),),
        (NoteClone("melody-1", "melody-1-copy", 750, 200, 60),),
        -4,
    )

    result = apply_material_patch(
        material,
        patch,
        target_counts={"note_count": 4, "attack_count": 4},
    )

    assert result.material_id == material.material_id
    assert result.duration_ms == material.duration_ms
    assert result.pedals == material.pedals
    assert result.derived_from == material.derived_from
    assert [(note.event_id, note.at_ms) for note in result.notes] == [
        ("melody-1", 0),
        ("bass-1", 250),
        ("melody-1-copy", 750),
        ("melody-2", 1_000),
    ]
    assert [note.velocity for note in result.notes] == [86, 66, 86, 88]


@pytest.mark.parametrize(
    "patch, message",
    [
        (
            MaterialPatch("B", (), (), (), 0),
            "material ID",
        ),
        (
            MaterialPatch("A", ("missing",), (), (), 0),
            "unknown removed",
        ),
        (
            MaterialPatch("A", (), (NoteEdit("missing", 250, 100),), (), 0),
            "unknown edited",
        ),
        (
            MaterialPatch("A", (), (), (NoteClone("missing", "new", 250, 100, 60),), 0),
            "unknown clone source",
        ),
        (
            MaterialPatch("A", (), (), (NoteClone("bass-1", "bass-2", 250, 100, 48),), 0),
            "duplicate event ID",
        ),
        (
            MaterialPatch("A", (), (NoteEdit("melody-1", 250, 300),), (), 0),
            "upper voice onset",
        ),
        (
            MaterialPatch("A", (), (NoteEdit("bass-1", 40, 300),), (), 0),
            "microtiming",
        ),
        (
            MaterialPatch("A", (), (NoteEdit("bass-1", -1, 300),), (), 0),
            "outside",
        ),
        (
            MaterialPatch("A", (), (), (NoteClone("bass-1", "new", 250, 100, 49),), 0),
            "pitch-class",
        ),
        (
            MaterialPatch("A", (), (), (NoteClone("bass-1", "new", 250, 100, 72),), 0),
            "pitch range",
        ),
        (
            MaterialPatch("A", (), (), (), 50),
            "velocity",
        ),
    ],
)
def test_patch_rejects_scope_and_performance_artifacts(patch: MaterialPatch, message: str) -> None:
    with pytest.raises(IntensityPatchError, match=message):
        apply_material_patch(_material(), patch)


def test_patch_rejects_duplicate_operations_and_wrong_target_counts() -> None:
    material = _material()
    duplicate = MaterialPatch("A", ("bass-1", "bass-1"), (), (), 0)
    with pytest.raises(IntensityPatchError, match="remove_ids are not unique"):
        apply_material_patch(material, duplicate)
    add_and_edit = MaterialPatch(
        "A",
        (),
        (NoteEdit("new", 250, 100),),
        (NoteClone("bass-1", "new", 750, 100, 48),),
        0,
    )
    with pytest.raises(IntensityPatchError, match="unknown edited"):
        apply_material_patch(material, add_and_edit)

    unchanged = MaterialPatch("A", (), (), (), 0)
    with pytest.raises(IntensityPatchError, match="note count"):
        apply_material_patch(
            material,
            unchanged,
            target_counts={"note_count": 3, "attack_count": 3},
        )
    with pytest.raises(IntensityPatchError, match="attack count"):
        apply_material_patch(
            material,
            unchanged,
            target_counts={"note_count": 4, "attack_count": 4},
        )


def test_patch_parser_rejects_full_material_output_and_unknown_syntax() -> None:
    with pytest.raises(IntensityPatchError, match="invalid material patch batch"):
        parse_material_patch_batch('material_batch("x", materials=[])')
    with pytest.raises(IntensityPatchError, match="invalid material patch batch"):
        parse_material_patch_batch(
            'material_patch_batch("x", patches=[material_patch("A", '
            "remove_ids=[], edits=[], additions=[], velocity_offset=0, notes=[])])"
        )


def test_patch_does_not_mutate_the_input_material() -> None:
    material = _material()
    before = replace(material)
    apply_material_patch(material, MaterialPatch("A", (), (), (), 0))
    assert material == before


@pytest.mark.parametrize(
    "note_count, attack_count, velocity_level",
    [(3, 2, 0.45), (4, 4, 0.60), (7, 6, 0.75)],
)
def test_deterministic_patch_planner_reaches_exact_counts_without_moving_upper_onsets(
    note_count: int, attack_count: int, velocity_level: float
) -> None:
    material = _material()

    first = plan_material_intensity_patch(
        material,
        target_counts={"note_count": note_count, "attack_count": attack_count},
        target_velocity_level=velocity_level,
    )
    second = plan_material_intensity_patch(
        material,
        target_counts={"note_count": note_count, "attack_count": attack_count},
        target_velocity_level=velocity_level,
    )
    revised = apply_material_patch(
        material,
        first,
        target_counts={"note_count": note_count, "attack_count": attack_count},
    )

    assert first == second
    assert len(revised.notes) == note_count
    assert len({note.at_ms for note in revised.notes}) == attack_count
    original_upper = {note.event_id: note.at_ms for note in material.notes if note.voice == "upper"}
    assert all(
        note.at_ms == original_upper[note.event_id]
        for note in revised.notes
        if note.event_id in original_upper
    )
    assert revised.pedals == material.pedals


def test_deterministic_patch_planner_rejects_impossible_targets() -> None:
    with pytest.raises(IntensityPatchError, match="target counts"):
        plan_material_intensity_patch(
            _material(),
            target_counts={"note_count": 2, "attack_count": 3},
            target_velocity_level=0.5,
        )
