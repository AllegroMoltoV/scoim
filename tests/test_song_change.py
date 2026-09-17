from __future__ import annotations

from dataclasses import replace

from llm_musical_composer.composition_ir import Composition, Material, Note, Part, Phrase, Use
from llm_musical_composer.song_change import (
    _normalized_positions,
    _rhythm,
    describe_song_change,
)


def _material(
    material_id: str,
    upper: tuple[tuple[int, int], ...],
    lower: tuple[tuple[int, int], ...],
    *,
    derived_from: str | None = None,
    velocity: int = 60,
) -> Material:
    notes = tuple(
        Note(f"{material_id}-u-{index}", at, 300, pitch, velocity, "upper")
        for index, (at, pitch) in enumerate(upper)
    ) + tuple(
        Note(f"{material_id}-l-{index}", at, 400, pitch, velocity, "lower")
        for index, (at, pitch) in enumerate(lower)
    )
    return Material(material_id, 1000, notes, derived_from=derived_from)


def _composition() -> Composition:
    materials = (
        _material("A", ((0, 60), (500, 62), (800, 64)), ((0, 48), (500, 50))),
        _material(
            "V",
            ((0, 60), (400, 62), (800, 64)),
            ((100, 48), (600, 50)),
            derived_from="A",
        ),
        _material("C", ((0, 67), (500, 60), (800, 66)), ((0, 42), (500, 54))),
    )
    return Composition(
        title="change",
        form=(Use("A"), Use("V"), Use("C"), Use("A")),
        materials=materials,
        parts=(
            Part("P1", "opening", 2, 0, 1),
            Part("P2", "development", 3, 1, 2),
            Part("P3", "climax", 5, 2, 3),
            Part("P4", "return", 2, 3, 4),
        ),
        phrases=(
            Phrase("S1", "P1", "statement", None, 0, 1),
            Phrase("V1", "P2", "variation", "S1", 1, 2, "rhythmic"),
            Phrase("C1", "P3", "contrast", None, 2, 3),
            Phrase("R1", "P4", "return", "S1", 3, 4),
        ),
    )


def _relation(report: dict[str, object], relation: str) -> dict[str, object]:
    return next(item for item in report["phrase_relations"] if item["relation"] == relation)


def test_describes_phrase_and_part_relations_without_a_total_score() -> None:
    report = describe_song_change(_composition())

    assert report["status"] == "measured"
    assert report["hold"] == {
        "duration_ms": 4000,
        "note_count": 20,
        "velocity_median": 60.0,
        "mean_polyphony": 1.7,
        "pitch_classes": [0, 2, 4, 6, 7],
    }
    assert [item["relation"] for item in report["phrase_relations"]] == [
        "variation",
        "contrast",
        "return",
    ]
    returned = _relation(report, "return")
    assert returned["upper_interval_similarity"] == 1.0
    assert returned["upper_rhythm_similarity"] == 1.0
    assert returned["upper_rhythm_position_distance"] == 0.0
    assert returned["upper_interval_distance_semitones"] == 0.0
    assert returned["lower_interval_similarity"] == 1.0
    assert returned["lower_rhythm_similarity"] == 1.0
    assert returned["upper_register_shift_semitones"] == 0.0
    assert len(report["part_relations"]) == 6
    assert "score" not in report
    assert "passes" not in report
    assert "target" not in report


def test_contrast_and_return_change_independently_with_note_count_held() -> None:
    original = _composition()
    original_report = describe_song_change(original)
    material_c = original.material_by_id["C"]
    stronger_c = replace(
        material_c,
        notes=tuple(
            replace(note, at_ms=(0, 120, 900, 0, 900)[index], pitch=note.pitch + index * 2)
            for index, note in enumerate(material_c.notes)
        ),
    )
    stronger = replace(
        original,
        materials=tuple(
            stronger_c if item.material_id == "C" else item for item in original.materials
        ),
    )

    before_contrast = _relation(original_report, "contrast")
    after_contrast = _relation(describe_song_change(stronger), "contrast")
    assert stronger.note_count == original.note_count
    assert (
        after_contrast["upper_rhythm_position_distance"]
        > before_contrast["upper_rhythm_position_distance"]
    )
    assert _relation(describe_song_change(stronger), "return") == _relation(
        original_report, "return"
    )

    broken_return = _material(
        "R",
        ((0, 65), (200, 59), (850, 68)),
        ((100, 45), (750, 53)),
        derived_from="A",
    )
    broken = replace(
        original,
        form=(*original.form[:3], Use("R")),
        materials=(*original.materials, broken_return),
    )
    broken_report = describe_song_change(broken)

    assert broken.note_count == original.note_count
    assert _relation(broken_report, "return")["upper_rhythm_similarity"] < 1.0
    assert _relation(broken_report, "contrast") == before_contrast


def test_velocity_labels_transposition_and_time_scale_do_not_fake_structure_change() -> None:
    original = _composition()
    report = describe_song_change(original)
    louder_c = replace(
        original.material_by_id["C"],
        notes=tuple(replace(note, velocity=90) for note in original.material_by_id["C"].notes),
    )
    louder = replace(
        original,
        materials=tuple(
            louder_c if item.material_id == "C" else item for item in original.materials
        ),
        parts=tuple(replace(part, energy=max(1, part.energy - 1)) for part in original.parts),
    )

    assert describe_song_change(louder)["phrase_relations"] == report["phrase_relations"]
    assert describe_song_change(louder)["part_relations"] == report["part_relations"]

    stretched_materials = tuple(
        replace(
            material,
            duration_ms=material.duration_ms * 2,
            notes=tuple(
                replace(
                    note,
                    at_ms=note.at_ms * 2,
                    duration_ms=note.duration_ms * 2,
                    pitch=note.pitch + 5,
                )
                for note in material.notes
            ),
        )
        for material in original.materials
    )
    stretched = replace(original, materials=stretched_materials)
    stretched_report = describe_song_change(stretched)

    assert stretched_report["phrase_relations"] == report["phrase_relations"]
    assert stretched_report["part_relations"] == report["part_relations"]
    assert stretched_report["hold"]["duration_ms"] == 8000


def test_missing_hierarchy_or_note_evidence_is_not_reported_as_zero_distance() -> None:
    without_hierarchy = replace(_composition(), parts=(), phrases=())
    empty_a = replace(_composition().material_by_id["A"], notes=())
    empty = replace(
        _composition(),
        materials=tuple(
            empty_a if item.material_id == "A" else item for item in _composition().materials
        ),
    )

    missing = describe_song_change(without_hierarchy)
    empty_report = describe_song_change(empty)

    assert missing["status"] == "unable_to_investigate"
    assert missing["phrase_relations"] == []
    assert empty_report["status"] == "measured_with_unavailable_metrics"
    assert _relation(empty_report, "variation")["upper_interval_similarity"] is None
    assert empty_report["unavailable_metrics"]


def test_degenerate_timing_and_missing_contrast_source_remain_unavailable() -> None:
    simultaneous = (
        Note("n1", 0, 100, 60, 60, "upper"),
        Note("n2", 0, 100, 62, 60, "upper"),
    )
    assert _rhythm(simultaneous) == ()
    assert _normalized_positions(simultaneous) == ()

    original = _composition()
    contrast_first = replace(
        original,
        phrases=(replace(original.phrases[2], start_use_index=0, end_use_index=1),),
    )
    report = describe_song_change(contrast_first)

    assert report["status"] == "measured_with_unavailable_metrics"
    assert report["phrase_relations"] == []
    assert report["unavailable_metrics"] == ["contrast:C1:source_statement"]


def test_empty_performance_reports_missing_velocity_instead_of_zero() -> None:
    original = _composition()
    empty = replace(
        original,
        materials=tuple(replace(material, notes=()) for material in original.materials),
    )

    report = describe_song_change(empty)

    assert report["hold"]["velocity_median"] is None
    assert "hold:velocity_median" in report["unavailable_metrics"]
