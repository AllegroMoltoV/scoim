from __future__ import annotations

from dataclasses import replace

from llm_musical_composer.music_dsl import parse_composition
from llm_musical_composer.section_contrast import evaluate_section_contrast
from tests.test_music_dsl import CONTRACT_ENDING_SOURCE, ENDING_SOURCE


def test_clear_relative_attack_and_register_contrast_passes() -> None:
    composition = parse_composition(CONTRACT_ENDING_SOURCE, require_section_contract=True)

    report = evaluate_section_contrast(composition)

    assert report["status"] == "pass"
    assert report["issues"] == []
    assert report["climax"]["section_index"] == 1
    assert set(report["climax"]["supporting_metrics"]) >= {"note_density", "velocity_median"}
    assert report["boundaries"][0]["attack_pattern_changed"] is True
    assert report["boundaries"][0]["clustered_note_ratio_shift"] >= 0.2
    assert report["boundaries"][0]["secondary_changes"]


def test_velocity_only_does_not_count_as_a_scene_change() -> None:
    composition = parse_composition(CONTRACT_ENDING_SOURCE, require_section_contract=True)
    material_b = composition.material_by_id["B"]
    grouped_notes = tuple(
        replace(note, at_ms=(index // 4) * 1000) for index, note in enumerate(material_b.notes)
    )
    grouped_b = replace(material_b, notes=grouped_notes)
    composition = replace(
        composition,
        materials=tuple(
            grouped_b if material.material_id == "B" else material
            for material in composition.materials
        ),
    )

    report = evaluate_section_contrast(composition)

    assert report["status"] == "fail"
    assert any("attack pattern" in issue for issue in report["issues"])
    assert (
        report["sections"][1]["features"]["velocity_median"]
        > report["sections"][0]["features"]["velocity_median"]
    )


def test_missing_contract_is_reported_as_unavailable() -> None:
    composition = parse_composition(ENDING_SOURCE)

    report = evaluate_section_contrast(composition)

    assert report["status"] == "unable_to_investigate"
    assert report["sections"] == []
    assert report["issues"] == ["section contract is missing"]


def test_empty_and_ambiguous_attacks_do_not_create_a_relative_scene_change() -> None:
    composition = parse_composition(CONTRACT_ENDING_SOURCE, require_section_contract=True)
    material_a = composition.material_by_id["A"]
    empty_a = replace(material_a, notes=())
    empty_b = replace(composition.material_by_id["B"], notes=())
    empty = replace(composition, materials=(empty_a, empty_b))

    empty_report = evaluate_section_contrast(empty)

    assert empty_report["status"] == "fail"
    assert empty_report["sections"][0]["features"]["observed_attack_style"] == "ambiguous"
    assert not any(boundary["scene_change_pass"] for boundary in empty_report["boundaries"])

    ambiguous_notes = tuple(
        replace(note, at_ms=0 if index < 3 else (index - 2) * 1000)
        for index, note in enumerate(material_a.notes[:6])
    )
    ambiguous_a = replace(material_a, notes=ambiguous_notes)
    ambiguous = replace(
        composition,
        materials=(ambiguous_a, composition.material_by_id["B"]),
    )

    ambiguous_report = evaluate_section_contrast(ambiguous)

    assert ambiguous_report["sections"][0]["features"]["observed_attack_style"] == "ambiguous"


def test_mixed_attack_patterns_can_pass_without_pure_binary_styles() -> None:
    composition = parse_composition(CONTRACT_ENDING_SOURCE, require_section_contract=True)
    material_a = composition.material_by_id["A"]
    material_b = composition.material_by_id["B"]
    mixed_a = replace(
        material_a,
        notes=tuple(
            replace(note, at_ms=(0 if index < 3 else 2500 if index < 6 else index * 700))
            for index, note in enumerate(material_a.notes)
        ),
    )
    mixed_b = replace(
        material_b,
        notes=tuple(
            replace(note, at_ms=(0 if index < 3 else index * 500))
            for index, note in enumerate(material_b.notes)
        ),
    )
    mixed = replace(composition, materials=(mixed_a, mixed_b))

    report = evaluate_section_contrast(mixed)

    assert report["status"] == "pass"
    assert 0 < report["sections"][0]["features"]["clustered_note_ratio"] < 1
    assert 0 < report["sections"][1]["features"]["clustered_note_ratio"] < 1


def test_false_climax_fails_energy_direction_and_activity_maxima() -> None:
    composition = parse_composition(CONTRACT_ENDING_SOURCE, require_section_contract=True)
    material_b = composition.material_by_id["B"]
    quiet_notes = tuple(
        replace(note, duration_ms=100, velocity=30) for note in material_b.notes[:2]
    )
    quiet_b = replace(material_b, notes=quiet_notes)
    false_climax = replace(
        composition,
        materials=(composition.material_by_id["A"], quiet_b),
    )

    report = evaluate_section_contrast(false_climax)

    assert report["status"] == "fail"
    assert report["climax"]["passes"] is False
    assert any("energy direction" in issue for issue in report["issues"])
    assert any("activity maxima" in issue for issue in report["issues"])


def test_equal_energy_boundary_and_density_change_are_reported_separately() -> None:
    composition = parse_composition(CONTRACT_ENDING_SOURCE, require_section_contract=True)
    material_b = composition.material_by_id["B"]
    dense_notes = tuple(
        replace(note, pitch=58 + index % 5, velocity=60)
        for index, note in enumerate(material_b.notes[:15])
    )
    dense_b = replace(material_b, duration_ms=9000, notes=dense_notes)
    opening = composition.form[0]
    equal_energy_form = (
        opening,
        replace(opening, role="contrast"),
        composition.form[1],
        composition.form[2],
    )
    dense = replace(
        composition, form=equal_energy_form, materials=(composition.materials[0], dense_b)
    )

    report = evaluate_section_contrast(dense)

    assert report["boundaries"][0]["energy_direction_supporting_metrics"] == []
    assert "note_density" in report["boundaries"][1]["secondary_changes"]
