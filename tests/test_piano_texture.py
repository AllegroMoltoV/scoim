from __future__ import annotations

from dataclasses import replace

from llm_musical_composer.composition_ir import Material, Note
from llm_musical_composer.music_dsl import THREE_MINUTE_POLICY, parse_composition
from llm_musical_composer.piano_texture import (
    _rhythm,
    _variation_pair,
    evaluate_piano_texture,
    evaluate_variation_contracts,
    voice_texture_features,
)
from tests.test_music_dsl import PHRASE_PARTS_SOURCE


def _notes(
    prefix: str,
    upper_onsets: tuple[int, ...],
    lower_onsets: tuple[int, ...],
    *,
    upper_pitches: tuple[int, ...] = (72, 74, 76, 77),
    lower_pitches: tuple[int, ...] = (48, 52, 50, 55),
) -> tuple[Note, ...]:
    upper = tuple(
        Note(f"{prefix}-u-{index}", onset, 350, upper_pitches[index], 84, "upper")
        for index, onset in enumerate(upper_onsets)
    )
    lower = tuple(
        Note(f"{prefix}-l-{index}", onset, 900, lower_pitches[index], 68, "lower")
        for index, onset in enumerate(lower_onsets)
    )
    return (*upper, *lower)


def test_voice_texture_rejects_lockstep_and_temporally_disjoint_layers() -> None:
    lockstep = voice_texture_features(_notes("s", (0, 500, 1_000, 1_500), (0, 500, 1_000, 1_500)))
    disjoint = voice_texture_features(
        _notes("d", (0, 500, 1_000, 1_500), (3_000, 3_500, 4_000, 4_500))
    )

    assert lockstep["near_onset_coupling"] == 1.0
    assert disjoint["cross_voice_active_ratio"] == 0.0


def test_voice_texture_accepts_two_layers_that_meet_and_separate() -> None:
    report = voice_texture_features(_notes("g", (0, 500, 1_000, 1_500), (0, 750, 1_500, 2_250)))

    assert report["status"] == "pass"
    assert 0.1 <= report["near_onset_coupling"] <= 0.85
    assert report["cross_voice_active_ratio"] >= 0.25
    assert report["pitch_median_separation"] >= 5


def test_voice_texture_reports_missing_labels_single_onsets_and_close_registers() -> None:
    empty = voice_texture_features(())
    single = voice_texture_features(
        (
            Note("u", 0, 400, 60, 80, "upper"),
            Note("l", 0, 800, 58, 70, "lower"),
            Note("unknown", 200, 300, 64, 75, None),
        )
    )

    assert empty["status"] == "fail"
    assert empty["duration_medians_ms"] == {"upper": None, "lower": None}
    assert any("label voice" in issue for issue in single["issues"])
    assert any("distinct onsets" in issue for issue in single["issues"])
    assert any("registers" in issue for issue in single["issues"])


def test_piano_texture_requires_both_labeled_voices_in_every_phrase() -> None:
    plan = parse_composition(
        PHRASE_PARTS_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    materials = tuple(
        replace(
            material,
            notes=_notes(material.material_id, (0, 500, 1_000, 1_500), (0, 750, 1_500, 2_250)),
        )
        for material in plan.materials
    )
    composition = replace(plan, materials=materials)

    assert evaluate_piano_texture(composition)["status"] == "pass"

    upper_only = tuple(note for note in materials[1].notes if note.voice == "upper")
    broken = replace(
        composition,
        materials=(materials[0], replace(materials[1], notes=upper_only), materials[2]),
    )
    report = evaluate_piano_texture(broken)

    assert report["status"] == "fail"
    assert any("both voices" in issue for issue in report["issues"])


def _variation_composition(kind: str, variation: tuple[Note, ...]):
    source = PHRASE_PARTS_SOURCE.replace(
        'phrase("V1", role="variation", derived_from="S1"',
        f'phrase("V1", role="variation", derived_from="S1", variation_kind="{kind}"',
    )
    plan = parse_composition(
        source,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    base = _notes("a", (0, 500, 1_000, 1_500), (0, 750, 1_500, 2_250))
    return replace(
        plan,
        materials=(
            replace(plan.materials[0], notes=base),
            replace(plan.materials[1], notes=variation),
            replace(
                plan.materials[2],
                notes=_notes("c", (0, 500, 1_000, 1_500), (0, 750, 1_500, 2_250)),
            ),
        ),
    )


def test_rhythmic_variation_must_change_rhythm_but_preserve_upper_contour() -> None:
    near_copy = _notes("b", (0, 500, 1_000, 1_500), (0, 750, 1_500, 2_250))
    changed = _notes("b", (0, 350, 1_050, 1_800), (0, 900, 1_400, 2_400))

    assert (
        evaluate_variation_contracts(_variation_composition("rhythmic", near_copy))["status"]
        == "fail"
    )
    assert (
        evaluate_variation_contracts(_variation_composition("rhythmic", changed))["status"]
        == "pass"
    )


def test_textural_variation_changes_lower_layer_without_losing_upper_motif() -> None:
    changed = _notes(
        "b",
        (0, 500, 1_000, 1_500),
        (250, 1_000, 1_750, 2_500),
        lower_pitches=(43, 50, 47, 52),
    )
    unrelated = _notes(
        "x",
        (0, 500, 1_000, 1_500),
        (250, 1_000, 1_750, 2_500),
        upper_pitches=(72, 67, 79, 65),
        lower_pitches=(43, 50, 47, 52),
    )

    assert (
        evaluate_variation_contracts(_variation_composition("textural", changed))["status"]
        == "pass"
    )
    assert (
        evaluate_variation_contracts(_variation_composition("textural", unrelated))["status"]
        == "fail"
    )


def test_registral_variation_preserves_pattern_and_moves_register() -> None:
    moved = _notes(
        "b",
        (0, 500, 1_000, 1_500),
        (0, 750, 1_500, 2_250),
        upper_pitches=(79, 81, 83, 84),
        lower_pitches=(43, 47, 45, 50),
    )

    assert (
        evaluate_variation_contracts(_variation_composition("registral", moved))["status"] == "pass"
    )


def test_registral_variation_allows_diatonic_interval_size_changes() -> None:
    diatonically_moved = _notes(
        "b",
        (0, 500, 1_000, 1_500),
        (0, 750, 1_500, 2_250),
        upper_pitches=(79, 82, 84, 86),
        lower_pitches=(43, 47, 45, 50),
    )

    report = evaluate_variation_contracts(_variation_composition("registral", diatonically_moved))

    assert report["status"] == "pass"
    assert report["pairs"][0]["upper_direction_similarity"] == 1.0


def test_variation_primitives_report_degenerate_and_unknown_cases() -> None:
    empty = Material("A", 5_500, ())
    one_onset = [
        Note("u1", 100, 300, 72, 80, "upper"),
        Note("u2", 100, 300, 76, 80, "upper"),
    ]
    regular = Material(
        "B",
        5_500,
        _notes("b", (0, 500, 1_000, 1_500), (0, 750, 1_500, 2_250)),
    )

    assert _rhythm([]) == ()
    assert _rhythm(one_onset) == (0, 0)
    assert "both voices" in _variation_pair(empty, empty, "rhythmic")["issues"][0]
    assert _variation_pair(regular, regular, "registral")["status"] == "fail"
    unrelated = Material(
        "C",
        5_500,
        _notes(
            "c",
            (0, 350, 1_050, 1_800),
            (0, 750, 1_500, 2_250),
            upper_pitches=(72, 67, 79, 65),
        ),
    )
    assert "preserve rhythm" in _variation_pair(regular, unrelated, "registral")["issues"][0]
    assert _variation_pair(regular, regular, "unknown")["status"] == "fail"


def test_legacy_variation_without_kind_is_reported_instead_of_guessed() -> None:
    plan = parse_composition(
        PHRASE_PARTS_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )

    report = evaluate_variation_contracts(plan)

    assert report["status"] == "fail"
    assert report["pairs"] == []
    assert "requires variation_kind" in report["issues"][0]


def test_variation_evaluation_skips_transitions_and_unrelated_materials() -> None:
    composition = _variation_composition(
        "rhythmic", _notes("b", (0, 350, 1_050, 1_800), (0, 900, 1_400, 2_400))
    )
    transition_form = list(composition.form)
    transition_form[1] = replace(transition_form[1], role="transition")
    unrelated_materials = (
        composition.materials[0],
        replace(composition.materials[1], derived_from=None),
        composition.materials[2],
    )

    assert (
        evaluate_variation_contracts(replace(composition, form=tuple(transition_form)))["pairs"]
        == []
    )
    assert (
        evaluate_variation_contracts(replace(composition, materials=unrelated_materials))["pairs"]
        == []
    )
