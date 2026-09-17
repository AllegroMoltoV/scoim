from __future__ import annotations

from dataclasses import replace

import pytest

from llm_musical_composer import piano_texture_register_placement as placement_module
from llm_musical_composer.performance_pipeline import (
    PiecePlan,
    PlanNode,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
)
from llm_musical_composer.piano_texture_pilot import (
    PianoTextureEvent as PianoTextureEventV1,
)
from llm_musical_composer.piano_texture_pilot import PianoTextureSpec as PianoTextureSpecV1
from llm_musical_composer.piano_texture_pilot import PianoTextureValidationError
from llm_musical_composer.piano_texture_register_placement import (
    REGISTER_ZONES,
    PianoTextureEventV2,
    PianoTexturePlacementResultV3,
    PianoTexturePlacementResultV4,
    PianoTexturePlacementResultV5,
    PianoTexturePlacementResultV6,
    PianoTexturePlacementResultV7,
    PianoTexturePlacementResultV8,
    PianoTextureSpecV2,
    combine_piano_texture_v2,
    convert_piano_texture_v1_to_v2,
    dump_piano_texture_spec_v2,
    low_spacing_violations,
    movement_distance,
    parse_piano_texture_spec_v2,
    place_piano_texture_v2,
    place_piano_texture_v3,
    place_piano_texture_v4,
    place_piano_texture_v5,
    place_piano_texture_v6,
    place_piano_texture_v7,
    place_piano_texture_v8,
    project_accompaniment_to_v2,
)


def _fixture(
    *,
    harmony: ScoreHarmony | None = None,
    foreground_voice: str = "upper",
) -> tuple[PiecePlan, ScoreSpec]:
    harmony = harmony or ScoreHarmony("h1", 0, 32, 2, "major")
    plan = PiecePlan(
        "placement-plan",
        "音域帯配置",
        2,
        "major",
        "whole",
        "tonic",
        (
            PlanNode("whole", None, 0, "whole"),
            PlanNode(
                "body",
                "whole",
                0,
                "statement",
                duration_weight=1,
                score_material_id="body",
            ),
        ),
    )
    foreground_pitch = 74 if foreground_voice == "upper" else 38
    material = ScoreMaterial(
        "body",
        32,
        (
            ScoreNote("foreground-1", 0, 8, foreground_pitch, foreground_voice),
            ScoreNote("foreground-2", 8, 8, foreground_pitch + 2, foreground_voice),
            ScoreNote("foreground-3", 16, 8, foreground_pitch - 2, foreground_voice),
            ScoreNote("foreground-4", 24, 8, foreground_pitch, foreground_voice),
            ScoreNote("old-root", 0, 8, 38, "lower" if foreground_voice == "upper" else "upper"),
            ScoreNote("old-fifth", 0, 8, 45, "lower" if foreground_voice == "upper" else "upper"),
        ),
        harmonies=(harmony,),
        foreground_voice=foreground_voice,
    )
    return plan, ScoreSpec("placement-score", 8, (material,))


def _v1_regression_texture() -> PianoTextureSpecV2:
    return PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "root", "h1", "accompaniment", "lower", 0, 8, "root", "low", ()
            ),
            PianoTextureEventV2(
                "fifth", "h1", "accompaniment", "lower", 0, 8, "fifth", "low", ()
            ),
            PianoTextureEventV2(
                "third", "h1", "accompaniment", "lower", 8, 8, "third", "low", ()
            ),
            PianoTextureEventV2(
                "root-2", "h1", "accompaniment", "lower", 8, 8, "root", "low", ()
            ),
        ),
    )


def test_v2_dsl_round_trips_and_has_no_exact_pitch_or_octave() -> None:
    source = dump_piano_texture_spec_v2(_v1_regression_texture())

    assert parse_piano_texture_spec_v2(source) == _v1_regression_texture()
    assert dump_piano_texture_spec_v2(parse_piano_texture_spec_v2(source)) == source
    assert "pitch=" not in source
    assert "octave=" not in source


@pytest.mark.parametrize(
    "source",
    (
        "piano_texture_spec_v2('body', events=[])",
        "piano_texture_spec_v2(material_id='body', events=[], pitch=38)",
        (
            "piano_texture_spec_v2(material_id='body', events=["
            "piano_texture_note_v2(event_id='x', harmony_id='h1', "
            "role='accompaniment', voice='lower', at_units=0.0, duration_units=8, "
            "degree='root', register_zone='bass', articulations=[])])"
        ),
        (
            "piano_texture_spec_v2(material_id='body', events=["
            "piano_texture_note_v2(event_id='x', harmony_id='h1', "
            "role='accompaniment', voice='lower', at_units=0, duration_units=8, "
            "degree='root', register_zone='unknown', articulations=[])])"
        ),
    ),
)
def test_v2_dsl_rejects_out_of_contract_values(source: str) -> None:
    with pytest.raises(PianoTextureValidationError):
        parse_piano_texture_spec_v2(source)


def test_register_zones_are_frozen_and_overlap() -> None:
    assert REGISTER_ZONES == {
        "bass": (21, 47, 36),
        "low": (36, 59, 48),
        "middle": (48, 71, 60),
        "high": (60, 108, 72),
    }
    assert 47 in range(REGISTER_ZONES["bass"][0], REGISTER_ZONES["bass"][1] + 1)
    assert 47 in range(REGISTER_ZONES["low"][0], REGISTER_ZONES["low"][1] + 1)


def test_movement_distance_handles_variable_chord_sizes() -> None:
    assert movement_distance((38, 45), (42,)) == 10
    assert movement_distance((), (42,)) == 0


def test_v1_low_spacing_regression_is_placed_without_changing_events() -> None:
    plan, score = _fixture()

    result = place_piano_texture_v2(plan, score, _v1_regression_texture())

    assert result.status == "placed"
    assert tuple(note.event_id for note in result.notes) == ("root", "fifth", "root-2", "third")
    assert len(result.notes) == 4
    assert {note.at_units for note in result.notes} == {0, 8}
    assert {note.pitch % 12 for note in result.notes if note.at_units == 0} == {2, 9}
    assert {note.pitch % 12 for note in result.notes if note.at_units == 8} == {2, 6}
    assert result.low_spacing_violations == 0


@pytest.mark.parametrize(
    "placer",
    (place_piano_texture_v2, place_piano_texture_v3, place_piano_texture_v4),
)
def test_placement_searches_only_inside_the_shared_allowed_pitch_range(placer) -> None:
    plan, score = _fixture()

    result = placer(
        plan,
        score,
        _v1_regression_texture(),
        allowed_pitch_range=(45, 59),
    )

    assert result.status == "placed"
    assert all(45 <= note.pitch <= 59 for note in result.notes)


@pytest.mark.parametrize(
    "placer",
    (place_piano_texture_v2, place_piano_texture_v3, place_piano_texture_v4),
)
def test_placement_stops_when_shared_allowed_pitch_range_has_no_solution(placer) -> None:
    plan, score = _fixture()

    result = placer(
        plan,
        score,
        _v1_regression_texture(),
        allowed_pitch_range=(51, 53),
    )

    assert result.status == "constraint_unplaceable"


def test_v1_conversion_preserves_symbolic_events_and_replaces_only_octave() -> None:
    plan, score = _fixture()
    source = PianoTextureSpecV1(
        "body",
        (
            PianoTextureEventV1(
                "root", "h1", "accompaniment", "lower", 0, 8, "root", 2, ("tenuto",)
            ),
            PianoTextureEventV1(
                "fifth", "h1", "accompaniment", "lower", 0, 8, "fifth", 2, ()
            ),
        ),
    )

    converted = convert_piano_texture_v1_to_v2(plan, score, source)

    assert converted == PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "root", "h1", "accompaniment", "lower", 0, 8, "root", "bass", ("tenuto",)
            ),
            PianoTextureEventV2(
                "fifth", "h1", "accompaniment", "lower", 0, 8, "fifth", "low", ()
            ),
        ),
    )


def test_low_third_is_retained_when_it_is_the_only_event() -> None:
    plan, score = _fixture()
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "third", "h1", "accompaniment", "lower", 0, 8, "third", "low", ()
            ),
        ),
    )

    result = place_piano_texture_v2(plan, score, texture)

    assert result.status == "placed"
    assert result.notes[0].pitch == 54


def test_diminished_fifth_is_not_treated_as_a_perfect_fifth_preference() -> None:
    plan, score = _fixture(harmony=ScoreHarmony("h1", 0, 32, 2, "diminished"))
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "dim-fifth", "h1", "accompaniment", "lower", 0, 8, "fifth", "low", ()
            ),
            PianoTextureEventV2(
                "root", "h1", "accompaniment", "lower", 0, 8, "root", "bass", ()
            ),
        ),
    )

    result = place_piano_texture_v2(plan, score, texture)

    assert result.status == "placed"
    assert min(result.notes, key=lambda note: note.pitch).pitch % 12 == 2


def test_sustained_low_note_is_included_in_later_spacing_check() -> None:
    plan, score = _fixture()
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "held-root", "h1", "accompaniment", "lower", 0, 16, "root", "bass", ()
            ),
            PianoTextureEventV2(
                "later-third", "h1", "accompaniment", "lower", 8, 8, "third", "low", ()
            ),
        ),
    )

    result = place_piano_texture_v2(plan, score, texture)

    assert result.status == "placed"
    assert tuple(note.pitch for note in result.notes) == (38, 54)
    assert result.low_spacing_violations == 0


def _foreground_spacing_fixture(
    foreground: ScoreNote,
) -> tuple[PiecePlan, ScoreSpec]:
    plan, score = _fixture(
        harmony=ScoreHarmony("h1", 0, 32, 0, "major"),
    )
    material = replace(
        score.materials[0],
        notes=(foreground,),
        foreground_voice="upper",
    )
    return plan, replace(score, materials=(material,))


def test_v4_backtracks_when_simultaneous_foreground_makes_low_spacing_invalid() -> None:
    foreground = ScoreNote("foreground", 0, 4, 41, "upper")
    plan, score = _foreground_spacing_fixture(foreground)
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "root", "h1", "accompaniment", "lower", 0, 4, "root", "bass", ()
            ),
        ),
    )

    result = place_piano_texture_v4(plan, score, texture)

    assert result.status == "placed"
    assert result.notes[0].pitch == 24
    assert low_spacing_violations((foreground, *result.notes)) == 0


def test_v4_backtracks_when_foreground_enters_after_a_low_accompaniment() -> None:
    foreground = ScoreNote("foreground", 1, 1, 41, "upper")
    plan, score = _foreground_spacing_fixture(foreground)
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "root", "h1", "accompaniment", "lower", 0, 4, "root", "bass", ()
            ),
        ),
    )

    result = place_piano_texture_v4(plan, score, texture)

    assert result.status == "placed"
    assert result.notes[0].pitch == 24
    assert low_spacing_violations((foreground, *result.notes)) == 0


def test_v4_keeps_a_path_that_becomes_valid_after_same_pitch_rearticulation() -> None:
    foreground = ScoreNote("foreground", 2, 1, 41, "upper")
    plan, score = _foreground_spacing_fixture(foreground)
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "root-old",
                "h1",
                "accompaniment",
                "lower",
                0,
                4,
                "root",
                "bass",
                (),
            ),
            PianoTextureEventV2(
                "root-new",
                "h1",
                "accompaniment",
                "lower",
                1,
                1,
                "root",
                "bass",
                (),
            ),
        ),
    )

    result = place_piano_texture_v4(plan, score, texture)

    assert result.status == "placed"
    assert tuple(note.pitch for note in result.notes) == (36, 36)
    assert tuple(note.duration_units for note in result.notes) == (1, 1)
    assert low_spacing_violations((foreground, *result.notes)) == 0


def test_constraint_unplaceable_does_not_delete_or_expand() -> None:
    plan, score = _fixture(foreground_voice="lower")
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "crossing", "h1", "accompaniment", "upper", 0, 8, "root", "bass", ()
            ),
        ),
    )

    result = place_piano_texture_v2(plan, score, texture)

    assert result.status == "constraint_unplaceable"
    assert result.failed_onset == 0
    assert result.notes == ()


def test_v3_rearticulates_same_physical_key_only_when_all_octaves_are_held() -> None:
    plan, score = _fixture()
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "root-1", "h1", "accompaniment", "lower", 0, 24, "root", "bass", ()
            ),
            PianoTextureEventV2(
                "root-2", "h1", "accompaniment", "lower", 8, 16, "root", "bass", ()
            ),
            PianoTextureEventV2(
                "root-3", "h1", "accompaniment", "lower", 16, 2, "root", "bass", ()
            ),
            PianoTextureEventV2(
                "root-4", "h1", "accompaniment", "lower", 20, 4, "root", "bass", ()
            ),
        ),
    )

    strict = place_piano_texture_v2(plan, score, texture)
    result = place_piano_texture_v3(plan, score, texture)

    assert strict.status == "greedy_unplaceable"
    assert isinstance(result, PianoTexturePlacementResultV3)
    assert result.status == "placed"
    assert len(result.notes) == 4
    assert len(result.rearticulations) == 1
    rearticulation = result.rearticulations[0]
    assert rearticulation.rearticulated_at_units == 16
    assert rearticulation.original_duration_units == 16
    assert rearticulation.final_duration_units == 8
    assert next(note for note in result.notes if note.event_id == "root-4").pitch == 26
    assert all(
        not (
            left.pitch == right.pitch
            and left.at_units < right.at_units + right.duration_units
            and right.at_units < left.at_units + left.duration_units
        )
        for index, left in enumerate(result.notes)
        for right in result.notes[index + 1 :]
    )


def test_v3_keeps_v2_result_when_another_octave_avoids_rearticulation() -> None:
    plan, score = _fixture()
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "root-1", "h1", "accompaniment", "lower", 0, 16, "root", "bass", ()
            ),
            PianoTextureEventV2(
                "root-2", "h1", "accompaniment", "lower", 8, 8, "root", "bass", ()
            ),
        ),
    )

    strict = place_piano_texture_v2(plan, score, texture)
    result = place_piano_texture_v3(plan, score, texture)

    assert strict.status == result.status == "placed"
    assert strict.notes == result.notes
    assert result.rearticulations == ()


def test_v4_keeps_the_ranked_v3_solution_when_no_backtracking_is_needed() -> None:
    plan, score = _fixture()
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "root", "h1", "accompaniment", "lower", 0, 8, "root", "low", ()
            ),
            PianoTextureEventV2(
                "fifth", "h1", "accompaniment", "lower", 8, 8, "fifth", "low", ()
            ),
        ),
    )

    legacy = place_piano_texture_v3(plan, score, texture)
    result = place_piano_texture_v4(plan, score, texture)

    assert isinstance(result, PianoTexturePlacementResultV4)
    assert result.status == "placed"
    assert result.notes == legacy.notes
    assert result.rearticulations == legacy.rearticulations
    assert result.backtrack_count == 0
    assert 0 < result.candidate_evaluation_count < 100_000


def test_v4_distinguishes_constraint_failure_from_search_exhaustion() -> None:
    plan, score = _fixture(foreground_voice="lower")
    impossible = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "crossing", "h1", "accompaniment", "upper", 0, 8, "root", "bass", ()
            ),
        ),
    )
    feasible = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "root", "h1", "accompaniment", "upper", 0, 8, "root", "high", ()
            ),
        ),
    )

    constraint = place_piano_texture_v4(plan, score, impossible)
    exhausted = place_piano_texture_v4(
        plan,
        score,
        feasible,
        maximum_candidate_evaluations=1,
    )

    assert constraint.status == "constraint_unplaceable"
    assert exhausted.status == "search_exhausted"
    assert exhausted.candidate_evaluation_count == 1


def test_v5_preserves_an_existing_v4_solution_without_projection() -> None:
    plan, score = _fixture()
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "root", "h1", "accompaniment", "lower", 0, 8, "root", "low", ()
            ),
            PianoTextureEventV2(
                "fifth", "h1", "accompaniment", "lower", 8, 8, "fifth", "low", ()
            ),
        ),
    )

    v4 = place_piano_texture_v4(plan, score, texture)
    v5 = place_piano_texture_v5(plan, score, texture)

    assert isinstance(v5, PianoTexturePlacementResultV5)
    assert v5.status == v4.status == "placed"
    assert v5.notes == v4.notes
    assert v5.rearticulations == v4.rearticulations
    assert v5.candidate_evaluation_count == v4.candidate_evaluation_count
    assert v5.backtrack_count == v4.backtrack_count
    assert v5.zone_projections == ()
    assert v5.fallback_status is None


def test_v5_does_not_project_when_v4_search_is_exhausted() -> None:
    plan, score = _fixture(foreground_voice="lower")
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "root", "h1", "accompaniment", "upper", 0, 8, "root", "high", ()
            ),
        ),
    )

    result = place_piano_texture_v5(
        plan,
        score,
        texture,
        maximum_candidate_evaluations=1,
    )

    assert result.status == "search_exhausted"
    assert result.primary_status == "search_exhausted"
    assert result.zone_projections == ()
    assert result.fallback_status is None


def _v6_spacing_case(
    *,
    root_pitch_class: int,
    quality: str,
    bass_degree: str,
    low_degree: str,
    low_at_units: int = 0,
) -> tuple[PiecePlan, ScoreSpec, PianoTextureSpecV2]:
    plan, score = _fixture(
        harmony=ScoreHarmony("h1", 0, 32, root_pitch_class, quality),
        foreground_voice="upper",
    )
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "bass", "h1", "accompaniment", "lower", 0, 8, bass_degree, "bass", ()
            ),
            PianoTextureEventV2(
                "low",
                "h1",
                "accompaniment",
                "lower",
                low_at_units,
                8,
                low_degree,
                "low",
                (),
            ),
        ),
    )
    return plan, score, texture


@pytest.mark.parametrize(
    ("root_pitch_class", "quality", "bass_degree", "low_degree"),
    (
        (0, "major", "fifth", "root"),
        (6, "diminished", "root", "fifth"),
        (5, "major", "third", "fifth"),
    ),
)
def test_v6_projects_only_low_event_when_candidate_pairs_are_too_close(
    root_pitch_class: int,
    quality: str,
    bass_degree: str,
    low_degree: str,
) -> None:
    plan, score, texture = _v6_spacing_case(
        root_pitch_class=root_pitch_class,
        quality=quality,
        bass_degree=bass_degree,
        low_degree=low_degree,
    )

    v5 = place_piano_texture_v5(
        plan, score, texture, allowed_pitch_range=(38, 86)
    )
    result = place_piano_texture_v6(
        plan, score, texture, allowed_pitch_range=(38, 86)
    )

    assert v5.status == "constraint_unplaceable"
    assert isinstance(result, PianoTexturePlacementResultV6)
    assert result.status == "placed"
    assert tuple(item.event_id for item in result.v6_zone_projections) == ("low",)
    assert result.v6_zone_projections[0].original_zone == "low"
    assert result.v6_zone_projections[0].effective_zone == "middle"
    assert result.v5_status == "constraint_unplaceable"
    assert result.v6_fallback_status == "placed"
    assert len(result.notes) == len(texture.events)
    assert result.low_spacing_violations == 0


def test_v6_keeps_v5_result_when_candidate_pair_is_seven_semitones_apart() -> None:
    plan, score, texture = _v6_spacing_case(
        root_pitch_class=2,
        quality="major",
        bass_degree="root",
        low_degree="fifth",
    )

    v5 = place_piano_texture_v5(
        plan, score, texture, allowed_pitch_range=(38, 86)
    )
    result = place_piano_texture_v6(
        plan, score, texture, allowed_pitch_range=(38, 86)
    )

    assert v5.status == result.status == "placed"
    assert result.notes == v5.notes
    assert result.rearticulations == v5.rearticulations
    assert result.candidate_evaluation_count == v5.candidate_evaluation_count
    assert result.backtrack_count == v5.backtrack_count
    assert result.zone_projections == v5.zone_projections
    assert result.v6_zone_projections == ()
    assert result.v6_fallback_status is None


def test_v6_does_not_project_non_overlapping_low_event() -> None:
    plan, score, texture = _v6_spacing_case(
        root_pitch_class=0,
        quality="major",
        bass_degree="fifth",
        low_degree="root",
        low_at_units=8,
    )

    result = place_piano_texture_v6(
        plan, score, texture, allowed_pitch_range=(38, 86)
    )

    assert result.v6_zone_projections == ()


def test_v6_does_not_project_when_low_candidate_set_is_empty() -> None:
    plan, score, texture = _v6_spacing_case(
        root_pitch_class=0,
        quality="major",
        bass_degree="fifth",
        low_degree="root",
    )

    result = place_piano_texture_v6(
        plan, score, texture, allowed_pitch_range=(38, 47)
    )

    assert result.status == "constraint_unplaceable"
    assert result.v6_zone_projections == ()
    assert result.v6_fallback_status is None


def test_v6_preserves_a_successful_v5_lower_foreground_projection() -> None:
    plan, score = _fixture(foreground_voice="lower")
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "root", "h1", "accompaniment", "upper", 0, 8, "root", "bass", ()
            ),
        ),
    )

    v5 = place_piano_texture_v5(plan, score, texture)
    result = place_piano_texture_v6(plan, score, texture)

    assert v5.status == result.status == "placed"
    assert result.notes == v5.notes
    assert result.zone_projections == v5.zone_projections
    assert len(result.zone_projections) == 1
    assert result.v6_zone_projections == ()
    assert result.v6_fallback_status is None


def test_v6_stops_after_one_failed_projection() -> None:
    plan, score, texture = _v6_spacing_case(
        root_pitch_class=0,
        quality="major",
        bass_degree="fifth",
        low_degree="root",
    )

    result = place_piano_texture_v6(
        plan, score, texture, allowed_pitch_range=(38, 59)
    )

    assert result.status == "constraint_unplaceable"
    assert tuple(item.event_id for item in result.v6_zone_projections) == ("low",)
    assert result.v6_fallback_status == "constraint_unplaceable"
    assert result.v6_fallback_candidate_evaluation_count == 1


def test_v7_uses_an_onset_feasible_zone_when_v6_projection_is_unplaceable() -> None:
    plan, score = _fixture(
        harmony=ScoreHarmony("h1", 0, 32, 2, "minor")
    )
    score = replace(
        score,
        materials=(
            replace(
                score.materials[0],
                notes=(ScoreNote("foreground", 0, 32, 62, "upper"),),
            ),
        ),
    )
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "fifth", "h1", "accompaniment", "lower", 0, 8, "fifth", "bass", ()
            ),
            PianoTextureEventV2(
                "root", "h1", "accompaniment", "lower", 0, 8, "root", "low", ()
            ),
            PianoTextureEventV2(
                "third", "h1", "accompaniment", "lower", 0, 8, "third", "middle", ()
            ),
        ),
    )

    v6 = place_piano_texture_v6(
        plan, score, texture, allowed_pitch_range=(41, 91)
    )
    result = place_piano_texture_v7(
        plan, score, texture, allowed_pitch_range=(41, 91)
    )

    assert v6.status == "constraint_unplaceable"
    assert isinstance(result, PianoTexturePlacementResultV7)
    assert result.status == "placed"
    assert result.v6_status == "constraint_unplaceable"
    assert result.v7_fallback_status == "placed"
    assert tuple(item.event_id for item in result.v7_zone_projections) == ("fifth",)
    assert result.v7_zone_projections[0].original_zone == "bass"
    assert result.v7_zone_projections[0].effective_zone == "low"
    assert {note.pitch for note in result.notes} == {50, 53, 57}
    assert result.low_spacing_violations == 0


def test_v7_preserves_a_successful_v6_result() -> None:
    plan, score, texture = _v6_spacing_case(
        root_pitch_class=0,
        quality="major",
        bass_degree="fifth",
        low_degree="root",
    )

    v6 = place_piano_texture_v6(
        plan, score, texture, allowed_pitch_range=(38, 86)
    )
    result = place_piano_texture_v7(
        plan, score, texture, allowed_pitch_range=(38, 86)
    )

    assert v6.status == result.status == "placed"
    assert result.notes == v6.notes
    assert result.rearticulations == v6.rearticulations
    assert result.v6_zone_projections == v6.v6_zone_projections
    assert result.v7_zone_projections == ()
    assert result.v7_fallback_status is None


def test_low_spacing_rechecks_every_sounding_change_point() -> None:
    foreground = ScoreNote("foreground", 0, 2, 50, "upper")
    fifth = ScoreNote("fifth", 0, 2, 45, "lower")
    supporting_root = ScoreNote("root", 0, 2, 38, "lower")
    short_root = ScoreNote("short-root", 0, 1, 38, "lower")

    assert low_spacing_violations((supporting_root, fifth, foreground)) == 0
    assert low_spacing_violations((fifth, foreground)) == 1
    assert low_spacing_violations((short_root, fifth, foreground)) == 1


def test_v8_retries_an_unexpanded_global_search_failure(monkeypatch) -> None:
    plan, score = _fixture()
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "root-1", "h1", "accompaniment", "lower", 0, 1, "root", "bass", ()
            ),
            PianoTextureEventV2(
                "root-2", "h1", "accompaniment", "lower", 1, 1, "root", "bass", ()
            ),
        ),
    )
    original = placement_module._valid_assignment

    def requires_expanded_first_pitch(material, events, pitches, placed):
        if events[0].at_units == 1 and placed and not any(
            note.event_id == "root-1" and note.pitch == 50 for _, note in placed
        ):
            return False
        return original(material, events, pitches, placed)

    monkeypatch.setattr(
        placement_module, "_valid_assignment", requires_expanded_first_pitch
    )

    v7 = place_piano_texture_v7(plan, score, texture)
    result = place_piano_texture_v8(plan, score, texture)

    assert v7.status == "search_unplaceable"
    assert v7.v7_fallback_status is None
    assert isinstance(result, PianoTexturePlacementResultV8)
    assert result.status == "placed"
    assert result.v7_status == "search_unplaceable"
    assert result.v8_fallback_status == "placed"
    assert tuple(item.event_id for item in result.v8_zone_projections) == ("root-1",)
    assert result.total_candidate_evaluation_count <= (
        result.maximum_total_candidate_evaluations
    )


def test_v8_does_not_repeat_a_failed_v7_expanded_search(monkeypatch) -> None:
    plan, score = _fixture(
        harmony=ScoreHarmony("h1", 0, 32, 2, "minor")
    )
    score = replace(
        score,
        materials=(
            replace(
                score.materials[0],
                notes=(ScoreNote("foreground", 0, 32, 62, "upper"),),
            ),
        ),
    )
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "fifth", "h1", "accompaniment", "lower", 0, 1, "fifth", "bass", ()
            ),
            PianoTextureEventV2(
                "root", "h1", "accompaniment", "lower", 0, 1, "root", "low", ()
            ),
            PianoTextureEventV2(
                "third", "h1", "accompaniment", "lower", 0, 1, "third", "middle", ()
            ),
            PianoTextureEventV2(
                "later", "h1", "accompaniment", "lower", 1, 1, "fifth", "bass", ()
            ),
        ),
    )
    original_assignment = placement_module._valid_assignment
    original_search = placement_module._place_piano_texture_v7_search
    calls = 0

    def rejects_later_history(material, events, pitches, placed):
        if events[0].at_units == 1 and placed:
            return False
        return original_assignment(material, events, pitches, placed)

    def counts_search(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_search(*args, **kwargs)

    monkeypatch.setattr(
        placement_module, "_valid_assignment", rejects_later_history
    )
    monkeypatch.setattr(
        placement_module, "_place_piano_texture_v7_search", counts_search
    )

    result = place_piano_texture_v8(
        plan, score, texture, allowed_pitch_range=(41, 91)
    )

    assert result.status == "search_unplaceable"
    assert result.v7_fallback_status == "search_unplaceable"
    assert result.v8_fallback_status is None
    assert calls == 1


def test_v7_backtracks_zone_and_pitch_together(monkeypatch) -> None:
    plan, score = _fixture(
        harmony=ScoreHarmony("h1", 0, 32, 2, "minor")
    )
    score = replace(
        score,
        materials=(
            replace(
                score.materials[0],
                notes=(ScoreNote("foreground", 0, 32, 62, "upper"),),
            ),
        ),
    )
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "opening-fifth", "h1", "accompaniment", "lower", 0, 1, "fifth", "bass", ()
            ),
            PianoTextureEventV2(
                "opening-root", "h1", "accompaniment", "lower", 0, 1, "root", "low", ()
            ),
            PianoTextureEventV2(
                "opening-third", "h1", "accompaniment", "lower", 0, 1, "third", "middle", ()
            ),
            PianoTextureEventV2(
                "fifth-1", "h1", "accompaniment", "lower", 1, 2, "fifth", "bass", ()
            ),
            PianoTextureEventV2(
                "fifth-2", "h1", "accompaniment", "lower", 2, 1, "fifth", "bass", ()
            ),
        ),
    )
    original = placement_module._valid_assignment

    def requires_first_fifth_pitch_57(material, events, pitches, placed):
        if (
            events[0].at_units == 2
            and placed
            and not any(
                note.event_id == "fifth-1" and note.pitch == 57
                for _, note in placed
            )
        ):
            return False
        return original(material, events, pitches, placed)

    monkeypatch.setattr(
        placement_module, "_valid_assignment", requires_first_fifth_pitch_57
    )

    result = place_piano_texture_v7(
        plan, score, texture, allowed_pitch_range=(41, 59)
    )

    assert result.status == "placed"
    assert any(
        item.event_id == "fifth-1" and item.effective_zone == "low"
        for item in result.v7_zone_projections
    )
    assert result.backtrack_count > 0


def test_v7_does_not_retry_v6_search_exhaustion() -> None:
    plan, score = _fixture(foreground_voice="lower")
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "root", "h1", "accompaniment", "upper", 0, 8, "root", "high", ()
            ),
        ),
    )

    result = place_piano_texture_v7(
        plan, score, texture, maximum_candidate_evaluations=1
    )

    assert result.status == "search_exhausted"
    assert result.v6_status == "search_exhausted"
    assert result.v7_zone_projections == ()
    assert result.v7_fallback_status is None


def test_v7_stops_after_one_failed_expanded_search(monkeypatch) -> None:
    plan, score = _fixture(
        harmony=ScoreHarmony("h1", 0, 32, 2, "minor")
    )
    score = replace(
        score,
        materials=(
            replace(
                score.materials[0],
                notes=(ScoreNote("foreground", 0, 32, 62, "upper"),),
            ),
        ),
    )
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "fifth", "h1", "accompaniment", "lower", 0, 1, "fifth", "bass", ()
            ),
            PianoTextureEventV2(
                "root", "h1", "accompaniment", "lower", 0, 1, "root", "low", ()
            ),
            PianoTextureEventV2(
                "third", "h1", "accompaniment", "lower", 0, 1, "third", "middle", ()
            ),
            PianoTextureEventV2(
                "later", "h1", "accompaniment", "lower", 1, 1, "fifth", "bass", ()
            ),
        ),
    )
    original = placement_module._valid_assignment

    def rejects_later_history(material, events, pitches, placed):
        if events[0].at_units == 1 and placed:
            return False
        return original(material, events, pitches, placed)

    monkeypatch.setattr(
        placement_module, "_valid_assignment", rejects_later_history
    )

    result = place_piano_texture_v7(
        plan, score, texture, allowed_pitch_range=(41, 91)
    )

    assert result.status == "search_unplaceable"
    assert result.v6_status == "constraint_unplaceable"
    assert result.v7_fallback_status == "search_unplaceable"
    assert result.v7_zone_projections == ()


def test_v6_does_not_retry_search_exhaustion() -> None:
    plan, score = _fixture(foreground_voice="lower")
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "root", "h1", "accompaniment", "upper", 0, 8, "root", "high", ()
            ),
        ),
    )

    v5 = place_piano_texture_v5(
        plan, score, texture, maximum_candidate_evaluations=1
    )
    result = place_piano_texture_v6(
        plan, score, texture, maximum_candidate_evaluations=1
    )

    assert v5.status == result.status == "search_exhausted"
    assert result.v6_zone_projections == ()
    assert result.v6_fallback_status is None


def test_v4_reports_fully_searched_global_failure(monkeypatch) -> None:
    plan, score = _fixture()
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "root-1", "h1", "accompaniment", "lower", 0, 1, "root", "low", ()
            ),
            PianoTextureEventV2(
                "root-2", "h1", "accompaniment", "lower", 1, 1, "root", "low", ()
            ),
        ),
    )
    original = placement_module._valid_assignment

    def no_second_history(material, events, pitches, placed):
        if events[0].at_units == 1 and placed:
            return False
        return original(material, events, pitches, placed)

    monkeypatch.setattr(placement_module, "_valid_assignment", no_second_history)

    result = place_piano_texture_v4(plan, score, texture)

    assert result.status == "search_unplaceable"
    assert result.candidate_evaluation_count < 100_000


def test_projection_rekeys_events_and_combination_preserves_fixed_foreground() -> None:
    plan, score = _fixture()

    projected = project_accompaniment_to_v2(plan, score, "body")
    placed = place_piano_texture_v2(plan, score, projected.texture)
    combined = combine_piano_texture_v2(plan, score, placed)

    assert projected.status == "assessed"
    assert projected.event_id_map == {
        "old-fifth": "projected-old-fifth",
        "old-root": "projected-old-root",
    }
    assert placed.status == "placed"
    material = combined.materials[0]
    assert tuple(note for note in material.notes if note.voice == "upper") == tuple(
        note for note in score.materials[0].notes if note.voice == "upper"
    )
    assert {note.event_id for note in material.notes if note.voice == "lower"} == {
        "projected-old-fifth",
        "projected-old-root",
    }


def test_seventh_is_allowed_only_for_major_seventh() -> None:
    plan, score = _fixture()
    texture = PianoTextureSpecV2(
        "body",
        (
            PianoTextureEventV2(
                "seventh", "h1", "accompaniment", "lower", 0, 8, "seventh", "low", ()
            ),
        ),
    )

    with pytest.raises(PianoTextureValidationError, match="degree"):
        place_piano_texture_v2(plan, score, texture)

    major_seventh_score = replace(
        score,
        materials=(
            replace(
                score.materials[0],
                harmonies=(ScoreHarmony("h1", 0, 32, 2, "major-seventh"),),
            ),
        ),
    )
    result = place_piano_texture_v2(plan, major_seventh_score, texture)
    assert result.status == "placed"
