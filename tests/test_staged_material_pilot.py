from __future__ import annotations

import pytest

from llm_musical_composer.performance_pipeline import (
    PiecePlan,
    PlanNode,
    ScoreDirection,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
)
from llm_musical_composer.staged_material_pilot import (
    HarmonicDraft,
    HarmonicEventDraft,
    MelodyDraft,
    MelodyEventDraft,
    StagedMaterialError,
    TextureDraft,
    TextureEventDraft,
    assemble_harmonies,
    assemble_melody,
    assemble_texture,
    build_fixed_context,
    build_texture_feasibility,
    dump_harmonic_draft,
    dump_melody_draft,
    dump_texture_draft,
    harmony_start_accompaniment_policy,
    parse_harmonic_draft,
    parse_melody_draft,
    parse_texture_draft,
    texture_budget_onset_capacity_reachability,
    texture_event_feasibility_violations,
    texture_feasibility_violations,
    texture_onset_capacity_violations,
    texture_preplacement_violations,
)


def _fixture() -> tuple[PiecePlan, ScoreSpec, ScoreMaterial]:
    plan = PiecePlan(
        "plan",
        "段階素材",
        0,
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
    target = ScoreMaterial(
        "body",
        16,
        (),
        directions=(ScoreDirection("d1", 0, "dynamic", "mp"),),
    )
    return plan, ScoreSpec("score", 8, (target,)), target


@pytest.mark.parametrize(
    ("value", "dump", "parse"),
    (
        (
            HarmonicDraft(
                (HarmonicEventDraft(0, 8, 0, "major"), HarmonicEventDraft(8, 8, 7, "major"))
            ),
            dump_harmonic_draft,
            parse_harmonic_draft,
        ),
        (
            MelodyDraft(
                "upper",
                (
                    MelodyEventDraft(0, 4, 60, ()),
                    MelodyEventDraft(4, 4, 62, ("tenuto",)),
                    MelodyEventDraft(8, 4, 67, ()),
                    MelodyEventDraft(12, 4, 64, ()),
                ),
            ),
            dump_melody_draft,
            parse_melody_draft,
        ),
        (
            TextureDraft(
                (
                    TextureEventDraft(0, 0, 8, "root", "low", ()),
                    TextureEventDraft(0, 0, 8, "fifth", "low", ()),
                    TextureEventDraft(1, 8, 8, "root", "low", ()),
                    TextureEventDraft(1, 8, 8, "fifth", "low", ()),
                )
            ),
            dump_texture_draft,
            parse_texture_draft,
        ),
    ),
)
def test_thin_drafts_round_trip_without_identifiers(
    value: object, dump: object, parse: object
) -> None:
    source = dump(value)
    assert parse(source) == value
    assert "event_id" not in source
    assert "harmony_id" not in source
    assert "material_id" not in source


def test_three_stages_assign_ids_and_build_a_material() -> None:
    plan, score, target = _fixture()
    harmonies = assemble_harmonies(
        "case",
        target,
        HarmonicDraft((HarmonicEventDraft(0, 8, 0, "major"), HarmonicEventDraft(8, 8, 7, "major"))),
        score,
    )
    melody = assemble_melody(
        "case",
        score.divisions,
        target,
        harmonies,
        MelodyDraft(
            "upper",
            (
                MelodyEventDraft(0, 4, 60, ()),
                MelodyEventDraft(4, 4, 62, ()),
                MelodyEventDraft(8, 4, 67, ()),
                MelodyEventDraft(12, 4, 71, ()),
            ),
        ),
        score,
    )
    result, material = assemble_texture(
        "case",
        plan,
        score,
        target,
        harmonies,
        melody,
        TextureDraft(
            (
                TextureEventDraft(0, 0, 8, "root", "low", ()),
                TextureEventDraft(0, 0, 8, "fifth", "low", ()),
                TextureEventDraft(1, 8, 8, "root", "low", ()),
                TextureEventDraft(1, 8, 8, "fifth", "low", ()),
            )
        ),
    )

    assert result.status == "placed"
    assert {harmony.harmony_id for harmony in material.harmonies} == {
        "pilot-case-h-001",
        "pilot-case-h-002",
    }
    assert all(note.event_id.startswith("pilot-case-") for note in material.notes)
    assert material.directions == target.directions


def test_texture_assembly_threads_the_shared_allowed_pitch_range_to_placement() -> None:
    plan, score, target = _fixture()
    harmonies = assemble_harmonies(
        "case",
        target,
        HarmonicDraft(
            (
                HarmonicEventDraft(0, 8, 0, "major"),
                HarmonicEventDraft(8, 8, 7, "major"),
            )
        ),
        score,
    )
    melody = (ScoreNote("melody", 0, 16, 72, "upper"),)
    draft = TextureDraft(
        (
            TextureEventDraft(0, 0, 8, "root", "low", ()),
            TextureEventDraft(0, 0, 8, "fifth", "low", ()),
            TextureEventDraft(1, 8, 8, "root", "low", ()),
            TextureEventDraft(1, 8, 8, "fifth", "low", ()),
        )
    )

    result, material = assemble_texture(
        "bounded",
        plan,
        score,
        target,
        harmonies,
        melody,
        draft,
        allowed_pitch_range=(51, 53),
    )

    assert result.status == "constraint_unplaceable"
    assert material.notes == melody


def test_texture_can_explicitly_use_v3_same_key_rearticulation() -> None:
    plan, score, target = _fixture()
    harmonies = assemble_harmonies(
        "case",
        target,
        HarmonicDraft(
            (
                HarmonicEventDraft(0, 12, 0, "major"),
                HarmonicEventDraft(12, 4, 7, "major"),
            )
        ),
        score,
    )
    melody = (
        ScoreNote("melody-1", 0, 16, 72, "upper"),
    )
    draft = TextureDraft(
        (
            TextureEventDraft(0, 0, 8, "root", "bass", ()),
            TextureEventDraft(0, 0, 3, "fifth", "low", ()),
            TextureEventDraft(0, 3, 6, "root", "bass", ()),
            TextureEventDraft(0, 6, 6, "root", "bass", ()),
            TextureEventDraft(1, 12, 4, "root", "bass", ()),
            TextureEventDraft(1, 12, 4, "fifth", "low", ()),
        )
    )

    strict, _ = assemble_texture(
        "strict",
        plan,
        score,
        target,
        harmonies,
        melody,
        draft,
    )
    result, material = assemble_texture(
        "rearticulated",
        plan,
        score,
        target,
        harmonies,
        melody,
        draft,
        placement_policy="same-key-rearticulation-v3",
    )

    assert strict.status == "greedy_unplaceable"
    assert result.status == "placed"
    assert len(result.rearticulations) == 1
    assert len(material.notes) == 7


def test_texture_can_explicitly_use_v4_bounded_backtracking() -> None:
    plan, score, target = _fixture()
    harmonies = assemble_harmonies(
        "case",
        target,
        HarmonicDraft(
            (
                HarmonicEventDraft(0, 12, 0, "major"),
                HarmonicEventDraft(12, 4, 7, "major"),
            )
        ),
        score,
    )
    melody = (ScoreNote("melody-1", 0, 16, 72, "upper"),)
    draft = TextureDraft(
        (
            TextureEventDraft(0, 0, 8, "root", "bass", ()),
            TextureEventDraft(0, 0, 3, "fifth", "low", ()),
            TextureEventDraft(0, 3, 6, "root", "bass", ()),
            TextureEventDraft(0, 6, 6, "root", "bass", ()),
            TextureEventDraft(1, 12, 4, "root", "bass", ()),
            TextureEventDraft(1, 12, 4, "fifth", "low", ()),
        )
    )

    result, material = assemble_texture(
        "backtracking",
        plan,
        score,
        target,
        harmonies,
        melody,
        draft,
        placement_policy="bounded-backtracking-v4",
    )

    assert result.status == "placed"
    assert result.candidate_evaluation_count > 0
    assert len(material.notes) == 7


def test_texture_can_explicitly_use_v6_candidate_spacing_projection() -> None:
    plan, score, target = _fixture()
    harmonies = assemble_harmonies(
        "case",
        target,
        HarmonicDraft(
            (
                HarmonicEventDraft(0, 8, 0, "major"),
                HarmonicEventDraft(8, 8, 0, "major"),
            )
        ),
        score,
    )
    melody = (ScoreNote("melody-1", 0, 16, 72, "upper"),)
    draft = TextureDraft(
        (
            TextureEventDraft(0, 0, 8, "fifth", "bass", ()),
            TextureEventDraft(0, 0, 8, "root", "low", ()),
            TextureEventDraft(1, 8, 8, "fifth", "bass", ()),
        )
    )

    result, material = assemble_texture(
        "v6-spacing",
        plan,
        score,
        target,
        harmonies,
        melody,
        draft,
        placement_policy="bidirectional-low-spacing-v6",
        allowed_pitch_range=(38, 86),
    )

    assert result.status == "placed"
    assert tuple(item.event_id for item in result.v6_zone_projections) == (
        "pilot-v6-spacing-a-002",
    )
    assert result.v6_fallback_status == "placed"
    assert len(material.notes) == 4


def test_texture_can_explicitly_use_v7_onset_feasible_zone_search() -> None:
    plan, score, target = _fixture()
    harmonies = assemble_harmonies(
        "case",
        target,
        HarmonicDraft(
            (
                HarmonicEventDraft(0, 8, 2, "minor"),
                HarmonicEventDraft(8, 8, 2, "minor"),
            )
        ),
        score,
    )
    melody = (ScoreNote("melody-1", 0, 16, 62, "upper"),)
    draft = TextureDraft(
        (
            TextureEventDraft(0, 0, 8, "fifth", "bass", ()),
            TextureEventDraft(0, 0, 8, "root", "low", ()),
            TextureEventDraft(0, 0, 8, "third", "middle", ()),
            TextureEventDraft(1, 8, 8, "fifth", "bass", ()),
        )
    )

    result, material = assemble_texture(
        "v7-onset",
        plan,
        score,
        target,
        harmonies,
        melody,
        draft,
        placement_policy="onset-feasible-zone-v7",
        allowed_pitch_range=(41, 91),
    )

    assert result.status == "placed"
    assert tuple(item.event_id for item in result.v7_zone_projections) == (
        "pilot-v7-onset-a-001",
    )
    assert len(material.notes) == 5

    v8_result, v8_material = assemble_texture(
        "v8-onset",
        plan,
        score,
        target,
        harmonies,
        melody,
        draft,
        placement_policy="search-aware-onset-zone-v8",
        allowed_pitch_range=(41, 91),
    )

    assert v8_result.status == "placed"
    assert v8_result.v7_status == "placed"
    assert v8_result.v8_fallback_status is None
    assert len(v8_material.notes) == 5


def test_melody_rejects_a_lower_register_with_no_safe_opposite_voice() -> None:
    _, score, target = _fixture()
    harmonies = assemble_harmonies(
        "case",
        target,
        HarmonicDraft((HarmonicEventDraft(0, 8, 0, "major"), HarmonicEventDraft(8, 8, 0, "major"))),
        score,
    )
    draft = MelodyDraft(
        "lower",
        (
            MelodyEventDraft(0, 4, 105, ()),
            MelodyEventDraft(4, 4, 106, ()),
            MelodyEventDraft(8, 4, 107, ()),
            MelodyEventDraft(12, 4, 108, ()),
        ),
    )

    with pytest.raises(StagedMaterialError, match="opposite voice"):
        assemble_melody("case", score.divisions, target, harmonies, draft, score)


def test_texture_rejects_single_note_attacks_without_a_varied_arpeggio() -> None:
    plan, score, target = _fixture()
    harmonies = assemble_harmonies(
        "case",
        target,
        HarmonicDraft((HarmonicEventDraft(0, 8, 0, "major"), HarmonicEventDraft(8, 8, 7, "major"))),
        score,
    )
    melody = assemble_melody(
        "case",
        score.divisions,
        target,
        harmonies,
        MelodyDraft(
            "upper",
            (
                MelodyEventDraft(0, 4, 60),
                MelodyEventDraft(4, 4, 62),
                MelodyEventDraft(8, 4, 67),
                MelodyEventDraft(12, 4, 71),
            ),
        ),
        score,
    )

    with pytest.raises(StagedMaterialError, match="chord attack or a varied arpeggio"):
        assemble_texture(
            "case",
            plan,
            score,
            target,
            harmonies,
            melody,
            TextureDraft(
                (
                    TextureEventDraft(0, 0, 8, "root", "low"),
                    TextureEventDraft(1, 8, 8, "root", "low"),
                )
            ),
        )


def test_texture_accepts_a_budget_derived_event_limit() -> None:
    plan, score, target = _fixture()
    harmonies = assemble_harmonies(
        "case",
        target,
        HarmonicDraft(
            (
                HarmonicEventDraft(0, 8, 0, "major"),
                HarmonicEventDraft(8, 8, 7, "major"),
            )
        ),
        score,
    )
    melody = assemble_melody(
        "case",
        score.divisions,
        target,
        harmonies,
        MelodyDraft(
            "upper",
            tuple(
                MelodyEventDraft(onset, 1, pitch)
                for onset, pitch in ((0, 60), (4, 62), (8, 67), (12, 71))
            ),
        ),
        score,
    )
    draft = TextureDraft(
        tuple(
            TextureEventDraft(
                0 if onset < 8 else 1,
                onset,
                1,
                "root" if onset % 2 == 0 else "fifth",
                "low",
            )
            for onset in range(13)
        )
    )

    with pytest.raises(StagedMaterialError, match="event count"):
        assemble_texture("case", plan, score, target, harmonies, melody, draft)

    placement, _ = assemble_texture(
        "case",
        plan,
        score,
        target,
        harmonies,
        melody,
        draft,
        maximum_event_count=13,
    )
    assert placement.status == "placed"


def test_lower_foreground_uses_the_highest_pitch_for_individual_zones() -> None:
    _, score, target = _fixture()
    harmonies = assemble_harmonies(
        "case",
        target,
        HarmonicDraft(
            (HarmonicEventDraft(0, 8, 0, "major"), HarmonicEventDraft(8, 8, 7, "major"))
        ),
        score,
    )
    melody = (
        ScoreNote("m1", 0, 4, 60, "lower"),
        ScoreNote("m2", 4, 4, 64, "lower"),
        ScoreNote("m3", 8, 4, 67, "lower"),
    )

    result = build_texture_feasibility(harmonies, melody)

    first = result["harmonies"][0]
    assert first["overlapping_foreground_pitch_range"] == [60, 64]
    assert "middle" not in first["individually_feasible_register_zones"]["root"]
    assert "high" in first["individually_feasible_register_zones"]["root"]
    assert "seventh" not in first["individually_feasible_register_zones"]


def test_upper_foreground_reports_individual_zones_not_joint_feasibility() -> None:
    plan, score, target = _fixture()
    harmonies = assemble_harmonies(
        "case",
        target,
        HarmonicDraft(
            (HarmonicEventDraft(0, 8, 0, "major"), HarmonicEventDraft(8, 8, 0, "major"))
        ),
        score,
    )
    melody = (
        ScoreNote("m1", 0, 8, 52, "upper"),
        ScoreNote("m2", 8, 8, 60, "upper"),
    )

    result = build_texture_feasibility(harmonies, melody)

    first = result["harmonies"][0]["individually_feasible_register_zones"]
    assert "middle" in first["root"]
    assert "low" in first["fifth"]
    assert result["advisory_scope"] == "whole_harmony_universal_safety_advisory"

    placement, _ = assemble_texture(
        "case",
        plan,
        score,
        target,
        harmonies,
        melody,
        TextureDraft(
            (
                TextureEventDraft(0, 0, 8, "root", "middle"),
                TextureEventDraft(0, 0, 8, "fifth", "low"),
                TextureEventDraft(1, 8, 8, "root", "middle"),
                TextureEventDraft(1, 8, 8, "fifth", "low"),
            )
        ),
    )
    assert placement.status == "constraint_unplaceable"
    assert placement.failed_onset == 0


def test_individual_zone_low_spacing_uses_the_seven_semitone_boundary() -> None:
    _, score, target = _fixture()
    harmonies = assemble_harmonies(
        "case",
        target,
        HarmonicDraft(
            (HarmonicEventDraft(0, 8, 7, "major"), HarmonicEventDraft(8, 8, 7, "major"))
        ),
        score,
    )

    six = build_texture_feasibility(
        harmonies, (ScoreNote("m1", 0, 16, 49, "upper"),)
    )
    seven = build_texture_feasibility(
        harmonies, (ScoreNote("m1", 0, 16, 50, "upper"),)
    )

    assert "low" not in six["harmonies"][0]["individually_feasible_register_zones"][
        "root"
    ]
    assert "low" in seven["harmonies"][0]["individually_feasible_register_zones"][
        "root"
    ]


def test_feasibility_includes_a_foreground_note_crossing_the_harmony_boundary() -> None:
    _, score, target = _fixture()
    harmonies = assemble_harmonies(
        "case",
        target,
        HarmonicDraft(
            (HarmonicEventDraft(0, 8, 0, "major"), HarmonicEventDraft(8, 8, 7, "major"))
        ),
        score,
    )
    melody = (ScoreNote("m1", 6, 4, 72, "lower"),)

    result = build_texture_feasibility(harmonies, melody)

    second = result["harmonies"][1]
    assert second["overlapping_foreground_pitch_range"] == [72, 72]
    assert "middle" not in second["individually_feasible_register_zones"]["root"]
    assert "high" in second["individually_feasible_register_zones"]["root"]


def test_texture_feasibility_lists_declared_zone_violations() -> None:
    _, score, target = _fixture()
    harmonies = assemble_harmonies(
        "case",
        target,
        HarmonicDraft(
            (HarmonicEventDraft(0, 8, 0, "major"), HarmonicEventDraft(8, 8, 7, "major"))
        ),
        score,
    )
    feasibility = build_texture_feasibility(
        harmonies, (ScoreNote("m1", 0, 16, 60, "lower"),)
    )
    draft = TextureDraft((TextureEventDraft(0, 0, 8, "root", "middle"),))

    violations = texture_feasibility_violations(draft, feasibility)

    assert violations == [
        {
            "event_index": 0,
            "harmony_index": 0,
            "degree": "root",
            "register_zone": "middle",
        }
    ]


def test_event_feasibility_does_not_reject_a_short_valid_event_from_advisory() -> None:
    harmonies = (ScoreHarmony("h1", 0, 16, 0, "major"),)
    melody = (
        ScoreNote("m1", 0, 8, 41, "lower"),
        ScoreNote("m2", 8, 8, 65, "lower"),
    )
    draft = TextureDraft((TextureEventDraft(0, 0, 8, "root", "middle"),))
    advisory = build_texture_feasibility(
        harmonies, melody, allowed_pitch_range=(38, 86)
    )

    assert "middle" not in advisory["harmonies"][0][
        "individually_feasible_register_zones"
    ]["root"]
    assert texture_feasibility_violations(draft, advisory)
    assert texture_event_feasibility_violations(
        draft,
        harmonies,
        melody,
        allowed_pitch_range=(38, 86),
    ) == []


def test_v23_shape_limits_new_accompaniment_attacks_to_two() -> None:
    harmonies = (ScoreHarmony("h1", 18, 12, 11, "minor"),)
    melody = (ScoreNote("m1", 18, 3, 59, "upper"),)

    feasibility = build_texture_feasibility(
        harmonies, melody, allowed_pitch_range=(43, 93)
    )

    assert feasibility["schema_version"] == 3
    assert feasibility["scope"] == "texture_generation_feasibility_v3"
    assert feasibility["onset_capacities"] == [
        {
            "harmony_index": 0,
            "harmony_id": "h1",
            "start_units": 18,
            "end_units": 21,
            "sounding_foreground_pitches": [59],
            "maximum_new_accompaniment_attack_count": 2,
            "allowed_pitch_range": [43, 93],
            "constraint_id": "new_accompaniment_onset_capacity_v1",
        },
        {
            "harmony_index": 0,
            "harmony_id": "h1",
            "start_units": 21,
            "end_units": 30,
            "sounding_foreground_pitches": [],
            "maximum_new_accompaniment_attack_count": 11,
            "allowed_pitch_range": [43, 93],
            "constraint_id": "new_accompaniment_onset_capacity_v1",
        },
    ]


def test_harmony_start_policy_exempts_only_capacity_zero_chord_tone_melody() -> None:
    harmonies = (ScoreHarmony("h1", 0, 8, 4, "diminished"),)
    capacity_zero = {
        "onset_capacities": [
            {
                "harmony_index": 0,
                "start_units": 0,
                "end_units": 8,
                "maximum_new_accompaniment_attack_count": 0,
            }
        ]
    }

    exempt = harmony_start_accompaniment_policy(
        harmonies,
        (ScoreNote("root", 0, 2, 52, "upper"),),
        capacity_zero,
    )
    non_chord = harmony_start_accompaniment_policy(
        harmonies,
        (ScoreNote("outside", 0, 2, 53, "upper"),),
        capacity_zero,
    )
    mixed = harmony_start_accompaniment_policy(
        harmonies,
        (
            ScoreNote("root", 0, 2, 52, "upper"),
            ScoreNote("outside", 0, 2, 53, "upper"),
        ),
        capacity_zero,
    )
    capacity_one = harmony_start_accompaniment_policy(
        harmonies,
        (ScoreNote("root", 0, 2, 52, "upper"),),
        {
            "onset_capacities": [
                {
                    "harmony_index": 0,
                    "start_units": 0,
                    "end_units": 8,
                    "maximum_new_accompaniment_attack_count": 1,
                }
            ]
        },
    )

    assert exempt["minimum_new_accompaniment_attacks"] == {"0": 0}
    assert exempt["melody_led_starts"] == [
        {
            "harmony_index": 0,
            "harmony_id": "h1",
            "at_units": 0,
            "maximum_new_accompaniment_attack_count": 0,
            "melody_pitches": [52],
            "chord_pitch_classes": [4, 7, 10],
            "reason": "capacity_zero_and_melody_attacks_chord_tones",
        }
    ]
    assert non_chord["minimum_new_accompaniment_attacks"] == {"0": 1}
    assert mixed["minimum_new_accompaniment_attacks"] == {"0": 1}
    assert capacity_one["minimum_new_accompaniment_attacks"] == {"0": 1}


def test_feasibility_schema_three_embeds_harmony_start_policy() -> None:
    harmonies = (ScoreHarmony("h1", 0, 8, 4, "diminished"),)
    melody = (ScoreNote("root", 0, 2, 52, "upper"),)

    feasibility = build_texture_feasibility(
        harmonies, melody, allowed_pitch_range=(46, 96)
    )

    assert feasibility["schema_version"] == 3
    assert feasibility["scope"] == "texture_generation_feasibility_v3"
    assert feasibility["onset_capacities"][0][
        "maximum_new_accompaniment_attack_count"
    ] == 0
    assert feasibility["harmony_start_accompaniment"][
        "minimum_new_accompaniment_attacks"
    ] == {"0": 0}


def test_texture_can_enter_after_melody_led_harmony_start() -> None:
    plan, score, target = _fixture()
    harmonies = (ScoreHarmony("h1", 0, 16, 4, "diminished"),)
    melody = (ScoreNote("root", 0, 2, 52, "upper"),)
    draft = TextureDraft(
        (
            TextureEventDraft(0, 2, 6, "root", "low"),
            TextureEventDraft(0, 2, 6, "fifth", "low"),
        )
    )

    placement, _ = assemble_texture(
        "melody-led",
        plan,
        score,
        target,
        harmonies,
        melody,
        draft,
        allowed_pitch_range=(46, 96),
    )

    assert placement.status == "placed"


def test_onset_capacity_rejects_three_new_attacks_but_allows_two() -> None:
    harmonies = (ScoreHarmony("h1", 18, 12, 11, "minor"),)
    melody = (ScoreNote("m1", 18, 3, 59, "upper"),)
    feasibility = build_texture_feasibility(
        harmonies, melody, allowed_pitch_range=(43, 93)
    )
    three = TextureDraft(
        (
            TextureEventDraft(0, 18, 3, "root", "bass"),
            TextureEventDraft(0, 18, 3, "third", "low"),
            TextureEventDraft(0, 18, 3, "fifth", "low"),
        )
    )
    two = TextureDraft(three.events[:2])

    assert texture_onset_capacity_violations(three, feasibility) == [
        {
            "harmony_index": 0,
            "at_units": 18,
            "actual_new_accompaniment_attack_count": 3,
            "maximum_new_accompaniment_attack_count": 2,
            "reason": "onset_capacity_exceeded",
        }
    ]
    assert texture_onset_capacity_violations(two, feasibility) == []


def test_onset_capacity_is_not_a_joint_degree_zone_sufficiency_claim() -> None:
    harmonies = (ScoreHarmony("h1", 18, 12, 11, "minor"),)
    melody = (ScoreNote("m1", 18, 3, 59, "upper"),)
    draft = TextureDraft(
        (
            TextureEventDraft(0, 18, 3, "root", "bass"),
            TextureEventDraft(0, 18, 3, "third", "low"),
        )
    )
    feasibility = build_texture_feasibility(
        harmonies, melody, allowed_pitch_range=(43, 93)
    )

    assert texture_event_feasibility_violations(
        draft, harmonies, melody, allowed_pitch_range=(43, 93)
    ) == []
    assert texture_onset_capacity_violations(draft, feasibility) == []


def test_preplacement_allows_joint_release_spacing_completed_by_root() -> None:
    harmonies = (ScoreHarmony("h1", 0, 6, 2, "major"),)
    melody = (ScoreNote("m1", 0, 6, 50, "upper"),)
    draft = TextureDraft(
        (
            TextureEventDraft(0, 0, 6, "root", "bass"),
            TextureEventDraft(0, 0, 6, "fifth", "low"),
        )
    )

    violations = texture_preplacement_violations(
        draft,
        harmonies,
        melody,
        allowed_pitch_range=(38, 45),
    )

    assert violations == {"event_feasibility": [], "onset_capacity": []}


def test_onset_capacity_does_not_overconstrain_high_or_lower_foreground() -> None:
    harmony = (ScoreHarmony("h1", 0, 8, 11, "minor"),)
    high = build_texture_feasibility(
        harmony,
        (ScoreNote("m1", 0, 8, 72, "upper"),),
        allowed_pitch_range=(43, 93),
    )
    lower = build_texture_feasibility(
        harmony,
        (ScoreNote("m2", 0, 8, 48, "lower"),),
        allowed_pitch_range=(43, 93),
    )

    assert high["onset_capacities"][0][
        "maximum_new_accompaniment_attack_count"
    ] >= 3
    assert lower["onset_capacities"][0][
        "maximum_new_accompaniment_attack_count"
    ] >= 3


def test_onset_capacity_counts_new_attacks_not_held_accompaniment() -> None:
    harmonies = (ScoreHarmony("h1", 0, 30, 11, "minor"),)
    melody = (ScoreNote("m1", 18, 3, 59, "upper"),)
    feasibility = build_texture_feasibility(
        harmonies, melody, allowed_pitch_range=(43, 93)
    )
    draft = TextureDraft(
        (
            TextureEventDraft(0, 0, 21, "root", "bass"),
            TextureEventDraft(0, 18, 3, "third", "low"),
            TextureEventDraft(0, 18, 3, "fifth", "low"),
        )
    )

    assert texture_onset_capacity_violations(draft, feasibility) == []


def test_texture_budget_reachability_respects_local_onset_capacity() -> None:
    melody = (
        ScoreNote("m1", 0, 1, 60, "upper"),
        ScoreNote("m2", 1, 1, 62, "upper"),
    )
    budget = {
        "combined_attack_group_count": 2,
        "attack_size_counts": {
            "one": 1,
            "two": 1,
            "three": 0,
            "four_or_more": 0,
        },
    }
    blocked = {
        "onset_capacities": [
            {
                "start_units": 0,
                "end_units": 2,
                "maximum_new_accompaniment_attack_count": 0,
            }
        ]
    }
    allowed = {
        "onset_capacities": [
            {
                "start_units": 0,
                "end_units": 2,
                "maximum_new_accompaniment_attack_count": 1,
            }
        ]
    }

    assert not texture_budget_onset_capacity_reachability(
        2, melody, budget, blocked
    )["reachable"]
    assert texture_budget_onset_capacity_reachability(
        2, melody, budget, allowed
    )["reachable"]


def test_texture_budget_reachability_respects_required_minimum_additions() -> None:
    melody = (ScoreNote("m1", 0, 1, 62, "upper"),)
    budget = {
        "combined_attack_group_count": 1,
        "attack_size_counts": {
            "one": 0,
            "two": 0,
            "three": 1,
            "four_or_more": 0,
        },
    }
    feasibility = {
        "onset_capacities": [
            {
                "start_units": 0,
                "end_units": 1,
                "maximum_new_accompaniment_attack_count": 2,
            }
        ]
    }

    assert not texture_budget_onset_capacity_reachability(
        1,
        melody,
        budget,
        feasibility,
        minimum_new_accompaniment_attacks={0: 3},
    )["reachable"]
    result = texture_budget_onset_capacity_reachability(
        1,
        melody,
        budget,
        feasibility,
        minimum_new_accompaniment_attacks={0: 2},
    )

    assert result["reachable"]
    assert result["minimum_new_accompaniment_attacks"] == {"0": 2}


def test_texture_budget_reachability_rejects_groups_only_assignable_after_ending() -> None:
    melody = (
        ScoreNote("m1", 0, 1, 60, "upper"),
        ScoreNote("m2", 0, 1, 64, "upper"),
        ScoreNote("m3", 1, 1, 62, "upper"),
        ScoreNote("m4", 1, 1, 65, "upper"),
        ScoreNote("m5", 2, 2, 60, "upper"),
    )
    budget = {
        "combined_attack_group_count": 4,
        "attack_size_counts": {
            "one": 1,
            "two": 2,
            "three": 1,
            "four_or_more": 0,
        },
    }
    feasibility = {
        "onset_capacities": [
            {
                "start_units": 0,
                "end_units": 4,
                "maximum_new_accompaniment_attack_count": 2,
            }
        ]
    }

    assert texture_budget_onset_capacity_reachability(
        4,
        melody,
        budget,
        feasibility,
        minimum_new_accompaniment_attacks={2: 2},
    )["reachable"]
    result = texture_budget_onset_capacity_reachability(
        4,
        melody,
        budget,
        feasibility,
        minimum_new_accompaniment_attacks={2: 2},
        last_usable_attack_units=2,
    )

    assert not result["reachable"]
    assert result["reason"] == "attack_size_buckets_unassignable"
    assert result["last_usable_attack_units"] == 2


@pytest.mark.parametrize("last_usable_attack_units", [-1, 4])
def test_texture_budget_reachability_rejects_invalid_last_usable_attack(
    last_usable_attack_units: int,
) -> None:
    budget = {
        "combined_attack_group_count": 0,
        "attack_size_counts": {
            "one": 0,
            "two": 0,
            "three": 0,
            "four_or_more": 0,
        },
    }
    feasibility = {
        "onset_capacities": [
            {
                "start_units": 0,
                "end_units": 4,
                "maximum_new_accompaniment_attack_count": 0,
            }
        ]
    }

    with pytest.raises(StagedMaterialError, match="last usable attack"):
        texture_budget_onset_capacity_reachability(
            4,
            (),
            budget,
            feasibility,
            last_usable_attack_units=last_usable_attack_units,
        )


def test_texture_budget_minimum_does_not_reject_four_or_more_group() -> None:
    melody = (ScoreNote("m1", 0, 1, 62, "upper"),)
    budget = {
        "combined_attack_group_count": 1,
        "attack_size_counts": {
            "one": 0,
            "two": 0,
            "three": 0,
            "four_or_more": 1,
        },
    }
    feasibility = {
        "onset_capacities": [
            {
                "start_units": 0,
                "end_units": 1,
                "maximum_new_accompaniment_attack_count": 3,
            }
        ]
    }

    assert texture_budget_onset_capacity_reachability(
        1,
        melody,
        budget,
        feasibility,
        minimum_new_accompaniment_attacks={0: 2},
    )["reachable"]


def test_fixed_context_does_not_depend_on_old_target_music() -> None:
    plan, score, target = _fixture()
    changed_target = ScoreMaterial(
        target.material_id,
        target.length_units,
        (),
        derived_from=target.derived_from,
        directions=target.directions,
    )
    changed = ScoreSpec(score.score_id, score.divisions, (changed_target,))

    assert build_fixed_context("case", plan, score, target.material_id) == build_fixed_context(
        "case", plan, changed, target.material_id
    )


def test_fixed_context_rejects_a_target_with_fixed_derivative_children() -> None:
    plan, score, target = _fixture()
    child = ScoreMaterial("child", 16, (), derived_from=target.material_id)
    expanded_score = ScoreSpec(score.score_id, score.divisions, (target, child))
    expanded_plan = PiecePlan(
        plan.plan_id,
        plan.title,
        plan.tonal_center,
        plan.mode,
        plan.root_node_id,
        plan.ending_intent,
        (
            *plan.nodes,
            PlanNode(
                "child",
                "whole",
                1,
                "variation",
                duration_weight=1,
                score_material_id="child",
            ),
        ),
    )

    with pytest.raises(StagedMaterialError, match="derivative children"):
        build_fixed_context("case", expanded_plan, expanded_score, target.material_id)
