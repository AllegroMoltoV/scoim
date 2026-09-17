from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import replace

import mido
import pytest

from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    PiecePlan,
    PipelineValidationError,
    PlanNode,
    ScoreDirection,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    analyze_voice_velocity,
    render_musicxml,
    render_performance,
    render_performance_smf,
    validate_musicxml_round_trip,
    validate_performance_spec,
    validate_piece_plan,
    validate_pipeline,
    validate_score_materials,
    validate_score_spec,
    validate_smf_round_trip,
)
from llm_musical_composer.recurrence_analysis import analyze_recurrences
from llm_musical_composer.recurrence_quality import (
    RecurrenceQualityError,
    analyze_boundary_breath,
    analyze_local_pulse,
    analyze_material_harmony,
    analyze_material_vertical_alignment,
    analyze_material_voice_texture,
    analyze_piano_texture_variety,
    analyze_rendered_harmony,
    analyze_section_contrast,
    analyze_section_pacing,
    analyze_vertical_alignment,
)


def _fixture() -> tuple[PiecePlan, ScoreSpec, PerformanceSpec]:
    plan = PiecePlan(
        plan_id="two-renditions",
        title="同じ主題の異なる演奏",
        tonal_center=9,
        mode="minor",
        root_node_id="root",
        ending_intent="tonic",
        nodes=(
            PlanNode("root", None, 0, "whole"),
            PlanNode("a1", "root", 0, "opening"),
            PlanNode(
                "a1-leaf",
                "a1",
                0,
                "statement",
                harmonic_focus=9,
                duration_weight=1,
                score_material_id="theme",
            ),
            PlanNode(
                "bridge",
                "root",
                1,
                "contrast",
                harmonic_focus=0,
                duration_weight=1,
                score_material_id="bridge-material",
            ),
            PlanNode("a2", "root", 2, "return", derived_from="a1"),
            PlanNode(
                "a2-leaf",
                "a2",
                0,
                "return",
                derived_from="a1-leaf",
                harmonic_focus=9,
                duration_weight=1,
                score_material_id="theme",
            ),
        ),
    )
    score = ScoreSpec(
        score_id="score",
        divisions=4,
        materials=(
            ScoreMaterial(
                material_id="theme",
                length_units=8,
                notes=(
                    ScoreNote("theme-u1", 0, 4, 64, "upper", articulations=("tenuto",)),
                    ScoreNote("theme-l1", 0, 4, 48, "lower"),
                    ScoreNote("theme-u2", 4, 2, 67, "upper", articulations=("staccato",)),
                    ScoreNote("theme-l2", 4, 4, 52, "lower"),
                ),
                directions=(
                    ScoreDirection("theme-dyn", 0, "dynamic", "mf"),
                    ScoreDirection("theme-breath", 6, "breath", "light"),
                ),
            ),
            ScoreMaterial(
                material_id="bridge-material",
                length_units=4,
                notes=(
                    ScoreNote("bridge-u", 0, 4, 69, "upper", articulations=("accent",)),
                    ScoreNote("bridge-l", 0, 4, 45, "lower"),
                ),
                directions=(ScoreDirection("bridge-dyn", 0, "dynamic", "f"),),
            ),
        ),
    )
    performance = PerformanceSpec(
        performance_id="savor-then-flow",
        target_duration_ms=12_000,
        default_velocity=64,
        timing_budget_id="subtle-v1",
        node_performances=(
            NodePerformance(
                "a1",
                timing_profile="savor",
                timing_amount="moderate",
                dynamics_profile="shape",
                articulation_profile="legato",
                pedal_profile="phrase_legato",
            ),
            NodePerformance(
                "bridge",
                timing_profile="build",
                timing_amount="subtle",
                dynamics_profile="build",
                articulation_profile="score",
                pedal_profile="clear",
            ),
            NodePerformance(
                "a2",
                timing_profile="flow",
                timing_amount="subtle",
                dynamics_profile="steady",
                articulation_profile="light",
                pedal_profile="phrase_legato",
            ),
        ),
    )
    return plan, score, performance


def test_pipeline_accepts_arbitrary_depth_and_reuses_score_material() -> None:
    plan, score, performance = _fixture()

    validate_pipeline(plan, score, performance)

    assert plan.nodes[-1].derived_from == "a1-leaf"
    assert plan.nodes[2].score_material_id == plan.nodes[-1].score_material_id == "theme"


def test_performance_spec_can_be_validated_before_a_score_exists() -> None:
    plan, _, performance = _fixture()

    validate_performance_spec(plan, performance)

    with pytest.raises(PipelineValidationError, match="known and unique"):
        validate_performance_spec(
            plan,
            replace(
                performance,
                node_performances=(NodePerformance("unknown", timing_profile="neutral"),),
            ),
        )


def test_key_release_percent_preserves_legacy_lineage_and_terminal_release() -> None:
    plan, score, performance = _fixture()
    legacy = render_performance(plan, score, performance)

    assert legacy.lineage[2] == "afba53022d2a0e65b92efddce96b68aeb6e3ce93073c283ba8cc0250517bf89b"

    release_plan = replace(
        plan,
        nodes=(*plan.nodes[:-1], replace(plan.nodes[-1], role="release")),
    )
    baseline = render_performance(release_plan, score, performance)
    calibrated = render_performance(
        release_plan,
        score,
        replace(performance, key_release_percent=50),
    )
    before = {note.event_id: note for note in baseline.notes}
    after = {note.event_id: note for note in calibrated.notes}

    assert after["a1-leaf:theme-l2"].duration_ms < before["a1-leaf:theme-l2"].duration_ms
    assert after["a2-leaf:theme-l2"].duration_ms == before["a2-leaf:theme-l2"].duration_ms
    assert calibrated.pedals == baseline.pedals
    assert calibrated.harmonies == baseline.harmonies
    assert calibrated.node_intervals == baseline.node_intervals
    assert (
        render_performance(
            release_plan,
            score,
            replace(performance, key_release_percent=50),
        )
        == calibrated
    )


def _velocity_policy_fixture(
    *, foreground_voice: str = "upper"
) -> tuple[PiecePlan, ScoreSpec, PerformanceSpec]:
    accompaniment_voice = "lower" if foreground_voice == "upper" else "upper"
    foreground_pitch = 72 if foreground_voice == "upper" else 48
    accompaniment_pitch = 48 if accompaniment_voice == "lower" else 72
    plan = PiecePlan(
        plan_id="velocity-policy",
        title="声部別強弱",
        tonal_center=0,
        mode="major",
        root_node_id="root",
        ending_intent="tonic",
        nodes=(
            PlanNode("root", None, 0, "whole"),
            PlanNode(
                "leaf",
                "root",
                0,
                "statement",
                harmonic_focus=0,
                duration_weight=1,
                score_material_id="material",
            ),
        ),
    )
    accompaniment_notes = []
    for index, onset in enumerate((0, 2, 4, 6, 8, 10)):
        accompaniment_notes.append(
            ScoreNote(
                f"a-{index}-root",
                onset,
                2,
                accompaniment_pitch,
                accompaniment_voice,
            )
        )
        if onset == 2:
            accompaniment_notes.append(
                ScoreNote(
                    "a-1-accent",
                    onset,
                    2,
                    accompaniment_pitch + 7,
                    accompaniment_voice,
                    articulations=("accent",),
                )
            )
    score = ScoreSpec(
        score_id="velocity-score",
        divisions=4,
        materials=(
            ScoreMaterial(
                material_id="material",
                length_units=12,
                notes=(
                    ScoreNote("f-0", 0, 6, foreground_pitch, foreground_voice),
                    ScoreNote("f-1", 6, 6, foreground_pitch + 2, foreground_voice),
                    *accompaniment_notes,
                ),
                harmonies=(
                    ScoreHarmony("h-1", 0, 6, 0, "major"),
                    ScoreHarmony("h-2", 6, 6, 5, "major"),
                ),
                foreground_voice=foreground_voice,
            ),
        ),
    )
    performance = PerformanceSpec(
        performance_id="velocity-performance",
        target_duration_ms=12_000,
        default_velocity=80,
        timing_budget_id="subtle-v1",
        node_performances=(),
    )
    return plan, score, performance


@pytest.mark.parametrize("foreground_voice", ["upper", "lower"])
def test_velocity_policy_shapes_only_the_declared_accompaniment_voice(
    foreground_voice: str,
) -> None:
    plan, score, legacy_performance = _velocity_policy_fixture(foreground_voice=foreground_voice)
    shaped_performance = replace(
        legacy_performance,
        velocity_policy_id="foreground-accompaniment-harmony-shape-v1",
    )

    legacy = render_performance(plan, score, legacy_performance)
    shaped = render_performance(plan, score, shaped_performance)
    legacy_by_id = {note.event_id: note for note in legacy.notes}
    shaped_by_id = {note.event_id: note for note in shaped.notes}

    assert shaped_by_id["leaf:f-0"].velocity == legacy_by_id["leaf:f-0"].velocity
    assert shaped_by_id["leaf:f-1"].velocity == legacy_by_id["leaf:f-1"].velocity
    assert shaped_by_id["leaf:a-0-root"].velocity == 74
    assert shaped_by_id["leaf:a-1-root"].velocity == 78
    assert shaped_by_id["leaf:a-1-accent"].velocity == 86
    assert shaped_by_id["leaf:a-2-root"].velocity == 74
    assert shaped.notes != legacy.notes
    assert shaped.pedals == legacy.pedals
    assert shaped.harmonies == legacy.harmonies
    assert shaped.node_intervals == legacy.node_intervals
    assert shaped.lineage[2] != legacy.lineage[2]


def test_velocity_policy_diagnostic_uses_occurrence_and_harmony_boundaries() -> None:
    plan, score, performance = _velocity_policy_fixture()
    shaped = replace(
        performance,
        velocity_policy_id="foreground-accompaniment-harmony-shape-v1",
    )

    report = analyze_voice_velocity(plan, score, shaped)

    assert report["schema_version"] == 1
    assert report["velocity_policy_id"] == "foreground-accompaniment-harmony-shape-v1"
    assert report["foreground"]["note_count"] == 2
    assert report["accompaniment"]["note_count"] == 7
    assert report["accompaniment_attack_groups"]["group_count"] == 6
    assert report["accompaniment_attack_groups"]["eligible_harmony_count"] == 2
    assert report["accompaniment_attack_groups"]["varied_eligible_harmony_count"] == 2
    assert report["accompaniment_attack_groups"]["longest_equal_run"] == 1
    assert report["shared_onset_velocity_difference"]["count"] == 2


def test_pipeline_stages_validate_semantics_before_later_ir_exists() -> None:
    plan, score, _ = _fixture()

    validate_piece_plan(plan)
    validate_score_spec(plan, score)

    invalid_plan = replace(plan, nodes=plan.nodes[:-1])
    with pytest.raises(PipelineValidationError, match="duration and score material"):
        validate_piece_plan(invalid_plan)

    invalid_score = replace(score, materials=score.materials[:-1])
    with pytest.raises(PipelineValidationError, match="unknown score material"):
        validate_score_spec(plan, invalid_score)


def test_score_spec_rejects_an_empty_material_referenced_by_a_plan_leaf() -> None:
    plan, score, _ = _fixture()
    empty_theme = replace(score.materials[0], notes=())
    invalid = replace(score, materials=(empty_theme, *score.materials[1:]))

    with pytest.raises(PipelineValidationError, match="referenced score material must contain"):
        validate_score_spec(plan, invalid)


def test_score_spec_allows_an_unreferenced_empty_material() -> None:
    plan, score, _ = _fixture()
    unused = ScoreMaterial(material_id="unused", length_units=4, notes=())
    compatible = replace(score, materials=(*score.materials, unused))

    validate_score_spec(plan, compatible)


def test_partial_score_materials_are_validated_without_all_plan_materials() -> None:
    _, score, _ = _fixture()
    partial = replace(score, materials=(score.materials[0],))

    validate_score_materials(partial)

    material = partial.materials[0]
    invalid = replace(
        partial,
        materials=(
            replace(
                material,
                notes=(replace(material.notes[0], pitch=109), *material.notes[1:]),
            ),
        ),
    )
    with pytest.raises(PipelineValidationError, match="supported piano range"):
        validate_score_materials(invalid)


def _harmonic_fixture() -> tuple[PiecePlan, ScoreSpec, PerformanceSpec]:
    plan, score, performance = _fixture()
    theme, bridge = score.materials
    harmonic_score = replace(
        score,
        materials=(
            replace(
                theme,
                notes=tuple(
                    replace(note, pitch=45)
                    if note.event_id == "theme-l1"
                    else replace(note, pitch=48)
                    if note.event_id == "theme-l2"
                    else note
                    for note in theme.notes
                ),
                harmonies=(
                    ScoreHarmony("theme-am", 0, 4, 9, "minor"),
                    ScoreHarmony("theme-c", 4, 4, 0, "major"),
                ),
                foreground_voice="upper",
            ),
            replace(
                bridge,
                harmonies=(ScoreHarmony("bridge-am", 0, 4, 9, "minor"),),
                foreground_voice="lower",
            ),
        ),
    )
    return plan, harmonic_score, performance


def test_score_harmony_contract_covers_material_without_gaps_or_duplicates() -> None:
    plan, score, performance = _harmonic_fixture()
    validate_pipeline(plan, score, performance)
    theme, bridge = score.materials

    invalid_themes = (
        replace(theme, foreground_voice=None),
        replace(
            theme, harmonies=(replace(theme.harmonies[0], duration_units=3), *theme.harmonies[1:])
        ),
        replace(theme, harmonies=(theme.harmonies[0], replace(theme.harmonies[1], at_units=3))),
        replace(
            theme,
            harmonies=(theme.harmonies[0], replace(theme.harmonies[1], harmony_id="theme-am")),
        ),
        replace(
            theme,
            harmonies=(replace(theme.harmonies[0], root_pitch_class=12), *theme.harmonies[1:]),
        ),
        replace(
            theme, harmonies=(replace(theme.harmonies[0], quality="seventh"), *theme.harmonies[1:])
        ),
    )
    for invalid_theme in invalid_themes:
        with pytest.raises(PipelineValidationError, match="harmony"):
            validate_pipeline(
                plan,
                replace(score, materials=(invalid_theme, bridge)),
                performance,
            )


def test_pipeline_accepts_major_seventh_and_musicxml_preserves_it(tmp_path) -> None:
    plan, score, performance = _harmonic_fixture()
    theme, bridge = score.materials
    major_seventh = replace(
        theme,
        harmonies=(
            replace(theme.harmonies[0], quality="major-seventh"),
            *theme.harmonies[1:],
        ),
    )
    changed = replace(score, materials=(major_seventh, bridge))

    validate_pipeline(plan, changed, performance)
    output = render_musicxml(plan, changed, tmp_path / "major-seventh.musicxml")

    root = ET.parse(output).getroot()
    assert root.findtext(".//harmony/kind") == "major-seventh"


def test_musicxml_repeats_local_harmony_at_each_material_occurrence(tmp_path) -> None:
    plan, score, _ = _harmonic_fixture()
    output = tmp_path / "harmony.musicxml"

    render_musicxml(plan, score, output)

    root = ET.parse(output).getroot()
    harmonies = root.findall(".//harmony")
    assert len(harmonies) == 5
    assert [item.findtext("root/root-step") for item in harmonies[:2]] == ["A", "C"]
    assert [item.findtext("kind") for item in harmonies[:2]] == ["minor", "major"]
    assert [item.findtext("offset") for item in harmonies[:2]] == [None, "4"]


def test_harmony_quality_distinguishes_accompaniment_and_register() -> None:
    _, score, _ = _harmonic_fixture()
    theme = score.materials[0]
    thick_theme = replace(
        theme,
        notes=(
            ScoreNote("u1", 0, 2, 69, "upper"),
            ScoreNote("u2", 2, 2, 71, "upper"),
            ScoreNote("u3", 4, 4, 72, "upper"),
            ScoreNote("l1", 0, 4, 45, "lower"),
            ScoreNote("l1-oct", 0, 4, 57, "lower"),
            ScoreNote("l2", 1, 1, 52, "lower"),
            ScoreNote("l3", 2, 1, 57, "lower"),
            ScoreNote("l4", 4, 4, 48, "lower"),
            ScoreNote("l4-fifth", 4, 4, 55, "lower"),
        ),
    )
    assessment = analyze_material_harmony(
        replace(score, materials=(thick_theme, score.materials[1])),
        "theme",
        low_pitch_boundary=48,
        minimum_low_spacing_semitones=7,
    )
    assert assessment.status == "assessed"
    assert assessment.accompaniment_chord_attacks >= 2
    assert assessment.arpeggio_candidates >= 1
    assert assessment.accompaniment_chord_tone_ratio == 1.0
    assert assessment.low_spacing_violations == 0
    assert assessment.unresolved_foreground_non_chord_tones == 0
    assert assessment.passes

    bad_theme = replace(
        thick_theme,
        notes=(
            *thick_theme.notes,
            ScoreNote("low-clash", 1, 1, 49, "lower"),
            ScoreNote("ending-nonchord", 7, 1, 70, "upper"),
        ),
    )
    rejected = analyze_material_harmony(
        replace(score, materials=(bad_theme, score.materials[1])),
        "theme",
        low_pitch_boundary=48,
        minimum_low_spacing_semitones=7,
    )
    assert rejected.accompaniment_chord_tone_ratio < 1.0
    assert rejected.low_spacing_violations >= 1
    assert rejected.unresolved_foreground_non_chord_tones >= 1
    assert not rejected.passes


def test_harmony_quality_keeps_missing_declarations_unassessed() -> None:
    _, score, _ = _fixture()

    assessment = analyze_material_harmony(
        score,
        "theme",
        low_pitch_boundary=48,
        minimum_low_spacing_semitones=7,
    )

    assert assessment.status == "unassessed"
    assert not assessment.passes
    with pytest.raises(RecurrenceQualityError, match="positive"):
        analyze_material_harmony(
            score,
            "theme",
            low_pitch_boundary=48,
            minimum_low_spacing_semitones=0,
        )
    with pytest.raises(RecurrenceQualityError, match="does not exist"):
        analyze_material_harmony(
            score,
            "missing",
            low_pitch_boundary=48,
            minimum_low_spacing_semitones=7,
        )


def test_harmony_legato_clears_old_chord_before_next_harmony() -> None:
    plan, score, performance = _harmonic_fixture()
    phrase = render_performance(plan, score, performance)
    assert not analyze_rendered_harmony(plan, score, phrase).passes

    harmonic_performance = replace(
        performance,
        node_performances=tuple(
            replace(item, pedal_profile="harmony_legato") if item.node_id in {"a1", "a2"} else item
            for item in performance.node_performances
        ),
    )
    rendered = render_performance(plan, score, harmonic_performance)
    assessment = analyze_rendered_harmony(plan, score, rendered)
    assert assessment.status == "assessed"
    assert assessment.pedal_carryover_violations == 0
    assert assessment.passes


def test_pipeline_tracks_contrast_target_separately_from_derivation() -> None:
    plan, score, performance = _fixture()
    contrasted = replace(
        plan,
        nodes=tuple(
            replace(node, contrasts_with="a1") if node.node_id == "bridge" else node
            for node in plan.nodes
        ),
    )

    validate_pipeline(contrasted, score, performance)

    bridge = next(node for node in contrasted.nodes if node.node_id == "bridge")
    assert bridge.contrasts_with == "a1"
    assert bridge.derived_from is None

    invalid = replace(
        contrasted,
        nodes=tuple(
            replace(node, contrasts_with="a1-leaf") if node.node_id == "bridge" else node
            for node in contrasted.nodes
        ),
    )
    with pytest.raises(PipelineValidationError, match=r"contrast target.*same depth"):
        validate_pipeline(invalid, score, performance)


def test_pipeline_rejects_cross_depth_derivation() -> None:
    plan, score, performance = _fixture()
    nodes = tuple(
        replace(node, derived_from="a1") if node.node_id == "a2-leaf" else node
        for node in plan.nodes
    )

    with pytest.raises(PipelineValidationError, match="same depth"):
        validate_pipeline(replace(plan, nodes=nodes), score, performance)


def test_score_rejects_overlapping_same_pitch_in_one_material() -> None:
    plan, score, performance = _fixture()
    theme = score.materials[0]
    overlapping = replace(
        theme,
        notes=(*theme.notes, ScoreNote("overlap", 2, 4, 64, "upper")),
    )

    with pytest.raises(PipelineValidationError, match="same pitch"):
        validate_pipeline(
            plan,
            replace(score, materials=(overlapping, score.materials[1])),
            performance,
        )


def test_pipeline_rejects_invalid_structure_contracts() -> None:
    plan, score, performance = _fixture()
    a1 = plan.nodes[1]
    a2 = plan.nodes[4]

    invalid_cases = (
        (replace(plan, ending_intent="fade"), "ending intent"),
        (
            replace(
                plan,
                nodes=tuple(
                    replace(node, role="unknown") if node.node_id == "a1" else node
                    for node in plan.nodes
                ),
            ),
            "role",
        ),
        (
            replace(
                plan,
                nodes=tuple(
                    replace(node, derived_from=a2.node_id) if node.node_id == a1.node_id else node
                    for node in plan.nodes
                ),
            ),
            "earlier",
        ),
        (
            replace(
                plan,
                nodes=tuple(
                    replace(node, derived_from=None) if node.node_id == a2.node_id else node
                    for node in plan.nodes
                ),
            ),
            "return",
        ),
    )

    for invalid_plan, message in invalid_cases:
        with pytest.raises(PipelineValidationError, match=message):
            validate_pipeline(invalid_plan, score, performance)


def test_pipeline_rejects_broken_references_and_boundaries() -> None:
    plan, score, performance = _fixture()
    plan_cases = (
        replace(plan, nodes=(*plan.nodes, plan.nodes[0])),
        replace(
            plan,
            nodes=tuple(
                replace(node, parent_id="missing") if node.node_id == "bridge" else node
                for node in plan.nodes
            ),
        ),
        replace(plan, tonal_center=12),
        replace(plan, root_node_id="missing"),
        replace(
            plan,
            nodes=tuple(
                replace(node, order=3) if node.node_id == "bridge" else node for node in plan.nodes
            ),
        ),
        replace(
            plan,
            nodes=tuple(
                replace(node, duration_weight=0) if node.node_id == "bridge" else node
                for node in plan.nodes
            ),
        ),
        replace(
            plan,
            nodes=tuple(
                replace(node, harmonic_focus=12) if node.node_id == "bridge" else node
                for node in plan.nodes
            ),
        ),
        replace(
            plan,
            nodes=tuple(
                replace(node, derived_from="missing") if node.node_id == "a2" else node
                for node in plan.nodes
            ),
        ),
    )
    for invalid_plan in plan_cases:
        with pytest.raises(PipelineValidationError):
            validate_pipeline(invalid_plan, score, performance)

    theme = score.materials[0]
    score_cases = (
        replace(score, score_id=""),
        replace(score, materials=(theme, theme)),
        replace(
            score,
            materials=(replace(theme, length_units=0), score.materials[1]),
        ),
    )
    for invalid_score in score_cases:
        with pytest.raises(PipelineValidationError):
            validate_pipeline(plan, invalid_score, performance)
    unknown_material_plan = replace(
        plan,
        nodes=tuple(
            replace(node, score_material_id="missing") if node.node_id == "bridge" else node
            for node in plan.nodes
        ),
    )
    with pytest.raises(PipelineValidationError):
        validate_pipeline(unknown_material_plan, score, performance)

    performance_cases = (
        replace(performance, target_duration_ms=0),
        replace(
            performance,
            node_performances=(
                replace(performance.node_performances[0], node_id="missing"),
                *performance.node_performances[1:],
            ),
        ),
        replace(
            performance,
            node_performances=(
                replace(performance.node_performances[0], timing_profile="unsupported"),
                *performance.node_performances[1:],
            ),
        ),
    )
    for invalid_performance in performance_cases:
        with pytest.raises(PipelineValidationError):
            validate_pipeline(plan, score, invalid_performance)


def test_pipeline_rejects_invalid_score_contracts() -> None:
    plan, score, performance = _fixture()
    theme = score.materials[0]
    bridge = score.materials[1]
    invalid_materials = (
        (
            replace(
                theme,
                notes=(replace(theme.notes[0], pitch=20), *theme.notes[1:]),
            ),
            bridge,
            "piano range",
        ),
        (
            theme,
            replace(bridge, notes=(replace(bridge.notes[0], event_id="theme-u1"), bridge.notes[1])),
            "event ids",
        ),
        (
            replace(
                theme,
                notes=(
                    replace(theme.notes[0], articulations=("unsupported",)),
                    *theme.notes[1:],
                ),
            ),
            bridge,
            "articulation",
        ),
        (
            replace(
                theme,
                directions=(replace(theme.directions[0], value="fff"), theme.directions[1]),
            ),
            bridge,
            "direction",
        ),
        (
            replace(theme, derived_from="bridge-material"),
            bridge,
            "earlier score material",
        ),
    )

    for first, second, message in invalid_materials:
        with pytest.raises(PipelineValidationError, match=message):
            validate_pipeline(
                plan,
                replace(score, materials=(first, second)),
                performance,
            )


def test_hierarchical_timing_changes_rendition_without_desynchronizing_voices() -> None:
    plan, score, performance = _fixture()

    rendered = render_performance(plan, score, performance)

    a1_notes = [note for note in rendered.notes if note.occurrence_node_id == "a1-leaf"]
    a2_notes = [note for note in rendered.notes if note.occurrence_node_id == "a2-leaf"]
    assert [note.pitch for note in a1_notes] == [note.pitch for note in a2_notes]
    assert len(a1_notes) == len(a2_notes) == 4
    assert a1_notes[0].at_ms == a1_notes[1].at_ms
    assert a2_notes[0].at_ms == a2_notes[1].at_ms
    a1_span = max(note.at_ms + note.duration_ms for note in a1_notes) - min(
        note.at_ms for note in a1_notes
    )
    a2_span = max(note.at_ms + note.duration_ms for note in a2_notes) - min(
        note.at_ms for note in a2_notes
    )
    assert a1_span > a2_span
    assert rendered.duration_ms == 12_000
    assert rendered.pedals[-1].at_ms == 12_000
    assert rendered.pedals[-1].value == 0
    assert {pedal.value for pedal in rendered.pedals} <= {0, 127}
    assert len(rendered.lineage) == 3
    assert all(len(digest) == 64 for digest in rendered.lineage)


def test_coordination_profile_rolls_a1_but_keeps_a2_vertical() -> None:
    plan, score, performance = _fixture()
    coordinated = replace(
        performance,
        node_performances=tuple(
            replace(item, coordination_profile="rolled")
            if item.node_id == "a1"
            else replace(item, coordination_profile="aligned")
            if item.node_id == "a2"
            else item
            for item in performance.node_performances
        ),
    )

    rendered = render_performance(plan, score, coordinated)
    a1_start = [note.at_ms for note in rendered.notes if note.occurrence_node_id == "a1-leaf"]
    a2_start = [note.at_ms for note in rendered.notes if note.occurrence_node_id == "a2-leaf"]
    alignment = analyze_vertical_alignment(
        plan,
        rendered,
        source_node_id="a1",
        target_node_id="a2",
        minimum_target_ratio=0.60,
        minimum_increase=0.25,
    )

    assert len(set(a1_start[:2])) == 2
    assert len(set(a2_start[:2])) == 1
    assert alignment.target_ratio == 1.0
    assert alignment.target_ratio - alignment.source_ratio >= 0.25
    assert alignment.passes


def test_flat_and_shaped_performances_preserve_score_identity() -> None:
    plan, score, shaped = _fixture()
    flat = replace(
        shaped,
        performance_id="flat",
        node_performances=tuple(
            replace(
                item,
                timing_profile="neutral",
                dynamics_profile="steady",
                articulation_profile="score",
            )
            for item in shaped.node_performances
        ),
    )

    flat_result = render_performance(plan, score, flat)
    shaped_result = render_performance(plan, score, shaped)

    assert [(note.pitch, note.voice) for note in flat_result.notes] == [
        (note.pitch, note.voice) for note in shaped_result.notes
    ]
    assert len(flat_result.notes) == len(shaped_result.notes)
    assert flat_result.duration_ms == shaped_result.duration_ms == 12_000
    assert [(note.at_ms, note.duration_ms, note.velocity) for note in flat_result.notes] != [
        (note.at_ms, note.duration_ms, note.velocity) for note in shaped_result.notes
    ]


def test_duration_weight_does_not_create_unwritten_silence() -> None:
    plan, score, performance = _fixture()
    weighted = replace(
        plan,
        nodes=tuple(
            replace(node, duration_weight=2) if node.node_id == "a1-leaf" else node
            for node in plan.nodes
        ),
    )
    flat = replace(
        performance,
        node_performances=(),
    )

    rendered = render_performance(weighted, score, flat)

    bridge_onset = min(note.at_ms for note in rendered.notes if note.occurrence_node_id == "bridge")
    assert bridge_onset == 4_800


def test_nearest_pedal_profile_owns_each_occurrence() -> None:
    plan, score, performance = _fixture()
    overridden = replace(
        performance,
        node_performances=(
            *performance.node_performances,
            NodePerformance("a1-leaf", pedal_profile="none"),
        ),
    )

    rendered = render_performance(plan, score, overridden)

    bridge_onset = min(note.at_ms for note in rendered.notes if note.occurrence_node_id == "bridge")
    assert not any(pedal.value == 127 and pedal.at_ms < bridge_onset for pedal in rendered.pedals)


def test_recurrence_analysis_distinguishes_performance_change_from_copy() -> None:
    plan, score, performance = _fixture()
    shaped = render_performance(plan, score, performance)

    shaped_assessment = {
        item.target_node_id: item for item in analyze_recurrences(plan, score, shaped)
    }["a2"]

    assert shaped_assessment.relation_status == "related"
    assert not shaped_assessment.score_difference
    assert shaped_assessment.performance_difference
    assert not shaped_assessment.exact_surface_copy

    a1_performance = performance.node_performances[0]
    copied_performance = replace(
        performance,
        node_performances=tuple(
            replace(
                item,
                timing_profile=a1_performance.timing_profile,
                timing_amount=a1_performance.timing_amount,
                dynamics_profile=a1_performance.dynamics_profile,
                articulation_profile=a1_performance.articulation_profile,
                pedal_profile=a1_performance.pedal_profile,
            )
            if item.node_id == "a2"
            else item
            for item in performance.node_performances
        ),
    )
    copied = render_performance(plan, score, copied_performance)
    copied_assessment = {
        item.target_node_id: item for item in analyze_recurrences(plan, score, copied)
    }["a2"]

    assert copied_assessment.relation_status == "related"
    assert not copied_assessment.performance_difference
    assert copied_assessment.exact_surface_copy


def test_recurrence_analysis_separates_identity_evidence_from_missing_evidence() -> None:
    plan, score, performance = _fixture()
    theme = score.materials[0]
    theme_prime = replace(
        theme,
        material_id="theme-prime",
        derived_from="theme",
        notes=tuple(
            replace(
                note,
                event_id=f"prime-{note.event_id}",
                at_units=5,
                duration_units=1,
            )
            if note.event_id == "theme-u2"
            else replace(note, event_id=f"prime-{note.event_id}")
            for note in theme.notes
        ),
    )
    varied_plan = replace(
        plan,
        nodes=tuple(
            replace(node, score_material_id="theme-prime") if node.node_id == "a2-leaf" else node
            for node in plan.nodes
        ),
    )
    varied_score = replace(score, materials=(*score.materials, theme_prime))
    rendered = render_performance(varied_plan, varied_score, performance)

    related = {
        item.target_node_id: item
        for item in analyze_recurrences(
            varied_plan,
            varied_score,
            rendered,
            declared_cues={"a2": ("pitch_contour",)},
        )
    }["a2"]
    unassessed = {
        item.target_node_id: item
        for item in analyze_recurrences(varied_plan, varied_score, rendered)
    }["a2"]

    assert related.score_difference
    assert related.satisfied_identity_cues == ("pitch_contour",)
    assert related.relation_status == "related"
    assert unassessed.relation_status == "unassessed"

    unrelated_prime = replace(
        theme_prime,
        notes=tuple(
            replace(note, pitch=60) if note.event_id == "prime-theme-u2" else note
            for note in theme_prime.notes
        ),
    )
    unrelated_score = replace(
        score,
        materials=(*score.materials, unrelated_prime),
    )
    unrelated_rendered = render_performance(
        varied_plan,
        unrelated_score,
        performance,
    )
    unrelated = {
        item.target_node_id: item
        for item in analyze_recurrences(
            varied_plan,
            unrelated_score,
            unrelated_rendered,
            declared_cues={"a2": ("pitch_contour",)},
        )
    }["a2"]

    assert unrelated.satisfied_identity_cues == ()
    assert unrelated.relation_status == "unrelated"


def test_motif_head_allows_large_continuation_change_without_accepting_full_contour() -> None:
    plan, score, performance = _fixture()
    source = ScoreMaterial(
        material_id="theme",
        length_units=12,
        notes=(
            *(
                ScoreNote(f"source-u{index}", index * 2, 2, pitch, "upper")
                for index, pitch in enumerate((64, 66, 67, 71, 69, 67))
            ),
            ScoreNote("source-l0", 0, 6, 48, "lower"),
            ScoreNote("source-l1", 6, 6, 52, "lower"),
        ),
    )
    target = ScoreMaterial(
        material_id="theme-prime",
        length_units=12,
        derived_from="theme",
        notes=(
            *(
                ScoreNote(f"target-u{index}", index * 2, 2, pitch, "upper")
                for index, pitch in enumerate((69, 71, 72, 76, 72, 74))
            ),
            ScoreNote("target-l0", 0, 4, 45, "lower"),
            ScoreNote("target-l1", 4, 4, 52, "lower"),
            ScoreNote("target-l2", 8, 4, 57, "lower"),
        ),
    )
    varied_plan = replace(
        plan,
        nodes=tuple(
            replace(node, score_material_id="theme-prime") if node.node_id == "a2-leaf" else node
            for node in plan.nodes
        ),
    )
    varied_score = replace(score, materials=(source, score.materials[1], target))

    rendered = render_performance(varied_plan, varied_score, performance)
    assessment = {
        item.target_node_id: item
        for item in analyze_recurrences(
            varied_plan,
            varied_score,
            rendered,
            declared_cues={"a2": ("motif_head", "pitch_contour")},
        )
    }["a2"]

    assert assessment.satisfied_identity_cues == ("motif_head",)
    assert assessment.score_difference
    assert assessment.performance_difference
    assert assessment.relation_status == "related"


def test_voice_texture_gate_distinguishes_anchors_from_full_tracking() -> None:
    _, score, _ = _fixture()

    coupled = analyze_material_voice_texture(score.materials[0])

    assert coupled.opening_anchor == 0
    assert coupled.closing_anchor == 4
    assert not coupled.internal_independence
    assert not coupled.passes

    independent = ScoreMaterial(
        material_id="independent",
        length_units=8,
        notes=(
            ScoreNote("u1", 0, 2, 64, "upper"),
            ScoreNote("l1", 0, 2, 48, "lower"),
            ScoreNote("u2", 2, 2, 67, "upper"),
            ScoreNote("l2", 4, 2, 52, "lower"),
            ScoreNote("u3", 6, 2, 69, "upper"),
            ScoreNote("l3", 6, 2, 45, "lower"),
        ),
    )

    separated = analyze_material_voice_texture(independent)

    assert separated.opening_anchor == 0
    assert separated.closing_anchor == 6
    assert separated.internal_independence
    assert separated.passes


def test_material_alignment_requires_vertical_attacks_and_internal_motion() -> None:
    material = ScoreMaterial(
        material_id="aligned-with-motion",
        length_units=8,
        notes=(
            ScoreNote("u0", 0, 2, 64, "upper"),
            ScoreNote("l0", 0, 4, 48, "lower"),
            ScoreNote("u1", 2, 2, 67, "upper"),
            ScoreNote("u2", 4, 2, 69, "upper"),
            ScoreNote("l1", 4, 4, 52, "lower"),
        ),
    )

    assessment = analyze_material_vertical_alignment(material, minimum_ratio=0.60)

    assert assessment.shared_attack_ratio == pytest.approx(0.8)
    assert assessment.internal_independence
    assert assessment.passes


def test_piano_texture_gate_rejects_constant_dyads_and_accepts_varied_attacks() -> None:
    constant_dyads = ScoreMaterial(
        material_id="dyads",
        length_units=8,
        notes=tuple(
            note
            for index, onset in enumerate((0, 2, 4, 6))
            for note in (
                ScoreNote(f"u{index}", onset, 2, 64 + index, "upper"),
                ScoreNote(f"l{index}", onset, 2, 48 + index, "lower"),
            )
        ),
    )
    varied = ScoreMaterial(
        material_id="varied",
        length_units=8,
        notes=(
            ScoreNote("u0", 0, 2, 69, "upper"),
            ScoreNote("l0", 0, 4, 45, "lower"),
            ScoreNote("l0-fifth", 0, 4, 52, "lower"),
            ScoreNote("u1", 2, 2, 72, "upper"),
            ScoreNote("u2", 4, 2, 76, "upper"),
            ScoreNote("u2-third", 4, 2, 72, "upper"),
            ScoreNote("l1", 6, 2, 50, "lower"),
        ),
    )

    rejected = analyze_piano_texture_variety(constant_dyads)
    accepted = analyze_piano_texture_variety(varied)

    assert rejected.attack_size_counts == ((2, 4),)
    assert not rejected.passes
    assert accepted.attack_size_counts == ((1, 2), (2, 1), (3, 1))
    assert accepted.maximum_polyphony >= 3
    assert accepted.passes


def test_section_contrast_requires_multiple_scene_axes_and_shared_context() -> None:
    plan, score, _performance = _fixture()
    contrasted_plan = replace(
        plan,
        nodes=tuple(
            replace(node, contrasts_with="a1") if node.node_id == "bridge" else node
            for node in plan.nodes
        ),
    )
    superficial = analyze_section_contrast(
        contrasted_plan,
        score,
        target_node_id="bridge",
    )
    role_inversion = ScoreMaterial(
        material_id="bridge-material",
        length_units=8,
        notes=(
            ScoreNote("b-u0", 0, 4, 64, "upper", articulations=("accent",)),
            ScoreNote("b-u1", 4, 4, 67, "upper", articulations=("accent",)),
            ScoreNote("b-l0", 0, 1, 48, "lower", articulations=("staccato",)),
            ScoreNote("b-l1", 1, 1, 50, "lower", articulations=("staccato",)),
            ScoreNote("b-l2", 3, 1, 52, "lower", articulations=("staccato",)),
            ScoreNote("b-l3", 5, 1, 53, "lower", articulations=("staccato",)),
            ScoreNote("b-l4", 7, 1, 55, "lower", articulations=("staccato",)),
        ),
    )
    contrasted_score = replace(score, materials=(score.materials[0], role_inversion))
    categorical = analyze_section_contrast(
        contrasted_plan,
        contrasted_score,
        target_node_id="bridge",
    )
    lower_focused = analyze_section_contrast(
        contrasted_plan,
        contrasted_score,
        target_node_id="bridge",
        feature_voice="lower",
        minimum_changed_axes=2,
    )

    assert not superficial.passes
    assert {"rhythm_grid", "hand_role", "register", "articulation"} <= set(categorical.changed_axes)
    assert categorical.shared_pitch_class_ratio >= 0.5
    assert 0.0 <= categorical.rhythm_grid_distance <= 1.0
    assert 0.0 <= categorical.articulation_ratio_delta <= 1.0
    assert categorical.passes
    assert {"rhythm_grid", "articulation"} <= set(lower_focused.changed_axes)
    assert lower_focused.passes
    with pytest.raises(RecurrenceQualityError, match="feature voice"):
        analyze_section_contrast(
            contrasted_plan,
            contrasted_score,
            target_node_id="bridge",
            feature_voice="middle",
        )


def test_narrative_budget_makes_return_brisk_independently_of_note_count() -> None:
    plan, score, performance = _fixture()
    narrative = replace(
        performance,
        timing_budget_id="narrative-v1",
        node_performances=tuple(
            replace(item, timing_amount="moderate") if item.node_id in {"a1", "a2"} else item
            for item in performance.node_performances
        ),
    )

    rendered = render_performance(plan, score, narrative)
    pacing = analyze_section_pacing(
        plan,
        score,
        rendered,
        source_node_id="a1",
        target_node_id="a2",
        maximum_target_ratio=0.85,
    )

    assert pacing.target_note_count == pacing.source_note_count
    assert pacing.target_ms_per_unit / pacing.source_ms_per_unit <= 0.85
    assert pacing.passes
    assert rendered.duration_ms == 12_000


@pytest.mark.parametrize("profile", ("neutral", "build", "release"))
def test_narrative_v2_preserves_v1_non_rubato_profile_shape(profile: str) -> None:
    plan, score, performance = _fixture()
    node_performances = (
        NodePerformance(
            "root",
            timing_profile=profile,
            timing_amount="moderate",
        ),
    )
    v1 = replace(
        performance,
        timing_budget_id="narrative-v1",
        node_performances=node_performances,
    )
    v2 = replace(v1, timing_budget_id="narrative-v2")

    assert render_performance(plan, score, v2).notes == render_performance(plan, score, v1).notes


def test_narrative_v2_adds_bounded_shared_rubato_without_changing_score_grid() -> None:
    plan, score, performance = _fixture()
    theme, bridge = score.materials
    pulse_theme = replace(
        theme,
        notes=(
            ScoreNote("u0", 0, 1, 69, "upper"),
            ScoreNote("l0-root", 0, 1, 45, "lower"),
            ScoreNote("l0-open", 0, 1, 57, "lower"),
            ScoreNote("u1", 2, 1, 72, "upper"),
            ScoreNote("l1", 2, 1, 52, "lower"),
            ScoreNote("u2", 4, 1, 71, "upper"),
            ScoreNote("l2", 4, 1, 57, "lower"),
            ScoreNote("u3", 6, 1, 69, "upper"),
            ScoreNote("l3", 6, 1, 52, "lower"),
        ),
    )
    pulse_score = replace(score, materials=(pulse_theme, bridge))
    node_performances = tuple(
        replace(item, timing_amount="moderate") if item.node_id in {"a1", "a2"} else item
        for item in performance.node_performances
    )
    v1 = replace(
        performance,
        timing_budget_id="narrative-v1",
        node_performances=node_performances,
    )
    v2 = replace(v1, timing_budget_id="narrative-v2")

    rendered_v1 = render_performance(plan, pulse_score, v1)
    rendered_v2 = render_performance(plan, pulse_score, v2)
    pulse_v1 = analyze_local_pulse(
        plan,
        pulse_score,
        rendered_v1,
        node_id="a1",
        voice="lower",
        minimum_variation_ratio=1.08,
        maximum_adjacent_ratio=1.25,
    )
    pulse_v2 = analyze_local_pulse(
        plan,
        pulse_score,
        rendered_v2,
        node_id="a1",
        voice="lower",
        minimum_variation_ratio=1.08,
        maximum_adjacent_ratio=1.30,
    )
    flow_v2 = analyze_local_pulse(
        plan,
        pulse_score,
        rendered_v2,
        node_id="a2",
        voice="lower",
        minimum_variation_ratio=1.0,
        maximum_adjacent_ratio=1.25,
    )
    pacing = analyze_section_pacing(
        plan,
        pulse_score,
        rendered_v2,
        source_node_id="a1",
        target_node_id="a2",
        maximum_target_ratio=0.85,
    )

    assert not pulse_v1.passes
    assert pulse_v1.simultaneous_attack_max_spread_ms == 0
    assert pulse_v2.simultaneous_attack_max_spread_ms == 0
    assert pulse_v2.variation_ratio >= 1.08
    assert pulse_v2.maximum_adjacent_ratio <= 1.30
    assert pulse_v2.passes
    assert flow_v2.variation_ratio < pulse_v2.variation_ratio
    assert pacing.passes
    assert rendered_v2.duration_ms == 12_000


def test_boundary_breath_uses_explicit_calibration_threshold() -> None:
    plan, score, _ = _fixture()

    accepted = analyze_boundary_breath(
        plan,
        score,
        "a1",
        minimum_gap_units=4,
    )
    rejected = analyze_boundary_breath(
        plan,
        score,
        "a1",
        minimum_gap_units=5,
    )

    assert accepted.new_attack_gap_units == 4
    assert accepted.passes
    assert rejected.new_attack_gap_units == 4
    assert not rejected.passes


def test_quality_diagnostics_preserve_unassessable_failures() -> None:
    plan, score, performance = _fixture()
    no_shared_attack = ScoreMaterial(
        material_id="no-shared",
        length_units=4,
        notes=(
            ScoreNote("upper", 0, 1, 64, "upper"),
            ScoreNote("lower", 2, 1, 48, "lower"),
        ),
    )

    texture = analyze_material_voice_texture(no_shared_attack)

    assert texture.opening_anchor is None
    assert texture.closing_anchor is None
    assert not texture.passes
    with pytest.raises(RecurrenceQualityError, match="negative"):
        analyze_boundary_breath(plan, score, "a1", minimum_gap_units=-1)
    with pytest.raises(RecurrenceQualityError, match="does not exist"):
        analyze_boundary_breath(plan, score, "missing", minimum_gap_units=0)
    empty_score = replace(
        score,
        materials=(
            replace(score.materials[0], notes=()),
            score.materials[1],
        ),
    )
    with pytest.raises(RecurrenceQualityError, match="no note attacks"):
        analyze_boundary_breath(plan, empty_score, "a1", minimum_gap_units=0)
    rendered = render_performance(plan, score, performance)
    with pytest.raises(RecurrenceQualityError, match="voice"):
        analyze_local_pulse(
            plan,
            score,
            rendered,
            node_id="a1",
            voice="middle",
            minimum_variation_ratio=1.0,
            maximum_adjacent_ratio=1.0,
        )
    with pytest.raises(RecurrenceQualityError, match="at least one"):
        analyze_local_pulse(
            plan,
            score,
            rendered,
            node_id="a1",
            voice="lower",
            minimum_variation_ratio=0.9,
            maximum_adjacent_ratio=1.0,
        )
    with pytest.raises(RecurrenceQualityError, match="does not exist"):
        analyze_local_pulse(
            plan,
            score,
            rendered,
            node_id="missing",
            voice="lower",
            minimum_variation_ratio=1.0,
            maximum_adjacent_ratio=1.0,
        )
    with pytest.raises(RecurrenceQualityError, match="three attack positions"):
        analyze_local_pulse(
            plan,
            score,
            rendered,
            node_id="a1",
            voice="lower",
            minimum_variation_ratio=1.0,
            maximum_adjacent_ratio=1.0,
        )


def test_musicxml_is_deterministic_and_contains_score_not_performance(tmp_path) -> None:
    plan, score, performance = _fixture()
    first = tmp_path / "first.musicxml"
    second = tmp_path / "second.musicxml"

    render_musicxml(plan, score, first)
    render_musicxml(plan, score, second)

    assert first.read_bytes() == second.read_bytes()
    root = ET.parse(first).getroot()
    assert root.tag == "score-partwise"
    assert root.attrib["version"] == "4.0"
    assert root.find(".//senza-misura") is not None
    assert len(root.findall(".//note/pitch")) == 10
    assert root.find(".//staccato") is not None
    assert root.find(".//tenuto") is not None
    assert root.find(".//grouping/feature[@type='node-id']") is not None
    assert "12000" not in first.read_text(encoding="utf-8")
    assert performance.target_duration_ms == 12_000


def test_musicxml_preserves_ties_and_accidentals(tmp_path) -> None:
    plan, score, _ = _fixture()
    theme = score.materials[0]
    tied_sharp = replace(theme.notes[0], pitch=66, tie="start")
    changed_score = replace(
        score,
        materials=(replace(theme, notes=(tied_sharp, *theme.notes[1:])), score.materials[1]),
    )
    output = tmp_path / "ties.musicxml"

    render_musicxml(plan, changed_score, output)

    root = ET.parse(output).getroot()
    assert root.find(".//note/tie[@type='start']") is not None
    assert root.find(".//notations/tied[@type='start']") is not None
    assert root.find(".//pitch/alter").text == "1"


def test_performance_smf_is_deterministic_and_round_trips(tmp_path) -> None:
    plan, score, performance = _fixture()
    rendered = render_performance(plan, score, performance)
    first = tmp_path / "first.mid"
    second = tmp_path / "second.mid"

    first_result = render_performance_smf(rendered, first)
    second_result = render_performance_smf(rendered, second)

    assert (
        hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    )
    midi = mido.MidiFile(first, charset="utf-8")
    assert midi.ticks_per_beat == 500
    assert first_result.duration_ms == second_result.duration_ms == 12_000
    assert first_result.note_count == second_result.note_count == 10
    absolute = 0
    final_pedal = None
    for message in midi.tracks[1]:
        absolute += message.time
        if message.type == "control_change" and message.control == 64:
            final_pedal = (absolute, message.value)
    assert final_pedal == (12_000, 0)


def _round_trip_fixture() -> tuple[PiecePlan, ScoreSpec, PerformanceSpec]:
    plan, score, performance = _fixture()
    theme, bridge = score.materials
    score = replace(
        score,
        materials=(
            replace(
                theme,
                notes=(replace(theme.notes[0], tie="start"), *theme.notes[1:]),
                harmonies=(ScoreHarmony("theme-h", 0, 8, 9, "minor"),),
                foreground_voice="upper",
            ),
            replace(
                bridge,
                harmonies=(ScoreHarmony("bridge-h", 0, 4, 0, "major"),),
                foreground_voice="upper",
            ),
        ),
    )
    return plan, score, performance


def test_rendered_outputs_pass_full_round_trip_validation(tmp_path) -> None:
    plan, score, performance = _round_trip_fixture()
    rendered = render_performance(plan, score, performance)
    musicxml = render_musicxml(plan, score, tmp_path / "score.musicxml")
    smf = render_performance_smf(rendered, tmp_path / "performance.mid").path

    musicxml_result = validate_musicxml_round_trip(plan, score, musicxml)
    smf_result = validate_smf_round_trip(rendered, smf)

    assert musicxml_result["status"] == "passed"
    assert smf_result["status"] == "passed"
    assert all(item["status"] == "passed" for item in musicxml_result["checks"])
    assert all(item["status"] == "passed" for item in smf_result["checks"])


@pytest.mark.parametrize(
    ("mutation", "expected_check"),
    (
        ("version", "musicxml.version"),
        ("part", "musicxml.part"),
        ("measure_order", "musicxml.measure_order"),
        ("divisions", "musicxml.divisions"),
        ("leaf", "musicxml.leaf_nodes"),
        ("pitch", "musicxml.notes"),
        ("voice", "musicxml.notes"),
        ("staff", "musicxml.notes"),
        ("duration", "musicxml.notes"),
        ("tie", "musicxml.notes"),
        ("articulation", "musicxml.notes"),
        ("harmony", "musicxml.harmonies"),
        ("direction", "musicxml.directions"),
        ("extra_direction", "musicxml.directions"),
    ),
)
def test_musicxml_round_trip_rejects_semantic_mutations(
    tmp_path,
    mutation: str,
    expected_check: str,
) -> None:
    plan, score, _ = _round_trip_fixture()
    path = render_musicxml(plan, score, tmp_path / f"{mutation}.musicxml")
    tree = ET.parse(path)
    root = tree.getroot()
    if mutation == "version":
        root.attrib["version"] = "3.1"
    elif mutation == "part":
        root.remove(root.find("part"))
    elif mutation == "measure_order":
        part = root.find("part")
        part[:] = (part[1], part[0], *part[2:])
    elif mutation == "divisions":
        root.find(".//divisions").text = "8"
    elif mutation == "leaf":
        root.find(".//grouping/feature[@type='node-id']").text = "other"
    elif mutation == "pitch":
        root.find(".//note/pitch/step").text = "D"
    elif mutation == "voice":
        root.find(".//note/voice").text = "2"
    elif mutation == "staff":
        root.find(".//note/staff").text = "2"
    elif mutation == "duration":
        root.find(".//note/duration").text = "3"
    elif mutation == "tie":
        root.find(".//note").remove(root.find(".//note/tie"))
    elif mutation == "articulation":
        root.find(".//notations/articulations").clear()
    elif mutation == "harmony":
        root.find(".//harmony/kind").text = "major"
    elif mutation == "direction":
        root.find(".//direction/direction-type/dynamics").clear()
    elif mutation == "extra_direction":
        measure = root.find(".//measure")
        direction = ET.SubElement(measure, "direction")
        direction_type = ET.SubElement(direction, "direction-type")
        dynamics = ET.SubElement(direction_type, "dynamics")
        ET.SubElement(dynamics, "pp")
    tree.write(path, encoding="utf-8", xml_declaration=True)

    result = validate_musicxml_round_trip(plan, score, path)

    assert result["status"] == "failed"
    assert any(
        item["check_id"] == expected_check and item["status"] == "failed"
        for item in result["checks"]
    )


@pytest.mark.parametrize(
    ("mutation", "expected_check"),
    (
        ("type", "smf.transport"),
        ("tracks", "smf.transport"),
        ("ticks", "smf.transport"),
        ("tempo", "smf.transport"),
        ("program", "smf.messages"),
        ("pitch", "smf.notes"),
        ("velocity", "smf.notes"),
        ("duration", "smf.notes"),
        ("pedal", "smf.pedals"),
        ("dangling", "smf.notes"),
        ("extra_channel", "smf.messages"),
        ("extra_track_name", "smf.messages"),
        ("ending", "smf.ending"),
    ),
)
def test_smf_round_trip_rejects_transport_and_event_mutations(
    tmp_path,
    mutation: str,
    expected_check: str,
) -> None:
    plan, score, performance = _round_trip_fixture()
    rendered = render_performance(plan, score, performance)
    path = render_performance_smf(rendered, tmp_path / f"{mutation}.mid").path
    midi = mido.MidiFile(path)
    if mutation == "type":
        midi.type = 2
    elif mutation == "tracks":
        midi.tracks.pop()
    elif mutation == "ticks":
        midi.ticks_per_beat = 480
    elif mutation == "tempo":
        next(item for item in midi.tracks[0] if item.type == "set_tempo").tempo = 600_000
    elif mutation == "program":
        next(item for item in midi.tracks[1] if item.type == "program_change").program = 1
    elif mutation == "pitch":
        next(item for item in midi.tracks[1] if item.type == "note_on" and item.velocity).note += 1
    elif mutation == "velocity":
        next(
            item for item in midi.tracks[1] if item.type == "note_on" and item.velocity
        ).velocity += 1
    elif mutation == "duration":
        next(item for item in midi.tracks[1] if item.type == "note_off").time += 1
    elif mutation == "pedal":
        next(
            item for item in midi.tracks[1] if item.type == "control_change" and item.control == 64
        ).value = 1
    elif mutation == "dangling":
        track = midi.tracks[1]
        track.remove(next(item for item in track if item.type == "note_off"))
    elif mutation == "extra_channel":
        midi.tracks[1].insert(1, mido.Message("aftertouch", value=1, time=0))
    elif mutation == "extra_track_name":
        midi.tracks[1].insert(1, mido.MetaMessage("track_name", name="Extra", time=0))
    elif mutation == "ending":
        next(item for item in midi.tracks[0] if item.type == "end_of_track").time += 1
    midi.save(path)

    result = validate_smf_round_trip(rendered, path)

    assert result["status"] == "failed"
    assert any(
        item["check_id"] == expected_check and item["status"] == "failed"
        for item in result["checks"]
    )
