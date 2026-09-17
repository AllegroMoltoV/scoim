from __future__ import annotations

from dataclasses import replace

import pytest

from llm_musical_composer.generic_pipeline_quality import (
    evaluate_generic_piece_plan_quality,
)
from llm_musical_composer.performance_pipeline import (
    PiecePlan,
    PlanNode,
    RenderedPerformance,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    render_musicxml,
    render_performance,
    render_performance_smf,
)
from llm_musical_composer.recurrence_analysis import analyze_recurrences
from llm_musical_composer.staged_material_pilot import (
    HarmonicDraft,
    HarmonicEventDraft,
    MelodyDraft,
    MelodyEventDraft,
    TextureDraft,
    TextureEventDraft,
)
from llm_musical_composer.whole_score_staged_generation import (
    MaterialIdentityCueV0,
    PerformanceOccurrenceDraftV0,
    WholeScoreStagedError,
    analyze_foreground_transition,
    assemble_whole_score_melodies,
    assemble_whole_score_performance,
    assemble_whole_score_skeleton,
    assemble_whole_score_textures,
    build_whole_score_context,
    melody_generation_order,
    validate_material_identity_cues,
)


def _plan() -> PiecePlan:
    return PiecePlan(
        "whole-score-plan",
        "曲全体",
        0,
        "major",
        "whole",
        "tonic",
        (
            PlanNode("whole", None, 0, "whole"),
            PlanNode("opening", "whole", 0, "opening"),
            PlanNode(
                "base",
                "opening",
                0,
                "statement",
                duration_weight=8,
                score_material_id="base-material",
            ),
            PlanNode(
                "base-repeat",
                "opening",
                1,
                "return",
                derived_from="base",
                duration_weight=5,
                score_material_id="base-material",
            ),
            PlanNode(
                "transition",
                "whole",
                1,
                "transition",
                duration_weight=2,
                score_material_id="transition-material",
            ),
            PlanNode(
                "return",
                "whole",
                2,
                "return",
                derived_from="opening",
            ),
            PlanNode(
                "derived",
                "return",
                0,
                "variation",
                derived_from="base",
                duration_weight=8,
                score_material_id="derived-material",
            ),
            PlanNode(
                "ending",
                "return",
                1,
                "release",
                duration_weight=3,
                score_material_id="ending-material",
            ),
        ),
    )


def _harmonic_draft(root: int, length: int) -> HarmonicDraft:
    first = length // 2
    return HarmonicDraft(
        (
            HarmonicEventDraft(0, first, root, "major"),
            HarmonicEventDraft(first, length - first, (root + 7) % 12, "major"),
        )
    )


def _ending_harmonic_draft(root: int, length: int) -> HarmonicDraft:
    return HarmonicDraft((HarmonicEventDraft(0, length, root, "major"),))


def test_context_separates_materials_occurrences_and_relations() -> None:
    context = build_whole_score_context(_plan())

    assert context.material_ids == (
        "base-material",
        "transition-material",
        "derived-material",
        "ending-material",
    )
    assert context.occurrence_weights["base-material"] == (8, 5)
    assert context.transition_material_ids == ("transition-material",)
    assert [(item.target_node_id, item.source_node_id) for item in context.relations] == [
        ("base-repeat", "base"),
        ("return", "opening"),
        ("derived", "base"),
    ]


def test_melody_order_defers_transition_until_both_neighbors_exist() -> None:
    context = build_whole_score_context(_plan())

    first_pass, second_pass = melody_generation_order(context)

    assert first_pass == ("base-material", "derived-material", "ending-material")
    assert second_pass == ("transition-material",)
    transition = context.transitions[0]
    assert transition.previous_material_id == "base-material"
    assert transition.next_material_id == "derived-material"
    assert transition.previous_node_id == "base-repeat"
    assert transition.next_node_id == "derived"


def test_skeleton_uses_runner_owned_ids_and_divisions() -> None:
    plan = _plan()
    drafts = (
        (12, _harmonic_draft(0, 12)),
        (6, _harmonic_draft(7, 6)),
        (12, _harmonic_draft(0, 12)),
        (6, _ending_harmonic_draft(0, 6)),
    )

    skeleton = assemble_whole_score_skeleton("case-a", plan, drafts)

    assert skeleton.score_id == "whole-score-case-a"
    assert skeleton.divisions == 12
    assert tuple(item.material_id for item in skeleton.materials) == (
        "base-material",
        "transition-material",
        "derived-material",
        "ending-material",
    )
    assert skeleton.materials[0].harmonies[0].harmony_id == "whole-case-a-h-001-001"


def test_skeleton_rejects_missing_or_extra_material_drafts() -> None:
    plan = _plan()
    drafts = ((12, _harmonic_draft(0, 12)),)

    with pytest.raises(WholeScoreStagedError, match="draft count"):
        assemble_whole_score_skeleton("case-a", plan, drafts)


def test_skeleton_allows_one_harmony_only_for_release_material() -> None:
    plan = _plan()
    accepted = (
        (12, _harmonic_draft(0, 12)),
        (6, _harmonic_draft(7, 6)),
        (12, _harmonic_draft(0, 12)),
        (6, _ending_harmonic_draft(0, 6)),
    )
    rejected = (
        (12, _ending_harmonic_draft(0, 12)),
        *accepted[1:],
    )

    assert len(assemble_whole_score_skeleton("ending", plan, accepted).materials[-1].harmonies) == 1
    with pytest.raises(WholeScoreStagedError, match="event count"):
        assemble_whole_score_skeleton("statement", plan, rejected)


def _melody_draft(pitches: tuple[int, int, int, int], length: int) -> MelodyDraft:
    step = length // 4
    return MelodyDraft(
        "upper",
        tuple(MelodyEventDraft(index * step, step, pitch) for index, pitch in enumerate(pitches)),
    )


def test_melody_collection_generates_each_material_once_without_old_score() -> None:
    plan = _plan()
    skeleton = assemble_whole_score_skeleton(
        "case-a",
        plan,
        (
            (12, _harmonic_draft(0, 12)),
            (6, _harmonic_draft(7, 6)),
            (12, _harmonic_draft(0, 12)),
            (6, _ending_harmonic_draft(0, 6)),
        ),
    )
    drafts = (
        _melody_draft((60, 64, 67, 71), 12),
        _melody_draft((67, 71, 74, 79), 6),
        _melody_draft((60, 64, 67, 72), 12),
        _melody_draft((60, 64, 67, 72), 6),
    )

    payload = assemble_whole_score_melodies("case-a", plan, skeleton, drafts)

    assert tuple(item.material_id for item in payload.materials) == skeleton_material_ids(skeleton)
    assert all(len(item.notes) == 4 for item in payload.materials)
    assert len({note.event_id for item in payload.materials for note in item.notes}) == 16
    assert all(item.directions == () for item in payload.materials)


def _texture_draft(harmony_starts: tuple[tuple[int, int], ...]) -> TextureDraft:
    return TextureDraft(
        tuple(
            event
            for harmony_index, (at_units, duration_units) in enumerate(harmony_starts)
            for event in (
                TextureEventDraft(
                    harmony_index,
                    at_units,
                    duration_units,
                    "root",
                    "bass",
                ),
                TextureEventDraft(
                    harmony_index,
                    at_units,
                    duration_units,
                    "fifth",
                    "low",
                ),
            )
        )
    )


def test_texture_collection_places_all_materials_after_melody_stage() -> None:
    plan = _plan()
    skeleton = assemble_whole_score_skeleton(
        "texture",
        plan,
        (
            (12, _harmonic_draft(0, 12)),
            (6, _harmonic_draft(7, 6)),
            (12, _harmonic_draft(0, 12)),
            (6, _ending_harmonic_draft(0, 6)),
        ),
    )
    melodies = assemble_whole_score_melodies(
        "texture",
        plan,
        skeleton,
        (
            _melody_draft((60, 64, 67, 71), 12),
            _melody_draft((67, 71, 74, 79), 6),
            _melody_draft((60, 64, 67, 72), 12),
            _melody_draft((60, 64, 67, 72), 6),
        ),
    )
    textures = (
        _texture_draft(((0, 6), (6, 6))),
        _texture_draft(((0, 3), (3, 3))),
        _texture_draft(((0, 6), (6, 6))),
        _texture_draft(((0, 6),)),
    )

    result = assemble_whole_score_textures(
        "texture",
        plan,
        skeleton,
        melodies,
        textures,
        allowed_pitch_range=(36, 79),
    )

    assert tuple(item.status for item in result.placements) == ("placed",) * 4
    assert tuple(len(item.notes) for item in result.score.materials) == (8, 8, 8, 6)
    assert result.low_spacing_violations == (0, 0, 0, 0)
    assert all(
        36 <= note.pitch <= 79
        for material in result.score.materials
        for note in material.notes
    )


def test_texture_collection_rejects_infeasible_zone_before_placement() -> None:
    plan = _plan()
    skeleton = assemble_whole_score_skeleton(
        "bad-texture",
        plan,
        (
            (12, _harmonic_draft(0, 12)),
            (6, _harmonic_draft(7, 6)),
            (12, _harmonic_draft(0, 12)),
            (6, _ending_harmonic_draft(0, 6)),
        ),
    )
    melodies = assemble_whole_score_melodies(
        "bad-texture",
        plan,
        skeleton,
        (
            _melody_draft((60, 64, 67, 71), 12),
            _melody_draft((67, 71, 74, 79), 6),
            _melody_draft((60, 64, 67, 72), 12),
            _melody_draft((60, 64, 67, 72), 6),
        ),
    )
    bad = replace(
        _texture_draft(((0, 6), (6, 6))),
        events=(
            TextureEventDraft(0, 0, 6, "root", "high"),
            TextureEventDraft(0, 0, 6, "fifth", "high"),
        ),
    )

    with pytest.raises(WholeScoreStagedError, match="event feasibility"):
        assemble_whole_score_textures(
            "bad-texture",
            plan,
            skeleton,
            melodies,
            (bad, *_texture_draft_sequence()[1:]),
        )


def _texture_draft_sequence() -> tuple[TextureDraft, ...]:
    return (
        _texture_draft(((0, 6), (6, 6))),
        _texture_draft(((0, 3), (3, 3))),
        _texture_draft(((0, 6), (6, 6))),
        _texture_draft(((0, 6),)),
    )


def _assembled_score(case_id: str = "performance") -> tuple[PiecePlan, ScoreSpec]:
    plan = _plan()
    skeleton = assemble_whole_score_skeleton(
        case_id,
        plan,
        (
            (12, _harmonic_draft(0, 12)),
            (6, _harmonic_draft(7, 6)),
            (12, _harmonic_draft(0, 12)),
            (6, _ending_harmonic_draft(0, 6)),
        ),
    )
    melodies = assemble_whole_score_melodies(
        case_id,
        plan,
        skeleton,
        (
            _melody_draft((60, 64, 67, 71), 12),
            _melody_draft((67, 71, 74, 79), 6),
            _melody_draft((60, 64, 67, 72), 12),
            _melody_draft((60, 64, 67, 72), 6),
        ),
    )
    result = assemble_whole_score_textures(
        case_id,
        plan,
        skeleton,
        melodies,
        _texture_draft_sequence(),
    )
    return plan, result.score


def test_performance_collection_owns_leaf_ids_and_180_second_target() -> None:
    plan, score = _assembled_score()
    drafts = _performance_draft_sequence()

    performance = assemble_whole_score_performance("performance", plan, score, drafts)

    assert performance.target_duration_ms == 180_000
    assert tuple(item.node_id for item in performance.node_performances) == (
        "base",
        "base-repeat",
        "transition",
        "derived",
        "ending",
    )
    assert performance.node_performances[0].timing_profile == "savor"
    assert performance.node_performances[1].timing_profile == "flow"


def _performance_draft_sequence() -> tuple[PerformanceOccurrenceDraftV0, ...]:
    return (
        PerformanceOccurrenceDraftV0(
            "savor", "subtle", "shape", "legato", "rolled", "harmony_legato"
        ),
        PerformanceOccurrenceDraftV0(
            "flow", "subtle", "steady", "score", "aligned", "harmony_legato"
        ),
        PerformanceOccurrenceDraftV0(
            "build", "subtle", "build", "score", "aligned", "harmony_legato"
        ),
        PerformanceOccurrenceDraftV0(
            "flow", "subtle", "shape", "legato", "aligned", "harmony_legato"
        ),
        PerformanceOccurrenceDraftV0(
            "release", "subtle", "release", "legato", "aligned", "harmony_legato"
        ),
    )


def test_performance_collection_rejects_wrong_occurrence_count() -> None:
    plan, score = _assembled_score("short-performance")

    with pytest.raises(WholeScoreStagedError, match="performance draft count"):
        assemble_whole_score_performance(
            "short-performance",
            plan,
            score,
            (PerformanceOccurrenceDraftV0(),),
        )


def test_whole_score_contract_renders_deterministic_180_second_outputs(tmp_path) -> None:
    plan, score = _assembled_score("render")
    performance = assemble_whole_score_performance(
        "render",
        plan,
        score,
        _performance_draft_sequence(),
    )
    rendered = render_performance(plan, score, performance)

    first_xml = render_musicxml(plan, score, tmp_path / "first" / "final.musicxml")
    first_midi = render_performance_smf(
        rendered, tmp_path / "first" / "final.mid"
    ).path
    second_xml = render_musicxml(plan, score, tmp_path / "second" / "final.musicxml")
    second_midi = render_performance_smf(
        rendered, tmp_path / "second" / "final.mid"
    ).path

    assert rendered.duration_ms == 180_000
    assert rendered.notes
    assert rendered.pedals[-1].value == 0
    assert first_xml.read_bytes() == second_xml.read_bytes()
    assert first_midi.read_bytes() == second_midi.read_bytes()


def skeleton_material_ids(skeleton) -> tuple[str, ...]:
    return tuple(item.material_id for item in skeleton.materials)


def test_identity_cues_require_fixed_relation_and_supported_foreground_cue() -> None:
    context = build_whole_score_context(_plan())
    cues = (
        MaterialIdentityCueV0(
            "derived",
            "base",
            "derived-material",
            "base-material",
            ("foreground_motif_head",),
        ),
    )

    assert validate_material_identity_cues(context, cues) == {"derived": ("foreground_motif_head",)}

    with pytest.raises(WholeScoreStagedError, match="unsupported"):
        validate_material_identity_cues(
            context,
            (replace(cues[0], cues=("motif_head",)),),
        )


def test_lower_foreground_cue_does_not_use_matching_upper_accompaniment() -> None:
    plan = PiecePlan(
        "lower-plan",
        "低音前景",
        0,
        "major",
        "whole",
        "tonic",
        (
            PlanNode("whole", None, 0, "whole"),
            PlanNode(
                "source",
                "whole",
                0,
                "statement",
                duration_weight=1,
                score_material_id="source-material",
            ),
            PlanNode(
                "target",
                "whole",
                1,
                "variation",
                derived_from="source",
                duration_weight=1,
                score_material_id="target-material",
            ),
        ),
    )
    upper = tuple(
        ScoreNote(f"u-{index}", index * 2, 1, pitch, "upper")
        for index, pitch in enumerate((72, 74, 76, 77))
    )
    source = ScoreMaterial(
        "source-material",
        8,
        upper
        + tuple(
            ScoreNote(f"sl-{index}", index * 2, 1, pitch, "lower")
            for index, pitch in enumerate((48, 50, 52, 53))
        ),
        foreground_voice="lower",
    )
    target = ScoreMaterial(
        "target-material",
        8,
        tuple(replace(note, event_id=f"copy-{note.event_id}") for note in upper)
        + tuple(
            ScoreNote(f"tl-{index}", index * 2, 1, pitch, "lower")
            for index, pitch in enumerate((48, 55, 49, 58))
        ),
        derived_from="source-material",
        foreground_voice="lower",
    )
    score = ScoreSpec("lower-score", 4, (source, target))
    rendered = RenderedPerformance(
        "performance",
        "低音前景",
        1_000,
        (),
        (),
        (),
        (),
        ("plan", "score", "performance"),
    )

    result = analyze_recurrences(
        plan,
        score,
        rendered,
        declared_cues={"target": ("foreground_motif_head",)},
    )

    assert result[0].satisfied_identity_cues == ()
    assert result[0].relation_status == "unrelated"


def test_piece_plan_quality_accepts_different_material_only_with_declared_cue() -> None:
    plan = _plan()

    without = evaluate_generic_piece_plan_quality(plan)
    with_cue = evaluate_generic_piece_plan_quality(
        plan,
        declared_cues={"derived": ("foreground_motif_head",)},
    )

    assert "unassessable-recurrence:derived" in without["failures"]
    assert "unassessable-recurrence:derived" not in with_cue["failures"]


def test_transition_diagnostic_uses_each_materials_own_foreground_voice() -> None:
    plan = _plan()
    context = build_whole_score_context(plan)
    score = ScoreSpec(
        "alternating-foreground",
        12,
        (
            ScoreMaterial(
                "base-material",
                12,
                (
                    ScoreNote("base-l", 0, 12, 48, "lower"),
                    ScoreNote("base-u", 9, 3, 67, "upper"),
                ),
                harmonies=(ScoreHarmony("base-h", 0, 12, 0, "major"),),
                foreground_voice="upper",
            ),
            ScoreMaterial(
                "transition-material",
                6,
                (
                    ScoreNote("transition-l1", 0, 2, 65, "lower"),
                    ScoreNote("transition-l2", 2, 2, 64, "lower"),
                    ScoreNote("transition-l3", 4, 2, 62, "lower"),
                    ScoreNote("transition-u", 0, 6, 84, "upper"),
                ),
                harmonies=(ScoreHarmony("transition-h", 0, 6, 7, "major"),),
                foreground_voice="lower",
            ),
            ScoreMaterial(
                "derived-material",
                12,
                (
                    ScoreNote("derived-l", 0, 12, 43, "lower"),
                    ScoreNote("derived-u", 0, 3, 60, "upper"),
                ),
                derived_from="base-material",
                harmonies=(ScoreHarmony("derived-h", 0, 12, 0, "major"),),
                foreground_voice="upper",
            ),
            ScoreMaterial(
                "ending-material",
                6,
                (
                    ScoreNote("ending-l", 0, 6, 48, "lower"),
                    ScoreNote("ending-u", 0, 6, 60, "upper"),
                ),
                harmonies=(ScoreHarmony("ending-h", 0, 6, 0, "major"),),
                foreground_voice="upper",
            ),
        ),
    )

    result = analyze_foreground_transition(
        plan,
        score,
        context.transitions[0],
        minimum_transition_attacks=3,
        maximum_boundary_leap=5,
        maximum_internal_leap=3,
    )

    assert (result.source_voice, result.transition_voice, result.target_voice) == (
        "upper",
        "lower",
        "upper",
    )
    assert result.source_to_transition_semitones == 2
    assert result.transition_to_target_semitones == 2
    assert result.passes
