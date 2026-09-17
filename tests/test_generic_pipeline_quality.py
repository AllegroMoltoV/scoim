from dataclasses import replace

from llm_musical_composer.generic_pipeline_quality import (
    evaluate_generic_piece_plan_quality,
    evaluate_generic_pipeline_quality,
    evaluate_generic_score_quality,
)
from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    PiecePlan,
    PlanNode,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    render_performance,
)


def _fixture() -> tuple[PiecePlan, ScoreSpec, PerformanceSpec]:
    plan = PiecePlan(
        "generic-plan",
        "形式に依存しない曲",
        9,
        "minor",
        "entire",
        "tonic",
        (
            PlanNode("entire", None, 0, "whole"),
            PlanNode(
                "first-scene",
                "entire",
                0,
                "statement",
                harmonic_focus=9,
                duration_weight=1,
                score_material_id="theme",
            ),
            PlanNode(
                "different-scene",
                "entire",
                1,
                "contrast",
                contrasts_with="first-scene",
                harmonic_focus=4,
                duration_weight=1,
                score_material_id="contrast",
            ),
            PlanNode(
                "recollected-scene",
                "entire",
                2,
                "return",
                derived_from="first-scene",
                harmonic_focus=9,
                duration_weight=1,
                score_material_id="theme",
            ),
            PlanNode(
                "tonic-ending",
                "entire",
                3,
                "release",
                harmonic_focus=9,
                duration_weight=1,
                score_material_id="ending",
            ),
        ),
    )
    theme = ScoreMaterial(
        "theme",
        8,
        (
            ScoreNote("t-l0", 0, 4, 45, "lower"),
            ScoreNote("t-l1", 0, 4, 52, "lower"),
            ScoreNote("t-u0", 0, 4, 69, "upper"),
            ScoreNote("t-u1", 4, 4, 69, "upper", articulations=("tenuto",)),
            ScoreNote("t-l2", 4, 4, 52, "lower", articulations=("tenuto",)),
        ),
        harmonies=(ScoreHarmony("theme-a", 0, 8, 9, "minor"),),
        foreground_voice="upper",
    )
    contrast = ScoreMaterial(
        "contrast",
        8,
        (
            ScoreNote("c-l0", 0, 4, 40, "lower"),
            ScoreNote("c-l1", 0, 4, 47, "lower"),
            ScoreNote("c-u0", 0, 2, 64, "upper"),
            ScoreNote("c-u1", 2, 2, 68, "upper"),
            ScoreNote("c-u2", 4, 4, 71, "upper"),
        ),
        harmonies=(ScoreHarmony("contrast-e", 0, 8, 4, "major"),),
        foreground_voice="upper",
    )
    ending = ScoreMaterial(
        "ending",
        8,
        (
            ScoreNote("e-l0", 0, 8, 45, "lower", articulations=("tenuto",)),
            ScoreNote("e-l1", 0, 8, 52, "lower", articulations=("tenuto",)),
            ScoreNote("e-u0", 0, 8, 69, "upper", articulations=("tenuto",)),
        ),
        harmonies=(ScoreHarmony("ending-a", 0, 8, 9, "minor"),),
        foreground_voice="upper",
    )
    score = ScoreSpec("generic-score", 4, (theme, contrast, ending))
    performance = PerformanceSpec(
        "generic-performance",
        180_000,
        64,
        "narrative-v2",
        (
            NodePerformance(
                "first-scene",
                timing_profile="savor",
                timing_amount="subtle",
                dynamics_profile="shape",
                articulation_profile="legato",
                coordination_profile="rolled",
                pedal_profile="harmony_legato",
            ),
            NodePerformance(
                "different-scene",
                timing_profile="build",
                timing_amount="subtle",
                dynamics_profile="build",
                articulation_profile="score",
                coordination_profile="score",
                pedal_profile="harmony_legato",
            ),
            NodePerformance(
                "recollected-scene",
                timing_profile="flow",
                timing_amount="subtle",
                dynamics_profile="steady",
                articulation_profile="legato",
                coordination_profile="aligned",
                pedal_profile="harmony_legato",
            ),
            NodePerformance(
                "tonic-ending",
                timing_profile="release",
                timing_amount="subtle",
                dynamics_profile="release",
                articulation_profile="legato",
                coordination_profile="aligned",
                pedal_profile="harmony_legato",
            ),
        ),
    )
    return plan, score, performance


def _evaluate(plan: PiecePlan, score: ScoreSpec, performance: PerformanceSpec) -> dict[str, object]:
    return evaluate_generic_pipeline_quality(
        plan, score, performance, render_performance(plan, score, performance)
    )


def test_generic_quality_passes_without_calibration_section_ids() -> None:
    plan, score, performance = _fixture()

    result = _evaluate(plan, score, performance)

    assert result["passes"] is True
    assert result["failures"] == []
    assert result["recurrences"][0]["relation_status"] == "related"
    assert result["recurrences"][0]["exact_surface_copy"] is False


def test_score_quality_passes_before_performance_generation() -> None:
    plan, score, _ = _fixture()

    result = evaluate_generic_score_quality(plan, score)

    assert result["passes"] is True
    assert result["failures"] == []
    assert result["ending_score"]["dedicated_single_harmony_material"] is True
    assert result["ending_score"]["minimum_nominal_duration_ms"] >= 2_000


def test_generic_quality_rejects_missing_declared_relation() -> None:
    plan, score, performance = _fixture()
    nodes = tuple(replace(node, contrasts_with=None) for node in plan.nodes)

    result = _evaluate(replace(plan, nodes=nodes), score, performance)

    assert "missing-contrast" in result["failures"]


def test_piece_plan_quality_detects_missing_relations_before_score_generation() -> None:
    plan, _, _ = _fixture()
    nodes = tuple(replace(node, contrasts_with=None) for node in plan.nodes)

    result = evaluate_generic_piece_plan_quality(replace(plan, nodes=nodes))

    assert result == {
        "schema_version": 1,
        "passes": False,
        "failures": ["missing-contrast"],
    }


def test_piece_plan_quality_rejects_leaf_recurrence_with_a_different_material() -> None:
    plan, _, _ = _fixture()
    nodes = tuple(
        replace(node, score_material_id="contrast")
        if node.node_id == "recollected-scene"
        else node
        for node in plan.nodes
    )

    result = evaluate_generic_piece_plan_quality(replace(plan, nodes=nodes))

    assert "unassessable-recurrence:recollected-scene" in result["failures"]


def _hierarchical_recurrence_plan(*, target_material: str, add_transition: bool) -> PiecePlan:
    nodes = [
        PlanNode("root", None, 0, "whole"),
        PlanNode("source", "root", 0, "statement"),
        PlanNode(
            "source-leaf",
            "source",
            0,
            "statement",
            duration_weight=1,
            score_material_id="theme",
        ),
        PlanNode("contrast", "root", 1, "contrast", contrasts_with="source"),
        PlanNode(
            "contrast-leaf",
            "contrast",
            0,
            "statement",
            duration_weight=1,
            score_material_id="contrast",
        ),
        PlanNode("return", "root", 2, "return", derived_from="source"),
        PlanNode(
            "return-leaf",
            "return",
            0,
            "variation",
            duration_weight=1,
            score_material_id=target_material,
        ),
    ]
    if add_transition:
        nodes.append(
            PlanNode(
                "return-transition",
                "return",
                1,
                "transition",
                duration_weight=1,
                score_material_id="new-transition",
            )
        )
    nodes.append(
        PlanNode(
            "ending",
            "root",
            3,
            "release",
            duration_weight=1,
            score_material_id="ending",
        )
    )
    return PiecePlan("nested", "階層回帰", 0, "major", "root", "tonic", tuple(nodes))


def test_piece_plan_quality_requires_same_material_evidence_in_nonleaf_recurrence() -> None:
    result = evaluate_generic_piece_plan_quality(
        _hierarchical_recurrence_plan(target_material="new-theme", add_transition=False)
    )

    assert "unassessable-recurrence:return" in result["failures"]


def test_piece_plan_quality_allows_an_extra_leaf_beside_same_material_evidence() -> None:
    result = evaluate_generic_piece_plan_quality(
        _hierarchical_recurrence_plan(target_material="theme", add_transition=True)
    )

    assert result["passes"] is True
    assert result["failures"] == []


def test_piece_plan_quality_rejects_leaf_to_nonleaf_recurrence() -> None:
    plan = _hierarchical_recurrence_plan(target_material="theme", add_transition=False)
    ending = next(node for node in plan.nodes if node.node_id == "ending")
    nodes = tuple(
        replace(ending, derived_from="source", score_material_id="theme")
        if node.node_id == ending.node_id
        else node
        for node in plan.nodes
    )

    result = evaluate_generic_piece_plan_quality(replace(plan, nodes=nodes))

    assert "unassessable-recurrence:ending" in result["failures"]


def test_piece_plan_quality_rejects_recurrence_on_the_final_leaf() -> None:
    plan, _, _ = _fixture()
    nodes = tuple(
        replace(node, derived_from="first-scene", score_material_id="theme")
        if node.node_id == "tonic-ending"
        else node
        for node in plan.nodes
    )

    result = evaluate_generic_piece_plan_quality(replace(plan, nodes=nodes))

    assert "final-leaf-recurrence:tonic-ending" in result["failures"]


def test_piece_plan_quality_allows_new_material_variation_without_lineage() -> None:
    plan, _, _ = _fixture()
    nodes = tuple(
        replace(node, role="variation") if node.node_id == "different-scene" else node
        for node in plan.nodes
    )

    result = evaluate_generic_piece_plan_quality(replace(plan, nodes=nodes))

    assert result["passes"] is True


def test_generic_quality_rejects_plan_without_recurrence() -> None:
    plan, score, performance = _fixture()
    nodes = tuple(
        replace(node, role="variation", derived_from=None)
        if node.node_id == "recollected-scene"
        else node
        for node in plan.nodes
    )

    result = _evaluate(replace(plan, nodes=nodes), score, performance)

    assert "missing-recurrence" in result["failures"]


def test_generic_quality_rejects_whole_piece_with_one_hand() -> None:
    plan, score, performance = _fixture()
    materials = tuple(
        replace(
            material,
            notes=tuple(note for note in material.notes if note.voice == "upper"),
            harmonies=(),
            foreground_voice=None,
        )
        for material in score.materials
    )
    performance = replace(
        performance,
        node_performances=tuple(
            replace(item, pedal_profile="phrase_legato") for item in performance.node_performances
        ),
    )

    result = _evaluate(plan, replace(score, materials=materials), performance)

    assert "missing-lower-voice" in result["failures"]


def test_generic_quality_rejects_exact_recurrence_and_wrong_duration() -> None:
    plan, score, performance = _fixture()
    common = NodePerformance(
        "first-scene",
        timing_profile="flow",
        timing_amount="subtle",
        dynamics_profile="steady",
        articulation_profile="legato",
        coordination_profile="aligned",
        pedal_profile="harmony_legato",
    )
    items = tuple(
        replace(common, node_id=item.node_id)
        if item.node_id in {"first-scene", "recollected-scene"}
        else item
        for item in performance.node_performances
    )
    changed = replace(performance, target_duration_ms=179_000, node_performances=items)

    result = _evaluate(plan, score, changed)

    assert "duration" in result["failures"]
    assert "exact-recurrence:recollected-scene" in result["failures"]


def test_generic_quality_rejects_low_register_mud() -> None:
    plan, score, performance = _fixture()
    theme, contrast, ending = score.materials
    muddy = replace(
        theme,
        notes=tuple(
            replace(note, pitch=48)
            if note.event_id == "t-l1"
            else replace(note, pitch=45)
            if note.event_id == "t-l0"
            else note
            for note in theme.notes
        ),
    )

    result = _evaluate(plan, replace(score, materials=(muddy, contrast, ending)), performance)

    assert "harmony:theme" in result["failures"]


def test_generic_quality_rejects_unsupported_foreground_tone() -> None:
    plan, score, performance = _fixture()
    theme, contrast, ending = score.materials
    unsupported = replace(
        theme,
        notes=tuple(
            replace(note, pitch=74) if note.event_id == "t-u0" else note for note in theme.notes
        ),
    )

    result = _evaluate(plan, replace(score, materials=(unsupported, contrast, ending)), performance)

    assert "unsupported-dissonance:theme" in result["failures"]
    assert len(result["failures"]) == len(set(result["failures"]))


def test_score_quality_rejects_unsupported_tone_without_performance() -> None:
    plan, score, _ = _fixture()
    theme, contrast, ending = score.materials
    unsupported = replace(
        theme,
        notes=tuple(
            replace(note, pitch=74) if note.event_id == "t-u0" else note for note in theme.notes
        ),
    )

    result = evaluate_generic_score_quality(
        plan, replace(score, materials=(unsupported, contrast, ending))
    )

    assert "unsupported-dissonance:theme" in result["failures"]


def test_score_quality_requires_a_dedicated_single_harmony_ending_material() -> None:
    plan, score, _ = _fixture()
    ending = score.materials[-1]
    reused_nodes = tuple(
        replace(node, score_material_id="theme") if node.node_id == "tonic-ending" else node
        for node in plan.nodes
    )

    result = evaluate_generic_score_quality(replace(plan, nodes=reused_nodes), score)

    assert "final-tonic-dedicated-material" in result["failures"]
    assert result["ending_score"]["dedicated_single_harmony_material"] is False
    assert ending.material_id == "ending"


def test_score_quality_keeps_harmony_less_material_unassessed() -> None:
    plan, score, _ = _fixture()
    theme, contrast, ending = score.materials
    harmony_less = replace(theme, harmonies=(), foreground_voice=None)

    result = evaluate_generic_score_quality(
        plan, replace(score, materials=(harmony_less, contrast, ending))
    )

    assert "theme" in result["unassessed_material_ids"]
    assert all(not failure.endswith(":theme") for failure in result["failures"])


def test_exact_nominal_two_second_ending_can_fail_after_articulation() -> None:
    plan, score, performance = _fixture()
    theme, contrast, ending = score.materials
    long_theme = replace(
        theme,
        length_units=200,
        harmonies=(replace(theme.harmonies[0], duration_units=200),),
    )
    long_contrast = replace(
        contrast,
        length_units=200,
        harmonies=(replace(contrast.harmonies[0], duration_units=200),),
    )
    boundary_ending = replace(
        ending,
        length_units=120,
        notes=tuple(replace(note, at_units=112, duration_units=8) for note in ending.notes),
        harmonies=(replace(ending.harmonies[0], duration_units=120),),
    )
    changed_score = replace(score, materials=(long_theme, long_contrast, boundary_ending))
    changed_performance = replace(
        performance,
        node_performances=tuple(
            replace(item, articulation_profile="light") if item.node_id == "tonic-ending" else item
            for item in performance.node_performances
        ),
    )

    score_result = evaluate_generic_score_quality(plan, changed_score)
    pipeline_result = _evaluate(plan, changed_score, changed_performance)

    assert score_result["ending_score"]["minimum_nominal_duration_ms"] == 2_000
    assert "final-tonic-nominal-duration" not in score_result["failures"]
    assert "final-tonic-note-duration" in pipeline_result["failures"]


def test_generic_quality_rejects_missing_pedal_and_short_final_tonic() -> None:
    plan, score, performance = _fixture()
    rendered = render_performance(plan, score, performance)
    final_attack = max(note.at_ms for note in rendered.notes)
    changed = replace(
        rendered,
        notes=tuple(
            replace(note, duration_ms=1) if note.at_ms == final_attack else note
            for note in rendered.notes
        ),
        pedals=(),
    )

    result = evaluate_generic_pipeline_quality(plan, score, performance, changed)

    assert "final-pedal-release" in result["failures"]
    assert "final-tonic-note-duration" in result["failures"]
    assert "final-tonic-pedal-duration" in result["failures"]


def test_generic_quality_requires_pedal_to_be_down_at_final_attack() -> None:
    plan, score, performance = _fixture()
    rendered = render_performance(plan, score, performance)
    final_attack = max(note.at_ms for note in rendered.notes)
    changed = replace(
        rendered,
        pedals=tuple(
            pedal
            for pedal in rendered.pedals
            if not (pedal.value >= 64 and pedal.at_ms <= final_attack)
        ),
    )

    result = evaluate_generic_pipeline_quality(plan, score, performance, changed)

    assert "final-tonic-pedal-down" in result["failures"]
