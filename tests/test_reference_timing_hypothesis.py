from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    PiecePlan,
    PlanNode,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    render_performance,
    render_performance_smf,
)
from llm_musical_composer.reference_decomposition import (
    build_observed_performance,
    load_observed_smf,
)
from llm_musical_composer.reference_timing_hypothesis import (
    align_known_score_note_events,
    align_known_score_timing,
    assess_sequential_performance_stage,
    build_timing_control_artifacts,
    monotonic_score_grid_family_member,
    monotonic_time_explanation,
    timing_candidate_equivalence,
    timing_semantic_fingerprint,
)


def _fixture() -> tuple[PiecePlan, ScoreSpec, PerformanceSpec]:
    plan = PiecePlan(
        plan_id="timing-fixture",
        title="timing fixture",
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
                duration_weight=1,
                score_material_id="material",
            ),
        ),
    )
    score = ScoreSpec(
        score_id="timing-score",
        divisions=4,
        materials=(
            ScoreMaterial(
                material_id="material",
                length_units=8,
                notes=(
                    ScoreNote("low-1", 0, 2, 48, "lower"),
                    ScoreNote("high-1", 0, 2, 64, "upper"),
                    ScoreNote("low-2", 2, 2, 52, "lower"),
                    ScoreNote("high-2", 2, 2, 67, "upper"),
                    ScoreNote("ending-low", 4, 4, 48, "lower"),
                    ScoreNote("ending-high", 4, 4, 60, "upper"),
                ),
            ),
        ),
    )
    performance = PerformanceSpec(
        performance_id="timing-performance",
        target_duration_ms=8_000,
        default_velocity=64,
        timing_budget_id="narrative-v1",
        node_performances=(
            NodePerformance(
                "root",
                timing_profile="neutral",
                timing_amount="subtle",
                coordination_profile="score",
            ),
        ),
    )
    return plan, score, performance


def test_known_timing_vocabulary_has_semantically_equivalent_spellings() -> None:
    plan, score, base = _fixture()
    score_coordination = base
    aligned_coordination = replace(
        base,
        node_performances=(replace(base.node_performances[0], coordination_profile="aligned"),),
    )
    neutral_moderate = replace(
        base,
        node_performances=(replace(base.node_performances[0], timing_amount="moderate"),),
    )
    root_savor = replace(
        base,
        node_performances=(
            replace(
                base.node_performances[0],
                timing_profile="savor",
                timing_amount="moderate",
            ),
        ),
    )

    candidates = {
        "score": score_coordination,
        "aligned": aligned_coordination,
        "neutral-moderate": neutral_moderate,
        "root-savor-v1": root_savor,
    }
    equivalence = timing_candidate_equivalence(plan, score, candidates)

    assert len(set(equivalence.semantic_fingerprints.values())) == 1
    assert equivalence.surface_candidate_count == 4
    assert equivalence.semantic_candidate_count == 1
    assert equivalence.status == "identifiable_within_candidate_space"
    assert len(equivalence.candidate_space_sha256) == 64


def test_different_local_score_ratios_can_explain_the_same_observed_times() -> None:
    observed_times = (0, 1_000, 2_000, 4_000)

    first = monotonic_time_explanation((0, 1, 3, 4), observed_times)
    second = monotonic_time_explanation((0, 2, 3, 4), observed_times)

    assert first.reconstructed_times == observed_times
    assert second.reconstructed_times == observed_times
    assert first.local_score_intervals != second.local_score_intervals
    assert first.segment_slopes != second.segment_slopes


def test_known_score_alignment_separates_shared_time_and_rolled_residuals(
    tmp_path: Path,
) -> None:
    plan, score, base = _fixture()
    rolled = replace(
        base,
        node_performances=(replace(base.node_performances[0], coordination_profile="rolled"),),
    )
    smf = tmp_path / "rolled.mid"
    render_performance_smf(render_performance(plan, score, rolled), smf)
    observed_smf = load_observed_smf(smf)
    observed_performance = build_observed_performance(observed_smf)

    evidence = align_known_score_timing(plan, score, observed_smf, observed_performance)

    assert evidence.status == "assessed"
    assert evidence.note_on_status == "assessed"
    assert evidence.note_off_status == "observed_not_attributed"
    assert evidence.cc64_status == "observed_not_attributed"
    assert len(evidence.attack_groups) == 3
    assert all(
        group.coordination_status == "rolled_v1_candidate" for group in evidence.attack_groups
    )
    assert all(group.invariant_spread_us == 45_000 for group in evidence.attack_groups)
    assert all(min(group.minimum_anchor_residuals_us) == 0 for group in evidence.attack_groups)
    assert all(
        group.minimum_anchor_us <= group.median_anchor_us for group in evidence.attack_groups
    )


def test_known_score_note_alignment_exposes_the_shared_event_mapping(
    tmp_path: Path,
) -> None:
    plan, score, base = _fixture()
    smf = tmp_path / "aligned.mid"
    render_performance_smf(render_performance(plan, score, base), smf)
    observed_smf = load_observed_smf(smf)
    observed_performance = build_observed_performance(observed_smf)

    alignment = align_known_score_note_events(plan, score, observed_performance)
    timing = align_known_score_timing(plan, score, observed_smf, observed_performance)

    assert alignment.status == "assessed"
    assert alignment.unmatched_expected_count == 0
    assert alignment.unmatched_observed_count == 0
    assert len(alignment.matches) == 6
    assert {match.expected.voice for match in alignment.matches} == {"lower", "upper"}
    assert {match.observed.note_on_event_id for match in alignment.matches} == {
        event_id for group in timing.attack_groups for event_id in group.evidence_event_ids
    }


def test_timing_fingerprint_changes_when_rolled_timing_changes() -> None:
    plan, score, base = _fixture()
    rolled = replace(
        base,
        node_performances=(replace(base.node_performances[0], coordination_profile="rolled"),),
    )

    assert timing_semantic_fingerprint(plan, score, base) != timing_semantic_fingerprint(
        plan, score, rolled
    )


def test_note_on_alignment_stays_assessed_when_note_off_pairing_is_ambiguous(
    tmp_path: Path,
) -> None:
    plan, score, base = _fixture()
    smf = tmp_path / "ambiguous-off.mid"
    rendered = render_performance(plan, score, base)
    render_performance_smf(rendered, smf)
    observed_smf = load_observed_smf(smf)
    observed_performance = replace(
        build_observed_performance(observed_smf),
        note_matching_status="ambiguous",
        ambiguous_note_count=1,
    )

    evidence = align_known_score_timing(plan, score, observed_smf, observed_performance)

    assert evidence.note_on_status == "assessed"
    assert evidence.note_off_status == "ambiguous_not_attributed"


def test_control_artifacts_separate_surface_collisions_from_score_time_ambiguity() -> None:
    controls = build_timing_control_artifacts()

    collisions = controls["finite_vocabulary_collisions"]
    non_identifiability = controls["non_identifiability"]
    assert collisions["surface_candidate_count"] == 4
    assert collisions["semantic_candidate_count"] == 1
    assert collisions["status"] == "identifiable_within_candidate_space"
    assert non_identifiability["same_observed_times"]
    assert non_identifiability["different_local_score_intervals"]
    assert non_identifiability["conclusion"] == "equivalent_candidates"


def test_score_unknown_time_mapping_has_an_unbounded_score_grid_family() -> None:
    near = monotonic_score_grid_family_member(3)
    far = monotonic_score_grid_family_member(100)

    assert near.reconstructed_times == far.reconstructed_times
    assert near.local_score_intervals != far.local_score_intervals


def test_performance_first_stage_is_rejected_by_reuse_criteria() -> None:
    assessment = assess_sequential_performance_stage()

    assert assessment["status"] == "sequential_performance_first_not_supported"
    assert assessment["score_unknown"]["candidate_set"] == "unbounded"
    assert assessment["score_unknown"]["reconstructable"] is False
    assert assessment["score_unknown"]["component_swappable"] is False
    assert assessment["recommendation"] == "coupled_score_timing_hypothesis"
