from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from llm_musical_composer.performance_pipeline import validate_pipeline
from llm_musical_composer.score_timing_hypothesis import GroupingAttack
from llm_musical_composer.timing_vocabulary_fit import (
    build_flat_timing_fixture,
    build_timing_dependency_counterexamples,
    classify_local_coordination,
    compare_timing_shapes,
    enumerate_flat_timing_vocabulary,
    run_timing_vocabulary_fit,
)


def test_flat_fixture_uses_one_node_that_is_both_root_and_leaf() -> None:
    plan, score = build_flat_timing_fixture((0, 25, 50, 100))

    assert len(plan.nodes) == 1
    assert plan.nodes[0].node_id == plan.root_node_id
    assert plan.nodes[0].score_material_id == score.materials[0].material_id
    assert score.materials[0].length_units == 101
    performance = enumerate_flat_timing_vocabulary(plan, score).surface_entries[0].performance
    validate_pipeline(plan, score, performance)


def test_flat_vocabulary_has_30_spellings_and_13_semantic_maps() -> None:
    plan, score = build_flat_timing_fixture((0, 25, 50, 75, 100))

    search = enumerate_flat_timing_vocabulary(plan, score)

    assert search.scope == "flat_single_node_minimum_tail"
    assert len(search.surface_entries) == 30
    assert len(search.semantic_maps) == 13
    assert all(
        entry.performance.node_performances[0].node_id == plan.root_node_id
        for entry in search.surface_entries
    )


def test_timing_distance_is_one_tick_distance_not_event_equivalence() -> None:
    within = compare_timing_shapes((0, 10_000, 20_000), (0, 10, 20))
    outside = compare_timing_shapes((0, 12_001, 20_000), (0, 10, 20))

    assert within.status == "within_one_transport_tick"
    assert within.maximum_absolute_error_us == 0
    assert outside.status == "more_than_one_transport_tick"
    assert outside.maximum_absolute_error_us == 2_001
    assert outside.over_one_transport_tick_count == 1


def test_fixed_counterexamples_expose_topology_and_tail_dependencies() -> None:
    result = build_timing_dependency_counterexamples()

    assert result["topology_dependency"]["status"] == "affirmative_evidence"
    assert result["topology_dependency"]["composed_map_in_flat_semantic_maps"] is False
    assert result["topology_dependency"]["nearest_flat_maximum_error_us"] > 0
    assert result["tail_length_dependency"]["status"] == "affirmative_evidence"
    assert result["tail_length_dependency"]["maximum_normalized_difference"] > 0
    assert result["tail_length_dependency"]["nearest_short_tail_vocabulary_fit"]["status"] == (
        "more_than_one_transport_tick"
    )
    assert (
        result["tail_length_dependency"]["nearest_short_tail_vocabulary_fit"][
            "maximum_absolute_error_us"
        ]
        > 1_000
    )


def test_local_coordination_separates_vertical_rolled_and_unmodeled() -> None:
    vertical = (
        GroupingAttack(0, ("a", "b", "c"), (0, 0, 0)),
        GroupingAttack(1_000_000, ("d",), (0,)),
    )
    rolled = (
        GroupingAttack(0, ("a", "b", "c"), (0, 22_000, 45_000)),
        GroupingAttack(1_000_000, ("d",), (0,)),
    )
    unmodeled = (
        GroupingAttack(0, ("a", "b", "c"), (0, 10_000, 45_000)),
        GroupingAttack(1_000_000, ("d",), (0,)),
    )

    assert classify_local_coordination(vertical) == {
        "status": "representable",
        "profile_semantics": ["aligned", "score"],
        "oracle_assumption": None,
    }
    assert classify_local_coordination(rolled) == {
        "status": "representable_under_pitch_rank_voice_assumption",
        "profile_semantics": ["rolled"],
        "oracle_assumption": "group members follow the renderer low-to-high ordering",
    }
    assert classify_local_coordination(unmodeled)["status"] == "not_represented"


def test_shape_comparison_affinely_aligns_endpoints_but_keeps_internal_difference() -> None:
    reference = (4_000, 14_000, 24_000)
    shifted_and_scaled = (0, 100, 200)
    bent = (0, 80, 200)

    assert compare_timing_shapes(reference, shifted_and_scaled).maximum_absolute_error_us == 0
    assert compare_timing_shapes(reference, bent).maximum_absolute_error_us == 2_000


def test_runner_is_byte_deterministic_and_keeps_upstream_failures_separate(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_result = run_timing_vocabulary_fit(workspace=Path("."), output_dir=first)
    second_result = run_timing_vocabulary_fit(workspace=Path("."), output_dir=second)

    assert first_result == second_result
    assert first_result["status"] == "pass"
    assert first_result["fixture_family_count"] == 4
    assert first_result["full_performance_mapping_status"] == (
        "deferred_until_form_and_tail_duration"
    )
    assert first_result["upstream_score_grid_mismatch_source_count"] >= 1
    assert first_result["grouping_conflict_candidate_count"] > 0
    assert first_result["candidate_count"] == sum(
        first_result[key]
        for key in (
            "flat_assessed_candidate_count",
            "upstream_score_grid_mismatch_candidate_count",
            "grouping_conflict_candidate_count",
        )
    )
    assert first_result["score_grid_recovered_source_count"] == 2
    assert first_result["score_grid_unrecovered_source_count"] == 2
    assert first_result["rolled_local_coordination_candidate_count"] > 0
    assert first_result["unrepresented_local_coordination_candidate_count"] > 0
    assert (
        first_result["root_end_unavailable_candidate_count"]
        == first_result["flat_assessed_candidate_count"]
    )
    first_files = {
        path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()
    }
    second_files = {
        path.relative_to(second): path.read_bytes() for path in second.rglob("*") if path.is_file()
    }
    assert first_files == second_files
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert {
        "pipeline_dsl.py",
        "reference_decomposition.py",
        "reference_timing_hypothesis.py",
        "run_state.py",
    }.issubset(manifest["inputs"])


def test_flat_vocabulary_entry_is_not_mutated_by_callers() -> None:
    plan, score = build_flat_timing_fixture((0, 1, 2))
    search = enumerate_flat_timing_vocabulary(plan, score)
    original = search.surface_entries[0]

    changed = replace(original.performance, performance_id="changed")

    assert original.performance.performance_id != changed.performance_id
