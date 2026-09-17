from __future__ import annotations

from fractions import Fraction

from llm_musical_composer.score_timing_hypothesis import GroupingAttack
from llm_musical_composer.score_timing_regime_diagnostics import (
    LeafSpan,
    assess_local_rhythm_profile,
    compare_interval_ratios,
    deduplicate_grouping_profiles,
    partition_leaf_internal_intervals,
    select_observed_vocabulary,
)


def test_partition_excludes_interval_ending_at_next_leaf_start() -> None:
    partition = partition_leaf_internal_intervals(
        positions=(0, 1, 2, 4, 5),
        leaves=(
            LeafSpan("a", "material-a", 0, 2),
            LeafSpan("b", "material-b", 2, 6),
        ),
    )

    assert partition.leaves[0].interval_indices == (0,)
    assert partition.leaves[1].interval_indices == (2, 3)
    assert partition.cross_boundary_indices == (1,)


def test_local_ratio_comparison_does_not_join_independent_scales() -> None:
    first = select_observed_vocabulary((100, 100))
    second = select_observed_vocabulary((300, 600))

    first_comparison = compare_interval_ratios(first.fit.intervals, (1, 1))
    second_comparison = compare_interval_ratios(second.fit.intervals, (1, 2))

    assert first_comparison.status == "equivalent"
    assert second_comparison.status == "equivalent"
    assert first.fit.scale != second.fit.scale


def test_observed_vocabulary_selection_does_not_need_expected_score() -> None:
    selected = select_observed_vocabulary((100, 150, 200, 300))

    assert selected.vocabulary_id == "binary-dotted"
    assert selected.fit.squared_error == 0
    assert selected.fit.normalized_intervals == (2, 3, 4, 6)


def test_identical_grouping_profiles_are_computed_once() -> None:
    groups = (
        GroupingAttack(0, ("a",), (0,)),
        GroupingAttack(100, ("b",), (0,)),
    )

    unique = deduplicate_grouping_profiles(
        {
            "attack-30ms": groups,
            "rolled-merge-60ms": groups,
            "rolled-separate": (
                GroupingAttack(0, ("a",), (0,)),
                GroupingAttack(110, ("b",), (0,)),
            ),
        }
    )

    assert tuple(unique) == ("attack-30ms", "rolled-separate")
    assert unique["attack-30ms"].aliases == (
        "attack-30ms",
        "rolled-merge-60ms",
    )


def test_ratio_comparison_is_invariant_to_global_fraction_scale() -> None:
    comparison = compare_interval_ratios(
        (Fraction(1, 2), Fraction(3, 2), Fraction(1)),
        (2, 6, 4),
    )

    assert comparison.status == "equivalent"
    assert comparison.matching_interval_count == 3


def test_local_leaf_ratios_can_improve_over_one_whole_song_grid() -> None:
    assessment = assess_local_rhythm_profile(
        positions=(0, 1, 2, 4, 8, 9, 10, 12),
        times_us=(0, 100, 200, 400, 900, 1_240, 1_580, 2_260),
        leaves=(
            LeafSpan("a", "material-a", 0, 8),
            LeafSpan("b", "material-b", 8, 13),
        ),
    )

    assert assessment["cross_boundary_count"] == 1
    assert assessment["local_exact_leaf_count"] == 2
    assert assessment["local_matching_interval_count"] > assessment[
        "global_matching_interval_count"
    ]
