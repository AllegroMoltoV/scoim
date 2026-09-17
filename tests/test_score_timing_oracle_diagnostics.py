from __future__ import annotations

from fractions import Fraction
from itertools import combinations, pairwise

from llm_musical_composer.score_timing_oracle_diagnostics import (
    fit_equal_piecewise_time_map,
    fit_piecewise_time_map,
    piecewise_anchor_squared_error,
)


def _brute_boundaries(
    positions: tuple[int, ...],
    times: tuple[int, ...],
    segment_count: int,
) -> tuple[Fraction, tuple[int, ...]]:
    interval_count = len(positions) - 1
    candidates = []
    for interior in combinations(range(2, interval_count - 1), segment_count - 1):
        boundaries = (0, *interior, interval_count)
        if any(right - left < 2 for left, right in pairwise(boundaries)):
            continue
        candidates.append(
            (
                piecewise_anchor_squared_error(positions, times, boundaries),
                boundaries,
            )
        )
    return min(candidates)


def test_piecewise_time_map_dynamic_program_matches_boundary_enumeration() -> None:
    positions = (0, 1, 2, 3, 4, 5, 6, 7)
    times = (0, 100, 200, 360, 520, 680, 780, 880)

    fit = fit_piecewise_time_map(positions, times, requested_segment_count=3)

    expected_error, expected_boundaries = _brute_boundaries(positions, times, 3)
    assert fit.anchor_squared_error == expected_error
    assert fit.boundaries == expected_boundaries
    assert fit.actual_segment_count == 3


def test_piecewise_time_map_reduces_segment_count_to_keep_two_intervals() -> None:
    fit = fit_piecewise_time_map(
        (0, 1, 2, 3, 4),
        (0, 100, 200, 400, 600),
        requested_segment_count=8,
    )

    assert fit.actual_segment_count == 2
    assert fit.boundaries[0] == 0
    assert fit.boundaries[-1] == 4
    assert all(right - left >= 2 for left, right in pairwise(fit.boundaries))
    assert len(fit.boundaries) <= 9


def test_piecewise_time_map_uses_lexicographically_earliest_equal_boundary() -> None:
    fit = fit_piecewise_time_map(
        tuple(range(7)),
        tuple(index * 100 for index in range(7)),
        requested_segment_count=2,
    )

    assert fit.anchor_squared_error == 0
    assert fit.boundaries == (0, 2, 6)


def test_equal_piecewise_time_map_keeps_sparse_deterministic_boundaries() -> None:
    fit = fit_equal_piecewise_time_map(
        tuple(range(11)),
        tuple(index * index for index in range(11)),
        requested_segment_count=4,
    )

    assert fit.boundaries == (0, 2, 5, 7, 10)
    assert fit.actual_segment_count == 4
    assert len(fit.boundaries) == 5
