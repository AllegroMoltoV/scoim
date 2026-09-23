import math
from fractions import Fraction
from itertools import pairwise

import pytest

from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    _profile_deviation,
    _time_map,
)
from llm_musical_composer.performance_timing import SparseTimeMap


def _performance(
    *nodes: NodePerformance, budget: str = "subtle-v1", duration_ms: int = 180_000
) -> PerformanceSpec:
    return PerformanceSpec("timing-probe", duration_ms, 64, budget, nodes)


def test_neutral_time_is_the_requested_fraction_of_total_duration() -> None:
    timing = SparseTimeMap(120, {}, _performance(), _profile_deviation)

    assert [timing[position] for position in (0, 1, 30, 60, 119, 120)] == [
        0,
        1500,
        45_000,
        90_000,
        178_500,
        180_000,
    ]
    with pytest.raises(IndexError):
        timing[-1]
    with pytest.raises(IndexError):
        timing[121]

    short = SparseTimeMap(120, {}, _performance(duration_ms=3), _profile_deviation)
    rounded_times = [short[position] for position in range(121)]
    assert rounded_times[0] == 0 and rounded_times[-1] == 3
    assert all(left <= right for left, right in pairwise(rounded_times))


@pytest.mark.parametrize(("profile", "sign"), [("build", 1), ("release", -1)])
def test_linear_curve_matches_its_closed_form_integral(profile: str, sign: int) -> None:
    performance = _performance(
        NodePerformance("root", timing_profile=profile, timing_amount="subtle")
    )
    timing = SparseTimeMap(80, {"root": (0, 80)}, performance, _profile_deviation)

    for position in (0, 7, 20, 31, 40, 60, 79, 80):
        x = Fraction(position, 80)
        # The unit-interval factor is 1 +/- (1/20) * (1/2 - x), with total area 1.
        cumulative_area = x + sign * Fraction(1, 40) * (x - x * x)
        assert timing[position] == round(180_000 * cumulative_area)


def test_cosine_savor_matches_its_closed_form_integral() -> None:
    performance = _performance(
        NodePerformance("root", timing_profile="savor", timing_amount="subtle")
    )
    timing = SparseTimeMap(100, {"root": (0, 100)}, performance, _profile_deviation)

    for position in (0, 13, 25, 50, 75, 87, 100):
        x = position / 100
        cumulative_area = x + 0.05 * math.sin(2 * math.pi * x) / (2 * math.pi)
        assert timing[position] == round(180_000 * cumulative_area)


def test_nested_curves_average_only_the_active_directions() -> None:
    performance = _performance(
        NodePerformance("root", timing_profile="build", timing_amount="subtle"),
        NodePerformance("child", timing_profile="release", timing_amount="subtle"),
    )
    timing = SparseTimeMap(
        80, {"root": (0, 80), "child": (20, 60)}, performance, _profile_deviation
    )
    amplitude = Fraction(1, 20)
    left, right = Fraction(1, 4), Fraction(3, 4)

    def root_integral(x: Fraction) -> Fraction:
        return x + amplitude * (x - x * x) / 2

    def middle_integral(x: Fraction) -> Fraction:
        # Within the child, the two deviations average to amplitude * (x - 1/2) / 2.
        return (
            root_integral(left)
            + x
            - left
            + amplitude * ((x - Fraction(1, 2)) ** 2 - Fraction(1, 16)) / 4
        )

    for position in (0, 10, 20, 30, 40, 50, 60, 70, 80):
        x = Fraction(position, 80)
        if x <= left:
            expected_area = root_integral(x)
        elif x <= right:
            expected_area = middle_integral(x)
        else:
            expected_area = middle_integral(right) + root_integral(x) - root_integral(right)
        assert timing[position] == round(180_000 * expected_area)


@pytest.mark.parametrize("budget", ["subtle-v1", "narrative-v1", "narrative-v2"])
def test_scaling_nested_score_coordinates_preserves_all_requested_times(budget: str) -> None:
    performance = _performance(
        NodePerformance("root", timing_profile="savor", timing_amount="moderate"),
        NodePerformance("child", timing_profile="build", timing_amount="subtle"),
        NodePerformance("leaf", timing_profile="release", timing_amount="moderate"),
        budget=budget,
    )
    intervals = {"root": (0, 12), "child": (3, 9), "leaf": (5, 7)}
    reference = SparseTimeMap(12, intervals, performance, _profile_deviation)
    expected = [reference[position] for position in range(13)]
    assert expected[0] == 0
    assert expected[-1] == 180_000
    assert all(left < right for left, right in pairwise(expected))

    for multiplier in (10, 10**15):
        scaled = SparseTimeMap(
            12 * multiplier,
            {
                name: (start * multiplier, end * multiplier)
                for name, (start, end) in intervals.items()
            },
            performance,
            _profile_deviation,
        )
        assert [scaled[position * multiplier] for position in range(13)] == expected


def test_huge_storage_grid_evaluates_only_a_few_requested_positions() -> None:
    total_units = 10**60
    positions = (0, total_units // 4, total_units // 2, 3 * total_units // 4, total_units)
    calls = 0

    def bounded_deviation(profile: str, amount: str, position: float, budget: str) -> float:
        nonlocal calls
        calls += 1
        # One affine curve needs two endpoints per requested integral, including normalization.
        assert calls <= 2 * (len(positions) + 1), "The time map is enumerating the storage grid"
        return _profile_deviation(profile, amount, position, budget)

    timing = SparseTimeMap(
        total_units,
        {"root": (0, total_units)},
        _performance(NodePerformance("root", timing_profile="build", timing_amount="subtle")),
        bounded_deviation,
    )

    actual = [timing[position] for position in positions]
    assert actual == [0, 45_844, 91_125, 135_844, 180_000]
    assert all(left <= right for left, right in pairwise(actual))


def test_sparse_savor_is_scale_invariant_while_legacy_midpoints_remain_grid_dependent() -> None:
    performance = _performance(
        NodePerformance("root", timing_profile="savor", timing_amount="moderate"),
        budget="narrative-v2",
    )
    old_coarse = _time_map(12, {"root": (0, 12)}, performance)
    old_fine = _time_map(120, {"root": (0, 120)}, performance)[::10]
    assert max(abs(left - right) for left, right in zip(old_coarse, old_fine, strict=True)) == 429

    coarse = SparseTimeMap(12, {"root": (0, 12)}, performance, _profile_deviation)
    fine = SparseTimeMap(120, {"root": (0, 120)}, performance, _profile_deviation)
    sparse_times = [coarse[position] for position in range(13)]
    assert sparse_times == [fine[10 * position] for position in range(13)]
    assert sparse_times != old_coarse
