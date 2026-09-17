"""既知楽譜位置に対する疎な区分線形時間写像を診断する。"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise


@dataclass(frozen=True)
class PiecewiseTimeMapFit:
    requested_segment_count: int
    actual_segment_count: int
    boundaries: tuple[int, ...]
    reconstructed_times: tuple[Fraction, ...]
    anchor_squared_error: Fraction
    anchor_mean_absolute_error: Fraction
    anchor_maximum_absolute_error: Fraction


@dataclass(frozen=True)
class _PrefixMoments:
    x: tuple[int, ...]
    y: tuple[int, ...]
    xx: tuple[int, ...]
    yy: tuple[int, ...]
    xy: tuple[int, ...]


def _prefix(values: tuple[int, ...]) -> tuple[int, ...]:
    result = [0]
    for value in values:
        result.append(result[-1] + value)
    return tuple(result)


def _moments(positions: tuple[int, ...], times: tuple[int, ...]) -> _PrefixMoments:
    return _PrefixMoments(
        x=_prefix(positions),
        y=_prefix(times),
        xx=_prefix(tuple(value * value for value in positions)),
        yy=_prefix(tuple(value * value for value in times)),
        xy=_prefix(tuple(x * y for x, y in zip(positions, times, strict=True))),
    )


def _range_sum(values: tuple[int, ...], start: int, end: int) -> int:
    return values[end + 1] - values[start]


def _segment_anchor_squared_error(
    positions: tuple[int, ...],
    times: tuple[int, ...],
    moments: _PrefixMoments,
    start: int,
    end: int,
) -> Fraction:
    slope = Fraction(times[end] - times[start], positions[end] - positions[start])
    intercept = Fraction(times[start]) - slope * positions[start]
    count = end - start + 1
    sum_x = _range_sum(moments.x, start, end)
    sum_y = _range_sum(moments.y, start, end)
    sum_xx = _range_sum(moments.xx, start, end)
    sum_yy = _range_sum(moments.yy, start, end)
    sum_xy = _range_sum(moments.xy, start, end)
    return (
        sum_yy
        - 2 * intercept * sum_y
        - 2 * slope * sum_xy
        + count * intercept * intercept
        + 2 * intercept * slope * sum_x
        + slope * slope * sum_xx
    )


def _validate(positions: tuple[int, ...], times: tuple[int, ...]) -> None:
    if len(positions) != len(times) or len(positions) < 3:
        raise ValueError("positions and times must have equal length >= 3")
    if any(right <= left for left, right in pairwise(positions)):
        raise ValueError("positions must be strictly increasing")
    if any(right < left for left, right in pairwise(times)):
        raise ValueError("times must be non-decreasing")


def _reconstruct(
    positions: tuple[int, ...],
    times: tuple[int, ...],
    boundaries: tuple[int, ...],
) -> tuple[Fraction, ...]:
    reconstructed = [Fraction(0)] * len(times)
    for start, end in pairwise(boundaries):
        slope = Fraction(times[end] - times[start], positions[end] - positions[start])
        for index in range(start, end + 1):
            reconstructed[index] = Fraction(times[start]) + slope * (
                positions[index] - positions[start]
            )
    return tuple(reconstructed)


def piecewise_anchor_squared_error(
    positions: tuple[int, ...],
    times: tuple[int, ...],
    boundaries: tuple[int, ...],
) -> Fraction:
    """指定境界で両端補間した発音アンカーの二乗誤差を返す。"""
    _validate(positions, times)
    if boundaries[0] != 0 or boundaries[-1] != len(positions) - 1:
        raise ValueError("boundaries must cover every position")
    if any(right - left < 2 for left, right in pairwise(boundaries)):
        raise ValueError("every segment must contain at least two intervals")
    reconstructed = _reconstruct(positions, times, boundaries)
    return sum(
        (Fraction(observed) - predicted) ** 2
        for observed, predicted in zip(times, reconstructed, strict=True)
    )


def _fit_from_boundaries(
    positions: tuple[int, ...],
    times: tuple[int, ...],
    requested_segment_count: int,
    boundaries: tuple[int, ...],
) -> PiecewiseTimeMapFit:
    reconstructed = _reconstruct(positions, times, boundaries)
    absolute_errors = tuple(
        abs(Fraction(observed) - predicted)
        for observed, predicted in zip(times, reconstructed, strict=True)
    )
    squared_error = sum(error * error for error in absolute_errors)
    return PiecewiseTimeMapFit(
        requested_segment_count=requested_segment_count,
        actual_segment_count=len(boundaries) - 1,
        boundaries=boundaries,
        reconstructed_times=reconstructed,
        anchor_squared_error=squared_error,
        anchor_mean_absolute_error=sum(absolute_errors) / len(absolute_errors),
        anchor_maximum_absolute_error=max(absolute_errors),
    )


def fit_equal_piecewise_time_map(
    positions: tuple[int, ...],
    times: tuple[int, ...],
    *,
    requested_segment_count: int,
) -> PiecewiseTimeMapFit:
    """区間番号を均等分割した現行対照の時間写像を返す。"""
    _validate(positions, times)
    if requested_segment_count < 1:
        raise ValueError("requested segment count must be positive")
    interval_count = len(positions) - 1
    actual_count = min(requested_segment_count, interval_count // 2)
    boundaries = (
        *(index * interval_count // actual_count for index in range(actual_count)),
        interval_count,
    )
    return _fit_from_boundaries(
        positions,
        times,
        requested_segment_count,
        boundaries,
    )


def fit_piecewise_time_map(
    positions: tuple[int, ...],
    times: tuple[int, ...],
    *,
    requested_segment_count: int,
) -> PiecewiseTimeMapFit:
    """全境界を動的計画法で比較し、アンカー誤差最小の写像を返す。"""
    _validate(positions, times)
    if requested_segment_count < 1:
        raise ValueError("requested segment count must be positive")
    interval_count = len(positions) - 1
    actual_count = min(requested_segment_count, interval_count // 2)
    moments = _moments(positions, times)
    previous: dict[int, tuple[Fraction, tuple[int, ...]]] = {
        end: (
            _segment_anchor_squared_error(positions, times, moments, 0, end),
            (0, end),
        )
        for end in range(2, interval_count + 1)
    }
    for segment_count in range(2, actual_count + 1):
        current: dict[int, tuple[Fraction, tuple[int, ...]]] = {}
        for end in range(2 * segment_count, interval_count + 1):
            candidates = []
            for start in range(2 * (segment_count - 1), end - 1):
                if start not in previous:
                    continue
                previous_error, previous_boundaries = previous[start]
                error = previous_error + _segment_anchor_squared_error(
                    positions,
                    times,
                    moments,
                    start,
                    end,
                )
                candidates.append((error, (*previous_boundaries, end)))
            current[end] = min(candidates)
        previous = current
    error, boundaries = previous[interval_count]
    fit = _fit_from_boundaries(
        positions,
        times,
        requested_segment_count,
        boundaries,
    )
    assert fit.anchor_squared_error == error
    return fit
