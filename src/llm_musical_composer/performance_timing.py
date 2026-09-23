"""Sparse integration of performance curves in normalized score coordinates."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .performance_pipeline import PerformanceSpec

SAVOR_CONTROL_POINTS = (
    (0.00, 1.00),
    (0.22, 0.80),
    (0.30, -1.00),
    (0.36, 0.20),
    (0.62, 1.00),
    (0.69, -0.80),
    (0.76, 0.20),
    (1.00, 1.00),
)

Curve = tuple[Fraction, Fraction, str, str]
Deviation = Callable[[str, str, float, str], float]


class TimingResolutionError(ValueError):
    """A positive score interval is lost at the output time resolution."""


class SparseTimeMap:
    """Evaluate requested integer positions without traversing the storage grid."""

    def __init__(
        self,
        total_units: int,
        intervals: dict[str, tuple[int, int]],
        performance: PerformanceSpec,
        deviation: Deviation,
    ) -> None:
        self.total_units = total_units
        self.target_ms = performance.target_duration_ms
        self.budget = performance.timing_budget_id
        self.deviation = deviation
        self.cache = {0: 0, total_units: self.target_ms}
        curves: list[Curve] = []
        points = {Fraction(0), Fraction(1)}
        for item in performance.node_performances:
            if item.timing_profile is None:
                continue
            left, right = intervals[item.node_id]
            start, end = Fraction(left, total_units), Fraction(right, total_units)
            curves.append((start, end, item.timing_profile, item.timing_amount or "subtle"))
            points.update((start, end))
            if item.timing_profile == "savor" and self.budget == "narrative-v2":
                points.update(
                    start + (end - start) * Fraction(str(x)) for x, _ in SAVOR_CONTROL_POINTS
                )
        self.points = sorted(points)
        self.active: list[tuple[Curve, ...]] = []
        self.cumulative = [0.0]
        for start, end in zip(self.points, self.points[1:], strict=False):
            midpoint = (start + end) / 2
            active = tuple(curve for curve in curves if curve[0] <= midpoint < curve[1])
            self.active.append(active)
            self.cumulative.append(self.cumulative[-1] + self._integral(start, end, active))
        self.area = self.cumulative[-1]

    def _integral(self, start: Fraction, end: Fraction, curves: tuple[Curve, ...]) -> float:
        area = float(end - start)
        if not curves:
            return area
        deviations = []
        for left, right, profile, amount in curves:
            span = right - left
            a, b = float((start - left) / span), float((end - left) / span)
            if profile == "savor" and self.budget not in {"narrative-v1", "narrative-v2"}:
                amplitude = self.deviation(profile, amount, 0.0, self.budget)
                integral = (
                    amplitude
                    * (math.sin(2 * math.pi * b) - math.sin(2 * math.pi * a))
                    / (2 * math.pi)
                )
                deviations.append(float(span) * integral)
            else:
                # All remaining curves are affine on these exact breakpoints.
                deviations.append(
                    area
                    * (
                        self.deviation(profile, amount, a, self.budget)
                        + self.deviation(profile, amount, b, self.budget)
                    )
                    / 2
                )
        # Existing budgets keep every mean factor above 0.5; no clamp is active.
        return area + math.fsum(deviations) / len(curves)

    def __getitem__(self, unit: int) -> int:
        if unit in self.cache:
            return self.cache[unit]
        if not 0 <= unit <= self.total_units:
            raise IndexError("score position is outside the time map")
        position = Fraction(unit, self.total_units)
        index = bisect_right(self.points, position) - 1
        area = self.cumulative[index] + self._integral(
            self.points[index], position, self.active[index]
        )
        result = round(self.target_ms * (area / self.area))
        self.cache[unit] = result
        return result
