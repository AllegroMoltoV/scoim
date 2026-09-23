"""Versioned allocation of score capacity, independent of structural ratios."""

from collections.abc import Mapping
from fractions import Fraction

LEGACY_TIMING = "ratio-units-v1"
QUANTIZED_TIMING = "quantized-score-v1"
# RFC 8785 restricts integer fields to the IEEE-754 safe domain.
MAX_SCORE_UNITS = 2**53 - 1


class ScoreCapacityError(ValueError):
    """The selected capacity cannot represent every structural section."""


def check_timing_contract(value: object) -> str:
    if value not in (LEGACY_TIMING, QUANTIZED_TIMING):
        raise ValueError(f"unsupported score timing contract: {value}")
    return str(value)


def allocate_score_units(
    relative_lengths: Mapping[str, int | float], total_score_units: int
) -> tuple[dict[str, int], dict[str, str]]:
    """Round cumulative boundaries, retaining exact rational error evidence."""
    if (
        isinstance(total_score_units, bool)
        or not isinstance(total_score_units, int)
        or not 1 <= total_score_units <= MAX_SCORE_UNITS
    ):
        raise ScoreCapacityError("total_score_units must be a positive JSON-safe integer")
    weights = {key: Fraction(str(value)) for key, value in relative_lengths.items()}
    if not weights or any(value <= 0 for value in weights.values()):
        raise ScoreCapacityError("section relative lengths must be positive")
    whole = sum(weights.values())
    cumulative = Fraction(0)
    previous = 0
    lengths: dict[str, int] = {}
    evidence: dict[str, str] = {}
    for section_id, weight in weights.items():
        ideal_start = total_score_units * cumulative / whole
        cumulative += weight
        ideal_end = total_score_units * cumulative / whole
        end = round(ideal_end)
        if end <= previous:
            raise ScoreCapacityError(
                f"section {section_id} rounds to zero units; "
                f"ideal_length={ideal_end - ideal_start}; total_score_units={total_score_units}"
            )
        lengths[section_id] = end - previous
        evidence[section_id] = (
            f"total_score_units={total_score_units}; rounding=nearest-even; "
            f"ideal_start={ideal_start}; ideal_end={ideal_end}; "
            f"start={previous}; end={end}; "
            f"start_error={previous - ideal_start}; end_error={end - ideal_end}"
        )
        previous = end
    return lengths, evidence
