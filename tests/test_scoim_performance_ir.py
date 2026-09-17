from dataclasses import replace

import pytest

from scoim.performance_ir import (
    PerformanceIrValidationError,
    PerformanceSpec,
    SectionPerformance,
    validate_performance_spec,
)
from scoim.profile_capabilities import PerformanceChoice, solo_piano_3m_v2_capabilities
from scoim.score_ir import PiecePlan, PlanNode


def test_performance_spec_accepts_directions_for_existing_sections() -> None:
    plan = PiecePlan(
        "plan-001",
        "小さな曲",
        0,
        "major",
        "whole",
        (
            PlanNode("whole", None, 0),
            PlanNode("statement", "whole", 0, 1, "score-unit-statement"),
        ),
    )
    performance = PerformanceSpec(
        performance_id="performance-001",
        target_duration_ms=180_000,
        default_velocity=64,
        timing_budget_id="timing-v1",
        section_performances=(
            SectionPerformance(
                section_id="whole",
                timing_profile="savor",
                timing_amount="moderate",
                dynamics_profile="steady",
                articulation_profile="score",
                coordination_profile="score",
                pedal_profile="harmony_legato",
            ),
        ),
    )

    validate_performance_spec(plan, performance)


def test_performance_spec_rejects_an_unknown_section() -> None:
    plan = PiecePlan(
        "plan-001",
        "小さな曲",
        0,
        "major",
        "whole",
        (
            PlanNode("whole", None, 0),
            PlanNode("statement", "whole", 0, 1, "score-unit-statement"),
        ),
    )
    performance = PerformanceSpec(
        "performance-001",
        180_000,
        64,
        "timing-v1",
        (SectionPerformance("missing"),),
    )

    with pytest.raises(PerformanceIrValidationError, match="unknown section"):
        validate_performance_spec(plan, performance)


def test_performance_spec_rejects_invalid_timing_metadata() -> None:
    plan = PiecePlan(
        "plan-001",
        "小さな曲",
        0,
        "major",
        "whole",
        (
            PlanNode("whole", None, 0),
            PlanNode("statement", "whole", 0, 1, "score-unit-statement"),
        ),
    )
    performance = PerformanceSpec(
        "performance-001",
        180_000,
        64,
        "timing-v1",
        (SectionPerformance("whole", timing_profile="unknown"),),
    )

    with pytest.raises(PerformanceIrValidationError, match="profile vocabulary"):
        validate_performance_spec(plan, performance)


def test_performance_spec_rejects_invalid_numeric_metadata() -> None:
    plan = PiecePlan(
        "plan-001",
        "小さな曲",
        0,
        "major",
        "whole",
        (PlanNode("whole", None, 0, 1, "score-unit-whole"),),
    )
    performance = PerformanceSpec("performance-001", 0, 64, "subtle-v1", ())

    with pytest.raises(PerformanceIrValidationError, match="metadata"):
        validate_performance_spec(plan, performance)


def test_performance_spec_rejects_a_non_integer_target_duration() -> None:
    plan = PiecePlan(
        "plan-001",
        "小さな曲",
        0,
        "major",
        "whole",
        (PlanNode("whole", None, 0, 1, "score-unit-whole"),),
    )
    performance = PerformanceSpec("performance-001", 180_000.0, 64, "subtle-v1", ())

    with pytest.raises(PerformanceIrValidationError, match="metadata"):
        validate_performance_spec(plan, performance)


def test_performance_spec_rejects_duplicate_section_directions() -> None:
    plan = PiecePlan(
        "plan-001",
        "小さな曲",
        0,
        "major",
        "whole",
        (PlanNode("whole", None, 0, 1, "score-unit-whole"),),
    )
    performance = PerformanceSpec(
        "performance-001",
        180_000,
        64,
        "subtle-v1",
        (SectionPerformance("whole"), SectionPerformance("whole")),
    )

    with pytest.raises(PerformanceIrValidationError, match="section IDs"):
        validate_performance_spec(plan, performance)


def test_performance_spec_uses_the_generation_profile_vocabulary() -> None:
    plan = PiecePlan(
        "plan-001",
        "小さな曲",
        0,
        "major",
        "whole",
        (PlanNode("whole", None, 0, 1, "score-unit-whole"),),
    )
    base = solo_piano_3m_v2_capabilities()
    timing = base.performance_aspects[0]
    timing_profile = timing.fields[0]
    capabilities = replace(
        base,
        performance_aspects=(
            replace(
                timing,
                fields=(
                    replace(
                        timing_profile,
                        choices=(
                            *timing_profile.choices,
                            PerformanceChoice("elastic", "大きく息をする。"),
                        ),
                    ),
                    *timing.fields[1:],
                ),
            ),
            *base.performance_aspects[1:],
        ),
    )
    performance = PerformanceSpec(
        "performance-001",
        180_000,
        64,
        "subtle-v1",
        (SectionPerformance("whole", timing_profile="elastic", timing_amount="subtle"),),
    )

    validate_performance_spec(plan, performance, capabilities)
