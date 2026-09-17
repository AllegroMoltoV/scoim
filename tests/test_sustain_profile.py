from __future__ import annotations

from dataclasses import replace

import pytest

from llm_musical_composer.composition_ir import Composition, Material, Note, Pedal, Use
from llm_musical_composer.sustain_profile import (
    evaluate_sustain_profile,
    validate_sustain_repair_scope,
)


def _material(*pedals: tuple[int, int], duration_ms: int = 10_000) -> Material:
    return Material(
        material_id="A",
        duration_ms=duration_ms,
        notes=(Note("n1", 0, 500, 60, 70),),
        pedals=tuple(
            Pedal(f"p{index}", at_ms, value) for index, (at_ms, value) in enumerate(pedals, start=1)
        ),
    )


def _composition(material: Material) -> Composition:
    return Composition(
        title="test",
        form=(
            Use("A", role="opening", energy=2),
            Use("A", role="return", energy=2),
            Use("A", role="climax", energy=5),
        ),
        materials=(material,),
        tonal_center=0,
        mode="major",
    )


def test_valid_sustain_profile_reports_held_time_and_repedal_gap() -> None:
    report = evaluate_sustain_profile((_material((0, 127), (3_000, 0), (3_200, 127), (9_800, 0)),))

    assert report["status"] == "pass"
    assert report["issues"] == []
    assert report["materials"] == [
        {
            "material_id": "A",
            "status": "pass",
            "pedal_on_ms": 9_600,
            "pedal_on_ratio": 0.96,
            "repedal_count": 1,
            "repedal_gaps_ms": [200],
            "issues": [],
        }
    ]


def test_values_below_64_are_not_treated_as_sustain() -> None:
    report = evaluate_sustain_profile((_material((0, 48), (3_000, 0), (3_200, 54), (9_800, 0)),))

    material = report["materials"][0]
    assert report["status"] == "fail"
    assert material["pedal_on_ratio"] == 0.0
    assert any("0 or 127" in issue for issue in material["issues"])
    assert any("85%" in issue for issue in material["issues"])


def test_low_coverage_fails_even_when_each_repedal_gap_is_valid() -> None:
    report = evaluate_sustain_profile(
        (
            _material(
                (0, 127),
                (1_000, 0),
                (1_400, 127),
                (2_400, 0),
                (2_800, 127),
                (3_800, 0),
                (4_200, 127),
                (5_200, 0),
                (5_600, 127),
                (9_800, 0),
            ),
        )
    )

    material = report["materials"][0]
    assert material["pedal_on_ratio"] == 0.82
    assert any("85%" in issue for issue in material["issues"])
    assert not any("between 100 and 400" in issue for issue in material["issues"])


def test_long_material_requires_at_least_one_repedal() -> None:
    report = evaluate_sustain_profile((_material((0, 127), (9_800, 0)),))

    assert any("at least one repedal" in issue for issue in report["materials"][0]["issues"])


@pytest.mark.parametrize("gap_ms", [99, 401])
def test_repedal_gap_outside_contract_fails(gap_ms: int) -> None:
    report = evaluate_sustain_profile(
        (_material((0, 127), (3_000, 0), (3_000 + gap_ms, 127), (9_800, 0)),)
    )

    assert any("between 100 and 400" in issue for issue in report["materials"][0]["issues"])


@pytest.mark.parametrize(
    ("pedals", "message"),
    [
        (((100, 127), (3_000, 0), (3_200, 127), (9_800, 0)), "start at 0 ms"),
        (((0, 127), (3_000, 0), (3_200, 127), (9_400, 0)), "final 500 ms"),
        (((0, 127), (3_000, 0), (3_200, 127)), "end with value 0"),
    ],
)
def test_material_must_start_held_and_end_released_near_boundary(
    pedals: tuple[tuple[int, int], ...], message: str
) -> None:
    report = evaluate_sustain_profile((_material(*pedals),))

    assert any(message in issue for issue in report["materials"][0]["issues"])


def test_short_material_does_not_require_repedal() -> None:
    report = evaluate_sustain_profile((_material((0, 127), (5_800, 0), duration_ms=6_000),))

    assert report["status"] == "pass"


def test_sustain_repair_scope_allows_only_pedal_changes() -> None:
    before = _composition(_material((0, 48), (9_800, 0)))
    after = replace(
        before,
        materials=(_material((0, 127), (3_000, 0), (3_200, 127), (9_800, 0)),),
    )

    assert validate_sustain_repair_scope(before, after) == []

    changed_note = replace(
        after,
        materials=(replace(after.materials[0], notes=(Note("n1", 0, 500, 61, 70),)),),
    )
    changed_form = replace(after, form=tuple(reversed(after.form)))
    changed_title = replace(after, title="changed")
    changed_material_set = replace(
        after,
        materials=(*after.materials, replace(after.materials[0], material_id="B")),
    )
    assert "material A notes changed" in validate_sustain_repair_scope(before, changed_note)
    assert "form changed" in validate_sustain_repair_scope(before, changed_form)
    assert "title changed" in validate_sustain_repair_scope(before, changed_title)
    assert validate_sustain_repair_scope(before, changed_material_set) == ["material set changed"]
