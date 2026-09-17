from __future__ import annotations

from llm_musical_composer.harmonic_skeleton import HarmonicMaterialV0, HarmonicSkeletonV0
from llm_musical_composer.performance_pipeline import PiecePlan, PlanNode, ScoreHarmony
from llm_musical_composer.tonal_hierarchy import (
    evaluate_brightness_resolution,
    evaluate_harmonic_skeleton_tonal_hierarchy,
    evaluate_piece_plan_tonal_hierarchy,
)


def _plan(*, mode: str = "minor") -> PiecePlan:
    return PiecePlan(
        "tonal-test",
        "tonal test",
        9,
        mode,
        "root",
        "tonic",
        (
            PlanNode("root", None, 0, "whole"),
            PlanNode("statement", "root", 0, "statement", duration_weight=1, score_material_id="s"),
            PlanNode("contrast", "root", 1, "contrast", duration_weight=1, score_material_id="c"),
            PlanNode("return", "root", 2, "return", duration_weight=1, score_material_id="r"),
            PlanNode("release", "root", 3, "release", duration_weight=1, score_material_id="z"),
        ),
    )


def _material(material_id: str, harmonies: tuple[tuple[int, str], ...]) -> HarmonicMaterialV0:
    duration = 12
    return HarmonicMaterialV0(
        material_id,
        duration * len(harmonies),
        None,
        tuple(
            ScoreHarmony(f"{material_id}-{index}", index * duration, duration, root, quality)
            for index, (root, quality) in enumerate(harmonies)
        ),
    )


def _skeleton(
    *,
    outside_scale: bool = False,
    missing_return_iv: bool = False,
    bad_ending: bool = False,
):
    iv = (2, "major")
    return HarmonicSkeletonV0(
        "tonal-test-score",
        12,
        (
            _material("s", ((9, "minor"), iv)),
            _material("c", ((0, "major"), iv)),
            _material(
                "r",
                (
                    ((1, "major") if outside_scale else (7, "major")),
                    ((7, "major") if missing_return_iv else iv),
                ),
            ),
            _material("z", (((9, "major") if bad_ending else (9, "minor")),)),
        ),
    )


def _target(status: str = "specified") -> dict[str, object]:
    return {
        "id": "tonal_hierarchy",
        "status": status,
        "plan_mode": "minor",
        "scale_policy": "dorian",
        "allowed_harmonies": [
            {"root_degree": 0, "quality": "minor"},
            {"root_degree": 2, "quality": "minor"},
            {"root_degree": 3, "quality": "major"},
            {"root_degree": 5, "quality": "major"},
            {"root_degree": 7, "quality": "minor"},
            {"root_degree": 9, "quality": "diminished"},
            {"root_degree": 10, "quality": "major"},
        ],
        "characteristic_harmony": {
            "required_roles": ["statement", "contrast", "return"],
            "root_degree": 5,
            "quality": "major",
            "minimum_per_role": 1,
        },
        "ending": {"root_degree": 0, "quality": "minor"},
    }


def test_dorian_piece_plan_and_harmony_skeleton_pass() -> None:
    plan = _plan()

    assert evaluate_piece_plan_tonal_hierarchy(plan, _target())["passes"]
    result = evaluate_harmonic_skeleton_tonal_hierarchy(plan, _skeleton(), _target())

    assert result["passes"]
    assert result["outside_harmonies"] == []
    assert result["characteristic_counts"] == {"statement": 1, "contrast": 1, "return": 1}


def test_piece_plan_mode_mismatch_fails_before_score_generation() -> None:
    result = evaluate_piece_plan_tonal_hierarchy(_plan(mode="major"), _target())

    assert not result["passes"]
    assert result["failures"] == ["plan-mode"]


def test_harmony_outside_mode_missing_characteristic_and_bad_ending_are_separate() -> None:
    plan = _plan()
    result = evaluate_harmonic_skeleton_tonal_hierarchy(
        plan,
        _skeleton(outside_scale=True, missing_return_iv=True, bad_ending=True),
        _target(),
    )

    assert not result["passes"]
    assert set(result["failures"]) == {
        "outside-harmony",
        "characteristic-harmony:return",
        "ending-harmony",
    }
    assert len(result["outside_harmonies"]) == 1


def test_unverified_reference_target_is_not_treated_as_a_hard_rule() -> None:
    target = _target("unverified_continuous_reference")

    assert evaluate_piece_plan_tonal_hierarchy(_plan(mode="major"), target) == {
        "status": "not_applicable",
        "passes": True,
        "failures": [],
    }
    assert evaluate_harmonic_skeleton_tonal_hierarchy(
        _plan(mode="major"), _skeleton(bad_ending=True), target
    ) == {
        "status": "not_applicable",
        "passes": True,
        "failures": [],
    }


def test_rendered_brightness_outside_middle_band_is_not_reached() -> None:
    target = {
        "status": "specified",
        "requested": 0,
        "target": {
            "normalized": 0.0,
            "acceptable_normalized_range": [-0.5, 0.5],
        },
    }

    achieved = evaluate_brightness_resolution(-0.34, target)
    missed = evaluate_brightness_resolution(0.51, target)

    assert achieved == {
        "status": "achieved",
        "passes": True,
        "requested": 0,
        "target_normalized": 0.0,
        "achieved_normalized": -0.34,
        "normalized_residual": -0.34,
        "acceptable_normalized_range": [-0.5, 0.5],
    }
    assert missed["status"] == "not_reached"
    assert not missed["passes"]
