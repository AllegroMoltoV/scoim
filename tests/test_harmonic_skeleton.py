from __future__ import annotations

from dataclasses import replace

import pytest

from llm_musical_composer.harmonic_skeleton import (
    HarmonicSkeletonError,
    apply_harmonic_skeleton,
    dump_harmonic_skeleton,
    parse_harmonic_skeleton,
    split_score_spec,
    validate_harmonic_skeleton,
)
from llm_musical_composer.performance_pipeline import (
    PiecePlan,
    PlanNode,
    ScoreDirection,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
)


def _fixture() -> tuple[PiecePlan, ScoreSpec]:
    plan = PiecePlan(
        "plan",
        "和声骨格",
        0,
        "major",
        "whole",
        "tonic",
        (
            PlanNode("whole", None, 0, "whole"),
            PlanNode(
                "first",
                "whole",
                0,
                "statement",
                duration_weight=1,
                score_material_id="base",
            ),
            PlanNode(
                "second",
                "whole",
                1,
                "variation",
                duration_weight=1,
                score_material_id="derived",
            ),
        ),
    )
    score = ScoreSpec(
        "score",
        8,
        (
            ScoreMaterial(
                "base",
                8,
                (ScoreNote("base-note", 0, 8, 60, "upper"),),
            ),
            ScoreMaterial(
                "derived",
                16,
                (
                    ScoreNote("upper", 0, 8, 64, "upper", articulations=("tenuto",)),
                    ScoreNote("lower", 0, 8, 48, "lower"),
                ),
                derived_from="base",
                directions=(ScoreDirection("dynamic", 0, "dynamic", "mp"),),
                harmonies=(
                    ScoreHarmony("h1", 0, 8, 0, "major"),
                    ScoreHarmony("h2", 8, 8, 7, "major-seventh"),
                ),
                foreground_voice="upper",
            ),
        ),
    )
    return plan, score


def test_skeleton_dsl_round_trips_and_contains_no_note_payload() -> None:
    plan, score = _fixture()
    skeleton, _ = split_score_spec(plan, score)

    source = dump_harmonic_skeleton(skeleton)

    assert parse_harmonic_skeleton(source) == skeleton
    assert dump_harmonic_skeleton(parse_harmonic_skeleton(source)) == source
    assert "pitch=" not in source
    assert "foreground_voice=" not in source
    assert "direction" not in source


@pytest.mark.parametrize(
    "source",
    (
        "harmonic_skeleton_v0('score', divisions=8, materials=[])",
        "harmonic_skeleton_v0(score_id='score', divisions=8.0, materials=[])",
        "harmonic_skeleton_v0(score_id='score', divisions=8, materials=[], extra=1)",
        "unknown(score_id='score', divisions=8, materials=[])",
        "harmonic_skeleton_v0(score_id=name, divisions=8, materials=[])",
        "harmonic_skeleton_v0(score_id='score', divisions=8, materials={})",
    ),
)
def test_skeleton_dsl_rejects_out_of_contract_syntax(source: str) -> None:
    with pytest.raises(HarmonicSkeletonError):
        parse_harmonic_skeleton(source)


def test_split_and_apply_are_exact_for_harmonized_and_legacy_materials() -> None:
    plan, score = _fixture()

    skeleton, payload = split_score_spec(plan, score)
    restored = apply_harmonic_skeleton(plan, payload, skeleton)

    assert restored == score
    assert skeleton.materials[0].harmonies == ()
    assert payload.materials[1].material_id == "derived"
    assert payload.materials[1].notes == score.materials[1].notes


def test_projection_ignores_notes_foreground_and_directions() -> None:
    plan, score = _fixture()
    changed = replace(
        score,
        materials=(
            replace(
                score.materials[0],
                notes=(ScoreNote("changed", 1, 4, 72, "lower", articulations=("accent",)),),
            ),
            replace(
                score.materials[1],
                notes=(
                    ScoreNote("changed-upper", 0, 4, 67, "lower"),
                    ScoreNote("changed-lower", 4, 4, 55, "upper"),
                ),
                directions=(ScoreDirection("breath", 8, "breath", "light"),),
                foreground_voice="lower",
            ),
        ),
    )

    original, _ = split_score_spec(plan, score)
    projected, _ = split_score_spec(plan, changed)

    assert projected == original


def test_projection_changes_when_a_skeleton_field_changes() -> None:
    plan, score = _fixture()
    original, _ = split_score_spec(plan, score)
    changed_score = replace(score, divisions=16)

    changed, _ = split_score_spec(plan, changed_score)

    assert changed != original


def test_validation_preserves_dependency_order_not_piece_playback_order() -> None:
    plan, score = _fixture()
    skeleton, _ = split_score_spec(plan, score)
    reversed_materials = replace(skeleton, materials=tuple(reversed(skeleton.materials)))

    with pytest.raises(HarmonicSkeletonError, match="derived_from"):
        validate_harmonic_skeleton(plan, reversed_materials)


def test_apply_rejects_payload_material_mismatch() -> None:
    plan, score = _fixture()
    skeleton, payload = split_score_spec(plan, score)
    incomplete = replace(payload, materials=payload.materials[:1])

    with pytest.raises(HarmonicSkeletonError, match="material IDs"):
        apply_harmonic_skeleton(plan, incomplete, skeleton)
