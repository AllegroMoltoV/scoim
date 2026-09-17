from __future__ import annotations

from copy import deepcopy

import pytest

from llm_musical_composer.generation_intent import (
    GenerationIntentError,
    candidate_use_allows_promotion,
    creative_targets_for_stage,
    creative_targets_for_step,
    creative_targets_sha256,
    merge_generation_intent,
    validate_generation_assessment,
)


def _target() -> dict[str, object]:
    return {
        "target_version": "reference-generation-target-v7",
        "controls": {},
        "semantic_targets": {
            "piece_plan": [],
            "score_spec": [
                {
                    "id": "attack_texture",
                    "kind": "distribution",
                    "generation_stage": "texture_collection",
                    "instructions": ["参照分布へ近づける"],
                    "reachability": {"status": "unverified"},
                }
            ],
            "performance_spec": [],
            "rendered_surface": [],
        },
    }


def _stage_target(
    stage: str,
    observable_id: str,
    *,
    operator: str = "shape",
    value: object | None = None,
    verification: str = "listening_only",
) -> dict[str, object]:
    observable: dict[str, object] = {
        "observable_id": observable_id,
        "scope": "whole_piece",
        "operator": operator,
    }
    if value is not None:
        observable["value"] = value
    return {
        "generation_stage": stage,
        "instructions": [f"{observable_id}を実現する"],
        "observable": observable,
        "verification": verification,
    }


def _intent() -> dict[str, object]:
    return {
        "schema_version": 1,
        "decisions": [
            {
                "decision_id": "late_climax",
                "statement": "後半に複数要因による山場を作る",
                "decision_kind": "long_form_climax",
                "stage_targets": [
                    _stage_target("piece_plan", "climax_location", value="late"),
                    _stage_target("harmonic_skeleton", "climax_harmonic_tension"),
                    _stage_target("melody_collection", "climax_melodic_register"),
                    _stage_target("texture_collection", "climax_texture"),
                    _stage_target("performance_spec", "climax_performance_shape"),
                ],
            },
            {
                "decision_id": "harmonic_contrast",
                "statement": "区分ごとの和声進行を区別する",
                "decision_kind": "section_harmonic_contrast",
                "stage_targets": [
                    _stage_target(
                        "harmonic_skeleton",
                        "section_harmonic_progression",
                        operator="differentiate",
                    )
                ],
            },
            {
                "decision_id": "melody_rhythm",
                "statement": "旋律に音価と休止の役割差を作る",
                "decision_kind": "melody_phrase_variety",
                "stage_targets": [
                    _stage_target(
                        "melody_collection",
                        "melody_duration_and_rest_roles",
                        operator="differentiate",
                        verification="mechanical",
                    )
                ],
            },
            {
                "decision_id": "middle_register",
                "statement": "低い伴奏と高い旋律の間に中間音域を使う",
                "decision_kind": "register_balance",
                "stage_targets": [
                    _stage_target(
                        "melody_collection",
                        "middle_register_melodic_support",
                        operator="include",
                    ),
                    _stage_target(
                        "texture_collection",
                        "middle_register_accompaniment_support",
                        operator="include",
                    ),
                ],
            },
        ],
    }


def test_long_form_climax_cannot_be_assigned_to_performance_only() -> None:
    intent = _intent()
    decision = intent["decisions"][0]
    assert isinstance(decision, dict)
    decision["stage_targets"] = [
        _stage_target("performance_spec", "climax_performance_shape")
    ]

    with pytest.raises(GenerationIntentError, match="missing required stages"):
        merge_generation_intent(_target(), intent)


def test_all_review_decisions_are_expanded_to_responsible_stages() -> None:
    merged = merge_generation_intent(_target(), _intent())

    expected = {
        "piece_plan": {"late_climax"},
        "harmonic_skeleton": {"late_climax", "harmonic_contrast"},
        "melody_collection": {"late_climax", "melody_rhythm", "middle_register"},
        "texture_collection": {"late_climax", "middle_register"},
        "performance_spec": {"late_climax"},
    }
    for stage, decision_ids in expected.items():
        targets = creative_targets_for_stage(merged, stage)
        assert {item["creative_decision_id"] for item in targets} == decision_ids
        assert creative_targets_sha256(merged, stage) == creative_targets_sha256(
            deepcopy(merged), stage
        )


def test_split_execution_steps_receive_their_logical_stage_targets() -> None:
    merged = merge_generation_intent(_target(), _intent())

    assert creative_targets_for_step(merged, "melody-collection-main") == (
        creative_targets_for_stage(merged, "melody_collection")
    )
    assert creative_targets_for_step(merged, "melody-collection-transition") == (
        creative_targets_for_stage(merged, "melody_collection")
    )
    assert creative_targets_for_step(merged, "texture-collection-batch-001") == (
        creative_targets_for_stage(merged, "texture_collection")
    )
    assert creative_targets_for_step(merged, "texture-collection-batch-006") == (
        creative_targets_for_stage(merged, "texture_collection")
    )


def test_generation_intent_cannot_shadow_reference_observable() -> None:
    intent = _intent()
    decision = intent["decisions"][1]
    assert isinstance(decision, dict)
    decision["stage_targets"] = [
        _stage_target("performance_spec", "attack_texture", operator="increase")
    ]
    decision["decision_kind"] = "performance_interpretation"

    with pytest.raises(GenerationIntentError, match="reference observable"):
        merge_generation_intent(_target(), intent)


def test_confirmed_assessment_requires_output_and_evaluator_provenance() -> None:
    target = creative_targets_for_stage(
        merge_generation_intent(_target(), _intent()), "melody_collection"
    )[0]

    with pytest.raises(GenerationIntentError, match="stage output SHA-256"):
        validate_generation_assessment(
            target,
            {"target_id": target["id"], "status": "achieved"},
        )


def test_delivery_only_target_cannot_be_marked_achieved() -> None:
    target = deepcopy(
        creative_targets_for_stage(
            merge_generation_intent(_target(), _intent()), "melody_collection"
        )[0]
    )
    target["verification"] = "delivery_only"
    assessment = {
        "target_id": target["id"],
        "status": "achieved",
        "stage_output_sha256": "a" * 64,
        "evaluation_input_sha256": "b" * 64,
        "evaluator_id": "test-evaluator",
        "evaluator_version": "1",
        "observation": {"value": True},
    }

    with pytest.raises(GenerationIntentError, match="delivery-only"):
        validate_generation_assessment(target, assessment)


def test_unrepresentable_assessment_uses_input_ir_provenance() -> None:
    target = creative_targets_for_stage(
        merge_generation_intent(_target(), _intent()), "melody_collection"
    )[0]
    record = {
        "target_id": target["id"],
        "status": "unrepresentable",
        "input_sha256": "c" * 64,
        "ir_version": "MelodyCollectionV0",
        "evaluator_id": "capability-check",
        "evaluator_version": "1",
        "reason": "現行語彙では表せない",
    }

    validate_generation_assessment(target, record)


def test_feasibility_only_candidate_cannot_be_promoted() -> None:
    assert candidate_use_allows_promotion("external_generation") is True
    assert candidate_use_allows_promotion("verified_fixture") is True
    assert candidate_use_allows_promotion("feasibility_only") is False
