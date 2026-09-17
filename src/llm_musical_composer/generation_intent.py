"""生成前方針を既存の段階別semantic targetへ決定的に統合する。"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from llm_musical_composer.run_state import sha256_json


class GenerationIntentError(ValueError):
    """生成前方針の構造、担当段階または来歴が不正である。"""


_STAGES = frozenset(
    {
        "piece_plan",
        "harmonic_skeleton",
        "melody_collection",
        "texture_collection",
        "performance_spec",
    }
)
_REQUIRED_STAGES = {
    "long_form_climax": _STAGES,
    "section_harmonic_contrast": frozenset({"harmonic_skeleton"}),
    "melody_phrase_variety": frozenset({"melody_collection"}),
    "register_balance": frozenset({"melody_collection", "texture_collection"}),
    "performance_interpretation": frozenset({"performance_spec"}),
}
_SEMANTIC_BUCKET = {
    "piece_plan": "piece_plan",
    "harmonic_skeleton": "score_spec",
    "melody_collection": "score_spec",
    "texture_collection": "score_spec",
    "performance_spec": "performance_spec",
}
_VERIFICATION_KINDS = frozenset({"mechanical", "listening_only", "delivery_only"})
_OPERATORS = frozenset(
    {"include", "avoid", "increase", "decrease", "differentiate", "shape"}
)
_SAFE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _objects(value: object, label: str) -> list[Mapping[str, object]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(not isinstance(item, Mapping) for item in value)
    ):
        raise GenerationIntentError(f"{label} must be a non-empty object list")
    return list(value)


def _strings(value: object, label: str) -> list[str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise GenerationIntentError(f"{label} must be a non-empty string list")
    return list(value)


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise GenerationIntentError(f"{label} is invalid")
    return value


def _semantic_target_lists(
    prompt_target: Mapping[str, object],
) -> dict[str, list[dict[str, Any]]]:
    raw = prompt_target.get("semantic_targets")
    if not isinstance(raw, Mapping):
        raise GenerationIntentError("semantic targets are missing")
    result: dict[str, list[dict[str, Any]]] = {}
    for bucket, items in raw.items():
        if not isinstance(bucket, str):
            raise GenerationIntentError("semantic target bucket is invalid")
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            raise GenerationIntentError(f"semantic target bucket is invalid: {bucket}")
        if any(not isinstance(item, Mapping) for item in items):
            raise GenerationIntentError(f"semantic target entry is invalid: {bucket}")
        result[bucket] = [dict(item) for item in items]
    for bucket in set(_SEMANTIC_BUCKET.values()):
        result.setdefault(bucket, [])
    return result


def _existing_identifiers(
    semantic_targets: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[set[str], set[str]]:
    target_ids: set[str] = set()
    observable_ids: set[str] = set()
    for items in semantic_targets.values():
        for item in items:
            target_id = item.get("id")
            if isinstance(target_id, str):
                if target_id in target_ids:
                    raise GenerationIntentError(f"semantic target ID is duplicated: {target_id}")
                target_ids.add(target_id)
                observable_ids.add(target_id)
            observable = item.get("observable")
            if isinstance(observable, Mapping):
                observable_id = observable.get("observable_id")
                if isinstance(observable_id, str):
                    observable_ids.add(observable_id)
    return target_ids, observable_ids


def _normalize_observable(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise GenerationIntentError("creative observable must be an object")
    allowed = {"observable_id", "scope", "operator", "value"}
    unknown = set(value) - allowed
    if unknown:
        raise GenerationIntentError(f"creative observable has unknown fields: {sorted(unknown)}")
    observable_id = _identifier(value.get("observable_id"), "observable ID")
    scope = _identifier(value.get("scope"), "observable scope")
    operator = value.get("operator")
    if operator not in _OPERATORS:
        raise GenerationIntentError("observable operator is invalid")
    result: dict[str, object] = {
        "observable_id": observable_id,
        "scope": scope,
        "operator": operator,
    }
    if "value" in value:
        result["value"] = deepcopy(value["value"])
    return result


def merge_generation_intent(
    prompt_target: Mapping[str, object], intent: Mapping[str, object]
) -> dict[str, Any]:
    """創作判断を段階別targetへ展開し、既存prompt targetへ統合する。"""

    if intent.get("schema_version") != 1:
        raise GenerationIntentError("generation intent schema version is invalid")
    decisions = _objects(intent.get("decisions"), "generation intent decisions")
    semantic_targets = _semantic_target_lists(prompt_target)
    target_ids, reference_observable_ids = _existing_identifiers(semantic_targets)
    decision_ids: set[str] = set()
    creative_observables: dict[tuple[str, str], dict[str, object]] = {}

    for decision in decisions:
        decision_id = _identifier(decision.get("decision_id"), "creative decision ID")
        if decision_id in decision_ids:
            raise GenerationIntentError(f"creative decision ID is duplicated: {decision_id}")
        decision_ids.add(decision_id)
        statement = decision.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            raise GenerationIntentError("creative decision statement is missing")
        decision_kind = decision.get("decision_kind")
        required_stages = _REQUIRED_STAGES.get(decision_kind)
        if required_stages is None:
            raise GenerationIntentError("creative decision kind is invalid")
        stage_targets = _objects(decision.get("stage_targets"), "creative stage targets")
        actual_stages = {item.get("generation_stage") for item in stage_targets}
        if not required_stages.issubset(actual_stages):
            missing = sorted(required_stages - actual_stages)
            raise GenerationIntentError(
                f"creative decision is missing required stages: {decision_id}: {missing}"
            )
        if any(stage not in _STAGES for stage in actual_stages):
            raise GenerationIntentError("creative generation stage is invalid")
        if len(actual_stages) != len(stage_targets):
            raise GenerationIntentError(
                f"creative decision repeats a generation stage: {decision_id}"
            )

        for stage_target in stage_targets:
            stage = str(stage_target["generation_stage"])
            verification = stage_target.get("verification")
            if verification not in _VERIFICATION_KINDS:
                raise GenerationIntentError("creative verification kind is invalid")
            instructions = _strings(
                stage_target.get("instructions"), "creative target instructions"
            )
            observable = _normalize_observable(stage_target.get("observable"))
            observable_id = str(observable["observable_id"])
            if observable_id in reference_observable_ids:
                raise GenerationIntentError(
                    f"creative target shadows a reference observable: {observable_id}"
                )
            observable_key = (observable_id, str(observable["scope"]))
            previous = creative_observables.get(observable_key)
            if previous is not None and previous != observable:
                raise GenerationIntentError(
                    f"creative observable has conflicting conditions: {observable_key}"
                )
            if previous is not None:
                raise GenerationIntentError(
                    f"creative observable is duplicated: {observable_key}"
                )
            creative_observables[observable_key] = observable

            target_id = f"creative_{decision_id}_{stage}"
            if target_id in target_ids:
                raise GenerationIntentError(f"creative target ID is duplicated: {target_id}")
            target_ids.add(target_id)
            semantic_targets[_SEMANTIC_BUCKET[stage]].append(
                {
                    "id": target_id,
                    "kind": "creative_intent",
                    "creative_decision_id": decision_id,
                    "decision_kind": decision_kind,
                    "statement": statement,
                    "generation_stage": stage,
                    "instructions": instructions,
                    "observable": observable,
                    "verification": verification,
                    "reachability": {"status": "unverified"},
                }
            )

    result = deepcopy(dict(prompt_target))
    result["semantic_targets"] = semantic_targets
    result["generation_intent"] = {
        "schema_version": 1,
        "source_sha256": sha256_json(intent),
        "decision_ids": list(decision_ids),
    }
    result["generation_intent"]["decision_ids"].sort()
    return result


def creative_targets_for_stage(
    prompt_target: Mapping[str, object], stage: str
) -> list[dict[str, Any]]:
    """論理段階へ渡すcreative targetを保存順で返す。"""

    if stage not in _STAGES:
        raise GenerationIntentError(f"unknown creative target stage: {stage}")
    if "semantic_targets" not in prompt_target:
        if "generation_intent" in prompt_target:
            raise GenerationIntentError("creative semantic targets are missing")
        return []
    semantic_targets = _semantic_target_lists(prompt_target)
    return [
        deepcopy(item)
        for items in semantic_targets.values()
        for item in items
        if item.get("kind") == "creative_intent"
        and item.get("generation_stage") == stage
    ]


def creative_targets_sha256(prompt_target: Mapping[str, object], stage: str) -> str:
    """論理段階へ渡すcreative target一覧の決定的hashを返す。"""

    return sha256_json(creative_targets_for_stage(prompt_target, stage))


def creative_targets_for_step(
    prompt_target: Mapping[str, object], step_id: str
) -> list[dict[str, Any]]:
    """実行stepを論理段階へ解決し、配達対象を返す。"""

    if step_id == "piece-plan":
        stage = "piece_plan"
    elif step_id == "harmonic-collection":
        stage = "harmonic_skeleton"
    elif step_id in {"melody-collection-main", "melody-collection-transition"}:
        stage = "melody_collection"
    elif step_id == "texture-collection" or step_id.startswith(
        "texture-collection-batch-"
    ):
        stage = "texture_collection"
    elif step_id == "performance-collection":
        stage = "performance_spec"
    else:
        raise GenerationIntentError(f"unknown generation step: {step_id}")
    return creative_targets_for_stage(prompt_target, stage)


def candidate_use_allows_promotion(candidate_use: str) -> bool:
    """候補用途が通常成果物への昇格を許すかを返す。"""

    if candidate_use not in {
        "external_generation",
        "verified_fixture",
        "feasibility_only",
    }:
        raise GenerationIntentError(f"unknown candidate use: {candidate_use}")
    return candidate_use != "feasibility_only"


def _require_sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise GenerationIntentError(f"{label} SHA-256 is invalid")


def _require_text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise GenerationIntentError(f"{label} is missing")


def validate_generation_assessment(
    target: Mapping[str, object], assessment: Mapping[str, object]
) -> None:
    """配達状態と、出力または入力へ結び付く評価状態を検証する。"""

    if target.get("kind") != "creative_intent":
        raise GenerationIntentError("assessment target is not creative intent")
    if assessment.get("target_id") != target.get("id"):
        raise GenerationIntentError("assessment target ID does not match")
    status = assessment.get("status")
    if status == "unassessed":
        return
    if status in {"achieved", "unmet"}:
        if target.get("verification") == "delivery_only" and status == "achieved":
            raise GenerationIntentError("delivery-only target cannot be achieved")
        _require_sha256(assessment.get("stage_output_sha256"), "stage output")
        _require_sha256(assessment.get("evaluation_input_sha256"), "evaluation input")
        _require_text(assessment.get("evaluator_id"), "evaluator ID")
        _require_text(assessment.get("evaluator_version"), "evaluator version")
        observation = assessment.get("observation")
        if not isinstance(observation, Mapping) or not observation:
            raise GenerationIntentError("assessment observation is missing")
        return
    if status == "unrepresentable":
        _require_sha256(assessment.get("input_sha256"), "input")
        _require_text(assessment.get("ir_version"), "IR version")
        _require_text(assessment.get("evaluator_id"), "evaluator ID")
        _require_text(assessment.get("evaluator_version"), "evaluator version")
        _require_text(assessment.get("reason"), "unrepresentable reason")
        return
    raise GenerationIntentError("assessment status is invalid")
