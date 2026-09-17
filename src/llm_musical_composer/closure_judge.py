"""Claude Sonnet 4.6 による局所的な終止感の未提示検証。"""

from __future__ import annotations

import argparse
import base64
import hashlib
import itertools
import json
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_musical_composer.closure_calibration import read_performance
from llm_musical_composer.piano_roll import render_pair_piano_roll

MODEL_ID = "jp.anthropic.claude-sonnet-4-6"
REGION = "ap-northeast-1"
PROFILE = "default"
EVALUATION_SHA256 = "2A791A53F5B2AE6A267A773B70121E6CDDF7EFBD5A82D32ECC277A59CE446FE9"
LABELS = ("A", "B", "同程度", "判定不能")
CONFIDENCE_VALUES = ("low", "medium", "high")
ORDER_NAMES = ("normal", "reversed")
MAX_OUTPUT_TOKENS = 500
REVISION_MAX_OUTPUT_TOKENS = 1_000
INPUT_USD_PER_MILLION = 3.0
OUTPUT_USD_PER_MILLION = 15.0
CACHE_WRITE_USD_PER_MILLION = 3.75
CACHE_READ_USD_PER_MILLION = 0.30
COST_SAFETY_MULTIPLIER = 1.25
EMERGENCY_PROJECTED_COST_USD = 10.0


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    split: str
    human_label: str
    human_reason: str | None
    source: str
    roles: Mapping[str, str]
    judge: Mapping[str, Any]


@dataclass(frozen=True)
class Calibration:
    few_shot: tuple[PairRecord, ...]
    holdout: tuple[PairRecord, ...]
    evaluation_sha256: str


@dataclass(frozen=True)
class Prediction:
    preferred: str
    confidence: str
    observed_evidence: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class SampleAssessment:
    closure_score: int
    sufficient_information: bool
    evidence_paths: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class RevisionPrediction:
    a: SampleAssessment
    b: SampleAssessment
    preferred: str


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _write_json_stable(path: Path, value: object) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"existing artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def _serialize_cli_result(value: object) -> str:
    """Windows の既定端末でも表示できる ASCII JSON を返す。"""
    return json.dumps(value, ensure_ascii=True, indent=2)


def load_calibration(
    root: Path, *, expected_evaluation_sha256: str = EVALUATION_SHA256
) -> Calibration:
    """固定された人間回答と分割を読み、矛盾や変更を拒否する。"""
    root = Path(root)
    evaluation_path = root / "evaluation-form.json"
    digest = _sha256_bytes(evaluation_path.read_bytes())
    if digest != expected_evaluation_sha256.upper():
        raise ValueError(
            f"evaluation SHA-256 mismatch: expected {expected_evaluation_sha256}, got {digest}"
        )
    evaluation = _read_json(evaluation_path)
    manifest = _read_json(root / "selection-manifest.json")
    answers = {item["pair_id"]: item for item in evaluation.get("answers", [])}
    records: list[PairRecord] = []
    for item in manifest.get("pairs", []):
        pair_id = str(item["pair_id"])
        answer = answers.get(pair_id)
        if answer is None:
            raise ValueError(f"missing human answer: {pair_id}")
        label = answer.get("answer")
        if label not in LABELS:
            raise ValueError(f"invalid human label for {pair_id}: {label!r}")
        split = str(item.get("split"))
        if split not in {"few_shot", "holdout"}:
            raise ValueError(f"invalid split for {pair_id}: {split}")
        roles = item.get("roles")
        if not isinstance(roles, dict) or set(roles) != {"A", "B"}:
            raise ValueError(f"invalid roles for {pair_id}")
        judge = _read_json(root / "judge" / f"{pair_id}.json")
        if judge.get("pair_id") != pair_id:
            raise ValueError(f"judge pair id mismatch: {pair_id}")
        records.append(
            PairRecord(
                pair_id=pair_id,
                split=split,
                human_label=label,
                human_reason=answer.get("reason"),
                source=str(item["source"]),
                roles=roles,
                judge=judge,
            )
        )
    if set(answers) != {record.pair_id for record in records}:
        raise ValueError("evaluation and selection manifest pair sets differ")
    records.sort(key=lambda record: record.pair_id)
    return Calibration(
        tuple(record for record in records if record.split == "few_shot"),
        tuple(record for record in records if record.split == "holdout"),
        digest,
    )


def _sample_payload(record: PairRecord, *, reversed_order: bool) -> dict[str, Any]:
    samples = record.judge["samples"]
    if reversed_order:
        samples = {"A": samples["B"], "B": samples["A"]}
    return {
        "scope": "局所的な終止感",
        "samples": samples,
        "full_structure": record.judge["full_structure"],
    }


def _normalize_sample_time_fields(sample: Mapping[str, Any]) -> dict[str, Any]:
    """旧校正データの無単位比率を、単位が明示された項目へ変換する。"""
    normalized = dict(sample)
    sample_duration_ms = normalized.get("duration_ms")
    terminal_events: list[dict[str, Any]] = []
    for value in normalized.get("terminal_events", []):
        event = dict(value)
        onset_ratio = event.pop("onset", event.get("onset_ratio"))
        duration_ratio = event.pop("duration", event.get("duration_ratio"))
        if onset_ratio is not None:
            event["onset_ratio"] = onset_ratio
        if duration_ratio is not None:
            event["duration_ratio"] = duration_ratio
        if isinstance(sample_duration_ms, int):
            if onset_ratio is not None and "onset_ms" not in event:
                event["onset_ms"] = round(float(onset_ratio) * sample_duration_ms)
            if duration_ratio is not None and "duration_ms" not in event:
                event["duration_ms"] = round(float(duration_ratio) * sample_duration_ms)
        terminal_events.append(event)
    if "terminal_events" in normalized:
        normalized["terminal_events"] = terminal_events
    pedal_events: list[dict[str, Any]] = []
    for value in normalized.get("pedal_events", []):
        event = dict(value)
        at_ratio = event.pop("at", event.get("at_ratio"))
        if at_ratio is not None:
            event["at_ratio"] = at_ratio
        if isinstance(sample_duration_ms, int) and at_ratio is not None and "at_ms" not in event:
            event["at_ms"] = round(float(at_ratio) * sample_duration_ms)
        pedal_events.append(event)
    if "pedal_events" in normalized:
        normalized["pedal_events"] = pedal_events
    return normalized


def _revision_sample_payload(record: PairRecord, *, reversed_order: bool) -> dict[str, Any]:
    payload = _sample_payload(record, reversed_order=reversed_order)
    payload["samples"] = {
        label: _normalize_sample_time_fields(sample) for label, sample in payload["samples"].items()
    }
    return payload


def _image_block(image: bytes) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(image).decode("ascii"),
        },
    }


def _json_text(value: object) -> dict[str, str]:
    return {
        "type": "text",
        "text": json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
    }


def build_request_body(
    calibration: Calibration,
    target_pair_id: str,
    *,
    reversed_order: bool,
    image_provider: Callable[[str, bool], bytes],
) -> dict[str, Any]:
    """回答を隠した対象と、人間回答付きの固定例から Bedrock 要求を作る。"""
    targets = [pair for pair in calibration.holdout if pair.pair_id == target_pair_id]
    if len(targets) != 1:
        raise ValueError(f"target must be one holdout pair: {target_pair_id}")
    content: list[dict[str, Any]] = []
    for example in calibration.few_shot:
        payload = _sample_payload(example, reversed_order=False)
        payload["human_label"] = example.human_label
        payload["human_reason"] = example.human_reason
        content.append(_json_text(payload))
        content.append(_image_block(image_provider(example.pair_id, False)))
    if content:
        content[-1]["cache_control"] = {"type": "ephemeral", "ttl": "5m"}
    target = targets[0]
    content.append(_json_text(_sample_payload(target, reversed_order=reversed_order)))
    content.append(_image_block(image_provider(target.pair_id, reversed_order)))
    schema = {
        "type": "object",
        "properties": {
            "preferred": {"type": "string", "enum": list(LABELS)},
            "confidence": {"type": "string", "enum": list(CONFIDENCE_VALUES)},
            "observed_evidence": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": ["preferred", "confidence", "observed_evidence", "reason"],
        "additionalProperties": False,
    }
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0,
        "system": (
            "あなたはピアノ演奏断片の局所的な終止感だけを比較する判定者です。"
            "好み、音質、曲全体の品質ではなく、末尾で音楽が区切れたように感じる強さを比べてください。"
            "画像の左がA、右がBです。情報が少なすぎる場合は判定不能を選んでください。"
            "先行する人間回答の基準を使い、最後の未回答例だけを判定してください。"
        ),
        "messages": [{"role": "user", "content": content}],
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
    }


def build_revision_request_body(
    calibration: Calibration,
    target_pair_id: str,
    *,
    reversed_order: bool,
    image_provider: Callable[[str, bool], bytes],
) -> dict[str, Any]:
    """単位を明示し、A と B を独立採点させる開発用要求を作る。"""
    targets = [pair for pair in calibration.holdout if pair.pair_id == target_pair_id]
    if len(targets) != 1:
        raise ValueError(f"target must be one holdout pair: {target_pair_id}")
    target = targets[0]
    content: list[dict[str, Any]] = []

    def append_demonstration(example: PairRecord) -> None:
        payload = _revision_sample_payload(example, reversed_order=False)
        payload["human_label"] = example.human_label
        payload["human_reason"] = example.human_reason
        content.append(_json_text(payload))
        content.append(_image_block(image_provider(example.pair_id, False)))

    for example in calibration.few_shot:
        append_demonstration(example)
    if content:
        content[-1]["cache_control"] = {"type": "ephemeral", "ttl": "5m"}
    abstention_examples = [
        pair
        for pair in calibration.holdout
        if pair.pair_id != target_pair_id and pair.human_label == "判定不能"
    ]
    for example in abstention_examples:
        append_demonstration(example)
    target_payload = _revision_sample_payload(target, reversed_order=reversed_order)
    target_payload["task"] = "score_each_sample_independently"
    content.append(_json_text(target_payload))
    content.append(_image_block(image_provider(target.pair_id, reversed_order)))

    assessment_schema = {
        "type": "object",
        "properties": {
            "closure_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "sufficient_information": {"type": "boolean"},
            "evidence_paths": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "reason": {"type": "string"},
        },
        "required": [
            "closure_score",
            "sufficient_information",
            "evidence_paths",
            "reason",
        ],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "assessments": {
                "type": "object",
                "properties": {"A": assessment_schema, "B": assessment_schema},
                "required": ["A", "B"],
                "additionalProperties": False,
            }
        },
        "required": ["assessments"],
        "additionalProperties": False,
    }
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": REVISION_MAX_OUTPUT_TOKENS,
        "temperature": 0,
        "system": (
            "あなたはピアノ演奏断片の局所的な終止感を評価します。"
            "AとBを比較して先に勝者を決めず、同じ基準で各断片を独立に1から5で採点してください。"
            "1は続きそうな状態、5は明確に区切れた状態です。これは音楽品質の点数ではありません。"
            "情報が少なすぎる、または曖昧で採点根拠を示せない場合はsufficient_informationをfalseにしてください。"
            "無音、長音、低音、密度低下のどれか一つだけで終止を断定しないでください。"
            "onset_ratio、duration_ratio、at_ratioは断片長に対する0から1の比率で、秒ではありません。"
            "実時間を述べる場合は入力の_ms項目だけを使い、単位をmsとしてください。"
            "各断片のevidence_pathsは最も重要なJSON上の位置1件だけにしてください。"
            "各断片のreasonは100文字以内の1文にしてください。"
            "先行する人間回答の基準を参考にし、最後の未回答例だけを採点してください。"
        ),
        "messages": [{"role": "user", "content": content}],
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
    }


def validate_prediction(value: object) -> Prediction:
    """構造化出力を厳密に検証する。"""
    if not isinstance(value, dict) or set(value) != {
        "preferred",
        "confidence",
        "observed_evidence",
        "reason",
    }:
        raise ValueError("prediction keys are invalid")
    if value["preferred"] not in LABELS:
        raise ValueError("prediction preferred value is invalid")
    if value["confidence"] not in CONFIDENCE_VALUES:
        raise ValueError("prediction confidence value is invalid")
    evidence = value["observed_evidence"]
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        raise ValueError("prediction observed_evidence must be a string array")
    if not isinstance(value["reason"], str):
        raise ValueError("prediction reason must be a string")
    return Prediction(value["preferred"], value["confidence"], tuple(evidence), value["reason"])


def _validate_sample_assessment(value: object) -> SampleAssessment:
    if not isinstance(value, dict) or set(value) != {
        "closure_score",
        "sufficient_information",
        "evidence_paths",
        "reason",
    }:
        raise ValueError("sample assessment keys are invalid")
    score = value["closure_score"]
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError("sample assessment closure_score must be an integer from 1 to 5")
    sufficient = value["sufficient_information"]
    if not isinstance(sufficient, bool):
        raise ValueError("sample assessment sufficient_information must be boolean")
    evidence = value["evidence_paths"]
    if (
        not isinstance(evidence, list)
        or not evidence
        or not all(isinstance(item, str) and item for item in evidence)
    ):
        raise ValueError("sample assessment evidence_paths must be a non-empty string array")
    reason = value["reason"]
    if not isinstance(reason, str) or not reason:
        raise ValueError("sample assessment reason must be a non-empty string")
    return SampleAssessment(score, sufficient, tuple(evidence), reason)


def validate_revision_prediction(value: object) -> RevisionPrediction:
    """独立採点を検証し、比較ラベルを決定的に導く。"""
    if not isinstance(value, dict) or set(value) != {"assessments"}:
        raise ValueError("revision prediction keys are invalid")
    assessments = value["assessments"]
    if not isinstance(assessments, dict) or set(assessments) != {"A", "B"}:
        raise ValueError("revision prediction assessments must contain A and B")
    first = _validate_sample_assessment(assessments["A"])
    second = _validate_sample_assessment(assessments["B"])
    if not first.sufficient_information or not second.sufficient_information:
        preferred = "判定不能"
    elif first.closure_score == second.closure_score:
        preferred = "同程度"
    elif first.closure_score > second.closure_score:
        preferred = "A"
    else:
        preferred = "B"
    return RevisionPrediction(first, second, preferred)


def normalize_preferred(preferred: str, *, reversed_order: bool) -> str:
    if preferred not in LABELS:
        raise ValueError(f"invalid preferred value: {preferred}")
    if not reversed_order:
        return preferred
    return {"A": "B", "B": "A"}.get(preferred, preferred)


def estimate_max_cost(
    input_tokens: Sequence[int],
    *,
    request_count: int,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    cost_cap_usd: float = 2.0,
) -> float:
    """実測入力と最大出力から課金上限を見積もり、上限超過を拒否する。"""
    if request_count != len(input_tokens):
        raise ValueError("request count and token counts differ")
    total = (
        sum(input_tokens) * INPUT_USD_PER_MILLION
        + request_count * max_output_tokens * OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    if total > cost_cap_usd:
        raise ValueError(f"estimated ${total:.6f} exceeds ${cost_cap_usd:.2f} cost cap")
    return total


def _usage_token_count(usage: Mapping[str, Any], name: str, *, required: bool) -> int:
    value = usage.get(name)
    if value is None and not required:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"usage {name} must be a non-negative integer")
    return value


def estimate_usage_cost(usage: Mapping[str, Any]) -> float:
    """Bedrock 応答の実測トークンを公開単価へ当てはめる。"""
    input_tokens = _usage_token_count(usage, "input_tokens", required=True)
    output_tokens = _usage_token_count(usage, "output_tokens", required=True)
    cache_write_tokens = _usage_token_count(usage, "cache_creation_input_tokens", required=False)
    cache_read_tokens = _usage_token_count(usage, "cache_read_input_tokens", required=False)
    return (
        input_tokens * INPUT_USD_PER_MILLION
        + output_tokens * OUTPUT_USD_PER_MILLION
        + cache_write_tokens * CACHE_WRITE_USD_PER_MILLION
        + cache_read_tokens * CACHE_READ_USD_PER_MILLION
    ) / 1_000_000


def project_initial_request_cost(
    usage: Mapping[str, Any],
    *,
    request_count: int = 16,
    safety_multiplier: float = COST_SAFETY_MULTIPLIER,
    emergency_limit_usd: float = EMERGENCY_PROJECTED_COST_USD,
) -> float:
    """最初の実測料金から全要求を保守的に予測し、異常値だけを拒否する。"""
    if request_count <= 0 or safety_multiplier <= 0 or emergency_limit_usd <= 0:
        raise ValueError("cost projection parameters must be positive")
    projected = estimate_usage_cost(usage) * request_count * safety_multiplier
    if projected > emergency_limit_usd:
        raise ValueError(
            f"projected ${projected:.6f} exceeds ${emergency_limit_usd:.2f} emergency limit"
        )
    return projected


def evaluate_predictions(
    calibration: Calibration,
    predictions: Mapping[tuple[str, str], Prediction],
) -> dict[str, Any]:
    """順序を入れ替えた両判定と、人間の未提示回答との一致を評価する。"""
    rows: list[dict[str, Any]] = []
    model_labels: list[str] = []
    human_labels: list[str] = []
    both_match = 0
    for pair in calibration.holdout:
        normal = predictions[(pair.pair_id, "normal")]
        reversed_prediction = predictions[(pair.pair_id, "reversed")]
        normalized_reversed = normalize_preferred(
            reversed_prediction.preferred, reversed_order=True
        )
        judgeable = pair.human_label != "判定不能"
        matches = normal.preferred == pair.human_label and normalized_reversed == pair.human_label
        if judgeable:
            human_labels.append(pair.human_label)
            model_labels.append(
                normal.preferred if normal.preferred == normalized_reversed else "不一致"
            )
            both_match += int(matches)
        rows.append(
            {
                "pair_id": pair.pair_id,
                "human_label": pair.human_label,
                "normal": asdict(normal),
                "reversed": asdict(reversed_prediction),
                "normalized_reversed": normalized_reversed,
                "judgeable": judgeable,
                "both_order_match": matches if judgeable else None,
            }
        )
    unique_permutations = set(itertools.permutations(human_labels))
    observed = sum(model == human for model, human in zip(model_labels, human_labels, strict=True))
    as_extreme = sum(
        sum(model == human for model, human in zip(model_labels, permuted, strict=True)) >= observed
        for permuted in unique_permutations
    )
    p_value = as_extreme / len(unique_permutations) if unique_permutations else 1.0
    return {
        "judgeable_count": len(human_labels),
        "both_order_match_count": both_match,
        "permutation_p_value": p_value,
        "passed": len(human_labels) >= 6 and both_match == len(human_labels) and p_value < 0.05,
        "rows": rows,
    }


_SECONDS_PATTERN = re.compile(r"(?<!ミリ)秒|\bseconds?\b", re.IGNORECASE)


def _revision_prediction_mentions_seconds(prediction: RevisionPrediction) -> bool:
    values = (
        *prediction.a.evidence_paths,
        prediction.a.reason,
        *prediction.b.evidence_paths,
        prediction.b.reason,
    )
    return any(_SECONDS_PATTERN.search(value) for value in values)


def evaluate_revision_predictions(
    calibration: Calibration,
    predictions: Mapping[tuple[str, str], RevisionPrediction],
) -> dict[str, Any]:
    """開発用再実行を、事前に固定した停止条件だけで評価する。"""
    rows: list[dict[str, Any]] = []
    judgeable_count = 0
    judgeable_order_consistent_count = 0
    order_consistent_count = 0
    both_order_match_count = 0
    abstention_both_order_count = 0
    unit_misinterpretation_count = 0
    for pair in calibration.holdout:
        normal = predictions[(pair.pair_id, "normal")]
        reversed_prediction = predictions[(pair.pair_id, "reversed")]
        normalized_reversed = normalize_preferred(
            reversed_prediction.preferred, reversed_order=True
        )
        order_consistent = normal.preferred == normalized_reversed
        judgeable = pair.human_label != "判定不能"
        both_order_match = (
            judgeable
            and normal.preferred == pair.human_label
            and normalized_reversed == pair.human_label
        )
        abstention_both_orders = (
            not judgeable and normal.preferred == "判定不能" and normalized_reversed == "判定不能"
        )
        unit_misinterpretation = _revision_prediction_mentions_seconds(
            normal
        ) or _revision_prediction_mentions_seconds(reversed_prediction)
        order_consistent_count += int(order_consistent)
        unit_misinterpretation_count += int(unit_misinterpretation)
        if judgeable:
            judgeable_count += 1
            judgeable_order_consistent_count += int(order_consistent)
            both_order_match_count += int(both_order_match)
        else:
            abstention_both_order_count += int(abstention_both_orders)
        rows.append(
            {
                "pair_id": pair.pair_id,
                "human_label": pair.human_label,
                "normal": asdict(normal),
                "reversed": asdict(reversed_prediction),
                "normalized_reversed": normalized_reversed,
                "judgeable": judgeable,
                "order_consistent": order_consistent,
                "both_order_match": both_order_match if judgeable else None,
                "abstention_both_orders": abstention_both_orders if not judgeable else None,
                "unit_misinterpretation": unit_misinterpretation,
            }
        )
    development_criteria_met = (
        judgeable_count >= 6
        and judgeable_order_consistent_count >= 5
        and abstention_both_order_count >= 1
        and unit_misinterpretation_count == 0
        and both_order_match_count > 2
    )
    return {
        "judgeable_count": judgeable_count,
        "order_consistent_count": order_consistent_count,
        "judgeable_order_consistent_count": judgeable_order_consistent_count,
        "both_order_match_count": both_order_match_count,
        "abstention_both_order_count": abstention_both_order_count,
        "unit_misinterpretation_count": unit_misinterpretation_count,
        "development_criteria_met": development_criteria_met,
        "formal_accuracy_evidence": False,
        "rows": rows,
    }


def execute_with_retry(
    command: list[str],
    *,
    output_path: Path,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = 3,
) -> subprocess.CompletedProcess[str]:
    """明示的な一時障害だけを再試行し、課金が曖昧な失敗は止める。"""
    runner = command_runner or (
        lambda arguments: subprocess.run(
            arguments, capture_output=True, text=True, timeout=330, check=False
        )
    )
    for attempt in range(1, max_attempts + 1):
        try:
            result = runner(command)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("ambiguous timeout; invocation may have been billed") from error
        if result.returncode == 0:
            return result
        if Path(output_path).exists():
            raise RuntimeError("ambiguous invocation failure with an output artifact")
        transient = any(
            marker in result.stderr
            for marker in ("ThrottlingException", "ServiceUnavailableException")
        )
        if not transient or attempt == max_attempts:
            raise RuntimeError(f"AWS command failed without safe retry: {result.stderr}")
        sleep(float(2 ** (attempt - 1)))
    raise AssertionError("unreachable")


def _extract_prediction(response: Mapping[str, Any]) -> Prediction:
    content = response.get("content")
    if not isinstance(content, list) or len(content) != 1 or not isinstance(content[0], dict):
        raise ValueError("unexpected Anthropic response content")
    text = content[0].get("text")
    if not isinstance(text, str):
        raise ValueError("Anthropic response has no JSON text")
    return validate_prediction(json.loads(text))


def _extract_revision_prediction(response: Mapping[str, Any]) -> RevisionPrediction:
    if response.get("stop_reason") == "max_tokens":
        raise ValueError("Anthropic response reached max_tokens before JSON completed")
    content = response.get("content")
    if not isinstance(content, list) or len(content) != 1 or not isinstance(content[0], dict):
        raise ValueError("unexpected Anthropic response content")
    text = content[0].get("text")
    if not isinstance(text, str):
        raise ValueError("Anthropic response has no JSON text")
    return validate_revision_prediction(json.loads(text))


def _aws_base(profile: str, region: str) -> list[str]:
    return [
        "aws",
        "--profile",
        profile,
        "--region",
        region,
        "--cli-connect-timeout",
        "30",
        "--cli-read-timeout",
        "300",
    ]


def _count_tokens_input(request_body: bytes) -> dict[str, Any]:
    return {"invokeModel": {"body": base64.b64encode(request_body).decode("ascii")}}


def _count_tokens_command(
    input_path: Path, *, profile: str, region: str, model_id: str
) -> list[str]:
    return [
        *_aws_base(profile, region),
        "bedrock-runtime",
        "count-tokens",
        "--model-id",
        model_id,
        "--input",
        f"file://{input_path.resolve()}",
    ]


def _order_requests_by_size(
    requests: Sequence[tuple[PairRecord, str, Path]],
) -> list[tuple[PairRecord, str, Path]]:
    """最大要求を先頭にし、最初の実測を保守的な料金予測へ使う。"""
    return sorted(
        requests,
        key=lambda item: (-item[2].stat().st_size, item[0].pair_id, item[1]),
    )


def _run_protocol_experiment(  # pragma: no cover - 固定スクリプトから実環境で検証する
    calibration_root: Path,
    output_root: Path,
    *,
    profile: str,
    region: str,
    model_id: str,
    request_builder: Callable[..., dict[str, Any]],
    prediction_extractor: Callable[[Mapping[str, Any]], Any],
    evaluator: Callable[[Calibration, Mapping[tuple[str, str], Any]], dict[str, Any]],
    result_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    calibration = load_calibration(calibration_root)
    if len(calibration.few_shot) != 8 or len(calibration.holdout) != 8:
        raise ValueError("production calibration must contain 8 few-shot and 8 holdout pairs")
    assets = output_root / "assets"

    def image_provider(pair_id: str, reversed_order: bool) -> bytes:
        if not reversed_order:
            return (calibration_root / "judge" / f"{pair_id}.png").read_bytes()
        path = assets / f"{pair_id}-reversed.png"
        if not path.exists():
            first = read_performance(calibration_root / "listen" / pair_id / "sample-B.mid")
            second = read_performance(calibration_root / "listen" / pair_id / "sample-A.mid")
            render_pair_piano_roll(first, second, path)
        return path.read_bytes()

    requests: list[tuple[PairRecord, str, Path]] = []
    for pair in calibration.holdout:
        for order in ORDER_NAMES:
            request = request_builder(
                calibration,
                pair.pair_id,
                reversed_order=order == "reversed",
                image_provider=image_provider,
            )
            serialized = json.dumps(request, ensure_ascii=False)
            forbidden = [record.source for record in calibration.few_shot + calibration.holdout]
            if any(value in serialized for value in (*forbidden, '"actual"', '"middle"')):
                raise RuntimeError("request leaks a source name or hidden role")
            path = output_root / "requests" / f"{pair.pair_id}-{order}.json"
            _write_json_stable(path, request)
            requests.append((pair, order, path))
    requests = _order_requests_by_size(requests)

    predictions: dict[tuple[str, str], Any] = {}
    usage_records: list[dict[str, Any]] = []
    initial_projected_cost_usd: float | None = None
    actual_estimated_cost_usd = 0.0
    for request_index, (pair, order, request_path) in enumerate(requests):
        response_path = output_root / "responses" / f"{pair.pair_id}-{order}.json"
        meta_path = output_root / "responses" / f"{pair.pair_id}-{order}.meta.json"
        request_sha = _sha256_bytes(request_path.read_bytes())
        if meta_path.exists():
            meta = _read_json(meta_path)
            if meta.get("request_sha256") != request_sha or not response_path.exists():
                raise RuntimeError("ambiguous existing response metadata")
            response = _read_json(response_path)
            prediction = prediction_extractor(response)
        else:
            if response_path.exists():
                raise RuntimeError("ambiguous response exists without metadata")
            response_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                *_aws_base(profile, region),
                "bedrock-runtime",
                "invoke-model",
                "--model-id",
                model_id,
                "--content-type",
                "application/json",
                "--accept",
                "application/json",
                "--body",
                f"fileb://{request_path.resolve()}",
                str(response_path.resolve()),
            ]
            result = execute_with_retry(command, output_path=response_path)
            response = _read_json(response_path)
            prediction = prediction_extractor(response)
            _write_json_stable(
                meta_path,
                {
                    "request_sha256": request_sha,
                    "model_id": model_id,
                    "prediction": asdict(prediction),
                    "usage": response.get("usage"),
                    "command_stdout": result.stdout,
                    "command_stderr": result.stderr,
                },
            )
        usage = response.get("usage")
        if not isinstance(usage, Mapping):
            raise ValueError("Anthropic response usage must be an object")
        request_cost = estimate_usage_cost(usage)
        if request_index == 0:
            initial_projected_cost_usd = project_initial_request_cost(
                usage, request_count=len(requests)
            )
        actual_estimated_cost_usd += request_cost
        predictions[(pair.pair_id, order)] = prediction
        usage_records.append(
            {
                "pair_id": pair.pair_id,
                "order": order,
                "usage": dict(usage),
                "public_global_price_estimate_usd": request_cost,
            }
        )
    evaluation = evaluator(calibration, predictions)
    result = {
        "status": "affirmative_evidence",
        "model_id": model_id,
        "region": region,
        "profile": profile,
        "request_count": len(requests),
        "initial_projected_cost_usd": initial_projected_cost_usd,
        "actual_estimated_cost_usd": actual_estimated_cost_usd,
        "price_basis": {
            "input_usd_per_million": INPUT_USD_PER_MILLION,
            "output_usd_per_million": OUTPUT_USD_PER_MILLION,
            "cache_write_usd_per_million": CACHE_WRITE_USD_PER_MILLION,
            "cache_read_usd_per_million": CACHE_READ_USD_PER_MILLION,
            "note": "public Global rates; the AWS invoice may differ for the JP profile",
        },
        "usage": usage_records,
        "evaluation": evaluation,
    }
    if result_metadata:
        result.update(result_metadata)
    _write_json_stable(output_root / "result.json", result)
    return result


def _run_experiment(  # pragma: no cover - 固定スクリプトから実環境で検証する
    calibration_root: Path,
    output_root: Path,
    *,
    profile: str,
    region: str,
    model_id: str,
) -> dict[str, Any]:
    return _run_protocol_experiment(
        calibration_root,
        output_root,
        profile=profile,
        region=region,
        model_id=model_id,
        request_builder=build_request_body,
        prediction_extractor=_extract_prediction,
        evaluator=evaluate_predictions,
    )


def _run_revision_experiment(  # pragma: no cover - 固定スクリプトから実環境で検証する
    calibration_root: Path,
    output_root: Path,
    *,
    profile: str,
    region: str,
    model_id: str,
) -> dict[str, Any]:
    calibration = load_calibration(calibration_root)
    label_counts_by_target: dict[str, dict[str, int]] = {}
    for target in calibration.holdout:
        examples = [
            *calibration.few_shot,
            *(
                pair
                for pair in calibration.holdout
                if pair.pair_id != target.pair_id and pair.human_label == "判定不能"
            ),
        ]
        counts = {label: 0 for label in LABELS}
        for example in examples:
            counts[example.human_label] += 1
        label_counts_by_target[target.pair_id] = counts
    return _run_protocol_experiment(
        calibration_root,
        output_root,
        profile=profile,
        region=region,
        model_id=model_id,
        request_builder=build_revision_request_body,
        prediction_extractor=_extract_revision_prediction,
        evaluator=evaluate_revision_predictions,
        result_metadata={
            "protocol": "independent-scores-v1",
            "development_only": True,
            "formal_accuracy_evidence": False,
            "demonstration_label_counts_by_target": label_counts_by_target,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI 入り口
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-root", type=Path, default=Path(".appendix/closure-calibration")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--protocol", choices=("legacy", "revision"), default="legacy")
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--model-id", default=MODEL_ID)
    arguments = parser.parse_args(argv)
    output_root = arguments.output_root or Path(
        ".appendix/closure-judge/20260811-sonnet46-v6"
        if arguments.protocol == "revision"
        else ".appendix/closure-judge/20260811-sonnet46-v3"
    )
    runner = _run_revision_experiment if arguments.protocol == "revision" else _run_experiment
    result = runner(
        arguments.calibration_root,
        output_root,
        profile=arguments.profile,
        region=arguments.region,
        model_id=arguments.model_id,
    )
    print(_serialize_cli_result(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
