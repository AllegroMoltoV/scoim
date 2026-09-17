from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from llm_musical_composer import closure_judge
from llm_musical_composer.closure_judge import (
    Calibration,
    PairRecord,
    Prediction,
    RevisionPrediction,
    SampleAssessment,
    build_request_body,
    build_revision_request_body,
    estimate_max_cost,
    estimate_usage_cost,
    evaluate_predictions,
    evaluate_revision_predictions,
    execute_with_retry,
    load_calibration,
    normalize_preferred,
    project_initial_request_cost,
    validate_prediction,
    validate_revision_prediction,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _calibration_root(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "calibration"
    answers = []
    pairs = []
    labels = ("A", "B", "同程度", "判定不能")
    for index in range(1, 5):
        pair_id = f"pair-{index:02d}"
        split = "few_shot" if index <= 2 else "holdout"
        answers.append(
            {
                "pair_id": pair_id,
                "answer": labels[index - 1],
                "reason": f"人間理由{index}",
            }
        )
        pairs.append(
            {
                "pair_id": pair_id,
                "source": f"secret-{index}.mid",
                "split": split,
                "roles": {"A": "actual", "B": "middle"},
            }
        )
        _write_json(
            root / "judge" / f"{pair_id}.json",
            {
                "pair_id": pair_id,
                "scope": "局所的な終止感",
                "samples": {
                    "A": {
                        "duration_ms": 1_000,
                        "note_count": index,
                        "marker": f"A{index}",
                        "terminal_events": [
                            {
                                "onset": 0.1,
                                "duration": 0.2,
                                "pitch": 60,
                                "velocity": 70,
                            }
                        ],
                        "pedal_events": [{"at": 0.5, "value": 0}],
                    },
                    "B": {
                        "duration_ms": 2_000,
                        "note_count": index + 10,
                        "marker": f"B{index}",
                        "terminal_events": [
                            {
                                "onset": 0.25,
                                "duration": 0.5,
                                "pitch": 48,
                                "velocity": 60,
                            }
                        ],
                        "pedal_events": [{"at": 0.75, "value": 0}],
                    },
                },
                "full_structure": [{"index": 0, "onset_count": index}],
            },
        )
        image = root / "judge" / f"{pair_id}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([index]))
    evaluation = {
        "allowed_values": ["A", "B", "同程度", "判定不能"],
        "answers": answers,
    }
    evaluation_path = root / "evaluation-form.json"
    _write_json(evaluation_path, evaluation)
    _write_json(root / "selection-manifest.json", {"pairs": pairs})
    digest = hashlib.sha256(evaluation_path.read_bytes()).hexdigest().upper()
    return root, digest


def test_load_calibration_fixes_hash_split_and_labels(tmp_path: Path) -> None:
    root, digest = _calibration_root(tmp_path)

    calibration = load_calibration(root, expected_evaluation_sha256=digest)

    assert [pair.pair_id for pair in calibration.few_shot] == ["pair-01", "pair-02"]
    assert [pair.pair_id for pair in calibration.holdout] == ["pair-03", "pair-04"]
    assert calibration.few_shot[0].human_label == "A"
    with pytest.raises(ValueError, match="SHA-256"):
        load_calibration(root, expected_evaluation_sha256="0" * 64)


def test_request_hides_target_answer_roles_and_source_and_only_reverses_target(
    tmp_path: Path,
) -> None:
    root, digest = _calibration_root(tmp_path)
    calibration = load_calibration(root, expected_evaluation_sha256=digest)

    def image_provider(pair_id: str, reversed_order: bool) -> bytes:
        return f"{pair_id}:{reversed_order}".encode()

    normal = build_request_body(
        calibration, "pair-03", reversed_order=False, image_provider=image_provider
    )
    reversed_body = build_request_body(
        calibration, "pair-03", reversed_order=True, image_provider=image_provider
    )
    normal_content = normal["messages"][0]["content"]
    reversed_content = reversed_body["messages"][0]["content"]
    normal_target = json.loads(normal_content[-2]["text"])
    reversed_target = json.loads(reversed_content[-2]["text"])
    normal_serialized = json.dumps(normal, ensure_ascii=False)

    assert normal_target["samples"]["A"]["marker"] == "A3"
    assert reversed_target["samples"]["A"]["marker"] == "B3"
    assert normal_content[:-2] == reversed_content[:-2]
    cached_blocks = [block for block in normal_content if "cache_control" in block]
    assert cached_blocks == [normal_content[-3]]
    assert normal_content[-3]["type"] == "image"
    assert normal_content[-3]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    assert "secret-3.mid" not in normal_serialized
    assert "actual" not in normal_serialized
    assert "人間理由3" not in normal_serialized
    assert '"human_label": "同程度"' not in normal_serialized
    assert "人間理由1" in normal_serialized
    assert normal["output_config"]["format"]["type"] == "json_schema"


def test_revision_request_uses_explicit_units_independent_scores_and_abstention_example(
    tmp_path: Path,
) -> None:
    root, digest = _calibration_root(tmp_path)
    calibration = load_calibration(root, expected_evaluation_sha256=digest)

    def image_provider(pair_id: str, reversed_order: bool) -> bytes:
        return f"{pair_id}:{reversed_order}".encode()

    normal = build_revision_request_body(
        calibration, "pair-03", reversed_order=False, image_provider=image_provider
    )
    reversed_body = build_revision_request_body(
        calibration, "pair-03", reversed_order=True, image_provider=image_provider
    )
    normal_content = normal["messages"][0]["content"]
    reversed_content = reversed_body["messages"][0]["content"]
    target = json.loads(normal_content[-2]["text"])
    terminal = target["samples"]["A"]["terminal_events"][0]
    pedal = target["samples"]["A"]["pedal_events"][0]
    serialized = json.dumps(normal, ensure_ascii=False)
    schema = normal["output_config"]["format"]["schema"]
    score_schema = schema["properties"]["assessments"]["properties"]["A"]["properties"][
        "closure_score"
    ]
    text_payloads = [
        json.loads(block["text"]) for block in normal_content if block["type"] == "text"
    ]
    demonstration_labels = [
        payload["human_label"] for payload in text_payloads if "human_label" in payload
    ]
    demonstration_reasons = [
        payload["human_reason"] for payload in text_payloads if "human_reason" in payload
    ]

    assert normal_content[:-2] == reversed_content[:-2]
    assert terminal["onset_ratio"] == 0.1
    assert terminal["duration_ratio"] == 0.2
    assert terminal["onset_ms"] == 100
    assert terminal["duration_ms"] == 200
    assert "onset" not in terminal
    assert "duration" not in terminal
    assert pedal == {"at_ratio": 0.5, "at_ms": 500, "value": 0}
    cached = [block for block in normal_content if "cache_control" in block]
    assert len(cached) == 1
    assert cached[0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    assert "判定不能" in demonstration_labels
    assert "人間理由4" in demonstration_reasons
    assert "同程度" not in demonstration_labels
    assert "人間理由3" not in demonstration_reasons
    assert "secret-3.mid" not in serialized
    assert "actual" not in serialized
    assert "秒ではありません" in normal["system"]
    assert normal["max_tokens"] == 1_000
    assert set(schema["properties"]) == {"assessments"}
    assert set(schema["properties"]["assessments"]["properties"]) == {"A", "B"}
    assert score_schema == {"type": "integer", "enum": [1, 2, 3, 4, 5]}


def test_prediction_validation_normalization_and_cost_cap() -> None:
    prediction = validate_prediction(
        {
            "preferred": "A",
            "confidence": "high",
            "observed_evidence": ["末尾の長音"],
            "reason": "区切れを感じる",
        }
    )

    assert prediction.preferred == "A"
    assert normalize_preferred("A", reversed_order=True) == "B"
    assert normalize_preferred("同程度", reversed_order=True) == "同程度"
    assert estimate_max_cost([10_000] * 16, request_count=16) == pytest.approx(0.6)
    with pytest.raises(ValueError, match="cost cap"):
        estimate_max_cost([50_000] * 16, request_count=16, cost_cap_usd=2.0)
    with pytest.raises(ValueError, match="prediction keys"):
        validate_prediction(
            {
                "preferred": "A",
                "confidence": "high",
                "observed_evidence": [],
                "reason": "x",
                "extra": True,
            }
        )


def _assessment(
    score: int, *, sufficient: bool = True, reason: str = "区切れを感じる"
) -> dict[str, object]:
    return {
        "closure_score": score,
        "sufficient_information": sufficient,
        "evidence_paths": ["samples.A.terminal_events[0]"],
        "reason": reason,
    }


def _revision_prediction(label: str, *, reason: str = "根拠") -> RevisionPrediction:
    if label == "A":
        scores = (5, 3)
        sufficient = (True, True)
    elif label == "B":
        scores = (3, 5)
        sufficient = (True, True)
    elif label == "同程度":
        scores = (4, 4)
        sufficient = (True, True)
    else:
        scores = (3, 3)
        sufficient = (False, True)
    return RevisionPrediction(
        a=SampleAssessment(scores[0], sufficient[0], ("samples.A.windows[15]",), reason),
        b=SampleAssessment(scores[1], sufficient[1], ("samples.B.windows[15]",), reason),
        preferred=label,
    )


def test_revision_prediction_derives_comparison_from_independent_assessments() -> None:
    prediction = validate_revision_prediction(
        {"assessments": {"A": _assessment(5), "B": _assessment(3)}}
    )
    tied = validate_revision_prediction({"assessments": {"A": _assessment(3), "B": _assessment(3)}})
    insufficient = validate_revision_prediction(
        {
            "assessments": {
                "A": _assessment(3, sufficient=False),
                "B": _assessment(2),
            }
        }
    )

    assert prediction.preferred == "A"
    assert prediction.a.closure_score == 5
    assert tied.preferred == "同程度"
    assert insufficient.preferred == "判定不能"
    with pytest.raises(ValueError, match="closure_score"):
        validate_revision_prediction({"assessments": {"A": _assessment(6), "B": _assessment(3)}})
    with pytest.raises(ValueError, match="evidence_paths"):
        invalid = _assessment(3)
        invalid["evidence_paths"] = []
        validate_revision_prediction({"assessments": {"A": invalid, "B": _assessment(3)}})


def test_revision_time_and_assessment_validation_edges() -> None:
    already_explicit = {
        "duration_ms": 1_000,
        "terminal_events": [
            {
                "onset_ratio": 0.2,
                "duration_ratio": 0.3,
                "onset_ms": 200,
                "duration_ms": 300,
            }
        ],
        "pedal_events": [{"at_ratio": 0.4, "at_ms": 400, "value": 80}],
    }
    assert closure_judge._normalize_sample_time_fields(already_explicit) == already_explicit
    assert closure_judge._normalize_sample_time_fields({"marker": "no-events"}) == {
        "marker": "no-events"
    }

    with pytest.raises(ValueError, match="prediction keys"):
        validate_revision_prediction([])
    with pytest.raises(ValueError, match="contain A and B"):
        validate_revision_prediction({"assessments": {"A": _assessment(3)}})
    invalid_sufficient = _assessment(3)
    invalid_sufficient["sufficient_information"] = "yes"
    with pytest.raises(ValueError, match="sufficient_information"):
        validate_revision_prediction(
            {"assessments": {"A": invalid_sufficient, "B": _assessment(3)}}
        )
    invalid_reason = _assessment(3)
    invalid_reason["reason"] = ""
    with pytest.raises(ValueError, match="reason"):
        validate_revision_prediction({"assessments": {"A": invalid_reason, "B": _assessment(3)}})

    milliseconds = _revision_prediction("A", reason="500ミリ秒、または500 milliseconds")
    seconds = _revision_prediction("A", reason="duration_ratio 0.5 seconds")
    assert closure_judge._revision_prediction_mentions_seconds(milliseconds) is False
    assert closure_judge._revision_prediction_mentions_seconds(seconds) is True


def test_actual_usage_cost_and_emergency_projection() -> None:
    usage = {
        "input_tokens": 10_000,
        "output_tokens": 200,
        "cache_creation_input_tokens": 8_000,
        "cache_read_input_tokens": 4_000,
    }
    expected = (10_000 * 3 + 200 * 15 + 8_000 * 3.75 + 4_000 * 0.30) / 1_000_000

    assert estimate_usage_cost(usage) == pytest.approx(expected)
    assert project_initial_request_cost(usage) == pytest.approx(expected * 1.25 * 16)
    with pytest.raises(ValueError, match="emergency limit"):
        project_initial_request_cost({"input_tokens": 3_000_000, "output_tokens": 100_000})
    with pytest.raises(ValueError, match="usage input_tokens"):
        estimate_usage_cost({"input_tokens": -1, "output_tokens": 1})


def test_request_execution_order_starts_with_largest_artifact(tmp_path: Path) -> None:
    pair_a = PairRecord("pair-a", "holdout", "A", None, "a.mid", {"A": "x", "B": "y"}, {})
    pair_b = PairRecord("pair-b", "holdout", "B", None, "b.mid", {"A": "x", "B": "y"}, {})
    small = tmp_path / "small.json"
    large_a = tmp_path / "large-a.json"
    large_b = tmp_path / "large-b.json"
    small.write_bytes(b"1")
    large_a.write_bytes(b"123")
    large_b.write_bytes(b"456")

    ordered = closure_judge._order_requests_by_size(
        [
            (pair_b, "normal", small),
            (pair_b, "reversed", large_b),
            (pair_a, "reversed", large_a),
        ]
    )

    assert [(pair.pair_id, order) for pair, order, _path in ordered] == [
        ("pair-a", "reversed"),
        ("pair-b", "reversed"),
        ("pair-b", "normal"),
    ]


def test_evaluation_requires_both_orders_and_uses_exact_permutation() -> None:
    labels = ("A", "B", "B", "同程度", "同程度", "同程度")
    holdout = tuple(
        PairRecord(
            pair_id=f"pair-{index:02d}",
            split="holdout",
            human_label=label,
            human_reason=None,
            source=f"hidden-{index}.mid",
            roles={"A": "actual", "B": "middle"},
            judge={"samples": {}, "full_structure": []},
        )
        for index, label in enumerate(labels, start=1)
    )
    calibration = Calibration((), holdout, "A" * 64)
    predictions: dict[tuple[str, str], Prediction] = {}
    for pair in holdout:
        predictions[(pair.pair_id, "normal")] = Prediction(
            pair.human_label, "high", ("evidence",), "reason"
        )
        reversed_label = {"A": "B", "B": "A"}.get(pair.human_label, pair.human_label)
        predictions[(pair.pair_id, "reversed")] = Prediction(
            reversed_label, "high", ("evidence",), "reason"
        )

    result = evaluate_predictions(calibration, predictions)

    assert result["judgeable_count"] == 6
    assert result["both_order_match_count"] == 6
    assert result["permutation_p_value"] == pytest.approx(1 / 60)
    assert result["passed"] is True
    predictions[("pair-01", "reversed")] = Prediction("A", "high", ("evidence",), "reason")
    assert evaluate_predictions(calibration, predictions)["passed"] is False


def test_revision_evaluation_applies_development_stop_conditions() -> None:
    labels = ("A", "B", "B", "同程度", "同程度", "同程度", "判定不能", "判定不能")
    holdout = tuple(
        PairRecord(
            pair_id=f"pair-{index:02d}",
            split="holdout",
            human_label=label,
            human_reason=None,
            source=f"hidden-{index}.mid",
            roles={"A": "actual", "B": "middle"},
            judge={"samples": {}, "full_structure": []},
        )
        for index, label in enumerate(labels, start=1)
    )
    calibration = Calibration((), holdout, "A" * 64)
    predictions: dict[tuple[str, str], RevisionPrediction] = {}
    for pair in holdout:
        predictions[(pair.pair_id, "normal")] = _revision_prediction(pair.human_label)
        reversed_label = {"A": "B", "B": "A"}.get(pair.human_label, pair.human_label)
        predictions[(pair.pair_id, "reversed")] = _revision_prediction(reversed_label)

    result = evaluate_revision_predictions(calibration, predictions)

    assert result["judgeable_count"] == 6
    assert result["order_consistent_count"] == 8
    assert result["both_order_match_count"] == 6
    assert result["abstention_both_order_count"] == 2
    assert result["unit_misinterpretation_count"] == 0
    assert result["development_criteria_met"] is True

    predictions[("pair-01", "normal")] = _revision_prediction("A", reason="duration_ratio 0.07秒")
    assert (
        evaluate_revision_predictions(calibration, predictions)["development_criteria_met"] is False
    )


def test_retry_only_for_explicit_transient_failure_without_output(tmp_path: Path) -> None:
    calls = []

    def transient_then_success(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 1, "", "ThrottlingException")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    result = execute_with_retry(
        ["aws", "bedrock-runtime"],
        output_path=tmp_path / "response.json",
        command_runner=transient_then_success,
        sleep=lambda _seconds: None,
    )

    assert result.returncode == 0
    assert len(calls) == 2

    def ambiguous_failure(command: list[str]) -> subprocess.CompletedProcess[str]:
        (tmp_path / "ambiguous.json").write_text("possibly billed", encoding="utf-8")
        return subprocess.CompletedProcess(command, 1, "", "timeout")

    with pytest.raises(RuntimeError, match="ambiguous"):
        execute_with_retry(
            ["aws", "bedrock-runtime"],
            output_path=tmp_path / "ambiguous.json",
            command_runner=ambiguous_failure,
            sleep=lambda _seconds: None,
        )


def test_calibration_rejects_inconsistent_records(tmp_path: Path) -> None:
    root, _ = _calibration_root(tmp_path)

    def load_after_mutation(mutate: object, match: str) -> None:
        manifest_path = root / "selection-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mutate(manifest)
        _write_json(manifest_path, manifest)
        evaluation_path = root / "evaluation-form.json"
        digest = hashlib.sha256(evaluation_path.read_bytes()).hexdigest().upper()
        with pytest.raises(ValueError, match=match):
            load_calibration(root, expected_evaluation_sha256=digest)

    load_after_mutation(lambda value: value["pairs"].pop(), "pair sets differ")
    root, _ = _calibration_root(tmp_path / "second")
    load_after_mutation(
        lambda value: value["pairs"][0].update({"split": "unknown"}), "invalid split"
    )
    root, _ = _calibration_root(tmp_path / "third")
    load_after_mutation(
        lambda value: value["pairs"][0].update({"roles": {"A": "actual"}}),
        "invalid roles",
    )
    root, _ = _calibration_root(tmp_path / "fourth")
    judge_path = root / "judge" / "pair-01.json"
    judge = json.loads(judge_path.read_text(encoding="utf-8"))
    judge["pair_id"] = "wrong"
    _write_json(judge_path, judge)
    evaluation_path = root / "evaluation-form.json"
    digest = hashlib.sha256(evaluation_path.read_bytes()).hexdigest().upper()
    with pytest.raises(ValueError, match="judge pair id mismatch"):
        load_calibration(root, expected_evaluation_sha256=digest)


def test_local_artifact_and_response_validation_edges(tmp_path: Path) -> None:
    stable = tmp_path / "stable.json"
    closure_judge._write_json_stable(stable, {"a": 1})
    closure_judge._write_json_stable(stable, {"a": 1})
    with pytest.raises(RuntimeError, match="differs"):
        closure_judge._write_json_stable(stable, {"a": 2})
    invalid_root = tmp_path / "array.json"
    invalid_root.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON root"):
        closure_judge._read_json(invalid_root)

    response = {
        "content": [
            {
                "text": json.dumps(
                    {
                        "preferred": "B",
                        "confidence": "medium",
                        "observed_evidence": ["長音"],
                        "reason": "末尾が安定している",
                    },
                    ensure_ascii=False,
                )
            }
        ]
    }
    assert closure_judge._extract_prediction(response).preferred == "B"
    revision_response = {
        "content": [
            {
                "text": json.dumps(
                    {"assessments": {"A": _assessment(2), "B": _assessment(4)}},
                    ensure_ascii=False,
                )
            }
        ]
    }
    assert closure_judge._extract_revision_prediction(revision_response).preferred == "B"
    with pytest.raises(ValueError, match="max_tokens"):
        closure_judge._extract_revision_prediction(
            {
                "stop_reason": "max_tokens",
                "content": [{"text": '{"assessments":'}],
            }
        )
    with pytest.raises(ValueError, match="content"):
        closure_judge._extract_prediction({"content": []})
    with pytest.raises(ValueError, match="no JSON text"):
        closure_judge._extract_prediction({"content": [{}]})
    assert closure_judge._aws_base("p", "r")[:5] == ["aws", "--profile", "p", "--region", "r"]
    count_input = closure_judge._count_tokens_input(b"request")
    assert set(count_input) == {"invokeModel"}
    assert "modelId" not in count_input
    count_path = tmp_path / "count.json"
    count_command = closure_judge._count_tokens_command(
        count_path, profile="default", region="ap-northeast-1", model_id="model"
    )
    model_index = count_command.index("--model-id")
    assert count_command[model_index + 1] == "model"
    cli_result = closure_judge._serialize_cli_result({"symbol": "≈"})
    assert "\\u2248" in cli_result
    cli_result.encode("cp932")


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (
            {
                "preferred": "X",
                "confidence": "high",
                "observed_evidence": [],
                "reason": "x",
            },
            "preferred",
        ),
        (
            {
                "preferred": "A",
                "confidence": "certain",
                "observed_evidence": [],
                "reason": "x",
            },
            "confidence",
        ),
        (
            {
                "preferred": "A",
                "confidence": "low",
                "observed_evidence": [1],
                "reason": "x",
            },
            "observed_evidence",
        ),
        (
            {
                "preferred": "A",
                "confidence": "low",
                "observed_evidence": [],
                "reason": 1,
            },
            "reason",
        ),
    ],
)
def test_prediction_rejects_invalid_fields(value: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_prediction(value)


def test_cost_normalization_empty_evaluation_and_retry_failures(tmp_path: Path) -> None:
    assert normalize_preferred("A", reversed_order=False) == "A"
    with pytest.raises(ValueError, match="invalid preferred"):
        normalize_preferred("X", reversed_order=True)
    with pytest.raises(ValueError, match="token counts"):
        estimate_max_cost([1], request_count=2)
    assert evaluate_predictions(Calibration((), (), "A" * 64), {})["permutation_p_value"] == 1

    def timeout(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, 330)

    with pytest.raises(RuntimeError, match="ambiguous timeout"):
        execute_with_retry(
            ["aws"],
            output_path=tmp_path / "timeout.json",
            command_runner=timeout,
            sleep=lambda _seconds: None,
        )

    def denied(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "AccessDeniedException")

    with pytest.raises(RuntimeError, match="without safe retry"):
        execute_with_retry(
            ["aws"],
            output_path=tmp_path / "denied.json",
            command_runner=denied,
            sleep=lambda _seconds: None,
        )

    with pytest.raises(RuntimeError, match="without safe retry"):
        execute_with_retry(
            ["aws"],
            output_path=tmp_path / "throttled.json",
            command_runner=lambda command: subprocess.CompletedProcess(
                command, 1, "", "ServiceUnavailableException"
            ),
            sleep=lambda _seconds: None,
            max_attempts=1,
        )
