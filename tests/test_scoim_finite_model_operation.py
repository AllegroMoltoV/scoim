import json
from pathlib import Path

import pytest

from llm_musical_composer.run_state import RunStore, StateConflictError, sha256_bytes
from scoim.finite_model_operation import (
    check_finite_model_operation_records,
    execute_finite_model_operation,
)
from scoim.proposal import ProposalRun
from scoim.validation import IssueCode, ValidationIssue


class SequencedRunner:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        self.prompts.append(prompt)
        response = self.responses[len(self.prompts) - 1]
        return ProposalRun(
            provider="fixed",
            model="fixed",
            model_settings={},
            started=True,
            terminal_state="completed",
            raw_response=json.dumps(response).encode(),
            events=b"",
            stderr=b"",
            issues=(),
        )


class FailingRunner:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        self.calls += 1
        return ProposalRun(
            provider="fixed",
            model="fixed",
            model_settings={},
            started=True,
            terminal_state="failed",
            raw_response=None,
            events=b"",
            stderr=b"provider failure",
            issues=(ValidationIssue(IssueCode.RUNNER_FAILED, "provider failure", "/runner"),),
        )


_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["value"],
    "properties": {"value": {"type": "integer", "minimum": 1}},
}


def test_finite_model_operation_accepts_a_valid_response_once(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run", max_calls=2)
    store.initialize({"max_calls": 2})
    runner = SequencedRunner([{"value": 1}])

    result = execute_finite_model_operation(
        store=store,
        operation_id="test-operation",
        prompt="return a value",
        schema=_SCHEMA,
        immutable_input={"source": "test"},
        runner=runner,
        validate_content=lambda _response: (),
    )

    assert result.outcome == "complete"
    assert result.response == {"value": 1}
    assert result.repaired is False
    assert runner.prompts == ["return a value"]


def test_finite_model_operation_records_the_accepted_attempt_and_validation(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run", max_calls=2)
    store.initialize({"max_calls": 2})
    runner = SequencedRunner([{"value": 1}])

    result = execute_finite_model_operation(
        store=store,
        operation_id="test-operation",
        prompt="return a value",
        schema=_SCHEMA,
        immutable_input={"source": "test"},
        runner=runner,
        validate_content=lambda _response: (),
    )

    assert result.outcome == "complete"
    attempt = store.run_dir / "attempts" / "test-operation" / "attempt-001"
    terminal = json.loads((attempt / "terminal.json").read_text(encoding="utf-8"))
    validation = json.loads((attempt / "validation.json").read_text(encoding="utf-8"))
    accepted = json.loads(
        (store.run_dir / "events" / "test-operation" / "accepted.json").read_text(encoding="utf-8")
    )

    assert validation == {
        "issues": [],
        "response_sha256": terminal["response_sha256"],
        "schema_sha256": accepted["schema_sha256"],
        "status": "valid",
    }
    assert accepted["accepted_attempt"] == "attempts/test-operation/attempt-001"
    assert accepted["response_sha256"] == terminal["response_sha256"]


def test_finite_model_operation_repairs_schema_invalid_content_once(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run", max_calls=2)
    store.initialize({"max_calls": 2})
    runner = SequencedRunner([{}, {"value": 2}])
    checked_values: list[int] = []

    def validate(response: dict[str, object]) -> tuple[ValidationIssue, ...]:
        checked_values.append(int(response["value"]))
        return ()

    result = execute_finite_model_operation(
        store=store,
        operation_id="test-operation",
        prompt="return a value",
        schema=_SCHEMA,
        immutable_input={"source": "test"},
        runner=runner,
        validate_content=validate,
    )

    assert result.outcome == "complete"
    assert result.response == {"value": 2}
    assert result.repaired is True
    assert len(runner.prompts) == 2
    assert "return a value" in runner.prompts[1]
    assert "required property" in runner.prompts[1]
    assert checked_values == [2]
    attempt_root = store.run_dir / "attempts" / "test-operation"
    first_validation = json.loads(
        (attempt_root / "attempt-001" / "validation.json").read_text(encoding="utf-8")
    )
    second_validation = json.loads(
        (attempt_root / "attempt-002" / "validation.json").read_text(encoding="utf-8")
    )
    accepted = json.loads(
        (store.run_dir / "events" / "test-operation" / "accepted.json").read_text(encoding="utf-8")
    )
    assert first_validation["status"] == "invalid"
    assert second_validation["status"] == "valid"
    assert accepted["accepted_attempt"] == "attempts/test-operation/attempt-002"


def test_finite_model_operation_records_both_failed_content_attempts(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run", max_calls=2)
    store.initialize({"max_calls": 2})
    runner = SequencedRunner([{"value": 0}, {"value": 0}])

    result = execute_finite_model_operation(
        store=store,
        operation_id="test-operation",
        prompt="return a value",
        schema=_SCHEMA,
        immutable_input={"source": "test"},
        runner=runner,
        validate_content=lambda _response: (),
    )

    assert result.outcome == "content_invalid"
    attempt_root = store.run_dir / "attempts" / "test-operation"
    assert [
        json.loads((path / "validation.json").read_text(encoding="utf-8"))["status"]
        for path in sorted(attempt_root.glob("attempt-*"))
    ] == ["invalid", "invalid"]
    assert not (store.run_dir / "events" / "test-operation" / "accepted.json").exists()


def test_finite_model_operation_sends_all_content_issues_and_rechecks(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run", max_calls=2)
    store.initialize({"max_calls": 2})
    runner = SequencedRunner([{"value": 1}, {"value": 2}])
    checked_values: list[int] = []

    def validate(response: dict[str, object]) -> tuple[ValidationIssue, ...]:
        value = int(response["value"])
        checked_values.append(value)
        if value == 1:
            return (
                ValidationIssue(IssueCode.SEMANTIC_INVALID, "first problem", "/a"),
                ValidationIssue(IssueCode.SEMANTIC_INVALID, "second problem", "/b"),
            )
        return ()

    result = execute_finite_model_operation(
        store=store,
        operation_id="test-operation",
        prompt="return a value",
        schema=_SCHEMA,
        immutable_input={"source": "test"},
        runner=runner,
        validate_content=validate,
    )

    assert result.outcome == "complete"
    assert checked_values == [1, 2]
    assert "first problem" in runner.prompts[1]
    assert "second problem" in runner.prompts[1]


def test_finite_model_operation_returns_typed_failure_for_schema_invalid_repair(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run", max_calls=2)
    store.initialize({"max_calls": 2})
    runner = SequencedRunner([{"value": 1}, {}])
    checked_values: list[int] = []

    def validate(response: dict[str, object]) -> tuple[ValidationIssue, ...]:
        value = int(response["value"])
        checked_values.append(value)
        return (ValidationIssue(IssueCode.SEMANTIC_INVALID, "change the value", "/value"),)

    result = execute_finite_model_operation(
        store=store,
        operation_id="test-operation",
        prompt="return a value",
        schema=_SCHEMA,
        immutable_input={"source": "test"},
        runner=runner,
        validate_content=validate,
    )

    assert result.outcome == "content_invalid"
    assert result.repaired is True
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert checked_values == [1]


def test_finite_model_operation_does_not_repair_a_runner_failure(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run", max_calls=2)
    store.initialize({"max_calls": 2})
    runner = FailingRunner()

    result = execute_finite_model_operation(
        store=store,
        operation_id="test-operation",
        prompt="return a value",
        schema=_SCHEMA,
        immutable_input={"source": "test"},
        runner=runner,
        validate_content=lambda _response: (),
    )

    assert result.outcome == "runner_failed"
    assert result.repaired is False
    assert runner.calls == 1


def test_finite_model_operation_reuses_an_accepted_response(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run", max_calls=2)
    store.initialize({"max_calls": 2})
    runner = SequencedRunner([{"value": 1}])
    arguments = {
        "store": store,
        "operation_id": "test-operation",
        "prompt": "return a value",
        "schema": _SCHEMA,
        "immutable_input": {"source": "test"},
        "runner": runner,
        "validate_content": lambda _response: (),
    }

    first = execute_finite_model_operation(**arguments)
    second = execute_finite_model_operation(**arguments)

    assert first.response == second.response == {"value": 1}
    assert len(runner.prompts) == 1


def test_finite_model_operation_rejects_an_accepted_response_without_its_validation(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run", max_calls=2)
    store.initialize({"max_calls": 2})
    runner = SequencedRunner([{"value": 1}])
    arguments = {
        "store": store,
        "operation_id": "test-operation",
        "prompt": "return a value",
        "schema": _SCHEMA,
        "immutable_input": {"source": "test"},
        "runner": runner,
        "validate_content": lambda _response: (),
    }
    execute_finite_model_operation(**arguments)
    validation_path = (
        store.run_dir / "attempts" / "test-operation" / "attempt-001" / "validation.json"
    )
    validation_path.unlink()

    with pytest.raises(StateConflictError, match="validation record"):
        execute_finite_model_operation(**arguments)


def test_finite_model_operation_records_can_be_checked_without_a_model(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run", max_calls=2)
    store.initialize({"max_calls": 2})
    runner = SequencedRunner([{"value": 1}])
    execute_finite_model_operation(
        store=store,
        operation_id="test-operation",
        prompt="return a value",
        schema=_SCHEMA,
        immutable_input={"source": "test"},
        runner=runner,
        validate_content=lambda _response: (),
    )

    valid = check_finite_model_operation_records(store.run_dir)

    assert valid.valid
    validation_path = (
        store.run_dir / "attempts" / "test-operation" / "attempt-001" / "validation.json"
    )
    validation_path.unlink()

    invalid = check_finite_model_operation_records(store.run_dir)

    assert not invalid.valid
    assert invalid.issues[0].code is IssueCode.LINEAGE_MISMATCH
    assert invalid.issues[0].path.endswith("/validation.json")


def test_finite_model_operation_record_rejects_an_attempt_from_another_operation(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run", max_calls=2)
    store.initialize({"max_calls": 2})
    runner = SequencedRunner([{"value": 1}, {"value": 1}])
    for operation_id in ("first-operation", "second-operation"):
        result = execute_finite_model_operation(
            store=store,
            operation_id=operation_id,
            prompt="return a value",
            schema=_SCHEMA,
            immutable_input={"source": "test"},
            runner=runner,
            validate_content=lambda _response: (),
        )
        assert result.outcome == "complete"
    accepted_path = store.run_dir / "events" / "first-operation" / "accepted.json"
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    accepted["accepted_attempt"] = "attempts/second-operation/attempt-001"
    accepted_path.write_text(json.dumps(accepted), encoding="utf-8")

    checked = check_finite_model_operation_records(store.run_dir)

    assert not checked.valid
    assert "attempt reference" in checked.issues[0].message


def test_finite_model_operation_stops_at_an_ambiguous_attempt(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run", max_calls=2)
    store.initialize({"max_calls": 2})
    store.reserve_attempt("test-operation", {"request": "test"}, "return a value")
    runner = SequencedRunner([])

    result = execute_finite_model_operation(
        store=store,
        operation_id="test-operation",
        prompt="return a value",
        schema=_SCHEMA,
        immutable_input={"source": "test"},
        runner=runner,
        validate_content=lambda _response: (),
    )

    assert result.outcome == "interrupted"
    assert len(runner.prompts) == 0


def test_finite_model_operation_recovers_a_completed_unaccepted_response(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run", max_calls=2)
    store.initialize({"max_calls": 2})
    attempt = store.reserve_attempt("test-operation", {"request": "test"}, "return a value")
    raw_response = json.dumps({"value": 1}).encode()
    (attempt / "response.staged.json").write_bytes(raw_response)
    store.finalize_attempt(
        attempt,
        "completed",
        returncode=0,
        response_sha256=sha256_bytes(raw_response),
    )
    runner = SequencedRunner([])

    result = execute_finite_model_operation(
        store=store,
        operation_id="test-operation",
        prompt="return a value",
        schema=_SCHEMA,
        immutable_input={"source": "test"},
        runner=runner,
        validate_content=lambda _response: (),
    )

    assert result.outcome == "complete"
    assert result.response == {"value": 1}
    assert len(runner.prompts) == 0


def test_finite_model_operation_rechecks_saved_schema_before_saved_content(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run", max_calls=2)
    store.initialize({"max_calls": 2})
    runner = SequencedRunner([{"value": 1}])
    checked_values: list[int] = []

    def validate(response: dict[str, object]) -> tuple[ValidationIssue, ...]:
        checked_values.append(int(response["value"]))
        return ()

    arguments = {
        "store": store,
        "operation_id": "test-operation",
        "prompt": "return a value",
        "schema": _SCHEMA,
        "immutable_input": {"source": "test"},
        "runner": runner,
        "validate_content": validate,
    }
    execute_finite_model_operation(**arguments)
    accepted_path = store.run_dir / "events" / "test-operation" / "accepted.json"
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    accepted["response"] = {}
    accepted_path.write_text(json.dumps(accepted), encoding="utf-8")

    with pytest.raises(StateConflictError, match="no longer validates"):
        execute_finite_model_operation(**arguments)

    assert checked_values == [1]


def test_finite_model_operation_repairs_a_saved_schema_invalid_response(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run", max_calls=2)
    store.initialize({"max_calls": 2})
    attempt = store.reserve_attempt("test-operation", {"request": "test"}, "return a value")
    raw_response = json.dumps({}).encode()
    (attempt / "response.staged.json").write_bytes(raw_response)
    store.finalize_attempt(
        attempt,
        "completed",
        returncode=0,
        response_sha256=sha256_bytes(raw_response),
    )
    runner = SequencedRunner([{"value": 2}])
    checked_values: list[int] = []

    def validate(response: dict[str, object]) -> tuple[ValidationIssue, ...]:
        checked_values.append(int(response["value"]))
        return ()

    result = execute_finite_model_operation(
        store=store,
        operation_id="test-operation",
        prompt="return a value",
        schema=_SCHEMA,
        immutable_input={"source": "test"},
        runner=runner,
        validate_content=validate,
    )

    assert result.outcome == "complete"
    assert result.repaired is True
    assert checked_values == [2]
