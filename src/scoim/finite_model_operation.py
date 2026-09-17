"""Finite, recorded model operation with one bounded content repair."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator

from llm_musical_composer.run_state import (
    RunStore,
    StateConflictError,
    sha256_bytes,
    sha256_json,
    sha256_text,
)

from .flow_proposal import _call_runner
from .proposal import ProposalRunner
from .validation import CheckResult, IssueCode, ValidationIssue

ContentValidator = Callable[
    [Mapping[str, object]],
    tuple[ValidationIssue, ...],
]


@dataclass(frozen=True, slots=True)
class FiniteModelOperationResult:
    outcome: str
    response: dict[str, object] | None
    repaired: bool
    issues: tuple[ValidationIssue, ...]


def check_finite_model_operation_records(run_dir: str | Path) -> CheckResult:
    """Check saved attempt hashes, validation records, and acceptance links offline."""
    root = Path(run_dir).resolve()
    issues: list[ValidationIssue] = []
    attempts_by_relative_path: dict[str, Path] = {}
    for attempt in sorted((root / "attempts").glob("*/attempt-*")):
        if not attempt.is_dir():
            continue
        relative_attempt = attempt.relative_to(root).as_posix()
        attempts_by_relative_path[relative_attempt] = attempt
        terminal = _load_record_for_check(root, attempt / "terminal.json", issues)
        if terminal is None:
            continue
        raw_path = attempt / "response.staged.json"
        validation_path = attempt / "validation.json"
        if terminal.get("status") != "completed" or not raw_path.is_file():
            continue
        response_sha256 = sha256_bytes(raw_path.read_bytes())
        if terminal.get("response_sha256") != response_sha256:
            issues.append(_lineage_issue(root, attempt / "terminal.json", "response hash differs"))
        validation = _load_record_for_check(root, validation_path, issues)
        if validation is None:
            continue
        if validation.get("response_sha256") != response_sha256:
            issues.append(_lineage_issue(root, validation_path, "response hash differs"))
        if validation.get("status") not in {"valid", "invalid"} or not isinstance(
            validation.get("issues"), list
        ):
            issues.append(_lineage_issue(root, validation_path, "result is invalid"))

    for accepted_path in sorted((root / "events").glob("*/accepted.json")):
        accepted = _load_record_for_check(root, accepted_path, issues)
        if accepted is None:
            continue
        attempt_value = accepted.get("accepted_attempt")
        operation_id = accepted_path.parent.name
        if (
            not isinstance(attempt_value, str)
            or attempt_value not in attempts_by_relative_path
            or not attempt_value.startswith(f"attempts/{operation_id}/attempt-")
        ):
            issues.append(_lineage_issue(root, accepted_path, "attempt reference is invalid"))
            continue
        attempt = attempts_by_relative_path[attempt_value]
        raw_path = attempt / "response.staged.json"
        validation_path = attempt / "validation.json"
        if not raw_path.is_file():
            issues.append(_lineage_issue(root, raw_path, "accepted response is missing"))
            continue
        response_sha256 = sha256_bytes(raw_path.read_bytes())
        if accepted.get("response_sha256") != response_sha256:
            issues.append(_lineage_issue(root, accepted_path, "response hash differs"))
        decoded, decode_issues = _decode_response(raw_path.read_bytes())
        if decode_issues or decoded != accepted.get("response"):
            issues.append(_lineage_issue(root, accepted_path, "response differs from attempt"))
        validation = _load_record_for_check(root, validation_path, issues)
        if validation is not None and (
            validation.get("status") != "valid" or validation.get("issues") != []
        ):
            issues.append(_lineage_issue(root, validation_path, "accepted result is not valid"))
        if validation is not None and (
            validation.get("schema_sha256") != accepted.get("schema_sha256")
        ):
            issues.append(_lineage_issue(root, accepted_path, "schema hash differs"))
    return CheckResult(not issues, tuple(issues))


def execute_finite_model_operation(
    *,
    store: RunStore,
    operation_id: str,
    prompt: str,
    schema: Mapping[str, object],
    immutable_input: Mapping[str, object],
    runner: ProposalRunner,
    validate_content: ContentValidator,
) -> FiniteModelOperationResult:
    """Run one typed operation and accept only a schema- and content-valid response."""
    accepted = _read_accepted_response(
        store,
        operation_id,
        prompt,
        schema,
        immutable_input,
        validate_content,
    )
    if accepted is not None:
        return accepted
    attempts = store.attempt_dirs(operation_id)
    ambiguous_attempts = [
        attempt for attempt in attempts if not (attempt / "terminal.json").is_file()
    ]
    if ambiguous_attempts:
        issue = ValidationIssue(
            IssueCode.RUNNER_FAILED,
            "An existing model attempt has no terminal result",
            f"/{ambiguous_attempts[0].relative_to(store.run_dir).as_posix()}",
        )
        return FiniteModelOperationResult("interrupted", None, False, (issue,))
    if attempts:
        latest = attempts[-1]
        terminal = json.loads((latest / "terminal.json").read_text(encoding="utf-8"))
        if terminal.get("status") != "completed":
            outcome = "interrupted" if terminal.get("status") == "interrupted" else "runner_failed"
            issue = _runner_issue(f"The saved {operation_id} attempt did not complete")
            return FiniteModelOperationResult(outcome, None, len(attempts) > 1, (issue,))
        staged_path = latest / "response.staged.json"
        if not staged_path.is_file():
            issue = _runner_issue(f"The saved {operation_id} response is missing")
            return FiniteModelOperationResult("interrupted", None, len(attempts) > 1, (issue,))
        raw_response = staged_path.read_bytes()
        recovered_response, recovered_issues = _decode_response(raw_response)
        if recovered_response is not None:
            recovered_issues += _response_issues(recovered_response, schema, validate_content)
        _record_attempt_validation(store, latest, schema, recovered_issues)
        if not recovered_issues and recovered_response is not None:
            return _accept_response(
                store,
                operation_id,
                prompt,
                schema,
                immutable_input,
                recovered_response,
                latest,
                repaired=len(attempts) > 1,
            )
        if len(attempts) > 1:
            return FiniteModelOperationResult("content_invalid", None, True, recovered_issues)
        return _repair_operation(
            store=store,
            operation_id=operation_id,
            original_prompt=prompt,
            previous_response=(
                recovered_response
                if recovered_response is not None
                else raw_response.decode("utf-8", errors="replace")
            ),
            previous_response_sha256=sha256_bytes(raw_response),
            issues=recovered_issues,
            schema=schema,
            immutable_input=immutable_input,
            runner=runner,
            validate_content=validate_content,
        )
    response, run, attempt = _call_runner(store, operation_id, prompt, schema, runner)
    if response is None:
        if _is_content_failure(run):
            assert run.raw_response is not None
            _record_attempt_validation(store, attempt, schema, run.issues)
            return _repair_operation(
                store=store,
                operation_id=operation_id,
                original_prompt=prompt,
                previous_response=run.raw_response.decode("utf-8", errors="replace"),
                previous_response_sha256=sha256_bytes(run.raw_response),
                issues=run.issues,
                schema=schema,
                immutable_input=immutable_input,
                runner=runner,
                validate_content=validate_content,
            )
        return FiniteModelOperationResult(
            "runner_failed",
            None,
            False,
            run.issues or (_runner_issue(f"The {operation_id} operation failed"),),
        )
    issues = _response_issues(response, schema, validate_content)
    _record_attempt_validation(store, attempt, schema, issues)
    if issues:
        return _repair_operation(
            store=store,
            operation_id=operation_id,
            original_prompt=prompt,
            previous_response=response,
            previous_response_sha256=sha256_json(response),
            issues=issues,
            schema=schema,
            immutable_input=immutable_input,
            runner=runner,
            validate_content=validate_content,
        )
    return _accept_response(
        store,
        operation_id,
        prompt,
        schema,
        immutable_input,
        response,
        attempt,
        repaired=False,
    )


def _repair_operation(
    *,
    store: RunStore,
    operation_id: str,
    original_prompt: str,
    previous_response: object,
    previous_response_sha256: str,
    issues: tuple[ValidationIssue, ...],
    schema: Mapping[str, object],
    immutable_input: Mapping[str, object],
    runner: ProposalRunner,
    validate_content: ContentValidator,
) -> FiniteModelOperationResult:
    store.snapshot_json(
        f"repairs/{operation_id}.json",
        {
            "policy_id": "whole-response-content-repair-v1",
            "source_response_sha256": previous_response_sha256,
            "issues": _issue_values(issues),
        },
    )
    repaired_response, run, attempt = _call_runner(
        store,
        operation_id,
        _repair_prompt(
            operation_id,
            original_prompt,
            previous_response,
            issues,
            immutable_input,
        ),
        schema,
        runner,
    )
    if repaired_response is None:
        if _is_content_failure(run):
            _record_attempt_validation(store, attempt, schema, run.issues)
        outcome = "content_invalid" if _is_content_failure(run) else "runner_failed"
        return FiniteModelOperationResult(
            outcome,
            None,
            True,
            run.issues or (_runner_issue(f"The {operation_id} repair failed"),),
        )
    repaired_issues = _response_issues(repaired_response, schema, validate_content)
    _record_attempt_validation(store, attempt, schema, repaired_issues)
    if repaired_issues:
        return FiniteModelOperationResult("content_invalid", None, True, repaired_issues)
    return _accept_response(
        store,
        operation_id,
        original_prompt,
        schema,
        immutable_input,
        repaired_response,
        attempt,
        repaired=True,
    )


def _accept_response(
    store: RunStore,
    operation_id: str,
    prompt: str,
    schema: Mapping[str, object],
    immutable_input: Mapping[str, object],
    response: dict[str, object],
    attempt: Path,
    *,
    repaired: bool,
) -> FiniteModelOperationResult:
    store.snapshot_json(
        f"events/{operation_id}/accepted.json",
        {
            "prompt_sha256": sha256_text(prompt),
            "schema_sha256": sha256_json(schema),
            "immutable_input_sha256": sha256_json(immutable_input),
            "accepted_attempt": attempt.relative_to(store.run_dir).as_posix(),
            "response_sha256": _attempt_response_sha256(attempt),
            "repaired": repaired,
            "response": response,
        },
    )
    return FiniteModelOperationResult("complete", response, repaired, ())


def _record_attempt_validation(
    store: RunStore,
    attempt: Path,
    schema: Mapping[str, object],
    issues: tuple[ValidationIssue, ...],
) -> None:
    store.snapshot_json(
        attempt.relative_to(store.run_dir) / "validation.json",
        _validation_record(_attempt_response_sha256(attempt), schema, issues),
    )


def _attempt_response_sha256(attempt: Path) -> str:
    return sha256_bytes((attempt / "response.staged.json").read_bytes())


def _validation_record(
    response_sha256: str,
    schema: Mapping[str, object],
    issues: tuple[ValidationIssue, ...],
) -> dict[str, object]:
    return {
        "status": "invalid" if issues else "valid",
        "response_sha256": response_sha256,
        "schema_sha256": sha256_json(schema),
        "issues": _issue_values(issues),
    }


def _read_record(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StateConflictError(f"saved {label} record is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise StateConflictError(f"saved {label} record must be an object: {path}")
    return value


def _load_record_for_check(
    root: Path,
    path: Path,
    issues: list[ValidationIssue],
) -> dict[str, object] | None:
    try:
        return _read_record(path, path.stem)
    except StateConflictError:
        issues.append(_lineage_issue(root, path, "record is missing or unreadable"))
        return None


def _lineage_issue(root: Path, path: Path, message: str) -> ValidationIssue:
    try:
        relative_path = path.relative_to(root).as_posix()
    except ValueError:
        relative_path = path.as_posix()
    return ValidationIssue(
        IssueCode.LINEAGE_MISMATCH,
        message,
        f"/{relative_path}",
    )


def _read_accepted_response(
    store: RunStore,
    operation_id: str,
    prompt: str,
    schema: Mapping[str, object],
    immutable_input: Mapping[str, object],
    validate_content: ContentValidator,
) -> FiniteModelOperationResult | None:
    path = store.run_dir / "events" / operation_id / "accepted.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    expected_hashes = {
        "prompt_sha256": sha256_text(prompt),
        "schema_sha256": sha256_json(schema),
        "immutable_input_sha256": sha256_json(immutable_input),
    }
    if any(record.get(key) != value for key, value in expected_hashes.items()):
        raise StateConflictError(f"accepted operation input changed: {operation_id}")
    response = record.get("response")
    if not isinstance(response, dict):
        raise StateConflictError(f"accepted operation response is invalid: {operation_id}")
    issues = _response_issues(response, schema, validate_content)
    if issues:
        raise StateConflictError(f"accepted operation response no longer validates: {operation_id}")
    attempt_value = record.get("accepted_attempt")
    attempt_by_path = {
        attempt.relative_to(store.run_dir).as_posix(): attempt
        for attempt in store.attempt_dirs(operation_id)
    }
    if not isinstance(attempt_value, str) or attempt_value not in attempt_by_path:
        raise StateConflictError(f"accepted operation attempt reference is invalid: {operation_id}")
    attempt = attempt_by_path[attempt_value]
    raw_response_path = attempt / "response.staged.json"
    validation_path = attempt / "validation.json"
    terminal_path = attempt / "terminal.json"
    if not raw_response_path.is_file() or not validation_path.is_file():
        raise StateConflictError(f"accepted operation validation record is missing: {operation_id}")
    if not terminal_path.is_file():
        raise StateConflictError(f"accepted operation terminal record is missing: {operation_id}")
    raw_response = raw_response_path.read_bytes()
    response_sha256 = sha256_bytes(raw_response)
    if record.get("response_sha256") != response_sha256:
        raise StateConflictError(f"accepted operation response hash is invalid: {operation_id}")
    decoded, decode_issues = _decode_response(raw_response)
    if decode_issues or decoded != response:
        raise StateConflictError(
            f"accepted operation response differs from its attempt: {operation_id}"
        )
    validation = _read_record(validation_path, "validation")
    expected_validation = _validation_record(response_sha256, schema, issues)
    if validation != expected_validation:
        raise StateConflictError(f"accepted operation validation record is invalid: {operation_id}")
    terminal = _read_record(terminal_path, "terminal")
    if terminal.get("status") != "completed" or terminal.get("response_sha256") != response_sha256:
        raise StateConflictError(f"accepted operation terminal record is invalid: {operation_id}")
    repaired = record.get("repaired")
    if not isinstance(repaired, bool):
        raise StateConflictError(f"accepted operation repair state is invalid: {operation_id}")
    return FiniteModelOperationResult("complete", response, repaired, ())


def _schema_issues(
    response: Mapping[str, object], schema: Mapping[str, object]
) -> tuple[ValidationIssue, ...]:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(response),
        key=lambda item: (tuple(str(part) for part in item.absolute_path), item.message),
    )
    return tuple(
        ValidationIssue(
            IssueCode.MODEL_OUTPUT_INVALID,
            error.message,
            "".join(f"/{part}" for part in error.absolute_path),
        )
        for error in errors
    )


def _response_issues(
    response: Mapping[str, object],
    schema: Mapping[str, object],
    validate_content: ContentValidator,
) -> tuple[ValidationIssue, ...]:
    schema_issues = _schema_issues(response, schema)
    if schema_issues:
        return schema_issues
    return validate_content(response)


def _decode_response(
    raw_response: bytes,
) -> tuple[dict[str, object] | None, tuple[ValidationIssue, ...]]:
    try:
        decoded = json.loads(raw_response.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, (ValidationIssue(IssueCode.MODEL_OUTPUT_INVALID, str(error), ""),)
    if not isinstance(decoded, dict):
        return None, (
            ValidationIssue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "Response must be an object",
                "",
            ),
        )
    return decoded, ()


def _runner_issue(message: str) -> ValidationIssue:
    return ValidationIssue(IssueCode.RUNNER_FAILED, message, "/runner")


def _is_content_failure(run: object) -> bool:
    issues = getattr(run, "issues", ())
    return (
        getattr(run, "terminal_state", None) == "completed"
        and getattr(run, "raw_response", None) is not None
        and bool(issues)
        and all(issue.code == IssueCode.MODEL_OUTPUT_INVALID for issue in issues)
    )


def _repair_prompt(
    operation_id: str,
    original_prompt: str,
    previous_response: object,
    issues: tuple[ValidationIssue, ...],
    immutable_input: Mapping[str, object],
) -> str:
    context = {
        "operation": operation_id,
        "original_prompt": original_prompt,
        "immutable_input": dict(immutable_input),
        "previous_response": previous_response,
        "all_issues": _issue_values(issues),
    }
    return (
        "前の応答には次の問題があります。すべて直した応答全体を、同じSchemaで"
        "1回だけ再提出してください。immutable_inputは変更しないでください。\n\n"
        f"入力: {json.dumps(context, ensure_ascii=False, sort_keys=True)}\n"
    )


def _issue_values(issues: tuple[ValidationIssue, ...]) -> list[dict[str, str]]:
    return [
        {"code": issue.code.value, "message": issue.message, "path": issue.path} for issue in issues
    ]
