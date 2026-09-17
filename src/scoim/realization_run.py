"""Durable work-run records for checked SCoIM realization operations."""

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from llm_musical_composer.run_state import (
    InterruptedAttemptError,
    RunLock,
    RunStore,
    StateConflictError,
    atomic_write_bytes,
    atomic_write_json,
    sha256_bytes,
    sha256_file,
    sha256_json,
    sha256_text,
)

from .proposal import ProposalRunner, RunnerPreflightError, preflight_runner
from .realization_script import realization_script_sha256
from .realization_workspace import (
    CheckedWorkspaceDiff,
    RealizationWorkspace,
    apply_checked_diff,
    checked_diff_from_dict,
    checked_diff_to_dict,
    create_workspace,
    workspace_from_dict,
    workspace_to_dict,
)
from .validation import ValidationIssue

_SAFE_INSTANCE_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_EVENT_ORDER = {
    "request.json": ("model_request", 0),
    "response.json": ("model_response", 1),
    "diff.json": ("checked_diff", 2),
    "review.json": ("review_decision", 3),
}


@dataclass(frozen=True, slots=True)
class OperationInstance:
    """One operation invocation declared by an immutable run spec."""

    instance_id: str
    operation: str
    targets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RealizationRunStatus:
    """Reconstructed summary of one work-in-progress realization run."""

    status: str
    workspace: RealizationWorkspace
    completed_operations: tuple[str, ...]
    awaiting_review: str | None
    issues: tuple[ValidationIssue, ...]


class RealizationRunStore:
    """Append immutable realization events and rebuild their current state."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir).resolve()

    def initialize(
        self,
        document: dict[str, object],
        *,
        operations: tuple[OperationInstance, ...],
        review_after: frozenset[str],
        model: str,
        max_calls: int,
        content_repair_limit: int = 0,
        initial_workspace: RealizationWorkspace | None = None,
        source_workspace_record_sha256: str | None = None,
    ) -> None:
        instance_ids = tuple(operation.instance_id for operation in operations)
        if len(set(instance_ids)) != len(instance_ids) or any(
            not _SAFE_INSTANCE_ID.fullmatch(instance_id) for instance_id in instance_ids
        ):
            raise ValueError("Operation instance IDs must be unique safe identifiers")
        if not review_after.issubset(instance_ids):
            raise ValueError("review_after contains an unknown operation instance")
        if content_repair_limit < 0:
            raise ValueError("content_repair_limit must not be negative")
        initial = initial_workspace if initial_workspace is not None else create_workspace(document)
        initial = workspace_from_dict(workspace_to_dict(initial))
        script_hash = realization_script_sha256(document)
        if initial.approved_script_sha256 != script_hash:
            raise ValueError("The initial workspace belongs to a different approved script")
        if initial_workspace is None and source_workspace_record_sha256 is not None:
            raise ValueError("A source workspace hash requires an initial workspace")
        store = RunStore(self.run_dir, max_calls=max_calls)
        store.initialize(
            {
                "schema_version": 1,
                "approved_script_sha256": script_hash,
                "initial_workspace_record_sha256": initial.workspace_record_sha256,
                "source_workspace_record_sha256": source_workspace_record_sha256,
                "operations": [
                    {
                        "instance_id": operation.instance_id,
                        "operation": operation.operation,
                        "targets": list(operation.targets),
                    }
                    for operation in operations
                ],
                "review_after": sorted(review_after),
                "model": model,
                "max_calls": max_calls,
                "content_repair_limit": content_repair_limit,
            }
        )
        store.snapshot_json("inputs/approved-script.json", document)
        store.snapshot_json("inputs/initial-workspace.json", workspace_to_dict(initial))
        self.rebuild()

    def record_request(self, instance_id: str, request: dict[str, object]) -> None:
        self._record(instance_id, "request.json", request)

    def record_response(self, instance_id: str, response: object) -> None:
        self._record(instance_id, "response.json", response)

    def record_checked_diff(self, instance_id: str, checked_diff: CheckedWorkspaceDiff) -> None:
        self._record(instance_id, "diff.json", checked_diff_to_dict(checked_diff))

    def record_review(
        self,
        instance_id: str,
        *,
        target_diff_sha256: str,
        decision: str,
        actor: str,
        replacement_diff: CheckedWorkspaceDiff | None = None,
    ) -> None:
        """Record one approval or checked replacement for a pending diff."""
        if decision not in {"approve", "replace"}:
            raise ValueError("Review decision must be approve or replace")
        if actor not in {"human", "llm"}:
            raise ValueError("Review actor must be human or llm")
        diff_path = self.run_dir / "events" / instance_id / "diff.json"
        current_diff = checked_diff_from_dict(
            cast(dict[str, object], self._event_payload(instance_id, diff_path))
        )
        if checked_diff_sha256(current_diff) != target_diff_sha256:
            raise ValueError("Review target does not match the saved checked diff")
        if (decision == "replace") != (replacement_diff is not None):
            raise ValueError("Only a replacement review may contain replacement_diff")
        if replacement_diff is not None and replacement_diff.write_keys != _operation_write_keys(
            self._operation(instance_id)
        ):
            raise StateConflictError(
                f"review replacement does not match declared targets: {instance_id}"
            )
        self._record(
            instance_id,
            "review.json",
            {
                "decision": decision,
                "actor": actor,
                "target_diff_sha256": target_diff_sha256,
                "replacement_diff": (
                    checked_diff_to_dict(replacement_diff) if replacement_diff is not None else None
                ),
            },
        )

    def pending_checked_diff(self) -> tuple[str, CheckedWorkspaceDiff]:
        """Return the diff currently awaiting a review decision."""
        status = self.rebuild()
        if status.awaiting_review is None:
            raise ValueError("The run is not awaiting review")
        instance_id = status.awaiting_review
        value = self._event_payload(
            instance_id,
            self.run_dir / "events" / instance_id / "diff.json",
        )
        if not isinstance(value, dict):
            raise StateConflictError("Saved checked diff is not an object")
        return instance_id, checked_diff_from_dict(value)

    def execute_model_call(
        self,
        instance_id: str,
        prompt: str,
        response_schema_path: Path,
        runner: ProposalRunner,
    ) -> dict[str, object]:
        """Execute at most one durable model attempt for an operation instance."""
        with RunLock(self.run_dir / ".run.lock"):
            self._operation(instance_id)
            store = self._store_from_spec()
            spec = store.read_spec()
            response_path = self.run_dir / "events" / instance_id / "response.json"
            if response_path.is_file():
                return cast(dict[str, object], self._event_payload(instance_id, response_path))
            attempts = store.attempt_dirs(instance_id)
            if attempts:
                recovered = _recover_completed_response(store, attempts)
                if recovered is not None:
                    self._record_unlocked(instance_id, "response.json", recovered)
                    return recovered
                raise InterruptedAttemptError(
                    f"Model attempt cannot be retried automatically: {instance_id}"
                )
            _ensure_runner_available(runner)
            request = _model_request(instance_id, spec, prompt, response_schema_path)
            self._record_unlocked(instance_id, "request.json", request)
            _, response = _execute_attempt(
                store, request, prompt, response_schema_path, runner, spec
            )
            self._record_unlocked(instance_id, "response.json", response)
            return cast(dict[str, object], response)

    def execute_checked_model_call(
        self,
        instance_id: str,
        prompt: str,
        response_schema_path: Path,
        runner: ProposalRunner,
        build_checked_diff: Callable[[Mapping[str, object]], CheckedWorkspaceDiff],
        build_repair_prompt: Callable[
            [str, Mapping[str, object], tuple[ValidationIssue, ...]], str
        ],
    ) -> tuple[dict[str, object], CheckedWorkspaceDiff]:
        """Run, inspect, and finitely repair one model-owned operation."""
        with RunLock(self.run_dir / ".run.lock"):
            self._operation(instance_id)
            store = self._store_from_spec()
            spec = store.read_spec()
            repair_limit = spec.get("content_repair_limit", 0)
            if not isinstance(repair_limit, int) or repair_limit < 0:
                raise StateConflictError("run spec contains an invalid content repair limit")
            response_path = self.run_dir / "events" / instance_id / "response.json"
            if response_path.is_file():
                response = cast(dict[str, object], self._event_payload(instance_id, response_path))
                return response, build_checked_diff(response)

            attempts = store.attempt_dirs(instance_id)
            if len(attempts) > repair_limit + 1:
                raise StateConflictError("Saved model attempts exceed the content repair limit")
            while True:
                if attempts:
                    attempt_dir = attempts[-1]
                    response = _recover_attempt_response(attempt_dir)
                    if response is None:
                        raise InterruptedAttemptError(
                            f"Model attempt cannot be retried automatically: {instance_id}"
                        )
                else:
                    _ensure_runner_available(runner)
                    request = _model_request(instance_id, spec, prompt, response_schema_path)
                    self._record_unlocked(instance_id, "request.json", request)
                    attempt_dir, response = _execute_attempt(
                        store, request, prompt, response_schema_path, runner, spec
                    )
                    attempts = store.attempt_dirs(instance_id)

                checked_diff = build_checked_diff(response)
                validation = _validation_record(attempt_dir, checked_diff.validation_issues)
                store.snapshot_json(
                    attempt_dir.relative_to(self.run_dir) / "validation.json", validation
                )
                if not checked_diff.validation_issues or len(attempts) >= repair_limit + 1:
                    self._record_unlocked(instance_id, "response.json", response)
                    return response, checked_diff

                repair_prompt = build_repair_prompt(
                    prompt, response, checked_diff.validation_issues
                )
                _ensure_runner_available(runner)
                request = _model_request(instance_id, spec, repair_prompt, response_schema_path)
                attempt_dir, response = _execute_attempt(
                    store, request, repair_prompt, response_schema_path, runner, spec
                )
                attempts = store.attempt_dirs(instance_id)

    def rebuild(self) -> RealizationRunStatus:
        store = self._store_from_spec()
        spec = store.read_spec()
        workspace = workspace_from_dict(
            _read_object(self.run_dir / "inputs" / "initial-workspace.json")
        )
        document = _read_object(self.run_dir / "inputs" / "approved-script.json")
        actual_script_hash = realization_script_sha256(document)
        if (
            spec.get("approved_script_sha256") != actual_script_hash
            or workspace.approved_script_sha256 != actual_script_hash
        ):
            raise StateConflictError(
                "approved script snapshot does not match the immutable run inputs"
            )
        expected_initial_hash = spec.get("initial_workspace_record_sha256")
        if (
            expected_initial_hash is not None
            and expected_initial_hash != workspace.workspace_record_sha256
        ):
            raise StateConflictError(
                "initial workspace snapshot does not match the immutable run spec"
            )
        completed: list[str] = []
        awaiting_review: str | None = None
        issues: tuple[ValidationIssue, ...] = ()
        status = "initialized"
        operations = cast(list[dict[str, object]], spec["operations"])
        review_after = set(cast(list[str], spec["review_after"]))
        for operation in operations:
            instance_id = cast(str, operation["instance_id"])
            event_dir = self.run_dir / "events" / instance_id
            if not (event_dir / "diff.json").is_file():
                status = "running" if event_dir.exists() else status
                break
            checked_diff = checked_diff_from_dict(
                cast(
                    dict[str, object],
                    self._event_payload(instance_id, event_dir / "diff.json"),
                )
            )
            expected_write_keys = _operation_write_keys(operation)
            if checked_diff.write_keys != expected_write_keys:
                raise StateConflictError(
                    f"checked diff does not match declared targets: {instance_id}"
                )
            if instance_id in review_after:
                review_path = event_dir / "review.json"
                if not review_path.is_file():
                    status = "awaiting_review"
                    awaiting_review = instance_id
                    break
                review = cast(dict[str, object], self._event_payload(instance_id, review_path))
                if review.get("target_diff_sha256") != checked_diff_sha256(checked_diff):
                    raise StateConflictError(
                        f"review target does not match saved diff: {instance_id}"
                    )
                decision = review.get("decision")
                replacement = review.get("replacement_diff")
                if decision == "approve" and replacement is not None:
                    raise StateConflictError(
                        f"approval review contains a replacement: {instance_id}"
                    )
                if decision == "replace":
                    if not isinstance(replacement, dict):
                        raise StateConflictError(
                            f"replacement review has no checked diff: {instance_id}"
                        )
                    checked_diff = checked_diff_from_dict(replacement)
                elif decision != "approve":
                    raise StateConflictError(f"review has an unsupported decision: {instance_id}")
                if checked_diff.write_keys != expected_write_keys:
                    raise StateConflictError(
                        f"review replacement does not match declared targets: {instance_id}"
                    )
            applied = apply_checked_diff(workspace, checked_diff)
            if applied.workspace is None:
                status = "failed"
                issues = applied.issues
                break
            workspace = applied.workspace
            completed.append(instance_id)
            status = "running"
        else:
            status = "completed"
        summary = RealizationRunStatus(
            status=status,
            workspace=workspace,
            completed_operations=tuple(completed),
            awaiting_review=awaiting_review,
            issues=issues,
        )
        atomic_write_json(
            self.run_dir / "realization-state.json",
            {
                "schema_version": 1,
                "status": summary.status,
                "completed_operations": list(summary.completed_operations),
                "awaiting_review": summary.awaiting_review,
                "issues": [
                    {
                        "code": issue.code.value,
                        "message": issue.message,
                        "path": issue.path,
                    }
                    for issue in summary.issues
                ],
                "workspace": workspace_to_dict(summary.workspace),
            },
        )
        return summary

    def _record(self, instance_id: str, filename: str, value: object) -> None:
        with RunLock(self.run_dir / ".run.lock"):
            self._operation(instance_id)
            self._record_unlocked(instance_id, filename, value)

    def _record_unlocked(self, instance_id: str, filename: str, value: object) -> None:
        if filename not in _EVENT_ORDER:
            raise ValueError(f"Unsupported realization event file: {filename}")
        event_type, offset = _EVENT_ORDER[filename]
        spec = self._store_from_spec().read_spec()
        operations = cast(list[dict[str, object]], spec["operations"])
        operation_index = next(
            index
            for index, operation in enumerate(operations)
            if operation.get("instance_id") == instance_id
        )
        self._store_from_spec().snapshot_json(
            Path("events") / instance_id / filename,
            {
                "schema_version": 1,
                "sequence": operation_index * len(_EVENT_ORDER) + offset,
                "event_type": event_type,
                "operation_instance_id": instance_id,
                "payload": value,
            },
        )
        self.rebuild()

    def _event_payload(self, instance_id: str, path: Path) -> object:
        filename = path.name
        if filename not in _EVENT_ORDER:
            raise StateConflictError(f"Unknown realization event file: {path}")
        expected_type, offset = _EVENT_ORDER[filename]
        spec = self._store_from_spec().read_spec()
        operations = cast(list[dict[str, object]], spec["operations"])
        operation_index = next(
            (
                index
                for index, operation in enumerate(operations)
                if operation.get("instance_id") == instance_id
            ),
            None,
        )
        record = _read_object(path)
        if (
            operation_index is None
            or record.get("schema_version") != 1
            or record.get("sequence") != operation_index * len(_EVENT_ORDER) + offset
            or record.get("event_type") != expected_type
            or record.get("operation_instance_id") != instance_id
            or "payload" not in record
        ):
            raise StateConflictError(f"Invalid realization event record: {path}")
        return record["payload"]

    def _store_from_spec(self) -> RunStore:
        spec = _read_object(self.run_dir / "run-spec.json")
        max_calls = spec.get("max_calls")
        if not isinstance(max_calls, int) or max_calls < 0:
            raise ValueError("run spec contains an invalid max_calls")
        return RunStore(self.run_dir, max_calls=max_calls)

    def _operation(self, instance_id: str) -> dict[str, object]:
        spec = self._store_from_spec().read_spec()
        operations = cast(list[dict[str, object]], spec["operations"])
        for operation in operations:
            if operation.get("instance_id") == instance_id:
                return operation
        raise ValueError(f"Unknown operation instance: {instance_id}")


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _operation_write_keys(operation: dict[str, object]) -> tuple[str, ...]:
    operation_name = operation.get("operation")
    targets = cast(list[str], operation.get("targets"))
    if operation_name == "plan":
        return ("/plan",)
    if operation_name == "harmony":
        return tuple(
            f"/harmonies/{target.replace('~', '~0').replace('/', '~1')}" for target in targets
        )
    if operation_name == "melody":
        return tuple(
            f"/melodies/{target.replace('~', '~0').replace('/', '~1')}" for target in targets
        )
    if operation_name == "transition-melody":
        return tuple(
            f"/transition_melodies/{target.replace('~', '~0').replace('/', '~1')}"
            for target in targets
        )
    if operation_name == "accompaniment":
        return tuple(
            f"/accompaniments/{target.replace('~', '~0').replace('/', '~1')}" for target in targets
        )
    if operation_name == "transition-accompaniment":
        return tuple(
            f"/transition_accompaniments/{target.replace('~', '~0').replace('/', '~1')}"
            for target in targets
        )
    if operation_name == "ending":
        return tuple(
            key
            for target in targets
            for key in (
                f"/harmonies/{target.replace('~', '~0').replace('/', '~1')}",
                f"/melodies/{target.replace('~', '~0').replace('/', '~1')}",
                f"/accompaniments/{target.replace('~', '~0').replace('/', '~1')}",
            )
        )
    if operation_name in {"performance-defaults", "performance"}:
        return tuple(
            f"/performances/{target.replace('~', '~0').replace('/', '~1')}" for target in targets
        )
    raise StateConflictError(f"Unknown realization operation: {operation_name}")


def checked_diff_sha256(checked_diff: CheckedWorkspaceDiff) -> str:
    """Return the immutable event fingerprint used by review decisions."""
    return sha256_json(checked_diff_to_dict(checked_diff))


def build_content_repair_prompt(
    original_prompt: str,
    upstream_workspace_sha256: str,
    response: Mapping[str, object],
    issues: tuple[ValidationIssue, ...],
) -> str:
    """Build the reproducible whole-response correction request."""
    context = {
        "immutable_upstream_workspace_sha256": upstream_workspace_sha256,
        "current_response": dict(response),
        "validation_issues": [
            {"code": issue.code.value, "message": issue.message, "path": issue.path}
            for issue in issues
        ],
    }
    return (
        "次の応答は内容検査に合格しませんでした。上流の入力と音楽上の意図を変えず、"
        "問題を解消した応答全体を、初回と同じSchemaで返してください。部分差分、説明文、"
        "Markdownは返さないでください。\n\n"
        "初回の指示:\n"
        + original_prompt
        + "\n\n検査結果と現在の応答:\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def _model_request(
    instance_id: str,
    spec: Mapping[str, object],
    prompt: str,
    response_schema_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "operation_instance_id": instance_id,
        "requested_model": spec["model"],
        "prompt_sha256": sha256_text(prompt),
        "response_schema_sha256": sha256_file(response_schema_path),
    }


def _execute_attempt(
    store: RunStore,
    request: Mapping[str, object],
    prompt: str,
    response_schema_path: Path,
    runner: ProposalRunner,
    spec: Mapping[str, object],
) -> tuple[Path, dict[str, object]]:
    instance_id = cast(str, request["operation_instance_id"])
    attempt_dir = store.reserve_attempt(instance_id, request, prompt)
    result = runner.run(prompt, response_schema_path)
    atomic_write_bytes(attempt_dir / "stdout.jsonl", result.events or b"")
    atomic_write_bytes(attempt_dir / "stderr.log", result.stderr or b"")
    atomic_write_json(
        attempt_dir / "runner.json",
        {
            "provider": result.provider,
            "model": result.model,
            "model_settings": dict(result.model_settings),
            "started": result.started,
            "terminal_state": result.terminal_state,
            "issues": [
                {
                    "code": issue.code.value,
                    "message": issue.message,
                    "path": issue.path,
                }
                for issue in result.issues
            ],
        },
    )
    if result.raw_response is not None:
        atomic_write_bytes(attempt_dir / "response.staged.json", result.raw_response)
    if result.model != spec["model"]:
        store.finalize_attempt(
            attempt_dir,
            "failed",
            returncode=None,
            detail="Model runner used a different model than the run spec",
        )
        raise StateConflictError("Model runner used a different model")
    if not result.started:
        store.finalize_attempt(
            attempt_dir,
            "failed",
            returncode=None,
            detail="Model runner failed before starting",
        )
        raise RuntimeError("Model runner failed before starting")
    if result.terminal_state in {"timeout", "interrupted"}:
        store.finalize_attempt(
            attempt_dir,
            "interrupted",
            returncode=None,
            detail=f"Model runner ended as {result.terminal_state}",
        )
        raise InterruptedAttemptError(f"Model attempt outcome is ambiguous: {instance_id}")
    if result.terminal_state != "completed" or result.raw_response is None:
        store.finalize_attempt(
            attempt_dir,
            "failed",
            returncode=None,
            detail=f"Model runner ended as {result.terminal_state}",
        )
        raise RuntimeError(f"Model attempt failed: {instance_id}")
    try:
        response = json.loads(result.raw_response.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        store.finalize_attempt(
            attempt_dir,
            "failed",
            returncode=None,
            detail=f"Invalid structured response: {error}",
        )
        raise ValueError("Model returned invalid structured JSON") from error
    if not isinstance(response, dict):
        store.finalize_attempt(
            attempt_dir,
            "failed",
            returncode=None,
            detail="Structured response is not an object",
        )
        raise ValueError("Model structured response must be an object")
    store.finalize_attempt(
        attempt_dir,
        "completed",
        returncode=0,
        response_sha256=sha256_bytes(result.raw_response),
    )
    return attempt_dir, cast(dict[str, object], response)


def _ensure_runner_available(runner: ProposalRunner) -> None:
    preflight_issues = preflight_runner(runner)
    if preflight_issues:
        raise RunnerPreflightError(preflight_issues)


def _recover_attempt_response(attempt: Path) -> dict[str, object] | None:
    terminal_path = attempt / "terminal.json"
    response_path = attempt / "response.staged.json"
    if not terminal_path.is_file() or not response_path.is_file():
        return None
    terminal = _read_object(terminal_path)
    content = response_path.read_bytes()
    if terminal.get("status") != "completed" or terminal.get("response_sha256") != sha256_bytes(
        content
    ):
        return None
    response = json.loads(content.decode("utf-8"))
    if not isinstance(response, dict):
        raise StateConflictError("Saved model response is not an object")
    return cast(dict[str, object], response)


def _validation_record(attempt_dir: Path, issues: tuple[ValidationIssue, ...]) -> dict[str, object]:
    raw_response = (attempt_dir / "response.staged.json").read_bytes()
    return {
        "schema_version": 1,
        "response_sha256": sha256_bytes(raw_response),
        "status": "invalid" if issues else "valid",
        "issues": [
            {"code": issue.code.value, "message": issue.message, "path": issue.path}
            for issue in issues
        ],
    }


def _recover_completed_response(store: RunStore, attempts: list[Path]) -> dict[str, object] | None:
    if len(attempts) != 1:
        raise StateConflictError("Multiple model attempts require manual review")
    return _recover_attempt_response(attempts[0])


def approve_pending_review(run_dir: Path, *, actor: str) -> RealizationRunStatus:
    """Approve the currently pending diff using no input other than its run directory."""
    run = RealizationRunStore(run_dir)
    instance_id, diff = run.pending_checked_diff()
    run.record_review(
        instance_id,
        target_diff_sha256=checked_diff_sha256(diff),
        decision="approve",
        actor=actor,
    )
    return run.rebuild()
