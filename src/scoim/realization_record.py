"""Self-contained audit records for completed staged realization runs."""

import base64
import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import cast

from jsonschema import Draft202012Validator

from .realization_operations import (
    build_accompaniment_checked_diff,
    build_ending_checked_diff,
    build_harmony_checked_diff,
    build_melody_checked_diff,
    build_performance_checked_diff,
    build_performance_defaults_checked_diff,
    build_plan_checked_diff,
    build_transition_accompaniment_checked_diff,
    build_transition_melody_checked_diff,
)
from .realization_run import (
    RealizationRunStore,
    build_content_repair_prompt,
    checked_diff_sha256,
)
from .realization_script import realization_script_sha256
from .realization_workspace import (
    CheckedWorkspaceDiff,
    RealizationWorkspace,
    apply_checked_diff,
    checked_diff_from_dict,
    checked_diff_to_dict,
    workspace_from_dict,
    workspace_to_dict,
)
from .validation import IssueCode, ValidationIssue

_PARENT_FILES = frozenset(
    {
        "run-spec.json",
        "run-state.json",
        "staged-realization-state.json",
        "inputs/approved-script.json",
    }
)
_ATTEMPT_FILES = frozenset(
    {
        "request.json",
        "prompt.md",
        "response.staged.json",
        "stdout.jsonl",
        "stderr.log",
        "runner.json",
        "terminal.json",
    }
)
_CONTENT_REPAIR_ATTEMPT_FILES = _ATTEMPT_FILES | {"validation.json"}
_CALL_POLICIES = frozenset({"one_per_operation", "bounded_content_repair_v1"})
_DETERMINISTIC_OPERATIONS = frozenset({"ending", "performance-defaults"})
_EXPECTED_STAGES = (
    {"stage_id": "plan-harmony", "run_path": "stages/01-plan-harmony"},
    {"stage_id": "melody", "run_path": "stages/02-melody"},
    {"stage_id": "accompaniment", "run_path": "stages/03-accompaniment"},
    {"stage_id": "ending", "run_path": "stages/04-ending"},
    {"stage_id": "performance", "run_path": "stages/05-performance"},
)


def package_staged_realization_record(
    run_dir: Path,
    frozen_response: Mapping[str, object],
) -> tuple[bytes | None, ValidationIssue | None]:
    """Validate and package one completed five-stage realization run."""
    root = Path(run_dir)
    try:
        if root.is_symlink() or not root.is_dir():
            return None, _issue(
                IssueCode.STORAGE_ERROR,
                "The staged realization record must be a real directory",
                "/realization_record_dir",
            )
        parent_files, issue = _collect_exact_files(root, _PARENT_FILES, {"stages"})
        if issue is not None:
            return None, issue
        spec = _object_from_bytes(parent_files["run-spec.json"])
        state = _object_from_bytes(parent_files["staged-realization-state.json"])
        document = _object_from_bytes(parent_files["inputs/approved-script.json"])
        if (
            set(spec)
            != {
                "schema_version",
                "approved_script_sha256",
                "profile",
                "model",
                "stages",
                "review_after",
                "call_policy",
            }
            or spec.get("schema_version") != 1
            or spec.get("profile") != "solo_piano_3m_v1"
            or spec.get("stages") != list(_EXPECTED_STAGES)
            or spec.get("call_policy") not in _CALL_POLICIES
            or state.get("status") != "completed"
            or spec.get("approved_script_sha256") != realization_script_sha256(document)
        ):
            return None, _issue(
                IssueCode.LINEAGE_MISMATCH,
                "The staged realization parent record is incomplete or inconsistent",
                "/realization_record/parent",
            )
        raw_workspace = frozen_response.get("workspace")
        if frozen_response.get("schema_version") != 3 or not isinstance(raw_workspace, Mapping):
            return None, _issue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "A version 3 frozen workspace is required",
                "/frozen_response",
            )
        final_hash = raw_workspace.get("workspace_record_sha256")
        if not _is_sha256(final_hash):
            return None, _issue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "The frozen workspace record hash is invalid",
                "/frozen_response/workspace/workspace_record_sha256",
            )

        stage_records: list[dict[str, object]] = []
        expected_source_hash: str | None = None
        stages = spec.get("stages")
        if not isinstance(stages, list):
            raise ValueError("The staged realization specification has no stages")
        for raw_stage in stages:
            if not isinstance(raw_stage, dict):
                raise ValueError("A staged realization stage is not an object")
            stage_id = cast(str, raw_stage["stage_id"])
            run_path = cast(str, raw_stage["run_path"])
            child_dir = root / run_path
            if child_dir.is_symlink() or not child_dir.is_dir():
                return None, _issue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "A realization stage must be a real directory",
                    f"/realization_record/stages/{stage_id}",
                )
            child_spec = _read_object(child_dir / "run-spec.json")
            expected_model = "deterministic-python" if stage_id == "ending" else spec.get("model")
            expected_review = cast(dict[str, list[str]], spec["review_after"])[stage_id]
            operations = child_spec.get("operations")
            model_operation_count = (
                sum(
                    1
                    for operation in operations
                    if isinstance(operation, dict)
                    and operation.get("operation") not in _DETERMINISTIC_OPERATIONS
                )
                if isinstance(operations, list)
                else -1
            )
            call_policy = cast(str, spec["call_policy"])
            expected_repair_limit = (
                1
                if call_policy == "bounded_content_repair_v1" and stage_id == "accompaniment"
                else 0
            )
            child_repair_limit = child_spec.get("content_repair_limit", 0)
            if (
                child_spec.get("model") != expected_model
                or child_spec.get("review_after") != expected_review
                or child_repair_limit != expected_repair_limit
                or child_spec.get("max_calls")
                != model_operation_count * (expected_repair_limit + 1)
            ):
                return None, _issue(
                    IssueCode.LINEAGE_MISMATCH,
                    "A realization stage does not match its parent execution policy",
                    f"/realization_record/stages/{stage_id}/run-spec.json",
                )
            allowed_files = _stage_file_contract(child_spec, child_dir)
            child_files, child_issue = _collect_exact_files(child_dir, allowed_files, set())
            if child_issue is not None:
                return None, child_issue
            evidence_issue = _validate_stage_evidence(stage_id, child_spec, child_files)
            if evidence_issue is not None:
                return None, evidence_issue
            child_state = _object_from_bytes(child_files["realization-state.json"])
            rebuild_issue = _temporary_rebuild_issue(stage_id, child_files, child_state)
            if rebuild_issue is not None:
                return None, rebuild_issue
            workspace = child_state.get("workspace")
            if not isinstance(workspace, dict):
                raise ValueError(f"Stage has no workspace: {stage_id}")
            output_hash = workspace.get("workspace_record_sha256")
            if (
                child_state.get("status") != "completed"
                or child_spec.get("approved_script_sha256") != spec["approved_script_sha256"]
                or child_spec.get("source_workspace_record_sha256") != expected_source_hash
                or not _is_sha256(output_hash)
            ):
                return None, _issue(
                    IssueCode.LINEAGE_MISMATCH,
                    "A realization stage is incomplete or breaks the workspace chain",
                    f"/realization_record/stages/{stage_id}",
                )
            expected_source_hash = cast(str, output_hash)
            stage_records.append(
                {
                    "stage_id": stage_id,
                    "run_path": run_path,
                    "files": _package_files(child_files),
                }
            )
        if expected_source_hash != final_hash or dict(raw_workspace) != workspace:
            return None, _issue(
                IssueCode.LINEAGE_MISMATCH,
                "The final staged workspace does not match the frozen response",
                "/frozen_response/workspace/workspace_record_sha256",
            )
        record = {
            "schema_version": 2,
            "profile": spec.get("profile"),
            "model": spec.get("model"),
            "approved_script_sha256": spec["approved_script_sha256"],
            "final_workspace_record_sha256": final_hash,
            "parent_files": _package_files(parent_files),
            "stages": stage_records,
        }
        return _json_bytes(record), None
    except (
        KeyError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        return None, _issue(
            IssueCode.STORAGE_ERROR,
            str(error),
            "/realization_record_dir",
        )


def verify_staged_realization_record(
    record: bytes,
    document: Mapping[str, object],
    frozen_response: Mapping[str, object],
) -> ValidationIssue | None:
    """Verify a packaged version 2 record without using its original run directory."""
    try:
        value = _object_from_bytes(record)
        expected_fields = {
            "schema_version",
            "profile",
            "model",
            "approved_script_sha256",
            "final_workspace_record_sha256",
            "parent_files",
            "stages",
        }
        if value.get("schema_version") != 2 or set(value) != expected_fields:
            return _issue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "The staged realization record fields are invalid",
                "/realization_record",
            )
        with tempfile.TemporaryDirectory(prefix="scoim-packaged-realization-") as temporary:
            root = Path(temporary) / "run"
            parent_files = _decode_packaged_files(value.get("parent_files"))
            _write_decoded_files(root, parent_files)
            stages = value.get("stages")
            if not isinstance(stages, list):
                raise ValueError("The packaged realization stages are not an array")
            for raw_stage in stages:
                if not isinstance(raw_stage, dict) or set(raw_stage) != {
                    "stage_id",
                    "run_path",
                    "files",
                }:
                    raise ValueError("A packaged realization stage is invalid")
                run_path = cast(str, raw_stage["run_path"])
                _safe_relative_path(run_path)
                child_files = _decode_packaged_files(raw_stage["files"])
                _write_decoded_files(root / run_path, child_files)
            packaged_document = _read_object(root / "inputs" / "approved-script.json")
            if packaged_document != dict(document):
                return _issue(
                    IssueCode.LINEAGE_MISMATCH,
                    "The realization record belongs to a different approved script",
                    "/realization_record/parent_files/inputs~1approved-script.json",
                )
            rebuilt, issue = package_staged_realization_record(root, frozen_response)
            if issue is not None:
                return issue
            assert rebuilt is not None
            if _object_from_bytes(rebuilt) != value:
                return _issue(
                    IssueCode.LINEAGE_MISMATCH,
                    "The realization record does not match its reconstructed run",
                    "/realization_record",
                )
    except (
        KeyError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        return _issue(
            IssueCode.MODEL_OUTPUT_INVALID,
            str(error),
            "/realization_record",
        )
    return None


def package_failed_staged_realization_record(
    run_dir: Path,
    issues: tuple[ValidationIssue, ...],
) -> tuple[bytes | None, ValidationIssue | None]:
    """Package one started staged run that ended before a frozen response existed."""

    root = Path(run_dir).resolve()
    try:
        files = _collect_failure_files(root)
        spec = _object_from_bytes(files["run-spec.json"])
        document = _object_from_bytes(files["inputs/approved-script.json"])
        failed_stage = _failed_stage_from_files(files)
        if (
            spec.get("schema_version") != 1
            or spec.get("approved_script_sha256") != realization_script_sha256(document)
            or spec.get("profile") != "solo_piano_3m_v1"
            or failed_stage is None
            or not issues
        ):
            raise ValueError("The failed staged realization record is incomplete")
        raw_stages = cast(list[dict[str, object]], spec["stages"])
        failed_run_path = next(
            cast(str, stage["run_path"])
            for stage in raw_stages
            if stage.get("stage_id") == failed_stage
        )
        failed_child_dir = root / failed_run_path
        failed_child_state = _read_object(failed_child_dir / "realization-state.json")
        if failed_child_state.get("status") == "failed":
            failed_child_spec = _read_object(failed_child_dir / "run-spec.json")
            allowed_files = _stage_file_contract(
                failed_child_spec, failed_child_dir, allow_failed=True
            )
            child_files, child_issue = _collect_exact_files(failed_child_dir, allowed_files, set())
            if child_issue is not None:
                return None, child_issue
            evidence_issue = _validate_stage_evidence(
                failed_stage,
                failed_child_spec,
                child_files,
                allow_failed=True,
            )
            if evidence_issue is not None:
                return None, evidence_issue
        record = {
            "schema_version": 3,
            "status": "failed",
            "profile": spec["profile"],
            "model": spec["model"],
            "approved_script_sha256": spec["approved_script_sha256"],
            "failed_stage": failed_stage,
            "issues": [
                {"code": issue.code.value, "message": issue.message, "path": issue.path}
                for issue in issues
            ],
            "files": _package_files(files),
        }
        return _json_bytes(record), None
    except (
        KeyError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        return None, _issue(IssueCode.STORAGE_ERROR, str(error), "/realization_record_dir")


def verify_failed_staged_realization_record(
    record: bytes,
    document: Mapping[str, object],
) -> ValidationIssue | None:
    """Verify a packaged failed staged run without using its original directory."""

    try:
        value = _object_from_bytes(record)
        if (
            set(value)
            != {
                "schema_version",
                "status",
                "profile",
                "model",
                "approved_script_sha256",
                "failed_stage",
                "issues",
                "files",
            }
            or value.get("schema_version") != 3
            or value.get("status") != "failed"
        ):
            raise ValueError("The failed staged realization record fields are invalid")
        files = _decode_packaged_files(value["files"])
        packaged_document = _object_from_bytes(files["inputs/approved-script.json"])
        spec = _object_from_bytes(files["run-spec.json"])
        if packaged_document != dict(document):
            raise ValueError("The failed staged realization belongs to a different script")
        if (
            value.get("profile") != spec.get("profile")
            or value.get("model") != spec.get("model")
            or value.get("approved_script_sha256") != realization_script_sha256(packaged_document)
            or spec.get("approved_script_sha256") != value.get("approved_script_sha256")
            or value.get("failed_stage") != _failed_stage_from_files(files)
        ):
            raise ValueError("The failed staged realization lineage does not match")
        raw_issues = value.get("issues")
        if not isinstance(raw_issues, list) or not raw_issues:
            raise ValueError("The failed staged realization has no issue")
        for raw_issue in raw_issues:
            if not isinstance(raw_issue, dict) or set(raw_issue) != {"code", "message", "path"}:
                raise ValueError("A failed staged realization issue is invalid")
        issues = tuple(
            ValidationIssue(
                IssueCode(cast(str, raw_issue["code"])),
                cast(str, raw_issue["message"]),
                cast(str, raw_issue["path"]),
            )
            for raw_issue in raw_issues
        )
        with tempfile.TemporaryDirectory(prefix="scoim-packaged-failed-realization-") as temporary:
            root = Path(temporary) / "run"
            _write_decoded_files(root, files)
            rebuilt, issue = package_failed_staged_realization_record(root, issues)
            if issue is not None:
                return issue
            assert rebuilt is not None
            if _object_from_bytes(rebuilt) != value:
                raise ValueError("The failed staged realization does not match its files")
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        return _issue(IssueCode.MODEL_OUTPUT_INVALID, str(error), "/realization_record")
    return None


def _collect_failure_files(root: Path) -> dict[str, bytes]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("The staged realization record must be a real directory")
    files: dict[str, bytes] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"A staged realization record entry is linked: {relative}")
        if path.is_file() and path.name != ".run.lock":
            files[relative] = path.read_bytes()
    for required in ("run-spec.json", "inputs/approved-script.json"):
        if required not in files:
            raise ValueError(f"A failed staged realization file is missing: {required}")
    return files


def _failed_stage_from_files(files: Mapping[str, bytes]) -> str | None:
    spec = _object_from_bytes(files["run-spec.json"])
    raw_stages = spec.get("stages")
    if not isinstance(raw_stages, list):
        return None
    stage_paths = {
        cast(str, stage["run_path"]): cast(str, stage["stage_id"])
        for stage in raw_stages
        if isinstance(stage, dict)
        and isinstance(stage.get("run_path"), str)
        and isinstance(stage.get("stage_id"), str)
    }
    for run_path, stage_id in stage_paths.items():
        state_path = f"{run_path}/realization-state.json"
        if state_path in files and _object_from_bytes(files[state_path]).get("status") == "failed":
            return stage_id
    failed: list[str] = []
    for relative, content in files.items():
        parts = PurePosixPath(relative).parts
        if len(parts) < 6 or parts[0] != "stages" or parts[-1] != "terminal.json":
            continue
        terminal = _object_from_bytes(content)
        if terminal.get("status") not in {"failed", "interrupted"}:
            continue
        runner_path = "/".join((*parts[:-1], "runner.json"))
        runner = _object_from_bytes(files[runner_path])
        if runner.get("started") is True:
            run_path = "/".join(parts[:2])
            failed.append(stage_paths.get(run_path, parts[1]))
    return sorted(set(failed))[0] if failed else None


def _stage_file_contract(
    spec: Mapping[str, object], root: Path, *, allow_failed: bool = False
) -> frozenset[str]:
    files = {
        "run-spec.json",
        "run-state.json",
        "realization-state.json",
        "inputs/approved-script.json",
        "inputs/initial-workspace.json",
    }
    operations = spec.get("operations")
    review_after = set(cast(list[str], spec.get("review_after", [])))
    if not isinstance(operations, list):
        raise ValueError("A realization stage has no operations")
    repair_limit = spec.get("content_repair_limit", 0)
    if not isinstance(repair_limit, int) or repair_limit < 0:
        raise ValueError("A realization stage has an invalid content repair limit")
    for raw_operation in operations:
        if not isinstance(raw_operation, dict):
            raise ValueError("A realization operation is not an object")
        instance_id = cast(str, raw_operation["instance_id"])
        operation = cast(str, raw_operation["operation"])
        event_root = f"events/{instance_id}"
        if allow_failed and not (root / event_root / "diff.json").is_file():
            break
        files.add(f"{event_root}/diff.json")
        if instance_id in review_after:
            files.add(f"{event_root}/review.json")
        if operation not in _DETERMINISTIC_OPERATIONS:
            files.update(
                {
                    f"schemas/{instance_id}.json",
                    f"{event_root}/request.json",
                    f"{event_root}/response.json",
                }
            )
            attempt_parent = root / "attempts" / instance_id
            attempts = sorted(path for path in attempt_parent.glob("attempt-*") if path.is_dir())
            expected_names = [f"attempt-{index:03d}" for index in range(1, len(attempts) + 1)]
            if (
                not attempts
                or [path.name for path in attempts] != expected_names
                or len(attempts) > repair_limit + 1
            ):
                raise ValueError(f"A realization operation has invalid attempts: {instance_id}")
            attempt_files = _CONTENT_REPAIR_ATTEMPT_FILES if repair_limit else _ATTEMPT_FILES
            files.update(
                f"attempts/{instance_id}/{attempt.name}/{filename}"
                for attempt in attempts
                for filename in attempt_files
            )
        if operation == "ending":
            files.add("diagnostics/ending-material.json")
    return frozenset(files)


def _validate_stage_evidence(
    stage_id: str,
    spec: Mapping[str, object],
    files: Mapping[str, bytes],
    *,
    allow_failed: bool = False,
) -> ValidationIssue | None:
    operations = cast(list[dict[str, object]], spec["operations"])
    if allow_failed:
        operations = [
            operation
            for operation in operations
            if f"events/{operation['instance_id']}/diff.json" in files
        ]
    document = _object_from_bytes(files["inputs/approved-script.json"])
    workspace = workspace_from_dict(_object_from_bytes(files["inputs/initial-workspace.json"]))
    failed_issues: tuple[ValidationIssue, ...] | None = None
    for operation_index, operation in enumerate(operations):
        instance_id = cast(str, operation["instance_id"])
        operation_name = cast(str, operation["operation"])
        targets = tuple(cast(list[str], operation["targets"]))
        prefix = f"/realization_record/stages/{stage_id}/operations/{instance_id}"
        event_root = f"events/{instance_id}"
        response_value: dict[str, object] = {}
        if operation["operation"] in _DETERMINISTIC_OPERATIONS:
            pass
        else:
            event_request = _event_payload(
                files[f"{event_root}/request.json"], instance_id, "model_request"
            )
            schema_bytes = files[f"schemas/{instance_id}.json"]
            schema = _object_from_bytes(schema_bytes)
            repair_limit = spec.get("content_repair_limit", 0)
            if not isinstance(repair_limit, int):
                raise ValueError("A realization stage has an invalid content repair limit")
            attempt_names = sorted(
                {
                    PurePosixPath(relative).parts[2]
                    for relative in files
                    if relative.startswith(f"attempts/{instance_id}/")
                }
            )
            original_prompt: str | None = None
            previous_response: dict[str, object] | None = None
            previous_issues: tuple[ValidationIssue, ...] | None = None
            for attempt_index, attempt_name in enumerate(attempt_names):
                attempt_root = f"attempts/{instance_id}/{attempt_name}"
                attempt_prefix = prefix if attempt_index == 0 else f"{prefix}/{attempt_name}"
                attempt_request = _object_from_bytes(files[f"{attempt_root}/request.json"])
                prompt = files[f"{attempt_root}/prompt.md"]
                prompt_text = prompt.decode("utf-8")
                if attempt_index == 0:
                    original_prompt = prompt_text
                else:
                    assert original_prompt is not None
                    assert previous_response is not None
                    assert previous_issues is not None
                    expected_repair_prompt = build_content_repair_prompt(
                        original_prompt,
                        workspace.workspace_record_sha256,
                        previous_response,
                        previous_issues,
                    )
                    if prompt_text != expected_repair_prompt:
                        return _issue(
                            IssueCode.LINEAGE_MISMATCH,
                            "The content repair prompt cannot be reproduced",
                            f"{attempt_prefix}/prompt.md",
                        )
                if (
                    (attempt_index == 0 and event_request != attempt_request)
                    or attempt_request.get("prompt_sha256") != hashlib.sha256(prompt).hexdigest()
                    or attempt_request.get("response_schema_sha256")
                    != hashlib.sha256(schema_bytes).hexdigest()
                ):
                    return _issue(
                        IssueCode.LINEAGE_MISMATCH,
                        "The saved request does not match its prompt or response schema",
                        f"{attempt_prefix}/prompt.md",
                    )
                raw_response = files[f"{attempt_root}/response.staged.json"]
                candidate_response = _object_from_bytes(raw_response)
                terminal = _object_from_bytes(files[f"{attempt_root}/terminal.json"])
                runner = _object_from_bytes(files[f"{attempt_root}/runner.json"])
                if (
                    terminal.get("status") != "completed"
                    or terminal.get("response_sha256") != hashlib.sha256(raw_response).hexdigest()
                ):
                    return _issue(
                        IssueCode.LINEAGE_MISMATCH,
                        "The terminal response hash does not match the raw response",
                        f"{attempt_prefix}/terminal.json",
                    )
                if runner.get("model") != spec.get("model"):
                    return _issue(
                        IssueCode.LINEAGE_MISMATCH,
                        "The runner model does not match the stage specification",
                        f"{attempt_prefix}/runner.json",
                    )
                if repair_limit == 0 and tuple(
                    Draft202012Validator(schema).iter_errors(candidate_response)
                ):
                    return _issue(
                        IssueCode.MODEL_OUTPUT_INVALID,
                        "The saved model response does not satisfy its response schema",
                        f"{attempt_prefix}/response.staged.json",
                    )
                candidate_diff = _rebuild_checked_diff(
                    operation_name,
                    document,
                    workspace,
                    targets,
                    candidate_response,
                )
                if repair_limit:
                    saved_validation = _object_from_bytes(files[f"{attempt_root}/validation.json"])
                    expected_validation = _validation_record(
                        raw_response, candidate_diff.validation_issues
                    )
                    if saved_validation != expected_validation:
                        return _issue(
                            IssueCode.LINEAGE_MISMATCH,
                            "The saved content validation cannot be reproduced",
                            f"{attempt_prefix}/validation.json",
                        )
                    if (
                        attempt_index < len(attempt_names) - 1
                        and not candidate_diff.validation_issues
                    ):
                        return _issue(
                            IssueCode.LINEAGE_MISMATCH,
                            "A content repair follows an already valid response",
                            f"{attempt_prefix}/validation.json",
                        )
                previous_response = candidate_response
                previous_issues = candidate_diff.validation_issues
                response_value = candidate_response
            event_response = _event_payload(
                files[f"{event_root}/response.json"], instance_id, "model_response"
            )
            if response_value != event_response:
                return _issue(
                    IssueCode.LINEAGE_MISMATCH,
                    "The raw model response does not match the response event",
                    f"{prefix}/response.staged.json",
                )
        checked_diff = _rebuild_checked_diff(
            operation_name,
            document,
            workspace,
            targets,
            response_value,
        )
        saved_diff_value = _event_payload(
            files[f"{event_root}/diff.json"], instance_id, "checked_diff"
        )
        if checked_diff_to_dict(checked_diff) != saved_diff_value:
            return _issue(
                IssueCode.LINEAGE_MISMATCH,
                "The saved checked diff cannot be reproduced from the model response",
                f"{prefix}/diff.json",
            )
        applied_diff = checked_diff
        review_path = f"{event_root}/review.json"
        if review_path in files:
            review = _event_payload(files[review_path], instance_id, "review_decision")
            if review.get("target_diff_sha256") != checked_diff_sha256(checked_diff):
                return _issue(
                    IssueCode.LINEAGE_MISMATCH,
                    "The review does not target the saved checked diff",
                    f"{prefix}/review.json",
                )
            if review.get("decision") == "replace":
                replacement = review.get("replacement_diff")
                if not isinstance(replacement, dict):
                    return _issue(
                        IssueCode.LINEAGE_MISMATCH,
                        "A replacement review has no checked diff",
                        f"{prefix}/review.json",
                    )
                applied_diff = checked_diff_from_dict(replacement)
            elif review.get("decision") != "approve" or review.get("replacement_diff") is not None:
                return _issue(
                    IssueCode.LINEAGE_MISMATCH,
                    "The review decision is inconsistent",
                    f"{prefix}/review.json",
                )
        applied = apply_checked_diff(workspace, applied_diff)
        if applied.workspace is None:
            if allow_failed and operation_index == len(operations) - 1:
                failed_issues = applied.issues
                break
            return _issue(
                IssueCode.LINEAGE_MISMATCH,
                "The saved checked diff cannot be applied to its recorded input",
                f"{prefix}/diff.json",
            )
        workspace = applied.workspace
    saved_state = _object_from_bytes(files["realization-state.json"])
    if saved_state.get("workspace") != workspace_to_dict(workspace):
        return _issue(
            IssueCode.LINEAGE_MISMATCH,
            "The rebuilt stage workspace does not match the saved state",
            f"/realization_record/stages/{stage_id}/realization-state.json",
        )
    if allow_failed:
        expected_issues = [
            {"code": issue.code.value, "message": issue.message, "path": issue.path}
            for issue in failed_issues or ()
        ]
        if saved_state.get("status") != "failed" or saved_state.get("issues") != expected_issues:
            return _issue(
                IssueCode.LINEAGE_MISMATCH,
                "The rebuilt failure does not match the saved stage state",
                f"/realization_record/stages/{stage_id}/realization-state.json",
            )
    return None


def _rebuild_checked_diff(
    operation: str,
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
    response: object,
) -> CheckedWorkspaceDiff:
    if operation == "plan":
        return build_plan_checked_diff(document, workspace, response)
    if operation == "harmony":
        return build_harmony_checked_diff(document, workspace, targets, response)
    if operation == "melody":
        return build_melody_checked_diff(document, workspace, targets, response)
    if operation == "transition-melody":
        return build_transition_melody_checked_diff(document, workspace, targets, response)
    if operation == "accompaniment":
        return build_accompaniment_checked_diff(document, workspace, targets, response)
    if operation == "transition-accompaniment":
        return build_transition_accompaniment_checked_diff(document, workspace, targets, response)
    if operation == "ending":
        return build_ending_checked_diff(document, workspace, targets)
    if operation == "performance-defaults":
        return build_performance_defaults_checked_diff(document, workspace)
    if operation == "performance":
        return build_performance_checked_diff(document, workspace, targets, response)
    raise ValueError(f"Unsupported realization operation: {operation}")


def _temporary_rebuild_issue(
    stage_id: str,
    files: Mapping[str, bytes],
    saved_state: Mapping[str, object],
) -> ValidationIssue | None:
    with tempfile.TemporaryDirectory(prefix="scoim-realization-record-") as temporary:
        run_dir = Path(temporary) / "run"
        for relative, content in files.items():
            path = run_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        try:
            rebuilt = RealizationRunStore(run_dir).rebuild()
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            return _issue(
                IssueCode.LINEAGE_MISMATCH,
                f"The realization stage cannot be rebuilt: {error}",
                f"/realization_record/stages/{stage_id}",
            )
        if rebuilt.status != saved_state.get("status") or workspace_to_dict(
            rebuilt.workspace
        ) != saved_state.get("workspace"):
            return _issue(
                IssueCode.LINEAGE_MISMATCH,
                "The rebuilt realization stage does not match its saved state",
                f"/realization_record/stages/{stage_id}/realization-state.json",
            )
    return None


def _event_payload(content: bytes, instance_id: str, event_type: str) -> dict[str, object]:
    event = _object_from_bytes(content)
    payload = event.get("payload")
    if (
        event.get("schema_version") != 1
        or event.get("event_type") != event_type
        or event.get("operation_instance_id") != instance_id
        or not isinstance(payload, dict)
    ):
        raise ValueError(f"Invalid realization event: {instance_id}/{event_type}")
    return cast(dict[str, object], payload)


def _validation_record(
    raw_response: bytes, issues: tuple[ValidationIssue, ...]
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "response_sha256": hashlib.sha256(raw_response).hexdigest(),
        "status": "invalid" if issues else "valid",
        "issues": [
            {"code": issue.code.value, "message": issue.message, "path": issue.path}
            for issue in issues
        ],
    }


def _collect_exact_files(
    root: Path,
    expected_files: frozenset[str],
    allowed_directories: set[str],
) -> tuple[dict[str, bytes], ValidationIssue | None]:
    actual_files: dict[str, bytes] = {}
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if relative.split("/", 1)[0] in allowed_directories:
            continue
        if path.is_symlink():
            return {}, _issue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "A staged realization record entry must not be a symbolic link",
                f"/realization_record/{relative}",
            )
        if path.is_dir():
            actual_directories.add(relative)
        elif relative != ".run.lock":
            actual_files[relative] = path.read_bytes()
    unknown_files = set(actual_files) - expected_files
    missing_files = expected_files - set(actual_files)
    expected_directories = {
        parent.as_posix()
        for relative in expected_files
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    unknown_directories = actual_directories - expected_directories
    if unknown_files or missing_files or unknown_directories:
        path = sorted(unknown_files or missing_files or unknown_directories)[0]
        return {}, _issue(
            IssueCode.MODEL_OUTPUT_INVALID,
            "The staged realization record file layout is not supported",
            f"/realization_record/{path}",
        )
    return actual_files, None


def _package_files(files: Mapping[str, bytes]) -> dict[str, dict[str, str]]:
    return {
        path: {
            "sha256": hashlib.sha256(content).hexdigest(),
            "content_base64": base64.b64encode(content).decode("ascii"),
        }
        for path, content in sorted(files.items())
    }


def _decode_packaged_files(value: object) -> dict[str, bytes]:
    if not isinstance(value, dict):
        raise ValueError("A packaged file collection is not an object")
    decoded: dict[str, bytes] = {}
    for relative, raw_record in value.items():
        if not isinstance(relative, str):
            raise ValueError("A packaged file path is not a string")
        _safe_relative_path(relative)
        if not isinstance(raw_record, dict) or set(raw_record) != {"sha256", "content_base64"}:
            raise ValueError(f"A packaged file record is invalid: {relative}")
        digest = raw_record["sha256"]
        encoded = raw_record["content_base64"]
        if not _is_sha256(digest) or not isinstance(encoded, str):
            raise ValueError(f"A packaged file record value is invalid: {relative}")
        try:
            content = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise ValueError(f"A packaged file is not valid base64: {relative}") from error
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"A packaged file hash does not match: {relative}")
        decoded[relative] = content
    return decoded


def _write_decoded_files(root: Path, files: Mapping[str, bytes]) -> None:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _safe_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"A packaged file path is unsafe: {value}")


def _read_object(path: Path) -> dict[str, object]:
    return _object_from_bytes(path.read_bytes())


def _object_from_bytes(content: bytes) -> dict[str, object]:
    value = json.loads(content.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("A saved JSON record is not an object")
    return cast(dict[str, object], value)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _issue(code: IssueCode, message: str, path: str) -> ValidationIssue:
    return ValidationIssue(code, message, path)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
