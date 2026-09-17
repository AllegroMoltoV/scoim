"""Model boundary for proposing a human-readable SCoIM flow."""

from __future__ import annotations

import copy
import json
import math
import re
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import cast

from jsonpointer import set_pointer
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from llm_musical_composer.run_state import (
    RunStore,
    atomic_write_bytes,
    atomic_write_json,
    sha256_bytes,
    sha256_json,
    sha256_text,
)

from .flow_validation import check_flow, flow_content_sha256
from .proposal import ProposalRun, ProposalRunner, preflight_runner
from .validation import IssueCode, ValidationIssue

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class FlowProposalRequest:
    """Inputs fixed by the caller before a flow proposal begins."""

    instruction: str
    document_id: str
    instrumentation: str
    target_duration: float


@dataclass(frozen=True, slots=True)
class FlowProposalResult:
    """Terminal result of one flow proposal operation."""

    proposed: bool
    document: dict[str, object] | None
    run_dir: Path | None
    content_sha256: str | None
    issues: tuple[ValidationIssue, ...]


def propose_flow(
    request: FlowProposalRequest,
    runner: ProposalRunner,
    run_dir: str | Path,
) -> FlowProposalResult:
    """Propose one flow and preserve the complete fixed-runner record."""
    destination = Path(run_dir).resolve()
    request_issues = _check_request(request)
    if request_issues:
        return FlowProposalResult(False, None, None, None, request_issues)
    if destination.exists():
        return _failure(IssueCode.STORAGE_CONFLICT, "The flow run already exists", "/run_dir")
    preflight_issues = preflight_runner(runner)
    if preflight_issues:
        return FlowProposalResult(False, None, None, None, preflight_issues)

    schema = _load_schema("flow-proposal-response-1.schema.json")
    prompt = _proposal_prompt(request)
    store = RunStore(destination, max_calls=2)
    spec = {
        "schema_version": 1,
        "operation": "flow-proposal",
        "document_id": request.document_id,
        "max_calls": 2,
    }
    store.initialize(spec)
    store.snapshot_json("inputs/request.json", asdict(request))
    response, run, attempt_dir = _call_runner(
        store,
        "flow-proposal",
        prompt,
        schema,
        runner,
    )
    if response is None:
        issues = run.issues or (
            ValidationIssue(IssueCode.RUNNER_FAILED, "The flow proposal failed", "/runner"),
        )
        store.record_step("publish-final", "failed", {}, {"issues": _issue_values(issues)})
        return FlowProposalResult(False, None, destination, None, issues)

    response_errors = _response_errors(response, schema)
    response_issues = _validation_issues(response_errors)
    if response_issues:
        repair_paths = _repair_paths(response, response_errors)
        if repair_paths is None:
            store.record_step(
                "publish-final",
                "failed",
                {"response_sha256": sha256_json(response)},
                {"issues": _issue_values(response_issues)},
            )
            return FlowProposalResult(False, None, destination, None, response_issues)
        ticket = {
            "policy_id": "validator-guided-local-repair-v1",
            "source_response_sha256": sha256_json(response),
            "allowed_paths": list(repair_paths),
        }
        store.snapshot_json("repairs/flow-proposal.json", ticket)
        repair_schema = _repair_schema(repair_paths)
        repair_prompt = _repair_prompt(response, ticket)
        repair_response, repair_run, attempt_dir = _call_runner(
            store,
            "flow-proposal",
            repair_prompt,
            repair_schema,
            runner,
        )
        if repair_response is None:
            issues = repair_run.issues or (
                ValidationIssue(IssueCode.RUNNER_FAILED, "The flow repair failed", "/runner"),
            )
            store.record_step("publish-final", "failed", {}, {"issues": _issue_values(issues)})
            return FlowProposalResult(False, None, destination, None, issues)
        repair_issues = _validation_issues(_response_errors(repair_response, repair_schema))
        if repair_issues:
            store.record_step(
                "publish-final", "failed", {}, {"issues": _issue_values(repair_issues)}
            )
            return FlowProposalResult(False, None, destination, None, repair_issues)
        repaired = _apply_replacements(response, repair_response, repair_paths)
        if repaired is None:
            issues = (
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "The repair response did not replace exactly the allowed paths",
                    "/replacements",
                ),
            )
            store.record_step("publish-final", "failed", {}, {"issues": _issue_values(issues)})
            return FlowProposalResult(False, None, destination, None, issues)
        response = repaired
        response_issues = _validation_issues(_response_errors(response, schema))
        if response_issues:
            store.record_step(
                "publish-final",
                "failed",
                {"response_sha256": sha256_json(response)},
                {"issues": _issue_values(response_issues)},
            )
            return FlowProposalResult(False, None, destination, None, response_issues)

    document = _normalize(request, response)
    validation = check_flow(document)
    if not validation.valid:
        store.record_step(
            "publish-final",
            "failed",
            {"response_sha256": sha256_json(response)},
            {"issues": _issue_values(validation.issues)},
        )
        return FlowProposalResult(False, None, destination, None, validation.issues)
    content_hash = flow_content_sha256(document)
    output_path = store.snapshot_json("outputs/flow-draft.json", document)
    store.record_step(
        "publish-final",
        "completed",
        {"response_sha256": sha256_json(response)},
        {
            "document_path": str(output_path.relative_to(destination)),
            "content_sha256": content_hash,
            "attempt": str(attempt_dir.relative_to(destination)),
        },
    )
    return FlowProposalResult(True, document, destination, content_hash, ())


def _check_request(request: FlowProposalRequest) -> tuple[ValidationIssue, ...]:
    if not request.instruction.strip():
        return (
            ValidationIssue(
                IssueCode.SCHEMA_INVALID, "Instruction must not be empty", "/instruction"
            ),
        )
    if _ID_PATTERN.fullmatch(request.document_id) is None:
        return (
            ValidationIssue(IssueCode.SCHEMA_INVALID, "Document ID is invalid", "/document_id"),
        )
    if _ID_PATTERN.fullmatch(request.instrumentation) is None:
        return (
            ValidationIssue(
                IssueCode.SCHEMA_INVALID, "Instrumentation is invalid", "/instrumentation"
            ),
        )
    duration = request.target_duration
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        return (
            ValidationIssue(
                IssueCode.SCHEMA_INVALID, "Duration must be positive", "/target_duration"
            ),
        )
    return ()


def _proposal_prompt(request: FlowProposalRequest) -> str:
    return (
        "短い依頼から、人が読んで曲の流れを判断できる楽曲台本を提案してください。"
        "専門用語、内部ID、音符、MIDI値は書かず、平易な言葉を使ってください。"
        "場面は演奏順に並べ、同じ音楽が戻る場合も、前と何を変えるかを書いてください。"
        "length_classはshort、medium、longのいずれかを使ってください。\n\n"
        f"依頼: {request.instruction}\n"
        f"編成: {request.instrumentation}\n"
        f"希望演奏時間: {request.target_duration:g}秒\n"
    )


def _normalize(
    request: FlowProposalRequest,
    response: Mapping[str, object],
) -> dict[str, object]:
    scenes = []
    for index, raw_scene in enumerate(cast(list[object], response["scenes"]), 1):
        scene = dict(cast(dict[str, object], raw_scene))
        scene["scene_id"] = f"scene-{index:03d}"
        scenes.append({"scene_id": scene.pop("scene_id"), **scene})
    return {
        "document_type": "flow",
        "schema_version": "0.1.0",
        "document_id": request.document_id,
        "revision": 1,
        "status": "draft",
        "approval": None,
        "title": response["title"],
        "overall_flow": response["overall_flow"],
        "instrumentation": request.instrumentation,
        "target_duration": request.target_duration,
        "scenes": scenes,
    }


def _load_schema(name: str) -> dict[str, object]:
    return cast(
        dict[str, object],
        json.loads(files("scoim").joinpath("schemas", name).read_text("utf-8")),
    )


def _call_runner(
    store: RunStore,
    step_id: str,
    prompt: str,
    schema: Mapping[str, object],
    runner: ProposalRunner,
) -> tuple[dict[str, object] | None, ProposalRun, Path]:
    with tempfile.TemporaryDirectory(prefix="scoim-flow-schema-") as temporary_name:
        schema_path = Path(temporary_name) / "response.schema.json"
        schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
        request = {
            "prompt_sha256": sha256_text(prompt),
            "response_schema_sha256": sha256_bytes(schema_path.read_bytes()),
        }
        attempt_dir = store.reserve_attempt(step_id, request, prompt)
        run = runner.run(prompt, schema_path)
    atomic_write_bytes(attempt_dir / "stdout.jsonl", run.events or b"")
    atomic_write_bytes(attempt_dir / "stderr.log", run.stderr or b"")
    atomic_write_json(
        attempt_dir / "runner.json",
        {
            "provider": run.provider,
            "model": run.model,
            "model_settings": dict(run.model_settings),
            "started": run.started,
            "terminal_state": run.terminal_state,
            "issues": _issue_values(run.issues),
        },
    )
    if run.raw_response is not None:
        atomic_write_bytes(attempt_dir / "response.staged.json", run.raw_response)
    if run.terminal_state != "completed" or run.raw_response is None or run.issues:
        terminal_state = (
            "interrupted" if run.terminal_state in {"timeout", "interrupted"} else "failed"
        )
        store.finalize_attempt(
            attempt_dir, terminal_state, returncode=None, detail=run.terminal_state
        )
        return None, run, attempt_dir
    try:
        decoded = json.loads(run.raw_response.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        store.finalize_attempt(
            attempt_dir, "completed", returncode=0, response_sha256=sha256_bytes(run.raw_response)
        )
        invalid = ProposalRun(
            run.provider,
            run.model,
            run.model_settings,
            run.started,
            run.terminal_state,
            run.raw_response,
            run.events,
            run.stderr,
            (ValidationIssue(IssueCode.MODEL_OUTPUT_INVALID, str(error), ""),),
        )
        return None, invalid, attempt_dir
    store.finalize_attempt(
        attempt_dir, "completed", returncode=0, response_sha256=sha256_bytes(run.raw_response)
    )
    if not isinstance(decoded, dict):
        invalid = ProposalRun(
            run.provider,
            run.model,
            run.model_settings,
            run.started,
            run.terminal_state,
            run.raw_response,
            run.events,
            run.stderr,
            (ValidationIssue(IssueCode.MODEL_OUTPUT_INVALID, "Response must be an object", ""),),
        )
        return None, invalid, attempt_dir
    return cast(dict[str, object], decoded), run, attempt_dir


def _response_errors(
    response: Mapping[str, object], schema: Mapping[str, object]
) -> tuple[ValidationError, ...]:
    return tuple(
        sorted(
            Draft202012Validator(schema).iter_errors(response),
            key=lambda item: (tuple(str(part) for part in item.absolute_path), item.message),
        )
    )


def _validation_issues(errors: tuple[ValidationError, ...]) -> tuple[ValidationIssue, ...]:
    return tuple(
        ValidationIssue(
            IssueCode.MODEL_OUTPUT_INVALID,
            error.message,
            "".join(f"/{_escape_pointer_token(part)}" for part in error.absolute_path),
        )
        for error in errors
    )


def _repair_paths(
    response: Mapping[str, object], errors: tuple[ValidationError, ...]
) -> tuple[str, ...] | None:
    repairable_validators = frozenset(
        {"enum", "type", "minLength", "maxLength", "minimum", "maximum", "pattern"}
    )
    paths: list[str] = []
    for error in errors:
        if error.validator not in repairable_validators or not error.absolute_path:
            return None
        path = "".join(f"/{_escape_pointer_token(part)}" for part in error.absolute_path)
        value = error.instance
        if isinstance(value, (dict, list)):
            return None
        paths.append(path)
    unique_paths = tuple(dict.fromkeys(paths))
    return unique_paths if unique_paths else None


def _repair_schema(paths: tuple[str, ...]) -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["replacements"],
        "properties": {
            "replacements": {
                "type": "array",
                "minItems": len(paths),
                "maxItems": len(paths),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["path", "value"],
                    "properties": {
                        "path": {"type": "string", "enum": list(paths)},
                        "value": {"type": ["string", "number", "boolean", "null"]},
                    },
                },
            }
        },
    }


def _repair_prompt(response: Mapping[str, object], ticket: Mapping[str, object]) -> str:
    return (
        "検査で不合格になった楽曲台本の応答を局所修正してください。"
        "allowed_pathsにある値だけを一度ずつ置き換え、ほかの内容は返さないでください。\n\n"
        f"元の応答: {json.dumps(response, ensure_ascii=False, sort_keys=True)}\n"
        f"修正票: {json.dumps(ticket, ensure_ascii=False, sort_keys=True)}\n"
    )


def _apply_replacements(
    response: Mapping[str, object],
    repair_response: Mapping[str, object],
    allowed_paths: tuple[str, ...],
) -> dict[str, object] | None:
    raw_replacements = repair_response.get("replacements")
    if not isinstance(raw_replacements, list):
        return None
    replacement_paths = [item.get("path") for item in raw_replacements if isinstance(item, dict)]
    if len(replacement_paths) != len(allowed_paths) or set(replacement_paths) != set(allowed_paths):
        return None
    repaired = cast(dict[str, object], copy.deepcopy(response))
    for raw_replacement in raw_replacements:
        replacement = cast(dict[str, object], raw_replacement)
        set_pointer(repaired, cast(str, replacement["path"]), replacement["value"], inplace=True)
    return repaired


def _escape_pointer_token(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _issue_values(issues: tuple[ValidationIssue, ...]) -> list[dict[str, str]]:
    return [
        {"code": issue.code.value, "message": issue.message, "path": issue.path} for issue in issues
    ]


def _failure(code: IssueCode, message: str, path: str) -> FlowProposalResult:
    return FlowProposalResult(False, None, None, None, (ValidationIssue(code, message, path),))
