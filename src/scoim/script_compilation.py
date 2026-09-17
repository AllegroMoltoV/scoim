"""Compile an approved human flow into one validated SCoIM script."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import cast

from jsonschema import Draft202012Validator

from llm_musical_composer.run_state import RunStore, sha256_json

from .flow_proposal import _call_runner, _issue_values
from .flow_validation import check_flow, flow_content_sha256
from .projection import ProjectionTarget, prepare_solo_piano_3m_structure
from .proposal import (
    ProposalRun,
    ProposalRunner,
    check_transfer_response,
    normalize_script_content,
    preflight_runner,
)
from .script_validation import check_script_document, script_content_sha256
from .solo_piano_performance import performance_operation_schedule_for_nodes
from .validation import IssueCode, ValidationIssue

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class CompilationRequest:
    """Inputs fixed before the compilation model operation starts."""

    approved_flow: Mapping[str, object]
    composition_id: str
    target_profile: str


@dataclass(frozen=True, slots=True)
class CompilationMetrics:
    """Downstream operation counts frozen from the validated structure."""

    has_transition: int
    has_absolute_performance_group: int
    comparative_waves: int
    performance_operations: int
    normal_calls: int
    maximum_calls: int


@dataclass(frozen=True, slots=True)
class CompilationResult:
    """Terminal result and evidence of one compilation operation."""

    compiled: bool
    document: dict[str, object] | None
    run_dir: Path | None
    response_record: dict[str, object]
    projection_targets: tuple[dict[str, object], ...]
    metrics: CompilationMetrics | None
    issues: tuple[ValidationIssue, ...]


def compile_script(
    request: CompilationRequest,
    runner: ProposalRunner,
    run_dir: str | Path,
) -> CompilationResult:
    """Run one bounded compilation attempt and validate every deterministic boundary."""
    destination = Path(run_dir)
    issues = _preflight_request(request, destination, runner)
    if issues:
        return CompilationResult(False, None, None, {}, (), None, issues)
    flow = dict(request.approved_flow)
    schema = _model_schema()
    store = RunStore(destination, max_calls=2)
    store.initialize(
        {
            "schema_version": 1,
            "operation": "script-compilation",
            "composition_id": request.composition_id,
            "source_flow_sha256": flow_content_sha256(flow),
            "max_calls": 2,
        }
    )
    store.snapshot_json("inputs/approved-flow.json", flow)
    response, run, attempt_dir = _call_runner(
        store, "script-compilation", _prompt(flow), schema, runner
    )
    attempts = 1
    attempt_paths = [str(attempt_dir.relative_to(destination))]
    recorded_responses: list[dict[str, object]] = []
    if response is None:
        failures = run.issues or (
            ValidationIssue(IssueCode.RUNNER_FAILED, "The compilation runner failed", "/runner"),
        )
        if not _repairable_content_failure(run, failures):
            record = {"attempts": 1, "issues": _issue_values(failures)}
            store.snapshot_json("outputs/compilation.json", record)
            store.record_step("publish-final", "failed", {}, record)
            return CompilationResult(False, None, destination, record, (), None, failures)
        initial_result = None
        initial_issues = failures
        current_response: object = run.raw_response.decode("utf-8", errors="replace")
        projection_targets: tuple[dict[str, object], ...] = ()
    else:
        recorded_responses.append(copy.deepcopy(response))
        initial_result = _validate_response(request, response)
        initial_issues = initial_result.issues
        current_response = response
        projection_targets = initial_result.projection_targets

    result = initial_result
    if result is None or not result.compiled:
        ticket = {
            "policy_id": "whole-response-content-repair-v1",
            "issues": _issue_values(initial_issues),
        }
        store.snapshot_json("repairs/script-compilation.json", ticket)
        repair_response, repair_run, repair_dir = _call_runner(
            store,
            "script-compilation",
            _repair_prompt(flow, current_response, initial_issues),
            schema,
            runner,
        )
        attempts = 2
        attempt_paths.append(str(repair_dir.relative_to(destination)))
        if repair_response is None:
            failures = repair_run.issues or (
                ValidationIssue(
                    IssueCode.RUNNER_FAILED, "The compilation repair failed", "/runner"
                ),
            )
            result = CompilationResult(False, None, None, {}, projection_targets, None, failures)
        else:
            recorded_responses.append(copy.deepcopy(repair_response))
            response = repair_response
            result = _validate_response(request, repair_response)
    assert result is not None
    record = {
        "attempts": attempts,
        "responses": recorded_responses,
        "attempt_paths": attempt_paths,
        "initial_issues": _issue_values(initial_issues),
        "issues": _issue_values(result.issues),
    }
    if response is not None:
        record["response_sha256"] = sha256_json(response)
    store.snapshot_json("outputs/compilation.json", record)
    if not result.compiled:
        store.record_step("publish-final", "failed", {}, record)
        return CompilationResult(
            False, None, destination, record, result.projection_targets, None, result.issues
        )
    assert result.document is not None and result.metrics is not None
    store.snapshot_json("outputs/validated-script.json", result.document)
    store.snapshot_json("outputs/projection-targets.json", list(result.projection_targets))
    store.record_step(
        "publish-final",
        "completed",
        {},
        {
            "script_content_sha256": script_content_sha256(result.document),
            "normal_calls": result.metrics.normal_calls,
            "maximum_calls": result.metrics.maximum_calls,
        },
    )
    return CompilationResult(
        True, result.document, destination, record, result.projection_targets, result.metrics, ()
    )


def _validate_response(
    request: CompilationRequest, response: Mapping[str, object]
) -> CompilationResult:
    errors = tuple(_response_validator().iter_errors(response))
    if errors:
        issues = tuple(
            ValidationIssue(
                IssueCode.MODEL_OUTPUT_INVALID,
                error.message,
                "".join(f"/{part!s}" for part in error.absolute_path),
            )
            for error in errors
        )
        return CompilationResult(False, None, None, {}, (), None, issues)
    transfer = cast(dict[str, object], response["script"])
    transfer_issues = check_transfer_response(transfer)
    if transfer_issues:
        return CompilationResult(False, None, None, {}, (), None, transfer_issues)
    flow = request.approved_flow
    source = cast(dict[str, object], flow["approval"])
    document = {
        "document_type": "script",
        "schema_version": "0.2.0",
        "document_id": request.composition_id,
        "revision": 1,
        "status": "validated",
        "source_flow": {
            "document_id": flow["document_id"],
            "revision": flow["revision"],
            "content_sha256": source["content_sha256"],
        },
        "script": normalize_script_content(
            transfer,
            instrumentation=cast(str, flow["instrumentation"]),
            target_duration_seconds=cast(float, flow["target_duration"]),
        ),
    }
    validation = check_script_document(document)
    if not validation.valid:
        return CompilationResult(False, None, None, {}, (), None, validation.issues)
    preparation = prepare_solo_piano_3m_structure(document)
    mapping_issues = _mapping_issues(
        flow, document, cast(list[object], response["scene_section_mappings"])
    )
    issues = preparation.issues + mapping_issues
    targets = _projection_ledger(
        flow, preparation.targets, cast(list[object], response["scene_section_mappings"])
    )
    if issues or not preparation.checks_complete:
        return CompilationResult(False, None, None, {}, targets, None, issues)
    try:
        schedule = performance_operation_schedule_for_nodes(document, preparation.nodes)
    except ValueError as error:
        issue = ValidationIssue(
            IssueCode.UNREPRESENTABLE,
            str(error),
            "/script/performance_setup/performance_directions",
        )
        return CompilationResult(False, None, None, {}, targets, None, (issue,))
    transition = int(bool(cast(dict[str, object], document["script"])["transitions"]))
    absolute = int(bool(schedule.absolute_node_ids))
    waves = len(schedule.comparative_waves)
    performance = absolute + waves
    normal = 4 + 2 * transition + performance
    metrics = CompilationMetrics(transition, absolute, waves, performance, normal, 2 * normal)
    return CompilationResult(True, document, None, {}, targets, metrics, ())


def _mapping_issues(
    flow: Mapping[str, object],
    document: Mapping[str, object],
    raw_mappings: list[object],
) -> tuple[ValidationIssue, ...]:
    mappings = [cast(dict[str, object], item) for item in raw_mappings]
    expected_scenes = [
        cast(str, scene["scene_id"]) for scene in cast(list[dict[str, object]], flow["scenes"])
    ]
    actual_scenes = [cast(str, item["scene_id"]) for item in mappings]
    issues: list[ValidationIssue] = []
    script = cast(dict[str, object], document["script"])
    if script["title"] != flow["title"]:
        issues.append(
            ValidationIssue(
                IssueCode.PROJECTION_TARGET_UNMET,
                "The compiled title must match the approved flow title",
                "/script/title",
            )
        )
    scene_mapping_mismatch = actual_scenes != expected_scenes
    if scene_mapping_mismatch:
        issues.append(
            ValidationIssue(
                IssueCode.PROJECTION_TARGET_UNMET,
                "Every flow scene must be mapped once and in order",
                "/scene_section_mappings",
            )
        )
    sections = cast(dict[str, dict[str, object]], script["sections"])
    ordered_leaves = [
        section_id
        for section_id, section in sorted(
            sections.items(),
            key=lambda item: (
                item[1]["parent_section_id"] is not None,
                cast(int, item[1]["order"]),
            ),
        )
        if not any(other["parent_section_id"] == section_id for other in sections.values())
    ]
    mapped = [
        cast(str, section_id)
        for item in mappings
        for section_id in cast(list[object], item["section_ids"])
    ]
    if mapped != ordered_leaves:
        issues.append(
            ValidationIssue(
                IssueCode.PROJECTION_TARGET_UNMET,
                "Every leaf section must be mapped once and in performance order",
                "/scene_section_mappings",
            )
        )
        return tuple(issues)
    if scene_mapping_mismatch:
        return tuple(issues)
    length_by_scene: dict[str, float] = {}
    for item in mappings:
        length_by_scene[cast(str, item["scene_id"])] = sum(
            float(sections[cast(str, sid)]["relative_length"])
            for sid in cast(list[object], item["section_ids"])
        )
    ranks = {"short": 0, "medium": 1, "long": 2}
    scenes = cast(list[dict[str, object]], flow["scenes"])
    for left in scenes:
        for right in scenes:
            if (
                ranks[cast(str, left["length_class"])] < ranks[cast(str, right["length_class"])]
                and length_by_scene[cast(str, left["scene_id"])]
                >= length_by_scene[cast(str, right["scene_id"])]
            ):
                issues.append(
                    ValidationIssue(
                        IssueCode.PROJECTION_TARGET_UNMET,
                        "Scene length classes must have strictly ordered total section lengths",
                        "/scene_section_mappings",
                    )
                )
                return tuple(issues)
    return tuple(issues)


def _projection_ledger(
    flow: Mapping[str, object],
    script_targets: tuple[ProjectionTarget, ...],
    mappings: list[object],
) -> tuple[dict[str, object], ...]:
    ledger: list[dict[str, object]] = []
    direct = {"title", "instrumentation", "target_duration"}
    for field in ("title", "overall_flow", "instrumentation", "target_duration"):
        ledger.append(
            {
                "source_path": f"/{field}",
                "value": copy.deepcopy(flow[field]),
                "verification": "verified" if field in direct else "unverified",
                "destination": "/script/title" if field == "title" else "/script/brief",
            }
        )
    mapping_by_scene = {
        cast(str, item["scene_id"]): cast(list[object], item["section_ids"])
        for item in cast(list[dict[str, object]], mappings)
    }
    for index, scene in enumerate(cast(list[dict[str, object]], flow["scenes"])):
        scene_id = cast(str, scene["scene_id"])
        destination = {"section_ids": mapping_by_scene.get(scene_id, [])}
        for field in (
            "scene_id",
            "name",
            "length_class",
            "heard_as",
            "relation_to_previous",
            "transition_to_next",
        ):
            ledger.append(
                {
                    "source_path": f"/scenes/{index}/{field}",
                    "value": copy.deepcopy(scene[field]),
                    "verification": "verified"
                    if field in {"scene_id", "length_class"}
                    else "unverified",
                    "destination": destination,
                }
            )
    ledger.extend(
        {
            "source_path": target.source_path,
            "value": copy.deepcopy(target.value),
            "verification": target.verification,
            "destination": list(target.generation_stages),
        }
        for target in script_targets
    )
    return tuple(ledger)


def _preflight_request(
    request: CompilationRequest, destination: Path, runner: ProposalRunner
) -> tuple[ValidationIssue, ...]:
    if destination.exists():
        return (
            ValidationIssue(
                IssueCode.STORAGE_CONFLICT, "The compilation run already exists", "/run_dir"
            ),
        )
    if _ID_PATTERN.fullmatch(request.composition_id) is None:
        return (
            ValidationIssue(
                IssueCode.SCHEMA_INVALID, "Composition ID is invalid", "/composition_id"
            ),
        )
    if request.target_profile != "solo_piano_3m_v1":
        return (
            ValidationIssue(
                IssueCode.UNSUPPORTED_PROFILE,
                "The compilation profile is unsupported",
                "/target_profile",
            ),
        )
    validation = check_flow(request.approved_flow)
    if not validation.valid:
        return validation.issues
    if request.approved_flow["status"] != "approved":
        return (
            ValidationIssue(
                IssueCode.SEMANTIC_INVALID,
                "Only an approved flow can be compiled",
                "/approved_flow/status",
            ),
        )
    return preflight_runner(runner)


def _prompt(flow: Mapping[str, object]) -> str:
    return (
        "承認済みの楽曲台本を、3分の独奏ピアノ用の楽曲設計データへ変換してください。"  # noqa: RUF001
        "場面の内容と順序を保ち、各場面が担当する葉区分をscene_section_mappingsへ示してください。"
        "IDはこの応答内だけで一貫させてください。\n\n"
        f"楽曲台本: {json.dumps(flow, ensure_ascii=False, sort_keys=True)}\n"
    )


def _load_schema(name: str) -> dict[str, object]:
    return cast(
        dict[str, object], json.loads(files("scoim").joinpath("schemas", name).read_text("utf-8"))
    )


def _model_schema() -> dict[str, object]:
    schema = copy.deepcopy(_load_schema("script-compilation-response-1.schema.json"))
    transfer = _load_schema("proposal-response-1.schema.json")
    transfer.pop("$schema", None)
    transfer.pop("$id", None)
    transfer_defs = cast(dict[str, object], transfer.pop("$defs"))
    schema["$defs"] = {"scriptTransfer": transfer, **transfer_defs}
    cast(dict[str, object], schema["properties"])["script"] = {"$ref": "#/$defs/scriptTransfer"}
    return schema


def _response_validator() -> Draft202012Validator:
    return Draft202012Validator(_model_schema())


def _repairable_content_failure(run: ProposalRun, issues: tuple[ValidationIssue, ...]) -> bool:
    return (
        run.started
        and run.terminal_state == "completed"
        and run.raw_response is not None
        and bool(issues)
        and all(issue.code is IssueCode.MODEL_OUTPUT_INVALID for issue in issues)
    )


def _repair_prompt(
    flow: Mapping[str, object],
    response: object,
    issues: tuple[ValidationIssue, ...],
) -> str:
    context = {
        "approved_flow": dict(flow),
        "current_response": response,
        "validation_issues": _issue_values(issues),
        "immutable_constraints": [
            "承認済み楽曲台本の内容と場面順を変えない",
            "応答内のIDと参照を一貫させる",
        ],
    }
    return (
        "構成変換の応答が内容検査に合格しませんでした。承認済み楽曲台本と場面順を"
        "変更せず、検査結果をすべて解消した応答全体を初回と同じSchemaで返してください。"
        "部分差分、説明文、Markdownは返さないでください。\n\n"
        "承認済み楽曲台本、元の応答、検査結果、変更しない条件:\n"
        f"{json.dumps(context, ensure_ascii=False, sort_keys=True)}\n"
    )
