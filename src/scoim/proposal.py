"""Natural-language proposal boundary for SCoIM music scripts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Protocol, cast

from jsonschema import Draft202012Validator

from .projection import check_solo_piano_3m_structure
from .validation import IssueCode, ValidationIssue, check, content_sha256

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class ProposalRequest:
    """Music intent fixed by the caller before one proposal run."""

    instruction: str
    document_id: str
    instrumentation: str
    target_duration_seconds: float
    target_profile: str | None = None


@dataclass(frozen=True, slots=True)
class ProposalRun:
    """Runner-reported conditions and outcome for one proposal call."""

    provider: str
    model: str
    model_settings: Mapping[str, object]
    started: bool
    terminal_state: str
    raw_response: bytes | None
    events: bytes | None
    stderr: bytes | None
    issues: tuple[ValidationIssue, ...]


class ProposalRunner(Protocol):
    """One model call that returns a proposal-transfer response."""

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun: ...


class PreflightProposalRunner(Protocol):
    """Optional capability checked immediately before reserving a model attempt."""

    def preflight(self) -> tuple[ValidationIssue, ...]: ...


class RunnerPreflightError(RuntimeError):
    """A runner could not start and no model attempt was reserved."""

    def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:
        super().__init__(issues[0].message if issues else "Runner preflight failed")
        self.issues = issues


def preflight_runner(runner: ProposalRunner) -> tuple[ValidationIssue, ...]:
    """Run the optional no-model-call preflight capability."""

    preflight = getattr(runner, "preflight", None)
    if preflight is None:
        return ()
    issues = preflight()
    if not isinstance(issues, tuple) or any(
        not isinstance(issue, ValidationIssue) for issue in issues
    ):
        raise TypeError("Runner preflight must return validation issues")
    return issues


@dataclass(frozen=True, slots=True)
class ProposalResult:
    """Result of proposing one draft music script."""

    proposed: bool
    document: dict[str, object] | None
    proposal_path: Path | None
    draft_path: Path | None
    content_sha256: str | None
    issues: tuple[ValidationIssue, ...]


def _prompt(request: ProposalRequest) -> str:
    profile_guidance = ""
    if request.target_profile == "solo_piano_3m_v1":
        profile_guidance = (
            "\nsolo_piano_3m_v1で実現できるように、区分roleはwhole、opening、statement、"
            "variation、contrast、transition、climax、return、releaseだけを使ってください。"
            "素材kindはtheme、contrast、transition、endingだけを使い、再現や変奏は"
            "variations関係で表してください。各葉区分には配置を一つだけ置いてください。"
            "最後のrelease葉には唯一のending素材を置き、その前に通常素材を置いてください。"
            "同じ素材を再利用する葉はrelative_lengthを同じにしてください。\n"
            "variationsのsourceとtargetは同じtypeにし、区分または配置が属する区分を"
            "同じ階層の深さに置き、両側で同じ素材を使ってください。一つの変奏先には"
            "一つの変奏元だけを指定してください。異なる素材間やtransition、endingへの"
            "発展関係はvariationsにせず、descriptionで説明してください。return区分には"
            "同じ深さにある先行区分からのvariationsを指定してください。すべての素材を"
            "一つ以上の葉区分へ配置してください。transition素材はtransition葉のconnector"
            "配置だけに使ってください。requirementsへ使う演奏指示は、target_typeと"
            "relative_to_typeをどちらもsectionにし、区分同士を比較してください。\n"
        )
    return (
        "SCoIMの楽曲台本の下書きを一つ提案してください。音符は書かず、曲の階層構成、"
        "再利用する素材、その配置、変奏、必要なつなぎ、対象を明示した演奏指示を設計してください。"
        "同じ素材を再現するときは完全な複製にせず、保つ点と変える点を関係として示してください。"
        "区分は演奏順に並べ、relative_lengthは子を持たない区分の構成比だけに使ってください。"
        "つなぎを作る場合は、つなぎ素材の配置だけを持つ専用の葉区分を作り、"
        "区分の演奏順を前の配置、つなぎ配置、次の配置の順にしてください。"
        "対象と比較対象と方向が明確な演奏指示だけ、requirementsへ型付き比較として追加してください。"
        "featureはonset_alignmentまたはloudness、relationはmoreまたはlessだけを使い、"
        "曖昧な指示を推測で変換せず、該当しない場合は空配列にしてください。"
        "応答は指定された出力Schemaだけに従ってください。\n\n"
        f"利用者の指示: {request.instruction}\n"
        f"編成: {request.instrumentation}\n"
        f"希望演奏時間: {request.target_duration_seconds:g}秒\n"
        f"{profile_guidance}"
    )


def _response_schema(request: ProposalRequest, schema: dict[str, object]) -> dict[str, object]:
    if request.target_profile is None:
        return schema
    if request.target_profile != "solo_piano_3m_v1":
        raise ValueError(f"Unsupported proposal profile: {request.target_profile}")
    definitions = cast(dict[str, object], schema["$defs"])
    section = cast(dict[str, object], definitions["section"])
    section_properties = cast(dict[str, object], section["properties"])
    section_properties["role"] = {
        "type": "string",
        "enum": [
            "climax",
            "contrast",
            "opening",
            "release",
            "return",
            "statement",
            "transition",
            "variation",
            "whole",
        ],
    }
    material = cast(dict[str, object], definitions["material"])
    material_properties = cast(dict[str, object], material["properties"])
    material_properties["kind"] = {
        "type": "string",
        "enum": ["contrast", "ending", "theme", "transition"],
    }
    return schema


def _collection(items: object, *, omit_null: frozenset[str] = frozenset()) -> dict[str, object]:
    collection: dict[str, object] = {}
    for raw_item in cast(list[object], items):
        item = cast(dict[str, object], raw_item)
        item_id = cast(str, item["id"])
        value = {
            key: member
            for key, member in item.items()
            if key != "id" and not (key in omit_null and member is None)
        }
        collection[item_id] = value
    return collection


def _sections(items: object) -> dict[str, object]:
    section_items = cast(list[object], items)
    branch_ids: set[str] = set()
    for raw_item in section_items:
        parent_id = cast(dict[str, object], raw_item)["parent_section_id"]
        if parent_id is not None:
            branch_ids.add(cast(str, parent_id))
    collection: dict[str, object] = {}
    next_order: dict[str | None, int] = {}
    for raw_item in section_items:
        item = cast(dict[str, object], raw_item)
        item_id = cast(str, item["id"])
        parent_id = cast(str | None, item["parent_section_id"])
        order = next_order.get(parent_id, 0)
        value = {
            key: member
            for key, member in item.items()
            if key != "id"
            and not (key == "relative_length" and (member is None or item_id in branch_ids))
        }
        value["order"] = order
        collection[item_id] = value
        next_order[parent_id] = order + 1
    return collection


def _directions(items: object) -> dict[str, object]:
    result: dict[str, object] = {}
    for raw_item in cast(list[object], items):
        item = cast(dict[str, object], raw_item)
        direction: dict[str, object] = {
            "target": {"type": item["target_type"], "id": item["target_id"]},
            "description": item["description"],
        }
        if item["relative_to_type"] is not None and item["relative_to_id"] is not None:
            direction["relative_to"] = {
                "type": item["relative_to_type"],
                "id": item["relative_to_id"],
            }
        result[cast(str, item["id"])] = direction
    return result


def normalize_script_content(
    response: Mapping[str, object],
    *,
    instrumentation: str,
    target_duration_seconds: float,
) -> dict[str, object]:
    """Normalize a model transfer response into the shared inner script shape."""
    return {
        "title": response["title"],
        "brief": response["brief"],
        "performance_setup": {
            "instrumentation": instrumentation,
            "target_duration_seconds": target_duration_seconds,
            "performance_directions": _directions(response["performance_directions"]),
        },
        "root_section_id": response["root_section_id"],
        "sections": _sections(response["sections"]),
        "materials": _collection(response["materials"]),
        "placements": _collection(response["placements"]),
        "variations": _collection(response["variations"]),
        "requirements": _collection(response["requirements"]),
        "transitions": _collection(response["transitions"]),
    }


def _normalize(request: ProposalRequest, response: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": "0.1.0",
        "document_id": request.document_id,
        "revision": 1,
        "status": "draft",
        "approval": None,
        "script": normalize_script_content(
            response,
            instrumentation=request.instrumentation,
            target_duration_seconds=request.target_duration_seconds,
        ),
    }


def _duplicate_id_issue(response: Mapping[str, object]) -> ValidationIssue | None:
    for collection_name in (
        "sections",
        "materials",
        "placements",
        "variations",
        "transitions",
        "performance_directions",
        "requirements",
    ):
        seen: set[str] = set()
        for index, raw_item in enumerate(cast(list[object], response[collection_name])):
            item = cast(dict[str, object], raw_item)
            item_id = cast(str, item["id"])
            if item_id in seen:
                return ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    f"Duplicate ID in {collection_name}: {item_id}",
                    f"/{collection_name}/{index}/id",
                )
            seen.add(item_id)
    return None


def _relative_reference_issue(response: Mapping[str, object]) -> ValidationIssue | None:
    directions = cast(list[object], response["performance_directions"])
    for index, raw_direction in enumerate(directions):
        direction = cast(dict[str, object], raw_direction)
        relative_type_is_null = direction["relative_to_type"] is None
        relative_id_is_null = direction["relative_to_id"] is None
        if relative_type_is_null != relative_id_is_null:
            field = "relative_to_type" if relative_type_is_null else "relative_to_id"
            return ValidationIssue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "Relative direction type and ID must both be null or both be present",
                f"/performance_directions/{index}/{field}",
            )
    return None


def check_transfer_response(response: Mapping[str, object]) -> tuple[ValidationIssue, ...]:
    """Check transfer invariants that JSON Schema cannot express."""
    issues = tuple(
        issue
        for issue in (_duplicate_id_issue(response), _relative_reference_issue(response))
        if issue is not None
    )
    return issues


def _failure(code: IssueCode, message: str, path: str = "") -> ProposalResult:
    return ProposalResult(
        proposed=False,
        document=None,
        proposal_path=None,
        draft_path=None,
        content_sha256=None,
        issues=(ValidationIssue(code, message, path),),
    )


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _issue_value(issue: ValidationIssue) -> dict[str, str]:
    return {"code": issue.code.value, "message": issue.message, "path": issue.path}


def _save_record(
    output_dir: Path,
    *,
    request: ProposalRequest,
    prompt: str,
    response_schema: bytes,
    run: ProposalRun,
    document: dict[str, object] | None,
    document_hash: str | None,
    issues: tuple[ValidationIssue, ...],
) -> tuple[Path, Path | None]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        files_to_write = {
            "request.json": _json_bytes(asdict(request)),
            "prompt.md": prompt.encode("utf-8"),
            "response-schema.json": response_schema,
            "validation.json": _json_bytes(
                {"valid": not issues, "issues": [_issue_value(issue) for issue in issues]}
            ),
        }
        if run.raw_response is not None:
            files_to_write["raw-response.txt"] = run.raw_response
        if document is not None:
            files_to_write["normalized-draft.json"] = _json_bytes(document)
        if run.events is not None:
            files_to_write["runner-events.jsonl"] = run.events
        if run.stderr is not None:
            files_to_write["stderr.log"] = run.stderr
        for relative_path, content in files_to_write.items():
            (temporary / relative_path).write_bytes(content)
        manifest = {
            "schema_version": 1,
            "status": "completed" if document is not None else "failed",
            "runner": {
                "provider": run.provider,
                "model": run.model,
                "model_settings": dict(run.model_settings),
                "terminal_state": run.terminal_state,
            },
            "draft": (
                {
                    "document_id": request.document_id,
                    "revision": 1,
                    "content_sha256": document_hash,
                }
                if document is not None
                else None
            ),
            "files": {
                relative_path: hashlib.sha256(content).hexdigest()
                for relative_path, content in sorted(files_to_write.items())
            },
        }
        (temporary / "manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    draft_path = output_dir / "normalized-draft.json" if document is not None else None
    return output_dir, draft_path


def _save_failed_result(
    output_dir: Path,
    *,
    request: ProposalRequest,
    prompt: str,
    response_schema: bytes,
    run: ProposalRun,
    issues: tuple[ValidationIssue, ...],
) -> ProposalResult:
    try:
        proposal_path, _ = _save_record(
            output_dir,
            request=request,
            prompt=prompt,
            response_schema=response_schema,
            run=run,
            document=None,
            document_hash=None,
            issues=issues,
        )
    except OSError as error:
        return _failure(IssueCode.STORAGE_ERROR, str(error), "/output_dir")
    return ProposalResult(False, None, proposal_path, None, None, issues)


def _recorded_invalid(
    output_dir: Path,
    *,
    request: ProposalRequest,
    prompt: str,
    response_schema: bytes,
    run: ProposalRun,
    message: str,
    path: str = "",
) -> ProposalResult:
    return _save_failed_result(
        output_dir,
        request=request,
        prompt=prompt,
        response_schema=response_schema,
        run=run,
        issues=(ValidationIssue(IssueCode.MODEL_OUTPUT_INVALID, message, path),),
    )


def propose_script(
    request: ProposalRequest,
    runner: ProposalRunner,
    output_dir: str | Path,
) -> ProposalResult:
    """Ask one runner for a proposal and normalize it to a valid draft."""
    output_path = Path(output_dir)
    if not request.instruction.strip():
        return _failure(
            IssueCode.SCHEMA_INVALID,
            "The proposal instruction must not be empty",
            "/instruction",
        )
    if _ID_PATTERN.fullmatch(request.document_id) is None:
        return _failure(
            IssueCode.SCHEMA_INVALID,
            "The proposal document ID is invalid",
            "/document_id",
        )
    if _ID_PATTERN.fullmatch(request.instrumentation) is None:
        return _failure(
            IssueCode.SCHEMA_INVALID,
            "The proposal instrumentation is invalid",
            "/instrumentation",
        )
    if (
        isinstance(request.target_duration_seconds, bool)
        or not isinstance(request.target_duration_seconds, (int, float))
        or not math.isfinite(request.target_duration_seconds)
        or request.target_duration_seconds <= 0
    ):
        return _failure(
            IssueCode.SCHEMA_INVALID,
            "The proposal duration must be a positive number",
            "/target_duration_seconds",
        )
    if request.target_profile not in {None, "solo_piano_3m_v1"}:
        return _failure(
            IssueCode.UNSUPPORTED_PROFILE,
            "The proposal target profile is not supported",
            "/target_profile",
        )
    if request.target_profile == "solo_piano_3m_v1" and (
        request.instrumentation != "solo_piano" or request.target_duration_seconds != 180
    ):
        return _failure(
            IssueCode.UNSUPPORTED_PROFILE,
            "solo_piano_3m_v1 requires solo_piano and 180 seconds",
            "/target_profile",
        )
    if output_path.exists():
        return _failure(
            IssueCode.STORAGE_CONFLICT,
            "The proposal output already exists",
            "/output_dir",
        )
    prompt = _prompt(request)
    schema_resource = files("scoim").joinpath("schemas", "proposal-response-1.schema.json")
    with as_file(schema_resource) as schema_path:
        schema = _response_schema(request, json.loads(schema_path.read_bytes()))
    schema_bytes = _json_bytes(schema)
    with tempfile.TemporaryDirectory(prefix="scoim-proposal-schema-") as temporary_dir:
        schema_path = Path(temporary_dir) / "proposal-response-1.schema.json"
        schema_path.write_bytes(schema_bytes)
        run = runner.run(prompt, schema_path)
    if run.terminal_state != "completed" or run.issues:
        if not run.started:
            return ProposalResult(False, None, None, None, None, run.issues)
        return _save_failed_result(
            output_path,
            request=request,
            prompt=prompt,
            response_schema=schema_bytes,
            run=run,
            issues=run.issues
            or (ValidationIssue(IssueCode.RUNNER_FAILED, "The runner failed", "/runner"),),
        )
    if run.raw_response is None:
        return _recorded_invalid(
            output_path,
            request=request,
            prompt=prompt,
            response_schema=schema_bytes,
            run=run,
            message="The proposal runner returned no response",
        )
    try:
        response = json.loads(run.raw_response.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return _recorded_invalid(
            output_path,
            request=request,
            prompt=prompt,
            response_schema=schema_bytes,
            run=run,
            message=f"The proposal response is not valid UTF-8 JSON: {error}",
        )
    errors = sorted(
        Draft202012Validator(schema).iter_errors(response),
        key=lambda item: list(item.path),
    )
    if errors:
        error = errors[0]
        path = "".join(f"/{part}" for part in error.absolute_path)
        return _recorded_invalid(
            output_path,
            request=request,
            prompt=prompt,
            response_schema=schema_bytes,
            run=run,
            message=error.message,
            path=path,
        )
    duplicate_issue = _duplicate_id_issue(cast(dict[str, object], response))
    if duplicate_issue is not None:
        return _recorded_invalid(
            output_path,
            request=request,
            prompt=prompt,
            response_schema=schema_bytes,
            run=run,
            message=duplicate_issue.message,
            path=duplicate_issue.path,
        )
    relative_issue = _relative_reference_issue(cast(dict[str, object], response))
    if relative_issue is not None:
        return _recorded_invalid(
            output_path,
            request=request,
            prompt=prompt,
            response_schema=schema_bytes,
            run=run,
            message=relative_issue.message,
            path=relative_issue.path,
        )
    document = _normalize(request, cast(dict[str, object], response))
    validation = check(document)
    if not validation.valid:
        issue = validation.issues[0]
        return _recorded_invalid(
            output_path,
            request=request,
            prompt=prompt,
            response_schema=schema_bytes,
            run=run,
            message=issue.message,
            path=issue.path,
        )
    if request.target_profile == "solo_piano_3m_v1":
        profile_validation = check_solo_piano_3m_structure(document)
        if not profile_validation.valid:
            issue = profile_validation.issues[0]
            return _recorded_invalid(
                output_path,
                request=request,
                prompt=prompt,
                response_schema=schema_bytes,
                run=run,
                message=issue.message,
                path=issue.path,
            )
    document_hash = content_sha256(document)
    try:
        proposal_path, draft_path = _save_record(
            output_path,
            request=request,
            prompt=prompt,
            response_schema=schema_bytes,
            run=run,
            document=document,
            document_hash=document_hash,
            issues=(),
        )
    except OSError as error:
        return _failure(IssueCode.STORAGE_ERROR, str(error), "/output_dir")
    return ProposalResult(
        proposed=True,
        document=document,
        proposal_path=proposal_path,
        draft_path=draft_path,
        content_sha256=document_hash,
        issues=(),
    )
