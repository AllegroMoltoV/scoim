"""Create one SCoIM generation trial from a structured model response."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from llm_musical_composer.run_state import InterruptedAttemptError, StateConflictError

from .outcome import ArtifactDisposition, ExecutionOutcome, TerminalState
from .projection import check_solo_piano_3m_structure
from .proposal import ProposalRun, ProposalRunner, RunnerPreflightError, preflight_runner
from .realization_record import (
    package_failed_staged_realization_record,
    package_staged_realization_record,
)
from .realization_script import check_realization_script, realization_script_sha256
from .realization_workspace import workspace_to_dict
from .requirements import resolve_requirements
from .staged_realization import advance_staged_realization, initialize_staged_realization
from .trial_bundle import (
    TrialBundleResult,
    create_failed_model_trial,
    create_trial_bundle,
    package_proposal_record,
)
from .typed_realization import (
    TypedRealizationError,
    build_typed_response_schema,
    material_lengths_for_plan,
    normalize_typed_model_response,
    preflight_typed_realization,
    validate_typed_model_response,
)
from .validation import IssueCode, ValidationIssue


def _prompt(document: Mapping[str, object], material_lengths: Mapping[str, int]) -> str:
    serialized = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
    script = cast(dict[str, object], document["script"])
    typed_requirements = json.dumps(
        [requirement.value() for requirement in resolve_requirements(script)],
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    instructions = (
        "SCoIMの承認済み楽曲台本から、solo_piano_3m_v1の音楽上の値を決めてください。",
        "",
        "台本の区分ID、親子関係、順序、素材ID、配置、相対長は変更しないでください。",
        "応答は指定されたJSON Schemaだけに従い、補足文を書かないでください。",
        "JSON Schemaが固定したキー以外のID、参照、検査式、実装用の記法は書かないでください。",
        "",
        "楽譜値について:",
        "- 1拍を12 unitsとして、各素材の範囲内に音符、和声、演奏記号を置いてください。",
        "- 各素材の固定長は次のとおりです: "
        + json.dumps(material_lengths, ensure_ascii=False, sort_keys=True),
        "- notesは実際に鳴らす音です。upperは主に右手、lowerは主に左手です。",
        "- harmoniesはその区間を支える和声です。低音では三度を密集させず、"
        "根音、五度、オクターブを優先してください。",
        "- 旋律の非和声音は、経過音、刺繍音、掛留など前後関係で理由が分かる形にしてください。",
        "",
        "演奏値について:",
        "- node_performancesは区分ごとの表情差だけを指定します。",
        "- 同じ素材の再登場は完全コピーにせず、台本の変奏指示に沿って演奏差を付けてください。",
        "- timing_profileとtiming_amountは組で使い、不要な区分は両方nullにしてください。",
        "- ペダルの踏み替えはフレーズや和声の境界に合わせ、旋律を唐突に切らないでください。",
        "",
        "plan_choiceについて:",
        "- harmonic_focus_by_sectionは区分ごとの中心音です。不要ならnullにしてください。",
        "- contrasts_with_by_sectionの整数は、Schemaに列挙された同じ深さの先行区分を"
        "0始まりで選ぶ番号です。比較先が不要ならnullにしてください。",
        "",
        "型付き検査要件について:",
        "onset_alignmentは同じ楽譜位置にある複数声部の"
        "実発音時刻がそろう度合い、loudnessは発音の強さです。moreは比較対象より"
        "強め、lessは弱める方向です。楽譜値と演奏値で満たしてください。",
        typed_requirements,
        "",
        "承認済み楽曲台本:",
        serialized,
    )
    return "\n".join(instructions) + "\n"


def _failure(issues: tuple[ValidationIssue, ...]) -> TrialBundleResult:
    return TrialBundleResult(
        created=False,
        persisted=False,
        trial_path=None,
        outcome=ExecutionOutcome(
            terminal_state=TerminalState.FAILED,
            artifact_disposition=ArtifactDisposition.NONE,
            issues=issues,
        ),
        bundle_path=None,
        manifest_path=None,
        trial_id=None,
        frozen_response_sha256=None,
    )


def _run_record(
    *,
    document: Mapping[str, object],
    prompt: str,
    schema: object,
    run: ProposalRun,
    issues: tuple[ValidationIssue, ...],
) -> bytes:
    record = {
        "schema_version": 1,
        "request": {
            "document_id": document["document_id"],
            "revision": document["revision"],
            "approved_content_sha256": realization_script_sha256(document),
            "prompt": prompt,
            "response_schema": schema,
        },
        "runner": {
            "provider": run.provider,
            "model": run.model,
            "model_settings": dict(run.model_settings),
            "started": run.started,
            "terminal_state": run.terminal_state,
        },
        "streams": {
            "raw_response_base64": (
                base64.b64encode(run.raw_response).decode("ascii")
                if run.raw_response is not None
                else None
            ),
            "events_base64": (
                base64.b64encode(run.events).decode("ascii") if run.events is not None else None
            ),
            "stderr_base64": (
                base64.b64encode(run.stderr).decode("ascii") if run.stderr is not None else None
            ),
        },
        "validation": {
            "valid": not issues,
            "issues": [
                {"code": issue.code.value, "message": issue.message, "path": issue.path}
                for issue in issues
            ],
        },
    }
    return (json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _persist_model_failure(
    *,
    document: Mapping[str, object],
    prompt: str,
    schema: object,
    run: ProposalRun,
    issues: tuple[ValidationIssue, ...],
    bundle_dir: Path,
    trial_id: str,
    proposal_record: bytes | None,
) -> TrialBundleResult:
    outcome = ExecutionOutcome(
        terminal_state=TerminalState.FAILED,
        artifact_disposition=ArtifactDisposition.NONE,
        issues=issues,
    )
    return create_failed_model_trial(
        document,
        bundle_dir,
        trial_id=trial_id,
        outcome=outcome,
        realization_record=_run_record(
            document=document,
            prompt=prompt,
            schema=schema,
            run=run,
            issues=issues,
        ),
        proposal_record=proposal_record,
    )


def _create_single_response_model_trial(
    document: Mapping[str, object],
    runner: ProposalRunner,
    bundle_dir: Path,
    *,
    trial_id: str,
    proposal_record_dir: Path | None = None,
) -> TrialBundleResult:
    """Call one structured runner and save the resulting generation trial."""

    if not isinstance(trial_id, str) or not trial_id.strip():
        return _failure(
            (
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "The trial ID must be a non-empty string",
                    "/trial_id",
                ),
            )
        )
    if Path(bundle_dir).exists():
        return _failure(
            (
                ValidationIssue(
                    IssueCode.STORAGE_CONFLICT,
                    "The trial bundle directory already exists",
                    "/bundle_dir",
                ),
            )
        )
    validation = check_realization_script(document)
    if not validation.valid:
        return _failure(validation.issues)
    script = cast(dict[str, object], document["script"])
    setup = cast(dict[str, object], script["performance_setup"])
    if setup["instrumentation"] != "solo_piano" or setup["target_duration_seconds"] != 180:
        return _failure(
            (
                ValidationIssue(
                    IssueCode.UNSUPPORTED_PROFILE,
                    "solo_piano_3m_v1 requires solo_piano and 180 seconds",
                    "/script/performance_setup",
                ),
            )
        )
    proposal_record = None
    if proposal_record_dir is not None:
        proposal_record, proposal_issue = package_proposal_record(proposal_record_dir, document)
        if proposal_issue is not None:
            return _failure((proposal_issue,))
    try:
        plan = preflight_typed_realization(document)
    except TypedRealizationError as error:
        return _failure((error.issue,))
    material_lengths = material_lengths_for_plan(plan)
    prompt = _prompt(document, material_lengths)
    schema = build_typed_response_schema(document)
    with TemporaryDirectory(prefix="scoim-response-schema-") as temporary_dir:
        schema_path = Path(temporary_dir) / "typed-realization-response-2.schema.json"
        schema_path.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run = runner.run(prompt, schema_path)
    if run.terminal_state != "completed" or run.issues:
        issues = run.issues or (
            ValidationIssue(
                IssueCode.RUNNER_FAILED,
                "The realization runner failed",
                "/runner",
            ),
        )
        if not run.started:
            return _failure(issues)
        return _persist_model_failure(
            document=document,
            prompt=prompt,
            schema=schema,
            run=run,
            issues=issues,
            bundle_dir=bundle_dir,
            trial_id=trial_id,
            proposal_record=proposal_record,
        )
    if run.raw_response is None:
        issues = (
            ValidationIssue(
                IssueCode.RUNNER_FAILED,
                "The realization runner returned no response",
                "/runner",
            ),
        )
        return _persist_model_failure(
            document=document,
            prompt=prompt,
            schema=schema,
            run=run,
            issues=issues,
            bundle_dir=bundle_dir,
            trial_id=trial_id,
            proposal_record=proposal_record,
        )
    try:
        response = json.loads(run.raw_response.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        issues = (
            ValidationIssue(
                IssueCode.MODEL_OUTPUT_INVALID,
                f"The realization response is not valid UTF-8 JSON: {error}",
                "/runner/response",
            ),
        )
        return _persist_model_failure(
            document=document,
            prompt=prompt,
            schema=schema,
            run=run,
            issues=issues,
            bundle_dir=bundle_dir,
            trial_id=trial_id,
            proposal_record=proposal_record,
        )
    issues = validate_typed_model_response(document, response)
    if issues:
        return _persist_model_failure(
            document=document,
            prompt=prompt,
            schema=schema,
            run=run,
            issues=issues,
            bundle_dir=bundle_dir,
            trial_id=trial_id,
            proposal_record=proposal_record,
        )
    assert isinstance(response, dict)
    frozen_response = normalize_typed_model_response(document, response)
    return create_trial_bundle(
        document,
        frozen_response,
        bundle_dir,
        trial_id=trial_id,
        realization_record=_run_record(
            document=document,
            prompt=prompt,
            schema=schema,
            run=run,
            issues=(),
        ),
        proposal_record=proposal_record,
    )


def create_model_trial(
    document: dict[str, object],
    runner: ProposalRunner,
    run_dir: Path,
    bundle_dir: Path,
    *,
    profile: str,
    model: str,
    trial_id: str,
    proposal_record_dir: Path | None = None,
    composition_manifest: bytes | None = None,
) -> TrialBundleResult:
    """Create one trial through the resumable realization workspace."""

    if not isinstance(trial_id, str) or not trial_id.strip():
        return _failure(
            (
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "The trial ID must be a non-empty string",
                    "/trial_id",
                ),
            )
        )
    if Path(bundle_dir).exists():
        return _failure(
            (
                ValidationIssue(
                    IssueCode.STORAGE_CONFLICT,
                    "The trial bundle directory already exists",
                    "/bundle_dir",
                ),
            )
        )
    resolved_run = Path(run_dir).resolve()
    resolved_bundle = Path(bundle_dir).resolve()
    if (
        resolved_run == resolved_bundle
        or resolved_run in resolved_bundle.parents
        or resolved_bundle in resolved_run.parents
    ):
        return _failure(
            (
                ValidationIssue(
                    IssueCode.STORAGE_CONFLICT,
                    "The work run and trial bundle must use separate non-nested directories",
                    "/run_dir",
                ),
            )
        )
    validation = check_realization_script(document)
    if not validation.valid:
        return _failure(validation.issues)
    if profile != "solo_piano_3m_v1":
        return _failure(
            (
                ValidationIssue(
                    IssueCode.UNSUPPORTED_PROFILE,
                    "The realization profile is not supported",
                    "/profile",
                ),
            )
        )
    profile_validation = check_solo_piano_3m_structure(document)
    if not profile_validation.valid:
        return _failure(profile_validation.issues)
    proposal_record = None
    if proposal_record_dir is not None:
        proposal_record, proposal_issue = package_proposal_record(proposal_record_dir, document)
        if proposal_issue is not None:
            return _failure((proposal_issue,))
    runner_issues = preflight_runner(runner)
    if runner_issues:
        return _failure(runner_issues)
    try:
        initialize_staged_realization(
            run_dir,
            document,
            profile=profile,
            model=model,
            review_after={},
        )
    except StateConflictError as error:
        return _failure((ValidationIssue(IssueCode.STORAGE_CONFLICT, str(error), "/run_dir"),))
    try:
        status = advance_staged_realization(run_dir, runner)
    except RunnerPreflightError as error:
        return _failure(error.issues)
    except (InterruptedAttemptError, RuntimeError, ValueError) as error:
        issues = _saved_staged_runner_issues(Path(run_dir), error)
        return _persist_failed_staged_trial(
            document,
            Path(run_dir),
            bundle_dir,
            trial_id=trial_id,
            issues=issues,
            proposal_record=proposal_record,
            composition_manifest=composition_manifest,
        )
    if status.status != "completed" or status.workspace is None:
        issues = status.issues or (
            ValidationIssue(
                IssueCode.RUNNER_FAILED,
                f"The staged realization stopped as {status.status}",
                "/runner",
            ),
        )
        return _persist_failed_staged_trial(
            document,
            Path(run_dir),
            bundle_dir,
            trial_id=trial_id,
            issues=issues,
            proposal_record=proposal_record,
            composition_manifest=composition_manifest,
        )
    frozen_response = {
        "schema_version": 3,
        "profile": profile,
        "workspace": workspace_to_dict(status.workspace),
    }
    realization_record, realization_issue = package_staged_realization_record(
        run_dir,
        frozen_response,
    )
    if realization_issue is not None or realization_record is None:
        assert realization_issue is not None
        return _failure((realization_issue,))
    return create_trial_bundle(
        document,
        frozen_response,
        bundle_dir,
        trial_id=trial_id,
        realization_record=realization_record,
        proposal_record=proposal_record,
        composition_manifest=composition_manifest,
    )


def _persist_failed_staged_trial(
    document: dict[str, object],
    run_dir: Path,
    bundle_dir: Path,
    *,
    trial_id: str,
    issues: tuple[ValidationIssue, ...],
    proposal_record: bytes | None,
    composition_manifest: bytes | None,
) -> TrialBundleResult:
    realization_record, record_issue = package_failed_staged_realization_record(run_dir, issues)
    if record_issue is not None or realization_record is None:
        assert record_issue is not None
        return _failure((record_issue,))
    return create_failed_model_trial(
        document,
        bundle_dir,
        trial_id=trial_id,
        outcome=ExecutionOutcome(
            TerminalState.FAILED,
            ArtifactDisposition.NONE,
            issues,
        ),
        realization_record=realization_record,
        proposal_record=proposal_record,
        composition_manifest=composition_manifest,
        realizer_version=5,
    )


def _saved_staged_runner_issues(
    run_dir: Path,
    error: Exception,
) -> tuple[ValidationIssue, ...]:
    for path in sorted(Path(run_dir).glob("stages/*/attempts/*/*/runner.json"), reverse=True):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict) or value.get("started") is not True:
            continue
        raw_issues = value.get("issues")
        if isinstance(raw_issues, list) and raw_issues:
            parsed: list[ValidationIssue] = []
            for raw in raw_issues:
                if not isinstance(raw, dict):
                    break
                try:
                    parsed.append(
                        ValidationIssue(
                            IssueCode(raw["code"]),
                            str(raw["message"]),
                            str(raw["path"]),
                        )
                    )
                except (KeyError, ValueError):
                    break
            else:
                return tuple(parsed)
        state = value.get("terminal_state")
        code = IssueCode.RUNNER_TIMEOUT if state == "timeout" else IssueCode.RUNNER_FAILED
        return (ValidationIssue(code, str(error), "/runner"),)
    return (ValidationIssue(IssueCode.MODEL_OUTPUT_INVALID, str(error), "/runner/response"),)
