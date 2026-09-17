"""Public orchestration for composing, realizing, and replaying one SCoIM request."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .composition_bundle import compose_flow, verify_composition_bundle
from .flow_validation import check_flow
from .phase8_bundle import replay_phase8_bundle
from .proposal import ProposalRunner
from .realization_generation import create_model_trial
from .script_compilation import CompilationRequest
from .trial_bundle import replay_trial_bundle
from .v2_public_realization import realize_v2_composition, realize_v2_flow
from .validation import IssueCode, ValidationIssue


@dataclass(frozen=True, slots=True)
class PublicRealizationResult:
    """Result paths are relative to one public output root."""

    succeeded: bool
    persisted: bool
    input_kind: str | None
    output_path: Path | None
    composition_bundle_path: Path | None
    trial_bundle_path: Path | None
    replay_path: Path | None
    artifacts: dict[str, str]
    issues: tuple[ValidationIssue, ...]


def _failure(
    code: IssueCode,
    message: str,
    path: str,
    *,
    input_kind: str | None = None,
    persisted: bool = False,
    output_path: Path | None = None,
    composition_bundle_path: Path | None = None,
    trial_bundle_path: Path | None = None,
) -> PublicRealizationResult:
    return PublicRealizationResult(
        False,
        persisted,
        input_kind,
        output_path,
        composition_bundle_path,
        trial_bundle_path,
        None,
        {},
        (ValidationIssue(code, message, path),),
    )


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("The input JSON must be an object")
    return value


def _artifact_paths(root: Path, artifact_dir: str) -> dict[str, str]:
    candidates = {
        "phase_04_melody": f"{artifact_dir}/phase-04-melody.mid",
        "phase_06_score": f"{artifact_dir}/phase-06-score.mid",
        "final_smf": f"{artifact_dir}/final.mid",
    }
    return {name: path for name, path in candidates.items() if (root / path).is_file()}


def _generate_from_composition(
    source: Path,
    output: Path,
    *,
    input_kind: str,
    composition_bundle_path: Path | None,
    runner: ProposalRunner | None,
    model: str | None,
    trial_id: str | None,
    profile: str,
) -> PublicRealizationResult:
    if runner is None or not model or not trial_id:
        return _failure(
            IssueCode.SCHEMA_INVALID,
            "Composition realization requires runner, model, and trial ID",
            "/arguments",
            input_kind=input_kind,
        )
    verification = verify_composition_bundle(source)
    if not verification.valid:
        issue = verification.issues[0]
        return _failure(issue.code, issue.message, issue.path, input_kind=input_kind)
    manifest = _read_object(source / "manifest.json")
    if manifest.get("terminal_state") != "succeeded":
        return _failure(
            IssueCode.SEMANTIC_INVALID,
            "Only a successful composition bundle can be realized",
            "/manifest/terminal_state",
            input_kind=input_kind,
        )
    source_resolved = source.resolve()
    output_resolved = output.resolve()
    if output_resolved == source_resolved or output_resolved.is_relative_to(source_resolved):
        return _failure(
            IssueCode.STORAGE_CONFLICT,
            "The output must be outside the source composition bundle",
            "/output",
            input_kind=input_kind,
        )
    try:
        document = _read_object(source / "validated-script.json")
        composition_manifest = (source / "manifest.json").read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        return _failure(
            IssueCode.STORAGE_ERROR,
            str(error),
            "/output",
            input_kind=input_kind,
            persisted=output.exists(),
            output_path=output if output.exists() else None,
            composition_bundle_path=composition_bundle_path,
        )
    trial = create_model_trial(
        document,
        runner,
        output / "realization-work",
        output / "trial",
        profile=profile,
        model=model,
        trial_id=trial_id,
        composition_manifest=composition_manifest,
    )
    return PublicRealizationResult(
        trial.created,
        output.exists() or trial.persisted,
        input_kind,
        output,
        composition_bundle_path,
        Path("trial") if trial.persisted else None,
        None,
        _artifact_paths(output, "trial/artifacts"),
        trial.issues,
    )


def _realize_composition(
    source: Path,
    output: Path,
    *,
    runner: ProposalRunner | None,
    model: str | None,
    trial_id: str | None,
    profile: str,
) -> PublicRealizationResult:
    return _generate_from_composition(
        source,
        output,
        input_kind="composition",
        composition_bundle_path=None,
        runner=runner,
        model=model,
        trial_id=trial_id,
        profile=profile,
    )


def _realize_flow(
    document: dict[str, object],
    output: Path,
    *,
    runner: ProposalRunner | None,
    model: str | None,
    trial_id: str | None,
    composition_id: str | None,
    profile: str,
) -> PublicRealizationResult:
    if runner is None or not model or not trial_id:
        return _failure(
            IssueCode.SCHEMA_INVALID,
            "Flow realization requires runner, model, and trial ID",
            "/arguments",
            input_kind="flow",
        )
    checked = check_flow(document)
    if not checked.valid:
        issue = checked.issues[0]
        return _failure(issue.code, issue.message, issue.path, input_kind="flow")
    if document.get("status") != "approved":
        return _failure(
            IssueCode.SEMANTIC_INVALID,
            "Only an approved flow can be realized",
            "/status",
            input_kind="flow",
        )
    identifier = composition_id or document.get("document_id")
    if not isinstance(identifier, str) or not identifier:
        return _failure(
            IssueCode.SCHEMA_INVALID,
            "The composition ID must be a non-empty string",
            "/composition_id",
            input_kind="flow",
        )
    composition = compose_flow(
        CompilationRequest(document, identifier, profile),
        runner,
        output / "composition",
    )
    if not composition.succeeded:
        return PublicRealizationResult(
            False,
            composition.persisted,
            "flow",
            output if output.exists() else None,
            Path("composition") if composition.persisted else None,
            None,
            None,
            {},
            composition.issues,
        )
    return _generate_from_composition(
        output / "composition",
        output,
        input_kind="flow",
        composition_bundle_path=Path("composition"),
        runner=runner,
        model=model,
        trial_id=trial_id,
        profile=profile,
    )


def realize(
    source: str | Path,
    destination: str | Path,
    *,
    runner: ProposalRunner | None = None,
    model: str | None = None,
    trial_id: str | None = None,
    composition_id: str | None = None,
    profile: str | None = None,
) -> PublicRealizationResult:
    """Run the operation declared by one explicitly typed public input."""

    input_path = Path(source)
    output = Path(destination)
    if profile not in {None, "solo_piano_3m_v1", "solo_piano_3m_v2"}:
        return _failure(
            IssueCode.UNSUPPORTED_PROFILE,
            "The realization profile is not supported",
            "/profile",
        )
    if output.exists() and (
        profile == "solo_piano_3m_v1" or not (output / "public-run.json").is_file()
    ):
        return _failure(
            IssueCode.STORAGE_CONFLICT,
            "The public realization output already exists",
            "/output",
        )
    try:
        if input_path.is_file():
            document = _read_object(input_path)
            if (
                document.get("document_type") == "flow"
                and document.get("schema_version") == "0.1.0"
            ):
                selected_profile = profile or "solo_piano_3m_v2"
                if selected_profile == "solo_piano_3m_v2":
                    return _realize_v2_flow(
                        document,
                        output,
                        runner=runner,
                        model=model,
                        trial_id=trial_id,
                        composition_id=composition_id,
                    )
                return _realize_flow(
                    document,
                    output,
                    runner=runner,
                    model=model,
                    trial_id=trial_id,
                    composition_id=composition_id,
                    profile=selected_profile,
                )
        elif input_path.is_dir():
            manifest = _read_object(input_path / "manifest.json")
            if manifest.get("bundle_type") == "composition":
                if (
                    manifest.get("schema_version") == 3
                    and manifest.get("target_profile") == "solo_piano_3m_v2"
                ):
                    if profile not in {None, "solo_piano_3m_v2"}:
                        return _failure(
                            IssueCode.UNSUPPORTED_PROFILE,
                            "The composition bundle does not match the requested profile",
                            "/profile",
                            input_kind="composition",
                        )
                    return _realize_v2_composition(
                        input_path,
                        output,
                        runner=runner,
                        model=model,
                        trial_id=trial_id,
                        composition_id=composition_id,
                    )
                if profile != "solo_piano_3m_v1":
                    return _failure(
                        IssueCode.UNSUPPORTED_PROFILE,
                        "The v1 composition bundle requires an explicit v1 profile",
                        "/profile",
                        input_kind="composition",
                    )
                return _realize_composition(
                    input_path,
                    output,
                    runner=runner,
                    model=model,
                    trial_id=trial_id,
                    profile=profile or "solo_piano_3m_v1",
                )
            if manifest.get("bundle_type") == "scoim-generation-trial":
                if any(
                    value is not None
                    for value in (runner, model, trial_id, composition_id, profile)
                ):
                    return _failure(
                        IssueCode.SCHEMA_INVALID,
                        "Trial replay does not accept generation arguments",
                        "/arguments",
                        input_kind="trial",
                    )
                if output.exists():
                    return _failure(
                        IssueCode.STORAGE_CONFLICT,
                        "The public realization output already exists",
                        "/output",
                        input_kind="trial",
                    )
                replayed = replay_phase8_bundle(input_path, output / "artifacts")
                return PublicRealizationResult(
                    replayed.replayed,
                    output.exists(),
                    "trial",
                    output if output.exists() else None,
                    None,
                    None,
                    Path("artifacts") if replayed.replayed else None,
                    {
                        name: path
                        for name, path in {
                            "final_musicxml": "artifacts/score.musicxml",
                            "final_smf": "artifacts/final.mid",
                        }.items()
                        if (output / path).is_file()
                    },
                    replayed.issues,
                )
            if manifest.get("bundle_type") == "realization" or (
                manifest.get("schema_version") == 3 and "bundle_type" not in manifest
            ):
                if any(
                    value is not None
                    for value in (runner, model, trial_id, composition_id, profile)
                ):
                    return _failure(
                        IssueCode.SCHEMA_INVALID,
                        "Trial replay does not accept generation arguments",
                        "/arguments",
                        input_kind="trial",
                    )
                if output.exists():
                    return _failure(
                        IssueCode.STORAGE_CONFLICT,
                        "The public realization output already exists",
                        "/output",
                        input_kind="trial",
                    )
                replayed = replay_trial_bundle(input_path, output / "artifacts")
                return PublicRealizationResult(
                    replayed.replayed,
                    output.exists(),
                    "trial",
                    output if output.exists() else None,
                    None,
                    None,
                    Path("artifacts") if replayed.replayed else None,
                    _artifact_paths(output, "artifacts"),
                    replayed.issues,
                )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        return _failure(IssueCode.STORAGE_ERROR, str(error), "/input")
    return _failure(
        IssueCode.UNSUPPORTED_SCHEMA_VERSION,
        "The input does not declare a supported SCoIM document or bundle contract",
        "/input",
    )


def _realize_v2_flow(
    document: dict[str, object],
    output: Path,
    *,
    runner: ProposalRunner | None,
    model: str | None,
    trial_id: str | None,
    composition_id: str | None,
) -> PublicRealizationResult:
    checked = check_flow(document)
    if not checked.valid:
        issue = checked.issues[0]
        return _failure(issue.code, issue.message, issue.path, input_kind="flow")
    if document.get("status") != "approved":
        return _failure(
            IssueCode.SEMANTIC_INVALID,
            "Only an approved flow can be realized",
            "/status",
            input_kind="flow",
        )
    identifier = composition_id or document.get("document_id")
    if not isinstance(identifier, str) or not identifier:
        return _failure(
            IssueCode.SCHEMA_INVALID,
            "The composition ID must be a non-empty string",
            "/composition_id",
            input_kind="flow",
        )
    if runner is None or not model or not trial_id:
        return _failure(
            IssueCode.SCHEMA_INVALID,
            "Flow realization requires runner, model, and trial ID",
            "/arguments",
            input_kind="flow",
        )
    result = realize_v2_flow(
        document,
        output,
        runner=runner,
        model=model,
        trial_id=trial_id,
        composition_id=identifier,
    )
    return PublicRealizationResult(
        result.succeeded,
        result.persisted,
        "flow",
        output if result.persisted else None,
        result.composition_bundle_path,
        result.trial_bundle_path,
        None,
        result.artifacts,
        result.issues,
    )


def _realize_v2_composition(
    source: Path,
    output: Path,
    *,
    runner: ProposalRunner | None,
    model: str | None,
    trial_id: str | None,
    composition_id: str | None,
) -> PublicRealizationResult:
    if composition_id is not None:
        return _failure(
            IssueCode.SCHEMA_INVALID,
            "A composition bundle already fixes its composition ID",
            "/composition_id",
            input_kind="composition",
        )
    if runner is None or not model or not trial_id:
        return _failure(
            IssueCode.SCHEMA_INVALID,
            "Composition realization requires runner, model, and trial ID",
            "/arguments",
            input_kind="composition",
        )
    source_resolved = source.resolve()
    output_resolved = output.resolve()
    if output_resolved == source_resolved or output_resolved.is_relative_to(source_resolved):
        return _failure(
            IssueCode.STORAGE_CONFLICT,
            "The output must be outside the source composition bundle",
            "/output",
            input_kind="composition",
        )
    result = realize_v2_composition(
        source,
        output,
        runner=runner,
        model=model,
        trial_id=trial_id,
    )
    return PublicRealizationResult(
        result.succeeded,
        result.persisted,
        "composition",
        output if result.persisted else None,
        result.composition_bundle_path,
        result.trial_bundle_path,
        None,
        result.artifacts,
        result.issues,
    )
