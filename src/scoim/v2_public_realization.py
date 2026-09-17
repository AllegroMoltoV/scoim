"""Public orchestration for the solo-piano v2 generation path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from llm_musical_composer.run_state import StateConflictError, sha256_file, sha256_json

from .finite_model_operation import check_finite_model_operation_records
from .phase3_realization import Phase3Request, realize_phase3
from .phase3_state import load_complete_phase3_state
from .phase4_realization import Phase4Request, realize_phase4
from .phase4_state import load_complete_phase4_state
from .phase5_realization import Phase5Request, realize_phase5
from .phase5_state import load_complete_phase5_state
from .phase6_realization import Phase6Request, realize_phase6
from .phase6_state import load_complete_phase6_run
from .phase7_realization import Phase7Request, realize_phase7
from .phase7_state import load_complete_phase7_run
from .phase8_bundle import Phase8BundleRequest, create_phase8_bundle, verify_phase8_bundle
from .projection_ledger import ProjectionLedgerEntry
from .proposal import ProposalRunner
from .runner_identity import IdentityCheckingRunner, read_runner_identity
from .script_0_4_compilation import Script04CompilationRequest, compile_script_0_4
from .v2_composition_bundle import (
    create_v2_composition_bundle,
    verify_v2_composition_bundle,
)
from .v2_public_run import PublicV2RunRequest, ensure_public_v2_run
from .validation import IssueCode, ValidationIssue

_PROFILE = "solo_piano_3m_v2"


@dataclass(frozen=True, slots=True)
class V2PublicRealizationResult:
    """Internal result mapped onto the stable public API at its boundary."""

    succeeded: bool
    persisted: bool
    composition_bundle_path: Path | None
    trial_bundle_path: Path | None
    artifacts: dict[str, str]
    issues: tuple[ValidationIssue, ...]


def realize_v2_composition(
    source: str | Path,
    destination: str | Path,
    *,
    runner: ProposalRunner,
    model: str,
    trial_id: str,
) -> V2PublicRealizationResult:
    """Realize one verified v2 composition bundle."""
    composition = Path(source).resolve()
    verification = verify_v2_composition_bundle(composition)
    if not verification.valid:
        return _failure(False, verification.issues)
    manifest = _read_object(composition / "manifest.json")
    composition_id = manifest.get("composition_id")
    if not isinstance(composition_id, str) or not composition_id:
        return _failure(
            False,
            (_issue(IssueCode.LINEAGE_MISMATCH, "composition ID is invalid", "/bundle"),),
        )
    return _realize(
        destination=destination,
        input_kind="composition",
        input_sha256=sha256_file(composition / "manifest.json"),
        composition_id=composition_id,
        trial_id=trial_id,
        runner=runner,
        model=model,
        composition=composition,
        composition_bundle_path=None,
    )


def realize_v2_flow(
    document: dict[str, object],
    destination: str | Path,
    *,
    runner: ProposalRunner,
    model: str,
    trial_id: str,
    composition_id: str,
) -> V2PublicRealizationResult:
    """Compile and realize one approved flow through the v2 path."""
    return _realize(
        destination=destination,
        input_kind="flow",
        input_sha256=sha256_json(document),
        composition_id=composition_id,
        trial_id=trial_id,
        runner=runner,
        model=model,
        flow=document,
        composition_bundle_path=Path("composition"),
    )


def _realize(
    *,
    destination: str | Path,
    input_kind: str,
    input_sha256: str,
    composition_id: str,
    trial_id: str,
    runner: ProposalRunner,
    model: str,
    composition_bundle_path: Path | None,
    flow: dict[str, object] | None = None,
    composition: Path | None = None,
) -> V2PublicRealizationResult:
    output = Path(destination).resolve()
    try:
        identity = read_runner_identity(runner)
    except (TypeError, ValueError) as error:
        return _failure(False, (_issue(IssueCode.RUNNER_INCOMPATIBLE, str(error), "/runner"),))
    if identity.model != model:
        return _failure(
            False,
            (_issue(IssueCode.LINEAGE_MISMATCH, "requested model differs from runner", "/model"),),
        )
    initialized = ensure_public_v2_run(
        output,
        PublicV2RunRequest(
            input_kind=input_kind,
            input_sha256=input_sha256,
            composition_id=composition_id,
            trial_id=trial_id,
            runner_identity=identity,
        ),
    )
    if not initialized.ready:
        return _failure(False, initialized.issues)
    checked_runner = IdentityCheckingRunner(runner, identity)
    work = output / "realization-work"
    try:
        if flow is not None:
            composition, compilation_issues = _compile_flow(
                flow,
                composition_id,
                checked_runner,
                work / "phase2",
                output / "composition",
            )
            if composition is None:
                return _failure(
                    True,
                    compilation_issues,
                    composition_bundle_path=composition_bundle_path,
                )
        assert composition is not None
        result = _realize_phases(
            composition,
            work,
            output / "trial",
            checked_runner,
            composition_id,
            trial_id,
        )
    except (KeyError, OSError, TypeError, ValueError, StateConflictError) as error:
        return _failure(
            True,
            (_issue(IssueCode.STORAGE_CONFLICT, str(error), "/output"),),
            composition_bundle_path=composition_bundle_path,
        )
    if result is not None:
        return _failure(
            True,
            result,
            composition_bundle_path=composition_bundle_path,
            trial_bundle_path=Path("trial") if (output / "trial").exists() else None,
        )
    return V2PublicRealizationResult(
        True,
        True,
        composition_bundle_path,
        Path("trial"),
        _artifacts(output),
        (),
    )


def _compile_flow(
    flow: dict[str, object],
    composition_id: str,
    runner: ProposalRunner,
    phase2_dir: Path,
    bundle_dir: Path,
) -> tuple[Path | None, tuple[ValidationIssue, ...]]:
    if not bundle_dir.exists():
        compiled = compile_script_0_4(
            Script04CompilationRequest(flow, composition_id),
            runner,
            phase2_dir,
        )
        if not compiled.compiled:
            return None, compiled.issues
        created = create_v2_composition_bundle(phase2_dir, bundle_dir)
        if not created.created:
            return None, created.issues
    verification = verify_v2_composition_bundle(bundle_dir)
    if not verification.valid:
        return None, verification.issues
    return bundle_dir, ()


def _realize_phases(
    composition: Path,
    work: Path,
    trial_dir: Path,
    runner: ProposalRunner,
    composition_id: str,
    trial_id: str,
) -> tuple[ValidationIssue, ...] | None:
    document = _read_object(composition / "validated-script.json")
    ledger = _read_ledger(composition / "projection-ledger.json")
    phase3_dir = work / "phase3"
    if not _is_complete(phase3_dir):
        phase3 = realize_phase3(Phase3Request(document, ledger), runner, phase3_dir)
        if not phase3.realized:
            return phase3.issues
    phase3_record_issues = _model_record_issues(phase3_dir, "phase3")
    if phase3_record_issues:
        return phase3_record_issues
    phase3_state = _read_object(phase3_dir / "outputs" / "phase3-state.json")
    ledger = _read_ledger(phase3_dir / "outputs" / "projection-ledger.json")
    load_complete_phase3_state(document, phase3_state, ledger)

    phase4_dir = work / "phase4"
    if not _is_complete(phase4_dir):
        phase4 = realize_phase4(Phase4Request(document, phase3_state, ledger), runner, phase4_dir)
        if not phase4.realized:
            return phase4.issues
    phase4_record_issues = _model_record_issues(phase4_dir, "phase4")
    if phase4_record_issues:
        return phase4_record_issues
    phase4_state = _read_object(phase4_dir / "outputs" / "phase4-state.json")
    ledger = _read_ledger(phase4_dir / "outputs" / "projection-ledger.json")
    load_complete_phase4_state(document, phase3_state, phase4_state, ledger)

    phase5_dir = work / "phase5"
    if not _is_complete(phase5_dir):
        phase5 = realize_phase5(
            Phase5Request(document, phase3_state, phase4_state, ledger), runner, phase5_dir
        )
        if not phase5.realized:
            return phase5.issues
    phase5_record_issues = _model_record_issues(phase5_dir, "phase5")
    if phase5_record_issues:
        return phase5_record_issues
    phase5_state = _read_object(phase5_dir / "outputs" / "phase5-state.json")
    ledger = _read_ledger(phase5_dir / "outputs" / "projection-ledger.json")
    load_complete_phase5_state(document, phase3_state, phase4_state, phase5_state, ledger)

    phase6_dir = work / "phase6"
    if phase6_dir.exists():
        load_complete_phase6_run(phase6_dir)
    else:
        phase6 = realize_phase6(
            Phase6Request(document, phase3_state, phase4_state, phase5_state, ledger),
            phase6_dir,
        )
        if not phase6.realized:
            return phase6.issues

    phase7_dir = work / "phase7"
    if not _is_complete(phase7_dir):
        phase7 = realize_phase7(Phase7Request(phase6_dir), runner, phase7_dir)
        if not phase7.realized:
            return phase7.issues
    load_complete_phase7_run(phase7_dir)

    if trial_dir.exists():
        verified = verify_phase8_bundle(trial_dir)
        return None if verified.valid else verified.issues
    created = create_phase8_bundle(
        Phase8BundleRequest(
            phase7_run_dir=phase7_dir,
            phase_run_dirs={
                "phase3": phase3_dir,
                "phase4": phase4_dir,
                "phase5": phase5_dir,
                "phase6": phase6_dir,
            },
            composition_id=composition_id,
            trial_id=trial_id,
            composition_manifest=(composition / "manifest.json").read_bytes(),
        ),
        trial_dir,
    )
    return None if created.created else created.issues


def _artifacts(output: Path) -> dict[str, str]:
    candidates = {
        "phase_04_foreground": "realization-work/phase4/outputs/foreground-preview.mid",
        "phase_06_score": "realization-work/phase6/outputs/score-preview.mid",
        "final_musicxml": "trial/artifacts/score.musicxml",
        "final_smf": "trial/artifacts/final.mid",
    }
    return {name: path for name, path in candidates.items() if (output / path).is_file()}


def _is_complete(run_dir: Path) -> bool:
    summary = run_dir / "outputs" / "realization.json"
    if not summary.is_file():
        return False
    return _read_object(summary).get("outcome") == "complete"


def _model_record_issues(
    run_dir: Path,
    phase_name: str,
) -> tuple[ValidationIssue, ...]:
    checked = check_finite_model_operation_records(run_dir)
    return tuple(
        ValidationIssue(
            IssueCode.LINEAGE_MISMATCH,
            f"{phase_name} model operation record is invalid: {issue.message}",
            issue.path,
        )
        for issue in checked.issues
    )


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return cast(dict[str, object], value)


def _read_ledger(path: Path) -> tuple[ProjectionLedgerEntry, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"expected a projection ledger array: {path}")
    return tuple(ProjectionLedgerEntry(**cast(dict[str, object], item)) for item in value)


def _issue(code: IssueCode, message: str, path: str) -> ValidationIssue:
    return ValidationIssue(code, message, path)


def _failure(
    persisted: bool,
    issues: tuple[ValidationIssue, ...],
    *,
    composition_bundle_path: Path | None = None,
    trial_bundle_path: Path | None = None,
) -> V2PublicRealizationResult:
    return V2PublicRealizationResult(
        False,
        persisted,
        composition_bundle_path,
        trial_bundle_path,
        {},
        issues,
    )
