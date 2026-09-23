"""Deterministic phase-6 assembly of validated score values."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_musical_composer.performance_timing import TimingResolutionError
from llm_musical_composer.run_state import atomic_write_json, sha256_file, sha256_json

from .neutral_score_preview import (
    NeutralPreviewSegment,
    check_neutral_score_preview,
    write_neutral_score_preview,
)
from .phase5_state import load_complete_phase5_state
from .phase6_state import load_complete_phase6_run
from .projection_ledger import (
    ProjectionLedgerEntry,
    ProjectionLedgerValidationError,
    validate_projection_ledger,
)
from .score_generation_context import require_current_generation_context
from .score_projection import ScoreProjectionError, build_score_spec
from .score_timing import QUANTIZED_TIMING
from .validation import IssueCode, ValidationIssue


@dataclass(frozen=True, slots=True)
class Phase6Request:
    validated_script: Mapping[str, object]
    phase3_state: Mapping[str, object]
    phase4_state: Mapping[str, object]
    phase5_state: Mapping[str, object]
    projection_ledger: Sequence[ProjectionLedgerEntry | Mapping[str, object]]
    target_profile: str = "solo_piano_3m_v2"


@dataclass(frozen=True, slots=True)
class Phase6RealizationResult:
    realized: bool
    outcome: str
    run_dir: Path | None
    state: dict[str, object] | None
    issues: tuple[ValidationIssue, ...]


def realize_phase6(
    request: Phase6Request,
    run_dir: str | Path,
) -> Phase6RealizationResult:
    """Assemble phase-3 through phase-5 values without creating musical content."""
    destination = Path(run_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"phase-6 run directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        loaded = load_complete_phase5_state(
            request.validated_script,
            request.phase3_state,
            request.phase4_state,
            request.phase5_state,
            request.projection_ledger,
            target_profile=request.target_profile,
        )
        notes_by_placement = dict(loaded.phase4.notes_by_material_placement)
        if not notes_by_placement.keys().isdisjoint(loaded.notes_by_material_placement):
            raise ValueError("phase-4 and phase-5 placement values overlap")
        notes_by_placement.update(loaded.notes_by_material_placement)
        harmonic_plan = loaded.phase4.phase3.plan
        require_current_generation_context(harmonic_plan.generation_context_contract)
        if harmonic_plan.timing_contract != QUANTIZED_TIMING:
            raise ValueError("new phase 6 requires the current timing contract; start a new trial")
        unit_ids = tuple(harmonic_plan.length_units_by_score_unit)
        score, new_evidence = build_score_spec(
            request.validated_script,
            harmonic_plan.piece_plan,
            score_id=f"{harmonic_plan.piece_plan.plan_id}-score",
            divisions=harmonic_plan.divisions,
            timing_contract=harmonic_plan.timing_contract,
            length_units_by_score_unit=harmonic_plan.length_units_by_score_unit,
            harmonies_by_score_unit=loaded.phase4.phase3.harmonies_by_score_unit,
            directions_by_score_unit={unit_id: () for unit_id in unit_ids},
            notes_by_material_placement=notes_by_placement,
            cumulative_projection_ledger=loaded.cumulative_projection_ledger,
        )
        cumulative_ledger = (*loaded.cumulative_projection_ledger, *new_evidence)
        validate_projection_ledger(cumulative_ledger)
    except (
        KeyError,
        TypeError,
        ValueError,
        ProjectionLedgerValidationError,
        ScoreProjectionError,
    ) as error:
        issue = ValidationIssue(IssueCode.SEMANTIC_INVALID, str(error), "/phase6_request")
        temporary = _new_temporary_directory(destination)
        _write_failure_record(temporary, "request_invalid", issue)
        return Phase6RealizationResult(
            False,
            "request_invalid",
            temporary,
            None,
            (issue,),
        )

    temporary = _new_temporary_directory(destination)
    atomic_write_json(
        temporary / "inputs" / "validated-script.json",
        dict(request.validated_script),
    )
    atomic_write_json(temporary / "inputs" / "phase3-state.json", dict(request.phase3_state))
    atomic_write_json(temporary / "inputs" / "phase4-state.json", dict(request.phase4_state))
    atomic_write_json(temporary / "inputs" / "phase5-state.json", dict(request.phase5_state))
    atomic_write_json(
        temporary / "inputs" / "projection-ledger.json",
        [asdict(entry) for entry in loaded.cumulative_projection_ledger],
    )
    score_path = temporary / "outputs" / "score-spec.json"
    ledger_path = temporary / "outputs" / "projection-ledger.json"
    atomic_write_json(score_path, asdict(score))
    atomic_write_json(ledger_path, [asdict(entry) for entry in cumulative_ledger])
    setup = request.validated_script["script"]["performance_setup"]
    preview_path = temporary / "outputs" / "score-preview.mid"
    preview_segments = tuple(
        NeutralPreviewSegment(
            unit.length_units,
            tuple(note for layer in unit.score_unit_layers for note in layer.notes),
        )
        for unit in score.score_units
    )
    target_seconds = float(setup["target_duration_seconds"])
    try:
        write_neutral_score_preview(
            preview_segments,
            target_seconds,
            preview_path,
            meta_track_name="SCoIM phase 6 score",
            note_track_name="Score",
            timing_contract=harmonic_plan.timing_contract,
        )
        check_neutral_score_preview(
            preview_segments,
            target_seconds,
            preview_path,
            timing_contract=harmonic_plan.timing_contract,
        )
    except ValueError as error:
        unrepresentable = isinstance(error, TimingResolutionError)
        outcome = "unrepresentable" if unrepresentable else "preview_invalid"
        code = IssueCode.UNREPRESENTABLE if unrepresentable else IssueCode.SEMANTIC_INVALID
        issue = ValidationIssue(code, str(error), "/score_preview")
        _write_failure_record(temporary, outcome, issue)
        return Phase6RealizationResult(
            False,
            outcome,
            temporary,
            None,
            (issue,),
        )
    state = {
        "outcome": "complete",
        "target_profile": request.target_profile,
        "input_script_sha256": sha256_json(request.validated_script),
        "input_phase3_state_sha256": sha256_json(request.phase3_state),
        "input_phase4_state_sha256": sha256_json(request.phase4_state),
        "input_phase5_state_sha256": sha256_json(request.phase5_state),
        "input_projection_ledger_sha256": sha256_json(
            [asdict(entry) for entry in loaded.cumulative_projection_ledger]
        ),
        "score_spec": {
            "path": "outputs/score-spec.json",
            "sha256": sha256_file(score_path),
        },
        "projection_ledger": {
            "path": "outputs/projection-ledger.json",
            "sha256": sha256_file(ledger_path),
        },
        "score_preview": {
            "path": "outputs/score-preview.mid",
            "sha256": sha256_file(preview_path),
        },
    }
    atomic_write_json(temporary / "outputs" / "phase6-state.json", state)
    atomic_write_json(
        temporary / "outputs" / "realization.json",
        {"outcome": "complete", "call_count": 0, "issues": []},
    )
    try:
        load_complete_phase6_run(temporary)
    except (KeyError, OSError, TypeError, ValueError) as error:
        issue = ValidationIssue(IssueCode.SEMANTIC_INVALID, str(error), "/phase6_state")
        failed_state = {**state, "outcome": "persisted_state_invalid"}
        atomic_write_json(temporary / "outputs" / "phase6-state.json", failed_state)
        _write_failure_record(temporary, "persisted_state_invalid", issue)
        return Phase6RealizationResult(
            False,
            "persisted_state_invalid",
            temporary,
            failed_state,
            (issue,),
        )
    temporary.replace(destination)
    return Phase6RealizationResult(True, "complete", destination, state, ())


def _new_temporary_directory(destination: Path) -> Path:
    return Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)).resolve()


def _write_failure_record(
    temporary: Path,
    outcome: str,
    issue: ValidationIssue,
) -> None:
    atomic_write_json(
        temporary / "outputs" / "realization.json",
        {
            "outcome": outcome,
            "call_count": 0,
            "issues": [
                {
                    "code": issue.code.value,
                    "message": issue.message,
                    "path": issue.path,
                }
            ],
        },
    )
