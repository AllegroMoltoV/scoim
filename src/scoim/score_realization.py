"""Generate, accept, and persist whole score work groups in dependency order."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from llm_musical_composer.run_state import RunStore, atomic_write_json, sha256_file, sha256_json

from .finite_model_operation import execute_finite_model_operation
from .neutral_score_preview import NeutralPreviewSegment, write_neutral_score_preview
from .phase3_state import load_complete_phase3_state
from .projection_ledger import ProjectionLedgerEntry
from .proposal import ProposalRunner, preflight_runner
from .score_generation_context import require_current_generation_context
from .score_group_pitch_placement import SCORE_GROUP_SEARCH_LIMIT
from .score_ir import ScoreNote
from .score_model_contracts import ScoreGroupCandidateResult, evaluate_score_group_candidate
from .score_state import (
    SCORE_STATE_SCHEMA_VERSION,
    assemble_score,
    load_complete_score_run,
    operation_comparisons,
    read_score_object,
    score_operation_request,
    score_operation_snapshot,
    score_run_spec,
    serialized_notes,
)
from .score_timing import QUANTIZED_TIMING
from .score_work_plan import build_score_work_plan
from .validation import IssueCode, ValidationIssue


@dataclass(frozen=True, slots=True)
class ScoreRequest:
    validated_script: Mapping[str, object]
    phase3_state: Mapping[str, object]
    projection_ledger: Sequence[ProjectionLedgerEntry | Mapping[str, object]]
    target_profile: str = "solo_piano_3m_v2"


@dataclass(frozen=True, slots=True)
class ScoreRealizationResult:
    realized: bool
    outcome: str
    run_dir: Path | None
    state: dict[str, object] | None
    issues: tuple[ValidationIssue, ...]


def realize_score(
    request: ScoreRequest,
    runner: ProposalRunner,
    run_dir: str | Path,
    *,
    max_new_operations: int | None = None,
) -> ScoreRealizationResult:
    """Accept each complete candidate atomically and reconstruct accepted work on resume."""
    if max_new_operations is not None and max_new_operations <= 0:
        raise ValueError("max_new_operations must be positive")
    try:
        phase3 = load_complete_phase3_state(
            request.validated_script,
            request.phase3_state,
            request.projection_ledger,
            target_profile=request.target_profile,
        )
        require_current_generation_context(phase3.plan.generation_context_contract)
        if phase3.plan.timing_contract != QUANTIZED_TIMING:
            raise ValueError("new score generation requires the current timing contract")
        work_plan = build_score_work_plan(request.validated_script)
        if work_plan.issues:
            return ScoreRealizationResult(False, "request_invalid", None, None, work_plan.issues)
    except (KeyError, TypeError, ValueError) as error:
        issue = ValidationIssue(IssueCode.SEMANTIC_INVALID, str(error), "/score_request")
        return ScoreRealizationResult(False, "request_invalid", None, None, (issue,))

    destination = Path(run_dir).resolve()
    spec = score_run_spec(
        request.validated_script,
        request.phase3_state,
        phase3.cumulative_projection_ledger,
        work_plan,
    )
    store = RunStore(destination, max_calls=cast(int, spec["max_calls"]))
    store.initialize(spec)
    store.snapshot_json("inputs/validated-script.json", dict(request.validated_script))
    store.snapshot_json("inputs/phase3-state.json", dict(request.phase3_state))
    store.snapshot_json(
        "inputs/projection-ledger.json",
        [asdict(entry) for entry in phase3.cumulative_projection_ledger],
    )
    if (destination / "outputs").exists():
        load_complete_score_run(destination)
        return ScoreRealizationResult(
            True,
            "complete",
            destination,
            read_score_object(destination / "outputs/score-state.json"),
            (),
        )
    if any(
        not (destination / "events" / operation.operation_id / "accepted.json").is_file()
        for operation in work_plan.operations
    ):
        preflight_issues = preflight_runner(runner)
        if preflight_issues:
            return _progress_failure(destination, "runner_failed", preflight_issues, {})

    notes: dict[str, tuple[ScoreNote, ...]] = {}
    responses: dict[str, Mapping[str, object]] = {}
    local_ledger: list[ProjectionLedgerEntry] = []
    group_hashes: dict[str, str] = {}
    new_operations = 0
    repair_count = 0
    for operation in work_plan.operations:
        was_accepted = (destination / "events" / operation.operation_id / "accepted.json").is_file()
        if (
            not was_accepted
            and max_new_operations is not None
            and new_operations >= max_new_operations
        ):
            state = _progress_state(spec, notes, responses, group_hashes, "paused")
            atomic_write_json(destination / "progress/score-state.json", state)
            return ScoreRealizationResult(False, "paused", destination, state, ())
        comparisons = operation_comparisons(operation, work_plan)
        prompt, schema, immutable_input = score_operation_request(
            request.validated_script, phase3, operation, comparisons, notes, responses, spec
        )
        evaluated: list[ScoreGroupCandidateResult] = []

        def validate(
            response: Mapping[str, object],
            *,
            operation=operation,
            comparisons=comparisons,
            evaluated=evaluated,
        ):
            candidate = evaluate_score_group_candidate(
                request.validated_script,
                phase3.plan,
                operation,
                comparisons,
                response,
                harmonies_by_score_unit=phase3.harmonies_by_score_unit,
                accepted_notes_by_placement=notes,
                search_limit=SCORE_GROUP_SEARCH_LIMIT,
            )
            evaluated[:] = [candidate]
            return candidate.issues

        result = execute_finite_model_operation(
            store=store,
            operation_id=operation.operation_id,
            prompt=prompt,
            schema=schema,
            immutable_input=immutable_input,
            runner=runner,
            validate_content=validate,
        )
        repair_count += int(result.repaired)
        if result.response is None:
            state = _progress_state(spec, notes, responses, group_hashes, result.outcome)
            return _progress_failure(destination, result.outcome, result.issues, state)
        if not evaluated:
            raise ValueError("the accepted score response was not evaluated")
        candidate = evaluated[0]
        snapshot = score_operation_snapshot(operation, candidate, prompt, schema, immutable_input)
        store.snapshot_json(f"groups/{operation.operation_id}.json", snapshot)
        group_hashes[operation.operation_id] = sha256_json(snapshot)
        if set(notes).intersection(candidate.notes_by_placement):
            raise ValueError("score work groups contain repeated placements")
        notes.update(candidate.notes_by_placement)
        responses.update(candidate.responses_by_placement)
        local_ledger.extend(candidate.projection_ledger)
        new_operations += int(not was_accepted)
        atomic_write_json(
            destination / "progress/score-state.json",
            _progress_state(spec, notes, responses, group_hashes, "running"),
        )

    temporary = Path(tempfile.mkdtemp(prefix=".outputs-", dir=destination)).resolve()
    try:
        score, cumulative = assemble_score(request.validated_script, phase3, notes, local_ledger)
        atomic_write_json(temporary / "score-spec.json", asdict(score))
        atomic_write_json(
            temporary / "projection-ledger.json", [asdict(entry) for entry in cumulative]
        )
        script = cast(Mapping[str, object], request.validated_script["script"])
        placements = cast(Mapping[str, Mapping[str, object]], script["material_placements"])
        setup = cast(Mapping[str, object], script["performance_setup"])
        for filename, only_foreground in (
            ("foreground-preview.mid", True),
            ("score-preview.mid", False),
        ):
            segments = tuple(
                NeutralPreviewSegment(
                    unit.length_units,
                    tuple(
                        note
                        for layer in unit.score_unit_layers
                        if not only_foreground
                        or placements[layer.source_material_placement_id]["role"] == "foreground"
                        for note in layer.notes
                    ),
                )
                for unit in score.score_units
            )
            write_neutral_score_preview(
                segments,
                float(cast(int, setup["target_duration_seconds"])),
                temporary / filename,
                meta_track_name="SCoIM score",
                note_track_name="Foreground" if only_foreground else "Score",
                timing_contract=phase3.plan.timing_contract,
            )
        state = _progress_state(spec, notes, responses, group_hashes, "complete")
        state.update(
            {
                field: {"path": f"outputs/{filename}", "sha256": sha256_file(temporary / filename)}
                for field, filename in (
                    ("score_spec", "score-spec.json"),
                    ("projection_ledger", "projection-ledger.json"),
                    ("foreground_preview", "foreground-preview.mid"),
                    ("score_preview", "score-preview.mid"),
                )
            }
        )
        atomic_write_json(temporary / "score-state.json", state)
        atomic_write_json(
            temporary / "realization.json",
            {
                "outcome": "complete",
                "call_count": store.call_attempt_count,
                "repair_count": repair_count,
                "operation_order": spec["operation_order"],
                "issues": [],
            },
        )
        load_complete_score_run(destination, outputs_dir=temporary)
        temporary.replace(destination / "outputs")
        temporary = None
        return ScoreRealizationResult(True, "complete", destination, state, ())
    except (KeyError, TypeError, ValueError) as error:
        issue = ValidationIssue(IssueCode.SEMANTIC_INVALID, str(error), "/score_outputs")
        state = _progress_state(spec, notes, responses, group_hashes, "persisted_state_invalid")
        return _progress_failure(destination, "persisted_state_invalid", (issue,), state)
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def _progress_state(
    spec: Mapping[str, object],
    notes: Mapping[str, tuple[ScoreNote, ...]],
    responses: Mapping[str, Mapping[str, object]],
    group_hashes: Mapping[str, str],
    outcome: str,
) -> dict[str, object]:
    return {
        "schema_version": SCORE_STATE_SCHEMA_VERSION,
        "outcome": outcome,
        "run_spec_sha256": sha256_json(spec),
        "notes_by_material_placement": serialized_notes(notes),
        "responses_by_material_placement": dict(responses),
        "group_sha256": dict(group_hashes),
    }


def _progress_failure(
    destination: Path,
    outcome: str,
    issues: tuple[ValidationIssue, ...],
    state: dict[str, object],
) -> ScoreRealizationResult:
    atomic_write_json(destination / "progress/score-state.json", state)
    atomic_write_json(
        destination / "progress/realization.json",
        {
            "outcome": outcome,
            "issues": [
                {"code": issue.code.value, "message": issue.message, "path": issue.path}
                for issue in issues
            ],
        },
    )
    return ScoreRealizationResult(False, outcome, destination, state, issues)
