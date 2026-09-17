"""Recorded phase-5 realization of placement-specific accompaniment layers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from llm_musical_composer.run_state import RunStore, atomic_write_json, sha256_json

from .finite_model_operation import execute_finite_model_operation
from .phase4_state import LoadedPhase4State, load_complete_phase4_state
from .phase5_model_contracts import (
    AccompanimentOperation,
    accompaniment_prompt,
    accompaniment_response_schema,
    build_accompaniment_operations,
    check_accompaniment_response,
)
from .phase5_pitch_placement import (
    Phase5PitchPlacementError,
    place_accompaniment_events,
)
from .phase5_state import accompaniment_projection_entries
from .projection_ledger import (
    ProjectionLedgerEntry,
    ProjectionLedgerValidationError,
    validate_expected_projection_targets,
    validate_projection_ledger,
)
from .proposal import ProposalRunner, preflight_runner
from .score_ir import ScoreNote
from .validation import IssueCode, ValidationIssue


@dataclass(frozen=True, slots=True)
class Phase5Request:
    validated_script: Mapping[str, object]
    phase3_state: Mapping[str, object]
    phase4_state: Mapping[str, object]
    projection_ledger: Sequence[ProjectionLedgerEntry | Mapping[str, object]]
    target_profile: str = "solo_piano_3m_v2"


@dataclass(frozen=True, slots=True)
class Phase5RealizationResult:
    realized: bool
    outcome: str
    run_dir: Path | None
    state: dict[str, object] | None
    issues: tuple[ValidationIssue, ...]


def realize_phase5(
    request: Phase5Request,
    runner: ProposalRunner,
    run_dir: str | Path,
    *,
    max_new_operations: int | None = None,
) -> Phase5RealizationResult:
    """Build phase-5 accompaniments, stopping at the first failed operation."""
    if max_new_operations is not None and max_new_operations <= 0:
        raise ValueError("max_new_operations must be positive")
    try:
        loaded = load_complete_phase4_state(
            request.validated_script,
            request.phase3_state,
            request.phase4_state,
            request.projection_ledger,
            target_profile=request.target_profile,
        )
        operations = build_accompaniment_operations(request.validated_script)
    except (KeyError, TypeError, ValueError, ProjectionLedgerValidationError) as error:
        issue = ValidationIssue(IssueCode.SEMANTIC_INVALID, str(error), "/phase5_request")
        return Phase5RealizationResult(False, "request_invalid", None, None, (issue,))

    destination = Path(run_dir).resolve()
    input_ledger = loaded.cumulative_projection_ledger
    operation_order = [operation.operation_id for operation in operations]
    max_calls = 2 * len(operation_order)
    store = RunStore(destination, max_calls=max_calls)
    spec = {
        "schema_version": 1,
        "operation": "phase5-accompaniment-realization",
        "target_profile": request.target_profile,
        "input_script_sha256": sha256_json(request.validated_script),
        "input_phase3_state_sha256": sha256_json(request.phase3_state),
        "input_phase4_state_sha256": sha256_json(request.phase4_state),
        "input_projection_ledger_sha256": sha256_json([asdict(entry) for entry in input_ledger]),
        "operation_order": operation_order,
        "dependencies": {
            operation.operation_id: (
                [operation.comparison_source.source_material_placement_id]
                if operation.comparison_source is not None
                else []
            )
            for operation in operations
        },
        "max_calls": max_calls,
    }
    store.initialize(spec)
    store.snapshot_json("inputs/validated-script.json", dict(request.validated_script))
    store.snapshot_json("inputs/phase3-state.json", dict(request.phase3_state))
    store.snapshot_json("inputs/phase4-state.json", dict(request.phase4_state))
    store.snapshot_json("inputs/projection-ledger.json", [asdict(entry) for entry in input_ledger])

    preflight_issues = preflight_runner(runner)
    if preflight_issues:
        return _publish_failure(store, destination, "runner_failed", preflight_issues, None)

    accepted_responses: dict[str, dict[str, object]] = {}
    notes_by_placement: dict[str, tuple[ScoreNote, ...]] = {}
    phase5_ledger: list[ProjectionLedgerEntry] = []
    repair_count = 0
    new_operation_count = 0
    foreground_by_unit = _foreground_notes_by_score_unit(request.validated_script, loaded)
    existing_by_unit = {
        score_unit_id: list(notes) for score_unit_id, notes in foreground_by_unit.items()
    }
    for operation in operations:
        was_accepted = _operation_was_accepted(store, operation.operation_id)
        if (
            not was_accepted
            and max_new_operations is not None
            and new_operation_count >= max_new_operations
        ):
            state = _write_state(
                destination,
                accepted_responses,
                notes_by_placement,
                phase5_ledger,
                outcome="paused",
                request=request,
            )
            return Phase5RealizationResult(False, "paused", destination, state, ())
        unit_notes = {unit_id: tuple(notes) for unit_id, notes in existing_by_unit.items()}
        comparison_response: Mapping[str, object] | None = None
        if operation.comparison_source is not None:
            comparison_response = accepted_responses.get(
                operation.comparison_source.source_material_placement_id
            )
            if comparison_response is None:
                issue = ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    "an accompaniment comparison source has not been accepted",
                    "/phase5_request",
                )
                state = _write_state(
                    destination,
                    accepted_responses,
                    notes_by_placement,
                    phase5_ledger,
                    outcome="request_invalid",
                    request=request,
                )
                return _publish_failure(store, destination, "request_invalid", (issue,), state)
        foreground_notes = tuple(foreground_by_unit.get(operation.score_unit_id, ()))
        prompt = accompaniment_prompt(
            request.validated_script,
            loaded.phase3.plan,
            operation,
            harmonies_by_score_unit=loaded.phase3.harmonies_by_score_unit,
            foreground_notes=foreground_notes,
            accepted_responses=accepted_responses,
            accepted_notes=notes_by_placement,
        )
        immutable_input = {
            "script_sha256": spec["input_script_sha256"],
            "phase3_state_sha256": spec["input_phase3_state_sha256"],
            "phase4_state_sha256": spec["input_phase4_state_sha256"],
            "material_placement_id": operation.material_placement_id,
            "score_unit_id": operation.score_unit_id,
            "length_units": loaded.phase3.plan.length_units_by_score_unit[operation.score_unit_id],
            "foreground_notes_sha256": sha256_json([asdict(note) for note in foreground_notes]),
            "comparison_response_sha256": (
                sha256_json(comparison_response) if comparison_response is not None else None
            ),
        }

        def validate_content(
            response: Mapping[str, object],
            *,
            current_operation: AccompanimentOperation = operation,
            current_unit_notes: Mapping[str, tuple[ScoreNote, ...]] = unit_notes,
        ) -> tuple[ValidationIssue, ...]:
            issues = check_accompaniment_response(
                response,
                length_units=loaded.phase3.plan.length_units_by_score_unit[
                    current_operation.score_unit_id
                ],
                harmonies=loaded.phase3.harmonies_by_score_unit[current_operation.score_unit_id],
            )
            if issues:
                return issues
            try:
                candidate = place_accompaniment_events(
                    current_operation.material_placement_id,
                    current_operation.score_unit_id,
                    response,
                    loaded.phase3.harmonies_by_score_unit[current_operation.score_unit_id],
                    existing_notes_by_score_unit=current_unit_notes,
                )
            except Phase5PitchPlacementError as error:
                return (
                    ValidationIssue(
                        IssueCode.MODEL_OUTPUT_INVALID,
                        f"{error.code}: {error}",
                        "/events",
                    ),
                )
            if current_operation.comparison_source is not None:
                source_id = current_operation.comparison_source.source_material_placement_id
                if _normalized_notes(candidate.notes) == _normalized_notes(
                    notes_by_placement[source_id]
                ):
                    return (
                        ValidationIssue(
                            IssueCode.MODEL_OUTPUT_INVALID,
                            "A reused accompaniment must not be an exact realized copy",
                            "/events",
                        ),
                    )
            return ()

        result = execute_finite_model_operation(
            store=store,
            operation_id=operation.operation_id,
            prompt=prompt,
            schema=accompaniment_response_schema(),
            immutable_input=immutable_input,
            runner=runner,
            validate_content=validate_content,
        )
        repair_count += int(result.repaired)
        if result.response is None:
            state = _write_state(
                destination,
                accepted_responses,
                notes_by_placement,
                phase5_ledger,
                outcome=result.outcome,
                request=request,
            )
            return _publish_failure(store, destination, result.outcome, result.issues, state)
        new_operation_count += int(not was_accepted)
        placed = place_accompaniment_events(
            operation.material_placement_id,
            operation.score_unit_id,
            result.response,
            loaded.phase3.harmonies_by_score_unit[operation.score_unit_id],
            existing_notes_by_score_unit=unit_notes,
        )
        accepted_responses[operation.material_placement_id] = result.response
        notes_by_placement[operation.material_placement_id] = placed.notes
        existing_by_unit.setdefault(operation.score_unit_id, []).extend(placed.notes)
        phase5_ledger.extend(accompaniment_projection_entries(operation, placed))
        store.snapshot_json(
            f"outputs/requests/{operation.material_placement_id}.json", result.response
        )
        store.snapshot_json(
            f"outputs/notes/{operation.material_placement_id}.json",
            [asdict(note) for note in placed.notes],
        )
        _write_state(
            destination,
            accepted_responses,
            notes_by_placement,
            phase5_ledger,
            outcome="running",
            request=request,
        )

    validate_projection_ledger(tuple(phase5_ledger))
    validate_expected_projection_targets(
        tuple(phase5_ledger),
        {
            "score_unit_layer": frozenset(
                f"score-unit-layer-{operation.material_placement_id}" for operation in operations
            ),
            "score_note": frozenset(
                note.score_note_id for notes in notes_by_placement.values() for note in notes
            ),
        },
    )
    cumulative_ledger = (*input_ledger, *phase5_ledger)
    validate_projection_ledger(cumulative_ledger)
    store.snapshot_json(
        "outputs/projection-ledger.json", [asdict(entry) for entry in cumulative_ledger]
    )
    state = _write_state(
        destination,
        accepted_responses,
        notes_by_placement,
        phase5_ledger,
        outcome="complete",
        request=request,
    )
    summary = {
        "outcome": "complete",
        "call_count": store.call_attempt_count,
        "repair_count": repair_count,
        "operation_order": operation_order,
        "issues": [],
    }
    store.snapshot_json("outputs/realization.json", summary)
    store.record_step("publish-final", "completed", {}, summary)
    return Phase5RealizationResult(True, "complete", destination, state, ())


def _foreground_notes_by_score_unit(
    document: Mapping[str, object], loaded: LoadedPhase4State
) -> dict[str, list[ScoreNote]]:
    script = cast(Mapping[str, object], document["script"])
    placements = cast(Mapping[str, Mapping[str, object]], script["material_placements"])
    result: dict[str, list[ScoreNote]] = {}
    for placement_id, notes in loaded.notes_by_material_placement.items():
        score_unit_id = f"score-unit-{placements[placement_id]['section_id']}"
        result.setdefault(score_unit_id, []).extend(notes)
    return result


def _write_state(
    destination: Path,
    responses: Mapping[str, Mapping[str, object]],
    notes_by_placement: Mapping[str, tuple[ScoreNote, ...]],
    phase5_ledger: list[ProjectionLedgerEntry],
    *,
    outcome: str,
    request: Phase5Request,
) -> dict[str, object]:
    state = {
        "outcome": outcome,
        "input_script_sha256": sha256_json(request.validated_script),
        "input_phase3_state_sha256": sha256_json(request.phase3_state),
        "input_phase4_state_sha256": sha256_json(request.phase4_state),
        "input_projection_ledger_sha256": sha256_json(
            [
                asdict(entry) if isinstance(entry, ProjectionLedgerEntry) else dict(entry)
                for entry in request.projection_ledger
            ]
        ),
        "requests_by_material_placement": {
            placement_id: dict(response) for placement_id, response in responses.items()
        },
        "notes_by_material_placement": {
            placement_id: [asdict(note) for note in notes]
            for placement_id, notes in notes_by_placement.items()
        },
        "projection_ledger": [asdict(entry) for entry in phase5_ledger],
    }
    atomic_write_json(destination / "outputs" / "phase5-state.json", state)
    return state


def _publish_failure(
    store: RunStore,
    destination: Path,
    outcome: str,
    issues: tuple[ValidationIssue, ...],
    state: dict[str, object] | None,
) -> Phase5RealizationResult:
    summary = {
        "outcome": outcome,
        "call_count": store.call_attempt_count,
        "issues": [
            {"code": issue.code.value, "message": issue.message, "path": issue.path}
            for issue in issues
        ],
    }
    store.snapshot_json("outputs/realization.json", summary)
    store.record_step("publish-final", "failed", {}, summary)
    return Phase5RealizationResult(False, outcome, destination, state, issues)


def _operation_was_accepted(store: RunStore, operation_id: str) -> bool:
    return (store.run_dir / "events" / operation_id / "accepted.json").is_file()


def _normalized_notes(notes: tuple[ScoreNote, ...]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        sorted(
            (
                note.at_units,
                note.duration_units,
                note.pitch,
                note.voice,
                note.articulations,
            )
            for note in notes
        )
    )
