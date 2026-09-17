"""Recorded phase-4 realization of placement-specific foreground layers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import cast

from llm_musical_composer.run_state import RunStore, atomic_write_json, sha256_json

from .finite_model_operation import execute_finite_model_operation
from .phase3_state import load_complete_phase3_state
from .phase4_model_contracts import (
    ForegroundOperation,
    build_foreground_notes,
    build_foreground_operations,
    check_foreground_response,
    foreground_prompt,
    foreground_response_schema,
)
from .phase4_preview import write_phase4_foreground_preview
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
class Phase4Request:
    validated_script: Mapping[str, object]
    phase3_state: Mapping[str, object]
    projection_ledger: Sequence[ProjectionLedgerEntry | Mapping[str, object]]
    target_profile: str = "solo_piano_3m_v2"


@dataclass(frozen=True, slots=True)
class Phase4RealizationResult:
    realized: bool
    outcome: str
    run_dir: Path | None
    state: dict[str, object] | None
    issues: tuple[ValidationIssue, ...]


def realize_phase4(
    request: Phase4Request,
    runner: ProposalRunner,
    run_dir: str | Path,
    *,
    max_new_operations: int | None = None,
) -> Phase4RealizationResult:
    """Build phase-4 foregrounds, stopping at the first failed operation."""
    if max_new_operations is not None and max_new_operations <= 0:
        raise ValueError("max_new_operations must be positive")
    try:
        loaded = load_complete_phase3_state(
            request.validated_script,
            request.phase3_state,
            request.projection_ledger,
            target_profile=request.target_profile,
        )
        plan = loaded.plan
        harmonies_by_score_unit = loaded.harmonies_by_score_unit
        input_ledger = loaded.cumulative_projection_ledger
        operations = build_foreground_operations(request.validated_script)
    except (KeyError, TypeError, ValueError, ProjectionLedgerValidationError) as error:
        issue = ValidationIssue(IssueCode.SEMANTIC_INVALID, str(error), "/phase4_request")
        return Phase4RealizationResult(False, "request_invalid", None, None, (issue,))

    destination = Path(run_dir).resolve()
    operation_order = [operation.operation_id for operation in operations]
    max_calls = 2 * len(operation_order)
    store = RunStore(destination, max_calls=max_calls)
    spec = {
        "schema_version": 1,
        "operation": "phase4-foreground-realization",
        "target_profile": request.target_profile,
        "input_script_sha256": sha256_json(request.validated_script),
        "input_phase3_state_sha256": sha256_json(request.phase3_state),
        "input_projection_ledger_sha256": sha256_json([asdict(entry) for entry in input_ledger]),
        "operation_order": operation_order,
        "dependencies": {
            operation.operation_id: [
                source.source_material_placement_id for source in operation.comparison_sources
            ]
            for operation in operations
        },
        "max_calls": max_calls,
    }
    store.initialize(spec)
    store.snapshot_json("inputs/validated-script.json", dict(request.validated_script))
    store.snapshot_json("inputs/phase3-state.json", dict(request.phase3_state))
    store.snapshot_json("inputs/projection-ledger.json", [asdict(entry) for entry in input_ledger])

    preflight_issues = preflight_runner(runner)
    if preflight_issues:
        return _publish_failure(store, destination, "runner_failed", preflight_issues, None)

    notes_by_placement: dict[str, tuple[ScoreNote, ...]] = {}
    phase4_ledger: list[ProjectionLedgerEntry] = []
    repair_count = 0
    new_operation_count = 0
    placements = cast(
        Mapping[str, Mapping[str, object]],
        cast(Mapping[str, object], request.validated_script["script"])["material_placements"],
    )
    for operation in operations:
        was_accepted = _operation_was_accepted(store, operation.operation_id)
        if (
            not was_accepted
            and max_new_operations is not None
            and new_operation_count >= max_new_operations
        ):
            state = _write_state(
                destination,
                notes_by_placement,
                phase4_ledger,
                outcome="paused",
                request=request,
            )
            return Phase4RealizationResult(False, "paused", destination, state, ())
        copy_comparison_notes = tuple(
            notes_by_placement[source.source_material_placement_id]
            for source in operation.comparison_sources
            if source.kind
            in {"implicit_reuse", "material_variation", "material_placement_variation"}
        )
        existing_unit_notes = tuple(
            sorted(
                (
                    note
                    for placement_id, notes in notes_by_placement.items()
                    if placements[placement_id]["section_id"] == operation.section_id
                    for note in notes
                ),
                key=lambda note: (
                    note.at_units,
                    note.duration_units,
                    note.pitch,
                    note.voice,
                    note.score_note_id,
                ),
            )
        )
        prompt = foreground_prompt(
            request.validated_script,
            plan,
            operation,
            harmonies_by_score_unit=harmonies_by_score_unit,
            accepted_notes_by_material_placement=notes_by_placement,
            existing_unit_notes=existing_unit_notes,
        )
        length_units = plan.length_units_by_score_unit[operation.score_unit_id]
        result = execute_finite_model_operation(
            store=store,
            operation_id=operation.operation_id,
            prompt=prompt,
            schema=foreground_response_schema(),
            immutable_input={
                "script_sha256": spec["input_script_sha256"],
                "phase3_state_sha256": spec["input_phase3_state_sha256"],
                "material_placement_id": operation.material_placement_id,
                "score_unit_id": operation.score_unit_id,
                "length_units": length_units,
                "comparison_sources": [
                    {
                        "kind": source.kind,
                        "material_placement_id": source.source_material_placement_id,
                        "notes_sha256": sha256_json(
                            [
                                asdict(note)
                                for note in notes_by_placement[source.source_material_placement_id]
                            ]
                        ),
                    }
                    for source in operation.comparison_sources
                ],
                "existing_unit_notes_sha256": sha256_json(
                    [asdict(note) for note in existing_unit_notes]
                ),
            },
            runner=runner,
            validate_content=partial(
                check_foreground_response,
                length_units=length_units,
                comparison_notes=copy_comparison_notes,
                existing_unit_notes=existing_unit_notes,
            ),
        )
        repair_count += int(result.repaired)
        if result.response is None:
            state = _write_state(
                destination,
                notes_by_placement,
                phase4_ledger,
                outcome=result.outcome,
                request=request,
            )
            return _publish_failure(store, destination, result.outcome, result.issues, state)
        new_operation_count += int(not was_accepted)
        notes = build_foreground_notes(operation.material_placement_id, result.response)
        notes_by_placement[operation.material_placement_id] = notes
        entries = _foreground_projection_entries(operation, notes)
        phase4_ledger.extend(entries)
        store.snapshot_json(
            f"outputs/notes/{operation.material_placement_id}.json",
            [asdict(note) for note in notes],
        )
        _write_state(
            destination,
            notes_by_placement,
            phase4_ledger,
            outcome="running",
            request=request,
        )

    validate_projection_ledger(tuple(phase4_ledger))
    validate_expected_projection_targets(
        tuple(phase4_ledger),
        {
            "score_unit_layer": frozenset(
                f"score-unit-layer-{operation.material_placement_id}" for operation in operations
            ),
            "score_note": frozenset(
                note.score_note_id for notes in notes_by_placement.values() for note in notes
            ),
        },
    )
    cumulative_ledger = (*input_ledger, *phase4_ledger)
    validate_projection_ledger(cumulative_ledger)
    write_phase4_foreground_preview(
        request.validated_script,
        plan,
        notes_by_placement,
        destination / "outputs" / "foreground-preview.mid",
    )
    store.snapshot_json(
        "outputs/projection-ledger.json", [asdict(entry) for entry in cumulative_ledger]
    )
    state = _write_state(
        destination,
        notes_by_placement,
        phase4_ledger,
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
    return Phase4RealizationResult(True, "complete", destination, state, ())


def _foreground_projection_entries(
    operation: ForegroundOperation, notes: tuple[ScoreNote, ...]
) -> tuple[ProjectionLedgerEntry, ...]:
    layer_id = f"score-unit-layer-{operation.material_placement_id}"
    return (
        ProjectionLedgerEntry(
            "material_placement",
            operation.material_placement_id,
            "score_unit_layer",
            layer_id,
            "foreground",
            "direct_id_equality",
            "passed",
            (
                f"material_placement_id={operation.material_placement_id}; "
                f"score_unit_layer_id={layer_id}"
            ),
        ),
        *(
            ProjectionLedgerEntry(
                "score_unit_layer",
                layer_id,
                "score_note",
                note.score_note_id,
                "foreground",
                "time_bounds_and_source_equality",
                "passed",
                (
                    f"score_unit_layer_id={layer_id}; at_units={note.at_units}; "
                    f"duration_units={note.duration_units}"
                ),
            )
            for note in notes
        ),
    )


def _write_state(
    destination: Path,
    notes_by_placement: Mapping[str, tuple[ScoreNote, ...]],
    phase4_ledger: list[ProjectionLedgerEntry],
    *,
    outcome: str,
    request: Phase4Request,
) -> dict[str, object]:
    state = {
        "outcome": outcome,
        "input_script_sha256": sha256_json(request.validated_script),
        "input_phase3_state_sha256": sha256_json(request.phase3_state),
        "input_projection_ledger_sha256": sha256_json(
            [
                asdict(entry) if isinstance(entry, ProjectionLedgerEntry) else dict(entry)
                for entry in request.projection_ledger
            ]
        ),
        "notes_by_material_placement": {
            placement_id: [asdict(note) for note in notes]
            for placement_id, notes in notes_by_placement.items()
        },
        "projection_ledger": [asdict(entry) for entry in phase4_ledger],
    }
    atomic_write_json(destination / "outputs" / "phase4-state.json", state)
    return state


def _publish_failure(
    store: RunStore,
    destination: Path,
    outcome: str,
    issues: tuple[ValidationIssue, ...],
    state: dict[str, object] | None,
) -> Phase4RealizationResult:
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
    return Phase4RealizationResult(False, outcome, destination, state, issues)


def _operation_was_accepted(store: RunStore, operation_id: str) -> bool:
    return (store.run_dir / "events" / operation_id / "accepted.json").is_file()
