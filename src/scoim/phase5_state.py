"""Load and validate the complete phase-5 boundary state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import cast

from llm_musical_composer.run_state import sha256_json

from .phase4_state import LoadedPhase4State, load_complete_phase4_state
from .phase5_model_contracts import (
    AccompanimentOperation,
    build_accompaniment_operations,
    check_accompaniment_response,
)
from .phase5_pitch_placement import Phase5PitchPlacementResult, place_accompaniment_events
from .projection_ledger import ProjectionLedgerEntry, validate_projection_ledger
from .score_ir import ScoreNote


@dataclass(frozen=True, slots=True)
class LoadedPhase5State:
    """Typed accompaniment values proven to belong to the supplied phase-4 state."""

    phase4: LoadedPhase4State
    requests_by_material_placement: dict[str, dict[str, object]]
    notes_by_material_placement: dict[str, tuple[ScoreNote, ...]]
    local_projection_ledger: tuple[ProjectionLedgerEntry, ...]
    cumulative_projection_ledger: tuple[ProjectionLedgerEntry, ...]


def load_complete_phase5_state(
    validated_script: Mapping[str, object],
    phase3_state: Mapping[str, object],
    phase4_state: Mapping[str, object],
    phase5_state: Mapping[str, object],
    projection_ledger: Sequence[ProjectionLedgerEntry | Mapping[str, object]],
    *,
    target_profile: str = "solo_piano_3m_v2",
) -> LoadedPhase5State:
    """Return phase-5 accompaniments after checking the completed boundary."""
    if target_profile != "solo_piano_3m_v2":
        raise ValueError("the phase-6 profile is unsupported")
    if phase5_state.get("outcome") != "complete":
        raise ValueError("phase 6 requires a complete phase-5 state")
    if phase5_state.get("input_script_sha256") != sha256_json(validated_script):
        raise ValueError("the phase-5 state does not match the validated script")
    if phase5_state.get("input_phase3_state_sha256") != sha256_json(phase3_state):
        raise ValueError("the phase-5 state does not match the phase-3 state")
    if phase5_state.get("input_phase4_state_sha256") != sha256_json(phase4_state):
        raise ValueError("the phase-5 state does not match the phase-4 state")

    cumulative = tuple(_ledger_entry(value) for value in projection_ledger)
    local = tuple(
        _ledger_entry(value)
        for value in cast(list[Mapping[str, object]], phase5_state["projection_ledger"])
    )
    validate_projection_ledger(cumulative)
    validate_projection_ledger(local)
    if len(local) > len(cumulative) or (local and cumulative[-len(local) :] != local):
        raise ValueError("phase-5 state does not match the cumulative projection ledger")
    phase4_cumulative = cumulative[: -len(local)] if local else cumulative
    if phase5_state.get("input_projection_ledger_sha256") != sha256_json(
        [asdict(entry) for entry in phase4_cumulative]
    ):
        raise ValueError("the phase-5 input ledger does not match the cumulative projection ledger")
    loaded_phase4 = load_complete_phase4_state(
        validated_script,
        phase3_state,
        phase4_state,
        phase4_cumulative,
        target_profile=target_profile,
    )

    raw_requests = cast(
        Mapping[str, Mapping[str, object]],
        phase5_state["requests_by_material_placement"],
    )
    raw_notes = cast(
        Mapping[str, list[Mapping[str, object]]],
        phase5_state["notes_by_material_placement"],
    )
    operations = build_accompaniment_operations(validated_script)
    expected_placement_ids = {operation.material_placement_id for operation in operations}
    if set(raw_requests) != expected_placement_ids or set(raw_notes) != expected_placement_ids:
        raise ValueError("phase-5 values do not exactly cover accompaniment placements")
    requests = {placement_id: dict(response) for placement_id, response in raw_requests.items()}
    notes = {
        placement_id: tuple(_score_note(value) for value in values)
        for placement_id, values in raw_notes.items()
    }
    script = cast(Mapping[str, object], validated_script["script"])
    placements = cast(Mapping[str, Mapping[str, object]], script["material_placements"])
    existing_by_unit: dict[str, list[ScoreNote]] = {}
    for placement_id, foreground_notes in loaded_phase4.notes_by_material_placement.items():
        score_unit_id = f"score-unit-{placements[placement_id]['section_id']}"
        existing_by_unit.setdefault(score_unit_id, []).extend(foreground_notes)
    expected_local: list[ProjectionLedgerEntry] = []
    for operation in operations:
        response = requests[operation.material_placement_id]
        length_units = loaded_phase4.phase3.plan.length_units_by_score_unit[operation.score_unit_id]
        harmonies = loaded_phase4.phase3.harmonies_by_score_unit[operation.score_unit_id]
        issues = check_accompaniment_response(
            response,
            length_units=length_units,
            harmonies=harmonies,
        )
        if issues:
            raise ValueError(issues[0].message)
        placed = place_accompaniment_events(
            operation.material_placement_id,
            operation.score_unit_id,
            response,
            harmonies,
            existing_notes_by_score_unit={
                unit_id: tuple(unit_notes) for unit_id, unit_notes in existing_by_unit.items()
            },
        )
        if placed.notes != notes[operation.material_placement_id]:
            raise ValueError("phase-5 state does not match reconstructed accompaniment notes")
        expected_local.extend(accompaniment_projection_entries(operation, placed))
        existing_by_unit.setdefault(operation.score_unit_id, []).extend(placed.notes)
    if local != tuple(expected_local):
        raise ValueError("phase-5 projection ledger does not match reconstructed values")
    return LoadedPhase5State(loaded_phase4, requests, notes, local, cumulative)


def accompaniment_projection_entries(
    operation: AccompanimentOperation,
    placed: Phase5PitchPlacementResult,
) -> tuple[ProjectionLedgerEntry, ...]:
    """Describe one deterministic phase-5 placement in the projection ledger."""
    layer_id = f"score-unit-layer-{operation.material_placement_id}"
    return (
        ProjectionLedgerEntry(
            "material_placement",
            operation.material_placement_id,
            "score_unit_layer",
            layer_id,
            "accompaniment",
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
                "accompaniment",
                "deterministic_harmony_register_placement",
                "passed",
                (
                    f"score_unit_layer_id={layer_id}; at_units={note.at_units}; "
                    f"duration_units={note.duration_units}; "
                    f"evaluated_candidates={placed.evaluated_candidate_count}"
                ),
            )
            for note in placed.notes
        ),
    )


def _ledger_entry(
    value: ProjectionLedgerEntry | Mapping[str, object],
) -> ProjectionLedgerEntry:
    if isinstance(value, ProjectionLedgerEntry):
        return value
    return ProjectionLedgerEntry(**cast(dict[str, object], value))


def _score_note(value: Mapping[str, object]) -> ScoreNote:
    return ScoreNote(
        score_note_id=cast(str, value["score_note_id"]),
        at_units=cast(int, value["at_units"]),
        duration_units=cast(int, value["duration_units"]),
        pitch=cast(int, value["pitch"]),
        voice=cast(str, value["voice"]),
        tie=cast(str | None, value.get("tie")),
        articulations=tuple(cast(list[str], value.get("articulations", []))),
    )
