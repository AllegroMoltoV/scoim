"""Load and validate the complete phase-4 boundary state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import cast

from llm_musical_composer.run_state import sha256_json

from .phase3_state import LoadedPhase3State, load_complete_phase3_state
from .phase4_model_contracts import build_foreground_operations
from .projection_ledger import (
    ProjectionLedgerEntry,
    validate_expected_projection_targets,
    validate_projection_ledger,
)
from .score_ir import ScoreNote


@dataclass(frozen=True, slots=True)
class LoadedPhase4State:
    """Typed foreground values proven to belong to the supplied phase-3 state."""

    phase3: LoadedPhase3State
    notes_by_material_placement: dict[str, tuple[ScoreNote, ...]]
    local_projection_ledger: tuple[ProjectionLedgerEntry, ...]
    cumulative_projection_ledger: tuple[ProjectionLedgerEntry, ...]


def load_complete_phase4_state(
    validated_script: Mapping[str, object],
    phase3_state: Mapping[str, object],
    phase4_state: Mapping[str, object],
    projection_ledger: Sequence[ProjectionLedgerEntry | Mapping[str, object]],
    *,
    target_profile: str = "solo_piano_3m_v2",
) -> LoadedPhase4State:
    """Return phase-4 foregrounds after checking completeness and lineage."""
    if target_profile != "solo_piano_3m_v2":
        raise ValueError("the phase-5 profile is unsupported")
    if phase4_state.get("outcome") != "complete":
        raise ValueError("phase 5 requires a complete phase-4 state")
    if phase4_state.get("input_script_sha256") != sha256_json(validated_script):
        raise ValueError("the phase-4 state does not match the validated script")
    if phase4_state.get("input_phase3_state_sha256") != sha256_json(phase3_state):
        raise ValueError("the phase-4 state does not match the phase-3 state")

    cumulative = tuple(_ledger_entry(value) for value in projection_ledger)
    local = tuple(
        _ledger_entry(value)
        for value in cast(list[Mapping[str, object]], phase4_state["projection_ledger"])
    )
    validate_projection_ledger(cumulative)
    validate_projection_ledger(local)
    if not local or len(local) > len(cumulative) or cumulative[-len(local) :] != local:
        raise ValueError("phase-4 state does not match the cumulative projection ledger")
    phase3_cumulative = cumulative[: -len(local)]
    if phase4_state.get("input_projection_ledger_sha256") != sha256_json(
        [asdict(entry) for entry in phase3_cumulative]
    ):
        raise ValueError("the phase-4 input ledger does not match the cumulative projection ledger")
    loaded_phase3 = load_complete_phase3_state(
        validated_script,
        phase3_state,
        phase3_cumulative,
        target_profile=target_profile,
    )

    script = cast(Mapping[str, object], validated_script["script"])
    placements = cast(Mapping[str, Mapping[str, object]], script["material_placements"])
    expected_placement_ids = {
        placement_id
        for placement_id, placement in placements.items()
        if placement["role"] == "foreground"
    }
    raw_notes = cast(
        Mapping[str, list[Mapping[str, object]]], phase4_state["notes_by_material_placement"]
    )
    if set(raw_notes) != expected_placement_ids:
        raise ValueError("phase-4 foregrounds do not exactly cover foreground placements")

    notes_by_placement: dict[str, tuple[ScoreNote, ...]] = {}
    note_ids: set[str] = set()
    for placement_id in sorted(raw_notes):
        placement = placements[placement_id]
        score_unit_id = f"score-unit-{placement['section_id']}"
        length_units = loaded_phase3.plan.length_units_by_score_unit[score_unit_id]
        notes = tuple(_score_note(value) for value in raw_notes[placement_id])
        if not notes:
            raise ValueError("a phase-4 foreground placement has no notes")
        for note in notes:
            if (
                note.at_units < 0
                or note.duration_units <= 0
                or note.at_units + note.duration_units > length_units
            ):
                raise ValueError("a phase-4 foreground note exceeds its score unit")
            if not 21 <= note.pitch <= 108:
                raise ValueError("a phase-4 foreground note is outside the piano range")
            if note.voice not in {"upper", "lower"}:
                raise ValueError("a phase-4 foreground note has an unsupported voice")
            if note.score_note_id in note_ids:
                raise ValueError("phase-4 foreground note IDs must be globally unique")
            note_ids.add(note.score_note_id)
        notes_by_placement[placement_id] = notes

    validate_expected_projection_targets(
        local,
        {
            "score_unit_layer": frozenset(
                f"score-unit-layer-{placement_id}" for placement_id in expected_placement_ids
            ),
            "score_note": frozenset(note_ids),
        },
    )
    expected_local = tuple(
        entry
        for operation in build_foreground_operations(validated_script)
        for entry in _expected_projection_entries(
            operation.material_placement_id,
            notes_by_placement[operation.material_placement_id],
        )
    )
    if local != expected_local:
        raise ValueError("phase-4 projection ledger does not match the foreground state")
    return LoadedPhase4State(loaded_phase3, notes_by_placement, local, cumulative)


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


def _expected_projection_entries(
    placement_id: str, notes: tuple[ScoreNote, ...]
) -> tuple[ProjectionLedgerEntry, ...]:
    layer_id = f"score-unit-layer-{placement_id}"
    return (
        ProjectionLedgerEntry(
            "material_placement",
            placement_id,
            "score_unit_layer",
            layer_id,
            "foreground",
            "direct_id_equality",
            "passed",
            f"material_placement_id={placement_id}; score_unit_layer_id={layer_id}",
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
