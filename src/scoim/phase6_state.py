"""Reload and verify a completed phase-6 run from persisted artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from llm_musical_composer.run_state import sha256_file, sha256_json

from .neutral_score_preview import NeutralPreviewSegment, check_neutral_score_preview
from .phase5_state import LoadedPhase5State, load_complete_phase5_state
from .projection_ledger import ProjectionLedgerEntry, validate_projection_ledger
from .score_ir import ScoreSpec, score_spec_from_json
from .score_projection import build_score_spec


@dataclass(frozen=True, slots=True)
class LoadedPhase6Run:
    """A persisted score proven to reconstruct from its saved upstream inputs."""

    validated_script: Mapping[str, object]
    phase5: LoadedPhase5State
    score: ScoreSpec
    cumulative_projection_ledger: tuple[ProjectionLedgerEntry, ...]


def load_complete_phase6_run(run_dir: str | Path) -> LoadedPhase6Run:
    """Reload one completed phase-6 directory without using process memory."""
    root = Path(run_dir).resolve()
    state = _read_json(root / "outputs" / "phase6-state.json")
    if state.get("outcome") != "complete":
        raise ValueError("phase 7 requires a complete phase-6 state")
    target_profile = cast(str, state["target_profile"])
    if target_profile != "solo_piano_3m_v2":
        raise ValueError("the saved phase-6 profile is unsupported")

    validated_script = _read_json(root / "inputs" / "validated-script.json")
    phase3_state = _read_json(root / "inputs" / "phase3-state.json")
    phase4_state = _read_json(root / "inputs" / "phase4-state.json")
    phase5_state = _read_json(root / "inputs" / "phase5-state.json")
    input_ledger_values = _read_json_list(root / "inputs" / "projection-ledger.json")
    input_ledger = tuple(_ledger_entry(value) for value in input_ledger_values)
    _require_hash(state, "input_script_sha256", validated_script)
    _require_hash(state, "input_phase3_state_sha256", phase3_state)
    _require_hash(state, "input_phase4_state_sha256", phase4_state)
    _require_hash(state, "input_phase5_state_sha256", phase5_state)
    _require_hash(
        state,
        "input_projection_ledger_sha256",
        [asdict(entry) for entry in input_ledger],
    )

    score_path = root / "outputs" / "score-spec.json"
    ledger_path = root / "outputs" / "projection-ledger.json"
    preview_path = root / "outputs" / "score-preview.mid"
    _require_artifact(state, "score_spec", "outputs/score-spec.json", score_path)
    _require_artifact(
        state,
        "projection_ledger",
        "outputs/projection-ledger.json",
        ledger_path,
    )
    _require_artifact(state, "score_preview", "outputs/score-preview.mid", preview_path)

    loaded_phase5 = load_complete_phase5_state(
        validated_script,
        phase3_state,
        phase4_state,
        phase5_state,
        input_ledger,
        target_profile=target_profile,
    )
    notes_by_placement = dict(loaded_phase5.phase4.notes_by_material_placement)
    if not notes_by_placement.keys().isdisjoint(loaded_phase5.notes_by_material_placement):
        raise ValueError("phase-4 and phase-5 placement values overlap")
    notes_by_placement.update(loaded_phase5.notes_by_material_placement)
    harmonic_plan = loaded_phase5.phase4.phase3.plan
    expected_score, new_evidence = build_score_spec(
        validated_script,
        harmonic_plan.piece_plan,
        score_id=f"{harmonic_plan.piece_plan.plan_id}-score",
        divisions=harmonic_plan.divisions,
        length_units_by_score_unit=harmonic_plan.length_units_by_score_unit,
        harmonies_by_score_unit=loaded_phase5.phase4.phase3.harmonies_by_score_unit,
        directions_by_score_unit={
            unit_id: () for unit_id in harmonic_plan.length_units_by_score_unit
        },
        notes_by_material_placement=notes_by_placement,
        cumulative_projection_ledger=loaded_phase5.cumulative_projection_ledger,
    )
    saved_score = score_spec_from_json(_read_json(score_path))
    if saved_score != expected_score:
        raise ValueError("saved phase-6 score does not match reconstructed values")
    saved_ledger = tuple(_ledger_entry(value) for value in _read_json_list(ledger_path))
    expected_ledger = (*loaded_phase5.cumulative_projection_ledger, *new_evidence)
    if saved_ledger != expected_ledger:
        raise ValueError("saved phase-6 projection ledger does not match reconstructed values")
    validate_projection_ledger(saved_ledger)

    setup = cast(Mapping[str, object], validated_script["script"])["performance_setup"]
    target_seconds = float(cast(Mapping[str, object], setup)["target_duration_seconds"])
    check_neutral_score_preview(
        tuple(
            NeutralPreviewSegment(
                unit.length_units,
                tuple(note for layer in unit.score_unit_layers for note in layer.notes),
            )
            for unit in saved_score.score_units
        ),
        target_seconds,
        preview_path,
    )
    return LoadedPhase6Run(validated_script, loaded_phase5, saved_score, saved_ledger)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, object], value)


def _read_json_list(path: Path) -> list[Mapping[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected a JSON array: {path}")
    return cast(list[Mapping[str, object]], value)


def _require_hash(state: Mapping[str, object], field: str, value: object) -> None:
    if state.get(field) != sha256_json(value):
        raise ValueError(f"saved phase-6 input hash does not match: {field}")


def _require_artifact(
    state: Mapping[str, object],
    field: str,
    expected_path: str,
    path: Path,
) -> None:
    descriptor = cast(Mapping[str, object], state[field])
    if descriptor.get("path") != expected_path:
        raise ValueError(f"saved phase-6 artifact path does not match: {field}")
    if descriptor.get("sha256") != sha256_file(path):
        raise ValueError(f"saved phase-6 artifact hash does not match: {field}")


def _ledger_entry(value: Mapping[str, object]) -> ProjectionLedgerEntry:
    return ProjectionLedgerEntry(**cast(dict[str, object], value))
