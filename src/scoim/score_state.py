"""Reconstruct the complete joint score boundary from accepted operations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from llm_musical_composer.run_state import (
    RunStore,
    StateConflictError,
    sha256_file,
    sha256_json,
    sha256_text,
)

from .finite_model_operation import (
    check_finite_model_operation_records,
    read_accepted_model_operation,
)
from .neutral_score_preview import NeutralPreviewSegment, check_neutral_score_preview
from .phase3_state import LoadedPhase3State, load_complete_phase3_state
from .projection_ledger import ProjectionLedgerEntry, validate_projection_ledger
from .score_generation_context import SECTION_RANGE_CONTEXT_V1
from .score_group_pitch_placement import SCORE_GROUP_SEARCH_LIMIT
from .score_ir import ScoreNote, ScoreSpec, score_spec_from_json
from .score_model_contracts import (
    ScoreGroupCandidateResult,
    evaluate_score_group_candidate,
    score_group_prompt,
    score_group_response_schema,
)
from .score_projection import build_score_spec
from .score_timing import QUANTIZED_TIMING
from .score_work_plan import (
    ScoreComparison,
    ScoreWorkOperation,
    ScoreWorkPlan,
    build_score_work_plan,
)
from .terminal_boundary import (
    SHARED_TERMINAL,
    TerminalBoundary,
    derive_score_terminal_boundary,
    validate_score_terminal_boundary,
)

SCORE_STATE_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class LoadedScoreRun:
    validated_script: Mapping[str, object]
    phase3: LoadedPhase3State
    score: ScoreSpec
    notes_by_material_placement: Mapping[str, tuple[ScoreNote, ...]]
    cumulative_projection_ledger: tuple[ProjectionLedgerEntry, ...]
    terminal_boundary: TerminalBoundary | None = None
    schema_version: int = SCORE_STATE_SCHEMA_VERSION


def score_run_spec(
    document: Mapping[str, object],
    phase3_state: Mapping[str, object],
    input_ledger: Sequence[ProjectionLedgerEntry],
    work_plan: ScoreWorkPlan,
    *,
    schema_version: int = SCORE_STATE_SCHEMA_VERSION,
) -> dict[str, object]:
    """Bind the immutable schedule and its musical inputs to this saved version."""
    return {
        "schema_version": schema_version,
        **({"terminal_contract": SHARED_TERMINAL} if schema_version >= 2 else {}),
        "operation": "score-realization",
        "target_profile": "solo_piano_3m_v2",
        "timing_contract": QUANTIZED_TIMING,
        "generation_context_contract": SECTION_RANGE_CONTEXT_V1,
        "input_script_sha256": sha256_json(document),
        "input_phase3_state_sha256": sha256_json(phase3_state),
        "input_projection_ledger_sha256": sha256_json([asdict(x) for x in input_ledger]),
        "operations": [asdict(operation) for operation in work_plan.operations],
        "comparisons": [asdict(comparison) for comparison in work_plan.comparisons],
        "operation_order": [operation.operation_id for operation in work_plan.operations],
        "max_calls": 2 * len(work_plan.operations),
        "content_repair_limit": 1,
        "pitch_search_limit": SCORE_GROUP_SEARCH_LIMIT,
    }


def operation_comparisons(
    operation: ScoreWorkOperation, work_plan: ScoreWorkPlan
) -> tuple[ScoreComparison, ...]:
    targets = set(operation.material_placement_ids)
    return tuple(
        comparison
        for comparison in work_plan.comparisons
        if targets.intersection(comparison.target_placement_ids)
    )


def score_operation_request(
    document: Mapping[str, object],
    phase3: LoadedPhase3State,
    operation: ScoreWorkOperation,
    comparisons: tuple[ScoreComparison, ...],
    notes: Mapping[str, tuple[ScoreNote, ...]],
    responses: Mapping[str, Mapping[str, object]],
    spec: Mapping[str, object],
) -> tuple[str, dict[str, object], dict[str, object]]:
    prompt = score_group_prompt(
        document,
        phase3.plan,
        operation,
        comparisons,
        harmonies_by_score_unit=phase3.harmonies_by_score_unit,
        accepted_notes_by_placement=notes,
        accepted_responses_by_placement=responses,
        shared_terminal=spec["schema_version"] >= 2,
    )
    schema = score_group_response_schema(document, operation)
    immutable_input = {
        **({"terminal_contract": SHARED_TERMINAL} if spec["schema_version"] >= 2 else {}),
        "script_sha256": spec["input_script_sha256"],
        "phase3_state_sha256": spec["input_phase3_state_sha256"],
        "operation": asdict(operation),
        "comparisons": [asdict(comparison) for comparison in comparisons],
        "accepted_notes_sha256": sha256_json(serialized_notes(notes)),
        "accepted_responses_sha256": sha256_json(responses),
        "pitch_search_limit": SCORE_GROUP_SEARCH_LIMIT,
    }
    return prompt, schema, immutable_input


def score_operation_snapshot(
    operation: ScoreWorkOperation,
    candidate: ScoreGroupCandidateResult,
    prompt: str,
    schema: Mapping[str, object],
    immutable_input: Mapping[str, object],
) -> dict[str, object]:
    """Connect one whole accepted candidate to its exact delivered request."""
    return {
        "operation_id": operation.operation_id,
        **(
            {
                "terminal_boundary": asdict(candidate.terminal_boundary)
                if candidate.terminal_boundary
                else None
            }
            if "terminal_contract" in immutable_input
            else {}
        ),
        "material_placement_ids": list(operation.material_placement_ids),
        "prompt_sha256": sha256_text(prompt),
        "schema_sha256": sha256_json(schema),
        "immutable_input_sha256": sha256_json(immutable_input),
        "notes_by_material_placement": serialized_notes(candidate.notes_by_placement),
        "responses_by_material_placement": dict(candidate.responses_by_placement),
        "projection_ledger": [asdict(entry) for entry in candidate.projection_ledger],
        "comparison_evidence": list(candidate.comparison_evidence),
        "evaluated_candidate_count_by_score_unit": dict(
            candidate.evaluated_candidate_count_by_score_unit
        ),
    }


def assemble_score(
    document: Mapping[str, object],
    phase3: LoadedPhase3State,
    notes: Mapping[str, tuple[ScoreNote, ...]],
    local_ledger: Sequence[ProjectionLedgerEntry],
) -> tuple[ScoreSpec, tuple[ProjectionLedgerEntry, ...]]:
    plan = phase3.plan
    cumulative = (*phase3.cumulative_projection_ledger, *local_ledger)
    score, assembly_ledger = build_score_spec(
        document,
        plan.piece_plan,
        score_id=f"{plan.piece_plan.plan_id}-score",
        divisions=plan.divisions,
        timing_contract=plan.timing_contract,
        length_units_by_score_unit=plan.length_units_by_score_unit,
        harmonies_by_score_unit=phase3.harmonies_by_score_unit,
        directions_by_score_unit={unit_id: () for unit_id in plan.length_units_by_score_unit},
        notes_by_material_placement=notes,
        cumulative_projection_ledger=cumulative,
    )
    complete_ledger = (*cumulative, *assembly_ledger)
    validate_projection_ledger(complete_ledger)
    return score, complete_ledger


def load_complete_score_run(
    run_dir: str | Path, *, outputs_dir: Path | None = None
) -> LoadedScoreRun:
    """Read a saved run, or its staged final outputs, without any writes."""
    root = Path(run_dir).resolve()
    output = outputs_dir if outputs_dir is not None else root / "outputs"
    state = read_score_object(output / "score-state.json")
    schema_version = state.get("schema_version")
    if schema_version not in {1, 2}:
        raise ValueError("the saved score state schema is unsupported")
    if state.get("outcome") != "complete":
        raise ValueError("a complete score state is required")
    document = read_score_object(root / "inputs/validated-script.json")
    phase3_state = read_score_object(root / "inputs/phase3-state.json")
    input_ledger = read_score_ledger(root / "inputs/projection-ledger.json")
    phase3 = load_complete_phase3_state(document, phase3_state, input_ledger)
    if phase3.plan.generation_context_contract != SECTION_RANGE_CONTEXT_V1:
        raise ValueError("the saved score requires its section-range generation context contract")
    if phase3.plan.timing_contract != QUANTIZED_TIMING:
        raise ValueError("the saved score requires the quantized timing contract")
    work_plan = build_score_work_plan(document)
    if work_plan.issues:
        raise ValueError(work_plan.issues[0].message)
    spec = score_run_spec(
        document, phase3_state, input_ledger, work_plan, schema_version=cast(int, schema_version)
    )
    if sha256_json(read_score_object(root / "run-spec.json")) != sha256_json(spec):
        raise ValueError("the saved score run specification does not match its inputs")
    if state.get("run_spec_sha256") != sha256_json(spec):
        raise ValueError("the saved score state does not match its run specification")
    records = check_finite_model_operation_records(root)
    if not records.valid:
        raise ValueError(
            f"saved score model operation record is invalid: {records.issues[0].message}"
        )
    operation_ids = {operation.operation_id for operation in work_plan.operations}
    accepted_ids = {path.parent.name for path in (root / "events").glob("*/accepted.json")}
    if accepted_ids != operation_ids:
        raise ValueError("accepted score operations do not exactly cover the work plan")
    group_ids = {path.stem for path in (root / "groups").glob("*.json")}
    if group_ids != operation_ids:
        raise ValueError("saved score groups do not exactly cover the work plan")
    store = RunStore(root, max_calls=cast(int, spec["max_calls"]))
    notes: dict[str, tuple[ScoreNote, ...]] = {}
    responses: dict[str, Mapping[str, object]] = {}
    local_ledger: list[ProjectionLedgerEntry] = []
    group_hashes: dict[str, str] = {}
    boundary = None
    for operation in work_plan.operations:
        comparisons = operation_comparisons(operation, work_plan)
        prompt, schema, immutable_input = score_operation_request(
            document, phase3, operation, comparisons, notes, responses, spec
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
                document,
                phase3.plan,
                operation,
                comparisons,
                response,
                harmonies_by_score_unit=phase3.harmonies_by_score_unit,
                accepted_notes_by_placement=notes,
                search_limit=SCORE_GROUP_SEARCH_LIMIT,
                shared_terminal=schema_version >= 2,
            )
            evaluated[:] = [candidate]
            return candidate.issues

        try:
            accepted = read_accepted_model_operation(
                store, operation.operation_id, prompt, schema, immutable_input, validate
            )
        except StateConflictError as error:
            raise ValueError(str(error)) from error
        if accepted is None or not evaluated:
            raise ValueError("a saved score operation has no validated accepted response")
        candidate = evaluated[0]
        boundary = candidate.terminal_boundary or boundary
        snapshot = score_operation_snapshot(operation, candidate, prompt, schema, immutable_input)
        saved_snapshot = read_score_object(root / "groups" / f"{operation.operation_id}.json")
        if sha256_json(saved_snapshot) != sha256_json(snapshot):
            raise ValueError("saved score comparison evidence or candidate does not match")
        group_hashes[operation.operation_id] = sha256_json(snapshot)
        if set(notes).intersection(candidate.notes_by_placement):
            raise ValueError("score work groups contain repeated placements")
        notes.update(candidate.notes_by_placement)
        responses.update(candidate.responses_by_placement)
        local_ledger.extend(candidate.projection_ledger)
    if state.get("group_sha256") != group_hashes:
        raise ValueError("saved score group hashes do not match reconstructed evidence")
    if sha256_json(state.get("notes_by_material_placement")) != sha256_json(
        serialized_notes(notes)
    ):
        raise ValueError("saved score notes do not match accepted candidates")
    if state.get("responses_by_material_placement") != responses:
        raise ValueError("saved score responses do not match accepted candidates")
    score, cumulative = assemble_score(document, phase3, notes, local_ledger)
    if schema_version >= 2:
        reconstructed = derive_score_terminal_boundary(document, phase3.plan.piece_plan, score)
        if boundary != reconstructed:
            raise ValueError("saved score candidate terminal boundary differs from assembled score")
        validate_score_terminal_boundary(document, phase3.plan.piece_plan, score, reconstructed)
        if state.get("terminal_boundary") != asdict(reconstructed) or state.get(
            "terminal_boundary_sha256"
        ) != sha256_json(asdict(reconstructed)):
            raise ValueError("saved score terminal boundary does not match accepted candidates")
    for field, filename in (
        ("score_spec", "score-spec.json"),
        ("projection_ledger", "projection-ledger.json"),
        ("foreground_preview", "foreground-preview.mid"),
        ("score_preview", "score-preview.mid"),
    ):
        descriptor = cast(Mapping[str, object], state[field])
        if descriptor.get("path") != f"outputs/{filename}":
            raise ValueError(f"saved score artifact path differs: {field}")
        if descriptor.get("sha256") != sha256_file(output / filename):
            raise ValueError(f"saved score artifact hash differs: {field}")
    if score_spec_from_json(read_score_object(output / "score-spec.json")) != score:
        raise ValueError("saved score does not match reconstructed candidates")
    if read_score_ledger(output / "projection-ledger.json") != cumulative:
        raise ValueError("saved score projection ledger does not match reconstructed evidence")
    script = cast(Mapping[str, object], document["script"])
    setup = cast(Mapping[str, object], script["performance_setup"])
    placements = cast(Mapping[str, Mapping[str, object]], script["material_placements"])
    for filename, only_foreground in (
        ("score-preview.mid", False),
        ("foreground-preview.mid", True),
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
        check_neutral_score_preview(
            segments,
            float(cast(int, setup["target_duration_seconds"])),
            output / filename,
            timing_contract=phase3.plan.timing_contract,
        )
    return LoadedScoreRun(
        document, phase3, score, notes, cumulative, boundary, cast(int, schema_version)
    )


def serialized_notes(notes: Mapping[str, tuple[ScoreNote, ...]]) -> dict[str, object]:
    return {
        placement_id: [asdict(note) for note in values] for placement_id, values in notes.items()
    }


def read_score_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, object], value)


def read_score_ledger(path: Path) -> tuple[ProjectionLedgerEntry, ...]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError(f"expected a projection ledger array: {path}")
    return tuple(ProjectionLedgerEntry(**value) for value in values)
