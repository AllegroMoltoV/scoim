"""Offline reconstruction and verification of a completed phase-7 run."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from llm_musical_composer.run_state import sha256_json, sha256_text

from .finite_model_operation import check_finite_model_operation_records
from .pedal_contract import HARMONY_RELEASE_PEDAL, LEGACY_PEDAL, pedal_vocabulary_version
from .performance_ir import (
    PerformanceSpec,
    RenderedPerformance,
    rendered_performance_to_json,
)
from .phase7_performance_contracts import (
    build_performance_choice_operations,
    build_performance_choices,
    performance_choices_prompt,
    performance_choices_response_schema,
    performance_operation_immutable_input,
)
from .profile_capabilities import (
    generation_profile_capabilities_from_json,
    solo_piano_3m_v2_capabilities,
)
from .projection_ledger import ProjectionLedgerEntry, validate_projection_ledger
from .score_ir import PiecePlan, ScoreSpec, piece_plan_from_json, score_spec_from_json
from .score_rendering import render_score_performance
from .score_timing import LEGACY_TIMING, QUANTIZED_TIMING
from .terminal_boundary import (
    SHARED_TERMINAL,
    TerminalBoundary,
    terminal_boundary_from_json,
    validate_score_terminal_boundary,
)


@dataclass(frozen=True, slots=True)
class LoadedPhase7Run:
    validated_script: Mapping[str, object]
    plan: PiecePlan
    score: ScoreSpec
    performance: PerformanceSpec
    rendered: RenderedPerformance
    rendered_json: Mapping[str, object]
    cumulative_projection_ledger: tuple[ProjectionLedgerEntry, ...]
    phase7_schema_version: int
    terminal_boundary: TerminalBoundary | None = None


def load_complete_phase7_run(run_dir: str | Path) -> LoadedPhase7Run:
    """Reconstruct a phase-7 result from its saved evidence without model access."""
    root = Path(run_dir).resolve()
    operation_records = check_finite_model_operation_records(root)
    if not operation_records.valid:
        raise ValueError(
            f"saved model operation record is invalid: {operation_records.issues[0].message}"
        )
    state = _read_object(root / "outputs" / "phase7-state.json")
    run_spec = _read_object(root / "run-spec.json")
    schema_version = run_spec.get("schema_version")
    if schema_version not in {1, 2, 3, 4, 5}:
        raise ValueError("the saved phase-7 schema version is unsupported")
    timing_contract = QUANTIZED_TIMING if schema_version >= 3 else LEGACY_TIMING
    if schema_version >= 3 and run_spec.get("timing_contract") != timing_contract:
        raise ValueError("the saved phase-7 timing contract does not match its version")
    if schema_version in {1, 2} and "timing_contract" in run_spec:
        raise ValueError("old phase-7 versions cannot declare the new timing contract")
    pedal_contract = HARMONY_RELEASE_PEDAL if schema_version >= 4 else LEGACY_PEDAL
    if schema_version >= 4 and run_spec.get("pedal_contract") != pedal_contract:
        raise ValueError("the saved phase-7 pedal contract does not match its version")
    if schema_version < 4 and "pedal_contract" in run_spec:
        raise ValueError("old phase-7 versions cannot declare a pedal contract")
    if state.get("outcome") != "complete":
        raise ValueError("phase 8 requires a complete phase-7 state")
    if state.get("target_profile") != "solo_piano_3m_v2":
        raise ValueError("the saved phase-7 profile is unsupported")

    document = _read_object(root / "inputs" / "validated-script.json")
    raw_plan = _read_object(root / "inputs" / "piece-plan.json")
    raw_score = _read_object(root / "inputs" / "score-spec.json")
    raw_input_ledger = _read_array(root / "inputs" / "projection-ledger.json")
    raw_capabilities = _read_object(root / "inputs" / "profile-capabilities.json")
    plan = piece_plan_from_json(raw_plan)
    score = score_spec_from_json(raw_score)
    boundary = None
    if schema_version == 5:
        if run_spec.get("terminal_contract") != SHARED_TERMINAL:
            raise ValueError("the saved phase-7 terminal contract differs")
        boundary = terminal_boundary_from_json(_read_object(root / "inputs/terminal-boundary.json"))
        validate_score_terminal_boundary(document, plan, score, boundary)
        boundary_hash = sha256_json(asdict(boundary))
        if any(
            record.get("input_terminal_boundary_sha256") != boundary_hash
            for record in (run_spec, state)
        ):
            raise ValueError("the saved phase-7 terminal boundary hash differs")
    elif "terminal_contract" in run_spec or (root / "inputs/terminal-boundary.json").exists():
        raise ValueError("old phase-7 versions cannot declare a terminal boundary")
    input_ledger = tuple(_ledger_entry(item) for item in raw_input_ledger)
    capabilities = generation_profile_capabilities_from_json(raw_capabilities)
    if capabilities != solo_piano_3m_v2_capabilities(pedal_vocabulary_version(pedal_contract)):
        raise ValueError("saved phase-7 capabilities do not match the saved contract")
    expected_input_hashes = {
        "input_script_sha256": sha256_json(document),
        "input_piece_plan_sha256": sha256_json(asdict(plan)),
        "input_score_spec_sha256": sha256_json(asdict(score)),
        "input_projection_ledger_sha256": sha256_json([asdict(entry) for entry in input_ledger]),
    }
    if any(state.get(field) != digest for field, digest in expected_input_hashes.items()):
        raise ValueError("saved phase-7 input hash does not match")

    operations = build_performance_choice_operations(document, plan)
    accepted: dict[str, Mapping[str, object]] = {}
    performance_id = f"{plan.plan_id}-performance"
    for operation in operations:
        target_ids = (operation.target_section_id,)
        prompt = performance_choices_prompt(
            document, plan, score, operation, accepted, capabilities
        )
        schema = performance_choices_response_schema(document, target_ids, capabilities)
        immutable_input = performance_operation_immutable_input(
            document, plan, score, operation, capabilities
        )
        record = _read_object(root / "events" / operation.operation_id / "accepted.json")
        if (
            record.get("prompt_sha256") != sha256_text(prompt)
            or record.get("schema_sha256") != sha256_json(schema)
            or record.get("immutable_input_sha256") != sha256_json(immutable_input)
        ):
            raise ValueError("saved phase-7 operation input does not match")
        response = cast(Mapping[str, object], record.get("response"))
        checked = build_performance_choices(
            document,
            plan,
            target_ids,
            response,
            capabilities,
            performance_id=performance_id,
        )
        if not checked.valid:
            raise ValueError("saved phase-7 operation response is invalid")
        accepted[operation.target_section_id] = cast(
            list[Mapping[str, object]], response["performances"]
        )[0]

    combined_response = {
        "performances": [accepted[operation.target_section_id] for operation in operations]
    }
    built = build_performance_choices(
        document,
        plan,
        tuple(operation.target_section_id for operation in operations),
        combined_response,
        capabilities,
        performance_id=performance_id,
    )
    if not built.valid or built.performance is None:
        raise ValueError("saved phase-7 responses cannot reconstruct a performance")
    saved_performance = _read_object(root / "outputs" / "performance-spec.json")
    if sha256_json(saved_performance) != sha256_json(asdict(built.performance)):
        raise ValueError("saved phase-7 performance does not match reconstructed values")

    rendered = render_score_performance(
        document,
        plan,
        score,
        built.performance,
        physical_key_contract=schema_version >= 2,
        timing_contract=timing_contract,
        pedal_contract=pedal_contract,
        terminal_boundary=boundary,
    )
    rendered_json = rendered_performance_to_json(
        rendered,
        score,
        schema_version=2 if schema_version >= 2 else 1,
    )
    saved_rendered = _read_object(root / "outputs" / "rendered-performance.json")
    if sha256_json(saved_rendered) != sha256_json(rendered_json):
        raise ValueError("saved phase-7 rendering does not match reconstructed values")
    cumulative_ledger = (*input_ledger, *built.projection_ledger)
    saved_ledger = tuple(
        _ledger_entry(item) for item in _read_array(root / "outputs" / "projection-ledger.json")
    )
    if saved_ledger != cumulative_ledger:
        raise ValueError("saved phase-7 projection ledger does not match")
    validate_projection_ledger(saved_ledger)
    output_hashes = {
        "performance_spec_sha256": sha256_json(asdict(built.performance)),
        "rendered_performance_sha256": sha256_json(rendered_json),
        "projection_ledger_sha256": sha256_json([asdict(entry) for entry in cumulative_ledger]),
    }
    if any(state.get(field) != digest for field, digest in output_hashes.items()):
        raise ValueError("saved phase-7 output hash does not match")
    return LoadedPhase7Run(
        document,
        plan,
        score,
        built.performance,
        rendered,
        rendered_json,
        cumulative_ledger,
        cast(int, schema_version),
        boundary,
    )


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, object], value)


def _read_array(path: Path) -> list[Mapping[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected a JSON array: {path}")
    return cast(list[Mapping[str, object]], value)


def _ledger_entry(value: Mapping[str, object]) -> ProjectionLedgerEntry:
    return ProjectionLedgerEntry(**cast(dict[str, object], value))
