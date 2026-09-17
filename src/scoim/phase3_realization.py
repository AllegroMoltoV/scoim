"""Recorded phase-3 realization from a validated script to plan and shared harmony."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import cast

from llm_musical_composer.run_state import RunStore, atomic_write_json, sha256_json

from .finite_model_operation import execute_finite_model_operation
from .generation_script_validation import check_generation_script_document
from .phase3_model_contracts import (
    HarmonicPlan,
    build_harmonic_plan,
    build_score_harmonies,
    check_harmony_response,
    harmony_projection_entries,
    harmony_prompt,
    harmony_response_schema,
    ordered_leaf_section_ids,
    overall_plan_prompt,
    overall_plan_response_schema,
)
from .profile_capabilities import (
    generation_profile_capabilities_from_json,
    generation_profile_capabilities_to_json,
    solo_piano_3m_v2_capabilities,
)
from .projection_ledger import (
    ProjectionLedgerEntry,
    ProjectionLedgerValidationError,
    validate_expected_projection_targets,
    validate_projection_ledger,
)
from .proposal import ProposalRunner, preflight_runner
from .score_ir import ScoreHarmony
from .validation import IssueCode, ValidationIssue


@dataclass(frozen=True, slots=True)
class Phase3Request:
    validated_script: Mapping[str, object]
    projection_ledger: tuple[ProjectionLedgerEntry, ...]
    target_profile: str = "solo_piano_3m_v2"
    divisions: int = 12
    units_per_duration_weight: int = 12


@dataclass(frozen=True, slots=True)
class Phase3RealizationResult:
    realized: bool
    outcome: str
    run_dir: Path | None
    state: dict[str, object] | None
    issues: tuple[ValidationIssue, ...]


def realize_phase3(
    request: Phase3Request,
    runner: ProposalRunner,
    run_dir: str | Path,
    *,
    max_new_operations: int | None = None,
) -> Phase3RealizationResult:
    """Build and record the phase-3 plan, stopping at the first failed operation."""
    if max_new_operations is not None and max_new_operations <= 0:
        raise ValueError("max_new_operations must be positive")
    destination = Path(run_dir).resolve()
    request_issues = _check_request(request)
    if request_issues:
        return Phase3RealizationResult(False, "request_invalid", None, None, request_issues)

    leaf_ids = ordered_leaf_section_ids(request.validated_script)
    operation_order = ["overall-plan"] + [
        f"harmony-score-unit-{section_id}" for section_id in leaf_ids
    ]
    max_calls = 2 * len(operation_order)
    store = RunStore(destination, max_calls=max_calls)
    spec = {
        "schema_version": 1,
        "operation": "phase3-realization",
        "target_profile": request.target_profile,
        "divisions": request.divisions,
        "units_per_duration_weight": request.units_per_duration_weight,
        "input_script_sha256": sha256_json(request.validated_script),
        "input_projection_ledger_sha256": sha256_json(
            [asdict(entry) for entry in request.projection_ledger]
        ),
        "operation_order": operation_order,
        "max_calls": max_calls,
    }
    store.initialize(spec)
    store.snapshot_json("inputs/validated-script.json", dict(request.validated_script))
    store.snapshot_json(
        "inputs/projection-ledger.json",
        [asdict(entry) for entry in request.projection_ledger],
    )
    capabilities_json = generation_profile_capabilities_to_json(solo_piano_3m_v2_capabilities())
    store.snapshot_json("inputs/profile-capabilities.json", capabilities_json)
    if (
        generation_profile_capabilities_from_json(capabilities_json)
        != solo_piano_3m_v2_capabilities()
    ):
        raise RuntimeError("saved generation profile capabilities did not round-trip")

    preflight_issues = preflight_runner(runner)
    if preflight_issues:
        return _publish_failure(store, destination, "runner_failed", preflight_issues, None)

    overall_schema = overall_plan_response_schema(leaf_count=len(leaf_ids))

    def validate_overall(response: Mapping[str, object]) -> tuple[ValidationIssue, ...]:
        try:
            build_harmonic_plan(
                request.validated_script,
                response,
                divisions=request.divisions,
                units_per_duration_weight=request.units_per_duration_weight,
            )
        except (KeyError, TypeError, ValueError) as error:
            return (ValidationIssue(IssueCode.SEMANTIC_INVALID, str(error), "/overall-plan"),)
        return ()

    overall_was_accepted = _operation_was_accepted(store, "overall-plan")
    overall_result = execute_finite_model_operation(
        store=store,
        operation_id="overall-plan",
        prompt=overall_plan_prompt(request.validated_script),
        schema=overall_schema,
        immutable_input=request.validated_script,
        runner=runner,
        validate_content=validate_overall,
    )
    if overall_result.response is None:
        return _publish_failure(
            store,
            destination,
            overall_result.outcome,
            overall_result.issues,
            None,
        )
    plan = build_harmonic_plan(
        request.validated_script,
        overall_result.response,
        divisions=request.divisions,
        units_per_duration_weight=request.units_per_duration_weight,
    )
    store.snapshot_json("outputs/harmonic-plan.json", _harmonic_plan_json(plan))
    store.snapshot_json("outputs/piece-plan.json", asdict(plan.piece_plan))

    harmonies_by_score_unit: dict[str, tuple[ScoreHarmony, ...]] = {}
    phase3_ledger = list(plan.projection_ledger)
    repair_count = int(overall_result.repaired)
    new_operation_count = int(not overall_was_accepted)
    for intent_index, intent in enumerate(plan.section_intents):
        previous = None
        if harmonies_by_score_unit:
            previous_harmony = next(reversed(harmonies_by_score_unit.values()))[-1]
            previous = {
                "root_pitch_class": previous_harmony.root_pitch_class,
                "quality": previous_harmony.quality,
            }
        required_final_tonic = (
            (plan.piece_plan.tonal_center, plan.piece_plan.mode)
            if intent_index == len(plan.section_intents) - 1
            else None
        )
        length_units = plan.length_units_by_score_unit[intent.score_unit_id]
        operation_id = f"harmony-{intent.score_unit_id}"
        harmony_was_accepted = _operation_was_accepted(store, operation_id)
        if (
            not harmony_was_accepted
            and max_new_operations is not None
            and new_operation_count >= max_new_operations
        ):
            state = _write_state(
                destination,
                plan,
                harmonies_by_score_unit,
                phase3_ledger,
                outcome="paused",
            )
            return Phase3RealizationResult(False, "paused", destination, state, ())
        harmony_result = execute_finite_model_operation(
            store=store,
            operation_id=operation_id,
            prompt=harmony_prompt(
                request.validated_script,
                plan,
                intent_index=intent_index,
                previous_final_harmony=previous,
            ),
            schema=harmony_response_schema(),
            immutable_input={
                "script_sha256": sha256_json(request.validated_script),
                "score_unit_id": intent.score_unit_id,
                "length_units": length_units,
                "previous_final_harmony": previous,
                "required_final_tonic": required_final_tonic,
            },
            runner=runner,
            validate_content=partial(
                check_harmony_response,
                length_units=length_units,
                required_final_tonic=required_final_tonic,
            ),
        )
        repair_count += int(harmony_result.repaired)
        if harmony_result.response is None:
            state = _write_state(
                destination,
                plan,
                harmonies_by_score_unit,
                phase3_ledger,
                outcome=harmony_result.outcome,
            )
            return _publish_failure(
                store,
                destination,
                harmony_result.outcome,
                harmony_result.issues,
                state,
            )
        new_operation_count += int(not harmony_was_accepted)
        harmonies = build_score_harmonies(intent.score_unit_id, harmony_result.response)
        harmonies_by_score_unit[intent.score_unit_id] = harmonies
        phase3_ledger.extend(harmony_projection_entries(intent.score_unit_id, harmonies))
        store.snapshot_json(
            f"outputs/harmonies/{intent.score_unit_id}.json",
            [asdict(harmony) for harmony in harmonies],
        )
        _write_state(
            destination,
            plan,
            harmonies_by_score_unit,
            phase3_ledger,
            outcome="running",
        )

    completed_ledger = tuple(request.projection_ledger) + tuple(phase3_ledger)
    validate_projection_ledger(completed_ledger)
    validate_expected_projection_targets(
        tuple(phase3_ledger),
        {
            "plan_node": frozenset(node.section_id for node in plan.piece_plan.nodes),
            "score_unit": frozenset(plan.length_units_by_score_unit),
            "score_harmony": frozenset(
                harmony.harmony_id
                for harmonies in harmonies_by_score_unit.values()
                for harmony in harmonies
            ),
        },
    )
    store.snapshot_json(
        "outputs/projection-ledger.json",
        [asdict(entry) for entry in completed_ledger],
    )
    state = _write_state(
        destination,
        plan,
        harmonies_by_score_unit,
        phase3_ledger,
        outcome="complete",
    )
    summary = {
        "outcome": "complete",
        "call_count": store.call_attempt_count,
        "repair_count": repair_count,
        "operation_order": operation_order,
        "issues": [],
    }
    store.snapshot_json("outputs/realization.json", summary)
    store.record_step(
        "publish-final",
        "completed",
        {"input_script_sha256": spec["input_script_sha256"]},
        summary,
    )
    return Phase3RealizationResult(True, "complete", destination, state, ())


def _check_request(request: Phase3Request) -> tuple[ValidationIssue, ...]:
    if request.target_profile != "solo_piano_3m_v2":
        return (
            ValidationIssue(
                IssueCode.UNSUPPORTED_PROFILE,
                "The phase-3 profile is unsupported",
                "/target_profile",
            ),
        )
    if request.divisions <= 0 or request.units_per_duration_weight <= 0:
        return (
            ValidationIssue(
                IssueCode.SEMANTIC_INVALID,
                "Phase-3 time grid values must be positive",
                "/time_grid",
            ),
        )
    script_check = check_generation_script_document(request.validated_script)
    if not script_check.valid:
        return script_check.issues
    try:
        validate_projection_ledger(request.projection_ledger)
        validate_expected_projection_targets(
            request.projection_ledger,
            expected_phase2_projection_targets(request.validated_script),
        )
    except ProjectionLedgerValidationError as error:
        return (
            ValidationIssue(
                IssueCode.SEMANTIC_INVALID,
                str(error),
                "/projection_ledger",
            ),
        )
    return ()


def expected_phase2_projection_targets(
    document: Mapping[str, object],
) -> dict[str, frozenset[str]]:
    script = cast(dict[str, object], document["script"])
    setup = cast(dict[str, object], script["performance_setup"])
    targets = {
        "sections": frozenset(cast(dict[str, object], script["sections"])),
        "materials": frozenset(cast(dict[str, object], script["materials"])),
        "material_placements": frozenset(cast(dict[str, object], script["material_placements"])),
        "script_element_variation_relations": frozenset(
            cast(dict[str, object], script["script_element_variation_relations"])
        ),
        "material_placement_transitions": frozenset(
            cast(dict[str, object], script["material_placement_transitions"])
        ),
        "performance_directions": frozenset(
            cast(dict[str, object], setup["performance_directions"])
        ),
    }
    comparison_requirements = script.get("performance_direction_comparison_requirements")
    if comparison_requirements is not None:
        targets["performance_direction_comparison_requirements"] = frozenset(
            cast(dict[str, object], comparison_requirements)
        )
    return targets


def _harmonic_plan_json(plan: HarmonicPlan) -> dict[str, object]:
    return {
        "piece_plan": asdict(plan.piece_plan),
        "overall_harmonic_story": plan.overall_harmonic_story,
        "section_intents": [asdict(intent) for intent in plan.section_intents],
        "divisions": plan.divisions,
        "length_units_by_score_unit": plan.length_units_by_score_unit,
        "projection_ledger": [asdict(entry) for entry in plan.projection_ledger],
    }


def _write_state(
    destination: Path,
    plan: HarmonicPlan,
    harmonies_by_score_unit: Mapping[str, tuple[ScoreHarmony, ...]],
    phase3_ledger: list[ProjectionLedgerEntry],
    *,
    outcome: str,
) -> dict[str, object]:
    state = {
        "outcome": outcome,
        "harmonic_plan": _harmonic_plan_json(plan),
        "harmonies_by_score_unit": {
            score_unit_id: [asdict(harmony) for harmony in harmonies]
            for score_unit_id, harmonies in harmonies_by_score_unit.items()
        },
        "projection_ledger": [asdict(entry) for entry in phase3_ledger],
    }
    atomic_write_json(destination / "outputs" / "phase3-state.json", state)
    return state


def _publish_failure(
    store: RunStore,
    destination: Path,
    outcome: str,
    issues: tuple[ValidationIssue, ...],
    state: dict[str, object] | None,
) -> Phase3RealizationResult:
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
    return Phase3RealizationResult(False, outcome, destination, state, issues)


def _operation_was_accepted(store: RunStore, operation_id: str) -> bool:
    return (store.run_dir / "events" / operation_id / "accepted.json").is_file()
