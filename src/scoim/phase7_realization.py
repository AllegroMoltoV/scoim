"""Recorded phase-7 selection and deterministic performance rendering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from llm_musical_composer.run_state import RunStore, sha256_json

from .finite_model_operation import execute_finite_model_operation
from .phase6_state import load_complete_phase6_run
from .phase7_performance_contracts import (
    build_performance_choice_operations,
    build_performance_choices,
    performance_choices_prompt,
    performance_choices_response_schema,
    performance_operation_immutable_input,
)
from .profile_capabilities import (
    generation_profile_capabilities_to_json,
    solo_piano_3m_v2_capabilities,
)
from .projection_ledger import validate_projection_ledger
from .proposal import ProposalRunner, preflight_runner
from .score_rendering import ScoreRenderingError, render_score_performance
from .validation import IssueCode, ValidationIssue


@dataclass(frozen=True, slots=True)
class Phase7Request:
    phase6_run_dir: str | Path
    target_profile: str = "solo_piano_3m_v2"


@dataclass(frozen=True, slots=True)
class Phase7RealizationResult:
    realized: bool
    outcome: str
    run_dir: Path | None
    state: dict[str, object] | None
    issues: tuple[ValidationIssue, ...]


def realize_phase7(
    request: Phase7Request,
    runner: ProposalRunner,
    run_dir: str | Path,
) -> Phase7RealizationResult:
    """Choose finite performance methods and render the checked phase-6 score."""
    if request.target_profile != "solo_piano_3m_v2":
        issue = ValidationIssue(
            IssueCode.UNSUPPORTED_PROFILE,
            "The phase-7 profile is unsupported",
            "/target_profile",
        )
        return Phase7RealizationResult(False, "request_invalid", None, None, (issue,))
    try:
        loaded = load_complete_phase6_run(request.phase6_run_dir)
        document = loaded.validated_script
        plan = loaded.phase5.phase4.phase3.plan.piece_plan
        score = loaded.score
        capabilities = solo_piano_3m_v2_capabilities()
        operations = build_performance_choice_operations(document, plan)
    except (KeyError, OSError, TypeError, ValueError) as error:
        issue = ValidationIssue(IssueCode.SEMANTIC_INVALID, str(error), "/phase7_request")
        return Phase7RealizationResult(False, "request_invalid", None, None, (issue,))

    destination = Path(run_dir).resolve()
    store = RunStore(destination, max_calls=2 * len(operations))
    operation_order = [operation.operation_id for operation in operations]
    spec = {
        "schema_version": 1,
        "operation": "phase7-performance-realization",
        "target_profile": request.target_profile,
        "input_script_sha256": sha256_json(document),
        "input_piece_plan_sha256": sha256_json(asdict(plan)),
        "input_score_spec_sha256": sha256_json(asdict(score)),
        "input_projection_ledger_sha256": sha256_json(
            [asdict(entry) for entry in loaded.cumulative_projection_ledger]
        ),
        "operation_order": operation_order,
        "max_calls": 2 * len(operations),
    }
    store.initialize(spec)
    store.snapshot_json("inputs/validated-script.json", dict(document))
    store.snapshot_json("inputs/piece-plan.json", asdict(plan))
    store.snapshot_json("inputs/score-spec.json", asdict(score))
    store.snapshot_json(
        "inputs/projection-ledger.json",
        [asdict(entry) for entry in loaded.cumulative_projection_ledger],
    )
    store.snapshot_json(
        "inputs/profile-capabilities.json",
        generation_profile_capabilities_to_json(capabilities),
    )
    if operations:
        preflight_issues = preflight_runner(runner)
        if preflight_issues:
            return _publish_failure(store, destination, "runner_failed", preflight_issues, None)

    accepted: dict[str, Mapping[str, object]] = {}
    repair_count = 0
    performance_id = f"{plan.plan_id}-performance"
    for operation in operations:
        target_ids = (operation.target_section_id,)
        prompt = performance_choices_prompt(
            document,
            plan,
            score,
            operation,
            accepted,
            capabilities,
        )
        schema = performance_choices_response_schema(document, target_ids, capabilities)

        def validate_content(
            response: Mapping[str, object],
            *,
            current_target_ids: tuple[str, ...] = target_ids,
        ) -> tuple[ValidationIssue, ...]:
            return build_performance_choices(
                document,
                plan,
                current_target_ids,
                response,
                capabilities,
                performance_id=performance_id,
            ).issues

        result = execute_finite_model_operation(
            store=store,
            operation_id=operation.operation_id,
            prompt=prompt,
            schema=schema,
            immutable_input=performance_operation_immutable_input(
                document, plan, score, operation, capabilities
            ),
            runner=runner,
            validate_content=validate_content,
        )
        repair_count += int(result.repaired)
        if result.response is None:
            return _publish_failure(store, destination, result.outcome, result.issues, None)
        accepted[operation.target_section_id] = cast(
            list[Mapping[str, object]], result.response["performances"]
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
        return _publish_failure(store, destination, "content_invalid", built.issues, None)
    cumulative_ledger = (*loaded.cumulative_projection_ledger, *built.projection_ledger)
    validate_projection_ledger(cumulative_ledger)
    try:
        rendered = render_score_performance(document, plan, score, built.performance)
    except (KeyError, TypeError, ValueError, ScoreRenderingError) as error:
        issue = ValidationIssue(IssueCode.SEMANTIC_INVALID, str(error), "/performance")
        return _publish_failure(store, destination, "render_invalid", (issue,), None)

    store.snapshot_json("outputs/performance-spec.json", asdict(built.performance))
    store.snapshot_json("outputs/rendered-performance.json", asdict(rendered))
    store.snapshot_json(
        "outputs/projection-ledger.json", [asdict(entry) for entry in cumulative_ledger]
    )
    state = {
        "outcome": "complete",
        "target_profile": request.target_profile,
        "input_script_sha256": spec["input_script_sha256"],
        "input_piece_plan_sha256": spec["input_piece_plan_sha256"],
        "input_score_spec_sha256": spec["input_score_spec_sha256"],
        "input_projection_ledger_sha256": spec["input_projection_ledger_sha256"],
        "performance_spec_sha256": sha256_json(asdict(built.performance)),
        "rendered_performance_sha256": sha256_json(asdict(rendered)),
        "projection_ledger_sha256": sha256_json([asdict(entry) for entry in cumulative_ledger]),
    }
    store.snapshot_json("outputs/phase7-state.json", state)
    summary = {
        "outcome": "complete",
        "call_count": store.call_attempt_count,
        "repair_count": repair_count,
        "operation_order": operation_order,
        "issues": [],
    }
    store.snapshot_json("outputs/realization.json", summary)
    store.record_step("publish-final", "completed", {}, summary)
    return Phase7RealizationResult(True, "complete", destination, state, ())


def _publish_failure(
    store: RunStore,
    destination: Path,
    outcome: str,
    issues: tuple[ValidationIssue, ...],
    state: dict[str, object] | None,
) -> Phase7RealizationResult:
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
    return Phase7RealizationResult(False, outcome, destination, state, issues)
