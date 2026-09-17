"""Finite two-operation compilation from an approved flow to script 0.3.0."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_musical_composer.run_state import RunStore, sha256_json

from .finite_model_operation import (
    ContentValidator,
    FiniteModelOperationResult,
    execute_finite_model_operation,
)
from .flow_validation import check_flow
from .profile_capabilities import (
    GenerationProfileCapabilities,
    generation_profile_capabilities_to_json,
    solo_piano_3m_v2_capabilities,
)
from .proposal import ProposalRunner, preflight_runner
from .script_0_3_model_contracts import (
    IndexedStructureResult,
    ScriptBuildResult,
    build_script_document,
    index_structure_candidate,
    relations_prompt,
    relations_response_schema,
    structure_prompt,
    structure_response_schema,
)
from .validation import IssueCode, ValidationIssue

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class ScriptCompilationContracts:
    structure_prompt: Callable[[Mapping[str, object], GenerationProfileCapabilities], str]
    structure_response_schema: Callable[[GenerationProfileCapabilities], dict[str, object]]
    index_structure_candidate: Callable[
        [
            Mapping[str, object],
            str,
            Mapping[str, object],
            GenerationProfileCapabilities,
        ],
        IndexedStructureResult,
    ]
    relations_prompt: Callable[
        [
            Mapping[str, object],
            Mapping[str, object],
            GenerationProfileCapabilities,
        ],
        str,
    ]
    relations_response_schema: Callable[
        [Mapping[str, object], GenerationProfileCapabilities], dict[str, object]
    ]
    build_script_document: Callable[
        [
            Mapping[str, object],
            Mapping[str, object],
            Mapping[str, object],
            GenerationProfileCapabilities,
        ],
        ScriptBuildResult,
    ]


_SCRIPT_0_3_CONTRACTS = ScriptCompilationContracts(
    structure_prompt,
    structure_response_schema,
    index_structure_candidate,
    relations_prompt,
    relations_response_schema,
    build_script_document,
)


@dataclass(frozen=True, slots=True)
class Script03CompilationRequest:
    approved_flow: Mapping[str, object]
    composition_id: str
    target_profile: str = "solo_piano_3m_v2"


@dataclass(frozen=True, slots=True)
class Script03CompilationResult:
    compiled: bool
    outcome: str
    document: dict[str, object] | None
    run_dir: Path | None
    issues: tuple[ValidationIssue, ...]


def compile_script_0_3(
    request: Script03CompilationRequest,
    runner: ProposalRunner,
    run_dir: str | Path,
) -> Script03CompilationResult:
    """Compile one approved flow through the two model-operation boundary."""
    return compile_script_with_contracts(
        request,
        runner,
        run_dir,
        schema_version="0.3.0",
        contracts=_SCRIPT_0_3_CONTRACTS,
    )


def compile_script_with_contracts(
    request: Script03CompilationRequest,
    runner: ProposalRunner,
    run_dir: str | Path,
    *,
    schema_version: str,
    contracts: ScriptCompilationContracts,
    max_new_operations: int | None = None,
) -> Script03CompilationResult:
    """Execute the shared two-operation boundary with versioned model contracts."""
    if max_new_operations is not None and max_new_operations <= 0:
        raise ValueError("max_new_operations must be positive")
    destination = Path(run_dir).resolve()
    issues = _check_request(request, destination)
    if issues:
        return Script03CompilationResult(False, "request_invalid", None, None, issues)
    preflight_issues = preflight_runner(runner)
    if preflight_issues:
        return Script03CompilationResult(False, "runner_failed", None, None, preflight_issues)

    capabilities = solo_piano_3m_v2_capabilities()
    store = RunStore(destination, max_calls=4)
    store.initialize(
        {
            "schema_version": 1,
            "operation": f"script-{schema_version}-compilation",
            "composition_id": request.composition_id,
            "target_profile": request.target_profile,
            "max_calls": 4,
        }
    )
    store.snapshot_json("inputs/approved-flow.json", dict(request.approved_flow))
    store.snapshot_json(
        "inputs/profile-capabilities.json",
        generation_profile_capabilities_to_json(capabilities),
    )
    repair_count = 0

    structure_was_accepted = _operation_was_accepted(store, "script-structure")
    structure_call = _call_operation(
        store,
        "script-structure",
        contracts.structure_prompt(request.approved_flow, capabilities),
        contracts.structure_response_schema(capabilities),
        request.approved_flow,
        runner,
        lambda response: (
            contracts.index_structure_candidate(
                request.approved_flow,
                request.composition_id,
                response,
                capabilities,
            ).issues
        ),
    )
    if structure_call.response is None:
        return _publish_failure(
            store,
            destination,
            structure_call.outcome,
            structure_call.issues,
            int(structure_call.repaired),
        )
    structure = structure_call.response
    structure_repaired = structure_call.repaired
    repair_count += int(structure_repaired)
    indexed_result = contracts.index_structure_candidate(
        request.approved_flow,
        request.composition_id,
        structure,
        capabilities,
    )
    if not indexed_result.valid or indexed_result.indexed_structure is None:
        return _publish_failure(
            store,
            destination,
            "content_invalid",
            indexed_result.issues,
            repair_count,
        )
    indexed = indexed_result.indexed_structure
    indexed_path = store.snapshot_json("outputs/indexed-structure.json", indexed)

    relation_was_accepted = _operation_was_accepted(store, "script-relations")
    if (
        not relation_was_accepted
        and max_new_operations is not None
        and int(not structure_was_accepted) >= max_new_operations
    ):
        return Script03CompilationResult(False, "paused", None, destination, ())

    relations_call = _call_operation(
        store,
        "script-relations",
        contracts.relations_prompt(request.approved_flow, indexed, capabilities),
        contracts.relations_response_schema(indexed, capabilities),
        indexed,
        runner,
        lambda response: (
            contracts.build_script_document(
                request.approved_flow,
                indexed,
                response,
                capabilities,
            ).issues
        ),
    )
    if relations_call.response is None:
        return _publish_failure(
            store,
            destination,
            relations_call.outcome,
            relations_call.issues,
            repair_count + int(relations_call.repaired),
        )
    relations = relations_call.response
    relations_repaired = relations_call.repaired
    repair_count += int(relations_repaired)
    built = contracts.build_script_document(
        request.approved_flow,
        indexed,
        relations,
        capabilities,
    )
    if not built.valid or built.document is None:
        return _publish_failure(
            store,
            destination,
            built.outcome,
            built.issues,
            repair_count,
        )

    ledger_path = store.snapshot_json(
        "outputs/projection-ledger.json",
        [asdict(entry) for entry in built.projection_ledger],
    )
    script_path = store.snapshot_json("outputs/validated-script.json", built.document)
    summary = {
        "outcome": "complete",
        "call_count": store.call_attempt_count,
        "repair_count": repair_count,
        "indexed_structure_path": str(indexed_path.relative_to(destination)),
        "projection_ledger_path": str(ledger_path.relative_to(destination)),
        "validated_script_path": str(script_path.relative_to(destination)),
        "attempts": [str(path.relative_to(destination)) for path in store.attempt_dirs()],
        "issues": [],
    }
    store.snapshot_json("outputs/compilation.json", summary)
    store.record_step(
        "publish-final",
        "completed",
        {"approved_flow_sha256": sha256_json(request.approved_flow)},
        summary,
    )
    return Script03CompilationResult(True, "complete", built.document, destination, ())


def _call_operation(
    store: RunStore,
    operation: str,
    prompt: str,
    schema: Mapping[str, object],
    immutable_input: Mapping[str, object],
    runner: ProposalRunner,
    validate_content: ContentValidator,
) -> FiniteModelOperationResult:
    return execute_finite_model_operation(
        store=store,
        operation_id=operation,
        prompt=prompt,
        schema=schema,
        immutable_input=immutable_input,
        runner=runner,
        validate_content=validate_content,
    )


def _check_request(
    request: Script03CompilationRequest, destination: Path
) -> tuple[ValidationIssue, ...]:
    if destination.exists() and not (destination / "run-spec.json").is_file():
        return (
            ValidationIssue(
                IssueCode.STORAGE_CONFLICT,
                "The compilation run already exists",
                "/run_dir",
            ),
        )
    if request.target_profile != "solo_piano_3m_v2":
        return (
            ValidationIssue(
                IssueCode.UNSUPPORTED_PROFILE,
                "The compilation profile is unsupported",
                "/target_profile",
            ),
        )
    if _ID_PATTERN.fullmatch(request.composition_id) is None:
        return (
            ValidationIssue(
                IssueCode.SCHEMA_INVALID,
                "The composition ID is invalid",
                "/composition_id",
            ),
        )
    flow_check = check_flow(request.approved_flow)
    if not flow_check.valid:
        return flow_check.issues
    if request.approved_flow["status"] != "approved":
        return (
            ValidationIssue(
                IssueCode.SEMANTIC_INVALID,
                "The input flow must be approved",
                "/approved_flow/status",
            ),
        )
    capabilities = solo_piano_3m_v2_capabilities()
    profile_issues: list[ValidationIssue] = []
    if request.approved_flow["instrumentation"] != capabilities.instrumentation:
        profile_issues.append(
            ValidationIssue(
                IssueCode.UNREPRESENTABLE,
                "The approved instrumentation is unsupported by the generation profile",
                "/approved_flow/instrumentation",
            )
        )
    if request.approved_flow["target_duration"] != capabilities.target_duration_seconds:
        profile_issues.append(
            ValidationIssue(
                IssueCode.UNREPRESENTABLE,
                "The approved duration is unsupported by the generation profile",
                "/approved_flow/target_duration",
            )
        )
    if profile_issues:
        return tuple(profile_issues)
    return ()


def _operation_was_accepted(store: RunStore, operation_id: str) -> bool:
    return (store.run_dir / "events" / operation_id / "accepted.json").is_file()


def _publish_failure(
    store: RunStore,
    destination: Path,
    outcome: str,
    issues: tuple[ValidationIssue, ...],
    repair_count: int = 0,
) -> Script03CompilationResult:
    summary = {
        "outcome": outcome,
        "call_count": store.call_attempt_count,
        "repair_count": repair_count,
        "issues": _issue_values(issues),
    }
    store.snapshot_json("outputs/compilation.json", summary)
    store.record_step("publish-final", "failed", {}, summary)
    return Script03CompilationResult(False, outcome, None, destination, issues)


def _issue_values(issues: tuple[ValidationIssue, ...]) -> list[dict[str, str]]:
    return [
        {"code": issue.code.value, "message": issue.message, "path": issue.path} for issue in issues
    ]
