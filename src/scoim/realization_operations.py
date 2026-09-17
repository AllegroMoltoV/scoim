"""Typed music operations that compile model choices into checked workspace diffs."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from jsonschema import Draft202012Validator

from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    PipelineValidationError,
    validate_performance_spec,
)
from llm_musical_composer.piano_texture_pilot import PianoTextureValidationError
from llm_musical_composer.run_state import RunStore, sha256_json

from .proposal import ProposalRunner
from .realization_run import (
    OperationInstance,
    RealizationRunStatus,
    RealizationRunStore,
    build_content_repair_prompt,
    checked_diff_sha256,
)
from .realization_script import check_realization_script, realization_script_sha256
from .realization_workspace import (
    AccompanimentValue,
    CheckedWorkspaceDiff,
    PatchOperation,
    PerformanceValue,
    RealizationWorkspace,
    state_key_sha256,
    upgrade_workspace_v3_to_v4,
)
from .solo_piano_accompaniment import (
    AccompanimentRequestEvent,
    place_workspace_accompaniment,
)
from .solo_piano_ending import ending_materialization_diagnostic, materialize_solo_piano_ending
from .solo_piano_performance import (
    performance_contexts,
    performance_operation_schedule,
    performance_piece_plan,
    performance_read_keys,
)
from .validation import IssueCode, ValidationIssue

_HARMONY_QUALITIES = ("major", "minor", "diminished", "major-seventh")


def initialize_minimal_realization_run(
    run_dir: Path,
    document: dict[str, object],
    *,
    review_after: frozenset[str],
    model: str,
    max_calls: int,
) -> RealizationRunStore:
    """Initialize the two-operation plan-and-harmony experiment."""
    harmony_targets = _non_ending_material_keys(document)
    operations = (
        OperationInstance("overall-plan", "plan", ("/plan",)),
        OperationInstance("harmony-all", "harmony", harmony_targets),
    )
    run = RealizationRunStore(run_dir)
    run.initialize(
        document,
        operations=operations,
        review_after=review_after,
        model=model,
        max_calls=max_calls,
    )
    return run


def initialize_melody_realization_run(
    run_dir: Path,
    document: dict[str, object],
    initial_workspace: RealizationWorkspace,
    *,
    source_workspace_record_sha256: str,
    review_after: frozenset[str],
    model: str,
    max_calls: int,
) -> RealizationRunStore:
    """Initialize ordinary and transition melody operations from saved harmony state."""
    if initial_workspace.schema_version not in {2, 3, 4}:
        raise ValueError("Melody realization requires a version 2, 3, or 4 workspace")
    transition_targets = _transition_keys(document)
    operations = (
        OperationInstance(
            "melody-all",
            "melody",
            _main_melody_material_keys(document),
        ),
    )
    if transition_targets:
        operations += (
            OperationInstance(
                "transition-melody-all",
                "transition-melody",
                transition_targets,
            ),
        )
    run = RealizationRunStore(run_dir)
    run.initialize(
        document,
        operations=operations,
        review_after=review_after,
        model=model,
        max_calls=max_calls,
        initial_workspace=initial_workspace,
        source_workspace_record_sha256=source_workspace_record_sha256,
    )
    return run


def initialize_accompaniment_realization_run(
    run_dir: Path,
    document: dict[str, object],
    initial_workspace: RealizationWorkspace,
    *,
    source_workspace_record_sha256: str,
    review_after: frozenset[str],
    model: str,
    max_calls: int,
    content_repair_limit: int = 0,
) -> RealizationRunStore:
    """Initialize ordinary and transition accompaniment operations."""
    if initial_workspace.schema_version not in {3, 4}:
        raise ValueError("Accompaniment realization requires a version 3 or 4 workspace")
    operations = (
        OperationInstance(
            "accompaniment-all",
            "accompaniment",
            _main_melody_material_keys(document),
        ),
    )
    transition_targets = _transition_keys(document)
    if transition_targets:
        operations += (
            OperationInstance(
                "transition-accompaniment-all",
                "transition-accompaniment",
                transition_targets,
            ),
        )
    run = RealizationRunStore(run_dir)
    run.initialize(
        document,
        operations=operations,
        review_after=review_after,
        model=model,
        max_calls=max_calls,
        content_repair_limit=content_repair_limit,
        initial_workspace=initial_workspace,
        source_workspace_record_sha256=source_workspace_record_sha256,
    )
    return run


def initialize_ending_realization_run(
    run_dir: Path,
    document: dict[str, object],
    initial_workspace: RealizationWorkspace,
    *,
    source_workspace_record_sha256: str,
    review_after: frozenset[str],
) -> RealizationRunStore:
    """Initialize one deterministic ending operation from saved accompaniment state."""
    if initial_workspace.schema_version not in {3, 4}:
        raise ValueError("Ending realization requires a version 3 or 4 workspace")
    run = RealizationRunStore(run_dir)
    run.initialize(
        document,
        operations=(
            OperationInstance(
                "ending-material",
                "ending",
                _ending_material_keys(document),
            ),
        ),
        review_after=review_after,
        model="deterministic-python",
        max_calls=0,
        initial_workspace=initial_workspace,
        source_workspace_record_sha256=source_workspace_record_sha256,
    )
    return run


def initialize_performance_realization_run(
    run_dir: Path,
    document: dict[str, object],
    initial_workspace: RealizationWorkspace,
    *,
    source_workspace_record_sha256: str,
    review_after: frozenset[str],
    model: str,
) -> RealizationRunStore:
    """Initialize deterministic defaults and dependency-ordered performance waves."""
    if initial_workspace.schema_version == 3:
        initial_workspace = upgrade_workspace_v3_to_v4(initial_workspace)
    if initial_workspace.schema_version != 4:
        raise ValueError("Performance realization requires a version 3 or 4 workspace")
    try:
        schedule = performance_operation_schedule(document, initial_workspace)
    except ValueError:
        schedule = None
    operations = (
        OperationInstance(
            "performance-defaults",
            "performance-defaults",
            schedule.default_node_ids if schedule is not None else (),
        ),
    )
    if schedule is not None and schedule.absolute_node_ids:
        operations += (
            OperationInstance(
                "performance-absolute",
                "performance",
                schedule.absolute_node_ids,
            ),
        )
    if schedule is not None:
        operations += tuple(
            OperationInstance(f"performance-comparative-{index}", "performance", targets)
            for index, targets in enumerate(schedule.comparative_waves, start=1)
        )
    model_calls = sum(operation.operation == "performance" for operation in operations)
    run = RealizationRunStore(run_dir)
    run.initialize(
        document,
        operations=operations,
        review_after=review_after,
        model=model,
        max_calls=model_calls,
        initial_workspace=initial_workspace,
        source_workspace_record_sha256=source_workspace_record_sha256,
    )
    return run


def advance_minimal_realization_run(run_dir: Path, runner: ProposalRunner) -> RealizationRunStatus:
    """Advance until the run completes, fails, or reaches a configured review point."""
    run = RealizationRunStore(run_dir)
    document = cast(
        dict[str, object],
        json.loads((run.run_dir / "inputs" / "approved-script.json").read_text(encoding="utf-8")),
    )
    spec = RunStore(run.run_dir).read_spec()
    operations = cast(list[dict[str, object]], spec["operations"])
    while True:
        status = run.rebuild()
        if status.status in {"awaiting_review", "failed", "completed"}:
            return status
        operation = operations[len(status.completed_operations)]
        instance_id = cast(str, operation["instance_id"])
        operation_name = cast(str, operation["operation"])
        targets = tuple(cast(list[str], operation["targets"]))
        if operation_name == "plan":
            schema = plan_response_schema(document)
            prompt = _plan_prompt(document)
        elif operation_name == "harmony":
            schema = harmony_response_schema(document, targets)
            prompt = _harmony_prompt(document, status.workspace, targets)
        else:
            raise ValueError(f"Unsupported realization operation: {operation_name}")
        schema_path = run.run_dir / "schemas" / f"{instance_id}.json"
        RunStore(run.run_dir, max_calls=cast(int, spec["max_calls"])).snapshot_json(
            schema_path.relative_to(run.run_dir), schema
        )
        response = run.execute_model_call(instance_id, prompt, schema_path, runner)
        if operation_name == "plan":
            checked_diff = build_plan_checked_diff(document, status.workspace, response)
        else:
            checked_diff = build_harmony_checked_diff(document, status.workspace, targets, response)
        run.record_checked_diff(instance_id, checked_diff)


def advance_melody_realization_run(run_dir: Path, runner: ProposalRunner) -> RealizationRunStatus:
    """Advance a melody-only workspace run to review, failure, or completion."""
    run = RealizationRunStore(run_dir)
    document = cast(
        dict[str, object],
        json.loads((run.run_dir / "inputs" / "approved-script.json").read_text(encoding="utf-8")),
    )
    spec = RunStore(run.run_dir).read_spec()
    operations = cast(list[dict[str, object]], spec["operations"])
    while True:
        status = run.rebuild()
        if status.status in {"awaiting_review", "failed", "completed"}:
            return status
        operation = operations[len(status.completed_operations)]
        instance_id = cast(str, operation["instance_id"])
        operation_name = cast(str, operation["operation"])
        targets = tuple(cast(list[str], operation["targets"]))
        if operation_name == "melody":
            if _melody_target_issues(document, status.workspace, targets):
                run.record_checked_diff(
                    instance_id,
                    build_melody_checked_diff(document, status.workspace, targets, {}),
                )
                continue
            schema = melody_response_schema(document, status.workspace, targets)
            prompt = _melody_prompt(document, status.workspace, targets)
        elif operation_name == "transition-melody":
            if _transition_melody_target_issues(document, status.workspace, targets):
                run.record_checked_diff(
                    instance_id,
                    build_transition_melody_checked_diff(document, status.workspace, targets, {}),
                )
                continue
            schema = transition_melody_response_schema(document, status.workspace, targets)
            prompt = _transition_melody_prompt(document, status.workspace, targets)
        else:
            raise ValueError(f"Unsupported melody realization operation: {operation_name}")
        schema_path = run.run_dir / "schemas" / f"{instance_id}.json"
        RunStore(run.run_dir, max_calls=cast(int, spec["max_calls"])).snapshot_json(
            schema_path.relative_to(run.run_dir), schema
        )
        response = run.execute_model_call(instance_id, prompt, schema_path, runner)
        if operation_name == "melody":
            checked_diff = build_melody_checked_diff(document, status.workspace, targets, response)
        else:
            checked_diff = build_transition_melody_checked_diff(
                document, status.workspace, targets, response
            )
        run.record_checked_diff(instance_id, checked_diff)


def advance_accompaniment_realization_run(
    run_dir: Path, runner: ProposalRunner
) -> RealizationRunStatus:
    """Advance an accompaniment-only workspace run to review, failure, or completion."""
    run = RealizationRunStore(run_dir)
    document = cast(
        dict[str, object],
        json.loads((run.run_dir / "inputs" / "approved-script.json").read_text(encoding="utf-8")),
    )
    spec = RunStore(run.run_dir).read_spec()
    operations = cast(list[dict[str, object]], spec["operations"])
    while True:
        status = run.rebuild()
        if status.status in {"awaiting_review", "failed", "completed"}:
            return status
        operation = operations[len(status.completed_operations)]
        instance_id = cast(str, operation["instance_id"])
        operation_name = cast(str, operation["operation"])
        targets = tuple(cast(list[str], operation["targets"]))
        if operation_name == "accompaniment":
            target_issues = _accompaniment_target_issues(document, status.workspace, targets)
            if target_issues:
                run.record_checked_diff(
                    instance_id,
                    build_accompaniment_checked_diff(document, status.workspace, targets, {}),
                )
                continue
            schema = accompaniment_response_schema(document, status.workspace, targets)
            prompt = _accompaniment_prompt(document, status.workspace, targets)
        elif operation_name == "transition-accompaniment":
            target_issues = _transition_accompaniment_target_issues(
                document, status.workspace, targets
            )
            if target_issues:
                run.record_checked_diff(
                    instance_id,
                    build_transition_accompaniment_checked_diff(
                        document, status.workspace, targets, {}
                    ),
                )
                continue
            schema = transition_accompaniment_response_schema(document, status.workspace, targets)
            prompt = _transition_accompaniment_prompt(document, status.workspace, targets)
        else:
            raise ValueError(f"Unsupported accompaniment operation: {operation_name}")
        schema_path = run.run_dir / "schemas" / f"{instance_id}.json"
        RunStore(run.run_dir, max_calls=cast(int, spec["max_calls"])).snapshot_json(
            schema_path.relative_to(run.run_dir), schema
        )
        checked_diff = _execute_accompaniment_model_operation(
            run,
            document,
            status.workspace,
            instance_id,
            operation_name,
            targets,
            prompt,
            schema_path,
            runner,
        )
        run.record_checked_diff(instance_id, checked_diff)


def _execute_accompaniment_model_operation(
    run: RealizationRunStore,
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    instance_id: str,
    operation_name: str,
    targets: tuple[str, ...],
    prompt: str,
    schema_path: Path,
    runner: ProposalRunner,
) -> CheckedWorkspaceDiff:
    def build_diff(response: Mapping[str, object]) -> CheckedWorkspaceDiff:
        if operation_name == "accompaniment":
            return build_accompaniment_checked_diff(document, workspace, targets, response)
        return build_transition_accompaniment_checked_diff(document, workspace, targets, response)

    def build_repair_prompt(
        original_prompt: str,
        response: Mapping[str, object],
        issues: tuple[ValidationIssue, ...],
    ) -> str:
        return build_content_repair_prompt(
            original_prompt,
            workspace.workspace_record_sha256,
            response,
            issues,
        )

    _, checked_diff = run.execute_checked_model_call(
        instance_id,
        prompt,
        schema_path,
        runner,
        build_diff,
        build_repair_prompt,
    )
    return checked_diff


def advance_ending_realization_run(run_dir: Path) -> RealizationRunStatus:
    """Advance the model-free ending operation to review, failure, or completion."""
    run = RealizationRunStore(run_dir)
    document = cast(
        dict[str, object],
        json.loads((run.run_dir / "inputs" / "approved-script.json").read_text(encoding="utf-8")),
    )
    spec = RunStore(run.run_dir).read_spec()
    operations = cast(list[dict[str, object]], spec["operations"])
    while True:
        status = run.rebuild()
        if status.status in {"awaiting_review", "failed", "completed"}:
            return status
        operation = operations[len(status.completed_operations)]
        instance_id = cast(str, operation["instance_id"])
        if operation["operation"] != "ending":
            raise ValueError(f"Unsupported ending operation: {operation['operation']}")
        targets = tuple(cast(list[str], operation["targets"]))
        try:
            materialization = materialize_solo_piano_ending(document, status.workspace)
        except (PianoTextureValidationError, ValueError):
            materialization = None
        if materialization is not None:
            RunStore(run.run_dir, max_calls=0).snapshot_json(
                "diagnostics/ending-material.json",
                ending_materialization_diagnostic(materialization),
            )
        run.record_checked_diff(
            instance_id,
            build_ending_checked_diff(document, status.workspace, targets),
        )


def advance_performance_realization_run(
    run_dir: Path, runner: ProposalRunner
) -> RealizationRunStatus:
    """Advance performance defaults and model waves to review, failure, or completion."""
    run = RealizationRunStore(run_dir)
    document = cast(
        dict[str, object],
        json.loads((run.run_dir / "inputs" / "approved-script.json").read_text(encoding="utf-8")),
    )
    spec = RunStore(run.run_dir).read_spec()
    operations = cast(list[dict[str, object]], spec["operations"])
    while True:
        status = run.rebuild()
        if status.status in {"awaiting_review", "failed", "completed"}:
            return status
        operation = operations[len(status.completed_operations)]
        instance_id = cast(str, operation["instance_id"])
        operation_name = cast(str, operation["operation"])
        targets = tuple(cast(list[str], operation["targets"]))
        if operation_name == "performance-defaults":
            run.record_checked_diff(
                instance_id,
                build_performance_defaults_checked_diff(document, status.workspace),
            )
            continue
        if operation_name != "performance":
            raise ValueError(f"Unsupported performance operation: {operation_name}")
        target_issues = _performance_target_issues(document, status.workspace, targets)
        if target_issues:
            run.record_checked_diff(
                instance_id,
                build_performance_checked_diff(document, status.workspace, targets, {}),
            )
            continue
        schema = performance_response_schema(document, status.workspace, targets)
        prompt = _performance_prompt(document, status.workspace, targets)
        schema_path = run.run_dir / "schemas" / f"{instance_id}.json"
        RunStore(run.run_dir, max_calls=cast(int, spec["max_calls"])).snapshot_json(
            schema_path.relative_to(run.run_dir), schema
        )
        response = run.execute_model_call(instance_id, prompt, schema_path, runner)
        run.record_checked_diff(
            instance_id,
            build_performance_checked_diff(document, status.workspace, targets, response),
        )


def replace_pending_review_with_response(
    run_dir: Path,
    response: object,
    *,
    actor: str,
) -> RealizationRunStatus:
    """Normalize a review replacement through the operation's regular checker."""
    run = RealizationRunStore(run_dir)
    status = run.rebuild()
    instance_id, original_diff = run.pending_checked_diff()
    document = cast(
        dict[str, object],
        json.loads((run.run_dir / "inputs" / "approved-script.json").read_text(encoding="utf-8")),
    )
    spec = RunStore(run.run_dir).read_spec()
    operations = cast(list[dict[str, object]], spec["operations"])
    operation = next(item for item in operations if item.get("instance_id") == instance_id)
    operation_name = cast(str, operation["operation"])
    targets = tuple(cast(list[str], operation["targets"]))
    if operation_name == "plan":
        replacement = build_plan_checked_diff(document, status.workspace, response)
    elif operation_name == "harmony":
        replacement = build_harmony_checked_diff(document, status.workspace, targets, response)
    elif operation_name == "melody":
        replacement = build_melody_checked_diff(document, status.workspace, targets, response)
    elif operation_name == "transition-melody":
        replacement = build_transition_melody_checked_diff(
            document, status.workspace, targets, response
        )
    elif operation_name == "accompaniment":
        replacement = build_accompaniment_checked_diff(
            document, status.workspace, targets, response
        )
    elif operation_name == "transition-accompaniment":
        replacement = build_transition_accompaniment_checked_diff(
            document, status.workspace, targets, response
        )
    elif operation_name == "performance":
        replacement = build_performance_checked_diff(document, status.workspace, targets, response)
    else:
        raise ValueError(f"Unsupported realization operation: {operation_name}")
    if replacement.validation_issues:
        raise ValueError(replacement.validation_issues[0].message)
    run.record_review(
        instance_id,
        target_diff_sha256=checked_diff_sha256(original_diff),
        decision="replace",
        actor=actor,
        replacement_diff=replacement,
    )
    return run.rebuild()


def plan_response_schema(document: Mapping[str, object]) -> dict[str, object]:
    """Build a strict response schema whose order is mapped to section keys by Python."""
    contrast_count = len(_contrast_section_keys(document))
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "tonal_center": {"type": "integer", "minimum": 0, "maximum": 11},
            "mode": {"type": "string", "enum": ["major", "minor"]},
            "contrast_descriptions": {
                "type": "array",
                "minItems": contrast_count,
                "maxItems": contrast_count,
                "items": {"type": "string", "minLength": 1},
            },
        },
        "required": ["tonal_center", "mode", "contrast_descriptions"],
        "additionalProperties": False,
    }


def build_plan_checked_diff(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    response: object,
) -> CheckedWorkspaceDiff:
    """Validate a plan response and compile it to a workspace-owned RFC 6902 path."""
    issues = _operation_context_issues(document, workspace)
    issues += _schema_issues(plan_response_schema(document), response)
    plan_value: object = None
    if not issues:
        choice = cast(Mapping[str, object], response)
        descriptions = cast(list[str], choice["contrast_descriptions"])
        plan_value = {
            "tonal_center": choice["tonal_center"],
            "mode": choice["mode"],
            "contrasts": dict(zip(_contrast_section_keys(document), descriptions, strict=True)),
        }
    return CheckedWorkspaceDiff(
        base_revision=workspace.revision,
        base_workspace_record_sha256=workspace.workspace_record_sha256,
        read_hashes=tuple(
            sorted(
                (
                    (
                        "/approved_script_sha256",
                        state_key_sha256(workspace, "/approved_script_sha256"),
                    ),
                    ("/plan", state_key_sha256(workspace, "/plan")),
                )
            )
        ),
        write_keys=("/plan",),
        patch=(PatchOperation("replace", "/plan", plan_value),),
        proposal_sha256=sha256_json(response),
        validation_issues=issues,
    )


def harmony_response_schema(
    document: Mapping[str, object], targets: tuple[str, ...]
) -> dict[str, object]:
    """Build a strict positional harmony response schema for caller-owned targets."""
    target_issues = _harmony_target_issues(document, targets)
    if target_issues:
        raise ValueError(target_issues[0].message)
    chord = {
        "type": "object",
        "properties": {
            "duration_units": {"type": "integer", "minimum": 1},
            "root_pitch_class": {"type": "integer", "minimum": 0, "maximum": 11},
            "quality": {"type": "string", "enum": list(_HARMONY_QUALITIES)},
        },
        "required": ["duration_units", "root_pitch_class", "quality"],
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "harmonies": {
                "type": "array",
                "minItems": len(targets),
                "maxItems": len(targets),
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "items": chord,
                },
            }
        },
        "required": ["harmonies"],
        "additionalProperties": False,
    }


def build_harmony_checked_diff(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
    response: object,
) -> CheckedWorkspaceDiff:
    """Validate positional harmony choices and bind them to material keys in Python."""
    issues = _operation_context_issues(document, workspace)
    issues += _harmony_target_issues(document, targets)
    if workspace.plan is None:
        issues += (
            ValidationIssue(
                IssueCode.SEMANTIC_INVALID,
                "Harmony generation requires a current overall plan",
                "/plan",
            ),
        )
    if not issues:
        issues += _schema_issues(harmony_response_schema(document, targets), response)
    harmonies: list[object] = []
    if not issues:
        harmonies = cast(list[object], cast(Mapping[str, object], response)["harmonies"])
    write_keys = tuple(f"/harmonies/{_pointer_token(target)}" for target in targets)
    current_harmonies = dict(workspace.harmonies)
    patch = tuple(
        PatchOperation(
            "replace" if target in current_harmonies else "add",
            key,
            harmonies[index] if index < len(harmonies) else [],
        )
        for index, (target, key) in enumerate(zip(targets, write_keys, strict=True))
    )
    return CheckedWorkspaceDiff(
        base_revision=workspace.revision,
        base_workspace_record_sha256=workspace.workspace_record_sha256,
        read_hashes=tuple(
            sorted(
                (
                    (
                        "/approved_script_sha256",
                        state_key_sha256(workspace, "/approved_script_sha256"),
                    ),
                    ("/plan", state_key_sha256(workspace, "/plan")),
                )
            )
        ),
        write_keys=write_keys,
        patch=patch,
        proposal_sha256=sha256_json(response),
        validation_issues=issues,
    )


def melody_response_schema(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
) -> dict[str, object]:
    """Build the strict positional schema for ordinary material melodies."""
    issues = _melody_target_issues(document, workspace, targets)
    if issues:
        raise ValueError(issues[0].message)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "melodies": {
                "type": "array",
                "minItems": len(targets),
                "maxItems": len(targets),
                "items": _melody_value_schema(),
            }
        },
        "required": ["melodies"],
        "additionalProperties": False,
    }


def build_melody_checked_diff(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
    response: object,
) -> CheckedWorkspaceDiff:
    """Validate ordinary melodies and bind positional values to material keys."""
    issues = _operation_context_issues(document, workspace)
    issues += _melody_target_issues(document, workspace, targets)
    if not issues:
        issues += _schema_issues(melody_response_schema(document, workspace, targets), response)
    melodies: list[object] = []
    if not issues:
        melodies = cast(list[object], cast(Mapping[str, object], response)["melodies"])
        issues += _melody_value_issues(workspace, targets, melodies)
    write_keys = tuple(f"/melodies/{_pointer_token(target)}" for target in targets)
    current = dict(workspace.melodies)
    patch = tuple(
        PatchOperation(
            "replace" if target in current else "add",
            key,
            melodies[index] if index < len(melodies) else {},
        )
        for index, (target, key) in enumerate(zip(targets, write_keys, strict=True))
    )
    common_reads = (
        ("/approved_script_sha256", state_key_sha256(workspace, "/approved_script_sha256")),
        ("/plan", state_key_sha256(workspace, "/plan")),
    )
    per_target_reads = {
        target: tuple(
            sorted(
                (
                    *common_reads,
                    (
                        f"/harmonies/{_pointer_token(target)}",
                        state_key_sha256(
                            workspace,
                            f"/harmonies/{_pointer_token(target)}",
                        ),
                    ),
                )
            )
        )
        for target in targets
        if target in dict(workspace.harmonies)
    }
    all_reads = tuple(sorted({item for reads in per_target_reads.values() for item in reads}))
    return CheckedWorkspaceDiff(
        base_revision=workspace.revision,
        base_workspace_record_sha256=workspace.workspace_record_sha256,
        read_hashes=all_reads,
        write_keys=write_keys,
        patch=patch,
        proposal_sha256=sha256_json(response),
        validation_issues=issues,
        write_read_hashes=tuple(
            (key, per_target_reads.get(target, ()))
            for target, key in zip(targets, write_keys, strict=True)
        ),
    )


def transition_melody_contexts(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
) -> tuple[dict[str, object], ...]:
    """Resolve placement-specific boundaries for transition melody targets."""
    issues = _transition_melody_target_issues(document, workspace, targets)
    if issues:
        raise ValueError(issues[0].message)
    script = cast(Mapping[str, object], document["script"])
    transitions = cast(Mapping[str, Mapping[str, object]], script["transitions"])
    placements = cast(Mapping[str, Mapping[str, object]], script["placements"])
    materials = cast(Mapping[str, Mapping[str, object]], script["materials"])
    melodies = dict(workspace.melodies)
    harmonies = dict(workspace.harmonies)
    contexts: list[dict[str, object]] = []
    for target in targets:
        transition = transitions[target]
        source_material = cast(
            str,
            placements[cast(str, transition["from_placement_id"])]["material_id"],
        )
        connector_material = cast(
            str,
            placements[cast(str, transition["connector_placement_id"])]["material_id"],
        )
        target_material = cast(
            str,
            placements[cast(str, transition["to_placement_id"])]["material_id"],
        )
        source_melody = melodies[source_material]
        target_melody = melodies[target_material]
        source_last_at = max(note.at_units for note in source_melody.notes)
        target_first_at = min(note.at_units for note in target_melody.notes)
        contexts.append(
            {
                "transition_key": target,
                "description": transition["description"],
                "source_material_key": source_material,
                "source_boundary": {
                    "foreground_voice": source_melody.foreground_voice,
                    "pitches": [
                        note.pitch
                        for note in source_melody.notes
                        if note.at_units == source_last_at
                    ],
                },
                "connector_material_key": connector_material,
                "connector_description": materials[connector_material]["description"],
                "connector_harmony": [
                    {
                        "duration_units": chord.duration_units,
                        "root_pitch_class": chord.root_pitch_class,
                        "quality": chord.quality,
                    }
                    for chord in harmonies[connector_material]
                ],
                "length_units": sum(
                    chord.duration_units for chord in harmonies[connector_material]
                ),
                "target_material_key": target_material,
                "target_boundary": {
                    "foreground_voice": target_melody.foreground_voice,
                    "pitches": [
                        note.pitch
                        for note in target_melody.notes
                        if note.at_units == target_first_at
                    ],
                },
            }
        )
    return tuple(contexts)


def transition_melody_response_schema(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
) -> dict[str, object]:
    """Build the positional schema for placement-specific transition melodies."""
    transition_melody_contexts(document, workspace, targets)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "transition_melodies": {
                "type": "array",
                "minItems": len(targets),
                "maxItems": len(targets),
                "items": _melody_value_schema(),
            }
        },
        "required": ["transition_melodies"],
        "additionalProperties": False,
    }


def build_transition_melody_checked_diff(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
    response: object,
) -> CheckedWorkspaceDiff:
    """Validate and bind placement-specific transition melodies."""
    issues = _operation_context_issues(document, workspace)
    issues += _transition_melody_target_issues(document, workspace, targets)
    contexts: tuple[dict[str, object], ...] = ()
    if not issues:
        contexts = transition_melody_contexts(document, workspace, targets)
    if not issues:
        issues += _schema_issues(
            transition_melody_response_schema(document, workspace, targets), response
        )
    melodies: list[object] = []
    if not issues:
        melodies = cast(list[object], cast(Mapping[str, object], response)["transition_melodies"])
        issues += _melody_lengths_issues(
            melodies,
            tuple(cast(int, context["length_units"]) for context in contexts),
            "transition_melodies",
        )
    write_keys = tuple(f"/transition_melodies/{_pointer_token(target)}" for target in targets)
    current = dict(workspace.transition_melodies)
    patch = tuple(
        PatchOperation(
            "replace" if target in current else "add",
            key,
            melodies[index] if index < len(melodies) else {},
        )
        for index, (target, key) in enumerate(zip(targets, write_keys, strict=True))
    )
    common_reads = (
        ("/approved_script_sha256", state_key_sha256(workspace, "/approved_script_sha256")),
        ("/plan", state_key_sha256(workspace, "/plan")),
    )
    per_target_reads: dict[str, tuple[tuple[str, str], ...]] = {}
    if not issues:
        for target, context in zip(targets, contexts, strict=True):
            dependency_keys = {
                f"/harmonies/{_pointer_token(cast(str, context['connector_material_key']))}",
                f"/melodies/{_pointer_token(cast(str, context['source_material_key']))}",
                f"/melodies/{_pointer_token(cast(str, context['target_material_key']))}",
            }
            per_target_reads[target] = tuple(
                sorted(
                    (
                        *common_reads,
                        *((key, state_key_sha256(workspace, key)) for key in dependency_keys),
                    )
                )
            )
    all_reads = tuple(sorted({item for reads in per_target_reads.values() for item in reads}))
    return CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        all_reads,
        write_keys,
        patch,
        sha256_json(response),
        issues,
        tuple(
            (key, per_target_reads.get(target, ()))
            for target, key in zip(targets, write_keys, strict=True)
        ),
    )


def accompaniment_response_schema(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
) -> dict[str, object]:
    """Build the positional schema for coarse ordinary accompaniment choices."""
    issues = _accompaniment_target_issues(document, workspace, targets)
    if issues:
        raise ValueError(issues[0].message)
    return _accompaniment_collection_schema("accompaniments", len(targets))


def _accompaniment_collection_schema(field: str, count: int) -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            field: {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": {
                        "events": {
                            "type": "array",
                            "minItems": 1,
                            "items": _accompaniment_request_event_schema(),
                        }
                    },
                    "required": ["events"],
                    "additionalProperties": False,
                },
            }
        },
        "required": [field],
        "additionalProperties": False,
    }


def build_accompaniment_checked_diff(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
    response: object,
) -> CheckedWorkspaceDiff:
    """Place coarse ordinary accompaniments and bind them to material keys."""
    issues = _operation_context_issues(document, workspace)
    issues += _accompaniment_target_issues(document, workspace, targets)
    if not issues:
        issues += _schema_issues(
            accompaniment_response_schema(document, workspace, targets), response
        )
    serialized_values: list[dict[str, object]] = []
    if not issues:
        raw_values = cast(
            list[Mapping[str, object]], cast(Mapping[str, object], response)["accompaniments"]
        )
        harmonies = dict(workspace.harmonies)
        melodies = dict(workspace.melodies)
        assert workspace.plan is not None
        for index, (target, raw_value) in enumerate(zip(targets, raw_values, strict=True)):
            requests = tuple(
                _parse_accompaniment_request(cast(Mapping[str, object], event))
                for event in cast(list[object], raw_value["events"])
            )
            try:
                placement = place_workspace_accompaniment(
                    material_key=target,
                    tonal_center=workspace.plan.tonal_center,
                    mode=workspace.plan.mode,
                    harmonies=harmonies[target],
                    melody=melodies[target],
                    requests=requests,
                )
            except (PianoTextureValidationError, ValueError) as error:
                issues += (
                    ValidationIssue(
                        IssueCode.MODEL_OUTPUT_INVALID,
                        str(error),
                        f"/response/accompaniments/{index}",
                    ),
                )
                break
            if placement.value is None:
                issues += (
                    ValidationIssue(
                        IssueCode.MODEL_OUTPUT_INVALID,
                        placement.reason or placement.status,
                        f"/response/accompaniments/{index}",
                    ),
                )
                break
            serialized_values.append(_accompaniment_value_dict(placement.value))
    write_keys = tuple(f"/accompaniments/{_pointer_token(target)}" for target in targets)
    current = dict(workspace.accompaniments)
    patch = tuple(
        PatchOperation(
            "replace" if target in current else "add",
            key,
            serialized_values[index] if index < len(serialized_values) else {},
        )
        for index, (target, key) in enumerate(zip(targets, write_keys, strict=True))
    )
    common_reads = (
        ("/approved_script_sha256", state_key_sha256(workspace, "/approved_script_sha256")),
        ("/plan", state_key_sha256(workspace, "/plan")),
    )
    per_target_reads = {
        target: tuple(
            sorted(
                (
                    *common_reads,
                    (
                        f"/harmonies/{_pointer_token(target)}",
                        state_key_sha256(workspace, f"/harmonies/{_pointer_token(target)}"),
                    ),
                    (
                        f"/melodies/{_pointer_token(target)}",
                        state_key_sha256(workspace, f"/melodies/{_pointer_token(target)}"),
                    ),
                )
            )
        )
        for target in targets
        if target in dict(workspace.harmonies) and target in dict(workspace.melodies)
    }
    all_reads = tuple(sorted({item for reads in per_target_reads.values() for item in reads}))
    return CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        all_reads,
        write_keys,
        patch,
        sha256_json(response),
        issues,
        tuple(
            (key, per_target_reads.get(target, ()))
            for target, key in zip(targets, write_keys, strict=True)
        ),
    )


def transition_accompaniment_contexts(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
) -> tuple[dict[str, object], ...]:
    """Resolve relation-specific melody and accompaniment boundaries for connectors."""
    issues = _transition_accompaniment_target_issues(document, workspace, targets)
    if issues:
        raise ValueError(issues[0].message)
    script = cast(Mapping[str, object], document["script"])
    transitions = cast(Mapping[str, Mapping[str, object]], script["transitions"])
    placements = cast(Mapping[str, Mapping[str, object]], script["placements"])
    harmonies = dict(workspace.harmonies)
    transition_melodies = dict(workspace.transition_melodies)
    accompaniments = dict(workspace.accompaniments)
    contexts: list[dict[str, object]] = []
    for target in targets:
        transition = transitions[target]
        source_material = cast(
            str, placements[cast(str, transition["from_placement_id"])]["material_id"]
        )
        connector_material = cast(
            str, placements[cast(str, transition["connector_placement_id"])]["material_id"]
        )
        target_material = cast(
            str, placements[cast(str, transition["to_placement_id"])]["material_id"]
        )
        source = accompaniments[source_material]
        destination = accompaniments[target_material]
        source_last = max(note.at_units for note in source.notes)
        destination_first = min(note.at_units for note in destination.notes)
        transition_melody = transition_melodies[target]
        contexts.append(
            {
                "transition_key": target,
                "description": transition["description"],
                "source_material_key": source_material,
                "source_boundary": _accompaniment_boundary(source, source_last),
                "connector_material_key": connector_material,
                "connector_harmony": [
                    {
                        "duration_units": chord.duration_units,
                        "root_pitch_class": chord.root_pitch_class,
                        "quality": chord.quality,
                    }
                    for chord in harmonies[connector_material]
                ],
                "transition_melody": {
                    "foreground_voice": transition_melody.foreground_voice,
                    "notes": [
                        {
                            "at_units": note.at_units,
                            "duration_units": note.duration_units,
                            "pitch": note.pitch,
                        }
                        for note in transition_melody.notes
                    ],
                },
                "target_material_key": target_material,
                "target_boundary": _accompaniment_boundary(destination, destination_first),
            }
        )
    return tuple(contexts)


def transition_accompaniment_response_schema(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
) -> dict[str, object]:
    """Build the positional schema for relation-specific transition accompaniments."""
    transition_accompaniment_contexts(document, workspace, targets)
    return _accompaniment_collection_schema("transition_accompaniments", len(targets))


def build_transition_accompaniment_checked_diff(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
    response: object,
) -> CheckedWorkspaceDiff:
    """Place and bind transition accompaniments to relation keys."""
    issues = _operation_context_issues(document, workspace)
    issues += _transition_accompaniment_target_issues(document, workspace, targets)
    contexts: tuple[dict[str, object], ...] = ()
    if not issues:
        contexts = transition_accompaniment_contexts(document, workspace, targets)
        issues += _schema_issues(
            transition_accompaniment_response_schema(document, workspace, targets), response
        )
    serialized_values: list[dict[str, object]] = []
    if not issues:
        raw_values = cast(
            list[Mapping[str, object]],
            cast(Mapping[str, object], response)["transition_accompaniments"],
        )
        harmonies = dict(workspace.harmonies)
        transition_melodies = dict(workspace.transition_melodies)
        assert workspace.plan is not None
        for index, (target, context, raw_value) in enumerate(
            zip(targets, contexts, raw_values, strict=True)
        ):
            requests = tuple(
                _parse_accompaniment_request(cast(Mapping[str, object], event))
                for event in cast(list[object], raw_value["events"])
            )
            connector_material = cast(str, context["connector_material_key"])
            try:
                placement = place_workspace_accompaniment(
                    material_key=f"transition-{target}",
                    tonal_center=workspace.plan.tonal_center,
                    mode=workspace.plan.mode,
                    harmonies=harmonies[connector_material],
                    melody=transition_melodies[target],
                    requests=requests,
                )
            except (PianoTextureValidationError, ValueError) as error:
                issues += (
                    ValidationIssue(
                        IssueCode.MODEL_OUTPUT_INVALID,
                        str(error),
                        f"/response/transition_accompaniments/{index}",
                    ),
                )
                break
            if placement.value is None:
                issues += (
                    ValidationIssue(
                        IssueCode.MODEL_OUTPUT_INVALID,
                        placement.reason or placement.status,
                        f"/response/transition_accompaniments/{index}",
                    ),
                )
                break
            serialized_values.append(_accompaniment_value_dict(placement.value))
    write_keys = tuple(f"/transition_accompaniments/{_pointer_token(target)}" for target in targets)
    current = dict(workspace.transition_accompaniments)
    patch = tuple(
        PatchOperation(
            "replace" if target in current else "add",
            key,
            serialized_values[index] if index < len(serialized_values) else {},
        )
        for index, (target, key) in enumerate(zip(targets, write_keys, strict=True))
    )
    common_reads = (
        ("/approved_script_sha256", state_key_sha256(workspace, "/approved_script_sha256")),
        ("/plan", state_key_sha256(workspace, "/plan")),
    )
    per_target_reads: dict[str, tuple[tuple[str, str], ...]] = {}
    if not issues:
        for target, context in zip(targets, contexts, strict=True):
            dependency_keys = {
                f"/harmonies/{_pointer_token(cast(str, context['connector_material_key']))}",
                f"/transition_melodies/{_pointer_token(target)}",
                f"/accompaniments/{_pointer_token(cast(str, context['source_material_key']))}",
                f"/accompaniments/{_pointer_token(cast(str, context['target_material_key']))}",
            }
            per_target_reads[target] = tuple(
                sorted(
                    (
                        *common_reads,
                        *((key, state_key_sha256(workspace, key)) for key in dependency_keys),
                    )
                )
            )
    all_reads = tuple(sorted({item for reads in per_target_reads.values() for item in reads}))
    return CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        all_reads,
        write_keys,
        patch,
        sha256_json(response),
        issues,
        tuple(
            (key, per_target_reads.get(target, ()))
            for target, key in zip(targets, write_keys, strict=True)
        ),
    )


def build_ending_checked_diff(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
) -> CheckedWorkspaceDiff:
    """Jointly derive the ending harmony, melody, and accompaniment."""
    issues = _operation_context_issues(document, workspace)
    materialization = None
    if len(targets) != 1 or len(set(targets)) != len(targets):
        issues += (
            ValidationIssue(
                IssueCode.SEMANTIC_INVALID,
                "Ending materialization requires exactly one target",
                "/targets",
            ),
        )
    if not issues:
        try:
            materialization = materialize_solo_piano_ending(document, workspace)
        except (PianoTextureValidationError, ValueError) as error:
            issues += (
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    str(error),
                    "/ending",
                ),
            )
    if materialization is not None and targets != (materialization.material_key,):
        issues += (
            ValidationIssue(
                IssueCode.SEMANTIC_INVALID,
                "Ending target does not match the final release material",
                "/targets/0",
            ),
        )
        materialization = None

    write_keys = tuple(
        key
        for target in targets
        for key in (
            f"/harmonies/{_pointer_token(target)}",
            f"/melodies/{_pointer_token(target)}",
            f"/accompaniments/{_pointer_token(target)}",
        )
    )
    payload: dict[str, object] = {}
    if materialization is not None:
        payload = {
            "harmonies": [
                {
                    "duration_units": chord.duration_units,
                    "root_pitch_class": chord.root_pitch_class,
                    "quality": chord.quality,
                }
                for chord in materialization.harmonies
            ],
            "melody": {
                "foreground_voice": materialization.melody.foreground_voice,
                "notes": [
                    {
                        "at_units": note.at_units,
                        "duration_units": note.duration_units,
                        "pitch": note.pitch,
                    }
                    for note in materialization.melody.notes
                ],
            },
            "accompaniment": _accompaniment_value_dict(materialization.accompaniment),
            "diagnostic": {
                "common_grid_units_per_weight": (materialization.common_grid_units_per_weight),
                "material_scale_factors": dict(materialization.material_scale_factors),
                "nominal_hold_ms": materialization.nominal_hold_ms,
                "placement_candidate_evaluation_count": (
                    materialization.placement_candidate_evaluation_count
                ),
            },
        }
    current_by_collection = {
        "harmonies": dict(workspace.harmonies),
        "melodies": dict(workspace.melodies),
        "accompaniments": dict(workspace.accompaniments),
    }
    values = (
        payload.get("harmonies", []),
        payload.get("melody", {}),
        payload.get("accompaniment", {}),
    )
    patch = tuple(
        PatchOperation(
            "replace" if target in current_by_collection[collection] else "add",
            f"/{collection}/{_pointer_token(target)}",
            values[index],
        )
        for target in targets
        for index, collection in enumerate(("harmonies", "melodies", "accompaniments"))
    )
    dependencies: tuple[tuple[str, str], ...] = ()
    if materialization is not None:
        dependency_keys = (
            "/approved_script_sha256",
            "/plan",
            f"/melodies/{_pointer_token(materialization.previous_material_key)}",
        )
        dependencies = tuple(
            sorted((key, state_key_sha256(workspace, key)) for key in dependency_keys)
        )
    return CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        dependencies,
        write_keys,
        patch,
        sha256_json(
            {
                "operation": "deterministic-solo-piano-ending-v1",
                "targets": list(targets),
                "result": payload,
            }
        ),
        issues,
        tuple((key, dependencies) for key in write_keys),
    )


def build_performance_defaults_checked_diff(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
) -> CheckedWorkspaceDiff:
    """Fill only non-directed performance nodes with explicit inherited defaults."""
    issues = _operation_context_issues(document, workspace)
    schedule = None
    if workspace.schema_version != 4:
        issues += (
            ValidationIssue(
                IssueCode.SEMANTIC_INVALID,
                "Performance generation requires a version 4 workspace",
                "/schema_version",
            ),
        )
    if not issues:
        try:
            schedule = performance_operation_schedule(document, workspace)
        except ValueError as error:
            issues += (ValidationIssue(IssueCode.UNREPRESENTABLE, str(error), "/performance"),)
    targets = schedule.default_node_ids if schedule is not None else ()
    write_keys = tuple(f"/performances/{_pointer_token(target)}" for target in targets)
    inherited = {
        "timing_profile": None,
        "timing_amount": None,
        "dynamics_profile": None,
        "articulation_profile": None,
        "coordination_profile": None,
        "pedal_profile": None,
    }
    current = dict(workspace.performances)
    patch = tuple(
        PatchOperation(
            "replace" if target in current else "add",
            key,
            inherited,
        )
        for target, key in zip(targets, write_keys, strict=True)
    )
    dependency_keys = ("/approved_script_sha256", "/plan")
    dependencies = (
        tuple(sorted((key, state_key_sha256(workspace, key)) for key in dependency_keys))
        if not issues
        else ()
    )
    return CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        dependencies,
        write_keys,
        patch,
        sha256_json(
            {
                "operation": "deterministic-performance-defaults-v1",
                "targets": list(targets),
                "value": inherited,
            }
        ),
        issues,
        tuple((key, dependencies) for key in write_keys),
    )


def performance_response_schema(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
) -> dict[str, object]:
    """Build a positional schema for per-node performance choices."""
    issues = _performance_target_issues(document, workspace, targets)
    if issues:
        raise ValueError(issues[0].message)
    nullable = lambda choices: {"enum": [None, *choices]}  # noqa: E731
    item = {
        "type": "object",
        "properties": {
            "timing_profile": nullable(["neutral", "savor", "flow", "build", "release"]),
            "timing_amount": nullable(["subtle", "moderate"]),
            "dynamics_profile": nullable(["steady", "shape", "build", "release"]),
            "articulation_profile": nullable(["score", "legato", "light"]),
            "coordination_profile": nullable(["score", "rolled", "aligned"]),
            "pedal_profile": nullable(["none", "phrase_legato", "harmony_legato", "clear"]),
        },
        "required": [
            "timing_profile",
            "timing_amount",
            "dynamics_profile",
            "articulation_profile",
            "coordination_profile",
            "pedal_profile",
        ],
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "performances": {
                "type": "array",
                "minItems": len(targets),
                "maxItems": len(targets),
                "items": item,
            }
        },
        "required": ["performances"],
        "additionalProperties": False,
    }


def build_performance_checked_diff(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
    response: object,
) -> CheckedWorkspaceDiff:
    """Validate positional performance choices and bind them to Python-owned nodes."""
    issues = _performance_target_issues(document, workspace, targets)
    values: list[Mapping[str, object]] = []
    if not issues:
        issues += _schema_issues(
            performance_response_schema(document, workspace, targets), response
        )
    if not issues:
        values = cast(
            list[Mapping[str, object]], cast(Mapping[str, object], response)["performances"]
        )
        try:
            plan = performance_piece_plan(document, workspace)
            proposed = dict(workspace.performances)
            for target, value in zip(targets, values, strict=True):
                proposed[target] = _performance_value(value)
            validate_performance_spec(
                plan,
                PerformanceSpec(
                    performance_id="workspace-performance",
                    target_duration_ms=180_000,
                    default_velocity=64,
                    timing_budget_id="narrative-v2",
                    node_performances=tuple(
                        NodePerformance(
                            node_id=node_id,
                            timing_profile=value.timing_profile,
                            timing_amount=value.timing_amount,
                            dynamics_profile=value.dynamics_profile,
                            articulation_profile=value.articulation_profile,
                            coordination_profile=value.coordination_profile,
                            pedal_profile=value.pedal_profile,
                        )
                        for node_id, value in sorted(proposed.items())
                    ),
                    key_release_percent=100,
                    velocity_policy_id="foreground-accompaniment-harmony-shape-v1",
                ),
            )
        except (PipelineValidationError, ValueError) as error:
            issues += (
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    str(error),
                    "/response/performances",
                ),
            )
    write_keys = tuple(f"/performances/{_pointer_token(target)}" for target in targets)
    current = dict(workspace.performances)
    patch = tuple(
        PatchOperation(
            "replace" if target in current else "add",
            key,
            dict(value),
        )
        for target, key, value in zip(targets, write_keys, values, strict=False)
    )
    per_target_dependencies: dict[str, tuple[tuple[str, str], ...]] = {}
    if not issues:
        try:
            for target in targets:
                per_target_dependencies[target] = tuple(
                    (key, state_key_sha256(workspace, key))
                    for key in performance_read_keys(document, workspace, target)
                )
        except ValueError as error:
            issues += (ValidationIssue(IssueCode.UNREPRESENTABLE, str(error), "/performance"),)
    dependencies = tuple(
        sorted(
            {
                item
                for target_dependencies in per_target_dependencies.values()
                for item in target_dependencies
            }
        )
    )
    return CheckedWorkspaceDiff(
        workspace.revision,
        workspace.workspace_record_sha256,
        dependencies,
        write_keys,
        patch,
        sha256_json(response),
        issues,
        tuple(
            (key, per_target_dependencies.get(target, ()))
            for target, key in zip(targets, write_keys, strict=True)
        ),
    )


def _performance_value(value: Mapping[str, object]) -> PerformanceValue:
    return PerformanceValue(
        timing_profile=cast(str | None, value["timing_profile"]),
        timing_amount=cast(str | None, value["timing_amount"]),
        dynamics_profile=cast(str | None, value["dynamics_profile"]),
        articulation_profile=cast(str | None, value["articulation_profile"]),
        coordination_profile=cast(str | None, value["coordination_profile"]),
        pedal_profile=cast(str | None, value["pedal_profile"]),
    )


def _performance_target_issues(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
) -> tuple[ValidationIssue, ...]:
    issues = _operation_context_issues(document, workspace)
    if workspace.schema_version != 4:
        issues += (
            ValidationIssue(
                IssueCode.SEMANTIC_INVALID,
                "Performance generation requires a version 4 workspace",
                "/schema_version",
            ),
        )
    if not targets or len(set(targets)) != len(targets):
        issues += (
            ValidationIssue(
                IssueCode.SEMANTIC_INVALID,
                "Performance targets must be a non-empty unique sequence",
                "/targets",
            ),
        )
    if issues:
        return issues
    try:
        schedule = performance_operation_schedule(document, workspace)
    except ValueError as error:
        return (ValidationIssue(IssueCode.UNREPRESENTABLE, str(error), "/performance"),)
    allowed_operations = (schedule.absolute_node_ids, *schedule.comparative_waves)
    if targets not in allowed_operations:
        return (
            ValidationIssue(
                IssueCode.SEMANTIC_INVALID,
                "Performance targets do not match a dependency-ordered operation",
                "/targets",
            ),
        )
    try:
        for target in targets:
            performance_read_keys(document, workspace, target)
    except ValueError as error:
        return (ValidationIssue(IssueCode.UNREPRESENTABLE, str(error), "/performance"),)
    return ()


def _accompaniment_boundary(value: AccompanimentValue, at_units: int) -> dict[str, object]:
    return {
        "foreground_voice": value.foreground_voice,
        "notes": [
            {
                "pitch": note.pitch,
                "duration_units": note.duration_units,
                "preferred_register_zone": note.preferred_register_zone,
            }
            for note in value.notes
            if note.at_units == at_units
        ],
    }


def _accompaniment_request_event_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "at_units": {"type": "integer", "minimum": 0},
            "preferred_duration_units": {"type": "integer", "minimum": 1},
            "degree": {"type": "string", "enum": ["root", "third", "fifth", "seventh"]},
            "preferred_register_zone": {
                "type": "string",
                "enum": ["bass", "low", "middle", "high"],
            },
            "articulations": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["normal", "staccato", "tenuto", "accent"],
                },
            },
        },
        "required": [
            "at_units",
            "preferred_duration_units",
            "degree",
            "preferred_register_zone",
            "articulations",
        ],
        "additionalProperties": False,
    }


def _parse_accompaniment_request(value: Mapping[str, object]) -> AccompanimentRequestEvent:
    return AccompanimentRequestEvent(
        cast(int, value["at_units"]),
        cast(int, value["preferred_duration_units"]),
        cast(str, value["degree"]),
        cast(str, value["preferred_register_zone"]),
        tuple(cast(list[str], value["articulations"])),
    )


def _accompaniment_value_dict(value: AccompanimentValue) -> dict[str, object]:
    return {
        "foreground_voice": value.foreground_voice,
        "notes": [
            {
                "at_units": note.at_units,
                "preferred_duration_units": note.preferred_duration_units,
                "duration_units": note.duration_units,
                "degree": note.degree,
                "preferred_register_zone": note.preferred_register_zone,
                "pitch": note.pitch,
                "articulations": list(note.articulations),
            }
            for note in value.notes
        ],
    }


def _operation_context_issues(
    document: Mapping[str, object], workspace: RealizationWorkspace
) -> tuple[ValidationIssue, ...]:
    validation = check_realization_script(document)
    if not validation.valid:
        return validation.issues
    if realization_script_sha256(document) != workspace.approved_script_sha256:
        return (
            ValidationIssue(
                IssueCode.LINEAGE_MISMATCH,
                "The workspace belongs to a different approved script",
                "/approved_script_sha256",
            ),
        )
    return ()


def _melody_value_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "foreground_voice": {"type": "string", "enum": ["upper", "lower"]},
            "notes": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "at_units": {"type": "integer", "minimum": 0},
                        "duration_units": {"type": "integer", "minimum": 1},
                        "pitch": {"type": "integer", "minimum": 21, "maximum": 108},
                    },
                    "required": ["at_units", "duration_units", "pitch"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["foreground_voice", "notes"],
        "additionalProperties": False,
    }


def _schema_issues(schema: Mapping[str, object], response: object) -> tuple[ValidationIssue, ...]:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(response),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return ()
    error = errors[0]
    pointer = "".join(
        f"/{str(part).replace('~', '~0').replace('/', '~1')}" for part in error.absolute_path
    )
    return (
        ValidationIssue(
            IssueCode.MODEL_OUTPUT_INVALID,
            error.message,
            f"/response{pointer}",
        ),
    )


def _harmony_target_issues(
    document: Mapping[str, object], targets: tuple[str, ...]
) -> tuple[ValidationIssue, ...]:
    if not targets or len(set(targets)) != len(targets):
        return (
            ValidationIssue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "Harmony targets must be a non-empty unique material sequence",
                "/targets",
            ),
        )
    script = cast(Mapping[str, object], document["script"])
    materials = cast(Mapping[str, Mapping[str, object]], script["materials"])
    for index, target in enumerate(targets):
        material = materials.get(target)
        if material is None:
            return (
                ValidationIssue(
                    IssueCode.NOT_FOUND,
                    f"Unknown harmony target: {target}",
                    f"/targets/{index}",
                ),
            )
        if material["kind"] == "ending":
            return (
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "Ending harmony is generated deterministically by Python",
                    f"/targets/{index}",
                ),
            )
    return ()


def _melody_target_issues(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
) -> tuple[ValidationIssue, ...]:
    if not targets or len(set(targets)) != len(targets):
        return (
            ValidationIssue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "Melody targets must be a non-empty unique material sequence",
                "/targets",
            ),
        )
    if workspace.plan is None:
        return (
            ValidationIssue(
                IssueCode.SEMANTIC_INVALID,
                "Melody generation requires a current overall plan",
                "/plan",
            ),
        )
    script = cast(Mapping[str, object], document["script"])
    materials = cast(Mapping[str, Mapping[str, object]], script["materials"])
    harmonies = dict(workspace.harmonies)
    for index, target in enumerate(targets):
        material = materials.get(target)
        if material is None:
            return (
                ValidationIssue(
                    IssueCode.NOT_FOUND,
                    f"Unknown melody target: {target}",
                    f"/targets/{index}",
                ),
            )
        if material["kind"] in {"transition", "ending"}:
            return (
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "Ordinary melody targets cannot be transition or ending materials",
                    f"/targets/{index}",
                ),
            )
        if target not in harmonies:
            return (
                ValidationIssue(
                    IssueCode.NOT_FOUND,
                    f"Melody target has no current harmony: {target}",
                    f"/harmonies/{_pointer_token(target)}",
                ),
            )
    return ()


def _melody_value_issues(
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
    melodies: list[object],
) -> tuple[ValidationIssue, ...]:
    harmonies = dict(workspace.harmonies)
    return _melody_lengths_issues(
        melodies,
        tuple(sum(chord.duration_units for chord in harmonies[target]) for target in targets),
        "melodies",
    )


def _accompaniment_target_issues(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
) -> tuple[ValidationIssue, ...]:
    if workspace.schema_version not in {3, 4}:
        return (
            ValidationIssue(
                IssueCode.SEMANTIC_INVALID,
                "Accompaniment generation requires a version 3 or 4 workspace",
                "/schema_version",
            ),
        )
    if not targets or len(set(targets)) != len(targets):
        return (
            ValidationIssue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "Accompaniment targets must be a non-empty unique material sequence",
                "/targets",
            ),
        )
    if workspace.plan is None:
        return (
            ValidationIssue(
                IssueCode.NOT_FOUND,
                "Accompaniment generation requires a current overall plan",
                "/plan",
            ),
        )
    script = cast(Mapping[str, object], document["script"])
    materials = cast(Mapping[str, Mapping[str, object]], script["materials"])
    harmonies = dict(workspace.harmonies)
    melodies = dict(workspace.melodies)
    for index, target in enumerate(targets):
        material = materials.get(target)
        if material is None:
            return (
                ValidationIssue(
                    IssueCode.NOT_FOUND,
                    f"Unknown accompaniment target: {target}",
                    f"/targets/{index}",
                ),
            )
        if material["kind"] in {"transition", "ending"}:
            return (
                ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "Ordinary accompaniment targets cannot be transition or ending materials",
                    f"/targets/{index}",
                ),
            )
        if target not in harmonies:
            return (
                ValidationIssue(
                    IssueCode.NOT_FOUND,
                    f"Accompaniment target has no current harmony: {target}",
                    f"/harmonies/{_pointer_token(target)}",
                ),
            )
        if target not in melodies:
            return (
                ValidationIssue(
                    IssueCode.NOT_FOUND,
                    f"Accompaniment target has no current melody: {target}",
                    f"/melodies/{_pointer_token(target)}",
                ),
            )
    return ()


def _transition_accompaniment_target_issues(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
) -> tuple[ValidationIssue, ...]:
    if workspace.schema_version not in {3, 4}:
        return (
            ValidationIssue(
                IssueCode.SEMANTIC_INVALID,
                "Transition accompaniment generation requires a version 3 or 4 workspace",
                "/schema_version",
            ),
        )
    if not targets or len(set(targets)) != len(targets):
        return (
            ValidationIssue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "Transition accompaniment targets must be a non-empty unique sequence",
                "/targets",
            ),
        )
    if workspace.plan is None:
        return (
            ValidationIssue(
                IssueCode.NOT_FOUND,
                "Transition accompaniment generation requires a current overall plan",
                "/plan",
            ),
        )
    script = cast(Mapping[str, object], document["script"])
    transitions = cast(Mapping[str, Mapping[str, object]], script["transitions"])
    placements = cast(Mapping[str, Mapping[str, object]], script["placements"])
    harmonies = dict(workspace.harmonies)
    transition_melodies = dict(workspace.transition_melodies)
    accompaniments = dict(workspace.accompaniments)
    for index, target in enumerate(targets):
        transition = transitions.get(target)
        if transition is None:
            return (
                ValidationIssue(
                    IssueCode.NOT_FOUND,
                    f"Unknown transition accompaniment target: {target}",
                    f"/targets/{index}",
                ),
            )
        try:
            source_material = cast(
                str, placements[cast(str, transition["from_placement_id"])]["material_id"]
            )
            connector_material = cast(
                str,
                placements[cast(str, transition["connector_placement_id"])]["material_id"],
            )
            target_material = cast(
                str, placements[cast(str, transition["to_placement_id"])]["material_id"]
            )
        except KeyError:
            return (
                ValidationIssue(
                    IssueCode.NOT_FOUND,
                    "A transition accompaniment placement cannot be resolved",
                    f"/script/transitions/{_pointer_token(target)}",
                ),
            )
        dependencies = (
            (connector_material in harmonies, f"/harmonies/{_pointer_token(connector_material)}"),
            (target in transition_melodies, f"/transition_melodies/{_pointer_token(target)}"),
            (
                source_material in accompaniments,
                f"/accompaniments/{_pointer_token(source_material)}",
            ),
            (
                target_material in accompaniments,
                f"/accompaniments/{_pointer_token(target_material)}",
            ),
        )
        for present, path in dependencies:
            if not present:
                return (
                    ValidationIssue(
                        IssueCode.NOT_FOUND,
                        "A transition accompaniment dependency is missing",
                        path,
                    ),
                )
    return ()


def _melody_lengths_issues(
    melodies: list[object],
    lengths: tuple[int, ...],
    response_field: str,
) -> tuple[ValidationIssue, ...]:
    for target_index, (raw_melody, length_units) in enumerate(zip(melodies, lengths, strict=True)):
        melody = cast(Mapping[str, object], raw_melody)
        notes = cast(list[Mapping[str, object]], melody["notes"])
        by_pitch: dict[int, list[tuple[int, int, int]]] = {}
        for note_index, note in enumerate(notes):
            at_units = cast(int, note["at_units"])
            end_units = at_units + cast(int, note["duration_units"])
            if end_units > length_units:
                return (
                    ValidationIssue(
                        IssueCode.MODEL_OUTPUT_INVALID,
                        "A melody note exceeds the material harmony length",
                        f"/response/{response_field}/{target_index}/notes/{note_index}",
                    ),
                )
            by_pitch.setdefault(cast(int, note["pitch"]), []).append(
                (at_units, end_units, note_index)
            )
        for same_pitch in by_pitch.values():
            previous_end = -1
            for at_units, end_units, note_index in sorted(same_pitch):
                if at_units < previous_end:
                    return (
                        ValidationIssue(
                            IssueCode.MODEL_OUTPUT_INVALID,
                            "The same melody pitch cannot overlap within one material",
                            f"/response/{response_field}/{target_index}/notes/{note_index}",
                        ),
                    )
                previous_end = end_units
    return ()


def _transition_melody_target_issues(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
) -> tuple[ValidationIssue, ...]:
    if not targets or len(set(targets)) != len(targets):
        return (
            ValidationIssue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "Transition melody targets must be a non-empty unique sequence",
                "/targets",
            ),
        )
    if workspace.plan is None:
        return (
            ValidationIssue(
                IssueCode.SEMANTIC_INVALID,
                "Transition melody generation requires a current overall plan",
                "/plan",
            ),
        )
    script = cast(Mapping[str, object], document["script"])
    transitions = cast(Mapping[str, Mapping[str, object]], script["transitions"])
    placements = cast(Mapping[str, Mapping[str, object]], script["placements"])
    materials = cast(Mapping[str, Mapping[str, object]], script["materials"])
    harmonies = dict(workspace.harmonies)
    melodies = dict(workspace.melodies)
    for index, target in enumerate(targets):
        transition = transitions.get(target)
        if transition is None:
            return (
                ValidationIssue(
                    IssueCode.NOT_FOUND,
                    f"Unknown transition melody target: {target}",
                    f"/targets/{index}",
                ),
            )
        try:
            source_material = cast(
                str,
                placements[cast(str, transition["from_placement_id"])]["material_id"],
            )
            connector_material = cast(
                str,
                placements[cast(str, transition["connector_placement_id"])]["material_id"],
            )
            target_material = cast(
                str,
                placements[cast(str, transition["to_placement_id"])]["material_id"],
            )
        except KeyError:
            return (
                ValidationIssue(
                    IssueCode.NOT_FOUND,
                    "A transition placement cannot be resolved",
                    f"/script/transitions/{_pointer_token(target)}",
                ),
            )
        if materials[connector_material]["kind"] != "transition":
            return (
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    "A connector placement must use a transition material",
                    f"/script/transitions/{_pointer_token(target)}/connector_placement_id",
                ),
            )
        if connector_material not in harmonies:
            return (
                ValidationIssue(
                    IssueCode.NOT_FOUND,
                    "A connector material has no current harmony",
                    f"/harmonies/{_pointer_token(connector_material)}",
                ),
            )
        for boundary_material in (source_material, target_material):
            if boundary_material not in melodies:
                return (
                    ValidationIssue(
                        IssueCode.NOT_FOUND,
                        "A transition boundary has no current ordinary melody",
                        f"/melodies/{_pointer_token(boundary_material)}",
                    ),
                )
    return ()


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _non_ending_material_keys(document: Mapping[str, object]) -> tuple[str, ...]:
    script = cast(Mapping[str, object], document["script"])
    materials = cast(Mapping[str, Mapping[str, object]], script["materials"])
    return tuple(sorted(key for key, material in materials.items() if material["kind"] != "ending"))


def _main_melody_material_keys(document: Mapping[str, object]) -> tuple[str, ...]:
    script = cast(Mapping[str, object], document["script"])
    materials = cast(Mapping[str, Mapping[str, object]], script["materials"])
    return tuple(
        sorted(
            key
            for key, material in materials.items()
            if material["kind"] not in {"transition", "ending"}
        )
    )


def _transition_keys(document: Mapping[str, object]) -> tuple[str, ...]:
    script = cast(Mapping[str, object], document["script"])
    transitions = cast(Mapping[str, object], script["transitions"])
    return tuple(sorted(transitions))


def _ending_material_keys(document: Mapping[str, object]) -> tuple[str, ...]:
    script = cast(Mapping[str, object], document["script"])
    materials = cast(Mapping[str, Mapping[str, object]], script["materials"])
    return tuple(sorted(key for key, material in materials.items() if material["kind"] == "ending"))


def _plan_prompt(document: Mapping[str, object]) -> str:
    script = cast(Mapping[str, object], document["script"])
    sections = cast(Mapping[str, Mapping[str, object]], script["sections"])
    contrasts = [
        {
            "section_key": key,
            "role": sections[key]["role"],
            "description": sections[key]["description"],
        }
        for key in _contrast_section_keys(document)
    ]
    context = {
        "title": script["title"],
        "brief": script["brief"],
        "contrasting_sections_in_output_order": contrasts,
    }
    return (
        "承認済みの楽曲台本について、曲全体の調性と対比の方針を決めてください。"
        "contrast_descriptionsは提示順に一つずつ書き、IDやJSON pathは出力しないでください。"
        "応答は指定Schemaだけに従ってください。\n\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def _harmony_prompt(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
) -> str:
    assert workspace.plan is not None
    script = cast(Mapping[str, object], document["script"])
    materials = cast(Mapping[str, Mapping[str, object]], script["materials"])
    context = {
        "plan": {
            "tonal_center": workspace.plan.tonal_center,
            "mode": workspace.plan.mode,
            "contrasts": dict(workspace.plan.contrasts),
        },
        "materials_in_output_order": [
            {
                "material_key": key,
                "kind": materials[key]["kind"],
                "description": materials[key]["description"],
            }
            for key in targets
        ],
    }
    return (
        "全体計画を守り、提示した素材それぞれの和音列を決めてください。harmoniesの外側の配列は"
        "提示順に対応させ、material ID、開始位置、終止音、JSON pathは出力しないでください。"
        "duration_unitsは各素材内の相対的な長さです。応答は指定Schemaだけに従ってください。\n\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def _melody_prompt(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
) -> str:
    script = cast(Mapping[str, object], document["script"])
    materials = cast(Mapping[str, Mapping[str, object]], script["materials"])
    harmonies = dict(workspace.harmonies)
    context = {
        "plan": {
            "tonal_center": workspace.plan.tonal_center if workspace.plan is not None else None,
            "mode": workspace.plan.mode if workspace.plan is not None else None,
            "contrasts": dict(workspace.plan.contrasts) if workspace.plan is not None else {},
        },
        "materials_in_output_order": [
            {
                "material_key": target,
                "kind": materials[target]["kind"],
                "description": materials[target]["description"],
                "length_units": sum(chord.duration_units for chord in harmonies[target]),
                "harmony": [
                    {
                        "duration_units": chord.duration_units,
                        "root_pitch_class": chord.root_pitch_class,
                        "quality": chord.quality,
                    }
                    for chord in harmonies[target]
                ],
            }
            for target in targets
        ],
    }
    return (
        "各素材について、全体計画とその素材自身の説明、和声だけを使い、再利用時の核となる"
        "前景旋律を作ってください。各出力を他素材の和声へ依存させないでください。melodiesは"
        "提示順に対応させ、material ID、音符ID、JSON pathは出力しないでください。音符は"
        "0以上から始まり、at_unitsとduration_unitsの合計がlength_unitsを超えてはいけません。"
        "応答は指定Schemaだけに従ってください。\n\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def _transition_melody_prompt(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
) -> str:
    context = {
        "plan": {
            "tonal_center": workspace.plan.tonal_center if workspace.plan is not None else None,
            "mode": workspace.plan.mode if workspace.plan is not None else None,
        },
        "transitions_in_output_order": transition_melody_contexts(document, workspace, targets),
    }
    return (
        "各つなぎについて、connector素材の和声を守り、source末尾からtarget冒頭へ自然に移る"
        "前景旋律を作ってください。同じconnector素材でも前後関係が違えば別の実現にしてかまいません。"
        "transition_melodiesは提示順に対応させ、transition ID、配置ID、素材ID、音符ID、JSON pathは"
        "出力しないでください。音符は0以上から始まり、at_unitsとduration_unitsの合計が"
        "length_unitsを超えてはいけません。応答は指定Schemaだけに従ってください。\n\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def _accompaniment_prompt(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
) -> str:
    script = cast(Mapping[str, object], document["script"])
    materials = cast(Mapping[str, Mapping[str, object]], script["materials"])
    harmonies = dict(workspace.harmonies)
    melodies = dict(workspace.melodies)
    context = {
        "plan": {
            "tonal_center": workspace.plan.tonal_center if workspace.plan is not None else None,
            "mode": workspace.plan.mode if workspace.plan is not None else None,
            "contrasts": dict(workspace.plan.contrasts) if workspace.plan is not None else {},
        },
        "materials_in_output_order": [
            {
                "material_key": target,
                "description": materials[target]["description"],
                "harmony": [
                    {
                        "duration_units": chord.duration_units,
                        "root_pitch_class": chord.root_pitch_class,
                        "quality": chord.quality,
                    }
                    for chord in harmonies[target]
                ],
                "melody": {
                    "foreground_voice": melodies[target].foreground_voice,
                    "notes": [
                        {
                            "at_units": note.at_units,
                            "duration_units": note.duration_units,
                            "pitch": note.pitch,
                        }
                        for note in melodies[target].notes
                    ],
                },
            }
            for target in targets
        ],
    }
    return (
        "各素材について、和声と前景旋律を支えるピアノ伴奏を作ってください。accompanimentsは"
        "提示順に対応させ、素材ID、和音ID、音符ID、絶対音高、JSON pathは出力しないでください。"
        "preferred_duration_unitsは保持したい音価の上限、preferred_register_zoneは希望音域です。"
        "Pythonが必要な音域投影、同音再打鍵時の音価短縮、絶対音高配置を決定します。"
        "各イベントは一つの和音区間内に収めてください。応答は指定Schemaだけに従ってください。\n\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def _transition_accompaniment_prompt(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
) -> str:
    context = {
        "plan": {
            "tonal_center": workspace.plan.tonal_center if workspace.plan is not None else None,
            "mode": workspace.plan.mode if workspace.plan is not None else None,
        },
        "transitions_in_output_order": transition_accompaniment_contexts(
            document, workspace, targets
        ),
    }
    return (
        "各つなぎについて、前後の伴奏境界を文脈として使い、つなぎ旋律と和声を支えるピアノ伴奏を"
        "作ってください。前後境界は参考情報であり、connector内部のeventsだけを返してください。"
        "transition_accompanimentsは提示順に対応させ、transition ID、配置ID、素材ID、和音ID、"
        "音符ID、絶対音高、JSON pathは出力しないでください。preferred_duration_unitsは保持したい"
        "音価の上限、preferred_register_zoneは希望音域です。各イベントは一つの和音区間内に"
        "収めてください。応答は指定Schemaだけに従ってください。\n\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def _performance_prompt(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
) -> str:
    context = {
        "targets_in_output_order": performance_contexts(document, workspace, targets),
        "vocabulary": {
            "timing_profile": [None, "neutral", "savor", "flow", "build", "release"],
            "timing_amount": [None, "subtle", "moderate"],
            "dynamics_profile": [None, "steady", "shape", "build", "release"],
            "articulation_profile": [None, "score", "legato", "light"],
            "coordination_profile": [None, "score", "rolled", "aligned"],
            "pedal_profile": [
                None,
                "none",
                "phrase_legato",
                "harmony_legato",
                "clear",
            ],
        },
    }
    return (
        "各対象区分の演奏意図を有限語彙へ翻訳してください。performancesは提示順に対応させ、"
        "ID、時刻、音符、JSON pathは出力しないでください。nullは祖先または既定への継承です。"
        "説明と比較根拠のない指定を追加しないでください。応答は指定Schemaだけに従ってください。\n\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def _contrast_section_keys(document: Mapping[str, object]) -> tuple[str, ...]:
    script = cast(Mapping[str, object], document["script"])
    sections = cast(Mapping[str, Mapping[str, object]], script["sections"])
    return tuple(
        section_id
        for section_id in _ordered_section_keys(document)
        if sections[section_id]["role"] == "contrast"
    )


def _ordered_section_keys(document: Mapping[str, object]) -> tuple[str, ...]:
    script = cast(Mapping[str, object], document["script"])
    sections = cast(Mapping[str, Mapping[str, object]], script["sections"])
    children: dict[str, list[str]] = {section_id: [] for section_id in sections}
    for section_id, section in sections.items():
        parent = section["parent_section_id"]
        if isinstance(parent, str):
            children[parent].append(section_id)
    for values in children.values():
        values.sort(key=lambda section_id: cast(int, sections[section_id]["order"]))
    ordered: list[str] = []

    def visit(section_id: str) -> None:
        ordered.append(section_id)
        for child_id in children[section_id]:
            visit(child_id)

    visit(cast(str, script["root_section_id"]))
    return tuple(ordered)
