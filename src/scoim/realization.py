"""Realize an approved SCoIM script from a frozen lower-stage response."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import jsonpatch
import jsonpointer
import rfc8785

from llm_musical_composer.generic_pipeline_quality import evaluate_generic_pipeline_quality
from llm_musical_composer.performance_pipeline import (
    PerformanceSpec,
    PiecePlan,
    PipelineValidationError,
    RenderedPerformance,
    ScoreSpec,
    ordered_leaf_schedule,
    render_musicxml,
    render_performance,
    render_performance_smf,
    resolve_effective_profile,
    validate_musicxml_round_trip,
    validate_pipeline,
    validate_smf_round_trip,
)
from llm_musical_composer.pipeline_dsl import (
    PipelineDslError,
    dump_performance_spec,
    dump_score_spec,
    parse_performance_spec,
    parse_score_spec,
)

from .outcome import ArtifactDisposition, ExecutionOutcome, TerminalState
from .projection import PlanChoice, ProjectionTarget, project_solo_piano_3m
from .realization_script import realization_script_sha256
from .requirements import resolve_requirements
from .solo_piano_requirements import evaluate_requirement
from .typed_realization import (
    TypedRealizationError,
    build_typed_ir,
    validate_typed_model_response,
)
from .validation import IssueCode, ValidationIssue
from .workspace_realization import (
    WorkspaceDiagnosticIR,
    WorkspaceRealizationError,
    build_workspace_diagnostic_irs,
    build_workspace_ir,
)

_FROZEN_RESPONSE_FIELDS = frozenset(
    {
        "schema_version",
        "profile",
        "plan_choice",
        "score_spec",
        "performance_spec",
        "target_assertions",
    }
)
_FROZEN_RESPONSE_V2_FIELDS = frozenset(
    {
        "schema_version",
        "profile",
        "plan_choice",
        "score_materials",
        "node_performances",
    }
)
_FROZEN_RESPONSE_V3_FIELDS = frozenset({"schema_version", "profile", "workspace"})
_PLAN_CHOICE_FIELDS = frozenset(
    {
        "tonal_center",
        "mode",
        "harmonic_focus_by_section",
        "contrasts_with_by_section",
    }
)
_SEMANTIC_VERIFICATIONS = frozenset(
    {"translated_mechanical_claim", "targeted_performance_difference"}
)


@dataclass(frozen=True, slots=True)
class RealizationResult:
    """Typed result of an offline realization attempt."""

    realized: bool
    outcome: ExecutionOutcome
    frozen_response_sha256: str | None
    musicxml_path: Path | None
    musicxml_sha256: str | None
    smf_path: Path | None
    smf_sha256: str | None
    diagnostics_path: Path | None
    realized_duration_ms: int | None
    target_results: tuple[dict[str, object], ...]

    @property
    def issues(self) -> tuple[ValidationIssue, ...]:
        """Return typed reasons from the single outcome source."""

        return self.outcome.issues


def _failure(
    code: IssueCode,
    message: str,
    path: str,
    *,
    response_sha256: str | None = None,
) -> RealizationResult:
    issue = ValidationIssue(code=code, message=message, path=path)
    return RealizationResult(
        realized=False,
        outcome=ExecutionOutcome(
            terminal_state=TerminalState.FAILED,
            artifact_disposition=ArtifactDisposition.NONE,
            issues=(issue,),
        ),
        frozen_response_sha256=response_sha256,
        musicxml_path=None,
        musicxml_sha256=None,
        smf_path=None,
        smf_sha256=None,
        diagnostics_path=None,
        realized_duration_ms=None,
        target_results=(),
    )


def _response_hash(response: Mapping[str, object]) -> str:
    return hashlib.sha256(rfc8785.dumps(response)).hexdigest()


def _string_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        return None
    return dict(value)


def _plan_choice(
    response: Mapping[str, object], response_sha256: str
) -> PlanChoice | RealizationResult:
    raw = _string_mapping(response.get("plan_choice"))
    path = "/frozen_response/plan_choice"
    if raw is None or set(raw) != _PLAN_CHOICE_FIELDS:
        return _failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "Invalid plan choice fields",
            path,
            response_sha256=response_sha256,
        )
    harmonic = _string_mapping(raw["harmonic_focus_by_section"])
    contrasts = _string_mapping(raw["contrasts_with_by_section"])
    if harmonic is None or contrasts is None:
        return _failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "Invalid plan choice mappings",
            path,
            response_sha256=response_sha256,
        )
    return PlanChoice(
        tonal_center=cast(int, raw["tonal_center"]),
        mode=cast(str, raw["mode"]),
        harmonic_focus_by_section=cast(dict[str, int], harmonic),
        contrasts_with_by_section=cast(dict[str, str], contrasts),
    )


def _dsl_source(
    response: Mapping[str, object], field: str, response_sha256: str
) -> str | RealizationResult:
    path = f"/frozen_response/{field}"
    raw = _string_mapping(response.get(field))
    if raw is None or set(raw) != {"format", "source"}:
        return _failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "Invalid DSL response fields",
            path,
            response_sha256=response_sha256,
        )
    if raw["format"] != "pipeline-dsl-v1" or not isinstance(raw["source"], str):
        return _failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "Invalid DSL response",
            path,
            response_sha256=response_sha256,
        )
    return raw["source"]


def _score_length_ratios_match(plan: PiecePlan, score: ScoreSpec) -> bool:
    leaves, _ = ordered_leaf_schedule(plan, score)
    materials = {item.material_id: item for item in score.materials}
    first = leaves[0][0]
    assert first.score_material_id is not None and first.duration_weight is not None
    first_length = materials[first.score_material_id].length_units
    for leaf, _, _ in leaves[1:]:
        assert leaf.score_material_id is not None and leaf.duration_weight is not None
        if (
            materials[leaf.score_material_id].length_units * first.duration_weight
            != first_length * leaf.duration_weight
        ):
            return False
    return True


def _children_by_parent(plan: PiecePlan) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for node in plan.nodes:
        if node.parent_id is not None:
            result.setdefault(node.parent_id, []).append(node.node_id)
    return result


def _leaf_ids_under(plan: PiecePlan, section_id: str) -> tuple[str, ...]:
    children = _children_by_parent(plan)
    result: list[str] = []

    def visit(node_id: str) -> None:
        if node_id not in children:
            result.append(node_id)
            return
        for child_id in children[node_id]:
            visit(child_id)

    visit(section_id)
    return tuple(result)


def _reference_section_id(
    reference: Mapping[str, object], script: Mapping[str, object]
) -> str | None:
    if reference["type"] == "section":
        return cast(str, reference["id"])
    if reference["type"] == "placement":
        placements = cast(dict[str, dict[str, object]], script["placements"])
        return cast(str, placements[cast(str, reference["id"])]["section_id"])
    return None


def _effective_profiles(
    plan: PiecePlan, performance: PerformanceSpec, node_id: str
) -> dict[str, str]:
    node_by_id = {item.node_id: item for item in plan.nodes}
    performance_by_node = {item.node_id: item for item in performance.node_performances}
    defaults = {
        "timing_profile": "neutral",
        "dynamics_profile": "steady",
        "articulation_profile": "score",
        "coordination_profile": "score",
        "pedal_profile": "none",
    }
    return {
        attribute: resolve_effective_profile(
            node_by_id[node_id], node_by_id, performance_by_node, attribute, default
        )[1]
        for attribute, default in defaults.items()
    }


def _stage_evidence(
    target: ProjectionTarget,
    stage: str,
    *,
    script: Mapping[str, object],
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
    rendered: RenderedPerformance,
    quality: Mapping[str, object],
    material_bindings: Mapping[str, tuple[str, ...]] | None,
) -> dict[str, object]:
    parts = jsonpointer.JsonPointer(target.source_path).parts
    nodes = {item.node_id: item for item in plan.nodes}
    materials = {item.material_id: item for item in score.materials}
    performed_by_node: dict[str, int] = {}
    for note in rendered.notes:
        performed_by_node[note.occurrence_node_id] = (
            performed_by_node.get(note.occurrence_node_id, 0) + 1
        )
    if len(parts) < 2 or parts[1] == "brief":
        if stage == "piece_plan":
            return {
                "entity_present": True,
                "node_count": len(nodes),
                "has_contrast": any(item.contrasts_with for item in plan.nodes),
                "has_recurrence": any(item.derived_from for item in plan.nodes),
            }
        if stage == "score_spec":
            voices = {note.voice for item in score.materials for note in item.notes}
            return {
                "entity_present": True,
                "material_count": len(materials),
                "note_count": sum(len(item.notes) for item in score.materials),
                "has_upper_and_lower": voices == {"upper", "lower"},
            }
        return {
            "entity_present": True,
            "duration_ms": rendered.duration_ms,
            "note_count": len(rendered.notes),
        }
    if parts[1] == "sections":
        section_id = parts[2]
        leaf_ids = _leaf_ids_under(plan, section_id)
        if stage == "piece_plan":
            return {"entity_present": section_id in nodes, "node": asdict(nodes[section_id])}
        if stage == "score_spec":
            material_ids = sorted(
                {cast(str, nodes[leaf_id].score_material_id) for leaf_id in leaf_ids}
            )
            return {
                "entity_present": True,
                "material_ids": material_ids,
                "note_count": sum(len(materials[item].notes) for item in material_ids),
            }
        return {
            "entity_present": True,
            "leaf_profiles": {
                leaf_id: _effective_profiles(plan, performance, leaf_id) for leaf_id in leaf_ids
            },
            "rendered_note_count": sum(performed_by_node.get(item, 0) for item in leaf_ids),
        }
    if parts[1] == "materials":
        material_id = parts[2]
        realized_ids = (
            material_bindings.get(material_id, ())
            if material_bindings is not None
            else (material_id,)
        )
        if len(realized_ids) != 1 or realized_ids[0] != material_id:
            return {
                "entity_present": bool(realized_ids)
                and all(item in materials for item in realized_ids),
                "realized_material_ids": list(realized_ids),
                "materials": [asdict(materials[item]) for item in realized_ids],
                "occurrence_node_ids": [
                    item.node_id for item in plan.nodes if item.score_material_id in realized_ids
                ],
            }
        return {
            "entity_present": material_id in materials,
            "material": asdict(materials[material_id]),
            "occurrence_node_ids": [
                item.node_id for item in plan.nodes if item.score_material_id == material_id
            ],
        }
    if parts[1] == "variations":
        variation = cast(dict[str, object], cast(dict[str, object], script["variations"])[parts[2]])
        source = cast(dict[str, object], variation["source"])
        target_reference = cast(dict[str, object], variation["target"])
        source_section = _reference_section_id(source, script)
        target_section = _reference_section_id(target_reference, script)
        recurrence = next(
            (
                item
                for item in cast(list[dict[str, object]], quality["recurrences"])
                if item["target_node_id"] == target_section
            ),
            None,
        )
        return {
            "entity_present": True,
            "source_section_id": source_section,
            "target_section_id": target_section,
            "same_score_material": bool(
                source_section
                and target_section
                and nodes[source_section].score_material_id
                == nodes[target_section].score_material_id
            ),
            "derived_from_matches": bool(
                source_section
                and target_section
                and nodes[target_section].derived_from == source_section
            ),
            "performance_difference": bool(recurrence and recurrence["performance_difference"]),
        }
    if parts[1] == "transitions":
        transition = cast(
            dict[str, object], cast(dict[str, object], script["transitions"])[parts[2]]
        )
        placements = cast(dict[str, dict[str, object]], script["placements"])
        leaves, _ = ordered_leaf_schedule(plan, score)
        leaf_ids = [item.node_id for item, _, _ in leaves]
        section_ids = [
            cast(str, placements[cast(str, transition[field])]["section_id"])
            for field in (
                "from_placement_id",
                "connector_placement_id",
                "to_placement_id",
            )
        ]
        positions = [leaf_ids.index(item) for item in section_ids]
        return {
            "entity_present": True,
            "section_ids": section_ids,
            "adjacent_in_order": positions[1] == positions[0] + 1
            and positions[2] == positions[1] + 1,
        }
    if "performance_directions" in parts:
        directions = cast(
            dict[str, dict[str, object]],
            cast(dict[str, object], script["performance_setup"])["performance_directions"],
        )
        direction = directions[parts[-2]]
        target_node = _reference_section_id(cast(dict[str, object], direction["target"]), script)
        relative = cast(dict[str, object] | None, direction.get("relative_to"))
        relative_node = _reference_section_id(relative, script) if relative else None
        target_profiles = _effective_profiles(plan, performance, cast(str, target_node))
        relative_profiles = (
            _effective_profiles(plan, performance, relative_node) if relative_node else None
        )
        return {
            "entity_present": True,
            "target_node_id": target_node,
            "relative_node_id": relative_node,
            "effective_profiles": target_profiles,
            "different_profile_count": 0
            if relative_profiles is None
            else sum(target_profiles[name] != relative_profiles[name] for name in target_profiles),
        }
    return {"entity_present": False}


def _target_evidence(
    targets: tuple[ProjectionTarget, ...],
    *,
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
    script: Mapping[str, object],
    rendered: RenderedPerformance,
    quality: Mapping[str, object],
    material_bindings: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, object]:
    return {
        target.target_id: {
            stage: {
                **_stage_evidence(
                    target,
                    stage,
                    script=script,
                    plan=plan,
                    score=score,
                    performance=performance,
                    rendered=rendered,
                    quality=quality,
                    material_bindings=material_bindings,
                )
            }
            for stage in target.generation_stages
        }
        for target in targets
        if target.verification in _SEMANTIC_VERIFICATIONS
    }


def _prepare_assertions(
    response: Mapping[str, object],
    targets: tuple[ProjectionTarget, ...],
    response_sha256: str,
) -> tuple[dict[str, dict[str, object]], RealizationResult | None]:
    assertions = _string_mapping(response.get("target_assertions"))
    required = {
        target.target_id: target
        for target in targets
        if target.verification in _SEMANTIC_VERIFICATIONS
    }
    if assertions is None or set(assertions) != set(required):
        return {}, _failure(
            IssueCode.UNREPRESENTABLE,
            "Every natural-language projection target needs exactly one assertion",
            "/frozen_response/target_assertions",
            response_sha256=response_sha256,
        )
    prepared: dict[str, dict[str, object]] = {}
    for target in required.values():
        escaped_target = jsonpointer.escape(target.target_id)
        assertion = _string_mapping(assertions[target.target_id])
        if assertion is None or set(assertion) != {"interpretation", "tests"}:
            return {}, _failure(
                IssueCode.MODEL_OUTPUT_INVALID,
                "Invalid target assertion fields",
                f"/frozen_response/target_assertions/{escaped_target}",
                response_sha256=response_sha256,
            )
        interpretation = assertion["interpretation"]
        tests_by_stage = _string_mapping(assertion["tests"])
        if (
            not isinstance(interpretation, str)
            or not interpretation.strip()
            or tests_by_stage is None
            or set(tests_by_stage) != set(target.generation_stages)
        ):
            return {}, _failure(
                IssueCode.MODEL_OUTPUT_INVALID,
                "A target assertion needs an interpretation and every target stage",
                f"/frozen_response/target_assertions/{escaped_target}",
                response_sha256=response_sha256,
            )
        checks: list[dict[str, object]] = []
        for stage in target.generation_stages:
            operations = tests_by_stage[stage]
            if not isinstance(operations, list) or not operations:
                return {}, _failure(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "Each target stage needs one or more test operations",
                    f"/frozen_response/target_assertions/{escaped_target}/tests/{stage}",
                    response_sha256=response_sha256,
                )
            prefix = f"/target_evidence/{escaped_target}/{jsonpointer.escape(stage)}/"
            for index, operation in enumerate(operations):
                operation_path = (
                    f"/frozen_response/target_assertions/{escaped_target}/tests/"
                    f"{jsonpointer.escape(stage)}/{index}"
                )
                if (
                    not isinstance(operation, Mapping)
                    or set(operation) != {"op", "path", "value"}
                    or operation.get("op") != "test"
                    or not isinstance(operation.get("path"), str)
                    or not cast(str, operation["path"]).startswith(prefix)
                ):
                    return {}, _failure(
                        IssueCode.MODEL_OUTPUT_INVALID,
                        "Only scoped RFC 6902 test operations are allowed",
                        operation_path,
                        response_sha256=response_sha256,
                    )
                checks.append(dict(operation))
        prepared[target.target_id] = {
            "interpretation": interpretation,
            "checks": checks,
        }
    return prepared, None


def _evaluate_targets(
    targets: tuple[ProjectionTarget, ...],
    assertions: dict[str, dict[str, object]] | None,
    facts: dict[str, object],
    *,
    script: Mapping[str, object],
    plan: PiecePlan,
    score: ScoreSpec,
    rendered: RenderedPerformance,
) -> tuple[list[dict[str, object]], tuple[ValidationIssue, ...]]:
    script_requirements = {
        requirement.source_path: requirement for requirement in resolve_requirements(script)
    }
    results: list[dict[str, object]] = []
    issues: list[ValidationIssue] = []
    assertion_facts = {"target_evidence": facts["target_evidence"]}
    for target in targets:
        if target.verification == "typed_requirement":
            observed = evaluate_requirement(
                plan,
                score,
                rendered,
                script_requirements[target.source_path],
            )
            results.append({"target_id": target.target_id, **observed.value()})
            if observed.status != "passed":
                issues.append(
                    ValidationIssue(
                        code=IssueCode.PROJECTION_TARGET_UNMET,
                        message="A typed requirement was not met by the rendered performance",
                        path=target.source_path,
                    )
                )
            continue
        if assertions is None and target.verification in _SEMANTIC_VERIFICATIONS:
            results.append(
                {
                    "target_id": target.target_id,
                    "status": "unverified",
                    "checks": [],
                }
            )
            continue
        if assertions is None:
            results.append({"target_id": target.target_id, "status": "passed", "checks": []})
            continue
        if target.target_id not in assertions:
            results.append({"target_id": target.target_id, "status": "passed", "checks": []})
            continue
        assertion = assertions[target.target_id]
        checks = cast(list[dict[str, object]], assertion["checks"])
        passed = True
        for operation in checks:
            try:
                jsonpatch.JsonPatch([operation]).apply(assertion_facts, in_place=False)
            except (jsonpatch.JsonPatchException, jsonpointer.JsonPointerException):
                passed = False
        results.append(
            {
                "target_id": target.target_id,
                "status": "passed" if passed else "failed",
                "interpretation": assertion["interpretation"],
                "checks": checks,
            }
        )
        if not passed:
            issues.append(
                ValidationIssue(
                    code=IssueCode.PROJECTION_TARGET_UNMET,
                    message="A projection target assertion did not match the realized facts",
                    path=target.source_path,
                )
            )
    return results, tuple(issues)


def _stable_round_trip(result: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in result.items() if key != "path"}


def realize_solo_piano_3m(
    document: Mapping[str, object],
    frozen_response: Mapping[str, object],
    output_dir: Path,
) -> RealizationResult:
    """Realize ``solo_piano_3m_v1`` without making an external model call."""

    response_version = frozen_response.get("schema_version")
    if response_version not in {1, 2, 3}:
        return _failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "The frozen response schema version must be 1, 2, or 3",
            "/frozen_response/schema_version",
        )
    expected_fields_by_version = {
        1: _FROZEN_RESPONSE_FIELDS,
        2: _FROZEN_RESPONSE_V2_FIELDS,
        3: _FROZEN_RESPONSE_V3_FIELDS,
    }
    expected_fields = expected_fields_by_version[response_version]
    unknown_fields = set(frozen_response) - expected_fields
    missing_fields = expected_fields - set(frozen_response)
    if unknown_fields or missing_fields:
        field = sorted(unknown_fields or missing_fields)[0]
        return _failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "The frozen response fields do not match the contract",
            f"/frozen_response/{field}",
        )
    if frozen_response["profile"] != "solo_piano_3m_v1":
        return _failure(
            IssueCode.UNSUPPORTED_PROFILE,
            "The frozen response profile is not supported",
            "/frozen_response/profile",
        )
    try:
        response_sha256 = _response_hash(frozen_response)
    except (TypeError, ValueError):
        return _failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "The frozen response is not canonical JSON data",
            "/frozen_response",
        )
    material_bindings: Mapping[str, tuple[str, ...]] | None = None
    workspace_conversion: dict[str, object] | None = None
    workspace_diagnostics: tuple[WorkspaceDiagnosticIR, WorkspaceDiagnosticIR] | None = None
    if response_version == 1:
        choice = _plan_choice(frozen_response, response_sha256)
        if isinstance(choice, RealizationResult):
            return choice
        projection = project_solo_piano_3m(document, plan_choice=choice)
        if not projection.projected:
            return RealizationResult(
                realized=False,
                outcome=ExecutionOutcome(
                    terminal_state=TerminalState.FAILED,
                    artifact_disposition=ArtifactDisposition.NONE,
                    issues=projection.issues,
                ),
                frozen_response_sha256=response_sha256,
                musicxml_path=None,
                musicxml_sha256=None,
                smf_path=None,
                smf_sha256=None,
                diagnostics_path=None,
                realized_duration_ms=None,
                target_results=(),
            )
        assert projection.piece_plan is not None
        plan = projection.piece_plan
        targets = projection.targets
        prepared_assertions, assertion_failure = _prepare_assertions(
            frozen_response, targets, response_sha256
        )
        if assertion_failure is not None:
            return assertion_failure
        score_source = _dsl_source(frozen_response, "score_spec", response_sha256)
        if isinstance(score_source, RealizationResult):
            return score_source
        performance_source = _dsl_source(frozen_response, "performance_spec", response_sha256)
        if isinstance(performance_source, RealizationResult):
            return performance_source
        try:
            score = parse_score_spec(score_source)
            performance = parse_performance_spec(performance_source)
            validate_pipeline(plan, score, performance)
        except (PipelineDslError, PipelineValidationError) as error:
            return _failure(
                IssueCode.MODEL_OUTPUT_INVALID,
                str(error),
                "/frozen_response",
                response_sha256=response_sha256,
            )
    elif response_version == 2:
        typed_payload = {
            key: frozen_response[key]
            for key in ("plan_choice", "score_materials", "node_performances")
        }
        typed_issues = validate_typed_model_response(
            document,
            typed_payload,
            path_prefix="/frozen_response",
        )
        if typed_issues:
            issue = typed_issues[0]
            return _failure(
                issue.code,
                issue.message,
                issue.path,
                response_sha256=response_sha256,
            )
        try:
            typed_ir = build_typed_ir(document, frozen_response)
        except TypedRealizationError as error:
            return _failure(
                error.issue.code,
                error.issue.message,
                error.issue.path,
                response_sha256=response_sha256,
            )
        plan = typed_ir.plan
        targets = typed_ir.targets
        score = typed_ir.score
        performance = typed_ir.performance
        prepared_assertions = None
        score_source = dump_score_spec(score)
        performance_source = dump_performance_spec(performance)
    else:
        raw_workspace = _string_mapping(frozen_response.get("workspace"))
        if raw_workspace is None:
            return _failure(
                IssueCode.MODEL_OUTPUT_INVALID,
                "The frozen workspace must be an object",
                "/frozen_response/workspace",
                response_sha256=response_sha256,
            )
        try:
            from .realization_workspace import workspace_from_dict

            workspace = workspace_from_dict(raw_workspace)
            workspace_ir = build_workspace_ir(document, workspace)
            workspace_diagnostics = build_workspace_diagnostic_irs(document, workspace)
        except ValueError as error:
            if isinstance(error, WorkspaceRealizationError):
                issue = error.issue
                return _failure(
                    issue.code,
                    issue.message,
                    issue.path,
                    response_sha256=response_sha256,
                )
            return _failure(
                IssueCode.MODEL_OUTPUT_INVALID,
                str(error),
                "/frozen_response/workspace",
                response_sha256=response_sha256,
            )
        plan = workspace_ir.plan
        targets = workspace_ir.targets
        score = workspace_ir.score
        performance = workspace_ir.performance
        prepared_assertions = None
        score_source = dump_score_spec(score)
        performance_source = dump_performance_spec(performance)
        material_bindings = workspace_ir.material_bindings
        workspace_conversion = {
            "common_units_per_weight": workspace_ir.common_units_per_weight,
            "material_bindings": {
                key: list(value) for key, value in workspace_ir.material_bindings.items()
            },
        }
    if not _score_length_ratios_match(plan, score):
        return _failure(
            IssueCode.PROJECTION_TARGET_UNMET,
            "The realized score lengths do not preserve the script ratios",
            "/script/sections",
            response_sha256=response_sha256,
        )
    rendered = render_performance(plan, score, performance)
    quality = evaluate_generic_pipeline_quality(plan, score, performance, rendered)
    issues: tuple[ValidationIssue, ...] = ()
    if not quality["passes"]:
        issues = (
            ValidationIssue(
                code=IssueCode.PROJECTION_TARGET_UNMET,
                message="The realized pipeline did not pass minimum quality",
                path="/quality",
            ),
        )
    output = Path(output_dir)
    if output.exists():
        return _failure(
            IssueCode.STORAGE_CONFLICT,
            "The realization output directory already exists",
            "/output_dir",
            response_sha256=response_sha256,
        )
    staging: Path | None = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".scoim-realization-", dir=output.parent))
        musicxml = render_musicxml(plan, score, staging / "score.musicxml")
        render_performance_smf(rendered, staging / "final.mid")
        preview_round_trips: dict[str, dict[str, object]] = {}
        if workspace_diagnostics is not None:
            for filename, preview in zip(
                ("phase-04-melody.mid", "phase-06-score.mid"),
                workspace_diagnostics,
                strict=True,
            ):
                preview_rendered = render_performance(
                    preview.plan, preview.score, preview.performance
                )
                preview_path = staging / filename
                render_performance_smf(preview_rendered, preview_path)
                preview_round_trip = _stable_round_trip(
                    validate_smf_round_trip(preview_rendered, preview_path)
                )
                if preview_round_trip["status"] != "passed":
                    return _failure(
                        IssueCode.PROJECTION_TARGET_UNMET,
                        "A diagnostic SMF did not round-trip",
                        f"/artifacts/{filename}",
                        response_sha256=response_sha256,
                    )
                preview_round_trips[filename] = preview_round_trip
        musicxml_round_trip = _stable_round_trip(
            validate_musicxml_round_trip(plan, score, musicxml)
        )
        smf_round_trip = _stable_round_trip(
            validate_smf_round_trip(rendered, staging / "final.mid")
        )
        if musicxml_round_trip["status"] != "passed" or smf_round_trip["status"] != "passed":
            return _failure(
                IssueCode.PROJECTION_TARGET_UNMET,
                "A rendered artifact did not round-trip",
                "/artifacts",
                response_sha256=response_sha256,
            )
        evidence = _target_evidence(
            targets,
            plan=plan,
            score=score,
            performance=performance,
            script=cast(dict[str, object], document["script"]),
            rendered=rendered,
            quality=quality,
            material_bindings=material_bindings,
        )
        facts = {"target_evidence": evidence}
        target_results, target_issues = _evaluate_targets(
            targets,
            prepared_assertions,
            facts,
            script=cast(dict[str, object], document["script"]),
            plan=plan,
            score=score,
            rendered=rendered,
        )
        issues += target_issues
        artifact_disposition = (
            ArtifactDisposition.DIAGNOSTIC_ONLY if issues else ArtifactDisposition.CANDIDATE
        )
        outcome = ExecutionOutcome(
            terminal_state=TerminalState.COMPLETED,
            artifact_disposition=artifact_disposition,
            issues=issues,
        )
        diagnostics = {
            "schema_version": 1,
            "profile": "solo_piano_3m_v1",
            "approved_content_sha256": realization_script_sha256(document),
            "frozen_response_sha256": response_sha256,
            "realized_duration_ms": rendered.duration_ms,
            "quality": quality,
            "artifacts": {"musicxml": musicxml_round_trip, "smf": smf_round_trip},
            **({"diagnostic_smf": preview_round_trips} if preview_round_trips else {}),
            "target_evidence": evidence,
            "targets": target_results,
            "outcome": outcome.to_dict(),
            **(
                {"workspace_conversion": workspace_conversion}
                if workspace_conversion is not None
                else {}
            ),
            "lineage": {
                "piece_plan": asdict(plan),
                "score_spec_sha256": hashlib.sha256(score_source.encode()).hexdigest(),
                "performance_spec_sha256": hashlib.sha256(performance_source.encode()).hexdigest(),
            },
        }
        (staging / "realization-diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, output)
        staging = None
    except OSError as error:
        return _failure(
            IssueCode.STORAGE_ERROR,
            str(error),
            "/output_dir",
            response_sha256=response_sha256,
        )
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    return RealizationResult(
        realized=True,
        outcome=outcome,
        frozen_response_sha256=response_sha256,
        musicxml_path=output / "score.musicxml",
        musicxml_sha256=cast(str, musicxml_round_trip["sha256"]),
        smf_path=output / "final.mid",
        smf_sha256=cast(str, smf_round_trip["sha256"]),
        diagnostics_path=output / "realization-diagnostics.json",
        realized_duration_ms=rendered.duration_ms,
        target_results=tuple(target_results),
    )
