"""Private model-transfer contracts for compiling script 0.3.0 documents."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from jsonschema import Draft202012Validator

from .profile_capabilities import (
    GenerationProfileCapabilities,
    check_generation_profile_document,
)
from .projection_ledger import ProjectionLedgerEntry, validate_projection_ledger
from .script_0_3_validation import check_script_0_3_document
from .validation import CheckResult, IssueCode, ValidationIssue

_ID_SCHEMA = {"type": "string", "pattern": "^[a-z][a-z0-9_-]*$"}
_TEXT_SCHEMA = {"type": "string", "minLength": 1}


@dataclass(frozen=True, slots=True)
class IndexedStructureResult:
    valid: bool
    indexed_structure: dict[str, object] | None
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True, slots=True)
class ScriptBuildResult:
    valid: bool
    outcome: str
    document: dict[str, object] | None
    projection_ledger: tuple[ProjectionLedgerEntry, ...]
    issues: tuple[ValidationIssue, ...]


def structure_prompt(
    approved_flow: Mapping[str, object], capabilities: GenerationProfileCapabilities
) -> str:
    """Build the first-operation prompt from the approved flow and typed capabilities."""
    context = {
        "approved_flow": dict(approved_flow),
        "profile_capabilities": {
            "profile_id": capabilities.profile_id,
            "material_placement_roles": sorted(capabilities.material_placement_roles),
            "minimum_material_placements_per_leaf": (
                capabilities.minimum_material_placements_per_leaf
            ),
            "maximum_material_placements_per_leaf": (
                capabilities.maximum_material_placements_per_leaf
            ),
        },
    }
    return (
        "承認済みの楽曲台本を、区分、マテリアル、マテリアル配置へ分解してください。"
        "同じマテリアル候補を複数の配置から参照して再利用を表してください。"
        "区分、マテリアル、配置の最終IDを返さないでください。"
        "場面の数と順序を変えず、各場面の区分を深さ優先順で返してください。\n\n"
        f"入力: {json.dumps(context, ensure_ascii=False, sort_keys=True)}\n"
    )


def structure_response_schema(
    capabilities: GenerationProfileCapabilities,
) -> dict[str, object]:
    """Build the ID-free response schema for the first model operation."""
    placement = {
        "type": "object",
        "additionalProperties": False,
        "required": ["material_index", "role"],
        "properties": {
            "material_index": {"type": "integer", "minimum": 0},
            "role": {"enum": sorted(capabilities.material_placement_roles)},
        },
    }
    section = {
        "type": "object",
        "additionalProperties": False,
        "required": ["depth", "role", "description", "relative_length", "placements"],
        "properties": {
            "depth": {"type": "integer", "minimum": 0},
            "role": dict(_ID_SCHEMA),
            "description": dict(_TEXT_SCHEMA),
            "relative_length": {
                "anyOf": [
                    {"type": "null"},
                    {"type": "number", "exclusiveMinimum": 0},
                ]
            },
            "placements": {"type": "array", "items": placement},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["materials", "scenes"],
        "properties": {
            "materials": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["description"],
                    "properties": {"description": dict(_TEXT_SCHEMA)},
                },
            },
            "scenes": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["sections"],
                    "properties": {"sections": {"type": "array", "minItems": 1, "items": section}},
                },
            },
        },
    }


def relations_prompt(
    approved_flow: Mapping[str, object],
    indexed_structure: Mapping[str, object],
    capabilities: GenerationProfileCapabilities,
) -> str:
    """Build the second-operation prompt around the Python-indexed structure."""
    context = {
        "approved_flow": dict(approved_flow),
        "indexed_structure": dict(indexed_structure),
        "profile_capabilities": {
            "profile_id": capabilities.profile_id,
            "performance_direction_target_types": sorted(
                capabilities.performance_direction_target_types
            ),
        },
    }
    return (
        "Pythonが確定した区分とマテリアル配置を変更しないでください。"
        "その構造へ、変奏関係、遷移、演奏指示、比較条件を追加してください。"
        "各要素の最終IDは返さず、入力にあるIDだけを参照してください。"
        "必要な関係をこの構造では正しく表せない場合は、無理に補わず"
        "outcomeをstructure_insufficientとして理由を返してください。\n\n"
        f"入力: {json.dumps(context, ensure_ascii=False, sort_keys=True)}\n"
    )


def relations_response_schema(
    indexed_structure: Mapping[str, object],
    capabilities: GenerationProfileCapabilities,
) -> dict[str, object]:
    """Build the second-operation schema from Python-assigned target IDs."""
    script = cast(dict[str, object], indexed_structure["script"])
    ids_by_type = {
        "section": sorted(cast(dict[str, object], script["sections"])),
        "material": sorted(cast(dict[str, object], script["materials"])),
        "material_placement": sorted(cast(dict[str, object], script["material_placements"])),
    }
    all_ids = sorted({item for values in ids_by_type.values() for item in values})
    reference = {
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "id"],
        "properties": {
            "type": {"enum": sorted(ids_by_type)},
            "id": {"enum": all_ids},
        },
    }
    variation = {
        "type": "object",
        "additionalProperties": False,
        "required": ["source", "target", "preserve", "change", "description"],
        "properties": {
            "source": reference,
            "target": reference,
            "preserve": {"type": "array", "minItems": 1, "items": dict(_TEXT_SCHEMA)},
            "change": {"type": "array", "minItems": 1, "items": dict(_TEXT_SCHEMA)},
            "description": dict(_TEXT_SCHEMA),
        },
    }
    transition = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "source_material_placement_id",
            "transition_material_placement_id",
            "target_material_placement_id",
            "description",
        ],
        "properties": {
            "source_material_placement_id": {"enum": ids_by_type["material_placement"]},
            "transition_material_placement_id": {"enum": ids_by_type["material_placement"]},
            "target_material_placement_id": {"enum": ids_by_type["material_placement"]},
            "description": dict(_TEXT_SCHEMA),
        },
    }
    direction_target_types = sorted(capabilities.performance_direction_target_types)
    direction_target_ids = sorted(
        {item for kind in direction_target_types for item in ids_by_type[kind]}
    )
    direction = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "target_type",
            "target_id",
            "relative_to_type",
            "relative_to_id",
            "description",
        ],
        "properties": {
            "target_type": {"enum": direction_target_types},
            "target_id": {"enum": direction_target_ids},
            "relative_to_type": {"anyOf": [{"type": "null"}, {"enum": direction_target_types}]},
            "relative_to_id": {"anyOf": [{"type": "null"}, {"enum": direction_target_ids}]},
            "description": dict(_TEXT_SCHEMA),
        },
    }
    comparison = {
        "type": "object",
        "additionalProperties": False,
        "required": ["performance_direction_index", "feature", "relation"],
        "properties": {
            "performance_direction_index": {"type": "integer", "minimum": 0},
            "feature": {"enum": ["onset_alignment", "loudness"]},
            "relation": {"enum": ["more", "less"]},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "outcome",
            "structure_insufficient_reason",
            "variation_relations",
            "material_placement_transitions",
            "performance_directions",
            "performance_comparison_requirements",
        ],
        "properties": {
            "outcome": {"enum": ["complete", "structure_insufficient"]},
            "structure_insufficient_reason": {"anyOf": [{"type": "null"}, dict(_TEXT_SCHEMA)]},
            "variation_relations": {"type": "array", "items": variation},
            "material_placement_transitions": {"type": "array", "items": transition},
            "performance_directions": {"type": "array", "items": direction},
            "performance_comparison_requirements": {
                "type": "array",
                "items": comparison,
            },
        },
    }


def check_structure_candidate(
    approved_flow: Mapping[str, object],
    response: Mapping[str, object],
    capabilities: GenerationProfileCapabilities,
) -> CheckResult:
    """List all machine-checkable problems in one first-operation response."""
    errors = sorted(
        Draft202012Validator(structure_response_schema(capabilities)).iter_errors(response),
        key=lambda error: (tuple(str(part) for part in error.absolute_path), error.message),
    )
    issues = [
        ValidationIssue(
            IssueCode.MODEL_OUTPUT_INVALID,
            error.message,
            "".join(f"/{_escape_pointer_token(part)}" for part in error.absolute_path),
        )
        for error in errors
    ]
    if errors:
        return CheckResult(False, tuple(issues))

    flow_scenes = cast(list[dict[str, object]], approved_flow["scenes"])
    response_scenes = cast(list[dict[str, object]], response["scenes"])
    if len(response_scenes) != len(flow_scenes):
        issues.append(
            ValidationIssue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "The structure must contain one entry for every approved flow scene",
                "/scenes",
            )
        )
    material_count = len(cast(list[object], response["materials"]))
    length_totals: list[float] = []
    for scene_index, scene in enumerate(response_scenes):
        sections = cast(list[dict[str, object]], scene["sections"])
        depths = [cast(int, section["depth"]) for section in sections]
        if depths and depths[0] != 0:
            issues.append(
                _model_issue(
                    "The first section in a scene must have depth zero",
                    f"/scenes/{scene_index}/sections/0/depth",
                )
            )
        for section_index, depth in enumerate(depths[1:], start=1):
            if depth > depths[section_index - 1] + 1:
                issues.append(
                    _model_issue(
                        "Section depth cannot increase by more than one",
                        f"/scenes/{scene_index}/sections/{section_index}/depth",
                    )
                )
        scene_length = 0.0
        for section_index, section in enumerate(sections):
            has_child = (
                section_index + 1 < len(sections)
                and depths[section_index + 1] > depths[section_index]
            )
            relative_length = section["relative_length"]
            placements = cast(list[dict[str, object]], section["placements"])
            path = f"/scenes/{scene_index}/sections/{section_index}"
            if has_child and (relative_length is not None or placements):
                issues.append(
                    _model_issue("A branch section cannot contain length or placements", path)
                )
            if not has_child:
                if relative_length is None:
                    issues.append(
                        _model_issue(
                            "A leaf section must contain a relative length",
                            f"{path}/relative_length",
                        )
                    )
                else:
                    scene_length += float(relative_length)
                if not capabilities.allows_material_placement_count(len(placements)):
                    issues.append(
                        _model_issue(
                            "A leaf section has an unsupported placement count",
                            f"{path}/placements",
                        )
                    )
            for placement_index, placement in enumerate(placements):
                material_index = cast(int, placement["material_index"])
                if not 0 <= material_index < material_count:
                    issues.append(
                        _model_issue(
                            "A placement refers to an unknown material index",
                            f"{path}/placements/{placement_index}/material_index",
                        )
                    )
        length_totals.append(scene_length)
    ranks = {"short": 0, "medium": 1, "long": 2}
    for left_index, left in enumerate(flow_scenes[: len(length_totals)]):
        for right_index, right in enumerate(flow_scenes[: len(length_totals)]):
            if (
                ranks[cast(str, left["length_class"])] < ranks[cast(str, right["length_class"])]
                and length_totals[left_index] > length_totals[right_index]
            ):
                issues.append(
                    _model_issue(
                        "Scene relative lengths reverse the approved length classes",
                        "/scenes",
                    )
                )
                break
    return CheckResult(not issues, tuple(issues))


def index_structure_candidate(
    approved_flow: Mapping[str, object],
    composition_id: str,
    response: Mapping[str, object],
    capabilities: GenerationProfileCapabilities,
) -> IndexedStructureResult:
    """Validate an ID-free response and assign deterministic final structure IDs."""
    checked = check_structure_candidate(approved_flow, response, capabilities)
    if not checked.valid:
        return IndexedStructureResult(False, None, checked.issues)
    flow_scenes = cast(list[dict[str, object]], approved_flow["scenes"])
    response_scenes = cast(list[dict[str, object]], response["scenes"])
    materials = {
        f"material-{index:03d}": {"description": material["description"]}
        for index, material in enumerate(
            cast(list[dict[str, object]], response["materials"]), start=1
        )
    }
    sections: dict[str, object] = {
        "section-root": {
            "parent_section_id": None,
            "order": 0,
            "role": "whole",
            "description": approved_flow["overall_flow"],
        }
    }
    placements: dict[str, object] = {}
    child_counts: dict[str, int] = {"section-root": 0}
    section_number = 0
    placement_number = 0
    scene_mappings: list[dict[str, object]] = []
    for scene_index, scene in enumerate(response_scenes):
        stack: list[str] = []
        scene_section_ids: list[str] = []
        for raw_section in cast(list[dict[str, object]], scene["sections"]):
            section_number += 1
            section_id = f"section-{section_number:03d}"
            depth = cast(int, raw_section["depth"])
            parent_id = "section-root" if depth == 0 else stack[depth - 1]
            order = child_counts.get(parent_id, 0)
            child_counts[parent_id] = order + 1
            section = {
                "parent_section_id": parent_id,
                "order": order,
                "role": raw_section["role"],
                "description": raw_section["description"],
            }
            if raw_section["relative_length"] is not None:
                section["relative_length"] = raw_section["relative_length"]
            sections[section_id] = section
            stack[depth:] = [section_id]
            scene_section_ids.append(section_id)
            for raw_placement in cast(list[dict[str, object]], raw_section["placements"]):
                placement_number += 1
                placement_id = f"material-placement-{placement_number:03d}"
                placements[placement_id] = {
                    "section_id": section_id,
                    "material_id": f"material-{cast(int, raw_placement['material_index']) + 1:03d}",
                    "role": raw_placement["role"],
                }
        scene_mappings.append(
            {
                "scene_id": flow_scenes[scene_index]["scene_id"],
                "section_ids": scene_section_ids,
            }
        )
    approval = cast(dict[str, object], approved_flow["approval"])
    indexed = {
        "document_type": "indexed_structure",
        "schema_version": 1,
        "composition_id": composition_id,
        "source_flow": {
            "document_id": approved_flow["document_id"],
            "revision": approved_flow["revision"],
            "content_sha256": approval["content_sha256"],
        },
        "script": {
            "title": approved_flow["title"],
            "brief": approved_flow["overall_flow"],
            "performance_setup": {
                "instrumentation": approved_flow["instrumentation"],
                "target_duration_seconds": approved_flow["target_duration"],
            },
            "root_section_id": "section-root",
            "sections": sections,
            "materials": materials,
            "material_placements": placements,
        },
        "scene_section_mappings": scene_mappings,
    }
    return IndexedStructureResult(True, indexed, ())


def build_script_document(
    approved_flow: Mapping[str, object],
    indexed_structure: Mapping[str, object],
    response: Mapping[str, object],
    capabilities: GenerationProfileCapabilities,
) -> ScriptBuildResult:
    """Validate the second response and construct an immutable script document."""
    schema_errors = sorted(
        Draft202012Validator(
            relations_response_schema(indexed_structure, capabilities)
        ).iter_errors(response),
        key=lambda error: (tuple(str(part) for part in error.absolute_path), error.message),
    )
    if schema_errors:
        issues = tuple(
            _model_issue(
                error.message,
                "".join(f"/{_escape_pointer_token(part)}" for part in error.absolute_path),
            )
            for error in schema_errors
        )
        return ScriptBuildResult(False, "content_invalid", None, (), issues)

    outcome = cast(str, response["outcome"])
    relation_fields = (
        "variation_relations",
        "material_placement_transitions",
        "performance_directions",
        "performance_comparison_requirements",
    )
    if outcome == "structure_insufficient":
        issues = []
        if response["structure_insufficient_reason"] is None:
            issues.append(
                _model_issue(
                    "A structure-insufficient response requires a reason",
                    "/structure_insufficient_reason",
                )
            )
        for field in relation_fields:
            if cast(list[object], response[field]):
                issues.append(
                    _model_issue(
                        "A structure-insufficient response cannot contain relation candidates",
                        f"/{field}",
                    )
                )
        if issues:
            return ScriptBuildResult(False, "content_invalid", None, (), tuple(issues))
        return ScriptBuildResult(False, outcome, None, (), ())
    if response["structure_insufficient_reason"] is not None:
        return ScriptBuildResult(
            False,
            "content_invalid",
            None,
            (),
            (
                _model_issue(
                    "A complete response cannot contain a structure-insufficient reason",
                    "/structure_insufficient_reason",
                ),
            ),
        )

    script = copy.deepcopy(cast(dict[str, object], indexed_structure["script"]))
    variations = {
        f"variation-{index:03d}": copy.deepcopy(item)
        for index, item in enumerate(
            cast(list[dict[str, object]], response["variation_relations"]), start=1
        )
    }
    transitions = {
        f"material-placement-transition-{index:03d}": copy.deepcopy(item)
        for index, item in enumerate(
            cast(list[dict[str, object]], response["material_placement_transitions"]),
            start=1,
        )
    }
    directions: dict[str, object] = {}
    direction_issues: list[ValidationIssue] = []
    raw_directions = cast(list[dict[str, object]], response["performance_directions"])
    for index, item in enumerate(raw_directions, start=1):
        target_type = item["target_type"]
        target_id = item["target_id"]
        relative_type = item["relative_to_type"]
        relative_id = item["relative_to_id"]
        if (relative_type is None) != (relative_id is None):
            direction_issues.append(
                _model_issue(
                    "A comparative direction requires both relative target fields",
                    f"/performance_directions/{index - 1}",
                )
            )
            continue
        direction: dict[str, object] = {
            "target": {"type": target_type, "id": target_id},
            "description": item["description"],
        }
        if relative_type is not None:
            direction["relative_to"] = {"type": relative_type, "id": relative_id}
        directions[f"performance-direction-{index:03d}"] = direction
    requirements: dict[str, object] = {}
    raw_requirements = cast(
        list[dict[str, object]], response["performance_comparison_requirements"]
    )
    for index, item in enumerate(raw_requirements, start=1):
        direction_index = cast(int, item["performance_direction_index"])
        if not 0 <= direction_index < len(raw_directions):
            direction_issues.append(
                _model_issue(
                    "A comparison refers to an unknown performance direction index",
                    f"/performance_comparison_requirements/{index - 1}/performance_direction_index",
                )
            )
            continue
        requirements[f"performance-comparison-{index:03d}"] = {
            "performance_direction_id": f"performance-direction-{direction_index + 1:03d}",
            "feature": item["feature"],
            "relation": item["relation"],
        }
    if direction_issues:
        return ScriptBuildResult(False, "content_invalid", None, (), tuple(direction_issues))
    setup = cast(dict[str, object], script["performance_setup"])
    setup["performance_directions"] = directions
    script["script_element_variation_relations"] = variations
    script["material_placement_transitions"] = transitions
    script["performance_direction_comparison_requirements"] = requirements
    document = {
        "document_type": "script",
        "schema_version": "0.3.0",
        "document_id": indexed_structure["composition_id"],
        "revision": 1,
        "status": "validated",
        "source_flow": copy.deepcopy(indexed_structure["source_flow"]),
        "script": script,
    }
    checked = check_script_0_3_document(document)
    if not checked.valid:
        return ScriptBuildResult(False, "content_invalid", None, (), checked.issues)
    profile_checked = check_generation_profile_document(document, capabilities)
    if not profile_checked.valid:
        return ScriptBuildResult(False, "content_invalid", None, (), profile_checked.issues)
    ledger = _build_projection_ledger(approved_flow, indexed_structure, document)
    validate_projection_ledger(ledger)
    return ScriptBuildResult(True, "complete", document, ledger, ())


def _build_projection_ledger(
    approved_flow: Mapping[str, object],
    indexed_structure: Mapping[str, object],
    document: Mapping[str, object],
) -> tuple[ProjectionLedgerEntry, ...]:
    entries: list[ProjectionLedgerEntry] = []
    direct_fields = {
        "title": "/script/title",
        "instrumentation": "/script/performance_setup/instrumentation",
        "target_duration": "/script/performance_setup/target_duration_seconds",
    }
    for field, target_id in direct_fields.items():
        entries.append(
            ProjectionLedgerEntry(
                "flow_field",
                f"/{field}",
                "script_field",
                target_id,
                "script",
                "direct_value_equality",
                "passed",
                f"{approved_flow[field]!r}={target_id}",
            )
        )
    entries.append(
        ProjectionLedgerEntry(
            "flow_field",
            "/overall_flow",
            "generation_context",
            "/script/brief",
            "script_relations",
            "natural_language_claim",
            "unverified",
            "forwarded to the script brief and model context",
        )
    )
    mappings = cast(list[dict[str, object]], indexed_structure["scene_section_mappings"])
    for scene_index in range(len(cast(list[object], approved_flow["scenes"]))):
        target = ",".join(cast(list[str], mappings[scene_index]["section_ids"]))
        for field in (
            "scene_id",
            "name",
            "length_class",
            "heard_as",
            "relation_to_previous",
            "transition_to_next",
        ):
            entries.append(
                ProjectionLedgerEntry(
                    "flow_scene_field",
                    f"/scenes/{scene_index}/{field}",
                    "generation_context",
                    target,
                    "script_relations",
                    "natural_language_claim",
                    "unverified",
                    "forwarded to the structure or relations model context",
                )
            )
    script = cast(dict[str, dict[str, object]], document["script"])
    for collection in (
        "sections",
        "materials",
        "material_placements",
        "script_element_variation_relations",
        "material_placement_transitions",
    ):
        for item_id in cast(dict[str, object], script[collection]):
            entries.append(
                ProjectionLedgerEntry(
                    "model_candidate",
                    item_id,
                    collection,
                    item_id,
                    "script",
                    "direct_id_equality",
                    "passed",
                    f"Python assigned and validated {item_id}",
                )
            )
    setup = cast(dict[str, object], script["performance_setup"])
    for item_id in cast(dict[str, object], setup["performance_directions"]):
        entries.append(
            ProjectionLedgerEntry(
                "model_candidate",
                item_id,
                "performance_directions",
                item_id,
                "script",
                "direct_id_equality",
                "passed",
                f"Python assigned and validated {item_id}",
            )
        )
    for item_id in cast(dict[str, object], script["performance_direction_comparison_requirements"]):
        entries.append(
            ProjectionLedgerEntry(
                "model_candidate",
                item_id,
                "performance_direction_comparison_requirements",
                item_id,
                "script",
                "direct_id_equality",
                "passed",
                f"Python assigned and validated {item_id}",
            )
        )
    return tuple(entries)


def _escape_pointer_token(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _model_issue(message: str, path: str) -> ValidationIssue:
    return ValidationIssue(IssueCode.MODEL_OUTPUT_INVALID, message, path)
