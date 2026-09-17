"""Private model-transfer contracts for compiling script 0.4.0 documents."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import cast

from jsonschema import Draft202012Validator

from .profile_capabilities import (
    GenerationProfileCapabilities,
    check_generation_profile_document,
)
from .projection_ledger import ProjectionLedgerEntry, validate_projection_ledger
from .score_operation_preflight import build_score_operation_plan
from .script_0_3_model_contracts import (
    IndexedStructureResult,
    ScriptBuildResult,
)
from .script_0_3_model_contracts import (
    check_structure_candidate as _check_structure_candidate_0_3,
)
from .script_0_3_model_contracts import (
    index_structure_candidate as _index_structure_candidate_0_3,
)
from .script_0_3_model_contracts import structure_prompt as _structure_prompt_0_3
from .script_0_3_model_contracts import (
    structure_response_schema as _structure_response_schema_0_3,
)
from .script_0_4_validation import check_script_0_4_document
from .validation import CheckResult, IssueCode, ValidationIssue

_TEXT_SCHEMA = {"type": "string", "minLength": 1}

__all__ = [
    "IndexedStructureResult",
    "ScriptBuildResult",
    "build_script_document",
    "check_structure_candidate",
    "index_structure_candidate",
    "relations_prompt",
    "relations_response_schema",
    "structure_prompt",
    "structure_response_schema",
]


def structure_response_schema(
    capabilities: GenerationProfileCapabilities,
) -> dict[str, object]:
    """Build the script-0.4-only structure transfer schema."""
    schema = copy.deepcopy(_structure_response_schema_0_3(capabilities))
    properties = cast(dict[str, object], schema["properties"])
    scenes = cast(dict[str, object], properties["scenes"])
    scene_item = cast(dict[str, object], scenes["items"])
    scene_properties = cast(dict[str, object], scene_item["properties"])
    sections = cast(dict[str, object], scene_properties["sections"])
    section_item = cast(dict[str, object], sections["items"])
    cast(list[str], section_item["required"]).append("structural_purpose")
    cast(dict[str, object], section_item["properties"])["structural_purpose"] = {
        "enum": ["regular", "transition_connector"]
    }
    return schema


def structure_prompt(
    approved_flow: Mapping[str, object], capabilities: GenerationProfileCapabilities
) -> str:
    """Build the script-0.4-only structure prompt."""
    base = _structure_prompt_0_3(approved_flow, capabilities)
    instructions, separator, context = base.partition("\n\n入力: ")
    return (
        instructions
        + "各区分のstructural_purposeはregularまたはtransition_connectorにしてください。"
        "独立した短い遷移区分が音楽上必要な場合だけtransition_connectorを使い、"
        "その区分にはforegroundの配置を一つだけ置いてください。"
        "transition_to_nextは聞こえ方の文章であり、それだけを理由に"
        "transition_connectorを作らないでください。" + separator + context
    )


def check_structure_candidate(
    approved_flow: Mapping[str, object],
    response: Mapping[str, object],
    capabilities: GenerationProfileCapabilities,
) -> CheckResult:
    """Validate the script-0.4 structure transfer and connector intent."""
    errors = sorted(
        Draft202012Validator(structure_response_schema(capabilities)).iter_errors(response),
        key=lambda error: (tuple(str(part) for part in error.absolute_path), error.message),
    )
    if errors:
        return CheckResult(
            False,
            tuple(
                _model_issue(
                    error.message,
                    "".join(f"/{_escape_pointer_token(part)}" for part in error.absolute_path),
                )
                for error in errors
            ),
        )

    stripped = _without_structural_purposes(response)
    base = _check_structure_candidate_0_3(approved_flow, stripped, capabilities)
    issues = list(base.issues)
    section_entries: list[tuple[dict[str, object], str, bool]] = []
    for scene_index, scene in enumerate(cast(list[dict[str, object]], response["scenes"])):
        sections = cast(list[dict[str, object]], scene["sections"])
        depths = [cast(int, section["depth"]) for section in sections]
        for section_index, section in enumerate(sections):
            has_child = (
                section_index + 1 < len(sections)
                and depths[section_index + 1] > depths[section_index]
            )
            section_entries.append(
                (section, f"/scenes/{scene_index}/sections/{section_index}", has_child)
            )

    leaf_entries = [entry for entry in section_entries if not entry[2]]
    for leaf_index, (section, path, _) in enumerate(leaf_entries):
        if section["structural_purpose"] != "transition_connector":
            continue
        placements = cast(list[dict[str, object]], section["placements"])
        if len(placements) != 1 or placements[0]["role"] != "foreground":
            issues.append(
                _model_issue(
                    "A transition connector must contain exactly one foreground placement",
                    f"{path}/placements",
                )
            )
        if not _has_nearby_regular_foreground(leaf_entries, leaf_index, -1):
            issues.append(
                _model_issue(
                    "A transition connector requires a preceding regular foreground placement",
                    f"{path}/structural_purpose",
                )
            )
        if not _has_nearby_regular_foreground(leaf_entries, leaf_index, 1):
            issues.append(
                _model_issue(
                    "A transition connector requires a following regular foreground placement",
                    f"{path}/structural_purpose",
                )
            )
    for section, path, has_child in section_entries:
        if has_child and section["structural_purpose"] != "regular":
            issues.append(
                _model_issue(
                    "A branch section must have regular structural purpose",
                    f"{path}/structural_purpose",
                )
            )
    return CheckResult(not issues, tuple(issues))


def index_structure_candidate(
    approved_flow: Mapping[str, object],
    composition_id: str,
    response: Mapping[str, object],
    capabilities: GenerationProfileCapabilities,
) -> IndexedStructureResult:
    """Assign stable IDs while retaining only private connector section IDs."""
    checked = check_structure_candidate(approved_flow, response, capabilities)
    if not checked.valid:
        return IndexedStructureResult(False, None, checked.issues)
    indexed_result = _index_structure_candidate_0_3(
        approved_flow,
        composition_id,
        _without_structural_purposes(response),
        capabilities,
    )
    if not indexed_result.valid or indexed_result.indexed_structure is None:
        return indexed_result
    indexed = indexed_result.indexed_structure
    connector_ids: list[str] = []
    section_number = 0
    for scene in cast(list[dict[str, object]], response["scenes"]):
        for section in cast(list[dict[str, object]], scene["sections"]):
            section_number += 1
            if section["structural_purpose"] == "transition_connector":
                connector_ids.append(f"section-{section_number:03d}")
    indexed["transition_connector_section_ids"] = connector_ids
    return IndexedStructureResult(True, indexed, ())


def relations_prompt(
    approved_flow: Mapping[str, object],
    indexed_structure: Mapping[str, object],
    capabilities: GenerationProfileCapabilities,
) -> str:
    """Build the second-operation prompt around the Python-indexed structure."""
    transition_candidate_groups = _transition_candidate_groups(indexed_structure)
    context = {
        "approved_flow": dict(approved_flow),
        "indexed_structure": dict(indexed_structure),
        "transition_candidate_groups": transition_candidate_groups,
        "profile_capabilities": {
            "profile_id": capabilities.profile_id,
            "performance_direction_target_types": sorted(
                capabilities.performance_direction_target_types
            ),
            "performance_aspects": [
                {
                    "aspect_id": aspect.aspect_id,
                    "description": aspect.description,
                }
                for aspect in capabilities.performance_aspects
            ],
            "score_relations": capabilities.score_relations.to_prompt_json(),
        },
    }
    return (
        "Pythonが確定した区分とマテリアル配置を変更しないでください。"
        "その構造へ、変奏関係、遷移、自然言語の演奏指示を追加してください。"
        "演奏指示は、提示した演奏要素だけを使い、楽譜を変えずに調整できる"
        "弾き方だけを記述してください。"
        "音域、旋律、伴奏音、和音の要望を演奏指示へ複写しないでください。"
        "それらの要望は区分とマテリアルの説明に残してください。"
        "マテリアル配置遷移は、transition_candidate_groupsの各groupから"
        "candidate_idをちょうど一つ選んでください。groupがない場合だけ空にしてください。"
        "遷移専用配置を変奏関係の元または先にしないでください。"
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
    transition_candidate_groups = _transition_candidate_groups(indexed_structure)
    transition_candidates = [
        candidate
        for group in transition_candidate_groups
        for candidate in cast(list[dict[str, str]], group["candidates"])
    ]
    transition_placement_ids = {
        candidate["transition_material_placement_id"] for candidate in transition_candidates
    }
    ids_by_type = {
        "section": sorted(cast(dict[str, object], script["sections"])),
        "material": sorted(cast(dict[str, object], script["materials"])),
        "material_placement": sorted(cast(dict[str, object], script["material_placements"])),
    }
    variation_ids_by_type = {
        kind: [
            item
            for item in values
            if kind != "material_placement" or item not in transition_placement_ids
        ]
        for kind, values in ids_by_type.items()
    }
    all_variation_ids = sorted(
        {item for values in variation_ids_by_type.values() for item in values}
    )
    reference = {
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "id"],
        "properties": {
            "type": {"enum": sorted(variation_ids_by_type)},
            "id": {"enum": all_variation_ids},
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
        "required": ["candidate_id", "description"],
        "properties": {
            "candidate_id": (
                {"enum": [candidate["candidate_id"] for candidate in transition_candidates]}
                if transition_candidates
                else {"type": "string"}
            ),
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
            "performance_aspects",
            "description",
        ],
        "properties": {
            "target_type": {"enum": direction_target_types},
            "target_id": {"enum": direction_target_ids},
            "relative_to_type": {"anyOf": [{"type": "null"}, {"enum": direction_target_types}]},
            "relative_to_id": {"anyOf": [{"type": "null"}, {"enum": direction_target_ids}]},
            "performance_aspects": {
                "type": "array",
                "minItems": 1,
                "items": {"enum": list(capabilities.performance_aspect_ids)},
            },
            "description": dict(_TEXT_SCHEMA),
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
        ],
        "properties": {
            "outcome": {"enum": ["complete", "structure_insufficient"]},
            "structure_insufficient_reason": {"anyOf": [{"type": "null"}, dict(_TEXT_SCHEMA)]},
            "variation_relations": {"type": "array", "items": variation},
            "material_placement_transitions": {
                "type": "array",
                "minItems": len(transition_candidate_groups),
                "maxItems": len(transition_candidate_groups),
                "items": transition,
            },
            "performance_directions": {"type": "array", "items": direction},
        },
    }


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
    candidate_groups = _transition_candidate_groups(indexed_structure)
    candidate_lookup: dict[str, tuple[str, dict[str, str]]] = {}
    for group in candidate_groups:
        connector_section_id = cast(str, group["transition_connector_section_id"])
        for candidate in cast(list[dict[str, str]], group["candidates"]):
            candidate_lookup[candidate["candidate_id"]] = (connector_section_id, candidate)
    transitions: dict[str, object] = {}
    transition_issues: list[ValidationIssue] = []
    selected_connector_ids: set[str] = set()
    for index, item in enumerate(
        cast(list[dict[str, object]], response["material_placement_transitions"]), start=1
    ):
        candidate_id = cast(str, item["candidate_id"])
        connector_section_id, candidate = candidate_lookup[candidate_id]
        if connector_section_id in selected_connector_ids:
            transition_issues.append(
                _model_issue(
                    "A transition connector must select exactly one candidate",
                    f"/material_placement_transitions/{index - 1}/candidate_id",
                )
            )
            continue
        selected_connector_ids.add(connector_section_id)
        transitions[f"material-placement-transition-{index:03d}"] = {
            "source_material_placement_id": candidate["source_material_placement_id"],
            "transition_material_placement_id": candidate["transition_material_placement_id"],
            "target_material_placement_id": candidate["target_material_placement_id"],
            "description": item["description"],
        }
    expected_connector_ids = {
        cast(str, group["transition_connector_section_id"]) for group in candidate_groups
    }
    if selected_connector_ids != expected_connector_ids:
        transition_issues.append(
            _model_issue(
                "Every transition connector must select exactly one candidate",
                "/material_placement_transitions",
            )
        )
    if transition_issues:
        return ScriptBuildResult(False, "content_invalid", None, (), tuple(transition_issues))
    directions: dict[str, object] = {}
    direction_issues: list[ValidationIssue] = []
    for index, item in enumerate(
        cast(list[dict[str, object]], response["performance_directions"]), start=1
    ):
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
            "target": {"type": item["target_type"], "id": item["target_id"]},
            "performance_aspects": copy.deepcopy(item["performance_aspects"]),
            "description": item["description"],
        }
        if relative_type is not None:
            direction["relative_to"] = {"type": relative_type, "id": relative_id}
        directions[f"performance-direction-{index:03d}"] = direction
    if direction_issues:
        return ScriptBuildResult(False, "content_invalid", None, (), tuple(direction_issues))

    setup = cast(dict[str, object], script["performance_setup"])
    setup["performance_directions"] = directions
    script["script_element_variation_relations"] = variations
    script["material_placement_transitions"] = transitions
    document = {
        "document_type": "script",
        "schema_version": "0.4.0",
        "document_id": indexed_structure["composition_id"],
        "revision": 1,
        "status": "validated",
        "source_flow": copy.deepcopy(indexed_structure["source_flow"]),
        "script": script,
    }
    checked = check_script_0_4_document(document)
    if not checked.valid:
        return ScriptBuildResult(False, "content_invalid", None, (), checked.issues)
    profile_checked = check_generation_profile_document(document, capabilities)
    if not profile_checked.valid:
        return ScriptBuildResult(False, "content_invalid", None, (), profile_checked.issues)
    operation_plan = build_score_operation_plan(document, capabilities)
    if operation_plan.issues:
        return ScriptBuildResult(False, "content_invalid", None, (), operation_plan.issues)
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
    return tuple(entries)


def _without_structural_purposes(
    response: Mapping[str, object],
) -> dict[str, object]:
    stripped = copy.deepcopy(dict(response))
    for scene in cast(list[dict[str, object]], stripped["scenes"]):
        for section in cast(list[dict[str, object]], scene["sections"]):
            section.pop("structural_purpose", None)
    return stripped


def _transition_candidate_groups(
    indexed_structure: Mapping[str, object],
) -> list[dict[str, object]]:
    script = cast(dict[str, object], indexed_structure["script"])
    placements = cast(dict[str, dict[str, object]], script["material_placements"])
    connector_section_ids = set(
        cast(list[str], indexed_structure.get("transition_connector_section_ids", []))
    )
    leaf_section_ids = _ordered_leaf_section_ids(script)
    foreground_by_section: dict[str, list[str]] = {}
    for placement_id, placement in placements.items():
        if placement["role"] == "foreground":
            foreground_by_section.setdefault(cast(str, placement["section_id"]), []).append(
                placement_id
            )
    for placement_ids in foreground_by_section.values():
        placement_ids.sort()

    groups: list[dict[str, object]] = []
    candidate_number = 0
    for leaf_index, section_id in enumerate(leaf_section_ids):
        if section_id not in connector_section_ids:
            continue
        source_ids = _nearest_regular_foreground_ids(
            leaf_section_ids,
            connector_section_ids,
            foreground_by_section,
            leaf_index,
            -1,
        )
        target_ids = _nearest_regular_foreground_ids(
            leaf_section_ids,
            connector_section_ids,
            foreground_by_section,
            leaf_index,
            1,
        )
        transition_ids = foreground_by_section.get(section_id, [])
        if not source_ids or not target_ids or len(transition_ids) != 1:
            continue
        candidates: list[dict[str, str]] = []
        for source_id in source_ids:
            for target_id in target_ids:
                candidate_number += 1
                candidates.append(
                    {
                        "candidate_id": f"transition-candidate-{candidate_number:03d}",
                        "source_material_placement_id": source_id,
                        "transition_material_placement_id": transition_ids[0],
                        "target_material_placement_id": target_id,
                    }
                )
        groups.append(
            {
                "transition_connector_section_id": section_id,
                "candidates": candidates,
            }
        )
    return groups


def _ordered_leaf_section_ids(script: Mapping[str, object]) -> list[str]:
    sections = cast(dict[str, dict[str, object]], script["sections"])
    children: dict[str, list[str]] = {}
    for section_id, section in sections.items():
        parent_id = section["parent_section_id"]
        if isinstance(parent_id, str):
            children.setdefault(parent_id, []).append(section_id)
    for section_ids in children.values():
        section_ids.sort(key=lambda item: (cast(int, sections[item]["order"]), item))

    leaves: list[str] = []

    def visit(section_id: str) -> None:
        child_ids = children.get(section_id, [])
        if not child_ids:
            leaves.append(section_id)
            return
        for child_id in child_ids:
            visit(child_id)

    visit(cast(str, script["root_section_id"]))
    return leaves


def _nearest_regular_foreground_ids(
    leaf_section_ids: list[str],
    connector_section_ids: set[str],
    foreground_by_section: Mapping[str, list[str]],
    start_index: int,
    step: int,
) -> list[str]:
    for index in range(start_index + step, len(leaf_section_ids) if step > 0 else -1, step):
        section_id = leaf_section_ids[index]
        if section_id in connector_section_ids:
            continue
        foreground_ids = foreground_by_section.get(section_id, [])
        if foreground_ids:
            return list(foreground_ids)
    return []


def _has_nearby_regular_foreground(
    leaf_entries: list[tuple[dict[str, object], str, bool]],
    start_index: int,
    step: int,
) -> bool:
    for index in range(start_index + step, len(leaf_entries) if step > 0 else -1, step):
        section = leaf_entries[index][0]
        if section["structural_purpose"] != "regular":
            continue
        if any(
            placement["role"] == "foreground"
            for placement in cast(list[dict[str, object]], section["placements"])
        ):
            return True
    return False


def _escape_pointer_token(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _model_issue(message: str, path: str) -> ValidationIssue:
    return ValidationIssue(IssueCode.MODEL_OUTPUT_INVALID, message, path)
