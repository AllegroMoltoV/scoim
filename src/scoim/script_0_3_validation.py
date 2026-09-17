"""Validation for immutable SCoIM script 0.3.0 documents."""

import hashlib
import json
from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from typing import cast

import rfc8785
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .validation import CheckResult, IssueCode, ValidationIssue

SCRIPT_0_3_DOCUMENT_TYPE = "script"
SCRIPT_0_3_SCHEMA_VERSION = "0.3.0"
_CONTENT_FIELDS = (
    "document_type",
    "schema_version",
    "document_id",
    "revision",
    "source_flow",
    "script",
)


def check_script_0_3_document(document: Mapping[str, object]) -> CheckResult:
    """Check a script 0.3.0 document through its version-specific boundary."""
    schema_version = document.get("schema_version")
    if isinstance(schema_version, str) and schema_version != SCRIPT_0_3_SCHEMA_VERSION:
        return CheckResult(
            False,
            (
                ValidationIssue(
                    IssueCode.UNSUPPORTED_SCHEMA_VERSION,
                    f"Unsupported script schema version: {schema_version!r}",
                    "/schema_version",
                ),
            ),
        )
    schema_issues = tuple(
        ValidationIssue(IssueCode.SCHEMA_INVALID, error.message, _error_pointer(error))
        for error in sorted(
            _validator().iter_errors(document),
            key=lambda item: (tuple(str(part) for part in item.absolute_path), item.message),
        )
    )
    if schema_issues:
        return CheckResult(False, schema_issues)
    script = cast(dict[str, object], document["script"])
    semantic_issues = (
        *_check_section_tree(script),
        *_check_placement_references(script),
        *_check_variation_relations(script),
        *_check_material_placement_transitions(script),
        *_check_performance_direction_references(script),
        *_check_performance_comparison_requirements(script),
    )
    return CheckResult(not semantic_issues, semantic_issues)


def script_0_3_content_sha256(document: Mapping[str, object]) -> str:
    """Hash the immutable content and source-flow lineage of a script 0.3.0 document."""
    content = {field: document[field] for field in _CONTENT_FIELDS}
    return hashlib.sha256(rfc8785.dumps(content)).hexdigest()


def _check_section_tree(script: dict[str, object]) -> tuple[ValidationIssue, ...]:
    sections = cast(dict[str, object], script["sections"])
    root_section_id = cast(str, script["root_section_id"])
    parent_by_id = {
        section_id: cast(dict[str, object], raw_section)["parent_section_id"]
        for section_id, raw_section in sections.items()
    }
    issues: list[ValidationIssue] = []
    if root_section_id not in sections:
        issues.append(
            ValidationIssue(
                IssueCode.SEMANTIC_INVALID,
                f"Root section does not exist: {root_section_id}",
                "/script/root_section_id",
            )
        )
    children_by_parent: dict[str, list[tuple[int, str]]] = {}
    for section_id, parent_id in parent_by_id.items():
        path = f"/script/sections/{_escape_pointer_token(section_id)}/parent_section_id"
        if section_id == root_section_id and parent_id is not None:
            issues.append(
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    "The designated root section must not have a parent",
                    path,
                )
            )
        elif section_id != root_section_id and parent_id is None:
            issues.append(
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    "Only the designated root section may have no parent",
                    path,
                )
            )
        if isinstance(parent_id, str):
            if parent_id not in sections:
                issues.append(
                    ValidationIssue(
                        IssueCode.SEMANTIC_INVALID,
                        f"Referenced parent section does not exist: {parent_id}",
                        path,
                    )
                )
            else:
                order = cast(int, cast(dict[str, object], sections[section_id])["order"])
                children_by_parent.setdefault(parent_id, []).append((order, section_id))

    for children in children_by_parent.values():
        for expected_order, (actual_order, section_id) in enumerate(sorted(children)):
            if actual_order != expected_order:
                issues.append(
                    ValidationIssue(
                        IssueCode.SEMANTIC_INVALID,
                        "Sibling order must be consecutive from zero; "
                        f"expected {expected_order}, got {actual_order}",
                        f"/script/sections/{_escape_pointer_token(section_id)}/order",
                    )
                )

    for section_id, raw_section in sections.items():
        section = cast(dict[str, object], raw_section)
        has_children = section_id in children_by_parent
        has_length = "relative_length" in section
        if has_children and has_length:
            issues.append(
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    "A branch section must derive its length from its children",
                    f"/script/sections/{_escape_pointer_token(section_id)}/relative_length",
                )
            )
        elif not has_children and not has_length:
            issues.append(
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    "A leaf section must have relative_length",
                    f"/script/sections/{_escape_pointer_token(section_id)}/relative_length",
                )
            )

    state: dict[str, str] = {}

    def visit(section_id: str) -> None:
        state[section_id] = "visiting"
        parent_id = parent_by_id[section_id]
        if isinstance(parent_id, str) and parent_id in sections:
            if state.get(parent_id) == "visiting":
                issues.append(
                    ValidationIssue(
                        IssueCode.SEMANTIC_INVALID,
                        f"Section containment cycle reaches: {parent_id}",
                        f"/script/sections/{_escape_pointer_token(section_id)}/parent_section_id",
                    )
                )
            elif state.get(parent_id) is None:
                visit(parent_id)
        state[section_id] = "done"

    for section_id in sections:
        if state.get(section_id) is None:
            visit(section_id)
    return tuple(issues)


def _check_placement_references(script: dict[str, object]) -> tuple[ValidationIssue, ...]:
    sections = cast(dict[str, object], script["sections"])
    materials = cast(dict[str, object], script["materials"])
    placements = cast(dict[str, object], script["material_placements"])
    branch_section_ids = {
        cast(str, cast(dict[str, object], section)["parent_section_id"])
        for section in sections.values()
        if cast(dict[str, object], section)["parent_section_id"] is not None
    }
    issues: list[ValidationIssue] = []
    for placement_id, raw_placement in placements.items():
        placement = cast(dict[str, object], raw_placement)
        section_id = cast(str, placement["section_id"])
        material_id = cast(str, placement["material_id"])
        path = f"/script/material_placements/{_escape_pointer_token(placement_id)}"
        if section_id not in sections:
            issues.append(
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    f"Referenced section does not exist: {section_id}",
                    f"{path}/section_id",
                )
            )
        elif section_id in branch_section_ids:
            issues.append(
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    "A material placement must target a leaf section",
                    f"{path}/section_id",
                )
            )
        if material_id not in materials:
            issues.append(
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    f"Referenced material does not exist: {material_id}",
                    f"{path}/material_id",
                )
            )
    return tuple(issues)


def _check_variation_relations(script: dict[str, object]) -> tuple[ValidationIssue, ...]:
    collections = {
        "section": cast(dict[str, object], script["sections"]),
        "material": cast(dict[str, object], script["materials"]),
        "material_placement": cast(dict[str, object], script["material_placements"]),
    }
    relations = cast(dict[str, object], script["script_element_variation_relations"])
    targets: set[tuple[str, str]] = set()
    issues: list[ValidationIssue] = []
    material_edges: dict[str, list[tuple[str, str]]] = {}
    spans_by_type = {
        "section": _section_performance_spans(script),
        "material_placement": _material_placement_spans(script),
    }
    for relation_id, raw_relation in relations.items():
        relation = cast(dict[str, object], raw_relation)
        source = cast(dict[str, object], relation["source"])
        target = cast(dict[str, object], relation["target"])
        path = f"/script/script_element_variation_relations/{_escape_pointer_token(relation_id)}"
        source_type = cast(str, source["type"])
        target_type = cast(str, target["type"])
        source_id = cast(str, source["id"])
        target_id = cast(str, target["id"])
        if source_type != target_type:
            issues.append(
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    "Variation source and target must have the same type",
                    f"{path}/target/type",
                )
            )
        if source_id == target_id:
            issues.append(
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    "Variation source and target must have different IDs",
                    f"{path}/target/id",
                )
            )
        target_key = (target_type, target_id)
        if target_key in targets:
            issues.append(
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    "A variation target must be unique",
                    f"{path}/target/id",
                )
            )
        targets.add(target_key)
        preserved = set(cast(list[str], relation["preserve"]))
        for index, changed in enumerate(cast(list[str], relation["change"])):
            if changed in preserved:
                issues.append(
                    ValidationIssue(
                        IssueCode.SEMANTIC_INVALID,
                        "Variation preserve and change items must be disjoint",
                        f"{path}/change/{index}",
                    )
                )
        for field, reference_type, reference_id in (
            ("source", source_type, source_id),
            ("target", target_type, target_id),
        ):
            if reference_id not in collections[reference_type]:
                issues.append(
                    ValidationIssue(
                        IssueCode.SEMANTIC_INVALID,
                        f"Referenced {reference_type} does not exist: {reference_id}",
                        f"{path}/{field}/id",
                    )
                )
        if source_type == target_type and source_type in spans_by_type:
            spans = spans_by_type[source_type]
            if (
                source_id in spans
                and target_id in spans
                and spans[source_id][1] >= spans[target_id][0]
            ):
                issues.append(
                    ValidationIssue(
                        IssueCode.SEMANTIC_INVALID,
                        f"A {source_type} variation source must precede its target",
                        f"{path}/target/id",
                    )
                )
        if (
            source_type == "material"
            and target_type == "material"
            and source_id in collections["material"]
            and target_id in collections["material"]
        ):
            material_edges.setdefault(source_id, []).append((target_id, relation_id))
    issues.extend(_check_material_variation_cycles(material_edges, collections["material"]))
    return tuple(issues)


def _check_material_variation_cycles(
    edges: dict[str, list[tuple[str, str]]], materials: dict[str, object]
) -> list[ValidationIssue]:
    state: dict[str, str] = {}
    issues: list[ValidationIssue] = []

    def visit(material_id: str) -> None:
        state[material_id] = "visiting"
        for target_id, relation_id in edges.get(material_id, []):
            if state.get(target_id) == "visiting":
                issues.append(
                    ValidationIssue(
                        IssueCode.SEMANTIC_INVALID,
                        f"Material variation cycle reaches: {target_id}",
                        "/script/script_element_variation_relations/"
                        f"{_escape_pointer_token(relation_id)}/target/id",
                    )
                )
            elif state.get(target_id) is None:
                visit(target_id)
        state[material_id] = "done"

    for material_id in materials:
        if state.get(material_id) is None:
            visit(material_id)
    return issues


def _check_material_placement_transitions(
    script: dict[str, object],
) -> tuple[ValidationIssue, ...]:
    placements = cast(dict[str, object], script["material_placements"])
    transitions = cast(dict[str, object], script["material_placement_transitions"])
    placement_count_by_section: dict[str, int] = {}
    for raw_placement in placements.values():
        section_id = cast(str, cast(dict[str, object], raw_placement)["section_id"])
        placement_count_by_section[section_id] = placement_count_by_section.get(section_id, 0) + 1
    issues: list[ValidationIssue] = []
    fields = (
        "source_material_placement_id",
        "transition_material_placement_id",
        "target_material_placement_id",
    )
    placement_spans = _material_placement_spans(script)
    for transition_id, raw_transition in transitions.items():
        transition = cast(dict[str, object], raw_transition)
        path = f"/script/material_placement_transitions/{_escape_pointer_token(transition_id)}"
        seen: set[str] = set()
        for field in fields:
            placement_id = cast(str, transition[field])
            if placement_id in seen:
                issues.append(
                    ValidationIssue(
                        IssueCode.SEMANTIC_INVALID,
                        "Transition material placements must be distinct",
                        f"{path}/{field}",
                    )
                )
            seen.add(placement_id)
            if placement_id not in placements:
                issues.append(
                    ValidationIssue(
                        IssueCode.SEMANTIC_INVALID,
                        f"Referenced material_placement does not exist: {placement_id}",
                        f"{path}/{field}",
                    )
                )
        transition_placement_id = cast(str, transition["transition_material_placement_id"])
        if transition_placement_id in placements:
            transition_placement = cast(dict[str, object], placements[transition_placement_id])
            section_id = cast(str, transition_placement["section_id"])
            if placement_count_by_section[section_id] != 1:
                issues.append(
                    ValidationIssue(
                        IssueCode.SEMANTIC_INVALID,
                        "A transition material placement must occupy a dedicated leaf section",
                        f"{path}/transition_material_placement_id",
                    )
                )
        source_id = cast(str, transition["source_material_placement_id"])
        target_id = cast(str, transition["target_material_placement_id"])
        if all(
            placement_id in placement_spans
            for placement_id in (source_id, transition_placement_id, target_id)
        ):
            source_span = placement_spans[source_id]
            transition_span = placement_spans[transition_placement_id]
            target_span = placement_spans[target_id]
            if source_span[1] >= transition_span[0] or transition_span[1] >= target_span[0]:
                issues.append(
                    ValidationIssue(
                        IssueCode.SEMANTIC_INVALID,
                        "Transition order must be source, transition, then target",
                        f"{path}/transition_material_placement_id",
                    )
                )
    return tuple(issues)


def _material_placement_spans(script: dict[str, object]) -> dict[str, tuple[int, int]]:
    section_spans = _section_performance_spans(script)
    placements = cast(dict[str, object], script["material_placements"])
    spans: dict[str, tuple[int, int]] = {}
    for placement_id, raw_placement in placements.items():
        section_id = cast(str, cast(dict[str, object], raw_placement)["section_id"])
        if section_id in section_spans:
            spans[placement_id] = section_spans[section_id]
    return spans


def _section_performance_spans(script: dict[str, object]) -> dict[str, tuple[int, int]]:
    sections = cast(dict[str, object], script["sections"])
    root_section_id = cast(str, script["root_section_id"])
    if root_section_id not in sections:
        return {}
    children: dict[str, list[str]] = {}
    for section_id, raw_section in sections.items():
        parent_id = cast(dict[str, object], raw_section)["parent_section_id"]
        if isinstance(parent_id, str) and parent_id in sections:
            children.setdefault(parent_id, []).append(section_id)
    for child_ids in children.values():
        child_ids.sort(key=lambda item: cast(dict[str, object], sections[item])["order"])
    spans: dict[str, tuple[int, int]] = {}
    visiting: set[str] = set()
    next_leaf_index = 0

    def visit(section_id: str) -> tuple[int, int] | None:
        nonlocal next_leaf_index
        if section_id in visiting:
            return None
        visiting.add(section_id)
        child_ids = children.get(section_id, [])
        if not child_ids:
            span = (next_leaf_index, next_leaf_index)
            next_leaf_index += 1
        else:
            child_spans = [span for child_id in child_ids if (span := visit(child_id))]
            if not child_spans:
                visiting.remove(section_id)
                return None
            span = (child_spans[0][0], child_spans[-1][1])
        visiting.remove(section_id)
        spans[section_id] = span
        return span

    visit(root_section_id)
    return spans


def _check_performance_direction_references(
    script: dict[str, object],
) -> tuple[ValidationIssue, ...]:
    collections = {
        "section": cast(dict[str, object], script["sections"]),
        "material_placement": cast(dict[str, object], script["material_placements"]),
    }
    setup = cast(dict[str, object], script["performance_setup"])
    directions = cast(dict[str, object], setup["performance_directions"])
    issues: list[ValidationIssue] = []
    for direction_id, raw_direction in directions.items():
        direction = cast(dict[str, object], raw_direction)
        path = (
            "/script/performance_setup/performance_directions/"
            f"{_escape_pointer_token(direction_id)}"
        )
        for field in ("target", "relative_to"):
            if field not in direction:
                continue
            reference = cast(dict[str, object], direction[field])
            reference_type = cast(str, reference["type"])
            reference_id = cast(str, reference["id"])
            if reference_id not in collections[reference_type]:
                issues.append(
                    ValidationIssue(
                        IssueCode.SEMANTIC_INVALID,
                        f"Referenced {reference_type} does not exist: {reference_id}",
                        f"{path}/{field}/id",
                    )
                )
    return tuple(issues)


def _check_performance_comparison_requirements(
    script: dict[str, object],
) -> tuple[ValidationIssue, ...]:
    setup = cast(dict[str, object], script["performance_setup"])
    directions = cast(dict[str, object], setup["performance_directions"])
    requirements = cast(dict[str, object], script["performance_direction_comparison_requirements"])
    sections = cast(dict[str, object], script["sections"])
    seen_direction_features: set[tuple[str, str]] = set()
    comparison_edges: dict[str, dict[str, set[str]]] = {}
    issues: list[ValidationIssue] = []
    for requirement_id, raw_requirement in sorted(requirements.items()):
        requirement = cast(dict[str, object], raw_requirement)
        path = (
            "/script/performance_direction_comparison_requirements/"
            f"{_escape_pointer_token(requirement_id)}"
        )
        direction_id = cast(str, requirement["performance_direction_id"])
        feature = cast(str, requirement["feature"])
        direction_feature = (direction_id, feature)
        if direction_feature in seen_direction_features:
            issues.append(
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    "A performance direction and feature can only be required once",
                    f"{path}/feature",
                )
            )
            continue
        seen_direction_features.add(direction_feature)
        if direction_id not in directions:
            issues.append(
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    f"Referenced performance direction does not exist: {direction_id}",
                    f"{path}/performance_direction_id",
                )
            )
            continue
        direction = cast(dict[str, object], directions[direction_id])
        if "relative_to" not in direction:
            issues.append(
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    "A comparison requirement needs a comparative performance direction",
                    f"{path}/performance_direction_id",
                )
            )
            continue
        target = cast(dict[str, object], direction["target"])
        reference = cast(dict[str, object], direction["relative_to"])
        target_type = cast(str, target["type"])
        reference_type = cast(str, reference["type"])
        target_id = cast(str, target["id"])
        reference_id = cast(str, reference["id"])
        if target_type != reference_type:
            issues.append(
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    "A comparison requirement must compare the same target type",
                    f"{path}/performance_direction_id",
                )
            )
            continue
        if target_id == reference_id:
            issues.append(
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    "A comparison requirement must compare different targets",
                    f"{path}/performance_direction_id",
                )
            )
            continue
        if target_type == "section" and (
            _is_section_ancestor(sections, target_id, reference_id)
            or _is_section_ancestor(sections, reference_id, target_id)
        ):
            issues.append(
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    "A comparison requirement cannot compare ancestor and descendant sections",
                    f"{path}/performance_direction_id",
                )
            )
            continue
        relation = cast(str, requirement["relation"])
        lower_id, higher_id = (
            (reference_id, target_id) if relation == "more" else (target_id, reference_id)
        )
        typed_lower_id = f"{target_type}:{lower_id}"
        typed_higher_id = f"{target_type}:{higher_id}"
        edges = comparison_edges.setdefault(feature, {})
        if _has_directed_path(edges, typed_higher_id, typed_lower_id):
            issues.append(
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    f"Requirements for {feature} contain a comparison cycle",
                    f"{path}/relation",
                )
            )
            continue
        edges.setdefault(typed_lower_id, set()).add(typed_higher_id)
    return tuple(issues)


def _is_section_ancestor(sections: dict[str, object], ancestor_id: str, descendant_id: str) -> bool:
    current_id = descendant_id
    seen: set[str] = set()
    while current_id in sections and current_id not in seen:
        seen.add(current_id)
        section = cast(dict[str, object], sections[current_id])
        parent_id = section["parent_section_id"]
        if parent_id == ancestor_id:
            return True
        if not isinstance(parent_id, str):
            return False
        current_id = parent_id
    return False


def _has_directed_path(edges: dict[str, set[str]], start_id: str, target_id: str) -> bool:
    pending = [start_id]
    visited: set[str] = set()
    while pending:
        current_id = pending.pop()
        if current_id == target_id:
            return True
        if current_id in visited:
            continue
        visited.add(current_id)
        pending.extend(edges.get(current_id, ()))
    return False


@cache
def _validator() -> Draft202012Validator:
    schema_text = files("scoim").joinpath("schemas", "script-0.3.0.schema.json").read_text("utf-8")
    schema = json.loads(schema_text)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _error_pointer(error: ValidationError) -> str:
    return "".join(f"/{_escape_pointer_token(part)}" for part in error.absolute_path)


def _escape_pointer_token(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")
