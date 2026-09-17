"""Validation results for SCoIM music-script documents."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from importlib.resources import files
from typing import cast

import rfc8785
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

SUPPORTED_SCHEMA_VERSION = "0.1.0"


class IssueCode(StrEnum):
    """Stable machine-readable validation issue codes."""

    UNSUPPORTED_SCHEMA_VERSION = "unsupported_schema_version"
    SCHEMA_INVALID = "schema_invalid"
    SEMANTIC_INVALID = "semantic_invalid"
    STALE_REVISION = "stale_revision"
    IMMUTABLE_APPROVED = "immutable_approved"
    MODEL_OUTPUT_INVALID = "model_output_invalid"
    RUNNER_UNAVAILABLE = "runner_unavailable"
    RUNNER_INCOMPATIBLE = "runner_incompatible"
    RUNNER_UNAUTHENTICATED = "runner_unauthenticated"
    RUNNER_TIMEOUT = "runner_timeout"
    RUNNER_FAILED = "runner_failed"
    PATCH_INVALID = "patch_invalid"
    QUERY_INVALID = "query_invalid"
    NOT_FOUND = "not_found"
    STORAGE_CONFLICT = "storage_conflict"
    STORAGE_ERROR = "storage_error"
    UNREPRESENTABLE = "unrepresentable"
    PROJECTION_TARGET_UNMET = "projection_target_unmet"
    LINEAGE_MISMATCH = "lineage_mismatch"
    UNSUPPORTED_PROFILE = "unsupported_profile"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One validation problem at an RFC 6901 JSON Pointer."""

    code: IssueCode
    message: str
    path: str


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Complete validation result for one document."""

    valid: bool
    issues: tuple[ValidationIssue, ...]


def check(document: Mapping[str, object]) -> CheckResult:
    """Check a decoded SCoIM document without raising for content errors."""
    schema_version = document.get("schema_version")
    if isinstance(schema_version, str) and schema_version != SUPPORTED_SCHEMA_VERSION:
        return CheckResult(
            valid=False,
            issues=(
                ValidationIssue(
                    code=IssueCode.UNSUPPORTED_SCHEMA_VERSION,
                    message=f"Unsupported schema version: {schema_version!r}",
                    path="/schema_version",
                ),
            ),
        )
    schema_issues = tuple(
        ValidationIssue(
            code=IssueCode.SCHEMA_INVALID,
            message=error.message,
            path=_error_pointer(error),
        )
        for error in sorted(
            _validator().iter_errors(document),
            key=lambda item: (tuple(str(part) for part in item.absolute_path), item.message),
        )
    )
    if schema_issues:
        return CheckResult(valid=False, issues=schema_issues)
    semantic_issues = _check_semantics(document)
    if semantic_issues:
        return CheckResult(valid=False, issues=semantic_issues)
    return CheckResult(valid=True, issues=())


@cache
def _validator() -> Draft202012Validator:
    schema_text = files("scoim").joinpath("schemas", "script-0.1.0.schema.json").read_text("utf-8")
    return Draft202012Validator(json.loads(schema_text))


def _error_pointer(error: ValidationError) -> str:
    return "".join(f"/{_escape_pointer_token(part)}" for part in error.absolute_path)


def _escape_pointer_token(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _check_semantics(document: Mapping[str, object]) -> tuple[ValidationIssue, ...]:
    script = cast(dict[str, object], document["script"])
    issues: list[ValidationIssue] = []
    if document["status"] == "draft" and "requirements" not in script:
        issues.append(
            ValidationIssue(
                code=IssueCode.SEMANTIC_INVALID,
                message="A draft must explicitly contain requirements",
                path="/script/requirements",
            )
        )
    issues.extend(check_script_semantics(script))
    issues.extend(_check_approval(document))
    return tuple(issues)


def check_script_semantics(script: dict[str, object]) -> tuple[ValidationIssue, ...]:
    """Check the shared inner script without document-state rules."""
    sections = cast(dict[str, object], script["sections"])
    materials = cast(dict[str, object], script["materials"])
    placements = cast(dict[str, object], script["placements"])
    issues: list[ValidationIssue] = []
    section_tree_issues = _check_section_tree(sections, cast(str, script["root_section_id"]))
    issues.extend(section_tree_issues)
    issues.extend(_check_relation_references(script))
    issues.extend(_check_requirement_references(script))
    if not section_tree_issues:
        issues.extend(_check_temporal_variation_order(script))
        issues.extend(_check_transition_order(script))
    branch_section_ids = {
        cast(str, cast(dict[str, object], raw_section)["parent_section_id"])
        for raw_section in sections.values()
        if cast(dict[str, object], raw_section)["parent_section_id"] is not None
    }
    for placement_id, raw_placement in placements.items():
        placement = cast(dict[str, object], raw_placement)
        for field, targets in (
            ("section_id", sections),
            ("material_id", materials),
        ):
            target_id = cast(str, placement[field])
            if target_id not in targets:
                issues.append(
                    ValidationIssue(
                        code=IssueCode.SEMANTIC_INVALID,
                        message=(
                            f"Referenced {field.removesuffix('_id')} does not exist: {target_id}"
                        ),
                        path=(f"/script/placements/{_escape_pointer_token(placement_id)}/{field}"),
                    )
                )
        section_id = cast(str, placement["section_id"])
        if section_id in branch_section_ids:
            issues.append(
                ValidationIssue(
                    code=IssueCode.SEMANTIC_INVALID,
                    message="A placement must target a leaf section",
                    path=(f"/script/placements/{_escape_pointer_token(placement_id)}/section_id"),
                )
            )
    return tuple(issues)


def _check_requirement_references(script: dict[str, object]) -> list[ValidationIssue]:
    requirements = cast(dict[str, object], script.get("requirements", {}))
    setup = cast(dict[str, object], script["performance_setup"])
    directions = cast(dict[str, object], setup["performance_directions"])
    sections = cast(dict[str, object], script["sections"])
    issues: list[ValidationIssue] = []
    seen_direction_features: set[tuple[str, str]] = set()
    comparison_edges: dict[str, dict[str, set[str]]] = {}
    for requirement_id, raw_requirement in sorted(requirements.items()):
        requirement = cast(dict[str, object], raw_requirement)
        direction_id = cast(str, requirement["performance_direction_id"])
        feature = cast(str, requirement["feature"])
        direction_feature = (direction_id, feature)
        if direction_feature in seen_direction_features:
            issues.append(
                ValidationIssue(
                    code=IssueCode.SEMANTIC_INVALID,
                    message="A performance direction and feature can only be required once",
                    path=(f"/script/requirements/{_escape_pointer_token(requirement_id)}/feature"),
                )
            )
            continue
        seen_direction_features.add(direction_feature)
        if direction_id not in directions:
            issues.append(
                ValidationIssue(
                    code=IssueCode.SEMANTIC_INVALID,
                    message=f"Referenced performance direction does not exist: {direction_id}",
                    path=(
                        f"/script/requirements/{_escape_pointer_token(requirement_id)}"
                        "/performance_direction_id"
                    ),
                )
            )
            continue
        direction = cast(dict[str, object], directions[direction_id])
        if "relative_to" not in direction:
            issues.append(
                ValidationIssue(
                    code=IssueCode.SEMANTIC_INVALID,
                    message="A requirement needs a comparative performance direction",
                    path=(
                        f"/script/requirements/{_escape_pointer_token(requirement_id)}"
                        "/performance_direction_id"
                    ),
                )
            )
            continue
        target = cast(dict[str, object], direction["target"])
        reference = cast(dict[str, object], direction["relative_to"])
        if target["type"] != "section" or reference["type"] != "section":
            issues.append(
                ValidationIssue(
                    code=IssueCode.SEMANTIC_INVALID,
                    message="A requirement can only compare sections",
                    path=(
                        f"/script/requirements/{_escape_pointer_token(requirement_id)}"
                        "/performance_direction_id"
                    ),
                )
            )
            continue
        if target["id"] == reference["id"]:
            issues.append(
                ValidationIssue(
                    code=IssueCode.SEMANTIC_INVALID,
                    message="A requirement must compare different sections",
                    path=(
                        f"/script/requirements/{_escape_pointer_token(requirement_id)}"
                        "/performance_direction_id"
                    ),
                )
            )
            continue
        target_id = cast(str, target["id"])
        reference_id = cast(str, reference["id"])
        if _is_section_ancestor(sections, target_id, reference_id) or _is_section_ancestor(
            sections, reference_id, target_id
        ):
            issues.append(
                ValidationIssue(
                    code=IssueCode.SEMANTIC_INVALID,
                    message="A requirement cannot compare ancestor and descendant sections",
                    path=(
                        f"/script/requirements/{_escape_pointer_token(requirement_id)}"
                        "/performance_direction_id"
                    ),
                )
            )
            continue
        relation = cast(str, requirement["relation"])
        lower_id, higher_id = (
            (reference_id, target_id) if relation == "more" else (target_id, reference_id)
        )
        edges = comparison_edges.setdefault(feature, {})
        if _has_directed_path(edges, higher_id, lower_id):
            issues.append(
                ValidationIssue(
                    code=IssueCode.SEMANTIC_INVALID,
                    message=f"Requirements for {feature} contain a comparison cycle",
                    path=(f"/script/requirements/{_escape_pointer_token(requirement_id)}/relation"),
                )
            )
            continue
        edges.setdefault(lower_id, set()).add(higher_id)
    return issues


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


def _check_approval(document: Mapping[str, object]) -> list[ValidationIssue]:
    if document["status"] == "draft" and document["approval"] is not None:
        return [
            ValidationIssue(
                code=IssueCode.SEMANTIC_INVALID,
                message="A draft must not contain approval metadata",
                path="/approval",
            )
        ]
    if document["status"] == "approved" and document["approval"] is None:
        return [
            ValidationIssue(
                code=IssueCode.SEMANTIC_INVALID,
                message="An approved document must contain approval metadata",
                path="/approval",
            )
        ]
    if document["status"] == "approved":
        approval = cast(dict[str, object], document["approval"])
        expected_hash = content_sha256(document)
        if approval["content_sha256"] != expected_hash:
            return [
                ValidationIssue(
                    code=IssueCode.SEMANTIC_INVALID,
                    message="Approval content hash does not match the document",
                    path="/approval/content_sha256",
                )
            ]
    return []


def content_sha256(document: Mapping[str, object]) -> str:
    hashed_content = {
        field: document[field] for field in ("schema_version", "document_id", "revision", "script")
    }
    return hashlib.sha256(rfc8785.dumps(hashed_content)).hexdigest()


def _check_relation_references(script: dict[str, object]) -> list[ValidationIssue]:
    collections = {
        "section": cast(dict[str, object], script["sections"]),
        "material": cast(dict[str, object], script["materials"]),
        "placement": cast(dict[str, object], script["placements"]),
    }
    placement_count_by_section: dict[str, int] = {}
    for raw_placement in collections["placement"].values():
        placement = cast(dict[str, object], raw_placement)
        section_id = cast(str, placement["section_id"])
        placement_count_by_section[section_id] = placement_count_by_section.get(section_id, 0) + 1
    issues: list[ValidationIssue] = []
    variations = cast(dict[str, object], script["variations"])
    variation_targets: set[tuple[str, str]] = set()
    for variation_id, raw_variation in variations.items():
        variation = cast(dict[str, object], raw_variation)
        source = cast(dict[str, object], variation["source"])
        target = cast(dict[str, object], variation["target"])
        if source["type"] != target["type"]:
            issues.append(
                ValidationIssue(
                    code=IssueCode.SEMANTIC_INVALID,
                    message="Variation source and target must have the same type",
                    path=(f"/script/variations/{_escape_pointer_token(variation_id)}/target/type"),
                )
            )
        if source["id"] == target["id"]:
            issues.append(
                ValidationIssue(
                    code=IssueCode.SEMANTIC_INVALID,
                    message="Variation source and target must have different IDs",
                    path=(f"/script/variations/{_escape_pointer_token(variation_id)}/target/id"),
                )
            )
        target_key = (cast(str, target["type"]), cast(str, target["id"]))
        if target_key in variation_targets:
            issues.append(
                ValidationIssue(
                    code=IssueCode.SEMANTIC_INVALID,
                    message="A variation target must be unique",
                    path=(f"/script/variations/{_escape_pointer_token(variation_id)}/target/id"),
                )
            )
        variation_targets.add(target_key)
        preserved = set(cast(list[str], variation["preserve"]))
        for change_index, changed_item in enumerate(cast(list[str], variation["change"])):
            if changed_item in preserved:
                issues.append(
                    ValidationIssue(
                        code=IssueCode.SEMANTIC_INVALID,
                        message="Variation preserve and change items must be disjoint",
                        path=(
                            f"/script/variations/{_escape_pointer_token(variation_id)}"
                            f"/change/{change_index}"
                        ),
                    )
                )
        for field in ("source", "target"):
            reference = cast(dict[str, object], variation[field])
            reference_type = cast(str, reference["type"])
            reference_id = cast(str, reference["id"])
            if reference_id not in collections[reference_type]:
                issues.append(
                    _missing_reference_issue(
                        collection="variations",
                        owner_id=variation_id,
                        field=f"{field}/id",
                        target_type=reference_type,
                        target_id=reference_id,
                    )
                )
    issues.extend(_check_material_variation_cycles(variations, collections["material"]))

    transitions = cast(dict[str, object], script["transitions"])
    for transition_id, raw_transition in transitions.items():
        transition = cast(dict[str, object], raw_transition)
        seen_placement_ids: set[str] = set()
        for field in (
            "from_placement_id",
            "connector_placement_id",
            "to_placement_id",
        ):
            reference_id = cast(str, transition[field])
            if reference_id in seen_placement_ids:
                issues.append(
                    ValidationIssue(
                        code=IssueCode.SEMANTIC_INVALID,
                        message="Transition placements must be distinct",
                        path=(
                            f"/script/transitions/{_escape_pointer_token(transition_id)}/{field}"
                        ),
                    )
                )
            seen_placement_ids.add(reference_id)
            if reference_id not in collections["placement"]:
                issues.append(
                    _missing_reference_issue(
                        collection="transitions",
                        owner_id=transition_id,
                        field=field,
                        target_type="placement",
                        target_id=reference_id,
                    )
                )
        connector_id = cast(str, transition["connector_placement_id"])
        if connector_id in collections["placement"]:
            connector = cast(dict[str, object], collections["placement"][connector_id])
            connector_section_id = cast(str, connector["section_id"])
            if placement_count_by_section[connector_section_id] != 1:
                issues.append(
                    ValidationIssue(
                        code=IssueCode.SEMANTIC_INVALID,
                        message="A transition connector must occupy a dedicated leaf section",
                        path=(
                            f"/script/transitions/{_escape_pointer_token(transition_id)}"
                            "/connector_placement_id"
                        ),
                    )
                )

    performance_setup = cast(dict[str, object], script["performance_setup"])
    directions = cast(dict[str, object], performance_setup["performance_directions"])
    for direction_id, raw_direction in directions.items():
        direction = cast(dict[str, object], raw_direction)
        for field in ("target", "relative_to"):
            if field not in direction:
                continue
            reference = cast(dict[str, object], direction[field])
            reference_type = cast(str, reference["type"])
            reference_id = cast(str, reference["id"])
            if reference_id not in collections[reference_type]:
                issues.append(
                    _missing_reference_issue(
                        collection="performance_setup/performance_directions",
                        owner_id=direction_id,
                        field=f"{field}/id",
                        target_type=reference_type,
                        target_id=reference_id,
                    )
                )
    return issues


def _check_material_variation_cycles(
    variations: dict[str, object], materials: dict[str, object]
) -> list[ValidationIssue]:
    outgoing: dict[str, list[tuple[str, str]]] = {}
    for variation_id, raw_variation in variations.items():
        variation = cast(dict[str, object], raw_variation)
        source = cast(dict[str, object], variation["source"])
        target = cast(dict[str, object], variation["target"])
        source_id = cast(str, source["id"])
        target_id = cast(str, target["id"])
        if (
            source["type"] == "material"
            and target["type"] == "material"
            and source_id in materials
            and target_id in materials
        ):
            outgoing.setdefault(source_id, []).append((target_id, variation_id))

    issues: list[ValidationIssue] = []
    state: dict[str, str] = {}

    def visit(material_id: str) -> None:
        state[material_id] = "visiting"
        for target_id, variation_id in outgoing.get(material_id, []):
            if state.get(target_id) == "visiting":
                issues.append(
                    ValidationIssue(
                        code=IssueCode.SEMANTIC_INVALID,
                        message=f"Material variation cycle reaches: {target_id}",
                        path=(
                            f"/script/variations/{_escape_pointer_token(variation_id)}/target/id"
                        ),
                    )
                )
            elif state.get(target_id) is None:
                visit(target_id)
        state[material_id] = "done"

    for material_id in materials:
        if state.get(material_id) is None:
            visit(material_id)
    return issues


def _check_temporal_variation_order(script: dict[str, object]) -> list[ValidationIssue]:
    section_spans = _section_performance_spans(script)
    placement_spans = _placement_performance_spans(script, section_spans)
    spans_by_type = {
        "section": section_spans,
        "placement": placement_spans,
    }

    issues: list[ValidationIssue] = []
    variations = cast(dict[str, object], script["variations"])
    for variation_id, raw_variation in variations.items():
        variation = cast(dict[str, object], raw_variation)
        source = cast(dict[str, object], variation["source"])
        target = cast(dict[str, object], variation["target"])
        reference_type = cast(str, source["type"])
        if reference_type != target["type"] or reference_type not in spans_by_type:
            continue
        source_id = cast(str, source["id"])
        target_id = cast(str, target["id"])
        spans = spans_by_type[reference_type]
        if source_id not in spans or target_id not in spans:
            continue
        if spans[source_id][1] >= spans[target_id][0]:
            issues.append(
                ValidationIssue(
                    code=IssueCode.SEMANTIC_INVALID,
                    message=f"A {reference_type} variation source must precede its target",
                    path=(f"/script/variations/{_escape_pointer_token(variation_id)}/target/id"),
                )
            )
    return issues


def _section_performance_spans(script: dict[str, object]) -> dict[str, tuple[int, int]]:
    sections = cast(dict[str, object], script["sections"])
    children_by_parent: dict[str, list[str]] = {}
    for section_id, raw_section in sections.items():
        section = cast(dict[str, object], raw_section)
        parent_id = section["parent_section_id"]
        if isinstance(parent_id, str):
            children_by_parent.setdefault(parent_id, []).append(section_id)
    for child_ids in children_by_parent.values():
        child_ids.sort(
            key=lambda section_id: cast(dict[str, object], sections[section_id])["order"]
        )

    section_spans: dict[str, tuple[int, int]] = {}
    next_leaf_index = 0

    def record_span(section_id: str) -> tuple[int, int]:
        nonlocal next_leaf_index
        child_ids = children_by_parent.get(section_id, [])
        if not child_ids:
            span = (next_leaf_index, next_leaf_index)
            next_leaf_index += 1
        else:
            child_spans = [record_span(child_id) for child_id in child_ids]
            span = (child_spans[0][0], child_spans[-1][1])
        section_spans[section_id] = span
        return span

    record_span(cast(str, script["root_section_id"]))
    return section_spans


def _placement_performance_spans(
    script: dict[str, object], section_spans: dict[str, tuple[int, int]]
) -> dict[str, tuple[int, int]]:
    placement_spans: dict[str, tuple[int, int]] = {}
    placements = cast(dict[str, object], script["placements"])
    for placement_id, raw_placement in placements.items():
        placement = cast(dict[str, object], raw_placement)
        section_id = cast(str, placement["section_id"])
        if section_id in section_spans:
            placement_spans[placement_id] = section_spans[section_id]
    return placement_spans


def _check_transition_order(script: dict[str, object]) -> list[ValidationIssue]:
    placement_spans = _placement_performance_spans(script, _section_performance_spans(script))
    issues: list[ValidationIssue] = []
    transitions = cast(dict[str, object], script["transitions"])
    for transition_id, raw_transition in transitions.items():
        transition = cast(dict[str, object], raw_transition)
        from_id = cast(str, transition["from_placement_id"])
        connector_id = cast(str, transition["connector_placement_id"])
        to_id = cast(str, transition["to_placement_id"])
        if any(
            placement_id not in placement_spans for placement_id in (from_id, connector_id, to_id)
        ):
            continue
        from_span = placement_spans[from_id]
        connector_span = placement_spans[connector_id]
        to_span = placement_spans[to_id]
        if from_span[1] >= connector_span[0] or connector_span[1] >= to_span[0]:
            issues.append(
                ValidationIssue(
                    code=IssueCode.SEMANTIC_INVALID,
                    message="Transition order must be from, connector, then to",
                    path=(
                        f"/script/transitions/{_escape_pointer_token(transition_id)}"
                        "/connector_placement_id"
                    ),
                )
            )
    return issues


def _missing_reference_issue(
    *, collection: str, owner_id: str, field: str, target_type: str, target_id: str
) -> ValidationIssue:
    return ValidationIssue(
        code=IssueCode.SEMANTIC_INVALID,
        message=f"Referenced {target_type} does not exist: {target_id}",
        path=(f"/script/{collection}/{_escape_pointer_token(owner_id)}/{field}"),
    )


def _check_section_tree(sections: dict[str, object], root_section_id: str) -> list[ValidationIssue]:
    parent_by_id = {
        section_id: cast(dict[str, object], raw_section)["parent_section_id"]
        for section_id, raw_section in sections.items()
    }
    issues: list[ValidationIssue] = []
    if root_section_id not in sections:
        issues.append(
            ValidationIssue(
                code=IssueCode.SEMANTIC_INVALID,
                message=f"Root section does not exist: {root_section_id}",
                path="/script/root_section_id",
            )
        )
    for section_id, parent_id in parent_by_id.items():
        if section_id == root_section_id and parent_id is not None:
            issues.append(
                ValidationIssue(
                    code=IssueCode.SEMANTIC_INVALID,
                    message="The designated root section must not have a parent",
                    path=(
                        f"/script/sections/{_escape_pointer_token(section_id)}/parent_section_id"
                    ),
                )
            )
        elif section_id != root_section_id and parent_id is None:
            issues.append(
                ValidationIssue(
                    code=IssueCode.SEMANTIC_INVALID,
                    message="Only the designated root section may have no parent",
                    path=(
                        f"/script/sections/{_escape_pointer_token(section_id)}/parent_section_id"
                    ),
                )
            )
        if parent_id is not None and parent_id not in sections:
            issues.append(
                ValidationIssue(
                    code=IssueCode.SEMANTIC_INVALID,
                    message=f"Referenced parent section does not exist: {parent_id}",
                    path=f"/script/sections/{_escape_pointer_token(section_id)}/parent_section_id",
                )
            )

    children_by_parent: dict[str, list[tuple[int, str]]] = {}
    for section_id, parent_id in parent_by_id.items():
        if isinstance(parent_id, str) and parent_id in sections:
            section = cast(dict[str, object], sections[section_id])
            children_by_parent.setdefault(parent_id, []).append(
                (cast(int, section["order"]), section_id)
            )
    for children in children_by_parent.values():
        for expected_order, (actual_order, section_id) in enumerate(sorted(children)):
            if actual_order != expected_order:
                issues.append(
                    ValidationIssue(
                        code=IssueCode.SEMANTIC_INVALID,
                        message=(
                            f"Sibling order must be consecutive from zero; "
                            f"expected {expected_order}, got {actual_order}"
                        ),
                        path=f"/script/sections/{_escape_pointer_token(section_id)}/order",
                    )
                )
    for section_id, raw_section in sections.items():
        section = cast(dict[str, object], raw_section)
        has_children = bool(children_by_parent.get(section_id))
        has_relative_length = "relative_length" in section
        if has_children and has_relative_length:
            issues.append(
                ValidationIssue(
                    code=IssueCode.SEMANTIC_INVALID,
                    message="A branch section must derive its length from its children",
                    path=(f"/script/sections/{_escape_pointer_token(section_id)}/relative_length"),
                )
            )
        elif not has_children and not has_relative_length:
            issues.append(
                ValidationIssue(
                    code=IssueCode.SEMANTIC_INVALID,
                    message="A leaf section must have relative_length",
                    path=(f"/script/sections/{_escape_pointer_token(section_id)}/relative_length"),
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
                        code=IssueCode.SEMANTIC_INVALID,
                        message=f"Section containment cycle reaches: {parent_id}",
                        path=(
                            f"/script/sections/{_escape_pointer_token(section_id)}"
                            "/parent_section_id"
                        ),
                    )
                )
            elif state.get(parent_id) is None:
                visit(parent_id)
        state[section_id] = "done"

    for section_id in sections:
        if state.get(section_id) is None:
            visit(section_id)
    return issues
