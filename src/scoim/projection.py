"""Projection of approved SCoIM scripts into the solo-piano profile."""

import copy
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from functools import reduce
from typing import cast

import jsonpointer

from llm_musical_composer.performance_pipeline import PiecePlan, PlanNode, validate_piece_plan

from .requirements import resolve_requirements
from .script_validation import check_script_document
from .validation import CheckResult, IssueCode, ValidationIssue, check

SUPPORTED_PLAN_ROLES = frozenset(
    {
        "whole",
        "opening",
        "statement",
        "variation",
        "contrast",
        "transition",
        "climax",
        "return",
        "release",
    }
)


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    """Result of projecting one approved script into a profile."""

    projected: bool
    piece_plan: PiecePlan | None
    targets: tuple["ProjectionTarget", ...]
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True, slots=True)
class ProjectionTarget:
    """One traceable script field and its required downstream verification."""

    target_id: str
    source_path: str
    generation_stages: tuple[str, ...]
    verification: str
    value: object


@dataclass(frozen=True, slots=True)
class PlanChoice:
    """Creative plan values supplied by a frozen lower-stage response."""

    tonal_center: int
    mode: str
    harmonic_focus_by_section: Mapping[str, int]
    contrasts_with_by_section: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StructurePreparation:
    """A key-free structural projection and its complete profile diagnostics."""

    prepared: bool
    nodes: tuple[PlanNode, ...]
    targets: tuple[ProjectionTarget, ...]
    issues: tuple[ValidationIssue, ...]
    checks_complete: bool


def _failure(code: IssueCode, message: str, path: str) -> ProjectionResult:
    return ProjectionResult(
        projected=False,
        piece_plan=None,
        targets=(),
        issues=(ValidationIssue(code=code, message=message, path=path),),
    )


def _target(
    path: str,
    value: object,
    stages: tuple[str, ...],
    verification: str,
) -> ProjectionTarget:
    return ProjectionTarget(path, path, stages, verification, copy.deepcopy(value))


def _projection_targets(script: dict[str, object]) -> tuple[ProjectionTarget, ...]:
    targets = [
        _target("/script/title", script["title"], ("piece_plan",), "direct_equality"),
        _target(
            "/script/brief",
            script["brief"],
            ("piece_plan", "score_spec", "performance_spec"),
            "translated_mechanical_claim",
        ),
        _target(
            "/script/root_section_id",
            script["root_section_id"],
            ("piece_plan",),
            "direct_equality",
        ),
    ]
    setup = cast(dict[str, object], script["performance_setup"])
    targets.extend(
        (
            _target(
                "/script/performance_setup/instrumentation",
                setup["instrumentation"],
                ("profile",),
                "profile_compatibility",
            ),
            _target(
                "/script/performance_setup/target_duration_seconds",
                setup["target_duration_seconds"],
                ("performance_spec", "rendered_performance"),
                "duration_equality",
            ),
        )
    )
    collections = (
        (
            "sections",
            ("piece_plan",),
            {
                "description": "translated_mechanical_claim",
                "parent_section_id": "direct_equality",
                "order": "direct_equality",
                "role": "direct_equality",
                "relative_length": "ratio_equality",
            },
        ),
        (
            "materials",
            ("score_spec",),
            {
                "kind": "translated_mechanical_claim",
                "description": "translated_mechanical_claim",
            },
        ),
        (
            "placements",
            ("piece_plan", "score_spec"),
            {
                "section_id": "direct_equality",
                "material_id": "direct_equality",
            },
        ),
        (
            "variations",
            ("piece_plan", "score_spec", "performance_spec"),
            {
                "source": "relation_equality",
                "target": "relation_equality",
                "preserve": "translated_mechanical_claim",
                "change": "translated_mechanical_claim",
                "description": "translated_mechanical_claim",
            },
        ),
        (
            "transitions",
            ("piece_plan", "score_spec"),
            {
                "from_placement_id": "relation_equality",
                "connector_placement_id": "relation_equality",
                "to_placement_id": "relation_equality",
                "description": "translated_mechanical_claim",
            },
        ),
    )
    for collection_name, stages, fields in collections:
        collection = cast(dict[str, object], script[collection_name])
        for item_id in sorted(collection):
            escaped_id = jsonpointer.escape(item_id)
            item_path = f"/script/{collection_name}/{escaped_id}"
            targets.append(_target(item_path, item_id, stages, "id_presence"))
            item = cast(dict[str, object], collection[item_id])
            for field_name, verification in fields.items():
                if field_name in item:
                    field_stages = (
                        ("piece_plan", "score_spec", "performance_spec")
                        if collection_name == "sections" and field_name == "description"
                        else stages
                    )
                    targets.append(
                        _target(
                            f"{item_path}/{field_name}",
                            item[field_name],
                            field_stages,
                            verification,
                        )
                    )
    directions = cast(dict[str, object], setup["performance_directions"])
    for direction_id in sorted(directions):
        escaped_id = jsonpointer.escape(direction_id)
        item_path = f"/script/performance_setup/performance_directions/{escaped_id}"
        targets.append(_target(item_path, direction_id, ("performance_spec",), "id_presence"))
        direction = cast(dict[str, object], directions[direction_id])
        for field_name in ("target", "relative_to", "description"):
            if field_name in direction:
                verification = (
                    "relation_equality"
                    if field_name != "description"
                    else "targeted_performance_difference"
                )
                targets.append(
                    _target(
                        f"{item_path}/{field_name}",
                        direction[field_name],
                        ("performance_spec",),
                        verification,
                    )
                )
    for requirement in resolve_requirements(script):
        targets.append(
            _target(
                requirement.source_path,
                requirement.value(),
                ("rendered_performance",),
                "typed_requirement",
            )
        )
    return tuple(targets)


def _validated_script(
    document: Mapping[str, object],
) -> tuple[dict[str, object] | None, CheckResult]:
    is_new = document.get("document_type") == "script"
    validation = check_script_document(document) if is_new else check(document)
    if not validation.valid:
        return None, validation
    return cast(dict[str, object], document["script"]), validation


def _prepare_nodes(script: dict[str, object]) -> StructurePreparation:
    sections = cast(dict[str, dict[str, object]], script["sections"])
    placements = cast(dict[str, dict[str, object]], script["placements"])
    issues: list[ValidationIssue] = []
    complete = True
    for section_id, section in sections.items():
        if section["role"] not in SUPPORTED_PLAN_ROLES:
            issues.append(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    "The section role is outside the solo_piano_3m_v1 vocabulary",
                    f"/script/sections/{jsonpointer.escape(section_id)}/role",
                )
            )

    children: dict[str, list[str]] = defaultdict(list)
    for section_id, section in sections.items():
        parent_id = section["parent_section_id"]
        if isinstance(parent_id, str):
            children[parent_id].append(section_id)
    for child_ids in children.values():
        child_ids.sort(key=lambda item: cast(int, sections[item]["order"]))
    by_section: dict[str, list[dict[str, object]]] = defaultdict(list)
    for placement in placements.values():
        by_section[cast(str, placement["section_id"])].append(placement)

    derived: dict[str, str] = {}
    derivation_paths: dict[str, str] = {}
    for variation_id, variation in cast(dict[str, dict[str, object]], script["variations"]).items():
        source = cast(dict[str, object], variation["source"])
        target = cast(dict[str, object], variation["target"])
        if source["type"] == "section":
            source_id, target_id = cast(str, source["id"]), cast(str, target["id"])
        elif source["type"] == "placement":
            source_id = cast(str, placements[cast(str, source["id"])]["section_id"])
            target_id = cast(str, placements[cast(str, target["id"])]["section_id"])
        else:
            continue
        path = f"/script/variations/{jsonpointer.escape(variation_id)}/target"
        if target_id in derived and derived[target_id] != source_id:
            issues.append(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    "Multiple variations require conflicting PiecePlan derivations",
                    path,
                )
            )
        else:
            derived[target_id] = source_id
            derivation_paths[target_id] = path
    for section_id, section in sections.items():
        if section["role"] == "return" and section_id not in derived:
            issues.append(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    "A return section requires a section or placement variation source",
                    f"/script/sections/{jsonpointer.escape(section_id)}/role",
                )
            )

    ordered: list[str] = []

    def visit(section_id: str) -> None:
        ordered.append(section_id)
        for child_id in children.get(section_id, []):
            visit(child_id)

    visit(cast(str, script["root_section_id"]))
    depths: dict[str, int] = {}
    for section_id in ordered:
        parent_id = sections[section_id]["parent_section_id"]
        depths[section_id] = 0 if parent_id is None else depths[cast(str, parent_id)] + 1
    for target_id, source_id in derived.items():
        if depths[target_id] != depths[source_id]:
            issues.append(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    "PiecePlan derivations must connect sections at the same depth",
                    derivation_paths[target_id],
                )
            )

    leaves = [section_id for section_id in ordered if not children.get(section_id)]
    bad_leaves = [section_id for section_id in leaves if len(by_section[section_id]) != 1]
    for section_id in bad_leaves:
        issues.append(
            ValidationIssue(
                IssueCode.UNREPRESENTABLE,
                "solo_piano_3m_v1 requires exactly one placement in each leaf section",
                f"/script/sections/{jsonpointer.escape(section_id)}",
            )
        )
    if bad_leaves:
        return StructurePreparation(False, (), _projection_targets(script), tuple(issues), False)

    def subtree_materials(section_id: str) -> frozenset[str]:
        if not children.get(section_id):
            return frozenset({cast(str, by_section[section_id][0]["material_id"])})
        return frozenset(
            material_id
            for child_id in children[section_id]
            for material_id in subtree_materials(child_id)
        )

    for target_id, source_id in derived.items():
        if not (subtree_materials(target_id) & subtree_materials(source_id)):
            issues.append(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    "A section variation between different materials needs "
                    "unsupported identity cues",
                    derivation_paths[target_id],
                )
            )

    lengths = {sid: Fraction(str(sections[sid]["relative_length"])) for sid in leaves}
    denominator = math.lcm(*(value.denominator for value in lengths.values()))
    unscaled = {
        sid: value.numerator * (denominator // value.denominator) for sid, value in lengths.items()
    }
    divisor = reduce(math.gcd, unscaled.values())
    weights = {sid: value // divisor for sid, value in unscaled.items()}
    by_material: dict[str, int] = {}
    for section_id in leaves:
        material_id = cast(str, by_section[section_id][0]["material_id"])
        if material_id in by_material and by_material[material_id] != weights[section_id]:
            issues.append(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    "Reused material cannot have different score-length ratios in the current IR",
                    f"/script/sections/{jsonpointer.escape(section_id)}/relative_length",
                )
            )
        else:
            by_material[material_id] = weights[section_id]
    nodes = tuple(
        PlanNode(
            node_id=sid,
            parent_id=cast(str | None, sections[sid]["parent_section_id"]),
            order=cast(int, sections[sid]["order"]),
            role=cast(str, sections[sid]["role"]),
            derived_from=derived.get(sid),
            contrasts_with=None,
            harmonic_focus=None,
            duration_weight=weights[sid] if sid in leaves else None,
            score_material_id=(
                cast(str, by_section[sid][0]["material_id"]) if sid in leaves else None
            ),
        )
        for sid in ordered
    )
    return StructurePreparation(
        not issues, nodes, _projection_targets(script), tuple(issues), complete
    )


def prepare_solo_piano_3m_structure(document: Mapping[str, object]) -> StructurePreparation:
    """Prepare and exhaustively check the key-free solo-piano structure."""
    script, validation = _validated_script(document)
    if script is None:
        return StructurePreparation(False, (), (), validation.issues, False)
    base = _prepare_nodes(script)
    issues = list(base.issues)
    setup = cast(dict[str, object], script["performance_setup"])
    if setup["instrumentation"] != "solo_piano":
        issues.append(
            ValidationIssue(
                IssueCode.UNSUPPORTED_PROFILE,
                "solo_piano_3m_v1 only supports solo_piano",
                "/script/performance_setup/instrumentation",
            )
        )
    if setup["target_duration_seconds"] != 180:
        issues.append(
            ValidationIssue(
                IssueCode.UNSUPPORTED_PROFILE,
                "solo_piano_3m_v1 only supports a 180-second target",
                "/script/performance_setup/target_duration_seconds",
            )
        )
    materials = cast(dict[str, dict[str, object]], script["materials"])
    placements = cast(dict[str, dict[str, object]], script["placements"])
    for material_id, material in materials.items():
        if material["kind"] not in {"theme", "contrast", "transition", "ending"}:
            issues.append(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    "The material kind is outside the solo_piano_3m_v1 vocabulary",
                    f"/script/materials/{jsonpointer.escape(material_id)}/kind",
                )
            )
    placed = {cast(str, item["material_id"]) for item in placements.values()}
    for material_id in sorted(set(materials) - placed):
        issues.append(
            ValidationIssue(
                IssueCode.UNREPRESENTABLE,
                "Every material must be placed by the solo-piano profile",
                f"/script/materials/{jsonpointer.escape(material_id)}",
            )
        )
    leaves = tuple(node for node in base.nodes if node.score_material_id is not None)
    if base.checks_complete:
        if len(leaves) < 2:
            issues.append(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    "The solo-piano profile requires music before a final release",
                    "/script/sections",
                )
            )
        elif leaves:
            final = leaves[-1]
            if final.role != "release":
                issues.append(
                    ValidationIssue(
                        IssueCode.UNREPRESENTABLE,
                        "The final leaf must have the release role",
                        f"/script/sections/{jsonpointer.escape(final.node_id)}/role",
                    )
                )
            ending_ids = {
                mid for mid, material in materials.items() if material["kind"] == "ending"
            }
            if ending_ids != {final.score_material_id}:
                issues.append(
                    ValidationIssue(
                        IssueCode.UNREPRESENTABLE,
                        "The final release must place the profile's only ending material",
                        "/script/materials",
                    )
                )
            ordinary = next(
                (
                    node
                    for node in reversed(leaves[:-1])
                    if materials[cast(str, node.score_material_id)]["kind"]
                    not in {"transition", "ending"}
                ),
                None,
            )
            if ordinary is None:
                issues.append(
                    ValidationIssue(
                        IssueCode.UNREPRESENTABLE,
                        "The final release requires preceding ordinary material",
                        f"/script/sections/{jsonpointer.escape(final.node_id)}",
                    )
                )
            assert final.duration_weight is not None
            total = sum(
                cast(int, node.duration_weight)
                for node in leaves
                if node.duration_weight is not None
            )
            duration_ms = float(cast(int | float, setup["target_duration_seconds"])) * 1_000
            if duration_ms * final.duration_weight < 2_000 * total:
                issues.append(
                    ValidationIssue(
                        IssueCode.UNREPRESENTABLE,
                        "The final release is too short for a 2,000 ms nominal ending",
                        f"/script/sections/{jsonpointer.escape(final.node_id)}/relative_length",
                    )
                )
    connectors = {
        cast(str, transition["connector_placement_id"])
        for transition in cast(dict[str, dict[str, object]], script["transitions"]).values()
    }
    sections = cast(dict[str, dict[str, object]], script["sections"])
    for placement_id, placement in placements.items():
        material_id = cast(str, placement["material_id"])
        section_id = cast(str, placement["section_id"])
        if (materials[material_id]["kind"] == "transition") != (placement_id in connectors) or (
            placement_id in connectors
        ) != (sections[section_id]["role"] == "transition"):
            issues.append(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    "Transition sections, connector placements, and transition materials "
                    "must match",
                    f"/script/placements/{jsonpointer.escape(placement_id)}",
                )
            )
    return StructurePreparation(
        not issues and base.checks_complete,
        base.nodes,
        base.targets,
        tuple(issues),
        base.checks_complete,
    )


def project_solo_piano_3m(
    document: Mapping[str, object], *, plan_choice: PlanChoice | None = None
) -> ProjectionResult:
    """Complete a prepared solo-piano structure with explicit creative choices."""
    is_new = document.get("document_type") == "script"
    script, validation = _validated_script(document)
    if script is None:
        return ProjectionResult(False, None, (), validation.issues)
    required_status = "validated" if is_new else "approved"
    if document["status"] != required_status:
        return _failure(
            IssueCode.SEMANTIC_INVALID,
            f"Only a {required_status} script can be projected",
            "/status",
        )
    setup = cast(dict[str, object], script["performance_setup"])
    if setup["instrumentation"] != "solo_piano":
        return _failure(
            IssueCode.UNSUPPORTED_PROFILE,
            "solo_piano_3m_v1 only supports solo_piano",
            "/script/performance_setup/instrumentation",
        )
    if setup["target_duration_seconds"] != 180:
        return _failure(
            IssueCode.UNSUPPORTED_PROFILE,
            "solo_piano_3m_v1 only supports a 180-second target",
            "/script/performance_setup/target_duration_seconds",
        )
    if plan_choice is None:
        return _failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "A frozen plan choice is required to complete the PiecePlan",
            "/plan_choice",
        )
    if plan_choice.mode not in {"major", "minor"}:
        return _failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "The plan choice mode must be major or minor",
            "/plan_choice/mode",
        )
    if (
        isinstance(plan_choice.tonal_center, bool)
        or not isinstance(plan_choice.tonal_center, int)
        or not 0 <= plan_choice.tonal_center <= 11
    ):
        return _failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "The plan choice tonal center must be a pitch class from 0 to 11",
            "/plan_choice/tonal_center",
        )
    prepared = _prepare_nodes(script)
    if not prepared.prepared:
        return ProjectionResult(False, None, prepared.targets, prepared.issues)
    sections = cast(dict[str, dict[str, object]], script["sections"])
    for section_id in plan_choice.harmonic_focus_by_section:
        if section_id not in sections:
            return _failure(
                IssueCode.MODEL_OUTPUT_INVALID,
                "The plan choice refers to an unknown section",
                f"/plan_choice/harmonic_focus_by_section/{jsonpointer.escape(section_id)}",
            )
    for section_id in plan_choice.contrasts_with_by_section:
        if section_id not in sections:
            return _failure(
                IssueCode.MODEL_OUTPUT_INVALID,
                "The plan choice refers to an unknown contrast section",
                f"/plan_choice/contrasts_with_by_section/{jsonpointer.escape(section_id)}",
            )
    positions = {node.node_id: index for index, node in enumerate(prepared.nodes)}
    depths: dict[str, int] = {}
    for node in prepared.nodes:
        depths[node.node_id] = 0 if node.parent_id is None else depths[node.parent_id] + 1
    for section_id, source_id in plan_choice.contrasts_with_by_section.items():
        if (
            source_id not in sections
            or sections[section_id]["role"] != "contrast"
            or positions[source_id] >= positions[section_id]
            or depths[source_id] != depths[section_id]
        ):
            return _failure(
                IssueCode.MODEL_OUTPUT_INVALID,
                "A contrast source must be an earlier section at the same depth",
                f"/plan_choice/contrasts_with_by_section/{jsonpointer.escape(section_id)}",
            )
    for section_id, focus in plan_choice.harmonic_focus_by_section.items():
        if isinstance(focus, bool) or not isinstance(focus, int) or not 0 <= focus <= 11:
            return _failure(
                IssueCode.MODEL_OUTPUT_INVALID,
                "Each harmonic focus must be a pitch class from 0 to 11",
                f"/plan_choice/harmonic_focus_by_section/{jsonpointer.escape(section_id)}",
            )
    nodes = tuple(
        PlanNode(
            node_id=node.node_id,
            parent_id=node.parent_id,
            order=node.order,
            role=node.role,
            derived_from=node.derived_from,
            contrasts_with=plan_choice.contrasts_with_by_section.get(node.node_id),
            harmonic_focus=plan_choice.harmonic_focus_by_section.get(node.node_id),
            duration_weight=node.duration_weight,
            score_material_id=node.score_material_id,
        )
        for node in prepared.nodes
    )
    plan = PiecePlan(
        plan_id=cast(str, document["document_id"]),
        title=cast(str, script["title"]),
        tonal_center=plan_choice.tonal_center,
        mode=plan_choice.mode,
        root_node_id=cast(str, script["root_section_id"]),
        ending_intent="tonic",
        nodes=nodes,
    )
    validate_piece_plan(plan)
    return ProjectionResult(True, plan, prepared.targets, ())


def check_solo_piano_3m_structure(document: Mapping[str, object]) -> CheckResult:
    """Check script-only constraints required by ``solo_piano_3m_v1``."""
    prepared = prepare_solo_piano_3m_structure(document)
    return CheckResult(prepared.prepared, prepared.issues)
