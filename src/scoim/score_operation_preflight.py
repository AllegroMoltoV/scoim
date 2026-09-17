"""Plan phase-4 and phase-5 relation work before model calls begin."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from .phase3_model_contracts import ordered_leaf_section_ids
from .profile_capabilities import GenerationProfileCapabilities
from .validation import IssueCode, ValidationIssue


@dataclass(frozen=True, slots=True)
class PlannedComparisonSource:
    """One already-generated placement needed by a later score operation."""

    kind: str
    source_material_placement_id: str
    relation_id: str | None = None
    description: str | None = None
    preserve: tuple[str, ...] = ()
    change: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PlannedLayerOperation:
    """One phase-4 or phase-5 layer operation in dependency order."""

    material_placement_id: str
    transition_id: str | None
    comparison_sources: tuple[PlannedComparisonSource, ...]


@dataclass(frozen=True, slots=True)
class ScoreOperationPlan:
    """Relation-planning result shared by phase 2, phase 4, and phase 5."""

    foreground_operations: tuple[PlannedLayerOperation, ...]
    accompaniment_operations: tuple[PlannedLayerOperation, ...]
    issues: tuple[ValidationIssue, ...]


def build_score_operation_plan(
    document: Mapping[str, object],
    capabilities: GenerationProfileCapabilities,
) -> ScoreOperationPlan:
    """List every relation that the selected score operations cannot realize."""
    relation_capabilities = capabilities.score_relations
    script = cast(Mapping[str, object], document["script"])
    placements = cast(Mapping[str, Mapping[str, object]], script["material_placements"])
    accompaniment_placements = {
        placement_id
        for placement_id, placement in placements.items()
        if placement["role"] == "accompaniment"
    }
    unsupported_explicit_placements = {
        placement_id
        for placement_id, placement in placements.items()
        if placement["role"] not in relation_capabilities.explicit_variation_roles
    }
    unsupported_explicit_materials = {
        cast(str, placements[placement_id]["material_id"])
        for placement_id in unsupported_explicit_placements
    }
    foreground_placements = {
        placement_id
        for placement_id, placement in placements.items()
        if placement["role"] == "foreground"
    }
    transitions = cast(Mapping[str, Mapping[str, object]], script["material_placement_transitions"])
    transition_placements = {
        cast(str, transition["transition_material_placement_id"])
        for transition in transitions.values()
    }
    normal_foreground_placements = foreground_placements - transition_placements
    normal_foreground_materials = {
        cast(str, placements[placement_id]["material_id"])
        for placement_id in normal_foreground_placements
    }
    relations = cast(
        Mapping[str, Mapping[str, object]], script["script_element_variation_relations"]
    )
    leaf_order = {
        section_id: index for index, section_id in enumerate(ordered_leaf_section_ids(document))
    }
    foreground_order = sorted(
        normal_foreground_placements,
        key=lambda placement_id: (
            leaf_order[cast(str, placements[placement_id]["section_id"])],
            placement_id,
        ),
    )
    accompaniment_order = sorted(
        accompaniment_placements,
        key=lambda placement_id: (
            leaf_order[cast(str, placements[placement_id]["section_id"])],
            placement_id,
        ),
    )
    foreground_sources: dict[str, list[PlannedComparisonSource]] = {
        placement_id: [] for placement_id in foreground_placements
    }
    explicit_foreground_targets: set[str] = set()
    first_foreground_by_material: dict[str, str] = {}
    for placement_id in foreground_order:
        material_id = cast(str, placements[placement_id]["material_id"])
        first_foreground_by_material.setdefault(material_id, placement_id)

    issues: list[ValidationIssue] = []
    for relation_id, relation in sorted(relations.items()):
        source = cast(Mapping[str, object], relation["source"])
        target = cast(Mapping[str, object], relation["target"])
        source_type = cast(str, source["type"])
        related_ids = {cast(str, source["id"]), cast(str, target["id"])}
        if (
            source_type == "material_placement" and related_ids & unsupported_explicit_placements
        ) or (source_type == "material" and related_ids & unsupported_explicit_materials):
            issues.append(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    "Explicit variation is unsupported by the generation profile: "
                    f"source={_relation_endpoint_description(source, placements)}, "
                    f"target={_relation_endpoint_description(target, placements)}",
                    f"/script/script_element_variation_relations/{relation_id}",
                )
            )
            continue
        if (
            source_type == "material_placement"
            and target["id"] in foreground_placements
            and source["id"] not in normal_foreground_placements
        ):
            issues.append(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    "A foreground variation source must be a normal foreground placement: "
                    f"source={_relation_endpoint_description(source, placements)}, "
                    f"target={_relation_endpoint_description(target, placements)}",
                    f"/script/script_element_variation_relations/{relation_id}/source/id",
                )
            )
            continue
        elif (
            source_type == "material"
            and target["id"] in normal_foreground_materials
            and source["id"] not in normal_foreground_materials
        ):
            issues.append(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    "A material variation source must have a normal foreground placement: "
                    f"source={_relation_endpoint_description(source, placements)}, "
                    f"target={_relation_endpoint_description(target, placements)}",
                    f"/script/script_element_variation_relations/{relation_id}/source/id",
                )
            )
            continue
        if source_type == "material_placement" and target["id"] in foreground_placements:
            target_id = cast(str, target["id"])
            explicit_foreground_targets.add(target_id)
            foreground_sources[target_id].append(
                _relation_source(
                    "material_placement_variation",
                    cast(str, source["id"]),
                    relation_id,
                    relation,
                )
            )
        elif source_type == "material":
            source_placement_id = first_foreground_by_material.get(cast(str, source["id"]))
            target_placement_ids = [
                placement_id
                for placement_id in foreground_order
                if placements[placement_id]["material_id"] == target["id"]
            ]
            if source_placement_id is not None:
                for target_id in target_placement_ids:
                    foreground_sources[target_id].append(
                        _relation_source(
                            "material_variation",
                            source_placement_id,
                            relation_id,
                            relation,
                        )
                    )

    previous_foreground_by_material: dict[str, str] = {}
    for placement_id in foreground_order:
        material_id = cast(str, placements[placement_id]["material_id"])
        previous_id = previous_foreground_by_material.get(material_id)
        if previous_id is not None and placement_id not in explicit_foreground_targets:
            foreground_sources[placement_id].append(
                PlannedComparisonSource("implicit_reuse", previous_id)
            )
        previous_foreground_by_material[material_id] = placement_id

    ordered_foreground = _topological_order(foreground_order, foreground_sources)
    foreground_operations = [
        PlannedLayerOperation(
            placement_id,
            None,
            tuple(foreground_sources[placement_id]),
        )
        for placement_id in ordered_foreground
    ]
    ordered_transitions = sorted(
        transitions.items(),
        key=lambda item: (
            leaf_order[
                cast(str, placements[item[1]["transition_material_placement_id"]]["section_id"])
            ],
            item[0],
        ),
    )
    for transition_id, transition in ordered_transitions:
        source_id = cast(str, transition["source_material_placement_id"])
        transition_placement_id = cast(str, transition["transition_material_placement_id"])
        target_id = cast(str, transition["target_material_placement_id"])
        related_ids = {source_id, transition_placement_id, target_id}
        if any(
            placements[placement_id]["role"] not in relation_capabilities.transition_roles
            for placement_id in related_ids
        ):
            issues.append(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    "Transition is unsupported by the generation profile: "
                    f"source={_placement_description(source_id, placements)}, "
                    f"connector={_placement_description(transition_placement_id, placements)}, "
                    f"target={_placement_description(target_id, placements)}",
                    f"/script/material_placement_transitions/{transition_id}",
                )
            )
            continue
        if (
            source_id not in normal_foreground_placements
            or target_id not in normal_foreground_placements
        ):
            issues.append(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    "A foreground transition requires normal foreground boundaries",
                    f"/script/material_placement_transitions/{transition_id}",
                )
            )
            continue
        foreground_operations.append(
            PlannedLayerOperation(
                transition_placement_id,
                transition_id,
                (
                    PlannedComparisonSource("transition_source", source_id, transition_id),
                    PlannedComparisonSource("transition_target", target_id, transition_id),
                ),
            )
        )

    accompaniment_operations: list[PlannedLayerOperation] = []
    previous_accompaniment_by_material: dict[str, str] = {}
    for placement_id in accompaniment_order:
        material_id = cast(str, placements[placement_id]["material_id"])
        previous_id = previous_accompaniment_by_material.get(material_id)
        sources = (
            (PlannedComparisonSource("implicit_reuse", previous_id),)
            if previous_id is not None
            else ()
        )
        accompaniment_operations.append(PlannedLayerOperation(placement_id, None, sources))
        previous_accompaniment_by_material[material_id] = placement_id
    return ScoreOperationPlan(
        tuple(foreground_operations),
        tuple(accompaniment_operations),
        tuple(issues),
    )


def _relation_source(
    kind: str,
    source_id: str,
    relation_id: str,
    relation: Mapping[str, object],
) -> PlannedComparisonSource:
    return PlannedComparisonSource(
        kind=kind,
        source_material_placement_id=source_id,
        relation_id=relation_id,
        description=cast(str, relation["description"]),
        preserve=tuple(cast(list[str], relation["preserve"])),
        change=tuple(cast(list[str], relation["change"])),
    )


def _relation_endpoint_description(
    endpoint: Mapping[str, object],
    placements: Mapping[str, Mapping[str, object]],
) -> str:
    endpoint_type = cast(str, endpoint["type"])
    endpoint_id = cast(str, endpoint["id"])
    if endpoint_type == "material_placement":
        role = cast(str, placements[endpoint_id]["role"])
        return f"{endpoint_type}:{endpoint_id} (role={role})"
    roles = sorted(
        {
            cast(str, placement["role"])
            for placement in placements.values()
            if placement["material_id"] == endpoint_id
        }
    )
    return f"{endpoint_type}:{endpoint_id} (roles={','.join(roles) or 'none'})"


def _placement_description(
    placement_id: str,
    placements: Mapping[str, Mapping[str, object]],
) -> str:
    return f"{placement_id} (role={placements[placement_id]['role']})"


def _topological_order(
    stable_order: list[str],
    sources_by_target: Mapping[str, list[PlannedComparisonSource]],
) -> tuple[str, ...]:
    pending = set(stable_order)
    completed: set[str] = set()
    result: list[str] = []
    while pending:
        ready = [
            placement_id
            for placement_id in stable_order
            if placement_id in pending
            and all(
                source.source_material_placement_id in completed
                for source in sources_by_target[placement_id]
            )
        ]
        if not ready:
            return tuple(result)
        placement_id = ready[0]
        pending.remove(placement_id)
        completed.add(placement_id)
        result.append(placement_id)
    return tuple(result)
