"""Plan atomic score candidates from section ranges and musical dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

from .generation_script_validation import check_generation_script_document
from .profile_capabilities import (
    GenerationProfileCapabilities,
    check_generation_profile_document,
    solo_piano_3m_v2_capabilities,
)
from .validation import IssueCode, ValidationIssue


@dataclass(frozen=True, slots=True)
class ScoreComparison:
    kind: str
    relation_id: str | None
    source_placement_ids: tuple[str, ...]
    target_placement_ids: tuple[str, ...]
    description: str | None = None
    preserve: tuple[str, ...] = ()
    change: tuple[str, ...] = ()
    source_section_id: str | None = None
    target_section_id: str | None = None


@dataclass(frozen=True, slots=True)
class ScoreWorkOperation:
    operation_id: str
    material_placement_ids: tuple[str, ...]
    dependency_operation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScoreWorkPlan:
    operations: tuple[ScoreWorkOperation, ...]
    comparisons: tuple[ScoreComparison, ...]
    issues: tuple[ValidationIssue, ...]


def section_leaf_ids(document: Mapping[str, object], section_id: str) -> tuple[str, ...]:
    """Return a validated section's complete leaf range in performance order."""
    script = cast(Mapping[str, object], document["script"])
    sections = cast(Mapping[str, Mapping[str, object]], script["sections"])
    if section_id not in sections:
        raise ValueError(f"unknown section: {section_id}")
    children: dict[str, list[str]] = {}
    for key, section in sections.items():
        parent = section["parent_section_id"]
        if isinstance(parent, str):
            children.setdefault(parent, []).append(key)
    for values in children.values():
        values.sort(key=lambda key: (cast(int, sections[key]["order"]), key))
    result: list[str] = []
    pending = [section_id]
    while pending:
        current = pending.pop()
        if current in children:
            pending.extend(reversed(children[current]))
        else:
            result.append(current)
    return tuple(result)


def ordered_placement_ids(document: Mapping[str, object]) -> tuple[str, ...]:
    """Use leaf order, role, then identifier for reproducible positional inputs."""
    script = cast(Mapping[str, object], document["script"])
    placements = cast(Mapping[str, Mapping[str, object]], script["material_placements"])
    leaves = section_leaf_ids(document, cast(str, script["root_section_id"]))
    rank = {key: index for index, key in enumerate(leaves)}
    return tuple(
        sorted(
            placements,
            key=lambda key: (
                rank[cast(str, placements[key]["section_id"])],
                0 if placements[key]["role"] == "foreground" else 1,
                key,
            ),
        )
    )


def build_score_work_plan(
    document: Mapping[str, object],
    capabilities: GenerationProfileCapabilities | None = None,
) -> ScoreWorkPlan:
    """Group complete variation targets and condense all dependency cycles."""
    checked = check_generation_script_document(document)
    if not checked.valid:
        return ScoreWorkPlan((), (), checked.issues)
    capabilities = capabilities or solo_piano_3m_v2_capabilities()
    checked = check_generation_profile_document(document, capabilities)
    if not checked.valid:
        return ScoreWorkPlan((), (), checked.issues)
    script = cast(Mapping[str, object], document["script"])
    placements = cast(Mapping[str, Mapping[str, object]], script["material_placements"])
    relations = cast(
        Mapping[str, Mapping[str, object]], script["script_element_variation_relations"]
    )
    transitions = cast(Mapping[str, Mapping[str, object]], script["material_placement_transitions"])
    order = ordered_placement_ids(document)
    rank = {key: index for index, key in enumerate(order)}
    connectors = {
        cast(str, value["transition_material_placement_id"]) for value in transitions.values()
    }
    foregrounds = tuple(key for key in order if placements[key]["role"] == "foreground")
    normal_foregrounds = tuple(key for key in foregrounds if key not in connectors)
    limits = capabilities.score_relations
    unsupported = {
        key for key in order if placements[key]["role"] not in limits.explicit_variation_roles
    }
    unsupported_materials = {placements[key]["material_id"] for key in unsupported}
    first_by_material: dict[str, str] = {}
    for key in normal_foregrounds:
        first_by_material.setdefault(cast(str, placements[key]["material_id"]), key)
    comparisons: list[ScoreComparison] = []
    issues: list[ValidationIssue] = []
    explicit_placement_targets: set[str] = set()

    def placement_range(section_id: str) -> tuple[str, ...]:
        leaves = set(section_leaf_ids(document, section_id))
        return tuple(key for key in order if placements[key]["section_id"] in leaves)

    for relation_id, relation in sorted(relations.items()):
        source = cast(Mapping[str, str], relation["source"])
        target = cast(Mapping[str, str], relation["target"])
        kind = source["type"]
        endpoints = {source["id"], target["id"]}
        if (kind == "material_placement" and endpoints & unsupported) or (
            kind == "material" and endpoints & unsupported_materials
        ):
            issues.append(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    f"Explicit {kind} variation is unsupported by the generation profile: "
                    f"source={source['id']}, target={target['id']}",
                    f"/script/script_element_variation_relations/{relation_id}",
                )
            )
            continue
        source_section = target_section = None
        if kind == "section":
            source_section, target_section = source["id"], target["id"]
            source_ids, target_ids = (
                placement_range(source_section),
                placement_range(target_section),
            )
        elif kind == "material_placement":
            if source["id"] not in normal_foregrounds or target["id"] not in normal_foregrounds:
                issues.append(
                    ValidationIssue(
                        IssueCode.UNREPRESENTABLE,
                        "A placement variation requires normal foreground placements",
                        f"/script/script_element_variation_relations/{relation_id}",
                    )
                )
                continue
            source_ids, target_ids = (source["id"],), (target["id"],)
            explicit_placement_targets.add(target["id"])
        else:
            source_id = first_by_material.get(source["id"])
            target_ids = tuple(
                key for key in normal_foregrounds if placements[key]["material_id"] == target["id"]
            )
            if source_id is None or not target_ids:
                issues.append(
                    ValidationIssue(
                        IssueCode.UNREPRESENTABLE,
                        "A material variation requires normal foreground placements",
                        f"/script/script_element_variation_relations/{relation_id}",
                    )
                )
                continue
            source_ids = (source_id,)
        comparisons.append(
            ScoreComparison(
                kind=f"{kind}_variation",
                relation_id=relation_id,
                source_placement_ids=source_ids,
                target_placement_ids=target_ids,
                description=cast(str, relation["description"]),
                preserve=tuple(cast(list[str], relation["preserve"])),
                change=tuple(cast(list[str], relation["change"])),
                source_section_id=source_section,
                target_section_id=target_section,
            )
        )

    previous: dict[tuple[str, str], str] = {}
    for key in order:
        if key in connectors:
            continue
        material_role = (
            cast(str, placements[key]["material_id"]),
            cast(str, placements[key]["role"]),
        )
        earlier = previous.get(material_role)
        if earlier is not None and key not in explicit_placement_targets:
            comparisons.append(ScoreComparison("implicit_reuse", None, (earlier,), (key,)))
        previous[material_role] = key

    for relation_id, relation in sorted(transitions.items()):
        source_id = cast(str, relation["source_material_placement_id"])
        target_id = cast(str, relation["target_material_placement_id"])
        connector_id = cast(str, relation["transition_material_placement_id"])
        if any(
            placements[key]["role"] not in limits.transition_roles
            for key in (source_id, connector_id, target_id)
        ):
            issues.append(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    "Transition is unsupported by the generation profile",
                    f"/script/material_placement_transitions/{relation_id}",
                )
            )
            continue
        if source_id not in normal_foregrounds or target_id not in normal_foregrounds:
            issues.append(
                ValidationIssue(
                    IssueCode.UNREPRESENTABLE,
                    "A foreground transition requires normal foreground boundaries",
                    f"/script/material_placement_transitions/{relation_id}",
                )
            )
            continue
        for kind, source_key in (
            ("transition_source", source_id),
            ("transition_target", target_id),
        ):
            comparisons.append(
                ScoreComparison(
                    kind,
                    relation_id,
                    (source_key,),
                    (connector_id,),
                    cast(str, relation["description"]),
                )
            )
    if issues:
        return ScoreWorkPlan((), tuple(comparisons), tuple(issues))

    edges = {
        (source, target)
        for comparison in comparisons
        for source in comparison.source_placement_ids
        for target in comparison.target_placement_ids
    }
    by_unit: dict[str, list[str]] = {}
    for key in order:
        by_unit.setdefault(cast(str, placements[key]["section_id"]), []).append(key)
    for keys in by_unit.values():
        foreground = [key for key in keys if placements[key]["role"] == "foreground"]
        accompaniment = [key for key in keys if placements[key]["role"] == "accompaniment"]
        edges.update(pairwise(foreground))
        edges.update(pairwise(accompaniment))
        edges.update((source, target) for source in foreground for target in accompaniment)

    parent = {key: key for key in order}

    def root(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def join(keys: tuple[str, ...]) -> None:
        roots = {root(key) for key in keys}
        first = min(roots, key=rank.__getitem__)
        for key in roots:
            parent[key] = first

    for comparison in comparisons:
        if comparison.kind == "section_variation":
            join(comparison.target_placement_ids)

    graph = {root(key): set() for key in order}
    for source, target in edges:
        if root(source) != root(target):
            graph[root(source)].add(root(target))
    for component in _strong_components(graph, rank):
        join(component)

    groups: dict[str, tuple[str, ...]] = {}
    for key in order:
        group = root(key)
        groups[group] = (*groups.get(group, ()), key)
    dependencies: dict[str, set[str]] = {key: set() for key in groups}
    for source, target in edges:
        if root(source) != root(target):
            dependencies[root(target)].add(root(source))
    operation_ids = {key: f"score-{key}" for key in groups}
    completed: set[str] = set()
    operations: list[ScoreWorkOperation] = []
    while len(completed) < len(groups):
        ready = [key for key in groups if key not in completed and dependencies[key] <= completed]
        if not ready:
            raise ValueError("score dependency condensation failed to cover every placement")
        key = min(ready, key=rank.__getitem__)
        operations.append(
            ScoreWorkOperation(
                operation_ids[key],
                groups[key],
                tuple(
                    operation_ids[dependency]
                    for dependency in sorted(dependencies[key], key=rank.__getitem__)
                ),
            )
        )
        completed.add(key)
    if sorted(
        key for operation in operations for key in operation.material_placement_ids
    ) != sorted(order):
        raise ValueError("score operations must cover every placement exactly once")
    return ScoreWorkPlan(tuple(operations), tuple(comparisons), ())


def _strong_components(
    graph: Mapping[str, set[str]], rank: Mapping[str, int]
) -> tuple[tuple[str, ...], ...]:
    """Iterative Kosaraju traversal avoids a call-stack bound on group count."""
    reverse: dict[str, set[str]] = {key: set() for key in graph}
    for source, targets in graph.items():
        for target in targets:
            reverse[target].add(source)
    visited: set[str] = set()
    finished: list[str] = []
    for initial in sorted(graph, key=rank.__getitem__):
        stack = [(initial, False)]
        while stack:
            key, expanded = stack.pop()
            if expanded:
                finished.append(key)
            elif key not in visited:
                visited.add(key)
                stack.append((key, True))
                stack.extend(
                    (child, False)
                    for child in sorted(graph[key], key=rank.__getitem__, reverse=True)
                    if child not in visited
                )
    visited.clear()
    components: list[tuple[str, ...]] = []
    for initial in reversed(finished):
        if initial in visited:
            continue
        component: list[str] = []
        stack = [initial]
        visited.add(initial)
        while stack:
            key = stack.pop()
            component.append(key)
            for child in reverse[key]:
                if child not in visited:
                    visited.add(child)
                    stack.append(child)
        components.append(tuple(sorted(component, key=rank.__getitem__)))
    return tuple(components)
