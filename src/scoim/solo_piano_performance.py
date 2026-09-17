"""Deterministic staging for solo-piano performance choices."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from llm_musical_composer.performance_pipeline import PiecePlan, PlanNode

from .projection import PlanChoice, project_solo_piano_3m
from .realization_workspace import RealizationWorkspace


@dataclass(frozen=True, slots=True)
class PerformanceDirectionGroup:
    """All script directions that resolve to one performance node."""

    node_id: str
    direction_ids: tuple[str, ...]
    descriptions: tuple[str, ...]
    comparison_node_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PerformanceOperationSchedule:
    """Model-free defaults followed by dependency-ordered model waves."""

    default_node_ids: tuple[str, ...]
    absolute_node_ids: tuple[str, ...]
    comparative_waves: tuple[tuple[str, ...], ...]
    direction_groups: tuple[PerformanceDirectionGroup, ...]


def performance_piece_plan(
    document: Mapping[str, object], workspace: RealizationWorkspace
) -> PiecePlan:
    if workspace.schema_version not in {3, 4}:
        raise ValueError("Performance staging requires a version 3 or 4 workspace")
    if workspace.plan is None:
        raise ValueError("Performance staging requires a current overall plan")
    projection = project_solo_piano_3m(
        document,
        plan_choice=PlanChoice(
            tonal_center=workspace.plan.tonal_center,
            mode=workspace.plan.mode,
            harmonic_focus_by_section={},
            contrasts_with_by_section={},
        ),
    )
    if not projection.projected or projection.piece_plan is None:
        message = projection.issues[0].message if projection.issues else "Projection failed"
        raise ValueError(message)
    return projection.piece_plan


def _resolve_node_id(reference: Mapping[str, object], script: Mapping[str, object]) -> str:
    reference_type = reference["type"]
    reference_id = cast(str, reference["id"])
    if reference_type == "section":
        return reference_id
    placements = cast(Mapping[str, Mapping[str, object]], script["placements"])
    if reference_type == "placement" and reference_id in placements:
        return cast(str, placements[reference_id]["section_id"])
    raise ValueError("Performance direction target cannot be resolved to a section")


def _direction_groups(
    document: Mapping[str, object], plan: PiecePlan
) -> tuple[PerformanceDirectionGroup, ...]:
    return _direction_groups_for_nodes(document, plan.nodes)


def _direction_groups_for_nodes(
    document: Mapping[str, object], nodes: Sequence[PlanNode]
) -> tuple[PerformanceDirectionGroup, ...]:
    script = cast(Mapping[str, object], document["script"])
    setup = cast(Mapping[str, object], script["performance_setup"])
    directions = cast(Mapping[str, Mapping[str, object]], setup["performance_directions"])
    grouped: dict[str, list[tuple[str, Mapping[str, object]]]] = defaultdict(list)
    for direction_id, direction in sorted(directions.items()):
        target = cast(Mapping[str, object], direction["target"])
        grouped[_resolve_node_id(target, script)].append((direction_id, direction))
    order = {node.node_id: index for index, node in enumerate(nodes)}
    unknown = set(grouped) - set(order)
    if unknown:
        raise ValueError(f"Unknown performance target node: {sorted(unknown)[0]}")
    result: list[PerformanceDirectionGroup] = []
    for node_id in sorted(grouped, key=order.__getitem__):
        items = grouped[node_id]
        comparisons = tuple(
            _resolve_node_id(cast(Mapping[str, object], direction["relative_to"]), script)
            for _, direction in items
            if "relative_to" in direction
        )
        result.append(
            PerformanceDirectionGroup(
                node_id=node_id,
                direction_ids=tuple(direction_id for direction_id, _ in items),
                descriptions=tuple(cast(str, direction["description"]) for _, direction in items),
                comparison_node_ids=comparisons,
            )
        )
    return tuple(result)


def _comparison_dependencies(plan: PiecePlan, source_node_id: str) -> set[str]:
    return _comparison_dependencies_for_nodes(plan.nodes, source_node_id)


def _comparison_dependencies_for_nodes(
    plan_nodes: Sequence[PlanNode], source_node_id: str
) -> set[str]:
    nodes = {node.node_id: node for node in plan_nodes}
    if source_node_id not in nodes:
        raise ValueError(f"Unknown performance comparison node: {source_node_id}")
    children: dict[str, list[str]] = defaultdict(list)
    for node in plan_nodes:
        if node.parent_id is not None:
            children[node.parent_id].append(node.node_id)
    subtree: set[str] = set()
    pending = [source_node_id]
    while pending:
        node_id = pending.pop()
        if node_id in subtree:
            continue
        subtree.add(node_id)
        pending.extend(children[node_id])
    dependencies = set(subtree)
    for node_id in tuple(subtree):
        parent_id = nodes[node_id].parent_id
        while parent_id is not None:
            dependencies.add(parent_id)
            parent_id = nodes[parent_id].parent_id
    return dependencies


def _children(plan: PiecePlan) -> dict[str, tuple[str, ...]]:
    mutable: dict[str, list[str]] = defaultdict(list)
    for node in plan.nodes:
        if node.parent_id is not None:
            mutable[node.parent_id].append(node.node_id)
    return {key: tuple(value) for key, value in mutable.items()}


def _leaf_ids(plan: PiecePlan, node_id: str) -> tuple[str, ...]:
    children = _children(plan)
    leaves: list[str] = []

    def visit(current: str) -> None:
        child_ids = children.get(current, ())
        if not child_ids:
            leaves.append(current)
            return
        for child_id in child_ids:
            visit(child_id)

    visit(node_id)
    return tuple(leaves)


def _ancestor_path(plan: PiecePlan, node_id: str) -> tuple[str, ...]:
    nodes = {node.node_id: node for node in plan.nodes}
    reverse_path: list[str] = []
    current: str | None = node_id
    while current is not None:
        reverse_path.append(current)
        current = nodes[current].parent_id
    return tuple(reversed(reverse_path))


def _transition_key_by_section(script: Mapping[str, object]) -> dict[str, str]:
    placements = cast(Mapping[str, Mapping[str, object]], script["placements"])
    transitions = cast(Mapping[str, Mapping[str, object]], script["transitions"])
    result: dict[str, str] = {}
    for transition_id, transition in transitions.items():
        placement_id = cast(str, transition["connector_placement_id"])
        section_id = cast(str, placements[placement_id]["section_id"])
        if section_id in result:
            raise ValueError("A transition section cannot represent multiple transition relations")
        result[section_id] = transition_id
    return result


def _score_summary(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    plan: PiecePlan,
    node_id: str,
) -> list[dict[str, object]]:
    script = cast(Mapping[str, object], document["script"])
    materials = cast(Mapping[str, Mapping[str, object]], script["materials"])
    nodes = {node.node_id: node for node in plan.nodes}
    transition_keys = _transition_key_by_section(script)
    harmonies = dict(workspace.harmonies)
    melodies = dict(workspace.melodies)
    accompaniments = dict(workspace.accompaniments)
    transition_melodies = dict(workspace.transition_melodies)
    transition_accompaniments = dict(workspace.transition_accompaniments)
    summaries: list[dict[str, object]] = []
    for leaf_id in _leaf_ids(plan, node_id):
        material_id = cast(str, nodes[leaf_id].score_material_id)
        material_kind = cast(str, materials[material_id]["kind"])
        if material_kind == "transition":
            relation_id = transition_keys.get(leaf_id)
            if relation_id is None:
                raise ValueError("A transition score summary cannot resolve its relation")
            melody = transition_melodies.get(relation_id)
            accompaniment = transition_accompaniments.get(relation_id)
        else:
            melody = melodies.get(material_id)
            accompaniment = accompaniments.get(material_id)
        harmony = harmonies.get(material_id)
        if harmony is None or melody is None or accompaniment is None:
            raise ValueError(f"Performance score summary is missing material for {leaf_id}")
        notes = (*melody.notes, *accompaniment.notes)
        positions = sorted({note.at_units for note in notes})
        max_simultaneous = max(
            (
                sum(
                    note.at_units <= position < note.at_units + note.duration_units
                    for note in notes
                )
                for position in positions
            ),
            default=0,
        )
        articulation_counts: dict[str, int] = defaultdict(int)
        for note in accompaniment.notes:
            for articulation in note.articulations:
                articulation_counts[articulation] += 1
        summaries.append(
            {
                "node_id": leaf_id,
                "material_id": material_id,
                "material_kind": material_kind,
                "relative_length": nodes[leaf_id].duration_weight,
                "duration_units": sum(chord.duration_units for chord in harmony),
                "foreground_voice": melody.foreground_voice,
                "harmony_count": len(harmony),
                "melody_note_count": len(melody.notes),
                "accompaniment_note_count": len(accompaniment.notes),
                "attack_position_count": len(positions),
                "max_simultaneous_note_count": max_simultaneous,
                "articulation_counts": dict(sorted(articulation_counts.items())),
            }
        )
    return summaries


def _score_value_keys(document: Mapping[str, object], plan: PiecePlan, node_id: str) -> set[str]:
    script = cast(Mapping[str, object], document["script"])
    materials = cast(Mapping[str, Mapping[str, object]], script["materials"])
    nodes = {node.node_id: node for node in plan.nodes}
    transition_keys = _transition_key_by_section(script)
    keys: set[str] = set()
    for leaf_id in _leaf_ids(plan, node_id):
        material_id = cast(str, nodes[leaf_id].score_material_id)
        escaped_material = material_id.replace("~", "~0").replace("/", "~1")
        keys.add(f"/harmonies/{escaped_material}")
        if materials[material_id]["kind"] == "transition":
            relation_id = transition_keys.get(leaf_id)
            if relation_id is None:
                raise ValueError("A transition dependency cannot resolve its relation")
            escaped_relation = relation_id.replace("~", "~0").replace("/", "~1")
            keys.add(f"/transition_melodies/{escaped_relation}")
            keys.add(f"/transition_accompaniments/{escaped_relation}")
        else:
            keys.add(f"/melodies/{escaped_material}")
            keys.add(f"/accompaniments/{escaped_material}")
    return keys


def performance_read_keys(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    target_node_id: str,
) -> tuple[str, ...]:
    """Return the exact workspace keys read to choose one node's performance."""
    plan = performance_piece_plan(document, workspace)
    groups = {group.node_id: group for group in _direction_groups(document, plan)}
    group = groups.get(target_node_id)
    if group is None:
        raise ValueError(f"Unknown directed performance node: {target_node_id}")
    keys = {"/approved_script_sha256", "/plan"}
    keys.update(_score_value_keys(document, plan, target_node_id))
    for source_id in group.comparison_node_ids:
        keys.update(_score_value_keys(document, plan, source_id))
        keys.update(
            f"/performances/{node_id.replace('~', '~0').replace('/', '~1')}"
            for node_id in _comparison_dependencies(plan, source_id)
        )
    missing = [key for key in sorted(keys) if not _workspace_key_exists(workspace, key)]
    if missing:
        raise ValueError(f"Performance dependency is not settled: {missing[0]}")
    return tuple(sorted(keys))


def _workspace_key_exists(workspace: RealizationWorkspace, key: str) -> bool:
    collection, _, escaped_item = key.lstrip("/").partition("/")
    if not escaped_item:
        return collection in {"approved_script_sha256", "plan"}
    item = escaped_item.replace("~1", "/").replace("~0", "~")
    values = getattr(workspace, collection, ())
    return item in dict(values)


def _comparison_performance_summary(
    plan: PiecePlan, workspace: RealizationWorkspace, source_node_id: str
) -> list[dict[str, object]]:
    performances = dict(workspace.performances)
    dependencies = _comparison_dependencies(plan, source_node_id)
    missing = dependencies - set(performances)
    if missing:
        raise ValueError(f"Comparison performance is not settled for {sorted(missing)[0]}")
    result: list[dict[str, object]] = []
    for leaf_id in _leaf_ids(plan, source_node_id):
        path = _ancestor_path(plan, leaf_id)
        timing = []
        for owner_node_id in path:
            value = performances[owner_node_id]
            if value.timing_profile is not None:
                timing.append(
                    {
                        "owner_node_id": owner_node_id,
                        "covered_leaf_node_ids": list(_leaf_ids(plan, owner_node_id)),
                        "timing_profile": value.timing_profile,
                        "timing_amount": value.timing_amount,
                    }
                )
        effective: dict[str, object] = {}
        for field in (
            "dynamics_profile",
            "articulation_profile",
            "coordination_profile",
            "pedal_profile",
        ):
            resolved_owner = None
            resolved_value = None
            for owner_node_id in reversed(path):
                candidate = getattr(performances[owner_node_id], field)
                if candidate is not None:
                    resolved_owner = owner_node_id
                    resolved_value = candidate
                    break
            effective[field] = {
                "owner_node_id": resolved_owner,
                "value": resolved_value,
            }
        result.append(
            {
                "leaf_node_id": leaf_id,
                "timing": timing,
                "effective_profiles": effective,
            }
        )
    return result


def performance_contexts(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    targets: tuple[str, ...],
) -> list[dict[str, object]]:
    """Build deterministic score and settled-comparison context for one model operation."""
    plan = performance_piece_plan(document, workspace)
    groups = {group.node_id: group for group in _direction_groups(document, plan)}
    script = cast(Mapping[str, object], document["script"])
    sections = cast(Mapping[str, Mapping[str, object]], script["sections"])
    nodes = {node.node_id: node for node in plan.nodes}
    contexts: list[dict[str, object]] = []
    for target in targets:
        if target not in groups:
            raise ValueError(f"Unknown directed performance node: {target}")
        group = groups[target]
        contexts.append(
            {
                "node_id": target,
                "role": nodes[target].role,
                "parent_node_id": nodes[target].parent_id,
                "description": sections[target]["description"],
                "direction_ids": list(group.direction_ids),
                "direction_descriptions": list(group.descriptions),
                "score_summary": _score_summary(document, workspace, plan, target),
                "comparison_sources": [
                    {
                        "node_id": source_id,
                        "description": sections[source_id]["description"],
                        "score_summary": _score_summary(document, workspace, plan, source_id),
                        "performance_by_leaf": _comparison_performance_summary(
                            plan, workspace, source_id
                        ),
                    }
                    for source_id in group.comparison_node_ids
                ],
            }
        )
    return contexts


def performance_operation_schedule(
    document: Mapping[str, object], workspace: RealizationWorkspace
) -> PerformanceOperationSchedule:
    """Resolve direction targets and order comparative targets by settled dependencies."""
    plan = performance_piece_plan(document, workspace)
    return performance_operation_schedule_for_nodes(document, plan.nodes)


def performance_operation_schedule_for_nodes(
    document: Mapping[str, object], nodes: Sequence[PlanNode]
) -> PerformanceOperationSchedule:
    """Build the model-operation schedule from validated structure only."""
    groups = _direction_groups_for_nodes(document, nodes)
    node_order = {node.node_id: index for index, node in enumerate(nodes)}
    directed = {group.node_id for group in groups}
    defaults = tuple(node.node_id for node in nodes if node.node_id not in directed)
    absolute = tuple(group.node_id for group in groups if not group.comparison_node_ids)
    remaining = {group.node_id: group for group in groups if group.comparison_node_ids}
    available = set(defaults) | set(absolute)
    waves: list[tuple[str, ...]] = []
    while remaining:
        ready = tuple(
            sorted(
                (
                    node_id
                    for node_id, group in remaining.items()
                    if all(
                        (_comparison_dependencies_for_nodes(nodes, source_id) & directed)
                        <= available
                        for source_id in group.comparison_node_ids
                    )
                ),
                key=node_order.__getitem__,
            )
        )
        if not ready:
            raise ValueError("Performance comparison dependencies cannot be ordered")
        waves.append(ready)
        available.update(ready)
        for node_id in ready:
            remaining.pop(node_id)
    return PerformanceOperationSchedule(defaults, absolute, tuple(waves), groups)
