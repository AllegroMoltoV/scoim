"""Convert a completed realization workspace into the existing piano IR."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

import jsonpointer

from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    PiecePlan,
    PipelineValidationError,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    validate_pipeline,
)

from .projection import PlanChoice, ProjectionTarget, project_solo_piano_3m
from .realization_script import check_realization_script, realization_script_sha256
from .realization_workspace import (
    AccompanimentValue,
    HarmonyChord,
    MelodyValue,
    RealizationWorkspace,
    workspace_from_dict,
    workspace_to_dict,
)
from .typed_realization import (
    TypedRealizationError,
    build_performance_spec_from_typed_response,
    ensure_distinct_event_boundaries,
)
from .validation import IssueCode, ValidationIssue

_DIVISIONS = 12
_TARGET_DURATION_MS = 180_000
_TRANSITION_MATERIAL_PREFIX = "__scoim_transition__"


class WorkspaceRealizationError(ValueError):
    """A completed workspace cannot be represented by the existing piano IR."""

    def __init__(self, issue: ValidationIssue) -> None:
        super().__init__(issue.message)
        self.issue = issue


@dataclass(frozen=True, slots=True)
class WorkspaceIR:
    """Existing three-stage IR reconstructed from one completed workspace."""

    plan: PiecePlan
    score: ScoreSpec
    performance: PerformanceSpec
    targets: tuple[ProjectionTarget, ...]
    common_units_per_weight: int
    material_bindings: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class WorkspaceDiagnosticIR:
    """One non-creative, auditionable view of a realization phase."""

    plan: PiecePlan
    score: ScoreSpec
    performance: PerformanceSpec


def _unrepresentable(message: str, path: str) -> WorkspaceRealizationError:
    return WorkspaceRealizationError(ValidationIssue(IssueCode.UNREPRESENTABLE, message, path))


def _checked_workspace(
    document: Mapping[str, object], workspace: RealizationWorkspace
) -> RealizationWorkspace:
    validation = check_realization_script(document)
    if not validation.valid:
        raise WorkspaceRealizationError(validation.issues[0])
    try:
        checked = workspace_from_dict(workspace_to_dict(workspace))
    except ValueError as error:
        raise _unrepresentable(str(error), "/frozen_response/workspace") from error
    if checked.schema_version != 4:
        raise _unrepresentable(
            "Workspace realization requires workspace schema version 4",
            "/frozen_response/workspace/schema_version",
        )
    if checked.approved_script_sha256 != realization_script_sha256(document):
        raise WorkspaceRealizationError(
            ValidationIssue(
                IssueCode.LINEAGE_MISMATCH,
                "The workspace does not belong to the approved script",
                "/frozen_response/workspace/approved_script_sha256",
            )
        )
    if checked.plan is None:
        raise _unrepresentable(
            "Workspace realization requires a completed overall plan",
            "/frozen_response/workspace/plan",
        )
    return checked


def _project(
    document: Mapping[str, object], workspace: RealizationWorkspace
) -> tuple[PiecePlan, tuple[ProjectionTarget, ...]]:
    assert workspace.plan is not None
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
        issue = projection.issues[0]
        raise WorkspaceRealizationError(issue)
    return projection.piece_plan, projection.targets


def _transition_sections(
    script: Mapping[str, object],
) -> tuple[dict[str, str], dict[str, str]]:
    placements = cast(Mapping[str, Mapping[str, object]], script["placements"])
    transitions = cast(Mapping[str, Mapping[str, object]], script["transitions"])
    relation_by_section: dict[str, str] = {}
    relation_by_placement: dict[str, str] = {}
    material_by_relation: dict[str, str] = {}
    for relation_id, transition in sorted(transitions.items()):
        placement_id = cast(str, transition["connector_placement_id"])
        if placement_id in relation_by_placement:
            raise _unrepresentable(
                "A connector placement cannot serve multiple transition relations",
                f"/script/transitions/{relation_id}/connector_placement_id",
            )
        placement = placements[placement_id]
        section_id = cast(str, placement["section_id"])
        material_id = cast(str, placement["material_id"])
        relation_by_placement[placement_id] = relation_id
        relation_by_section[section_id] = relation_id
        material_by_relation[relation_id] = material_id
    return relation_by_section, material_by_relation


def _expanded_plan(
    plan: PiecePlan,
    relation_by_section: Mapping[str, str],
) -> PiecePlan:
    return replace(
        plan,
        nodes=tuple(
            replace(
                node,
                score_material_id=f"{_TRANSITION_MATERIAL_PREFIX}{relation_by_section[node.node_id]}",
            )
            if node.node_id in relation_by_section
            else node
            for node in plan.nodes
        ),
    )


def _material_sources(
    document: Mapping[str, object],
    plan: PiecePlan,
    workspace: RealizationWorkspace,
    relation_by_section: Mapping[str, str],
    material_by_relation: Mapping[str, str],
) -> tuple[
    dict[str, tuple[str, str | None]],
    dict[str, tuple[str, ...]],
]:
    script = cast(Mapping[str, object], document["script"])
    materials = cast(Mapping[str, Mapping[str, object]], script["materials"])
    sources: dict[str, tuple[str, str | None]] = {}
    bindings: dict[str, list[str]] = {material_id: [] for material_id in materials}
    for node in plan.nodes:
        if node.score_material_id is None:
            continue
        relation_id = relation_by_section.get(node.node_id)
        source_material_id = (
            material_by_relation[relation_id] if relation_id is not None else node.score_material_id
        )
        realized_material_id = (
            f"{_TRANSITION_MATERIAL_PREFIX}{relation_id}"
            if relation_id is not None
            else source_material_id
        )
        previous = sources.get(realized_material_id)
        source = (source_material_id, relation_id)
        if previous is not None and previous != source:
            raise _unrepresentable(
                "A realized material has conflicting workspace sources",
                f"/script/sections/{node.node_id}",
            )
        sources[realized_material_id] = source
        if realized_material_id not in bindings[source_material_id]:
            bindings[source_material_id].append(realized_material_id)
    missing = [material_id for material_id, values in bindings.items() if not values]
    if missing:
        raise _unrepresentable(
            "A script material has no realized placement",
            f"/script/materials/{missing[0]}",
        )
    return sources, {key: tuple(value) for key, value in bindings.items()}


def _require_exact_keys(
    actual_items: tuple[tuple[str, object], ...],
    expected: set[str],
    field: str,
) -> None:
    actual = {key for key, _ in actual_items}
    if actual != expected:
        key = sorted(actual ^ expected)[0]
        raise _unrepresentable(
            "The completed workspace keys do not match the approved script",
            f"/frozen_response/workspace/{field}/{jsonpointer.escape(key)}",
        )


def _ensure_complete_keys(
    document: Mapping[str, object],
    workspace: RealizationWorkspace,
    plan: PiecePlan,
) -> None:
    script = cast(Mapping[str, object], document["script"])
    materials = cast(Mapping[str, Mapping[str, object]], script["materials"])
    transitions = cast(Mapping[str, object], script["transitions"])
    material_ids = set(materials)
    transition_material_ids = {
        material_id
        for material_id, material in materials.items()
        if material["kind"] == "transition"
    }
    direct_material_ids = material_ids - transition_material_ids
    _require_exact_keys(workspace.harmonies, material_ids, "harmonies")
    _require_exact_keys(workspace.melodies, direct_material_ids, "melodies")
    _require_exact_keys(workspace.accompaniments, direct_material_ids, "accompaniments")
    _require_exact_keys(
        workspace.transition_melodies,
        set(transitions),
        "transition_melodies",
    )
    _require_exact_keys(
        workspace.transition_accompaniments,
        set(transitions),
        "transition_accompaniments",
    )
    _require_exact_keys(
        workspace.performances,
        {node.node_id for node in plan.nodes},
        "performances",
    )


def _workspace_values(
    workspace: RealizationWorkspace,
    source_material_id: str,
    relation_id: str | None,
) -> tuple[tuple[HarmonyChord, ...], MelodyValue, AccompanimentValue]:
    harmonies = dict(workspace.harmonies)
    melodies = dict(workspace.melodies)
    accompaniments = dict(workspace.accompaniments)
    transition_melodies = dict(workspace.transition_melodies)
    transition_accompaniments = dict(workspace.transition_accompaniments)
    try:
        harmony = harmonies[source_material_id]
        if relation_id is None:
            melody = melodies[source_material_id]
            accompaniment = accompaniments[source_material_id]
        else:
            melody = transition_melodies[relation_id]
            accompaniment = transition_accompaniments[relation_id]
    except KeyError as error:
        key = cast(str, error.args[0])
        raise _unrepresentable(
            "The completed workspace is missing a required score value",
            f"/frozen_response/workspace/{key}",
        ) from error
    if melody.foreground_voice != accompaniment.foreground_voice:
        raise _unrepresentable(
            "Melody and accompaniment foreground voices do not match",
            f"/frozen_response/workspace/accompaniments/{source_material_id}",
        )
    return harmony, melody, accompaniment


def _common_grid(
    plan: PiecePlan,
    sources: Mapping[str, tuple[str, str | None]],
    workspace: RealizationWorkspace,
) -> tuple[int, dict[str, int]]:
    weights: dict[str, int] = {}
    lengths: dict[str, int] = {}
    for node in plan.nodes:
        if node.score_material_id is None or node.duration_weight is None:
            continue
        material_id = node.score_material_id
        weights[material_id] = node.duration_weight
        source_material_id, relation_id = sources[material_id]
        harmony, _, _ = _workspace_values(workspace, source_material_id, relation_id)
        lengths[material_id] = sum(chord.duration_units for chord in harmony)
    common_units = math.lcm(
        *(
            length // math.gcd(length, weights[material_id])
            for material_id, length in lengths.items()
        )
    )
    total_units = sum(
        cast(int, node.duration_weight) * common_units
        for node in plan.nodes
        if node.duration_weight is not None
    )
    if total_units > _TARGET_DURATION_MS:
        raise _unrepresentable(
            "The projected score grid exceeds the solo-piano resource limit",
            "/script/sections",
        )
    return common_units, {
        material_id: weights[material_id] * common_units // length
        for material_id, length in lengths.items()
    }


def _score_material(
    realized_material_id: str,
    values: tuple[tuple[HarmonyChord, ...], MelodyValue, AccompanimentValue],
    scale: int,
) -> ScoreMaterial:
    harmony_value, melody, accompaniment = values
    at_units = 0
    harmonies: list[ScoreHarmony] = []
    for index, chord in enumerate(harmony_value):
        duration_units = chord.duration_units * scale
        harmonies.append(
            ScoreHarmony(
                harmony_id=f"{realized_material_id}-harmony-{index:03d}",
                at_units=at_units,
                duration_units=duration_units,
                root_pitch_class=chord.root_pitch_class,
                quality=chord.quality,
            )
        )
        at_units += duration_units
    notes = tuple(
        ScoreNote(
            event_id=f"{realized_material_id}-melody-{index:03d}",
            at_units=note.at_units * scale,
            duration_units=note.duration_units * scale,
            pitch=note.pitch,
            voice=melody.foreground_voice,
        )
        for index, note in enumerate(melody.notes)
    ) + tuple(
        ScoreNote(
            event_id=f"{realized_material_id}-accompaniment-{index:03d}",
            at_units=note.at_units * scale,
            duration_units=note.duration_units * scale,
            pitch=note.pitch,
            voice="lower" if melody.foreground_voice == "upper" else "upper",
            articulations=note.articulations,
        )
        for index, note in enumerate(accompaniment.notes)
    )
    return ScoreMaterial(
        material_id=realized_material_id,
        length_units=at_units,
        notes=tuple(
            sorted(notes, key=lambda item: (item.at_units, item.voice, item.pitch, item.event_id))
        ),
        harmonies=tuple(harmonies),
        foreground_voice=melody.foreground_voice,
    )


def build_workspace_ir(
    document: Mapping[str, object], workspace: RealizationWorkspace
) -> WorkspaceIR:
    """Build validated existing IR from one completed workspace without creative repair."""

    checked = _checked_workspace(document, workspace)
    plan, targets = _project(document, checked)
    _ensure_complete_keys(document, checked, plan)
    script = cast(Mapping[str, object], document["script"])
    relation_by_section, material_by_relation = _transition_sections(script)
    expanded_plan = _expanded_plan(plan, relation_by_section)
    sources, bindings = _material_sources(
        document,
        plan,
        checked,
        relation_by_section,
        material_by_relation,
    )
    common_units, scales = _common_grid(expanded_plan, sources, checked)
    score = ScoreSpec(
        score_id=f"{expanded_plan.plan_id}-score",
        divisions=_DIVISIONS,
        materials=tuple(
            _score_material(
                material_id,
                _workspace_values(checked, source_material_id, relation_id),
                scales[material_id],
            )
            for material_id, (source_material_id, relation_id) in sources.items()
        ),
    )
    performance_payload = {
        node_id: {
            "timing_profile": value.timing_profile,
            "timing_amount": value.timing_amount,
            "dynamics_profile": value.dynamics_profile,
            "articulation_profile": value.articulation_profile,
            "coordination_profile": value.coordination_profile,
            "pedal_profile": value.pedal_profile,
        }
        for node_id, value in checked.performances
    }
    try:
        performance = build_performance_spec_from_typed_response(expanded_plan, performance_payload)
        validate_pipeline(expanded_plan, score, performance)
        ensure_distinct_event_boundaries(expanded_plan, score, performance)
    except (PipelineValidationError, TypedRealizationError) as error:
        if isinstance(error, TypedRealizationError):
            raise WorkspaceRealizationError(error.issue) from error
        raise _unrepresentable(str(error), "/frozen_response/workspace") from error
    return WorkspaceIR(
        plan=expanded_plan,
        score=score,
        performance=performance,
        targets=targets,
        common_units_per_weight=common_units,
        material_bindings=MappingProxyType(bindings),
    )


def build_workspace_diagnostic_irs(
    document: Mapping[str, object], workspace: RealizationWorkspace
) -> tuple[WorkspaceDiagnosticIR, WorkspaceDiagnosticIR]:
    """Build melody-only and score-only previews without later creative values."""

    complete = build_workspace_ir(document, workspace)
    script = cast(Mapping[str, object], document["script"])
    materials = cast(Mapping[str, Mapping[str, object]], script["materials"])
    ending_material_ids = {
        realized_id
        for material_id, realized_ids in complete.material_bindings.items()
        if materials[material_id]["kind"] == "ending"
        for realized_id in realized_ids
    }
    removed_node_ids = {
        node.node_id
        for node in complete.plan.nodes
        if node.score_material_id in ending_material_ids
    }
    while True:
        child_ids = {
            node.parent_id
            for node in complete.plan.nodes
            if node.node_id not in removed_node_ids and node.parent_id is not None
        }
        newly_empty = {
            node.node_id
            for node in complete.plan.nodes
            if node.node_id not in removed_node_ids
            and node.score_material_id is None
            and node.node_id not in child_ids
        }
        if not newly_empty:
            break
        removed_node_ids.update(newly_empty)
    order_by_parent: dict[str | None, int] = {}
    melody_nodes = []
    for node in complete.plan.nodes:
        if node.node_id in removed_node_ids:
            continue
        order = order_by_parent.get(node.parent_id, 0)
        order_by_parent[node.parent_id] = order + 1
        melody_nodes.append(replace(node, order=order))
    melody_plan = replace(complete.plan, nodes=tuple(melody_nodes))
    melody_score = replace(
        complete.score,
        score_id=f"{complete.score.score_id}-melody-preview",
        materials=tuple(
            replace(
                material,
                notes=(
                    ()
                    if material.material_id in ending_material_ids
                    else tuple(
                        note for note in material.notes if note.voice == material.foreground_voice
                    )
                ),
            )
            for material in complete.score.materials
            if material.material_id not in ending_material_ids
        ),
    )
    total_weight = sum(
        cast(int, node.duration_weight)
        for node in complete.plan.nodes
        if node.duration_weight is not None
    )
    melody_weight = sum(
        cast(int, node.duration_weight)
        for node in melody_plan.nodes
        if node.duration_weight is not None
    )
    melody_performance = replace(
        complete.performance,
        performance_id=f"{complete.performance.performance_id}-neutral-preview",
        target_duration_ms=round(
            complete.performance.target_duration_ms * melody_weight / total_weight
        ),
        node_performances=tuple(
            NodePerformance(node_id=node.node_id) for node in melody_plan.nodes
        ),
    )
    score_performance = replace(
        complete.performance,
        performance_id=f"{complete.performance.performance_id}-neutral-preview",
        node_performances=tuple(
            NodePerformance(node_id=node.node_id) for node in complete.plan.nodes
        ),
    )
    validate_pipeline(melody_plan, melody_score, melody_performance)
    validate_pipeline(complete.plan, complete.score, score_performance)
    return (
        WorkspaceDiagnosticIR(melody_plan, melody_score, melody_performance),
        WorkspaceDiagnosticIR(complete.plan, complete.score, score_performance),
    )
