"""PiecePlanの派生関係を楽譜差、同一性手掛かり、演奏差へ分けて診断する。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from typing import Literal

from llm_musical_composer.performance_pipeline import (
    PiecePlan,
    PlanNode,
    RenderedPerformance,
    ScoreMaterial,
    ScoreSpec,
)

RelationStatus = Literal["related", "unrelated", "unassessed"]
SUPPORTED_CUES = frozenset(
    {"foreground_motif_head", "motif_head", "pitch_contour", "rhythm_skeleton"}
)


class RecurrenceAnalysisError(ValueError):
    """再現関係の診断入力が契約外であることを表す。"""


@dataclass(frozen=True)
class RecurrenceAssessment:
    target_node_id: str
    source_node_id: str
    descendant_correspondence: tuple[tuple[str, str], ...]
    score_difference: bool
    satisfied_identity_cues: tuple[str, ...]
    performance_difference: bool
    exact_surface_copy: bool
    relation_status: RelationStatus


def _children(plan: PiecePlan) -> dict[str, list[PlanNode]]:
    result: dict[str, list[PlanNode]] = defaultdict(list)
    for node in plan.nodes:
        if node.parent_id is not None:
            result[node.parent_id].append(node)
    for items in result.values():
        items.sort(key=lambda item: item.order)
    return result


def _subtree_nodes(
    root: PlanNode,
    children: dict[str, list[PlanNode]],
) -> tuple[PlanNode, ...]:
    result: list[PlanNode] = []

    def visit(node: PlanNode) -> None:
        result.append(node)
        for child in children[node.node_id]:
            visit(child)

    visit(root)
    return tuple(result)


def _leaves(
    root: PlanNode,
    children: dict[str, list[PlanNode]],
) -> tuple[PlanNode, ...]:
    return tuple(node for node in _subtree_nodes(root, children) if not children[node.node_id])


def _material_ancestors(
    material_id: str,
    materials: dict[str, ScoreMaterial],
) -> tuple[str, ...]:
    result: list[str] = []
    current = materials[material_id]
    while current.derived_from is not None:
        result.append(current.derived_from)
        current = materials[current.derived_from]
    return tuple(result)


def _correspondence(
    target: PlanNode,
    source: PlanNode,
    children: dict[str, list[PlanNode]],
    materials: dict[str, ScoreMaterial],
) -> tuple[tuple[str, str], ...]:
    source_leaves = _leaves(source, children)
    source_by_id = {leaf.node_id: leaf for leaf in source_leaves}
    pairs: list[tuple[str, str]] = []
    for target_leaf in _leaves(target, children):
        if target_leaf.derived_from in source_by_id:
            pairs.append((target_leaf.node_id, target_leaf.derived_from))
            continue
        assert target_leaf.score_material_id is not None
        ancestors = _material_ancestors(target_leaf.score_material_id, materials)
        matching = next(
            (
                source_leaf
                for source_leaf in source_leaves
                if source_leaf.score_material_id == target_leaf.score_material_id
                or source_leaf.score_material_id in ancestors
            ),
            None,
        )
        if matching is not None:
            pairs.append((target_leaf.node_id, matching.node_id))
    return tuple(pairs)


def _score_signature(
    root: PlanNode,
    children: dict[str, list[PlanNode]],
    materials: dict[str, ScoreMaterial],
) -> tuple[tuple[object, ...], tuple[object, ...], int]:
    notes: list[tuple[object, ...]] = []
    directions: list[tuple[object, ...]] = []
    offset = 0
    for leaf in _leaves(root, children):
        assert leaf.score_material_id is not None
        material = materials[leaf.score_material_id]
        notes.extend(
            (
                offset + note.at_units,
                note.duration_units,
                note.pitch,
                note.voice,
                note.tie,
                note.articulations,
            )
            for note in material.notes
        )
        directions.extend(
            (offset + direction.at_units, direction.kind, direction.value)
            for direction in material.directions
        )
        offset += material.length_units
    return tuple(sorted(notes)), tuple(sorted(directions)), offset


def _pitch_contour(
    signature: tuple[tuple[object, ...], tuple[object, ...], int],
) -> tuple[int, ...]:
    notes = signature[0]
    pitches = [int(note[2]) for note in sorted(notes) if note[3] == "upper"]
    return tuple(second - first for first, second in pairwise(pitches))


def _motif_head(
    signature: tuple[tuple[object, ...], tuple[object, ...], int],
) -> tuple[int, ...]:
    by_onset: dict[int, list[int]] = defaultdict(list)
    for note in signature[0]:
        if note[3] == "upper":
            by_onset[int(note[0])].append(int(note[2]))
    melody = [max(by_onset[onset]) for onset in sorted(by_onset)]
    if len(melody) < 4:
        return ()
    head = melody[:4]
    return tuple(pitch - head[0] for pitch in head)


def _foreground_motif_head(
    root: PlanNode,
    children: dict[str, list[PlanNode]],
    materials: dict[str, ScoreMaterial],
) -> tuple[int, ...]:
    by_onset: dict[int, list[int]] = defaultdict(list)
    offset = 0
    for leaf in _leaves(root, children):
        assert leaf.score_material_id is not None
        material = materials[leaf.score_material_id]
        if material.foreground_voice is None:
            return ()
        foreground = tuple(
            note for note in material.notes if note.voice == material.foreground_voice
        )
        select = max if material.foreground_voice == "upper" else min
        local: dict[int, list[int]] = defaultdict(list)
        for note in foreground:
            local[note.at_units].append(note.pitch)
        for onset in sorted(local):
            by_onset[offset + onset].append(select(local[onset]))
        offset += material.length_units
    melody = [max(by_onset[onset]) for onset in sorted(by_onset)]
    if len(melody) < 4:
        return ()
    head = melody[:4]
    return tuple(pitch - head[0] for pitch in head)


def _rhythm_skeleton(
    signature: tuple[tuple[object, ...], tuple[object, ...], int],
) -> tuple[tuple[Fraction, Fraction, object], ...]:
    notes, _, total = signature
    return tuple(
        (
            Fraction(int(note[0]), total),
            Fraction(int(note[1]), total),
            note[3],
        )
        for note in notes
    )


def _performance_signature(
    root: PlanNode,
    children: dict[str, list[PlanNode]],
    rendered: RenderedPerformance,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    nodes = _subtree_nodes(root, children)
    node_ids = {node.node_id for node in nodes}
    leaf_ids = {node.node_id for node in nodes if not children[node.node_id]}
    notes = [note for note in rendered.notes if note.occurrence_node_id in leaf_ids]
    if not notes:
        return (), ()
    start = min(note.at_ms for note in notes)
    end = max(note.at_ms + note.duration_ms for note in notes)
    span = max(1, end - start)
    note_signature = tuple(
        sorted(
            (
                Fraction(note.at_ms - start, span),
                Fraction(note.duration_ms, span),
                note.pitch,
                note.velocity,
                note.voice,
            )
            for note in notes
        )
    )
    pedal_signature = tuple(
        sorted(
            (
                Fraction(pedal.at_ms - start, span),
                pedal.value,
            )
            for pedal in rendered.pedals
            if pedal.occurrence_node_id in node_ids
        )
    )
    return note_signature, pedal_signature


def _satisfied_cues(
    requested: tuple[str, ...],
    source_signature: tuple[tuple[object, ...], tuple[object, ...], int],
    target_signature: tuple[tuple[object, ...], tuple[object, ...], int],
) -> tuple[str, ...]:
    unknown = set(requested) - SUPPORTED_CUES
    if unknown:
        raise RecurrenceAnalysisError(f"unsupported identity cues: {', '.join(sorted(unknown))}")
    satisfied: list[str] = []
    source_head = _motif_head(source_signature)
    if "motif_head" in requested and source_head and source_head == _motif_head(target_signature):
        satisfied.append("motif_head")
    if "pitch_contour" in requested and _pitch_contour(source_signature) == _pitch_contour(
        target_signature
    ):
        satisfied.append("pitch_contour")
    if "rhythm_skeleton" in requested and _rhythm_skeleton(source_signature) == _rhythm_skeleton(
        target_signature
    ):
        satisfied.append("rhythm_skeleton")
    return tuple(satisfied)


def analyze_recurrences(
    plan: PiecePlan,
    score: ScoreSpec,
    rendered: RenderedPerformance,
    *,
    declared_cues: dict[str, tuple[str, ...]] | None = None,
) -> tuple[RecurrenceAssessment, ...]:
    """各derived_fromを、参照、楽譜差、保持手掛かり、演奏差へ分解する。"""
    cue_map = declared_cues or {}
    node_by_id = {node.node_id: node for node in plan.nodes}
    children = _children(plan)
    materials = {material.material_id: material for material in score.materials}
    results: list[RecurrenceAssessment] = []
    for target in plan.nodes:
        if target.derived_from is None:
            continue
        source = node_by_id[target.derived_from]
        correspondence = _correspondence(target, source, children, materials)
        source_score = _score_signature(source, children, materials)
        target_score = _score_signature(target, children, materials)
        score_difference = source_score != target_score
        paired_score_identity = any(
            _score_signature(node_by_id[source_leaf], children, materials)
            == _score_signature(node_by_id[target_leaf], children, materials)
            for target_leaf, source_leaf in correspondence
        )
        requested = cue_map.get(target.node_id, ())
        satisfied_set = set(_satisfied_cues(requested, source_score, target_score))
        source_foreground_head = _foreground_motif_head(source, children, materials)
        if (
            "foreground_motif_head" in requested
            and source_foreground_head
            and source_foreground_head == _foreground_motif_head(target, children, materials)
        ):
            satisfied_set.add("foreground_motif_head")
        for cue in requested:
            if cue == "foreground_motif_head":
                if correspondence and all(
                    (head := _foreground_motif_head(node_by_id[source_leaf], children, materials))
                    and head == _foreground_motif_head(node_by_id[target_leaf], children, materials)
                    for target_leaf, source_leaf in correspondence
                ):
                    satisfied_set.add(cue)
                continue
            if correspondence and all(
                cue
                in _satisfied_cues(
                    (cue,),
                    _score_signature(node_by_id[source_leaf], children, materials),
                    _score_signature(node_by_id[target_leaf], children, materials),
                )
                for target_leaf, source_leaf in correspondence
            ):
                satisfied_set.add(cue)
        satisfied = tuple(cue for cue in requested if cue in satisfied_set)
        performance_difference = _performance_signature(
            source, children, rendered
        ) != _performance_signature(target, children, rendered)
        if not correspondence:
            status: RelationStatus = "unrelated"
        elif not score_difference or paired_score_identity:
            status = "related"
        elif not requested:
            status = "unassessed"
        elif satisfied:
            status = "related"
        else:
            status = "unrelated"
        results.append(
            RecurrenceAssessment(
                target_node_id=target.node_id,
                source_node_id=source.node_id,
                descendant_correspondence=correspondence,
                score_difference=score_difference,
                satisfied_identity_cues=satisfied,
                performance_difference=performance_difference,
                exact_surface_copy=not score_difference and not performance_difference,
                relation_status=status,
            )
        )
    return tuple(results)
