"""既存ScoreSpecを使わず、曲全体の段階生成入力を組み立てる。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise

from llm_musical_composer.harmonic_skeleton import (
    HarmonicMaterialV0,
    HarmonicSkeletonV0,
    MaterialPayloadV0,
    ScorePayloadV0,
    apply_harmonic_skeleton,
    validate_harmonic_skeleton,
)
from llm_musical_composer.performance_pipeline import (
    HARMONY_INTERVALS,
    NodePerformance,
    PerformanceSpec,
    PiecePlan,
    PlanNode,
    ScoreHarmony,
    ScoreMaterial,
    ScoreSpec,
    validate_piece_plan,
    validate_pipeline,
    validate_score_spec,
)
from llm_musical_composer.piano_texture_register_placement import (
    PianoTexturePlacementResult,
)
from llm_musical_composer.staged_material_pilot import (
    HarmonicDraft,
    MelodyDraft,
    TextureDraft,
    assemble_melody,
    assemble_texture,
    full_low_spacing_violations,
    texture_preplacement_violations,
)

WHOLE_SCORE_DIVISIONS = 12
WHOLE_SCORE_TARGET_DURATION_MS = 180_000
SUPPORTED_IDENTITY_CUES = frozenset({"foreground_motif_head", "rhythm_skeleton"})


class WholeScoreStagedError(ValueError):
    """曲全体の段階生成契約に違反した入力。"""


@dataclass(frozen=True)
class MaterialRelationV0:
    target_node_id: str
    source_node_id: str
    material_pairs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class TransitionContextV0:
    node_id: str
    material_id: str
    previous_node_id: str
    previous_material_id: str
    next_node_id: str
    next_material_id: str


@dataclass(frozen=True)
class WholeScoreContextV0:
    material_ids: tuple[str, ...]
    occurrence_weights: dict[str, tuple[int, ...]]
    transition_material_ids: tuple[str, ...]
    release_material_ids: tuple[str, ...]
    relations: tuple[MaterialRelationV0, ...]
    transitions: tuple[TransitionContextV0, ...]


@dataclass(frozen=True)
class MaterialIdentityCueV0:
    target_node_id: str
    source_node_id: str
    target_material_id: str
    source_material_id: str
    cues: tuple[str, ...]


@dataclass(frozen=True)
class ForegroundTransitionAssessmentV0:
    source_node_id: str
    transition_node_id: str
    target_node_id: str
    source_voice: str
    transition_voice: str
    target_voice: str
    transition_attack_count: int
    source_to_transition_semitones: int
    maximum_internal_leap: int
    transition_to_target_semitones: int
    same_pitch_restrike: bool
    passes: bool


@dataclass(frozen=True)
class WholeScoreTextureResultV0:
    score: ScoreSpec
    placements: tuple[PianoTexturePlacementResult, ...]
    low_spacing_violations: tuple[int, ...]


@dataclass(frozen=True)
class PerformanceOccurrenceDraftV0:
    timing_profile: str | None = None
    timing_amount: str | None = None
    dynamics_profile: str | None = None
    articulation_profile: str | None = None
    coordination_profile: str | None = None
    pedal_profile: str | None = None


@dataclass(frozen=True)
class WholeHarmonicMaterialDraftV0:
    """モデルが一つの和声段階で生成する素材長と和声列。"""

    length_units: int
    draft: HarmonicDraft


def _children(plan: PiecePlan) -> dict[str, tuple[PlanNode, ...]]:
    grouped: dict[str, list[PlanNode]] = {node.node_id: [] for node in plan.nodes}
    for node in plan.nodes:
        if node.parent_id is not None:
            grouped[node.parent_id].append(node)
    return {
        node_id: tuple(sorted(items, key=lambda item: item.order))
        for node_id, items in grouped.items()
    }


def _leaves(node: PlanNode, children: dict[str, tuple[PlanNode, ...]]) -> tuple[PlanNode, ...]:
    descendants = children[node.node_id]
    if not descendants:
        return (node,)
    return tuple(leaf for child in descendants for leaf in _leaves(child, children))


def _material_pairs(
    target: PlanNode,
    source: PlanNode,
    children: dict[str, tuple[PlanNode, ...]],
) -> tuple[tuple[str, str], ...]:
    source_leaves = _leaves(source, children)
    source_by_node = {leaf.node_id: leaf for leaf in source_leaves}
    pairs: list[tuple[str, str]] = []
    for target_leaf in _leaves(target, children):
        source_leaf = source_by_node.get(target_leaf.derived_from or "")
        if source_leaf is None and target_leaf.score_material_id is not None:
            source_leaf = next(
                (
                    candidate
                    for candidate in source_leaves
                    if candidate.score_material_id == target_leaf.score_material_id
                ),
                None,
            )
        if (
            source_leaf is not None
            and target_leaf.score_material_id is not None
            and source_leaf.score_material_id is not None
        ):
            pair = (target_leaf.score_material_id, source_leaf.score_material_id)
            if pair not in pairs:
                pairs.append(pair)
    return tuple(pairs)


def build_whole_score_context(plan: PiecePlan) -> WholeScoreContextV0:
    """PiecePlanから全曲生成に必要な固定素材文脈を作る。"""

    validate_piece_plan(plan)
    node_by_id = {node.node_id: node for node in plan.nodes}
    children = _children(plan)
    ordered_leaves = _leaves(node_by_id[plan.root_node_id], children)

    material_ids: list[str] = []
    weights: dict[str, list[int]] = {}
    for leaf in ordered_leaves:
        assert leaf.score_material_id is not None and leaf.duration_weight is not None
        if leaf.score_material_id not in weights:
            material_ids.append(leaf.score_material_id)
            weights[leaf.score_material_id] = []
        weights[leaf.score_material_id].append(leaf.duration_weight)
    if any(
        leaf.role == "transition" and len(weights[leaf.score_material_id]) != 1
        for leaf in ordered_leaves
    ):
        raise WholeScoreStagedError(
            "transition material must be dedicated to one leaf"
        )

    relations = tuple(
        MaterialRelationV0(
            node.node_id,
            node.derived_from,
            _material_pairs(node, node_by_id[node.derived_from], children),
        )
        for node in plan.nodes
        if node.derived_from is not None
    )

    transitions: list[TransitionContextV0] = []
    for index, leaf in enumerate(ordered_leaves):
        if leaf.role != "transition":
            continue
        if index == 0 or index + 1 == len(ordered_leaves):
            raise WholeScoreStagedError("transition must have previous and next material")
        assert leaf.score_material_id is not None
        previous = ordered_leaves[index - 1].score_material_id
        following = ordered_leaves[index + 1].score_material_id
        assert previous is not None and following is not None
        transitions.append(
            TransitionContextV0(
                leaf.node_id,
                leaf.score_material_id,
                ordered_leaves[index - 1].node_id,
                previous,
                ordered_leaves[index + 1].node_id,
                following,
            )
        )

    transition_ids = tuple(
        material_id
        for material_id in material_ids
        if any(item.material_id == material_id for item in transitions)
    )
    release_ids = tuple(
        material_id
        for material_id in material_ids
        if all(
            leaf.role == "release"
            for leaf in ordered_leaves
            if leaf.score_material_id == material_id
        )
    )
    return WholeScoreContextV0(
        tuple(material_ids),
        {material_id: tuple(values) for material_id, values in weights.items()},
        transition_ids,
        release_ids,
        relations,
        tuple(transitions),
    )


def melody_generation_order(
    context: WholeScoreContextV0,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """通常素材を先に、transition素材を後に処理する順序を返す。"""

    transition_ids = set(context.transition_material_ids)
    dependencies: dict[str, set[str]] = {item: set() for item in context.material_ids}
    for relation in context.relations:
        for target, source in relation.material_pairs:
            if target != source and target in dependencies and source in dependencies:
                dependencies[target].add(source)

    pending = [item for item in context.material_ids if item not in transition_ids]
    ordered: list[str] = []
    while pending:
        ready = next(
            (
                item
                for item in pending
                if dependencies[item].issubset(set(ordered) | transition_ids)
            ),
            None,
        )
        if ready is None:
            raise WholeScoreStagedError("material dependency order is cyclic")
        ordered.append(ready)
        pending.remove(ready)
    return tuple(ordered), context.transition_material_ids


def _derived_materials(context: WholeScoreContextV0) -> dict[str, str]:
    result: dict[str, str] = {}
    for relation in context.relations:
        for target, source in relation.material_pairs:
            if target == source:
                continue
            previous = result.setdefault(target, source)
            if previous != source:
                raise WholeScoreStagedError("material has multiple relation sources")
    return result


def assemble_whole_score_skeleton(
    case_id: str,
    plan: PiecePlan,
    drafts: tuple[tuple[int, HarmonicDraft], ...],
) -> HarmonicSkeletonV0:
    """IDを持たない素材別和声応答からHarmonicSkeletonV0を作る。"""

    if not case_id:
        raise WholeScoreStagedError("case ID must not be empty")
    context = build_whole_score_context(plan)
    if len(drafts) != len(context.material_ids):
        raise WholeScoreStagedError("harmonic draft count must match PiecePlan materials")
    derived = _derived_materials(context)
    materials: list[HarmonicMaterialV0] = []
    for material_index, (material_id, (length_units, draft)) in enumerate(
        zip(context.material_ids, drafts, strict=True),
        1,
    ):
        minimum_events = 1 if material_id in context.release_material_ids else 2
        if length_units <= 0 or not minimum_events <= len(draft.events) <= 8:
            raise WholeScoreStagedError("harmonic draft length or event count is invalid")
        cursor = 0
        harmonies: list[ScoreHarmony] = []
        for harmony_index, event in enumerate(draft.events, 1):
            if (
                event.at_units != cursor
                or event.duration_units <= 0
                or not 0 <= event.root_pitch_class <= 11
                or event.quality not in HARMONY_INTERVALS
            ):
                raise WholeScoreStagedError("harmonic draft coverage or vocabulary is invalid")
            cursor += event.duration_units
            harmonies.append(
                ScoreHarmony(
                    f"whole-{case_id}-h-{material_index:03d}-{harmony_index:03d}",
                    event.at_units,
                    event.duration_units,
                    event.root_pitch_class,
                    event.quality,
                )
            )
        if cursor != length_units:
            raise WholeScoreStagedError("harmonic draft must fill its material")
        materials.append(
            HarmonicMaterialV0(
                material_id,
                length_units,
                derived.get(material_id),
                tuple(harmonies),
            )
        )
    skeleton = HarmonicSkeletonV0(
        f"whole-score-{case_id}",
        WHOLE_SCORE_DIVISIONS,
        tuple(materials),
    )
    try:
        validate_harmonic_skeleton(plan, skeleton)
    except ValueError as error:
        raise WholeScoreStagedError(str(error)) from error
    return skeleton


def validate_material_identity_cues(
    context: WholeScoreContextV0,
    cues: tuple[MaterialIdentityCueV0, ...],
) -> dict[str, tuple[str, ...]]:
    """固定済み関係に対する同一性手掛かりを検査する。"""

    relations = {
        (relation.target_node_id, relation.source_node_id): set(relation.material_pairs)
        for relation in context.relations
    }
    result: dict[str, list[str]] = {}
    seen: set[tuple[str, str, str, str]] = set()
    for item in cues:
        key = (item.target_node_id, item.source_node_id)
        pair = (item.target_material_id, item.source_material_id)
        identity = (*key, *pair)
        if identity in seen:
            raise WholeScoreStagedError("identity cue relation is duplicated")
        seen.add(identity)
        if key not in relations or pair not in relations[key]:
            raise WholeScoreStagedError("identity cue relation is not declared by PiecePlan")
        if (
            not item.cues
            or len(set(item.cues)) != len(item.cues)
            or any(cue not in SUPPORTED_IDENTITY_CUES for cue in item.cues)
        ):
            raise WholeScoreStagedError("identity cue is empty, duplicated, or unsupported")
        selected = result.setdefault(item.target_node_id, [])
        for cue in item.cues:
            if cue not in selected:
                selected.append(cue)
    return {node_id: tuple(values) for node_id, values in result.items()}


def assemble_whole_score_melodies(
    case_id: str,
    plan: PiecePlan,
    skeleton: HarmonicSkeletonV0,
    drafts: tuple[MelodyDraft, ...],
) -> ScorePayloadV0:
    """素材IDを持たない旋律応答を全曲payloadへ結合する。"""

    validate_harmonic_skeleton(plan, skeleton)
    if len(drafts) != len(skeleton.materials):
        raise WholeScoreStagedError("melody draft count must match harmonic materials")
    placeholders = tuple(
        ScoreMaterial(
            item.material_id,
            item.length_units,
            (),
            derived_from=item.derived_from,
            harmonies=item.harmonies,
        )
        for item in skeleton.materials
    )
    score = ScoreSpec(skeleton.score_id, skeleton.divisions, placeholders)
    payloads: list[MaterialPayloadV0] = []
    for index, (material, draft) in enumerate(
        zip(skeleton.materials, drafts, strict=True),
        1,
    ):
        target = next(item for item in placeholders if item.material_id == material.material_id)
        try:
            notes = assemble_melody(
                f"{case_id}-{index:03d}",
                skeleton.divisions,
                target,
                material.harmonies,
                draft,
                score,
            )
        except ValueError as error:
            raise WholeScoreStagedError(str(error)) from error
        payloads.append(
            MaterialPayloadV0(
                material.material_id,
                notes,
                (),
                draft.foreground_voice,
            )
        )
    return ScorePayloadV0(tuple(payloads))


def assemble_whole_score_textures(
    case_id: str,
    plan: PiecePlan,
    skeleton: HarmonicSkeletonV0,
    melodies: ScorePayloadV0,
    drafts: tuple[TextureDraft, ...],
    *,
    maximum_event_counts: Mapping[str, int] | None = None,
    placement_policy: str = "strict-v2",
    allowed_pitch_range: tuple[int, int] | None = None,
) -> WholeScoreTextureResultV0:
    """旋律済み全曲payloadへ素材別伴奏を安全配置してScoreSpecを作る。"""

    if not case_id:
        raise WholeScoreStagedError("case ID must not be empty")
    try:
        score = apply_harmonic_skeleton(plan, melodies, skeleton)
    except ValueError as error:
        raise WholeScoreStagedError(str(error)) from error
    if len(drafts) != len(score.materials):
        raise WholeScoreStagedError("texture draft count must match score materials")
    placements: list[PianoTexturePlacementResult] = []
    spacing: list[int] = []
    for index, (material, draft) in enumerate(
        zip(score.materials, drafts, strict=True),
        1,
    ):
        foreground = tuple(
            note for note in material.notes if note.voice == material.foreground_voice
        )
        try:
            preplacement = texture_preplacement_violations(
                draft,
                material.harmonies,
                foreground,
                allowed_pitch_range=allowed_pitch_range,
            )
            if preplacement["event_feasibility"]:
                raise WholeScoreStagedError(
                    "texture draft violates event feasibility contract: "
                    f"{preplacement['event_feasibility']}"
                )
            if preplacement["onset_capacity"]:
                raise WholeScoreStagedError(
                    "texture draft exceeds onset capacity contract: "
                    f"{preplacement['onset_capacity']}"
                )
            placement, assembled = assemble_texture(
                f"{case_id}-{index:03d}",
                plan,
                score,
                material,
                material.harmonies,
                foreground,
                draft,
                maximum_event_count=(
                    maximum_event_counts.get(material.material_id)
                    if maximum_event_counts is not None
                    else None
                ),
                placement_policy=placement_policy,
                allowed_pitch_range=allowed_pitch_range,
            )
        except ValueError as error:
            raise WholeScoreStagedError(str(error)) from error
        if placement.status != "placed":
            raise WholeScoreStagedError(
                f"texture placement failed for {material.material_id}: {placement.reason}"
            )
        violation_count = full_low_spacing_violations(assembled)
        if violation_count:
            raise WholeScoreStagedError(
                f"texture low spacing failed for {material.material_id}"
            )
        score = ScoreSpec(
            score.score_id,
            score.divisions,
            tuple(
                assembled if item.material_id == material.material_id else item
                for item in score.materials
            ),
        )
        placements.append(placement)
        spacing.append(violation_count)
    try:
        validate_score_spec(plan, score)
    except ValueError as error:
        raise WholeScoreStagedError(str(error)) from error
    return WholeScoreTextureResultV0(score, tuple(placements), tuple(spacing))


def assemble_whole_score_performance(
    case_id: str,
    plan: PiecePlan,
    score: ScoreSpec,
    drafts: tuple[PerformanceOccurrenceDraftV0, ...],
) -> PerformanceSpec:
    """演奏順のIDなし応答を、180秒のPerformanceSpecへ結合する。"""

    if not case_id:
        raise WholeScoreStagedError("case ID must not be empty")
    validate_score_spec(plan, score)
    node_by_id = {node.node_id: node for node in plan.nodes}
    leaves = _leaves(node_by_id[plan.root_node_id], _children(plan))
    if len(drafts) != len(leaves):
        raise WholeScoreStagedError("performance draft count must match PiecePlan occurrences")
    performance = PerformanceSpec(
        f"whole-performance-{case_id}",
        WHOLE_SCORE_TARGET_DURATION_MS,
        64,
        "narrative-v2",
        tuple(
            NodePerformance(
                leaf.node_id,
                draft.timing_profile,
                draft.timing_amount,
                draft.dynamics_profile,
                draft.articulation_profile,
                draft.coordination_profile,
                draft.pedal_profile,
            )
            for leaf, draft in zip(leaves, drafts, strict=True)
        ),
    )
    try:
        validate_pipeline(plan, score, performance)
    except ValueError as error:
        raise WholeScoreStagedError(str(error)) from error
    return performance


def _foreground_line(material: ScoreMaterial) -> tuple[tuple[int, int], ...]:
    voice = material.foreground_voice
    if voice not in {"upper", "lower"}:
        raise WholeScoreStagedError("transition material has no foreground voice")
    grouped: dict[int, list[int]] = defaultdict(list)
    for note in material.notes:
        if note.voice == voice:
            grouped[note.at_units].append(note.pitch)
    if not grouped:
        raise WholeScoreStagedError("transition comparison needs foreground notes")
    select = max if voice == "upper" else min
    return tuple((onset, select(grouped[onset])) for onset in sorted(grouped))


def analyze_foreground_transition(
    plan: PiecePlan,
    score: ScoreSpec,
    transition: TransitionContextV0,
    *,
    minimum_transition_attacks: int,
    maximum_boundary_leap: int,
    maximum_internal_leap: int,
) -> ForegroundTransitionAssessmentV0:
    """素材ごとの前景声部を使い、接続句の境界と内部輪郭を診断する。"""

    validate_piece_plan(plan)
    validate_score_spec(plan, score)
    if minimum_transition_attacks < 2 or min(maximum_boundary_leap, maximum_internal_leap) < 0:
        raise WholeScoreStagedError("transition thresholds are invalid")
    nodes = {node.node_id: node for node in plan.nodes}
    expected = {
        transition.previous_node_id: transition.previous_material_id,
        transition.node_id: transition.material_id,
        transition.next_node_id: transition.next_material_id,
    }
    if any(
        node_id not in nodes or nodes[node_id].score_material_id != material_id
        for node_id, material_id in expected.items()
    ):
        raise WholeScoreStagedError("transition context does not match PiecePlan")
    materials = {material.material_id: material for material in score.materials}
    try:
        source = materials[transition.previous_material_id]
        middle = materials[transition.material_id]
        target = materials[transition.next_material_id]
    except KeyError as error:
        raise WholeScoreStagedError("transition material does not exist in score") from error
    source_line = _foreground_line(source)
    middle_line = _foreground_line(middle)
    target_line = _foreground_line(target)
    source_leap = abs(source_line[-1][1] - middle_line[0][1])
    internal_leap = max(
        (abs(right[1] - left[1]) for left, right in pairwise(middle_line)),
        default=0,
    )
    target_leap = abs(middle_line[-1][1] - target_line[0][1])
    restrike = middle_line[-1][1] == target_line[0][1]
    passes = (
        len(middle_line) >= minimum_transition_attacks
        and source_leap <= maximum_boundary_leap
        and internal_leap <= maximum_internal_leap
        and target_leap <= maximum_boundary_leap
        and not restrike
    )
    assert source.foreground_voice is not None
    assert middle.foreground_voice is not None
    assert target.foreground_voice is not None
    return ForegroundTransitionAssessmentV0(
        transition.previous_node_id,
        transition.node_id,
        transition.next_node_id,
        source.foreground_voice,
        middle.foreground_voice,
        target.foreground_voice,
        len(middle_line),
        source_leap,
        internal_leap,
        target_leap,
        restrike,
        passes,
    )
