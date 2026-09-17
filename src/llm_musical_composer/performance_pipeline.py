"""楽曲計画、譜面、演奏指定を分離して MusicXML と SMF を生成する。"""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import mido

TICKS_PER_BEAT = 500
TEMPO = 500_000
HARMONY_INTERVALS: dict[str, tuple[int, ...]] = {
    "major": (0, 4, 7),
    "minor": (0, 3, 7),
    "diminished": (0, 3, 6),
    "major-seventh": (0, 4, 7, 11),
}


class PipelineValidationError(ValueError):
    """多段 IR の契約に違反した入力を表す。"""


def harmony_pitch_classes(root_pitch_class: int, quality: str) -> frozenset[int]:
    """宣言済み和声品質を音高クラス集合へ変換する。"""
    try:
        intervals = HARMONY_INTERVALS[quality]
    except KeyError as error:
        raise PipelineValidationError("score harmony quality is invalid") from error
    return frozenset((root_pitch_class + interval) % 12 for interval in intervals)


@dataclass(frozen=True)
class PlanNode:
    node_id: str
    parent_id: str | None
    order: int
    role: str
    derived_from: str | None = None
    contrasts_with: str | None = None
    harmonic_focus: int | None = None
    duration_weight: int | None = None
    score_material_id: str | None = None


@dataclass(frozen=True)
class PiecePlan:
    plan_id: str
    title: str
    tonal_center: int
    mode: str
    root_node_id: str
    ending_intent: str
    nodes: tuple[PlanNode, ...]


@dataclass(frozen=True)
class ScoreNote:
    event_id: str
    at_units: int
    duration_units: int
    pitch: int
    voice: str
    tie: str | None = None
    articulations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScoreDirection:
    direction_id: str
    at_units: int
    kind: str
    value: str


@dataclass(frozen=True)
class ScoreHarmony:
    harmony_id: str
    at_units: int
    duration_units: int
    root_pitch_class: int
    quality: str


@dataclass(frozen=True)
class ScoreMaterial:
    material_id: str
    length_units: int
    notes: tuple[ScoreNote, ...]
    derived_from: str | None = None
    directions: tuple[ScoreDirection, ...] = ()
    harmonies: tuple[ScoreHarmony, ...] = ()
    foreground_voice: str | None = None


@dataclass(frozen=True)
class ScoreSpec:
    score_id: str
    divisions: int
    materials: tuple[ScoreMaterial, ...]


@dataclass(frozen=True)
class NodePerformance:
    node_id: str
    timing_profile: str | None = None
    timing_amount: str | None = None
    dynamics_profile: str | None = None
    articulation_profile: str | None = None
    coordination_profile: str | None = None
    pedal_profile: str | None = None


@dataclass(frozen=True)
class PerformanceSpec:
    performance_id: str
    target_duration_ms: int
    default_velocity: int
    timing_budget_id: str
    node_performances: tuple[NodePerformance, ...]
    key_release_percent: int = 100
    velocity_policy_id: str = "legacy-unison-v1"


@dataclass(frozen=True)
class PerformedNote:
    event_id: str
    occurrence_node_id: str
    at_ms: int
    duration_ms: int
    pitch: int
    velocity: int
    voice: str


@dataclass(frozen=True)
class PerformedPedal:
    event_id: str
    occurrence_node_id: str
    at_ms: int
    value: int


@dataclass(frozen=True)
class RenderedHarmony:
    occurrence_node_id: str
    harmony_id: str
    start_ms: int
    end_ms: int
    root_pitch_class: int
    quality: str
    accompaniment_voice: str


@dataclass(frozen=True)
class RenderedPerformance:
    performance_id: str
    title: str
    duration_ms: int
    notes: tuple[PerformedNote, ...]
    pedals: tuple[PerformedPedal, ...]
    harmonies: tuple[RenderedHarmony, ...]
    node_intervals: tuple[tuple[str, int, int], ...]
    lineage: tuple[str, str, str]


@dataclass(frozen=True)
class PerformanceRenderResult:
    path: Path
    duration_ms: int
    note_count: int


def _fail(message: str) -> None:
    raise PipelineValidationError(message)


def _plan_indexes(
    plan: PiecePlan,
) -> tuple[dict[str, PlanNode], dict[str, list[PlanNode]], dict[str, int]]:
    by_id: dict[str, PlanNode] = {}
    children: dict[str, list[PlanNode]] = defaultdict(list)
    depths: dict[str, int] = {}
    for node in plan.nodes:
        if not node.node_id or node.node_id in by_id:
            _fail("plan node ids must be non-empty and unique")
        if node.parent_id is None:
            depth = 0
        elif node.parent_id not in by_id:
            _fail("parent nodes must appear before their children")
        else:
            depth = depths[node.parent_id] + 1
            children[node.parent_id].append(node)
        by_id[node.node_id] = node
        depths[node.node_id] = depth
    return by_id, children, depths


def _validate_score_materials(
    score: ScoreSpec, *, require_harmony_foreground_voice: bool = True
) -> set[str]:
    """PiecePlanの全素材が揃う前でも、ScoreSpec内部の意味契約を検査する。"""

    if not score.score_id or score.divisions <= 0 or not score.materials:
        _fail("score metadata is invalid")
    material_ids: set[str] = set()
    score_event_ids: set[str] = set()
    harmony_ids: set[str] = set()
    for material in score.materials:
        if not material.material_id or material.material_id in material_ids:
            _fail("score material ids must be non-empty and unique")
        material_ids.add(material.material_id)
        if material.derived_from is not None and material.derived_from not in material_ids:
            _fail("derived_from must reference an earlier score material")
        if material.length_units <= 0:
            _fail("score material length must be positive")
        if require_harmony_foreground_voice:
            if bool(material.harmonies) != (material.foreground_voice is not None):
                _fail("score harmony and foreground voice must be declared together")
        elif material.foreground_voice is not None:
            _fail("role-neutral score material must not declare a foreground voice")
        if material.foreground_voice not in {None, "upper", "lower"}:
            _fail("score harmony foreground voice is invalid")
        harmony_cursor = 0
        for harmony in material.harmonies:
            if not harmony.harmony_id or harmony.harmony_id in harmony_ids:
                _fail("score harmony ids must be non-empty and unique")
            harmony_ids.add(harmony.harmony_id)
            if (
                harmony.at_units != harmony_cursor
                or harmony.duration_units <= 0
                or not 0 <= harmony.root_pitch_class <= 11
                or harmony.quality not in HARMONY_INTERVALS
            ):
                _fail("score harmony coverage or vocabulary is invalid")
            harmony_cursor += harmony.duration_units
        if material.harmonies and harmony_cursor != material.length_units:
            _fail("score harmony coverage must fill the material")
        for note in material.notes:
            if not note.event_id or note.event_id in score_event_ids:
                _fail("score event ids must be non-empty and unique")
            score_event_ids.add(note.event_id)
            if (
                note.at_units < 0
                or note.duration_units <= 0
                or note.at_units + note.duration_units > material.length_units
                or not 21 <= note.pitch <= 108
                or note.voice not in {"upper", "lower"}
                or note.tie not in {None, "start", "stop", "continue"}
            ):
                _fail("score note is outside the supported piano range or contract")
            if len(set(note.articulations)) != len(note.articulations) or any(
                item not in {"normal", "staccato", "tenuto", "accent"}
                for item in note.articulations
            ):
                _fail("score articulation is outside the supported vocabulary")
        grouped: dict[tuple[int, str], list[ScoreNote]] = defaultdict(list)
        for note in material.notes:
            grouped[(note.pitch, note.voice)].append(note)
        for same_pitch in grouped.values():
            ordered = sorted(same_pitch, key=lambda item: item.at_units)
            if any(
                current.at_units < previous.at_units + previous.duration_units
                for previous, current in pairwise(ordered)
            ):
                _fail("the same pitch cannot overlap within one score material")
        direction_ids: set[str] = set()
        for direction in material.directions:
            if (
                not direction.direction_id
                or direction.direction_id in direction_ids
                or not 0 <= direction.at_units <= material.length_units
                or direction.kind not in {"dynamic", "breath"}
                or (
                    direction.kind == "dynamic"
                    and direction.value not in {"pp", "p", "mp", "mf", "f", "ff"}
                )
                or (direction.kind == "breath" and direction.value not in {"light", "full"})
            ):
                _fail("score direction is invalid")
            direction_ids.add(direction.direction_id)
    return material_ids


def _validate_performance_spec(nodes: dict[str, PlanNode], performance: PerformanceSpec) -> None:
    if (
        not performance.performance_id
        or performance.target_duration_ms <= 0
        or not 1 <= performance.default_velocity <= 127
        or performance.timing_budget_id not in {"subtle-v1", "narrative-v1", "narrative-v2"}
        or performance.velocity_policy_id
        not in {
            "legacy-unison-v1",
            "foreground-accompaniment-harmony-shape-v1",
        }
    ):
        if performance.velocity_policy_id not in {
            "legacy-unison-v1",
            "foreground-accompaniment-harmony-shape-v1",
        }:
            _fail("performance velocity policy is invalid")
        _fail("performance metadata is invalid")
    if not 40 <= performance.key_release_percent <= 100:
        _fail("performance key release percent is outside the supported range")
    seen_performance_nodes: set[str] = set()
    for item in performance.node_performances:
        if item.node_id not in nodes or item.node_id in seen_performance_nodes:
            _fail("performance node references must be known and unique")
        seen_performance_nodes.add(item.node_id)
        if (
            item.timing_profile not in {None, "neutral", "savor", "flow", "build", "release"}
            or item.timing_amount not in {None, "subtle", "moderate"}
            or item.dynamics_profile not in {None, "steady", "shape", "build", "release"}
            or item.articulation_profile not in {None, "score", "legato", "light"}
            or item.coordination_profile not in {None, "score", "rolled", "aligned"}
            or item.pedal_profile not in {None, "none", "phrase_legato", "harmony_legato", "clear"}
            or (item.timing_profile is None and item.timing_amount is not None)
        ):
            _fail("performance profile is outside the supported vocabulary")


def _validate_pipeline_stages(
    plan: PiecePlan,
    score: ScoreSpec | None,
    performance: PerformanceSpec | None,
    *,
    require_harmony_foreground_voice: bool = True,
) -> None:
    """多段 IR 間の参照、木構造、譜面イベント、演奏指定を検証する。"""
    roles = {
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
    if (
        not plan.plan_id
        or not plan.title
        or plan.mode not in {"major", "minor"}
        or plan.ending_intent != "tonic"
    ):
        _fail("plan metadata or ending intent is invalid")
    if not 0 <= plan.tonal_center <= 11 or not plan.nodes:
        _fail("tonal center or plan nodes are invalid")
    nodes, children, depths = _plan_indexes(plan)
    roots = [node for node in plan.nodes if node.parent_id is None]
    if len(roots) != 1 or roots[0].node_id != plan.root_node_id:
        _fail("plan must contain exactly the declared root")
    for parent_id, items in children.items():
        orders = sorted(item.order for item in items)
        if orders != list(range(len(items))):
            _fail(f"children of {parent_id} must have contiguous order values")
    node_positions = {node.node_id: index for index, node in enumerate(plan.nodes)}
    for node in plan.nodes:
        is_leaf = not children[node.node_id]
        has_leaf_payload = node.duration_weight is not None and node.score_material_id is not None
        if is_leaf != has_leaf_payload or (is_leaf and node.duration_weight <= 0):
            _fail("only leaves must define a positive duration and score material")
        if node.role not in roles:
            _fail("plan node role is outside the supported vocabulary")
        if node.harmonic_focus is not None and not 0 <= node.harmonic_focus <= 11:
            _fail("harmonic focus must be a pitch class")
        if node.role == "return" and node.derived_from is None:
            _fail("a return node must derive from an earlier node")
        if node.derived_from is not None:
            if node.derived_from not in nodes:
                _fail("derived_from must reference an earlier plan node")
            if node_positions[node.derived_from] >= node_positions[node.node_id]:
                _fail("derived_from must reference an earlier plan node")
            if depths[node.derived_from] != depths[node.node_id]:
                _fail("derived_from must reference a node at the same depth")
        if node.contrasts_with is not None:
            if node.contrasts_with not in nodes:
                _fail("contrast target must reference an earlier plan node")
            if node_positions[node.contrasts_with] >= node_positions[node.node_id]:
                _fail("contrast target must reference an earlier plan node")
            if depths[node.contrasts_with] != depths[node.node_id]:
                _fail("contrast target must reference a node at the same depth")

    if score is None:
        if performance is not None:
            _validate_performance_spec(nodes, performance)
        return
    material_ids = _validate_score_materials(
        score,
        require_harmony_foreground_voice=require_harmony_foreground_voice,
    )
    if any(
        node.score_material_id not in material_ids
        for node in plan.nodes
        if node.score_material_id is not None
    ):
        _fail("plan leaf references an unknown score material")
    materials_by_id = {material.material_id: material for material in score.materials}
    if any(
        not materials_by_id[node.score_material_id].notes
        for node in plan.nodes
        if node.score_material_id is not None
    ):
        _fail("referenced score material must contain at least one note")

    if performance is not None:
        _validate_performance_spec(nodes, performance)


def validate_piece_plan(plan: PiecePlan) -> None:
    """後続IRがなくてもPiecePlanの意味契約を検査する。"""

    _validate_pipeline_stages(plan, None, None)


def validate_score_materials(score: ScoreSpec) -> None:
    """PiecePlanの全素材が揃う前にScoreSpec内部だけを検査する。"""

    _validate_score_materials(score)


def validate_score_spec(plan: PiecePlan, score: ScoreSpec) -> None:
    """PerformanceSpecがなくてもPiecePlanとScoreSpecの意味契約を検査する。"""

    _validate_pipeline_stages(plan, score, None)


def validate_performance_spec(plan: PiecePlan, performance: PerformanceSpec) -> None:
    """ScoreSpecがなくてもPiecePlanとPerformanceSpecの意味契約を検査する。"""

    _validate_pipeline_stages(plan, None, performance)


def validate_pipeline(plan: PiecePlan, score: ScoreSpec, performance: PerformanceSpec) -> None:
    """多段IR間の参照、木構造、譜面イベント、演奏指定を検証する。"""

    _validate_pipeline_stages(plan, score, performance)


def ordered_leaf_schedule(
    plan: PiecePlan, score: ScoreSpec
) -> tuple[list[tuple[PlanNode, int, int]], dict[str, tuple[int, int]]]:
    """構成木を演奏順の葉と楽譜時間区間へ展開する。"""
    nodes, children, _ = _plan_indexes(plan)
    materials = {material.material_id: material for material in score.materials}
    leaves: list[tuple[PlanNode, int, int]] = []
    intervals: dict[str, tuple[int, int]] = {}
    cursor = 0

    def visit(node: PlanNode) -> tuple[int, int]:
        nonlocal cursor
        start = cursor
        descendants = sorted(children[node.node_id], key=lambda item: item.order)
        if descendants:
            for child in descendants:
                visit(child)
        else:
            assert node.score_material_id is not None
            assert node.duration_weight is not None
            cursor += materials[node.score_material_id].length_units
            leaves.append((node, start, cursor))
        intervals[node.node_id] = (start, cursor)
        return start, cursor

    visit(nodes[plan.root_node_id])
    return leaves, intervals


def _profile_deviation(
    profile: str,
    amount: str,
    position: float,
    timing_budget_id: str,
) -> float:
    if timing_budget_id in {"narrative-v1", "narrative-v2"}:
        amplitude = 0.10 if amount == "subtle" else 0.18
        if profile == "savor" and timing_budget_id == "narrative-v1":
            return amplitude
        if profile == "flow":
            return -amplitude
    else:
        amplitude = 0.05 if amount == "subtle" else 0.09
    if profile == "savor" and timing_budget_id == "narrative-v2":
        control_points = (
            (0.00, 1.00),
            (0.22, 0.80),
            (0.30, -1.00),
            (0.36, 0.20),
            (0.62, 1.00),
            (0.69, -0.80),
            (0.76, 0.20),
            (1.00, 1.00),
        )
        for (left_x, left_y), (right_x, right_y) in pairwise(control_points):
            if left_x <= position <= right_x:
                local = (position - left_x) / (right_x - left_x)
                return amplitude * (left_y + (right_y - left_y) * local)
        return amplitude * control_points[-1][1]
    if profile == "savor":
        return amplitude * math.cos(2.0 * math.pi * position)
    if profile == "flow":
        return -0.5 * amplitude
    if profile == "build":
        return amplitude * (0.5 - position)
    if profile == "release":
        return amplitude * (position - 0.5)
    return 0.0


def _time_map(
    total_units: int,
    intervals: dict[str, tuple[int, int]],
    performance: PerformanceSpec,
) -> list[int]:
    factors: list[float] = []
    for unit in range(total_units):
        deviations: list[float] = []
        midpoint = unit + 0.5
        for item in performance.node_performances:
            if item.timing_profile is None:
                continue
            start, end = intervals[item.node_id]
            if start <= midpoint < end:
                position = (midpoint - start) / (end - start)
                deviations.append(
                    _profile_deviation(
                        item.timing_profile,
                        item.timing_amount or "subtle",
                        position,
                        performance.timing_budget_id,
                    )
                )
        factors.append(max(0.5, 1.0 + (sum(deviations) / len(deviations) if deviations else 0.0)))
    scale = performance.target_duration_ms / sum(factors)
    mapped = [0]
    elapsed = 0.0
    for factor in factors:
        elapsed += factor * scale
        mapped.append(round(elapsed))
    mapped[-1] = performance.target_duration_ms
    return mapped


def resolve_effective_profile(
    node: PlanNode,
    node_by_id: dict[str, PlanNode],
    performance_by_node: dict[str, NodePerformance],
    attribute: str,
    default: str,
) -> tuple[str | None, str]:
    """葉に最も近い明示指定を祖先方向へ解決する。"""
    current: PlanNode | None = node
    while current is not None:
        if current.node_id in performance_by_node:
            value = getattr(performance_by_node[current.node_id], attribute)
            if value is not None:
                return current.node_id, value
        current = node_by_id.get(current.parent_id) if current.parent_id is not None else None
    return None, default


def dynamic_velocity_unclamped(
    material: ScoreMaterial, note: ScoreNote, default: int, profile: str
) -> int:
    """direction、profile、accent適用後かつMIDI範囲へ収める前の値を返す。"""
    values = {"pp": 40, "p": 48, "mp": 56, "mf": 68, "f": 80, "ff": 92}
    dynamic = default
    for direction in sorted(material.directions, key=lambda item: item.at_units):
        if direction.kind == "dynamic" and direction.at_units <= note.at_units:
            dynamic = values.get(direction.value, default)
    position = note.at_units / material.length_units
    if profile == "shape":
        dynamic += round(4 * math.sin(math.pi * position))
    elif profile == "build":
        dynamic += round(8 * position)
    elif profile == "release":
        dynamic += round(8 * (1.0 - position))
    if "accent" in note.articulations:
        dynamic += 8
    return dynamic


def _dynamic_velocity(material: ScoreMaterial, note: ScoreNote, default: int, profile: str) -> int:
    return min(127, max(1, dynamic_velocity_unclamped(material, note, default, profile)))


def material_velocity_adjustments(
    material: ScoreMaterial,
    velocity_policy_id: str,
) -> dict[str, int]:
    """声部役割と和声内の打鍵順から、音符ごとの相対velocityを決める。"""

    adjustments = {note.event_id: 0 for note in material.notes}
    if (
        velocity_policy_id != "foreground-accompaniment-harmony-shape-v1"
        or material.foreground_voice is None
        or not material.harmonies
    ):
        return adjustments
    accompaniment_voice = "lower" if material.foreground_voice == "upper" else "upper"
    for harmony in material.harmonies:
        onsets = sorted(
            {
                note.at_units
                for note in material.notes
                if note.voice == accompaniment_voice
                and harmony.at_units <= note.at_units < harmony.at_units + harmony.duration_units
            }
        )
        onset_adjustments: dict[int, int] = {}
        for index, onset in enumerate(onsets):
            position = index / (len(onsets) - 1) if len(onsets) > 1 else 0.0
            onset_adjustments[onset] = -6 + round(4 * math.sin(math.pi * position))
        for note in material.notes:
            if note.voice == accompaniment_voice and note.at_units in onset_adjustments:
                adjustments[note.event_id] = onset_adjustments[note.at_units]
    return adjustments


def _gate_ratio(note: ScoreNote, profile: str) -> float:
    ratio = 0.9
    if "tenuto" in note.articulations:
        ratio = 0.98
    if "staccato" in note.articulations:
        ratio = 0.55
    if profile == "legato":
        return max(ratio, 0.96)
    if profile == "light":
        return min(ratio, 0.78)
    return ratio


def _performance_lineage_source(performance: PerformanceSpec) -> str:
    """旧100%成果物のlineage文字列表現を維持する。"""

    if performance.velocity_policy_id != "legacy-unison-v1":
        return repr(performance)
    fields = (
        "PerformanceSpec("
        f"performance_id={performance.performance_id!r}, "
        f"target_duration_ms={performance.target_duration_ms!r}, "
        f"default_velocity={performance.default_velocity!r}, "
        f"timing_budget_id={performance.timing_budget_id!r}, "
        f"node_performances={performance.node_performances!r}"
    )
    if performance.key_release_percent != 100:
        fields += f", key_release_percent={performance.key_release_percent!r}"
    return fields + ")"


def _coordination_offsets(material: ScoreMaterial, profile: str) -> dict[str, int]:
    if profile != "rolled":
        return {note.event_id: 0 for note in material.notes}
    by_onset: dict[int, list[ScoreNote]] = defaultdict(list)
    for note in material.notes:
        by_onset[note.at_units].append(note)
    offsets: dict[str, int] = {}
    for notes in by_onset.values():
        ordered = sorted(notes, key=lambda item: (item.voice == "upper", item.pitch))
        divisor = max(1, len(ordered) - 1)
        for index, note in enumerate(ordered):
            offsets[note.event_id] = round(45 * index / divisor)
    return offsets


def performance_time_map(
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
) -> tuple[int, ...]:
    """局所打鍵差を加える前の楽譜単位ごとの演奏時刻を診断用に返す。"""
    validate_pipeline(plan, score, performance)
    _, intervals = ordered_leaf_schedule(plan, score)
    total_units = intervals[plan.root_node_id][1]
    return tuple(_time_map(total_units, intervals, performance))


def render_performance(
    plan: PiecePlan, score: ScoreSpec, performance: PerformanceSpec
) -> RenderedPerformance:
    """譜面上の同時性を保ったまま、階層的な演奏指定を絶対時刻へ写像する。"""
    validate_pipeline(plan, score, performance)
    return _render_performance_unchecked(plan, score, performance)


def render_role_neutral_performance(
    plan: PiecePlan, score: ScoreSpec, performance: PerformanceSpec
) -> RenderedPerformance:
    """声部を前景または伴奏へ読み替えず、検証済みの譜面を演奏へ写像する。"""
    _validate_pipeline_stages(
        plan,
        score,
        performance,
        require_harmony_foreground_voice=False,
    )
    return _render_performance_unchecked(plan, score, performance)


def render_role_neutral_performance_with_pedal_sources(
    plan: PiecePlan, score: ScoreSpec, performance: PerformanceSpec
) -> tuple[RenderedPerformance, dict[str, str]]:
    """Render without role inference and return each pedal event's originating leaf."""
    _validate_pipeline_stages(
        plan,
        score,
        performance,
        require_harmony_foreground_voice=False,
    )
    return _render_performance_unchecked_with_pedal_sources(plan, score, performance)


def _render_performance_unchecked(
    plan: PiecePlan, score: ScoreSpec, performance: PerformanceSpec
) -> RenderedPerformance:
    return _render_performance_unchecked_with_pedal_sources(plan, score, performance)[0]


def _render_performance_unchecked_with_pedal_sources(
    plan: PiecePlan, score: ScoreSpec, performance: PerformanceSpec
) -> tuple[RenderedPerformance, dict[str, str]]:
    leaves, intervals = ordered_leaf_schedule(plan, score)
    materials = {material.material_id: material for material in score.materials}
    node_by_id = {node.node_id: node for node in plan.nodes}
    performance_by_node = {item.node_id: item for item in performance.node_performances}
    total_units = intervals[plan.root_node_id][1]
    time_map = _time_map(total_units, intervals, performance)
    final_leaf_id = leaves[-1][0].node_id
    notes: list[PerformedNote] = []
    rendered_harmonies: list[RenderedHarmony] = []
    for leaf, start, _ in leaves:
        assert leaf.score_material_id is not None
        material = materials[leaf.score_material_id]
        _, articulation = resolve_effective_profile(
            leaf,
            node_by_id,
            performance_by_node,
            "articulation_profile",
            "score",
        )
        _, dynamics = resolve_effective_profile(
            leaf,
            node_by_id,
            performance_by_node,
            "dynamics_profile",
            "steady",
        )
        _, coordination = resolve_effective_profile(
            leaf,
            node_by_id,
            performance_by_node,
            "coordination_profile",
            "score",
        )
        coordination_offsets = _coordination_offsets(material, coordination)
        velocity_adjustments = material_velocity_adjustments(
            material, performance.velocity_policy_id
        )
        final_onset = max(note.at_units for note in material.notes)
        if material.foreground_voice is not None:
            accompaniment_voice = "lower" if material.foreground_voice == "upper" else "upper"
            for harmony in material.harmonies:
                rendered_harmonies.append(
                    RenderedHarmony(
                        occurrence_node_id=leaf.node_id,
                        harmony_id=harmony.harmony_id,
                        start_ms=time_map[start + harmony.at_units],
                        end_ms=time_map[start + harmony.at_units + harmony.duration_units],
                        root_pitch_class=harmony.root_pitch_class,
                        quality=harmony.quality,
                        accompaniment_voice=accompaniment_voice,
                    )
                )
        for note in material.notes:
            onset_unit = start + note.at_units
            ending_unit = onset_unit + note.duration_units
            onset = time_map[onset_unit] + coordination_offsets[note.event_id]
            mapped_duration = time_map[ending_unit] - onset
            preserves_terminal_release = (
                leaf.node_id == final_leaf_id
                and note.at_units == final_onset
                and note.at_units + note.duration_units == material.length_units
            )
            release_factor = (
                1.0 if preserves_terminal_release else performance.key_release_percent / 100
            )
            notes.append(
                PerformedNote(
                    event_id=f"{leaf.node_id}:{note.event_id}",
                    occurrence_node_id=leaf.node_id,
                    at_ms=onset,
                    duration_ms=max(
                        1,
                        round(mapped_duration * _gate_ratio(note, articulation) * release_factor),
                    ),
                    pitch=note.pitch,
                    velocity=min(
                        127,
                        max(
                            1,
                            dynamic_velocity_unclamped(
                                material,
                                note,
                                performance.default_velocity,
                                dynamics,
                            )
                            + velocity_adjustments[note.event_id],
                        ),
                    ),
                    voice=note.voice,
                )
            )
    pedal_segments: list[tuple[str | None, str, int, int, ScoreMaterial, str]] = []
    for leaf, start_unit, end_unit in leaves:
        owner, profile = resolve_effective_profile(
            leaf,
            node_by_id,
            performance_by_node,
            "pedal_profile",
            "none",
        )
        if (
            pedal_segments
            and profile != "harmony_legato"
            and pedal_segments[-1][0:2] == (owner, profile)
            and pedal_segments[-1][3] == start_unit
        ):
            previous = pedal_segments[-1]
            pedal_segments[-1] = (*previous[:3], end_unit, *previous[4:])
        else:
            assert leaf.score_material_id is not None
            pedal_segments.append(
                (
                    owner,
                    profile,
                    start_unit,
                    end_unit,
                    materials[leaf.score_material_id],
                    leaf.node_id,
                )
            )
    pedals: list[PerformedPedal] = []
    pedal_source_node_ids: dict[str, str] = {}
    for index, (_, profile, start_unit, end_unit, material, source_node_id) in enumerate(
        pedal_segments
    ):
        if profile == "none":
            continue
        start_ms, end_ms = time_map[start_unit], time_map[end_unit]
        owner = pedal_segments[index][0]
        assert owner is not None
        if profile == "harmony_legato":
            if not material.harmonies:
                _fail("harmony_legato requires score harmony")
            for harmony_index, harmony in enumerate(material.harmonies):
                harmony_start_unit = start_unit + harmony.at_units
                harmony_end_unit = harmony_start_unit + harmony.duration_units
                harmony_start_ms = time_map[harmony_start_unit]
                harmony_end_ms = time_map[harmony_end_unit]
                down_ms = (
                    harmony_start_ms
                    if harmony_index == 0
                    else min(harmony_end_ms - 1, harmony_start_ms + 80)
                )
                down = PerformedPedal(
                    f"pedal-{index}-{harmony_index}-down",
                    owner,
                    down_ms,
                    127,
                )
                pedals.append(down)
                pedal_source_node_ids[down.event_id] = source_node_id
                release_ms = harmony_end_ms
                up = PerformedPedal(
                    f"pedal-{index}-{harmony_index}-up",
                    owner,
                    release_ms,
                    0,
                )
                pedals.append(up)
                pedal_source_node_ids[up.event_id] = source_node_id
            continue
        down = PerformedPedal(f"pedal-{index}-down", owner, start_ms, 127)
        pedals.append(down)
        pedal_source_node_ids[down.event_id] = source_node_id
        release = max(start_ms, end_ms - 200)
        up = PerformedPedal(f"pedal-{index}-up", owner, release, 0)
        pedals.append(up)
        pedal_source_node_ids[up.event_id] = source_node_id
    final_up = PerformedPedal(
        "pedal-final-up",
        plan.root_node_id,
        performance.target_duration_ms,
        0,
    )
    pedals.append(final_up)
    pedal_source_node_ids[final_up.event_id] = final_leaf_id
    return (
        RenderedPerformance(
            performance_id=performance.performance_id,
            title=plan.title,
            duration_ms=performance.target_duration_ms,
            notes=tuple(sorted(notes, key=lambda item: (item.at_ms, item.voice, item.pitch))),
            pedals=tuple(sorted(pedals, key=lambda item: (item.at_ms, item.value, item.event_id))),
            harmonies=tuple(rendered_harmonies),
            node_intervals=tuple(
                (
                    node.node_id,
                    time_map[intervals[node.node_id][0]],
                    time_map[intervals[node.node_id][1]],
                )
                for node in plan.nodes
            ),
            lineage=(
                hashlib.sha256(repr(plan).encode("utf-8")).hexdigest(),
                hashlib.sha256(repr(score).encode("utf-8")).hexdigest(),
                hashlib.sha256(
                    _performance_lineage_source(performance).encode("utf-8")
                ).hexdigest(),
            ),
        ),
        pedal_source_node_ids,
    )


def _velocity_summary(values: list[int]) -> dict[str, object]:
    counts = Counter(values)
    ordered_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "note_count": len(values),
        "distinct_velocity_count": len(counts),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "most_common": [
            {"velocity": velocity, "count": count} for velocity, count in ordered_counts[:10]
        ],
    }


def _rounded_mean(values: list[int]) -> int:
    return math.floor(sum(values) / len(values) + 0.5)


def analyze_voice_velocity(
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
) -> dict[str, object]:
    """出現と和声境界を保って、前景・伴奏のvelocityを診断する。"""

    rendered = render_performance(plan, score, performance)
    leaves, _ = ordered_leaf_schedule(plan, score)
    materials = {material.material_id: material for material in score.materials}
    performed = {note.event_id: note for note in rendered.notes}
    foreground_values: list[int] = []
    accompaniment_values: list[int] = []
    adjacent_equal = 0
    adjacent_total = 0
    longest_equal_run = 0
    group_count = 0
    eligible_harmony_count = 0
    varied_eligible_harmony_count = 0
    shared_differences: list[int] = []

    for leaf, _, _ in leaves:
        assert leaf.score_material_id is not None
        material = materials[leaf.score_material_id]
        if material.foreground_voice is None or not material.harmonies:
            continue
        accompaniment_voice = "lower" if material.foreground_voice == "upper" else "upper"
        by_voice_and_onset: dict[tuple[str, int], list[int]] = defaultdict(list)
        for note in material.notes:
            value = performed[f"{leaf.node_id}:{note.event_id}"].velocity
            by_voice_and_onset[(note.voice, note.at_units)].append(value)
            if note.voice == material.foreground_voice:
                foreground_values.append(value)
            elif note.voice == accompaniment_voice:
                accompaniment_values.append(value)
        foreground_onsets = {
            onset for voice, onset in by_voice_and_onset if voice == material.foreground_voice
        }
        accompaniment_onsets = {
            onset for voice, onset in by_voice_and_onset if voice == accompaniment_voice
        }
        for onset in sorted(foreground_onsets & accompaniment_onsets):
            foreground_value = _rounded_mean(by_voice_and_onset[(material.foreground_voice, onset)])
            accompaniment_value = _rounded_mean(by_voice_and_onset[(accompaniment_voice, onset)])
            shared_differences.append(foreground_value - accompaniment_value)

        for harmony in material.harmonies:
            representatives = [
                _rounded_mean(by_voice_and_onset[(accompaniment_voice, onset)])
                for onset in sorted(accompaniment_onsets)
                if harmony.at_units <= onset < harmony.at_units + harmony.duration_units
            ]
            group_count += len(representatives)
            if len(representatives) >= 3:
                eligible_harmony_count += 1
                if len(set(representatives)) >= 2:
                    varied_eligible_harmony_count += 1
            if representatives:
                current_run = 1
                longest_equal_run = max(longest_equal_run, current_run)
                for previous, current in pairwise(representatives):
                    adjacent_total += 1
                    if previous == current:
                        adjacent_equal += 1
                        current_run += 1
                    else:
                        current_run = 1
                    longest_equal_run = max(longest_equal_run, current_run)

    difference_counts = Counter(shared_differences)
    return {
        "schema_version": 1,
        "velocity_policy_id": performance.velocity_policy_id,
        "foreground": _velocity_summary(foreground_values),
        "accompaniment": _velocity_summary(accompaniment_values),
        "accompaniment_attack_groups": {
            "group_count": group_count,
            "adjacent_pair_count": adjacent_total,
            "adjacent_equal_count": adjacent_equal,
            "adjacent_equal_ratio": (adjacent_equal / adjacent_total if adjacent_total else None),
            "longest_equal_run": longest_equal_run,
            "eligible_harmony_count": eligible_harmony_count,
            "varied_eligible_harmony_count": varied_eligible_harmony_count,
        },
        "shared_onset_velocity_difference": {
            "count": len(shared_differences),
            "minimum": min(shared_differences) if shared_differences else None,
            "maximum": max(shared_differences) if shared_differences else None,
            "most_common": [
                {"difference": difference, "count": count}
                for difference, count in sorted(
                    difference_counts.items(), key=lambda item: (-item[1], item[0])
                )[:10]
            ],
        },
    }


def _pitch_xml(parent: ET.Element, pitch: int) -> None:
    names = (
        ("C", 0),
        ("C", 1),
        ("D", 0),
        ("D", 1),
        ("E", 0),
        ("F", 0),
        ("F", 1),
        ("G", 0),
        ("G", 1),
        ("A", 0),
        ("A", 1),
        ("B", 0),
    )
    step, alter = names[pitch % 12]
    pitch_element = ET.SubElement(parent, "pitch")
    ET.SubElement(pitch_element, "step").text = step
    if alter:
        ET.SubElement(pitch_element, "alter").text = str(alter)
    ET.SubElement(pitch_element, "octave").text = str((pitch // 12) - 1)


def _write_harmony(measure: ET.Element, harmony: ScoreHarmony) -> None:
    names = (
        ("C", 0),
        ("C", 1),
        ("D", 0),
        ("D", 1),
        ("E", 0),
        ("F", 0),
        ("F", 1),
        ("G", 0),
        ("G", 1),
        ("A", 0),
        ("A", 1),
        ("B", 0),
    )
    step, alter = names[harmony.root_pitch_class]
    element = ET.SubElement(measure, "harmony")
    root = ET.SubElement(element, "root")
    ET.SubElement(root, "root-step").text = step
    if alter:
        ET.SubElement(root, "root-alter").text = str(alter)
    ET.SubElement(element, "kind").text = harmony.quality
    if harmony.at_units:
        ET.SubElement(element, "offset").text = str(harmony.at_units)


def _note_type(duration: int, divisions: int) -> str:
    options = (
        (4 * divisions, "whole"),
        (2 * divisions, "half"),
        (divisions, "quarter"),
        (max(1, divisions // 2), "eighth"),
    )
    return min(options, key=lambda item: abs(item[0] - duration))[1]


def _write_voice(
    measure: ET.Element, notes: list[ScoreNote], voice_number: str, divisions: int
) -> int:
    cursor = 0
    by_onset: dict[int, list[ScoreNote]] = defaultdict(list)
    for note in notes:
        by_onset[note.at_units].append(note)
    for onset, chord in sorted(by_onset.items()):
        if onset > cursor:
            forward = ET.SubElement(measure, "forward")
            ET.SubElement(forward, "duration").text = str(onset - cursor)
        elif onset < cursor:
            backup = ET.SubElement(measure, "backup")
            ET.SubElement(backup, "duration").text = str(cursor - onset)
        ordered_chord = sorted(chord, key=lambda item: (-item.duration_units, item.pitch))
        for index, note in enumerate(ordered_chord):
            note_element = ET.SubElement(measure, "note")
            if index:
                ET.SubElement(note_element, "chord")
            _pitch_xml(note_element, note.pitch)
            if note.tie is not None:
                ET.SubElement(note_element, "tie", type=note.tie)
            ET.SubElement(note_element, "duration").text = str(note.duration_units)
            ET.SubElement(note_element, "voice").text = voice_number
            ET.SubElement(note_element, "type").text = _note_type(note.duration_units, divisions)
            ET.SubElement(note_element, "staff").text = voice_number
            if note.articulations or note.tie is not None:
                notations = ET.SubElement(note_element, "notations")
                if note.tie is not None:
                    ET.SubElement(notations, "tied", type=note.tie)
                visible_articulations = tuple(
                    item for item in note.articulations if item != "normal"
                )
                if visible_articulations:
                    articulations = ET.SubElement(notations, "articulations")
                    for articulation in visible_articulations:
                        ET.SubElement(articulations, articulation)
        cursor = onset + ordered_chord[0].duration_units
    return cursor


def render_musicxml(plan: PiecePlan, score: ScoreSpec, output_path: Path) -> Path:
    """演奏時刻を含まない譜面層を決定的な MusicXML として書き出す。"""
    placeholder = PerformanceSpec("validation", 1, 64, "subtle-v1", ())
    validate_pipeline(plan, score, placeholder)
    return _render_musicxml_unchecked(plan, score, output_path)


def render_role_neutral_musicxml(plan: PiecePlan, score: ScoreSpec, output_path: Path) -> Path:
    """声部を前景または伴奏へ読み替えず、検証済みの譜面を書き出す。"""
    placeholder = PerformanceSpec("validation", 1, 64, "subtle-v1", ())
    _validate_pipeline_stages(
        plan,
        score,
        placeholder,
        require_harmony_foreground_voice=False,
    )
    return _render_musicxml_unchecked(plan, score, output_path)


def _render_musicxml_unchecked(plan: PiecePlan, score: ScoreSpec, output_path: Path) -> Path:
    leaves, _ = ordered_leaf_schedule(plan, score)
    materials = {material.material_id: material for material in score.materials}
    root = ET.Element("score-partwise", version="4.0")
    ET.SubElement(root, "movement-title").text = plan.title
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Piano"
    part = ET.SubElement(root, "part", id="P1")
    for number, (leaf, _, _) in enumerate(leaves, start=1):
        assert leaf.score_material_id is not None
        material = materials[leaf.score_material_id]
        measure = ET.SubElement(part, "measure", number=str(number))
        if number == 1:
            attributes = ET.SubElement(measure, "attributes")
            ET.SubElement(attributes, "divisions").text = str(score.divisions)
            time = ET.SubElement(attributes, "time")
            ET.SubElement(time, "senza-misura")
            ET.SubElement(attributes, "staves").text = "2"
        grouping = ET.SubElement(measure, "grouping", type="start", number="1")
        ET.SubElement(grouping, "feature", type="node-id").text = leaf.node_id
        for harmony in material.harmonies:
            _write_harmony(measure, harmony)
        for direction in sorted(material.directions, key=lambda item: item.at_units):
            direction_element = ET.SubElement(measure, "direction")
            direction_type = ET.SubElement(direction_element, "direction-type")
            if direction.kind == "dynamic":
                dynamics = ET.SubElement(direction_type, "dynamics")
                ET.SubElement(dynamics, direction.value)
            else:
                ET.SubElement(direction_type, "breath-mark").text = direction.value
            if direction.at_units:
                ET.SubElement(direction_element, "offset").text = str(direction.at_units)
        upper = [note for note in material.notes if note.voice == "upper"]
        lower = [note for note in material.notes if note.voice == "lower"]
        upper_cursor = _write_voice(measure, upper, "1", score.divisions)
        if upper_cursor:
            backup = ET.SubElement(measure, "backup")
            ET.SubElement(backup, "duration").text = str(upper_cursor)
        _write_voice(measure, lower, "2", score.divisions)
        ET.SubElement(measure, "grouping", type="stop", number="1")
    ET.indent(root, space="  ")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output


def _append_absolute_events(
    track: mido.MidiTrack, events: list[tuple[int, int, mido.Message]]
) -> None:
    previous = 0
    for tick, _, message in sorted(events, key=lambda item: (item[0], item[1])):
        message.time = tick - previous
        track.append(message)
        previous = tick


def render_performance_smf(
    rendered: RenderedPerformance, output_path: Path
) -> PerformanceRenderResult:
    """絶対時刻の演奏層を 1 tick = 1 ms の決定的な SMF として書き出す。"""
    midi = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT, charset="utf-8")
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name=rendered.title, time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=TEMPO, time=0))
    meta.append(mido.MetaMessage("end_of_track", time=rendered.duration_ms))
    midi.tracks.append(meta)
    piano = mido.MidiTrack()
    piano.append(mido.MetaMessage("track_name", name="Piano", time=0))
    piano.append(mido.Message("program_change", channel=0, program=0, time=0))
    events: list[tuple[int, int, mido.Message]] = []
    for note in rendered.notes:
        events.append(
            (
                note.at_ms,
                2,
                mido.Message("note_on", channel=0, note=note.pitch, velocity=note.velocity),
            )
        )
        events.append(
            (
                note.at_ms + note.duration_ms,
                0,
                mido.Message("note_off", channel=0, note=note.pitch, velocity=0),
            )
        )
    for pedal in rendered.pedals:
        events.append(
            (
                pedal.at_ms,
                1,
                mido.Message("control_change", channel=0, control=64, value=pedal.value),
            )
        )
    _append_absolute_events(piano, events)
    piano.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(piano)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output)
    reread = mido.MidiFile(output, charset="utf-8")
    if reread.ticks_per_beat != TICKS_PER_BEAT:
        raise ValueError("rendered SMF did not preserve ticks_per_beat")
    return PerformanceRenderResult(output, rendered.duration_ms, len(rendered.notes))


def _artifact_result(path: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": (
            "passed" if checks and all(item["status"] == "passed" for item in checks) else "failed"
        ),
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        "checks": checks,
    }


def _comparison_check(check_id: str, expected: Any, actual: Any) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "status": "passed" if expected == actual else "failed",
        "expected": expected,
        "actual": actual,
    }


def _dict_sort_key(item: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(item[name]) for name in sorted(item))


def _xml_pitch(note: ET.Element) -> int | None:
    pitch = note.find("pitch")
    if pitch is None:
        return None
    steps = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    step = pitch.findtext("step")
    octave = pitch.findtext("octave")
    if step not in steps or octave is None:
        return None
    return (int(octave) + 1) * 12 + steps[step] + int(pitch.findtext("alter", "0"))


def _xml_root_pitch_class(harmony: ET.Element) -> int | None:
    steps = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    step = harmony.findtext("root/root-step")
    if step not in steps:
        return None
    return (steps[step] + int(harmony.findtext("root/root-alter", "0"))) % 12


def _expected_musicxml_contents(
    plan: PiecePlan, score: ScoreSpec
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    leaves, _ = ordered_leaf_schedule(plan, score)
    materials = {item.material_id: item for item in score.materials}
    leaf_ids: list[str] = []
    notes: list[dict[str, Any]] = []
    harmonies: list[dict[str, Any]] = []
    directions: list[dict[str, Any]] = []
    for measure_number, (leaf, _, _) in enumerate(leaves, start=1):
        assert leaf.score_material_id is not None
        material = materials[leaf.score_material_id]
        leaf_ids.append(leaf.node_id)
        for note in material.notes:
            notes.append(
                {
                    "measure": measure_number,
                    "leaf": leaf.node_id,
                    "at": note.at_units,
                    "duration": note.duration_units,
                    "pitch": note.pitch,
                    "voice": "1" if note.voice == "upper" else "2",
                    "staff": "1" if note.voice == "upper" else "2",
                    "tie": note.tie,
                    "tied": note.tie,
                    "articulations": sorted(
                        item for item in note.articulations if item != "normal"
                    ),
                }
            )
        for harmony in material.harmonies:
            harmonies.append(
                {
                    "measure": measure_number,
                    "leaf": leaf.node_id,
                    "at": harmony.at_units,
                    "root_pitch_class": harmony.root_pitch_class,
                    "quality": harmony.quality,
                }
            )
        for direction in material.directions:
            directions.append(
                {
                    "measure": measure_number,
                    "leaf": leaf.node_id,
                    "at": direction.at_units,
                    "kind": direction.kind,
                    "value": direction.value,
                }
            )
    return (
        leaf_ids,
        sorted(notes, key=_dict_sort_key),
        sorted(harmonies, key=_dict_sort_key),
        sorted(directions, key=_dict_sort_key),
    )


def validate_musicxml_round_trip(plan: PiecePlan, score: ScoreSpec, path: Path) -> dict[str, Any]:
    """MusicXMLを再読込みし、ScoreSpecを全leafへ展開した意味と照合する。"""
    source = Path(path)
    try:
        root = ET.parse(source).getroot()
    except (OSError, ET.ParseError) as error:
        return _artifact_result(
            source,
            [_comparison_check("musicxml.parse", "readable MusicXML", str(error))],
        )
    leaf_ids, expected_notes, expected_harmonies, expected_directions = _expected_musicxml_contents(
        plan, score
    )
    parts = root.findall("part")
    part_ok = (
        len(parts) == 1
        and parts[0].get("id") == "P1"
        and root.findtext("part-list/score-part[@id='P1']/part-name") == "Piano"
    )
    measures = parts[0].findall("measure") if part_ok else []
    actual_measure_order = [item.get("number") for item in measures]
    actual_leaf_ids = [
        item.findtext("grouping[@type='start']/feature[@type='node-id']") for item in measures
    ]
    actual_divisions = measures[0].findtext("attributes/divisions") if measures else None
    actual_notes: list[dict[str, Any]] = []
    actual_harmonies: list[dict[str, Any]] = []
    actual_directions: list[dict[str, Any]] = []
    for measure_number, measure in enumerate(measures, start=1):
        leaf_id = actual_leaf_ids[measure_number - 1]
        cursor = 0
        previous_onset = 0
        for child in measure:
            if child.tag == "forward":
                cursor += int(child.findtext("duration", "0"))
            elif child.tag == "backup":
                cursor -= int(child.findtext("duration", "0"))
            elif child.tag == "note":
                duration = int(child.findtext("duration", "0"))
                is_chord = child.find("chord") is not None
                onset = previous_onset if is_chord else cursor
                if not is_chord:
                    previous_onset = onset
                    cursor += duration
                ties = [item.get("type") for item in child.findall("tie")]
                tied = [item.get("type") for item in child.findall("notations/tied")]
                articulations = sorted(
                    item.tag for item in child.findall("notations/articulations/*")
                )
                actual_notes.append(
                    {
                        "measure": measure_number,
                        "leaf": leaf_id,
                        "at": onset,
                        "duration": duration,
                        "pitch": _xml_pitch(child),
                        "voice": child.findtext("voice"),
                        "staff": child.findtext("staff"),
                        "tie": ties[0] if len(ties) == 1 else ties or None,
                        "tied": tied[0] if len(tied) == 1 else tied or None,
                        "articulations": articulations,
                    }
                )
        for harmony in measure.findall("harmony"):
            actual_harmonies.append(
                {
                    "measure": measure_number,
                    "leaf": leaf_id,
                    "at": int(harmony.findtext("offset", "0")),
                    "root_pitch_class": _xml_root_pitch_class(harmony),
                    "quality": harmony.findtext("kind"),
                }
            )
        for direction in measure.findall("direction"):
            dynamics = direction.findall("direction-type/dynamics/*")
            breath = direction.find("direction-type/breath-mark")
            if len(dynamics) == 1 and breath is None:
                kind, value = "dynamic", dynamics[0].tag
            elif not dynamics and breath is not None:
                kind, value = "breath", breath.text or ""
            else:
                kind, value = "invalid", None
            actual_directions.append(
                {
                    "measure": measure_number,
                    "leaf": leaf_id,
                    "at": int(direction.findtext("offset", "0")),
                    "kind": kind,
                    "value": value,
                }
            )
    checks = [
        _comparison_check("musicxml.version", "4.0", root.get("version")),
        _comparison_check("musicxml.part", True, part_ok),
        _comparison_check(
            "musicxml.measure_order",
            [str(index) for index in range(1, len(leaf_ids) + 1)],
            actual_measure_order,
        ),
        _comparison_check("musicxml.divisions", str(score.divisions), actual_divisions),
        _comparison_check("musicxml.leaf_nodes", leaf_ids, actual_leaf_ids),
        _comparison_check(
            "musicxml.notes", expected_notes, sorted(actual_notes, key=_dict_sort_key)
        ),
        _comparison_check(
            "musicxml.harmonies",
            expected_harmonies,
            sorted(actual_harmonies, key=_dict_sort_key),
        ),
        _comparison_check(
            "musicxml.directions",
            expected_directions,
            sorted(actual_directions, key=_dict_sort_key),
        ),
    ]
    return _artifact_result(source, checks)


def _absolute_track(track: mido.MidiTrack) -> list[tuple[int, mido.Message]]:
    absolute = 0
    result: list[tuple[int, mido.Message]] = []
    for message in track:
        absolute += message.time
        result.append((absolute, message))
    return result


def validate_smf_round_trip(rendered: RenderedPerformance, path: Path) -> dict[str, Any]:
    """SMFを再読込みし、RenderedPerformanceと搬送・演奏イベントを照合する。"""
    source = Path(path)
    try:
        midi = mido.MidiFile(source, charset="utf-8")
    except (OSError, EOFError, ValueError) as error:
        return _artifact_result(
            source,
            [_comparison_check("smf.parse", "readable SMF", str(error))],
        )
    tracks = [_absolute_track(track) for track in midi.tracks]
    meta = tracks[0] if tracks else []
    piano = tracks[1] if len(tracks) > 1 else []
    tempo = [(at, message.tempo) for at, message in meta if message.type == "set_tempo"]
    transport_actual = {
        "type": midi.type,
        "tracks": len(midi.tracks),
        "ticks_per_beat": midi.ticks_per_beat,
        "tempo": tempo,
    }
    transport_expected = {
        "type": 1,
        "tracks": 2,
        "ticks_per_beat": TICKS_PER_BEAT,
        "tempo": [(0, TEMPO)],
    }
    meta_types = [message.type for _, message in meta]
    piano_types = [message.type for _, message in piano]
    programs = [
        (at, message.channel, message.program)
        for at, message in piano
        if message.type == "program_change"
    ]
    piano_track_names = [
        (at, message.name) for at, message in piano if message.type == "track_name"
    ]
    messages_actual = {
        "meta_types": meta_types,
        "piano_types_allowed": all(
            item
            in {
                "track_name",
                "program_change",
                "note_on",
                "note_off",
                "control_change",
                "end_of_track",
            }
            for item in piano_types
        ),
        "channel_zero": all(
            not hasattr(message, "channel") or message.channel == 0 for _, message in piano
        ),
        "cc64_only": all(
            message.type != "control_change" or message.control == 64 for _, message in piano
        ),
        "programs": programs,
        "track_names": piano_track_names,
    }
    messages_expected = {
        "meta_types": ["track_name", "set_tempo", "end_of_track"],
        "piano_types_allowed": True,
        "channel_zero": True,
        "cc64_only": True,
        "programs": [(0, 0, 0)],
        "track_names": [(0, "Piano")],
    }
    active: dict[int, list[tuple[int, int]]] = defaultdict(list)
    actual_notes: list[tuple[int, int, int, int]] = []
    unmatched_note_off: list[tuple[int, int]] = []
    actual_pedals: list[tuple[int, int]] = []
    ordering_ok = True
    priorities_by_tick: dict[int, list[int]] = defaultdict(list)
    for at, message in piano:
        if message.type == "note_on" and message.velocity > 0:
            active[message.note].append((at, message.velocity))
            priorities_by_tick[at].append(2)
        elif message.type in {"note_off", "note_on"}:
            priorities_by_tick[at].append(0)
            if active[message.note]:
                onset, velocity = active[message.note].pop(0)
                actual_notes.append((message.note, onset, at - onset, velocity))
            else:
                unmatched_note_off.append((message.note, at))
        elif message.type == "control_change" and message.control == 64:
            priorities_by_tick[at].append(1)
            actual_pedals.append((at, message.value))
    for priorities in priorities_by_tick.values():
        ordering_ok = ordering_ok and priorities == sorted(priorities)
    dangling = sorted((pitch, *event) for pitch, events in active.items() for event in events)
    expected_notes = sorted(
        (item.pitch, item.at_ms, item.duration_ms, item.velocity) for item in rendered.notes
    )
    expected_pedals = sorted((item.at_ms, item.value) for item in rendered.pedals)
    meta_end = [at for at, message in meta if message.type == "end_of_track"]
    piano_end = [at for at, message in piano if message.type == "end_of_track"]
    ending_actual = {
        "meta_end": meta_end,
        "piano_end": piano_end,
        "dangling": dangling,
        "unmatched_note_off": unmatched_note_off,
        "final_pedal": actual_pedals[-1] if actual_pedals else None,
    }
    ending_expected = {
        "meta_end": [rendered.duration_ms],
        "piano_end": [rendered.duration_ms],
        "dangling": [],
        "unmatched_note_off": [],
        "final_pedal": (rendered.duration_ms, 0),
    }
    checks = [
        _comparison_check("smf.transport", transport_expected, transport_actual),
        _comparison_check("smf.messages", messages_expected, messages_actual),
        _comparison_check("smf.notes", expected_notes, sorted(actual_notes)),
        _comparison_check("smf.pedals", expected_pedals, sorted(actual_pedals)),
        _comparison_check("smf.ordering", True, ordering_ok),
        _comparison_check("smf.ending", ending_expected, ending_actual),
    ]
    return _artifact_result(source, checks)
